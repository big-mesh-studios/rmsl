import { Node, ShaderType } from "../../core";
import { add, scale, sliceAttribute, isWrapped, Value } from "../cpu-rasterizer";
import { componentCountOf, CpuDrawBuffer, CpuShaderContext } from "../cpu";
import { compileJSRoutine, CompileJSOptions } from "./js";

/**
 * Options shared by both stages of a {@link compileJS} pair — the same
 * fields `compileJSRoutine` itself takes, minus `name`/`stage`/`params`,
 * which `compileJS` fixes itself.
 *
 * `attributeTypes` exists because — unlike `compileWasmFn`, whose `params`
 * list carries each attribute's `ShaderType` for `compileWasm` to read back
 * — `compileJSFn` exposes no such metadata (JS has no fixed-address layout
 * to describe): the caller has to say how many components each attribute
 * slot packs, the same way `rasterizeTriangles` already requires it.
 */
export interface CompileJSRasterOptions {
  attributeTypes: Record<string, ShaderType>;
  derivatives?: CompileJSOptions["derivatives"];
  reentrant?: CompileJSOptions["reentrant"];
}

/**
 * One draw call's inputs — mirrors `WasmRasterContext`, minus `textures`'
 * WASM-specific shape (`CpuShaderContext["textures"]` here instead).
 */
export interface JsRasterContext {
  attributes: Record<string, ArrayLike<number>>;
  uniforms?: Record<string, unknown>;
  textures?: CpuShaderContext["textures"];
}

/**
 * Same as `WasmRasterDrawOptions`: `clear`/`clearDepth` default to
 * `false`, so several `draw()` calls in a row compose onto both buffers
 * by default — mirroring how a WebGPU render pass declares
 * `loadOp`/`depthLoadOp` together, per pass, rather than clearing as a
 * separate operation.
 */
export interface JsRasterDrawOptions {
  /** Non-indexed triangle list, so a multiple of 3. */
  vertexCount: number;
  width: number;
  height: number;
  out?: CpuDrawBuffer;
  clear?: boolean;
  clearDepth?: boolean;
}

/**
 * The JS-side counterpart to `WasmRasterRoutine` — same shape, same
 * semantics (near-plane clipping, a persistent LEQUAL depth buffer, both
 * clears declared per `draw()` call via {@link JsRasterDrawOptions}), a
 * plain-JS implementation of the same algorithm `rasterizer.wat` compiles
 * to WASM bytecode. Always produces `vec4` output, matching
 * `compileWasm`'s own fragment-result requirement.
 */
export interface JsRasterRoutine {
  draw(ctx: JsRasterContext, options: JsRasterDrawOptions): CpuDrawBuffer;
}

/** Homogeneous-clip-space near-plane epsilon — see rasterizer.md's clip-pass design (`rasterizer.wat`'s `W_CLIP_EPS`). */
const W_CLIP_EPS = 1e-5;

interface ClipVertex {
  position: number[];
  varyings: Record<string, Value>;
}

function lerpValue(a: Value, b: Value, t: number): Value {
  return typeof a === "number" ? a + ((b as number) - a) * t : a.map((x, i) => x + ((b as number[])[i] - x) * t);
}

function lerpVertex(a: ClipVertex, b: ClipVertex, t: number): ClipVertex {
  const varyings: Record<string, Value> = {};
  for (const slot in a.varyings) varyings[slot] = lerpValue(a.varyings[slot], b.varyings[slot], t);
  return { position: a.position.map((x, i) => x + (b.position[i] - x) * t), varyings };
}

/**
 * One edge (`a` -> `b`) of a triangle's single-plane Sutherland-Hodgman
 * clip against `w > W_CLIP_EPS`: keep `a` if it's on the inside, and
 * whenever the edge crosses the plane (`a`/`b` disagree), emit the cut
 * point — same as `rasterizer.wat`'s `$clipEdge`.
 */
function clipEdge(poly: ClipVertex[], a: ClipVertex, b: ClipVertex): void {
  const aIn = a.position[3] > W_CLIP_EPS;
  const bIn = b.position[3] > W_CLIP_EPS;
  if (aIn) poly.push(a);
  if (aIn !== bIn) {
    const t = (W_CLIP_EPS - a.position[3]) / (b.position[3] - a.position[3]);
    poly.push(lerpVertex(a, b, t));
  }
}

/**
 * Clips one triangle, appending 0, 1 (3 vertices), or 2 (a fan-
 * triangulated quad, 4 vertices) resulting triangles to `out` — the same
 * fan order `rasterizer.wat`'s `$rasterize` clip pass uses.
 */
function clipTriangle(v0: ClipVertex, v1: ClipVertex, v2: ClipVertex, out: ClipVertex[][]): void {
  const poly: ClipVertex[] = [];
  clipEdge(poly, v0, v1);
  clipEdge(poly, v1, v2);
  clipEdge(poly, v2, v0);
  if (poly.length >= 3) out.push([poly[0]!, poly[1]!, poly[2]!]);
  if (poly.length === 4) out.push([poly[0]!, poly[2]!, poly[3]!]);
}

/**
 * Compiles a vertex/fragment `Fn` pair and rasterizes them exactly like
 * `compileWasm`'s `WasmRasterRoutine` does, in plain JS instead of a
 * shared WASM module — see rasterizer.md for the algorithm both share.
 */
export function compileJS(
  vertexFn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  fragmentFn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  options: CompileJSRasterOptions,
): JsRasterRoutine {
  const vertexRoutine = compileJSRoutine(vertexFn, {
    name: "vtx",
    params: [],
    stage: "vertex",
    derivatives: options.derivatives,
    reentrant: options.reentrant,
  });
  const fragmentRoutine = compileJSRoutine(fragmentFn, {
    name: "frag",
    params: [],
    derivatives: options.derivatives,
    reentrant: options.reentrant,
  });

  const widths: Record<string, number> = {};
  for (const slot in options.attributeTypes) widths[slot] = componentCountOf(options.attributeTypes[slot]!);

  // Persists across draw() calls, like WasmRasterRoutine's depth buffer —
  // grown (and re-cleared) only when a draw() needs more pixels than it
  // currently holds, never moved or shrunk otherwise.
  let depthBuffer: Float64Array | null = null;
  // Persists across draw() calls too, same as WasmRasterRoutine's output
  // buffer address does by default — several draw() calls in a row
  // compose onto it unless `out` (caller-owned) or `clear` says otherwise.
  let colorBuffer: Float64Array | null = null;

  function draw(ctx: JsRasterContext, options: JsRasterDrawOptions): CpuDrawBuffer {
    const { vertexCount, width, height, out } = options;
    const { attributes, uniforms, textures } = ctx;

    // vertex pass
    const vertices: ClipVertex[] = new Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) {
      const attrs: Record<string, unknown> = {};
      for (const slot in attributes) attrs[slot] = sliceAttribute(attributes[slot]!, i, widths[slot]!);

      const raw = vertexRoutine.invoke({ attributes: attrs, uniforms, textures });
      // A vertex Fn that never calls builtinPosition() itself has its plain
      // `return vec4(...)` become the position instead (assertStageResult).
      const position = (isWrapped(raw) ? (raw.position ?? raw.value) : raw) as number[] | undefined;
      if (!position) {
        throw new Error(
          "[RMSL] compileJS: vertexFn never wrote a position (builtinPosition(), or a plain vec4 return)",
        );
      }
      vertices[i] = { position, varyings: (isWrapped(raw) && raw.varyings) || {} };
    }

    // near-plane clip pass
    const clippedTriangles: ClipVertex[][] = [];
    for (let t = 0; t + 2 < vertexCount; t += 3) {
      clipTriangle(vertices[t]!, vertices[t + 1]!, vertices[t + 2]!, clippedTriangles);
    }

    const pixelCount = width * height;
    if (!depthBuffer || depthBuffer.length < pixelCount) {
      depthBuffer = new Float64Array(pixelCount).fill(Infinity);
    } else if (options.clearDepth) {
      depthBuffer.fill(Infinity);
    }

    if (!out && (!colorBuffer || colorBuffer.length < pixelCount * 4)) {
      colorBuffer = new Float64Array(pixelCount * 4);
    }
    const result = out ?? colorBuffer!;
    if (options.clear) result.fill(0, 0, pixelCount * 4);

    for (const [v0, v1, v2] of clippedTriangles) {
      const w0 = v0!.position[3]!,
        w1 = v1!.position[3]!,
        w2 = v2!.position[3]!;

      // Clip space -> NDC (perspective divide) -> screen space, y flipped
      // to match a canvas's top-down row order.
      const s0x = ((v0!.position[0]! / w0) * 0.5 + 0.5) * width;
      const s0y = (1 - ((v0!.position[1]! / w0) * 0.5 + 0.5)) * height;
      const s1x = ((v1!.position[0]! / w1) * 0.5 + 0.5) * width;
      const s1y = (1 - ((v1!.position[1]! / w1) * 0.5 + 0.5)) * height;
      const s2x = ((v2!.position[0]! / w2) * 0.5 + 0.5) * width;
      const s2y = (1 - ((v2!.position[1]! / w2) * 0.5 + 0.5)) * height;

      const area = (s1x - s0x) * (s2y - s0y) - (s1y - s0y) * (s2x - s0x);
      if (area === 0) continue; // degenerate (zero-area) triangle: skip it

      const minX = Math.max(0, Math.floor(Math.min(s0x, s1x, s2x)));
      const maxX = Math.min(width - 1, Math.ceil(Math.max(s0x, s1x, s2x)));
      const minY = Math.max(0, Math.floor(Math.min(s0y, s1y, s2y)));
      const maxY = Math.min(height - 1, Math.ceil(Math.max(s0y, s1y, s2y)));

      const invW0 = 1 / w0,
        invW1 = 1 / w1,
        invW2 = 1 / w2;
      // NDC depth (z/w) is affine in screen space, like x/w and y/w above,
      // so it interpolates with plain barycentric weights — no invW needed.
      const depth0 = v0!.position[2]! / w0,
        depth1 = v1!.position[2]! / w1,
        depth2 = v2!.position[2]! / w2;
      const vary0 = v0!.varyings,
        vary1 = v1!.varyings,
        vary2 = v2!.varyings;

      for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
          const px = x + 0.5,
            py = y + 0.5;
          const e0 = (s1x - px) * (s2y - py) - (s1y - py) * (s2x - px);
          const e1 = (s2x - px) * (s0y - py) - (s2y - py) * (s0x - px);
          const e2 = (s0x - px) * (s1y - py) - (s0y - py) * (s1x - px);
          const inside = area > 0 ? e0 >= 0 && e1 >= 0 && e2 >= 0 : e0 <= 0 && e1 <= 0 && e2 <= 0;
          if (!inside) continue;

          const b0 = e0 / area,
            b1 = e1 / area,
            b2 = e2 / area;
          const invW = b0 * invW0 + b1 * invW1 + b2 * invW2;
          const pixelDepth = b0 * depth0 + b1 * depth1 + b2 * depth2;

          const pixelIndex = y * width + x;
          // LEQUAL depth test: closer-or-equal wins.
          if (pixelDepth > depthBuffer[pixelIndex]!) continue;
          depthBuffer[pixelIndex] = pixelDepth;

          const varyings: Record<string, unknown> = {};
          for (const slot in vary0) {
            const perspSum = add(
              add(scale(vary0[slot]!, b0 * invW0), scale(vary1[slot]!, b1 * invW1)),
              scale(vary2[slot]!, b2 * invW2),
            );
            varyings[slot] = scale(perspSum, 1 / invW);
          }

          const raw = fragmentRoutine.invoke({ varyings, uniforms, textures, fragCoord: [px, py] });
          const color = ((isWrapped(raw) ? raw.value : raw) ?? 0) as Value;
          const base = pixelIndex * 4;
          if (typeof color === "number") result[base] = color;
          else for (let c = 0; c < 4; c++) result[base + c] = color[c] ?? 0;
        }
      }
    }

    return result;
  }

  return { draw };
}
