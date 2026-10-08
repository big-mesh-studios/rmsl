import { Node, ShaderType } from "../../core";
import { DrawClearOptions, DrawCountOptions, TRANSPARENT_BLACK } from "../adapter";
import {
  componentCountOf,
  CpuDrawBuffer,
  CpuShaderContext,
  elementKindOf,
  isAggregate,
  isResultObject,
  scalarKindOf,
  vertexPosition,
} from "../cpu";
import { compileJSProgram, CompileJSOptions } from "./js";

/** Homogeneous-clip-space near-plane epsilon — see rasterizer.md's clip-pass design (`rasterizer.wat`'s `W_CLIP_EPS`). */
const W_CLIP_EPS = 1e-5;

/**
 * Options shared by both stages of a {@link compileJS} pair — the same
 * fields `compileJSRoutine` itself takes, minus `name`/`stage`/`params`,
 * which `compileJS` fixes itself.
 *
 * `attributeTypes` exists because — unlike `compileWasmFn`, whose `params`
 * list carries each attribute's `ShaderType` for `compileWasm` to read back
 * — `compileJSFn` exposes no such metadata (JS has no fixed-address layout
 * to describe): the caller has to say how many components each attribute
 * slot packs.
 */
export interface CompileJSRasterOptions {
  attributeTypes: Record<string, ShaderType>;
  derivatives?: CompileJSOptions["derivatives"];
  reentrant?: CompileJSOptions["reentrant"];
  float?: CompileJSOptions["float"];
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
 * Same as `WasmRasterDrawOptions`: `clear` and `clearDepth` default to
 * `true`, so each `draw()` starts from a cleared colour and depth buffer,
 * as a three.js render does. Pass `false` to compose several draws onto
 * both buffers, occlusion included — mirroring how a WebGPU render pass
 * declares `loadOp`/`depthLoadOp` together, per pass, rather than clearing
 * as a separate operation.
 */
export interface JsRasterDrawOptions extends DrawCountOptions, DrawClearOptions {
  width: number;
  height: number;
  out?: CpuDrawBuffer;
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

/**
 * A vertex as the rasterizer keeps it: its clip-space position, and each
 * varying in the order of the draw's varying slots, a scalar as one component.
 */
interface ClipVertex {
  position: Float64Array;
  varyings: Float64Array[];
}

/** The target of a rasterizer between draws. */
const NO_TARGET = new Float64Array(0);

/**
 * Whether an edge running `(dx, dy)`, in a triangle wound so that its inside
 * lies where every edge function is positive, owns the pixel centres on it:
 * one that runs down the screen, or left along it. The two triangles sharing
 * an edge run it in opposite directions, so exactly one owns it, as WebGPU
 * gives a pixel on a shared edge to one triangle.
 */
function ownsEdge(dx: number, dy: number): boolean {
  return dy > 0 || (dy === 0 && dx < 0);
}

/** A vertex with room for a position and varyings of these widths. */
function makeVertex(widths: readonly number[]): ClipVertex {
  return { position: new Float64Array(4), varyings: widths.map((w) => new Float64Array(w)) };
}

/** The vertex a fraction `t` of the way from `a` to `b`. */
function lerpVertex(a: ClipVertex, b: ClipVertex, t: number): ClipVertex {
  const out = makeVertex(a.varyings.map((v) => v.length));
  for (let i = 0; i < 4; i++) out.position[i] = a.position[i]! + (b.position[i]! - a.position[i]!) * t;
  for (let k = 0; k < a.varyings.length; k++) {
    const x = a.varyings[k]!;
    const y = b.varyings[k]!;
    for (let i = 0; i < x.length; i++) out.varyings[k]![i] = x[i]! + (y[i]! - x[i]!) * t;
  }
  return out;
}

/**
 * One edge (`a` -> `b`) of a triangle's single-plane Sutherland-Hodgman
 * clip against `w > W_CLIP_EPS`: keep `a` if it's on the inside, and
 * whenever the edge crosses the plane (`a`/`b` disagree), emit the cut
 * point — same as `rasterizer.wat`'s `$clipEdge`.
 */
function clipEdge(poly: ClipVertex[], a: ClipVertex, b: ClipVertex): void {
  const aIn = a.position[3]! > W_CLIP_EPS;
  const bIn = b.position[3]! > W_CLIP_EPS;
  if (aIn) poly.push(a);
  if (aIn !== bIn) {
    const t = (W_CLIP_EPS - a.position[3]!) / (b.position[3]! - a.position[3]!);
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

// === Exported functions ===

/**
 * Compiles a vertex/fragment `Fn` pair and rasterizes them exactly like
 * `compileWasm`'s `WasmRasterRoutine` does, in plain JS instead of a
 * shared WASM module — see rasterizer.md for the algorithm both share.
 *
 * A draw allocates nothing for each vertex or fragment: the vertices, the
 * varyings a fragment reads and the context each stage is called with are
 * made once and filled again, and a stage's result is read where it left it.
 */
export function compileJS(
  vertexFn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  fragmentFn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  options: CompileJSRasterOptions,
): JsRasterRoutine {
  const stageOptions = {
    params: [],
    derivatives: options.derivatives,
    reentrant: options.reentrant,
    float: options.float,
  };
  const vertexStage = compileJSProgram(vertexFn, { ...stageOptions, name: "vtx", stage: "vertex" });
  const fragmentStage = compileJSProgram(fragmentFn, { ...stageOptions, name: "frag", stage: "fragment" });
  const writesDepth = fragmentStage.writesDepth;

  const attributeSlots = Object.keys(options.attributeTypes);
  const attributeWidths = attributeSlots.map((slot) => componentCountOf(options.attributeTypes[slot]!));
  const widths: Record<string, number> = {};
  attributeSlots.forEach((slot, i) => (widths[slot] = attributeWidths[i]!));
  /** The attributes of the vertex being shaded: a vector in an array of its own, filled for each vertex. */
  const vertexAttributes: Record<string, number | Float64Array | undefined> = {};
  attributeSlots.forEach((slot, i) => {
    vertexAttributes[slot] = attributeWidths[i]! > 1 ? new Float64Array(attributeWidths[i]!) : 0;
  });
  const vertexCtx: CpuShaderContext = { attributes: vertexAttributes };

  /** The varying slots the vertex stage writes, and their widths, as its program declares them. */
  const varyingSlots = Object.keys(vertexStage.varyingTypes);
  const varyingWidths = varyingSlots.map((slot) => componentCountOf(vertexStage.varyingTypes[slot]!));
  /** Whether each varying is an integer one, which a fragment reads flat, as its triangle's first vertex wrote it. */
  const varyingFlat = varyingSlots.map((slot) => {
    const type = vertexStage.varyingTypes[slot]!;
    return (isAggregate(type) ? elementKindOf(type) : scalarKindOf(type)) !== "float";
  });
  /** The varyings of the fragment being shaded: a vector in an array of its own, filled for each fragment. */
  const fragmentArrays = varyingWidths.map((w) => new Float64Array(w));
  const fragmentVaryings: Record<string, number | Float64Array> = {};
  varyingSlots.forEach((slot, k) => {
    fragmentVaryings[slot] = varyingWidths[k]! > 1 ? fragmentArrays[k]! : 0;
  });
  const fragCoord = new Float64Array(2);
  const fragmentCtx: CpuShaderContext = { varyings: fragmentVaryings, fragCoord };
  /** One vertex for each the draw shades, made the first time a draw has that many and kept. */
  const vertices: ClipVertex[] = [];

  // Persists across draw() calls of one size, like WasmRasterRoutine's depth
  // buffer: a draw of another width or height clears it, as its pixels lie elsewhere.
  let depthBuffer: Float64Array | null = null;
  let depthWidth = 0;
  let depthHeight = 0;
  // Persists across draw() calls too, same as WasmRasterRoutine's output
  // buffer address does — a draw that passes `clear: false` composes onto it.
  let colorBuffer: Float64Array | null = null;

  /** What the triangle being rasterized writes into, set by each draw. */
  let target: CpuDrawBuffer = NO_TARGET;
  let targetWidth = 0;
  let targetHeight = 0;

  /**
   * Rasterizes one triangle, already clipped to the near plane, into `target`.
   * `first` is the first vertex of the triangle it was clipped from, which gives the flat varyings.
   */
  function rasterize(v0: ClipVertex, v1: ClipVertex, v2: ClipVertex, first: ClipVertex): void {
    const width = targetWidth;
    const height = targetHeight;
    const w0 = v0.position[3]!,
      w1 = v1.position[3]!,
      w2 = v2.position[3]!;

    // Clip space -> NDC (perspective divide) -> screen space, y flipped
    // to match a canvas's top-down row order.
    const s0x = ((v0.position[0]! / w0) * 0.5 + 0.5) * width;
    const s0y = (1 - ((v0.position[1]! / w0) * 0.5 + 0.5)) * height;
    const s1x = ((v1.position[0]! / w1) * 0.5 + 0.5) * width;
    const s1y = (1 - ((v1.position[1]! / w1) * 0.5 + 0.5)) * height;
    const s2x = ((v2.position[0]! / w2) * 0.5 + 0.5) * width;
    const s2y = (1 - ((v2.position[1]! / w2) * 0.5 + 0.5)) * height;

    const area = (s1x - s0x) * (s2y - s0y) - (s1y - s0y) * (s2x - s0x);
    if (area === 0) return; // degenerate (zero-area) triangle: skip it
    // Each edge function, times `wind`, is positive inside whichever way the triangle winds.
    const wind = area > 0 ? 1 : -1;
    const owns0 = ownsEdge((s2x - s1x) * wind, (s2y - s1y) * wind);
    const owns1 = ownsEdge((s0x - s2x) * wind, (s0y - s2y) * wind);
    const owns2 = ownsEdge((s1x - s0x) * wind, (s1y - s0y) * wind);

    const minX = Math.max(0, Math.floor(Math.min(s0x, s1x, s2x)));
    const maxX = Math.min(width - 1, Math.ceil(Math.max(s0x, s1x, s2x)));
    const minY = Math.max(0, Math.floor(Math.min(s0y, s1y, s2y)));
    const maxY = Math.min(height - 1, Math.ceil(Math.max(s0y, s1y, s2y)));

    const invW0 = 1 / w0,
      invW1 = 1 / w1,
      invW2 = 1 / w2;
    // NDC depth (z/w) is affine in screen space, like x/w and y/w above,
    // so it interpolates with plain barycentric weights — no invW needed.
    const depth0 = v0.position[2]! / w0,
      depth1 = v1.position[2]! / w1,
      depth2 = v2.position[2]! / w2;
    const slots = varyingSlots;
    const depths = depthBuffer!;
    // A triangle wholly outside depth 0 to 1 draws nothing. One that crosses it drops each
    // pixel outside it, which is what clipping there drops, as depth over w is affine in
    // screen space; one inside it draws every pixel, whatever the rounding of its depth.
    const nearest = Math.min(depth0, depth1, depth2);
    const farthest = Math.max(depth0, depth1, depth2);
    if (farthest < 0 || nearest > 1) return;
    const crossesDepthRange = nearest < 0 || farthest > 1;

    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const px = x + 0.5,
          py = y + 0.5;
        const e0 = (s1x - px) * (s2y - py) - (s1y - py) * (s2x - px);
        const e1 = (s2x - px) * (s0y - py) - (s2y - py) * (s0x - px);
        const e2 = (s0x - px) * (s1y - py) - (s0y - py) * (s1x - px);
        const f0 = e0 * wind,
          f1 = e1 * wind,
          f2 = e2 * wind;
        if (f0 < 0 || f1 < 0 || f2 < 0) continue;
        if ((f0 === 0 && !owns0) || (f1 === 0 && !owns1) || (f2 === 0 && !owns2)) continue;

        const b0 = e0 / area,
          b1 = e1 / area,
          b2 = e2 / area;
        const invW = b0 * invW0 + b1 * invW1 + b2 * invW2;
        const pixelDepth = b0 * depth0 + b1 * depth1 + b2 * depth2;

        if (crossesDepthRange && (pixelDepth < 0 || pixelDepth > 1)) continue;
        const pixelIndex = y * width + x;
        // LEQUAL depth test: closer-or-equal wins. A stage that writes its depth is tested once it has run.
        if (!writesDepth && pixelDepth > depths[pixelIndex]!) continue;

        // Perspective-correct: each vertex's varying weighted by its barycentric over w, the sum over w again.
        const p0 = b0 * invW0,
          p1 = b1 * invW1,
          p2 = b2 * invW2,
          perspective = 1 / invW;
        for (let k = 0; k < slots.length; k++) {
          const a = v0.varyings[k]!,
            b = v1.varyings[k]!,
            c = v2.varyings[k]!,
            out = fragmentArrays[k]!;
          if (varyingFlat[k]) out.set(first.varyings[k]!);
          else for (let i = 0; i < out.length; i++) out[i] = (a[i]! * p0 + b[i]! * p1 + c[i]! * p2) * perspective;
          if (out.length === 1) fragmentVaryings[slots[k]!] = out[0]!;
        }
        fragCoord[0] = px;
        fragCoord[1] = py;

        const raw = fragmentStage.runInPlace(fragmentCtx);
        // A fragment that discards leaves the pixel and its depth as they were.
        if (raw === null) continue;
        const written = isResultObject(raw) ? raw.fragDepth : undefined;
        // A written depth is clamped to the depth range, as WebGPU clamps it.
        const depth = typeof written === "number" ? Math.min(Math.max(written, 0), 1) : pixelDepth;
        if (writesDepth && depth > depths[pixelIndex]!) continue;
        depths[pixelIndex] = depth;
        // A fragment that writes no colour leaves the pixel as it was.
        const color = ArrayBuffer.isView(raw)
          ? (raw as Float64Array)
          : ((isResultObject(raw) ? raw.value : undefined) as Float64Array);
        if (!color) continue;
        const base = pixelIndex * 4;
        for (let c = 0; c < 4; c++) target[base + c] = color[c] ?? 0;
      }
    }
  }

  /**
   * Points both stages' contexts at the inputs of `from` that every vertex and
   * fragment shares. The contexts own their attributes and varyings.
   */
  function shareInputs(from: Pick<JsRasterContext, "uniforms" | "textures">): void {
    vertexCtx.uniforms = fragmentCtx.uniforms = from.uniforms;
    vertexCtx.textures = fragmentCtx.textures = from.textures;
  }

  function draw(ctx: JsRasterContext, options: JsRasterDrawOptions): CpuDrawBuffer {
    shareInputs(ctx);
    try {
      return drawShaded(ctx, options);
    } finally {
      // The contexts are kept for the next draw, so they let go of this one's inputs.
      shareInputs({});
      target = NO_TARGET;
    }
  }

  function drawShaded(ctx: JsRasterContext, options: JsRasterDrawOptions): CpuDrawBuffer {
    const { width, height, out } = options;
    const { attributes } = ctx;
    const first = options.first ?? 0;
    let firstSlot: string | undefined;
    for (const slot in attributes) {
      firstSlot = slot;
      break;
    }
    const inferredCount = firstSlot ? Math.floor(attributes[firstSlot]!.length / widths[firstSlot]!) - first : 0;
    const vertexCount = options.count ?? inferredCount;

    // vertex pass
    for (let i = 0; i < vertexCount; i++) {
      for (let a = 0; a < attributeSlots.length; a++) {
        const slot = attributeSlots[a]!;
        const buffer = attributes[slot];
        const w = attributeWidths[a]!;
        if (buffer === undefined) vertexAttributes[slot] = undefined;
        else if (w === 1) vertexAttributes[slot] = buffer[i + first]!;
        else {
          let into = vertexAttributes[slot];
          if (!(into instanceof Float64Array)) into = vertexAttributes[slot] = new Float64Array(w);
          for (let k = 0; k < w; k++) into[k] = buffer[(i + first) * w + k]!;
        }
      }
      const raw = vertexStage.runInPlace(vertexCtx);
      const position = vertexPosition(raw);
      const varyings = isResultObject(raw) ? raw.varyings : undefined;
      const vertex = (vertices[i] ??= makeVertex(varyingWidths));
      for (let c = 0; c < 4; c++) vertex.position[c] = position[c]!;
      for (let k = 0; k < varyingSlots.length; k++) {
        // The stage gives 0 for a varying this vertex does not write, and a bool as true or false.
        const value = varyings![varyingSlots[k]!];
        const into = vertex.varyings[k]!;
        if (typeof value === "number") into[0] = value;
        else if (typeof value === "boolean") into[0] = value ? 1 : 0;
        else for (let c = 0; c < into.length; c++) into[c] = (value as ArrayLike<number>)[c]!;
      }
    }

    const pixelCount = width * height;
    if (!depthBuffer || depthBuffer.length < pixelCount) {
      depthBuffer = new Float64Array(pixelCount).fill(Infinity);
    } else if (options.clearDepth !== false || width !== depthWidth || height !== depthHeight) {
      depthBuffer.fill(Infinity);
    }
    depthWidth = width;
    depthHeight = height;

    if (!out && (!colorBuffer || colorBuffer.length < pixelCount * 4)) {
      colorBuffer = new Float64Array(pixelCount * 4);
    }
    const result = out ?? colorBuffer!;
    if (options.clear !== false) {
      const clearColor = options.clearColor ?? TRANSPARENT_BLACK;
      for (let i = 0; i < pixelCount * 4; i += 4) {
        result[i] = clearColor[0]!;
        result[i + 1] = clearColor[1]!;
        result[i + 2] = clearColor[2]!;
        result[i + 3] = clearColor[3]!;
      }
    }

    target = result;
    targetWidth = width;
    targetHeight = height;
    for (let t = 0; t + 2 < vertexCount; t += 3) {
      const v0 = vertices[t]!,
        v1 = vertices[t + 1]!,
        v2 = vertices[t + 2]!;
      // A triangle wholly in front of the near plane is rasterized as it is; only one that crosses it is clipped.
      if (v0.position[3]! > W_CLIP_EPS && v1.position[3]! > W_CLIP_EPS && v2.position[3]! > W_CLIP_EPS) {
        rasterize(v0, v1, v2, v0);
        continue;
      }
      const clipped: ClipVertex[][] = [];
      clipTriangle(v0, v1, v2, clipped);
      for (const [c0, c1, c2] of clipped) rasterize(c0!, c1!, c2!, v0);
    }

    return result;
  }

  return { draw };
}
