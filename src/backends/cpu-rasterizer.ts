import { componentCountOf, CpuDrawBuffer, CpuRoutine, CpuShaderContext } from "./cpu";
import { ShaderType } from "../core";

type Value = number | number[];

function scale(v: Value, s: number): Value {
  return typeof v === "number" ? v * s : v.map((x) => x * s);
}

function add(a: Value, b: Value): Value {
  return typeof a === "number" ? a + (b as number) : a.map((x, i) => x + (b as number[])[i]);
}

function sliceAttribute(buffer: ArrayLike<number>, index: number, width: number): Value {
  if (width === 1) return buffer[index];
  const out = new Array(width);
  for (let i = 0; i < width; i++) out[i] = buffer[index * width + i];
  return out;
}

type WrappedResult = { value?: Value; position?: number[]; varyings?: Record<string, Value> };

/**
 * Both compileJS and compileWasm only wrap a call's result in
 * `{ value, position, varyings, ... }` when the program actually needs to
 * report more than a bare value — a write to a varying/output/position, or
 * (WASM specifically) an aggregate return type. A program that just reads
 * and returns (a `vec4(vColor, 1.0)` fragment, say) hands back the plain
 * value instead, so callers can't assume the wrapped shape.
 */
function isWrapped(result: unknown): result is WrappedResult {
  return typeof result === "object" && result !== null && !Array.isArray(result);
}

export interface RasterizeTrianglesOptions {
  /** One flat typed/plain array per attribute slot, e.g. `pos.name: Float32Array`. */
  attributes: Record<string, ArrayLike<number>>;
  /** Each attribute's ShaderType, to know how many components it packs. */
  attributeTypes: Record<string, ShaderType>;
  uniforms?: Record<string, unknown>;
  textures?: CpuShaderContext["textures"];
  width: number;
  height: number;
  /** The fragment program's output width — vec4 -> 4, vec3 -> 3, float -> 1. */
  componentCount: number;
  out?: CpuDrawBuffer;
}

/**
 * Runs `vertex` once per vertex, then `fragment` once per pixel each
 * triangle covers, with perspective-correct interpolated varyings —
 * software equivalent of what createGlsl/createWgsl's GPU rasterizer does
 * for the same vertex()/fragment() pair. A plain (non-indexed) triangle
 * list, no depth test or near/far clipping — the same scope createGlsl's
 * default draw already has, meant for shader-output comparison rather than
 * a general rasterizer.
 */
export function rasterizeTriangles(
  vertex: CpuRoutine,
  fragment: CpuRoutine,
  options: RasterizeTrianglesOptions,
): CpuDrawBuffer {
  const { attributes, attributeTypes, uniforms, textures, width, height, componentCount } = options;

  const widths: Record<string, number> = {};
  for (const slot in attributeTypes) widths[slot] = componentCountOf(attributeTypes[slot]);

  const firstSlot = Object.keys(attributes)[0];
  if (!firstSlot) throw new Error("[RMSL] rasterizeTriangles needs at least one attribute buffer");
  const vertexCount = Math.floor(attributes[firstSlot].length / widths[firstSlot]);

  const positions: number[][] = new Array(vertexCount);
  const varyingsPerVertex: Record<string, Value>[] = new Array(vertexCount);
  for (let i = 0; i < vertexCount; i++) {
    const attrs: Record<string, unknown> = {};

    for (const slot in attributes) {
      attrs[slot] = sliceAttribute(attributes[slot], i, widths[slot]);
    }

    const raw = vertex.invoke({ attributes: attrs, uniforms, textures });
    // A vertex Fn that never calls builtinPosition() itself has its plain
    // `return vec4(...)` become the position instead (assertStageResult in
    // shared.ts requires exactly that).
    const position = (isWrapped(raw) ? (raw.position ?? raw.value) : raw) as number[] | undefined;

    if (!position) {
      throw new Error("[RMSL] vertex program never wrote a position (builtinPosition(), or a plain vec4 return)");
    }

    positions[i] = position;
    varyingsPerVertex[i] = (isWrapped(raw) && raw.varyings) || {};
  }

  const out = options.out ?? new Float64Array(width * height * componentCount);

  for (let t = 0; t + 2 < vertexCount; t += 3) {
    const p0 = positions[t];
    const p1 = positions[t + 1];
    const p2 = positions[t + 2];

    const w0 = p0[3];
    const w1 = p1[3];
    const w2 = p2[3];

    // Clip space -> NDC (perspective divide) -> screen space, y flipped to
    // match a canvas's top-down row order.
    const s0 = [((p0[0] / w0) * 0.5 + 0.5) * width, (1 - ((p0[1] / w0) * 0.5 + 0.5)) * height];
    const s1 = [((p1[0] / w1) * 0.5 + 0.5) * width, (1 - ((p1[1] / w1) * 0.5 + 0.5)) * height];
    const s2 = [((p2[0] / w2) * 0.5 + 0.5) * width, (1 - ((p2[1] / w2) * 0.5 + 0.5)) * height];

    const area = (s1[0] - s0[0]) * (s2[1] - s0[1]) - (s1[1] - s0[1]) * (s2[0] - s0[0]);
    if (area === 0) continue;

    const minX = Math.max(0, Math.floor(Math.min(s0[0], s1[0], s2[0])));
    const maxX = Math.min(width - 1, Math.ceil(Math.max(s0[0], s1[0], s2[0])));
    const minY = Math.max(0, Math.floor(Math.min(s0[1], s1[1], s2[1])));
    const maxY = Math.min(height - 1, Math.ceil(Math.max(s0[1], s1[1], s2[1])));

    const vary0 = varyingsPerVertex[t],
      vary1 = varyingsPerVertex[t + 1],
      vary2 = varyingsPerVertex[t + 2];
    const invW0 = 1 / w0,
      invW1 = 1 / w1,
      invW2 = 1 / w2;

    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const px = x + 0.5,
          py = y + 0.5;
        const e0 = (s1[0] - px) * (s2[1] - py) - (s1[1] - py) * (s2[0] - px);
        const e1 = (s2[0] - px) * (s0[1] - py) - (s2[1] - py) * (s0[0] - px);
        const e2 = (s0[0] - px) * (s1[1] - py) - (s0[1] - py) * (s1[0] - px);
        const inside = area > 0 ? e0 >= 0 && e1 >= 0 && e2 >= 0 : e0 <= 0 && e1 <= 0 && e2 <= 0;
        if (!inside) continue;

        const b0 = e0 / area,
          b1 = e1 / area,
          b2 = e2 / area;
        const invW = b0 * invW0 + b1 * invW1 + b2 * invW2;

        const varyings: Record<string, unknown> = {};
        for (const slot in vary0) {
          const perspSum = add(
            add(scale(vary0[slot], b0 * invW0), scale(vary1[slot], b1 * invW1)),
            scale(vary2[slot], b2 * invW2),
          );
          varyings[slot] = scale(perspSum, 1 / invW);
        }

        const raw = fragment.invoke({ varyings, uniforms, textures, fragCoord: [px, py] });
        const color = ((isWrapped(raw) ? raw.value : raw) ?? 0) as Value;
        const base = (y * width + x) * componentCount;
        if (typeof color === "number") out[base] = color;
        else for (let c = 0; c < componentCount; c++) out[base + c] = color[c] ?? 0;
      }
    }
  }

  return out;
}
