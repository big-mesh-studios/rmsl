/**
 * Shared types for the two CPU backends (`js.ts` and
 * `wasm.ts`): what a host passes in and reads back, and the shape a
 * compiled function itself takes.
 *
 * Both backends read the same host-supplied context and texture data, and
 * both produce a callable with the same two-part shape — call it once per
 * value, or `draw()` it once per pixel over a whole image — so the contract
 * is described here once rather than per backend.
 */
import { MATRIX_DIMENSIONS, TYPE_WIDTH } from "../core";

/** Values a host supplies to a compiled CPU function. */
export type CpuShaderContext = {
  params?: Record<string, unknown>;
  uniforms?: Record<string, unknown>;
  varyings?: Record<string, unknown>;
  attributes?: Record<string, unknown>;
  textures?: Record<string, CpuTextureData>;
  /** Pixel being evaluated, which `fragCoord()` reads on the CPU target. */
  fragCoord?: [number, number];
  /**
   * Backing arrays for `storage()` slots, one per name passed to `storage()`.
   * A `storage()`-based program is called once per element with `index` set
   * to that element's position — the same per-invocation semantics WGSL's
   * compute path gives it, just driven by a host-side loop instead of the
   * GPU's own dispatch.
   */
  storages?: Record<string, ArrayLike<number> & { [i: number]: number }>;
  /** The current element index, which `invocationIndex()` reads on the CPU target. */
  index?: number;
};

/** How a coordinate outside the image is turned into one inside it. */
export type CpuTextureWrap = "clamp" | "repeat" | "mirror";

/**
 * Texture data a CPU target samples from.
 *
 * Sampling is described the way the renderers describe it to themselves —
 * the `SamplerState` the scene layer derives from a texture's three.js
 * constants — so both CPU backends are driven by one reading of what the
 * texture asked for, rather than each deciding for itself what a constant
 * means.
 */
export type CpuTextureData = {
  data: ArrayLike<number>;
  width: number;
  height: number;
  depth?: number;
  /**
   * How many of the four channels each texel stores, which is what says where
   * one texel ends and the next begins. Four by default; a single-channel
   * texture is one. The channels a texel does not store read the way a sampler
   * reports them on a device: zero for green and blue, one for alpha.
   */
  channels?: 1 | 2 | 3 | 4;
  /**
   * What to do when the sampled point falls between texels: `"nearest"` (the
   * default) takes the texel it lands in, `"linear"` blends the neighbours.
   *
   * `minFilter` is accepted so a sampler state can be handed over whole, and
   * ignored: choosing between the two needs the footprint of the pixel being
   * shaded, which a single CPU evaluation has no way to know.
   */
  magFilter?: "nearest" | "linear";
  minFilter?: "nearest" | "linear";
  /** What happens outside `0..1`, per axis. All default to `"clamp"`. */
  wrapS?: CpuTextureWrap;
  wrapT?: CpuTextureWrap;
  wrapR?: CpuTextureWrap;
};

/**
 * What a compiled CPU function returns when the program writes outputs, a
 * position or the fragment depth; otherwise the Fn's bare return value.
 */
export type CpuShaderResult = {
  value?: unknown;
  outputs?: Record<string, unknown>;
  varyings?: Record<string, unknown>;
  position?: number[];
  fragDepth?: number;
};

/** The typed array `draw()` fills, matching the result's declared kind. */
export type CpuDrawBuffer = Float64Array | Int32Array | Uint32Array;

/**
 * The runtime face of a compiled CPU function, common to `compileJS` and
 * `compileWasm`: a plain callable per invocation, plus `draw()` for
 * rendering the result over a whole `width x height` pixel grid without a
 * JS call per pixel from the host side.
 *
 * `draw()` feeds each pixel's center — `(x + 0.5, y + 0.5)` — in as
 * `fragCoord`, holding every other input (uniforms, textures, ...) fixed
 * across the grid, and packs the result into one flat row-major buffer of
 * `width * height * componentCount` elements.
 *
 * Pass `out` to write into an existing buffer instead of allocating a new
 * one — e.g. a view over a `SharedArrayBuffer` so several workers can each
 * draw a row range into disjoint regions of one shared buffer. `out` must
 * already have the matching typed-array kind and be at least
 * `width * height * componentCount` elements; it is returned unchanged.
 */
export type CpuRenderer = ((ctx: CpuShaderContext) => number | boolean | CpuShaderResult) & {
  draw(ctx: CpuShaderContext, width: number, height: number, out?: CpuDrawBuffer): CpuDrawBuffer;
};

/** A compiled function's scalar element kind, at the WASM/typed-array level. */
export type ScalarKind = "float" | "int" | "uint" | "bool";

export function scalarKindOf(t: string | undefined): ScalarKind {
  return t === "int" || t === "uint" || t === "bool" ? t : "float";
}

/** Element kind of a shader type's components; non-int vectors default to float. */
export function elementKindOf(t: string): ScalarKind {
  if (t.startsWith("ivec")) return "int";
  if (t.startsWith("uvec")) return "uint";
  if (t.startsWith("bvec")) return "bool";
  return "float";
}

/** Component count: 1 for scalars, TYPE_WIDTH for vectors, rows*cols for matrices. */
export function componentCountOf(t: string): number {
  const width = TYPE_WIDTH[t];
  if (width !== undefined) return width;
  const shape = MATRIX_DIMENSIONS[t];
  if (shape !== undefined) return shape[0] * shape[1];
  return 1;
}

/**
 * True when the type is an aggregate: a vector or matrix made up of more than
 * one scalar component (as opposed to a single scalar like f32/i32/u32/b32).
 */
export function isAggregate(t: string): boolean {
  return componentCountOf(t) > 1;
}
