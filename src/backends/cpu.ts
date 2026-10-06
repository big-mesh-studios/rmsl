import { MATRIX_DIMENSIONS, Node, OutputStruct, ShaderType, TYPE_WIDTH } from "../core";

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
   * Backing arrays for `storage()` slots, keyed by each storage node's `.name`.
   * A `storage()`-based program runs once per element with `index` set to
   * that element's position — the same per-invocation semantics WGSL's
   * compute path gives it, driven by `CpuRoutine.compute()` instead of the
   * GPU's own dispatch. Any invocation may read or write any element.
   */
  storages?: Record<string, ArrayLike<number> & { [i: number]: number }>;
  /**
   * Storage buffers already in the WASM routine's own memory, by slot: the
   * byte address of the first element and the element count. A slot listed
   * here is read and written in place instead of being copied in from
   * `storages` and back out. Ignored by the JS target.
   */
  storageBuffers?: Record<string, { address: number; length: number }>;
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
 * The JavaScript value a shader type carries on the CPU: scalars are numbers
 * (or booleans), vectors and matrices are flat arrays — matrices in the
 * column-major order the rest of the library uses.
 */
export type CpuValue<A extends ShaderType> = A extends "float" | "int" | "uint"
  ? number
  : A extends "bool"
    ? boolean
    : A extends "bvec2" | "bvec3" | "bvec4"
      ? boolean[]
      : A extends `${string}sampler${string}`
        ? never
        : A extends "void"
          ? void
          : number[];

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
 * The runtime face of a compiled CPU function, common to `compileJSRoutine` and
 * `compileWasmRoutine` — not tied to any one stage or use: a plain compute
 * program runs through `compute()`, once per `storage()` index, with its
 * return value ignored (side effects land in `ctx.storages`), a vertex/fragment
 * program's `run()` is called once per vertex/pixel for its return
 * value, and `draw()` runs the whole grid in one call rather than one JS
 * call per pixel from the host side.
 *
 * `draw()` feeds each pixel's center — `(x + 0.5, y + 0.5)` — in as
 * `fragCoord`, holding every other input (uniforms, textures, ...) fixed
 * across the grid, and packs the result into one flat row-major buffer of
 * `width * height * componentCount` elements.
 *
 * Pass `out` to write into an existing buffer instead of allocating a new
 * one — e.g. a view over a `SharedArrayBuffer` so several workers can each
 * fill a row range into disjoint regions of one shared buffer. `out` must
 * already have the matching typed-array kind and be at least
 * `width * height * componentCount` elements; it is returned unchanged.
 */
export type CpuRoutine<A extends ShaderType = ShaderType> = {
  /** Runs the program once and returns its value. The value is the caller's: a later call does not change it. */
  run(ctx: CpuShaderContext): CpuValue<A>;
  /**
   * Runs the program once per pixel of a `width x height` grid, feeding each
   * pixel's center in as `fragCoord`, and packs the results into one flat
   * row-major buffer. The buffer is the caller's: a later `draw` does not
   * change it. Pass `out` to fill a buffer of your own instead.
   */
  draw(ctx: CpuShaderContext, width: number, height: number, out?: CpuDrawBuffer): CpuDrawBuffer;
  /**
   * Runs a compute program once per index in `0..count`, in index order, with
   * `invocationIndex()` reading that index, and leaves the results in
   * `ctx.storages`. The same as calling `run()` once per index, except that
   * a backend which keeps storage in memory of its own copies each buffer in
   * and out once for the whole dispatch rather than once per invocation, and
   * the WASM backend runs the loop itself inside the module.
   */
  compute(ctx: CpuShaderContext, count: number): void;
  /**
   * The shader type of one element of each storage buffer the program reads,
   * by slot. A compute adapter needs it to count the elements of the flat
   * typed array the host passes for a slot.
   */
  storageTypes?: Readonly<Record<string, ShaderType>>;
};

/**
 * A {@link CpuRoutine} compiled for a stage: its `run` returns the value, or
 * the {@link CpuShaderResult} a program that writes outputs, a position or a
 * depth hands back.
 */
export type CpuStageRoutine = Omit<CpuRoutine, "run"> & {
  run(ctx: CpuShaderContext): CpuValue<ShaderType> | CpuShaderResult | null;
};

/** A compiled function's scalar element kind, at the WASM/typed-array level. */
export type ScalarKind = "float" | "int" | "uint" | "bool";

export function scalarKindOf(t: string | undefined): ScalarKind {
  return t === "int" || t === "uint" || t === "bool" ? t : "float";
}

/**
 * The kind of value a type's components hold, scalar and vector types alike:
 * `int` and `ivec3` are both "int". Every backend asks this of a type name,
 * so it is answered here once.
 */
export function componentKindOf(t: string | undefined): ScalarKind {
  if (t === undefined) return "float";
  if (t === "bool" || t.startsWith("bvec")) return "bool";
  if (t === "int" || t.startsWith("ivec")) return "int";
  if (t === "uint" || t.startsWith("uvec")) return "uint";
  return "float";
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

/** A compile function of either CPU target for a program with no stage: it gives the {@link CpuRoutine} they share. */
export type CompileCpuRoutine = (
  fn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  options: { name: string; params: Array<{ name: string; type: ShaderType }> },
) => CpuRoutine;

/** What a vertex stage hands on: its position, and the varyings the fragment stage reads, by slot. */
export type VertexResult = {
  position: number[];
  varyings: Record<string, unknown>;
};

/**
 * The values an `outputStruct` writes, by position: one for each member, of
 * the JavaScript type its shader type has.
 */
export type CpuOutputs<M extends readonly Node<ShaderType>[]> = {
  -readonly [K in keyof M]: M[K] extends Node<infer A> ? CpuValue<A> : never;
};

/**
 * What a fragment stage writes for one fragment: its colour as the four
 * channels of a `vec4`, the values of its `outputStruct` by position, and the
 * depth it wrote. A stage that returns an `outputStruct` has no colour, and a
 * stage that returns a colour has no outputs. `R` is what the stage returns.
 */
export type FragmentResult<R = unknown> =
  R extends OutputStruct<infer M extends readonly Node<ShaderType>[]>
    ? { value: undefined; outputs: CpuOutputs<M>; fragDepth?: number }
    : unknown extends R
      ? { value: number[] | undefined; outputs: unknown[]; fragDepth?: number }
      : R extends Node<"void">
        ? { value: undefined; outputs: []; fragDepth?: number }
        : { value: number[]; outputs: []; fragDepth?: number };

/** A compiled vertex program, run once per vertex. */
export type VertexStage = {
  run(ctx: CpuShaderContext): VertexResult;
};

/** A compiled fragment program, run once per fragment. `null` is a fragment that discarded. */
export type FragmentStage<R = unknown> = {
  run(ctx: CpuShaderContext): FragmentResult<R> | null;
  /**
   * Runs the program once for each pixel of a `width x height` grid, with
   * `fragCoord()` at the centre of each pixel, and packs the colours into one
   * flat row-major buffer: a full-screen pass, with no triangles to rasterize.
   * Every other input, uniforms and textures included, is the same for every
   * pixel. A pixel that discards is zero in every channel. The buffer is the
   * caller's: a later `quad` does not change it. Pass `out` to fill a buffer
   * of your own instead.
   */
  quad(ctx: CpuShaderContext, width: number, height: number, out?: CpuDrawBuffer): CpuDrawBuffer;
};

/** A compiled compute program, run once per index of a dispatch. */
export type ComputeStage = {
  /**
   * Runs the program once per index in `0..count`, in index order, with
   * `invocationIndex()` reading that index, and leaves the results in
   * `ctx.storages`. It returns nothing.
   */
  dispatch(ctx: CpuShaderContext, count: number): void;
  /** The shader type of one element of each storage buffer the program reads, by slot. */
  storageTypes: Readonly<Record<string, ShaderType>>;
};

const isResultObject = (raw: unknown): raw is CpuShaderResult =>
  typeof raw === "object" && raw !== null && !Array.isArray(raw);

/** The {@link VertexResult} of what a routine compiled for the vertex stage returned. */
export function toVertexResult(raw: CpuValue<ShaderType> | CpuShaderResult | null): VertexResult {
  // A vertex stage that never writes the position itself has its `vec4` result become the position.
  const wrapped = isResultObject(raw);
  const position = (wrapped ? (raw.position ?? raw.value) : raw) as number[] | undefined;
  if (!position) {
    throw new Error("[RMSL] A vertex stage never wrote a position, with builtinPosition() or a vec4 result.");
  }
  return { position, varyings: (wrapped && (raw.varyings as Record<string, unknown>)) || {} };
}

/** The values a routine wrote to its outputs, in the order of their locations. */
function outputsInOrder(outputs: Record<string, unknown> | undefined): unknown[] {
  return Object.keys(outputs ?? {})
    .sort((a, b) => Number(a.replace(/\D/g, "")) - Number(b.replace(/\D/g, "")))
    .map((slot) => outputs![slot]);
}

/** The {@link FragmentResult} of what a routine compiled for the fragment stage returned, `null` when it discarded. */
export function toFragmentResult<R>(raw: CpuValue<ShaderType> | CpuShaderResult | null): FragmentResult<R> | null {
  if (raw === null) return null;
  const result: { value: number[] | undefined; outputs: unknown[]; fragDepth?: number } = isResultObject(raw)
    ? { value: Array.isArray(raw.value) ? (raw.value as number[]) : undefined, outputs: outputsInOrder(raw.outputs) }
    : { value: Array.isArray(raw) ? (raw as number[]) : undefined, outputs: [] };
  if (isResultObject(raw) && raw.fragDepth !== undefined) result.fragDepth = raw.fragDepth;
  return result as FragmentResult<R>;
}

/**
 * The typed array a grid of `A` fills: floats and matrices in a `Float64Array`,
 * integers and booleans in an `Int32Array`, unsigned integers in a
 * `Uint32Array`, with every component of every pixel one after another.
 */
export type GridBuffer<A extends ShaderType> = A extends "uint" | `uvec${string}`
  ? Uint32Array
  : A extends "int" | "bool" | `ivec${string}` | `bvec${string}`
    ? Int32Array
    : Float64Array;

/**
 * A program of `fragCoord()` evaluated over a grid of pixels, for what a
 * fragment stage does not give: the buffer takes the type of the result, and
 * there is no colour to convert it to.
 */
export type CpuGrid<A extends ShaderType = ShaderType> = {
  /**
   * Evaluates the program once for each pixel of a `width x height` grid, with
   * `fragCoord()` at the centre of each pixel, and packs the results into one
   * flat row-major buffer of `width * height * componentCount` elements. A
   * pixel that discards is zero in every channel. Every other input, uniforms
   * and textures included, is the same for every pixel. The buffer is the
   * caller's: a later `fill` does not change it. Pass `out` to fill a buffer
   * of your own instead; it must be the type `A` has and hold at least that
   * many elements, and it is returned.
   */
  fill(ctx: CpuShaderContext, width: number, height: number, out?: GridBuffer<A>): GridBuffer<A>;
};
