import { MATRIX_DIMENSIONS, Node, OutputStruct, ShaderType, TYPE_WIDTH } from "../core";

/** Values a host supplies to a compiled CPU function. */
export type CpuShaderContext = {
  params?: Record<string, unknown>;
  uniforms?: Record<string, unknown>;
  varyings?: Record<string, unknown>;
  attributes?: Record<string, unknown>;
  textures?: Record<string, CpuTextureData>;
  /** Pixel being evaluated, which `fragCoord()` reads on the CPU target. */
  fragCoord?: [number, number] | Float64Array;
  /**
   * Backing arrays for `storage()` slots, keyed by each storage node's `.name`.
   * A `storage()`-based program runs once per element with `index` set to
   * that element's position — the same per-invocation semantics WGSL's
   * compute path gives it, driven by a compute stage instead of the GPU's own
   * dispatch. Any invocation may read or write any element. Each buffer is one
   * flat array that holds the components of its elements one after another.
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
 * The JavaScript value a shader type carries on the CPU, at the float width
 * `W` the program was compiled at: scalars are numbers (or booleans), vectors
 * and matrices typed arrays of their kind — matrices in the column-major order
 * the rest of the library uses. A boolean vector holds 1 and 0.
 */
export type CpuValue<A extends ShaderType, W extends FloatWidth = "f64"> = A extends "float" | "int" | "uint"
  ? number
  : A extends "bool"
    ? boolean
    : A extends `${string}sampler${string}`
      ? never
      : A extends "void"
        ? void
        : A extends `ivec${string}` | `bvec${string}`
          ? Int32Array
          : A extends `uvec${string}`
            ? Uint32Array
            : FloatArray<W>;

/**
 * What a compiled CPU function returns when the program writes outputs, a
 * position or the fragment depth; otherwise the Fn's bare return value.
 */
export type CpuProgramResult = {
  value?: unknown;
  outputs?: Record<string, unknown>;
  varyings?: Record<string, unknown>;
  position?: ArrayLike<number>;
  fragDepth?: number;
};

/** The typed array a grid fills, matching the result's declared kind. */
export type CpuDrawBuffer = Float64Array | Float32Array | Int32Array | Uint32Array;

/** The width a CPU program computes a `float` in, as its compile's `float` option picks. */
export type FloatWidth = "f64" | "f32";

/** The typed array a float vector or matrix is held in at a width: a `Float32Array` at `"f32"`. */
export type FloatArray<W extends FloatWidth> = W extends "f32" ? Float32Array : Float64Array;

/**
 * The typed array a vector or matrix of `type` is held in on the CPU: floats
 * in a `Float64Array`, or a `Float32Array` when `float32`, integers and
 * booleans in an `Int32Array`, and unsigned integers in a `Uint32Array`.
 */
export function typedArrayOf(type: string, float32: boolean): TypedArrayConstructor {
  return typedArrayOfKind(elementKindOf(type), float32);
}

/** The constructors of the typed arrays a CPU target holds a value in. */
export type TypedArrayConstructor =
  Float64ArrayConstructor | Float32ArrayConstructor | Int32ArrayConstructor | Uint32ArrayConstructor;

/** {@link typedArrayOf} for the kind of a component: `"float"`, `"int"`, `"uint"` or `"bool"`. */
export function typedArrayOfKind(kind: ScalarKind, float32: boolean): TypedArrayConstructor {
  if (kind === "float") return float32 ? Float32Array : Float64Array;
  return kind === "uint" ? Uint32Array : Int32Array;
}

/** A copy of the components of a vector or matrix of `type`, in the typed array it is held in; a boolean is 1 or 0. */
export function typedValue(value: ArrayLike<number | boolean>, type: string, float32: boolean): CpuDrawBuffer {
  const Typed = typedArrayOf(type, float32);
  const out = new Typed(value.length);
  for (let i = 0; i < value.length; i++) out[i] = Number(value[i]);
  return out;
}

/**
 * A compiled `Fn` as a function of a context: `compileJSRoutine` and
 * `compileWasmRoutine` give one. It reads its parameters and uniforms from
 * the context, and returns the value of its program, typed by the type the
 * program returns. A program that reads what only a stage has is not a
 * routine: a vertex, fragment or compute program compiles as a stage, and a
 * program of `fragCoord()` as a grid.
 */
export type CpuRoutine<A extends ShaderType = ShaderType, W extends FloatWidth = "f64"> = (
  ctx: CpuShaderContext,
) => CpuValue<A, W>;

/**
 * A program as the compilers build it, for the stages and the grid to take
 * what each of them gives from: `run` returns the value, or the
 * {@link CpuProgramResult} of a program that writes outputs, a position or a
 * depth, or `null` for a discarded fragment. Not public: a routine, a stage and
 * a grid each give a part of it.
 */
export type CpuProgram = {
  run(ctx: CpuShaderContext): CpuValue<ShaderType> | CpuProgramResult | null;
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
  options: { name: string; params: Array<{ name: string; type: ShaderType }>; float?: FloatWidth },
) => CpuRoutine<ShaderType, FloatWidth>;

/** What a vertex stage hands on: its position, and the varyings the fragment stage reads, by slot. */
export type VertexResult<W extends FloatWidth = "f64"> = {
  position: FloatArray<W>;
  varyings: Record<string, unknown>;
};

/**
 * The values an `outputStruct` writes, by position: one for each member, of
 * the JavaScript type its shader type has.
 */
export type CpuOutputs<M extends readonly Node<ShaderType>[], W extends FloatWidth = "f64"> = {
  -readonly [K in keyof M]: M[K] extends Node<infer A> ? CpuValue<A, W> : never;
};

/**
 * What a fragment stage writes for one fragment: its colour as the four
 * channels of a `vec4`, the values of its `outputStruct` by position, and the
 * depth it wrote. A stage that returns an `outputStruct` has no colour, and a
 * stage that returns a colour has no outputs. `R` is what the stage returns.
 */
export type FragmentResult<R = unknown, W extends FloatWidth = "f64"> =
  R extends OutputStruct<infer M extends readonly Node<ShaderType>[]>
    ? { value: undefined; outputs: CpuOutputs<M, W>; fragDepth?: number }
    : unknown extends R
      ? { value: FloatArray<W> | undefined; outputs: unknown[]; fragDepth?: number }
      : R extends Node<"void">
        ? { value: undefined; outputs: []; fragDepth?: number }
        : { value: FloatArray<W>; outputs: []; fragDepth?: number };

/** A compiled vertex program: runs once per vertex, and returns what the stage hands on. */
export type VertexStage<W extends FloatWidth = "f64"> = (ctx: CpuShaderContext) => VertexResult<W>;

/** A compiled fragment program: runs once per fragment, and returns `null` for a fragment that discarded. */
export type FragmentStage<R = unknown, W extends FloatWidth = "f64"> = (
  ctx: CpuShaderContext,
) => FragmentResult<R, W> | null;

/**
 * A compiled compute program: runs once per index in `0..count`, in index
 * order, with `invocationIndex()` reading that index, and leaves the results in
 * `ctx.storages`. It returns nothing.
 */
export type ComputeStage = {
  (ctx: CpuShaderContext, count: number): void;
  /** The shader type of one element of each storage buffer the program reads, by slot. */
  readonly storageTypes: Readonly<Record<string, ShaderType>>;
};

const isResultObject = (raw: unknown): raw is CpuProgramResult =>
  typeof raw === "object" && raw !== null && !Array.isArray(raw) && !ArrayBuffer.isView(raw);

/** Whether a value is a vector or matrix: an array, or the typed view a JS slot is. */
const isVector = (value: unknown): value is ArrayLike<number> => Array.isArray(value) || ArrayBuffer.isView(value);

/** The {@link VertexResult} of what a routine compiled for the vertex stage returned. */
export function toVertexResult(raw: CpuValue<ShaderType> | CpuProgramResult | null): VertexResult {
  // A vertex stage that never writes the position itself has its `vec4` result become the position.
  const wrapped = isResultObject(raw);
  const position = (wrapped ? (raw.position ?? raw.value) : raw) as VertexResult["position"] | undefined;
  if (!position) {
    throw new Error("[RMSL] A vertex stage never wrote a position, with builtinPosition() or a vec4 result.");
  }
  return { position, varyings: (wrapped && (raw.varyings as Record<string, unknown>)) || {} };
}

/** The outputs of a fragment that wrote none, which every such fragment shares: a per-pixel call allocates nothing for it. */
const NO_OUTPUTS: unknown[] = Object.freeze([]) as unknown as unknown[];

/** The values a routine wrote to its outputs, in the order of their locations. */
function outputsInOrder(outputs: Record<string, unknown> | undefined): unknown[] {
  if (outputs === undefined) return NO_OUTPUTS;
  return Object.keys(outputs)
    .sort((a, b) => Number(a.replace(/\D/g, "")) - Number(b.replace(/\D/g, "")))
    .map((slot) => outputs![slot]);
}

/** The {@link FragmentResult} of what a routine compiled for the fragment stage returned, `null` when it discarded. */
export function toFragmentResult<R>(raw: CpuValue<ShaderType> | CpuProgramResult | null): FragmentResult<R> | null {
  if (raw === null) return null;
  const result: { value: ArrayLike<number> | undefined; outputs: unknown[]; fragDepth?: number } = isResultObject(raw)
    ? { value: isVector(raw.value) ? raw.value : undefined, outputs: outputsInOrder(raw.outputs) }
    : { value: isVector(raw) ? raw : undefined, outputs: NO_OUTPUTS };
  if (isResultObject(raw) && raw.fragDepth !== undefined) result.fragDepth = raw.fragDepth;
  return result as unknown as FragmentResult<R>;
}

/**
 * The typed array a grid of `A` fills: floats and matrices in a `Float64Array`,
 * integers and booleans in an `Int32Array`, unsigned integers in a
 * `Uint32Array`, with every component of every pixel one after another.
 */
export type GridBuffer<A extends ShaderType, W extends FloatWidth = "f64"> = A extends "uint" | `uvec${string}`
  ? Uint32Array
  : A extends "int" | "bool" | `ivec${string}` | `bvec${string}`
    ? Int32Array
    : FloatArray<W>;

/**
 * A program of `fragCoord()` evaluated over a grid of pixels. It evaluates the
 * program once for each pixel of a `width x height` grid, with `fragCoord()` at
 * the centre of each pixel, and packs the results into one flat row-major
 * buffer of `width * height * componentCount` elements, typed by the result. A
 * pixel that discards is zero in every channel. Every other input, uniforms and
 * textures included, is the same for every pixel. The buffer is the caller's: a
 * later call does not change it. Pass `out` to fill a buffer of your own
 * instead; it must be the type `A` has and hold at least that many elements,
 * and it is returned.
 */
export type CpuGrid<A extends ShaderType = ShaderType, W extends FloatWidth = "f64"> = (
  ctx: CpuShaderContext,
  width: number,
  height: number,
  out?: GridBuffer<A, W>,
) => GridBuffer<A, W>;
