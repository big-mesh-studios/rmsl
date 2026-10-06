import { AttributeNode, ShaderType, UniformArrayNode, UniformNode, UniformValue } from "../core";
import type { CpuTextureData } from "./cpu";

/**
 * A uniform way to drive any of the four backends (CPU/JS, WASM, GLSL,
 * WGSL) without homogenizing what makes them different: JS/WASM compute
 * synchronously into a caller-supplied buffer, WGSL's compute pass is
 * async, and drawing needs a canvas while computing doesn't. `compute` is
 * optional — GL genuinely has no compute path, so `createGlsl`'s returned
 * object has no `compute` property at all, not even a throwing one —
 * callers feature-detect it the same way they'd check `navigator.gpu`.
 * `draw`, by contrast, is required: every adapter this interface actually
 * has (GL, WGSL, CPU/JS/WASM) always returns one, throwing at call time
 * only if that particular construction wasn't given anything to draw.
 */
export interface Adapter<TBuffer, TDrawOptions = void> {
  /**
   * One-time setup (device/context/pipeline creation) for `draw`. Creates
   * its own offscreen canvas when none is given, so a draw-capable adapter
   * still runs headless (tests, benches, backend comparisons) without the
   * caller wiring up DOM first.
   */
  attach(canvas?: HTMLCanvasElement): void | Promise<void>;

  /**
   * Passing the `uniform()`/`uniformArray()` node itself (rather than its
   * slot name) lets `value`'s type be inferred from the node's own
   * `ShaderType` instead of widened to `number | number[]`. The plain-string
   * overload stays for `uniformRaw()` slots or callers that only have a
   * name on hand — there the value shape is on the caller to get right.
   */
  setUniform<T extends ShaderType>(uniform: UniformNode<T>, value: UniformValue<T>): void;
  setUniform<T extends ShaderType>(uniform: UniformArrayNode<T>, value: UniformValue<T>[]): void;
  setUniform(slot: string, value: number | number[]): void;

  /**
   * Same reasoning as `setUniform`: passing the `attribute()` node lets the
   * slot name come from the node instead of being retyped by hand. Attribute
   * data is always a flat `TypedArray` regardless of `ShaderType`, so unlike
   * `setUniform` this doesn't narrow `data`'s type — it only removes the
   * chance of a slot-name typo.
   */
  setAttribute<T extends ShaderType>(attribute: AttributeNode<T>, data: TypedArray): void;
  setAttribute(slot: string, data: TypedArray): void;

  /**
   * Gives a sampler uniform the texture it reads. Naming the sampler by its
   * node or by its slot works as it does for `setUniform`. The texture is
   * described the way a CPU target samples it (`CpuTextureData`), and a GPU
   * target uploads it to a texture of its own. An adapter that has no
   * texture to give, such as `createWgslContext`, leaves this out.
   */
  setTexture?(sampler: UniformNode<ShaderType> | string, texture: CpuTextureData): void;

  /**
   * With `out`, writes the result into it and returns it, so chaining into
   * the next adapter's `setAttribute` needs no extra variable. Without it,
   * just runs — for a backend that keeps its result GPU-resident (a WGSL
   * compute pass writing storage buffers a `draw` reads directly), forcing a
   * readback into `out` on every call would be the one thing this interface
   * isn't supposed to do: take away a backend's own advantage to look uniform.
   *
   * `count` is the number of invocations to dispatch. Given none, an adapter
   * runs one invocation per element of the first storage buffer the host
   * passed — the count TSL's caller writes beside `instancedArray(count,
   * type)`. A GPU target rounds up to whole workgroups, so its last
   * workgroup also runs the invocations past the count; the program's own
   * bounds check is what skips them.
   */
  compute?: (out?: TBuffer, count?: number) => TBuffer | void | Promise<TBuffer | void>;

  /**
   * Renders into whatever `attach` set up. `TDrawOptions` is each
   * backend's own beyond {@link DrawCountOptions} — a GL adapter's own
   * `mode`, a GPU adapter's `instanceCount`, have nothing in common with
   * each other, so there's no shared shape to force those into; a
   * backend without a meaningful options shape leaves it `void`.
   */
  draw(options?: TDrawOptions): void | Promise<void>;

  destroy(): void;
}

/**
 * How many vertices a draw call covers, and where to start — the one
 * piece every draw-capable adapter's own `TDrawOptions` actually shares
 * (`GlslDrawOptions`, `WgslDrawOptions`, `WasmDrawOptions`,
 * `JsDrawOptions` all extend this), so each backend's own `count`/`first`
 * mean the same thing instead of just happening to be spelled the same.
 * `count` left unset defaults to the first attribute the host
 * passed, less `first` — every backend that draws infers it that way.
 */
export interface DrawCountOptions {
  /** First vertex to draw. Defaults to 0. */
  first?: number;
  /** Vertices to draw. Defaults to the first attribute the host passed, less `first`. */
  count?: number;
}

/** Red, green, blue and alpha, each from 0 to 1. */
export type ClearColor = readonly [number, number, number, number];

/** What a draw clears a target to when it gives no `clearColor`: transparent black. */
export const TRANSPARENT_BLACK: ClearColor = [0, 0, 0, 0];

/**
 * How a draw treats the colour of its target, the one piece every
 * draw-capable adapter's own `TDrawOptions` shares besides
 * {@link DrawCountOptions}. Clearing is on by default, as `autoClear` is in
 * three.js; an application that composes several draws into one frame passes
 * `clear: false` to the later ones.
 */
export interface DrawClearOptions {
  /** Whether to clear the colour of the target before drawing. Defaults to `true`. */
  clear?: boolean;
  /** The colour a clear fills the target with. Defaults to transparent black. */
  clearColor?: ClearColor;
}

/**
 * What `setUniform`'s first parameter is once its overloads collapse into
 * one implementation signature: a uniform (array) node, or a raw slot name.
 * Each adapter implementation takes this instead of `any`, and derives the
 * slot via `slotOf` below.
 */
export type UniformOrSlot = UniformNode<ShaderType> | UniformArrayNode<ShaderType> | string;

/** Same idea as `UniformOrSlot`, for `setAttribute`. */
export type AttributeOrSlot = AttributeNode<ShaderType> | string;

/**
 * The storage slots a `compute(out)` call asks to have read back: the keys of
 * `out`. A key that names no storage slot throws, where skipping it would
 * leave a misspelled slot's array unwritten without a word.
 */
export function requestedStorageSlots(out: object, slots: Iterable<string>): string[] {
  let known = new Set(slots);
  let requested = Object.keys(out);
  let unknown = requested.filter((slot) => !known.has(slot));
  if (unknown.length > 0) {
    throw new Error(
      `[RMSL] compute(out) was given ${unknown.map((s) => `"${s}"`).join(", ")}, which the program has no storage slot for. ` +
        `Its storage slots are: ${[...known].map((s) => `"${s}"`).join(", ") || "none"}.`,
    );
  }
  return requested;
}

export function slotOf(uniformOrAttribute: UniformOrSlot | AttributeOrSlot): string {
  return typeof uniformOrAttribute === "string" ? uniformOrAttribute : uniformOrAttribute.name;
}

export type TypedArray =
  Float32Array | Float64Array | Int32Array | Uint32Array | Int16Array | Uint16Array | Int8Array | Uint8Array;
