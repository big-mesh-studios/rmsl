import { AttributeNode, ShaderType, UniformArrayNode, UniformNode, UniformValue } from "../core";

// === Backend adapter ===
// A uniform way to drive any of the four backends (CPU/JS, WASM, GLSL,
// WGSL) without homogenizing what makes them different: JS/WASM compute
// synchronously into a caller-supplied buffer, WGSL's compute pass is async,
// and drawing needs a canvas while computing doesn't. `compute`/`draw` are
// optional so a backend only implements what it actually supports —
// callers feature-detect the same way they'd check `navigator.gpu`.
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
   * With `out`, writes the result into it and returns it, so chaining into
   * the next adapter's `setAttribute` needs no extra variable. Without it,
   * just runs — for a backend that keeps its result GPU-resident (a WGSL
   * compute pass writing storage buffers a `draw` reads directly), forcing a
   * readback into `out` on every call would be the one thing this interface
   * isn't supposed to do: take away a backend's own advantage to look uniform.
   */
  compute?: (out?: TBuffer) => TBuffer | void | Promise<TBuffer | void>;

  /**
   * Renders into whatever `attach` set up. `TDrawOptions` is each
   * backend's own — a GL adapter's draw call (mode, instancing, indexed
   * vs. array draws) has nothing in common with a WGSL render pipeline's,
   * so there is no shared options shape to force either one into; a
   * backend without a meaningful options shape leaves it `void`.
   */
  draw?: (options?: TDrawOptions) => void | Promise<void>;

  destroy(): void;
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

export function slotOf(uniformOrAttribute: UniformOrSlot | AttributeOrSlot): string {
  return typeof uniformOrAttribute === "string" ? uniformOrAttribute : uniformOrAttribute.name;
}

export type TypedArray =
  | Float32Array
  | Float64Array
  | Int32Array
  | Uint32Array
  | Int16Array
  | Uint16Array
  | Int8Array
  | Uint8Array;
