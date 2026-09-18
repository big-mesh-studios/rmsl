import { AttributeNode, ShaderType, UniformArrayNode, UniformNode, UniformValue } from "../core";

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
 * `count` left unset defaults to whatever the widest `setAttribute` call
 * implied — every backend that draws infers it that way.
 */
export interface DrawCountOptions {
  /** First vertex to draw. Defaults to 0. */
  first?: number;
  /** Vertices to draw. Defaults to everything the widest `setAttribute` call implied. */
  count?: number;
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
  Float32Array | Float64Array | Int32Array | Uint32Array | Int16Array | Uint16Array | Int8Array | Uint8Array;
