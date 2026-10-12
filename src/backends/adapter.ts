import {
  AttributeNode,
  ReadOnlyStorageNode,
  ShaderType,
  StorageNode,
  UniformArrayNode,
  UniformNode,
  UniformValue,
  StorageBufferAttribute,
} from "../core";
import type { CpuTextureData } from "./cpu";
import { assertHostArray } from "./vertex-format";

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
export interface Adapter<TBuffer, TDrawOptions = void, TFloatArray extends Float32Array | Float64Array = Float32Array> {
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
   * Same reasoning as `setUniform`: passing the `attribute()` or storage node
   * lets the slot name come from the node instead of being retyped by hand,
   * and narrows `data` to the one array the node's type takes, which the
   * adapter refuses any other array for (see {@link HostArray}). A slot name
   * takes any flat `TypedArray`, checked once the adapter knows its type.
   */
  setAttribute<T extends ShaderType>(attribute: AttributeOrStorage<T>, data: HostArray<T, TFloatArray>): void;
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
 * `count` left unset defaults to the vertices of the first attribute the
 * program reads, in the order the host passed them, less `first` — every
 * backend that draws infers it that way.
 */
export interface DrawCountOptions {
  /** First vertex to draw. Defaults to 0. */
  first?: number;
  /** Vertices to draw. Defaults to those of the first attribute the program reads, in the order the host passed them, less `first`. */
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

/**
 * The array an attribute of type `T`, or a storage buffer of elements of type
 * `T`, takes: an `Int32Array` for an `int` or `ivecN`, a `Uint32Array` for a
 * `uint` or `uvecN`, and `TFloatArray` for any other, a `Float32Array` unless
 * the adapter also takes a `Float64Array`.
 */
export type HostArray<T extends ShaderType, TFloatArray extends Float32Array | Float64Array = Float32Array> = T extends
  "int" | "ivec2" | "ivec3" | "ivec4"
  ? Int32Array
  : T extends "uint" | "uvec2" | "uvec3" | "uvec4"
    ? Uint32Array
    : TFloatArray;

/** A node whose slot `setAttribute` fills: an attribute, or a storage buffer. */
export type AttributeOrStorage<T extends ShaderType> = AttributeNode<T> | StorageNode<T> | ReadOnlyStorageNode<T>;

/** Same idea as `UniformOrSlot`, for `setAttribute`. */
export type AttributeOrSlot = AttributeOrStorage<ShaderType> | string;

/**
 * Refuses `data` for a node `setAttribute` was given unless it is the array
 * of the node's kind: an attribute's declared type, or a storage buffer's
 * element type. A slot name passes here, to be checked where the adapter
 * learns its type. `float64` admits a `Float64Array` for float components.
 */
export function assertNodeArray(attribute: AttributeOrSlot, data: ArrayLike<number>, float64 = false): void {
  if (typeof attribute === "string") return;
  if ("attribute" in attribute && attribute.attribute instanceof StorageBufferAttribute) {
    const type = attribute.attribute.elementType;
    if (type) assertHostArray("storage buffer", attribute.name, type, data, float64);
    return;
  }
  const type = attributeTypeOf(attribute);
  if (type !== undefined) assertHostArray("attribute", attribute.name, type, data, float64);
}

/** The declared type of an attribute node, or nothing for a storage node or a slot name. */
export function attributeTypeOf(attribute: AttributeOrSlot): string | undefined {
  if (typeof attribute === "string") return undefined;
  const node = attribute as { type?: string; _t?: string };
  return node.type === "attribute" ? node._t : undefined;
}

/**
 * The storage slots a `compute(out)` call asks to have read back: the keys of
 * `out`. A key that names no storage slot throws, where skipping it would
 * leave a misspelled slot's array unwritten without a word.
 */
export function requestedStorageSlots(out: object, slots: Iterable<string>): string[] {
  let known = new Set(slots);
  let requested = Object.keys(out);
  if (requested.some((slot) => !known.has(slot))) throw unknownStorageSlots(out, known);
  return requested;
}

/** The error for a `compute(out)` whose keys name a slot outside `slots`, naming each such key. */
export function unknownStorageSlots(out: object, slots: Iterable<string>): Error {
  // The set requestedStorageSlots has already made is used as it is.
  let known = slots instanceof Set ? (slots as Set<string>) : new Set(slots);
  let unknown = Object.keys(out).filter((slot) => !known.has(slot));
  return new Error(
    `[RMSL] compute(out) was given ${unknown.map((s) => `"${s}"`).join(", ")}, which the program has no storage slot for. ` +
      `Its storage slots are: ${[...known].map((s) => `"${s}"`).join(", ") || "none"}.`,
  );
}

export function slotOf(uniformOrAttribute: UniformOrSlot | AttributeOrSlot): string {
  return typeof uniformOrAttribute === "string" ? uniformOrAttribute : uniformOrAttribute.name;
}

export type TypedArray =
  Float32Array | Float64Array | Int32Array | Uint32Array | Int16Array | Uint16Array | Int8Array | Uint8Array;
