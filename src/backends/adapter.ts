// === Backend adapter ===
// A uniform way to drive any of the four backends (CPU/JS, WASM, GLSL,
// WGSL) without homogenizing what makes them different: JS/WASM compute
// synchronously into a caller-supplied buffer, WGSL's compute pass is async,
// and drawing needs a canvas while computing doesn't. `compute`/`draw` are
// optional so a backend only implements what it actually supports —
// callers feature-detect the same way they'd check `navigator.gpu`.
export interface Adapter<TBuffer> {
  /**
   * One-time setup (device/context/pipeline creation) for `draw`. Creates
   * its own offscreen canvas when none is given, so a draw-capable adapter
   * still runs headless (tests, benches, backend comparisons) without the
   * caller wiring up DOM first.
   */
  attach(canvas?: HTMLCanvasElement): void | Promise<void>;

  setUniform(slot: string, value: number | number[]): void;
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

  /** Renders into whatever `attach` set up. */
  draw?: () => void | Promise<void>;

  destroy(): void;
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
