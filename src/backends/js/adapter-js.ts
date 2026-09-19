import { AttributeNode, Node, ShaderType, UniformArrayNode, UniformNode, UniformValue } from "../../core";
import { Adapter, AttributeOrSlot, DrawCountOptions, slotOf, TypedArray, UniformOrSlot } from "../adapter";
import { AdapterResult, bufferToImageData, CpuAdapter, createCpuAdapter } from "../adapter-cpu";
import { compileJS, CompileJSRasterOptions, JsRasterContext } from "./rasterizer";
import { compileJSRoutine, CompileJSOptions } from "./js";

export interface CreateJsRoutineOptions {
  /** A fragCoord() program, evaluated once per pixel by `draw()`'s `.batch()` call. */
  batch: Node<ShaderType> | readonly Node<ShaderType>[];
  name?: string;
  params?: CompileJSOptions["params"];
  derivatives?: CompileJSOptions["derivatives"];
  reentrant?: CompileJSOptions["reentrant"];
}

/**
 * Compiles a `fragCoord()` program with {@link compileJSRoutine} and
 * wraps it in a {@link createCpuAdapter} — a plain CPU-callable evaluated
 * once per pixel/sample via `.batch()`, not a wgpu pipeline shape. See
 * {@link createJsCompute} for the `storage()`/`invocationIndex()` shape
 * and {@link createJs} for the vertex/fragment render shape — those each
 * got their own dedicated entry point rather than living as options here
 * for the same reason (see `src/backends/wasm/adapter-wasm.ts`'s own
 * split, which this mirrors).
 */
export function createJsRoutine(options: CreateJsRoutineOptions): CpuAdapter {
  const batch = compileJSRoutine(() => options.batch, {
    name: options.name ?? "batch",
    params: options.params ?? [],
    stage: "fragment",
    derivatives: options.derivatives,
    reentrant: options.reentrant,
  });

  return createCpuAdapter({ batch });
}

export interface CreateJsComputeOptions {
  name?: string;
  params?: CompileJSOptions["params"];
  derivatives?: CompileJSOptions["derivatives"];
  reentrant?: CompileJSOptions["reentrant"];
}

/**
 * `setAttribute`/`setUniform` collect draw state the way {@link CpuAdapter}
 * does; `compute()` is the only invocation method, since a `storage()`/
 * `invocationIndex()` program has no `draw()`/`attach()` counterpart —
 * unlike `CpuAdapter`, that's not just unused here, it's not part of the
 * type at all.
 */
export interface JsComputeAdapter {
  setUniform<T extends ShaderType>(uniform: UniformNode<T>, value: UniformValue<T>): void;
  setUniform<T extends ShaderType>(uniform: UniformArrayNode<T>, value: UniformValue<T>[]): void;
  setUniform(slot: string, value: number | number[]): void;
  setAttribute<T extends ShaderType>(attribute: AttributeNode<T>, data: TypedArray): void;
  setAttribute(slot: string, data: TypedArray): void;
  compute(out?: AdapterResult): AdapterResult | void;
}

/**
 * Compiles a `storage()`/`invocationIndex()` program with
 * {@link compileJSRoutine} and wraps it in a {@link createCpuAdapter} —
 * the wgpu-compute-pipeline-shaped counterpart to {@link createJs}'s
 * render-pipeline shape, and the JS-side sibling of `createWasmCompute`.
 */
export function createJsCompute(
  compute: Node<ShaderType> | readonly Node<ShaderType>[],
  options: CreateJsComputeOptions = {},
): JsComputeAdapter {
  const computeRoutine = compileJSRoutine(() => compute, {
    name: options.name ?? "compute",
    params: options.params ?? [],
    derivatives: options.derivatives,
    reentrant: options.reentrant,
  });

  return createCpuAdapter({ compute: computeRoutine });
}

/**
 * `draw()`'s own options — `count`/`first` from `DrawCountOptions`, same
 * as GL/WGSL: unset `count` defaults to whatever the widest `setAttribute`
 * call implied. `clear`/`clearDepth` default to `true`: the common case
 * for a single-material adapter is one `draw()` call, one whole frame —
 * mirroring how a WebGPU render pass declares `loadOp`/`depthLoadOp`
 * together, per pass, rather than clearing as a separate operation. Pass
 * `false` to composite several `draw()` calls into one frame instead
 * (several materials sharing a framebuffer/depth buffer); `compileJS`'s
 * own `JsRasterRoutine` is the lower-level primitive that composability
 * is built on.
 */
export interface JsDrawOptions extends DrawCountOptions {
  clear?: boolean;
  clearDepth?: boolean;
}

/**
 * Compiles a vertex/fragment `Fn` pair with {@link compileJS} and wraps
 * the resulting `JsRasterRoutine` in the uniform `Adapter` interface — the
 * JS-side counterpart to `createWasm`. `setAttribute`/`setUniform` collect
 * draw state, `attach` opens a 2D canvas context, and `draw()` runs one
 * frame into it. See {@link JsDrawOptions} for `count`/`first`/`clear`/`clearDepth`.
 */
export interface JsAdapter extends Adapter<void, JsDrawOptions> {
  attach(canvas?: HTMLCanvasElement): void;
  draw(options?: JsDrawOptions): void;
}

export function createJs(
  vertexFn: (...args: any[]) => any,
  fragmentFn: (...args: any[]) => any,
  options: CompileJSRasterOptions,
): JsAdapter {
  const routine = compileJS(vertexFn, fragmentFn, options);

  const uniforms: Record<string, unknown> = {};
  const attributes: Record<string, TypedArray> = {};
  let canvas: HTMLCanvasElement | null = null;
  let ctx2d: CanvasRenderingContext2D | null = null;

  function setUniform<T extends ShaderType>(uniform: UniformNode<T>, value: UniformValue<T>): void;
  function setUniform<T extends ShaderType>(uniform: UniformArrayNode<T>, value: UniformValue<T>[]): void;
  function setUniform(slot: string, value: number | number[]): void;
  function setUniform(uniform: UniformOrSlot, value: unknown): void {
    uniforms[slotOf(uniform)] = value;
  }

  function setAttribute<T extends ShaderType>(attribute: AttributeNode<T>, data: TypedArray): void;
  function setAttribute(slot: string, data: TypedArray): void;
  function setAttribute(attribute: AttributeOrSlot, data: TypedArray): void {
    attributes[slotOf(attribute)] = data;
  }

  return {
    attach(givenCanvas) {
      canvas = givenCanvas ?? document.createElement("canvas");
      const context = canvas.getContext("2d");
      if (!context) throw new Error("[RMSL] 2D canvas context unavailable");
      ctx2d = context;
    },

    setUniform,
    setAttribute,

    draw(drawOptions) {
      if (!canvas || !ctx2d) throw new Error("[RMSL] createJs: attach() was never called");
      const ctx: JsRasterContext = { attributes, uniforms };
      const buffer = routine.draw(ctx, {
        count: drawOptions?.count,
        first: drawOptions?.first,
        width: canvas.width,
        height: canvas.height,
        clear: drawOptions?.clear ?? true,
        clearDepth: drawOptions?.clearDepth ?? true,
      });
      ctx2d.putImageData(bufferToImageData(buffer, canvas.width, canvas.height), 0, 0);
    },

    destroy() {},
  };
}
