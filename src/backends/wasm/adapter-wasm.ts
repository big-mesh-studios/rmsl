import { AttributeNode, Node, ShaderType, UniformArrayNode, UniformNode, UniformValue } from "../../core";
import { Adapter, AttributeOrSlot, DrawCountOptions, slotOf, TypedArray, UniformOrSlot } from "../adapter";
import { bufferToImageData, CpuAdapter, createCpuAdapter } from "../adapter-cpu";
import { compileWasm, CompileWasmOptions, WasmRasterContext } from "./rasterizer";
import { compileWasmRoutine, CompileWasmFnOptions } from "./wasm";

export interface CreateWasmRoutineOptions {
  /** A storage()/invocationIndex() program to run as `compute()`. */
  compute?: Node<ShaderType> | readonly Node<ShaderType>[];
  /** A fragCoord() program, evaluated once per pixel by `draw()`'s `.batch()` call. */
  batch?: Node<ShaderType> | readonly Node<ShaderType>[];
  computeName?: string;
  batchName?: string;
  params?: CompileWasmFnOptions["params"];
  derivatives?: CompileWasmFnOptions["derivatives"];
  reentrant?: CompileWasmFnOptions["reentrant"];
  memory?: CompileWasmFnOptions["memory"];
  sharedMemory?: CompileWasmFnOptions["sharedMemory"];
  maxMemoryPages?: CompileWasmFnOptions["maxMemoryPages"];
  gpuUniformLayout?: CompileWasmFnOptions["gpuUniformLayout"];
}

/** Compiles `compute`/`batch` with {@link compileWasmRoutine} and wraps them in a {@link createCpuAdapter}. */
export function createWasmRoutine(options: CreateWasmRoutineOptions): CpuAdapter {
  if (!options.compute && !options.batch) {
    throw new Error("[RMSL] createWasmRoutine needs a `compute` program, a `batch` program, or both");
  }

  const shared = {
    params: options.params ?? [],
    derivatives: options.derivatives,
    reentrant: options.reentrant,
    memory: options.memory,
    sharedMemory: options.sharedMemory,
    maxMemoryPages: options.maxMemoryPages,
    gpuUniformLayout: options.gpuUniformLayout,
  };

  const compute = options.compute
    ? compileWasmRoutine(() => options.compute!, { name: options.computeName ?? "compute", ...shared })
    : undefined;

  const batch = options.batch
    ? compileWasmRoutine(() => options.batch!, { name: options.batchName ?? "batch", stage: "fragment", ...shared })
    : undefined;

  return createCpuAdapter({ compute, batch });
}

/**
 * `draw()`'s own options — `count`/`first` from `DrawCountOptions`, same
 * as GL/WGSL: unset `count` defaults to whatever the widest `setAttribute`
 * call implied. `clear`/`clearDepth` default to `true`: the common case
 * for a single-material adapter is one `draw()` call, one whole frame —
 * mirroring how a WebGPU render pass declares `loadOp`/`depthLoadOp`
 * together, per pass, rather than clearing as a separate operation. Pass
 * `false` to composite several `draw()` calls into one frame instead
 * (several materials sharing a framebuffer/depth buffer); `compileWasm`'s
 * own `WasmRasterRoutine` is the lower-level primitive that composability
 * is built on.
 */
export interface WasmDrawOptions extends DrawCountOptions {
  clear?: boolean;
  clearDepth?: boolean;
}

/**
 * Compiles a vertex/fragment `Fn` pair with {@link compileWasm} and wraps
 * the resulting `WasmRasterRoutine` in the uniform `Adapter` interface —
 * `setAttribute`/`setUniform` collect draw state, `attach` opens a 2D
 * canvas context, and `draw()` runs one frame into it. See
 * {@link WasmDrawOptions} for `count`/`first`/`clear`/`clearDepth`.
 */
export interface WasmAdapter extends Adapter<void, WasmDrawOptions> {
  attach(canvas?: HTMLCanvasElement): void;
  draw(options?: WasmDrawOptions): void;
}

export function createWasm(
  vertexFn: (...args: any[]) => any,
  fragmentFn: (...args: any[]) => any,
  options: CompileWasmOptions = {},
): WasmAdapter {
  const routine = compileWasm(vertexFn, fragmentFn, options);

  const uniforms: Record<string, number | number[]> = {};
  const attributes: Record<string, TypedArray> = {};
  let canvas: HTMLCanvasElement | null = null;
  let ctx2d: CanvasRenderingContext2D | null = null;

  function setUniform<T extends ShaderType>(uniform: UniformNode<T>, value: UniformValue<T>): void;
  function setUniform<T extends ShaderType>(uniform: UniformArrayNode<T>, value: UniformValue<T>[]): void;
  function setUniform(slot: string, value: number | number[]): void;
  function setUniform(uniform: UniformOrSlot, value: unknown): void {
    uniforms[slotOf(uniform)] = value as number | number[];
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
      if (!canvas || !ctx2d) throw new Error("[RMSL] createWasm: attach() was never called");
      const ctx: WasmRasterContext = { attributes, uniforms };
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
