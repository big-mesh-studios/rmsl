import { AttributeNode, Node, ShaderType, UniformArrayNode, UniformNode, UniformValue } from "../../core";
import { Adapter, AttributeOrSlot, slotOf, TypedArray, UniformOrSlot } from "../adapter";
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

/** `draw()`'s vertex count — a real rasterizer draw call, unlike `CreateWasmRoutineOptions`'s
 * per-pixel `batch` program, actually runs over geometry, so it needs one. */
export interface WasmDrawOptions {
  /** Vertex count for this draw — a non-indexed triangle list, so a multiple of 3. */
  vertexCount: number;
}

/**
 * Compiles a vertex/fragment `Fn` pair with {@link compileWasm} and wraps
 * the resulting `WasmRasterRoutine` in the uniform `Adapter` interface —
 * `setAttribute`/`setUniform` collect draw state, `attach` opens a 2D
 * canvas context, and `draw({ vertexCount })` runs one frame into it.
 *
 * Each `draw()` call auto-clears both the depth buffer and the output
 * buffer first, treating one call as one whole frame — the common case
 * for a single-material adapter. A caller composing several materials
 * into one shared framebuffer/depth buffer needs `compileWasm`'s own
 * `WasmRasterRoutine` directly, where both clears are explicit instead
 * (`clearDepth()`, and `draw()`'s own `clear` argument).
 */
export function createWasm(
  vertexFn: (...args: any[]) => any,
  fragmentFn: (...args: any[]) => any,
  options: CompileWasmOptions = {},
): Adapter<void, WasmDrawOptions> {
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
      if (!drawOptions) throw new Error("[RMSL] createWasm: draw() needs a { vertexCount }");
      routine.clearDepth();
      const ctx: WasmRasterContext = { attributes, uniforms };
      const buffer = routine.draw(ctx, drawOptions.vertexCount, canvas.width, canvas.height, undefined, true);
      ctx2d.putImageData(bufferToImageData(buffer, canvas.width, canvas.height), 0, 0);
    },

    destroy() {},
  };
}
