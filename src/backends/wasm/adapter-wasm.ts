import { Node, ShaderType } from "../../core";
import { CpuAdapter, createCpuAdapter } from "../adapter-cpu";
import { compileWasm, CompileWasmFnOptions } from "./wasm";

export interface CreateWasmOptions {
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

/** Compiles `compute`/`batch` with {@link compileWasm} and wraps them in a {@link createCpuAdapter}. */
export function createWasm(options: CreateWasmOptions): CpuAdapter {
  if (!options.compute && !options.batch) {
    throw new Error("[RMSL] createWasm needs a `compute` program, a `batch` program, or both");
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
    ? compileWasm(() => options.compute!, { name: options.computeName ?? "compute", ...shared })
    : undefined;

  const batch = options.batch
    ? compileWasm(() => options.batch!, { name: options.batchName ?? "batch", stage: "fragment", ...shared })
    : undefined;

  return createCpuAdapter({ compute, batch });
}
