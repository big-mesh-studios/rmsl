// === WASM adapter ===
// Wraps compileWasm + the shared CPU adapter loop (adapter-cpu.ts); see
// adapter-js.ts's comment — the same reasoning applies here, `compute`
// and `draw` are two different root graphs a caller may give either or
// both of.
import { Node, ShaderType } from "../core";
import { CpuAdapter, createCpuAdapter } from "./adapter-cpu";
import { compileWasm, CompileWasmFnOptions } from "./wasm";

export interface CreateWasmOptions {
  /** A storage()/invocationIndex() program to run as `compute()`. */
  compute?: Node<ShaderType> | readonly Node<ShaderType>[];
  /** A fragCoord() program, evaluated once per pixel by `draw()`. */
  draw?: Node<ShaderType> | readonly Node<ShaderType>[];
  computeName?: string;
  drawName?: string;
  params?: CompileWasmFnOptions["params"];
  derivatives?: CompileWasmFnOptions["derivatives"];
  reentrant?: CompileWasmFnOptions["reentrant"];
  memory?: CompileWasmFnOptions["memory"];
  sharedMemory?: CompileWasmFnOptions["sharedMemory"];
  maxMemoryPages?: CompileWasmFnOptions["maxMemoryPages"];
  gpuUniformLayout?: CompileWasmFnOptions["gpuUniformLayout"];
}

export function createWasm(options: CreateWasmOptions): CpuAdapter {
  if (!options.compute && !options.draw) {
    throw new Error("[RMSL] createWasm needs a `compute` program, a `draw` program, or both");
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

  const draw = options.draw
    ? compileWasm(() => options.draw!, { name: options.drawName ?? "draw", stage: "fragment", ...shared })
    : undefined;

  return createCpuAdapter({ compute, draw });
}
