// === WASM adapter ===
// Wraps compileWasm + the shared CPU adapter loop (adapter-cpu.ts) so this
// takes a root graph and compiles it internally, the same contract
// createGlsl/createWgsl have — not an already-compiled callable, which is
// what the generic loop underneath actually needs but no other adapter
// constructor asks a caller for.
import { Node, ShaderType } from "../core";
import { Adapter } from "./adapter";
import { AdapterResult, createCpuAdapter } from "./adapter-cpu";
import { compileWasm, CompileWasmFnOptions } from "./wasm";

export type CreateWasmOptions = Partial<Omit<CompileWasmFnOptions, "name" | "params">> &
  Pick<Partial<CompileWasmFnOptions>, "name" | "params">;

export function createWasm(
  root: Node<ShaderType> | readonly Node<ShaderType>[],
  options: CreateWasmOptions = {},
): Adapter<AdapterResult> {
  const step = compileWasm(() => root, {
    name: options.name ?? "main",
    params: options.params ?? [],
    stage: options.stage,
    derivatives: options.derivatives,
    reentrant: options.reentrant,
    memory: options.memory,
    sharedMemory: options.sharedMemory,
    maxMemoryPages: options.maxMemoryPages,
    gpuUniformLayout: options.gpuUniformLayout,
  });
  return createCpuAdapter(step);
}
