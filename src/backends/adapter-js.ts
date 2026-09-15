// === JS adapter ===
// Wraps compileJS + the shared CPU adapter loop (adapter-cpu.ts) so this
// takes a root graph and compiles it internally, the same contract
// createGlsl/createWgsl have — not an already-compiled callable, which is
// what the generic loop underneath actually needs but no other adapter
// constructor asks a caller for.
import { Node, ShaderType } from "../core";
import { CpuAdapter, createCpuAdapter } from "./adapter-cpu";
import { compileJS, CompileJSOptions } from "./js";

export type CreateJsOptions = Partial<Omit<CompileJSOptions, "name" | "params">> &
  Pick<Partial<CompileJSOptions>, "name" | "params">;

export function createJs(
  root: Node<ShaderType> | readonly Node<ShaderType>[],
  options: CreateJsOptions = {},
): CpuAdapter {
  const step = compileJS(() => root, {
    name: options.name ?? "main",
    params: options.params ?? [],
    stage: options.stage,
    derivatives: options.derivatives,
    reentrant: options.reentrant,
  });
  return createCpuAdapter(step);
}
