// === JS adapter ===
// Wraps compileJS + the shared CPU adapter loop (adapter-cpu.ts) so this
// takes root graphs and compiles them internally, the same contract
// createGlsl/createWgsl have — not already-compiled callables, which is
// what the generic loop underneath actually needs but no other adapter
// constructor asks a caller for. Options mirror createWgsl's shape for
// the same reason: `compute` and `draw` are two different root graphs (a
// storage()/invocationIndex() program vs. a fragCoord() one), so a
// caller may give either or both.
import { Node, ShaderType } from "../../core";
import { CpuAdapter, createCpuAdapter } from "../adapter-cpu";
import { compileJS, CompileJSOptions } from "./js";

export interface CreateJsOptions {
  /** A storage()/invocationIndex() program to run as `compute()`. */
  compute?: Node<ShaderType> | readonly Node<ShaderType>[];
  /** A fragCoord() program, evaluated once per pixel by `draw()`. */
  draw?: Node<ShaderType> | readonly Node<ShaderType>[];
  computeName?: string;
  drawName?: string;
  params?: CompileJSOptions["params"];
  derivatives?: CompileJSOptions["derivatives"];
  reentrant?: CompileJSOptions["reentrant"];
}

export function createJs(options: CreateJsOptions): CpuAdapter {
  if (!options.compute && !options.draw) {
    throw new Error("[RMSL] createJs needs a `compute` program, a `draw` program, or both");
  }

  const compute = options.compute
    ? compileJS(() => options.compute!, {
        name: options.computeName ?? "compute",
        params: options.params ?? [],
        derivatives: options.derivatives,
        reentrant: options.reentrant,
      })
    : undefined;

  const draw = options.draw
    ? compileJS(() => options.draw!, {
        name: options.drawName ?? "draw",
        params: options.params ?? [],
        stage: "fragment",
        derivatives: options.derivatives,
        reentrant: options.reentrant,
      })
    : undefined;

  return createCpuAdapter({ compute, draw });
}
