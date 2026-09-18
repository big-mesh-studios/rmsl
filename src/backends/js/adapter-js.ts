import { Node, ShaderType } from "../../core";
import { CpuAdapter, createCpuAdapter } from "../adapter-cpu";
import { compileJSRoutine, CompileJSOptions } from "./js";

export interface CreateJsRoutineOptions {
  /** A storage()/invocationIndex() program to run as `compute()`. */
  compute?: Node<ShaderType> | readonly Node<ShaderType>[];
  /** A fragCoord() program, evaluated once per pixel by `draw()`'s `.batch()` call. */
  batch?: Node<ShaderType> | readonly Node<ShaderType>[];
  computeName?: string;
  batchName?: string;
  params?: CompileJSOptions["params"];
  derivatives?: CompileJSOptions["derivatives"];
  reentrant?: CompileJSOptions["reentrant"];
}

/** Compiles `compute`/`batch` with {@link compileJSRoutine} and wraps them in a {@link createCpuAdapter}. */
export function createJsRoutine(options: CreateJsRoutineOptions): CpuAdapter {
  if (!options.compute && !options.batch) {
    throw new Error("[RMSL] createJsRoutine needs a `compute` program, a `batch` program, or both");
  }

  const compute = options.compute
    ? compileJSRoutine(() => options.compute!, {
        name: options.computeName ?? "compute",
        params: options.params ?? [],
        derivatives: options.derivatives,
        reentrant: options.reentrant,
      })
    : undefined;

  const batch = options.batch
    ? compileJSRoutine(() => options.batch!, {
        name: options.batchName ?? "batch",
        params: options.params ?? [],
        stage: "fragment",
        derivatives: options.derivatives,
        reentrant: options.reentrant,
      })
    : undefined;

  return createCpuAdapter({ compute, batch });
}
