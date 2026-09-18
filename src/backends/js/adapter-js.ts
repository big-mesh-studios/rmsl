import { Node, ShaderType } from "../../core";
import { CpuAdapter, createCpuAdapter } from "../adapter-cpu";
import { compileJS, CompileJSOptions } from "./js";

export interface CreateJsOptions {
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

/** Compiles `compute`/`batch` with {@link compileJS} and wraps them in a {@link createCpuAdapter}. */
export function createJs(options: CreateJsOptions): CpuAdapter {
  if (!options.compute && !options.batch) {
    throw new Error("[RMSL] createJs needs a `compute` program, a `batch` program, or both");
  }

  const compute = options.compute
    ? compileJS(() => options.compute!, {
        name: options.computeName ?? "compute",
        params: options.params ?? [],
        derivatives: options.derivatives,
        reentrant: options.reentrant,
      })
    : undefined;

  const batch = options.batch
    ? compileJS(() => options.batch!, {
        name: options.batchName ?? "batch",
        params: options.params ?? [],
        stage: "fragment",
        derivatives: options.derivatives,
        reentrant: options.reentrant,
      })
    : undefined;

  return createCpuAdapter({ compute, batch });
}
