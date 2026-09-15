/**
 * The JS (CPU) backend's public entry point, mirroring `wgsl.ts` — compile
 * and adapter together, kept off the main barrel since they're specific to
 * this one backend. The `Cpu*` context types are shared with the WASM
 * backend (both compile to the same callable shape — see adapter-cpu.ts),
 * so `wasm.ts` re-exports them too rather than making one backend the
 * "real" home for something both need.
 */
export { compileJS, compileJSFn } from "./backends/js";
export type { CompileJSOptions } from "./backends/js";

export type {
  CpuDrawBuffer,
  CpuRenderer,
  CpuShaderContext,
  CpuShaderResult,
  CpuTextureData,
  CpuTextureWrap,
} from "./backends/cpu";

export { createJs } from "./backends/adapter-js";
export type { CreateJsOptions } from "./backends/adapter-js";
export type { AdapterResult as CpuAdapterResult, CpuAdapter } from "./backends/adapter-cpu";

export type { Adapter, TypedArray } from "./backends/adapter";
