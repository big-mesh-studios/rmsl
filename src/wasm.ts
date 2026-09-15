/**
 * The WASM (CPU) backend's public entry point, mirroring `wgsl.ts` — compile
 * and adapter together, kept off the main barrel since they're specific to
 * this one backend. `createCpu` and the `Cpu*` context types are shared
 * with the JS backend (both compile to the same callable shape — see
 * adapter-cpu.ts); see `js.ts` for the same re-export.
 */
export { compileWasm, compileWasmFn, instantiateWasm } from "./backends/wasm";
export type { CompiledWasm, WasmParam } from "./backends/wasm";

export type {
  CpuDrawBuffer,
  CpuRenderer,
  CpuShaderContext,
  CpuShaderResult,
  CpuTextureData,
  CpuTextureWrap,
} from "./backends/cpu";

export { createCpu } from "./backends/adapter-cpu";
export type { AdapterResult as CpuAdapterResult } from "./backends/adapter-cpu";

export type { Adapter, TypedArray } from "./backends/adapter";
