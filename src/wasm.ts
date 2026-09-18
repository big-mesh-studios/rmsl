export { compileWasm, compileWasmFn, instantiateWasm } from "./backends/wasm/wasm";
export type { CompiledWasm, WasmParam } from "./backends/wasm/wasm";

export type {
  CpuDrawBuffer,
  CpuRoutine,
  CpuShaderContext,
  CpuShaderResult,
  CpuTextureData,
  CpuTextureWrap,
} from "./backends/cpu";

export { createWasm } from "./backends/wasm/adapter-wasm";
export type { CreateWasmOptions } from "./backends/wasm/adapter-wasm";
export type { AdapterResult as CpuAdapterResult, CpuAdapter } from "./backends/adapter-cpu";

export type { Adapter, TypedArray } from "./backends/adapter";

export { rasterizeTriangles } from "./backends/cpu-rasterizer";
export type { RasterizeTrianglesOptions } from "./backends/cpu-rasterizer";
