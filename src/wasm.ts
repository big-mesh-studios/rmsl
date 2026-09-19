export { compileWasmRoutine, compileWasmFn, instantiateWasmRoutine } from "./backends/wasm/wasm";
export type { CompiledWasm, WasmParam } from "./backends/wasm/wasm";

export type {
  CpuDrawBuffer,
  CpuRoutine,
  CpuShaderContext,
  CpuShaderResult,
  CpuTextureData,
  CpuTextureWrap,
} from "./backends/cpu";

export { createWasmRoutine, createWasm } from "./backends/wasm/adapter-wasm";
export type { CreateWasmRoutineOptions, WasmAdapter, WasmDrawOptions } from "./backends/wasm/adapter-wasm";
export type { AdapterResult as CpuAdapterResult, CpuAdapter } from "./backends/adapter-cpu";

export type { Adapter, TypedArray } from "./backends/adapter";

export { rasterizeTriangles } from "./backends/js/rasterizer";
export type { RasterizeTrianglesOptions } from "./backends/js/rasterizer";

export { compileWasm } from "./backends/wasm/rasterizer";
export type {
  CompileWasmOptions,
  WasmRasterContext,
  WasmRasterDrawOptions,
  WasmRasterRoutine,
} from "./backends/wasm/rasterizer";
