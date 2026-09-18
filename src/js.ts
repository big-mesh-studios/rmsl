export { compileJS, compileJSFn } from "./backends/js/js";
export type { CompileJSOptions } from "./backends/js/js";

export type {
  CpuDrawBuffer,
  CpuRoutine,
  CpuShaderContext,
  CpuShaderResult,
  CpuTextureData,
  CpuTextureWrap,
} from "./backends/cpu";

export type { CpuAdapter, AdapterResult as CpuAdapterResult } from "./backends/adapter-cpu";
export { createJs } from "./backends/js/adapter-js";
export type { CreateJsOptions } from "./backends/js/adapter-js";

export type { Adapter, TypedArray } from "./backends/adapter";

export { rasterizeTriangles } from "./backends/cpu-rasterizer";
export type { RasterizeTrianglesOptions } from "./backends/cpu-rasterizer";
