export { compileJSRoutine, compileJSFn } from "./backends/js/js";
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
export { createJsRoutine, createJs } from "./backends/js/adapter-js";
export type { CreateJsRoutineOptions, JsAdapter, JsDrawOptions } from "./backends/js/adapter-js";

export type { Adapter, TypedArray } from "./backends/adapter";

export { compileJS, rasterizeTriangles } from "./backends/js/rasterizer";
export type {
  CompileJSRasterOptions,
  JsRasterContext,
  RasterizeTrianglesOptions,
  JsRasterDrawOptions,
  JsRasterRoutine,
} from "./backends/js/rasterizer";
