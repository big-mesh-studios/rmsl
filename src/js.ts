export { compileJSRoutine, compileJSFn, compileJSVertex, compileJSFragment, compileJSCompute } from "./backends/js/js";
export type { CompileJSOptions, CompileJSStageOptions, JsVertexStage, JsFragmentStage } from "./backends/js/js";

export type {
  ComputeStage,
  CpuDrawBuffer,
  CpuRoutine,
  CpuShaderContext,
  CpuShaderResult,
  CpuTextureData,
  CpuTextureWrap,
  FragmentResult,
  FragmentStage,
  VertexResult,
  VertexStage,
} from "./backends/cpu";

export type { CpuAdapter, AdapterResult as CpuAdapterResult } from "./backends/adapter-cpu";
export { createJsRoutine, createJsCompute, createJs } from "./backends/js/adapter-js";
export type {
  CreateJsRoutineOptions,
  CreateJsComputeOptions,
  JsComputeAdapter,
  JsAdapter,
  JsDrawOptions,
} from "./backends/js/adapter-js";

export type { Adapter, TypedArray } from "./backends/adapter";

export { compileJS } from "./backends/js/rasterizer";
export type {
  CompileJSRasterOptions,
  JsRasterContext,
  JsRasterDrawOptions,
  JsRasterRoutine,
} from "./backends/js/rasterizer";
