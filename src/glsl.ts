/**
 * The GLSL backend's public entry point, mirroring `wgsl.ts` — compile and
 * adapter together, kept off the main barrel since they're specific to
 * this one backend.
 */
export { compileGlsl } from "./backends/glsl/glsl";
export type { CompileGLSLOptions, GLSLPrecision } from "./backends/glsl/glsl";

export { compileGlslFn } from "./backends/glsl/glsl";

export { createGlsl } from "./backends/glsl/adapter-glsl";
export type { GlslAdapter, GlslDrawOptions } from "./backends/glsl/adapter-glsl";

export type { Adapter, TypedArray } from "./backends/adapter";
