/**
 * The GLSL backend's public entry point, mirroring `wgsl.ts` — compile and
 * adapter together, kept off the main barrel since they're specific to
 * this one backend.
 */
export { compileGlsl } from "./backends/glsl";
export type { CompileGLSLOptions, GLSLPrecision } from "./backends/glsl";

export { compileGlslFn } from "./backends/glsl";

export { createGlsl } from "./backends/adapter-glsl";
export type { GlslAdapter, GlslDrawOptions } from "./backends/adapter-glsl";

export type { Adapter, TypedArray } from "./backends/adapter";
