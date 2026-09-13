/**
 * RMSL's public entry point. A thin barrel: everything the package exports
 * lives in one of the files below, split by concern (the DSL core, the
 * compiler internals every backend shares, and the four backends
 * themselves). Kept as an explicit named re-export list, not `export *`,
 * so a helper one file needs from another doesn't silently become public
 * API just because it had to be `export`ed to cross a file boundary.
 *
 * Types and values are re-exported separately (`export type { }` vs
 * `export { }`): a bundler doing real cross-module analysis (Rollup, for
 * `pnpm build`) errors on re-exporting a name that has no runtime binding,
 * which every `interface`/`type` here is.
 */

export type {
  ShaderType, FloatLike, Vec2Like, Vec3Like, Vec4Like, IntLike, UintLike, BooleanLike,
  IVec2Like, IVec3Like, IVec4Like, UVec2Like, UVec3Like, UVec4Like, Mat3Like, Mat4Like,
  Sampler2DLike, Sampler3DLike, ISampler2DLike, USampler3DLike,
  BaseNode, VariableNode, UniformNode, UniformArrayNode, AttributeNode, VaryingNode,
} from "./rmsl-core";
export {
  isUniformNode, isAttributeNode, isVaryingNode,
  Node, var_, assertBlockScope, Fn,
  float, vec2, vec3, vec4, int, uint, ivec2, ivec3, ivec4, uvec2, uvec3, uvec4, bool,
  mat2, mat2x3, mat2x4, mat3x2, mat3, mat3x4, mat4x2, mat4x3, mat4, bvec2, bvec3, bvec4,
  add, sub, mul, div, mod, equal, notEqual, lessThan, greaterThan, lessThanEqual, greaterThanEqual,
  and, or, xor, not, bitAnd, bitOr, bitXor, bitNot, shiftLeft, shiftRight,
  abs, sign, floor, ceil, fract, round, trunc, radians, degrees, sqrt, inverseSqrt, inversesqrt,
  exp, log, exp2, log2, negate, oneMinus, reciprocal, cbrt, saturate, lengthSq, normalize,
  dFdx, dFdy, fwidth, sin, cos, tan, asin, acos, sinh, cosh, tanh, asinh, acosh, atanh, atan,
  pow, pow2, pow3, pow4, min, max, step, reflect, refract, faceForward, difference,
  dot, cross, distance, length, mix, clamp, smoothstep, all, any, transpose, determinant,
  inverse, element, select, luminance, rand, interleavedGradientNoise,
  premultiplyAlpha, unpremultiplyAlpha, textureLoad, textureSize,
  PI, TWO_PI, PI2, HALF_PI, EPSILON, INFINITY,
  uniformArray, uniform, uniformRaw, time, attribute, attributeRaw, varying, varyingRaw,
  output, builtinPosition, builtinFragDepth, fragCoord, screenCoordinate, screenSize, screenUV, uv,
  If, For, Loop, While, Switch, Discard, Break, Continue, Return,
} from "./rmsl-core";

export type { VertexRoot, CompileFnOptions } from "./rmsl-compiler-shared";

export type { GLSLPrecision, CompileGLSLOptions } from "./rmsl-glsl";
export { compileGLSL } from "./rmsl-glsl";

export type { WgslUniformMember, WgslUniformDeclaration, CompileWGSLOptions } from "./rmsl-wgsl";
export { wgslUniformLayout, compileWGSL } from "./rmsl-wgsl";

export type { JsShaderContext, JsTextureWrap, JsTextureData, JsShaderResult, CompileJSOptions } from "./rmsl-compile-js";
export { compileJSFn, compileJS } from "./rmsl-compile-js";

export { compileGLSLFn, compileWGSLFn } from "./rmsl-standalone-fn";

export type { WasmParam, CompiledWasm } from "./rmsl-wasm";
export { compileWasmFn, compileWasm } from "./rmsl-wasm";
