const words = (list: string) => list.split(/\s+/).filter(Boolean);

/**
 * The keywords and the words kept for later of GLSL ES 3.00 and WGSL, which
 * name nothing a program declares, a member of a struct included.
 */
export const KEYWORDS: ReadonlySet<string> = new Set([
  ...words(`attribute const uniform varying break continue do for while switch case default if else in out inout
    float int void bool true false invariant discard return mat2 mat3 mat4 mat2x2 mat2x3 mat2x4 mat3x2 mat3x3 mat3x4
    mat4x2 mat4x3 mat4x4 vec2 vec3 vec4 ivec2 ivec3 ivec4 bvec2 bvec3 bvec4 uint uvec2 uvec3 uvec4 lowp mediump highp
    precision sampler2D sampler3D samplerCube sampler2DShadow samplerCubeShadow sampler2DArray sampler2DArrayShadow
    isampler2D isampler3D isamplerCube isampler2DArray usampler2D usampler3D usamplerCube usampler2DArray struct
    layout centroid flat smooth common partition active asm class union enum typedef template this resource goto
    inline noinline public static extern external interface long short double half fixed unsigned superp input
    output hvec2 hvec3 hvec4 dvec2 dvec3 dvec4 fvec2 fvec3 fvec4 sampler1D sampler1DShadow sampler1DArray
    sampler2DRect sampler2DRectShadow sizeof cast namespace using filter`),
  ...words(`alias case const const_assert continue continuing default diagnostic discard else enable false fn for if
    let loop override requires return struct switch true var while self`),
]);

/**
 * The words WGSL reserves for later, which name nothing a WGSL module declares,
 * a member of a struct included.
 */
export const WGSL_RESERVED_WORDS: ReadonlySet<string> = new Set(
  words(`NULL Self abstract active alignas alignof as asm asm_fragment async attribute auto await become
    binding_array cast catch class co_await co_return co_yield coherent column_major common compile
    compile_fragment concept const_cast consteval constexpr constinit crate debugger decltype delete demote
    demote_to_helper do dynamic_cast enum explicit export extends extern external fallthrough filter final
    finally friend from fxgroup get goto groupshared highp impl implements import inline instanceof interface
    layout lowp macro macro_rules match mediump meta mod module move mut mutable namespace new nil noexcept
    noinline nointerpolation noperspective null nullptr of operator package packoffset partition pass patch
    pixelfragment precise precision premerge priv protected pub public readonly ref regardless register
    reinterpret_cast require resource restrict self set shared sizeof smooth snorm static static_assert
    static_cast std subroutine super target template this thread_local throw trait try type typedef typeid
    typename typeof union unless unorm unsafe unsized use using varying virtual volatile wgsl where with
    writeonly yield`),
);

/**
 * The names a function cannot be declared under in GLSL ES 3.00 or WGSL: the
 * {@link KEYWORDS} and the built-in functions and types of either language. One
 * list serves both, so a program names its functions alike on every target.
 */
export const RESERVED_FUNCTION_NAMES: ReadonlySet<string> = new Set([
  ...KEYWORDS,
  "main",
  ...words(`radians degrees sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh pow exp log exp2 log2 sqrt
    inversesqrt abs sign floor trunc round roundEven ceil fract mod modf min max clamp mix step smoothstep isnan
    isinf floatBitsToInt floatBitsToUint intBitsToFloat uintBitsToFloat length distance dot cross normalize
    faceforward reflect refract matrixCompMult outerProduct transpose determinant inverse lessThan lessThanEqual
    greaterThan greaterThanEqual equal notEqual any all not texture textureProj textureLod textureOffset
    textureProjLod texelFetch textureSize textureGrad dFdx dFdy fwidth packSnorm2x16 unpackSnorm2x16 packUnorm2x16
    unpackUnorm2x16 packHalf2x16 unpackHalf2x16`),
  ...words(`alias case const const_assert continue continuing default diagnostic discard else enable false fn for if
    let loop override requires return struct switch true var while f32 f16 i32 u32 array ptr atomic sampler
    sampler_comparison texture_2d texture_3d texture_cube texture_storage_2d texture_depth_2d acosh asinh atanh
    arrayLength atan2 faceForward fma frexp inverseSqrt ldexp quantizeToF16 reverseBits saturate select
    textureDimensions textureLoad textureNumLevels textureSample textureSampleLevel textureStore workgroupBarrier
    storageBarrier atomicLoad atomicStore atomicAdd extractBits insertBits countOneBits countLeadingZeros
    countTrailingZeros firstLeadingBit firstTrailingBit pack4x8snorm unpack4x8snorm fwidthCoarse fwidthFine
    dpdx dpdy self`),
]);

/**
 * Whether `name` cannot be declared in GLSL ES 3.00 or WGSL, as a function or a
 * variable: a word of {@link RESERVED_FUNCTION_NAMES} or {@link WGSL_RESERVED_WORDS}, a name GLSL keeps for
 * itself with the prefix `gl_` or two underscores in a row, or a name WGSL keeps
 * with two leading underscores.
 */
export const isReservedName = (name: string): boolean =>
  RESERVED_FUNCTION_NAMES.has(name) || WGSL_RESERVED_WORDS.has(name) || name.startsWith("gl_") || name.includes("__");

/**
 * The name a program declares in place of `name` when GLSL or WGSL cannot
 * declare `name`: a name is a hint the compiler may change. It is `name` unless
 * that is {@link isReservedName}, and then `name` with every run of underscores
 * made one, a `gl_` prefix made `_gl_`, and a keyword or built-in followed by an
 * underscore. Two names can map to one, so the caller numbers the clashes.
 */
export function safeName(name: string): string {
  if (!isReservedName(name)) return name;
  let safe = name.replace(/_{2,}/g, "_");
  if (safe.startsWith("gl_")) safe = `_${safe}`;
  return RESERVED_FUNCTION_NAMES.has(safe) || WGSL_RESERVED_WORDS.has(safe) ? `${safe}_` : safe;
}
