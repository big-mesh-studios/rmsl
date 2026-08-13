const __brand = Symbol();

// === Shader Types (strings, not TS enums) ===
export type ShaderType =
  | "float" | "vec2" | "vec3" | "vec4"
  | "int" | "uint" | "bool"
  | "ivec2" | "ivec3" | "ivec4"
  | "uvec2" | "uvec3" | "uvec4"
  | "bvec2" | "bvec3" | "bvec4"
  | "mat2" | "mat2x3" | "mat2x4"
  | "mat3x2" | "mat3" | "mat3x4"
  | "mat4x2" | "mat4x3" | "mat4"
  | "sampler2D" | "sampler3D" | "samplerCube"
  | "isampler2D" | "isampler3D" | "isamplerCube"
  | "usampler2D" | "usampler3D" | "usamplerCube"
  | "void";

// === Like types (raw JS values | Node) ===
export type FloatLike = number | BaseNode<"float">;
export type Vec2Like = [number, number] | BaseNode<"vec2">;
export type Vec3Like = [number, number, number] | BaseNode<"vec3">;
export type Vec4Like = [number, number, number, number] | BaseNode<"vec4">;
export type IntLike = number | BaseNode<"int">;
export type UintLike = number | BaseNode<"uint">;
export type BooleanLike = boolean | BaseNode<"bool">;
export type IVec2Like = [number, number] | BaseNode<"ivec2">;
export type IVec3Like = [number, number, number] | BaseNode<"ivec3">;
export type IVec4Like = [number, number, number, number] | BaseNode<"ivec4">;
export type UVec2Like = [number, number] | BaseNode<"uvec2">;
export type UVec3Like = [number, number, number] | BaseNode<"uvec3">;
export type UVec4Like = [number, number, number, number] | BaseNode<"uvec4">;
export type Mat3Like = number[] | BaseNode<"mat3">;
export type Mat4Like = number[] | BaseNode<"mat4">;
export type Sampler2DLike = BaseNode<"sampler2D"> | Node<"sampler2D">;
export type Sampler3DLike = BaseNode<"sampler3D"> | Node<"sampler3D">;
export type ISampler2DLike = BaseNode<"isampler2D"> | Node<"isampler2D">;
export type USampler3DLike = BaseNode<"usampler3D"> | Node<"usampler3D">;

// === BaseNode ===
export interface BaseNode<A extends ShaderType> {
  [__brand]: A;
  _t: string;
  type: string;
  params?: BaseNode<ShaderType>[];
  value?: unknown;
}

// === Typed node types with variable name access ===
/**
 * A uniform, attribute or varying. Carries its type's operations directly, so
 * it can be used wherever a `Node<A>` can.
 */
export type VariableNode<A extends ShaderType> = Node<A> & {
  name: string;
};

// Aliases rather than interfaces: `Node<A>` resolves through an indexed access,
// and an interface may only extend a type whose members are statically known.
export type UniformNode<A extends ShaderType> = VariableNode<A>;

/**
 * A uniform array. Not a `Node<A>` itself — the array as a whole has no value,
 * only its elements do, so it exposes `element()` rather than the operations of
 * its element type.
 */
export interface UniformArrayNode<A extends ShaderType> {
  readonly name: string;
  readonly length: number;
  element(index: IntLike | FloatLike): Node<A>;
}
export type AttributeNode<A extends ShaderType> = VariableNode<A>;
export type VaryingNode<A extends ShaderType> = VariableNode<A>;

// === Type guards for node type checking ===
export function isUniformNode<T extends ShaderType>(node: Node<T> | VariableNode<T>): node is UniformNode<T> {
  return node.type === "uniform" && "name" in node;
}

export function isAttributeNode<T extends ShaderType>(node: Node<T> | VariableNode<T>): node is AttributeNode<T> {
  return node.type === "attribute" && "name" in node;
}

export function isVaryingNode<T extends ShaderType>(node: Node<T> | VariableNode<T>): node is VaryingNode<T> {
  return node.type === "varying" && "name" in node;
}

// === Per-type swizzle sets ===
type Vec3Swizzles = {
  readonly x: Node<"float">; readonly y: Node<"float">; readonly z: Node<"float">;
  readonly r: Node<"float">; readonly g: Node<"float">; readonly b: Node<"float">;
  readonly xy: Node<"vec2">; readonly xz: Node<"vec2">; readonly yz: Node<"vec2">;
  readonly xyz: Node<"vec3">; readonly rgb: Node<"vec3">;
};

type Vec4Swizzles = {
  readonly x: Node<"float">; readonly y: Node<"float">; readonly z: Node<"float">; readonly w: Node<"float">;
  readonly r: Node<"float">; readonly g: Node<"float">; readonly b: Node<"float">; readonly a: Node<"float">;
  readonly xy: Node<"vec2">; readonly xz: Node<"vec2">; readonly xw: Node<"vec2">;
  readonly yz: Node<"vec2">; readonly yw: Node<"vec2">; readonly zw: Node<"vec2">;
  readonly xyz: Node<"vec3">; readonly xyw: Node<"vec3">; readonly xzw: Node<"vec3">; readonly yzw: Node<"vec3">;
  readonly rgba: Node<"vec4">; readonly rgb: Node<"vec3">;
};

type Vec2Swizzles = {
  readonly x: Node<"float">; readonly y: Node<"float">;
  readonly r: Node<"float">; readonly g: Node<"float">;
  readonly xy: Node<"vec2">;
};

type IVec2Swizzles = {
  readonly x: Node<"int">; readonly y: Node<"int">;
  readonly r: Node<"int">; readonly g: Node<"int">;
  readonly xy: Node<"ivec2">;
};

type UVec2Swizzles = {
  readonly x: Node<"uint">; readonly y: Node<"uint">;
  readonly r: Node<"uint">; readonly g: Node<"uint">;
  readonly xy: Node<"uvec2">;
};

type IVec3Swizzles = {
  readonly x: Node<"int">; readonly y: Node<"int">; readonly z: Node<"int">;
  readonly r: Node<"int">; readonly g: Node<"int">; readonly b: Node<"int">;
  readonly xy: Node<"ivec2">; readonly xz: Node<"ivec2">; readonly yz: Node<"ivec2">;
  readonly xyz: Node<"ivec3">; readonly rgb: Node<"ivec3">;
};

type UVec3Swizzles = {
  readonly x: Node<"uint">; readonly y: Node<"uint">; readonly z: Node<"uint">;
  readonly r: Node<"uint">; readonly g: Node<"uint">; readonly b: Node<"uint">;
  readonly xy: Node<"uvec2">; readonly xz: Node<"uvec2">; readonly yz: Node<"uvec2">;
  readonly xyz: Node<"uvec3">; readonly rgb: Node<"uvec3">;
};

type IVec4Swizzles = {
  readonly x: Node<"int">; readonly y: Node<"int">; readonly z: Node<"int">; readonly w: Node<"int">;
  readonly r: Node<"int">; readonly g: Node<"int">; readonly b: Node<"int">; readonly a: Node<"int">;
  readonly xy: Node<"ivec2">; readonly xz: Node<"ivec2">; readonly xw: Node<"ivec2">;
  readonly yz: Node<"ivec2">; readonly yw: Node<"ivec2">; readonly zw: Node<"ivec2">;
  readonly xyz: Node<"ivec3">; readonly xyw: Node<"ivec3">; readonly xzw: Node<"ivec3">; readonly yzw: Node<"ivec3">;
  readonly rgba: Node<"ivec4">; readonly rgb: Node<"ivec3">;
};

type UVec4Swizzles = {
  readonly x: Node<"uint">; readonly y: Node<"uint">; readonly z: Node<"uint">; readonly w: Node<"uint">;
  readonly r: Node<"uint">; readonly g: Node<"uint">; readonly b: Node<"uint">; readonly a: Node<"uint">;
  readonly xy: Node<"uvec2">; readonly xz: Node<"uvec2">; readonly xw: Node<"uvec2">;
  readonly yz: Node<"uvec2">; readonly yw: Node<"uvec2">; readonly zw: Node<"uvec2">;
  readonly xyz: Node<"uvec3">; readonly xyw: Node<"uvec3">; readonly xzw: Node<"uvec3">; readonly yzw: Node<"uvec3">;
  readonly rgba: Node<"uvec4">; readonly rgb: Node<"uvec3">;
};

// === Node (branded + conditional methods + swizzles) ===
/**
 * Which operations each shader type carries.
 *
 * A registry rather than a chain of conditionals. Defunctionalising the
 * dispatch into a lookup provides a single indexed access, and the interface's
 * members stay lazy.
 *
 * Every ShaderType needs an entry, so a new type cannot be added without
 * saying what it supports.
 */
interface NodeOps {
  float: ArithOps<"float"> & FloatMathOps<"float"> & ComparisonOps<"bool", FloatLike>;
  vec2: ArithOps<"vec2"> & FloatMathOps<"vec2"> & ComparisonOps<"bvec2", Vec2Like | FloatLike> & VecCommonOps<"vec2"> & Vec2Swizzles;
  vec3: ArithOps<"vec3"> & FloatMathOps<"vec3"> & ComparisonOps<"bvec3", Vec3Like | FloatLike> & VecCommonOps<"vec3"> & Vec3Ops & Vec3Swizzles;
  vec4: ArithOps<"vec4"> & FloatMathOps<"vec4"> & ComparisonOps<"bvec4", Vec4Like | FloatLike> & VecCommonOps<"vec4"> & Vec4Swizzles;
  int: IntOps;
  uint: UintOps;
  bool: BoolOps;
  ivec2: IVecOps<"ivec2"> & ComparisonOps<"bvec2", IVec2Like | IntLike> & IVec2Swizzles;
  ivec3: IVecOps<"ivec3"> & ComparisonOps<"bvec3", IVec3Like | IntLike> & IVec3Swizzles;
  ivec4: IVecOps<"ivec4"> & ComparisonOps<"bvec4", IVec4Like | IntLike> & IVec4Swizzles;
  uvec2: UVecOps<"uvec2"> & ComparisonOps<"bvec2", UVec2Like | UintLike> & UVec2Swizzles;
  uvec3: UVecOps<"uvec3"> & ComparisonOps<"bvec3", UVec3Like | UintLike> & UVec3Swizzles;
  uvec4: UVecOps<"uvec4"> & ComparisonOps<"bvec4", UVec4Like | UintLike> & UVec4Swizzles;
  bvec2: BoolVecOps<"bvec2">;
  bvec3: BoolVecOps<"bvec3">;
  bvec4: BoolVecOps<"bvec4">;
  mat2: MatOps<"mat2", "vec2">;
  mat2x3: RectMatOps<"vec2", "vec3", "mat3x2">;
  mat2x4: RectMatOps<"vec2", "vec4", "mat4x2">;
  mat3x2: RectMatOps<"vec3", "vec2", "mat2x3">;
  mat3: MatOps<"mat3", "vec3">;
  mat3x4: RectMatOps<"vec3", "vec4", "mat4x3">;
  mat4x2: RectMatOps<"vec4", "vec2", "mat2x4">;
  mat4x3: RectMatOps<"vec4", "vec3", "mat3x4">;
  mat4: MatOps<"mat4", "vec4">;
  sampler2D: SamplerOps;
  sampler3D: Sampler3DOps;
  samplerCube: CubeSamplerOps;
  isampler2D: ISampler2DOps;
  isampler3D: ISampler3DOps;
  isamplerCube: ISamplerCubeOps;
  usampler2D: USampler2DOps;
  usampler3D: USampler3DOps;
  usamplerCube: USamplerCubeOps;
  void: {};
}

export type Node<A extends ShaderType> = BaseNode<A> & NodeOps[A] & NodeMethods<A>;

// === Operation interfaces (shared across Node types) ===
interface ArithOps<A extends ShaderType> {
  add(other: FloatLike | Vec2Like | Vec3Like | Vec4Like): Node<A>;
  sub(other: FloatLike | Vec2Like | Vec3Like | Vec4Like): Node<A>;
  mult(other: FloatLike | Vec2Like | Vec3Like | Vec4Like): Node<A>;
  div(other: FloatLike | Vec2Like | Vec3Like | Vec4Like): Node<A>;
  negate(): Node<A>;
}

interface FloatMathOps<A extends ShaderType> {
  sin(): Node<A>; cos(): Node<A>; tan(): Node<A>;
  asin(): Node<A>; acos(): Node<A>; atan(): Node<A>;
  sinh(): Node<A>; cosh(): Node<A>; tanh(): Node<A>;
  asinh(): Node<A>; acosh(): Node<A>; atanh(): Node<A>;
  abs(): Node<A>; sign(): Node<A>;
  floor(): Node<A>; ceil(): Node<A>; fract(): Node<A>;
  round(): Node<A>; trunc(): Node<A>;
  radians(): Node<A>; degrees(): Node<A>;
  sqrt(): Node<A>; inversesqrt(): Node<A>;
  exp(): Node<A>; log(): Node<A>; exp2(): Node<A>; log2(): Node<A>;
  cbrt(): Node<A>;
  reciprocal(): Node<A>;
  oneMinus(): Node<A>;
  difference(other: Node<A> | FloatLike): Node<A>;
  lengthSq(): Node<A>;
  saturate(): Node<A>;
  pow(e: FloatLike): Node<A>;
  pow2(): Node<A>; pow3(): Node<A>; pow4(): Node<A>;
  min(other: FloatLike): Node<A>;
  max(other: FloatLike): Node<A>;
  mod(other: FloatLike): Node<A>;
  mix(b: Node<A>, t: FloatLike): Node<A>;
  clamp(min: FloatLike, max: FloatLike): Node<A>;
  // Declared here rather than on VecCommonOps so floats get them too, and so
  // there is only one declaration: both interfaces apply to the vector types,
  // and two declarations disagreeing about the return type leaves the caller
  // with whichever the checker resolves first.
  //
  // Both edge forms are valid — GLSL has step(genType, genType) alongside
  // step(float, genType), and likewise for smoothstep.
  step(edge: Node<A> | FloatLike): Node<A>;
  smoothstep(edge0: Node<A> | FloatLike, edge1: Node<A> | FloatLike): Node<A>;
  fwidth(): Node<A>;
  // Derivative functions. Meaningful in a fragment stage on both backends;
  // GLSL names them dFdx/dFdy and WGSL dpdx/dpdy.
  dFdx(): Node<A>;
  dFdy(): Node<A>;
}

/**
 * Component-wise comparisons, parameterised by their result type and by what
 * they accept.
 *
 * The operand is a parameter because the result width follows the wider of the
 * two sides. A vector may be compared against a scalar — the scalar is
 * broadcast, which is what the caller means — but a scalar compared against a
 * vector would produce a boolean per component while the receiver's row here
 * promises a single `bool`, so the two disagreed. Naming the operand per type
 * makes that combination a type error rather than a node whose runtime type
 * contradicts its declared one.
 *
 * Attached per concrete node type below rather than derived with a conditional:
 * resolving `Node<Conditional<A>>` for a generic `A` forces the checker to
 * expand the whole `Node` intersection at every call site, which exhausts its
 * heap.
 */
interface ComparisonOps<R extends ShaderType, Operand> {
  lessThan(other: Operand): Node<R>;
  greaterThan(other: Operand): Node<R>;
  lessThanEqual(other: Operand): Node<R>;
  greaterThanEqual(other: Operand): Node<R>;
  equal(other: Operand): Node<R>;
  notEqual(other: Operand): Node<R>;
}

interface VecCommonOps<A extends "vec2" | "vec3" | "vec4"> {
  dot(other: Node<A>): Node<"float">;
  length(): Node<"float">;
  normalize(): Node<A>;
  distance(other: Node<A>): Node<"float">;
  reflect(normal: Node<A>): Node<A>;
  refract(normal: Node<A>, eta: FloatLike): Node<A>;
  clamp(min: Node<A> | FloatLike, max: Node<A> | FloatLike): Node<A>;
  mix(b: Node<A>, t: FloatLike): Node<A>;
  element(i: IntLike): Node<"float">;
  // step/smoothstep live on FloatMathOps, which also applies to every vector
  // type.
}

interface Vec3Ops {
  cross(other: Node<"vec3">): Node<"vec3">;
}

/**
 * A square matrix. `Vec` is the vector of its own width: what one of its
 * columns is, and what multiplying it by a vector both takes and gives.
 */
interface MatOps<Self extends ShaderType, Vec extends ShaderType> {
  mult(other: Node<Self>): Node<Self>;
  mult(other: Node<Vec>): Node<Vec>;
  multVec(other: Node<"vec3">): Node<"vec3">;
  multVec4(other: Node<"vec4">): Node<"vec4">;
  element(i: IntLike): Node<Vec>;
  inverse(): Node<Self>;
  transpose(): Node<Self>;
  determinant(): Node<"float">;
}

/**
 * A matrix that is not square. A matCxR holds C columns of R rows, so it
 * multiplies a vecC to give a vecR, one of its columns is a vecR, and
 * transposing it gives a matRxC.
 *
 * There is deliberately no inverse: only a square matrix has one, and neither
 * target language offers the overload.
 *
 * Written out per type rather than derived from `Self` with conditionals. A
 * conditional that resolves to a `Node` makes the checker expand the whole
 * intersection at every use, which is what exhausted its heap before.
 */
interface RectMatOps<
  Operand extends ShaderType,
  Column extends ShaderType,
  Transposed extends ShaderType,
> {
  mult(other: Node<Operand>): Node<Column>;
  element(i: IntLike): Node<Column>;
  transpose(): Node<Transposed>;
}

/** A cube map is sampled with a direction rather than a surface coordinate. */
interface CubeSamplerOps {
  texture(coords: Vec3Like): Node<"vec4">;
  textureLod(coords: Vec3Like, lod: FloatLike): Node<"vec4">;
}

/** A 3D texture is sampled at its volume coordinate. */
interface Sampler3DOps {
  texture(coords: Vec3Like): Node<"vec4">;
  textureLod(coords: Vec3Like, lod: FloatLike): Node<"vec4">;
}

interface IntOps {
  add(other: IntLike): Node<"int">;
  sub(other: IntLike): Node<"int">;
  mult(other: IntLike): Node<"int">;
  div(other: IntLike): Node<"int">;
  mod(other: IntLike): Node<"int">;
  negate(): Node<"int">;
  abs(): Node<"int">;
  min(other: IntLike): Node<"int">;
  max(other: IntLike): Node<"int">;
  clamp(min: IntLike, max: IntLike): Node<"int">;
  bitAnd(other: IntLike): Node<"int">;
  bitOr(other: IntLike): Node<"int">;
  bitXor(other: IntLike): Node<"int">;
  shiftLeft(other: IntLike): Node<"int">;
  shiftRight(other: IntLike): Node<"int">;
  bitNot(): Node<"int">;
  lessThan(other: IntLike): Node<"bool">;
  greaterThan(other: IntLike): Node<"bool">;
  lessThanEqual(other: IntLike): Node<"bool">;
  greaterThanEqual(other: IntLike): Node<"bool">;
  equal(other: IntLike): Node<"bool">;
  notEqual(other: IntLike): Node<"bool">;
}

interface UintOps {
  add(other: UintLike): Node<"uint">;
  sub(other: UintLike): Node<"uint">;
  mult(other: UintLike): Node<"uint">;
  div(other: UintLike): Node<"uint">;
  mod(other: UintLike): Node<"uint">;
  min(other: UintLike): Node<"uint">;
  max(other: UintLike): Node<"uint">;
  clamp(min: UintLike, max: UintLike): Node<"uint">;
  bitAnd(other: UintLike): Node<"uint">;
  bitOr(other: UintLike): Node<"uint">;
  bitXor(other: UintLike): Node<"uint">;
  shiftLeft(other: UintLike): Node<"uint">;
  shiftRight(other: UintLike): Node<"uint">;
  bitNot(): Node<"uint">;
  lessThan(other: UintLike): Node<"bool">;
  greaterThan(other: UintLike): Node<"bool">;
  lessThanEqual(other: UintLike): Node<"bool">;
  greaterThanEqual(other: UintLike): Node<"bool">;
  equal(other: UintLike): Node<"bool">;
  notEqual(other: UintLike): Node<"bool">;
}

/**
 * Component-wise integer vector operations. The operand is broadcast alongside
 * each component, so an `ivec3` may be added to a whole number as well as to
 * another `ivec3`. Comparisons reduce to a boolean vector of the same width.
 */
interface IVecOps<A extends "ivec2" | "ivec3" | "ivec4"> {
  add(other: IntLike | IVec2Like | IVec3Like | IVec4Like): Node<A>;
  sub(other: IntLike | IVec2Like | IVec3Like | IVec4Like): Node<A>;
  mult(other: IntLike | IVec2Like | IVec3Like | IVec4Like): Node<A>;
  div(other: IntLike | IVec2Like | IVec3Like | IVec4Like): Node<A>;
  mod(other: IntLike | IVec2Like | IVec3Like | IVec4Like): Node<A>;
  negate(): Node<A>;
  abs(): Node<A>;
  min(other: IntLike | IVec2Like | IVec3Like | IVec4Like): Node<A>;
  max(other: IntLike | IVec2Like | IVec3Like | IVec4Like): Node<A>;
  clamp(min: IntLike | IVec2Like | IVec3Like | IVec4Like, max: IntLike | IVec2Like | IVec3Like | IVec4Like): Node<A>;
  bitAnd(other: IntLike | IVec2Like | IVec3Like | IVec4Like): Node<A>;
  bitOr(other: IntLike | IVec2Like | IVec3Like | IVec4Like): Node<A>;
  bitXor(other: IntLike | IVec2Like | IVec3Like | IVec4Like): Node<A>;
  shiftLeft(other: IntLike | IVec2Like | IVec3Like | IVec4Like): Node<A>;
  shiftRight(other: IntLike | IVec2Like | IVec3Like | IVec4Like): Node<A>;
  bitNot(): Node<A>;
  element(i: IntLike): Node<"int">;
}

interface UVecOps<A extends "uvec2" | "uvec3" | "uvec4"> {
  add(other: UintLike | UVec2Like | UVec3Like | UVec4Like): Node<A>;
  sub(other: UintLike | UVec2Like | UVec3Like | UVec4Like): Node<A>;
  mult(other: UintLike | UVec2Like | UVec3Like | UVec4Like): Node<A>;
  div(other: UintLike | UVec2Like | UVec3Like | UVec4Like): Node<A>;
  mod(other: UintLike | UVec2Like | UVec3Like | UVec4Like): Node<A>;
  min(other: UintLike | UVec2Like | UVec3Like | UVec4Like): Node<A>;
  max(other: UintLike | UVec2Like | UVec3Like | UVec4Like): Node<A>;
  clamp(min: UintLike | UVec2Like | UVec3Like | UVec4Like, max: UintLike | UVec2Like | UVec3Like | UVec4Like): Node<A>;
  bitAnd(other: UintLike | UVec2Like | UVec3Like | UVec4Like): Node<A>;
  bitOr(other: UintLike | UVec2Like | UVec3Like | UVec4Like): Node<A>;
  bitXor(other: UintLike | UVec2Like | UVec3Like | UVec4Like): Node<A>;
  shiftLeft(other: UintLike | UVec2Like | UVec3Like | UVec4Like): Node<A>;
  shiftRight(other: UintLike | UVec2Like | UVec3Like | UVec4Like): Node<A>;
  bitNot(): Node<A>;
  element(i: IntLike): Node<"uint">;
}

interface SamplerOps {
  texture(coords: Vec2Like): Node<"vec4">;
  textureLod(coords: Vec2Like, lod: FloatLike): Node<"vec4">;
}

/**
 * A signed or unsigned integer texture. Integer textures are not filterable in
 * either language, so `texture()`/`textureLod()` compile to an unfiltered
 * fetch (`texelFetch` in GLSL, `textureLoad` in WGSL — which there needs no
 * sampler) and return an integer vector. Coordinates are texel coordinates and
 * must be integers, matching how the underlying fetch is parameterised in both
 * languages, and the LOD is an int. Each is written out per dimension so a 2D
 * sampler cannot be given an `ivec3`/`uvec3` — a conditional would make the
 * checker expand the whole `Node` intersection at every use, exhausting its
 * heap.
 */
interface ISampler2DOps {
  texture(coords: IVec2Like): Node<"ivec4">;
  textureLod(coords: IVec2Like, lod: IntLike): Node<"ivec4">;
}

interface ISampler3DOps {
  texture(coords: IVec3Like): Node<"ivec4">;
  textureLod(coords: IVec3Like, lod: IntLike): Node<"ivec4">;
}

interface ISamplerCubeOps {
  texture(coords: IVec3Like): Node<"ivec4">;
  textureLod(coords: IVec3Like, lod: IntLike): Node<"ivec4">;
}

interface USampler2DOps {
  texture(coords: UVec2Like): Node<"uvec4">;
  textureLod(coords: UVec2Like, lod: IntLike): Node<"uvec4">;
}

interface USampler3DOps {
  texture(coords: UVec3Like): Node<"uvec4">;
  textureLod(coords: UVec3Like, lod: IntLike): Node<"uvec4">;
}

interface USamplerCubeOps {
  texture(coords: UVec3Like): Node<"uvec4">;
  textureLod(coords: UVec3Like, lod: IntLike): Node<"uvec4">;
}

interface BoolOps {
  and(other: BooleanLike): Node<"bool">;
  or(other: BooleanLike): Node<"bool">;
  not(): Node<"bool">;
  xor(other: BooleanLike): Node<"bool">;
}

/**
 * A component-wise comparison result. There is deliberately no implicit path
 * to `bool`: "is this vector less than that one" has no single answer, so the
 * reduction is spelled out with `all()` or `any()`.
 */
interface BoolVecOps<A extends ShaderType> {
  /** True when every component is true. */
  all(): Node<"bool">;
  /** True when at least one component is true. */
  any(): Node<"bool">;
  /** Negates each component. */
  not(): Node<A>;
  /** Component-wise logical xor. */
  xor(other: Node<A>): Node<A>;
}

interface NodeMethods<A extends ShaderType> {
  toVar(): Node<A>;
  assign(value: BaseNode<A> | Node<A>): void;
  // === Conversions (cast to a different type) ===
  toFloat(): Node<"float">;
  toInt(): Node<"int">;
  toUint(): Node<"uint">;
  toBool(): Node<"bool">;
  toVec2(): Node<"vec2">;
  toVec3(): Node<"vec3">;
  toVec4(): Node<"vec4">;
  toIVec2(): Node<"ivec2">;
  toIVec3(): Node<"ivec3">;
  toIVec4(): Node<"ivec4">;
  toUVec2(): Node<"uvec2">;
  toUVec3(): Node<"uvec3">;
  toUVec4(): Node<"uvec4">;
  toBVec2(): Node<"bvec2">;
  toBVec3(): Node<"bvec3">;
  toBVec4(): Node<"bvec4">;
  toMat2(): Node<"mat2">;
  toMat3(): Node<"mat3">;
  toMat4(): Node<"mat4">;
  convert<T extends ShaderType>(target: T): Node<T>;
}

// === NodeImpl - defines all methods, Node<A> hides typed subset ===
class NodeImpl<A extends ShaderType> implements BaseNode<A> {
  declare [__brand]: A;
  _t: string;
  type: string;
  params?: BaseNode<ShaderType>[];
  value?: unknown;

  constructor(config: { _t: string; type: string; params?: BaseNode<ShaderType>[]; value?: unknown }) {
    this._t = config._t;
    this.type = config.type;
    this.params = config.params;
    this.value = config.value;
  }

  // === ArithOps ===
  add(other: any): any { return op("add", this, other); }
  sub(other: any): any { return op("sub", this, other); }
  mult(other: any): any {
    // A matCxR times a vecC gives a vecR. The result type is determined by the
    // vector dimension, not the matrix type.
    let shape = MATRIX_DIMENSIONS[this._t];
    let otherType = other?._t;
    if (
      shape !== undefined && typeof otherType === "string"
      && otherType.startsWith("vec") && TYPE_WIDTH[otherType] === shape[0]
    ) {
      return node({
        _t: `vec${shape[1]}`,
        type: "matVecMul",
        params: [this as BaseNode<ShaderType>, wrapValue(other) as BaseNode<ShaderType>],
      });
    }
    return op("mult", this, other);
  }
  div(other: any): any { return op("div", this, other); }
  negate(): any { return op("negate", this); }

  // === FloatMathOps ===
  sin() { return op1("sin", this); }
  cos() { return op1("cos", this); }
  tan() { return op1("tan", this); }
  asin() { return op1("asin", this); }
  acos() { return op1("acos", this); }
  atan() { return op1("atan", this); }
  sinh() { return op1("sinh", this); }
  cosh() { return op1("cosh", this); }
  tanh() { return op1("tanh", this); }
  asinh() { return op1("asinh", this); }
  acosh() { return op1("acosh", this); }
  atanh() { return op1("atanh", this); }
  abs() { return op1("abs", this); }
  sign() { return op1("sign", this); }
  floor() { return op1("floor", this); }
  ceil() { return op1("ceil", this); }
  fract() { return op1("fract", this); }
  round() { return op1("round", this); }
  trunc() { return op1("trunc", this); }
  radians() { return op("mult", this, 0.017453292519943295); }
  degrees() { return op("mult", this, 57.29577951308232); }
  sqrt() { return op1("sqrt", this); }
  inversesqrt() { return op1("inversesqrt", this); }
  exp() { return op1("exp", this); }
  log() { return op1("log", this); }
  exp2() { return op1("exp2", this); }
  log2() { return op1("log2", this); }
  cbrt() { return op("mult", this.sign(), op("pow", this.abs(), 1.0 / 3.0)); }
  reciprocal() { return op("div", 1, this); }
  oneMinus() { return op("sub", 1, this); }
  difference(other: any) { return op1("abs", op("sub", this, other)); }
  lengthSq(): any {
    // For a vector this is the squared length, the dot of itself; for a scalar
    // it is simply its square — neither language offers a scalar `dot`.
    return (TYPE_WIDTH[this._t] ?? 1) > 1 ? op("dot", this, this) : op("mult", this, this);
  }
  saturate() { return op("clamp", this, 0, 1); }
  pow(e: any) { return op("pow", this, e); }
  pow2() { return op("mult", this, this); }
  pow3() { return op("mult", this, this, this); }
  pow4() { return op("mult", this, this, this, this); }
  min(other: any) { return op("min", this, other); }
  max(other: any) { return op("max", this, other); }
  mod(other: any) { return op("mod", this, other); }
  dFdx() { return op1("dFdx", this); }
  dFdy() { return op1("dFdy", this); }

  // === Comparison ops ===
  lessThan(other: any) { return comp("lessThan", this, other); }
  greaterThan(other: any) { return comp("greaterThan", this, other); }
  lessThanEqual(other: any) { return comp("lessThanEqual", this, other); }
  greaterThanEqual(other: any) { return comp("greaterThanEqual", this, other); }
  equal(other: any) { return comp("equal", this, other); }
  notEqual(other: any) { return comp("notEqual", this, other); }

  // === VecCommonOps ===
  dot(other: any): any { return op("dot", this, other); }
  length(): any { return op1("length", this); }
  normalize(): any { return op1("normalize", this); }
  distance(other: any): any { return op("distance", this, other); }
  reflect(normal: any): any { return op("reflect", this, normal); }
  refract(normal: any, eta: any): any { return op("refract", this, normal, eta); }
  clamp(minV: any, maxV: any): any { return op("clamp", this, minV, maxV); }
  mix(b: any, t: any): any { return op("mix", this, b, t); }
  step(edge: any): any { return op("step", edge, this); }
  smoothstep(edge0: any, edge1: any): any { return op("smoothstep", edge0, edge1, this); }
  fwidth(): any { return op1("fwidth", this); }

  // === Vec3Ops ===
  cross(other: any): any { return op("cross", this, other); }

  // === MatOps ===
  multVec(other: any): any {
    return node({ _t: "vec3", type: "matVecMul", params: [this as BaseNode<ShaderType>, wrapValue(other) as BaseNode<ShaderType>] });
  }
  multVec4(other: any): any {
    return node({ _t: "vec4", type: "matVecMul", params: [this as BaseNode<ShaderType>, wrapValue(other) as BaseNode<ShaderType>] });
  }
  // The argument is an index, so a plain number is always typed as an integer.
  // A matrix indexes to one of its columns; a vector to one of its components.
  element(i: any): any {
    let index = typeof i === "number" ? node({ _t: "int", type: "int", value: i | 0 }) : i;
    let isVector = /^(vec|ivec|uvec|bvec)[234]$/.test(this._t);
    return op(isVector ? "vectorElement" : "matrixElement", this, index);
  }
  inverse() { return op1("inverse", this); }
  transpose() { return op1("transpose", this); }
  determinant() { return op1("determinant", this); }

  // === IntOps ===
  bitAnd(other: any) { return op("bitAnd", this, other); }
  bitOr(other: any) { return op("bitOr", this, other); }
  bitXor(other: any) { return op("bitXor", this, other); }
  shiftLeft(other: any) { return op("shiftLeft", this, other); }
  shiftRight(other: any) { return op("shiftRight", this, other); }
  bitNot(): any {
    return node({
      _t: this._t,
      type: "bitNot",
      params: [this as BaseNode<ShaderType>],
    });
  }

  // === SamplerOps ===
  texture(coords: any): any {
    return node({
      _t: textureResultType(this._t),
      type: "texture",
      params: [this as BaseNode<ShaderType>, wrapValue(coords) as BaseNode<ShaderType>],
    });
  }
  textureLod(coords: any, lod: any): any {
    return node({
      _t: textureResultType(this._t),
      type: "textureLod",
      params: [this as BaseNode<ShaderType>, wrapValue(coords) as BaseNode<ShaderType>, wrapValue(lod) as BaseNode<ShaderType>],
    });
  }

  // === BoolOps ===
  and(other: any): any { return op("and", this, other); }
  or(other: any): any { return op("or", this, other); }
  not(): any { return op1("not", this); }
  xor(other: any): any {
    // Neither language has a logical xor: `(a || b) && !(a && b)` is the same
    // truth table for scalars and component-wise for boolean vectors.
    return op(
      "and",
      op("or", this, other),
      op1("not", op("and", this, other)),
    );
  }
  all(): any { return node({ _t: "bool", type: "all", params: [this as BaseNode<ShaderType>] }); }
  any(): any { return node({ _t: "bool", type: "any", params: [this as BaseNode<ShaderType>] }); }

  // === NodeMethods ===
  assign(value: BaseNode<A>): void {
    assertBlockScope("assign", (blockScope) => {
      blockScope.push(new NodeImpl({
        _t: "void",
        type: "assign",
        params: [this, value as BaseNode<ShaderType>],
      }));
    });
  }

  toVar(): Node<A> {
    let v: Node<A>;
    assertBlockScope("toVar", (blockScope) => {
      let name = `_rmsl_${nextVarId++}`;
      v = var_(name, this._t) as Node<A>;
      blockScope.push(new NodeImpl({
        _t: "void",
        type: "let",
        params: [(v as BaseNode<ShaderType>), (this as BaseNode<ShaderType>)],
      }));
    });
    return v!;
  }

  // === Conversions (cast to a different type) ===
  toFloat(): any { return node({ _t: "float", type: "construct", params: [this as BaseNode<ShaderType>] }); }
  toInt(): any { return node({ _t: "int", type: "construct", params: [this as BaseNode<ShaderType>] }); }
  toUint(): any { return node({ _t: "uint", type: "construct", params: [this as BaseNode<ShaderType>] }); }
  toBool(): any { return node({ _t: "bool", type: "construct", params: [this as BaseNode<ShaderType>] }); }
  toVec2(): any { return node({ _t: "vec2", type: "construct", params: [this as BaseNode<ShaderType>] }); }
  toVec3(): any { return node({ _t: "vec3", type: "construct", params: [this as BaseNode<ShaderType>] }); }
  toVec4(): any { return node({ _t: "vec4", type: "construct", params: [this as BaseNode<ShaderType>] }); }
  toIVec2(): any { return node({ _t: "ivec2", type: "construct", params: [this as BaseNode<ShaderType>] }); }
  toIVec3(): any { return node({ _t: "ivec3", type: "construct", params: [this as BaseNode<ShaderType>] }); }
  toIVec4(): any { return node({ _t: "ivec4", type: "construct", params: [this as BaseNode<ShaderType>] }); }
  toUVec2(): any { return node({ _t: "uvec2", type: "construct", params: [this as BaseNode<ShaderType>] }); }
  toUVec3(): any { return node({ _t: "uvec3", type: "construct", params: [this as BaseNode<ShaderType>] }); }
  toUVec4(): any { return node({ _t: "uvec4", type: "construct", params: [this as BaseNode<ShaderType>] }); }
  toBVec2(): any { return node({ _t: "bvec2", type: "construct", params: [this as BaseNode<ShaderType>] }); }
  toBVec3(): any { return node({ _t: "bvec3", type: "construct", params: [this as BaseNode<ShaderType>] }); }
  toBVec4(): any { return node({ _t: "bvec4", type: "construct", params: [this as BaseNode<ShaderType>] }); }
  toMat2(): any { return node({ _t: "mat2", type: "construct", params: [this as BaseNode<ShaderType>] }); }
  toMat3(): any { return node({ _t: "mat3", type: "construct", params: [this as BaseNode<ShaderType>] }); }
  toMat4(): any { return node({ _t: "mat4", type: "construct", params: [this as BaseNode<ShaderType>] }); }
  convert<T extends ShaderType>(target: T): any {
    return node({ _t: target, type: "construct", params: [this as BaseNode<ShaderType>] });
  }

  // === Swizzles (gated by Node<"vec3"> / Node<"vec4"> type) ===
  get x(): Node<"float"> { return swizzle(this, "x"); }
  get y(): Node<"float"> { return swizzle(this, "y"); }
  get z(): Node<"float"> { return swizzle(this, "z"); }
  get w(): Node<"float"> { return swizzle(this, "w"); }
  get r(): Node<"float"> { return swizzle(this, "r"); }
  get g(): Node<"float"> { return swizzle(this, "g"); }
  get b(): Node<"float"> { return swizzle(this, "b"); }
  get a(): Node<"float"> { return swizzle(this, "a"); }
  get xy(): Node<"vec2"> { return swizzle(this, "xy"); }
  get xz(): Node<"vec2"> { return swizzle(this, "xz"); }
  get xw(): Node<"vec2"> { return swizzle(this, "xw"); }
  get yz(): Node<"vec2"> { return swizzle(this, "yz"); }
  get yw(): Node<"vec2"> { return swizzle(this, "yw"); }
  get zw(): Node<"vec2"> { return swizzle(this, "zw"); }
  get xyz(): Node<"vec3"> { return swizzle(this, "xyz"); }
  get xyw(): Node<"vec3"> { return swizzle(this, "xyw"); }
  get xzw(): Node<"vec3"> { return swizzle(this, "xzw"); }
  get yzw(): Node<"vec3"> { return swizzle(this, "yzw"); }
  get rgba(): Node<"vec4"> { return swizzle(this, "rgba"); }
  get rgb(): Node<"vec3"> { return swizzle(this, "rgb"); }
}

// Cast constructor so `new Node<T>(...)` returns `Node<T>` with conditional methods
export const Node = NodeImpl as unknown as new <A extends ShaderType>(config: {
  _t?: string;
  type: string;
  params?: BaseNode<ShaderType>[];
  value?: unknown;
}) => Node<A>;

// === Helpers ===
function node<A extends ShaderType>(config: {
  _t?: string;
  type: string;
  params?: BaseNode<ShaderType>[];
  value?: unknown;
  name?: string;
}): Node<A> {
  let result = new Node<A>({ _t: config._t ?? config.type, ...config } as any);
  if (config.name !== undefined) {
    (result as any).name = config.name;
  }
  return result;
}

export function var_<A extends ShaderType>(varName: string, brandType: string): Node<A> {
  return new Node<A>({
    _t: brandType,
    type: "var",
    value: { varName, varType: brandType },
  });
}

function isNode(x: any): x is BaseNode<ShaderType> {
  return typeof x === 'object' && x !== null && '_t' in x && 'type' in x;
}

/**
 * What sampling a texture gives back. A float texture samples to a vec4; an
 * integer texture to an integer vector of the same width — signed for an
 * `isampler*`, unsigned for a `usampler*`.
 */
function textureResultType(samplerType: string): string {
  if (samplerType.startsWith("isampler")) return "ivec4";
  if (samplerType.startsWith("usampler")) return "uvec4";
  return "vec4";
}

// === Value wrapping (convert raw JS -> Node for AST) ===
type ExtractType<V> =
  V extends FloatLike ? "float" :
  V extends Vec2Like ? "vec2" :
  V extends Vec3Like ? "vec3" :
  V extends Vec4Like ? "vec4" :
  V extends IntLike ? "int" :
  V extends UintLike ? "uint" :
  V extends BooleanLike ? "bool" :
  V extends Mat3Like ? "mat3" :
  V extends Mat4Like ? "mat4" :
  "void";

function wrapValue<V>(x: V): Node<ExtractType<V>> {
  if (x === undefined || x === null) {
    return node({ _t: "void", type: "void" }) as any;
  }
  if (typeof x === "boolean") {
    return node({ _t: "bool", type: "bool", value: x }) as any;
  }
  if (typeof x === "number") {
    return node({ _t: "float", type: "float", value: x }) as any;
  }
  if (Array.isArray(x)) {
    if (x.length === 3) {
      return node({ _t: "vec3", type: "vec3", value: x }) as any;
    }
    if (x.length === 4) {
      return node({ _t: "vec4", type: "vec4", value: x }) as any;
    }
    if (x.length === 2) {
      return node({ _t: "vec2", type: "vec2", value: x }) as any;
    }
    if (x.length === 9) {
      return node({ _t: "mat3", type: "mat3", value: x }) as any;
    }
    if (x.length === 16) {
      return node({ _t: "mat4", type: "mat4", value: x }) as any;
    }
    return node({ _t: "float", type: "float", value: x[0] }) as any;
  }
  return x as any;
}

/**
 * Ops whose result type is not the type of their first operand.
 *
 * Most ops are type-preserving — `vec3 + vec3` is a vec3 — so the default is to
 * inherit from the first operand. These reduce instead, and their `Node` type
 * parameter says so. Without an entry here the node's runtime `_t` disagrees
 * with its declared type, and downstream code that switches on `_t` (variable
 * declarations, the scalar-vs-vector split in comparison codegen) picks the
 * wrong branch.
 */
const REDUCING_OPS: Record<string, string | ((operandType: string) => string)> = {
  dot: "float",
  length: "float",
  distance: "float",
  // The determinant of a square matrix is a scalar.
  determinant: "float",
  // Transposing swaps columns for rows, so a matCxR becomes a matRxC. A square
  // matrix keeps its type, which is why this only matters once the non-square
  // ones are reachable.
  transpose: (operandType) => {
    let shape = MATRIX_DIMENSIONS[operandType];
    if (shape === undefined) return operandType;
    let [columns, rows] = shape;
    return columns === rows ? operandType : `mat${rows}x${columns}`;
  },
  // A matrix column, so it has as many components as the matrix has rows —
  // a mat2x3 is two columns of three, and indexing it gives a vec3. Expressed
  // as a function because unlike the others it depends on the operand.
  matrixElement: (operandType) => {
    let shape = MATRIX_DIMENSIONS[operandType];
    return shape === undefined ? "float" : `vec${shape[1]}`;
  },
  // One component of a vector, so a scalar of its own kind.
  vectorElement: (operandType) => {
    if (operandType.startsWith("ivec")) return "int";
    if (operandType.startsWith("uvec")) return "uint";
    return "float";
  },
};

/** The result type of an op, given the type of the operand that defines it. */
function resultType(op: string, operandType: string): string {
  let reducing = REDUCING_OPS[op];
  if (reducing === undefined) return operandType;
  return typeof reducing === "function" ? reducing(operandType) : reducing;
}

/** Component count per type, for the operations whose width follows it. */
const TYPE_WIDTH: Record<string, number> = {
  float: 1, int: 1, uint: 1, bool: 1,
  vec2: 2, vec3: 3, vec4: 4,
  ivec2: 2, ivec3: 3, ivec4: 4,
  uvec2: 2, uvec3: 3, uvec4: 4,
  bvec2: 2, bvec3: 3, bvec4: 4,
};

/**
 * Where an op's defining operand sits, when it is not the first.
 *
 * Params are emitted in the order the target language expects, and GLSL takes
 * the value last in `step(edge, x)` and `smoothstep(e0, e1, x)`. Reading the
 * type from the first operand there gives the edge's, so `vec3.step(0.5)`
 * produced a node typed float around a call that returns vec3.
 */
const VALUE_OPERAND: Record<string, number> = {
  step: 1,
  smoothstep: 2,
};

/**
 * Ops whose operands must all share the defining operand's type.
 *
 * Their signatures accept `Node<A> | FloatLike`, so a scalar can be passed
 * where a vector is expected — `vec3.step(0.5)`. GLSL tolerates some of those
 * and WGSL none of them, so rather than patch each backend the scalar is
 * broadcast once here and both receive operands that already agree.
 *
 * Ops taking a genuinely scalar argument are absent by design: `mix(a, b, t)`
 * and `refract(i, n, eta)` declare that argument `FloatLike`, and broadcasting
 * it would produce `refract(vec3, vec3, vec3)`, which neither language has.
 */
const UNIFORM_OPERAND_OPS = new Set([
  "step", "smoothstep", "clamp", "min", "max", "pow", "mod",
]);

/**
 * Give a plain JavaScript number the type of the operand it sits beside.
 *
 * A number carries no shader type of its own and wrapValue can only guess — it
 * picks float. Beside an integer operand that guess is wrong: the two operands
 * disagree, codegen inserts a conversion, and the result stops matching the
 * node's own type, as in `int x = (float(u) % 2.0)`.
 *
 * A number the operand's type cannot represent is refused rather than quietly
 * reinterpreted.
 */
function typedOperand(value: any, operandType: string): BaseNode<ShaderType> {
  // A bare number beside an integer vector is broadcast as its component type —
  // the int of an ivec, the uint of a uvec — rather than as the vector itself,
  // since `op`/`comp` construct the broadcast from whatever this wraps.
  let scalarType = /^ivec/.test(operandType) ? "int"
    : /^uvec/.test(operandType) ? "uint"
    : operandType;
  let isIntegral = scalarType === "int" || scalarType === "uint";
  if (typeof value !== "number" || !isIntegral) {
    return wrapValue(value) as BaseNode<ShaderType>;
  }
  if (!Number.isInteger(value)) {
    throw new Error(
      `[RMSL] ${value} is not a whole number, but the operand beside it is an `
      + `${operandType}. Convert the operand to a float, or use a whole number.`,
    );
  }
  if (scalarType === "uint" && value < 0) {
    throw new Error(
      `[RMSL] ${value} is negative, but the operand beside it is unsigned. `
      + `Use a signed operand, or a literal that is not negative.`,
    );
  }
  return node({ _t: scalarType, type: scalarType, value }) as BaseNode<ShaderType>;
}

function op(type: string, ...args: any[]): Node<ShaderType> {
  let first = wrapValue(args[0]) as BaseNode<ShaderType>;
  let firstT = (first as any)?._t || "float";
  let params = [first, ...args.slice(1).map(a => typedOperand(a, firstT))];
  // The operand that defines the op's type — usually the first, but `step` and
  // `smoothstep` take the value last because that is the argument order both
  // languages expect.
  let valueIndex = VALUE_OPERAND[type] ?? 0;
  let valueT = (params[valueIndex] as any)?._t ?? firstT;

  if (UNIFORM_OPERAND_OPS.has(type) && (TYPE_WIDTH[valueT] ?? 1) > 1) {
    params = params.map(p =>
      (TYPE_WIDTH[(p as any)?._t] ?? 1) === 1
        ? node({ _t: valueT, type: "construct", params: [p] }) as BaseNode<ShaderType>
        : p,
    );
  }

  return node({ _t: resultType(type, valueT), type, params });
}

function op1(type: string, a: any): Node<ShaderType> {
  let wrapped = wrapValue(a) as BaseNode<ShaderType>;
  let t = (wrapped as any)?._t || "float";
  return node({ _t: resultType(type, t), type, params: [wrapped] });
}

/**
 * Comparisons are component-wise, so comparing vectors yields one boolean per
 * component — `bvec3` for vec3 — and only scalars reduce to a single `bool`.
 * This mirrors GLSL, where `lessThan(vec3, vec3)` is a bvec3, and matches how
 * Three.js's TSL types the same operations.
 */
function comp(type: string, a: any, b: any): Node<ShaderType> {
  // Comparisons need the same literal typing arithmetic gets: an unsigned
  // operand compared against a plain number must be typed accordingly.
  let first = wrapValue(a) as BaseNode<ShaderType>;
  let params = [first, typedOperand(b, (first as any)?._t || "float")];
  let widths = params.map(p => TYPE_WIDTH[(p as any)?._t] ?? 1);
  let width = Math.max(widths[0], widths[1]);

  // Neither language compares a vector against a scalar: GLSL has no
  // lessThan(vec3, float) and WGSL no `operator < (vec3<f32>, f32)`. The
  // signatures accept the mix, so the scalar is broadcast to the vector's
  // width — `lessThan(v, vec3(0.5))` — which is what the caller meant.
  if (width > 1) {
    let wide = (params[widths[0] >= widths[1] ? 0 : 1] as any)._t as ShaderType;
    params = params.map((p, i) =>
      widths[i] === 1
        ? node({ _t: wide, type: "construct", params: [p] }) as BaseNode<ShaderType>
        : p,
    );
  }

  return node({ _t: width > 1 ? `bvec${width}` : "bool", type, params });
}

function swizzle<A extends ShaderType>(src: BaseNode<ShaderType>, pattern: string): Node<A> {
  // A single component of an integer vector is that integer scalar, not a
  // float, so the result type is derived from the source's component prefix
  // rather than assumed float.
  let srcT = (src as any)?._t || "float";
  let prefix = /^ivec/.test(srcT) ? "i" : /^uvec/.test(srcT) ? "u" : "";
  let outType = pattern.length === 1
    ? prefix === "i" ? "int" as const : prefix === "u" ? "uint" as const : "float" as const
    : prefix === "i" ? `ivec${pattern.length}` as const
    : prefix === "u" ? `uvec${pattern.length}` as const
    : `vec${pattern.length}` as const;
  return node({
    _t: outType,
    type: "swizzle",
    params: [src],
    value: pattern,
  }) as Node<A>;
}

// === Block scope (same pattern as story-lang) ===
let blockScope: BaseNode<ShaderType>[] | undefined = undefined;
let nextVarId = 0;

export function assertBlockScope(
  fnName: string,
  fn: (blockScope: BaseNode<ShaderType>[]) => void,
) {
  if (blockScope === undefined) {
    throw new Error(`${fnName} must be called inside an Fn(() => { ... }) scope.`);
  }
  fn(blockScope);
}

// === Fn - macro that captures statements into a seq node ===
// Supports single return: Fn(() => { ...; return x; }) -> () => Node<A>
// Supports multi return: Fn(() => { ...; return [a, b]; }) -> () => [Node<A>, Node<B>]
// Supports parameters: Fn((a: Node<"float">, b: Node<"float">) => a.add(b)) -> (a, b) => Node<"float">
export function Fn<T extends any[], const R>(fn: (...args: T) => R): (...args: T) => R {
  return ((...args: T) => {
    let oldBlockScope = blockScope;
    try {
      let scope: BaseNode<ShaderType>[] = [];
      blockScope = scope;
      let r = fn(...args);
      if (Array.isArray(r)) {
        return r.map((_, i) => {
          let item = wrapValue((r as any[])[i]) as BaseNode<ShaderType>;
          return node({
            _t: item._t || "void",
            type: "seq",
            params: [...scope, item],
          }) as Node<ShaderType>;
        }) as R;
      }
      let wrappedR = wrapValue(r as any) as BaseNode<ShaderType>;
      let returnType = wrappedR._t || "void";
      let seqNode = node({
        _t: returnType,
        type: "seq",
        params: [...scope, wrappedR],
      }) as any;
      return seqNode;
    } finally {
      blockScope = oldBlockScope;
    }
  });
}

function buildBlock(body: () => void): Node<"void"> {
  let oldBlockScope = blockScope;
  blockScope = [];
  try {
    body();
    return node({
      _t: "void",
      type: "seq",
      params: [...blockScope!],
    }) as Node<"void">;
  } finally {
    blockScope = oldBlockScope;
  }
}

// === Literal constructors (with overloads) ===
export function float(v: number | Node<"int">): Node<"float"> {
  if (isNode(v)) {
    return node({ _t: "float", type: "construct", params: [v] }) as Node<"float">;
  }
  return node({ _t: "float", type: "float", value: v }) as Node<"float">;
}
export function vec2(x?: FloatLike | Node<"vec2"> | Node<"vec3"> | Node<"vec4">, y?: FloatLike): Node<"vec2"> {
  if (x === undefined) {
    return node({ _t: "vec2", type: "construct", params: [wrapValue(0)] }) as Node<"vec2">;
  }
  if (isNode(x)) {
    let params = [x as BaseNode<ShaderType>];
    if (y !== undefined) params.push(wrapValue(y) as BaseNode<ShaderType>);
    return node({ _t: "vec2", type: "construct", params }) as Node<"vec2">;
  }
  if (y === undefined) {
    return node({ _t: "vec2", type: "construct", params: [wrapValue(x)] }) as Node<"vec2">;
  }
  if (typeof y === "number") {
    return node({ _t: "vec2", type: "vec2", value: [x, y] }) as Node<"vec2">;
  }
  return node({ _t: "vec2", type: "construct", params: [wrapValue(x), y as BaseNode<ShaderType>] }) as Node<"vec2">;
}
export function vec3(x?: FloatLike | Node<"vec3"> | Node<"vec4">, y?: FloatLike, z?: FloatLike): Node<"vec3"> {
  if (x === undefined) {
    return node({ _t: "vec3", type: "construct", params: [wrapValue(0)] }) as Node<"vec3">;
  }
  if (isNode(x)) {
    let params = [x as BaseNode<ShaderType>];
    if (y !== undefined) params.push(wrapValue(y) as BaseNode<ShaderType>);
    if (z !== undefined) params.push(wrapValue(z) as BaseNode<ShaderType>);
    return node({ _t: "vec3", type: "construct", params }) as Node<"vec3">;
  }
  if (y === undefined) {
    return node({ _t: "vec3", type: "construct", params: [wrapValue(x)] }) as Node<"vec3">;
  }
  if (typeof y === "number" && (z === undefined || typeof z === "number")) {
    let values: number[] = [x, y];
    if (z !== undefined) values.push(z);
    return node({ _t: "vec3", type: "vec3", value: values }) as Node<"vec3">;
  }
  let params = [wrapValue(x) as BaseNode<ShaderType>];
  if (y !== undefined) params.push(wrapValue(y) as BaseNode<ShaderType>);
  if (z !== undefined) params.push(wrapValue(z) as BaseNode<ShaderType>);
  return node({ _t: "vec3", type: "construct", params }) as Node<"vec3">;
}
export function vec4(x?: FloatLike | Node<"vec2"> | Node<"vec3"> | Node<"vec4">, y?: FloatLike, z?: FloatLike, w?: FloatLike): Node<"vec4"> {
  if (x === undefined) {
    return node({ _t: "vec4", type: "construct", params: [wrapValue(0)] }) as Node<"vec4">;
  }
  if (isNode(x)) {
    let params = [x as BaseNode<ShaderType>];
    if (y !== undefined) params.push(wrapValue(y) as BaseNode<ShaderType>);
    if (z !== undefined) params.push(wrapValue(z) as BaseNode<ShaderType>);
    if (w !== undefined) params.push(wrapValue(w) as BaseNode<ShaderType>);
    return node({ _t: "vec4", type: "construct", params }) as Node<"vec4">;
  }
  if (y === undefined) {
    return node({ _t: "vec4", type: "construct", params: [wrapValue(x)] }) as Node<"vec4">;
  }
  if (typeof y === "number" && (z === undefined || typeof z === "number") && (w === undefined || typeof w === "number")) {
    let values: number[] = [x, y];
    if (z !== undefined) values.push(z);
    if (w !== undefined) values.push(w);
    return node({ _t: "vec4", type: "vec4", value: values }) as Node<"vec4">;
  }
  let params = [wrapValue(x) as BaseNode<ShaderType>];
  if (y !== undefined) params.push(wrapValue(y) as BaseNode<ShaderType>);
  if (z !== undefined) params.push(wrapValue(z) as BaseNode<ShaderType>);
  if (w !== undefined) params.push(wrapValue(w) as BaseNode<ShaderType>);
  return node({ _t: "vec4", type: "construct", params }) as Node<"vec4">;
}
export function int(v: number | Node<"float">): Node<"int"> {
  if (isNode(v)) {
    return node({ _t: "int", type: "construct", params: [v] }) as Node<"int">;
  }
  return node({ _t: "int", type: "int", value: v | 0 }) as Node<"int">;
}
export function uint(v: number | Node<"float"> | Node<"int">): Node<"uint"> {
  if (isNode(v)) {
    return node({ _t: "uint", type: "construct", params: [v] }) as Node<"uint">;
  }
  if (v < 0) {
    throw new Error(
      `[RMSL] uint(${v}) is negative. An unsigned literal cannot be negative.`,
    );
  }
  return node({ _t: "uint", type: "uint", value: v | 0 }) as Node<"uint">;
}

/**
 * Build an integer-vector constructor the same way vec2/3/4 are built: a single
 * number (or node) is broadcast via a construct, while a full set of number
 * arguments becomes a literal so the result folds like any other constant.
 */
function makeIntVecConstructor<T extends ShaderType>(
  t: T,
  width: number,
  scalarType: "int" | "uint",
): (...args: any[]) => Node<T> {
  return (...args: any[]): Node<T> => {
    if (args.length === 0) {
      return node({
        _t: t,
        type: "construct",
        params: [node({ _t: scalarType, type: scalarType, value: 0 })],
      }) as Node<T>;
    }
    if (args.length === 1 && isNode(args[0])) {
      return node({ _t: t, type: "construct", params: [args[0] as BaseNode<ShaderType>] }) as Node<T>;
    }
    if (args.length === 1 && typeof args[0] === "number") {
      return node({
        _t: t,
        type: "construct",
        params: [node({ _t: scalarType, type: scalarType, value: args[0] | 0 })],
      }) as Node<T>;
    }
    if (args.length <= width && args.every((a) => typeof a === "number")) {
      if (scalarType === "uint") {
        for (let a of args) {
          if (a < 0) {
            throw new Error(
              `[RMSL] ${a} is negative, but ${t} components are unsigned. `
              + `Use a signed vector, or values that are not negative.`,
            );
          }
        }
      }
      return node({ _t: t, type: t, value: args.map((a) => a | 0) }) as Node<T>;
    }
    return node({
      _t: t,
      type: "construct",
      params: args.map((a: any) =>
        isNode(a) ? a as BaseNode<ShaderType> : wrapValue(a) as BaseNode<ShaderType>,
      ),
    }) as Node<T>;
  };
}

export const ivec2 = makeIntVecConstructor<"ivec2">("ivec2", 2, "int");
export const ivec3 = makeIntVecConstructor<"ivec3">("ivec3", 3, "int");
export const ivec4 = makeIntVecConstructor<"ivec4">("ivec4", 4, "int");
export const uvec2 = makeIntVecConstructor<"uvec2">("uvec2", 2, "uint");
export const uvec3 = makeIntVecConstructor<"uvec3">("uvec3", 3, "uint");
export const uvec4 = makeIntVecConstructor<"uvec4">("uvec4", 4, "uint");
export function boolean(v: boolean | Node<"float"> | Node<"int"> | Node<"uint">): Node<"bool"> {
  if (isNode(v)) {
    return node({ _t: "bool", type: "construct", params: [v] }) as Node<"bool">;
  }
  return node({ _t: "bool", type: "bool", value: v }) as Node<"bool">;
}
function makeMatConstructor<T extends ShaderType>(t: T, size: number, defaultVal: number[]): (...args: any[]) => Node<T> {
  return (...args: any[]): Node<T> => {
    if (args.length === 1 && isNode(args[0])) {
      return node({ _t: t, type: "construct", params: [args[0] as BaseNode<ShaderType>] }) as Node<T>;
    }
    if (args.length === 1 && typeof args[0] === "number") {
      return node({ _t: t, type: "construct", params: [wrapValue(args[0])] }) as Node<T>;
    }
    if (args.length === 0) {
      return node({ _t: t, type: t, value: defaultVal }) as Node<T>;
    }
    return node({ _t: t, type: t, value: args }) as Node<T>;
  };
}
export const mat2 = makeMatConstructor("mat2", 4, [1,0,0,1]);
export const mat2x3 = makeMatConstructor("mat2x3", 6, [1,0,0,0,1,0]);
export const mat2x4 = makeMatConstructor("mat2x4", 8, [1,0,0,0,0,1,0,0]);
export const mat3x2 = makeMatConstructor("mat3x2", 6, [1,0,0,0,1,0]);
export function mat3(...args: any[]): Node<"mat3"> {
  if (args.length === 1 && isNode(args[0])) {
    return node({ _t: "mat3", type: "construct", params: [args[0] as BaseNode<ShaderType>] }) as Node<"mat3">;
  }
  if (args.length === 3 && args.every((a: any) => isNode(a))) {
    return node({ _t: "mat3", type: "construct", params: args.map((a: any) => a as BaseNode<ShaderType>) }) as Node<"mat3">;
  }
  if (args.length === 1 && typeof args[0] === "number") {
    return node({ _t: "mat3", type: "construct", params: [wrapValue(args[0])] }) as Node<"mat3">;
  }
  if (args.length === 0) {
    return node({ _t: "mat3", type: "mat3", value: [1,0,0,0,1,0,0,0,1] }) as Node<"mat3">;
  }
  return node({ _t: "mat3", type: "mat3", value: args }) as Node<"mat3">;
}
export const mat3x4 = makeMatConstructor("mat3x4", 12, [1,0,0,0,0,1,0,0,0,0,1,0]);
export const mat4x2 = makeMatConstructor("mat4x2", 8, [1,0,0,0,0,1,0,0]);
export const mat4x3 = makeMatConstructor("mat4x3", 12, [1,0,0,0,0,1,0,0,0,0,1,0]);
export function mat4(...args: any[]): Node<"mat4"> {
  if (args.length === 1 && isNode(args[0])) {
    return node({ _t: "mat4", type: "construct", params: [args[0] as BaseNode<ShaderType>] }) as Node<"mat4">;
  }
  if (args.length === 4 && args.every((a: any) => isNode(a))) {
    return node({ _t: "mat4", type: "construct", params: args.map((a: any) => a as BaseNode<ShaderType>) }) as Node<"mat4">;
  }
  if (args.length === 1 && typeof args[0] === "number") {
    return node({ _t: "mat4", type: "construct", params: [wrapValue(args[0])] }) as Node<"mat4">;
  }
  if (args.length === 0) {
    return node({ _t: "mat4", type: "mat4", value: [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1] }) as Node<"mat4">;
  }
  return node({ _t: "mat4", type: "mat4", value: args }) as Node<"mat4">;
}

// === Uniforms, Attributes, Varyings ===
let nextUniformId = 0;
let nextAttrId = 0;
let nextVaryingId = 0;

/**
 * A uniform holding several values of one type.
 *
 * Declaring N separate uniforms instead costs N slots, and WGSL allows only 12
 * uniform buffers per stage; an array is one slot however long it is. It also
 * lets the shader loop over the elements rather than unrolling a test per
 * value.
 *
 * The length is given rather than the values, since the host writes the
 * contents by name — unlike TSL's `uniformArray(values, type)`, where the node
 * owns the data.
 *
 *   const bricks = uniformArray("vec4", 24);
 *   bricks.element(i)            // indexed by a node, inside a loop
 *   bricks.element(3)            // or by a constant
 */
export function uniformArray<T extends ShaderType>(
  shaderType: T,
  length: number,
): UniformArrayNode<T> {
  if (!Number.isInteger(length) || length < 1) {
    throw new Error(`[RMSL] uniformArray length must be a positive integer, got ${length}`);
  }
  if (isSamplerType(shaderType)) {
    throw new Error(
      `[RMSL] uniformArray cannot hold a texture. WGSL has no array of separate`
      + ` texture bindings without an extension, so there is no spelling both`
      + ` backends share — Three.js does not offer one either. Declare each`
      + ` texture on its own, or use a layered array texture, which both`
      + ` languages do have.`,
    );
  }
  let id = nextUniformId++;
  let slot = `_rmsl_u${id}`;
  const arrayNode = node({
    _t: shaderType,
    type: "uniformArray",
    value: { id, slot, shaderType, length },
    name: slot,
  }) as any;
  arrayNode.length = length;
  arrayNode.element = (index: IntLike | FloatLike) =>
    node({
      _t: shaderType,
      type: "uniformArrayElement",
      params: [arrayNode, wrapValue(index) as BaseNode<ShaderType>],
    });
  return arrayNode as UniformArrayNode<T>;
}

export function uniform<T extends ShaderType>(shaderType: T): UniformNode<T> {
  let id = nextUniformId++;
  const result = node({
    _t: shaderType,
    type: "uniform",
    value: { id, slot: `_rmsl_u${id}`, shaderType },
    name: `_rmsl_u${id}`,
  });
  return result as unknown as UniformNode<T>;
}

export function uniformRaw<T extends ShaderType>(name: string, shaderType: T): UniformNode<T> {
  let id = nextUniformId++;
  const result = node({
    _t: shaderType,
    type: "uniform",
    value: { id, slot: name, shaderType },
    name: name,
  });
  return result as unknown as UniformNode<T>;
}

export function attribute<T extends ShaderType>(shaderType: T): AttributeNode<T> {
  let id = nextAttrId++;
  const result = node({
    _t: shaderType,
    type: "attribute",
    value: { id, slot: `_rmsl_a${id}`, shaderType },
    name: `_rmsl_a${id}`,
  });
  return result as unknown as AttributeNode<T>;
}

export function varying<T extends ShaderType>(shaderType: T): VaryingNode<T> {
  let id = nextVaryingId++;
  const result = node({
    _t: shaderType,
    type: "varying",
    value: { id, slot: `_rmsl_v${id}`, shaderType },
    name: `_rmsl_v${id}`,
  });
  return result as unknown as VaryingNode<T>;
}

// === Outputs ===
let nextOutputId = 0;

export function output<T extends ShaderType>(shaderType: T): Node<T> {
  let id = nextOutputId++;
  return node({
    _t: shaderType,
    type: "output",
    value: { id, slot: `_rmsl_o${id}`, shaderType, location: id },
  }) as Node<T>;
}

export function builtinPosition(): Node<"vec4"> {
  return node({
    _t: "vec4",
    type: "builtinPosition",
  }) as Node<"vec4">;
}

export function builtinFragDepth(): Node<"float"> {
  return node({
    _t: "float",
    type: "builtinFragDepth",
  }) as Node<"float">;
}

// === Control Flow ===

type ElseIfChain = {
  ElseIf: (cond: BooleanLike, body: () => void) => ElseIfChain;
  elseIf: (cond: BooleanLike, body: () => void) => ElseIfChain;
  Else: (body: () => void) => void;
  else_: (body: () => void) => void;
};

export function If(cond: BooleanLike, body: () => void): ElseIfChain {
  let ifNode = node({
    _t: "void",
    type: "if",
    params: [
      wrapValue(cond) as BaseNode<ShaderType>,
      buildBlock(body) as BaseNode<ShaderType>,
    ],
  });
  assertBlockScope("If", (scope) => { scope.push(ifNode); });
  let deepestIf = ifNode;
  const chain: ElseIfChain = {
    ElseIf: (nextCond, nextBody) => {
      let nextIf = node({
        _t: "void",
        type: "if",
        params: [
          wrapValue(nextCond) as BaseNode<ShaderType>,
          buildBlock(nextBody) as BaseNode<ShaderType>,
        ],
      });
      deepestIf.params![2] = nextIf as BaseNode<ShaderType>;
      deepestIf = nextIf;
      return chain;
    },
    elseIf: (nextCond, nextBody) => {
      let nextIf = node({
        _t: "void",
        type: "if",
        params: [
          wrapValue(nextCond) as BaseNode<ShaderType>,
          buildBlock(nextBody) as BaseNode<ShaderType>,
        ],
      });
      deepestIf.params![2] = nextIf as BaseNode<ShaderType>;
      deepestIf = nextIf;
      return chain;
    },
    Else: (elseBody) => {
      deepestIf.params![2] = buildBlock(elseBody) as BaseNode<ShaderType>;
    },
    else_: (elseBody) => {
      deepestIf.params![2] = buildBlock(elseBody) as BaseNode<ShaderType>;
    },
  };
  return chain;
}

/**
 * Lowercase alias for `If`, so control flow reads like keywords:
 * `if_ (cond) { ... }.elseIf (...).else_ (...)`. The trailing underscore avoids
 * the reserved word `if`; `elseIf` needs none.
 */
export function if_(cond: BooleanLike, body: () => void): ElseIfChain {
  return If(cond, body);
}

export function For<T extends Node<ShaderType>>(
  init: () => T,
  cond: (v: T) => BooleanLike,
  update: (v: T) => void,
  body: (v: T) => void,
): void {
  assertBlockScope("For", (scope) => {
    let oldBlockScope = blockScope;
    let initScope: BaseNode<ShaderType>[] = [];
    blockScope = initScope;
    let v: T;
    try {
      v = init();
    } finally {
      blockScope = oldBlockScope;
    }
    let initNode = node({ _t: "void", type: "seq", params: [...initScope] }) as Node<"void">;
    let condNode = wrapValue(cond(v)) as BaseNode<ShaderType>;
    let updateNode = buildBlock(() => update(v));
    let bodyNode = buildBlock(() => body(v));
    scope.push(node({
      _t: "void",
      type: "for",
      params: [initNode, condNode, updateNode, bodyNode],
    }));
  });
}

/** Lowercase alias for `For` — `for_` avoids the reserved word `for`. */
export function for_<T extends Node<ShaderType>>(
  init: () => T,
  cond: (v: T) => BooleanLike,
  update: (v: T) => void,
  body: (v: T) => void,
): void {
  return For(init, cond, update, body);
}

export function While(cond: BooleanLike, body: () => void): void {
  assertBlockScope("While", (scope) => {
    let condNode = wrapValue(cond) as BaseNode<ShaderType>;
    let bodyNode = buildBlock(body);
    scope.push(node({
      _t: "void",
      type: "while",
      params: [condNode, bodyNode],
    }));
  });
}

/** Lowercase alias for `While` — `while_` avoids the reserved word `while`. */
export function while_(cond: BooleanLike, body: () => void): void {
  return While(cond, body);
}

type SwitchCase = { values: BaseNode<ShaderType>[]; body: Node<"void"> };

type SwitchChain = {
  Case: (values: IntLike | readonly IntLike[], body: () => void) => SwitchChain;
  case_: (values: IntLike | readonly IntLike[], body: () => void) => SwitchChain;
  Default: (body: () => void) => void;
  default_: (body: () => void) => void;
};

/**
 * Multi-way branch on an integer selector.
 *
 *   Switch(level, (s) => {
 *     s.Case(0, () => { colour.assign(black); });
 *     s.Case(1, 2, () => { colour.assign(grey); });
 *     s.Default(() => { colour.assign(white); });
 *   });
 *
 * Compiles to an if/else-if chain comparing the selector with each case value —
 * the same lowering Three.js's TSL uses for its `Switch`/`Case`/`Default` — so
 * there is no fall-through and no `break_()` inside a case.
 */
export function Switch(
  selector: Node<"int"> | Node<"uint">,
  body: (chain: SwitchChain) => void,
): SwitchChain {
  let cases: SwitchCase[] = [];
  let defaultBody: Node<"void"> | undefined;
  const addCase = (values: IntLike | readonly IntLike[], caseBody: () => void): SwitchChain => {
    let vals = (Array.isArray(values) ? values : [values]) as IntLike[];
    cases.push({
      values: vals.map(v => wrapValue(v) as BaseNode<ShaderType>),
      body: buildBlock(caseBody),
    });
    return chain;
  };
  const chain: SwitchChain = {
    Case: addCase,
    case_: addCase,
    Default: (dBody) => { defaultBody = buildBlock(dBody); },
    default_: (dBody) => { defaultBody = buildBlock(dBody); },
  };
  body(chain);

  let root = node({ _t: "void", type: "if", params: [] });
  let cursor = root;
  let selectorNode = wrapValue(selector) as BaseNode<ShaderType>;
  for (let c of cases) {
    let cond: BaseNode<ShaderType> | undefined;
    for (let v of c.values) {
      let eq = comp("equal", selectorNode, v);
      cond = cond === undefined ? eq : (op("or", cond, eq) as BaseNode<ShaderType>);
    }
    let ifNode = node({ _t: "void", type: "if", params: [cond!, c.body] });
    cursor.params![2] = ifNode;
    cursor = ifNode;
  }
  if (defaultBody !== undefined) {
    cursor.params![2] = defaultBody;
  }
  let switchNode = root.params![2] as BaseNode<ShaderType>;
  assertBlockScope("Switch", (scope) => { scope.push(switchNode); });
  return chain;
}

/**
 * Lowercase alias for `Switch` — `switch_` avoids the reserved word `switch`.
 * Inside the body, `case_` and `default_` mirror `Case` and `Default`.
 */
export function switch_(
  selector: Node<"int"> | Node<"uint">,
  body: (chain: SwitchChain) => void,
): SwitchChain {
  return Switch(selector, body);
}

export function discard(): void {
  assertBlockScope("discard", (scope) => {
    scope.push(node({ _t: "void", type: "discard" }));
  });
}

export function break_(): void {
  assertBlockScope("break", (scope) => {
    scope.push(node({ _t: "void", type: "break" }));
  });
}

export function continue_(): void {
  assertBlockScope("continue", (scope) => {
    scope.push(node({ _t: "void", type: "continue" }));
  });
}

// ==== COMPILERS ====

// ========== GLSL Compiler ==========
/**
 * What compiling one node yields: statements to emit, how to refer to it, and
 * its operator precedence (higher = tighter binding, for bracket reduction).
 *
 * `prec` is omitted for atoms (literals, variables, function calls) — callers
 * default it to `PREC_ATOM` so they never need wrapping.
 */
interface CompiledNode {
  decls: string[];
  body: string[];
  expr: string;
  prec?: number;
}

/**
 * Precedence values for bracket reduction. Higher number = tighter binding.
 *
 * GLSL and WGSL share the same relative ordering, so one table covers both.
 */
const PRECEDENCE: Record<string, number> = {
  or: 10,
  and: 20,
  bitOr: 30,
  bitXor: 40,
  bitAnd: 50,
  equal: 60,
  notEqual: 60,
  lessThan: 60,
  greaterThan: 60,
  lessThanEqual: 60,
  greaterThanEqual: 60,
  shiftLeft: 70,
  shiftRight: 70,
  add: 80,
  sub: 80,
  mult: 90,
  div: 90,
  mod: 90,
};

/** Precedence for unary operators (negate, not). Tighter than all binary ops. */
const PREC_UNARY = 100;

/** Precedence for atoms — never needs wrapping. */
const PREC_ATOM = 200;

/**
 * Wrap a child expression in parens when its precedence is lower than (or equal
 * to) the parent operator's, otherwise the child would be parsed differently.
 */
function wrapExpr(
  childPrec: number | undefined,
  parentPrec: number,
  expr: string,
): string {
  return (childPrec ?? PREC_ATOM) <= parentPrec ? `(${expr})` : expr;
}

interface CompileCtx {
  nextId: number;
  shaderStage: "vertex" | "fragment";
  /** `length` is set only for uniform arrays, and gives their element count. */
  uniforms: Map<number, { type: string; slot: string; length?: number }>;
  attributes: Map<number, { type: string; slot: string }>;
  varyings: Map<number, { type: string; slot: string }>;
  outputs: Map<number, { type: string; slot: string; location: number }>;
  wgslSamplers: Map<string, { textureSlot: string; samplerSlot: string }>;
  varDefs: Map<string, string>;
  /**
   * What each node already compiled to, keyed by the node itself.
   *
   * The graph is a directed acyclic graph, not a tree: `Fn` returning an array
   * gives every element the whole block scope, so one node is reachable from
   * several roots — as the same object, not a copy. Compiling it once per root
   * repeats whatever it does, which for a declaration is a redefinition and for
   * an assignment or a loop is the work happening twice.
   *
   * So a node is compiled the first time it is reached and its statements are
   * emitted there. Later arrivals get its expression alone, since the
   * statements producing that expression are already in the output.
   */
  memo: Map<BaseNode<ShaderType>, CompiledNode>;
  /**
   * Names of WGSL helper functions the shader needs, emitted ahead of the entry
   * point. GLSL provides some builtins that WGSL does not, so they are written
   * out on demand rather than always.
   */
  wgslHelpers: Set<string>;
  /**
   * Whether the program assigned the position itself. A vertex stage that has
   * done so needs no implicit write, and its result is free to be anything.
   */
  positionWritten: boolean;
  inFn: boolean;
  fragDepthUsed: boolean;
}

let typeToGLSL: Record<string, string> = {
  float: "float", vec2: "vec2", vec3: "vec3", vec4: "vec4",
  int: "int", uint: "uint", bool: "bool",
  ivec2: "ivec2", ivec3: "ivec3", ivec4: "ivec4",
  uvec2: "uvec2", uvec3: "uvec3", uvec4: "uvec4",
  bvec2: "bvec2", bvec3: "bvec3", bvec4: "bvec4",
  mat2: "mat2", mat2x3: "mat2x3", mat2x4: "mat2x4",
  mat3x2: "mat3x2", mat3: "mat3", mat3x4: "mat3x4",
  mat4x2: "mat4x2", mat4x3: "mat4x3", mat4: "mat4",
  sampler2D: "sampler2D", sampler3D: "sampler3D", samplerCube: "samplerCube",
  isampler2D: "isampler2D", isampler3D: "isampler3D", isamplerCube: "isamplerCube",
  usampler2D: "usampler2D", usampler3D: "usampler3D", usamplerCube: "usamplerCube",
  void: "void",
};

function glslType(brand: any): string {
  return typeToGLSL[brand as string] ?? "float";
}

// === Constant folding ===
function isLeafLiteral(n: BaseNode<ShaderType>): boolean {
  return (n.type === "float" || n.type === "int" || n.type === "uint" || n.type === "bool") && !n.params;
}

function tryFold(n: BaseNode<ShaderType>): BaseNode<ShaderType> | null {
  let params = n.params ?? [];
  if (!params.every(isLeafLiteral)) return null;
  let p0 = params[0]?.value;
  let p1 = params[1]?.value;
  let t = n._t;
  if (t === "float" || t === "int" || t === "uint") {
    let a = p0 as number;
    let b = p1 as number;
    switch (n.type) {
      case "add": return mkNode({ _t: t, type: t, value: t === "int" || t === "uint" ? (a + b) | 0 : a + b });
      case "sub": return mkNode({ _t: t, type: t, value: t === "int" || t === "uint" ? (a - b) | 0 : a - b });
      case "mult": return mkNode({ _t: t, type: t, value: t === "int" || t === "uint" ? (a * b) | 0 : a * b });
      case "div": return mkNode({ _t: t, type: t, value: t === "int" || t === "uint" ? (a / b) | 0 : a / b });
      case "negate": return mkNode({ _t: t, type: t, value: t === "int" || t === "uint" ? (-a) | 0 : -a });
      // JavaScript's % truncates toward zero. The float operation is floored,
      // following GLSL's mod(), so folding it with % would give a literal that
      // disagrees with what the same expression computes when its operands are
      // not constants. The integer path keeps % because that is what both
      // backends emit for integers.
      case "mod": return mkNode({
        _t: t,
        type: t,
        value: t === "int" || t === "uint" ? (a % b) | 0 : a - b * Math.floor(a / b),
      });
      case "sin": return mkNode({ _t: t, type: t, value: Math.sin(a) });
      case "cos": return mkNode({ _t: t, type: t, value: Math.cos(a) });
      case "tan": return mkNode({ _t: t, type: t, value: Math.tan(a) });
      case "asin": return mkNode({ _t: t, type: t, value: Math.asin(a) });
      case "acos": return mkNode({ _t: t, type: t, value: Math.acos(a) });
      case "atan": return mkNode({ _t: t, type: t, value: Math.atan(a) });
      case "sinh": return mkNode({ _t: t, type: t, value: Math.sinh(a) });
      case "cosh": return mkNode({ _t: t, type: t, value: Math.cosh(a) });
      case "tanh": return mkNode({ _t: t, type: t, value: Math.tanh(a) });
      case "asinh": return mkNode({ _t: t, type: t, value: Math.asinh(a) });
      case "acosh": return mkNode({ _t: t, type: t, value: Math.acosh(a) });
      case "atanh": return mkNode({ _t: t, type: t, value: Math.atanh(a) });
      case "abs": return mkNode({ _t: t, type: t, value: Math.abs(a) });
      case "sign": return mkNode({ _t: t, type: t, value: Math.sign(a) });
      case "floor": return mkNode({ _t: t, type: t, value: Math.floor(a) });
      case "ceil": return mkNode({ _t: t, type: t, value: Math.ceil(a) });
      case "round": return mkNode({ _t: t, type: t, value: Math.round(a) });
      case "trunc": return mkNode({ _t: t, type: t, value: Math.trunc(a) });
      case "fract": return mkNode({ _t: t, type: t, value: a - Math.floor(a) });
      case "sqrt": return mkNode({ _t: t, type: t, value: Math.sqrt(a) });
      case "inversesqrt": return mkNode({ _t: t, type: t, value: 1 / Math.sqrt(a) });
      case "exp": return mkNode({ _t: t, type: t, value: Math.exp(a) });
      case "log": return mkNode({ _t: t, type: t, value: Math.log(a) });
      case "exp2": return mkNode({ _t: t, type: t, value: Math.pow(2, a) });
      case "log2": return mkNode({ _t: t, type: t, value: Math.log2(a) });
      case "pow": return mkNode({ _t: t, type: t, value: Math.pow(a, b) });
      case "min": return mkNode({ _t: t, type: t, value: Math.min(a, b) });
      case "max": return mkNode({ _t: t, type: t, value: Math.max(a, b) });
      case "dot": return mkNode({ _t: t, type: t, value: a * b });
    }
  }
  return null;
}

function mkNode(config: { _t?: string; type: string; params?: BaseNode<ShaderType>[]; value?: unknown }): BaseNode<ShaderType> {
  return new NodeImpl({ _t: config._t ?? config.type, type: config.type, params: config.params, value: config.value }) as BaseNode<ShaderType>;
}

/**
 * Render a for-loop's update clause.
 *
 * The clause is authored as statements — `(i) => i.assign(i.add(1))` — so it
 * arrives with its work in `body` and only a bare variable reference in `expr`.
 * Emitting `expr` alone drops the increment and produces an infinite loop.
 *
 * GLSL's update slot accepts a comma expression, so every statement survives.
 * WGSL's grammar allows exactly one update statement, so callers there keep
 * the last.
 */
function forUpdateStatements(update: CompiledNode): string[] {
  // A nested block cannot go in either language's update slot: GLSL's takes an
  // expression, and accepting one in WGSL alone would make a program that runs
  // on one backend and not the other.
  if (update.body.some(line => line.includes("{"))) {
    throw new Error(
      "[RMSL] A for-loop's update cannot contain a block. Move the branch into "
      + "the loop body, or write the loop with While.",
    );
  }
  return update.body;
}

/** Drop a trailing semicolon, for the slots that take an expression. */
function withoutSemicolon(statement: string): string {
  return statement.endsWith(";") ? statement.slice(0, -1) : statement;
}

/**
 * Reject a vertex stage whose result cannot reach the position output.
 *
 * That output is a vec4 and writing it is not optional, so a vertex shader
 * returning anything else is unambiguously a mistake: a value was produced and
 * has nowhere to go. Skipping the write instead would link cleanly and draw
 * nothing, which is the silent-corruption failure mode the unhandled-node case
 * throws to avoid.
 *
 * A fragment stage is deliberately not checked. A shader with no colour output
 * is legal, so "no result" there is a choice rather than a mistake.
 */
function assertStageResult(
  shaderStage: "vertex" | "fragment",
  lastType: string | undefined,
  positionWritten: boolean,
): void {
  if (shaderStage !== "vertex") return;
  // The program set the position itself, so its result has nowhere it needs to
  // go and can be anything, including nothing.
  if (positionWritten) return;
  // Otherwise the result becomes the position, and has to be able to.
  if (lastType === "vec4") return;
  // Whether a stage produced a value is a question about its type, not the text
  // it compiled to. A vertex shader returning zero or returning nothing both
  // fail the same check.
  throw new Error(
    `[RMSL] A vertex shader has to produce a position. This one `
    + (lastType === undefined || lastType === "void"
      ? `returns nothing and never assigns builtinPosition(). Return a vec4, or `
        + `assign builtinPosition() yourself.`
      : `returns ${lastType}, which cannot become one. Wrap it — for example `
        + `vec4(value, 1.0).`),
  );
}

/** Which component each accessor letter names, in both spellings. */
const COMPONENT_INDEX: Record<string, number> = {
  x: 0, y: 1, z: 2, w: 3,
  r: 0, g: 1, b: 2, a: 3,
};

/**
 * Resolve a chain of swizzles down to the variable underneath it.
 *
 * Only a variable can be assigned to. `a.xyz` is a value, so `a.xyz.xy = e`
 * has to become a write to `a` — and which components of `a` that is takes
 * composing the patterns: the outer pattern indexes into the inner one, so
 * `a.yzw.xy` selects the first two of y, z, w, which is `a.yz`.
 */
function resolveSwizzleTarget(
  target: any,
): { base: BaseNode<ShaderType>; pattern: string } {
  let pattern = target.value as string;
  let base = target.params![0];
  while (base?.type === "swizzle") {
    let inner = base.value as string;
    pattern = [...pattern].map(c => inner[COMPONENT_INDEX[c]]).join("");
    base = base.params![0];
  }
  return { base, pattern };
}

/**
 * Only a square matrix has an inverse, and neither language offers an overload
 * for the rest. Asked in both backends, so the two cannot come to different
 * answers about the same program — one emitting a call no driver accepts while
 * the other refuses it.
 *
 * Returns the matrix's size, which the WGSL side needs to pick its helper.
 */
function assertSquareMatrix(operandType: string | undefined): number {
  let shape = MATRIX_DIMENSIONS[operandType as string];
  if (shape === undefined || shape[0] !== shape[1]) {
    throw new Error(
      `[RMSL] inverse() needs a square matrix, but this one is `
      + `${operandType ?? "untyped"}.`,
    );
  }
  return shape[0];
}

/**
 * The position is the vertex stage's output. A fragment stage cannot read it:
 * GLSL's gl_Position is write-only there and WGSL has no such value at all.
 * Emitting it anyway produced an identifier neither backend declares.
 */
function assertPositionIsReadable(ctx: CompileCtx): void {
  if (ctx.shaderStage === "vertex") return;
  throw new Error(
    "[RMSL] builtinPosition() is the vertex stage's output position, and a "
    + "fragment stage cannot read it. Pass the value you need through a "
    + "varying() instead.",
  );
}

function compileGLSLStage(
  node: BaseNode<ShaderType> | ShaderType extends never ? never : any,
  ctx: CompileCtx,
): CompiledNode {
  if (node === undefined || node === null) {
    return { decls: [], body: [], expr: "0.0" };
  }
  if (typeof node === "boolean") {
    return { decls: [], body: [], expr: node ? "true" : "false" };
  }
  if (typeof node === "number") {
    return { decls: [], body: [], expr: node.toString() };
  }
  if (Array.isArray(node)) {
    return { decls: [], body: [], expr: `vec3(${node.join(", ")})` };
  }

  // Reached before: its statements are already in the output, so only the
  // expression naming the result is handed back. Emitting them again would
  // redeclare a variable, or run an assignment or a loop a second time.
  let seen = ctx.memo.get(node);
  if (seen) return { decls: [], body: [], expr: seen.expr, prec: seen.prec };

  let result = compileGLSLNode(node, ctx);
  ctx.memo.set(node, result);
  return result;
}

function compileGLSLNode(
  node: BaseNode<ShaderType> | ShaderType extends never ? never : any,
  ctx: CompileCtx,
): CompiledNode {
  // Constant folding
  let folded = tryFold(node);
  if (folded) node = folded;

  switch (node.type) {
    case "float": {
      let s = String(node.value);
      if (!s.includes('.') && !s.includes('e')) s += '.0';
      return { decls: [], body: [], expr: s };
    }
    case "int": return { decls: [], body: [], expr: String(node.value) };
    case "uint": return { decls: [], body: [], expr: String(node.value) + "u" };
    case "bool": return { decls: [], body: [], expr: node.value ? "true" : "false" };
    case "vec2": return { decls: [], body: [], expr: `vec2(${(node.value as number[]).join(", ")})` };
    case "vec3": return { decls: [], body: [], expr: `vec3(${(node.value as number[]).join(", ")})` };
    case "vec4": return { decls: [], body: [], expr: `vec4(${(node.value as number[]).join(", ")})` };
    case "ivec2": return { decls: [], body: [], expr: `ivec2(${(node.value as number[]).join(", ")})` };
    case "ivec3": return { decls: [], body: [], expr: `ivec3(${(node.value as number[]).join(", ")})` };
    case "ivec4": return { decls: [], body: [], expr: `ivec4(${(node.value as number[]).join(", ")})` };
    case "uvec2": return { decls: [], body: [], expr: `uvec2(${(node.value as number[]).map(v => `${v}u`).join(", ")})` };
    case "uvec3": return { decls: [], body: [], expr: `uvec3(${(node.value as number[]).map(v => `${v}u`).join(", ")})` };
    case "uvec4": return { decls: [], body: [], expr: `uvec4(${(node.value as number[]).map(v => `${v}u`).join(", ")})` };
    case "mat2": return { decls: [], body: [], expr: `mat2(${(node.value as number[]).join(", ")})` };
    case "mat2x3": return { decls: [], body: [], expr: `mat2x3(${(node.value as number[]).join(", ")})` };
    case "mat2x4": return { decls: [], body: [], expr: `mat2x4(${(node.value as number[]).join(", ")})` };
    case "mat3x2": return { decls: [], body: [], expr: `mat3x2(${(node.value as number[]).join(", ")})` };
    case "mat3": return { decls: [], body: [], expr: `mat3(${(node.value as number[]).join(", ")})` };
    case "mat3x4": return { decls: [], body: [], expr: `mat3x4(${(node.value as number[]).join(", ")})` };
    case "mat4x2": return { decls: [], body: [], expr: `mat4x2(${(node.value as number[]).join(", ")})` };
    case "mat4x3": return { decls: [], body: [], expr: `mat4x3(${(node.value as number[]).join(", ")})` };
    case "mat4": return { decls: [], body: [], expr: `mat4(${(node.value as number[]).join(", ")})` };
    case "void": return { decls: [], body: [], expr: "0.0" };

    case "construct": {
      let params = (node.params ?? []).map((p: any) => compileGLSLStage(p, ctx));
      let t = glslType(node._t as string);
      let args = params.map((p: any) => p.expr).join(", ");
      return {
        decls: params.flatMap((p: any) => p.decls),
        body: params.flatMap((p: any) => p.body),
        expr: `${t}(${args})`,
      };
    }

    case "var": {
      let varInfo = (node.value as any);
      let varName = varInfo?.varName;
      if (varName && !ctx.varDefs.has(varName)) {
        ctx.varDefs.set(varName, varInfo?.varType || "float");
      }
      return { decls: [], body: [], expr: varName };
    }

    case "uniform": {
      let v = node.value as any;
      if (!ctx.uniforms.has(v.id)) {
        ctx.uniforms.set(v.id, { type: glslType(v.shaderType), slot: v.slot });
      }
      return { decls: [], body: [], expr: v.slot };
    }

    case "uniformArray": {
      // Registered on first reference like any uniform; `length` makes the
      // declaration `uniform vec4 name[24];` rather than a single value.
      let v = node.value as any;
      if (!ctx.uniforms.has(v.id)) {
        ctx.uniforms.set(v.id, {
          type: glslType(v.shaderType),
          slot: v.slot,
          length: v.length,
        });
      }
      return { decls: [], body: [], expr: v.slot };
    }

    case "uniformArrayElement": {
      let arr = compileGLSLStage(node.params![0], ctx);
      let index = compileGLSLStage(node.params![1], ctx);
      // GLSL indexes with an int; a float loop counter has to be converted.
      let indexType = (node.params![1] as any)?._t;
      let indexExpr = indexType === "int" || indexType === "uint"
        ? index.expr
        : `int(${index.expr})`;
      return {
        decls: [...arr.decls, ...index.decls],
        body: [...arr.body, ...index.body],
        expr: `${arr.expr}[${indexExpr}]`,
      };
    }

    case "attribute": {
      let v = node.value as any;
      if (!ctx.attributes.has(v.id)) {
        ctx.attributes.set(v.id, { type: glslType(v.shaderType), slot: v.slot });
      }
      return { decls: [], body: [], expr: v.slot };
    }

    case "varying": {
      let v = node.value as any;
      if (!ctx.varyings.has(v.id)) {
        ctx.varyings.set(v.id, { type: glslType(v.shaderType), slot: v.slot });
      }
      return { decls: [], body: [], expr: v.slot };
    }

    case "output": {
      let v = node.value as any;
      if (!ctx.outputs.has(v.id)) {
        ctx.outputs.set(v.id, { type: glslType(v.shaderType), slot: v.slot, location: v.location });
      }
      return { decls: [], body: [], expr: v.slot };
    }

    case "builtinPosition": {
      assertPositionIsReadable(ctx);
      return { decls: [], body: [], expr: "gl_Position" };
    }

    case "builtinFragDepth": {
      if (ctx.shaderStage !== "fragment") {
        throw new Error("builtinFragDepth() can only be used in fragment shaders");
      }
      return { decls: [], body: [], expr: "gl_FragDepth" };
    }

    case "swizzle": {
      let src = compileGLSLStage(node.params![0], ctx);
      let pattern = node.value as string;
      let srcExpr = (src.prec ?? PREC_ATOM) < PREC_ATOM ? `(${src.expr})` : src.expr;
      return { decls: src.decls, body: src.body, expr: `${srcExpr}.${pattern}`, prec: PREC_ATOM };
    }

    case "negate": {
      let a = compileGLSLStage(node.params![0], ctx);
      let childExpr = wrapExpr(a.prec, PREC_UNARY, a.expr);
      return { decls: a.decls, body: a.body, expr: `-${childExpr}`, prec: PREC_UNARY };
    }
    case "not": {
      let a = compileGLSLStage(node.params![0], ctx);
      // GLSL's `!` takes a bool only; boolean vectors go through not().
      let operandType = (node.params![0] as any)?._t;
      let isBoolVector = operandType === "bvec2" || operandType === "bvec3" || operandType === "bvec4";
      if (isBoolVector) {
        return { decls: a.decls, body: a.body, expr: `not(${a.expr})`, prec: PREC_ATOM };
      }
      let childExpr = wrapExpr(a.prec, PREC_UNARY, a.expr);
      return { decls: a.decls, body: a.body, expr: `!${childExpr}`, prec: PREC_UNARY };
    }

    case "all": {
      let a = compileGLSLStage(node.params![0], ctx);
      return { decls: a.decls, body: a.body, expr: `all(${a.expr})` };
    }

    case "any": {
      let a = compileGLSLStage(node.params![0], ctx);
      return { decls: a.decls, body: a.body, expr: `any(${a.expr})` };
    }

    // Binary math ops (same pattern for all)
    case "add": return binaryGLSL(node, ctx, "+");
    case "sub": return binaryGLSL(node, ctx, "-");
    case "mult": return binaryGLSL(node, ctx, "*");
    case "div": return binaryGLSL(node, ctx, "/");
    case "mod": {
      // GLSL's % is integer-only; floats need the mod() builtin.
      let operandType = (node.params![0] as any)?._t;
      let isInteger = operandType === "int" || operandType === "uint";
      return isInteger
        ? binaryGLSL(node, ctx, "%")
        : binaryGLSL(node, ctx, "mod", true);
    }
    case "pow": return binaryGLSL(node, ctx, "pow", true);
    case "min": return binaryGLSL(node, ctx, "min", true);
    case "max": return binaryGLSL(node, ctx, "max", true);
    case "dot": return binaryGLSL(node, ctx, "dot", true);
    case "cross": return binaryGLSL(node, ctx, "cross", true);
    case "distance": return binaryGLSL(node, ctx, "distance", true);
    case "reflect": return binaryGLSL(node, ctx, "reflect", true);
    case "refract": return ternaryGLSL(node, ctx, "refract");
    case "mix": return ternaryGLSL(node, ctx, "mix");
    case "step": return binaryGLSL(node, ctx, "step", true);
    case "smoothstep": return ternaryGLSL(node, ctx, "smoothstep");
    case "clamp": return ternaryGLSL(node, ctx, "clamp");
    // Comparison ops
    case "lessThan": return comparisonGLSL(node, ctx, "<", "lessThan");
    case "greaterThan": return comparisonGLSL(node, ctx, ">", "greaterThan");
    case "lessThanEqual": return comparisonGLSL(node, ctx, "<=", "lessThanEqual");
    case "greaterThanEqual": return comparisonGLSL(node, ctx, ">=", "greaterThanEqual");
    case "equal": return comparisonGLSL(node, ctx, "==", "equal");
    case "notEqual": return comparisonGLSL(node, ctx, "!=", "notEqual");

    case "and": return binaryGLSL(node, ctx, "&&");
    case "or": return binaryGLSL(node, ctx, "||");
    case "bitAnd": return binaryGLSL(node, ctx, "&");
    case "bitOr": return binaryGLSL(node, ctx, "|");
    case "bitXor": return binaryGLSL(node, ctx, "^");
    case "shiftLeft": return binaryGLSL(node, ctx, "<<");
    case "shiftRight": return binaryGLSL(node, ctx, ">>");

    case "matVecMul": {
      let mat = compileGLSLStage(node.params![0], ctx);
      let vec = compileGLSLStage(node.params![1], ctx);
      let matType = (node.params![0] as any)?._t || "mat4";
      let vecType = (node.params![1] as any)?._t || "vec3";
      let prec = PRECEDENCE.mult;
      let matExpr = wrapExpr(mat.prec, prec, mat.expr);
      let vecExpr = wrapExpr(vec.prec, prec, vec.expr);
      if (matType === "mat4" && vecType === "vec3") {
        return {
          decls: [...mat.decls, ...vec.decls],
          body: [...mat.body, ...vec.body],
          expr: `(${matExpr} * vec4(${vec.expr}, 1.0)).xyz`,
          prec: PREC_ATOM,
        };
      }
      return {
        decls: [...mat.decls, ...vec.decls],
        body: [...mat.body, ...vec.body],
        expr: `${matExpr} * ${vecExpr}`,
        prec,
      };
    }

    // Unary math ops
    case "sin": return unaryGLSL(node, ctx, "sin");
    case "cos": return unaryGLSL(node, ctx, "cos");
    case "tan": return unaryGLSL(node, ctx, "tan");
    case "asin": return unaryGLSL(node, ctx, "asin");
    case "acos": return unaryGLSL(node, ctx, "acos");
    case "atan": return unaryGLSL(node, ctx, "atan");
    case "sinh": return unaryGLSL(node, ctx, "sinh");
    case "cosh": return unaryGLSL(node, ctx, "cosh");
    case "tanh": return unaryGLSL(node, ctx, "tanh");
    case "asinh": return unaryGLSL(node, ctx, "asinh");
    case "acosh": return unaryGLSL(node, ctx, "acosh");
    case "atanh": return unaryGLSL(node, ctx, "atanh");
    case "abs": return unaryGLSL(node, ctx, "abs");
    case "sign": return unaryGLSL(node, ctx, "sign");
    case "floor": return unaryGLSL(node, ctx, "floor");
    case "ceil": return unaryGLSL(node, ctx, "ceil");
    case "fract": return unaryGLSL(node, ctx, "fract");
    case "round": return unaryGLSL(node, ctx, "round");
    case "trunc": return unaryGLSL(node, ctx, "trunc");
    case "sqrt": return unaryGLSL(node, ctx, "sqrt");
    case "inversesqrt": return unaryGLSL(node, ctx, "inversesqrt");
    case "exp": return unaryGLSL(node, ctx, "exp");
    case "log": return unaryGLSL(node, ctx, "log");
    case "exp2": return unaryGLSL(node, ctx, "exp2");
    case "log2": return unaryGLSL(node, ctx, "log2");
    case "normalize": return unaryGLSL(node, ctx, "normalize");
    case "length": return unaryGLSL(node, ctx, "length");
    case "transpose": return unaryGLSL(node, ctx, "transpose");
    case "inverse":
      assertSquareMatrix((node.params![0] as any)?._t);
      return unaryGLSL(node, ctx, "inverse");
    case "determinant": return unaryGLSL(node, ctx, "determinant");
    case "fwidth": return unaryGLSL(node, ctx, "fwidth");
    case "dFdx": return unaryGLSL(node, ctx, "dFdx");
    case "dFdy": return unaryGLSL(node, ctx, "dFdy");
    case "faceForward": return binaryGLSL(node, ctx, "faceforward", true);
    case "bitNot": {
      let a = compileGLSLStage(node.params![0], ctx);
      let childExpr = wrapExpr(a.prec, PREC_UNARY, a.expr);
      return { decls: a.decls, body: a.body, expr: `~${childExpr}`, prec: PREC_UNARY };
    }

    case "matrixElement": {
      let mat = compileGLSLStage(node.params![0], ctx);
      let idx = compileGLSLStage(node.params![1], ctx);
      let idxExpr = idx.expr;
      let idxType = (node.params![1] as any)?._t || "float";
      if (idxType === "float") idxExpr = `int(${idxExpr})`;
      let matExpr = (mat.prec ?? PREC_ATOM) < PREC_ATOM ? `(${mat.expr})` : mat.expr;
      return {
        decls: [...mat.decls, ...idx.decls],
        body: [...mat.body, ...idx.body],
        expr: `${matExpr}[${idxExpr}]`,
        prec: PREC_ATOM,
      };
    }

    case "vectorElement": {
      let src = compileGLSLStage(node.params![0], ctx);
      let idx = compileGLSLStage(node.params![1], ctx);
      let idxExpr = idx.expr;
      let idxType = (node.params![1] as any)?._t || "float";
      if (idxType === "float") idxExpr = `int(${idxExpr})`;
      let srcExpr = (src.prec ?? PREC_ATOM) < PREC_ATOM ? `(${src.expr})` : src.expr;
      return {
        decls: [...src.decls, ...idx.decls],
        body: [...src.body, ...idx.body],
        expr: `${srcExpr}[${idxExpr}]`,
        prec: PREC_ATOM,
      };
    }

    case "texture": {
      // Float textures sample; integer textures are not filterable, so they are
      // fetched at texel coordinates with lod 0 — the coordinate convention the
      // WGSL backend uses too, keeping the two faithful to each other.
      let sampler = compileGLSLStage(node.params![0], ctx);
      let coords = compileGLSLStage(node.params![1], ctx);
      let samplerType = (node.params![0] as any)?._t || "sampler2D";
      let isIntegerSampler = samplerType.startsWith("isampler") || samplerType.startsWith("usampler");
      if (!isIntegerSampler) return binaryGLSL(node, ctx, "texture", true);
      let width = samplerType.endsWith("2D") ? 2 : 3;
      let idxExpr = coords.expr;
      let idxType = (node.params![1] as any)?._t || "float";
      if (idxType !== "int") idxExpr = `ivec${width}(${idxExpr})`;
      return {
        decls: [...sampler.decls, ...coords.decls],
        body: [...sampler.body, ...coords.body],
        expr: `texelFetch(${sampler.expr}, ${idxExpr}, 0)`,
        prec: PREC_ATOM,
      };
    }
    case "textureLod": {
      let sampler = compileGLSLStage(node.params![0], ctx);
      let coords = compileGLSLStage(node.params![1], ctx);
      let lod = compileGLSLStage(node.params![2], ctx);
      // An integer texture is not filterable, so GLSL samples it with texelFetch
      // at integer coordinates (and an explicit lod) rather than texture().
      let samplerType = (node.params![0] as any)?._t || "sampler2D";
      let isIntegerSampler = samplerType.startsWith("isampler") || samplerType.startsWith("usampler");
      if (!isIntegerSampler) {
        return {
          decls: [...sampler.decls, ...coords.decls, ...lod.decls],
          body: [...sampler.body, ...coords.body, ...lod.body],
          expr: `textureLod(${sampler.expr}, ${coords.expr}, ${lod.expr})`,
        };
      }
      let width = samplerType.endsWith("2D") ? 2 : 3;
      let lodExpr = lod.expr;
      let lodType = (node.params![2] as any)?._t || "float";
      if (lodType === "float") lodExpr = `int(${lodExpr})`;
      let idxExpr = coords.expr;
      let idxType = (node.params![1] as any)?._t || "float";
      if (idxType !== "int") idxExpr = `ivec${width}(${idxExpr})`;
      return {
        decls: [...sampler.decls, ...coords.decls, ...lod.decls],
        body: [...sampler.body, ...coords.body, ...lod.body],
        expr: `texelFetch(${sampler.expr}, ${idxExpr}, ${lodExpr})`,
      };
    }

    case "let": {
      let lhs = compileGLSLStage(node.params![0], ctx);
      let rhs = compileGLSLStage(node.params![1], ctx);
      let vt = (node.params![0] as any)._t || "float";
      let rhsType = (node.params![1] as any)?._t || "float";
      let t = glslType(vt);
      let rhsExpr = rhs.expr;
      if (vt === "float" && (rhsType === "int" || rhsType === "uint")) {
        rhsExpr = `float(${rhsExpr})`;
      }
      return {
        decls: [...lhs.decls, ...rhs.decls],
        body: [...lhs.body, ...rhs.body, `${t} ${lhs.expr} = ${rhsExpr};`],
        expr: lhs.expr,
      };
    }

    case "assign": {
      // An explicit write to the position tells the stage check that the
      // program has taken care of it.
      if ((node.params![0] as any)?.type === "builtinPosition") {
        ctx.positionWritten = true;
      }
      let lhs = compileGLSLStage(node.params![0], ctx);
      let rhs = compileGLSLStage(node.params![1], ctx);
      return {
        decls: [...lhs.decls, ...rhs.decls],
        body: [...lhs.body, ...rhs.body, `${lhs.expr} = ${rhs.expr};`],
        expr: lhs.expr,
      };
    }

    case "seq": {
      let params = node.params ?? [];
      let allDecls: string[] = [];
      let allBody: string[] = [];
      let expr = "0.0";
      for (let p of params) {
        let r = compileGLSLStage(p, ctx);
        allDecls.push(...r.decls);
        allBody.push(...r.body);
        expr = r.expr;
      }
      return { decls: allDecls, body: allBody, expr };
    }

    case "if": {
      let cond = compileGLSLStage(node.params![0], ctx);
      let body = compileGLSLStage(node.params![1], ctx);
      let elseBody = node.params!.length >= 3 && node.params![2] !== undefined
        ? compileGLSLStage(node.params![2], ctx)
        : { decls: [] as string[], body: [] as string[], expr: "" };
      let lines: string[] = [
        ...cond.body,
        `if (${cond.expr}) {`,
        ...body.body.map(l => "  " + l),
        "}",
      ];
      if (elseBody.body.length > 0) {
        lines.push("else {");
        lines.push(...elseBody.body.map(l => "  " + l));
        lines.push("}");
      }
      return {
        decls: [...cond.decls, ...body.decls, ...elseBody.decls],
        body: lines,
        expr: "0.0",
      };
    }

    case "for": {
      let init = compileGLSLStage(node.params![0], ctx);
      let cond = compileGLSLStage(node.params![1], ctx);
      let update = compileGLSLStage(node.params![2], ctx);
      let body = compileGLSLStage(node.params![3], ctx);
      let initExpr = init.expr;
      let initBody = init.body;
      if (init.body.length > 0) {
        let lastStmt = init.body[init.body.length - 1];
        if (lastStmt.endsWith(';')) {
          initExpr = lastStmt.slice(0, -1);
          initBody = init.body.slice(0, -1);
        }
      }
      return {
        decls: [...init.decls, ...cond.decls, ...update.decls, ...body.decls],
        body: [
          ...initBody,
          ...cond.body,
          `for (${initExpr}; ${cond.expr}; ${forUpdateStatements(update).map(withoutSemicolon).join(", ")}) {`,
          ...body.body.map(l => "  " + l),
          "}",
        ],
        expr: "0.0",
      };
    }

    case "while": {
      let cond = compileGLSLStage(node.params![0], ctx);
      let body = compileGLSLStage(node.params![1], ctx);
      return {
        decls: [...cond.decls, ...body.decls],
        body: [
          ...cond.body,
          `while (${cond.expr}) {`,
          ...body.body.map(l => "  " + l),
          "}",
        ],
        expr: "0.0",
      };
    }

    case "discard": {
      return { decls: [], body: ["discard;"], expr: "0.0" };
    }

    case "break": {
      return { decls: [], body: ["break;"], expr: "0.0" };
    }

    case "continue": {
      return { decls: [], body: ["continue;"], expr: "0.0" };
    }

    default:
      // Emitting a placeholder here would silently corrupt the shader: an
      // unhandled node becomes the literal 0.0 and the program still "compiles".
      // Every node type the public API can build has a case above, so reaching
      // this means the compiler lost one.
      throw new Error(`[RMSL] Unsupported node type in GLSL compiler: "${node.type}"`);
  }
}

function binaryGLSL(
  node: BaseNode<ShaderType>,
  ctx: CompileCtx,
  op: string,
  isFn?: boolean,
): CompiledNode {
  let lhs = compileGLSLStage(node.params![0], ctx);
  let rhs = compileGLSLStage(node.params![1], ctx);
  let lhsType = (node.params![0] as any)?._t || "float";
  let rhsType = (node.params![1] as any)?._t || "float";
  let lhsExpr = lhs.expr;
  let rhsExpr = rhs.expr;
  if (lhsType === "float" && (rhsType === "int" || rhsType === "uint")) {
    rhsExpr = `float(${rhsExpr})`;
  } else if ((lhsType === "int" || lhsType === "uint") && rhsType === "float") {
    lhsExpr = `float(${lhsExpr})`;
  } else if ((lhsType === "int" || lhsType === "uint") && (rhsType === "int" || rhsType === "uint")) {
    // GLSL would promote int+uint to uint, but the node is typed after its
    // first operand, so the other side is converted to match.
    if (lhsType === "int" && rhsType === "uint") rhsExpr = `int(${rhsExpr})`;
    if (lhsType === "uint" && rhsType === "int") lhsExpr = `uint(${lhsExpr})`;
  }
  if (isFn) {
    return {
      decls: [...lhs.decls, ...rhs.decls],
      body: [...lhs.body, ...rhs.body],
      expr: `${op}(${lhsExpr}, ${rhsExpr})`,
      prec: PREC_ATOM,
    };
  }
  let prec = PRECEDENCE[node.type] ?? 0;
  lhsExpr = wrapExpr(lhs.prec, prec, lhsExpr);
  rhsExpr = wrapExpr(rhs.prec, prec, rhsExpr);
  return {
    decls: [...lhs.decls, ...rhs.decls],
    body: [...lhs.body, ...rhs.body],
    expr: `${lhsExpr} ${op} ${rhsExpr}`,
    prec,
  };
}

function comparisonGLSL(
  node: BaseNode<ShaderType>,
  ctx: CompileCtx,
  op: string,
  fnName: string,
): CompiledNode {
  let a = compileGLSLStage(node.params![0], ctx);
  let b = compileGLSLStage(node.params![1], ctx);
  let lhsType = (node.params![0] as any)?._t || "float";
  let rhsType = (node.params![1] as any)?._t || "float";
  // Comparisons are per-component for any vector, float or integer.
  let isVec = (TYPE_WIDTH[lhsType] ?? 1) > 1;
  let lhsExpr = a.expr;
  let rhsExpr = b.expr;
  if (!isVec && lhsType === "float" && (rhsType === "int" || rhsType === "uint")) {
    rhsExpr = `float(${rhsExpr})`;
  } else if (!isVec && (lhsType === "int" || lhsType === "uint") && rhsType === "float") {
    lhsExpr = `float(${lhsExpr})`;
  }
  if (isVec) {
    return {
      decls: [...a.decls, ...b.decls],
      body: [...a.body, ...b.body],
      expr: `${fnName}(${lhsExpr}, ${rhsExpr})`,
      prec: PREC_ATOM,
    };
  }
  let prec = PRECEDENCE[node.type] ?? 0;
  lhsExpr = wrapExpr(a.prec, prec, lhsExpr);
  rhsExpr = wrapExpr(b.prec, prec, rhsExpr);
  return {
    decls: [...a.decls, ...b.decls],
    body: [...a.body, ...b.body],
    expr: `${lhsExpr} ${op} ${rhsExpr}`,
    prec,
  };
}

function ternaryGLSL(
  node: BaseNode<ShaderType>,
  ctx: CompileCtx,
  fn: string,
): { decls: string[]; body: string[]; expr: string } {
  let a = compileGLSLStage(node.params![0], ctx);
  let b = compileGLSLStage(node.params![1], ctx);
  let c = compileGLSLStage(node.params![2], ctx);
  let aType = (node.params![0] as any)?._t || "float";
  let bType = (node.params![1] as any)?._t || "float";
  let cType = (node.params![2] as any)?._t || "float";
  let aExpr = a.expr;
  let bExpr = b.expr;
  let cExpr = c.expr;
  if (aType === "float") {
    if (bType === "int" || bType === "uint") bExpr = `float(${bExpr})`;
    if (cType === "int" || cType === "uint") cExpr = `float(${cExpr})`;
  } else if (aType === "int" || aType === "uint") {
    if (bType === "float") aExpr = `float(${aExpr})`;
    if (cType === "float") { /* keep as-is or convert both */ }
  }
  return {
    decls: [...a.decls, ...b.decls, ...c.decls],
    body: [...a.body, ...b.body, ...c.body],
    expr: `${fn}(${aExpr}, ${bExpr}, ${cExpr})`,
  };
}

function unaryGLSL(
  node: BaseNode<ShaderType>,
  ctx: CompileCtx,
  fn: string,
): { decls: string[]; body: string[]; expr: string } {
  let a = compileGLSLStage(node.params![0], ctx);
  return {
    decls: a.decls,
    body: a.body,
    expr: `${fn}(${a.expr})`,
  };
}

function compileGLSLWithStage(
  root: Node<ShaderType> | readonly Node<ShaderType>[],
  shaderStage: "vertex" | "fragment",
): string {
  let ctx: CompileCtx = {
    nextId: 0,
    shaderStage,
    uniforms: new Map(),
    attributes: new Map(),
    varyings: new Map(),
    outputs: new Map(),
    wgslSamplers: new Map(),
    varDefs: new Map(),
    memo: new Map(),
    wgslHelpers: new Set(),
    positionWritten: false,
    inFn: false,
    fragDepthUsed: false,
  };

  let nodes = Array.isArray(root) ? root : [root];
  let results = nodes.map(n => compileGLSLStage(n, ctx));
  let allBody: string[] = [];
  let lastExpr = "0.0";
  // The stage output is a fixed type (vec4 for gl_Position and the implicit
  // fragment colour), so the final expression's type decides whether it can be
  // assigned there at all. Emitting it unchecked produces shaders that do not
  // compile — `gl_Position = <vec3>` and `result._rmsl_fragColor = <f32>`.
  let lastType: string | undefined;
  for (let i = 0; i < results.length; i++) {
    allBody.push(...results[i].decls, ...results[i].body);
    lastExpr = results[i].expr;
    lastType = (nodes[i] as any)?._t;
  }
  assertStageResult(shaderStage, lastType, ctx.positionWritten);
  // A vec4-typed node always has a value, so its type alone settles this. An
  // explicit write means the implicit one would be a second, conflicting
  // assignment.
  let hasVec4Result = lastType === "vec4" && !ctx.positionWritten;
  // A fragment stage that declares no output of its own still has to put its
  // colour somewhere, and GLSL ES 3.00 removed gl_FragColor, so an output is
  // declared for it.
  let emitImplicitColor =
    shaderStage === "fragment" && ctx.outputs.size === 0 && hasVec4Result;

  let lines: string[] = [];
  lines.push("#version 300 es");
  lines.push("precision highp float;");
  // No default precision covers every sampler. GLSL ES predeclares one for
  // sampler2D and samplerCube in a fragment stage, but not sampler3D, and the
  // vertex language predeclares none; Chromium's WebGL2 compiler rejects a
  // sampler with no precision at all ("No precision specified"). Each sampler
  // type a shader actually uses gets a high precision declared for the stages
  // that use it.
  let glslSamplerTypes = [...new Set(
    [...ctx.uniforms.values()]
      .map(info => info.type)
      .filter(t => /^(i|u)?sampler2D$|^(i|u)?sampler3D$|^(i|u)?samplerCube$/.test(t)),
  )].sort();
  for (let samplerType of glslSamplerTypes) {
    lines.push(`precision highp ${samplerType};`);
  }
  lines.push("");

  ctx.uniforms.forEach((info) => {
    lines.push(
      info.length !== undefined
        ? `uniform ${info.type} ${info.slot}[${info.length}];`
        : `uniform ${info.type} ${info.slot};`,
    );
  });
  ctx.attributes.forEach((info) => {
    lines.push(`in ${info.type} ${info.slot};`);
  });
  ctx.varyings.forEach((info) => {
    if (shaderStage === "vertex") {
      lines.push(`out ${info.type} ${info.slot};`);
    } else {
      lines.push(`in ${info.type} ${info.slot};`);
    }
  });
  // Numbered per shader, not from the id the output was declared with.
  let outputLocation = 0;
  ctx.outputs.forEach((info) => {
    if (info && info.slot && info.type) {
      // The qualifier names a draw buffer, which only a fragment stage has.
      // GLSL ES 3.00 rejects one on a vertex output, where the value is simply
      // another thing passed on to the fragment stage.
      lines.push(shaderStage === "fragment"
        ? `layout(location=${outputLocation++}) out ${info.type} ${info.slot};`
        : `out ${info.type} ${info.slot};`);
    }
  });
  if (emitImplicitColor) {
    lines.push("layout(location=0) out vec4 _rmsl_fragColor;");
  }
  if (
    ctx.uniforms.size > 0 || ctx.attributes.size > 0 || ctx.outputs.size > 0
    || emitImplicitColor
  ) {
    lines.push("");
  }

  if (shaderStage === "vertex") {
    lines.push("void main(void) {");
    for (let line of allBody) {
      lines.push("  " + line);
    }
    if (hasVec4Result) {
      lines.push(`  gl_Position = ${lastExpr};`);
    }
    lines.push("}");
  } else {
    lines.push("void main(void) {");
    for (let line of allBody) {
      lines.push("  " + line);
    }
    // Only the implicit output is written from the stage result. A declared
    // output belongs to the program, which assigns it itself — writing the
    // trailing expression into every declared slot ignored its type and
    // overwrote whatever the program had already put there.
    if (emitImplicitColor) {
      lines.push(`  _rmsl_fragColor = ${lastExpr};`);
    }
    lines.push("}");
  }
  return lines.join("\n");
}

/**
 * What a vertex stage may be handed.
 *
 * Its result becomes the position, so a vec4 is the ordinary case, and anything
 * else is refused here rather than at run time.
 *
 * Void is the other way to satisfy a vertex stage: assign builtinPosition()
 * yourself and return nothing. A function whose body returns nothing has that
 * type, so the two cases are exactly the two the signature admits. Whether an
 * assignment actually happened is not something a signature can see, so that
 * half stays a run-time check.
 *
 * Several values may be returned at once, of which the last becomes the
 * position — so that is the one constrained, and the values before it are
 * whatever the shader needed on the way there. Saying so requires knowing which
 * value is last, which is why `Fn` infers an array return as a tuple.
 */
export type VertexRoot =
  | Node<"vec4">
  | readonly [...Node<ShaderType>[], Node<"vec4">]
  | void;

export const compileGLSL: {
  (root: Node<ShaderType> | readonly Node<ShaderType>[]): string;
  vertex(root: VertexRoot): string;
  fragment(root: Node<ShaderType> | readonly Node<ShaderType>[]): string;
} = Object.assign(
  (root: Node<ShaderType> | readonly Node<ShaderType>[]) => compileGLSLWithStage(root, "fragment"),
  {
    // The value is always a node; void only describes a body that returned
    // nothing, which still compiles to one.
    vertex: (root: VertexRoot) => compileGLSLWithStage(root as Node<ShaderType>, "vertex"),
    fragment: (root: Node<ShaderType> | readonly Node<ShaderType>[]) => compileGLSLWithStage(root, "fragment"),
  },
);

// ========== WGSL Compiler ==========
let typeToWGSL: Record<string, string> = {
  float: "f32", vec2: "vec2<f32>", vec3: "vec3<f32>", vec4: "vec4<f32>",
  int: "i32", uint: "u32", bool: "bool",
  ivec2: "vec2<i32>", ivec3: "vec3<i32>", ivec4: "vec4<i32>",
  uvec2: "vec2<u32>", uvec3: "vec3<u32>", uvec4: "vec4<u32>",
  bvec2: "vec2<bool>", bvec3: "vec3<bool>", bvec4: "vec4<bool>",
  mat2: "mat2x2<f32>", mat2x3: "mat2x3<f32>", mat2x4: "mat2x4<f32>",
  mat3x2: "mat3x2<f32>", mat3: "mat3x3<f32>", mat3x4: "mat3x4<f32>",
  mat4x2: "mat4x2<f32>", mat4x3: "mat4x3<f32>", mat4: "mat4x4<f32>",
  sampler2D: "texture_2d<f32>", sampler3D: "texture_3d<f32>", samplerCube: "texture_cube<f32>",
  isampler2D: "texture_2d<i32>", isampler3D: "texture_3d<i32>", isamplerCube: "texture_cube<i32>",
  usampler2D: "texture_2d<u32>", usampler3D: "texture_3d<u32>", usamplerCube: "texture_cube<u32>",
  void: "void",
};

/**
 * `[columns, rows]` per matrix type. A GLSL/WGSL `matCxR` is C columns of R
 * rows, and the square names are the C === R shorthand.
 *
 * Every matrix type is listed.
 */
const MATRIX_DIMENSIONS: Record<string, [number, number]> = {
  mat2: [2, 2], mat2x3: [2, 3], mat2x4: [2, 4],
  mat3x2: [3, 2], mat3: [3, 3], mat3x4: [3, 4],
  mat4x2: [4, 2], mat4x3: [4, 3], mat4: [4, 4],
};

/**
 * Expand a single-scalar matrix constructor for WGSL.
 *
 * GLSL reads `mat4(1.0)` as a diagonal — the identity scaled by the scalar.
 * WGSL has no such overload and requires every component, so the one argument
 * is written out as the full diagonal, column by column.
 *
 * Only a *scalar* argument means a diagonal. A lone matrix argument is a copy
 * or truncation — `mat3(someMat4)` — which WGSL spells the same way GLSL does,
 * so it passes through. Expanding it instead produced
 * `mat3x3<f32>(m, 0f, 0f, 0f, m, ...)`, a constructor that does not exist.
 */
function wgslMatrixArgs(
  type: string,
  args: string[],
  sourceType: string | undefined,
): string[] {
  let shape = MATRIX_DIMENSIONS[type];
  if (shape === undefined || args.length !== 1) return args;
  if (TYPE_WIDTH[sourceType as string] !== 1) return args;
  let [columns, rows] = shape;
  let scalar = args[0];
  let out: string[] = [];
  for (let col = 0; col < columns; col++) {
    for (let row = 0; row < rows; row++) out.push(row === col ? scalar : "0f");
  }
  return out;
}

/** Struct type and binding name holding every uniform in a WGSL shader. */
const WGSL_UNIFORM_STRUCT = "_RmslUniforms";
const WGSL_UNIFORM_BINDING = "_rmsl_uniforms";

/** Byte size and alignment of each WGSL type, per the spec's layout rules. */
const WGSL_LAYOUT: Record<string, { size: number; align: number }> = {
  f32: { size: 4, align: 4 },
  i32: { size: 4, align: 4 },
  u32: { size: 4, align: 4 },
  "vec2<f32>": { size: 8, align: 8 },
  "vec3<f32>": { size: 12, align: 16 },
  "vec4<f32>": { size: 16, align: 16 },
  // The carriers. A bool is not host-shareable, so it travels as an unsigned
  // integer of the same width, and a narrow array element travels widened —
  // both of which arrive here as the type they are stored as.
  "vec2<u32>": { size: 8, align: 8 },
  "vec3<u32>": { size: 12, align: 16 },
  "vec4<u32>": { size: 16, align: 16 },
  "vec2<i32>": { size: 8, align: 8 },
  "vec3<i32>": { size: 12, align: 16 },
  "vec4<i32>": { size: 16, align: 16 },
  // A matCxR is C columns of vecR, and each column takes a whole multiple of
  // its own alignment — so a column of three floats occupies sixteen bytes,
  // not twelve.
  "mat2x2<f32>": { size: 16, align: 8 },
  "mat2x3<f32>": { size: 32, align: 16 },
  "mat2x4<f32>": { size: 32, align: 16 },
  "mat3x2<f32>": { size: 24, align: 8 },
  "mat3x3<f32>": { size: 48, align: 16 },
  "mat3x4<f32>": { size: 48, align: 16 },
  "mat4x2<f32>": { size: 32, align: 8 },
  "mat4x3<f32>": { size: 64, align: 16 },
  "mat4x4<f32>": { size: 64, align: 16 },
};

export interface WgslUniformMember {
  /** Generated slot name, matching the `name` on the uniform node. */
  name: string;
  /** Element type. For an array this is the element's type, not the array's. */
  type: string;
  /** Byte offset within the uniform buffer. */
  offset: number;
  /** Bytes occupied in total, so an array's whole extent rather than one element. */
  size: number;
  /** Element count, present only for a uniform array. */
  length?: number;
  /**
   * Bytes between consecutive elements, present only for a uniform array.
   *
   * Not the same as the element size: WGSL rounds the stride of an array in
   * the uniform address space up to 16, so `array<f32, 4>` spans 64 bytes with
   * each element alone in its own slot.
   */
  stride?: number;
}

/**
 * Element types that cannot be array members in WGSL's uniform address space,
 * and what to store instead.
 *
 * Elements there must be 16-byte aligned. Dawn accepts `array<vec3<f32>, N>`
 * — a vec3 aligns to 16 even though it occupies 12 — but rejects anything
 * narrower, so f32, i32, u32 and vec2 are widened to a four-component vector
 * and the value read back out of its leading components.
 *
 * The same approach TSL takes, where it is called the padded type.
 */
const WGSL_ARRAY_PADDING: Record<
  string,
  { stored: string; read: (element: string) => string }
> = {
  f32: { stored: "vec4<f32>", read: e => `${e}.x` },
  i32: { stored: "vec4<i32>", read: e => `${e}.x` },
  u32: { stored: "vec4<u32>", read: e => `${e}.x` },
  "vec2<f32>": { stored: "vec4<f32>", read: e => `${e}.xy` },
  "vec2<i32>": { stored: "vec4<i32>", read: e => `${e}.xy` },
  "vec2<u32>": { stored: "vec4<u32>", read: e => `${e}.xy` },
  // A bool is not host-shareable at all, so it travels as an unsigned integer
  // and is compared back, the same substitution a single bool uniform makes.
  // Reading is a comparison rather than a suffix, which is why these are
  // written as functions.
  bool: { stored: "vec4<u32>", read: e => `(${e}.x != 0u)` },
  "vec2<bool>": { stored: "vec4<u32>", read: e => `(${e}.xy != vec2<u32>(0u))` },
  "vec3<bool>": { stored: "vec4<u32>", read: e => `(${e}.xyz != vec3<u32>(0u))` },
  "vec4<bool>": { stored: "vec4<u32>", read: e => `(${e} != vec4<u32>(0u))` },
};

/** How a member is written in the struct: `array<T, N>` for arrays, else `T`. */
function wgslMemberType(m: WgslUniformMember): string {
  if (m.length === undefined) return m.type;
  const stored = WGSL_ARRAY_PADDING[m.type]?.stored ?? m.type;
  return `array<${stored}, ${m.length}>`;
}

/**
 * Place uniforms in one struct and report where each lands.
 *
 * WGSL caps uniform *buffers* at 12 per stage — the spec minimum, and what
 * real devices report — so a binding per uniform stops working at the
 * thirteenth. One struct is one binding no matter how many members, which is
 * how WebGPU code is written by hand.
 *
 * Members are ordered by descending alignment so the natural WGSL layout adds
 * no padding between them, and the offsets are returned because a caller
 * writing the buffer has no other way to know them.
 */
/**
 * Whether a uniform is a texture, which cannot go in the uniform address space
 * and keeps a binding of its own. Asked in both places that emit uniforms, so
 * the two cannot disagree about it.
 */
function isSamplerType(type: string): boolean {
  return /^(i|u)?sampler(2D|3D|Cube)$/.test(type);
}

function isWgslTexture(type: string): boolean {
  return type === "texture_2d<f32>" || type === "texture_3d<f32>" || type === "texture_cube<f32>"
    || type === "texture_2d<i32>" || type === "texture_3d<i32>" || type === "texture_cube<i32>"
    || type === "texture_2d<u32>" || type === "texture_3d<u32>" || type === "texture_cube<u32>";
}

export function wgslUniformLayout(
  members: { slot: string; type: string; length?: number }[],
): { members: WgslUniformMember[]; size: number } {
  // An array in the uniform address space has its element stride rounded up to
  // 16, so `array<f32, 4>` occupies 64 bytes rather than 16 — each element sits
  // in its own 16-byte slot. Callers writing the buffer need the stride, not
  // just the element size.
  const shapeOf = (m: { type: string; length?: number }) => {
    // An element too narrow to align is stored widened, so its footprint
    // follows what it is stored as rather than what it was declared as.
    const stored = m.length === undefined
      ? m.type
      : WGSL_ARRAY_PADDING[m.type]?.stored ?? m.type;
    const base = WGSL_LAYOUT[stored];
    // Guessing here is the worst thing this function could do. A wrong size is
    // not a shader that fails to build, it is one that reads whatever happens
    // to lie at that address, and the caller has no way to notice.
    if (base === undefined) {
      throw new Error(
        `[RMSL] no uniform layout is known for ${m.type}. Its size and`
        + ` alignment have to be added to WGSL_LAYOUT before it can be packed`
        + ` into a uniform buffer.`,
      );
    }
    if (m.length === undefined) return { ...base, stride: base.size };
    const stride = Math.ceil(base.size / 16) * 16;
    return { size: stride * m.length, align: Math.max(base.align, 16), stride };
  };

  // Widest alignment first, so the gaps between members stay small. Members
  // that align the same keep the order they were declared in.
  const ordered = members
    .map((m, declaredAt) => ({ m, declaredAt }))
    .sort((a, b) => {
      const byAlign = shapeOf(b.m).align - shapeOf(a.m).align;
      return byAlign !== 0 ? byAlign : a.declaredAt - b.declaredAt;
    })
    .map(({ m }) => m);

  const out: WgslUniformMember[] = [];
  let offset = 0;
  for (const m of ordered) {
    const { size, align, stride } = shapeOf(m);
    offset = Math.ceil(offset / align) * align;
    out.push({
      name: m.slot,
      type: m.type,
      offset,
      size,
      ...(m.length !== undefined ? { length: m.length, stride } : {}),
    });
    offset += size;
  }
  // A uniform struct is itself aligned to its largest member, and an array
  // member aligns to sixteen whatever it holds — so this has to ask for the
  // member's real alignment rather than its element type's, or the struct comes
  // out shorter than the buffer the shader reads.
  const structAlign = ordered.reduce((a, m) => Math.max(a, shapeOf(m).align), 4);
  return { members: out, size: Math.ceil(offset / structAlign) * structAlign };
}

function wgslType(brand: any): string {
  return typeToWGSL[brand as string] ?? "f32";
}

function compileWGSLStage(
  node: BaseNode<ShaderType> | any,
  ctx: CompileCtx,
): CompiledNode {
  if (node === undefined || node === null) {
    return { decls: [], body: [], expr: "0.0" };
  }
  if (typeof node === "boolean") {
    return { decls: [], body: [], expr: node ? "true" : "false" };
  }
  if (typeof node === "number") {
    return { decls: [], body: [], expr: Number.isInteger(node) ? `${node}i` : `${node}f` };
  }
  if (Array.isArray(node)) {
    return { decls: [], body: [], expr: `vec3<f32>(${node.join(", ")})` };
  }

  // Reached before: its statements are already in the output, so only the
  // expression naming the result is handed back. Emitting them again would
  // redeclare a variable, or run an assignment or a loop a second time.
  let seen = ctx.memo.get(node);
  if (seen) return { decls: [], body: [], expr: seen.expr, prec: seen.prec };

  let result = compileWGSLNode(node, ctx);
  ctx.memo.set(node, result);
  return result;
}

function compileWGSLNode(
  node: BaseNode<ShaderType> | any,
  ctx: CompileCtx,
): CompiledNode {
  // Constant folding
  let folded = tryFold(node);
  if (folded) node = folded;

  switch (node.type) {
    case "float": return { decls: [], body: [], expr: `${node.value}f` };
    case "int": return { decls: [], body: [], expr: `${node.value}i` };
    case "uint": return { decls: [], body: [], expr: `${node.value}u` };
    case "bool": return { decls: [], body: [], expr: node.value ? "true" : "false" };
    case "vec2": return { decls: [], body: [], expr: `vec2<f32>(${(node.value as number[]).join(", ")})` };
    case "vec3": return { decls: [], body: [], expr: `vec3<f32>(${(node.value as number[]).join(", ")})` };
    case "vec4": return { decls: [], body: [], expr: `vec4<f32>(${(node.value as number[]).join(", ")})` };
    case "ivec2": return { decls: [], body: [], expr: `vec2<i32>(${(node.value as number[]).map(v => `${v}i`).join(", ")})` };
    case "ivec3": return { decls: [], body: [], expr: `vec3<i32>(${(node.value as number[]).map(v => `${v}i`).join(", ")})` };
    case "ivec4": return { decls: [], body: [], expr: `vec4<i32>(${(node.value as number[]).map(v => `${v}i`).join(", ")})` };
    case "uvec2": return { decls: [], body: [], expr: `vec2<u32>(${(node.value as number[]).map(v => `${v}u`).join(", ")})` };
    case "uvec3": return { decls: [], body: [], expr: `vec3<u32>(${(node.value as number[]).map(v => `${v}u`).join(", ")})` };
    case "uvec4": return { decls: [], body: [], expr: `vec4<u32>(${(node.value as number[]).map(v => `${v}u`).join(", ")})` };
    case "mat2": return { decls: [], body: [], expr: `mat2x2<f32>(${(node.value as number[]).join(", ")})` };
    case "mat2x3": return { decls: [], body: [], expr: `mat2x3<f32>(${(node.value as number[]).join(", ")})` };
    case "mat2x4": return { decls: [], body: [], expr: `mat2x4<f32>(${(node.value as number[]).join(", ")})` };
    case "mat3x2": return { decls: [], body: [], expr: `mat3x2<f32>(${(node.value as number[]).join(", ")})` };
    case "mat3": return { decls: [], body: [], expr: `mat3x3<f32>(${(node.value as number[]).join(", ")})` };
    case "mat3x4": return { decls: [], body: [], expr: `mat3x4<f32>(${(node.value as number[]).join(", ")})` };
    case "mat4x2": return { decls: [], body: [], expr: `mat4x2<f32>(${(node.value as number[]).join(", ")})` };
    case "mat4x3": return { decls: [], body: [], expr: `mat4x3<f32>(${(node.value as number[]).join(", ")})` };
    case "mat4": return { decls: [], body: [], expr: `mat4x4<f32>(${(node.value as number[]).join(", ")})` };
    case "void": return { decls: [], body: [], expr: "0.0" };

    case "construct": {
      let params = (node.params ?? []).map((p: any) => compileWGSLStage(p, ctx));
      let t = wgslType(node._t as string);

      // GLSL truncates with vec3(someVec4); WGSL has no narrowing constructor,
      // so the components are selected explicitly.
      let target = TYPE_WIDTH[node._t as string];
      let sourceType = (node.params?.[0] as any)?._t;
      let source = TYPE_WIDTH[sourceType];
      if (
        params.length === 1 && target !== undefined && source !== undefined
        && source > target && target > 1
        && /^(vec|ivec|uvec|bvec)/.test(sourceType ?? "")
      ) {
        return {
          decls: params[0].decls,
          body: params[0].body,
          expr: `${params[0].expr}.${"xyzw".slice(0, target)}`,
        };
      }

      let args = wgslMatrixArgs(
        node._t as string,
        params.map((p: any) => p.expr),
        sourceType,
      ).join(", ");
      return {
        decls: params.flatMap((p: any) => p.decls),
        body: params.flatMap((p: any) => p.body),
        expr: `${t}(${args})`,
      };
    }

    case "var": {
      let varInfo = (node.value as any);
      let varName = varInfo?.varName;
      if (varName && !ctx.varDefs.has(varName)) {
        ctx.varDefs.set(varName, wgslType(varInfo?.varType || "float"));
      }
      return { decls: [], body: [], expr: varName };
    }

    case "uniform": {
      let v = node.value as any;
      // WGSL restricts the uniform address space to host-shareable types, and
      // neither bool nor a boolean vector is one. GLSL allows both, so they are
      // carried as unsigned integers and compared back on read — the difference
      // stays inside the compiler. The comparison is component-wise for a
      // vector, so it gives back a boolean vector of the same width.
      let width = TYPE_WIDTH[v?.shaderType] ?? 1;
      let isBoolean = v?.shaderType === "bool" || v?.shaderType?.startsWith("bvec");
      let carrier = width === 1 ? "u32" : `vec${width}<u32>`;
      let zero = width === 1 ? "0u" : `${carrier}(0u)`;
      if (v && v.id != null && !ctx.uniforms.has(v.id)) {
        ctx.uniforms.set(v.id, {
          type: isBoolean ? carrier : wgslType(v.shaderType),
          slot: v.slot,
        });
      }
      if (!v?.slot) return { decls: [], body: [], expr: "uniform<f32>" };
      // Value uniforms are members of one struct rather than a binding each, so
      // their references are qualified. Textures keep a binding of their own —
      // they cannot live in the uniform address space — and stay unqualified.
      let isTexture = isSamplerType(v.shaderType);
      let ref = isTexture ? v.slot : `${WGSL_UNIFORM_BINDING}.${v.slot}`;
      return {
        decls: [],
        body: [],
        expr: isBoolean ? `(${ref} != ${zero})` : ref,
      };
    }

    case "uniformArray": {
      let v = node.value as any;
      if (v && v.id != null && !ctx.uniforms.has(v.id)) {
        ctx.uniforms.set(v.id, {
          type: wgslType(v.shaderType),
          slot: v.slot,
          length: v.length,
        });
      }
      return { decls: [], body: [], expr: `${WGSL_UNIFORM_BINDING}.${v.slot}` };
    }

    case "uniformArrayElement": {
      let arr = compileWGSLStage(node.params![0], ctx);
      let index = compileWGSLStage(node.params![1], ctx);
      // WGSL indexes with i32 or u32; a float loop counter has to be converted.
      let indexType = (node.params![1] as any)?._t;
      let indexExpr = indexType === "int" || indexType === "uint"
        ? index.expr
        : `i32(${index.expr})`;
      // An element too narrow to align is stored widened, so the value is read
      // back out of the leading components — the padding never reaches the
      // caller, who asked for a float and gets a float.
      let elementType = wgslType((node.params![0] as any)?._t);
      let element = `${arr.expr}[${indexExpr}]`;
      let read = WGSL_ARRAY_PADDING[elementType]?.read;
      return {
        decls: [...arr.decls, ...index.decls],
        body: [...arr.body, ...index.body],
        expr: read ? read(element) : element,
      };
    }

    case "attribute": {
      let v = node.value as any;
      if (v && v.id != null && !ctx.attributes.has(v.id)) {
        ctx.attributes.set(v.id, { type: wgslType(v.shaderType), slot: v.slot });
      }
      return { decls: [], body: [], expr: ctx.shaderStage === "vertex" ? `input.${v.slot}` : v.slot };
    }

    case "varying": {
      let v = node.value as any;
      if (v && v.id != null && !ctx.varyings.has(v.id)) {
        ctx.varyings.set(v.id, { type: wgslType(v.shaderType), slot: v.slot });
      }
      let slot = v?.slot || "vec3<f32>(0.0, 0.0, 0.0)";
      let expr = ctx.shaderStage === "vertex" ? `result.${slot}` : slot;
      return { decls: [], body: [], expr };
    }

    case "output": {
      let v = node.value as any;
      if (v && v.id != null && !ctx.outputs.has(v.id)) {
        ctx.outputs.set(v.id, { type: wgslType(v.shaderType), slot: v.slot, location: v.location });
      }
      return { decls: [], body: [], expr: `result.${v?.slot}` };
    }

    case "builtinPosition": {
      // WGSL has no free-standing `position`; in a vertex stage it is a member
      // of the output struct.
      assertPositionIsReadable(ctx);
      return { decls: [], body: [], expr: "result.position" };
    }

    case "builtinFragDepth": {
      if (ctx.shaderStage !== "fragment") {
        throw new Error("builtinFragDepth() can only be used in fragment shaders");
      }
      ctx.fragDepthUsed = true;
      return { decls: [], body: [], expr: "result._rmsl_fragDepth" };
    }

    case "swizzle": {
      let src = compileWGSLStage(node.params![0], ctx);
      let pattern = node.value as string;
      let srcExpr = (src.prec ?? PREC_ATOM) < PREC_ATOM ? `(${src.expr})` : src.expr;
      return { decls: src.decls, body: src.body, expr: `${srcExpr}.${pattern}`, prec: PREC_ATOM };
    }

    case "negate": {
      let a = compileWGSLStage(node.params![0], ctx);
      let childExpr = wrapExpr(a.prec, PREC_UNARY, a.expr);
      return { decls: a.decls, body: a.body, expr: `-${childExpr}`, prec: PREC_UNARY };
    }
    case "not": {
      // Unlike GLSL, WGSL's `!` is defined for vecN<bool> too.
      let a = compileWGSLStage(node.params![0], ctx);
      let childExpr = wrapExpr(a.prec, PREC_UNARY, a.expr);
      return { decls: a.decls, body: a.body, expr: `!${childExpr}`, prec: PREC_UNARY };
    }

    case "all": {
      let a = compileWGSLStage(node.params![0], ctx);
      return { decls: a.decls, body: a.body, expr: `all(${a.expr})` };
    }

    case "any": {
      let a = compileWGSLStage(node.params![0], ctx);
      return { decls: a.decls, body: a.body, expr: `any(${a.expr})` };
    }

    case "add": return binaryWGSL(node, ctx, "+");
    case "sub": return binaryWGSL(node, ctx, "-");
    case "mult": return binaryWGSL(node, ctx, "*");
    case "div": return binaryWGSL(node, ctx, "/");
    case "mod": {
      let operandType = (node.params![0] as any)?._t;
      if (operandType === "int" || operandType === "uint") {
        return binaryWGSL(node, ctx, "%");
      }
      // WGSL's % truncates toward zero, GLSL's mod() floors, and floored is
      // what this operation is named for — the result takes the sign of the
      // divisor. A helper carries that, rather than the subtraction being
      // written inline: inline repeats both operands twice each, so an
      // expensive operand is evaluated four times. A call is an ordinary
      // expression, so it still fits wherever the operator did.
      let helper = `_rmsl_mod_${operandType}`;
      if (helper in WGSL_HELPERS) {
        ctx.wgslHelpers.add(helper);
        return binaryWGSL(node, ctx, helper, true);
      }
      return binaryWGSL(node, ctx, "%");
    }
    case "pow": return binaryWGSL(node, ctx, "pow", true);
    case "min": return binaryWGSL(node, ctx, "min", true);
    case "max": return binaryWGSL(node, ctx, "max", true);
    case "dot": return binaryWGSL(node, ctx, "dot", true);
    case "cross": return binaryWGSL(node, ctx, "cross", true);
    case "distance": return binaryWGSL(node, ctx, "distance", true);
    case "reflect": return binaryWGSL(node, ctx, "reflect", true);
    case "refract": return ternaryWGSL(node, ctx, "refract");
    case "mix": return ternaryWGSL(node, ctx, "mix");
    case "step": return binaryWGSL(node, ctx, "step", true);
    case "smoothstep": return ternaryWGSL(node, ctx, "smoothstep");
    case "clamp": return ternaryWGSL(node, ctx, "clamp");
    // Comparison ops
    case "lessThan": return binaryWGSL(node, ctx, "<");
    case "greaterThan": return binaryWGSL(node, ctx, ">");
    case "lessThanEqual": return binaryWGSL(node, ctx, "<=");
    case "greaterThanEqual": return binaryWGSL(node, ctx, ">=");
    case "equal": return binaryWGSL(node, ctx, "==");
    case "notEqual": return binaryWGSL(node, ctx, "!=");

    case "and": return binaryWGSL(node, ctx, "&&");
    case "or": return binaryWGSL(node, ctx, "||");
    case "bitAnd": return binaryWGSL(node, ctx, "&");
    case "bitOr": return binaryWGSL(node, ctx, "|");
    case "bitXor": return binaryWGSL(node, ctx, "^");
    // WGSL takes the shift amount as u32 even when the value shifted is i32,
    // so the right operand is converted. GLSL accepts either.
    case "shiftLeft": return shiftWGSL(node, ctx, "<<");
    case "shiftRight": return shiftWGSL(node, ctx, ">>");

    case "matVecMul": {
      let mat = compileWGSLStage(node.params![0], ctx);
      let vec = compileWGSLStage(node.params![1], ctx);
      let matType = (node.params![0] as any)?._t || "mat4";
      let vecType = (node.params![1] as any)?._t || "vec3";
      let prec = PRECEDENCE.mult;
      let matExpr = wrapExpr(mat.prec, prec, mat.expr);
      let vecExpr = wrapExpr(vec.prec, prec, vec.expr);
      if (matType === "mat4" && vecType === "vec3") {
        return {
          decls: [...mat.decls, ...vec.decls],
          body: [...mat.body, ...vec.body],
          expr: `(${matExpr} * vec4<f32>(${vec.expr}, 1.0)).xyz`,
          prec: PREC_ATOM,
        };
      }
      return {
        decls: [...mat.decls, ...vec.decls],
        body: [...mat.body, ...vec.body],
        expr: `${matExpr} * ${vecExpr}`,
        prec,
      };
    }

    case "sin": return unaryWGSL(node, ctx, "sin");
    case "cos": return unaryWGSL(node, ctx, "cos");
    case "tan": return unaryWGSL(node, ctx, "tan");
    case "asin": return unaryWGSL(node, ctx, "asin");
    case "acos": return unaryWGSL(node, ctx, "acos");
    case "atan": return unaryWGSL(node, ctx, "atan");
    case "sinh": return unaryWGSL(node, ctx, "sinh");
    case "cosh": return unaryWGSL(node, ctx, "cosh");
    case "tanh": return unaryWGSL(node, ctx, "tanh");
    case "asinh": return unaryWGSL(node, ctx, "asinh");
    case "acosh": return unaryWGSL(node, ctx, "acosh");
    case "atanh": return unaryWGSL(node, ctx, "atanh");
    case "abs": return unaryWGSL(node, ctx, "abs");
    case "sign": return unaryWGSL(node, ctx, "sign");
    case "floor": return unaryWGSL(node, ctx, "floor");
    case "ceil": return unaryWGSL(node, ctx, "ceil");
    case "fract": return unaryWGSL(node, ctx, "fract");
    case "round": return unaryWGSL(node, ctx, "round");
    case "trunc": return unaryWGSL(node, ctx, "trunc");
    case "sqrt": return unaryWGSL(node, ctx, "sqrt");
    case "inversesqrt": return unaryWGSL(node, ctx, "inverseSqrt");
    case "exp": return unaryWGSL(node, ctx, "exp");
    case "log": return unaryWGSL(node, ctx, "log");
    case "exp2": return unaryWGSL(node, ctx, "exp2");
    case "log2": return unaryWGSL(node, ctx, "log2");
    case "normalize": return unaryWGSL(node, ctx, "normalize");
    case "length": return unaryWGSL(node, ctx, "length");
    case "transpose": return unaryWGSL(node, ctx, "transpose");
    case "inverse": {
      // No inverse() builtin in WGSL, so one is written out per matrix size and
      // pulled in on demand.
      let operand = compileWGSLStage(node.params![0], ctx);
      let size = assertSquareMatrix((node.params![0] as any)?._t);
      let helper = `_rmsl_inverse${size}`;
      ctx.wgslHelpers.add(helper);
      return {
        decls: operand.decls,
        body: operand.body,
        expr: `${helper}(${operand.expr})`,
      };
    }
    case "determinant": return unaryWGSL(node, ctx, "determinant");
    case "fwidth": return unaryWGSL(node, ctx, "fwidth");
    case "dFdx": return unaryWGSL(node, ctx, "dpdx");
    case "dFdy": return unaryWGSL(node, ctx, "dpdy");
    case "faceForward": return binaryWGSL(node, ctx, "faceForward", true);
    case "bitNot": {
      let a = compileWGSLStage(node.params![0], ctx);
      let childExpr = wrapExpr(a.prec, PREC_UNARY, a.expr);
      return { decls: a.decls, body: a.body, expr: `~${childExpr}`, prec: PREC_UNARY };
    }

    case "matrixElement": {
      let mat = compileWGSLStage(node.params![0], ctx);
      let idx = compileWGSLStage(node.params![1], ctx);
      let idxExpr = idx.expr;
      let idxType = (node.params![1] as any)?._t || "float";
      if (idxType === "float") idxExpr = `i32(${idxExpr})`;
      let matExpr = (mat.prec ?? PREC_ATOM) < PREC_ATOM ? `(${mat.expr})` : mat.expr;
      return {
        decls: [...mat.decls, ...idx.decls],
        body: [...mat.body, ...idx.body],
        expr: `${matExpr}[${idxExpr}]`,
        prec: PREC_ATOM,
      };
    }

    case "vectorElement": {
      let src = compileWGSLStage(node.params![0], ctx);
      let idx = compileWGSLStage(node.params![1], ctx);
      let idxExpr = idx.expr;
      let idxType = (node.params![1] as any)?._t || "float";
      if (idxType === "float") idxExpr = `i32(${idxExpr})`;
      let srcExpr = (src.prec ?? PREC_ATOM) < PREC_ATOM ? `(${src.expr})` : src.expr;
      return {
        decls: [...src.decls, ...idx.decls],
        body: [...src.body, ...idx.body],
        expr: `${srcExpr}[${idxExpr}]`,
        prec: PREC_ATOM,
      };
    }

    case "texture":
    case "textureLod": {
      let samplerNode = node.params![0];
      let samplerCompiled = compileWGSLStage(samplerNode, ctx);
      let coords = compileWGSLStage(node.params![1], ctx);
      let samplerSlot = (samplerNode.value as any)?.slot;
      let samplerType = (samplerNode as any)?._t || "sampler2D";
      let isIntegerSampler = samplerType.startsWith("isampler") || samplerType.startsWith("usampler");
      // Integer textures are not filterable, so they are read with textureLoad,
      // which takes integer texel coordinates and needs no sampler binding.
      if (isIntegerSampler) {
        let width = samplerType.endsWith("2D") ? 2 : 3;
        let coordsType = (node.params![1] as any)?._t || "ivec2";
        let coordsExpr = coords.expr;
        if (coordsType !== `ivec${width}`) {
          coordsExpr = `vec${width}<i32>(${coordsExpr})`;
        }
        if (node.type === "texture") {
          return {
            decls: [...samplerCompiled.decls, ...coords.decls],
            body: [...samplerCompiled.body, ...coords.body],
            expr: `textureLoad(${samplerCompiled.expr}, ${coordsExpr}, 0i)`,
            prec: PREC_ATOM,
          };
        }
        let lod = compileWGSLStage(node.params![2], ctx);
        let lodExpr = lod.expr;
        let lodType = (node.params![2] as any)?._t || "float";
        if (lodType !== "int") lodExpr = `i32(${lodExpr})`;
        return {
          decls: [...samplerCompiled.decls, ...coords.decls, ...lod.decls],
          body: [...samplerCompiled.body, ...coords.body, ...lod.body],
          expr: `textureLoad(${samplerCompiled.expr}, ${coordsExpr}, ${lodExpr})`,
          prec: PREC_ATOM,
        };
      }
      if (samplerSlot && !ctx.wgslSamplers.has(samplerSlot)) {
        ctx.wgslSamplers.set(samplerSlot, {
          textureSlot: samplerSlot,
          samplerSlot: samplerSlot + "_s",
        });
      }
      let samplerName = samplerSlot ? samplerSlot + "_s" : "sampler";
      if (node.type === "texture") {
        return {
          decls: [...samplerCompiled.decls, ...coords.decls],
          body: [...samplerCompiled.body, ...coords.body],
          expr: `textureSample(${samplerCompiled.expr}, ${samplerName}, ${coords.expr})`,
        };
      } else {
        let lod = compileWGSLStage(node.params![2], ctx);
        return {
          decls: [...samplerCompiled.decls, ...coords.decls, ...lod.decls],
          body: [...samplerCompiled.body, ...coords.body, ...lod.body],
          expr: `textureSampleLevel(${samplerCompiled.expr}, ${samplerName}, ${coords.expr}, ${lod.expr})`,
        };
      }
    }

    case "let": {
      let lhs = compileWGSLStage(node.params![0], ctx);
      let rhs = compileWGSLStage(node.params![1], ctx);
      let vt = (node.params![0] as any)._t || "float";
      let t = wgslType(vt);
      let varName = (node.params![0] as any).varName || lhs.expr;
      ctx.varDefs.set(varName, t);
      return {
        decls: [...lhs.decls, ...rhs.decls],
        body: [...lhs.body, ...rhs.body, `var ${varName}: ${t} = ${rhs.expr};`],
        expr: varName,
      };
    }

    case "assign": {
      // An explicit write to the position tells the stage check that the
      // program has taken care of it.
      if ((node.params![0] as any)?.type === "builtinPosition") {
        ctx.positionWritten = true;
      }
      let rhs = compileWGSLStage(node.params![1], ctx);

      // WGSL only makes a single component assignable: `v.x = e` is a
      // reference, but a multi-component swizzle like `v.xy` is a value, so
      // `v.xy = e` is rejected. GLSL allows it, so the write is split into one
      // assignment per component. The right-hand side is bound to a temporary
      // first, otherwise an expression with side effects would run once per
      // component.
      let target = node.params![0] as any;

      // The swizzle itself is never compiled here — it would emit the very
      // `v.xy` form WGSL rejects, and any statements it produced would be
      // dropped along with it. The chain is resolved to the variable
      // underneath it, and the base compiled instead.
      if (target?.type === "swizzle") {
        let resolved = resolveSwizzleTarget(target);
        let base = compileWGSLStage(resolved.base, ctx);

        // A single component is directly assignable, so it needs no splitting.
        if (resolved.pattern.length === 1) {
          return {
            decls: [...base.decls, ...rhs.decls],
            body: [
              ...base.body,
              ...rhs.body,
              `${base.expr}.${resolved.pattern} = ${rhs.expr};`,
            ],
            expr: base.expr,
          };
        }

        let temp = `_rmsl_sw${ctx.nextId++}`;
        let rhsType = wgslType((node.params![1] as any)?._t ?? "float");
        let lines = [
          ...base.body,
          ...rhs.body,
          `var ${temp}: ${rhsType} = ${rhs.expr};`,
          ...[...resolved.pattern].map(
            (component, i) => `${base.expr}.${component} = ${temp}[${i}];`,
          ),
        ];
        return {
          decls: [...base.decls, ...rhs.decls],
          body: lines,
          expr: base.expr,
        };
      }

      let lhs = compileWGSLStage(node.params![0], ctx);
      return {
        decls: [...lhs.decls, ...rhs.decls],
        body: [...lhs.body, ...rhs.body, `${lhs.expr} = ${rhs.expr};`],
        expr: lhs.expr,
      };
    }

    case "seq": {
      let params = node.params ?? [];
      let allDecls: string[] = [];
      let allBody: string[] = [];
      let expr = "0.0";
      for (let p of params) {
        let r = compileWGSLStage(p, ctx);
        allDecls.push(...r.decls);
        allBody.push(...r.body);
        expr = r.expr;
      }
      return { decls: allDecls, body: allBody, expr };
    }

    case "if": {
      let cd = compileWGSLStage(node.params![0], ctx);
      let body = compileWGSLStage(node.params![1], ctx);
      let elseBody = node.params!.length >= 3 && node.params![2] !== undefined
        ? compileWGSLStage(node.params![2], ctx)
        : { decls: [] as string[], body: [] as string[], expr: "" };
      let lines: string[] = [
        ...cd.body,
        `if (${cd.expr}) {`,
        ...body.body.map(l => "  " + l),
        "}",
      ];
      if (elseBody.body.length > 0) {
        lines.push("else {");
        lines.push(...elseBody.body.map(l => "  " + l));
        lines.push("}");
      }
      return {
        decls: [...cd.decls, ...body.decls, ...elseBody.decls],
        body: lines,
        expr: "0.0",
      };
    }

    case "for": {
      let init = compileWGSLStage(node.params![0], ctx);
      let cd = compileWGSLStage(node.params![1], ctx);
      let update = compileWGSLStage(node.params![2], ctx);
      let body = compileWGSLStage(node.params![3], ctx);
      let initExpr = init.expr;
      let initBody = init.body;
      if (init.body.length > 0) {
        let lastStmt = init.body[init.body.length - 1];
        if (lastStmt.endsWith(';')) {
          let converted = lastStmt.slice(0, -1);
          // WGSL for-init needs var not let (skip let prefix)
          initExpr = converted;
          initBody = init.body.slice(0, -1);
        }
      }
      let updates = forUpdateStatements(update);
      let decls = [...init.decls, ...cd.decls, ...update.decls, ...body.decls];

      // WGSL's for-header holds a single update statement. More than one goes
      // in a continuing block instead, which runs after the body on every
      // iteration — including after a continue, which appending them to the
      // body would not.
      if (updates.length > 1) {
        return {
          decls,
          body: [
            ...initBody,
            "{",
            `  ${initExpr};`,
            "  loop {",
            ...cd.body.map(l => "    " + l),
            `    if (!(${cd.expr})) { break; }`,
            ...body.body.map(l => "    " + l),
            "    continuing {",
            ...updates.map(l => "      " + l),
            "    }",
            "  }",
            "}",
          ],
          expr: "0.0",
        };
      }

      let header = updates.length === 1 ? withoutSemicolon(updates[0]) : "";
      return {
        decls,
        body: [
          ...initBody,
          `for (${initExpr}; ${cd.expr}; ${header}) {`,
          ...body.body.map(l => "  " + l),
          "}",
        ],
        expr: "0.0",
      };
    }

    case "while": {
      let cd = compileWGSLStage(node.params![0], ctx);
      let body = compileWGSLStage(node.params![1], ctx);
      return {
        decls: [...cd.decls, ...body.decls],
        body: [
          ...cd.body,
          `while (${cd.expr}) {`,
          ...body.body.map(l => "  " + l),
          "}",
        ],
        expr: "0.0",
      };
    }

    case "discard": {
      return { decls: [], body: ["discard;"], expr: "0.0" };
    }

    case "break": {
      return { decls: [], body: ["break;"], expr: "0.0" };
    }

    case "continue": {
      return { decls: [], body: ["continue;"], expr: "0.0" };
    }

    default:
      // Emitting a placeholder here would silently corrupt the shader: an
      // unhandled node becomes the literal 0.0 and the program still "compiles".
      // Every node type the public API can build has a case above, so reaching
      // this means the compiler lost one.
      throw new Error(`[RMSL] Unsupported node type in WGSL compiler: "${node.type}"`);
  }
}

/**
 * WGSL helper functions, emitted only when a shader uses them.
 *
 * GLSL has `inverse()` as a builtin and WGSL does not, so the matrix inverses
 * are written out here — cofactor expansion over a column-major matrix, the
 * same formulation as the mat4Inverse used on the JS side.
 */
const WGSL_HELPERS: Record<string, string> = {
  // A floored modulus, which is what GLSL's mod() computes and what WGSL's %
  // does not. One per width, because the operands are broadcast to match.
  _rmsl_mod_float: `fn _rmsl_mod_float(x: f32, y: f32) -> f32 {
  return x - y * floor(x / y);
}`,
  _rmsl_mod_vec2: `fn _rmsl_mod_vec2(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
  return x - y * floor(x / y);
}`,
  _rmsl_mod_vec3: `fn _rmsl_mod_vec3(x: vec3<f32>, y: vec3<f32>) -> vec3<f32> {
  return x - y * floor(x / y);
}`,
  _rmsl_mod_vec4: `fn _rmsl_mod_vec4(x: vec4<f32>, y: vec4<f32>) -> vec4<f32> {
  return x - y * floor(x / y);
}`,
  _rmsl_inverse2: `fn _rmsl_inverse2(m: mat2x2<f32>) -> mat2x2<f32> {
  let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
  let inv = 1.0 / det;
  return mat2x2<f32>(
    vec2<f32>(m[1][1] * inv, -m[0][1] * inv),
    vec2<f32>(-m[1][0] * inv, m[0][0] * inv),
  );
}`,
  _rmsl_inverse3: `fn _rmsl_inverse3(m: mat3x3<f32>) -> mat3x3<f32> {
  let a00 = m[0][0]; let a01 = m[0][1]; let a02 = m[0][2];
  let a10 = m[1][0]; let a11 = m[1][1]; let a12 = m[1][2];
  let a20 = m[2][0]; let a21 = m[2][1]; let a22 = m[2][2];
  let b01 = a22 * a11 - a12 * a21;
  let b11 = -a22 * a10 + a12 * a20;
  let b21 = a21 * a10 - a11 * a20;
  let det = a00 * b01 + a01 * b11 + a02 * b21;
  let inv = 1.0 / det;
  return mat3x3<f32>(
    vec3<f32>(b01 * inv, (-a22 * a01 + a02 * a21) * inv, (a12 * a01 - a02 * a11) * inv),
    vec3<f32>(b11 * inv, (a22 * a00 - a02 * a20) * inv, (-a12 * a00 + a02 * a10) * inv),
    vec3<f32>(b21 * inv, (-a21 * a00 + a01 * a20) * inv, (a11 * a00 - a01 * a10) * inv),
  );
}`,
  _rmsl_inverse4: `fn _rmsl_inverse4(m: mat4x4<f32>) -> mat4x4<f32> {
  let a00 = m[0][0]; let a01 = m[0][1]; let a02 = m[0][2]; let a03 = m[0][3];
  let a10 = m[1][0]; let a11 = m[1][1]; let a12 = m[1][2]; let a13 = m[1][3];
  let a20 = m[2][0]; let a21 = m[2][1]; let a22 = m[2][2]; let a23 = m[2][3];
  let a30 = m[3][0]; let a31 = m[3][1]; let a32 = m[3][2]; let a33 = m[3][3];
  let b00 = a00 * a11 - a01 * a10;
  let b01 = a00 * a12 - a02 * a10;
  let b02 = a00 * a13 - a03 * a10;
  let b03 = a01 * a12 - a02 * a11;
  let b04 = a01 * a13 - a03 * a11;
  let b05 = a02 * a13 - a03 * a12;
  let b06 = a20 * a31 - a21 * a30;
  let b07 = a20 * a32 - a22 * a30;
  let b08 = a20 * a33 - a23 * a30;
  let b09 = a21 * a32 - a22 * a31;
  let b10 = a21 * a33 - a23 * a31;
  let b11 = a22 * a33 - a23 * a32;
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  let inv = 1.0 / det;
  return mat4x4<f32>(
    vec4<f32>((a11 * b11 - a12 * b10 + a13 * b09) * inv,
              (-a01 * b11 + a02 * b10 - a03 * b09) * inv,
              (a31 * b05 - a32 * b04 + a33 * b03) * inv,
              (-a21 * b05 + a22 * b04 - a23 * b03) * inv),
    vec4<f32>((-a10 * b11 + a12 * b08 - a13 * b07) * inv,
              (a00 * b11 - a02 * b08 + a03 * b07) * inv,
              (-a30 * b05 + a32 * b02 - a33 * b01) * inv,
              (a20 * b05 - a22 * b02 + a23 * b01) * inv),
    vec4<f32>((a10 * b10 - a11 * b08 + a13 * b06) * inv,
              (-a00 * b10 + a01 * b08 - a03 * b06) * inv,
              (a30 * b04 - a31 * b02 + a33 * b00) * inv,
              (-a20 * b04 + a21 * b02 - a23 * b00) * inv),
    vec4<f32>((-a10 * b09 + a11 * b07 - a12 * b06) * inv,
              (a00 * b09 - a01 * b07 + a02 * b06) * inv,
              (-a30 * b03 + a31 * b01 - a32 * b00) * inv,
              (a20 * b03 - a21 * b01 + a22 * b00) * inv),
  );
}`,
};

/**
 * A WGSL shift. The value keeps its own type, but the shift amount must be
 * u32 — `i32 << i32` has no overload — so the right operand is converted when
 * it is not already unsigned.
 */
function shiftWGSL(
  node: BaseNode<ShaderType>,
  ctx: CompileCtx,
  op: string,
): CompiledNode {
  let lhs = compileWGSLStage(node.params![0], ctx);
  let rhs = compileWGSLStage(node.params![1], ctx);
  let amountType = (node.params![1] as any)?._t;
  let rhsExpr = amountType === "uint" ? rhs.expr : `u32(${rhs.expr})`;
  let prec = PRECEDENCE[node.type] ?? 0;
  let lhsExpr = wrapExpr(lhs.prec, prec, lhs.expr);
  rhsExpr = wrapExpr(rhs.prec, prec, rhsExpr);
  return {
    decls: [...lhs.decls, ...rhs.decls],
    body: [...lhs.body, ...rhs.body],
    expr: `${lhsExpr} ${op} ${rhsExpr}`,
    prec,
  };
}

function binaryWGSL(
  node: BaseNode<ShaderType>,
  ctx: CompileCtx,
  op: string,
  isFn?: boolean,
): CompiledNode {
  let lhs = compileWGSLStage(node.params![0], ctx);
  let rhs = compileWGSLStage(node.params![1], ctx);
  let lhsType = (node.params![0] as any)?._t || "float";
  let rhsType = (node.params![1] as any)?._t || "float";
  let rhsExpr = rhs.expr;
  let lhsExpr = lhs.expr;
  if (lhsType !== rhsType) {
    if (lhsType === "float" && (rhsType === "int" || rhsType === "uint")) {
      rhsExpr = `f32(${rhs.expr})`;
    } else if ((lhsType === "int" || lhsType === "uint") && rhsType === "float") {
      // The node's type follows its first operand, so the float side is
      // converted to match rather than the result promoted to float.
      rhsExpr = lhsType === "int" ? `i32(${rhs.expr})` : `u32(${rhs.expr})`;
    } else if ((lhsType === "int" || lhsType === "uint") && (rhsType === "int" || rhsType === "uint")) {
      // WGSL has no mixed signed/unsigned arithmetic; convert to the type the
      // node is declared as (its first operand's).
      rhsExpr = lhsType === "int" ? `i32(${rhs.expr})` : `u32(${rhs.expr})`;
    }
  }
  if (isFn) {
    return {
      decls: [...lhs.decls, ...rhs.decls],
      body: [...lhs.body, ...rhs.body],
      expr: `${op}(${lhsExpr}, ${rhsExpr})`,
      prec: PREC_ATOM,
    };
  }
  let prec = PRECEDENCE[node.type] ?? 0;
  lhsExpr = wrapExpr(lhs.prec, prec, lhsExpr);
  rhsExpr = wrapExpr(rhs.prec, prec, rhsExpr);
  return {
    decls: [...lhs.decls, ...rhs.decls],
    body: [...lhs.body, ...rhs.body],
    expr: `${lhsExpr} ${op} ${rhsExpr}`,
    prec,
  };
}

function ternaryWGSL(
  node: BaseNode<ShaderType>,
  ctx: CompileCtx,
  fn: string,
): { decls: string[]; body: string[]; expr: string } {
  let a = compileWGSLStage(node.params![0], ctx);
  let b = compileWGSLStage(node.params![1], ctx);
  let c = compileWGSLStage(node.params![2], ctx);
  let aType = (node.params![0] as any)?._t || "float";
  let bType = (node.params![1] as any)?._t || "float";
  let cType = (node.params![2] as any)?._t || "float";
  let aExpr = a.expr;
  let bExpr = b.expr;
  let cExpr = c.expr;
  if (aType === "float") {
    if (bType === "int" || bType === "uint") bExpr = `f32(${bExpr})`;
    if (cType === "int" || cType === "uint") cExpr = `f32(${cExpr})`;
  }
  return {
    decls: [...a.decls, ...b.decls, ...c.decls],
    body: [...a.body, ...b.body, ...c.body],
    expr: `${fn}(${aExpr}, ${bExpr}, ${cExpr})`,
  };
}

function unaryWGSL(
  node: BaseNode<ShaderType>,
  ctx: CompileCtx,
  fn: string,
): { decls: string[]; body: string[]; expr: string } {
  let a = compileWGSLStage(node.params![0], ctx);
  return {
    decls: a.decls,
    body: a.body,
    expr: `${fn}(${a.expr})`,
  };
}

function compileWGSLWithStage(
  root: Node<ShaderType> | readonly Node<ShaderType>[],
  shaderStage: "vertex" | "fragment",
): string {
  let ctx: CompileCtx = {
    nextId: 0,
    shaderStage,
    uniforms: new Map(),
    attributes: new Map(),
    varyings: new Map(),
    outputs: new Map(),
    wgslSamplers: new Map(),
    varDefs: new Map(),
    memo: new Map(),
    wgslHelpers: new Set(),
    positionWritten: false,
    inFn: false,
    fragDepthUsed: false,
  };

  let nodes = Array.isArray(root) ? root : [root];
  let results = nodes.map(n => compileWGSLStage(n, ctx));
  let allBody: string[] = [];
  let lastExpr = "0.0";
  // The stage output is a fixed type (vec4 for gl_Position and the implicit
  // fragment colour), so the final expression's type decides whether it can be
  // assigned there at all. Emitting it unchecked produces shaders that do not
  // compile — `gl_Position = <vec3>` and `result._rmsl_fragColor = <f32>`.
  let lastType: string | undefined;
  for (let i = 0; i < results.length; i++) {
    allBody.push(...results[i].decls, ...results[i].body);
    lastExpr = results[i].expr;
    lastType = (nodes[i] as any)?._t;
  }
  assertStageResult(shaderStage, lastType, ctx.positionWritten);
  // A vec4-typed node always has a value, so its type alone settles this. An
  // explicit write means the implicit one would be a second, conflicting
  // assignment.
  let hasVec4Result = lastType === "vec4" && !ctx.positionWritten;

  let lines: string[] = [];
  let texBinding = 0;
  let samplerBinding = 0;
  let sortedUniforms = [...ctx.uniforms.entries()].sort((a, b) => a[1].slot.localeCompare(b[1].slot));

  // Textures keep their own bindings; everything else goes in one struct,
  // because WGSL allows only 12 uniform buffers per stage.
  let textures = sortedUniforms.filter(([, i]) => isWgslTexture(i.type));
  let plain = sortedUniforms.filter(([, i]) => !isWgslTexture(i.type));

  for (let [, info] of textures) {
    lines.push(`@group(1) @binding(${texBinding++}) var ${info.slot}: ${info.type};`);
  }
  if (plain.length > 0) {
    let layout = wgslUniformLayout(
      plain.map(([, i]) => ({ slot: i.slot, type: i.type, length: i.length })),
    );
    lines.push(`struct ${WGSL_UNIFORM_STRUCT} {`);
    for (let m of layout.members) lines.push(`  ${m.name}: ${wgslMemberType(m)},`);
    lines.push("};");
    lines.push(`@group(0) @binding(0) var<uniform> ${WGSL_UNIFORM_BINDING}: ${WGSL_UNIFORM_STRUCT};`);
  }
  ctx.wgslSamplers.forEach((info) => {
    lines.push(`@group(2) @binding(${samplerBinding++}) var ${info.samplerSlot}: sampler;`);
  });
  if (ctx.uniforms.size > 0 || ctx.wgslSamplers.size > 0 || ctx.outputs.size > 0) {
    lines.push("");
  }

  // Helpers standing in for GLSL builtins WGSL lacks. Sorted so identical
  // shaders produce identical source regardless of the order ops were reached.
  for (const helper of [...ctx.wgslHelpers].sort()) {
    lines.push(WGSL_HELPERS[helper], "");
  }

  if (shaderStage === "vertex") {
    if (ctx.attributes.size > 0) {
      lines.push("struct VertexInput {");
      let attrLoc = 0;
      ctx.attributes.forEach((info) => {
        lines.push(`  @location(${attrLoc++}) ${info.slot}: ${info.type},`);
      });
      lines.push("};");
      lines.push("");
    }
    lines.push("struct VertexOutput {");
    lines.push("  @builtin(position) position: vec4<f32>,");
    // Everything the stage passes on shares one set of slots, because they all
    // become members of this one structure. Numbering varyings and declared
    // outputs separately handed the same slot to one of each.
    let outgoingLocation = 0;
    let sortedVaryings = [...ctx.varyings.entries()].sort((a, b) => a[1].slot.localeCompare(b[1].slot));
    for (let [, info] of sortedVaryings) {
      lines.push(`  @location(${outgoingLocation++}) ${info.slot}: ${info.type},`);
    }
    ctx.outputs.forEach((info) => {
      if (info && info.slot && info.type) {
        lines.push(`  @location(${outgoingLocation++}) ${info.slot}: ${info.type},`);
      }
    });
    lines.push("};");
    lines.push("");
    lines.push("@vertex");
    if (ctx.attributes.size > 0) {
      lines.push("fn main(input: VertexInput) -> VertexOutput {");
    } else {
      lines.push("fn main() -> VertexOutput {");
    }
    lines.push("  var result: VertexOutput;");
    for (let line of allBody) {
      lines.push("  " + line);
    }
    if (hasVec4Result) {
      lines.push(`  result.position = ${lastExpr};`);
    }
    lines.push("  return result;");
    lines.push("}");
  } else {
    // A fragment stage only gets a return struct when it has something to put
    // in it. WGSL forbids empty structs, so a shader with no declared output
    // and a non-vec4 result becomes a plain `@fragment fn main()` that returns
    // nothing — the WGSL equivalent of the GLSL branch emitting no assignment.
    let emitImplicitColor = ctx.outputs.size === 0 && hasVec4Result;
    let hasFragmentOutput = ctx.outputs.size > 0 || emitImplicitColor || ctx.fragDepthUsed;

    if (hasFragmentOutput) {
      lines.push("struct FragmentOutput {");
      // Numbered per shader. The implicit colour below takes location 0, and
      // only exists when there are no declared outputs, so the two cannot clash.
      let fragmentOutputLocation = 0;
      ctx.outputs.forEach((info) => {
        if (info && info.slot && info.type) {
          lines.push(`  @location(${fragmentOutputLocation++}) ${info.slot}: ${info.type},`);
        }
      });
      if (emitImplicitColor) {
        lines.push("  @location(0) _rmsl_fragColor: vec4<f32>,");
      }
      if (ctx.fragDepthUsed) {
        lines.push("  @builtin(frag_depth) _rmsl_fragDepth: f32,");
      }
      lines.push("};");
      lines.push("");
    }

    lines.push("@fragment");
    let fragParams = "";
    let fragVaryingLoc = 0;
    let sortedFVaryings = [...ctx.varyings.entries()].sort((a, b) => a[1].slot.localeCompare(b[1].slot));
    for (let [, info] of sortedFVaryings) {
      if (fragParams) fragParams += ", ";
      fragParams += `@location(${fragVaryingLoc++}) ${info.slot}: ${info.type}`;
    }
    lines.push(`fn main(${fragParams})${hasFragmentOutput ? " -> FragmentOutput" : ""} {`);
    if (hasFragmentOutput) {
      lines.push("  var result: FragmentOutput;");
    }
    for (let line of allBody) {
      lines.push("  " + line);
    }
    if (emitImplicitColor) {
      lines.push(`  result._rmsl_fragColor = ${lastExpr};`);
    }
    if (ctx.fragDepthUsed && !allBody.some(l => l.includes("_rmsl_fragDepth ="))) {
      lines.push("  result._rmsl_fragDepth = 1.0;");
    }
    if (hasFragmentOutput) {
      lines.push("  return result;");
    }
    lines.push("}");
  }
  return lines.join("\n");
}

export const compileWGSL: {
  (root: Node<ShaderType> | readonly Node<ShaderType>[]): string;
  vertex(root: VertexRoot): string;
  fragment(root: Node<ShaderType> | readonly Node<ShaderType>[]): string;
} = Object.assign(
  (root: Node<ShaderType> | readonly Node<ShaderType>[]) => compileWGSLWithStage(root, "fragment"),
  {
    vertex: (root: VertexRoot) => compileWGSLWithStage(root as Node<ShaderType>, "vertex"),
    fragment: (root: Node<ShaderType> | readonly Node<ShaderType>[]) => compileWGSLWithStage(root, "fragment"),
  },
);

// === Standalone function compilers (for Three.js glslFn/wgslFn embedding) ===

export type CompileFnOptions = {
  name: string;
  params: Array<{ name: string; type: ShaderType }>;
};

function compileFnBody(
  result: Node<ShaderType>,
  params: Array<{ name: string; type: ShaderType }>,
  name: string,
  language: "glsl" | "wgsl",
): string {
  if (Array.isArray(result)) {
    throw new Error(
      "compileGLSLFn/compileWGSLFn does not support multi-return functions. "
      + "Define separate functions for each return value.",
    );
  }

  if (language === "glsl") {
    const ctx: CompileCtx = {
      nextId: 0,
      shaderStage: "fragment",
      uniforms: new Map(),
      attributes: new Map(),
      varyings: new Map(),
      outputs: new Map(),
      wgslSamplers: new Map(),
      varDefs: new Map(),
      memo: new Map(),
      wgslHelpers: new Set(),
      positionWritten: false,
      inFn: false,
      fragDepthUsed: false,
    };
    const compiled = compileGLSLStage(result, ctx);
    const returnType = glslType((result as any)._t || "float");
    const paramStr = params.map(p => `${glslType(p.type)} ${p.name}`).join(", ");
    let code = "";
    ctx.uniforms.forEach((info) => {
      code += info.length !== undefined
        ? `uniform ${info.type} ${info.slot}[${info.length}];\n`
        : `uniform ${info.type} ${info.slot};\n`;
    });
    if (ctx.uniforms.size > 0) {
      code += "\n";
    }
    code += `${returnType} ${name}(${paramStr}) {\n`;
    for (const line of compiled.body) {
      code += `  ${line}\n`;
    }
    if (compiled.expr !== "0.0") {
      code += `  return ${compiled.expr};\n`;
    } else {
      code += `  return ${returnType}(0);\n`;
    }
    code += `}`;
    return code;
  } else {
    const ctx: CompileCtx = {
      nextId: 0,
      shaderStage: "fragment",
      uniforms: new Map(),
      attributes: new Map(),
      varyings: new Map(),
      outputs: new Map(),
      wgslSamplers: new Map(),
      varDefs: new Map(),
      memo: new Map(),
      wgslHelpers: new Set(),
      positionWritten: false,
      inFn: false,
      fragDepthUsed: false,
    };
    const compiled = compileWGSLStage(result, ctx);
    const returnType = wgslType((result as any)._t || "float");
    const paramStr = params.map(p => `${p.name}: ${wgslType(p.type)}`).join(", ");
    // Helpers standing in for GLSL builtins WGSL lacks, emitted ahead of the
    // function that calls them. The whole-shader path does the same at its own
    // top level; a function emitted on its own has to carry them itself, or it
    // calls something that was never defined. Sorted so identical input gives
    // identical output regardless of the order ops were reached.
    let code = "";
    for (const helper of [...ctx.wgslHelpers].sort()) {
      code += `${WGSL_HELPERS[helper]}\n\n`;
    }
    code += `fn ${name}(${paramStr}) -> ${returnType} {\n`;
    for (const line of compiled.decls) {
      code += `  ${line}\n`;
    }
    for (const line of compiled.body) {
      code += `  ${line}\n`;
    }
    if (compiled.expr !== "0.0") {
      code += `  return ${compiled.expr};\n`;
    } else {
      code += `  return ${returnType}();\n`;
    }
    code += `}\n`;
    // Same single-struct packing as a full shader, for the same reason: one
    // binding per uniform runs out at twelve.
    let sortedUniforms = [...ctx.uniforms.entries()].sort((a, b) => a[1].slot.localeCompare(b[1].slot));
    // A texture is sampled through a companion sampler, so both are declared
    // or neither resolves. The whole-shader path does the same, in the same
    // binding groups.
    let samplerDecls = "";
    let samplerBinding = 0;
    ctx.wgslSamplers.forEach((info) => {
      samplerDecls += `@group(2) @binding(${samplerBinding++}) var ${info.samplerSlot}: sampler;\n`;
    });
    let textureDecls = "";
    let texBinding = 0;
    for (let [, info] of sortedUniforms.filter(([, i]) => isWgslTexture(i.type))) {
      textureDecls += `@group(1) @binding(${texBinding++}) var ${info.slot}: ${info.type};\n`;
    }
    if (textureDecls || samplerDecls) code = textureDecls + samplerDecls + "\n" + code;
    let plainUniforms = sortedUniforms.filter(([, i]) => !isWgslTexture(i.type));
    if (plainUniforms.length > 0) {
      let layout = wgslUniformLayout(
        plainUniforms.map(([, i]) => ({ slot: i.slot, type: i.type, length: i.length })),
      );
      let struct = `struct ${WGSL_UNIFORM_STRUCT} {\n`
        + layout.members.map(m => `  ${m.name}: ${wgslMemberType(m)},\n`).join("")
        + `};\n`
        + `@group(0) @binding(0) var<uniform> ${WGSL_UNIFORM_BINDING}: ${WGSL_UNIFORM_STRUCT};\n\n`;
      code = struct + code;
    }
    return code;
  }
}

export function compileGLSLFn(
  fn: (...args: any[]) => Node<ShaderType>,
  options: CompileFnOptions,
): string {
  const paramNodes = options.params.map(p => var_(p.name, p.type));
  const result = fn(...paramNodes);
  return compileFnBody(result, options.params, options.name, "glsl");
}

export function compileWGSLFn(
  fn: (...args: any[]) => Node<ShaderType>,
  options: CompileFnOptions,
): string {
  const paramNodes = options.params.map(p => var_(p.name, p.type));
  const result = fn(...paramNodes);
  return compileFnBody(result, options.params, options.name, "wgsl");
}
