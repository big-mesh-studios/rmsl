// === RMSL core: shader types, the Node graph, and the DSL surface ===
// (TSL-style free functions, literals, uniforms/attributes/varyings, control
// flow) that all four compiler backends walk. No dependencies of its own —
// everything else in the compiler imports from here.
export const __brand = Symbol();

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
/**
 * The `stpq` spelling of the texture-coordinate accessors, shared across the
 * float and integer vector types. Each letter is one component, so a
 * single-letter pattern is that scalar and a multi-letter one is the matching
 * vector of the same prefix — `ivec3.st` is an `ivec2`, like `.xy`.
 */
export type Stpq2<S extends ShaderType, V extends ShaderType> = {
  readonly s: Node<S>; readonly t: Node<S>;
  readonly st: Node<V>;
};

export type Stpq3<S extends ShaderType, V2 extends ShaderType, V3 extends ShaderType> = Stpq2<S, V2> & {
  readonly p: Node<S>;
  readonly sp: Node<V2>; readonly tp: Node<V2>;
  readonly stp: Node<V3>;
};

export type Stpq4<S extends ShaderType, V2 extends ShaderType, V3 extends ShaderType, V4 extends ShaderType> = Stpq3<S, V2, V3> & {
  readonly q: Node<S>;
  readonly sq: Node<V2>; readonly tq: Node<V2>; readonly pq: Node<V2>;
  readonly stq: Node<V3>; readonly spq: Node<V3>; readonly tpq: Node<V3>;
  readonly stpq: Node<V4>;
};

export type Vec3Swizzles = {
  readonly x: Node<"float">; readonly y: Node<"float">; readonly z: Node<"float">;
  readonly r: Node<"float">; readonly g: Node<"float">; readonly b: Node<"float">;
  readonly xy: Node<"vec2">; readonly xz: Node<"vec2">; readonly yz: Node<"vec2">;
  readonly xyz: Node<"vec3">; readonly rgb: Node<"vec3">;
} & Stpq3<"float", "vec2", "vec3">;

export type Vec4Swizzles = {
  readonly x: Node<"float">; readonly y: Node<"float">; readonly z: Node<"float">; readonly w: Node<"float">;
  readonly r: Node<"float">; readonly g: Node<"float">; readonly b: Node<"float">; readonly a: Node<"float">;
  readonly xy: Node<"vec2">; readonly xz: Node<"vec2">; readonly xw: Node<"vec2">;
  readonly yz: Node<"vec2">; readonly yw: Node<"vec2">; readonly zw: Node<"vec2">;
  readonly xyz: Node<"vec3">; readonly xyw: Node<"vec3">; readonly xzw: Node<"vec3">; readonly yzw: Node<"vec3">;
  readonly rgba: Node<"vec4">; readonly rgb: Node<"vec3">;
} & Stpq4<"float", "vec2", "vec3", "vec4">;

export type Vec2Swizzles = {
  readonly x: Node<"float">; readonly y: Node<"float">;
  readonly r: Node<"float">; readonly g: Node<"float">;
  readonly xy: Node<"vec2">;
} & Stpq2<"float", "vec2">;

export type IVec2Swizzles = {
  readonly x: Node<"int">; readonly y: Node<"int">;
  readonly r: Node<"int">; readonly g: Node<"int">;
  readonly xy: Node<"ivec2">;
} & Stpq2<"int", "ivec2">;

export type UVec2Swizzles = {
  readonly x: Node<"uint">; readonly y: Node<"uint">;
  readonly r: Node<"uint">; readonly g: Node<"uint">;
  readonly xy: Node<"uvec2">;
} & Stpq2<"uint", "uvec2">;

export type IVec3Swizzles = {
  readonly x: Node<"int">; readonly y: Node<"int">; readonly z: Node<"int">;
  readonly r: Node<"int">; readonly g: Node<"int">; readonly b: Node<"int">;
  readonly xy: Node<"ivec2">; readonly xz: Node<"ivec2">; readonly yz: Node<"ivec2">;
  readonly xyz: Node<"ivec3">; readonly rgb: Node<"ivec3">;
} & Stpq3<"int", "ivec2", "ivec3">;

export type UVec3Swizzles = {
  readonly x: Node<"uint">; readonly y: Node<"uint">; readonly z: Node<"uint">;
  readonly r: Node<"uint">; readonly g: Node<"uint">; readonly b: Node<"uint">;
  readonly xy: Node<"uvec2">; readonly xz: Node<"uvec2">; readonly yz: Node<"uvec2">;
  readonly xyz: Node<"uvec3">; readonly rgb: Node<"uvec3">;
} & Stpq3<"uint", "uvec2", "uvec3">;

export type IVec4Swizzles = {
  readonly x: Node<"int">; readonly y: Node<"int">; readonly z: Node<"int">; readonly w: Node<"int">;
  readonly r: Node<"int">; readonly g: Node<"int">; readonly b: Node<"int">; readonly a: Node<"int">;
  readonly xy: Node<"ivec2">; readonly xz: Node<"ivec2">; readonly xw: Node<"ivec2">;
  readonly yz: Node<"ivec2">; readonly yw: Node<"ivec2">; readonly zw: Node<"ivec2">;
  readonly xyz: Node<"ivec3">; readonly xyw: Node<"ivec3">; readonly xzw: Node<"ivec3">; readonly yzw: Node<"ivec3">;
  readonly rgba: Node<"ivec4">; readonly rgb: Node<"ivec3">;
} & Stpq4<"int", "ivec2", "ivec3", "ivec4">;

export type UVec4Swizzles = {
  readonly x: Node<"uint">; readonly y: Node<"uint">; readonly z: Node<"uint">; readonly w: Node<"uint">;
  readonly r: Node<"uint">; readonly g: Node<"uint">; readonly b: Node<"uint">; readonly a: Node<"uint">;
  readonly xy: Node<"uvec2">; readonly xz: Node<"uvec2">; readonly xw: Node<"uvec2">;
  readonly yz: Node<"uvec2">; readonly yw: Node<"uvec2">; readonly zw: Node<"uvec2">;
  readonly xyz: Node<"uvec3">; readonly xyw: Node<"uvec3">; readonly xzw: Node<"uvec3">; readonly yzw: Node<"uvec3">;
  readonly rgba: Node<"uvec4">; readonly rgb: Node<"uvec3">;
} & Stpq4<"uint", "uvec2", "uvec3", "uvec4">;

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
export interface NodeOps {
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
  mat3: MatOps<"mat3", "vec3", "vec2">;
  mat3x4: RectMatOps<"vec3", "vec4", "mat4x3">;
  mat4x2: RectMatOps<"vec4", "vec2", "mat2x4">;
  mat4x3: RectMatOps<"vec4", "vec3", "mat3x4">;
  mat4: MatOps<"mat4", "vec4", "vec3">;
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
export interface ArithOps<A extends ShaderType> {
  add(other: FloatLike | Vec2Like | Vec3Like | Vec4Like): Node<A>;
  sub(other: FloatLike | Vec2Like | Vec3Like | Vec4Like): Node<A>;
  mul(other: FloatLike | Vec2Like | Vec3Like | Vec4Like): Node<A>;
  div(other: FloatLike | Vec2Like | Vec3Like | Vec4Like): Node<A>;
  negate(): Node<A>;
}

export interface FloatMathOps<A extends ShaderType> {
  sin(): Node<A>; cos(): Node<A>; tan(): Node<A>;
  asin(): Node<A>; acos(): Node<A>; atan(other?: FloatLike): Node<A>;
  sinh(): Node<A>; cosh(): Node<A>; tanh(): Node<A>;
  asinh(): Node<A>; acosh(): Node<A>; atanh(): Node<A>;
  abs(): Node<A>; sign(): Node<A>;
  floor(): Node<A>; ceil(): Node<A>; fract(): Node<A>;
  round(): Node<A>; trunc(): Node<A>;
  radians(): Node<A>; degrees(): Node<A>;
  sqrt(): Node<A>; inverseSqrt(): Node<A>; inversesqrt(): Node<A>;
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
export interface ComparisonOps<R extends ShaderType, Operand> {
  lessThan(other: Operand): Node<R>;
  greaterThan(other: Operand): Node<R>;
  lessThanEqual(other: Operand): Node<R>;
  greaterThanEqual(other: Operand): Node<R>;
  equal(other: Operand): Node<R>;
  notEqual(other: Operand): Node<R>;
}

export interface VecCommonOps<A extends "vec2" | "vec3" | "vec4"> {
  dot(other: Node<A>): Node<"float">;
  length(): Node<"float">;
  normalize(): Node<A>;
  distance(other: Node<A>): Node<"float">;
  reflect(normal: Node<A>): Node<A>;
  refract(normal: Node<A>, eta: FloatLike): Node<A>;
  faceForward(incident: Node<A>, reference: Node<A>): Node<A>;
  clamp(min: Node<A> | FloatLike, max: Node<A> | FloatLike): Node<A>;
  mix(b: Node<A>, t: FloatLike): Node<A>;
  element(i: IntLike): Node<"float">;
  // step/smoothstep live on FloatMathOps, which also applies to every vector
  // type.
}

export interface Vec3Ops {
  cross(other: Node<"vec3">): Node<"vec3">;
}

/**
 * A square matrix. `Vec` is the vector of its own width: what one of its
 * columns is, and what multiplying it by a vector both takes and gives.
 *
 * `Shorter` is the vector one component short of a column — a position with its
 * homogeneous coordinate implied. `mat4 * vec3` and `mat3 * vec2` are both the
 * ordinary "transform a position" multiply, so they are spelled `mul` like any
 * other vector multiply rather than a method of their own.
 */
export interface MatOps<Self extends ShaderType, Vec extends ShaderType, Shorter extends ShaderType = never> {
  mul(other: Node<Self>): Node<Self>;
  mul(other: Node<Vec>): Node<Vec>;
  mul(other: Node<Shorter>): Node<Shorter>;
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
export interface RectMatOps<
  Operand extends ShaderType,
  Column extends ShaderType,
  Transposed extends ShaderType,
> {
  mul(other: Node<Operand>): Node<Column>;
  element(i: IntLike): Node<Column>;
  transpose(): Node<Transposed>;
}

/** A cube map is sampled with a direction rather than a surface coordinate. */
export interface CubeSamplerOps {
  texture(coords: Vec3Like): Node<"vec4">;
  textureLod(coords: Vec3Like, lod: FloatLike): Node<"vec4">;
}

/** A 3D texture is sampled at its volume coordinate. */
export interface Sampler3DOps {
  texture(coords: Vec3Like): Node<"vec4">;
  textureLod(coords: Vec3Like, lod: FloatLike): Node<"vec4">;
}

export interface IntOps {
  add(other: IntLike): Node<"int">;
  sub(other: IntLike): Node<"int">;
  mul(other: IntLike): Node<"int">;
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

export interface UintOps {
  add(other: UintLike): Node<"uint">;
  sub(other: UintLike): Node<"uint">;
  mul(other: UintLike): Node<"uint">;
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
export interface IVecOps<A extends "ivec2" | "ivec3" | "ivec4"> {
  add(other: IntLike | IVec2Like | IVec3Like | IVec4Like): Node<A>;
  sub(other: IntLike | IVec2Like | IVec3Like | IVec4Like): Node<A>;
  mul(other: IntLike | IVec2Like | IVec3Like | IVec4Like): Node<A>;
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

export interface UVecOps<A extends "uvec2" | "uvec3" | "uvec4"> {
  add(other: UintLike | UVec2Like | UVec3Like | UVec4Like): Node<A>;
  sub(other: UintLike | UVec2Like | UVec3Like | UVec4Like): Node<A>;
  mul(other: UintLike | UVec2Like | UVec3Like | UVec4Like): Node<A>;
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

export interface SamplerOps {
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
export interface ISampler2DOps {
  texture(coords: IVec2Like): Node<"ivec4">;
  textureLod(coords: IVec2Like, lod: IntLike): Node<"ivec4">;
}

export interface ISampler3DOps {
  texture(coords: IVec3Like): Node<"ivec4">;
  textureLod(coords: IVec3Like, lod: IntLike): Node<"ivec4">;
}

export interface ISamplerCubeOps {
  texture(coords: IVec3Like): Node<"ivec4">;
  textureLod(coords: IVec3Like, lod: IntLike): Node<"ivec4">;
}

export interface USampler2DOps {
  texture(coords: UVec2Like): Node<"uvec4">;
  textureLod(coords: UVec2Like, lod: IntLike): Node<"uvec4">;
}

export interface USampler3DOps {
  texture(coords: UVec3Like): Node<"uvec4">;
  textureLod(coords: UVec3Like, lod: IntLike): Node<"uvec4">;
}

export interface USamplerCubeOps {
  texture(coords: UVec3Like): Node<"uvec4">;
  textureLod(coords: UVec3Like, lod: IntLike): Node<"uvec4">;
}

export interface BoolOps {
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
export interface BoolVecOps<A extends ShaderType> {
  /** True when every component is true. */
  all(): Node<"bool">;
  /** True when at least one component is true. */
  any(): Node<"bool">;
  /** Negates each component. */
  not(): Node<A>;
  /** Component-wise logical xor. */
  xor(other: Node<A>): Node<A>;
}

export interface NodeMethods<A extends ShaderType> {
  /**
   * Assigns the expression to a variable and returns its reference.
   *
   * Without a name the variable gets an auto-generated `_rmsl_N` slot. A name
   * is emitted verbatim into the shader for easier debugging; a duplicate name
   * gets a number appended (`color`, `color1`, `color2`, ...).
   */
  toVar(name?: string): Node<A>;
  /** TSL's shorthand for `toVar()`. */
  var(name?: string): Node<A>;
  assign(value: BaseNode<A> | Node<A>): void;
  // === Compound assignments (as TSL's `addAssign`/`mulAssign`/...) ===
  addAssign(other: FloatLike | IntLike | UintLike | Vec2Like | Vec3Like | Vec4Like): void;
  subAssign(other: FloatLike | IntLike | UintLike | Vec2Like | Vec3Like | Vec4Like): void;
  mulAssign(other: FloatLike | IntLike | UintLike | Vec2Like | Vec3Like | Vec4Like): void;
  divAssign(other: FloatLike | IntLike | UintLike | Vec2Like | Vec3Like | Vec4Like): void;
  modAssign(other: FloatLike | IntLike | UintLike | Vec2Like | Vec3Like | Vec4Like): void;
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
  /**
   * TSL's `select()`: a conditional value. `cond.select(a, b)` is `a` when
   * `cond` is true and `b` otherwise. The condition may be a single bool or a
   * boolean vector, in which case the selection is component-wise.
   */
  select<T extends ShaderType>(
    ifTrue: BaseNode<T> | number | readonly number[],
    ifFalse: BaseNode<T> | number | readonly number[],
  ): Node<T>;
}

// === NodeImpl - defines all methods, Node<A> hides typed subset ===
export class NodeImpl<A extends ShaderType> implements BaseNode<A> {
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
  mul(other: any): any {
    // A matCxR times a vecC gives a vecR. The result type is determined by the
    // vector dimension, not the matrix type. A vector one component short of
    // the column width is a position with its homogeneous coordinate implied —
    // `mat4 * vec3` and `mat3 * vec2` — promoted here and truncated by the
    // compilers' matVecMul cases.
    let shape = MATRIX_DIMENSIONS[this._t];
    let otherType = other?._t;
    if (
      shape !== undefined && typeof otherType === "string"
      && otherType.startsWith("vec")
    ) {
      let width = TYPE_WIDTH[otherType];
      let columns = shape[0];
      let rows = shape[1];
      if (width === columns) {
        return node({
          _t: `vec${rows}`,
          type: "matVecMul",
          params: [this as BaseNode<ShaderType>, wrapValue(other) as BaseNode<ShaderType>],
        });
      }
      if (width === columns - 1) {
        return node({
          _t: `vec${Math.min(rows, width)}`,
          type: "matVecMul",
          params: [this as BaseNode<ShaderType>, wrapValue(other) as BaseNode<ShaderType>],
        });
      }
      throw new Error(
        `[RMSL] A ${this._t} cannot multiply a ${otherType}: the vector must have `
        + `the matrix's column width or one fewer component (a position with its `
        + `homogeneous coordinate implied).`,
      );
    }
    return op("mul", this, other);
  }
  div(other: any): any { return op("div", this, other); }
  negate(): any { return op("negate", this); }

  // === FloatMathOps ===
  sin() { return op1("sin", this); }
  cos() { return op1("cos", this); }
  tan() { return op1("tan", this); }
  asin() { return op1("asin", this); }
  acos() { return op1("acos", this); }
  atan(other?: any) { return other === undefined ? op1("atan", this) : op("atan2", this, other); }
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
  radians() { return op("mul", this, 0.017453292519943295); }
  degrees() { return op("mul", this, 57.29577951308232); }
  sqrt() { return op1("sqrt", this); }
  inverseSqrt() { return op1("inverseSqrt", this); }
  inversesqrt() { return op1("inverseSqrt", this); }
  exp() { return op1("exp", this); }
  log() { return op1("log", this); }
  exp2() { return op1("exp2", this); }
  log2() { return op1("log2", this); }
  cbrt() { return op("mul", this.sign(), op("pow", this.abs(), 1.0 / 3.0)); }
  reciprocal() { return op("div", 1, this); }
  oneMinus() { return op("sub", 1, this); }
  difference(other: any) { return op1("abs", op("sub", this, other)); }
  lengthSq(): any {
    // For a vector this is the squared length, the dot of itself; for a scalar
    // it is simply its square — neither language offers a scalar `dot`.
    return (TYPE_WIDTH[this._t] ?? 1) > 1 ? op("dot", this, this) : op("mul", this, this);
  }
  saturate() { return op("clamp", this, 0, 1); }
  pow(e: any) { return op("pow", this, e); }
  pow2() { return op("mul", this, this); }
  pow3() { return op("mul", this, this, this); }
  pow4() { return op("mul", this, this, this, this); }
  min(other: any) { return op("min", this, other); }
  max(other: any) { return op("max", this, other); }
  mod(other: any): any { return op("mod", this, other); }
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
  faceForward(incident: any, reference: any): any { return op("faceForward", this, incident, reference); }
  clamp(minV: any, maxV: any): any { return op("clamp", this, minV, maxV); }
  mix(b: any, t: any): any { return op("mix", this, b, t); }
  step(edge: any): any { return op("step", edge, this); }
  smoothstep(edge0: any, edge1: any): any { return op("smoothstep", edge0, edge1, this); }
  fwidth(): any { return op1("fwidth", this); }

  // === Vec3Ops ===
  cross(other: any): any { return op("cross", this, other); }

  // === MatOps ===
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

  toVar(name?: string): Node<A> {
    let v: Node<A>;
    assertBlockScope("toVar", (blockScope) => {
      let varName = claimVarName(name);
      v = var_(varName, this._t) as Node<A>;
      blockScope.push(new NodeImpl({
        _t: "void",
        type: "let",
        params: [(v as BaseNode<ShaderType>), (this as BaseNode<ShaderType>)],
      }));
    });
    return v!;
  }

  var(name?: string): Node<A> { return this.toVar(name); }

  // === Compound assignments ===
  addAssign(other: any) { this.assign(this.add(other)); }
  subAssign(other: any) { this.assign(this.sub(other)); }
  mulAssign(other: any) { this.assign(this.mul(other)); }
  divAssign(other: any) { this.assign(this.div(other)); }
  modAssign(other: any) { this.assign(this.mod(other)); }

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
  select(ifTrue: any, ifFalse: any): any {
    let a = wrapValue(ifTrue) as BaseNode<ShaderType>;
    let b = wrapValue(ifFalse) as BaseNode<ShaderType>;
    // The result type follows the branches, not the condition. `vec3(0).equal(1).select(v, w)`
    // is a vec3 no matter that the selector is a bvec3.
    let t = (a as any)?._t || (b as any)?._t || this._t;
    return node({
      _t: t,
      type: "select",
      params: [this as BaseNode<ShaderType>, a, b],
    });
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

// The `stpq` swizzles are added on the prototype rather than written out as
// getters, so the 25 patterns share one definition. The `swizzle()` helper
// types the result from the source's prefix and the pattern's length, which is
// what the explicit `x`/`xy`/`xyz` getters above do individually.
for (const pattern of ["s", "t", "p", "q", "st", "sp", "sq", "tp", "tq", "pq", "stp", "stq", "spq", "tpq", "stpq"]) {
  Object.defineProperty(NodeImpl.prototype, pattern, {
    get(this: NodeImpl<ShaderType>) { return swizzle(this, pattern); },
  });
}

// Cast constructor so `new Node<T>(...)` returns `Node<T>` with conditional methods
export const Node = NodeImpl as unknown as new <A extends ShaderType>(config: {
  _t?: string;
  type: string;
  params?: BaseNode<ShaderType>[];
  value?: unknown;
}) => Node<A>;

// === Helpers ===
export function node<A extends ShaderType>(config: {
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

export function isNode(x: any): x is BaseNode<ShaderType> {
  return typeof x === 'object' && x !== null && '_t' in x && 'type' in x;
}

/**
 * What sampling a texture gives back. A float texture samples to a vec4; an
 * integer texture to an integer vector of the same width — signed for an
 * `isampler*`, unsigned for a `usampler*`.
 */
export function textureResultType(samplerType: string): string {
  if (samplerType.startsWith("isampler")) return "ivec4";
  if (samplerType.startsWith("usampler")) return "uvec4";
  return "vec4";
}

// === Value wrapping (convert raw JS -> Node for AST) ===
export type ExtractType<V> =
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

export function wrapValue<V>(x: V): Node<ExtractType<V>> {
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
export const REDUCING_OPS: Record<string, string | ((operandType: string) => string)> = {
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
export function resultType(op: string, operandType: string): string {
  let reducing = REDUCING_OPS[op];
  if (reducing === undefined) return operandType;
  return typeof reducing === "function" ? reducing(operandType) : reducing;
}

/** Component count per type, for the operations whose width follows it. */
export const TYPE_WIDTH: Record<string, number> = {
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
export const VALUE_OPERAND: Record<string, number> = {
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
export const UNIFORM_OPERAND_OPS = new Set([
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
export function typedOperand(value: any, operandType: string): BaseNode<ShaderType> {
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

export function op(type: string, ...args: any[]): Node<ShaderType> {
  let first = wrapValue(args[0]) as BaseNode<ShaderType>;
  let firstT = (first as any)?._t || "float";
  let params = [first, ...args.slice(1).map(a => typedOperand(a, firstT))];
  // The operand that defines the op's type — usually the first, but `step` and
  // `smoothstep` take the value last because that is the argument order both
  // languages expect. The result of a type-preserving op follows the *widest*
  // operand, so a scalar broadcast beside a vector keeps the vector type:
  // `1 - vec3` (oneMinus) and `1 / vec3` (reciprocal) are still vec3.
  let valueIndex = VALUE_OPERAND[type] ?? 0;
  let valueT = (params[valueIndex] as any)?._t ?? firstT;
  let widthOf = (p: BaseNode<ShaderType>) =>
    TYPE_WIDTH[(p as any)?._t] ?? (MATRIX_DIMENSIONS[(p as any)?._t] ? 16 : 1);
  let widest = params[0];
  for (const p of params) {
    if (widthOf(p) > widthOf(widest)) widest = p;
  }
  valueT = (widest as any)?._t ?? valueT;

  if (UNIFORM_OPERAND_OPS.has(type) && (TYPE_WIDTH[valueT] ?? 1) > 1) {
    params = params.map(p =>
      (TYPE_WIDTH[(p as any)?._t] ?? 1) === 1
        ? node({ _t: valueT, type: "construct", params: [p] }) as BaseNode<ShaderType>
        : p,
    );
  }

  return node({ _t: resultType(type, valueT), type, params });
}

export function op1(type: string, a: any): Node<ShaderType> {
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
export function comp(type: string, a: any, b: any): Node<ShaderType> {
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

export function swizzle<A extends ShaderType>(src: BaseNode<ShaderType>, pattern: string): Node<A> {
  // A single component of an integer or boolean vector is that scalar type,
  // not a float, so the result type is derived from the source's component
  // prefix rather than assumed float.
  let srcT = (src as any)?._t || "float";
  let prefix = /^ivec/.test(srcT) ? "i" : /^uvec/.test(srcT) ? "u" : /^bvec/.test(srcT) ? "b" : "";
  let outType = pattern.length === 1
    ? prefix === "i" ? "int" as const : prefix === "u" ? "uint" as const : prefix === "b" ? "bool" as const : "float" as const
    : prefix === "i" ? `ivec${pattern.length}` as const
    : prefix === "u" ? `uvec${pattern.length}` as const
    : prefix === "b" ? `bvec${pattern.length}` as const
    : `vec${pattern.length}` as const;
  return node({
    _t: outType,
    type: "swizzle",
    params: [src],
    value: pattern,
  }) as Node<A>;
}

// === Block scope (same pattern as story-lang) ===
export let blockScope: BaseNode<ShaderType>[] | undefined = undefined;
export let nextVarId = 0;

/**
 * Variable names already claimed by `toVar()` in the current top-level `Fn`.
 *
 * Cleared whenever a top-level `Fn` starts, so the same source produces the
 * same names in every compile. User-supplied names are deduped here by appending
 * a number; the `_rmsl_` generated names are checked against it too so the two
 * sources can never collide.
 */
export let usedVarNames = new Set<string>();

/**
 * The `_rmsl_` prefix is reserved for everything the compiler invents —
 * uniforms, attributes, varyings, outputs, scratch vars and helpers. A user
 * variable name must not use it, or it could collide with one of those.
 */
export const RESERVED_VAR_PREFIX = "_rmsl_";

/** Pick a name for a `toVar()`, claiming it in `usedVarNames`. */
export function claimVarName(name: string | undefined): string {
  if (name !== undefined) {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) {
      throw new Error(
        `toVar("${name}") must be a valid identifier (letters, digits and ` +
        `underscore, not starting with a digit).`,
      );
    }
    if (name.startsWith(RESERVED_VAR_PREFIX)) {
      throw new Error(
        `toVar("${name}") uses the reserved "${RESERVED_VAR_PREFIX}" prefix, ` +
        `which the compiler keeps for its own names.`,
      );
    }
    let candidate = name;
    for (let i = 1; usedVarNames.has(candidate); i++) {
      candidate = `${name}${i}`;
    }
    usedVarNames.add(candidate);
    return candidate;
  }
  let candidate = `_rmsl_${nextVarId++}`;
  while (usedVarNames.has(candidate)) {
    candidate = `_rmsl_${nextVarId++}`;
  }
  usedVarNames.add(candidate);
  return candidate;
}

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
    // A top-level Fn starts a fresh name registry, so each compiled program
    // gets its own deterministic set of user-named variables. Nested Fns keep
    // the outer registry, since their variables share the outer program.
    if (oldBlockScope === undefined) usedVarNames.clear();
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

export function buildBlock(body: () => void): Node<"void"> {
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
export function makeIntVecConstructor<T extends ShaderType>(
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
export function bool(v: boolean | Node<"float"> | Node<"int"> | Node<"uint">): Node<"bool"> {
  if (isNode(v)) {
    return node({ _t: "bool", type: "construct", params: [v] }) as Node<"bool">;
  }
  return node({ _t: "bool", type: "bool", value: v }) as Node<"bool">;
}
export function makeMatConstructor<T extends ShaderType>(t: T, size: number, defaultVal: number[]): (...args: any[]) => Node<T> {
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

export function makeBoolVecConstructor<T extends ShaderType>(t: T, width: number): (...args: any[]) => Node<T> {
  return (...args: any[]): Node<T> => {
    if (args.length === 0) {
      return node({
        _t: t,
        type: "construct",
        params: [node({ _t: "bool", type: "bool", value: false })],
      }) as Node<T>;
    }
    if (args.length === 1 && isNode(args[0])) {
      return node({ _t: t, type: "construct", params: [args[0] as BaseNode<ShaderType>] }) as Node<T>;
    }
    if (args.length <= width && args.every((a) => typeof a === "boolean")) {
      return node({ _t: t, type: t, value: args }) as Node<T>;
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

export const bvec2 = makeBoolVecConstructor<"bvec2">("bvec2", 2);
export const bvec3 = makeBoolVecConstructor<"bvec3">("bvec3", 3);
export const bvec4 = makeBoolVecConstructor<"bvec4">("bvec4", 4);

// === TSL free-function API ===
/**
 * The free-function forms Three.js's TSL exports, so `mul(a, b)`, `sin(x)`,
 * `mix(a, b, t)` and the rest compile as they do in `three/tsl`. Each delegates
 * to the equivalent method on a wrapped operand, so a plain number, boolean or
 * array is accepted anywhere a node is.
 *
 * The argument order follows TSL — `step(edge, x)`, `smoothstep(low, high, x)`
 * and `mix(a, b, t)` all take the value last, as both GLSL and WGSL spell them.
 */
export type MathLike =
  | number
  | boolean
  | readonly number[]
  | Node<ShaderType>;

/**
 * Wrap a raw value as a node for method delegation. The free functions then
 * call the matching method on it. Their results are typed `any` — a node whose
 * type is not known until its operands are inspected cannot be narrowed to a
 * single `Node<A>`, and the union `Node<ShaderType>` has no methods — so the
 * operations that reduce (dot, length, all, ...) declare the narrower type and
 * the rest leave the node untyped, the way TSL's own free functions do.
 */
export function toNode(v: MathLike): any {
  return wrapValue(v);
}

export function add(a: MathLike, b: MathLike, ...rest: MathLike[]): any {
  let r = toNode(a).add(b);
  for (const x of rest) r = r.add(x);
  return r;
}
export function sub(a: MathLike, b: MathLike, ...rest: MathLike[]): any {
  let r = toNode(a).sub(b);
  for (const x of rest) r = r.sub(x);
  return r;
}
export function mul(a: MathLike, b: MathLike, ...rest: MathLike[]): any {
  let r = toNode(a).mul(b);
  for (const x of rest) r = r.mul(x);
  return r;
}
export function div(a: MathLike, b: MathLike, ...rest: MathLike[]): any {
  let r = toNode(a).div(b);
  for (const x of rest) r = r.div(x);
  return r;
}
export function mod(a: MathLike, b: MathLike): any {
  return toNode(a).mod(b);
}

export function equal(a: MathLike, b: MathLike): any {
  return toNode(a).equal(b);
}
export function notEqual(a: MathLike, b: MathLike): any {
  return toNode(a).notEqual(b);
}
export function lessThan(a: MathLike, b: MathLike): any {
  return toNode(a).lessThan(b);
}
export function greaterThan(a: MathLike, b: MathLike): any {
  return toNode(a).greaterThan(b);
}
export function lessThanEqual(a: MathLike, b: MathLike): any {
  return toNode(a).lessThanEqual(b);
}
export function greaterThanEqual(a: MathLike, b: MathLike): any {
  return toNode(a).greaterThanEqual(b);
}

export function and(a: MathLike, b: MathLike, ...rest: MathLike[]): any {
  let r = toNode(a).and(b);
  for (const x of rest) r = r.and(x);
  return r;
}
export function or(a: MathLike, b: MathLike, ...rest: MathLike[]): any {
  let r = toNode(a).or(b);
  for (const x of rest) r = r.or(x);
  return r;
}
export function xor(a: MathLike, b: MathLike): any {
  return toNode(a).xor(b);
}
export function not(a: MathLike): any {
  return toNode(a).not();
}

export function bitAnd(a: MathLike, b: MathLike): any {
  return toNode(a).bitAnd(b);
}
export function bitOr(a: MathLike, b: MathLike): any {
  return toNode(a).bitOr(b);
}
export function bitXor(a: MathLike, b: MathLike): any {
  return toNode(a).bitXor(b);
}
export function bitNot(a: MathLike): any {
  return toNode(a).bitNot();
}
export function shiftLeft(a: MathLike, b: MathLike): any {
  return toNode(a).shiftLeft(b);
}
export function shiftRight(a: MathLike, b: MathLike): any {
  return toNode(a).shiftRight(b);
}

export function abs(a: MathLike): any { return toNode(a).abs(); }
export function sign(a: MathLike): any { return toNode(a).sign(); }
export function floor(a: MathLike): any { return toNode(a).floor(); }
export function ceil(a: MathLike): any { return toNode(a).ceil(); }
export function fract(a: MathLike): any { return toNode(a).fract(); }
export function round(a: MathLike): any { return toNode(a).round(); }
export function trunc(a: MathLike): any { return toNode(a).trunc(); }
export function radians(a: MathLike): any { return toNode(a).radians(); }
export function degrees(a: MathLike): any { return toNode(a).degrees(); }
export function sqrt(a: MathLike): any { return toNode(a).sqrt(); }
export function inverseSqrt(a: MathLike): any { return toNode(a).inverseSqrt(); }
/** GLSL-style alias for `inverseSqrt`, which TSL also exports. */
export function inversesqrt(a: MathLike): any { return toNode(a).inverseSqrt(); }
export function exp(a: MathLike): any { return toNode(a).exp(); }
export function log(a: MathLike): any { return toNode(a).log(); }
export function exp2(a: MathLike): any { return toNode(a).exp2(); }
export function log2(a: MathLike): any { return toNode(a).log2(); }
export function negate(a: MathLike): any { return toNode(a).negate(); }
export function oneMinus(a: MathLike): any { return toNode(a).oneMinus(); }
export function reciprocal(a: MathLike): any { return toNode(a).reciprocal(); }
export function cbrt(a: MathLike): any { return toNode(a).cbrt(); }
export function saturate(a: MathLike): any { return toNode(a).saturate(); }
export function lengthSq(a: MathLike): any { return toNode(a).lengthSq(); }
export function normalize(a: MathLike): any { return toNode(a).normalize(); }
export function dFdx(a: MathLike): any { return toNode(a).dFdx(); }
export function dFdy(a: MathLike): any { return toNode(a).dFdy(); }
export function fwidth(a: MathLike): any { return toNode(a).fwidth(); }

export function sin(a: MathLike): any { return toNode(a).sin(); }
export function cos(a: MathLike): any { return toNode(a).cos(); }
export function tan(a: MathLike): any { return toNode(a).tan(); }
export function asin(a: MathLike): any { return toNode(a).asin(); }
export function acos(a: MathLike): any { return toNode(a).acos(); }
export function sinh(a: MathLike): any { return toNode(a).sinh(); }
export function cosh(a: MathLike): any { return toNode(a).cosh(); }
export function tanh(a: MathLike): any { return toNode(a).tanh(); }
export function asinh(a: MathLike): any { return toNode(a).asinh(); }
export function acosh(a: MathLike): any { return toNode(a).acosh(); }
export function atanh(a: MathLike): any { return toNode(a).atanh(); }

/** `atan(y)` is the single-argument arctangent; `atan(y, x)` is `atan2`. */
export function atan(y: MathLike, x?: MathLike): any {
  return x === undefined ? toNode(y).atan() : toNode(y).atan(x);
}

export function pow(x: MathLike, e: MathLike): any { return toNode(x).pow(e); }
export function pow2(x: MathLike): any { return toNode(x).pow2(); }
export function pow3(x: MathLike): any { return toNode(x).pow3(); }
export function pow4(x: MathLike): any { return toNode(x).pow4(); }
export function min(a: MathLike, b: MathLike, ...rest: MathLike[]): any {
  let r = toNode(a).min(b);
  for (const x of rest) r = r.min(x);
  return r;
}
export function max(a: MathLike, b: MathLike, ...rest: MathLike[]): any {
  let r = toNode(a).max(b);
  for (const x of rest) r = r.max(x);
  return r;
}
export function step(edge: MathLike, x: MathLike): any {
  return toNode(x).step(edge);
}
export function reflect(incident: MathLike, normal: MathLike): any {
  return toNode(incident).reflect(normal);
}
export function refract(incident: MathLike, normal: MathLike, eta: MathLike): any {
  return toNode(incident).refract(normal, eta);
}
export function faceForward(n: MathLike, incident: MathLike, reference: MathLike): any {
  return toNode(n).faceForward(incident, reference);
}
export function difference(a: MathLike, b: MathLike): any {
  return toNode(a).difference(b);
}
export function dot(a: MathLike, b: MathLike): Node<"float"> {
  return toNode(a).dot(b);
}
export function cross(a: MathLike, b: MathLike): any {
  return toNode(a).cross(b);
}
export function distance(a: MathLike, b: MathLike): Node<"float"> {
  return toNode(a).distance(b);
}
export function length(a: MathLike): Node<"float"> {
  return toNode(a).length();
}
export function mix(a: MathLike, b: MathLike, t: MathLike): any {
  return toNode(a).mix(b, t);
}
export function clamp(x: MathLike, low: MathLike = 0, high: MathLike = 1): any {
  return toNode(x).clamp(low, high);
}
export function smoothstep(low: MathLike, high: MathLike, x: MathLike): any {
  return toNode(x).smoothstep(low, high);
}

export function all(x: MathLike): Node<"bool"> { return toNode(x).all(); }
export function any(x: MathLike): Node<"bool"> { return toNode(x).any(); }

export function transpose(m: MathLike): any { return toNode(m).transpose(); }
export function determinant(m: MathLike): Node<"float"> { return toNode(m).determinant(); }
export function inverse(m: MathLike): any { return toNode(m).inverse(); }

export function element(a: MathLike, i: IntLike): any {
  return toNode(a).element(i);
}

/** TSL's conditional: `select(cond, a, b)` is `a` when `cond`, else `b`. */
export function select(cond: MathLike, a: MathLike, b: MathLike): any {
  return toNode(cond).select(a, b);
}

/**
 * Rec. 709 luminance of a colour. A vec4 is reduced over its rgb; a vec3 (or
 * anything narrower) is used as it is. The coefficients match the current
 * working colour space's primaries in three.js, which for both sRGB and
 * linear-sRGB is Rec. 709.
 */
export function luminance(color: MathLike, luminanceCoefficients: MathLike = [0.2126, 0.7152, 0.0722]): Node<"float"> {
  let c = toNode(color);
  let rgb = (c as any)?._t === "vec4" ? c.rgb : c;
  return dot(rgb, luminanceCoefficients);
}

/**
 * A deterministic hash of the given uv, in `[0, 1)`, from TSL's `rand`.
 */
export function rand(uv: MathLike): Node<"float"> {
  let dt = dot(toNode(uv).xy, vec2(12.9898, 78.233));
  return fract(sin(mod(dt, PI)).mul(43758.5453));
}

/**
 * Interleaved gradient noise (Jimenez 2014), a cheap per-pixel dithering hash
 * in `[0, 1)`. Takes a pixel-space position.
 */
export function interleavedGradientNoise(position: MathLike): Node<"float"> {
  return fract(float(52.9829189).mul(fract(dot(toNode(position), vec2(0.06711056, 0.00583715)))));
}

/** Multiply a colour's rgb by its alpha, leaving alpha alone. */
export function premultiplyAlpha(color: MathLike): any {
  let c = toNode(color);
  return vec4(c.rgb.mul(c.a), c.a);
}

/** Reverse `premultiplyAlpha`, guarding against a zero alpha. */
export function unpremultiplyAlpha(color: MathLike): any {
  let c = toNode(color);
  return c.a.equal(0).select(vec4(0), vec4(c.rgb.div(c.a), c.a));
}

/**
 * Fetch a single texel at integer coordinates, without filtering — TSL's
 * `textureLoad`. The float-sampler counterpart of the integer samplers' texel
 * fetch; both backends emit their unfiltered load (`texelFetch` / `textureLoad`).
 */
export function textureLoad(samplerNode: Sampler2DLike, coords: IVec2Like | UVec2Like): Node<"vec4">;
export function textureLoad(samplerNode: Sampler3DLike, coords: IVec3Like | UVec3Like): Node<"vec4">;
export function textureLoad(samplerNode: ISampler2DLike, coords: IVec2Like): Node<"ivec4">;
export function textureLoad(samplerNode: USampler3DLike, coords: UVec3Like): Node<"uvec4">;
export function textureLoad(
  samplerNode: Sampler2DLike | Sampler3DLike | ISampler2DLike | USampler3DLike,
  coords: IVec2Like | IVec3Like | UVec2Like | UVec3Like,
): any {
  let sampler: any = samplerNode;
  let samplerType = (sampler as any)?._t || "sampler2D";
  return node({
    _t: textureResultType(samplerType),
    type: "textureLoad",
    params: [sampler as BaseNode<ShaderType>, wrapValue(coords) as BaseNode<ShaderType>],
  });
}

/**
 * Dimensions of a texture, in texels — `uvec2` for a 2D or cube texture,
 * `uvec3` for a 3D one.
 */
export function textureSize(samplerNode: Sampler2DLike): Node<"uvec2">;
export function textureSize(samplerNode: Sampler3DLike): Node<"uvec3">;
export function textureSize(
  samplerNode: Sampler2DLike | Sampler3DLike | ISampler2DLike | USampler3DLike,
): any {
  let sampler: any = samplerNode;
  let samplerType = (sampler as any)?._t || "sampler2D";
  let width = samplerType.endsWith("2D") || samplerType.endsWith("Cube") ? 2 : 3;
  return node({
    _t: width === 2 ? "uvec2" : "uvec3",
    type: "textureSize",
    params: [sampler as BaseNode<ShaderType>],
  });
}

// === TSL constants ===
/** π as a float node. */
export const PI = float(Math.PI);
/** 2π as a float node. */
export const TWO_PI = float(Math.PI * 2);
/** @deprecated Alias for `TWO_PI`, kept because TSL still exports it. */
export const PI2 = float(Math.PI * 2);
/** π/2 as a float node. */
export const HALF_PI = float(Math.PI * 0.5);
/** A small float used to handle floating-point precision errors. */
export const EPSILON = float(1e-6);
/** A large float standing in for infinity, as TSL uses it. */
export const INFINITY = float(1e6);

// === Uniforms, Attributes, Varyings ===
export let nextUniformId = 0;
export let nextAttrId = 0;
export let nextVaryingId = 0;

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

/**
 * A shared per-frame clock, seconds since the start, as TSL's `time`. One node
 * for every shader that references it, so the host updates a single uniform.
 * Created lazily so merely importing rmsl never consumes a uniform slot.
 */
export let _timeUniform: UniformNode<"float"> | undefined;
export function time(): UniformNode<"float"> {
  return (_timeUniform ??= uniform("float"));
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

export function attributeRaw<T extends ShaderType>(name: string, shaderType: T): AttributeNode<T> {
  let id = nextAttrId++;
  const result = node({
    _t: shaderType,
    type: "attribute",
    value: { id, slot: name, shaderType },
    name: name,
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

export function varyingRaw<T extends ShaderType>(name: string, shaderType: T): VaryingNode<T> {
  let id = nextVaryingId++;
  const result = node({
    _t: shaderType,
    type: "varying",
    value: { id, slot: name, shaderType },
    name: name,
  });
  return result as unknown as VaryingNode<T>;
}

// === Outputs ===
export let nextOutputId = 0;

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

/**
 * The fragment's position in the framebuffer, in pixels — `gl_FragCoord.xy`.
 * The origin is the lower-left of the framebuffer on both backends, which is
 * what a screen-space pass samples its texture with.
 */
export function fragCoord(): Node<"vec2"> {
  return node({
    _t: "vec2",
    type: "fragCoord",
  }) as Node<"vec2">;
}

/**
 * Pixel coordinates of the current fragment (a fragment-stage-only builtin).
 * Alias of `fragCoord()`, named as TSL does.
 */
export function screenCoordinate(): Node<"vec2"> {
  return fragCoord();
}

/** Drawing-buffer size in pixels, a `vec2` uniform the host must bind. */
export function screenSize(): Node<"vec2"> {
  return uniform("vec2");
}

/** Normalized fragment coordinate — `fragCoord() / screenSize()`. */
export function screenUV(): Node<"vec2"> {
  return div(screenCoordinate(), screenSize());
}

/** TSL's fullscreen-quad `uv()`: the normalized screen position. */
export function uv(): Node<"vec2"> {
  return screenUV();
}

// === Control Flow ===

export type ElseIfChain = {
  ElseIf: (cond: BooleanLike, body: () => void) => ElseIfChain;
  Else: (body: () => void) => void;
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
    Else: (elseBody) => {
      deepestIf.params![2] = buildBlock(elseBody) as BaseNode<ShaderType>;
    },
  };
  return chain;
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

/**
 * TSL's counting loop: `Loop(count, (i) => { ... })` iterates `count` times
 * with `i` an `int` index from 0. Lowered to the same `For` machinery.
 */
export function Loop(
  count: IntLike | FloatLike,
  body: (i: Node<"int">) => void,
): void {
  For(
    () => int(0).toVar(),
    (i) => i.lessThan(count as any),
    (i) => { i.assign(i.add(int(1))); },
    (i) => body(i),
  );
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

export type SwitchCase = { values: BaseNode<ShaderType>[]; body: Node<"void"> };

export type SwitchChain = {
  Case: (values: IntLike | readonly IntLike[], body: () => void) => SwitchChain;
  Default: (body: () => void) => void;
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
 * there is no fall-through and no `Break()` inside a case.
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
      // `typedOperand`, not `wrapValue`: a bare number here is a case value
      // beside an int/uint selector, the exact situation `typedOperand`
      // exists for — `wrapValue` alone always defaults a plain number to
      // `float`, which would type every case value as `float` regardless of
      // the selector, silently mismatched against it (GLSL/WGSL happened to
      // paper over this with an implicit cast at comparison codegen; the
      // WASM backend does not, and surfaced it as a real type error).
      values: vals.map(v => typedOperand(v, selector._t) as BaseNode<ShaderType>),
      body: buildBlock(caseBody),
    });
    return chain;
  };
  const chain: SwitchChain = {
    Case: addCase,
    Default: (dBody) => { defaultBody = buildBlock(dBody); },
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

export function Discard(): void {
  assertBlockScope("Discard", (scope) => {
    scope.push(node({ _t: "void", type: "discard" }));
  });
}

export function Break(): void {
  assertBlockScope("Break", (scope) => {
    scope.push(node({ _t: "void", type: "break" }));
  });
}

export function Continue(): void {
  assertBlockScope("Continue", (scope) => {
    scope.push(node({ _t: "void", type: "continue" }));
  });
}

export function Return(): void {
  assertBlockScope("Return", (scope) => {
    scope.push(node({ _t: "void", type: "return" }));
  });
}


/**
 * `[columns, rows]` per matrix type. A GLSL/WGSL `matCxR` is C columns of R
 * rows, and the square names are the C === R shorthand.
 *
 * Every matrix type is listed.
 */
export const MATRIX_DIMENSIONS: Record<string, [number, number]> = {
  mat2: [2, 2], mat2x3: [2, 3], mat2x4: [2, 4],
  mat3x2: [3, 2], mat3: [3, 3], mat3x4: [3, 4],
  mat4x2: [4, 2], mat4x3: [4, 3], mat4: [4, 4],
};

/**
 * Whether a uniform is a texture, which cannot go in the uniform address space
 * and keeps a binding of its own. Asked in both places that emit uniforms, so
 * the two cannot disagree about it.
 */
export function isSamplerType(type: string): boolean {
  return /^(i|u)?sampler(2D|3D|Cube)$/.test(type);
}
