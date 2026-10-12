import { withArticle } from "./article";
export const __brand = Symbol();

// === Shader Types (strings, not TS enums) ===
export type ShaderType =
  | "float"
  | "vec2"
  | "vec3"
  | "vec4"
  | "int"
  | "uint"
  | "bool"
  | "ivec2"
  | "ivec3"
  | "ivec4"
  | "uvec2"
  | "uvec3"
  | "uvec4"
  | "bvec2"
  | "bvec3"
  | "bvec4"
  | "mat2"
  | "mat2x3"
  | "mat2x4"
  | "mat3x2"
  | "mat3"
  | "mat3x4"
  | "mat4x2"
  | "mat4x3"
  | "mat4"
  | "sampler2D"
  | "sampler3D"
  | "samplerCube"
  | "isampler2D"
  | "isampler3D"
  | "isamplerCube"
  | "usampler2D"
  | "usampler3D"
  | "usamplerCube"
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
/** Nine numbers, a `mat3`'s columns one after another. */
type Mat3Array = readonly [number, number, number, number, number, number, number, number, number];
/** Sixteen numbers, a `mat4`'s columns one after another. */
type Mat4Array = readonly [...Mat3Array, number, number, number, number, number, number, number];
/**
 * A JavaScript array given where a node goes: the lengths a vector, a `mat3`
 * or a `mat4` has, so an array of another length is refused as it is written.
 */
export type VectorArray =
  | readonly [number, number]
  | readonly [number, number, number]
  | readonly [number, number, number, number]
  | Mat3Array
  | Mat4Array;
export type Mat3Like = Mat3Array | BaseNode<"mat3">;
export type Mat4Like = Mat4Array | BaseNode<"mat4">;
export type Sampler2DLike = BaseNode<"sampler2D"> | Node<"sampler2D">;
export type Sampler3DLike = BaseNode<"sampler3D"> | Node<"sampler3D">;
export type ISampler2DLike = BaseNode<"isampler2D"> | Node<"isampler2D">;
export type USampler3DLike = BaseNode<"usampler3D"> | Node<"usampler3D">;
export type SamplerCubeLike = BaseNode<"samplerCube"> | Node<"samplerCube">;
export type ISampler3DLike = BaseNode<"isampler3D"> | Node<"isampler3D">;
export type ISamplerCubeLike = BaseNode<"isamplerCube"> | Node<"isamplerCube">;
export type USampler2DLike = BaseNode<"usampler2D"> | Node<"usampler2D">;
export type USamplerCubeLike = BaseNode<"usamplerCube"> | Node<"usamplerCube">;

/** What the host gives a boolean component: `true` or `false`, or 1 or 0, which encode them exactly. */
export type BooleanValue = boolean | 0 | 1;

/**
 * The plain-JS shape `setUniform`/`setUniform` on a uniform array element
 * accepts for a given `ShaderType` — a scalar for scalar types, a fixed-width
 * tuple for vecs, a flat array for mats (column-major, like `Mat3Like`/
 * `Mat4Like` above). Samplers and `void` have no numeric uniform value, so
 * they resolve to `never` — set a sampler through its own binding, not
 * `setUniform`.
 */
export type UniformValue<A extends ShaderType> = A extends "float" | "int" | "uint"
  ? number
  : A extends "bool"
    ? BooleanValue
    : A extends "bvec2"
      ? [BooleanValue, BooleanValue] | Int32Array | Uint32Array
      : A extends "bvec3"
        ? [BooleanValue, BooleanValue, BooleanValue] | Int32Array | Uint32Array
        : A extends "bvec4"
          ? [BooleanValue, BooleanValue, BooleanValue, BooleanValue] | Int32Array | Uint32Array
          : A extends keyof typeof MATRIX_SHAPES
            ? MatrixComponents<(typeof MATRIX_SHAPES)[A][0], (typeof MATRIX_SHAPES)[A][1]> | HostVectorArray<A>
            : A extends keyof typeof VECTOR_WIDTHS
              ? Components<(typeof VECTOR_WIDTHS)[A]> | HostVectorArray<A>
              : never;

/** A tuple of `N` numbers: the components of a vector or matrix value, given as a plain array. */
type Components<N extends number, T extends number[] = []> = T["length"] extends N ? T : Components<N, [...T, number]>;

/** A tuple of the components of a matrix of `C` columns of `R` rows, a plain array of them column after column. */
type MatrixComponents<
  C extends number,
  R extends number,
  T extends unknown[] = [],
  Columns extends unknown[] = [],
> = Columns["length"] extends C ? T : MatrixComponents<C, R, [...T, ...Components<R>], [...Columns, unknown]>;

/** The typed arrays that hold a vector or matrix of type `A` as it is: integers in their own kind, floats at either width. */
type HostVectorArray<A extends ShaderType> = A extends `ivec${string}`
  ? Int32Array
  : A extends `uvec${string}`
    ? Uint32Array
    : Float32Array | Float64Array;

/**
 * Any value `setUniform` takes by slot name, where the type of the slot is
 * not known: a scalar, a vector or matrix, or a uniform array of them, or
 * `null` or `undefined`, which set it to zero. The adapter checks its shape
 * against the slot's type.
 */
export type UniformSlotValue =
  number | boolean | ArrayLike<number | boolean> | readonly ArrayLike<number | boolean>[] | null | undefined;

/** A parameter of a routine, as the host names it where the routine is compiled. */
export type RoutineParam = { readonly name: string; readonly type: ShaderType };

/**
 * The parameters `run` takes, by name. Given parameters whose names are
 * known, each takes the value a uniform of its type takes, and may be left
 * out or `null`, which reads zero. Otherwise any name takes any value a
 * parameter can hold.
 */
export type RoutineParams<P extends readonly RoutineParam[]> = string extends P[number]["name"]
  ? Record<string, UniformSlotValue>
  : { [K in P[number] as K["name"]]?: UniformValue<K["type"]> | null };

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
/** The precision a uniform can ask for its GLSL declaration, as TSL's `setPrecision` takes it. */
export type UniformPrecision = "low" | "medium" | "high";

export type UniformNode<A extends ShaderType> = VariableNode<A> & {
  /** Sets the precision of this uniform's GLSL declaration, over the shader's default, and returns it. */
  setPrecision(precision: UniformPrecision): UniformNode<A>;
};

/**
 * A uniform array. Not a `Node<A>` itself — the array as a whole has no value,
 * only its elements do, so it exposes `element()` rather than the operations of
 * its element type.
 */
export interface UniformArrayNode<A extends ShaderType> {
  readonly name: string;
  readonly length: number;
  element(index: IntLike | UintLike | FloatLike): Node<A>;
  /** Sets the precision of this array's GLSL declaration, over the shader's default, and returns it. */
  setPrecision(precision: UniformPrecision): UniformArrayNode<A>;
}
export type AttributeNode<A extends ShaderType> = VariableNode<A>;
/** A varying: written by the vertex stage, read by the fragment stage. */
export type VaryingNode<A extends ShaderType> = Var<A> & { name: string };

// === Type guards for node type checking ===
export function isUniformNode<T extends ShaderType>(node: BaseNode<T>): node is UniformNode<T> {
  return node.type === "uniform" && "name" in node;
}

export function isAttributeNode<T extends ShaderType>(node: BaseNode<T>): node is AttributeNode<T> {
  return node.type === "attribute" && "name" in node;
}

export function isVaryingNode<T extends ShaderType>(node: BaseNode<T>): node is VaryingNode<T> {
  return node.type === "varying" && "name" in node;
}

// === Per-type swizzle sets ===
/** A vector's component letters, by width, in each spelling: `xyzw`, `rgba` and `stpq`. */
type SwizzleLetters = {
  2: ["x" | "y", "r" | "g", "s" | "t"];
  3: ["x" | "y" | "z", "r" | "g" | "b", "s" | "t" | "p"];
  4: ["x" | "y" | "z" | "w", "r" | "g" | "b" | "a", "s" | "t" | "p" | "q"];
};

/** Every swizzle of one to four of the letters `L`: one letter is the scalar, more the vector of that width. */
type SwizzlesOf<
  L extends string,
  S extends ShaderType,
  V2 extends ShaderType,
  V3 extends ShaderType,
  V4 extends ShaderType,
> = {
  readonly [K in L]: Node<S>;
} & { readonly [K in `${L}${L}`]: Node<V2> } & { readonly [K in `${L}${L}${L}`]: Node<V3> } & {
  readonly [K in `${L}${L}${L}${L}`]: Node<V4>;
};

/**
 * Every swizzle of a vector of width `N`, as in TSL, GLSL and WGSL: any one to
 * four of its components, in any order and with repeats, in one spelling —
 * `v.zyx`, `v.rr`, `v.ts` — but not mixing spellings, as `v.xg`.
 */
export type Swizzles<
  N extends 2 | 3 | 4,
  S extends ShaderType,
  V2 extends ShaderType,
  V3 extends ShaderType,
  V4 extends ShaderType,
> = SwizzlesOf<SwizzleLetters[N][0], S, V2, V3, V4> &
  SwizzlesOf<SwizzleLetters[N][1], S, V2, V3, V4> &
  SwizzlesOf<SwizzleLetters[N][2], S, V2, V3, V4>;

export type Vec2Swizzles = Swizzles<2, "float", "vec2", "vec3", "vec4">;
export type Vec3Swizzles = Swizzles<3, "float", "vec2", "vec3", "vec4">;
export type Vec4Swizzles = Swizzles<4, "float", "vec2", "vec3", "vec4">;
export type IVec2Swizzles = Swizzles<2, "int", "ivec2", "ivec3", "ivec4">;
export type IVec3Swizzles = Swizzles<3, "int", "ivec2", "ivec3", "ivec4">;
export type IVec4Swizzles = Swizzles<4, "int", "ivec2", "ivec3", "ivec4">;
export type UVec2Swizzles = Swizzles<2, "uint", "uvec2", "uvec3", "uvec4">;
export type UVec3Swizzles = Swizzles<3, "uint", "uvec2", "uvec3", "uvec4">;
export type UVec4Swizzles = Swizzles<4, "uint", "uvec2", "uvec3", "uvec4">;
export type BVec2Swizzles = Swizzles<2, "bool", "bvec2", "bvec3", "bvec4">;
export type BVec3Swizzles = Swizzles<3, "bool", "bvec2", "bvec3", "bvec4">;
export type BVec4Swizzles = Swizzles<4, "bool", "bvec2", "bvec3", "bvec4">;

// === Writable nodes ===
/** Every pattern of two of the letters `L`, each at most once. */
type Distinct2<L extends string> = { [A in L]: `${A}${Exclude<L, A>}` }[L];
/** Every pattern of three of the letters `L`, each at most once. */
type Distinct3<L extends string> = { [A in L]: `${A}${Distinct2<Exclude<L, A>>}` }[L];
/** Every pattern of four of the letters `L`, each at most once. */
type Distinct4<L extends string> = { [A in L]: `${A}${Distinct3<Exclude<L, A>>}` }[L];

/** The swizzles of the letters `L` that can be written through: each names a component at most once. */
type WritableSwizzlesOf<
  L extends string,
  S extends ShaderType,
  V2 extends ShaderType,
  V3 extends ShaderType,
  V4 extends ShaderType,
> = { readonly [K in L]: Var<S> } & { readonly [K in Distinct2<L>]: Var<V2> } & {
  readonly [K in Distinct3<L>]: Var<V3>;
} & { readonly [K in Distinct4<L>]: Var<V4> };

/** The swizzles of a writable vector of width `N` that can be written through, in each spelling. */
type WritableSwizzles<
  N extends 2 | 3 | 4,
  S extends ShaderType,
  V2 extends ShaderType,
  V3 extends ShaderType,
  V4 extends ShaderType,
> = WritableSwizzlesOf<SwizzleLetters[N][0], S, V2, V3, V4> &
  WritableSwizzlesOf<SwizzleLetters[N][1], S, V2, V3, V4> &
  WritableSwizzlesOf<SwizzleLetters[N][2], S, V2, V3, V4>;

/** Assigning to a writable node, as TSL's `assign`, `addAssign`, `mulAssign`, … */
export interface AssignOps<A extends ShaderType> {
  assign(value: BaseNode<A> | Node<A>): void;
  addAssign(other: FloatLike | IntLike | UintLike | Vec2Like | Vec3Like | Vec4Like): void;
  subAssign(other: FloatLike | IntLike | UintLike | Vec2Like | Vec3Like | Vec4Like): void;
  mulAssign(other: FloatLike | IntLike | UintLike | Vec2Like | Vec3Like | Vec4Like): void;
  divAssign(other: FloatLike | IntLike | UintLike | Vec2Like | Vec3Like | Vec4Like): void;
  modAssign(other: FloatLike | IntLike | UintLike | Vec2Like | Vec3Like | Vec4Like): void;
}

/** A writable vector: assignable, and so is each of its components and each swizzle naming a component once. */
type WritableVec<
  A extends ShaderType,
  N extends 2 | 3 | 4,
  S extends ShaderType,
  V2 extends ShaderType,
  V3 extends ShaderType,
  V4 extends ShaderType,
> = AssignOps<A> & { element(i: IntLike): Var<S> } & WritableSwizzles<N, S, V2, V3, V4>;

/** A writable matrix: assignable, and so is each of its columns. */
type WritableMat<A extends ShaderType, Column extends ShaderType> = AssignOps<A> & { element(i: IntLike): Var<Column> };

/**
 * What a writable node of each type adds to its {@link NodeOps}: a registry,
 * as `NodeOps` is, so a new type says whether and how it can be written.
 */
export interface WritableOps {
  float: AssignOps<"float">;
  int: AssignOps<"int">;
  uint: AssignOps<"uint">;
  bool: AssignOps<"bool">;
  vec2: WritableVec<"vec2", 2, "float", "vec2", "vec3", "vec4">;
  vec3: WritableVec<"vec3", 3, "float", "vec2", "vec3", "vec4">;
  vec4: WritableVec<"vec4", 4, "float", "vec2", "vec3", "vec4">;
  ivec2: WritableVec<"ivec2", 2, "int", "ivec2", "ivec3", "ivec4">;
  ivec3: WritableVec<"ivec3", 3, "int", "ivec2", "ivec3", "ivec4">;
  ivec4: WritableVec<"ivec4", 4, "int", "ivec2", "ivec3", "ivec4">;
  uvec2: WritableVec<"uvec2", 2, "uint", "uvec2", "uvec3", "uvec4">;
  uvec3: WritableVec<"uvec3", 3, "uint", "uvec2", "uvec3", "uvec4">;
  uvec4: WritableVec<"uvec4", 4, "uint", "uvec2", "uvec3", "uvec4">;
  bvec2: WritableVec<"bvec2", 2, "bool", "bvec2", "bvec3", "bvec4">;
  bvec3: WritableVec<"bvec3", 3, "bool", "bvec2", "bvec3", "bvec4">;
  bvec4: WritableVec<"bvec4", 4, "bool", "bvec2", "bvec3", "bvec4">;
  mat2: WritableMat<"mat2", "vec2">;
  mat2x3: WritableMat<"mat2x3", "vec3">;
  mat2x4: WritableMat<"mat2x4", "vec4">;
  mat3x2: WritableMat<"mat3x2", "vec2">;
  mat3: WritableMat<"mat3", "vec3">;
  mat3x4: WritableMat<"mat3x4", "vec4">;
  mat4x2: WritableMat<"mat4x2", "vec2">;
  mat4x3: WritableMat<"mat4x3", "vec3">;
  mat4: WritableMat<"mat4", "vec4">;
  sampler2D: {};
  sampler3D: {};
  samplerCube: {};
  isampler2D: {};
  isampler3D: {};
  isamplerCube: {};
  usampler2D: {};
  usampler3D: {};
  usamplerCube: {};
  void: {};
}

/**
 * A node that can be assigned to: a variable, a storage element or a stage
 * output, or a component, column or swizzle of one that names each component
 * once. Its writable members come first, so where both declare one, as
 * `element()`, the writable one is the one a call resolves to.
 */
export type Var<A extends ShaderType> = BaseNode<A> & VarOps[A] & NodeMethods<A>;

/**
 * Each type's writable members followed by its {@link NodeOps}, in one
 * registry, so a `Var<A>` holds a single indexed access, as `Node<A>` does.
 * Two would make TypeScript intersect their constraints, a union of every
 * type's members with another, whenever it relates a `Var<T>` of a generic `T`.
 * Written out rather than mapped: TypeScript reduces an indexed access into a
 * mapped type back into the two.
 */
interface VarOps {
  float: WritableOps["float"] & NodeOps["float"];
  vec2: WritableOps["vec2"] & NodeOps["vec2"];
  vec3: WritableOps["vec3"] & NodeOps["vec3"];
  vec4: WritableOps["vec4"] & NodeOps["vec4"];
  int: WritableOps["int"] & NodeOps["int"];
  uint: WritableOps["uint"] & NodeOps["uint"];
  bool: WritableOps["bool"] & NodeOps["bool"];
  ivec2: WritableOps["ivec2"] & NodeOps["ivec2"];
  ivec3: WritableOps["ivec3"] & NodeOps["ivec3"];
  ivec4: WritableOps["ivec4"] & NodeOps["ivec4"];
  uvec2: WritableOps["uvec2"] & NodeOps["uvec2"];
  uvec3: WritableOps["uvec3"] & NodeOps["uvec3"];
  uvec4: WritableOps["uvec4"] & NodeOps["uvec4"];
  bvec2: WritableOps["bvec2"] & NodeOps["bvec2"];
  bvec3: WritableOps["bvec3"] & NodeOps["bvec3"];
  bvec4: WritableOps["bvec4"] & NodeOps["bvec4"];
  mat2: WritableOps["mat2"] & NodeOps["mat2"];
  mat2x3: WritableOps["mat2x3"] & NodeOps["mat2x3"];
  mat2x4: WritableOps["mat2x4"] & NodeOps["mat2x4"];
  mat3x2: WritableOps["mat3x2"] & NodeOps["mat3x2"];
  mat3: WritableOps["mat3"] & NodeOps["mat3"];
  mat3x4: WritableOps["mat3x4"] & NodeOps["mat3x4"];
  mat4x2: WritableOps["mat4x2"] & NodeOps["mat4x2"];
  mat4x3: WritableOps["mat4x3"] & NodeOps["mat4x3"];
  mat4: WritableOps["mat4"] & NodeOps["mat4"];
  sampler2D: WritableOps["sampler2D"] & NodeOps["sampler2D"];
  sampler3D: WritableOps["sampler3D"] & NodeOps["sampler3D"];
  samplerCube: WritableOps["samplerCube"] & NodeOps["samplerCube"];
  isampler2D: WritableOps["isampler2D"] & NodeOps["isampler2D"];
  isampler3D: WritableOps["isampler3D"] & NodeOps["isampler3D"];
  isamplerCube: WritableOps["isamplerCube"] & NodeOps["isamplerCube"];
  usampler2D: WritableOps["usampler2D"] & NodeOps["usampler2D"];
  usampler3D: WritableOps["usampler3D"] & NodeOps["usampler3D"];
  usamplerCube: WritableOps["usamplerCube"] & NodeOps["usamplerCube"];
  void: WritableOps["void"] & NodeOps["void"];
}

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
  float: FloatArithOps & FloatMathOps<"float"> & FloatComparisonOps;
  vec2: ArithOps<"vec2", Vec2Like | FloatLike> &
    FloatMathOps<"vec2"> &
    ComparisonOps<"bvec2", Vec2Like | FloatLike> &
    VecCommonOps<"vec2"> &
    Vec2Swizzles;
  vec3: ArithOps<"vec3", Vec3Like | FloatLike> &
    FloatMathOps<"vec3"> &
    ComparisonOps<"bvec3", Vec3Like | FloatLike> &
    VecCommonOps<"vec3"> &
    Vec3Ops &
    Vec3Swizzles;
  vec4: ArithOps<"vec4", Vec4Like | FloatLike> &
    FloatMathOps<"vec4"> &
    ComparisonOps<"bvec4", Vec4Like | FloatLike> &
    VecCommonOps<"vec4"> &
    Vec4Swizzles;
  int: IntOps;
  uint: UintOps;
  bool: BoolOps;
  ivec2: IVecOps<"ivec2"> & ComparisonOps<"bvec2", IVec2Like | IntLike> & IVec2Swizzles;
  ivec3: IVecOps<"ivec3"> & ComparisonOps<"bvec3", IVec3Like | IntLike> & IVec3Swizzles;
  ivec4: IVecOps<"ivec4"> & ComparisonOps<"bvec4", IVec4Like | IntLike> & IVec4Swizzles;
  uvec2: UVecOps<"uvec2"> & ComparisonOps<"bvec2", UVec2Like | UintLike> & UVec2Swizzles;
  uvec3: UVecOps<"uvec3"> & ComparisonOps<"bvec3", UVec3Like | UintLike> & UVec3Swizzles;
  uvec4: UVecOps<"uvec4"> & ComparisonOps<"bvec4", UVec4Like | UintLike> & UVec4Swizzles;
  bvec2: BoolVecOps<"bvec2"> & BVec2Swizzles;
  bvec3: BoolVecOps<"bvec3"> & BVec3Swizzles;
  bvec4: BoolVecOps<"bvec4"> & BVec4Swizzles;
  mat2: MatOps<"mat2", "vec2">;
  mat2x3: RectMatOps<"mat2x3", "vec2", "vec3", "mat3x2">;
  mat2x4: RectMatOps<"mat2x4", "vec2", "vec4", "mat4x2">;
  mat3x2: RectMatOps<"mat3x2", "vec3", "vec2", "mat2x3">;
  mat3: MatOps<"mat3", "vec3", "vec2">;
  mat3x4: RectMatOps<"mat3x4", "vec3", "vec4", "mat4x3">;
  mat4x2: RectMatOps<"mat4x2", "vec4", "vec2", "mat2x4">;
  mat4x3: RectMatOps<"mat4x3", "vec4", "vec3", "mat3x4">;
  mat4: MatOps<"mat4", "vec4", "vec3">;
  sampler2D: {};
  sampler3D: {};
  samplerCube: {};
  isampler2D: {};
  isampler3D: {};
  isamplerCube: {};
  usampler2D: {};
  usampler3D: {};
  usamplerCube: {};
  void: {};
}

export type Node<A extends ShaderType> = BaseNode<A> & NodeOps[A] & NodeMethods<A>;

// === Operation interfaces (shared across Node types) ===
export interface ArithOps<A extends ShaderType, Operand> {
  add(other: Operand): Node<A>;
  sub(other: Operand): Node<A>;
  mul(other: Operand): Node<A>;
  div(other: Operand): Node<A>;
  negate(): Node<A>;
}

/** A float's arithmetic, which takes the width of a vector operand: `float.mul(vec4)` is a `vec4`. */
export interface FloatArithOps {
  add(other: FloatLike): Node<"float">;
  add(other: Vec2Like): Node<"vec2">;
  add(other: Vec3Like): Node<"vec3">;
  add(other: Vec4Like): Node<"vec4">;
  sub(other: FloatLike): Node<"float">;
  sub(other: Vec2Like): Node<"vec2">;
  sub(other: Vec3Like): Node<"vec3">;
  sub(other: Vec4Like): Node<"vec4">;
  mul(other: FloatLike): Node<"float">;
  mul(other: Vec2Like): Node<"vec2">;
  mul(other: Vec3Like): Node<"vec3">;
  mul(other: Vec4Like): Node<"vec4">;
  div(other: FloatLike): Node<"float">;
  div(other: Vec2Like): Node<"vec2">;
  div(other: Vec3Like): Node<"vec3">;
  div(other: Vec4Like): Node<"vec4">;
  negate(): Node<"float">;
}

/** A float's comparisons, which take the width of a vector operand: `float.lessThan(vec3)` is a `bvec3`. */
export interface FloatComparisonOps {
  lessThan(other: FloatLike): Node<"bool">;
  lessThan(other: Vec2Like): Node<"bvec2">;
  lessThan(other: Vec3Like): Node<"bvec3">;
  lessThan(other: Vec4Like): Node<"bvec4">;
  greaterThan(other: FloatLike): Node<"bool">;
  greaterThan(other: Vec2Like): Node<"bvec2">;
  greaterThan(other: Vec3Like): Node<"bvec3">;
  greaterThan(other: Vec4Like): Node<"bvec4">;
  lessThanEqual(other: FloatLike): Node<"bool">;
  lessThanEqual(other: Vec2Like): Node<"bvec2">;
  lessThanEqual(other: Vec3Like): Node<"bvec3">;
  lessThanEqual(other: Vec4Like): Node<"bvec4">;
  greaterThanEqual(other: FloatLike): Node<"bool">;
  greaterThanEqual(other: Vec2Like): Node<"bvec2">;
  greaterThanEqual(other: Vec3Like): Node<"bvec3">;
  greaterThanEqual(other: Vec4Like): Node<"bvec4">;
  equal(other: FloatLike): Node<"bool">;
  equal(other: Vec2Like): Node<"bvec2">;
  equal(other: Vec3Like): Node<"bvec3">;
  equal(other: Vec4Like): Node<"bvec4">;
  notEqual(other: FloatLike): Node<"bool">;
  notEqual(other: Vec2Like): Node<"bvec2">;
  notEqual(other: Vec3Like): Node<"bvec3">;
  notEqual(other: Vec4Like): Node<"bvec4">;
}

export interface FloatMathOps<A extends ShaderType> {
  sin(): Node<A>;
  cos(): Node<A>;
  tan(): Node<A>;
  asin(): Node<A>;
  acos(): Node<A>;
  atan(other?: FloatLike): Node<A>;
  sinh(): Node<A>;
  cosh(): Node<A>;
  tanh(): Node<A>;
  asinh(): Node<A>;
  acosh(): Node<A>;
  atanh(): Node<A>;
  abs(): Node<A>;
  sign(): Node<A>;
  floor(): Node<A>;
  ceil(): Node<A>;
  fract(): Node<A>;
  round(): Node<A>;
  trunc(): Node<A>;
  radians(): Node<A>;
  degrees(): Node<A>;
  sqrt(): Node<A>;
  inverseSqrt(): Node<A>;
  inversesqrt(): Node<A>;
  exp(): Node<A>;
  log(): Node<A>;
  exp2(): Node<A>;
  log2(): Node<A>;
  cbrt(): Node<A>;
  reciprocal(): Node<A>;
  oneMinus(): Node<A>;
  difference(other: Node<A> | FloatLike): Node<A>;
  lengthSq(): Node<A>;
  saturate(): Node<A>;
  pow(e: FloatLike): Node<A>;
  pow2(): Node<A>;
  pow3(): Node<A>;
  pow4(): Node<A>;
  min(other: FloatLike): Node<A>;
  max(other: FloatLike): Node<A>;
  mod(other: FloatLike): Node<A>;
  mix(b: Node<A>, t: FloatLike): Node<A>;
  clamp(min: FloatLike, max: FloatLike): Node<A>;
  /**
   * Declared here rather than on VecCommonOps so floats get them too, with
   * only one declaration each. GLSL/WGSL both allow either edge form
   * (`step(genType, genType)` or `step(float, genType)`, and likewise for
   * smoothstep).
   */
  step(edge: Node<A> | FloatLike): Node<A>;
  smoothstep(edge0: Node<A> | FloatLike, edge1: Node<A> | FloatLike): Node<A>;
  fwidth(): Node<A>;
  /**
   * Meaningful in a fragment stage on both backends; GLSL names them
   * dFdx/dFdy and WGSL dpdx/dpdy.
   */
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
export type MatrixType = "mat2" | "mat3" | "mat4" | "mat2x3" | "mat2x4" | "mat3x2" | "mat3x4" | "mat4x2" | "mat4x3";

type MatrixColumns<M extends MatrixType> = M extends `mat${infer C}x${string}`
  ? C extends `${infer N extends number}`
    ? N
    : never
  : M extends `mat${infer N extends number}`
    ? N
    : never;

type MatrixRows<M extends MatrixType> = M extends `mat${string}x${infer R}`
  ? R extends `${infer N extends number}`
    ? N
    : never
  : M extends `mat${infer N extends number}`
    ? N
    : never;

type MatName<C extends number, R extends number> = C extends R ? `mat${C}` : `mat${C}x${R}`;

/** The result type of a matCxR times a matC2xR2, or `never` when the shapes don't meet. */
export type MatrixProduct<A extends MatrixType, B extends MatrixType> =
  MatrixColumns<A> extends MatrixRows<B> ? MatName<MatrixColumns<B>, MatrixRows<A>> : never;

type MatMulResult<Self extends MatrixType, O extends MatrixType> =
  MatrixProduct<Self, O> extends infer P ? (P extends ShaderType ? Node<P> : never) : never;

export interface MatOps<Self extends MatrixType, Vec extends ShaderType, Shorter extends ShaderType = never> {
  mul<O extends MatrixType>(other: Node<O>): MatMulResult<Self, O>;
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
  Self extends MatrixType,
  Operand extends ShaderType,
  Column extends ShaderType,
  Transposed extends ShaderType,
> {
  mul<O extends MatrixType>(other: Node<O>): MatMulResult<Self, O>;
  mul(other: Node<Operand>): Node<Column>;
  element(i: IntLike): Node<Column>;
  transpose(): Node<Transposed>;
}

/** A float, a signed integer, or a vector of either, as `abs`, `sign` and `negate` take it. */
export type SignedValueLike = FloatValueLike | IntLike | IVec2Like | IVec3Like | IVec4Like;

/** An integer or integer vector of either kind, as the bitwise operators and the shifts take it. */
export type IntegerValueLike =
  IntLike | UintLike | IVec2Like | IVec3Like | IVec4Like | UVec2Like | UVec3Like | UVec4Like;

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
  shiftLeft(other: IntLike | UintLike): Node<"int">;
  shiftRight(other: IntLike | UintLike): Node<"int">;
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
  shiftLeft(other: UintLike | IntLike): Node<"uint">;
  shiftRight(other: UintLike | IntLike): Node<"uint">;
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
  shiftLeft(other: IntegerValueLike): Node<A>;
  shiftRight(other: IntegerValueLike): Node<A>;
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
  shiftLeft(other: IntegerValueLike): Node<A>;
  shiftRight(other: IntegerValueLike): Node<A>;
  bitNot(): Node<A>;
  element(i: IntLike): Node<"uint">;
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
  toVar(name?: string): Var<A>;
  /** TSL's shorthand for `toVar()`. */
  var(name?: string): Var<A>;
  /** This program, dispatched once per index in `0..count`, as TSL's `.compute()`. */
  compute(count: number, workgroupSize?: number): ComputeNode;
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
    ifTrue: BaseNode<T> | number | VectorArray,
    ifFalse: (BaseNode<T> | number | VectorArray) & OfOneKind<T>,
  ): Node<T>;
}

/** The kind of the components of `T`: `"int"`, `"uint"`, `"bool"`, or `"float"` for a float, float vector or matrix. */
type ComponentKindOf<T extends ShaderType> = T extends "int" | "ivec2" | "ivec3" | "ivec4"
  ? "int"
  : T extends "uint" | "uvec2" | "uvec3" | "uvec4"
    ? "uint"
    : T extends "bool" | "bvec2" | "bvec3" | "bvec4"
      ? "bool"
      : "float";

/**
 * `unknown` when every type in `T` has one component kind, `never` when `T`
 * mixes two. Two numbers leave `T` at every shader type, which mixes nothing.
 */
type OfOneKind<T extends ShaderType> = ShaderType extends T
  ? unknown
  : true extends IsUnion<ComponentKindOf<T>>
    ? never
    : unknown;

/** Whether `T` is a union of two or more types. */
type IsUnion<T, U = T> = T extends unknown ? ([U] extends [T] ? false : true) : never;

// === NodeImpl - defines all methods, Node<A> hides typed subset ===
/**
 * Every kind of node rmsl builds. A node of any other kind is refused where it
 * is made, so a new kind cannot be added without listing it here, and
 * `deserialize` refuses data that names a kind not on the list.
 */
export const NODE_TYPES: ReadonlySet<string> = new Set([
  "abs",
  "acos",
  "acosh",
  "add",
  "all",
  "and",
  "any",
  "asin",
  "asinh",
  "assign",
  "atan",
  "atan2",
  "atanh",
  "attribute",
  "bitAnd",
  "bitNot",
  "bitOr",
  "bitXor",
  "bool",
  "break",
  "builtinFragDepth",
  "builtinPosition",
  "bvec2",
  "bvec3",
  "bvec4",
  "ceil",
  "clamp",
  "construct",
  "continue",
  "cos",
  "cosh",
  "cross",
  "dFdx",
  "dFdy",
  "degrees",
  "determinant",
  "discard",
  "distance",
  "div",
  "dot",
  "equal",
  "exp",
  "exp2",
  "faceForward",
  "float",
  "floor",
  "for",
  "fract",
  "fragCoord",
  "fwidth",
  "greaterThan",
  "greaterThanEqual",
  "if",
  "instanceIndex",
  "int",
  "inverse",
  "inverseSqrt",
  "invocationIndex",
  "ivec2",
  "ivec3",
  "ivec4",
  "length",
  "lessThan",
  "lessThanEqual",
  "let",
  "log",
  "log2",
  "mat2",
  "mat2x3",
  "mat2x4",
  "mat3",
  "mat3x2",
  "mat3x4",
  "mat4",
  "mat4x2",
  "mat4x3",
  "matVecMul",
  "matrixElement",
  "max",
  "min",
  "mix",
  "mod",
  "mul",
  "negate",
  "normalize",
  "not",
  "notEqual",
  "or",
  "output",
  "outputStruct",
  "pow",
  "radians",
  "reflect",
  "refract",
  "return",
  "round",
  "select",
  "seq",
  "shiftLeft",
  "shiftRight",
  "sign",
  "sin",
  "sinh",
  "smoothstep",
  "sqrt",
  "step",
  "storage",
  "storageElement",
  "sub",
  "swizzle",
  "tan",
  "tanh",
  "texture",
  "textureLoad",
  "textureLod",
  "textureSize",
  "transpose",
  "trunc",
  "uint",
  "uniform",
  "uniformArray",
  "uniformArrayElement",
  "uvec2",
  "uvec3",
  "uvec4",
  "var",
  "varying",
  "vec2",
  "vec3",
  "vec4",
  "vectorElement",
  "vertexIndex",
  "void",
  "while",
]);

export class NodeImpl<A extends ShaderType> implements BaseNode<A> {
  declare [__brand]: A;
  _t: string;
  type: string;
  params?: BaseNode<ShaderType>[];
  value?: unknown;

  constructor(config: { _t: string; type: string; params?: BaseNode<ShaderType>[]; value?: unknown }) {
    if (!NODE_TYPES.has(config.type)) throw new Error(`[RMSL] "${config.type}" is not a kind of node rmsl has`);
    this._t = config._t;
    this.type = config.type;
    this.params = config.params;
    this.value = config.value;
  }

  // === ArithOps ===
  add(other: any): any {
    return op("add", this, other);
  }
  sub(other: any): any {
    return op("sub", this, other);
  }
  mul(other: any): any {
    // A matCxR times a vecC gives a vecR. A vector one component short of the
    // column width (`mat4 * vec3`) is a position with its homogeneous
    // coordinate implied, promoted here and truncated by matVecMul.
    let shape = MATRIX_DIMENSIONS[this._t];
    let otherType = other?._t;
    if (shape !== undefined && typeof otherType === "string" && otherType.startsWith("vec")) {
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
        `[RMSL] A ${this._t} cannot multiply ${withArticle(otherType)}: the vector must have ` +
          `the matrix's column width or one fewer component (a position with its ` +
          `homogeneous coordinate implied).`,
      );
    }
    let otherShape = typeof otherType === "string" ? MATRIX_DIMENSIONS[otherType] : undefined;
    if (shape !== undefined && otherShape !== undefined) {
      let [c1, r1] = shape;
      let [c2, r2] = otherShape;
      if (c1 !== r2) {
        throw new Error(
          `[RMSL] A ${this._t} cannot multiply ${withArticle(otherType)}: the left has ${c1} ` +
            `column(s) but the right has ${r2} row(s), and a matrix product needs ` +
            `the left's columns to equal the right's rows.`,
        );
      }
      let resultType = c2 === r1 ? `mat${c2}` : `mat${c2}x${r1}`;
      return node({
        _t: resultType,
        type: "mul",
        params: [this as BaseNode<ShaderType>, wrapValue(other) as BaseNode<ShaderType>],
      });
    }
    return op("mul", this, other);
  }
  div(other: any): any {
    return op("div", this, other);
  }
  negate(): any {
    return op("negate", this);
  }

  // === FloatMathOps ===
  sin() {
    return op1("sin", this);
  }
  cos() {
    return op1("cos", this);
  }
  tan() {
    return op1("tan", this);
  }
  asin() {
    return op1("asin", this);
  }
  acos() {
    return op1("acos", this);
  }
  atan(other?: any) {
    return other === undefined ? op1("atan", this) : op("atan2", this, other);
  }
  sinh() {
    return op1("sinh", this);
  }
  cosh() {
    return op1("cosh", this);
  }
  tanh() {
    return op1("tanh", this);
  }
  asinh() {
    return op1("asinh", this);
  }
  acosh() {
    return op1("acosh", this);
  }
  atanh() {
    return op1("atanh", this);
  }
  abs() {
    return op1("abs", this);
  }
  sign() {
    return op1("sign", this);
  }
  floor() {
    return op1("floor", this);
  }
  ceil() {
    return op1("ceil", this);
  }
  fract() {
    return op1("fract", this);
  }
  round() {
    return op1("round", this);
  }
  trunc() {
    return op1("trunc", this);
  }
  radians() {
    return op1("radians", this);
  }
  degrees() {
    return op1("degrees", this);
  }
  sqrt() {
    return op1("sqrt", this);
  }
  inverseSqrt() {
    return op1("inverseSqrt", this);
  }
  inversesqrt() {
    return op1("inverseSqrt", this);
  }
  exp() {
    return op1("exp", this);
  }
  log() {
    return op1("log", this);
  }
  exp2() {
    return op1("exp2", this);
  }
  log2() {
    return op1("log2", this);
  }
  cbrt() {
    assertOperands("cbrt", [this as BaseNode<ShaderType>]);
    return op("mul", this.sign(), op("pow", this.abs(), 1.0 / 3.0));
  }
  reciprocal() {
    assertOperands("reciprocal", [this as BaseNode<ShaderType>]);
    return op("div", 1, this);
  }
  oneMinus() {
    assertOperands("oneMinus", [this as BaseNode<ShaderType>]);
    return op("sub", 1, this);
  }
  difference(other: any) {
    return op1("abs", op("sub", this, other));
  }
  lengthSq(): any {
    // For a vector this is the squared length, the dot of itself; for a scalar
    // it is simply its square — neither language offers a scalar `dot`.
    assertOperands("lengthSq", [this as BaseNode<ShaderType>]);
    return (TYPE_WIDTH[this._t] ?? 1) > 1 ? op("dot", this, this) : op("mul", this, this);
  }
  saturate() {
    return op("clamp", this, 0, 1);
  }
  pow(e: any) {
    return op("pow", this, e);
  }
  pow2() {
    return op("mul", this, this);
  }
  pow3() {
    return op("mul", this, this, this);
  }
  pow4() {
    return op("mul", this, this, this, this);
  }
  min(other: any) {
    return op("min", this, other);
  }
  max(other: any) {
    return op("max", this, other);
  }
  mod(other: any): any {
    return op("mod", this, other);
  }
  dFdx() {
    return op1("dFdx", this);
  }
  dFdy() {
    return op1("dFdy", this);
  }

  // === Comparison ops ===
  lessThan(other: any) {
    return comp("lessThan", this, other);
  }
  greaterThan(other: any) {
    return comp("greaterThan", this, other);
  }
  lessThanEqual(other: any) {
    return comp("lessThanEqual", this, other);
  }
  greaterThanEqual(other: any) {
    return comp("greaterThanEqual", this, other);
  }
  equal(other: any) {
    return comp("equal", this, other);
  }
  notEqual(other: any) {
    return comp("notEqual", this, other);
  }

  // === VecCommonOps ===
  dot(other: any): any {
    return op("dot", this, other);
  }
  length(): any {
    return op1("length", this);
  }
  normalize(): any {
    return op1("normalize", this);
  }
  distance(other: any): any {
    return op("distance", this, other);
  }
  reflect(normal: any): any {
    return op("reflect", this, normal);
  }
  refract(normal: any, eta: any): any {
    return op("refract", this, normal, eta);
  }
  faceForward(incident: any, reference: any): any {
    return op("faceForward", this, incident, reference);
  }
  clamp(minV: any, maxV: any): any {
    return op("clamp", this, minV, maxV);
  }
  mix(b: any, t: any): any {
    return op("mix", this, b, t);
  }
  step(edge: any): any {
    return op("step", edge, this);
  }
  smoothstep(edge0: any, edge1: any): any {
    return op("smoothstep", edge0, edge1, this);
  }
  fwidth(): any {
    return op1("fwidth", this);
  }

  // === Vec3Ops ===
  cross(other: any): any {
    return op("cross", this, other);
  }

  // === MatOps ===
  // The argument is an index, so a plain number is always typed as an integer.
  // A matrix indexes to one of its columns; a vector to one of its components.
  element(i: any): any {
    const index =
      typeof i === "number"
        ? node({
            _t: "int",
            type: "int",
            value: i | 0,
          })
        : i;

    if (this.type === "storage") {
      return node({
        _t: this._t,
        type: "storageElement",
        params: [this as BaseNode<ShaderType>, wrapValue(index, "element") as BaseNode<ShaderType>],
      });
    }

    const isVector = /^(vec|ivec|uvec|bvec)[234]$/.test(this._t);

    return op(isVector ? "vectorElement" : "matrixElement", this, index);
  }
  inverse() {
    return op1("inverse", this);
  }
  transpose() {
    return op1("transpose", this);
  }
  determinant() {
    return op1("determinant", this);
  }

  // === IntOps ===
  bitAnd(other: any) {
    return op("bitAnd", this, other);
  }
  bitOr(other: any) {
    return op("bitOr", this, other);
  }
  bitXor(other: any) {
    return op("bitXor", this, other);
  }
  shiftLeft(other: any) {
    return op("shiftLeft", this, other);
  }
  shiftRight(other: any) {
    return op("shiftRight", this, other);
  }
  bitNot(): any {
    assertOperands("bitNot", [this as BaseNode<ShaderType>]);
    return node({
      _t: this._t,
      type: "bitNot",
      params: [this as BaseNode<ShaderType>],
    });
  }

  // === Uniform precision ===
  setPrecision(precision: UniformPrecision): any {
    if (this.type !== "uniform" && this.type !== "uniformArray") {
      throw new Error(
        `[RMSL] setPrecision: only a uniform has a precision of its own, not ${withArticle(this.type)} node`,
      );
    }
    if (precision !== "low" && precision !== "medium" && precision !== "high") {
      throw new Error(
        `[RMSL] setPrecision: a precision is "low", "medium" or "high", not ${JSON.stringify(precision)}`,
      );
    }
    const shaderType = (this.value as { shaderType: string }).shaderType;
    if (shaderType === "bool" || shaderType.startsWith("bvec")) {
      throw new Error(
        `[RMSL] setPrecision: ${withArticle(shaderType)} uniform takes no precision, which GLSL gives only to numbers`,
      );
    }
    (this.value as { precision?: UniformPrecision }).precision = precision;
    return this;
  }

  // === BoolOps ===
  and(other: any): any {
    return op("and", this, other);
  }
  or(other: any): any {
    return op("or", this, other);
  }
  not(): any {
    return op1("not", this);
  }
  xor(other: any): any {
    // Neither language has a logical xor: `(a || b) && !(a && b)` is the same
    // truth table for scalars and component-wise for boolean vectors.
    return op("and", op("or", this, other), op1("not", op("and", this, other)));
  }
  all(): any {
    return node({ _t: "bool", type: "all", params: [this as BaseNode<ShaderType>] });
  }
  any(): any {
    return node({ _t: "bool", type: "any", params: [this as BaseNode<ShaderType>] });
  }

  // === NodeMethods ===
  assign(value: BaseNode<A>): void {
    assertBlockScope("assign", (blockScope) => {
      blockScope.push(
        new NodeImpl({
          _t: "void",
          type: "assign",
          params: [this as BaseNode<ShaderType>, typedOperand(value, this._t, "assign") as BaseNode<ShaderType>],
        }),
      );
    });
  }

  compute(count: number, workgroupSize?: number): ComputeNode {
    return compute(this as unknown as Node<ShaderType>, count, workgroupSize);
  }

  toVar(name?: string): Node<A> {
    let v: Node<A>;
    assertBlockScope("toVar", (blockScope) => {
      let varName = claimVarName(name);
      v = var_(varName, this._t) as Node<A>;
      blockScope.push(
        new NodeImpl({
          _t: "void",
          type: "let",
          params: [v as BaseNode<ShaderType>, this as BaseNode<ShaderType>],
        }),
      );
    });
    return v!;
  }

  var(name?: string): Node<A> {
    return this.toVar(name);
  }

  // === Compound assignments ===
  addAssign(other: any) {
    this.assign(this.add(other));
  }
  subAssign(other: any) {
    this.assign(this.sub(other));
  }
  mulAssign(other: any) {
    this.assign(this.mul(other));
  }
  divAssign(other: any) {
    this.assign(this.div(other));
  }
  modAssign(other: any) {
    this.assign(this.mod(other));
  }

  // === Conversions (cast to a different type) ===
  toFloat(): any {
    return node({ _t: "float", type: "construct", params: [this as BaseNode<ShaderType>] });
  }
  toInt(): any {
    return node({ _t: "int", type: "construct", params: [this as BaseNode<ShaderType>] });
  }
  toUint(): any {
    return node({ _t: "uint", type: "construct", params: [this as BaseNode<ShaderType>] });
  }
  toBool(): any {
    return node({ _t: "bool", type: "construct", params: [this as BaseNode<ShaderType>] });
  }
  toVec2(): any {
    return node({ _t: "vec2", type: "construct", params: [this as BaseNode<ShaderType>] });
  }
  toVec3(): any {
    return node({ _t: "vec3", type: "construct", params: [this as BaseNode<ShaderType>] });
  }
  toVec4(): any {
    return node({ _t: "vec4", type: "construct", params: [this as BaseNode<ShaderType>] });
  }
  toIVec2(): any {
    return node({ _t: "ivec2", type: "construct", params: [this as BaseNode<ShaderType>] });
  }
  toIVec3(): any {
    return node({ _t: "ivec3", type: "construct", params: [this as BaseNode<ShaderType>] });
  }
  toIVec4(): any {
    return node({ _t: "ivec4", type: "construct", params: [this as BaseNode<ShaderType>] });
  }
  toUVec2(): any {
    return node({ _t: "uvec2", type: "construct", params: [this as BaseNode<ShaderType>] });
  }
  toUVec3(): any {
    return node({ _t: "uvec3", type: "construct", params: [this as BaseNode<ShaderType>] });
  }
  toUVec4(): any {
    return node({ _t: "uvec4", type: "construct", params: [this as BaseNode<ShaderType>] });
  }
  toBVec2(): any {
    return node({ _t: "bvec2", type: "construct", params: [this as BaseNode<ShaderType>] });
  }
  toBVec3(): any {
    return node({ _t: "bvec3", type: "construct", params: [this as BaseNode<ShaderType>] });
  }
  toBVec4(): any {
    return node({ _t: "bvec4", type: "construct", params: [this as BaseNode<ShaderType>] });
  }
  toMat2(): any {
    return node({ _t: "mat2", type: "construct", params: [this as BaseNode<ShaderType>] });
  }
  toMat3(): any {
    return node({ _t: "mat3", type: "construct", params: [this as BaseNode<ShaderType>] });
  }
  toMat4(): any {
    return node({ _t: "mat4", type: "construct", params: [this as BaseNode<ShaderType>] });
  }
  convert<T extends ShaderType>(target: T): any {
    return node({ _t: target, type: "construct", params: [this as BaseNode<ShaderType>] });
  }
  select(ifTrue: any, ifFalse: any): any {
    let a = toNodeBeside(ifTrue, ifFalse, "select") as BaseNode<ShaderType>;
    let b = typedOperand(ifFalse, (a as any)?._t ?? "float", "select");
    assertOperands("select", [this as BaseNode<ShaderType>, a, b]);
    // The result type follows the branches, not the condition. `vec3(0).equal(1).select(v, w)`
    // is a vec3 no matter that the selector is a bvec3.
    let t = (a as any)?._t || (b as any)?._t || this._t;
    return node({
      _t: t,
      type: "select",
      params: [this as BaseNode<ShaderType>, a, b],
    });
  }
}

/** Every swizzle pattern of one to four letters in one spelling, as the `Swizzles` types list them. */
function swizzlePatterns(letters: string): string[] {
  let patterns = [...letters];
  let all = [...patterns];
  for (let length = 2; length <= 4; length++) {
    patterns = patterns.flatMap((prefix) => [...letters].map((letter) => prefix + letter));
    all.push(...patterns);
  }
  return all;
}

// Defined on the prototype rather than written out: 1020 patterns, one getter
// shape. Which of them a node offers is up to its type (`Swizzles`).
for (const pattern of ["xyzw", "rgba", "stpq"].flatMap(swizzlePatterns)) {
  Object.defineProperty(NodeImpl.prototype, pattern, {
    get(this: NodeImpl<ShaderType>) {
      return swizzle(this, pattern);
    },
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

export function var_<A extends ShaderType>(varName: string, brandType: string): Var<A> {
  return new Node<A>({
    _t: brandType,
    type: "var",
    value: { varName, varType: brandType },
  }) as Var<A>;
}

export function isNode(x: any): x is BaseNode<ShaderType> {
  return typeof x === "object" && x !== null && "_t" in x && "type" in x;
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
export type ExtractType<V> = V extends FloatLike
  ? "float"
  : V extends Vec2Like
    ? "vec2"
    : V extends Vec3Like
      ? "vec3"
      : V extends Vec4Like
        ? "vec4"
        : V extends IntLike
          ? "int"
          : V extends UintLike
            ? "uint"
            : V extends BooleanLike
              ? "bool"
              : V extends Mat3Like
                ? "mat3"
                : V extends Mat4Like
                  ? "mat4"
                  : "void";

/** The vector or matrix type a JavaScript array of each length is. */
const ARRAY_TYPES: Record<number, "vec2" | "vec3" | "vec4" | "mat3" | "mat4"> = {
  2: "vec2",
  3: "vec3",
  4: "vec4",
  9: "mat3",
  16: "mat4",
};

/** What an element of an array is, for the message that refuses it. */
function describeElement(element: unknown): string {
  if (element === undefined) return "an empty or undefined element";
  if (Array.isArray(element)) return "a nested array";
  return `${withArticle(typeof element)}, ${JSON.stringify(element)}`;
}

/**
 * `x` as a node: a number as a float, a boolean as a bool, and a JavaScript
 * array as the vector or matrix of its length. `operation` names the call that
 * was given `x`, for the message that refuses an array.
 */
export function wrapValue<V>(x: V, operation?: string): Node<ExtractType<V>> {
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
    const type = ARRAY_TYPES[x.length];
    const givenTo = operation === undefined ? "" : ` given to ${operation}()`;
    if (type === undefined) {
      throw new Error(
        `[RMSL] A JavaScript array of length ${x.length}${givenTo} is no vector or matrix. Give 2, 3 or 4 numbers for a vector, or 9 or 16 for a mat3 or mat4. Other matrices come from mat2, mat2x3 and the other matrix constructors.`,
      );
    }
    const at = x.findIndex((element) => typeof element !== "number" && !isNode(element));
    if (at >= 0) {
      throw new Error(
        `[RMSL] A JavaScript array${givenTo} as ${withArticle(type)} holds numbers or nodes, and this one holds ${describeElement(x[at])}. Build the ${type} from nodes with ${type}(...) instead.`,
      );
    }
    if (x.every((element) => typeof element === "number")) return node({ _t: type, type, value: x }) as any;
    // An array that holds nodes is built as its constructor builds it.
    const construct = { vec2, vec3, vec4, mat3, mat4 }[type] as (...elements: unknown[]) => unknown;
    return construct(...x) as any;
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

/** Component count of each scalar and vector type, as literals the types read. */
const VECTOR_WIDTHS = {
  float: 1,
  int: 1,
  uint: 1,
  bool: 1,
  vec2: 2,
  vec3: 3,
  vec4: 4,
  ivec2: 2,
  ivec3: 3,
  ivec4: 4,
  uvec2: 2,
  uvec3: 3,
  uvec4: 4,
  bvec2: 2,
  bvec3: 3,
  bvec4: 4,
} as const satisfies Record<string, number>;

/** Component count per type, for the operations whose width follows it. */
export const TYPE_WIDTH: Record<string, number> = VECTOR_WIDTHS;

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
export const UNIFORM_OPERAND_OPS = new Set(["step", "smoothstep", "clamp", "min", "max", "pow", "mod"]);

/**
 * Give a plain JavaScript number the type of the operand it sits beside.
 *
 * A number carries no shader type of its own and wrapValue can only guess — it
 * picks float. Beside an integer operand that guess is wrong: the two operands
 * disagree, codegen inserts a conversion, and the result stops matching the
 * node's own type, as in `int x = (float(u) % 2.0)`.
 *
 * A number the operand's type cannot represent is refused rather than quietly
 * reinterpreted. `operation` names the call that was given `value`, as
 * {@link wrapValue} takes it.
 */
export function typedOperand(value: any, operandType: string, operation?: string): BaseNode<ShaderType> {
  // A bare number beside an integer vector is broadcast as its component type —
  // the int of an ivec, the uint of a uvec — rather than as the vector itself,
  // since `op`/`comp` construct the broadcast from whatever this wraps.
  let scalarType = /^ivec/.test(operandType) ? "int" : /^uvec/.test(operandType) ? "uint" : operandType;
  let isIntegral = scalarType === "int" || scalarType === "uint";
  if (typeof value !== "number" || !isIntegral) {
    return wrapValue(value, operation) as BaseNode<ShaderType>;
  }
  if (!Number.isInteger(value)) {
    throw new Error(
      `[RMSL] ${value} is not a whole number, but the operand beside it is an ` +
        `${operandType}. Convert the operand to a float, or use a whole number.`,
    );
  }
  if (scalarType === "uint" && value < 0) {
    throw new Error(
      `[RMSL] ${value} is negative, but the operand beside it is unsigned. ` +
        `Use a signed operand, or a literal that is not negative.`,
    );
  }
  return node({
    _t: scalarType,
    type: scalarType,
    value: integerLiteral(scalarType as "int" | "uint", value),
  }) as BaseNode<ShaderType>;
}

/** The comparisons, which `comp` builds. */
const COMPARISON_OPS = new Set(["equal", "notEqual", "lessThan", "greaterThan", "lessThanEqual", "greaterThanEqual"]);

/** The operations whose operands share one component kind: signed integer, unsigned integer or float. `select` checks its two branches. */
const SAME_KIND_OPS = new Set([
  "add",
  "sub",
  "mul",
  "div",
  "mod",
  "min",
  "max",
  "clamp",
  "bitAnd",
  "bitOr",
  "bitXor",
  "select",
  ...COMPARISON_OPS,
]);

/** The operations GLSL ES 3.00 and WGSL define only for floats and float vectors. */
const FLOAT_ONLY_OPS = new Set([
  "radians",
  "degrees",
  "sin",
  "cos",
  "tan",
  "asin",
  "acos",
  "atan",
  "atan2",
  "sinh",
  "cosh",
  "tanh",
  "asinh",
  "acosh",
  "atanh",
  "pow",
  "exp",
  "log",
  "exp2",
  "log2",
  "sqrt",
  "inverseSqrt",
  "floor",
  "ceil",
  "fract",
  "round",
  "trunc",
  "mix",
  "step",
  "smoothstep",
  "length",
  "distance",
  "dot",
  "cross",
  "normalize",
  "faceForward",
  "reflect",
  "refract",
  "dFdx",
  "dFdy",
  "fwidth",
  "cbrt",
  "reciprocal",
  "oneMinus",
  "lengthSq",
]);

/** The operations GLSL ES 3.00 and WGSL define only for integers and integer vectors. A shift's amount may be of either integer kind. */
const INTEGER_ONLY_OPS = new Set(["bitAnd", "bitOr", "bitXor", "bitNot", "shiftLeft", "shiftRight"]);

/** The operations GLSL ES 3.00 and WGSL define only for floats and signed integers, a matrix included for `negate`. */
const SIGNED_OPS = new Set(["abs", "sign", "negate"]);

/** The operations whose two operands are of one type, so a scalar does not broadcast beside a vector. */
const ONE_TYPE_OPS = new Set(["dot", "distance"]);

/** The operations whose operands are vectors of one width, or a scalar beside a vector. A `select` may pick a scalar or a vector. */
const ONE_WIDTH_OPS = new Set(
  [
    ...SAME_KIND_OPS,
    ...FLOAT_ONLY_OPS,
    ...INTEGER_ONLY_OPS,
    "and",
    "or",
    "negate",
    "vectorElement",
    "matrixElement",
  ].filter((type) => type !== "select"),
);

/** The name the program calls an operation by, where its node is named otherwise. */
const OPERATION_NAMES: Record<string, string> = { atan2: "atan" };

type ComponentKind = "int" | "uint" | "float";

/** The kind of the components of `type`: `"int"` for an int or ivec, `"uint"` for a uint or uvec, `"float"` for a float, float vector or matrix, otherwise `undefined`. */
function componentKind(type: string | undefined): ComponentKind | undefined {
  if (type === undefined) return undefined;
  if (/^(int|ivec[234])$/.test(type)) return "int";
  if (/^(uint|uvec[234])$/.test(type)) return "uint";
  if (/^(float|vec[234]|mat)/.test(type)) return "float";
  return undefined;
}

/** The method that converts a value of `type` to the same width of `kind`, such as `toUVec3()` for an ivec3 to uint, or `undefined` for a matrix. */
function conversionTo(type: string, kind: ComponentKind): string | undefined {
  if (MATRIX_DIMENSIONS[type]) return undefined;
  let width = /vec([234])$/.exec(type)?.[1];
  if (width) return `to${{ int: "IVec", uint: "UVec", float: "Vec" }[kind]}${width}()`;
  return `to${{ int: "Int", uint: "Uint", float: "Float" }[kind]}()`;
}

/**
 * Why an operation of `type` cannot take operands of `operandTypes`, or
 * `undefined` when it can. The builders ask it as they make a node, so the
 * refusal points at the call, and every compiler asks it again of each node
 * it compiles, so a graph `deserialize` rebuilt is refused alike.
 *
 * A function defined only for floats refuses an integer or boolean operand,
 * an operation defined only for integers a float or boolean one, and one
 * defined only for signed values an unsigned or boolean one. Operands of two
 * kinds are refused, as are vectors of two widths.
 */
export function operandRefusal(type: string, operandTypes: readonly (string | undefined)[]): string | undefined {
  let name = OPERATION_NAMES[type] ?? type;
  let types = (type === "select" ? operandTypes.slice(1) : operandTypes).filter((t): t is string => t !== undefined);
  if (FLOAT_ONLY_OPS.has(type)) {
    let wrong = types.find((t) => !/^(float|vec[234])$/.test(t));
    if (wrong !== undefined) {
      let fix = MATRIX_DIMENSIONS[wrong]
        ? "Take it a column at a time, with element(i)."
        : `Convert it with ${conversionTo(wrong, "float")} first.`;
      return `${name}() takes a float or a float vector, not ${wrong}. ${fix}`;
    }
    if (ONE_TYPE_OPS.has(type) && types[0] !== types[1] && types.includes("float")) {
      return `${name}() takes two arguments of one type, not ${types[0]} and ${types[1]}.`;
    }
  }
  if (INTEGER_ONLY_OPS.has(type)) {
    let isInteger = (t: string) => componentKind(t) === "int" || componentKind(t) === "uint";
    let wrong = types.find((t) => !isInteger(t));
    if (wrong !== undefined) {
      let beside = types.find(isInteger);
      let fix = conversionTo(wrong, beside ? componentKind(beside)! : "int");
      return `${name}() takes an integer or an integer vector, not ${wrong}.${fix ? ` Convert it with ${fix} first.` : ""}`;
    }
  }
  if (SIGNED_OPS.has(type)) {
    let wrong = types.find((t) => componentKind(t) !== "int" && componentKind(t) !== "float");
    if (wrong !== undefined) {
      return `${name}() takes a float or a signed integer, scalar or vector, not ${wrong}. Convert it with ${conversionTo(wrong, "int")} first.`;
    }
  }
  if (SAME_KIND_OPS.has(type)) {
    let typed = types.flatMap((t) => {
      let kind = componentKind(t);
      return kind ? [{ t, kind }] : [];
    });
    let other = typed.find((p) => p.kind !== typed[0].kind);
    if (other) {
      let first = typed[0];
      let fixes = [
        [first.t, conversionTo(first.t, other.kind)],
        [other.t, conversionTo(other.t, first.kind)],
      ].filter(([, fix]) => fix !== undefined);
      return (
        `${name}() takes operands of one kind, not ${first.t} and ${other.t}. ` +
        `Convert ${fixes.map(([t, fix]) => `the ${t} with ${fix}`).join(" or ")} first.`
      );
    }
  }
  if (ONE_WIDTH_OPS.has(type) && new Set(types.map((t) => TYPE_WIDTH[t] ?? 1).filter((w) => w > 1)).size > 1) {
    return `${name}() takes vectors of one width, not ${types.join(" and ")}. Swizzle one of them to the width of the other.`;
  }
  return undefined;
}

/** Throw the refusal {@link operandRefusal} gives an operation of `type` on `params`, if any. */
function assertOperands(type: string, params: readonly BaseNode<ShaderType>[]): void {
  let refusal = operandRefusal(
    type,
    params.map((p) => (p as any)?._t),
  );
  if (refusal) throw new Error(`[RMSL] ${refusal}`);
}

export function op(type: string, ...args: any[]): Node<ShaderType> {
  let first = wrapValue(args[0], type) as BaseNode<ShaderType>;
  let firstT = (first as any)?._t || "float";
  let params = [first, ...args.slice(1).map((a) => typedOperand(a, firstT, type))];
  assertOperands(type, params);
  // The result follows the *widest* operand, so a scalar broadcast beside a
  // vector keeps the vector type (`1 - vec3` is still vec3).
  let valueIndex = VALUE_OPERAND[type] ?? 0;
  let valueT = (params[valueIndex] as any)?._t ?? firstT;
  let widthOf = (p: BaseNode<ShaderType>) => TYPE_WIDTH[(p as any)?._t] ?? (MATRIX_DIMENSIONS[(p as any)?._t] ? 16 : 1);
  let widest = params[0];
  for (const p of params) {
    if (widthOf(p) > widthOf(widest)) widest = p;
  }
  valueT = (widest as any)?._t ?? valueT;

  if (UNIFORM_OPERAND_OPS.has(type) && (TYPE_WIDTH[valueT] ?? 1) > 1) {
    params = params.map((p) =>
      (TYPE_WIDTH[(p as any)?._t] ?? 1) === 1
        ? (node({ _t: valueT, type: "construct", params: [p] }) as BaseNode<ShaderType>)
        : p,
    );
  }

  return node({ _t: resultType(type, valueT), type, params });
}

export function op1(type: string, a: any): Node<ShaderType> {
  let wrapped = wrapValue(a, type) as BaseNode<ShaderType>;
  assertOperands(type, [wrapped]);
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
  let first = wrapValue(a, type) as BaseNode<ShaderType>;
  let params = [first, typedOperand(b, (first as any)?._t || "float", type)];
  assertOperands(type, params);
  let widths = params.map((p) => TYPE_WIDTH[(p as any)?._t] ?? 1);
  let width = Math.max(widths[0], widths[1]);

  // Neither language compares a vector against a scalar, so a scalar operand
  // is broadcast to the vector's width first.
  if (width > 1) {
    let wide = (params[widths[0] >= widths[1] ? 0 : 1] as any)._t as ShaderType;
    params = params.map((p, i) =>
      widths[i] === 1 ? (node({ _t: wide, type: "construct", params: [p] }) as BaseNode<ShaderType>) : p,
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
  let outType =
    pattern.length === 1
      ? prefix === "i"
        ? ("int" as const)
        : prefix === "u"
          ? ("uint" as const)
          : prefix === "b"
            ? ("bool" as const)
            : ("float" as const)
      : prefix === "i"
        ? (`ivec${pattern.length}` as const)
        : prefix === "u"
          ? (`uvec${pattern.length}` as const)
          : prefix === "b"
            ? (`bvec${pattern.length}` as const)
            : (`vec${pattern.length}` as const);
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
        `toVar("${name}") must be a valid identifier (letters, digits and ` + `underscore, not starting with a digit).`,
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

/**
 * The variables each `ElseIf` condition in the current top-level `Fn` made,
 * with the `If` whose chain they belong to.
 */
let elseIfVariables: { chain: BaseNode<ShaderType>; names: Set<string> }[] = [];

/**
 * Throws if `roots` use a variable an `ElseIf` condition made anywhere outside
 * that condition's `If` chain. Such a variable belongs to the `ElseIf`, as
 * `let i` in `for (let i = 0; ...)` belongs to its loop.
 */
function assertElseIfVariablesInChain(roots: unknown) {
  for (const { chain, names } of elseIfVariables) {
    const usedOutside = someNode(
      roots,
      (n) => n.type === "var" && names.has((n.value as { varName: string }).varName),
      [chain],
    );
    if (usedOutside) {
      throw new Error(
        "[RMSL] A variable made in an ElseIf condition is used after its If chain: it belongs to that ElseIf. Make it before the If to use it after the chain",
      );
    }
  }
}

export function assertBlockScope(fnName: string, fn: (blockScope: BaseNode<ShaderType>[]) => void) {
  if (blockScope === undefined) {
    throw new Error(`${fnName} must be called inside an Fn(() => { ... }) scope.`);
  }
  fn(blockScope);
}

/**
 * Captures a function body's statements into a `seq` node.
 *
 * Supports a single return (`Fn(() => { ...; return x; })` becomes
 * `() => Node<A>`), a multi return (`return [a, b]` becomes
 * `() => [Node<A>, Node<B>]`), no return at all (`Fn(() => { ... })` becomes
 * `() => Node<"void">`, which a compute program or a vertex stage that assigns
 * `builtinPosition()` itself is), and parameters (`Fn((a: Node<"float">, b:
 * Node<"float">) => a.add(b))` becomes `(a, b) => Node<"float">`).
 *
 * Called inside another Fn, its statements go into the caller's block where
 * it is called, and the call gives back its return alone, so they run there
 * once, whether or where the return is read.
 */
export function Fn<T extends any[], const R>(fn: (...args: T) => R): (...args: T) => FnResult<R> {
  return (...args: T): any => {
    let oldBlockScope = blockScope;
    // A top-level Fn starts a fresh name registry, so each compiled program
    // gets its own deterministic set of user-named variables. Nested Fns keep
    // the outer registry, since their variables share the outer program. The
    // same goes for the variables ElseIf conditions make.
    let oldElseIfVariables = elseIfVariables;
    if (oldBlockScope === undefined) {
      usedVarNames.clear();
      elseIfVariables = [];
    }
    try {
      let scope: BaseNode<ShaderType>[] = [];
      blockScope = scope;
      let r = fn(...args);
      if (oldBlockScope !== undefined) {
        oldBlockScope.push(...scope);
        return Array.isArray(r) ? r.map((item) => wrapValue(item)) : wrapValue(r as any);
      }
      let result: unknown;
      if (Array.isArray(r)) {
        result = r.map((_, i) => {
          let item = wrapValue((r as any[])[i]) as BaseNode<ShaderType>;
          return node({
            _t: item._t || "void",
            type: "seq",
            params: [...scope, item],
          }) as Node<ShaderType>;
        });
      } else {
        let wrappedR = wrapValue(r as any) as BaseNode<ShaderType>;
        result = node({
          _t: wrappedR._t || "void",
          type: "seq",
          params: [...scope, wrappedR],
        });
      }
      assertElseIfVariablesInChain(result);
      return result;
    } finally {
      blockScope = oldBlockScope;
      elseIfVariables = oldElseIfVariables;
    }
  };
}

/**
 * Whether `T` is `any`. Nothing else makes `1 & T` accept `0`: intersecting
 * with `any` gives `any`, while intersecting with any other type gives
 * something `0` cannot be.
 */
type IsAny<T> = 0 extends 1 & T ? true : false;

/**
 * Whether a body's return type is "returns nothing". Tuple-wrapped so a union
 * return is judged whole rather than member by member. `any` is excluded, since
 * it would otherwise match `void` too.
 */
type ReturnsNothing<R> = IsAny<R> extends true ? false : [R] extends [void] ? true : false;

/** The node `Fn` makes of a value its body returns, as `wrapValue` makes it; a node stays as it is. */
type ReturnedNode<V> = IsAny<V> extends true ? V : V extends BaseNode<ShaderType> ? V : Node<ExtractType<V>>;

/**
 * What calling an `Fn` gives back: the node of the body's return, a root for
 * each element of an array it returns, or `Node<"void">` for a body that
 * returns nothing, since the call still produces the `seq` node holding its
 * statements.
 */
export type FnResult<R> =
  ReturnsNothing<R> extends true
    ? Node<"void">
    : IsAny<R> extends true
      ? R
      : R extends readonly unknown[]
        ? { -readonly [K in keyof R]: ReturnedNode<R[K]> }
        : ReturnedNode<R>;

/** Runs `build` in a block of its own: what it returns, and the statements it made. */
function captureStatements<T>(build: () => T): { value: T; statements: BaseNode<ShaderType>[] } {
  let oldBlockScope = blockScope;
  let statements: BaseNode<ShaderType>[] = [];
  blockScope = statements;
  try {
    return { value: build(), statements };
  } finally {
    blockScope = oldBlockScope;
  }
}

/** The component of a vector each swizzle letter names, in the spellings `xyzw`, `rgba` and `stpq`. */
function swizzleComponent(letter: string): number {
  return Math.max("xyzw".indexOf(letter), "rgba".indexOf(letter), "stpq".indexOf(letter));
}

/**
 * The place a write by index through a swizzle, as `m.element(1).yx.element(i)`,
 * writes: a write by index into the vector the swizzle reads, where index `k`
 * of `.yx` is component `"yx"[k]` of it. Every target writes by index into a
 * vector, so each writes the same component. A run-time index outside the
 * swizzle reaches its last component, as a CPU target reaches the last element.
 * The graph keeps the write through the swizzle, which the compilers refuse
 * as they refuse its read, and each compiles this place in its stead.
 */
export function swizzleWritePlace(target: BaseNode<ShaderType>): BaseNode<ShaderType> {
  // A swizzle of a swizzle names components of the vector the last one reads.
  let pattern = target.params![0].value as string;
  let base = target.params![0].params![0]!;
  while (base.type === "swizzle") {
    const inner = base.value as string;
    pattern = [...pattern].map((c) => inner[swizzleComponent(c)]).join("");
    base = base.params![0]!;
  }
  const components = [...pattern].map(swizzleComponent);
  const index = target.params![1]!;
  const component = (c: number) => typedOperand(c, index._t) as BaseNode<ShaderType>;
  let mapped: BaseNode<ShaderType>;
  if (typeof index.value === "number" && Number.isInteger(index.value) && components[index.value] !== undefined) {
    mapped = component(components[index.value]!);
  } else {
    mapped = component(components[components.length - 1]!);
    for (let k = components.length - 2; k >= 0; k--) {
      mapped = (comp("equal", index, component(k)) as any).select(component(components[k]!), mapped);
    }
  }
  return node({ _t: target._t, type: "vectorElement", params: [base, mapped] }) as BaseNode<ShaderType>;
}

export function buildBlock(body: () => void): Node<"void"> {
  const { statements } = captureStatements(body);
  return node({ _t: "void", type: "seq", params: statements }) as Node<"void">;
}

// === Literal constructors (with overloads) ===
export function float(v: number | Node<"int"> | Node<"uint"> | Node<"bool">): Node<"float"> {
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
    if (y !== undefined) params.push(wrapValue(y, "vec2") as BaseNode<ShaderType>);
    return node({ _t: "vec2", type: "construct", params }) as Node<"vec2">;
  }
  if (y === undefined) {
    return node({ _t: "vec2", type: "construct", params: [wrapValue(x, "vec2")] }) as Node<"vec2">;
  }
  if (typeof y === "number") {
    return node({ _t: "vec2", type: "vec2", value: [x, y] }) as Node<"vec2">;
  }
  return node({
    _t: "vec2",
    type: "construct",
    params: [wrapValue(x, "vec2"), y as BaseNode<ShaderType>],
  }) as Node<"vec2">;
}
export function vec3(x?: FloatLike | Node<"vec3"> | Node<"vec4">, y?: FloatLike, z?: FloatLike): Node<"vec3"> {
  if (x === undefined) {
    return node({ _t: "vec3", type: "construct", params: [wrapValue(0)] }) as Node<"vec3">;
  }
  if (isNode(x)) {
    let params = [x as BaseNode<ShaderType>];
    if (y !== undefined) params.push(wrapValue(y, "vec3") as BaseNode<ShaderType>);
    if (z !== undefined) params.push(wrapValue(z, "vec3") as BaseNode<ShaderType>);
    return node({ _t: "vec3", type: "construct", params }) as Node<"vec3">;
  }
  if (y === undefined) {
    return node({ _t: "vec3", type: "construct", params: [wrapValue(x, "vec3")] }) as Node<"vec3">;
  }
  if (typeof y === "number" && (z === undefined || typeof z === "number")) {
    let values: number[] = [x, y];
    if (z !== undefined) values.push(z);
    return node({ _t: "vec3", type: "vec3", value: values }) as Node<"vec3">;
  }
  let params = [wrapValue(x, "vec3") as BaseNode<ShaderType>];
  if (y !== undefined) params.push(wrapValue(y, "vec3") as BaseNode<ShaderType>);
  if (z !== undefined) params.push(wrapValue(z, "vec3") as BaseNode<ShaderType>);
  return node({ _t: "vec3", type: "construct", params }) as Node<"vec3">;
}
export function vec4(
  x?: FloatLike | Node<"vec2"> | Node<"vec3"> | Node<"vec4">,
  y?: FloatLike,
  z?: FloatLike,
  w?: FloatLike,
): Node<"vec4"> {
  if (x === undefined) {
    return node({ _t: "vec4", type: "construct", params: [wrapValue(0)] }) as Node<"vec4">;
  }
  if (isNode(x)) {
    let params = [x as BaseNode<ShaderType>];
    if (y !== undefined) params.push(wrapValue(y, "vec4") as BaseNode<ShaderType>);
    if (z !== undefined) params.push(wrapValue(z, "vec4") as BaseNode<ShaderType>);
    if (w !== undefined) params.push(wrapValue(w, "vec4") as BaseNode<ShaderType>);
    return node({ _t: "vec4", type: "construct", params }) as Node<"vec4">;
  }
  if (y === undefined) {
    return node({ _t: "vec4", type: "construct", params: [wrapValue(x, "vec4")] }) as Node<"vec4">;
  }
  if (
    typeof y === "number" &&
    (z === undefined || typeof z === "number") &&
    (w === undefined || typeof w === "number")
  ) {
    let values: number[] = [x, y];
    if (z !== undefined) values.push(z);
    if (w !== undefined) values.push(w);
    return node({ _t: "vec4", type: "vec4", value: values }) as Node<"vec4">;
  }
  let params = [wrapValue(x, "vec4") as BaseNode<ShaderType>];
  if (y !== undefined) params.push(wrapValue(y, "vec4") as BaseNode<ShaderType>);
  if (z !== undefined) params.push(wrapValue(z, "vec4") as BaseNode<ShaderType>);
  if (w !== undefined) params.push(wrapValue(w, "vec4") as BaseNode<ShaderType>);
  return node({ _t: "vec4", type: "construct", params }) as Node<"vec4">;
}
/** The number as an integer of `type`, truncated toward zero. A number the type cannot hold is refused. */
function integerLiteral(type: "int" | "uint", v: number): number {
  let [low, high] = type === "int" ? [-2147483648, 2147483647] : [0, 4294967295];
  let truncated = Math.trunc(v);
  if (!(truncated >= low && truncated <= high)) {
    throw new Error(`[RMSL] ${v} is outside the range of ${type}, ${low} to ${high}. Use a number inside it.`);
  }
  return truncated;
}

export function int(v: number | Node<"float"> | Node<"uint"> | Node<"bool">): Node<"int"> {
  if (isNode(v)) {
    return node({ _t: "int", type: "construct", params: [v] }) as Node<"int">;
  }
  return node({ _t: "int", type: "int", value: integerLiteral("int", v) }) as Node<"int">;
}
export function uint(v: number | Node<"float"> | Node<"int"> | Node<"bool">): Node<"uint"> {
  if (isNode(v)) {
    return node({ _t: "uint", type: "construct", params: [v] }) as Node<"uint">;
  }
  if (v < 0) {
    throw new Error(`[RMSL] uint(${v}) is negative. An unsigned literal cannot be negative.`);
  }
  return node({ _t: "uint", type: "uint", value: integerLiteral("uint", v) }) as Node<"uint">;
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
  let toComponent = (v: number) => integerLiteral(scalarType, v);
  return (...args: any[]): Node<T> => {
    // Every number, whatever form the call takes: uvec2(-1) is as wrong as uvec2(-1, 0).
    if (scalarType === "uint") {
      for (let a of args) {
        if (typeof a === "number" && a < 0) {
          throw new Error(
            `[RMSL] ${a} is negative, but ${t} components are unsigned. ` +
              `Use a signed vector, or values that are not negative.`,
          );
        }
      }
    }
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
        params: [node({ _t: scalarType, type: scalarType, value: toComponent(args[0]) })],
      }) as Node<T>;
    }
    if (args.length <= width && args.every((a) => typeof a === "number")) {
      return node({ _t: t, type: t, value: args.map(toComponent) }) as Node<T>;
    }
    return node({
      _t: t,
      type: "construct",
      params: args.map((a: any) =>
        isNode(a) ? (a as BaseNode<ShaderType>) : (wrapValue(a, t) as BaseNode<ShaderType>),
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
/**
 * A matrix given its values, one per component in column order. All numbers make a literal. With a scalar node
 * among them the values become the column vectors the matrix is built from, which every target constructs.
 */
function matrixOfValues(t: ShaderType, values: any[]): any {
  if (!values.some(isNode)) return node({ _t: t, type: t, value: values });
  let [columns, rows] = MATRIX_DIMENSIONS[t];
  if (values.length !== columns * rows) {
    throw new Error(`[RMSL] ${t}() takes ${columns * rows} values, or ${columns} columns, not ${values.length}.`);
  }
  let vector = rows === 2 ? vec2 : rows === 3 ? vec3 : vec4;
  let params = Array.from({ length: columns }, (_, c) => (vector as any)(...values.slice(c * rows, (c + 1) * rows)));
  return node({ _t: t, type: "construct", params });
}

/** `vector` as a float vector of its width: an integer or boolean vector converted, as GLSL converts it in a constructor. */
function asFloatVector(vector: BaseNode<ShaderType>): BaseNode<ShaderType> {
  const integerOrBoolean = /^[iub]vec([234])$/.exec(vector._t);
  if (integerOrBoolean === null) return vector;
  return node({ _t: `vec${integerOrBoolean[1]}` as ShaderType, type: "construct", params: [vector] });
}

/**
 * The columns of a matrix of type `t` built from one vector node: its
 * components in column order, converted to float, which only a vector with
 * one component for each of the matrix's has, as `mat2(vec4)`.
 */
function columnsOfVector(t: ShaderType, vector: BaseNode<ShaderType>): BaseNode<ShaderType>[] {
  const [columns, rows] = MATRIX_DIMENSIONS[t];
  if (TYPE_WIDTH[vector._t] !== columns * rows) {
    throw new Error(
      `[RMSL] ${t}() takes ${columns} columns of ${rows} components, a matrix, a scalar or one vector of ` +
        `${columns * rows} components, not ${withArticle(vector._t)}.`,
    );
  }
  const floats = asFloatVector(vector) as any;
  return Array.from(
    { length: columns },
    (_, c) => floats["xyzw".slice(c * rows, (c + 1) * rows)] as BaseNode<ShaderType>,
  );
}

/**
 * Build a matrix constructor: from a matrix, a scalar, a vector holding every
 * component, its columns as vector nodes, or its values one per component.
 * A column must be a vector with one component for each row; an integer or
 * boolean one converts to float. Given nothing,
 * it builds the identity, column by column.
 */
export function makeMatConstructor<T extends ShaderType>(
  t: T,
  size: number,
  columns: number,
): (...args: any[]) => Node<T> {
  const rows = size / columns;
  const identity = Array.from({ length: size }, (_, i) => (Math.floor(i / rows) === i % rows ? 1 : 0));
  return (...args: any[]): Node<T> => {
    if (args.length === 1 && isNode(args[0])) {
      const arg = args[0] as BaseNode<ShaderType>;
      const params = /^vec[234]$|^[iub]vec[234]$/.test(arg._t) ? columnsOfVector(t, arg) : [arg];
      return node({ _t: t, type: "construct", params }) as Node<T>;
    }
    if (args.length === columns && args.every((a: any) => isNode(a))) {
      for (const column of args as BaseNode<ShaderType>[]) {
        if (!/^[iub]?vec[234]$/.test(column._t) || TYPE_WIDTH[column._t] !== rows) {
          throw new Error(
            `[RMSL] ${t}() takes ${columns} columns, each a vector of ${rows} components, not ${withArticle(column._t)}.`,
          );
        }
      }
      return node({ _t: t, type: "construct", params: args.map((a: any) => asFloatVector(a)) }) as Node<T>;
    }
    if (args.length === 1 && typeof args[0] === "number") {
      return node({ _t: t, type: "construct", params: [wrapValue(args[0])] }) as Node<T>;
    }
    if (args.length === 0) {
      return node({ _t: t, type: t, value: identity }) as Node<T>;
    }
    return matrixOfValues(t, args) as Node<T>;
  };
}
export const mat2 = makeMatConstructor("mat2", 4, 2);
export const mat2x3 = makeMatConstructor("mat2x3", 6, 2);
export const mat2x4 = makeMatConstructor("mat2x4", 8, 2);
export const mat3x2 = makeMatConstructor("mat3x2", 6, 3);
export const mat3 = makeMatConstructor("mat3", 9, 3);
export const mat3x4 = makeMatConstructor("mat3x4", 12, 3);
export const mat4x2 = makeMatConstructor("mat4x2", 8, 4);
export const mat4x3 = makeMatConstructor("mat4x3", 12, 4);
export const mat4 = makeMatConstructor("mat4", 16, 4);

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
        isNode(a) ? (a as BaseNode<ShaderType>) : (wrapValue(a, t) as BaseNode<ShaderType>),
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
export type MathLike = number | boolean | VectorArray | Node<ShaderType>;

/**
 * Wrap a raw value as a node for method delegation. The free functions then
 * call the matching method on it. Their results are typed `any` — a node whose
 * type is not known until its operands are inspected cannot be narrowed to a
 * single `Node<A>`, and the union `Node<ShaderType>` has no methods — so the
 * operations that reduce (dot, length, all, ...) declare the narrower type and
 * the rest leave the node untyped, the way TSL's own free functions do.
 */
export function toNode(v: MathLike | BaseNode<ShaderType>, operation?: string): any {
  return wrapValue(v, operation);
}

/**
 * `toNode` for the first operand of a free function. A number beside an
 * integer node takes that node's type, as a number after it does.
 */
function toNodeBeside(a: MathLike | BaseNode<ShaderType>, b: unknown, operation?: string): any {
  let beside = (b as { _t?: string } | null | undefined)?._t;
  return typeof a === "number" && beside !== undefined ? typedOperand(a, beside, operation) : wrapValue(a, operation);
}

export function add(a: MathLike, b: MathLike, ...rest: MathLike[]): any {
  let r = toNodeBeside(a, b, "add").add(b);
  for (const x of rest) r = r.add(x);
  return r;
}
export function sub(a: MathLike, b: MathLike, ...rest: MathLike[]): any {
  let r = toNodeBeside(a, b, "sub").sub(b);
  for (const x of rest) r = r.sub(x);
  return r;
}
export function mul(a: MathLike, b: MathLike, ...rest: MathLike[]): any {
  let r = toNodeBeside(a, b, "mul").mul(b);
  for (const x of rest) r = r.mul(x);
  return r;
}
export function div(a: MathLike, b: MathLike, ...rest: MathLike[]): any {
  let r = toNodeBeside(a, b, "div").div(b);
  for (const x of rest) r = r.div(x);
  return r;
}
export function mod(a: MathLike, b: MathLike): any {
  return toNodeBeside(a, b, "mod").mod(b);
}

export function equal(a: MathLike, b: MathLike): any {
  return toNodeBeside(a, b, "equal").equal(b);
}
export function notEqual(a: MathLike, b: MathLike): any {
  return toNodeBeside(a, b, "notEqual").notEqual(b);
}
export function lessThan(a: MathLike, b: MathLike): any {
  return toNodeBeside(a, b, "lessThan").lessThan(b);
}
export function greaterThan(a: MathLike, b: MathLike): any {
  return toNodeBeside(a, b, "greaterThan").greaterThan(b);
}
export function lessThanEqual(a: MathLike, b: MathLike): any {
  return toNodeBeside(a, b, "lessThanEqual").lessThanEqual(b);
}
export function greaterThanEqual(a: MathLike, b: MathLike): any {
  return toNodeBeside(a, b, "greaterThanEqual").greaterThanEqual(b);
}

export function and(a: MathLike, b: MathLike, ...rest: MathLike[]): any {
  let r = toNodeBeside(a, b, "and").and(b);
  for (const x of rest) r = r.and(x);
  return r;
}
export function or(a: MathLike, b: MathLike, ...rest: MathLike[]): any {
  let r = toNodeBeside(a, b, "or").or(b);
  for (const x of rest) r = r.or(x);
  return r;
}
export function xor(a: MathLike, b: MathLike): any {
  return toNodeBeside(a, b, "xor").xor(b);
}
export function not(a: MathLike): any {
  return toNode(a, "not").not();
}

export function bitAnd(a: IntegerValueLike, b: IntegerValueLike): any {
  return toNodeBeside(a, b, "bitAnd").bitAnd(b);
}
export function bitOr(a: IntegerValueLike, b: IntegerValueLike): any {
  return toNodeBeside(a, b, "bitOr").bitOr(b);
}
export function bitXor(a: IntegerValueLike, b: IntegerValueLike): any {
  return toNodeBeside(a, b, "bitXor").bitXor(b);
}
export function bitNot(a: IntegerValueLike): any {
  return toNode(a, "bitNot").bitNot();
}
export function shiftLeft(a: IntegerValueLike, b: IntegerValueLike): any {
  return (typeof a === "number" ? int(a) : toNode(a, "shiftLeft")).shiftLeft(b);
}
export function shiftRight(a: IntegerValueLike, b: IntegerValueLike): any {
  return (typeof a === "number" ? int(a) : toNode(a, "shiftRight")).shiftRight(b);
}

export function abs(a: SignedValueLike): any {
  return toNode(a, "abs").abs();
}
export function sign(a: SignedValueLike): any {
  return toNode(a, "sign").sign();
}
export function floor(a: FloatValueLike): any {
  return toNode(a, "floor").floor();
}
export function ceil(a: FloatValueLike): any {
  return toNode(a, "ceil").ceil();
}
export function fract(a: FloatValueLike): any {
  return toNode(a, "fract").fract();
}
export function round(a: FloatValueLike): any {
  return toNode(a, "round").round();
}
export function trunc(a: FloatValueLike): any {
  return toNode(a, "trunc").trunc();
}
export function radians(a: FloatValueLike): any {
  return toNode(a, "radians").radians();
}
export function degrees(a: FloatValueLike): any {
  return toNode(a, "degrees").degrees();
}
export function sqrt(a: FloatValueLike): any {
  return toNode(a, "sqrt").sqrt();
}
export function inverseSqrt(a: FloatValueLike): any {
  return toNode(a, "inversesqrt").inverseSqrt();
}
/** GLSL-style alias for `inverseSqrt`, which TSL also exports. */
export function inversesqrt(a: FloatValueLike): any {
  return toNode(a, "inversesqrt").inverseSqrt();
}
export function exp(a: FloatValueLike): any {
  return toNode(a, "exp").exp();
}
export function log(a: FloatValueLike): any {
  return toNode(a, "log").log();
}
export function exp2(a: FloatValueLike): any {
  return toNode(a, "exp2").exp2();
}
export function log2(a: FloatValueLike): any {
  return toNode(a, "log2").log2();
}
export function negate(a: SignedValueLike | Node<MatrixType>): any {
  return toNode(a, "negate").negate();
}
export function oneMinus(a: FloatValueLike): any {
  return toNode(a, "oneMinus").oneMinus();
}
export function reciprocal(a: FloatValueLike): any {
  return toNode(a, "reciprocal").reciprocal();
}
export function cbrt(a: FloatValueLike): any {
  return toNode(a, "cbrt").cbrt();
}
export function saturate(a: MathLike): any {
  return toNode(a, "saturate").saturate();
}
export function lengthSq(a: FloatValueLike): any {
  return toNode(a, "lengthSq").lengthSq();
}
export function normalize(a: FloatValueLike): any {
  return toNode(a, "normalize").normalize();
}
export function dFdx(a: FloatValueLike): any {
  return toNode(a, "dFdx").dFdx();
}
export function dFdy(a: FloatValueLike): any {
  return toNode(a, "dFdy").dFdy();
}
export function fwidth(a: FloatValueLike): any {
  return toNode(a, "fwidth").fwidth();
}

export function sin(a: FloatValueLike): any {
  return toNode(a, "sin").sin();
}
export function cos(a: FloatValueLike): any {
  return toNode(a, "cos").cos();
}
export function tan(a: FloatValueLike): any {
  return toNode(a, "tan").tan();
}
export function asin(a: FloatValueLike): any {
  return toNode(a, "asin").asin();
}
export function acos(a: FloatValueLike): any {
  return toNode(a, "acos").acos();
}
export function sinh(a: FloatValueLike): any {
  return toNode(a, "sinh").sinh();
}
export function cosh(a: FloatValueLike): any {
  return toNode(a, "cosh").cosh();
}
export function tanh(a: FloatValueLike): any {
  return toNode(a, "tanh").tanh();
}
export function asinh(a: FloatValueLike): any {
  return toNode(a, "asinh").asinh();
}
export function acosh(a: FloatValueLike): any {
  return toNode(a, "acosh").acosh();
}
export function atanh(a: FloatValueLike): any {
  return toNode(a, "atanh").atanh();
}

/** `atan(y)` is the single-argument arctangent; `atan(y, x)` is `atan2`. */
export function atan(y: FloatValueLike, x?: FloatValueLike): any {
  return x === undefined ? toNode(y, "atan").atan() : toNode(y, "atan").atan(x);
}

export function pow(x: FloatValueLike, e: FloatValueLike): any {
  return toNode(x, "pow").pow(e);
}
export function pow2(x: MathLike): any {
  return toNode(x, "pow2").pow2();
}
export function pow3(x: MathLike): any {
  return toNode(x, "pow3").pow3();
}
export function pow4(x: MathLike): any {
  return toNode(x, "pow4").pow4();
}
export function min(a: MathLike, b: MathLike, ...rest: MathLike[]): any {
  let r = toNodeBeside(a, b, "min").min(b);
  for (const x of rest) r = r.min(x);
  return r;
}
export function max(a: MathLike, b: MathLike, ...rest: MathLike[]): any {
  let r = toNodeBeside(a, b, "max").max(b);
  for (const x of rest) r = r.max(x);
  return r;
}
export function step(edge: FloatValueLike, x: FloatValueLike): any {
  return toNode(x, "step").step(edge);
}
export function reflect(incident: FloatValueLike, normal: FloatValueLike): any {
  return toNode(incident, "reflect").reflect(normal);
}
export function refract(incident: FloatValueLike, normal: FloatValueLike, eta: FloatValueLike): any {
  return toNode(incident, "refract").refract(normal, eta);
}
export function faceForward(n: FloatValueLike, incident: FloatValueLike, reference: FloatValueLike): any {
  return toNode(n, "faceForward").faceForward(incident, reference);
}
export function difference(a: MathLike, b: MathLike): any {
  return toNodeBeside(a, b, "difference").difference(b);
}
/** A float or a float vector of each width, as `length`, `distance` and `dot` take it. */
export type FloatScalarLike = number | Node<"float">;
export type FloatVec2Like = readonly [number, number] | Node<"vec2">;
export type FloatVec3Like = readonly [number, number, number] | Node<"vec3">;
export type FloatVec4Like = readonly [number, number, number, number] | Node<"vec4">;
export type FloatValueLike = FloatScalarLike | FloatVec2Like | FloatVec3Like | FloatVec4Like;

export function dot(a: FloatScalarLike, b: FloatScalarLike): Node<"float">;
export function dot(a: FloatVec2Like, b: FloatVec2Like): Node<"float">;
export function dot(a: FloatVec3Like, b: FloatVec3Like): Node<"float">;
export function dot(a: FloatVec4Like, b: FloatVec4Like): Node<"float">;
export function dot(a: FloatValueLike, b: FloatValueLike): Node<"float"> {
  return toNodeBeside(a as MathLike, b as MathLike, "dot").dot(b);
}
export function cross(a: FloatVec3Like, b: FloatVec3Like): any {
  return toNodeBeside(a, b, "cross").cross(b);
}
export function distance(a: FloatScalarLike, b: FloatScalarLike): Node<"float">;
export function distance(a: FloatVec2Like, b: FloatVec2Like): Node<"float">;
export function distance(a: FloatVec3Like, b: FloatVec3Like): Node<"float">;
export function distance(a: FloatVec4Like, b: FloatVec4Like): Node<"float">;
export function distance(a: FloatValueLike, b: FloatValueLike): Node<"float"> {
  return toNodeBeside(a as MathLike, b as MathLike, "distance").distance(b);
}
export function length(a: FloatValueLike): Node<"float"> {
  return toNode(a as MathLike, "length").length();
}
export function mix(a: FloatValueLike, b: FloatValueLike, t: FloatValueLike): any {
  return toNodeBeside(a, b, "mix").mix(b, t);
}
export function clamp(x: MathLike, low: MathLike = 0, high: MathLike = 1): any {
  return toNode(x, "clamp").clamp(low, high);
}
export function smoothstep(low: FloatValueLike, high: FloatValueLike, x: FloatValueLike): any {
  return toNode(x, "smoothstep").smoothstep(low, high);
}

export function all(x: MathLike): Node<"bool"> {
  return toNode(x, "all").all();
}
export function any(x: MathLike): Node<"bool"> {
  return toNode(x, "any").any();
}

export function transpose(m: MathLike): any {
  return toNode(m, "transpose").transpose();
}
export function determinant(m: MathLike): Node<"float"> {
  return toNode(m, "determinant").determinant();
}
export function inverse(m: MathLike): any {
  return toNode(m, "inverse").inverse();
}

export function element(a: MathLike, i: IntLike): any {
  return toNode(a, "element").element(i);
}

/** TSL's conditional: `select(cond, a, b)` is `a` when `cond`, else `b`. */
export function select(cond: MathLike, a: MathLike, b: MathLike): any {
  return toNode(cond, "select").select(a, b);
}

/**
 * Rec. 709 luminance of a colour. A vec4 is reduced over its rgb; a vec3 (or
 * anything narrower) is used as it is. The coefficients match the current
 * working colour space's primaries in three.js, which for both sRGB and
 * linear-sRGB is Rec. 709.
 */
export function luminance(
  color: FloatVec3Like | FloatVec4Like,
  luminanceCoefficients: FloatVec3Like = [0.2126, 0.7152, 0.0722],
): Node<"float"> {
  let c = toNode(color, "luminance");
  let rgb = (c as any)?._t === "vec4" ? c.rgb : c;
  return dot(rgb as Node<"vec3">, luminanceCoefficients);
}

/**
 * A deterministic hash of the given uv, in `[0, 1)`, from TSL's `rand`.
 */
export function rand(uv: MathLike): Node<"float"> {
  let dt = dot(toNode(uv, "rand").xy, vec2(12.9898, 78.233));
  return fract(sin(mod(dt, PI)).mul(43758.5453));
}

/**
 * Interleaved gradient noise (Jimenez 2014), a cheap per-pixel dithering hash
 * in `[0, 1)`. Takes a pixel-space position.
 */
export function interleavedGradientNoise(position: MathLike): Node<"float"> {
  return fract(
    float(52.9829189).mul(fract(dot(toNode(position, "interleavedGradientNoise"), vec2(0.06711056, 0.00583715)))),
  );
}

/** Multiply a colour's rgb by its alpha, leaving alpha alone. */
export function premultiplyAlpha(color: MathLike): any {
  let c = toNode(color, "premultiplyAlpha");
  return vec4(c.rgb.mul(c.a), c.a);
}

/** Reverse `premultiplyAlpha`, guarding against a zero alpha. */
export function unpremultiplyAlpha(color: MathLike): any {
  let c = toNode(color, "unpremultiplyAlpha");
  return c.a.equal(0).select(vec4(0), vec4(c.rgb.div(c.a), c.a));
}

/**
 * Throws unless `value` is a sampler that `texture` and `textureLevel` can read:
 * an integer cube map has no texel fetch on GLSL or WGSL.
 */
function assertSampledTexture(fn: string, value: BaseNode<ShaderType>): void {
  const type = (value as { _t?: string } | undefined)?._t;
  if (type === undefined || !isSamplerType(type)) {
    const what = type === undefined ? "not a node" : withArticle(type);
    throw new Error(`[RMSL] ${fn}: its first argument is ${what}, not a sampler`);
  }
  if (type === "isamplerCube" || type === "usamplerCube") {
    throw new Error(
      `[RMSL] ${fn}: ${withArticle(type)} is an integer cube map, which neither GLSL nor WGSL can read a texel of`,
    );
  }
}

/**
 * Samples a texture at a coordinate of its dimension, as TSL's `texture(value,
 * uv)`. A cube map takes a direction. An integer texture cannot be filtered,
 * so it is read unfiltered (`texelFetch` in GLSL, `textureLoad` in WGSL) at
 * integer texel coordinates, and gives an integer vector.
 */
export function texture(value: Sampler2DLike, uv: Vec2Like): Node<"vec4">;
export function texture(value: Sampler3DLike | SamplerCubeLike, uv: Vec3Like): Node<"vec4">;
export function texture(value: ISampler2DLike, uv: IVec2Like): Node<"ivec4">;
export function texture(value: ISampler3DLike, uv: IVec3Like): Node<"ivec4">;
export function texture(value: USampler2DLike, uv: UVec2Like): Node<"uvec4">;
export function texture(value: USampler3DLike, uv: UVec3Like): Node<"uvec4">;
export function texture(value: BaseNode<ShaderType>, uv: unknown): any {
  assertSampledTexture("texture", value);
  return node({
    _t: textureResultType(value._t),
    type: "texture",
    params: [value, wrapValue(uv as any, "texture") as BaseNode<ShaderType>],
  });
}

/**
 * Samples a texture at a coordinate of its dimension and at mip level `level`,
 * as TSL's `textureLevel(value, uv, level)`. An integer texture takes an
 * integer level.
 */
export function textureLevel(value: Sampler2DLike, uv: Vec2Like, level: FloatLike): Node<"vec4">;
export function textureLevel(value: Sampler3DLike | SamplerCubeLike, uv: Vec3Like, level: FloatLike): Node<"vec4">;
export function textureLevel(value: ISampler2DLike, uv: IVec2Like, level: IntLike): Node<"ivec4">;
export function textureLevel(value: ISampler3DLike, uv: IVec3Like, level: IntLike): Node<"ivec4">;
export function textureLevel(value: USampler2DLike, uv: UVec2Like, level: IntLike): Node<"uvec4">;
export function textureLevel(value: USampler3DLike, uv: UVec3Like, level: IntLike): Node<"uvec4">;
export function textureLevel(value: BaseNode<ShaderType>, uv: unknown, level: unknown): any {
  assertSampledTexture("textureLevel", value);
  return node({
    _t: textureResultType(value._t),
    type: "textureLod",
    params: [
      value,
      wrapValue(uv as any, "textureLevel") as BaseNode<ShaderType>,
      wrapValue(level as any, "textureLevel") as BaseNode<ShaderType>,
    ],
  });
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
    params: [sampler as BaseNode<ShaderType>, wrapValue(coords, "textureLoad") as BaseNode<ShaderType>],
  });
}

/**
 * Dimensions of a texture, in texels — `uvec2` for a 2D or cube texture,
 * `uvec3` for a 3D one.
 */
export function textureSize(samplerNode: Sampler2DLike): Node<"uvec2">;
export function textureSize(samplerNode: Sampler3DLike): Node<"uvec3">;
export function textureSize(samplerNode: Sampler2DLike | Sampler3DLike | ISampler2DLike | USampler3DLike): any {
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

/** Draws an id from the counter `uniform()` and `uniformArray()` draw from, for a node rebuilt by `deserialize()`. */
export function allocUniformId(): number {
  return nextUniformId++;
}
export function allocAttrId(): number {
  return nextAttrId++;
}
export function allocVaryingId(): number {
  return nextVaryingId++;
}

/** Gives a `uniformArray()` node its `element(index)`, for `uniformArray()` and for a node rebuilt by `deserialize()`. */
export function attachUniformArrayElement<T extends ShaderType>(target: any, shaderType: T): void {
  target.element = (index: IntLike | FloatLike) =>
    node({
      _t: shaderType,
      type: "uniformArrayElement",
      params: [target, wrapValue(index, "element") as BaseNode<ShaderType>],
    });
}

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
export function uniformArray<T extends ShaderType>(shaderType: T, length: number): UniformArrayNode<T> {
  if (!Number.isInteger(length) || length < 1) {
    throw new Error(`[RMSL] uniformArray length must be a positive integer, got ${length}`);
  }
  if (isSamplerType(shaderType)) {
    throw new Error(
      `[RMSL] uniformArray cannot hold a texture. WGSL has no array of separate` +
        ` texture bindings without an extension, so there is no spelling both` +
        ` backends share — Three.js does not offer one either. Declare each` +
        ` texture on its own, or use a layered array texture, which both` +
        ` languages do have.`,
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
  attachUniformArrayElement(arrayNode, shaderType);
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

export type StorageAccess = "read" | "write" | "read_write";

/**
 * A storage buffer node. Not a `Node<A>` itself: the buffer as a whole has no
 * value, only its elements do, so it exposes `element()` rather than the
 * operations of its element type.
 */
export interface StorageNode<A extends ShaderType> {
  readonly name: string;
  readonly access: StorageAccess;
  /** The buffer this node reads and writes. Every node over the same attribute shares one buffer. */
  attribute: StorageBufferAttribute;
  element(index: IntLike | UintLike | FloatLike): Var<A>;
  /** Makes this node read-only, as TSL's `toReadOnly()`, and returns it. */
  toReadOnly(): ReadOnlyStorageNode<A>;
}

/** A storage buffer node made read-only with `toReadOnly()`: its elements can't be assigned to. */
export interface ReadOnlyStorageNode<A extends ShaderType> {
  readonly name: string;
  readonly access: StorageAccess;
  /** The buffer this node reads. Every node over the same attribute shares one buffer. */
  attribute: StorageBufferAttribute;
  element(index: IntLike | UintLike | FloatLike): Node<A>;
  toReadOnly(): ReadOnlyStorageNode<A>;
}

/** A typed-array constructor a storage buffer's elements can be stored in. */
export type StorageArrayClass = Float32ArrayConstructor | Int32ArrayConstructor | Uint32ArrayConstructor;

let nextStorageBufferId = 0;

/**
 * The buffer behind a {@link storage} node, as TSL's `StorageBufferAttribute`:
 * `count` elements of `itemSize` components each. Built from a typed array,
 * that array is the buffer's initial contents; built from a count, the buffer
 * holds a zeroed array of that many elements, on the host, for the application
 * to fill, as TSL's does.
 */
export class StorageBufferAttribute {
  readonly id = nextStorageBufferId++;
  readonly count: number;
  readonly itemSize: number;
  readonly arrayClass: StorageArrayClass;
  readonly array: Float32Array | Int32Array | Uint32Array;
  /**
   * The type of the attribute's elements, set by the first {@link storage}
   * node over it, or `null` before one. A GPU buffer's layout depends on it:
   * WGSL gives a `vec3`, and each column of three in a matrix, 16 bytes.
   */
  elementType: ShaderType | null = null;

  constructor(
    countOrArray: number | Float32Array | Int32Array | Uint32Array,
    itemSize = 1,
    arrayClass: StorageArrayClass = Float32Array,
  ) {
    this.itemSize = itemSize;
    if (typeof countOrArray === "number") {
      this.count = countOrArray;
      this.arrayClass = arrayClass;
      this.array = new arrayClass(countOrArray * itemSize);
    } else {
      if (countOrArray.length % itemSize !== 0) {
        throw new Error(
          `[RMSL] a typed array of ${countOrArray.length} values isn't a whole number of elements of itemSize ${itemSize}`,
        );
      }
      this.count = countOrArray.length / itemSize;
      this.arrayClass = countOrArray.constructor as StorageArrayClass;
      this.array = countOrArray;
    }
  }
}

/**
 * As TSL's `StorageInstancedBufferAttribute`. It differs from
 * {@link StorageBufferAttribute} only in how a render pipeline steps through
 * it — once per instance rather than once per vertex.
 */
export class StorageInstancedBufferAttribute extends StorageBufferAttribute {}

/**
 * A storage buffer node over `attribute`, as TSL's `storage()`. Read and
 * written through `.element(i)`; read-write unless {@link StorageNode.toReadOnly}
 * is called.
 */
export function storage<T extends ShaderType>(attribute: StorageBufferAttribute, shaderType: T): StorageNode<T> {
  assertStorageLayout(attribute, shaderType);
  if (attribute.elementType !== null && attribute.elementType !== shaderType) {
    throw new Error(
      `[RMSL] an attribute holds one element type: this one holds ${attribute.elementType}, so a storage node over it can't be ${withArticle(shaderType)}`,
    );
  }
  attribute.elementType = shaderType;
  const slot = `_rmsl_b${attribute.id}`;
  const value = { slot, shaderType, access: "read_write" as StorageAccess, attribute };

  const storageNode = node({ _t: shaderType, type: "storage", value, name: slot });
  const result = storageNode as unknown as StorageNode<T>;

  Object.defineProperty(result, "access", { get: () => value.access, enumerable: true });
  result.attribute = attribute;
  result.element = (index: IntLike | UintLike | FloatLike) =>
    node({
      _t: shaderType,
      type: "storageElement",
      params: [storageNode, wrapValue(index, "element") as BaseNode<ShaderType>],
    }) as Var<T>;
  result.toReadOnly = () => {
    value.access = "read";
    return result;
  };

  return result;
}

/** Components in one element of `shaderType`: 1 for a scalar, the width of a vector, columns × rows of a matrix. */
function componentsOf(shaderType: ShaderType): number {
  const shape = MATRIX_DIMENSIONS[shaderType];
  return shape ? shape[0] * shape[1] : (TYPE_WIDTH[shaderType] ?? 1);
}

/**
 * Throws unless `attribute` holds elements of `shaderType`: its `itemSize`
 * the type's component count, its array class the type's component type.
 * Every backend lays a buffer out from the attribute, so a mismatch would
 * read or write past an element, or reinterpret its bits.
 */
function assertStorageLayout(attribute: StorageBufferAttribute, shaderType: ShaderType): void {
  if (shaderType === "bool" || shaderType.startsWith("bvec")) {
    throw new Error(
      `[RMSL] a storage buffer can't hold ${shaderType} elements, as WGSL's can't; store them as uint (or uvec) instead`,
    );
  }
  const components = componentsOf(shaderType);
  if (attribute.itemSize !== components) {
    throw new Error(
      `[RMSL] ${withArticle(shaderType)} storage node needs ${components} components per element, but its attribute has itemSize ${attribute.itemSize}`,
    );
  }
  const arrayClass = storageArrayClass(shaderType);
  if (attribute.arrayClass !== arrayClass) {
    throw new Error(
      `[RMSL] ${withArticle(shaderType)} storage node needs ${withArticle(arrayClass.name)} attribute, but its attribute holds ${withArticle(attribute.arrayClass.name)}`,
    );
  }
}

function storageArrayClass(shaderType: ShaderType): StorageArrayClass {
  if (shaderType === "int" || shaderType.startsWith("ivec")) return Int32Array;
  if (shaderType === "uint" || shaderType.startsWith("uvec")) return Uint32Array;
  return Float32Array;
}

function storageArrayNode<T extends ShaderType>(
  Attribute: typeof StorageBufferAttribute,
  countOrArray: number | Float32Array | Int32Array | Uint32Array,
  shaderType: T,
): StorageNode<T> {
  const attribute = new Attribute(countOrArray, componentsOf(shaderType), storageArrayClass(shaderType));
  return storage(attribute, shaderType);
}

/**
 * A storage node over a new {@link StorageBufferAttribute} of `countOrArray`
 * elements of `shaderType`, as TSL's `attributeArray()`.
 */
export function attributeArray<T extends ShaderType = "float">(
  countOrArray: number | Float32Array | Int32Array | Uint32Array,
  shaderType: T = "float" as T,
): StorageNode<T> {
  return storageArrayNode(StorageBufferAttribute, countOrArray, shaderType);
}

/**
 * A storage node over a new {@link StorageInstancedBufferAttribute} of
 * `countOrArray` elements of `shaderType`, as TSL's `instancedArray()`.
 */
export function instancedArray<T extends ShaderType = "float">(
  countOrArray: number | Float32Array | Int32Array | Uint32Array,
  shaderType: T = "float" as T,
): StorageNode<T> {
  return storageArrayNode(StorageInstancedBufferAttribute, countOrArray, shaderType);
}

/**
 * A compute program with its dispatch size, as TSL's `ComputeNode`. Runs
 * `computeNode` once per index in `0..count`; `countNode` is the uniform the
 * compiled program checks each index against, so `count` can change without
 * recompiling.
 */
export class ComputeNode {
  readonly isComputeNode = true;
  readonly countNode: UniformNode<"uint"> = uniform("uint");

  constructor(
    readonly computeNode: Node<ShaderType>,
    public count: number,
    readonly workgroupSize = 64,
  ) {}
}

/** `node` dispatched once per index in `0..count`, as TSL's `compute()`. */
export function compute(node: Node<ShaderType>, count: number, workgroupSize = 64): ComputeNode {
  return new ComputeNode(node, count, workgroupSize);
}

export function invocationIndex(): Node<"uint"> {
  return node({
    _t: "uint",
    type: "invocationIndex",
  }) as Node<"uint">;
}

/** Refuses an empty name, which declares an input that no target can read by name. */
function assertRawName(name: string, fn: string): void {
  if (name === "") throw new Error(`[RMSL] ${fn}: the name is empty`);
}

export function uniformRaw<T extends ShaderType>(name: string, shaderType: T): UniformNode<T> {
  assertRawName(name, "uniformRaw");
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
 * for every shader that references it, named `_rmsl_time` in every process, so
 * the host updates a single uniform, read too by a graph `deserialize()` rebuilt.
 * Created lazily so merely importing rmsl never consumes a uniform slot.
 */
export let _timeUniform: UniformNode<"float"> | undefined;
export function time(): UniformNode<"float"> {
  return (_timeUniform ??= uniformRaw("_rmsl_time", "float"));
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
  assertRawName(name, "attributeRaw");
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
  assertRawName(name, "varyingRaw");
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
declare const outputMembers: unique symbol;

/** What `outputStruct` returns: a fragment stage's result, with the members it writes in the type. */
export type OutputStruct<M extends readonly BaseNode<ShaderType>[] = readonly BaseNode<ShaderType>[]> = Node<"void"> & {
  readonly [outputMembers]: M;
};

/**
 * The result of a fragment stage that writes several values, one for each
 * render target, at the location of its position among the members: TSL's
 * `outputStruct`. A fragment stage returns it.
 */
export function outputStruct<const M extends readonly Node<ShaderType>[]>(...members: M): OutputStruct<M> {
  return node({
    _t: "void",
    type: "outputStruct",
    params: members.map((member) => wrapValue(member as any) as BaseNode<ShaderType>),
  }) as unknown as OutputStruct<M>;
}

export function builtinPosition(): Var<"vec4"> {
  return node({
    _t: "vec4",
    type: "builtinPosition",
  }) as Var<"vec4">;
}

export function builtinFragDepth(): Var<"float"> {
  return node({
    _t: "float",
    type: "builtinFragDepth",
  }) as Var<"float">;
}

/** The index of the vertex being processed, as TSL's `vertexIndex` (a vertex-stage-only builtin). */
export function vertexIndex(): Node<"uint"> {
  return node({ _t: "uint", type: "vertexIndex" }) as Node<"uint">;
}

/** The index of the instance being drawn, as TSL's `instanceIndex` (a vertex-stage-only builtin). */
export function instanceIndex(): Node<"uint"> {
  return node({ _t: "uint", type: "instanceIndex" }) as Node<"uint">;
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

/**
 * The drawing-buffer size in pixels, as TSL's `screenSize`. One `vec2` uniform
 * for every shader that references it, named `_rmsl_screenSize` in every
 * process, so the host binds a single uniform by its `.name`, as it updates
 * `time()`. Created lazily, as `time()` is.
 */
let _screenSizeUniform: UniformNode<"vec2"> | undefined;
export function screenSize(): UniformNode<"vec2"> {
  return (_screenSizeUniform ??= uniformRaw("_rmsl_screenSize", "vec2"));
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

/**
 * The statements building a condition, split into what runs where the
 * condition is tested and the declarations of the variables it made. Each
 * such variable is declared beforehand, at zero, and assigned where the
 * condition is tested; declared beforehand, it stays in scope after the
 * construct, as one made in the enclosing block would. Any other statement
 * runs where the condition is tested, as it is.
 */
function splitCondition(made: readonly BaseNode<ShaderType>[]) {
  const declarations: BaseNode<ShaderType>[] = [];
  const assignments: BaseNode<ShaderType>[] = [];
  for (const statement of made) {
    if (statement.type !== "let") {
      assignments.push(statement);
      continue;
    }
    const [variable, value] = statement.params! as [Node<ShaderType>, BaseNode<ShaderType>];
    declarations.push(
      node({
        _t: "void",
        type: "let",
        params: [variable, (float(0) as any).convert(variable._t)],
      }) as BaseNode<ShaderType>,
    );
    assignments.push(node({ _t: "void", type: "assign", params: [variable, value] }) as BaseNode<ShaderType>);
  }
  return { declarations, assignments };
}

/**
 * Calls `visit` on each node reachable from the roots through `params`,
 * stopping at the first that returns true. Returns whether one did. Nodes in
 * `skip` are neither visited nor walked into.
 */
export function someNode(roots: unknown, visit: (node: any) => boolean | void, skip: Iterable<unknown> = []): boolean {
  const visited = new Set<unknown>(skip);
  const walk = (node: any): boolean => {
    if (!node || typeof node !== "object" || visited.has(node)) return false;
    visited.add(node);
    if (Array.isArray(node)) return node.some(walk);
    return visit(node) === true || (Array.isArray(node.params) && node.params.some(walk));
  };
  return walk(roots);
}

/** `test`, preceded where it is tested by `statements`: a `seq` of them, or `test` alone. */
function withStatements(statements: BaseNode<ShaderType>[], test: BaseNode<ShaderType>): BaseNode<ShaderType> {
  return statements.length === 0
    ? test
    : (node({ _t: test._t, type: "seq", params: [...statements, test] }) as BaseNode<ShaderType>);
}

/**
 * A conditional, as TSL's `If`, chained with `ElseIf` and `Else`.
 *
 * Each link has to follow the one before it directly, in the same block:
 * anything written between them, a variable included, is an error. A
 * variable an `ElseIf` condition makes belongs to that `ElseIf`, as `let i`
 * in `for (let i = 0; ...)` belongs to its loop: it is computed in the else
 * branch, ahead of the test, and using it after the chain is an error.
 *
 * `ElseIf` and `Else` are getters so they can tell the two apart: JavaScript
 * reads `chain.ElseIf` before it evaluates the condition passed to it, so
 * what was written before the read sits between the links, and what was
 * written after it belongs to the condition.
 */
export function If(cond: BooleanLike, body: () => void): ElseIfChain {
  let ifNode = node({
    _t: "void",
    type: "if",
    params: [wrapValue(cond) as BaseNode<ShaderType>, buildBlock(body) as BaseNode<ShaderType>],
  });
  let outer: BaseNode<ShaderType>[] = [];
  assertBlockScope("If", (scope) => {
    scope.push(ifNode);
    outer = scope;
  });
  /** Where the chain ends in the enclosing block: what follows is written after it. */
  const end = outer.length;
  let deepestIf = ifNode;
  const assertFollows = (link: string) => {
    if (blockScope !== outer) {
      throw new Error(`[RMSL] ${link} has to follow its If directly: it was called from inside another block`);
    }
    if (outer.length !== end) {
      throw new Error(
        `[RMSL] ${link} has to follow its If directly: nothing can be written between them, a variable included. Make a value the chain reads before the If`,
      );
    }
  };
  const chain: ElseIfChain = {
    get ElseIf() {
      assertFollows("ElseIf");
      return (nextCond: BooleanLike, nextBody: () => void) => {
        const test = wrapValue(nextCond) as BaseNode<ShaderType>;
        const made = outer.splice(end);
        const names = new Set(
          made.flatMap((m) =>
            m.type === "let" ? [(m.params![0] as { value: { varName: string } }).value.varName] : [],
          ),
        );
        if (names.size > 0) elseIfVariables.push({ chain: ifNode, names });
        let nextIf = node({
          _t: "void",
          type: "if",
          params: [test, buildBlock(nextBody) as BaseNode<ShaderType>],
        });
        deepestIf.params![2] = withStatements(made, nextIf as BaseNode<ShaderType>);
        deepestIf = nextIf;
        return chain;
      };
    },
    get Else() {
      assertFollows("Else");
      return (elseBody: () => void) => {
        deepestIf.params![2] = buildBlock(elseBody) as BaseNode<ShaderType>;
      };
    },
  };
  return chain;
}

/**
 * A loop's condition, built in a block of its own: the variables it made,
 * declared at zero for the enclosing block, ahead of the loop, and the
 * condition, preceded by their assignments and any other statement it made,
 * which every backend runs before every test.
 */
function loopCondition(cond: () => BooleanLike) {
  const { value, statements } = captureStatements(cond);
  const { declarations, assignments } = splitCondition(statements);
  return { declarations, condition: withStatements(assignments, wrapValue(value) as BaseNode<ShaderType>) };
}

/** Why a `For` whose update holds a block is refused. */
export const FOR_UPDATE_BLOCK_MESSAGE =
  "[RMSL] A for-loop's update cannot contain a block. Move the branch into the loop body, or write the loop with While.";

/** Why a `For` whose update leaves the loop or the function is refused. */
export const FOR_UPDATE_JUMP_MESSAGE =
  "[RMSL] A for-loop's update cannot contain a break, continue, discard or return. Move it into the loop body.";

/** The statements that run a body of their own again on every iteration. */
export const LOOP_STATEMENTS: ReadonlySet<string> = new Set(["for", "while"]);

/** The statements that open a block of their own on every target. */
const BLOCK_STATEMENTS = new Set(["if", ...LOOP_STATEMENTS]);

/** The statements that act on the loop around them. */
export const LOOP_JUMP_STATEMENTS: ReadonlySet<string> = new Set(["break", "continue"]);

/** The statements that leave the loop or the function they are in. */
const JUMP_STATEMENTS = new Set([...LOOP_JUMP_STATEMENTS, "discard", "return"]);

/**
 * Refuses `node`, a node of the update of a `for`, when it is a block or a
 * statement that leaves the loop or the function, neither of which the
 * update slot of a `for` takes.
 */
export function assertForUpdateNode(node: { type?: string }): void {
  if (BLOCK_STATEMENTS.has(node.type!)) throw new Error(FOR_UPDATE_BLOCK_MESSAGE);
  if (JUMP_STATEMENTS.has(node.type!)) throw new Error(FOR_UPDATE_JUMP_MESSAGE);
}

/**
 * A loop of rmsl's own, beside TSL's `Loop`: `init` makes the loop variable, `cond`
 * tests it before every iteration, `update` steps it after every one. A
 * variable `cond` makes is computed before every test, and stays in scope
 * after the loop, and in `update`. The update is the update slot of a GLSL,
 * WGSL or JavaScript `for`, which takes no block and no `break`, `continue`,
 * `discard` or `return`, so an update that holds one is refused here, before
 * any target compiles it.
 */
export function For<T extends Node<ShaderType>>(
  init: () => T,
  cond: (v: T) => BooleanLike,
  update: (v: T) => void,
  body: (v: T) => void,
): void {
  assertBlockScope("For", (scope) => {
    const { value: v, statements: initStatements } = captureStatements(init);
    let initNode = node({ _t: "void", type: "seq", params: initStatements }) as Node<"void">;
    // Condition, update, body: the order they were always built in, which names their variables.
    const { declarations, condition } = loopCondition(() => cond(v));
    let updateNode = buildBlock(() => update(v));
    someNode(updateNode, assertForUpdateNode);
    let bodyNode = buildBlock(() => body(v));
    scope.push(...declarations);
    scope.push(
      node({
        _t: "void",
        type: "for",
        params: [initNode, condition, updateNode, bodyNode],
      }),
    );
  });
}

/** The type of a `Loop` range's index. */
export type LoopIndexType = "int" | "uint" | "float";

/**
 * A range of TSL's object shape of `Loop`. With only `end`, it counts up from
 * 0; with only `start`, it counts down from `start - 1` to 0. Its `update` is
 * a step in the direction of its `condition`, or a function that steps the
 * index itself.
 */
export type LoopRange<T extends LoopIndexType = LoopIndexType, N extends string = string> = {
  start?: number | Node<ShaderType>;
  end?: number | Node<ShaderType>;
  type?: T;
  name?: N;
  condition?: "<" | "<=" | ">" | ">=";
  update?: number | Node<ShaderType> | ((indices: Record<string, Node<any>>) => void);
};

/** The index a `Loop` range of type `T` gives its body. */
type LoopIndex<T> = T extends "float" ? Var<"float"> : T extends "uint" ? Var<"uint"> : Var<"int">;

/**
 * TSL's `Loop`, in each of its shapes, with its indices passed as `{ i }`:
 *
 *   Loop(count, ({ i }) => { ... })                      // i from 0 while i < count
 *   Loop(condition, () => { ... })                       // while the bool holds
 *   Loop({ start, end, type, condition, update }, ({ i }) => { ... })
 *   Loop(4, 2, ({ i, j }) => { ... })                    // one nested loop per range
 *
 * A `bool` condition is built once, before the loop, as TSL builds it.
 */
export function Loop(condition: Node<"bool">, body: () => void): void;
export function Loop(count: IntLike | FloatLike, body: (indices: { i: Var<"int"> }) => void): void;
export function Loop<T extends LoopIndexType = "int", N extends string = "i">(
  range: LoopRange<T, N>,
  body: (indices: { [K in N]: LoopIndex<T> }) => void,
): void;
export function Loop(
  ...params: [...ranges: (IntLike | FloatLike | LoopRange)[], body: (indices: Record<string, Var<"int">>) => void]
): void;
export function Loop(...params: any[]): void {
  const body = params.pop() as (indices: Record<string, Node<any>>) => void;
  const ranges = params as (IntLike | FloatLike | Node<"bool"> | LoopRange)[];
  const indices: Record<string, Node<any>> = {};
  const names = ranges.map((range, depth) =>
    isNode(range) && (range as BaseNode<ShaderType>)._t === "bool"
      ? undefined
      : (isLoopRange(range) && range.name) || "ijklmnopqrstuvwxyz"[depth]!,
  );
  const named = names.filter((name) => name !== undefined);
  const repeated = named.find((name, i) => named.indexOf(name) !== i);
  if (repeated !== undefined) {
    throw new Error(
      `[RMSL] Loop: two ranges name their index "${repeated}", so the inner one hides the outer from the body`,
    );
  }
  const nest = (depth: number): void => {
    if (depth === ranges.length) return body(indices);
    const range = ranges[depth];
    if (isNode(range) && (range as BaseNode<ShaderType>)._t === "bool") {
      return While(range as Node<"bool">, () => nest(depth + 1));
    }
    const name = names[depth]!;
    const { start, end, condition, step } = loopBounds(range as IntLike | FloatLike | LoopRange);
    For(
      () => start.toVar(),
      (index) => compare(index, condition, end),
      (index) => {
        // The update is built before the body, so it records the index itself.
        indices[name] = index;
        if (typeof step === "function") step(indices);
        else index.assign(condition.includes("<") ? index.add(step) : index.sub(step));
      },
      (index) => {
        indices[name] = index;
        nest(depth + 1);
      },
    );
  };
  nest(0);
}

/** Whether a `Loop` param is TSL's object of `start`, `end` and the rest, rather than a count. */
function isLoopRange(range: unknown): range is LoopRange {
  return typeof range === "object" && range !== null && !isNode(range) && !Array.isArray(range);
}

/** The number a `Loop` bound stands for, when it is a number or a literal such as `int(10)`. */
function constantOf(value: number | Node<ShaderType> | undefined): number | undefined {
  if (typeof value === "number") return value;
  const literal = value as BaseNode<ShaderType> | undefined;
  return literal && literal.type === literal._t && typeof literal.value === "number" ? literal.value : undefined;
}

/** A comparison of a `Loop` index with its end, by TSL's operator. */
function compare(index: Node<any>, condition: string, end: Node<any>): Node<"bool"> {
  if (condition === "<") return index.lessThan(end);
  if (condition === "<=") return index.lessThanEqual(end);
  if (condition === ">") return index.greaterThan(end);
  return index.greaterThanEqual(end);
}

/** The start, end, condition and step of a `Loop` range, with what TSL fills in where the range gives none. */
function loopBounds(range: IntLike | FloatLike | LoopRange) {
  if (!isLoopRange(range)) {
    if (typeof range === "number") assertWholeBound(range, "int");
    // A count is built as the index's type, as TSL builds it.
    const end = isNode(range) && range._t !== "int" ? int(range as Node<"float">) : range;
    return { start: int(0), end: end as Node<any>, condition: "<", step: int(1) as Node<any> };
  }
  const type = range.type ?? "int";
  if (type !== "int" && type !== "uint" && type !== "float") {
    throw new Error(`[RMSL] Loop: an index is an int, a uint or a float, not ${JSON.stringify(type)}`);
  }
  if (range.start === undefined && range.end === undefined) {
    throw new Error('[RMSL] Loop: a range names neither "start" nor "end", so it has nothing to count from or to');
  }
  // Each bound is built as the index's type, as TSL builds it.
  const of = (value: number | Node<ShaderType>): Node<any> => {
    if (typeof value === "number" && type !== "float") assertWholeBound(value, type);
    if (isNode(value) && value._t === type) return value;
    return type === "float" ? float(value as number) : type === "uint" ? uint(value as number) : int(value as number);
  };
  let start = range.start;
  let end = range.end;
  let condition: string | undefined = range.condition;
  if (start !== undefined && end === undefined) {
    // Only a start counts down to 0, from the value below it.
    start = isNode(start) ? (of(start) as any).sub(1) : start - 1;
    end = 0;
    condition = ">=";
  } else if (start === undefined) {
    start = 0;
    condition = "<";
  }
  const [first, last] = [constantOf(start!), constantOf(end!)];
  condition ??= first !== undefined && last !== undefined && first > last ? ">=" : "<";
  if (!["<", "<=", ">", ">="].includes(condition)) {
    throw new Error(`[RMSL] Loop: a condition is "<", "<=", ">" or ">=", not ${JSON.stringify(condition)}`);
  }
  const update = range.update;
  if (typeof update === "string") {
    throw new Error(
      "[RMSL] Loop: an update is a number, a node or a function, not a string: TSL writes a string into the shader it emits, which the JS and WASM targets have none of",
    );
  }
  const step = typeof update === "function" ? update : of(update ?? 1);
  if (type === "uint" && condition === ">=" && constantOf(end!) === 0) {
    throw new Error(
      '[RMSL] Loop: a uint range that tests ">=" against an end of 0 never ends, since every uint is at least 0. Give the index the type "int"',
    );
  }
  return { start: of(start!), end: of(end ?? 0), condition, step };
}

/** Throws for a number with a fraction given as a bound of an integer `Loop` index, which TSL would round. */
function assertWholeBound(value: number, type: string): void {
  if (!Number.isInteger(value)) {
    throw new Error(
      `[RMSL] Loop: ${value} is not a whole number, but the index is ${withArticle(type)}. Use a whole number, or a float node, which converts to the index's type`,
    );
  }
}

/**
 * A loop of rmsl's own that runs `body` while `cond` holds, beside TSL's
 * `Loop(condition, body)`. A condition
 * given as a node is built before the loop, so a variable it makes is
 * computed once; given as a function, a variable it makes is computed before
 * every test, and stays in scope after the loop.
 */
export function While(cond: BooleanLike | (() => BooleanLike), body: () => void): void {
  assertBlockScope("While", (scope) => {
    const { declarations, condition } =
      typeof cond === "function"
        ? loopCondition(cond)
        : { declarations: [], condition: wrapValue(cond) as BaseNode<ShaderType> };
    let bodyNode = buildBlock(body);
    scope.push(...declarations);
    scope.push(
      node({
        _t: "void",
        type: "while",
        params: [condition, bodyNode],
      }),
    );
  });
}

/** The chain `Switch` returns, which adds its cases as TSL's does. */
export type SwitchChain = {
  /** Runs `body` when the selector equals one of the values before it. */
  Case: <V extends IntLike[]>(...params: [...values: V, body: () => void]) => SwitchChain;
  /** Runs `body` when no case matched. No case can follow it. */
  Default: (body: () => void) => SwitchChain;
};

/**
 * Multi-way branch on an integer selector, written as TSL writes it:
 *
 *   Switch(level)
 *     .Case(0, () => { colour.assign(black); })
 *     .Case(1, 2, () => { colour.assign(grey); })
 *     .Default(() => { colour.assign(white); });
 *
 * Compiles to an if/else-if chain comparing the selector with each case value,
 * the lowering TSL uses, so there is no fall-through and no `Break()` inside a
 * case. The chain stands where its first `Case` or `Default` is added, as
 * TSL's does, and each later one must follow the one before it in that block.
 */
export function Switch(selector: Node<"int"> | Node<"uint">): SwitchChain {
  let scope: BaseNode<ShaderType>[] | undefined;
  assertBlockScope("Switch", (s) => {
    scope = s;
  });
  const block = scope!;
  /** The length of the block once the previous `Case` or `Default` was added. */
  let after: number | undefined;
  const selectorNode = wrapValue(selector) as BaseNode<ShaderType>;
  let tail: BaseNode<ShaderType> | undefined;
  let closed = false;
  /** Checks that a `Case` or `Default` is added where it can still reach the program. */
  const check = (name: string) => {
    if (closed) throw new Error(`[RMSL] ${name}() after Default(): a Switch takes no case after its Default.`);
    if (blockScope !== block) {
      throw new Error(
        `[RMSL] ${name}() must be called from the block that holds its Switch(), as Switch(x).Case(...).Default(...).`,
      );
    }
    if (after !== undefined && block.length !== after) {
      throw new Error(
        `[RMSL] ${name}() after a statement that follows the case before it: the chain would run before that statement.`,
      );
    }
  };
  /** Puts `branch` at the end of the chain, the first one at the end of the block. */
  const append = (branch: BaseNode<ShaderType>) => {
    if (tail === undefined) block.push(branch);
    else tail.params![2] = branch;
    tail = branch;
    after = block.length;
  };
  const chain: SwitchChain = {
    Case: (...params) => {
      check("Case");
      const caseBody = params[params.length - 1] as () => void;
      const values = params.slice(0, -1) as IntLike[];
      if (values.length === 0) {
        throw new Error("[RMSL] Case() needs at least one value: a case with no values can never match.");
      }
      let cond: BaseNode<ShaderType> | undefined;
      for (const value of values) {
        // `typedOperand`, not `wrapValue`: a bare number here is a case value
        // beside an int/uint selector, and `wrapValue` alone would default it
        // to `float`, mismatched against the selector.
        const eq = comp("equal", selectorNode, typedOperand(value, selector._t) as BaseNode<ShaderType>);
        cond = cond === undefined ? eq : (op("or", cond, eq) as BaseNode<ShaderType>);
      }
      append(node({ _t: "void", type: "if", params: [cond!, buildBlock(caseBody)] }));
      return chain;
    },
    Default: (defaultBody) => {
      check("Default");
      // A Default with no case before it always runs: its body stands alone.
      append(buildBlock(defaultBody));
      closed = true;
      return chain;
    },
  };
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
const MATRIX_SHAPES = {
  mat2: [2, 2],
  mat2x3: [2, 3],
  mat2x4: [2, 4],
  mat3x2: [3, 2],
  mat3: [3, 3],
  mat3x4: [3, 4],
  mat4x2: [4, 2],
  mat4x3: [4, 3],
  mat4: [4, 4],
} as const satisfies Record<string, readonly [number, number]>;

/** `[columns, rows]` per matrix type, looked up by any type name. */
export const MATRIX_DIMENSIONS: Readonly<Record<string, readonly [number, number]>> = MATRIX_SHAPES;

/**
 * Whether a uniform is a texture, which cannot go in the uniform address space
 * and keeps a binding of its own. Asked in both places that emit uniforms, so
 * the two cannot disagree about it.
 */
export function isSamplerType(type: string): boolean {
  return /^(i|u)?sampler(2D|3D|Cube)$/.test(type);
}

/** Whether a sampler type reads integers, which no filtering sampler goes with. */
export function isIntegerSamplerType(type: string): boolean {
  return type.startsWith("isampler") || type.startsWith("usampler");
}
