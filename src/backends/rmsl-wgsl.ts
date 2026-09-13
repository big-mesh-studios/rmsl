// ========== WGSL Compiler ==========
import { BaseNode, MATRIX_DIMENSIONS, Node, ShaderType, TYPE_WIDTH, isSamplerType } from "../rmsl-core";
import {
  CompileCtx, CompiledNode, PRECEDENCE, PREC_ATOM, PREC_UNARY, VertexRoot,
  assertPositionIsReadable, assertSquareMatrix, assertStageResult,
  forUpdateStatements, resolveSwizzleTarget, tryFold, withoutSemicolon, wrapExpr,
} from "../rmsl-compiler-shared";
import { AllocRules, planLayout } from "../rmsl-layout";
export let typeToWGSL: Record<string, string> = {
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
 * The vertex input locations a WGSL attribute consumes. A matrix occupies one
 * per column — a `mat4x4<f32>` spans four consecutive `@location`s — so the
 * location counter advances by its column count rather than one.
 */
export function wgslAttributeLocationCount(type: string): number {
  return wgslMatrixColumns(type)?.count ?? 1;
}

/**
 * The columns a matrix attribute arrives as, or null for anything else.
 *
 * WGSL takes no matrix at a `@location`: a vertex input is a scalar or a
 * vector, so a `mat4x4<f32>` instance transform comes in as its four
 * `vec4<f32>` columns at four consecutive locations, and is put back together
 * in the shader. That is also how the buffer is laid out on both backends —
 * one 64-byte record per instance, read as four columns — so nothing about the
 * data changes, only how the shader receives it.
 */
export function wgslMatrixColumns(type: string): { count: number; columnType: string } | null {
  const match = /^mat(\d)x(\d)<(.+)>$/.exec(type);
  if (!match) return null;
  return { count: Number(match[1]), columnType: `vec${match[2]}<${match[3]}>` };
}

/** The name a matrix attribute's `index`th column is declared under. */
export function wgslMatrixColumnSlot(slot: string, index: number): string {
  return `${slot}_${index}`;
}

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
/**
 * The helper that cuts one matrix down to a smaller one, or null when the
 * construction is not that.
 *
 * GLSL writes the normal-matrix idiom as `mat3(modelMatrix)`, dropping the
 * fourth column and row. WGSL has no constructor that takes a matrix at all, so
 * the columns have to be cut down by hand — through a helper rather than
 * inline, so however large the source expression is, it is evaluated once.
 */
export function wgslMatrixNarrowing(target: string, source: string | undefined): string | null {
  const to = MATRIX_DIMENSIONS[target];
  const from = source === undefined ? undefined : MATRIX_DIMENSIONS[source];
  if (to === undefined || from === undefined) return null;
  if (to[0] >= from[0] || to[1] >= from[1]) return null;
  const helper = `_rmsl_${target}_from_${source}`;
  return helper in WGSL_HELPERS ? helper : null;
}

export function wgslMatrixArgs(
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
export const WGSL_UNIFORM_STRUCT = "_RmslUniforms";
export const WGSL_UNIFORM_BINDING = "_rmsl_uniforms";

/** Byte size and alignment of each WGSL type, per the spec's layout rules. */
export const WGSL_LAYOUT: Record<string, { size: number; align: number }> = {
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
export const WGSL_ARRAY_PADDING: Record<
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
export function wgslMemberType(m: WgslUniformMember): string {
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


export function isWgslTexture(type: string): boolean {
  return type === "texture_2d<f32>" || type === "texture_3d<f32>" || type === "texture_cube<f32>"
    || type === "texture_2d<i32>" || type === "texture_3d<i32>" || type === "texture_cube<i32>"
    || type === "texture_2d<u32>" || type === "texture_3d<u32>" || type === "texture_cube<u32>";
}

/**
 * WGSL's placement rules for `planLayout` (src/rmsl-layout.ts): reorder by
 * descending alignment to minimize padding, widen an array element too
 * narrow to align (see `WGSL_ARRAY_PADDING`), round an array element's
 * stride up to 16, and align the whole struct to at least 4. `type` here is
 * always a WGSL type spelling (`"f32"`, `"vec3<f32>"`, ...), matching what
 * every caller of `wgslUniformLayout` already has on hand.
 */
const WGSL_UNIFORM_RULES: AllocRules = {
  sizeAndAlignOf(type) {
    const base = WGSL_LAYOUT[type];
    // Guessing here is the worst thing this function could do. A wrong size is
    // not a shader that fails to build, it is one that reads whatever happens
    // to lie at that address, and the caller has no way to notice.
    if (base === undefined) {
      throw new Error(
        `[RMSL] no uniform layout is known for ${type}. Its size and`
        + ` alignment have to be added to WGSL_LAYOUT before it can be packed`
        + ` into a uniform buffer.`,
      );
    }
    return base;
  },
  reorderByAlignment: true,
  widenNarrowArrayElements: type => WGSL_ARRAY_PADDING[type]?.stored ?? type,
  arrayStrideRoundedTo: 16,
  structAlignMinimum: 4,
};

export function wgslUniformLayout(
  members: { slot: string; type: string; length?: number }[],
): { members: WgslUniformMember[]; size: number } {
  const placed = planLayout(members.map(m => ({ slot: m.slot, type: m.type, length: m.length })), WGSL_UNIFORM_RULES);
  return {
    members: placed.members.map(m => ({
      name: m.slot,
      type: m.type,
      offset: m.offset,
      size: m.size,
      ...(m.length !== undefined ? { length: m.length, stride: m.stride } : {}),
    })),
    size: placed.size,
  };
}

export function wgslType(brand: any): string {
  return typeToWGSL[brand as string] ?? "f32";
}

/**
 * WGSL only allows the `xyzw` and `rgba` swizzle spellings, so the texture-
 * coordinate `stpq` set has to be translated before it is emitted. `s` is the
 * same component as `x` and so on.
 */
export function wgslSwizzle(pattern: string): string {
  let out = "";
  for (const c of pattern) {
    out += c === "s" ? "x" : c === "t" ? "y" : c === "p" ? "z" : c === "q" ? "w" : c;
  }
  return out;
}

/**
 * A varying's inter-stage location. Generated varyings carry their slot id in
 * their name (`_rmsl_v3` is location 3); a `varyingRaw` one has no numeric
 * suffix, so the id stored alongside the slot is used instead. Either way the
 * vertex and fragment both compute the location from the same value, so a
 * fragment reading a subset of the varyings still numbers them the same way
 * the vertex does.
 */
export function varyingLocation(info: { id?: number; slot: string }): number {
  return info.id ?? Number(/^_rmsl_v(\d+)$/.exec(info.slot)?.[1] ?? 0);
}

export function compileWGSLStage(
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

export function compileWGSLNode(
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
    case "bvec2": return { decls: [], body: [], expr: `vec2<bool>(${(node.value as boolean[]).map(v => v ? "true" : "false").join(", ")})` };
    case "bvec3": return { decls: [], body: [], expr: `vec3<bool>(${(node.value as boolean[]).map(v => v ? "true" : "false").join(", ")})` };
    case "bvec4": return { decls: [], body: [], expr: `vec4<bool>(${(node.value as boolean[]).map(v => v ? "true" : "false").join(", ")})` };
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
      // so the components are selected explicitly. All the way down to a
      // scalar: `float(v)` reads `v.x` in GLSL, where WGSL takes only a scalar
      // operand and so needs the component pulled out before converting it.
      let target = TYPE_WIDTH[node._t as string];
      let sourceType = (node.params?.[0] as any)?._t;
      let source = TYPE_WIDTH[sourceType];
      if (
        params.length === 1 && target !== undefined && source !== undefined
        && source > target && target >= 1
        && /^(vec|ivec|uvec|bvec)/.test(sourceType ?? "")
      ) {
        let narrowed = `${params[0].expr}.${"xyzw".slice(0, target)}`;
        return {
          decls: params[0].decls,
          body: params[0].body,
          expr: target === 1 ? `${t}(${narrowed})` : narrowed,
        };
      }

      // The same narrowing one step up: a matrix cut down to a smaller matrix,
      // which GLSL spells as a constructor and WGSL has no spelling for.
      let narrowing = params.length === 1
        ? wgslMatrixNarrowing(node._t as string, sourceType)
        : null;
      if (narrowing) {
        ctx.wgslHelpers.add(narrowing);
        return {
          decls: params[0].decls,
          body: params[0].body,
          expr: `${narrowing}(${params[0].expr})`,
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
      if (ctx.shaderStage !== "vertex") return { decls: [], body: [], expr: v.slot };
      // A matrix attribute is rebuilt from its columns at the top of `main`,
      // under its own slot name, so a reference to it is a plain local read.
      const isMatrix = wgslMatrixColumns(wgslType(v.shaderType)) !== null;
      return { decls: [], body: [], expr: isMatrix ? v.slot : `input.${v.slot}` };
    }

    case "varying": {
      let v = node.value as any;
      if (v && v.id != null && !ctx.varyings.has(v.id)) {
        ctx.varyings.set(v.id, { id: v.id, type: wgslType(v.shaderType), slot: v.slot });
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

    case "fragCoord": {
      if (ctx.shaderStage !== "fragment") {
        throw new Error("fragCoord() can only be used in fragment shaders");
      }
      ctx.fragCoordUsed = true;
      return { decls: [], body: [], expr: "_rmsl_fragCoordInput.xy" };
    }

    case "swizzle": {
      let src = compileWGSLStage(node.params![0], ctx);
      let pattern = wgslSwizzle(node.value as string);
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
    case "mul": return binaryWGSL(node, ctx, "*");
    case "div": return binaryWGSL(node, ctx, "/");
    case "atan2": return binaryWGSL(node, ctx, "atan2", true);
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
    case "select": {
      let cond = compileWGSLStage(node.params![0], ctx);
      let a = compileWGSLStage(node.params![1], ctx);
      let b = compileWGSLStage(node.params![2], ctx);
      // WGSL's select takes the false value first and supports vector selectors
      // natively — `select(b, a, cond)` is `cond ? a : b`.
      let aW = TYPE_WIDTH[(node.params![1] as any)?._t] ?? 1;
      let bW = TYPE_WIDTH[(node.params![2] as any)?._t] ?? 1;
      let w = Math.max(aW, bW);
      let aExpr = a.expr;
      let bExpr = b.expr;
      if (aW === 1 && w > 1) aExpr = `vec${w}<f32>(${aExpr})`;
      if (bW === 1 && w > 1) bExpr = `vec${w}<f32>(${bExpr})`;
      return {
        decls: [...cond.decls, ...a.decls, ...b.decls],
        body: [...cond.body, ...a.body, ...b.body],
        expr: `select(${bExpr}, ${aExpr}, ${cond.expr})`,
        prec: PREC_ATOM,
      };
    }
    // Comparison ops
    case "lessThan": return binaryWGSL(node, ctx, "<");
    case "greaterThan": return binaryWGSL(node, ctx, ">");
    case "lessThanEqual": return binaryWGSL(node, ctx, "<=");
    case "greaterThanEqual": return binaryWGSL(node, ctx, ">=");
    case "equal": return binaryWGSL(node, ctx, "==");
    case "notEqual": return binaryWGSL(node, ctx, "!=");

    case "and": return logicalWGSL(node, ctx, "&&");
    case "or": return logicalWGSL(node, ctx, "||");
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
      let prec = PRECEDENCE.mul;
      let matExpr = wrapExpr(mat.prec, prec, mat.expr);
      let vecExpr = wrapExpr(vec.prec, prec, vec.expr);
      let shape = MATRIX_DIMENSIONS[matType];
      let width = TYPE_WIDTH[vecType] ?? 0;
      if (shape !== undefined && width === shape[0] - 1) {
        // A position vector one component short of the matrix's column width is
        // promoted with an implied homogeneous 1, and the extra result component
        // dropped — `mat4 * vec3` compiles to `(m * vec4<f32>(v, 1.0)).xyz`.
        let expr = `(${matExpr} * vec${shape[0]}<f32>(${vecExpr}, 1.0))`;
        if (width < shape[1]) expr += `.${"xyzw".slice(0, width)}`;
        return {
          decls: [...mat.decls, ...vec.decls],
          body: [...mat.body, ...vec.body],
          expr,
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
    case "inverseSqrt": return unaryWGSL(node, ctx, "inverseSqrt");
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
    // faceForward(n, i, nref) takes three vectors; a binary emitter would drop
    // the reference and hand Dawn a call it refuses to compile.
    case "faceForward": return ternaryWGSL(node, ctx, "faceForward");
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
    case "textureLoad": {
      // Unfiltered texel fetch; like the integer samplers above this needs no
      // sampler binding.
      let samplerNode = node.params![0];
      let samplerCompiled = compileWGSLStage(samplerNode, ctx);
      let coords = compileWGSLStage(node.params![1], ctx);
      let samplerType = (samplerNode as any)?._t || "sampler2D";
      let width = samplerType.endsWith("2D") ? 2 : 3;
      let coordsType = (node.params![1] as any)?._t || `ivec${width}`;
      let coordsExpr = coords.expr;
      if (coordsType !== `ivec${width}` && coordsType !== `uvec${width}`) coordsExpr = `vec${width}<i32>(${coordsExpr})`;
      return {
        decls: [...samplerCompiled.decls, ...coords.decls],
        body: [...samplerCompiled.body, ...coords.body],
        expr: `textureLoad(${samplerCompiled.expr}, ${coordsExpr}, 0)`,
        prec: PREC_ATOM,
      };
    }
    case "textureSize": {
      let sampler = compileWGSLStage(node.params![0], ctx);
      return {
        decls: sampler.decls,
        body: sampler.body,
        expr: `textureDimensions(${sampler.expr})`,
        prec: PREC_ATOM,
      };
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

    case "return": {
      return { decls: [], body: ["return;"], expr: "0.0" };
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
export const WGSL_HELPERS: Record<string, string> = {
  // A matrix cut down to a smaller one, which GLSL writes as `mat3(m)` — the
  // normal-matrix idiom above all. WGSL takes no matrix in a matrix
  // constructor, so each column is truncated and passed on its own.
  _rmsl_mat3_from_mat4: `fn _rmsl_mat3_from_mat4(m: mat4x4<f32>) -> mat3x3<f32> {
  return mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
}`,
  _rmsl_mat2_from_mat4: `fn _rmsl_mat2_from_mat4(m: mat4x4<f32>) -> mat2x2<f32> {
  return mat2x2<f32>(m[0].xy, m[1].xy);
}`,
  _rmsl_mat2_from_mat3: `fn _rmsl_mat2_from_mat3(m: mat3x3<f32>) -> mat2x2<f32> {
  return mat2x2<f32>(m[0].xy, m[1].xy);
}`,
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
export function shiftWGSL(
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

/**
 * A WGSL logical operator.
 *
 * WGSL gives `&&` and `||` the *same* precedence, and refuses to mix them in
 * one expression without explicit parentheses — unlike C/GLSL, where `&&`
 * binds tighter. An `and` nested under an `or` (as xor's expansion produces)
 * therefore must be parenthesised even though its precedence number is higher
 * than its parent's.
 */
export function logicalWGSL(
  node: BaseNode<ShaderType>,
  ctx: CompileCtx,
  op: string,
): CompiledNode {
  let lhs = compileWGSLStage(node.params![0], ctx);
  let rhs = compileWGSLStage(node.params![1], ctx);
  let prec = PRECEDENCE[node.type] ?? 0;
  const child = (c: CompiledNode, raw: BaseNode<ShaderType> | undefined): string => {
    if (raw?.type === "and" || raw?.type === "or") return `(${c.expr})`;
    return wrapExpr(c.prec, prec, c.expr);
  };
  return {
    decls: [...lhs.decls, ...rhs.decls],
    body: [...lhs.body, ...rhs.body],
    expr: `${child(lhs, node.params![0])} ${op} ${child(rhs, node.params![1])}`,
    prec,
  };
}

export function binaryWGSL(
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

export function ternaryWGSL(
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

export function unaryWGSL(
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

export function compileWGSLWithStage(
  root: Node<ShaderType> | readonly Node<ShaderType>[],
  shaderStage: "vertex" | "fragment",
  options?: CompileWGSLOptions,
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
    fragCoordUsed: false,
    jsParams: new Set(),
    jsHelpers: new Set(),
    outTarget: null,
    derivatives: "throw",
    reentrant: false,
    jsNeedsRes: false,
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
  // The struct a stage declares is normally what that stage reads. A program
  // compiled as two stages sharing one uniform buffer cannot work that way: the
  // members each stage happens to read differ, so the same byte offset would
  // mean a different value in each, and in the buffer the host packs. Passing
  // the program's whole uniform set makes all three agree — a member a stage
  // never reads costs it nothing.
  let declared = options?.uniforms
    ? sharedUniformMembers(options.uniforms, plain.map(([, i]) => i))
    : plain.map(([, i]) => ({ slot: i.slot, type: i.type, length: i.length }));
  if (declared.length > 0) {
    let layout = wgslUniformLayout(declared);
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
      // Attributes are emitted in creation order (numeric slot id), not the
      // order the graph first referenced them — the renderer's pipeline layout
      // numbers its vertex buffers from the material's attribute list, which is
      // creation order too, so the two must line up. This order is also the one
      // `@location` values are handed out in, so a mat4 skipping four shifts
      // only what follows it.
      let attrLoc = 0;
      const vertexInputs = [...ctx.attributes.entries()].sort((a, b) => a[0] - b[0]);
      for (const [, info] of vertexInputs) {
        const matrix = wgslMatrixColumns(info.type);
        if (matrix) {
          for (let column = 0; column < matrix.count; column++) {
            lines.push(`  @location(${attrLoc + column}) ${wgslMatrixColumnSlot(info.slot, column)}: ${matrix.columnType},`);
          }
        } else {
          lines.push(`  @location(${attrLoc}) ${info.slot}: ${info.type},`);
        }
        attrLoc += wgslAttributeLocationCount(info.type);
      }
      lines.push("};");
      lines.push("");
    }
    lines.push("struct VertexOutput {");
    lines.push("  @builtin(position) position: vec4<f32>,");
    // A varying's location is its slot id — `_rmsl_v2` lives at location 2 —
    // not its rank in this stage's sorted list. The fragment numbers its inputs
    // the same way, so a stage reading a subset of the vertex's varyings still
    // agrees on where each one is; rank-based numbering only matched when both
    // stages carried the full set. Declared outputs share the struct, so they
    // start one past the highest varying slot.
    let sortedVaryings = [...ctx.varyings.entries()].sort((a, b) => a[1].slot.localeCompare(b[1].slot));
    let outgoingLocation = 0;
    for (let [, info] of sortedVaryings) {
      outgoingLocation = Math.max(outgoingLocation, varyingLocation(info) + 1);
      lines.push(`  @location(${varyingLocation(info)}) ${info.slot}: ${info.type},`);
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
    // Put each matrix attribute back together from the columns it arrived in,
    // before anything reads it.
    for (const [, info] of [...ctx.attributes.entries()].sort((a, b) => a[0] - b[0])) {
      const matrix = wgslMatrixColumns(info.type);
      if (!matrix) continue;
      const columns = Array.from(
        { length: matrix.count },
        (_, column) => `input.${wgslMatrixColumnSlot(info.slot, column)}`,
      );
      lines.push(`  let ${info.slot} = ${info.type}(${columns.join(", ")});`);
    }
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
    let sortedFVaryings = [...ctx.varyings.entries()].sort((a, b) => a[1].slot.localeCompare(b[1].slot));
    for (let [, info] of sortedFVaryings) {
      if (fragParams) fragParams += ", ";
      fragParams += `@location(${varyingLocation(info)}) ${info.slot}: ${info.type}`;
    }
    // fragCoord() reads the fragment's position in the framebuffer, which WGSL
    // passes in as a builtin parameter rather than a global.
    if (ctx.fragCoordUsed) {
      if (fragParams) fragParams += ", ";
      fragParams += "@builtin(position) _rmsl_fragCoordInput: vec4<f32>";
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

/**
 * A uniform of the program being compiled, as the WGSL struct declares it.
 * `slot` is the uniform node's name and `type` its WGSL type; `length` is set
 * for a uniform array.
 */
export type WgslUniformDeclaration = { slot: string; type: string; length?: number };

export type CompileWGSLOptions = {
  /**
   * Every uniform of the program, not only the ones this stage reads.
   *
   * A vertex and a fragment stage compiled from one program share a single
   * uniform buffer at `@group(0) @binding(0)`. Each stage declaring only what
   * it reads gives the two different structs — and a third layout again in
   * whatever the host packs — so a member lands at one offset in one stage and
   * another offset in the other. Pass the whole set to both stages, and to
   * `wgslUniformLayout` when packing the buffer, and all three agree.
   */
  uniforms?: WgslUniformDeclaration[];
};

export const compileWGSL: {
  (root: Node<ShaderType> | readonly Node<ShaderType>[], options?: CompileWGSLOptions): string;
  vertex(root: VertexRoot, options?: CompileWGSLOptions): string;
  fragment(root: Node<ShaderType> | readonly Node<ShaderType>[], options?: CompileWGSLOptions): string;
} = Object.assign(
  (root: Node<ShaderType> | readonly Node<ShaderType>[], options?: CompileWGSLOptions) =>
    compileWGSLWithStage(root, "fragment", options),
  {
    vertex: (root: VertexRoot, options?: CompileWGSLOptions) =>
      compileWGSLWithStage(root as Node<ShaderType>, "vertex", options),
    fragment: (root: Node<ShaderType> | readonly Node<ShaderType>[], options?: CompileWGSLOptions) =>
      compileWGSLWithStage(root, "fragment", options),
  },
);

/**
 * The struct members to declare when a caller has named the program's whole
 * uniform set: that set, checked against what this stage actually reads.
 *
 * A uniform the stage reads but the caller left out would compile to a
 * reference to a member that does not exist, which the driver reports as a
 * syntax error somewhere in generated code. Saying so here names the slot.
 */
export function sharedUniformMembers(
  declared: WgslUniformDeclaration[],
  used: { slot: string; type: string; length?: number }[],
): WgslUniformDeclaration[] {
  let names = new Set(declared.map(u => u.slot));
  for (let uniform of used) {
    if (!names.has(uniform.slot)) {
      throw new Error(
        `[RMSL] the uniform "${uniform.slot}" is read by this stage but missing`
        + ` from the uniforms passed to the compiler. Pass every uniform of the`
        + ` program, so both stages and the host agree on the buffer layout.`,
      );
    }
  }
  return declared;
}

