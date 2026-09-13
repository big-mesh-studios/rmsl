// ========== GLSL Compiler ==========
import { BaseNode, MATRIX_DIMENSIONS, Node, ShaderType, TYPE_WIDTH } from "../rmsl-core";
import {
  CompileCtx,
  CompiledNode,
  PRECEDENCE,
  PREC_ATOM,
  PREC_UNARY,
  VertexRoot,
  assertPositionIsReadable,
  assertSquareMatrix,
  assertStageResult,
  forUpdateStatements,
  tryFold,
  withoutSemicolon,
  wrapExpr,
} from "./shared";
export let typeToGLSL: Record<string, string> = {
  float: "float",
  vec2: "vec2",
  vec3: "vec3",
  vec4: "vec4",
  int: "int",
  uint: "uint",
  bool: "bool",
  ivec2: "ivec2",
  ivec3: "ivec3",
  ivec4: "ivec4",
  uvec2: "uvec2",
  uvec3: "uvec3",
  uvec4: "uvec4",
  bvec2: "bvec2",
  bvec3: "bvec3",
  bvec4: "bvec4",
  mat2: "mat2",
  mat2x3: "mat2x3",
  mat2x4: "mat2x4",
  mat3x2: "mat3x2",
  mat3: "mat3",
  mat3x4: "mat3x4",
  mat4x2: "mat4x2",
  mat4x3: "mat4x3",
  mat4: "mat4",
  sampler2D: "sampler2D",
  sampler3D: "sampler3D",
  samplerCube: "samplerCube",
  isampler2D: "isampler2D",
  isampler3D: "isampler3D",
  isamplerCube: "isamplerCube",
  usampler2D: "usampler2D",
  usampler3D: "usampler3D",
  usamplerCube: "usamplerCube",
  void: "void",
};

export function glslType(brand: any): string {
  return typeToGLSL[brand as string] ?? "float";
}

export function compileGLSLStage(
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

export function compileGLSLNode(
  node: BaseNode<ShaderType> | ShaderType extends never ? never : any,
  ctx: CompileCtx,
): CompiledNode {
  // Constant folding
  let folded = tryFold(node);
  if (folded) node = folded;

  switch (node.type) {
    case "float": {
      let s = String(node.value);
      if (!s.includes(".") && !s.includes("e")) s += ".0";
      return { decls: [], body: [], expr: s };
    }
    case "int":
      return { decls: [], body: [], expr: String(node.value) };
    case "uint":
      return { decls: [], body: [], expr: String(node.value) + "u" };
    case "bool":
      return { decls: [], body: [], expr: node.value ? "true" : "false" };
    case "vec2":
      return { decls: [], body: [], expr: `vec2(${(node.value as number[]).join(", ")})` };
    case "vec3":
      return { decls: [], body: [], expr: `vec3(${(node.value as number[]).join(", ")})` };
    case "vec4":
      return { decls: [], body: [], expr: `vec4(${(node.value as number[]).join(", ")})` };
    case "ivec2":
      return { decls: [], body: [], expr: `ivec2(${(node.value as number[]).join(", ")})` };
    case "ivec3":
      return { decls: [], body: [], expr: `ivec3(${(node.value as number[]).join(", ")})` };
    case "ivec4":
      return { decls: [], body: [], expr: `ivec4(${(node.value as number[]).join(", ")})` };
    case "uvec2":
      return { decls: [], body: [], expr: `uvec2(${(node.value as number[]).map((v) => `${v}u`).join(", ")})` };
    case "uvec3":
      return { decls: [], body: [], expr: `uvec3(${(node.value as number[]).map((v) => `${v}u`).join(", ")})` };
    case "uvec4":
      return { decls: [], body: [], expr: `uvec4(${(node.value as number[]).map((v) => `${v}u`).join(", ")})` };
    case "bvec2":
      return {
        decls: [],
        body: [],
        expr: `bvec2(${(node.value as boolean[]).map((v) => (v ? "true" : "false")).join(", ")})`,
      };
    case "bvec3":
      return {
        decls: [],
        body: [],
        expr: `bvec3(${(node.value as boolean[]).map((v) => (v ? "true" : "false")).join(", ")})`,
      };
    case "bvec4":
      return {
        decls: [],
        body: [],
        expr: `bvec4(${(node.value as boolean[]).map((v) => (v ? "true" : "false")).join(", ")})`,
      };
    case "mat2":
      return { decls: [], body: [], expr: `mat2(${(node.value as number[]).join(", ")})` };
    case "mat2x3":
      return { decls: [], body: [], expr: `mat2x3(${(node.value as number[]).join(", ")})` };
    case "mat2x4":
      return { decls: [], body: [], expr: `mat2x4(${(node.value as number[]).join(", ")})` };
    case "mat3x2":
      return { decls: [], body: [], expr: `mat3x2(${(node.value as number[]).join(", ")})` };
    case "mat3":
      return { decls: [], body: [], expr: `mat3(${(node.value as number[]).join(", ")})` };
    case "mat3x4":
      return { decls: [], body: [], expr: `mat3x4(${(node.value as number[]).join(", ")})` };
    case "mat4x2":
      return { decls: [], body: [], expr: `mat4x2(${(node.value as number[]).join(", ")})` };
    case "mat4x3":
      return { decls: [], body: [], expr: `mat4x3(${(node.value as number[]).join(", ")})` };
    case "mat4":
      return { decls: [], body: [], expr: `mat4(${(node.value as number[]).join(", ")})` };
    case "void":
      return { decls: [], body: [], expr: "0.0" };

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
      let varInfo = node.value as any;
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
      let indexExpr = indexType === "int" || indexType === "uint" ? index.expr : `int(${index.expr})`;
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
        ctx.varyings.set(v.id, { id: v.id, type: glslType(v.shaderType), slot: v.slot });
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

    case "fragCoord": {
      if (ctx.shaderStage !== "fragment") {
        throw new Error("fragCoord() can only be used in fragment shaders");
      }
      return { decls: [], body: [], expr: "gl_FragCoord.xy" };
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
    case "add":
      return binaryGLSL(node, ctx, "+");
    case "sub":
      return binaryGLSL(node, ctx, "-");
    case "mul":
      return binaryGLSL(node, ctx, "*");
    case "div":
      return binaryGLSL(node, ctx, "/");
    case "atan2":
      return binaryGLSL(node, ctx, "atan", true);
    case "mod": {
      // GLSL's % is integer-only; floats need the mod() builtin.
      let operandType = (node.params![0] as any)?._t;
      let isInteger = operandType === "int" || operandType === "uint";
      return isInteger ? binaryGLSL(node, ctx, "%") : binaryGLSL(node, ctx, "mod", true);
    }
    case "pow":
      return binaryGLSL(node, ctx, "pow", true);
    case "min":
      return binaryGLSL(node, ctx, "min", true);
    case "max":
      return binaryGLSL(node, ctx, "max", true);
    case "dot":
      return binaryGLSL(node, ctx, "dot", true);
    case "cross":
      return binaryGLSL(node, ctx, "cross", true);
    case "distance":
      return binaryGLSL(node, ctx, "distance", true);
    case "reflect":
      return binaryGLSL(node, ctx, "reflect", true);
    case "refract":
      return ternaryGLSL(node, ctx, "refract");
    case "mix":
      return ternaryGLSL(node, ctx, "mix");
    case "step":
      return binaryGLSL(node, ctx, "step", true);
    case "smoothstep":
      return ternaryGLSL(node, ctx, "smoothstep");
    case "clamp":
      return ternaryGLSL(node, ctx, "clamp");
    case "select": {
      let cond = compileGLSLStage(node.params![0], ctx);
      let a = compileGLSLStage(node.params![1], ctx);
      let b = compileGLSLStage(node.params![2], ctx);
      let condType = (node.params![0] as any)?._t || "bool";
      // A boolean vector selects component-wise. GLSL's `?:` takes a scalar
      // bool only, so a vector condition is widened to a float vector and mixed
      // — the branches swap because mix(x, y, a) picks y where the selector is
      // nonzero, and the branch here is `cond ? a : b`.
      if (condType !== "bool") {
        let width = TYPE_WIDTH[(node.params![1] as any)?._t] ?? 3;
        let aExpr = a.expr;
        let bExpr = b.expr;
        let condExpr = cond.expr;
        // Mixed scalar/vector branches are promoted to the wider of the two.
        let aW = TYPE_WIDTH[(node.params![1] as any)?._t] ?? 1;
        let bW = TYPE_WIDTH[(node.params![2] as any)?._t] ?? 1;
        let w = Math.max(aW, bW, width);
        if (aW === 1 && w > 1) aExpr = `vec${w}(${aExpr})`;
        if (bW === 1 && w > 1) bExpr = `vec${w}(${bExpr})`;
        let cExpr = condType.startsWith("bvec") || condType.startsWith("vec") ? `vec${w}(${condExpr})` : condExpr;
        return {
          decls: [...cond.decls, ...a.decls, ...b.decls],
          body: [...cond.body, ...a.body, ...b.body],
          expr: `mix(${bExpr}, ${aExpr}, ${cExpr})`,
          prec: PREC_ATOM,
        };
      }
      let prec = PRECEDENCE[node.type] ?? 0;
      let condExpr = wrapExpr(cond.prec, prec, cond.expr);
      let aExpr = wrapExpr(a.prec, prec, a.expr);
      let bExpr = wrapExpr(b.prec, prec, b.expr);
      return {
        decls: [...cond.decls, ...a.decls, ...b.decls],
        body: [...cond.body, ...a.body, ...b.body],
        expr: `${condExpr} ? ${aExpr} : ${bExpr}`,
        // The ternary binds loosest, so a select nested inside any operator
        // must be wrapped by that operator. Advertising PREC_ATOM would leave
        // `a * (c ? x : y)` unparenthesised — `a * c ? x : y` is a different
        // expression.
        prec,
      };
    }
    // Comparison ops
    case "lessThan":
      return comparisonGLSL(node, ctx, "<", "lessThan");
    case "greaterThan":
      return comparisonGLSL(node, ctx, ">", "greaterThan");
    case "lessThanEqual":
      return comparisonGLSL(node, ctx, "<=", "lessThanEqual");
    case "greaterThanEqual":
      return comparisonGLSL(node, ctx, ">=", "greaterThanEqual");
    case "equal":
      return comparisonGLSL(node, ctx, "==", "equal");
    case "notEqual":
      return comparisonGLSL(node, ctx, "!=", "notEqual");

    case "and":
      return binaryGLSL(node, ctx, "&&");
    case "or":
      return binaryGLSL(node, ctx, "||");
    case "bitAnd":
      return binaryGLSL(node, ctx, "&");
    case "bitOr":
      return binaryGLSL(node, ctx, "|");
    case "bitXor":
      return binaryGLSL(node, ctx, "^");
    case "shiftLeft":
      return binaryGLSL(node, ctx, "<<");
    case "shiftRight":
      return binaryGLSL(node, ctx, ">>");

    case "matVecMul": {
      let mat = compileGLSLStage(node.params![0], ctx);
      let vec = compileGLSLStage(node.params![1], ctx);
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
        // dropped — `mat4 * vec3` compiles to `(m * vec4(v, 1.0)).xyz`.
        let expr = `(${matExpr} * vec${shape[0]}(${vecExpr}, 1.0))`;
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

    // Unary math ops
    case "sin":
      return unaryGLSL(node, ctx, "sin");
    case "cos":
      return unaryGLSL(node, ctx, "cos");
    case "tan":
      return unaryGLSL(node, ctx, "tan");
    case "asin":
      return unaryGLSL(node, ctx, "asin");
    case "acos":
      return unaryGLSL(node, ctx, "acos");
    case "atan":
      return unaryGLSL(node, ctx, "atan");
    case "sinh":
      return unaryGLSL(node, ctx, "sinh");
    case "cosh":
      return unaryGLSL(node, ctx, "cosh");
    case "tanh":
      return unaryGLSL(node, ctx, "tanh");
    case "asinh":
      return unaryGLSL(node, ctx, "asinh");
    case "acosh":
      return unaryGLSL(node, ctx, "acosh");
    case "atanh":
      return unaryGLSL(node, ctx, "atanh");
    case "abs":
      return unaryGLSL(node, ctx, "abs");
    case "sign":
      return unaryGLSL(node, ctx, "sign");
    case "floor":
      return unaryGLSL(node, ctx, "floor");
    case "ceil":
      return unaryGLSL(node, ctx, "ceil");
    case "fract":
      return unaryGLSL(node, ctx, "fract");
    case "round":
      return unaryGLSL(node, ctx, "round");
    case "trunc":
      return unaryGLSL(node, ctx, "trunc");
    case "sqrt":
      return unaryGLSL(node, ctx, "sqrt");
    case "inverseSqrt":
      return unaryGLSL(node, ctx, "inversesqrt");
    case "exp":
      return unaryGLSL(node, ctx, "exp");
    case "log":
      return unaryGLSL(node, ctx, "log");
    case "exp2":
      return unaryGLSL(node, ctx, "exp2");
    case "log2":
      return unaryGLSL(node, ctx, "log2");
    case "normalize":
      return unaryGLSL(node, ctx, "normalize");
    case "length":
      return unaryGLSL(node, ctx, "length");
    case "transpose":
      return unaryGLSL(node, ctx, "transpose");
    case "inverse":
      assertSquareMatrix((node.params![0] as any)?._t);
      return unaryGLSL(node, ctx, "inverse");
    case "determinant":
      return unaryGLSL(node, ctx, "determinant");
    case "fwidth":
      return unaryGLSL(node, ctx, "fwidth");
    case "dFdx":
      return unaryGLSL(node, ctx, "dFdx");
    case "dFdy":
      return unaryGLSL(node, ctx, "dFdy");
    // faceforward(n, i, nref) takes three vectors, so a binary emitter would
    // silently drop the reference — the exact bug the validator exists to catch.
    case "faceForward":
      return ternaryGLSL(node, ctx, "faceforward");
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
    case "textureLoad": {
      // Unfiltered texel fetch at integer coordinates — the float-sampler
      // counterpart of the integer texelFetch above.
      let sampler = compileGLSLStage(node.params![0], ctx);
      let coords = compileGLSLStage(node.params![1], ctx);
      let samplerType = (node.params![0] as any)?._t || "sampler2D";
      let width = samplerType.endsWith("2D") ? 2 : 3;
      let idxExpr = coords.expr;
      let idxType = (node.params![1] as any)?._t || "ivec2";
      if (idxType !== `ivec${width}` && idxType !== `uvec${width}`) idxExpr = `ivec${width}(${idxExpr})`;
      return {
        decls: [...sampler.decls, ...coords.decls],
        body: [...sampler.body, ...coords.body],
        expr: `texelFetch(${sampler.expr}, ${idxExpr}, 0)`,
        prec: PREC_ATOM,
      };
    }
    case "textureSize": {
      let sampler = compileGLSLStage(node.params![0], ctx);
      return {
        decls: sampler.decls,
        body: sampler.body,
        expr: `textureSize(${sampler.expr}, 0)`,
        prec: PREC_ATOM,
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
      let elseBody =
        node.params!.length >= 3 && node.params![2] !== undefined
          ? compileGLSLStage(node.params![2], ctx)
          : { decls: [] as string[], body: [] as string[], expr: "" };
      let lines: string[] = [...cond.body, `if (${cond.expr}) {`, ...body.body.map((l) => "  " + l), "}"];
      if (elseBody.body.length > 0) {
        lines.push("else {");
        lines.push(...elseBody.body.map((l) => "  " + l));
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
        if (lastStmt.endsWith(";")) {
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
          ...body.body.map((l) => "  " + l),
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
        body: [...cond.body, `while (${cond.expr}) {`, ...body.body.map((l) => "  " + l), "}"],
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
      throw new Error(`[RMSL] Unsupported node type in GLSL compiler: "${node.type}"`);
  }
}

export function binaryGLSL(node: BaseNode<ShaderType>, ctx: CompileCtx, op: string, isFn?: boolean): CompiledNode {
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

export function comparisonGLSL(node: BaseNode<ShaderType>, ctx: CompileCtx, op: string, fnName: string): CompiledNode {
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

export function ternaryGLSL(
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
    if (cType === "float") {
      /* keep as-is or convert both */
    }
  }
  return {
    decls: [...a.decls, ...b.decls, ...c.decls],
    body: [...a.body, ...b.body, ...c.body],
    expr: `${fn}(${aExpr}, ${bExpr}, ${cExpr})`,
  };
}

export function unaryGLSL(
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

/**
 * Which shader precision a GLSL program is compiled with. Mirrors three.js's
 * `precision` option: `"highp"` for the most accurate math, `"mediump"`/`"lowp"`
 * for the faster, cheaper fragment math that mobile GPUs often need. WGSL has
 * no precision qualifiers, so it applies to GLSL output only.
 */
export type GLSLPrecision = "lowp" | "mediump" | "highp";

/** Options for the GLSL shader compilers. */
export interface CompileGLSLOptions {
  /**
   * The float and sampler precision to declare in the shader header. Defaults
   * to `"highp"`.
   */
  precision?: GLSLPrecision;
}

export function compileGLSLWithStage(
  root: Node<ShaderType> | readonly Node<ShaderType>[],
  shaderStage: "vertex" | "fragment",
  options: CompileGLSLOptions = {},
): string {
  const precision = options.precision ?? "highp";
  if (precision !== "lowp" && precision !== "mediump" && precision !== "highp") {
    throw new Error(`[RMSL] unknown precision "${precision}" — use "lowp", "mediump" or "highp".`);
  }
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
  let results = nodes.map((n) => compileGLSLStage(n, ctx));
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
  let emitImplicitColor = shaderStage === "fragment" && ctx.outputs.size === 0 && hasVec4Result;

  let lines: string[] = [];
  lines.push("#version 300 es");
  lines.push(`precision ${precision} float;`);
  // No default precision covers every sampler. GLSL ES predeclares one for
  // sampler2D and samplerCube in a fragment stage, but not sampler3D, and the
  // vertex language predeclares none; Chromium's WebGL2 compiler rejects a
  // sampler with no precision at all ("No precision specified"). Each sampler
  // type a shader actually uses gets a precision declared for the stages that
  // use it.
  let glslSamplerTypes = [
    ...new Set(
      [...ctx.uniforms.values()]
        .map((info) => info.type)
        .filter((t) => /^(i|u)?sampler2D$|^(i|u)?sampler3D$|^(i|u)?samplerCube$/.test(t)),
    ),
  ].sort();
  for (let samplerType of glslSamplerTypes) {
    lines.push(`precision ${precision} ${samplerType};`);
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
      lines.push(
        shaderStage === "fragment"
          ? `layout(location=${outputLocation++}) out ${info.type} ${info.slot};`
          : `out ${info.type} ${info.slot};`,
      );
    }
  });
  if (emitImplicitColor) {
    lines.push("layout(location=0) out vec4 _rmsl_fragColor;");
  }
  if (ctx.uniforms.size > 0 || ctx.attributes.size > 0 || ctx.outputs.size > 0 || emitImplicitColor) {
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

export const compileGLSL: {
  (root: Node<ShaderType> | readonly Node<ShaderType>[], options?: CompileGLSLOptions): string;
  vertex(root: VertexRoot, options?: CompileGLSLOptions): string;
  fragment(root: Node<ShaderType> | readonly Node<ShaderType>[], options?: CompileGLSLOptions): string;
} = Object.assign(
  (root: Node<ShaderType> | readonly Node<ShaderType>[], options?: CompileGLSLOptions) =>
    compileGLSLWithStage(root, "fragment", options),
  {
    // The value is always a node; void only describes a body that returned
    // nothing, which still compiles to one.
    vertex: (root: VertexRoot, options?: CompileGLSLOptions) =>
      compileGLSLWithStage(root as Node<ShaderType>, "vertex", options),
    fragment: (root: Node<ShaderType> | readonly Node<ShaderType>[], options?: CompileGLSLOptions) =>
      compileGLSLWithStage(root, "fragment", options),
  },
);
