import { BaseNode, MATRIX_DIMENSIONS, Node, ShaderType, TYPE_WIDTH } from "../../core";
import {
  CpuDrawBuffer,
  CpuRoutine,
  CpuShaderContext,
  CpuProgramResult,
  CpuGrid,
  CpuProgram,
  CpuValue,
  GridBuffer,
  ComputeStage,
  FragmentResult,
  FragmentStage,
  VertexResult,
  VertexStage,
  FloatWidth,
  toFragmentResult,
  toVertexResult,
  typedValue,
  typedArrayOfKind,
  componentCountOf,
  componentKindOf,
  elementKindOf,
  isAggregate,
  scalarKindOf,
} from "../cpu";
import {
  CompileCtx,
  CompileFnOptions,
  CompiledNode,
  PRECEDENCE,
  PREC_ATOM,
  PREC_UNARY,
  assertNotInAComputeStage,
  COMPUTE_REFUSES,
  DEGREES_PER_RADIAN,
  RADIANS_PER_DEGREE,
  assertPositionIsReadable,
  assertSquareMatrix,
  assertLiteralIndexInRange,
  assertAssignable,
  parameterNode,
  assertStageResult,
  prepareRoots,
  assertReadsNoStageInput,
  assertOneDeclarationPerName,
  numberClashingVariables,
  forUpdateStatements,
  loopTest,
  resolveSwizzleTarget,
  roundHalfToEven,
  tryFold,
  withoutSemicolon,
  wrapExpr,
} from "../shared";
import { shareNodes } from "../share";

/** Which component each swizzle accessor names, in all three spellings. */
export const JS_COMPONENT_INDEX: Record<string, number> = {
  x: 0,
  y: 1,
  z: 2,
  w: 3,
  r: 0,
  g: 1,
  b: 2,
  a: 3,
  s: 0,
  t: 1,
  p: 2,
  q: 3,
};

/** Length of the JS array a value of this type occupies (0 for a scalar). */
export function jsArrayLength(brand: string | undefined): number {
  if (!brand) return 0;
  let width = TYPE_WIDTH[brand];
  if (width) return width;
  let shape = MATRIX_DIMENSIONS[brand];
  if (shape) return shape[0] * shape[1];
  return 0;
}

/** Whether a value of this type is carried as a JS array rather than a number. */
export function jsIsArrayType(brand: string | undefined): boolean {
  return jsArrayLength(brand) > 1;
}

/** Zero-array initializer for a hoisted scratch slot, "" for a scalar. */
export function jsScratchLiteral(brand: string | undefined): string {
  let n = jsArrayLength(brand);
  return n > 1 ? `[${Array(n).fill(0).join(", ")}]` : "";
}

/** Node types that read an existing array rather than producing one. */
export const JS_ARRAY_LEAF_TYPES = new Set([
  "vec2",
  "vec3",
  "vec4",
  "ivec2",
  "ivec3",
  "ivec4",
  "uvec2",
  "uvec3",
  "uvec4",
  "bvec2",
  "bvec3",
  "bvec4",
  "mat2",
  "mat2x3",
  "mat2x4",
  "mat3x2",
  "mat3",
  "mat3x4",
  "mat4x2",
  "mat4x3",
  "mat4",
  "var",
  "uniform",
  "uniformArray",
  "uniformArrayElement",
  "attribute",
  "varying",
  "output",
  "builtinPosition",
  "storage",
]);

export function isJSArrayLeaf(node: any): boolean {
  return !!node && JS_ARRAY_LEAF_TYPES.has(node.type);
}

/** Element-wise operations the JS vector helpers implement, per index. */
export const JS_ELEM: Record<string, { argc: number; fn: (xs: string[]) => string; helper?: string }> = {
  add: { argc: 2, fn: (xs) => `${xs[0]} + ${xs[1]}` },
  sub: { argc: 2, fn: (xs) => `${xs[0]} - ${xs[1]}` },
  mul: { argc: 2, fn: (xs) => `${xs[0]} * ${xs[1]}` },
  div: { argc: 2, fn: (xs) => `${xs[0]} / ${xs[1]}` },
  // Integer operations wrap to 32 bits and follow WGSL where JS numbers differ:
  // `x / 0` is `x`, `x % 0` is `0`, and `INT_MIN / -1` is `INT_MIN`.
  iadd: { argc: 2, fn: (xs) => `(${xs[0]} + ${xs[1]}) | 0` },
  isub: { argc: 2, fn: (xs) => `(${xs[0]} - ${xs[1]}) | 0` },
  imul: { argc: 2, fn: (xs) => `Math.imul(${xs[0]}, ${xs[1]})` },
  idiv: { argc: 2, fn: (xs) => `_idiv(${xs[0]}, ${xs[1]})`, helper: "idiv" },
  imod: { argc: 2, fn: (xs) => `_imod(${xs[0]}, ${xs[1]})`, helper: "imod" },
  uadd: { argc: 2, fn: (xs) => `(${xs[0]} + ${xs[1]}) >>> 0` },
  usub: { argc: 2, fn: (xs) => `(${xs[0]} - ${xs[1]}) >>> 0` },
  umul: { argc: 2, fn: (xs) => `Math.imul(${xs[0]}, ${xs[1]}) >>> 0` },
  udiv: { argc: 2, fn: (xs) => `_udiv(${xs[0]}, ${xs[1]})`, helper: "udiv" },
  umod: { argc: 2, fn: (xs) => `_umod(${xs[0]}, ${xs[1]})`, helper: "umod" },
  // Bitwise operations: JS already works on 32 bits here and takes shift
  // amounts modulo 32; a uint result only needs reading back unsigned.
  iand: { argc: 2, fn: (xs) => `${xs[0]} & ${xs[1]}` },
  ior: { argc: 2, fn: (xs) => `${xs[0]} | ${xs[1]}` },
  ixor: { argc: 2, fn: (xs) => `${xs[0]} ^ ${xs[1]}` },
  ishl: { argc: 2, fn: (xs) => `${xs[0]} << ${xs[1]}` },
  ishr: { argc: 2, fn: (xs) => `${xs[0]} >> ${xs[1]}` },
  inot: { argc: 1, fn: (xs) => `~${xs[0]}` },
  uand: { argc: 2, fn: (xs) => `(${xs[0]} & ${xs[1]}) >>> 0` },
  uor: { argc: 2, fn: (xs) => `(${xs[0]} | ${xs[1]}) >>> 0` },
  uxor: { argc: 2, fn: (xs) => `(${xs[0]} ^ ${xs[1]}) >>> 0` },
  ushl: { argc: 2, fn: (xs) => `(${xs[0]} << ${xs[1]}) >>> 0` },
  ushr: { argc: 2, fn: (xs) => `${xs[0]} >>> ${xs[1]}` },
  unot: { argc: 1, fn: (xs) => `~${xs[0]} >>> 0` },
  min: { argc: 2, fn: (xs) => `Math.min(${xs[0]}, ${xs[1]})` },
  max: { argc: 2, fn: (xs) => `Math.max(${xs[0]}, ${xs[1]})` },
  pow: { argc: 2, fn: (xs) => `Math.pow(${xs[0]}, ${xs[1]})` },
  atan2: { argc: 2, fn: (xs) => `Math.atan2(${xs[0]}, ${xs[1]})` },
  // Floored, matching GLSL's mod() — JS % truncates toward zero.
  mod: { argc: 2, fn: (xs) => `${xs[0]} - ${xs[1]} * Math.floor(${xs[0]} / ${xs[1]})` },
  // step(edge, x): 0 while x < edge, 1 from there on.
  step: { argc: 2, fn: (xs) => `${xs[1]} < ${xs[0]} ? 0 : 1` },
  clamp: { argc: 3, fn: (xs) => `Math.min(Math.max(${xs[0]}, ${xs[1]}), ${xs[2]})` },
  mix: { argc: 3, fn: (xs) => `${xs[0]} + ${xs[2]} * (${xs[1]} - ${xs[0]})` },
  smoothstep: { argc: 3, fn: (xs) => `_smoothstep(${xs[0]}, ${xs[1]}, ${xs[2]})`, helper: "smoothstep" },
  neg: { argc: 1, fn: (xs) => `-${xs[0]}` },
  ineg: { argc: 1, fn: (xs) => `-${xs[0]} | 0` },
  iabs: { argc: 1, fn: (xs) => `Math.abs(${xs[0]}) | 0` },
  abs: { argc: 1, fn: (xs) => `Math.abs(${xs[0]})` },
  sign: { argc: 1, fn: (xs) => `Math.sign(${xs[0]})` },
  floor: { argc: 1, fn: (xs) => `Math.floor(${xs[0]})` },
  ceil: { argc: 1, fn: (xs) => `Math.ceil(${xs[0]})` },
  round: { argc: 1, fn: (xs) => `_rmsl_roundEven(${xs[0]})`, helper: "roundEven" },
  trunc: { argc: 1, fn: (xs) => `Math.trunc(${xs[0]})` },
  radians: { argc: 1, fn: (xs) => `(${xs[0]} * ${RADIANS_PER_DEGREE})` },
  degrees: { argc: 1, fn: (xs) => `(${xs[0]} * ${DEGREES_PER_RADIAN})` },
  fract: { argc: 1, fn: (xs) => `${xs[0]} - Math.floor(${xs[0]})` },
  sqrt: { argc: 1, fn: (xs) => `Math.sqrt(${xs[0]})` },
  rsqrt: { argc: 1, fn: (xs) => `1 / Math.sqrt(${xs[0]})` },
  exp: { argc: 1, fn: (xs) => `Math.exp(${xs[0]})` },
  log: { argc: 1, fn: (xs) => `Math.log(${xs[0]})` },
  exp2: { argc: 1, fn: (xs) => `Math.pow(2, ${xs[0]})` },
  log2: { argc: 1, fn: (xs) => `Math.log2(${xs[0]})` },
  sin: { argc: 1, fn: (xs) => `Math.sin(${xs[0]})` },
  cos: { argc: 1, fn: (xs) => `Math.cos(${xs[0]})` },
  tan: { argc: 1, fn: (xs) => `Math.tan(${xs[0]})` },
  asin: { argc: 1, fn: (xs) => `Math.asin(${xs[0]})` },
  acos: { argc: 1, fn: (xs) => `Math.acos(${xs[0]})` },
  atan: { argc: 1, fn: (xs) => `Math.atan(${xs[0]})` },
  sinh: { argc: 1, fn: (xs) => `Math.sinh(${xs[0]})` },
  cosh: { argc: 1, fn: (xs) => `Math.cosh(${xs[0]})` },
  tanh: { argc: 1, fn: (xs) => `Math.tanh(${xs[0]})` },
  asinh: { argc: 1, fn: (xs) => `Math.asinh(${xs[0]})` },
  acosh: { argc: 1, fn: (xs) => `Math.acosh(${xs[0]})` },
  atanh: { argc: 1, fn: (xs) => `Math.atanh(${xs[0]})` },
};

/**
 * The precedence of the outermost operator in each scalar `JS_ELEM` form that
 * is written as a formula, so a parent operator brackets it when it must:
 * `(a + b) | 0` is an `|`, not an addition, and `a < (a + b) | 0` would
 * compare before the `|`. `fract` is a subtraction and `rsqrt` a division.
 * A form missing here is a call, which never needs brackets.
 */
const JS_FORM_PREC: Record<string, number> = {
  fract: PRECEDENCE.sub!,
  rsqrt: PRECEDENCE.div!,
  iadd: PRECEDENCE.bitOr!,
  isub: PRECEDENCE.bitOr!,
  uadd: PRECEDENCE.shiftRight!,
  usub: PRECEDENCE.shiftRight!,
  umul: PRECEDENCE.shiftRight!,
  iand: PRECEDENCE.bitAnd!,
  ior: PRECEDENCE.bitOr!,
  ixor: PRECEDENCE.bitXor!,
  ishl: PRECEDENCE.shiftLeft!,
  ishr: PRECEDENCE.shiftRight!,
  inot: PREC_UNARY,
  uand: PRECEDENCE.shiftRight!,
  uor: PRECEDENCE.shiftRight!,
  uxor: PRECEDENCE.shiftRight!,
  ushl: PRECEDENCE.shiftRight!,
  ushr: PRECEDENCE.shiftRight!,
  unot: PRECEDENCE.shiftRight!,
  ineg: PRECEDENCE.bitOr!,
  iabs: PRECEDENCE.bitOr!,
};

/** A scalar integer `JS_ELEM` form applied to compiled operands, bracketed as its operator needs. */
function jsIntegerForm(op: string, operands: CompiledNode[]): { expr: string; prec?: number } {
  return { expr: JS_ELEM[op]!.fn(operands.map(jsOperand)), prec: JS_FORM_PREC[op] };
}

export function jsZeroes(width: number): string {
  return Array(width).fill(0).join(", ");
}

/**
 * Source for one JS helper function, named for what it computes.
 *
 * Every array-producing helper takes a trailing `out` array it writes into —
 * the slot the caller allocated — and returns it. Called without `out`, it
 * allocates one itself, which is the path expressions take. So an assignment
 * compiled with a target slot allocates nothing.
 */
export function jsHelperSource(name: string): string {
  let m = /^v(\d+)([a-zA-Z]+)(?:_([vs]+))?$/.exec(name);
  if (m) {
    let width = Number(m[1]);
    let op = m[2]!;
    let e = JS_ELEM[op];
    if (e) {
      let args = "abcdef".slice(0, e.argc).split("");
      // One letter per operand, `v` for a vector and `s` for a scalar; every operand a vector without one.
      let shape = m[3] ?? "v".repeat(e.argc);
      let lines: string[] = [];
      for (let i = 0; i < width; i++) {
        let xs = args.map((a, k) => (shape[k] === "s" ? a : `${a}[${i}]`));
        lines.push(`  out[${i}] = ${e.fn(xs)};`);
      }
      return (
        `function _${name}(${args.join(", ")}, out) {\n` +
        `  out = out || [${jsZeroes(width)}];\n${lines.join("\n")}\n  return out;\n}`
      );
    }
    if (op === "norm") {
      // Read the length before writing out, so out may alias the input.
      return (
        `function _${name}(a, out) {\n` +
        `  out = out || new Array(${width});\n` +
        `  let l = 0;\n` +
        `  for (let i = 0; i < ${width}; i++) l += a[i] * a[i];\n` +
        `  l = Math.sqrt(l);\n` +
        `  if (l > 0) { for (let i = 0; i < ${width}; i++) out[i] = a[i] / l; }\n` +
        `  else { for (let i = 0; i < ${width}; i++) out[i] = a[i]; }\n` +
        `  return out;\n}`
      );
    }
    if (op === "reflect") {
      // reflect(i, n) = i - 2 * dot(n, i) * n
      return (
        `function _${name}(i, n, out) {\n` +
        `  out = out || new Array(${width});\n` +
        `  let d = 0;\n` +
        `  for (let j = 0; j < ${width}; j++) d += n[j] * i[j];\n` +
        `  for (let j = 0; j < ${width}; j++) out[j] = i[j] - 2 * d * n[j];\n` +
        `  return out;\n}`
      );
    }
    if (op === "refract") {
      // refract(i, n, eta): k = 1 - eta^2 (1 - dot^2); eta*i - (eta*dot + sqrt(k))*n
      return (
        `function _${name}(i, n, eta, out) {\n` +
        `  out = out || new Array(${width});\n` +
        `  let d = 0;\n` +
        `  for (let j = 0; j < ${width}; j++) d += n[j] * i[j];\n` +
        `  let k = 1 - eta * eta * (1 - d * d);\n` +
        `  if (k < 0) { for (let j = 0; j < ${width}; j++) out[j] = 0; }\n` +
        `  else { let r = eta * d + Math.sqrt(k); for (let j = 0; j < ${width}; j++) out[j] = eta * i[j] - r * n[j]; }\n` +
        `  return out;\n}`
      );
    }
    if (op === "faceforward") {
      // faceforward(n, i, nref) = dot(nref, i) < 0 ? n : -n
      return (
        `function _${name}(n, i, nref, out) {\n` +
        `  out = out || new Array(${width});\n` +
        `  let d = 0;\n` +
        `  for (let j = 0; j < ${width}; j++) d += nref[j] * i[j];\n` +
        `  let s = d < 0 ? 1 : -1;\n` +
        `  for (let j = 0; j < ${width}; j++) out[j] = s * n[j];\n` +
        `  return out;\n}`
      );
    }
    if (op === "cross") {
      if (width !== 3) {
        throw new Error(`[RMSL] cross() needs a vec3 on the JS target, got width ${width}.`);
      }
      return (
        `function _v3cross(a, b, out) {\n` +
        `  out = out || [0, 0, 0];\n` +
        // Read before written: `out` may be `a` or `b`.
        `  let a0 = a[0], a1 = a[1], a2 = a[2], b0 = b[0], b1 = b[1], b2 = b[2];\n` +
        `  out[0] = a1 * b2 - a2 * b1;\n` +
        `  out[1] = a2 * b0 - a0 * b2;\n` +
        `  out[2] = a0 * b1 - a1 * b0;\n` +
        `  return out;\n}`
      );
    }
    throw new Error(`[RMSL] Unknown JS vector helper: ${name}`);
  }

  let bm = /^b(\d+)(and|or|not|eq|neq)$/.exec(name);
  if (bm) {
    let width = Number(bm[1]);
    let op = bm[2];
    let oneArg = op === "not";
    let params = oneArg ? "a" : "a, b";
    let lines: string[] = [];
    for (let i = 0; i < width; i++) {
      let body = oneArg
        ? `!a[${i}]`
        : op === "and"
          ? `a[${i}] && b[${i}]`
          : op === "or"
            ? `a[${i}] || b[${i}]`
            : op === "eq"
              ? `a[${i}] === b[${i}]`
              : `a[${i}] !== b[${i}]`;
      lines.push(`  out[${i}] = ${body};`);
    }
    return (
      `function _${name}(${params}, out) {\n` +
      `  out = out || [${Array(width).fill("false").join(", ")}];\n${lines.join("\n")}\n  return out;\n}`
    );
  }

  switch (name) {
    case "smoothstep":
      return `function _smoothstep(e0, e1, x) {\n  let t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);\n  return t * t * (3 - 2 * t);\n}`;
    case "idiv":
      return `function _idiv(a, b) {\n  return b === 0 || (a === -2147483648 && b === -1) ? a : (a / b) | 0;\n}`;
    case "imod":
      return `function _imod(a, b) {\n  return b === 0 ? 0 : (a % b) | 0;\n}`;
    case "udiv":
      return `function _udiv(a, b) {\n  return b === 0 ? a : (a / b) >>> 0;\n}`;
    case "umod":
      return `function _umod(a, b) {\n  return b === 0 ? 0 : a % b;\n}`;
    // A typed array's own set copies without boxing what it reads, whatever kind of array the source is.
    case "copy":
      return `function _copy(src, out) {\n  if (src.length === out.length && ArrayBuffer.isView(out)) out.set(src);\n  else for (let i = 0; i < src.length; i++) out[i] = src[i];\n  return out;\n}`;
    case "fr":
      return `function _fr(a) {\n  for (let i = 0; i < a.length; i++) a[i] = Math.fround(a[i]);\n  return a;\n}`;
    case "load":
      return `function _load(src, at, out) {\n  for (let i = 0; i < out.length; i++) out[i] = src[at + i];\n  return out;\n}`;
    case "vdot":
      return `function _vdot(a, b) {\n  let s = 0;\n  for (let i = 0; i < a.length; i++) s += a[i] * b[i];\n  return s;\n}`;
    case "vlen":
      return `function _vlen(a) {\n  let s = 0;\n  for (let i = 0; i < a.length; i++) s += a[i] * a[i];\n  return Math.sqrt(s);\n}`;
    case "vdist":
      return `function _vdist(a, b) {\n  let s = 0;\n  for (let i = 0; i < a.length; i++) { let d = a[i] - b[i]; s += d * d; }\n  return Math.sqrt(s);\n}`;
    case "ball":
      return `function _ball(v) {\n  for (let i = 0; i < v.length; i++) if (!v[i]) return false;\n  return true;\n}`;
    case "bselect":
      return `function _bselect(cond, a, b, out) {\n  out = out || new Array(a.length);\n  for (let i = 0; i < a.length; i++) out[i] = cond[i] ? a[i] : b[i];\n  return out;\n}`;
    case "bany":
      return `function _bany(v) {\n  for (let i = 0; i < v.length; i++) if (v[i]) return true;\n  return false;\n}`;
    case "matDiag":
      return `function _matDiag(s, size, stride) {\n  let m = new Array(size).fill(0);\n  for (let i = 0; i < size; i += stride) m[i] = s;\n  return m;\n}`;
    case "tex2d":
      return `function _tex2d(tex, uv, out) {
  out = out || [0, 0, 0, 0];
  let w = tex.width, h = tex.height, s = _unorm(tex), c = _chan(tex);
  if (tex.magFilter === "linear") {
    let fx = uv[0] * w - 0.5, fy = uv[1] * h - 0.5;
    let x0 = Math.floor(fx), y0 = Math.floor(fy);
    let tx = fx - x0, ty = fy - y0;
    let xa = _wrap(x0, w, tex.wrapS) * c, xb = _wrap(x0 + 1, w, tex.wrapS) * c;
    let ya = _wrap(y0, h, tex.wrapT) * w * c, yb = _wrap(y0 + 1, h, tex.wrapT) * w * c;
    for (let i = 0; i < 4; i++) {
      if (i >= c) { out[i] = i === 3 ? 1 : 0; continue; }
      let lower = tex.data[ya + xa + i] + (tex.data[ya + xb + i] - tex.data[ya + xa + i]) * tx;
      let upper = tex.data[yb + xa + i] + (tex.data[yb + xb + i] - tex.data[yb + xa + i]) * tx;
      out[i] = (lower + (upper - lower) * ty) / s;
    }
    return out;
  }
  let x = _wrap(Math.floor(uv[0] * w), w, tex.wrapS);
  let y = _wrap(Math.floor(uv[1] * h), h, tex.wrapT);
  return _texel(tex, (y * w + x) * c, c, s, out);
}`;
    case "texFetch2d":
      return `function _texFetch2d(tex, uv, out) {
  out = out || [0, 0, 0, 0];
  let x = Math.floor(uv[0]);
  let y = Math.floor(uv[1]);
  if (x < 0 || y < 0 || x >= tex.width || y >= tex.height) return out;
  let c = _chan(tex);
  return _texel(tex, (y * tex.width + x) * c, c, 1, out);
}`;
    case "tex3d":
      return `function _tex3d(tex, uvw, out) {
  out = out || [0, 0, 0, 0];
  let w = tex.width, h = tex.height, d = tex.depth, s = _unorm(tex), c = _chan(tex);
  if (tex.magFilter === "linear") {
    let fx = uvw[0] * w - 0.5, fy = uvw[1] * h - 0.5, fz = uvw[2] * d - 0.5;
    let x0 = Math.floor(fx), y0 = Math.floor(fy), z0 = Math.floor(fz);
    let tx = fx - x0, ty = fy - y0, tz = fz - z0;
    let xa = _wrap(x0, w, tex.wrapS) * c, xb = _wrap(x0 + 1, w, tex.wrapS) * c;
    let ya = _wrap(y0, h, tex.wrapT) * w * c, yb = _wrap(y0 + 1, h, tex.wrapT) * w * c;
    let za = _wrap(z0, d, tex.wrapR) * h * w * c, zb = _wrap(z0 + 1, d, tex.wrapR) * h * w * c;
    for (let i = 0; i < 4; i++) {
      if (i >= c) { out[i] = i === 3 ? 1 : 0; continue; }
      let near = _lerp2(tex.data, za + ya + xa + i, za + ya + xb + i, za + yb + xa + i, za + yb + xb + i, tx, ty);
      let far = _lerp2(tex.data, zb + ya + xa + i, zb + ya + xb + i, zb + yb + xa + i, zb + yb + xb + i, tx, ty);
      out[i] = (near + (far - near) * tz) / s;
    }
    return out;
  }
  let x = _wrap(Math.floor(uvw[0] * w), w, tex.wrapS);
  let y = _wrap(Math.floor(uvw[1] * h), h, tex.wrapT);
  let z = _wrap(Math.floor(uvw[2] * d), d, tex.wrapR);
  return _texel(tex, ((z * h + y) * w + x) * c, c, s, out);
}`;
    // One bilinear tap of a volume, so the trilinear blend above is two of
    // these and a step between them.
    case "lerp2":
      return `function _lerp2(data, a, b, c, e, tx, ty) {
  let lower = data[a] + (data[b] - data[a]) * tx;
  let upper = data[c] + (data[e] - data[c]) * tx;
  return lower + (upper - lower) * ty;
}`;
    // A direction to the face it lands on plus the 0..1 coordinate within
    // that face — the standard cube-map face-selection algorithm (major
    // axis picks the face; the other two components, divided by it, are the
    // face-local coordinate), matching the face order both GPU backends and
    // three.js's own CubeTexture agree on: +X,-X,+Y,-Y,+Z,-Z.
    case "cubeFace":
      return `function _cubeFace(x, y, z, out) {
  let ax = Math.abs(x), ay = Math.abs(y), az = Math.abs(z);
  let ma, uc, vc, face;
  if (ax >= ay && ax >= az) {
    ma = ax;
    if (x >= 0) { uc = -z; vc = -y; face = 0; } else { uc = z; vc = -y; face = 1; }
  } else if (ay >= ax && ay >= az) {
    ma = ay;
    if (y >= 0) { uc = x; vc = z; face = 2; } else { uc = x; vc = -z; face = 3; }
  } else {
    ma = az;
    if (z >= 0) { uc = x; vc = -y; face = 4; } else { uc = -x; vc = -y; face = 5; }
  }
  out[0] = face;
  out[1] = 0.5 * (uc / ma + 1);
  out[2] = 0.5 * (vc / ma + 1);
  return out;
}`;
    // A cube map is 6 square faces of the same width/height, stored back to
    // back in `data` (no depth field — 6 is a property of being a cube, not
    // of the texture). Sampling never blends across a face edge, the way a
    // volume texture blends across z, so this is `tex2d`'s bilinear tap
    // applied within one face (always clamped at its edges — cube maps have
    // no wrap mode on either GPU backend either) rather than `tex3d`'s
    // trilinear one.
    case "texCube":
      return `const _texCubeFace = [0, 0, 0];
function _texCube(tex, dir, out) {
  out = out || [0, 0, 0, 0];
  let f = _cubeFace(dir[0], dir[1], dir[2], _texCubeFace);
  let face = f[0], u = f[1], v = f[2];
  let w = tex.width, h = tex.height, s = _unorm(tex), c = _chan(tex);
  let base = face * w * h * c;
  if (tex.magFilter === "linear") {
    let fx = u * w - 0.5, fy = v * h - 0.5;
    let x0 = Math.floor(fx), y0 = Math.floor(fy);
    let tx = fx - x0, ty = fy - y0;
    let xa = _wrap(x0, w) * c, xb = _wrap(x0 + 1, w) * c;
    let ya = _wrap(y0, h) * w * c, yb = _wrap(y0 + 1, h) * w * c;
    for (let i = 0; i < 4; i++) {
      if (i >= c) { out[i] = i === 3 ? 1 : 0; continue; }
      let lower = tex.data[base + ya + xa + i] + (tex.data[base + ya + xb + i] - tex.data[base + ya + xa + i]) * tx;
      let upper = tex.data[base + yb + xa + i] + (tex.data[base + yb + xb + i] - tex.data[base + yb + xa + i]) * tx;
      out[i] = (lower + (upper - lower) * ty) / s;
    }
    return out;
  }
  let x = _wrap(Math.floor(u * w), w);
  let y = _wrap(Math.floor(v * h), h);
  return _texel(tex, base + (y * w + x) * c, c, s, out);
}`;
    // A texel index brought inside the image, the way a sampler's address mode
    // does it: the edge stretched, the image tiled, or tiled and flipped.
    case "wrap":
      return `function _wrap(i, n, mode) {
  if (mode === "repeat") return ((i % n) + n) % n;
  if (mode === "mirror") {
    let period = ((i % (2 * n)) + 2 * n) % (2 * n);
    return period < n ? period : 2 * n - 1 - period;
  }
  return i < 0 ? 0 : (i > n - 1 ? n - 1 : i);
}`;
    case "texFetch3d":
      return `function _texFetch3d(tex, uvw, out) {
  out = out || [0, 0, 0, 0];
  let x = Math.floor(uvw[0]);
  let y = Math.floor(uvw[1]);
  let z = Math.floor(uvw[2]);
  if (x < 0 || y < 0 || z < 0 || x >= tex.width || y >= tex.height || z >= tex.depth) return out;
  let c = _chan(tex);
  return _texel(tex, ((z * tex.height + y) * tex.width + x) * c, c, 1, out);
}`;
    case "texFetchUnorm2d":
      return `function _texFetchUnorm2d(tex, uv, out) {
  out = out || [0, 0, 0, 0];
  let x = Math.floor(uv[0]);
  let y = Math.floor(uv[1]);
  if (x < 0 || y < 0 || x >= tex.width || y >= tex.height) return out;
  let c = _chan(tex);
  return _texel(tex, (y * tex.width + x) * c, c, _unorm(tex), out);
}`;
    case "texFetchUnorm3d":
      return `function _texFetchUnorm3d(tex, uvw, out) {
  out = out || [0, 0, 0, 0];
  let x = Math.floor(uvw[0]);
  let y = Math.floor(uvw[1]);
  let z = Math.floor(uvw[2]);
  if (x < 0 || y < 0 || z < 0 || x >= tex.width || y >= tex.height || z >= tex.depth) return out;
  let c = _chan(tex);
  return _texel(tex, ((z * tex.height + y) * tex.width + x) * c, c, _unorm(tex), out);
}`;
    // What a float sampler divides a texel by. An 8-bit texture is uploaded to
    // both backends as a normalized format, so the shader reads 0..1 from data
    // stored as 0..255; anything else is taken as the value it already is.
    case "unorm":
      return `function _unorm(tex) {\n  let d = tex.data;\n  return (d instanceof Uint8Array || d instanceof Uint8ClampedArray) ? 255 : 1;\n}`;
    // How many channels a texel occupies, which is also the stride from one to
    // the next. Four unless the data says otherwise.
    case "chan":
      return `function _chan(tex) {\n  return tex.channels || 4;\n}`;
    // One texel read out as the four channels a shader sees. A texture that
    // stores fewer fills the rest the way a sampler does on a device: zero for
    // green and blue, one for alpha.
    case "texel":
      return `function _texel(tex, o, c, s, out) {
  out[0] = tex.data[o] / s;
  out[1] = c > 1 ? tex.data[o + 1] / s : 0;
  out[2] = c > 2 ? tex.data[o + 2] / s : 0;
  out[3] = c > 3 ? tex.data[o + 3] / s : 1;
  return out;
}`;
    case "texSize":
      return `function _texSize(tex, out) {\n  out = out || [0, 0, 0];\n  out[0] = tex.width;\n  out[1] = tex.height;\n  if (tex.depth !== undefined) out[2] = tex.depth;\n  return out;\n}`;
    case "mat2x2inv":
      return `function _mat2x2inv(m, out) {\n  out = out || new Array(4);\n  let a00 = m[0], a01 = m[1], a10 = m[2], a11 = m[3];\n  let inv = 1 / (a00 * a11 - a01 * a10);\n  out[0] = a11 * inv;\n  out[1] = -a01 * inv;\n  out[2] = -a10 * inv;\n  out[3] = a00 * inv;\n  return out;\n}`;
    case "mat3x3inv":
      return `function _mat3x3inv(m, out) {\n  out = out || new Array(9);\n  let a00 = m[0], a01 = m[1], a02 = m[2];\n  let a10 = m[3], a11 = m[4], a12 = m[5];\n  let a20 = m[6], a21 = m[7], a22 = m[8];\n  let b01 = a22 * a11 - a12 * a21;\n  let b11 = -a22 * a10 + a12 * a20;\n  let b21 = a21 * a10 - a11 * a20;\n  let det = a00 * b01 + a01 * b11 + a02 * b21;\n  let inv = 1 / det;\n  out[0] = b01 * inv;\n  out[1] = (-a22 * a01 + a02 * a21) * inv;\n  out[2] = (a12 * a01 - a02 * a11) * inv;\n  out[3] = b11 * inv;\n  out[4] = (a22 * a00 - a02 * a20) * inv;\n  out[5] = (-a12 * a00 + a02 * a10) * inv;\n  out[6] = b21 * inv;\n  out[7] = (-a21 * a00 + a01 * a20) * inv;\n  out[8] = (a11 * a00 - a01 * a10) * inv;\n  return out;\n}`;
    case "mat4x4inv":
      return `function _mat4x4inv(m, out) {\n  out = out || new Array(16);\n  let a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];\n  let a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];\n  let a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];\n  let a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];\n  let b00 = a00 * a11 - a01 * a10;\n  let b01 = a00 * a12 - a02 * a10;\n  let b02 = a00 * a13 - a03 * a10;\n  let b03 = a01 * a12 - a02 * a11;\n  let b04 = a01 * a13 - a03 * a11;\n  let b05 = a02 * a13 - a03 * a12;\n  let b06 = a20 * a31 - a21 * a30;\n  let b07 = a20 * a32 - a22 * a30;\n  let b08 = a20 * a33 - a23 * a30;\n  let b09 = a21 * a32 - a22 * a31;\n  let b10 = a21 * a33 - a23 * a31;\n  let b11 = a22 * a33 - a23 * a32;\n  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;\n  let inv = 1 / det;\n  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * inv;\n  out[1] = (-a01 * b11 + a02 * b10 - a03 * b09) * inv;\n  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * inv;\n  out[3] = (-a21 * b05 + a22 * b04 - a23 * b03) * inv;\n  out[4] = (-a10 * b11 + a12 * b08 - a13 * b07) * inv;\n  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * inv;\n  out[6] = (-a30 * b05 + a32 * b02 - a33 * b01) * inv;\n  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * inv;\n  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * inv;\n  out[9] = (-a00 * b10 + a01 * b08 - a03 * b06) * inv;\n  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * inv;\n  out[11] = (-a20 * b04 + a21 * b02 - a23 * b00) * inv;\n  out[12] = (-a10 * b09 + a11 * b07 - a12 * b06) * inv;\n  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * inv;\n  out[14] = (-a30 * b03 + a31 * b01 - a32 * b00) * inv;\n  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * inv;\n  return out;\n}`;
    case "mat2x2det":
      return `function _mat2x2det(m) {\n  return m[0] * m[3] - m[1] * m[2];\n}`;
    case "mat3x3det":
      return `function _mat3x3det(m) {\n  let a00 = m[0], a01 = m[1], a02 = m[2];\n  let a10 = m[3], a11 = m[4], a12 = m[5];\n  let a20 = m[6], a21 = m[7], a22 = m[8];\n  return a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20);\n}`;
    case "mat4x4det":
      return `function _mat4x4det(m) {\n  let a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];\n  let a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];\n  let a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];\n  let a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];\n  let b00 = a00 * a11 - a01 * a10;\n  let b01 = a00 * a12 - a02 * a10;\n  let b02 = a00 * a13 - a03 * a10;\n  let b03 = a01 * a12 - a02 * a11;\n  let b04 = a01 * a13 - a03 * a11;\n  let b05 = a02 * a13 - a03 * a12;\n  let b06 = a20 * a31 - a21 * a30;\n  let b07 = a20 * a32 - a22 * a30;\n  let b08 = a20 * a33 - a23 * a30;\n  let b09 = a21 * a32 - a22 * a31;\n  let b10 = a21 * a33 - a23 * a31;\n  let b11 = a22 * a33 - a23 * a32;\n  return b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;\n}`;
  }

  // A matCLxRL times a matCRxRR (CL === RR) is a matCRxRL product.
  let mmMul = /^matmul(\d+)x(\d+)x(\d+)x(\d+)$/.exec(name);
  if (mmMul) {
    let cL = Number(mmMul[1]);
    let rL = Number(mmMul[2]);
    let cR = Number(mmMul[3]);
    let rR = Number(mmMul[4]);
    // Column-major product: out[col*rL + row] = sum_k a[k*rL+row] * b[col*rR+k].
    let lines: string[] = [];
    for (let col = 0; col < cR; col++) {
      for (let row = 0; row < rL; row++) {
        let terms: string[] = [];
        for (let k = 0; k < cL; k++) terms.push(`a[${k * rL + row}] * b[${col * rR + k}]`);
        lines.push(`  out[${col * rL + row}] = ${terms.join(" + ")};`);
      }
    }
    // A product written into one of its own operands reads that operand while it
    // writes it, so it reads a copy: one array for each operand, made once.
    let copyA = `_${name}A`;
    let copyB = `_${name}B`;
    return (
      `const ${copyA} = new Array(${cL * rL}), ${copyB} = new Array(${cR * rR});\n` +
      `function _${name}(a, b, out) {\n` +
      `  out = out || new Array(${cR * rL});\n` +
      `  if (out === a) { for (let i = 0; i < ${cL * rL}; i++) ${copyA}[i] = a[i]; a = ${copyA}; }\n` +
      `  if (out === b) { for (let i = 0; i < ${cR * rR}; i++) ${copyB}[i] = b[i]; b = ${copyB}; }\n${lines.join("\n")}\n  return out;\n}`
    );
  }

  let mmT = /^mat(\d+)x(\d+)T$/.exec(name);
  if (mmT) {
    let cols = Number(mmT[1]);
    let rows = Number(mmT[2]);
    // Transpose: out[r*cols + c] = m[c*rows + r], from locals, since `out` may be `m`.
    let locals = Array.from({ length: cols * rows }, (_, i) => `m${i} = m[${i}]`);
    let lines: string[] = [];
    for (let c = 0; c < cols; c++)
      for (let r = 0; r < rows; r++) {
        lines.push(`  out[${r * cols + c}] = m${c * rows + r};`);
      }
    return (
      `function _${name}(m, out) {\n` +
      `  out = out || new Array(${cols * rows});\n  let ${locals.join(", ")};\n${lines.join("\n")}\n  return out;\n}`
    );
  }

  let mvm = /^mat(\d+)x(\d+)mv(\d+)$/.exec(name);
  if (mvm) {
    let cols = Number(mvm[1]);
    let rows = Number(mvm[2]);
    let vlen = Number(mvm[3]);
    // A shorter vector (mat4 * vec3) implies w = 1 and drops the w row.
    let outRows = vlen < cols ? vlen : rows;
    let locals = Array.from({ length: vlen }, (_, i) => `x${i} = v[${i}]`);
    let lines: string[] = [];
    for (let row = 0; row < outRows; row++) {
      let terms: string[] = [];
      for (let c = 0; c < vlen; c++) terms.push(`m[${c * rows + row}] * x${c}`);
      if (vlen < cols) terms.push(`m[${vlen * rows + row}]`);
      lines.push(`  out[${row}] = ${terms.join(" + ")};`);
    }
    return (
      `function _${name}(m, v, out) {\n` +
      `  out = out || new Array(${outRows});\n` +
      `  let ${locals.join(", ")};\n${lines.join("\n")}\n  return out;\n}`
    );
  }

  if (name === "roundEven") {
    // The function constant folding uses, so folded and run-time rounding cannot differ.
    const source = roundHalfToEven.toString().replace(/^function[^(]*/, "function _rmsl_roundEven");
    if (!source.startsWith("function _rmsl_roundEven(")) {
      throw new Error("[RMSL] roundHalfToEven must be a function declaration to become the JS rounding helper");
    }
    return source;
  }

  throw new Error(`[RMSL] Unknown JS helper: ${name}`);
}

export function jsRequireHelper(ctx: CompileCtx, name: string): void {
  ctx.jsHelpers.add(name);
  // A vector helper calls the helper its `JS_ELEM` entry names.
  const elementHelper = JS_ELEM[/^v\d+([a-zA-Z]+)(?:_[vs]+)?$/.exec(name)?.[1] ?? ""]?.helper;
  if (elementHelper) ctx.jsHelpers.add(elementHelper);
}

/** A constant array, declared once beside the function and read from there, so reading it allocates nothing. */
function jsConstant(ctx: CompileCtx, literal: string): string {
  ctx.jsConstants ??= new Map();
  for (const [name, text] of ctx.jsConstants) if (text === literal) return name;
  const name = `_rmsl_k${ctx.jsConstants.size}`;
  ctx.jsConstants.set(name, literal);
  return name;
}

/**
 * The declarations of a function's slots. A scalar is a local of the function.
 * A vector or matrix is a typed view of its kind into one `ArrayBuffer`, made
 * once with the declarations: `Float64Array`, or `Float32Array` when `float32`,
 * `Int32Array` for integers and for booleans as 1 or 0, and `Uint32Array`. The
 * 8-byte views come first, so each lies on a multiple of its element size.
 */
function jsSlotDeclarations(
  varDefs: Map<string, string>,
  float32: boolean,
  keyword: "let" | "var",
): { views: string[]; scalars: string[] } {
  const typed: { name: string; type: string; bytes: number; length: number }[] = [];
  const names: string[] = [];
  for (const [name, brand] of varDefs) {
    const length = jsArrayLength(brand);
    if (length <= 1) {
      names.push(name);
      continue;
    }
    const Typed = typedArrayOfKind(elementKindOf(brand), float32);
    typed.push({ name, type: Typed.name, bytes: Typed.BYTES_PER_ELEMENT, length });
  }
  // A scalar is a local of the function: V8 boxes a float stored in a variable the closure keeps.
  const scalars = names.length ? [`let ${names.join(", ")};`] : [];
  if (typed.length === 0) return { views: [], scalars };
  typed.sort((a, b) => b.bytes - a.bytes);
  let offset = 0;
  const views = typed.map(({ name, type, bytes, length }) => {
    const line = `${keyword} ${name} = new ${type}(_rmsl_slots, ${offset}, ${length});`;
    offset += bytes * length;
    return line;
  });
  return {
    views: [`${keyword === "var" ? "var" : "const"} _rmsl_slots = new ArrayBuffer(${offset});`, ...views],
    scalars,
  };
}

/** A fresh hoisted slot for an intermediate value, registered for preallocation. */
export function jsNewTemp(ctx: CompileCtx, brand: string): string {
  let name = `_rmsl_t${ctx.nextId++}`;
  ctx.varDefs.set(name, brand);
  return name;
}

/** Whether `expr` is a name, a member of one, or a number, so reading it again costs nothing and changes nothing. */
function jsIsReference(expr: string): boolean {
  return /^[_$a-zA-Z][\w$]*(\.[\w$]+|\[(\d+|"[^"]*")\])*$/.test(expr) || /^-?\d+(\.\d+)?(e[+-]?\d+)?$/.test(expr);
}

/**
 * `compiled` as an expression that can be read several times: a reference is
 * read as it is, and any other expression is stored in a hoisted slot first,
 * so the program emits it once however many components take from it.
 */
function jsReadable(compiled: CompiledNode, brand: string | undefined, ctx: CompileCtx): CompiledNode {
  if (jsIsReference(compiled.expr) || brand === undefined) return compiled;
  const slot = jsNewTemp(ctx, brand);
  if (jsIsArrayType(brand)) {
    // An array slot is a view that is written into, never rebound.
    jsRequireHelper(ctx, "copy");
    return { ...compiled, body: [...compiled.body, `_copy(${compiled.expr}, ${slot});`], expr: slot };
  }
  return { ...compiled, body: [...compiled.body, `${slot} = ${compiled.expr};`], expr: slot };
}

/**
 * An index computed at run time, kept inside `count` items as WASM keeps it:
 * truncated, and past the end or negative (a huge unsigned number) selecting
 * the last item.
 */
function jsBoundedIndex(index: string, count: number): string {
  return `Math.min((${index}) >>> 0, ${count - 1})`;
}

/**
 * A vector or a matrix column as the target of a component write: `at(k)` is
 * the expression holding its `k`th component. Reading a column gives a copy
 * (`slice`), so a column's components are addressed in the matrix itself,
 * through its index evaluated once, and bounded to the matrix, into a
 * temporary.
 */
function jsAssignable(node: any, ctx: CompileCtx): CompiledNode & { at(k: string): string } {
  if (node.type === "storageElement" && jsIsArrayType(node._t)) {
    let element = jsStorageElement(node, ctx);
    return {
      decls: element.decls,
      body: element.body,
      expr: "0",
      at: (k) => `${element.buffer}[${element.start} + ${k}]`,
    };
  }
  if (node.type !== "matrixElement") {
    let target = jsCompileTarget(node, ctx);
    return { ...target, at: (k) => `${target.expr}[${k}]` };
  }
  assertLiteralIndexInRange(node.params![0], node.params![1]);
  let mat = jsAssignable(node.params![0], ctx);
  let idx = compileJSStage(node.params![1], ctx);
  let [columns, rows] = MATRIX_DIMENSIONS[node.params![0]._t];
  let column = jsNewTemp(ctx, "int");
  return {
    decls: [...mat.decls, ...idx.decls],
    body: [...mat.body, ...idx.body, `${column} = ${jsBoundedIndex(idx.expr, columns)};`],
    expr: mat.expr,
    at: (k) => mat.at(`${column} * ${rows} + ${k}`),
  };
}

/**
 * An operand whose components a read reaches one by one: `at(k)` is component
 * `k`. A vector or matrix storage element is read in its buffer, without a copy.
 */
function jsComponents(node: any, ctx: CompileCtx): CompiledNode & { at(k: string): string; inBuffer: boolean } {
  if (node?.type === "storageElement" && jsIsArrayType(node._t)) {
    let element = jsStorageElement(node, ctx);
    return { ...element, at: (k) => `${element.buffer}[${element.start} + ${k}]`, inBuffer: true };
  }
  let src = jsCompileOperand(node, ctx);
  let srcExpr = (src.prec ?? PREC_ATOM) < PREC_ATOM ? `(${src.expr})` : src.expr;
  return { ...src, at: (k) => `${srcExpr}[${k}]`, inBuffer: false };
}

/** The index of a storage element, a float index without its fraction as WGSL's `u32()` and WASM give it. */
function jsStorageIndex(node: any, index: string): string {
  // As WGSL's u32(): the fraction dropped, and a negative index clamped to 0.
  return node.params![1]?._t === "float" ? `Math.trunc(Math.max(${index}, 0))` : `(${index})`;
}

/**
 * A vector or matrix element of a storage buffer, which holds its components
 * one after another: they lie in `buffer` from `start`, evaluated once.
 */
function jsStorageElement(node: any, ctx: CompileCtx): CompiledNode & { buffer: string; start: string } {
  let buffer = jsCompileOperand(node.params![0], ctx);
  let idx = jsCompileOperand(node.params![1], ctx);
  let start = jsNewTemp(ctx, "int");
  return {
    decls: [...buffer.decls, ...idx.decls],
    body: [...buffer.body, ...idx.body, `${start} = ${jsStorageIndex(node, idx.expr)} * ${componentCountOf(node._t)};`],
    expr: start,
    buffer: buffer.expr,
    start,
  };
}

/**
 * Compile an operand for a vector/matrix operation.
 *
 * An array-typed operand that itself computes something needs its own scratch
 * slot, in a plain expression as under an assignment: the parent's helper may
 * write into its target while it reads its operands, so an operand may never
 * share that slot, and a value that is not stored in one is a new array on every
 * call. Leaves (variables, uniforms, constants) are references and need no slot.
 */
export function jsCompileOperand(node: any, ctx: CompileCtx): CompiledNode {
  let saved = ctx.outTarget;
  let isArrayOp = jsIsArrayType(node?._t) && !isJSArrayLeaf(node);
  if (isArrayOp) {
    let temp = jsNewTemp(ctx, node._t);
    ctx.outTarget = temp;
    let result = compileJSStage(node, ctx);
    ctx.outTarget = saved;
    return result;
  }
  ctx.outTarget = null;
  let result = compileJSStage(node, ctx);
  ctx.outTarget = saved;
  return result;
}

/** A leaf reference read as a value — copied into the target under out-mode. */
export function jsLeafRef(expr: string, brand: string | undefined, ctx: CompileCtx): CompiledNode {
  if (ctx.outTarget && jsIsArrayType(brand)) {
    jsRequireHelper(ctx, "copy");
    return { decls: [], body: [`_copy(${expr}, ${ctx.outTarget});`], expr: ctx.outTarget };
  }
  return { decls: [], body: [], expr };
}

export function isPlainJSIdentifier(s: string): boolean {
  return /^[_$a-zA-Z][_$a-zA-Z0-9]*$/.test(s);
}

/**
 * One component converted between two types' scalar kinds, as a constructor
 * converts it in both shading languages: a boolean reads as 1 or 0, a signed
 * integer truncates toward zero, an unsigned one wraps, and every non-zero
 * value is true.
 *
 * JavaScript numbers hold all four kinds, so without this a boolean stays a
 * boolean — arithmetic on it still coerces, but a comparison does not, and
 * `false !== 0` is true.
 */
export function jsComponentCast(expr: string, sourceType: string | undefined, targetType: string): string {
  let from = componentKindOf(sourceType);
  let to = componentKindOf(targetType);
  if (from === to) return expr;
  if (from === "bool") expr = `(${expr} ? 1 : 0)`;
  if (to === "bool") return `(${expr} !== 0)`;
  // uint and int convert to each other keeping the bits, as WGSL's do.
  // A float truncates toward zero and clamps to the range WebGPU clamps to, with NaN as 0.
  if (to === "int" && from === "float")
    return `(Math.trunc(Math.min(Math.max(${expr}, -2147483648), 2147483520)) || 0)`;
  if (to === "uint" && from === "float") return `(Math.trunc(Math.min(Math.max(${expr}, 0), 4294967040)) || 0)`;
  if (to === "int") return from === "uint" ? `((${expr}) | 0)` : `Math.trunc(${expr})`;
  if (to === "uint") return `((${expr}) >>> 0)`;
  return expr;
}

/**
 * An operand safe to drop into a formula.
 *
 * A call's arguments are separated by commas, so an operand of any shape can go
 * there unchanged. A formula is different: written into
 * `a - b * Math.floor(a / b)`, an operand that is itself a sum binds to the
 * neighbouring term rather than arriving whole, and `(x + y) mod m` quietly
 * becomes `x + y - m * floor(x + y / m)`.
 *
 * So anything that is not already a single value gets brackets. A literal, a
 * variable and a call are left alone, which is what the absent precedence on
 * those means.
 */
export function jsOperand(compiled: CompiledNode): string {
  return (compiled.prec ?? PREC_ATOM) < PREC_ATOM ? `(${compiled.expr})` : compiled.expr;
}

/**
 * The operands of a scalar form that writes an operand out twice, `mod` twice
 * and `mix` and `smoothstep` once, by position. Each is stored first, so the
 * form computes it once.
 */
const JS_REPEATED_OPERANDS: Record<string, number[]> = { mod: [0, 1], mix: [0], smoothstep: [0] };

export function jsScalarBinary(node: BaseNode<ShaderType>, ctx: CompileCtx, op: string): CompiledNode {
  const repeated = JS_REPEATED_OPERANDS[op] ?? [];
  const operand = (i: number) => {
    const compiled = compileJSStage(node.params![i], ctx);
    return repeated.includes(i) ? jsReadable(compiled, node.params![i]?._t, ctx) : compiled;
  };
  let a = operand(0);
  let b = operand(1);
  let c = node.params![2] ? operand(2) : null;
  let decls = [...a.decls, ...b.decls, ...(c ? c.decls : [])];
  let body = [...a.body, ...b.body, ...(c ? c.body : [])];
  let expr: string;
  switch (op) {
    case "add":
    case "sub":
    case "mul":
    case "div": {
      let sym = op === "add" ? "+" : op === "sub" ? "-" : op === "mul" ? "*" : "/";
      let prec = PRECEDENCE[node.type] ?? 0;
      expr = `${wrapExpr(a.prec, prec, a.expr)} ${sym} ${wrapExpr(b.prec, prec, b.expr)}`;
      break;
    }
    // The formula-shaped cases below take their operands through jsOperand, so
    // an operand that is itself an expression arrives whole. The call-shaped
    // ones do not need it: a comma already separates their arguments.
    case "iadd":
    case "isub":
    case "imul":
    case "uadd":
    case "usub":
    case "umul": {
      let form = jsIntegerForm(op, [a, b]);
      return { decls, body, expr: form.expr, prec: form.prec };
    }
    case "idiv":
    case "imod":
    case "udiv":
    case "umod":
      jsRequireHelper(ctx, op);
      expr = `_${op}(${a.expr}, ${b.expr})`;
      break;
    case "min":
      expr = `Math.min(${a.expr}, ${b.expr})`;
      break;
    case "max":
      expr = `Math.max(${a.expr}, ${b.expr})`;
      break;
    case "pow":
      expr = `Math.pow(${a.expr}, ${b.expr})`;
      break;
    case "atan2":
      expr = `Math.atan2(${a.expr}, ${b.expr})`;
      break;
    case "mod":
      expr = `(${jsOperand(a)} - ${jsOperand(b)} * Math.floor(${jsOperand(a)} / ${jsOperand(b)}))`;
      break;
    case "step":
      expr = `(${b.expr} < ${a.expr} ? 0 : 1)`;
      break;
    case "clamp":
      expr = `Math.min(Math.max(${a.expr}, ${b.expr}), ${c!.expr})`;
      break;
    case "mix":
      expr = `(${jsOperand(a)} + ${jsOperand(c!)} * (${jsOperand(b)} - ${jsOperand(a)}))`;
      break;
    case "smoothstep": {
      // Written out in the function: V8 boxes a float returned from a call it does not inline.
      let t = jsNewTemp(ctx, "float");
      body.push(`${t} = Math.min(Math.max((${c!.expr} - ${a.expr}) / (${b.expr} - ${a.expr}), 0), 1);`);
      expr = `(${t} * ${t} * (3 - 2 * ${t}))`;
      break;
    }
    default:
      throw new Error(`[RMSL] Unknown JS scalar op: ${op}`);
  }
  return { decls, body, expr, prec: PRECEDENCE[node.type] };
}

export function jsVectorBinary(node: BaseNode<ShaderType>, ctx: CompileCtx, op: string, width: number): CompiledNode {
  let a = jsCompileOperand(node.params![0], ctx);
  let b = jsCompileOperand(node.params![1], ctx);
  let c = node.params![2] ? jsCompileOperand(node.params![2], ctx) : null;
  // The helper is written for the shape of each operand, which the types give.
  let shape = (node.params ?? []).map((p) => (jsArrayLength(p?._t) > 1 ? "v" : "s")).join("");
  if (ctx.outTarget) {
    let written = jsElementwise(ctx.outTarget, op, width, shape, [a, b, ...(c ? [c] : [])], node.params!, ctx);
    if (written) return written;
  }
  let helper = `v${width}${op}_${shape}`;
  jsRequireHelper(ctx, helper);
  let args = c ? `${a.expr}, ${b.expr}, ${c.expr}` : `${a.expr}, ${b.expr}`;
  let decls = [...a.decls, ...b.decls, ...(c ? c.decls : [])];
  let body = [...a.body, ...b.body, ...(c ? c.body : [])];
  if (ctx.outTarget) {
    return { decls, body: [...body, `_${helper}(${args}, ${ctx.outTarget});`], expr: ctx.outTarget };
  }
  return { decls, body, expr: `_${helper}(${args})` };
}

/**
 * The integer variant of an arithmetic operation — `iadd`, `udiv` and so on —
 * when `node` produces `int`/`uint` or a vector of them, and `op` unchanged
 * otherwise. JS numbers are doubles, so integer arithmetic needs its own
 * operations to wrap to 32 bits and match WGSL's division by zero.
 */
export function jsIntegerOp(node: BaseNode<ShaderType>, op: string): string {
  let kind = componentKindOf(node._t as string);
  return kind === "int" ? `i${op}` : kind === "uint" ? `u${op}` : op;
}

export function jsBinaryOp(node: BaseNode<ShaderType>, ctx: CompileCtx, op: string): CompiledNode {
  let width = Math.max(jsArrayLength(node.params![0]?._t), jsArrayLength(node.params![1]?._t));
  if (width <= 1) return jsScalarBinary(node, ctx, op);
  return jsVectorBinary(node, ctx, op, width);
}

/** Matrix times matrix, a separate operator from element-wise mult. */
export function jsMatMul(node: BaseNode<ShaderType>, ctx: CompileCtx): CompiledNode {
  let aType = node.params![0]?._t;
  let bType = node.params![1]?._t;
  let [cL, rL] = MATRIX_DIMENSIONS[aType];
  let [cR, rR] = MATRIX_DIMENSIONS[bType];
  let a = jsCompileOperand(node.params![0], ctx);
  let b = jsCompileOperand(node.params![1], ctx);
  let name = `matmul${cL}x${rL}x${cR}x${rR}`;
  jsRequireHelper(ctx, name);
  if (ctx.outTarget) {
    return {
      decls: [...a.decls, ...b.decls],
      body: [...a.body, ...b.body, `_${name}(${a.expr}, ${b.expr}, ${ctx.outTarget});`],
      expr: ctx.outTarget,
    };
  }
  return { decls: [...a.decls, ...b.decls], body: [...a.body, ...b.body], expr: `_${name}(${a.expr}, ${b.expr})` };
}

export function jsMatrixUnary(node: BaseNode<ShaderType>, ctx: CompileCtx, suffix: string): CompiledNode {
  let brand = node.params![0]?._t;
  let [c, r] = MATRIX_DIMENSIONS[brand];
  let name = `mat${c}x${r}${suffix}`;
  jsRequireHelper(ctx, name);
  let a = jsCompileOperand(node.params![0], ctx);
  if (ctx.outTarget) {
    return {
      decls: a.decls,
      body: [...a.body, `_${name}(${a.expr}, ${ctx.outTarget});`],
      expr: ctx.outTarget,
    };
  }
  return { decls: a.decls, body: a.body, expr: `_${name}(${a.expr})` };
}

/**
 * An element-wise operation written out in the function, one line for each
 * component of `target`, or `null` where it cannot be. V8 boxes a float it
 * passes to a call it does not inline, and a large function inlines few. A
 * component reads only the same component of each operand, so `target` may be
 * one of them. A scalar operand that is not a local or a number is stored in a
 * local first, since it may read a component of `target` written before it.
 */
function jsElementwise(
  target: string,
  op: string,
  width: number,
  shape: string,
  operands: CompiledNode[],
  nodes: any[],
  ctx: CompileCtx,
): CompiledNode | null {
  let e = JS_ELEM[op];
  if (!e || !isPlainJSIdentifier(target)) return null;
  let decls = operands.flatMap((o) => o.decls);
  let body = operands.flatMap((o) => o.body);
  let reads: string[] = [];
  for (let k = 0; k < operands.length; k++) {
    let o = operands[k]!;
    if (shape[k] === "v") {
      if (!jsIsReference(o.expr)) return null;
      reads.push(o.expr);
    } else if (isPlainJSIdentifier(o.expr) || /^-?\d+(\.\d+)?(e[+-]?\d+)?$/.test(o.expr)) reads.push(o.expr);
    else {
      let local = jsNewTemp(ctx, nodes[k]?._t ?? "float");
      body.push(`${local} = ${o.expr};`);
      reads.push(local);
    }
  }
  if (e.helper) jsRequireHelper(ctx, e.helper);
  for (let i = 0; i < width; i++) {
    let xs = reads.map((r, k) => (shape[k] === "v" ? `${r}[${i}]` : r));
    body.push(`${target}[${i}] = ${e.fn(xs)};`);
  }
  return { decls, body, expr: target };
}

export function jsUnaryMath(node: BaseNode<ShaderType>, ctx: CompileCtx, suffix: string): CompiledNode {
  let width = jsArrayLength(node.params![0]?._t);
  if (width <= 1) {
    let a = compileJSStage(node.params![0], ctx);
    let e = JS_ELEM[suffix];
    if (!e) throw new Error(`[RMSL] Unknown JS unary op: ${suffix}`);
    if (e.helper) jsRequireHelper(ctx, e.helper);
    // `fract` writes its operand out twice.
    if (suffix === "fract") a = jsReadable(a, node.params![0]?._t, ctx);
    return { decls: a.decls, body: a.body, expr: e.fn([`(${a.expr})`]), prec: JS_FORM_PREC[suffix] };
  }
  let a = jsCompileOperand(node.params![0], ctx);
  if (ctx.outTarget) {
    let written = jsElementwise(ctx.outTarget, suffix, width, "v", [a], node.params!, ctx);
    if (written) return written;
  }
  jsRequireHelper(ctx, `v${width}${suffix}`);
  if (ctx.outTarget) {
    return {
      decls: a.decls,
      body: [...a.body, `_v${width}${suffix}(${a.expr}, ${ctx.outTarget});`],
      expr: ctx.outTarget,
    };
  }
  return { decls: a.decls, body: a.body, expr: `_v${width}${suffix}(${a.expr})` };
}

/** A unary operation producing a vector written through an `out` slot. */
export function jsVecOutOp(node: BaseNode<ShaderType>, ctx: CompileCtx, suffix: string): CompiledNode {
  let width = jsArrayLength(node.params![0]?._t);
  jsRequireHelper(ctx, `v${width}${suffix}`);
  let args = (node.params ?? []).map((p) => jsCompileOperand(p, ctx));
  let decls = args.flatMap((a) => a.decls);
  let body = args.flatMap((a) => a.body);
  if (ctx.outTarget) {
    return {
      decls,
      body: [...body, `_v${width}${suffix}(${args.map((a) => a.expr).join(", ")}, ${ctx.outTarget});`],
      expr: ctx.outTarget,
    };
  }
  return { decls, body, expr: `_v${width}${suffix}(${args.map((a) => a.expr).join(", ")})` };
}

/** dot/length/distance — reduce to a scalar, so never written into a target. */
export function jsVecReduce(node: BaseNode<ShaderType>, ctx: CompileCtx, helper: string): CompiledNode {
  let a = jsCompileOperand(node.params![0], ctx);
  let b = node.params![1] ? jsCompileOperand(node.params![1], ctx) : null;
  let decls = b ? [...a.decls, ...b.decls] : a.decls;
  let body = b ? [...a.body, ...b.body] : a.body;
  // A sum written out in the function: V8 boxes a float returned from a call it does not inline.
  // It starts from 0 and adds in order, as the helper does, so the bits stay the same.
  let width = jsArrayLength(node.params![0]?._t);
  if (helper !== "ball" && helper !== "bany" && jsIsReference(a.expr) && (!b || jsIsReference(b.expr))) {
    let term = (i: number) =>
      helper === "vdot"
        ? `${a.expr}[${i}] * ${b!.expr}[${i}]`
        : helper === "vlen"
          ? `${a.expr}[${i}] * ${a.expr}[${i}]`
          : `(${a.expr}[${i}] - ${b!.expr}[${i}]) * (${a.expr}[${i}] - ${b!.expr}[${i}])`;
    let sum = `0 + ${Array.from({ length: width }, (_, i) => term(i)).join(" + ")}`;
    return { decls, body, expr: helper === "vdot" ? `(${sum})` : `Math.sqrt(${sum})`, prec: PREC_ATOM };
  }
  jsRequireHelper(ctx, helper);
  return {
    decls,
    body,
    expr: b ? `_${helper}(${a.expr}, ${b.expr})` : `_${helper}(${a.expr})`,
  };
}

export function jsComparison(node: BaseNode<ShaderType>, ctx: CompileCtx, op: string): CompiledNode {
  let width = Math.max(jsArrayLength(node.params![0]?._t), jsArrayLength(node.params![1]?._t));
  if (width <= 1) {
    let a = compileJSStage(node.params![0], ctx);
    let b = compileJSStage(node.params![1], ctx);
    let prec = PRECEDENCE[node.type] ?? 0;
    return {
      decls: [...a.decls, ...b.decls],
      body: [...a.body, ...b.body],
      expr: `${wrapExpr(a.prec, prec, a.expr)} ${op} ${wrapExpr(b.prec, prec, b.expr)}`,
      prec,
    };
  }
  let a = jsCompileOperand(node.params![0], ctx);
  let b = jsCompileOperand(node.params![1], ctx);
  if (ctx.outTarget) {
    let lines = Array.from(
      { length: width },
      (_, i) => `${ctx.outTarget}[${i}] = ${a.expr}[${i}] ${op} ${b.expr}[${i}];`,
    );
    return {
      decls: [...a.decls, ...b.decls],
      body: [...a.body, ...b.body, ...lines],
      expr: ctx.outTarget,
    };
  }
  let pieces = Array.from({ length: width }, (_, i) => `${a.expr}[${i}] ${op} ${b.expr}[${i}]`).join(", ");
  return {
    decls: [...a.decls, ...b.decls],
    body: [...a.body, ...b.body],
    expr: `[${pieces}]`,
  };
}

/** The element-wise helper name for each bitwise operator, before its `i`/`u` prefix. */
const JS_BITWISE_NAMES: Record<string, string> = {
  "&": "and",
  "|": "or",
  "^": "xor",
  "<<": "shl",
  ">>": "shr",
  "~": "not",
};

export function jsBitwise(node: BaseNode<ShaderType>, ctx: CompileCtx, op: string): CompiledNode {
  let width = jsArrayLength(node._t);
  if (width > 1) {
    let name = jsIntegerOp(node, JS_BITWISE_NAMES[op]!);
    return op === "~" ? jsUnaryMath(node, ctx, name) : jsVectorBinary(node, ctx, name, width);
  }
  let name = jsIntegerOp(node, JS_BITWISE_NAMES[op]!);
  let operands = (node.params ?? []).slice(0, op === "~" ? 1 : 2).map((p) => compileJSStage(p, ctx));
  let form = jsIntegerForm(name, operands);
  return {
    decls: operands.flatMap((o) => o.decls),
    body: operands.flatMap((o) => o.body),
    expr: form.expr,
    prec: form.prec,
  };
}

export function compileJSStage(node: any, ctx: CompileCtx): CompiledNode {
  if (node === undefined || node === null) {
    return { decls: [], body: [], expr: "0" };
  }
  if (typeof node === "boolean") {
    return { decls: [], body: [], expr: node ? "true" : "false" };
  }
  if (typeof node === "number") {
    return { decls: [], body: [], expr: String(node) };
  }
  if (Array.isArray(node)) {
    return { decls: [], body: [], expr: `[${node.join(", ")}]` };
  }

  // A target is compiled as what it names: a read of the same node is cached as its value, rounded or copied.
  let target = ctx.jsTarget === node;
  let seen = target ? undefined : ctx.memo.get(node);
  if (seen) {
    // A statement runs once. A seq's statements have run, and its value is read as any value is.
    if (node._t === "void") return { decls: [], body: [], expr: seen.expr, prec: seen.prec };
    if (node.type === "seq") return compileJSStage(node.params[node.params.length - 1], ctx);
    if (seen.jsEpoch === undefined || seen.jsEpoch === ctx.jsEpoch) {
      if (seen.jsEpoch !== undefined) ctx.jsReadsSlot = true;
      return { decls: [], body: [], expr: seen.expr, prec: seen.prec };
    }
  }

  let outer = ctx.jsReadsSlot;
  ctx.jsReadsSlot = false;
  let result = compileJSNode(node, ctx);
  if (target) {
    ctx.jsReadsSlot = outer || ctx.jsReadsSlot;
    return result;
  }
  result = jsTypedInput(node, result, ctx);
  if (ctx.jsFloat32) result = jsRound32(node, result, ctx);
  // A value computed into a slot holds what it was there, so it is reused only
  // until a write or the end of the block it was computed in.
  let readsSlot = ctx.jsReadsSlot || result.body.length > 0;
  ctx.jsReadsSlot = outer || readsSlot;
  ctx.memo.set(node, readsSlot ? { ...result, jsEpoch: ctx.jsEpoch } : result);
  if (node.type === "let" || node.type === "assign") ctx.jsEpoch++;
  return result;
}

/** A component of a boolean vector, which its slot holds as 1 or 0, read as `true` or `false`; a target as it is. */
function jsBooleanComponent(node: any, read: CompiledNode, ctx: CompileCtx): CompiledNode {
  if (node._t !== "bool" || ctx.jsTarget === node) return read;
  return { ...read, expr: `!!${wrapExpr(read.prec, PREC_UNARY, read.expr)}`, prec: PREC_UNARY };
}

/** `node` compiled as the target of an assignment: what it names, to be written, and not its value rounded. */
function jsCompileTarget(node: any, ctx: CompileCtx): CompiledNode {
  let saved = ctx.jsTarget;
  ctx.jsTarget = node;
  try {
    return compileJSStage(node, ctx);
  } finally {
    ctx.jsTarget = saved;
  }
}

/** The node types whose value the host passes in, which a program at `float: "f32"` rounds as it reads them. */
const JS_HOST_INPUTS = new Set(["uniform", "uniformArrayElement", "attribute", "varying", "storageElement", "var"]);

/**
 * `result`, the value of `node`, in a typed slot of its kind when it is a
 * vector or matrix the host passed in. A helper then reads only typed arrays of
 * one kind: V8 boxes each number it reads through a load that has seen arrays
 * of many kinds, as the host's plain arrays and the slots together are.
 */
function jsTypedInput(node: any, result: CompiledNode, ctx: CompileCtx): CompiledNode {
  if (node?.type === "storage" || node?.type === "uniformArray") return result;
  let t = node?._t as string | undefined;
  if (!t || !jsIsArrayType(t) || isPlainJSIdentifier(result.expr)) return result;
  if (!JS_HOST_INPUTS.has(node.type) && !result.expr.startsWith("ctx.")) return result;
  let temp = jsNewTemp(ctx, t);
  // Copied here, one line a component: this load sees only the arrays this input arrives in, so it
  // stays specialised, where one shared copy would see every kind and box what it read.
  let source = jsIsReference(result.expr) ? result.expr : null;
  let body = [...result.body];
  if (!source) {
    source = jsNewTemp(ctx, "float");
    body.push(`${source} = ${result.expr};`);
  }
  for (let i = 0; i < jsArrayLength(t); i++) body.push(`${temp}[${i}] = ${source}[${i}];`);
  return { ...result, body, expr: temp };
}

/**
 * `result`, the value of `node`, rounded to 32 bits, as `float: "f32"` asks. A
 * scalar is wrapped in `Math.fround`. A vector or matrix in a slot is rounded
 * by the slot, a Float32Array; one read from the host's context is copied into
 * a slot, so the host's own array is never written.
 */
function jsRound32(node: any, result: CompiledNode, ctx: CompileCtx): CompiledNode {
  // A whole buffer or uniform array carries the type of its element, but is no value.
  if (node?.type === "storage" || node?.type === "uniformArray") return result;
  let t = node?._t as string | undefined;
  if (t === "float") {
    if (node.type === "float") return result;
    if (node.type === "var" && !ctx.jsParams.has(node.value?.varName)) return result;
    if (/^Math\.fround\([^()]*\)$/.test(result.expr)) return result;
    return { ...result, expr: `Math.fround(${result.expr})`, prec: PREC_ATOM };
  }
  if (!t || !jsIsArrayType(t) || elementKindOf(t) !== "float") return result;
  if (/^(vec|mat)/.test(node.type) && node.type === t) return result;
  // A slot is a Float32Array, which rounds each value it stores, and a host input is copied into one.
  if (isPlainJSIdentifier(result.expr)) return result;
  jsRequireHelper(ctx, "fr");
  return { ...result, expr: `_fr(${result.expr})`, prec: PREC_ATOM };
}

/**
 * Compile a block that may run any number of times, or not at all: a value
 * computed into a slot on one side of it is not reused on the other.
 */
function compileJSBoundary(node: any, ctx: CompileCtx): CompiledNode {
  ctx.jsEpoch++;
  let result = compileJSStage(node, ctx);
  ctx.jsEpoch++;
  return result;
}

/** A negative literal is a negation, and brackets like one: `-(-7)`, not `--7`. */
function jsLiteralPrec(value: number): number | undefined {
  return value < 0 || Object.is(value, -0) ? PREC_UNARY : undefined;
}

export function compileJSNode(
  node: BaseNode<ShaderType> | ShaderType extends never ? never : any,
  ctx: CompileCtx,
): CompiledNode {
  let folded = tryFold(node, ctx.jsFloat32);
  if (folded) node = folded;

  switch (node.type) {
    case "float": {
      let value = ctx.jsFloat32 ? Math.fround(node.value as number) : (node.value as number);
      return { decls: [], body: [], expr: String(value), prec: jsLiteralPrec(value) };
    }
    case "int":
    case "uint":
      return { decls: [], body: [], expr: String(node.value), prec: jsLiteralPrec(node.value as number) };
    case "bool":
      return { decls: [], body: [], expr: node.value ? "true" : "false" };
    case "vec2":
    case "vec3":
    case "vec4":
    case "ivec2":
    case "ivec3":
    case "ivec4":
    case "uvec2":
    case "uvec3":
    case "uvec4":
    case "bvec2":
    case "bvec3":
    case "bvec4":
    case "mat2":
    case "mat2x3":
    case "mat2x4":
    case "mat3x2":
    case "mat3":
    case "mat3x4":
    case "mat4x2":
    case "mat4x3":
    case "mat4": {
      let values = node.value as number[];
      if (ctx.jsFloat32 && elementKindOf(node._t) === "float") values = values.map(Math.fround);
      if (ctx.outTarget) {
        let lines = values.map((v, i) => `${ctx.outTarget}[${i}] = ${JSON.stringify(v)};`);
        return { decls: [], body: lines, expr: ctx.outTarget };
      }
      // A constant is a typed array of its kind, as a slot is, so a helper reads one kind of array.
      let typed = typedArrayOfKind(elementKindOf(node._t), ctx.jsFloat32 === true).name;
      let literal = `new ${typed}([${values.map((v) => (typeof v === "boolean" ? (v ? 1 : 0) : JSON.stringify(v))).join(", ")}])`;
      return { decls: [], body: [], expr: jsConstant(ctx, literal) };
    }
    case "void":
      return { decls: [], body: [], expr: "0" };

    case "construct": {
      let targetType = node._t as string;
      // Scalar conversions (float/int/uint/bool casts).
      if ((TYPE_WIDTH[targetType] ?? 0) === 1) {
        let source = node.params?.[0] as BaseNode<ShaderType> | undefined;
        // A scalar constructor given a vector reads its first component —
        // `float(v)` is `v.x` — rather than refusing it.
        if ((TYPE_WIDTH[source?._t as string] ?? 1) > 1) {
          let c = jsCompileOperand(source, ctx);
          return {
            decls: c.decls,
            body: c.body,
            expr: jsComponentCast(`${c.expr}[0]`, source?._t, targetType),
          };
        }
        let p = compileJSStage(source, ctx);
        return {
          decls: p.decls,
          body: p.body,
          expr: jsComponentCast(p.expr, source?._t, targetType),
        };
      }
      let width = TYPE_WIDTH[targetType];
      if (width !== undefined) {
        let params = node.params ?? [];
        // GLSL and WGSL broadcast a lone scalar operand across every component
        // — vec3(2.0) is (2.0, 2.0, 2.0) — so the JS backend must too. Multiple
        // operands instead fill components in order, zero-filling the rest.
        if (params.length === 1 && (TYPE_WIDTH[params[0]?._t] ?? 1) <= 1) {
          let c = jsReadable(jsCompileOperand(params[0], ctx), params[0]?._t, ctx);
          let broadcast = jsComponentCast(c.expr, params[0]?._t, targetType);
          let target = ctx.outTarget ?? jsNewTemp(ctx, targetType);
          let writes = Array.from({ length: width }, (_, i) => `${target}[${i}] = ${broadcast};`);
          return { decls: c.decls, body: [...c.body, ...writes], expr: target };
        }
        // Vector construct: expand every operand's components into one array.
        let compiled = params.map((p: BaseNode<ShaderType>) => {
          let w = TYPE_WIDTH[p?._t] ?? 1;
          let c = jsCompileOperand(p, ctx);
          return { c: w > 1 ? jsReadable(c, p?._t, ctx) : c, w, t: p?._t as string | undefined };
        });
        let pieces: string[] = [];
        let decls: string[] = [];
        let body: string[] = [];
        for (let { c, w, t } of compiled) {
          decls.push(...c.decls);
          body.push(...c.body);
          if (w <= 1) pieces.push(jsComponentCast(c.expr, t, targetType));
          else for (let i = 0; i < w; i++) pieces.push(jsComponentCast(`${c.expr}[${i}]`, t, targetType));
        }
        while (pieces.length < width) pieces.push("0");
        pieces = pieces.slice(0, width);
        let target = ctx.outTarget ?? jsNewTemp(ctx, targetType);
        let writes = pieces.map((piece, i) => `${target}[${i}] = ${piece};`);
        return { decls, body: [...body, ...writes], expr: target };
      }
      let shape = MATRIX_DIMENSIONS[targetType];
      if (shape) {
        let [cols, rows] = shape;
        let size = cols * rows;
        if ((node.params ?? []).length === 1) {
          let src = node.params![0];
          if (MATRIX_DIMENSIONS[src?._t] !== undefined) {
            // A matrix source: copy (or truncate/extend through the same shape).
            let c = compileJSStage(src, ctx);
            let target = ctx.outTarget ?? jsNewTemp(ctx, targetType);
            jsRequireHelper(ctx, "copy");
            return { decls: c.decls, body: [...c.body, `_copy(${c.expr}, ${target});`], expr: target };
          }
          // A scalar source: the diagonal. Zero the whole slot first — it is a
          // hoisted slot and could carry stale off-diagonal values from a
          // previous call.
          let s = compileJSStage(src, ctx);
          let target = ctx.outTarget ?? jsNewTemp(ctx, targetType);
          let zeroAll = Array(size)
            .fill(0)
            .map((_, i) => `${target}[${i}] = 0;`);
          let diag: string[] = [];
          for (let col = 0; col < cols; col++)
            for (let row = 0; row < rows; row++) {
              if (col === row) diag.push(`${target}[${col * rows + row}] = ${s.expr};`);
            }
          return { decls: s.decls, body: [...s.body, ...zeroAll, ...diag], expr: target };
        }
        // Column-wise construction: each param is one column vector.
        let compiled = (node.params ?? []).map((p: BaseNode<ShaderType>) =>
          jsReadable(jsCompileOperand(p, ctx), p?._t, ctx),
        );
        let decls = compiled.flatMap((c: CompiledNode) => c.decls);
        let body = compiled.flatMap((c: CompiledNode) => c.body);
        // Column-major flat layout: column 0's components first, then column 1.
        let pieces: string[] = [];
        for (let col = 0; col < cols; col++)
          for (let row = 0; row < rows; row++) {
            pieces.push(`${compiled[col].expr}[${row}]`);
          }
        if (ctx.outTarget) {
          let writes = pieces.map((piece, i) => `${ctx.outTarget}[${i}] = ${piece};`);
          return { decls, body: [...body, ...writes], expr: ctx.outTarget };
        }
        return { decls, body, expr: `[${pieces.join(", ")}]` };
      }
      throw new Error(`[RMSL] Unsupported construct target in JS compiler: "${targetType}"`);
    }

    case "var": {
      let varInfo = node.value as any;
      let varName = varInfo?.varName;
      if (ctx.jsParams.has(varName)) {
        return { decls: [], body: [], expr: `ctx.params[${JSON.stringify(varName)}]` };
      }
      return jsLeafRef(varName, node._t, ctx);
    }

    case "uniform":
    case "uniformArray": {
      let v = node.value as any;
      return jsLeafRef(`ctx.uniforms[${JSON.stringify(v.slot)}]`, v.shaderType ?? node._t, ctx);
    }

    case "uniformArrayElement": {
      let arr = jsCompileOperand(node.params![0], ctx);
      let idx = jsCompileOperand(node.params![1], ctx);
      let element = `${arr.expr}[${idx.expr}]`;
      if (ctx.outTarget && jsIsArrayType(node._t)) {
        jsRequireHelper(ctx, "copy");
        return {
          decls: [...arr.decls, ...idx.decls],
          body: [...arr.body, ...idx.body, `_copy(${element}, ${ctx.outTarget});`],
          expr: ctx.outTarget,
        };
      }
      return { decls: [...arr.decls, ...idx.decls], body: [...arr.body, ...idx.body], expr: element };
    }

    case "storage": {
      let v = node.value as any;
      ctx.storageTypes ??= new Map();
      ctx.storageTypes.set(v.slot, v.shaderType ?? node._t);
      return jsLeafRef(`ctx.storages[${JSON.stringify(v.slot)}]`, v.shaderType ?? node._t, ctx);
    }

    case "storageElement": {
      if (jsIsArrayType(node._t)) {
        let element = jsStorageElement(node, ctx);
        let target = ctx.outTarget ?? jsNewTemp(ctx, node._t);
        jsRequireHelper(ctx, "load");
        return {
          decls: element.decls,
          body: [...element.body, `_load(${element.buffer}, ${element.start}, ${target});`],
          expr: target,
        };
      }
      let arr = jsCompileOperand(node.params![0], ctx);
      let idx = jsCompileOperand(node.params![1], ctx);
      return {
        decls: [...arr.decls, ...idx.decls],
        body: [...arr.body, ...idx.body],
        expr: `${arr.expr}[${jsStorageIndex(node, idx.expr)}]`,
      };
    }

    case "invocationIndex": {
      return { decls: [], body: [], expr: "ctx.index" };
    }

    case "attribute": {
      let v = node.value as any;
      assertNotInAComputeStage(ctx.shaderStage, COMPUTE_REFUSES.attribute);
      return jsLeafRef(`ctx.attributes[${JSON.stringify(v.slot)}]`, v.shaderType ?? node._t, ctx);
    }

    case "varying": {
      let v = node.value as any;
      assertNotInAComputeStage(ctx.shaderStage, COMPUTE_REFUSES.varying);
      let slot = v?.slot;
      // In a vertex stage a varying is an output, collected in the result so
      // the host can read it back; in a fragment stage it is an input.
      if (ctx.shaderStage === "vertex") {
        ctx.jsNeedsRes = true;
        ctx.varyings.set(slot, { id: 0, type: v.shaderType ?? node._t, slot });
        return jsLeafRef(`res.varyings[${JSON.stringify(slot)}]`, v.shaderType ?? node._t, ctx);
      }
      return jsLeafRef(`ctx.varyings[${JSON.stringify(slot)}]`, v.shaderType ?? node._t, ctx);
    }

    case "output": {
      let v = node.value as any;
      ctx.jsNeedsRes = true;
      if (v.id != null) ctx.outputs.set(v.id, { type: v.shaderType, slot: v.slot, location: v.location });
      return jsLeafRef(`res.outputs[${JSON.stringify(v.slot)}]`, v.shaderType ?? node._t, ctx);
    }

    case "builtinPosition": {
      assertPositionIsReadable(ctx);
      ctx.jsNeedsRes = true;
      return jsLeafRef("res.position", "vec4", ctx);
    }

    case "builtinFragDepth": {
      if (ctx.shaderStage !== "fragment") {
        throw new Error("builtinFragDepth() can only be used in fragment shaders");
      }
      ctx.jsNeedsRes = true;
      return { decls: [], body: [], expr: "res.fragDepth" };
    }

    case "fragCoord": {
      if (ctx.shaderStage !== "fragment") {
        throw new Error("fragCoord() can only be used in fragment shaders");
      }
      // The CPU target has no framebuffer; the caller passes the pixel being
      // evaluated as ctx.fragCoord, defaulting to the origin.
      return { decls: [], body: [], expr: "(ctx.fragCoord || [0, 0])" };
    }

    case "swizzle": {
      let src = jsComponents(node.params![0], ctx);
      let pattern = node.value as string;
      if (pattern.length === 1) {
        let read = { decls: src.decls, body: src.body, expr: src.at(`${JS_COMPONENT_INDEX[pattern]}`) };
        return jsBooleanComponent(node, read, ctx);
      }
      let idx = [...pattern].map((ch) => JS_COMPONENT_INDEX[ch]);
      if (ctx.outTarget) {
        let reads = idx.map((j) => src.at(`${j}`));
        let body = [...src.body];
        // A swizzle of the target itself reads every component before it writes one.
        if (reads.some((read) => read.startsWith(`${ctx.outTarget}[`))) {
          reads = reads.map((read) => {
            let local = jsNewTemp(ctx, elementKindOf(node._t));
            body.push(`${local} = ${read};`);
            return local;
          });
        }
        let lines = reads.map((read, i) => `${ctx.outTarget}[${i}] = ${read};`);
        return { decls: src.decls, body: [...body, ...lines], expr: ctx.outTarget };
      }
      return { decls: src.decls, body: src.body, expr: `[${idx.map((j) => src.at(`${j}`)).join(", ")}]` };
    }

    case "negate": {
      let integer = jsIntegerOp(node, "neg") === "ineg";
      if (jsArrayLength(node.params![0]?._t) <= 1) {
        let a = compileJSStage(node.params![0], ctx);
        if (integer) return { decls: a.decls, body: a.body, ...jsIntegerForm("ineg", [a]) };
        return { decls: a.decls, body: a.body, expr: `-${wrapExpr(a.prec, PREC_UNARY, a.expr)}`, prec: PREC_UNARY };
      }
      return jsUnaryMath(node, ctx, integer ? "ineg" : "neg");
    }

    case "not": {
      let width = jsArrayLength(node.params![0]?._t);
      if (width <= 1) {
        let a = compileJSStage(node.params![0], ctx);
        return { decls: a.decls, body: a.body, expr: `!${wrapExpr(a.prec, PREC_UNARY, a.expr)}`, prec: PREC_UNARY };
      }
      jsRequireHelper(ctx, `b${width}not`);
      let a = jsCompileOperand(node.params![0], ctx);
      if (ctx.outTarget) {
        return {
          decls: a.decls,
          body: [...a.body, `_b${width}not(${a.expr}, ${ctx.outTarget});`],
          expr: ctx.outTarget,
        };
      }
      return { decls: a.decls, body: a.body, expr: `_b${width}not(${a.expr})` };
    }

    case "all":
      return jsVecReduce(node, ctx, "ball");
    case "any":
      return jsVecReduce(node, ctx, "bany");

    case "add":
      return jsBinaryOp(node, ctx, jsIntegerOp(node, "add"));
    case "sub":
      return jsBinaryOp(node, ctx, jsIntegerOp(node, "sub"));
    case "mul": {
      let aType = node.params![0]?._t;
      let bType = node.params![1]?._t;
      let aIsMat = MATRIX_DIMENSIONS[aType] !== undefined;
      let bIsMat = MATRIX_DIMENSIONS[bType] !== undefined;
      if (aIsMat && bIsMat) return jsMatMul(node, ctx);
      if (aIsMat || bIsMat) {
        // Matrix times scalar scales every element.
        return jsVectorBinary(node, ctx, "mul", jsArrayLength(aIsMat ? aType : bType));
      }
      return jsBinaryOp(node, ctx, jsIntegerOp(node, "mul"));
    }
    case "div":
      return jsBinaryOp(node, ctx, jsIntegerOp(node, "div"));
    case "mod":
      return jsBinaryOp(node, ctx, jsIntegerOp(node, "mod"));
    case "pow":
      return jsBinaryOp(node, ctx, "pow");
    case "atan2":
      return jsBinaryOp(node, ctx, "atan2");
    case "min":
      return jsBinaryOp(node, ctx, "min");
    case "max":
      return jsBinaryOp(node, ctx, "max");
    case "dot":
      return jsVecReduce(node, ctx, "vdot");
    case "cross":
      return jsVecOutOp(node, ctx, "cross");
    case "distance":
      return jsVecReduce(node, ctx, "vdist");
    case "reflect":
      return jsVecOutOp(node, ctx, "reflect");
    case "refract":
      return jsVecOutOp(node, ctx, "refract");
    case "mix":
      return jsBinaryOp(node, ctx, "mix");
    case "step":
      return jsBinaryOp(node, ctx, "step");
    case "smoothstep":
      return jsBinaryOp(node, ctx, "smoothstep");
    case "clamp":
      return jsBinaryOp(node, ctx, "clamp");
    case "select": {
      let cond = jsCompileOperand(node.params![0], ctx);
      let a = jsCompileOperand(node.params![1], ctx);
      let b = jsCompileOperand(node.params![2], ctx);
      let condType = (node.params![0] as any)?._t || "bool";
      // A scalar condition is a plain ternary; a boolean vector selects per
      // component, which JS has no operator for, so a helper walks the arrays.
      if (condType !== "bool") {
        jsRequireHelper(ctx, "bselect");
        // The helper writes into a slot, the assignment's own or a hoisted one, so a call allocates nothing.
        let target = ctx.outTarget ?? jsNewTemp(ctx, node._t);
        return {
          decls: [...cond.decls, ...a.decls, ...b.decls],
          body: [...cond.body, ...a.body, ...b.body, `_bselect(${cond.expr}, ${a.expr}, ${b.expr}, ${target});`],
          expr: target,
        };
      }
      return {
        decls: [...cond.decls, ...a.decls, ...b.decls],
        body: [...cond.body, ...a.body, ...b.body],
        expr: `(${cond.expr} ? ${a.expr} : ${b.expr})`,
      };
    }
    case "faceForward":
      return jsVecOutOp(node, ctx, "faceforward");

    case "lessThan":
      return jsComparison(node, ctx, "<");
    case "greaterThan":
      return jsComparison(node, ctx, ">");
    case "lessThanEqual":
      return jsComparison(node, ctx, "<=");
    case "greaterThanEqual":
      return jsComparison(node, ctx, ">=");
    case "equal":
      return jsComparison(node, ctx, "===");
    case "notEqual":
      return jsComparison(node, ctx, "!==");

    case "and": {
      let width = jsArrayLength(node.params![0]?._t);
      if (width <= 1) {
        let a = compileJSStage(node.params![0], ctx);
        let b = compileJSStage(node.params![1], ctx);
        let prec = PRECEDENCE[node.type] ?? 0;
        return {
          decls: [...a.decls, ...b.decls],
          body: [...a.body, ...b.body],
          expr: `${wrapExpr(a.prec, prec, a.expr)} && ${wrapExpr(b.prec, prec, b.expr)}`,
          prec,
        };
      }
      jsRequireHelper(ctx, `b${width}and`);
      let a = jsCompileOperand(node.params![0], ctx);
      let b = jsCompileOperand(node.params![1], ctx);
      if (ctx.outTarget) {
        return {
          decls: [...a.decls, ...b.decls],
          body: [...a.body, ...b.body, `_b${width}and(${a.expr}, ${b.expr}, ${ctx.outTarget});`],
          expr: ctx.outTarget,
        };
      }
      return {
        decls: [...a.decls, ...b.decls],
        body: [...a.body, ...b.body],
        expr: `_b${width}and(${a.expr}, ${b.expr})`,
      };
    }
    case "or": {
      let width = jsArrayLength(node.params![0]?._t);
      if (width <= 1) {
        let a = compileJSStage(node.params![0], ctx);
        let b = compileJSStage(node.params![1], ctx);
        let prec = PRECEDENCE[node.type] ?? 0;
        return {
          decls: [...a.decls, ...b.decls],
          body: [...a.body, ...b.body],
          expr: `${wrapExpr(a.prec, prec, a.expr)} || ${wrapExpr(b.prec, prec, b.expr)}`,
          prec,
        };
      }
      jsRequireHelper(ctx, `b${width}or`);
      let a = jsCompileOperand(node.params![0], ctx);
      let b = jsCompileOperand(node.params![1], ctx);
      if (ctx.outTarget) {
        return {
          decls: [...a.decls, ...b.decls],
          body: [...a.body, ...b.body, `_b${width}or(${a.expr}, ${b.expr}, ${ctx.outTarget});`],
          expr: ctx.outTarget,
        };
      }
      return {
        decls: [...a.decls, ...b.decls],
        body: [...a.body, ...b.body],
        expr: `_b${width}or(${a.expr}, ${b.expr})`,
      };
    }

    case "bitAnd":
      return jsBitwise(node, ctx, "&");
    case "bitOr":
      return jsBitwise(node, ctx, "|");
    case "bitXor":
      return jsBitwise(node, ctx, "^");
    case "shiftLeft":
      return jsBitwise(node, ctx, "<<");
    case "shiftRight":
      return jsBitwise(node, ctx, ">>");
    case "bitNot":
      return jsBitwise(node, ctx, "~");

    case "matVecMul": {
      let aType = node.params![0]?._t;
      let bType = node.params![1]?._t;
      let [c, r] = MATRIX_DIMENSIONS[aType];
      let vlen = TYPE_WIDTH[bType] ?? c;
      let mat = jsCompileOperand(node.params![0], ctx);
      let vec = jsCompileOperand(node.params![1], ctx);
      let name = `mat${c}x${r}mv${vlen}`;
      jsRequireHelper(ctx, name);
      if (ctx.outTarget) {
        return {
          decls: [...mat.decls, ...vec.decls],
          body: [...mat.body, ...vec.body, `_${name}(${mat.expr}, ${vec.expr}, ${ctx.outTarget});`],
          expr: ctx.outTarget,
        };
      }
      return {
        decls: [...mat.decls, ...vec.decls],
        body: [...mat.body, ...vec.body],
        expr: `_${name}(${mat.expr}, ${vec.expr})`,
      };
    }

    case "sin":
      return jsUnaryMath(node, ctx, "sin");
    case "cos":
      return jsUnaryMath(node, ctx, "cos");
    case "tan":
      return jsUnaryMath(node, ctx, "tan");
    case "asin":
      return jsUnaryMath(node, ctx, "asin");
    case "acos":
      return jsUnaryMath(node, ctx, "acos");
    case "atan":
      return jsUnaryMath(node, ctx, "atan");
    case "sinh":
      return jsUnaryMath(node, ctx, "sinh");
    case "cosh":
      return jsUnaryMath(node, ctx, "cosh");
    case "tanh":
      return jsUnaryMath(node, ctx, "tanh");
    case "asinh":
      return jsUnaryMath(node, ctx, "asinh");
    case "acosh":
      return jsUnaryMath(node, ctx, "acosh");
    case "atanh":
      return jsUnaryMath(node, ctx, "atanh");
    case "abs":
      return jsUnaryMath(node, ctx, jsIntegerOp(node, "abs") === "iabs" ? "iabs" : "abs");
    case "sign":
      return jsUnaryMath(node, ctx, "sign");
    case "floor":
      return jsUnaryMath(node, ctx, "floor");
    case "ceil":
      return jsUnaryMath(node, ctx, "ceil");
    case "fract":
      return jsUnaryMath(node, ctx, "fract");
    case "round":
      return jsUnaryMath(node, ctx, "round");
    case "trunc":
      return jsUnaryMath(node, ctx, "trunc");
    case "radians":
      return jsUnaryMath(node, ctx, "radians");
    case "degrees":
      return jsUnaryMath(node, ctx, "degrees");
    case "sqrt":
      return jsUnaryMath(node, ctx, "sqrt");
    case "inverseSqrt":
      return jsUnaryMath(node, ctx, "rsqrt");
    case "exp":
      return jsUnaryMath(node, ctx, "exp");
    case "log":
      return jsUnaryMath(node, ctx, "log");
    case "exp2":
      return jsUnaryMath(node, ctx, "exp2");
    case "log2":
      return jsUnaryMath(node, ctx, "log2");
    case "normalize":
      return jsVecOutOp(node, ctx, "norm");
    case "length":
      return jsVecReduce(node, ctx, "vlen");
    case "transpose":
      return jsMatrixUnary(node, ctx, "T");
    case "inverse":
      assertSquareMatrix(node.params![0]?._t);
      return jsMatrixUnary(node, ctx, "inv");
    case "determinant": {
      let brand = node.params![0]?._t;
      let [c, r] = MATRIX_DIMENSIONS[brand];
      let name = `mat${c}x${r}det`;
      jsRequireHelper(ctx, name);
      let a = compileJSStage(node.params![0], ctx);
      return { decls: a.decls, body: a.body, expr: `_${name}(${a.expr})` };
    }

    case "fwidth":
    case "dFdx":
    case "dFdy": {
      if (ctx.derivatives === "zero") {
        let width = jsArrayLength(node.params![0]?._t);
        if (width > 1) {
          if (ctx.outTarget) {
            let lines = Array.from({ length: width }, (_, i) => `${ctx.outTarget}[${i}] = 0;`);
            return { decls: [], body: lines, expr: ctx.outTarget };
          }
          return { decls: [], body: [], expr: `[${jsZeroes(width)}]` };
        }
        return { decls: [], body: [], expr: "0" };
      }
      throw new Error(
        `[RMSL] ${node.type} has no meaning on the CPU target. Compile with ` +
          `{ derivatives: "zero" } to evaluate it as 0.`,
      );
    }

    case "matrixElement": {
      assertLiteralIndexInRange(node.params![0], node.params![1]);
      let mat = jsComponents(node.params![0], ctx);
      let idx = jsCompileOperand(node.params![1], ctx);
      let brand = node.params![0]?._t;
      let [columns, rows] = MATRIX_DIMENSIONS[brand];
      let column = mat.inBuffer ? jsBoundedIndex(idx.expr, columns) : `(${idx.expr})`;
      let target = ctx.outTarget ?? jsNewTemp(ctx, node._t);
      let lines = Array.from(
        { length: rows },
        (_, row) => `${target}[${row}] = ${mat.at(`${column} * ${rows} + ${row}`)};`,
      );
      return { decls: [...mat.decls, ...idx.decls], body: [...mat.body, ...idx.body, ...lines], expr: target };
    }

    case "vectorElement": {
      assertLiteralIndexInRange(node.params![0], node.params![1]);
      let src = jsComponents(node.params![0], ctx);
      let idx = jsCompileOperand(node.params![1], ctx);
      let component = src.inBuffer ? jsBoundedIndex(idx.expr, TYPE_WIDTH[node.params![0]._t]) : idx.expr;
      let read = { decls: [...src.decls, ...idx.decls], body: [...src.body, ...idx.body], expr: src.at(component) };
      return jsBooleanComponent(node, read, ctx);
    }

    case "texture":
    case "textureLod": {
      let samplerNode = node.params![0];
      let samplerType = samplerNode?._t || "sampler2D";
      let slot = (samplerNode.value as any)?.slot;
      let isInteger = samplerType.startsWith("isampler") || samplerType.startsWith("usampler");
      let is3D = samplerType.endsWith("3D");
      let isCube = samplerType.endsWith("Cube");
      if (isCube && isInteger) {
        throw new Error("[RMSL] The JS target supports samplerCube, not isamplerCube/usamplerCube, yet.");
      }
      if (!samplerType.endsWith("2D") && !is3D && !isCube) {
        throw new Error("[RMSL] The JS target supports sampler2D/sampler3D/samplerCube textures only.");
      }
      let texRef = `ctx.textures[${JSON.stringify(slot)}]`;
      let coords = jsCompileOperand(node.params![1], ctx);
      let helper = isCube
        ? "texCube"
        : is3D
          ? isInteger
            ? "texFetch3d"
            : "tex3d"
          : isInteger
            ? "texFetch2d"
            : "tex2d";
      jsRequireHelper(ctx, helper);
      jsRequireHelper(ctx, "chan");
      jsRequireHelper(ctx, "texel");
      if (isCube) jsRequireHelper(ctx, "cubeFace");
      if (!isInteger) {
        jsRequireHelper(ctx, "unorm");
        jsRequireHelper(ctx, "wrap");
        if (is3D) jsRequireHelper(ctx, "lerp2");
      }
      if (ctx.outTarget) {
        return {
          decls: coords.decls,
          body: [...coords.body, `_${helper}(${texRef}, ${coords.expr}, ${ctx.outTarget});`],
          expr: ctx.outTarget,
        };
      }
      return { decls: coords.decls, body: coords.body, expr: `_${helper}(${texRef}, ${coords.expr})` };
    }

    case "textureLoad": {
      let samplerNode = node.params![0];
      let samplerType = samplerNode?._t || "sampler2D";
      let slot = (samplerNode.value as any)?.slot;
      if (!samplerType.endsWith("2D") && !samplerType.endsWith("3D")) {
        throw new Error("[RMSL] The JS target supports sampler2D/sampler3D textures only.");
      }
      let is3D = samplerType.endsWith("3D");
      // A texel fetch through a float sampler still reads a normalized texture,
      // as `texelFetch`/`textureLoad` do on either backend; an integer sampler
      // has no normalization to undo.
      let isInteger = samplerType.startsWith("isampler") || samplerType.startsWith("usampler");
      let helper = isInteger ? (is3D ? "texFetch3d" : "texFetch2d") : is3D ? "texFetchUnorm3d" : "texFetchUnorm2d";
      jsRequireHelper(ctx, helper);
      jsRequireHelper(ctx, "chan");
      jsRequireHelper(ctx, "texel");
      if (!isInteger) jsRequireHelper(ctx, "unorm");
      let texRef = `ctx.textures[${JSON.stringify(slot)}]`;
      let coords = jsCompileOperand(node.params![1], ctx);
      if (ctx.outTarget) {
        return {
          decls: coords.decls,
          body: [...coords.body, `_${helper}(${texRef}, ${coords.expr}, ${ctx.outTarget});`],
          expr: ctx.outTarget,
        };
      }
      return { decls: coords.decls, body: coords.body, expr: `_${helper}(${texRef}, ${coords.expr})` };
    }

    case "textureSize": {
      let samplerNode = node.params![0];
      let slot = (samplerNode.value as any)?.slot;
      let is3D = (samplerNode as any)?._t?.endsWith("3D");
      jsRequireHelper(ctx, "texSize");
      let texRef = `ctx.textures[${JSON.stringify(slot)}]`;
      if (ctx.outTarget) {
        return {
          decls: [],
          body: [`_texSize(${texRef}, ${ctx.outTarget});`],
          expr: ctx.outTarget,
        };
      }
      return { decls: [], body: [], expr: `_texSize(${texRef})` };
    }

    case "let": {
      let lhsNode = node.params![0];
      let varName = (lhsNode.value as any)?.varName || (lhsNode as any)?.name;
      ctx.varDefs.set(varName, lhsNode._t);
      let rhsNode = node.params![1];
      if (jsIsArrayType(rhsNode?._t)) {
        let saved = ctx.outTarget;
        ctx.outTarget = varName;
        let rhs = compileJSStage(rhsNode, ctx);
        ctx.outTarget = saved;
        if (rhs.expr !== varName) {
          jsRequireHelper(ctx, "copy");
          return { decls: rhs.decls, body: [...rhs.body, `_copy(${rhs.expr}, ${varName});`], expr: varName };
        }
        return { decls: rhs.decls, body: rhs.body, expr: varName };
      }
      let rhs = compileJSStage(rhsNode, ctx);
      return { decls: rhs.decls, body: [...rhs.body, `${varName} = ${rhs.expr};`], expr: varName };
    }

    case "assign": {
      let targetNode = node.params![0];
      if (targetNode?.type === "builtinPosition") ctx.positionWritten = true;
      assertAssignable(targetNode, ctx.shaderStage);
      let rhsNode = node.params![1];

      // A swizzle, a column, or a component of a column by index: single
      // components assign directly, several split into per-component writes
      // (JS has no `v.xy = e`). A component index is compiled after the
      // column's, as `m[i][j]` evaluates.
      let parts: { base: any; components?: string[]; index?: any } | undefined;
      if (targetNode?.type === "swizzle") {
        let resolved = resolveSwizzleTarget(targetNode);
        parts = { base: resolved.base, components: [...resolved.pattern].map((ch) => `${JS_COMPONENT_INDEX[ch]}`) };
      } else if (targetNode?.type === "matrixElement") {
        let [, rows] = MATRIX_DIMENSIONS[targetNode.params![0]._t];
        parts = { base: targetNode, components: Array.from({ length: rows }, (_, row) => `${row}`) };
      } else if (
        targetNode?.type === "vectorElement" &&
        (targetNode.params![0]?.type === "matrixElement" || targetNode.params![0]?.type === "storageElement")
      ) {
        parts = { base: targetNode.params![0], index: targetNode.params![1] };
      } else if (targetNode?.type === "storageElement" && jsIsArrayType(targetNode._t)) {
        parts = {
          base: targetNode,
          components: Array.from({ length: componentCountOf(targetNode._t) }, (_, k) => `${k}`),
        };
      }
      if (parts) {
        let base = jsAssignable(parts.base, ctx);
        let components = parts.components ?? [];
        if (parts.index) {
          assertLiteralIndexInRange(parts.base, parts.index);
          let idx = compileJSStage(parts.index, ctx);
          base = { ...base, decls: [...base.decls, ...idx.decls], body: [...base.body, ...idx.body] };
          components = [jsBoundedIndex(idx.expr, TYPE_WIDTH[parts.base._t])];
        }
        if (components.length === 1) {
          let rhs = compileJSStage(rhsNode, ctx);
          return {
            decls: [...base.decls, ...rhs.decls],
            body: [...base.body, ...rhs.body, `${base.at(components[0]!)} = ${rhs.expr};`],
            expr: base.expr,
          };
        }
        let temp = jsNewTemp(ctx, rhsNode?._t || "float");
        let saved = ctx.outTarget;
        ctx.outTarget = temp;
        let rhs = compileJSStage(rhsNode, ctx);
        ctx.outTarget = saved;
        if (rhs.expr !== temp) jsRequireHelper(ctx, "copy");
        let fill = rhs.expr === temp ? [] : [`_copy(${rhs.expr}, ${temp});`];
        let writes = components.map((k, i) => `${base.at(k)} = ${temp}[${i}];`);
        return {
          decls: [...base.decls, ...rhs.decls],
          body: [...base.body, ...rhs.body, ...fill, ...writes],
          expr: base.expr,
        };
      }

      let lhs = jsCompileTarget(targetNode, ctx);
      // Only a plain variable slot is written through out-mode helpers; an
      // external sink (res.position, res.outputs[...], ctx.varyings[...]) takes
      // the whole value in one assignment.
      if (jsIsArrayType(rhsNode?._t) && isPlainJSIdentifier(lhs.expr)) {
        let saved = ctx.outTarget;
        ctx.outTarget = lhs.expr;
        let rhs = compileJSStage(rhsNode, ctx);
        ctx.outTarget = saved;
        if (rhs.expr !== lhs.expr) {
          jsRequireHelper(ctx, "copy");
          return {
            decls: [...lhs.decls, ...rhs.decls],
            body: [...lhs.body, ...rhs.body, `_copy(${rhs.expr}, ${lhs.expr});`],
            expr: lhs.expr,
          };
        }
        return { decls: [...lhs.decls, ...rhs.decls], body: [...lhs.body, ...rhs.body], expr: lhs.expr };
      }
      if (jsIsArrayType(rhsNode?._t)) {
        // The sink takes a slot of its own, so a write through it leaves what it was assigned from as it was.
        let slot = jsNewTemp(ctx, rhsNode._t);
        let saved = ctx.outTarget;
        ctx.outTarget = slot;
        let rhs = compileJSStage(rhsNode, ctx);
        ctx.outTarget = saved;
        if (rhs.expr !== slot) jsRequireHelper(ctx, "copy");
        let fill = rhs.expr === slot ? [] : [`_copy(${rhs.expr}, ${slot});`];
        return {
          decls: [...lhs.decls, ...rhs.decls],
          body: [...lhs.body, ...rhs.body, ...fill, `${lhs.expr} = ${slot};`],
          expr: lhs.expr,
        };
      }
      let rhs = compileJSStage(rhsNode, ctx);
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
      let expr = "0";
      let saved = ctx.outTarget;
      for (let [i, p] of params.entries()) {
        // Only the value goes to the target; a statement before it has its own.
        ctx.outTarget = i === params.length - 1 ? saved : null;
        let r = compileJSStage(p, ctx);
        ctx.outTarget = saved;
        allDecls.push(...r.decls);
        allBody.push(...r.body);
        expr = r.expr;
      }
      return { decls: allDecls, body: allBody, expr };
    }

    case "if": {
      let cond = compileJSStage(node.params![0], ctx);
      let body = compileJSBoundary(node.params![1], ctx);
      let elseBody =
        node.params!.length >= 3 && node.params![2] !== undefined
          ? compileJSBoundary(node.params![2], ctx)
          : { decls: [] as string[], body: [] as string[], expr: "0" };
      let lines: string[] = [...cond.body, `if (${cond.expr}) {`, ...body.body.map((l) => "  " + l), "}"];
      if (elseBody.body.length > 0) {
        lines.push("else {");
        lines.push(...elseBody.body.map((l) => "  " + l));
        lines.push("}");
      }
      return {
        decls: [...cond.decls, ...body.decls, ...elseBody.decls],
        body: lines,
        expr: "0",
      };
    }

    case "for": {
      let init = compileJSStage(node.params![0], ctx);
      let cond = compileJSStage(node.params![1], ctx);
      let update = compileJSBoundary(node.params![2], ctx);
      let body = compileJSBoundary(node.params![3], ctx);
      // An init that makes no statement, such as a variable made before the loop, leaves the header's init empty.
      let initExpr = "";
      let initBody = init.body;
      if (init.body.length > 0) {
        let lastStmt = init.body[init.body.length - 1];
        if (lastStmt.endsWith(";")) {
          initExpr = lastStmt.slice(0, -1);
          initBody = init.body.slice(0, -1);
        }
      }
      // The update stays in the header, so a continue still runs it.
      let { header, guard } = loopTest(cond);
      return {
        decls: [...init.decls, ...cond.decls, ...update.decls, ...body.decls],
        body: [
          ...initBody,
          `for (${initExpr}; ${header}; ${forUpdateStatements(update).map(withoutSemicolon).join(", ")}) {`,
          ...[...guard, ...body.body].map((l) => "  " + l),
          "}",
        ],
        expr: "0",
      };
    }

    case "while": {
      let cond = compileJSStage(node.params![0], ctx);
      let body = compileJSBoundary(node.params![1], ctx);
      let { header, guard } = loopTest(cond);
      return {
        decls: [...cond.decls, ...body.decls],
        body: [`while (${header}) {`, ...[...guard, ...body.body].map((l) => "  " + l), "}"],
        expr: "0",
      };
    }

    case "discard": {
      return { decls: [], body: ["return null;"], expr: "0" };
    }

    case "break": {
      return { decls: [], body: ["break;"], expr: "0" };
    }

    case "continue": {
      return { decls: [], body: ["continue;"], expr: "0" };
    }

    case "return": {
      return { decls: [], body: ["return;"], expr: "0" };
    }

    default:
      throw new Error(`[RMSL] Unsupported node type in JS compiler: "${node.type}"`);
  }
}

export type CompileJSOptions = CompileFnOptions & {
  stage?: "vertex" | "fragment" | "compute";
  derivatives?: "throw" | "zero";
  reentrant?: boolean;
  /**
   * The width the program computes a `float` in: `"f64"`, the default, or
   * `"f32"`, which rounds every float value to 32 bits as a GPU holds it.
   */
  float?: "f64" | "f32";
};

/**
 * Compile an Fn to JavaScript: a self-contained expression that evaluates to
 * the callable. The expression is the scratch slots and helper functions in a
 * closure, then `return function <name>(ctx) { ... }`, so a caller evaluates
 * it with `new Function(source)()` or embeds it and assigns the result.
 */
/**
 * `compileJSFn`'s real body, also handing back the root node's result type —
 * needed by `compileJSProgram`'s `draw()` and computed here, from the one time `fn`
 * is actually called. A second call to read it back afterward is not an
 * option: `fn` routinely has side effects on the caller's own closure (the
 * `let tex; Fn(() => { tex = uniform(...); ... })` idiom this whole test
 * suite uses), so calling it twice leaves the caller's own reference
 * pointing at a second, different uniform than the one actually compiled in.
 */
function compileJSFnDetailed(
  fn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  options: CompileJSOptions,
): {
  source: string;
  resultType: ShaderType | undefined;
  storageTypes: Record<string, ShaderType>;
  resultTypes: JsResultTypes;
} {
  let stage = options.stage ?? "fragment";
  let derivatives = options.derivatives ?? "throw";
  let reentrant = options.reentrant ?? false;
  const paramNodes = options.params.map((p) => parameterNode(p.name, p.type));
  const rawResult = fn(...paramNodes);
  const rawNodes: Node<ShaderType>[] = Array.isArray(rawResult) ? rawResult : [rawResult];
  if (options.kind) assertReadsNoStageInput(rawNodes, options.kind);
  // Without a stage the function is a plain function of its context, whose
  // result can be any value.
  const resultNodes = numberClashingVariables(shareNodes(prepareRoots(options.stage, rawNodes)));

  const ctx: CompileCtx = {
    nextId: 0,
    shaderStage: stage,
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
    jsParams: new Set(options.params.map((p) => p.name)),
    jsHelpers: new Set(),
    outTarget: null,
    jsEpoch: 0,
    jsReadsSlot: false,
    derivatives,
    reentrant,
    jsNeedsRes: false,
    jsFloat32: options.float === "f32",
  };

  assertOneDeclarationPerName(resultNodes);
  // The value of the function is written into a hoisted slot too, so a vector or a matrix result allocates nothing.
  const compiledList = resultNodes.map((n, i) => {
    if (i !== resultNodes.length - 1 || !jsIsArrayType((n as any)?._t) || isJSArrayLeaf(n))
      return compileJSStage(n, ctx);
    ctx.outTarget = jsNewTemp(ctx, (n as any)._t);
    const compiled = compileJSStage(n, ctx);
    ctx.outTarget = null;
    return compiled;
  });
  const lastCompiled = compiledList[compiledList.length - 1];
  const lastType = (resultNodes[resultNodes.length - 1] as any)?._t;
  if (options.stage !== undefined) assertStageResult(stage, lastType, ctx.positionWritten, ctx.outputs.size > 0);

  const body: string[] = [];
  // The object a stage returns its outputs in is made once, as its slots are, or once a call with `reentrant`.
  const res = ctx.jsNeedsRes ? "var res = { outputs: {}, varyings: {} };" : "";
  if (res && reentrant) body.push(res);
  const slots = jsSlotDeclarations(ctx.varDefs, ctx.jsFloat32 === true, reentrant ? "var" : "let");
  body.push(...slots.scalars);
  if (reentrant) body.push(...slots.views);
  for (const compiled of compiledList) body.push(...compiled.decls, ...compiled.body);
  if (ctx.jsNeedsRes) {
    // a program that returns nothing has no value
    if (lastType !== "void") body.push(`res.value = ${lastCompiled.expr};`);
    body.push("return res;");
  } else {
    body.push(`return ${lastCompiled.expr};`);
  }

  let scratch = reentrant ? "" : [...(res ? [res] : []), ...slots.views].join("\n");
  let helpers = [...ctx.jsHelpers]
    .sort()
    .map((name) => jsHelperSource(name))
    .join("\n\n");

  let constants = [...(ctx.jsConstants ?? [])].map(([name, literal]) => `const ${name} = ${literal};`).join("\n");

  let parts: string[] = [];
  if (constants) parts.push(constants);
  if (scratch) parts.push(scratch);
  if (helpers) parts.push(helpers);
  parts.push(`return function ${options.name}(ctx) {\n${body.map((l) => "  " + l).join("\n")}\n};`);
  return {
    source: parts.join("\n\n"),
    resultType: lastType as ShaderType | undefined,
    storageTypes: Object.fromEntries(ctx.storageTypes ?? []) as Record<string, ShaderType>,
    resultTypes: {
      value: lastType as ShaderType | undefined,
      varyings: Object.fromEntries([...ctx.varyings.values()].map((v) => [v.slot, v.type])),
      outputs: Object.fromEntries([...ctx.outputs.values()].map((o) => [o.slot, o.type])),
      float32: ctx.jsFloat32 === true,
    },
  };
}

export function compileJSFn(
  fn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  options: CompileJSOptions,
): string {
  return compileJSFnDetailed(fn, options).source;
}

/** A JS program: what the stages and the grid take what they give from, and the call that skips the copy of `run`. */
export interface JsProgram extends CpuProgram {
  /**
   * Like `run`, but the result lives in the scratch slots the next call
   * overwrites. For a caller that reads each result at once.
   */
  runInPlace(ctx: CpuShaderContext): CpuValue<ShaderType> | CpuProgramResult | null;
}

/** The types of what a compiled JS function returns: its value, and the varyings and outputs it writes by slot. */
type JsResultTypes = {
  value: ShaderType | undefined;
  varyings: Record<string, string>;
  outputs: Record<string, string>;
  float32: boolean;
};

/**
 * A copy of what one call returned, so the next call does not change it: each
 * vector or matrix in a new typed array of its kind, from the type the compile
 * gave it.
 */
function ownedResult(raw: unknown, types: JsResultTypes): unknown {
  const copy = (value: unknown, type: string | undefined): unknown =>
    type !== undefined && (Array.isArray(value) || ArrayBuffer.isView(value))
      ? typedValue(value as ArrayLike<number>, type, types.float32)
      : value;
  if (raw === null || typeof raw !== "object") return raw;
  if (Array.isArray(raw) || ArrayBuffer.isView(raw)) return copy(raw, types.value);
  const result = raw as CpuProgramResult;
  const owned: CpuProgramResult = {};
  if ("value" in result) owned.value = copy(result.value, types.value);
  if (result.position !== undefined) owned.position = copy(result.position, "vec4") as number[];
  if (result.varyings) {
    owned.varyings = {};
    for (const slot in result.varyings) owned.varyings[slot] = copy(result.varyings[slot], types.varyings[slot]);
  }
  if (result.outputs) {
    owned.outputs = {};
    for (const slot in result.outputs) owned.outputs[slot] = copy(result.outputs[slot], types.outputs[slot]);
  }
  if (result.fragDepth !== undefined) owned.fragDepth = result.fragDepth;
  return owned;
}

/**
 * Compile an Fn to an actual callable program, with the scratch slots and
 * helper functions baked into its closure.
 *
 * `run` is called as `run(ctx)` where `ctx` is a `CpuShaderContext`, and the
 * value it returns is the caller's own: a later call does not change it. The
 * scratch slots are shared across calls, so a call must finish before the
 * next one starts. Pass `{ reentrant: true }` for per-call bindings instead.
 *
 * Also carries `draw()`, the whole-image entry point: one JS call per pixel,
 * feeding `fragCoord` in and packing every result into one flat row-major
 * buffer, and `compute()`, one call per invocation.
 */
export function compileJSProgram(
  fn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  options: CompileJSOptions,
): JsProgram {
  const { source, resultType, storageTypes, resultTypes } = compileJSFnDetailed(fn, options);
  const factory = new Function(source) as () => (ctx: CpuShaderContext) => number | boolean | CpuProgramResult | null;
  const runScratch = factory();

  /** The result of one call, copied out of the scratch slots the next call writes into. */
  function run(ctx: CpuShaderContext): number | boolean | CpuProgramResult | null {
    return ownedResult(runScratch(ctx), resultTypes) as number | boolean | CpuProgramResult | null;
  }

  function draw(ctx: CpuShaderContext, width: number, height: number, out?: CpuDrawBuffer): CpuDrawBuffer {
    if (resultType === undefined || resultType === "void") {
      throw new Error("[RMSL] compileJSGrid: this function produces no value to render — the grid needs a result.");
    }
    const componentCount = componentCountOf(resultType);
    const kind = isAggregate(resultType) ? elementKindOf(resultType) : scalarKindOf(resultType);
    const buffer: CpuDrawBuffer =
      out ??
      (kind === "float"
        ? new (resultTypes.float32 ? Float32Array : Float64Array)(width * height * componentCount)
        : kind === "uint"
          ? new Uint32Array(width * height * componentCount)
          : new Int32Array(width * height * componentCount));

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        // pixel centers land at (x + 0.5, y + 0.5) — the same convention
        // compileWasmRoutine's draw() and fragCoordMemory in wasm.ts use.
        const result = runScratch({ ...ctx, fragCoord: [x + 0.5, y + 0.5] });
        if (result === null) {
          // a discarded fragment leaves the pixel at zero in every channel
          buffer.fill(0, (y * width + x) * componentCount, (y * width + x + 1) * componentCount);
          continue;
        }
        const raw =
          typeof result === "object" && result !== null && "value" in result
            ? (result as CpuProgramResult).value
            : result;
        const values = Array.isArray(raw) || ArrayBuffer.isView(raw) ? (raw as ArrayLike<unknown>) : [raw];
        const base = (y * width + x) * componentCount;
        for (let k = 0; k < componentCount; k++) {
          buffer[base + k] = kind === "bool" ? (values[k] ? 1 : 0) : (values[k] as number);
        }
      }
    }
    return buffer;
  }

  /**
   * Runs each invocation on `ctx` itself, so a dispatch allocates nothing:
   * `ctx.index` is each invocation's index while it runs, and what it was
   * before once the dispatch returns or throws.
   */
  function compute(ctx: CpuShaderContext, count: number): void {
    const index = ctx.index;
    try {
      for (let i = 0; i < count; i++) {
        ctx.index = i;
        runScratch(ctx);
      }
    } finally {
      ctx.index = index;
    }
  }

  // A reentrant routine declares its variables per call, so nothing is shared to copy out of.
  return { run, runInPlace: runScratch, draw, compute, storageTypes };
}

/** What a stage compile function takes: the options of a routine, without the stage, which the function names. */
export type CompileJSStageOptions = Omit<CompileJSOptions, "stage" | "kind">;

/**
 * Compiles an `Fn` as a vertex stage: a function that returns the position and
 * the varyings the program writes.
 */
export function compileJSVertex<W extends FloatWidth = "f64">(
  fn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  options: CompileJSStageOptions & { float?: W },
): VertexStage<W> {
  const program = compileJSProgram(fn, { ...options, stage: "vertex" });
  return (ctx) => toVertexResult(program.run(ctx)) as VertexResult<W>;
}

/**
 * Compiles an `Fn` as a fragment stage: a function that returns the colour and
 * the members of the `outputStruct` the program returns, or `null` for a
 * discarded fragment.
 */
export function compileJSFragment<R extends Node<ShaderType>, W extends FloatWidth = "f64">(
  fn: (...args: any[]) => R,
  options: CompileJSStageOptions & { float?: W },
): FragmentStage<R, W>;
export function compileJSFragment<W extends FloatWidth = "f64">(
  fn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  options: CompileJSStageOptions & { float?: W },
): FragmentStage<unknown, W>;
export function compileJSFragment(
  fn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  options: CompileJSStageOptions,
): FragmentStage {
  const program = compileJSProgram(fn, { ...options, stage: "fragment" });
  return (ctx) => toFragmentResult(program.run(ctx));
}

/**
 * Compiles an `Fn` as a compute stage: a function that runs the program once
 * per index of a count, and returns nothing. It reads `invocationIndex()` and
 * writes `storage()`, and its `storageTypes` name the buffers it reads.
 */
export function compileJSCompute(
  fn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  options: CompileJSStageOptions,
): ComputeStage {
  const program = compileJSProgram(fn, { ...options, stage: "compute" });
  return Object.assign((ctx: CpuShaderContext, count: number) => program.compute(ctx, count), {
    storageTypes: program.storageTypes ?? {},
  });
}

/** Compiles an `Fn` of `fragCoord()` as a grid: one result for each pixel, in a buffer the type of the result. */
export function compileJSGrid<A extends ShaderType, W extends FloatWidth = "f64">(
  fn: (...args: any[]) => Node<A>,
  options: CompileJSStageOptions & { float?: W },
): CpuGrid<A, W> {
  const program = compileJSProgram(fn, { ...options, kind: "grid" });
  return (ctx, width, height, out) => program.draw(ctx, width, height, out) as GridBuffer<A, W>;
}

/**
 * Compiles an `Fn` to a function of a context: it reads its parameters and
 * uniforms from `ctx`, and returns its value, typed by the type the program
 * returns. A program that reads what only a stage has, such as `fragCoord()`
 * or a varying, is refused: compile it as a stage or as a grid.
 */
export function compileJSRoutine<A extends ShaderType, W extends FloatWidth = "f64">(
  fn: (...args: any[]) => Node<A>,
  options: CompileJSStageOptions & { float?: W },
): CpuRoutine<A, W>;
export function compileJSRoutine<W extends FloatWidth = "f64">(
  fn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  options: CompileJSStageOptions & { float?: W },
): CpuRoutine<ShaderType, W>;
export function compileJSRoutine(
  fn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  options: CompileJSStageOptions,
): CpuRoutine {
  const program = compileJSProgram(fn, { ...options, kind: "routine" });
  return (ctx) => program.run(ctx) as never;
}
