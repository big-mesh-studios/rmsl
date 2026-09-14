// ========== JS Compiler ==========
// The third backend: compile a node graph to a JavaScript function that the
// host can call on the CPU, one fragment at a time.
import { BaseNode, MATRIX_DIMENSIONS, Node, ShaderType, TYPE_WIDTH, var_ } from "../core";
import {
  componentCountOf,
  CpuDrawBuffer,
  CpuRenderer,
  CpuShaderContext,
  CpuShaderResult,
  elementKindOf,
  isAggregate,
  scalarKindOf,
} from "./cpu";
import {
  CompileCtx,
  CompileFnOptions,
  CompiledNode,
  PRECEDENCE,
  PREC_ATOM,
  PREC_UNARY,
  assertPositionIsReadable,
  assertSquareMatrix,
  assertStageResult,
  forUpdateStatements,
  resolveSwizzleTarget,
  tryFold,
  withoutSemicolon,
  wrapExpr,
} from "./shared";
// ========== JS Compiler ==========
/**
 * The third backend: compile a node graph to a JavaScript function that the
 * host can call on the CPU, one fragment at a time. Its purpose is screen
 * picking from a ray-marched scene — feed the per-pixel varyings and uniforms
 * into the compiled function and read the colour/depth back, no GPU round-trip.
 *
 * Values are plain numbers (scalars), arrays (vectors) and flat column-major
 * arrays (matrices) — the same representation `wrapValue` and the apps use.
 * Internal `toVar()` variables live in per-program scratch slots outside the
 * callable, and vector/matrix helpers write into a caller-supplied output
 * array, so a per-pixel evaluation allocates nothing beyond the result.
 */

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
]);

export function isJSArrayLeaf(node: any): boolean {
  return !!node && JS_ARRAY_LEAF_TYPES.has(node.type);
}

/** Element-wise operations the JS vector helpers implement, per index. */
export const JS_ELEM: Record<string, { argc: number; fn: (xs: string[]) => string }> = {
  add: { argc: 2, fn: (xs) => `${xs[0]} + ${xs[1]}` },
  sub: { argc: 2, fn: (xs) => `${xs[0]} - ${xs[1]}` },
  mul: { argc: 2, fn: (xs) => `${xs[0]} * ${xs[1]}` },
  div: { argc: 2, fn: (xs) => `${xs[0]} / ${xs[1]}` },
  // Integer division truncates, following GLSL/WGSL, not JS's float `/`.
  idiv: { argc: 2, fn: (xs) => `Math.trunc(${xs[0]} / ${xs[1]})` },
  min: { argc: 2, fn: (xs) => `Math.min(${xs[0]}, ${xs[1]})` },
  max: { argc: 2, fn: (xs) => `Math.max(${xs[0]}, ${xs[1]})` },
  pow: { argc: 2, fn: (xs) => `Math.pow(${xs[0]}, ${xs[1]})` },
  atan2: { argc: 2, fn: (xs) => `Math.atan2(${xs[0]}, ${xs[1]})` },
  // Floored, matching GLSL's mod() — JS % truncates toward zero.
  mod: { argc: 2, fn: (xs) => `${xs[0]} - ${xs[1]} * Math.floor(${xs[0]} / ${xs[1]})` },
  imod: { argc: 2, fn: (xs) => `${xs[0]} % ${xs[1]}` },
  // step(edge, x): 0 while x < edge, 1 from there on.
  step: { argc: 2, fn: (xs) => `${xs[1]} < ${xs[0]} ? 0 : 1` },
  clamp: { argc: 3, fn: (xs) => `Math.min(Math.max(${xs[0]}, ${xs[1]}), ${xs[2]})` },
  mix: { argc: 3, fn: (xs) => `${xs[0]} + ${xs[2]} * (${xs[1]} - ${xs[0]})` },
  smoothstep: {
    argc: 3,
    fn: (xs) =>
      `(function(t){ return t * t * (3 - 2 * t); })(Math.min(Math.max((${xs[2]} - ${xs[0]}) / (${xs[1]} - ${xs[0]}), 0), 1))`,
  },
  neg: { argc: 1, fn: (xs) => `-${xs[0]}` },
  abs: { argc: 1, fn: (xs) => `Math.abs(${xs[0]})` },
  sign: { argc: 1, fn: (xs) => `Math.sign(${xs[0]})` },
  floor: { argc: 1, fn: (xs) => `Math.floor(${xs[0]})` },
  ceil: { argc: 1, fn: (xs) => `Math.ceil(${xs[0]})` },
  round: { argc: 1, fn: (xs) => `Math.round(${xs[0]})` },
  trunc: { argc: 1, fn: (xs) => `Math.trunc(${xs[0]})` },
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
  let m = /^v(\d+)([a-zA-Z]+)$/.exec(name);
  if (m) {
    let width = Number(m[1]);
    let op = m[2];
    let e = JS_ELEM[op];
    if (e) {
      let args = "abcdef".slice(0, e.argc).split("");
      let lines: string[] = [];
      for (let i = 0; i < width; i++) {
        let xs = args.map((a) => `(typeof ${a} === "number" ? ${a} : ${a}[${i}])`);
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
        `  out[0] = a[1] * b[2] - a[2] * b[1];\n` +
        `  out[1] = a[2] * b[0] - a[0] * b[2];\n` +
        `  out[2] = a[0] * b[1] - a[1] * b[0];\n` +
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
    case "copy":
      return `function _copy(src, out) {\n  for (let i = 0; i < src.length; i++) out[i] = src[i];\n  return out;\n}`;
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
      return `function _texCube(tex, dir, out) {
  out = out || [0, 0, 0, 0];
  let f = _cubeFace(dir[0], dir[1], dir[2], [0, 0, 0]);
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
      return `function _mat2x2inv(m, out) {\n  out = out || new Array(4);\n  let det = m[0] * m[3] - m[1] * m[2];\n  let inv = 1 / det;\n  out[0] = m[3] * inv;\n  out[1] = -m[1] * inv;\n  out[2] = -m[2] * inv;\n  out[3] = m[0] * inv;\n  return out;\n}`;
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
    return (
      `function _${name}(a, b, out) {\n` +
      `  out = out || new Array(${cR * rL});\n` +
      `  if (out === a) a = a.slice();\n` +
      `  if (out === b) b = b.slice();\n${lines.join("\n")}\n  return out;\n}`
    );
  }

  let mmT = /^mat(\d+)x(\d+)T$/.exec(name);
  if (mmT) {
    let cols = Number(mmT[1]);
    let rows = Number(mmT[2]);
    // Transpose: out[r*cols + c] = m[c*rows + r].
    let lines: string[] = [];
    for (let c = 0; c < cols; c++)
      for (let r = 0; r < rows; r++) {
        lines.push(`  out[${r * cols + c}] = m[${c * rows + r}];`);
      }
    return (
      `function _${name}(m, out) {\n` +
      `  out = out || new Array(${cols * rows});\n${lines.join("\n")}\n  return out;\n}`
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

  throw new Error(`[RMSL] Unknown JS helper: ${name}`);
}

export function jsRequireHelper(ctx: CompileCtx, name: string): void {
  ctx.jsHelpers.add(name);
}

/** A fresh hoisted slot for an intermediate value, registered for preallocation. */
export function jsNewTemp(ctx: CompileCtx, brand: string): string {
  let name = `_rmsl_t${ctx.nextId++}`;
  ctx.varDefs.set(name, brand);
  return name;
}

/**
 * Compile an operand for a vector/matrix operation.
 *
 * In a plain expression (`outTarget` null) operands compile as expressions and
 * allocate. Under an assignment, an array-typed operand that itself computes
 * something needs its own scratch slot — the parent's helper writes into the
 * target while reading its operands, so an operand may never share that slot.
 * Leaves (variables, uniforms, literals) are references and need no slot.
 */
export function jsCompileOperand(node: any, ctx: CompileCtx): CompiledNode {
  let saved = ctx.outTarget;
  let isArrayOp = jsIsArrayType(node?._t) && !isJSArrayLeaf(node);
  if (ctx.outTarget && isArrayOp) {
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

/** The scalar kind a type's components hold, vector and scalar types alike. */
export function componentKind(type: string | undefined): "float" | "int" | "uint" | "bool" {
  if (type === undefined) return "float";
  if (type === "bool" || type.startsWith("bvec")) return "bool";
  if (type === "int" || type.startsWith("ivec")) return "int";
  if (type === "uint" || type.startsWith("uvec")) return "uint";
  return "float";
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
  let from = componentKind(sourceType);
  let to = componentKind(targetType);
  if (from === to) return expr;
  if (from === "bool") expr = `(${expr} ? 1 : 0)`;
  if (to === "bool") return `(${expr} !== 0)`;
  if (to === "int") return `Math.trunc(${expr})`;
  if (to === "uint") return `(${expr} >>> 0)`;
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

export function jsScalarBinary(node: BaseNode<ShaderType>, ctx: CompileCtx, op: string): CompiledNode {
  let a = compileJSStage(node.params![0], ctx);
  let b = compileJSStage(node.params![1], ctx);
  let c = node.params![2] ? compileJSStage(node.params![2], ctx) : null;
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
    case "idiv":
      expr = `Math.trunc(${jsOperand(a)} / ${jsOperand(b)})`;
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
    case "imod":
      expr = `(${jsOperand(a)} % ${jsOperand(b)})`;
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
    case "smoothstep":
      expr = `(function(t){ return t * t * (3 - 2 * t); })(Math.min(Math.max((${jsOperand(c!)} - ${jsOperand(a)}) / (${jsOperand(b)} - ${jsOperand(a)}), 0), 1))`;
      break;
    default:
      throw new Error(`[RMSL] Unknown JS scalar op: ${op}`);
  }
  return { decls, body, expr, prec: PRECEDENCE[node.type] };
}

export function jsVectorBinary(node: BaseNode<ShaderType>, ctx: CompileCtx, op: string, width: number): CompiledNode {
  let a = jsCompileOperand(node.params![0], ctx);
  let b = jsCompileOperand(node.params![1], ctx);
  let c = node.params![2] ? jsCompileOperand(node.params![2], ctx) : null;
  jsRequireHelper(ctx, `v${width}${op}`);
  let args = c ? `${a.expr}, ${b.expr}, ${c.expr}` : `${a.expr}, ${b.expr}`;
  let decls = [...a.decls, ...b.decls, ...(c ? c.decls : [])];
  let body = [...a.body, ...b.body, ...(c ? c.body : [])];
  if (ctx.outTarget) {
    return { decls, body: [...body, `_v${width}${op}(${args}, ${ctx.outTarget});`], expr: ctx.outTarget };
  }
  return { decls, body, expr: `_v${width}${op}(${args})` };
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

export function jsUnaryMath(node: BaseNode<ShaderType>, ctx: CompileCtx, suffix: string): CompiledNode {
  let width = jsArrayLength(node.params![0]?._t);
  if (width <= 1) {
    let a = compileJSStage(node.params![0], ctx);
    let e = JS_ELEM[suffix];
    if (!e) throw new Error(`[RMSL] Unknown JS unary op: ${suffix}`);
    return { decls: a.decls, body: a.body, expr: e.fn([`(${a.expr})`]) };
  }
  jsRequireHelper(ctx, `v${width}${suffix}`);
  let a = jsCompileOperand(node.params![0], ctx);
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
  jsRequireHelper(ctx, helper);
  let a = jsCompileOperand(node.params![0], ctx);
  let b = node.params![1] ? jsCompileOperand(node.params![1], ctx) : null;
  let decls = b ? [...a.decls, ...b.decls] : a.decls;
  let body = b ? [...a.body, ...b.body] : a.body;
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

export function jsBitwise(node: BaseNode<ShaderType>, ctx: CompileCtx, op: string): CompiledNode {
  let a = compileJSStage(node.params![0], ctx);
  let isUint = node._t === "uint";
  let mask = isUint ? ">>> 0" : "| 0";
  if (op === "~") {
    return { decls: a.decls, body: a.body, expr: `(~(${a.expr})) ${mask}`, prec: PREC_UNARY };
  }
  let b = compileJSStage(node.params![1], ctx);
  let sym = op === ">>" && isUint ? ">>>" : op;
  return {
    decls: [...a.decls, ...b.decls],
    body: [...a.body, ...b.body],
    expr: `((${a.expr}) ${sym} (${b.expr})) ${mask}`,
    prec: PRECEDENCE[node.type] ?? 0,
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

  let seen = ctx.memo.get(node);
  if (seen) return { decls: [], body: [], expr: seen.expr, prec: seen.prec };

  let result = compileJSNode(node, ctx);
  ctx.memo.set(node, result);
  return result;
}

export function compileJSNode(
  node: BaseNode<ShaderType> | ShaderType extends never ? never : any,
  ctx: CompileCtx,
): CompiledNode {
  let folded = tryFold(node);
  if (folded) node = folded;

  switch (node.type) {
    case "float":
      return { decls: [], body: [], expr: String(node.value) };
    case "int":
      return { decls: [], body: [], expr: String(node.value) };
    case "uint":
      return { decls: [], body: [], expr: String(node.value) };
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
      if (ctx.outTarget) {
        let lines = values.map((v, i) => `${ctx.outTarget}[${i}] = ${JSON.stringify(v)};`);
        return { decls: [], body: lines, expr: ctx.outTarget };
      }
      return { decls: [], body: [], expr: `[${values.map((v) => JSON.stringify(v)).join(", ")}]` };
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
          let c = jsCompileOperand(params[0], ctx);
          let broadcast = jsComponentCast(c.expr, params[0]?._t, targetType);
          if (ctx.outTarget) {
            let writes = Array.from({ length: width }, (_, i) => `${ctx.outTarget}[${i}] = ${broadcast};`);
            return { decls: c.decls, body: [...c.body, ...writes], expr: ctx.outTarget };
          }
          return { decls: c.decls, body: c.body, expr: `[${Array(width).fill(broadcast).join(", ")}]` };
        }
        // Vector construct: expand every operand's components into one array.
        let compiled = params.map((p: BaseNode<ShaderType>) => ({
          c: jsCompileOperand(p, ctx),
          w: TYPE_WIDTH[p?._t] ?? 1,
          t: p?._t as string | undefined,
        }));
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
        if (ctx.outTarget) {
          let writes = pieces.map((piece, i) => `${ctx.outTarget}[${i}] = ${piece};`);
          return { decls, body: [...body, ...writes], expr: ctx.outTarget };
        }
        return { decls, body, expr: `[${pieces.join(", ")}]` };
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
            if (ctx.outTarget) {
              jsRequireHelper(ctx, "copy");
              return { decls: c.decls, body: [...c.body, `_copy(${c.expr}, ${ctx.outTarget});`], expr: ctx.outTarget };
            }
            return { decls: c.decls, body: c.body, expr: `${c.expr}.slice()` };
          }
          // A scalar source: the diagonal. Zero the whole slot first — it is a
          // hoisted slot and could carry stale off-diagonal values from a
          // previous call.
          let s = compileJSStage(src, ctx);
          if (ctx.outTarget) {
            let zeroAll = Array(size)
              .fill(0)
              .map((_, i) => `${ctx.outTarget}[${i}] = 0;`);
            let diag: string[] = [];
            for (let col = 0; col < cols; col++)
              for (let row = 0; row < rows; row++) {
                if (col === row) diag.push(`${ctx.outTarget}[${col * rows + row}] = ${s.expr};`);
              }
            return { decls: s.decls, body: [...s.body, ...zeroAll, ...diag], expr: ctx.outTarget };
          }
          jsRequireHelper(ctx, "matDiag");
          return { decls: s.decls, body: s.body, expr: `_matDiag(${s.expr}, ${size}, ${rows + 1})` };
        }
        // Column-wise construction: each param is one column vector.
        let compiled = (node.params ?? []).map((p: BaseNode<ShaderType>) => jsCompileOperand(p, ctx));
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

    case "attribute": {
      let v = node.value as any;
      return jsLeafRef(`ctx.attributes[${JSON.stringify(v.slot)}]`, v.shaderType ?? node._t, ctx);
    }

    case "varying": {
      let v = node.value as any;
      let slot = v?.slot;
      // In a vertex stage a varying is an output, collected in the result so
      // the host can read it back; in a fragment stage it is an input.
      if (ctx.shaderStage === "vertex") {
        ctx.jsNeedsRes = true;
        return jsLeafRef(`res.varyings[${JSON.stringify(slot)}]`, v.shaderType ?? node._t, ctx);
      }
      return jsLeafRef(`ctx.varyings[${JSON.stringify(slot)}]`, v.shaderType ?? node._t, ctx);
    }

    case "output": {
      let v = node.value as any;
      ctx.jsNeedsRes = true;
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
      let src = jsCompileOperand(node.params![0], ctx);
      let pattern = node.value as string;
      let srcExpr = (src.prec ?? PREC_ATOM) < PREC_ATOM ? `(${src.expr})` : src.expr;
      if (pattern.length === 1) {
        return { decls: src.decls, body: src.body, expr: `${srcExpr}[${JS_COMPONENT_INDEX[pattern]}]` };
      }
      let idx = [...pattern].map((ch) => JS_COMPONENT_INDEX[ch]);
      if (ctx.outTarget) {
        let lines = idx.map((j, i) => `${ctx.outTarget}[${i}] = ${srcExpr}[${j}];`);
        return { decls: src.decls, body: [...src.body, ...lines], expr: ctx.outTarget };
      }
      return { decls: src.decls, body: src.body, expr: `[${idx.map((j) => `${srcExpr}[${j}]`).join(", ")}]` };
    }

    case "negate": {
      if (jsArrayLength(node.params![0]?._t) <= 1) {
        let a = compileJSStage(node.params![0], ctx);
        return { decls: a.decls, body: a.body, expr: `-${wrapExpr(a.prec, PREC_UNARY, a.expr)}`, prec: PREC_UNARY };
      }
      return jsUnaryMath(node, ctx, "neg");
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
      return jsBinaryOp(node, ctx, "add");
    case "sub":
      return jsBinaryOp(node, ctx, "sub");
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
      return jsBinaryOp(node, ctx, "mul");
    }
    case "div": {
      let t = node.params![0]?._t;
      return jsBinaryOp(node, ctx, t === "int" || t === "uint" ? "idiv" : "div");
    }
    case "mod": {
      let t = node.params![0]?._t;
      return jsBinaryOp(node, ctx, t === "int" || t === "uint" ? "imod" : "mod");
    }
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
        return {
          decls: [...cond.decls, ...a.decls, ...b.decls],
          body: [...cond.body, ...a.body, ...b.body],
          expr: `_bselect(${cond.expr}, ${a.expr}, ${b.expr})`,
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
      return jsUnaryMath(node, ctx, "abs");
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
      let mat = jsCompileOperand(node.params![0], ctx);
      let idx = jsCompileOperand(node.params![1], ctx);
      let brand = node.params![0]?._t;
      let [, rows] = MATRIX_DIMENSIONS[brand];
      let matExpr = (mat.prec ?? PREC_ATOM) < PREC_ATOM ? `(${mat.expr})` : mat.expr;
      if (ctx.outTarget) {
        let lines = Array.from(
          { length: rows },
          (_, row) => `${ctx.outTarget}[${row}] = ${matExpr}[(${idx.expr}) * ${rows} + ${row}];`,
        );
        return { decls: [...mat.decls, ...idx.decls], body: [...mat.body, ...idx.body, ...lines], expr: ctx.outTarget };
      }
      return {
        decls: [...mat.decls, ...idx.decls],
        body: [...mat.body, ...idx.body],
        expr: `${matExpr}.slice((${idx.expr}) * ${rows}, (${idx.expr}) * ${rows} + ${rows})`,
      };
    }

    case "vectorElement": {
      let src = jsCompileOperand(node.params![0], ctx);
      let idx = jsCompileOperand(node.params![1], ctx);
      let srcExpr = (src.prec ?? PREC_ATOM) < PREC_ATOM ? `(${src.expr})` : src.expr;
      return { decls: [...src.decls, ...idx.decls], body: [...src.body, ...idx.body], expr: `${srcExpr}[${idx.expr}]` };
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
      let rhsNode = node.params![1];

      // A swizzle target: single components assign directly, multi-component
      // ones split into per-component writes (JS has no `v.xy = e`).
      if (targetNode?.type === "swizzle") {
        let resolved = resolveSwizzleTarget(targetNode);
        let base = compileJSStage(resolved.base, ctx);
        if (resolved.pattern.length === 1) {
          let rhs = compileJSStage(rhsNode, ctx);
          return {
            decls: [...base.decls, ...rhs.decls],
            body: [
              ...base.body,
              ...rhs.body,
              `${base.expr}[${JS_COMPONENT_INDEX[resolved.pattern[0]]}] = ${rhs.expr};`,
            ],
            expr: base.expr,
          };
        }
        let temp = jsNewTemp(ctx, rhsNode?._t || "float");
        let saved = ctx.outTarget;
        ctx.outTarget = temp;
        let rhs = compileJSStage(rhsNode, ctx);
        ctx.outTarget = saved;
        let fill = rhs.expr === temp ? [] : [`${temp} = ${rhs.expr};`];
        let writes = [...resolved.pattern].map((ch, i) => `${base.expr}[${JS_COMPONENT_INDEX[ch]}] = ${temp}[${i}];`);
        return {
          decls: [...base.decls, ...rhs.decls],
          body: [...base.body, ...rhs.body, ...fill, ...writes],
          expr: base.expr,
        };
      }

      let lhs = compileJSStage(targetNode, ctx);
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
      for (let p of params) {
        let r = compileJSStage(p, ctx);
        allDecls.push(...r.decls);
        allBody.push(...r.body);
        expr = r.expr;
      }
      return { decls: allDecls, body: allBody, expr };
    }

    case "if": {
      let cond = compileJSStage(node.params![0], ctx);
      let body = compileJSStage(node.params![1], ctx);
      let elseBody =
        node.params!.length >= 3 && node.params![2] !== undefined
          ? compileJSStage(node.params![2], ctx)
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
      let update = compileJSStage(node.params![2], ctx);
      let body = compileJSStage(node.params![3], ctx);
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
        expr: "0",
      };
    }

    case "while": {
      let cond = compileJSStage(node.params![0], ctx);
      let body = compileJSStage(node.params![1], ctx);
      return {
        decls: [...cond.decls, ...body.decls],
        body: [...cond.body, `while (${cond.expr}) {`, ...body.body.map((l) => "  " + l), "}"],
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
  stage?: "vertex" | "fragment";
  derivatives?: "throw" | "zero";
  reentrant?: boolean;
};

/**
 * Compile an Fn to JavaScript: a self-contained expression that evaluates to
 * the callable. The expression is the scratch slots and helper functions in a
 * closure, then `return function <name>(ctx) { ... }`, so a caller evaluates
 * it with `new Function(source)()` or embeds it and assigns the result.
 */
/**
 * `compileJSFn`'s real body, also handing back the root node's result type —
 * needed by `compileJS`'s `draw()` and computed here, from the one time `fn`
 * is actually called. A second call to read it back afterward is not an
 * option: `fn` routinely has side effects on the caller's own closure (the
 * `let tex; Fn(() => { tex = uniform(...); ... })` idiom this whole test
 * suite uses), so calling it twice leaves the caller's own reference
 * pointing at a second, different uniform than the one actually compiled in.
 */
function compileJSFnDetailed(
  fn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  options: CompileJSOptions,
): { source: string; resultType: ShaderType | undefined } {
  let stage = options.stage ?? "fragment";
  let derivatives = options.derivatives ?? "throw";
  let reentrant = options.reentrant ?? false;
  const paramNodes = options.params.map((p) => var_(p.name, p.type));
  const rawResult = fn(...paramNodes);
  const resultNodes: Node<ShaderType>[] = Array.isArray(rawResult) ? rawResult : [rawResult];

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
    derivatives,
    reentrant,
    jsNeedsRes: false,
  };

  const compiledList = resultNodes.map((n) => compileJSStage(n, ctx));
  const lastCompiled = compiledList[compiledList.length - 1];
  const lastType = (resultNodes[resultNodes.length - 1] as any)?._t;
  assertStageResult(stage, lastType, ctx.positionWritten);

  const body: string[] = [];
  if (ctx.jsNeedsRes) body.push("var res = { outputs: {}, varyings: {} };");
  if (reentrant) {
    for (const [v, brand] of ctx.varDefs) {
      let init = jsScratchLiteral(brand);
      body.push(init ? `var ${v} = ${init};` : `var ${v} = 0;`);
    }
  }
  for (const compiled of compiledList) body.push(...compiled.decls, ...compiled.body);
  if (ctx.jsNeedsRes) {
    body.push(`res.value = ${lastCompiled.expr};`);
    body.push("return res;");
  } else {
    body.push(`return ${lastCompiled.expr};`);
  }

  let scratch = reentrant
    ? ""
    : [...ctx.varDefs]
        .map(([v, brand]) => {
          let init = jsScratchLiteral(brand);
          return init ? `let ${v} = ${init};` : `let ${v};`;
        })
        .join("\n");
  let helpers = [...ctx.jsHelpers]
    .sort()
    .map((name) => jsHelperSource(name))
    .join("\n\n");

  let parts: string[] = [];
  if (scratch) parts.push(scratch);
  if (helpers) parts.push(helpers);
  parts.push(`return function ${options.name}(ctx) {\n${body.map((l) => "  " + l).join("\n")}\n};`);
  return {
    source: parts.join("\n\n"),
    resultType: lastType as ShaderType | undefined,
  };
}

export function compileJSFn(
  fn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  options: CompileJSOptions,
): string {
  return compileJSFnDetailed(fn, options).source;
}

/**
 * Compile an Fn to an actual callable function, with the scratch slots and
 * helper functions baked into its closure.
 *
 * The result is called as `fn(ctx)` where `ctx` is a `CpuShaderContext`. Its
 * scratch slots are shared across calls, so a call must finish before the next
 * one starts — for screen picking one call per click that is the point. Pass
 * `{ reentrant: true }` for per-call bindings instead.
 *
 * Also carries `draw()`, the same whole-image entry point `compileWasm`'s
 * result has: one JS call per pixel, feeding `fragCoord` in and packing every
 * result into one flat row-major buffer — see `CpuRenderer`.
 */
export function compileJS(
  fn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  options: CompileJSOptions,
): CpuRenderer {
  const { source, resultType } = compileJSFnDetailed(fn, options);
  const factory = new Function(source) as () => (ctx: CpuShaderContext) => number | boolean | CpuShaderResult;
  const callable = factory() as CpuRenderer;

  callable.draw = (ctx: CpuShaderContext, width: number, height: number, out?: CpuDrawBuffer): CpuDrawBuffer => {
    if (resultType === undefined) {
      throw new Error("[RMSL] compileJS: this function produces no value to render — draw() needs a result.");
    }
    const componentCount = componentCountOf(resultType);
    const kind = isAggregate(resultType) ? elementKindOf(resultType) : scalarKindOf(resultType);
    const buffer: CpuDrawBuffer =
      out ??
      (kind === "float"
        ? new Float64Array(width * height * componentCount)
        : kind === "uint"
          ? new Uint32Array(width * height * componentCount)
          : new Int32Array(width * height * componentCount));

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        // pixel centers land at (x + 0.5, y + 0.5) — the same convention
        // compileWasm's draw() and fragCoordMemory in wasm.ts use.
        const result = callable({ ...ctx, fragCoord: [x + 0.5, y + 0.5] });
        const raw =
          typeof result === "object" && result !== null && "value" in result
            ? (result as CpuShaderResult).value
            : result;
        const values = Array.isArray(raw) ? raw : [raw];
        const base = (y * width + x) * componentCount;
        for (let k = 0; k < componentCount; k++) {
          buffer[base + k] = kind === "bool" ? (values[k] ? 1 : 0) : (values[k] as number);
        }
      }
    }
    return buffer;
  };

  return callable;
}
