/**
 * What a CPU rasterizer draws for one triangle, computed exactly, so that a
 * sweep has something to hold the rasterizers to that shares no code or
 * rounding with them. Every 64-bit float is a dyadic rational, so each sign
 * is computed exactly with `BigInt`, and each weight and depth is the exact
 * ratio rounded once.
 *
 * The rules are the canon's: the top-left rule and the depth range from 0 to 1
 * applied to the triangle as its vertices place it, a component smaller than
 * 2^-300 of the largest of its vertex taken as 0, and a vertex at `vec4(0)`
 * placed at the centre at depth 0 with all of the weight, when the other two
 * lie in the clip volume. Each weight is a vertex's barycentric over its w,
 * over the sum of the three, as WebGPU interpolates.
 */

export { random } from "./inline-fn-fuzz";

/** A dyadic rational, `m * 2^e`. */
type Dyadic = { m: bigint; e: number };

const ZERO: Dyadic = { m: 0n, e: 0 };

function dyadic(x: number): Dyadic {
  if (!Number.isFinite(x)) throw new Error(`not finite: ${x}`);
  if (x === 0) return ZERO;
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, x);
  const bits = view.getBigUint64(0);
  const sign = bits >> 63n ? -1n : 1n;
  const exponent = Number((bits >> 52n) & 0x7ffn);
  const fraction = bits & ((1n << 52n) - 1n);
  if (exponent === 0) return { m: sign * fraction, e: -1074 };
  return { m: sign * (fraction | (1n << 52n)), e: exponent - 1075 };
}

const integer = (n: number): Dyadic => ({ m: BigInt(n), e: 0 });
const mul = (a: Dyadic, b: Dyadic): Dyadic => ({ m: a.m * b.m, e: a.e + b.e });
function add(a: Dyadic, b: Dyadic): Dyadic {
  if (a.m === 0n) return b;
  if (b.m === 0n) return a;
  if (a.e < b.e) return { m: a.m + (b.m << BigInt(b.e - a.e)), e: a.e };
  return { m: (a.m << BigInt(a.e - b.e)) + b.m, e: b.e };
}
const neg = (a: Dyadic): Dyadic => ({ m: -a.m, e: a.e });
const sub = (a: Dyadic, b: Dyadic): Dyadic => add(a, neg(b));
const sign = (a: Dyadic): number => (a.m > 0n ? 1 : a.m < 0n ? -1 : 0);
const abs = (a: Dyadic): Dyadic => (a.m < 0n ? neg(a) : a);

/** `a / b` rounded to a 64-bit float, near enough for a tolerance of 1e-12. */
function ratio(a: Dyadic, b: Dyadic): number {
  if (a.m === 0n) return 0;
  const negative = a.m < 0n !== b.m < 0n;
  const n = a.m < 0n ? -a.m : a.m;
  const d = b.m < 0n ? -b.m : b.m;
  const shift = 64 + d.toString(2).length - n.toString(2).length;
  const q = shift >= 0 ? (n << BigInt(shift)) / d : n / (d << BigInt(-shift));
  let exponent = a.e - b.e - shift;
  let value = Number(q);
  // Scale by 2^exponent in steps a 64-bit float holds.
  while (exponent > 1000) {
    value *= 2 ** 1000;
    exponent -= 1000;
  }
  while (exponent < -1000) {
    value *= 2 ** -1000;
    exponent += 1000;
  }
  value *= 2 ** exponent;
  return negative ? -value : value;
}

/** The line through two points in homogeneous x, y and w: the cross product of the two. */
function lineThrough(i: Dyadic[], j: Dyadic[]): Dyadic[] {
  return [
    sub(mul(i[1]!, j[3]!), mul(i[3]!, j[1]!)),
    sub(mul(i[3]!, j[0]!), mul(i[0]!, j[3]!)),
    sub(mul(i[0]!, j[1]!), mul(i[1]!, j[0]!)),
  ];
}

const at = (l: Dyadic[], p: Dyadic[]): Dyadic => add(add(mul(l[0]!, p[0]!), mul(l[1]!, p[1]!)), mul(l[2]!, p[3]!));

/** The ray of the centre of pixel `(x, y)`, `(ndcX, ndcY, 0, 1)` times `width * height`, so it is whole. */
function ray(x: number, y: number, width: number, height: number): Dyadic[] {
  return [integer((2 * x + 1 - width) * height), integer((height - 2 * y - 1) * width), ZERO, integer(width * height)];
}

/** A vertex's components, each smaller than 2^-300 of the largest of the four taken as 0. */
function flushed(v: readonly number[]): Dyadic[] {
  const d = v.map(dyadic);
  const largest = d.reduce((m, c) => (compare(abs(c), m) > 0 ? abs(c) : m), ZERO);
  return d.map((c) => (compare({ m: abs(c).m, e: abs(c).e + 300 }, largest) < 0 ? ZERO : c));
}

const compare = (a: Dyadic, b: Dyadic): number => sign(sub(a, b));

const isZero = (v: readonly number[]) => v.every((c) => c === 0);

/** Whether `v` lies in WebGPU's clip volume: x and y from -w to w, z from 0 to w. */
const insideClipVolume = (v: readonly number[]) =>
  Math.abs(v[0]!) <= v[3]! && Math.abs(v[1]!) <= v[3]! && v[2]! >= 0 && v[2]! <= v[3]!;

/** What a rasterizer draws for one triangle: which pixels, and each one's weights and depth. */
export interface ReferenceFragments {
  /** 1 for each pixel the triangle covers, row by row from the top. */
  covered: Uint8Array;
  /** The weight of each vertex at each pixel, three a pixel, summing to 1. */
  weights: Float64Array;
  /** The depth, z over w, at each pixel. */
  depth: Float64Array;
}

/**
 * The fragments the triangle at `positions`, three clip positions of four
 * components, draws on a target of `width` by `height`. `null` for a triangle
 * the canon leaves to the clip: one in a plane with the eye that is not a
 * vertex at `vec4(0)` and two vertices in the clip volume.
 */
export function referenceFragments(
  positions: readonly number[],
  width: number,
  height: number,
): ReferenceFragments | null {
  const given = [0, 1, 2].map((k) => positions.slice(4 * k, 4 * k + 4));
  const zeros = given.filter(isZero).length;
  // A vertex at vec4(0) with the other two in the clip volume lies at the centre, at depth 0, with all the weight.
  const centre = zeros === 1 && given.every((v) => isZero(v) || insideClipVolume(v)) ? given.findIndex(isZero) : -1;
  if (zeros > 0 && centre < 0) return null;
  const vertex = given.map((v, k) => (k === centre ? [ZERO, ZERO, ZERO, integer(1)] : flushed(v)));
  const lines = [0, 1, 2].map((e) => lineThrough(vertex[(e + 1) % 3]!, vertex[(e + 2) % 3]!));
  const s = sign(at(lines[0]!, vertex[0]!));
  const covered = new Uint8Array(width * height);
  const weights = new Float64Array(width * height * 3);
  const depth = new Float64Array(width * height);
  if (s === 0) return centre >= 0 ? { covered, weights, depth } : null;
  // Top-left: an edge owns its centres when its inside lies to its right on the screen, or below a horizontal one.
  const owns = lines.map((l) => {
    const ax = sign(l[0]!) * s,
      by = -sign(l[1]!) * s;
    return ax > 0 || (ax === 0 && by > 0);
  });
  // The weights take the given vertices, whose w weigh them; the lines take the flushed ones.
  const weighed = given.map((v, k) => (k === centre ? [ZERO, ZERO, ZERO, ZERO] : v.map(dyadic)));
  const weightLines = [0, 1, 2].map((e) => lineThrough(weighed[(e + 1) % 3]!, weighed[(e + 2) % 3]!));
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const r = ray(x, y, width, height);
      const values = lines.map((l) => at(l, r));
      let inside = true;
      for (let e = 0; e < 3; e++) {
        const v = sign(values[e]!) * s;
        if (v < 0 || (v === 0 && !owns[e])) inside = false;
      }
      if (!inside) continue;
      // The point's z and w, each times one factor of the sign of s: in front of the eye, at depth z / w from 0 to 1.
      let z = ZERO,
        w = ZERO;
      for (let e = 0; e < 3; e++) {
        z = add(z, mul(values[e]!, vertex[e]![2]!));
        w = add(w, mul(values[e]!, vertex[e]![3]!));
      }
      if (sign(w) * s <= 0 || sign(z) * s < 0 || sign(sub(z, w)) * s > 0) continue;
      const p = y * width + x;
      covered[p] = 1;
      depth[p] = ratio(z, w);
      if (centre >= 0) {
        weights[p * 3 + centre] = 1;
        continue;
      }
      const lambda = weightLines.map((l) => at(l, r));
      const total = lambda.reduce(add, ZERO);
      for (let e = 0; e < 3; e++) weights[p * 3 + e] = ratio(lambda[e]!, total);
    }
  return { covered, weights, depth };
}

/** Which pixel centres lie exactly on the line through the projections of clip positions `a` and `b`. */
export function exactlyOnLine(a: readonly number[], b: readonly number[], width: number, height: number): Uint8Array {
  const l = lineThrough(a.map(dyadic), b.map(dyadic));
  const out = new Uint8Array(width * height);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) if (sign(at(l, ray(x, y, width, height))) === 0) out[y * width + x] = 1;
  return out;
}

/** The pixels the triangle at `positions` covers, or `null` where {@link referenceFragments} gives none. */
export function exactCoverage(positions: readonly number[], width: number, height: number): Uint8Array | null {
  return referenceFragments(positions, width, height)?.covered ?? null;
}

/** How far `p` lies inside plane `plane` of the clip volume, as {@link insideClipVolume} takes them: w + x, w - x, w + y, w - y, z, w - z. */
function planeDistance(p: readonly Dyadic[], plane: number): Dyadic {
  const w = p[3]!;
  if (plane === 0) return add(w, p[0]!);
  if (plane === 1) return sub(w, p[0]!);
  if (plane === 2) return add(w, p[1]!);
  if (plane === 3) return sub(w, p[1]!);
  if (plane === 4) return p[2]!;
  return sub(w, p[2]!);
}

/**
 * The winding of the part of the triangle at `positions` inside the clip volume, in normalized
 * device coordinates with y up: 1 counter-clockwise, -1 clockwise, 0 for a part of no area or none.
 * It clips exactly, each cut vertex `da * b - db * a`, the cut point times `da - db`, which is
 * positive, and sums the polygon's area over the product of its w, also positive. A vertex at
 * `vec4(0)`, the one point of the volume whose w is 0, lies at the centre.
 */
export function windingInClipVolume(positions: readonly number[]): number {
  let polygon = [0, 1, 2].map((k) => positions.slice(4 * k, 4 * k + 4).map(dyadic));
  for (let plane = 0; plane < 6 && polygon.length > 0; plane++) {
    const next: Dyadic[][] = [];
    for (let i = 0; i < polygon.length; i++) {
      const a = polygon[i]!,
        b = polygon[(i + 1) % polygon.length]!;
      const da = planeDistance(a, plane),
        db = planeDistance(b, plane);
      if (sign(da) >= 0) next.push(a);
      if (sign(da) >= 0 !== sign(db) >= 0)
        next.push(a.map((c, k) => sub(mul(da, b[k]!), mul(db, c))).map((c) => (sign(da) >= 0 ? c : neg(c))));
    }
    polygon = next;
  }
  const n = polygon.length;
  let area = ZERO;
  for (let i = 0; i < n; i++) {
    const p = polygon[i]!,
      q = polygon[(i + 1) % n]!;
    if (sign(p[3]!) === 0 || sign(q[3]!) === 0) continue;
    let term = sub(mul(p[0]!, q[1]!), mul(q[0]!, p[1]!));
    for (let j = 0; j < n; j++)
      if (j !== i && j !== (i + 1) % n && sign(polygon[j]![3]!) !== 0) term = mul(term, polygon[j]![3]!);
    area = add(area, term);
  }
  return sign(area);
}
