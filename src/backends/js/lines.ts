/**
 * The lines of a triangle in homogeneous x, y and w, and the exact sign of each
 * at the ray through a pixel centre, as `rasterizer.wat` computes them.
 *
 * A pixel centre `(x + 0.5, y + 0.5)` on a target of `width` by `height` has
 * the ray `X = (2x + 1 - width) * height`, `Y = (height - 2y - 1) * width`,
 * `W = width * height`, whole numbers a 64-bit float holds exactly. The line
 * opposite vertex `e` is the cross product of the other two, `(a, b, c)`, and
 * its value at a ray is `aX + bY + cW`. A float filter decides the sign of a
 * value far from 0; within its bound of rounding, the value is summed exactly
 * as an expansion of floats, as Shewchuk's adaptive predicates do. No value
 * crosses a function call as a float, so nothing boxes.
 */

/** The bound of rounding of a line's value, over the sum of its terms' sizes: three times what it can reach. */
const LINE_ROUNDING = 2 ** -48;

/**
 * The least a line's float value keeps, over the sum of its terms' sizes: 2^32 times its bound of rounding,
 * so a value kept is within 2^-32 of itself. A smaller value is summed exactly, as a weight takes it.
 */
const LINE_FILTER = 2 ** -16;

/**
 * The bound of rounding of the plane of depth's value, over the sum of its terms' sizes: its three
 * coefficients each rounded once from an exact sum, and the sum of their products.
 */
const PLANE_ROUNDING = 2 ** -50;

/** The most a depth kept from the float value of its plane may round by, over the point's w: 2^-32 of the depth range. */
const DEPTH_FILTER = 2 ** -32;

/** Dekker's splitter for a 64-bit float, `2^27 + 1`. */
const SPLITTER = 134217729;

/** A component smaller than this, times the largest of its vertex, counts as 0. */
const SMALLEST_COMPONENT = 2 ** -300;

/** The room for an expansion of one line's value: 24 terms and one more. */
const EXPANSION = 32;

/** What a rasterizer keeps of the triangle it draws by its lines; made once and filled for each triangle. */
export interface Lines {
  /** The vertices, each scaled by a power of two that brings its largest component near 1: x, y, z, w. */
  points: Float64Array;
  /** The power of two each vertex was scaled down by. */
  exponents: Int32Array;
  /** Each line's three coefficients, four terms each whose sum is exact: `(3e + k) * 4`. */
  terms: Float64Array;
  /** Each coefficient rounded, `3e + k`. */
  coefficients: Float64Array;
  /** The size of each coefficient's terms, which bounds its rounding. */
  sizes: Float64Array;
  /** Whether each line owns the pixel centres on it, by the top-left rule. */
  owns: Uint8Array;
  /** The weight of each vertex for each unit of its line, by which of the three lines are not 0: `mask * 3 + e`. */
  ratios: Float64Array;
  /**
   * The depth over w as a plane on the screen: the point's z along the ray `(X, Y, W)` is
   * `plane[0] X + plane[1] Y + plane[2] W`, and its w is `plane[3] W`, the triangle's determinant times `W`.
   */
  plane: Float64Array;
  /** Per line, the part of its value and of its terms' sizes that a row of pixels shares: `2e` and `2e + 1`. */
  rows: Float64Array;
  /** At the pixel being tested: each line's value. */
  values: Float64Array;
  /** The exact expansion of each line's value, and its length. */
  expansions: Float64Array;
  lengths: Int32Array;
  /** Terms to sum exactly, and the sum. */
  pending: Float64Array;
  sum: Float64Array;
}

/** Makes the arrays a rasterizer keeps for the lines of a triangle. */
export function makeLines(): Lines {
  return {
    points: new Float64Array(12),
    exponents: new Int32Array(3),
    terms: new Float64Array(36),
    coefficients: new Float64Array(9),
    sizes: new Float64Array(9),
    owns: new Uint8Array(3),
    ratios: new Float64Array(24),
    plane: new Float64Array(4),
    rows: new Float64Array(6),
    values: new Float64Array(3),
    expansions: new Float64Array(3 * EXPANSION),
    lengths: new Int32Array(3),
    pending: new Float64Array(320),
    sum: new Float64Array(320),
  };
}

const VIEW = new DataView(new ArrayBuffer(8));

/** The exponent of `from[i]`, finite and not 0: `floor(log2(|x|))`, subnormals included. */
function exponentOf(from: Float64Array, i: number): number {
  VIEW.setFloat64(0, from[i]!);
  const biased = (VIEW.getUint16(0) >> 4) & 0x7ff;
  if (biased !== 0) return biased - 1023;
  VIEW.setFloat64(0, from[i]! * 2 ** 64);
  return ((VIEW.getUint16(0) >> 4) & 0x7ff) - 1023 - 64;
}

/** Multiplies `into[i]` by `2^n`, in steps that each power of two a 64-bit float holds. */
function multiplyByPowerOfTwo(into: Float64Array, i: number, n: number): void {
  while (n > 1000) {
    into[i] = into[i]! * 2 ** 1000;
    n -= 1000;
  }
  while (n < -1000) {
    into[i] = into[i]! * 2 ** -1000;
    n += 1000;
  }
  VIEW.setUint32(0, (n + 1023) << 20);
  VIEW.setUint32(4, 0);
  into[i] = into[i]! * VIEW.getFloat64(0);
}

/**
 * Writes `a * b`, of `from[i]` and `by[j]`, into `into[k]`, and the rounding of that
 * product into `into[k + 1]`, so the two sum to it exactly: Dekker's product.
 */
function product(from: Float64Array, i: number, by: Float64Array, j: number, into: Float64Array, k: number): void {
  const a = from[i]!,
    b = by[j]!;
  const p = a * b;
  let c = SPLITTER * a;
  const aHigh = c - (c - a),
    aLow = a - aHigh;
  c = SPLITTER * b;
  const bHigh = c - (c - b),
    bLow = b - bHigh;
  into[k] = p;
  into[k + 1] = aLow * bLow - (p - aHigh * bHigh - aLow * bHigh - aHigh * bLow);
}

/** {@link product} of `from[i]` and a whole number `n` that is small enough to be exact. */
function productByWhole(from: Float64Array, i: number, n: number, into: Float64Array, k: number): void {
  const a = from[i]!,
    b = n;
  const p = a * b;
  let c = SPLITTER * a;
  const aHigh = c - (c - a),
    aLow = a - aHigh;
  c = SPLITTER * b;
  const bHigh = c - (c - b),
    bLow = b - bHigh;
  into[k] = p;
  into[k + 1] = aLow * bLow - (p - aHigh * bHigh - aLow * bHigh - aHigh * bLow);
}

/**
 * Adds `count` terms from `terms` exactly into an empty expansion at `sum[at]`, and gives its length:
 * components smallest first, none of them 0 but a lone one, whose last has the sign of the sum.
 */
function addExactly(terms: Float64Array, count: number, sum: Float64Array, at: number): number {
  let length = 0;
  for (let t = 0; t < count; t++) {
    let q = terms[t]!;
    if (q === 0) continue;
    let kept = 0;
    for (let i = 0; i < length; i++) {
      const e = sum[at + i]!;
      const s = q + e;
      const bVirtual = s - q;
      const error = q - (s - bVirtual) + (e - bVirtual);
      q = s;
      if (error !== 0) sum[at + kept++] = error;
    }
    if (q !== 0 || kept === 0) sum[at + kept++] = q;
    length = kept;
  }
  return length;
}

/** The sum of the expansion at `sum[at]` of `length` components, as a float: within a few roundings of it. */
function estimateOf(sum: Float64Array, at: number, length: number): number {
  let estimate = 0;
  for (let i = 0; i < length; i++) estimate += sum[at + i]!;
  return estimate;
}

/** The sign of the expansion at `sum[at]` of `length` components. */
function signOf(sum: Float64Array, at: number, length: number): number {
  if (length === 0) return 0;
  if (sum[at + length - 1]! > 0) return 1;
  return sum[at + length - 1]! < 0 ? -1 : 0;
}

/**
 * Prepares the lines of the triangle `p0`, `p1`, `p2`, with the vertex `centre`, if not -1, at the
 * centre of the target in place of `vec4(0)`. Gives the triangle's sign, 0 when its vertices lie in
 * one plane with the eye.
 */
export function prepareLines(
  lines: Lines,
  p0: Float64Array,
  p1: Float64Array,
  p2: Float64Array,
  centre: number,
): number {
  const { points, exponents, terms, coefficients, sizes, pending, sum } = lines;
  for (let k = 0; k < 3; k++) {
    const p = k === 0 ? p0 : k === 1 ? p1 : p2;
    const o = 4 * k;
    if (k === centre) {
      points[o] = points[o + 1] = points[o + 2] = 0;
      points[o + 3] = 1;
      exponents[k] = 0;
      continue;
    }
    pending[0] = Math.max(Math.abs(p[0]!), Math.abs(p[1]!), Math.abs(p[2]!), Math.abs(p[3]!));
    if (pending[0] === 0) return 0;
    const exponent = exponentOf(pending, 0);
    exponents[k] = exponent;
    for (let c = 0; c < 4; c++) {
      points[o + c] = p[c]!;
      multiplyByPowerOfTwo(points, o + c, -exponent);
    }
    // The largest component now lies from 1 to 2, so the least that counts is 2^-300 of it.
    const least =
      Math.max(Math.abs(points[o]!), Math.abs(points[o + 1]!), Math.abs(points[o + 2]!), Math.abs(points[o + 3]!)) *
      SMALLEST_COMPONENT;
    for (let c = 0; c < 4; c++) if (Math.abs(points[o + c]!) < least) points[o + c] = 0;
  }
  // Line e, opposite vertex e, through vertices i and j: a = yi wj - wi yj, b = wi xj - xi wj, c = xi yj - yi xj.
  for (let e = 0; e < 3; e++) {
    const i = 4 * ((e + 1) % 3),
      j = 4 * ((e + 2) % 3);
    for (let k = 0; k < 3; k++) {
      // The two components each product takes, of x (0), y (1) and w (3).
      const u = k === 0 ? 1 : k === 1 ? 3 : 0,
        v = k === 0 ? 3 : k === 1 ? 0 : 1;
      const t = (3 * e + k) * 4;
      product(points, i + u, points, j + v, terms, t);
      product(points, i + v, points, j + u, terms, t + 2);
      terms[t + 2] = -terms[t + 2]!;
      terms[t + 3] = -terms[t + 3]!;
      coefficients[3 * e + k] = terms[t]! + terms[t + 2]! + (terms[t + 1]! + terms[t + 3]!);
      sizes[3 * e + k] = Math.abs(terms[t]!) + Math.abs(terms[t + 2]!);
    }
  }
  // The sign of the triangle: line 0 at vertex 0, exactly.
  let n = 0;
  for (let k = 0; k < 3; k++)
    for (let q = 0; q < 4; q++) {
      product(terms, k * 4 + q, points, k === 2 ? 3 : k, pending, n);
      n += 2;
    }
  const determinant = addExactly(pending, n, sum, 0);
  const orientation = signOf(sum, 0, determinant);
  if (orientation === 0) return 0;
  const { plane } = lines;
  plane[3] = estimateOf(sum, 0, determinant);
  // The point's z along the ray is the sum of each vertex's z times its line: each coefficient summed exactly once.
  for (let k = 0; k < 3; k++) {
    n = 0;
    for (let e = 0; e < 3; e++)
      for (let q = 0; q < 4; q++) {
        product(terms, (3 * e + k) * 4 + q, points, 4 * e + 2, pending, n);
        n += 2;
      }
    plane[k] = estimateOf(sum, 0, addExactly(pending, n, sum, 0));
  }
  // An edge owns its centres when its inside lies to its right on the screen, or below it when horizontal.
  for (let e = 0; e < 3; e++) {
    const a = signOfTerms(lines, 3 * e * 4) * orientation,
      b = -signOfTerms(lines, (3 * e + 1) * 4) * orientation;
    lines.owns[e] = a > 0 || (a === 0 && b > 0) ? 1 : 0;
  }
  // The weight of a vertex for each unit of its line is its scale over the greatest scale among the
  // vertices whose lines are not 0 at the pixel: a smaller vertex, of a smaller w, weighs more.
  const { ratios } = lines;
  for (let mask = 1; mask < 8; mask++) {
    let least = 0x7fffffff;
    for (let e = 0; e < 3; e++) if (mask & (1 << e)) least = Math.min(least, exponents[e]!);
    for (let e = 0; e < 3; e++) {
      const r = mask * 3 + e;
      ratios[r] = mask & (1 << e) ? 1 : 0;
      if (mask & (1 << e)) multiplyByPowerOfTwo(ratios, r, Math.max(least - exponents[e]!, -1100));
    }
  }
  return orientation;
}

/** The sign of the sum of the four terms of a coefficient, from `terms[t]`. */
function signOfTerms(lines: Lines, t: number): number {
  const { terms, pending, sum } = lines;
  for (let q = 0; q < 4; q++) pending[q] = terms[t + q]!;
  return signOf(sum, 0, addExactly(pending, 4, sum, 0));
}

/** Writes the exact expansion of line `e`'s value at the ray `(X, Y, W)` and gives its length. */
function expandLine(lines: Lines, e: number, X: number, Y: number, W: number): number {
  const { terms, pending, expansions } = lines;
  let n = 0;
  for (let k = 0; k < 3; k++) {
    const whole = k === 0 ? X : k === 1 ? Y : W;
    for (let q = 0; q < 4; q++) {
      productByWhole(terms, (3 * e + k) * 4 + q, whole, pending, n);
      n += 2;
    }
  }
  const length = addExactly(pending, n, expansions, e * EXPANSION);
  lines.lengths[e] = length;
  return length;
}

/** Writes the part of each line's float value and of its terms' sizes that the row of pixels at `Y` shares. */
export function prepareRow(lines: Lines, Y: number, W: number): void {
  const { coefficients: c, sizes: m, rows } = lines;
  for (let e = 0; e < 3; e++) {
    rows[2 * e] = c[3 * e + 1]! * Y + c[3 * e + 2]! * W;
    rows[2 * e + 1] = m[3 * e + 1]! * Math.abs(Y) + m[3 * e + 2]! * W;
  }
}

/**
 * Tests line `e` at the ray `(X, Y, W)`, in the row {@link prepareRow} prepared: writes its value into
 * `values`, and gives its sign, exactly. The float value is kept when it is more than
 * {@link LINE_FILTER} of its terms' sizes; any other is summed exactly.
 */
export function lineSign(lines: Lines, e: number, X: number, Y: number, W: number): number {
  const { coefficients: c, sizes: m, rows, values } = lines;
  const value = c[3 * e]! * X + rows[2 * e]!;
  const size = m[3 * e]! * Math.abs(X) + rows[2 * e + 1]!;
  if (value > LINE_FILTER * size || value < -LINE_FILTER * size) {
    values[e] = value;
    return value > 0 ? 1 : -1;
  }
  return exactLineSign(lines, e, X, Y, W);
}

/** The sign of line `e` at the ray `(X, Y, W)`, summed exactly; writes the float sum of its expansion into `values`. */
function exactLineSign(lines: Lines, e: number, X: number, Y: number, W: number): number {
  const { values, expansions } = lines;
  const length = expandLine(lines, e, X, Y, W);
  values[e] = estimateOf(expansions, e * EXPANSION, length);
  return signOf(expansions, e * EXPANSION, length);
}

/**
 * The depth at the ray `(X, Y, W)` whose lines {@link lineSign} has tested, held to 0 to 1, where coverage
 * put it: the point's z by the triangle's {@link Lines.plane}, over its w. The float value of the plane is
 * kept when its bound of rounding is within {@link DEPTH_FILTER} of that w; any other z is summed exactly.
 */
export function depthAt(lines: Lines, X: number, Y: number, W: number): number {
  const { plane } = lines;
  const x = plane[0]! * X,
    y = plane[1]! * Y,
    w = plane[2]! * W;
  const divisor = plane[3]! * W;
  let z = x + y + w;
  if (PLANE_ROUNDING * (Math.abs(x) + Math.abs(y) + Math.abs(w)) > DEPTH_FILTER * Math.abs(divisor))
    z = estimateOf(lines.sum, 0, expandPoint(lines, 2, X, Y, W));
  return Math.min(1, Math.max(0, z / divisor));
}

/**
 * Whether every vertex lies at a depth from 0 to 1 in front of the eye: z from 0 to w, and w above 0.
 * A point inside the triangle's three lines is then a sum of its vertices with weights of one sign, so
 * it lies at such a depth too, and needs no {@link pointSign}.
 */
export function liesInDepthRange(lines: Lines): boolean {
  const { points } = lines;
  for (let k = 0; k < 3; k++) {
    const z = points[4 * k + 2]!,
      w = points[4 * k + 3]!;
    if (!(w > 0 && z >= 0 && z <= w)) return false;
  }
  return true;
}

/**
 * The exact sign, at the ray `(X, Y, W)` whose lines {@link lineSign} has tested, of the point's z
 * (`component` 2) or its w minus its z (-1): each the sum over the vertices of that component times
 * the vertex's line. The same sum of w is the triangle's sign times `W`, so it needs no test.
 */
export function pointSign(lines: Lines, component: number, X: number, Y: number, W: number): number {
  const { points, sizes, rows, values, sum } = lines;
  // Each vertex's component c, and the size of its terms, which for w minus z is |w| + |z|.
  const z = component === -1 ? 2 : component,
    w = component === -1 ? 3 : component,
    minus = component === -1 ? 1 : 0;
  const c0 = points[w]! - minus * points[z]!,
    c1 = points[4 + w]! - minus * points[4 + z]!,
    c2 = points[8 + w]! - minus * points[8 + z]!;
  const m0 = Math.abs(points[w]!) + minus * Math.abs(points[z]!),
    m1 = Math.abs(points[4 + w]!) + minus * Math.abs(points[4 + z]!),
    m2 = Math.abs(points[8 + w]!) + minus * Math.abs(points[8 + z]!);
  const value = c0 * values[0]! + c1 * values[1]! + c2 * values[2]!;
  // Each line's bound of rounding is at least 2^-48 of its value too, which covers the rounding of this sum.
  const absX = Math.abs(X);
  const bound =
    m0 * (LINE_ROUNDING * (sizes[0]! * absX + rows[1]!)) +
    m1 * (LINE_ROUNDING * (sizes[3]! * absX + rows[3]!)) +
    m2 * (LINE_ROUNDING * (sizes[6]! * absX + rows[5]!));
  if (value > bound) return 1;
  if (value < -bound) return -1;
  // A bound of 0 leaves every component 0, as a line of a triangle that is not flat is not 0: so is the sum.
  if (bound === 0) return 0;
  return signOf(sum, 0, expandPoint(lines, component, X, Y, W));
}

/**
 * Writes into `sum` the exact expansion, at the ray `(X, Y, W)`, of the point's z (`component` 2) or
 * its w minus its z (-1): the sum over the vertices of that component times the vertex's line. Gives
 * its length.
 */
function expandPoint(lines: Lines, component: number, X: number, Y: number, W: number): number {
  const { points, expansions, pending, sum } = lines;
  // Each line's expansion first, as an expansion takes `pending` for its own terms.
  for (let e = 0; e < 3; e++) expandLine(lines, e, X, Y, W);
  let n = 0;
  for (let e = 0; e < 3; e++) {
    const length = lines.lengths[e]!;
    for (let i = 0; i < length; i++) {
      const at = e * EXPANSION + i;
      if (component === -1) {
        product(expansions, at, points, 4 * e + 3, pending, n);
        product(expansions, at, points, 4 * e + 2, pending, n + 2);
        pending[n + 2] = -pending[n + 2]!;
        pending[n + 3] = -pending[n + 3]!;
        n += 4;
      } else {
        product(expansions, at, points, 4 * e + component, pending, n);
        n += 2;
      }
    }
  }
  return addExactly(pending, n, sum, 0);
}
