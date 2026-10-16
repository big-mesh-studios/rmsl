/** A call of a built-in function as the JavaScript run of a program made it: its arguments and its value. */
export interface BuiltinCall {
  name: string;
  args: number[];
  value: number;
}

const F32_MIN_NORMAL = 2 ** -126;

const f32Bits = new DataView(new ArrayBuffer(4));

/**
 * The distance from `x`, rounded to 32 bits, to the next 32-bit float away
 * from zero. A correctly rounded step may give either neighbouring float, so
 * the bounds below count its rounding as one of these.
 */
export function f32Ulp(x: number): number {
  const a = Math.abs(Math.fround(x));
  if (!Number.isFinite(a)) return Infinity;
  if (a < F32_MIN_NORMAL) return 2 ** -149;
  f32Bits.setFloat32(0, a);
  const exponent = ((f32Bits.getUint32(0) >>> 23) & 0xff) - 127;
  return 2 ** (exponent - 23);
}

function isNormal(x: number): boolean {
  return Number.isFinite(x) && Math.abs(x) >= F32_MIN_NORMAL;
}

/** The widest `n / d` moves when `n` is off by up to `en` and `d` by up to `ed`, or Infinity when `d` may reach zero. */
function quotientError(n: number, d: number, en: number, ed: number): number {
  if (Math.abs(d) <= ed) return Infinity;
  const q = n / d;
  let widest = 0;
  for (const nn of [n - en, n + en]) {
    for (const dd of [d - ed, d + ed]) widest = Math.max(widest, Math.abs(nn / dd - q));
  }
  return widest;
}

/** The widest `log` moves when its argument `a` is off by up to `ea`, or Infinity when the argument may reach zero. */
function logOfPerturbed(a: number, ea: number): number {
  return ea >= a ? Infinity : -Math.log1p(-ea / a);
}

function atan2Bound(y: number, x: number, value: number): number {
  y = Math.fround(y);
  const xSize = Math.abs(Math.fround(x));
  if (!isNormal(y) || xSize < F32_MIN_NORMAL || xSize > 2 ** 126) return Infinity;
  return 4096 * f32Ulp(value);
}

function sinCosBound(x: number): number {
  return Math.abs(Math.fround(x)) <= Math.PI ? 2 ** -11 : Infinity;
}

function expBound(x: number, value: number): number {
  return (3 + 2 * Math.abs(x)) * f32Ulp(value);
}

function logBound(x: number, value: number): number {
  const argument = Math.fround(x);
  if (!(argument > 0)) return Infinity;
  return argument >= 0.5 && argument <= 2 ? 2 ** -21 : 3 * f32Ulp(value);
}

/** `sqrt(1 - x * x)`, and its error: the rounding of each step, and `sqrt` as `1 / inverseSqrt`. */
function complementRoot(x: number): { root: number; error: number } {
  const square = x * x;
  const difference = 1 - square;
  const root = Math.sqrt(difference);
  const differenceError = f32Ulp(square) + f32Ulp(difference);
  return { root, error: differenceError / (2 * root) + 4.5 * f32Ulp(root) };
}

/** The bound of `asin` or `acos` through the `atan2` that defines it, with the error of its `sqrt` argument carried through. */
function arcBound(x: number, value: number, floor: number, rootIsY: boolean): number {
  const { root, error } = complementRoot(x);
  const inherited = rootIsY ? atan2Bound(root, x, value) : atan2Bound(x, root, value);
  // ∂atan2/∂ of either argument is the other over x² + root², which is 1 here.
  return Math.max(floor, inherited + Math.abs(x) * error);
}

/** The error of `exp(x) ∓ exp(-x)` halved, the formula of `sinh` and `cosh`. */
function hyperbolicBound(x: number, sign: 1 | -1): number {
  const plus = Math.exp(x);
  const minus = Math.exp(-x);
  return 0.5 * (expBound(x, plus) + expBound(-x, minus) + f32Ulp(plus + sign * minus));
}

/** The bound of `log(x + sqrt(x * x ± 1))`, the formula of `asinh` and `acosh`. */
function inverseHyperbolicBound(x: number, sign: 1 | -1, value: number): number {
  const square = x * x;
  const radicand = square + sign;
  const root = Math.sqrt(radicand);
  const radicandError = f32Ulp(square) + f32Ulp(radicand);
  const rootError = radicandError / (2 * root) + 4.5 * f32Ulp(root);
  const argument = x + root;
  const argumentError = rootError + f32Ulp(argument);
  return logBound(argument, value) + logOfPerturbed(argument, argumentError);
}

/** A GPU target, whose specification decides the error it may give a built-in call. */
export type GpuTarget = "wgsl" | "glsl";

/** The built-in functions whose precision GLSL ES 3.00 leaves undefined. */
const UNDEFINED_ON_GLSL = new Set(["sin", "cos", "tan", "asin", "acos", "atan", "atan2", "asinh", "acosh", "atanh"]);

/**
 * The error the specification of `target` allows one call of a built-in
 * function, as an absolute error at its value: WGSL's table of the accuracy of
 * `f32` built-in functions, or GLSL ES 3.00's at `highp`. Infinity where the
 * specification gives the call no bound.
 */
export function builtinCallBound(call: BuiltinCall, target: GpuTarget = "wgsl"): number {
  const { name, args, value } = call;
  const x = args[0]!;
  if (target === "glsl" && UNDEFINED_ON_GLSL.has(name)) return Infinity;
  switch (name) {
    case "atan":
      return 4096 * f32Ulp(value);
    case "atan2":
      return atan2Bound(x, args[1]!, value);
    case "asin":
      return arcBound(x, value, 6.81e-5, false);
    case "acos":
      return arcBound(x, value, 6.77e-5, true);
    case "sin":
    case "cos":
      return sinCosBound(x);
    case "tan": {
      if (sinCosBound(x) === Infinity) return Infinity;
      return quotientError(Math.sin(x), Math.cos(x), 2 ** -11, 2 ** -11) + 2.5 * f32Ulp(value);
    }
    case "exp":
    case "exp2":
      return expBound(x, value);
    case "log":
    case "log2":
      return logBound(x, value);
    case "pow": {
      const y = args[1]!;
      const log = Math.log2(x);
      const product = y * log;
      const productError = Math.abs(y) * logBound(x, log) + f32Ulp(product);
      return value * (2 ** productError - 1) + expBound(product, value);
    }
    case "sinh":
      return hyperbolicBound(x, -1);
    case "cosh":
      return hyperbolicBound(x, 1);
    case "tanh": {
      const inherited =
        quotientError(Math.sinh(x), Math.cosh(x), hyperbolicBound(x, -1), hyperbolicBound(x, 1)) + 2.5 * f32Ulp(value);
      return target === "glsl" ? inherited : Math.max(1e-5, inherited);
    }
    case "asinh":
      return inverseHyperbolicBound(x, 1, value);
    case "acosh":
      return inverseHyperbolicBound(x, -1, value);
    case "atanh": {
      const numerator = 1 + x;
      const denominator = 1 - x;
      const quotient = numerator / denominator;
      const quotientBound =
        quotientError(numerator, denominator, f32Ulp(numerator), f32Ulp(denominator)) + 2.5 * f32Ulp(quotient);
      return 0.5 * (logBound(quotient, 2 * value) + logOfPerturbed(quotient, quotientBound));
    }
    default:
      return 0;
  }
}

/** The built-in functions of `Math` whose WGSL error can exceed a few units in the last place. */
const TRACKED = [
  "atan",
  "atan2",
  "asin",
  "acos",
  "sin",
  "cos",
  "tan",
  "exp",
  "log",
  "log2",
  "pow",
  "sinh",
  "cosh",
  "tanh",
  "asinh",
  "acosh",
  "atanh",
] as const;

/** What a run of a program through `recordingMath` records, and the one call it moves. */
export interface CallRecording {
  calls: BuiltinCall[];
  /** Whether the program calls `pow`, which tells a `Math.pow(2, x)` of `pow` from one of `exp2`. */
  powIsCalled: boolean;
  /** The call, by its place in the run, whose value is moved by `delta`. */
  perturb?: { index: number; delta: number };
}

/**
 * A stand-in for `Math` that records each call of a tracked function into
 * `recording.calls`, and moves the value of the call `recording.perturb`
 * names. The JavaScript target writes `exp2(x)` as `Math.pow(2, x)`, so a
 * `pow` with a base of 2 reads as `exp2` unless the program calls `pow`.
 */
export function recordingMath(recording: CallRecording): Math {
  const math = Object.create(Math) as Math & Record<string, (...args: number[]) => number>;
  for (const name of TRACKED) {
    const original = (Math as unknown as Record<string, (...args: number[]) => number>)[name]!;
    math[name] = (...args: number[]) => {
      const value = original(...args);
      const call =
        name === "pow" && args[0] === 2 && !recording.powIsCalled
          ? { name: "exp2", args: [args[1]!], value }
          : { name, args, value };
      const { perturb, calls } = recording;
      const moved = perturb !== undefined && perturb.index === calls.length ? value + perturb.delta : value;
      calls.push(call);
      return moved;
    };
  }
  return math;
}

/**
 * Difference allowed between a result of `target` and the CPU one, for each
 * component: `baseTolerance` of the component's size, widened by the sum over
 * the calls of the most the component moves when `run` gives that call its
 * value plus or minus its bound on `target`. Null when a call has no bound
 * there, or when moving one gives a component that is not finite.
 *
 * `run` runs the program, moving the value of the call `perturb` names, and
 * returns the components of its result.
 */
export function carriedTolerances(
  components: readonly number[],
  calls: readonly BuiltinCall[],
  run: (perturb: { index: number; delta: number }) => readonly number[],
  baseTolerance: (magnitude: number) => number,
  target: GpuTarget,
): number[] | null {
  const bounds = calls.map((call) => builtinCallBound(call, target));
  // A bound that is NaN, as `pow(0, 2)` and `asin(2)` give, is no bound either.
  if (bounds.some((bound) => !(bound < Infinity))) return null;
  const widening = components.map(() => 0);
  bounds.forEach((bound, index) => {
    if (bound === 0) return;
    const shifts = components.map(() => 0);
    for (const delta of [bound, -bound]) {
      const moved = run({ index, delta });
      components.forEach((component, i) => {
        const shift = Math.abs(moved[i]! - component);
        shifts[i] = Math.max(shifts[i]!, Number.isFinite(moved[i]!) ? shift : Infinity);
      });
    }
    shifts.forEach((shift, i) => (widening[i] = widening[i]! + shift));
  });
  if (widening.some((w) => w === Infinity)) return null;
  return components.map((component, i) => baseTolerance(component) + widening[i]!);
}
