import { describe, it, expect, afterAll } from "vitest";
import {
  runWGSL,
  closeEvaluators,
  floatTolerance,
  EVALUATION_SKIPPED,
  evaluateJS,
  evaluateGLSL,
  evaluateWGSL,
  evaluateWASM,
  accuracyOf,
  type Build,
} from "./shader-eval";
import { builtinCallBound, f32Ulp, type GpuTarget } from "./float-accuracy";
import {
  vec2,
  vec3,
  mat2,
  mat3,
  sin,
  cos,
  tan,
  atan,
  asin,
  acos,
  exp,
  exp2,
  log,
  log2,
  pow,
  sinh,
  cosh,
  tanh,
  asinh,
  acosh,
  atanh,
  determinant,
  inverse,
  step,
  clamp,
  abs,
  select,
  float,
  Fn,
  type Node,
} from "../rmsl";

afterAll(async () => {
  await closeEvaluators();
});

describe("float tolerance", () => {
  // A 32-bit float carries 24 bits of mantissa, so the gap between neighbouring
  // representable values at |x| is about |x| * 2^-23. A tolerance below that is
  // asking two independent GPU implementations to agree bit for bit, which they
  // are under no obligation to do — pow is commonly evaluated as
  // exp2(y * log2(x)) and lands a unit or two either side.
  const ulp = (x: number) => Math.abs(x) * Math.pow(2, -23);
  /**
   * @canon spec-the-float-tolerance-allows-a-few-units-in-the-last-place
   */
  it("allows at least one unit in the last place, at every magnitude", () => {
    for (const magnitude of [1, 10, 1024, 65536, 1e6]) {
      expect(floatTolerance(magnitude), `magnitude ${magnitude}`).toBeGreaterThan(ulp(magnitude));
    }
  });

  /**
   * The case that motivated this: pow(2, 10) is 1024, where one unit in the
   * last place is 1.22e-4 — larger than the flat 1e-5 the harness used to
   * allow, so a correct backend could fail.
   *
   * @canon spec-the-float-tolerance-allows-a-few-units-in-the-last-place
   */
  it("is looser than a unit in the last place at 1024", () => {
    expect(floatTolerance(1024)).toBeGreaterThan(1.22e-4);
  });

  /**
   * Near zero the relative gap collapses, so there has to be a floor.
   *
   * @canon spec-the-float-tolerance-allows-a-few-units-in-the-last-place
   */
  it("stays usable near zero", () => {
    expect(floatTolerance(0)).toBeGreaterThan(0);
    expect(floatTolerance(0.001)).toBeGreaterThan(0);
  });

  /**
   * Loose enough to survive a driver, tight enough to catch a real mistake:
   * it must not admit an off-by-one, and 1 versus 2 is the smallest of those.
   *
   * @canon spec-the-float-tolerance-allows-a-few-units-in-the-last-place
   */
  it("stays tight enough to catch a wrong answer", () => {
    expect(floatTolerance(1)).toBeLessThan(0.5);
    expect(floatTolerance(1024)).toBeLessThan(0.5);
  });
});

/** How much a program's built-in calls widen the tolerance of its first component. */
function widening(build: Build, args: number[], target: GpuTarget = "wgsl"): number {
  const { js, tolerances } = accuracyOf(build, args);
  return tolerances[target]![0]! - floatTolerance(typeof js === "number" ? js : js[0]!);
}

const unbounded = (build: Build, args: number[]) => accuracyOf(build, args).unbounded;

/** `actual` within a billionth of `expected`, for a widening that the carry computes by a difference of floats. */
function expectNear(actual: number, expected: number) {
  expect(Math.abs(actual - expected), `${actual} against ${expected}`).toBeLessThan(Math.abs(expected) * 1e-9);
}

describe("tolerance of a built-in call", () => {
  /**
   * Dawn on lavapipe gave 0.7853949069976807 for `atan2(1, 1)`, 55 units in
   * the last place from π/4, which a tolerance of a millionth refused.
   *
   * @canon spec-atan-and-atan2-widen-the-tolerance-by-4096-units-in-the-last-place
   */
  it("widens by 4096 units in the last place for atan and atan2", () => {
    expectNear(
      widening((a, b) => atan(a, b), [1, 1]),
      4096 * 2 ** -24,
    );
    expectNear(
      widening((a) => atan(a), [3]),
      4096 * 2 ** -23,
    );
    const { js, tolerances } = accuracyOf((a, b) => atan(a, b), [1, 1]);
    expect(Math.abs(0.7853949069976807 - (js as number))).toBeLessThan(tolerances.wgsl![0]!);
    expect(unbounded((a, b) => atan(a, b), [1, 1])).toBe(false);
    expect(unbounded((a, b) => atan(a, b), [0, 1])).toBe(true);
    expect(unbounded((a, b) => atan(a, b), [1, 0])).toBe(true);
    expect(unbounded((a, b) => atan(a, b), [1, 2 ** 127])).toBe(true);
  });

  /**
   * Near 0, `asin` and `acos` take their absolute errors, which differ; near
   * π/4 the 4096 units of `atan2` are wider.
   *
   * @canon spec-asin-and-acos-widen-the-tolerance-by-the-worse-of-their-absolute-error-and-atan2s
   */
  it("widens by the worse of the absolute error of asin and acos and that of atan2", () => {
    const asinSmall = widening((a) => asin(a), [0.01]);
    expect(asinSmall).toBeGreaterThan(6.81e-5 * (1 - 1e-9));
    expect(asinSmall).toBeLessThan(6.81e-5 + 1e-6);
    const acosSmall = widening((a) => acos(a), [0.99]);
    expect(acosSmall).toBeGreaterThan(6.77e-5 * (1 - 1e-9));
    expect(acosSmall).toBeLessThan(6.81e-5 * (1 - 1e-9));
    const x = 0.7;
    const root = Math.sqrt(1 - x * x);
    const rootError = (f32Ulp(x * x) + f32Ulp(1 - x * x)) / (2 * root) + 4.5 * f32Ulp(root);
    expectNear(
      widening((a) => asin(a), [x]),
      4096 * f32Ulp(Math.asin(x)) + x * rootError,
    );
    expect(unbounded((a) => asin(a), [0])).toBe(true);
    expect(unbounded((a) => acos(a), [1])).toBe(true);
    expect(unbounded((a) => acos(a), [0])).toBe(true);
    expect(unbounded((a) => asin(a), [0.5])).toBe(false);
  });

  /**
   * @canon spec-sin-and-cos-widen-the-tolerance-by-2-to-the-minus-11
   */
  it("widens by 2^-11 for sin and cos from -π to π, and has no bound outside", () => {
    expectNear(
      widening((a) => sin(a), [0.5]),
      2 ** -11,
    );
    expectNear(
      widening((a) => cos(a), [-3.14159]),
      2 ** -11,
    );
    expect(unbounded((a) => sin(a), [3.2])).toBe(true);
    // π rounded to 32 bits lies above π.
    expect(unbounded((a) => sin(a), [Math.PI])).toBe(true);
    expect(unbounded((a) => cos(a), [-3.2])).toBe(true);
  });

  /**
   * A GPU computes 3.0004 + 0.1411925918 as 3.14159274, outside [-π, π], so
   * WGSL gives `sin` of it no bound; computed in 64 bits and rounded once, the
   * sum is 3.14159250, inside.
   *
   * @canon bug-the-harness-checks-the-range-of-sin-on-the-cpus-argument
   */
  it.fails("has no bound for sin of an argument a GPU computes outside [-π, π]", () => {
    expect(unbounded((a, b) => sin(a.add(b)), [3.0004, 0.1411925918])).toBe(true);
  });

  /**
   * @canon spec-tan-widens-the-tolerance-by-the-error-of-sin-over-cos
   */
  it("widens tan by what sin over cos moves when each is off by 2^-11", () => {
    const s = Math.sin(1);
    const c = Math.cos(1);
    const firstOrder = (2 ** -11 * (Math.abs(s) + Math.abs(c))) / (c * c);
    const w = widening((a) => tan(a), [1]);
    expect(w).toBeGreaterThan(firstOrder);
    expect(w).toBeLessThan(firstOrder * 1.01);
    expect(unbounded((a) => tan(a), [1.5707])).toBe(true);
    expect(unbounded((a) => tan(a), [4])).toBe(true);
  });

  /**
   * `exp2(3)` takes the bound of `exp2`, not that of `pow(2, 3)`, though the
   * JavaScript target writes it as `Math.pow(2, 3)`.
   *
   * @canon spec-exp-or-exp2-without-pow-widens-the-tolerance-by-3-plus-twice-the-argument
   */
  it("widens by 3 + 2|x| units in the last place for exp and exp2", () => {
    expectNear(
      widening((a) => exp(a), [2]),
      7 * 2 ** -21,
    );
    expectNear(
      widening((a) => exp2(a), [3]),
      9 * 2 ** -20,
    );
    expectNear(
      widening((a) => exp2(a), [-3]),
      9 * 2 ** -26,
    );
  });

  /**
   * JavaScript computes `exp2(a)` as `Math.pow(2, a)`, the call `pow(2, a)`
   * makes, so beside a `pow` it takes the bound of `pow(2, a)`.
   *
   * @canon exception-exp2-takes-the-bound-of-pow-in-a-program-that-calls-pow
   */
  it("widens exp2 by the bound of pow(2, x) in a program that calls pow", () => {
    const exp2Of3 = builtinCallBound({ name: "exp2", args: [3], value: 8 });
    const powOf2And3 = builtinCallBound({ name: "pow", args: [2, 3], value: 8 });
    const powOfQuarter = builtinCallBound({ name: "pow", args: [0.25, 2], value: 0.0625 });
    expect(powOf2And3).toBeGreaterThan(exp2Of3);
    expectNear(
      widening((a, b) => exp2(a).add(pow(b, 2)), [3, 0.25]),
      powOf2And3 + powOfQuarter,
    );
    expectNear(
      widening((a) => exp2(a), [3]),
      exp2Of3,
    );
  });

  /**
   * @canon spec-log-and-log2-widen-the-tolerance-by-their-absolute-or-relative-bound
   */
  it("widens by 2^-21 for log and log2 from 0.5 to 2, and 3 units in the last place elsewhere", () => {
    expectNear(
      widening((a) => log(a), [1.5]),
      2 ** -21,
    );
    expectNear(
      widening((a) => log2(a), [0.5]),
      2 ** -21,
    );
    expectNear(
      widening((a) => log2(a), [2]),
      2 ** -21,
    );
    expectNear(
      widening((a) => log(a), [4]),
      3 * 2 ** -23,
    );
    expectNear(
      widening((a) => log2(a), [0.25]),
      3 * 2 ** -22,
    );
  });

  /**
   * `pow(2, 10)` is 1024: `log2(2)` may be off by 2^-21, and `exp2` carries
   * ten times that to the result.
   *
   * @canon spec-pow-widens-the-tolerance-by-the-error-of-exp2-of-y-times-log2-x
   */
  it("widens pow by the error of exp2 of y times log2 x", () => {
    const w = widening((a, b) => pow(a, b), [2, 10]);
    const carried = 1024 * Math.LN2 * (10 * 2 ** -21 + f32Ulp(10));
    expect(w).toBeGreaterThan(carried + 23 * f32Ulp(1024));
    expect(w).toBeLessThan((carried + 23 * f32Ulp(1024)) * 1.001);
    expect(widening((a, b) => pow(a, b), [2, 3])).toBeGreaterThan(widening((a) => exp2(a), [3]));
    expect(unbounded((a, b) => pow(a, b), [0, 2])).toBe(true);
    expect(unbounded((a, b) => pow(a, b), [-8, 3])).toBe(true);
    // The NaN of the bound, not of the result, makes it unbounded: step hides it.
    expect(unbounded((a) => step(-1, pow(a, 2)), [0])).toBe(true);
  });

  /**
   * Near 0, `exp(x)` and `exp(-x)` nearly cancel, so `sinh` may be off by
   * far more than units in the last place of its value.
   *
   * @canon spec-sinh-and-cosh-widen-the-tolerance-by-the-error-of-their-exponentials
   */
  it("widens sinh and cosh by the error of their exponentials", () => {
    const x = 0.001;
    const plus = Math.exp(x);
    const minus = Math.exp(-x);
    const exponentials = (3 + 2 * x) * (f32Ulp(plus) + f32Ulp(minus));
    expectNear(
      widening((a) => sinh(a), [x]),
      0.5 * (exponentials + f32Ulp(plus - minus)),
    );
    expectNear(
      widening((a) => cosh(a), [x]),
      0.5 * (exponentials + f32Ulp(plus + minus)),
    );
    expect(widening((a) => sinh(a), [x])).toBeGreaterThan(100 * f32Ulp(Math.sinh(x)));
  });

  /**
   * @canon spec-tanh-widens-the-tolerance-by-the-worse-of-1e-5-and-the-error-of-sinh-over-cosh
   */
  it("widens tanh by the worse of 1e-5 and the error of sinh over cosh", () => {
    expectNear(
      widening((a) => tanh(a), [0.5]),
      1e-5,
    );
    expect(widening((a) => tanh(a), [50])).toBeGreaterThan(1e-5);
    expect(unbounded((a) => tanh(a), [90])).toBe(true);
  });

  /**
   * `asinh(-100)` takes `log` of 100.005 - 100, which loses most of its bits.
   *
   * @canon spec-the-inverse-hyperbolic-functions-widen-the-tolerance-by-the-error-of-their-logarithm
   */
  it("widens the inverse hyperbolic functions by the error of their logarithm", () => {
    expect(widening((a) => asinh(a), [-100])).toBeGreaterThan(5e-3);
    // log(x + sqrt(x * x ± 1)), each correctly rounded step off by a unit in
    // the last place, sqrt as 1 / inverseSqrt off by 4.5, then the bound of log.
    const viaLog = (x: number, sign: 1 | -1, value: number) => {
      const radicand = x * x + sign;
      const root = Math.sqrt(radicand);
      const rootError = (f32Ulp(x * x) + f32Ulp(radicand)) / (2 * root) + 4.5 * f32Ulp(root);
      const argument = x + root;
      const logBound = argument >= 0.5 && argument <= 2 ? 2 ** -21 : 3 * f32Ulp(value);
      return logBound - Math.log1p(-(rootError + f32Ulp(argument)) / argument);
    };
    expectNear(
      widening((a) => asinh(a), [3]),
      viaLog(3, 1, Math.asinh(3)),
    );
    expectNear(
      widening((a) => acosh(a), [3]),
      viaLog(3, -1, Math.acosh(3)),
    );
    // atanh(0.5) is log((1 + x) / (1 - x)) / 2: the quotient 3 moves most when
    // 1.5 and 0.5 are each off by a unit in the last place the opposite way.
    const quotientError = (1.5 + f32Ulp(1.5)) / (0.5 - f32Ulp(0.5)) - 3 + 2.5 * f32Ulp(3);
    expectNear(
      widening((a) => atanh(a), [0.5]),
      0.5 * (3 * f32Ulp(Math.log(3)) - Math.log1p(-quotientError / 3)),
    );
    expect(widening((a) => asinh(a), [0.5])).toBeGreaterThanOrEqual(2 ** -21);
    expect(widening((a) => acosh(a), [3])).toBeGreaterThanOrEqual(3 * f32Ulp(Math.acosh(3)));
    const atanhHalf = widening((a) => atanh(a), [0.5]);
    expect(atanhHalf).toBeGreaterThan(0.5 * (3 * f32Ulp(Math.log(3)) + (2.5 * f32Ulp(3)) / 3));
    expect(atanhHalf).toBeLessThan(4e-7);
    expect(unbounded((a) => asinh(a), [-1e4])).toBe(true);
    expect(unbounded((a) => acosh(a), [0.5])).toBe(true);
    expect(unbounded((a) => step(0.5, acosh(a)), [0.5])).toBe(true);
    // A WGSL acosh that rounds each step of log(x + sqrt(x * x - 1)) away from
    // the exact result, as WGSL allows, stays within the tolerance near 1.
    for (const x of [Math.fround(1.0001), Math.fround(1.00001)]) {
      const up = (v: number) => v + f32Ulp(v);
      const square = up(Math.fround(x * x));
      const radicand = up(Math.fround(square - 1));
      const root = Math.sqrt(radicand) + 4.5 * f32Ulp(Math.sqrt(radicand));
      const conforming = Math.log(up(Math.fround(x + root))) + 2 ** -21;
      const { js, tolerances } = accuracyOf((a) => acosh(a), [x]);
      expect(Math.abs(conforming - (js as number)), `acosh(${x})`).toBeLessThan(tolerances.wgsl![0]!);
    }
    expect(unbounded((a) => atanh(a), [1])).toBe(true);
  });

  /**
   * Each call's bound moves the result as the rest of the program carries it:
   * a sum adds the bounds of its calls, a product scales them, and a
   * component that does not read a call is not widened by it.
   *
   * @canon spec-each-call-widens-the-tolerance-by-what-its-error-moves-the-result
   */
  it("widens by what each call's error moves the result, summed over the calls", () => {
    expectNear(
      widening((a, b) => sin(a).add(atan(b, b)), [0.5, 1]),
      2 ** -11 + 4096 * 2 ** -24,
    );
    expectNear(
      widening((a, b) => atan(a, b).mul(2), [1, 1]),
      2 * 4096 * 2 ** -24,
    );
    expectNear(
      widening((a, b) => atan(a, b).mul(0.25), [1, 1]),
      0.25 * 4096 * 2 ** -24,
    );
    expectNear(
      widening((a) => atan(a).mul(1000).sub(785), [1]),
      1000 * 4096 * 2 ** -24,
    );
    expectNear(
      widening((a) => sin(a).add(5), [1e-6]),
      2 ** -11,
    );
    // A component that is not finite whatever the move stays bounded, compared by equality.
    expect(unbounded((a) => vec2(sin(a), a.div(0)), [0.5])).toBe(false);
    // One that a move turns finite has no bound: 1 / sin(0) is infinite, 1 / sin(2^-11) is not.
    expect(unbounded((a) => float(1).div(sin(a)), [0])).toBe(true);
    const { js, tolerances } = accuracyOf((a) => vec2(sin(a), a.mul(4)), [0.5]);
    expect(tolerances.wgsl![1]).toBe(floatTolerance((js as Float64Array)[1]!));
    // sin(0) moved down by 2^-11 takes the square root of a negative number.
    expect(unbounded((a) => sin(a).sqrt(), [0])).toBe(true);
    expect(unbounded((a) => sin(a).add(1).sqrt(), [0])).toBe(false);
    // A result that is not finite without any call is no unbounded call.
    expect(unbounded((a) => a.div(0).add(a), [1])).toBe(false);
  });

  /**
   * @canon spec-a-call-glsl-es-bounds-as-wgsl-does-widens-the-glsl-tolerance-alike
   */
  it("widens the GLSL tolerance as the WGSL one for the built-ins GLSL ES bounds alike", () => {
    const programs: [Build, number[]][] = [
      [(a) => exp(a), [2]],
      [(a) => exp2(a), [3]],
      [(a) => log(a), [1.5]],
      [(a) => log2(a), [4]],
      [(a, b) => pow(a, b), [2, 10]],
      [(a) => sinh(a), [0.001]],
      [(a) => cosh(a), [2]],
    ];
    for (const [build, args] of programs) {
      const { tolerances } = accuracyOf(build, args);
      expect(tolerances.glsl, `${build}`).toEqual(tolerances.wgsl);
      expect(widening(build, args, "glsl"), `${build}`).toBeGreaterThan(0);
    }
  });

  /**
   * @canon spec-tanh-on-glsl-widens-the-tolerance-without-the-floor-of-1e-5
   */
  it("widens tanh on GLSL by the error of sinh over cosh alone", () => {
    const glsl = widening((a) => tanh(a), [0.5], "glsl");
    expect(glsl).toBeGreaterThan(0);
    expect(glsl).toBeLessThan(1e-6);
    expectNear(
      widening((a) => tanh(a), [0.5]),
      1e-5,
    );
    expectNear(
      widening((a) => tanh(a), [50], "glsl"),
      widening((a) => tanh(a), [50]),
    );
  });

  /**
   * @canon spec-the-components-a-call-without-a-glsl-es-precision-moves-are-not-compared-on-glsl
   */
  it("leaves open on GLSL the components a call GLSL ES gives no precision moves", () => {
    const open: [Build, number[]][] = [
      [(a) => sin(a), [0.5]],
      [(a) => cos(a), [0.5]],
      [(a) => tan(a), [0.5]],
      [(a) => asin(a), [0.5]],
      [(a) => acos(a), [0.5]],
      [(a) => atan(a), [0.5]],
      [(a, b) => atan(a, b), [1, 1]],
      [(a) => asinh(a), [0.5]],
      [(a) => acosh(a), [2]],
      [(a) => atanh(a), [0.5]],
      [(a) => determinant(mat2(vec2(a, 1), vec2(2, 4))), [0.3]],
      [(a) => inverse(mat2(vec2(a, 1), vec2(2, 4))), [0.3]],
    ];
    for (const [build, args] of open) {
      const glsl = accuracyOf(build, args).tolerances.glsl;
      expect(glsl === null || glsl.every((t) => t === Infinity), `${build}`).toBe(true);
    }
    // The other components are compared: here a * 4, which neither call reaches.
    const m = (a: Node<"float">) => mat2(vec2(a, 1), vec2(2, 4));
    const mixed: Build[] = [(a) => vec2(sin(a), a.mul(4)), (a) => vec2(inverse(m(a)).element(0).x, a.mul(4))];
    for (const build of mixed) {
      const glsl = accuracyOf(build, [0.5]).tolerances.glsl!;
      expect(glsl[0], `${build}`).toBe(Infinity);
      expect(glsl[1], `${build}`).toBe(floatTolerance(2));
    }
    // Each of the three values reaches a component the other two leave as it was:
    // -Infinity alone flips step(0.4, x), +Infinity alone step(x, 0.6), and NaN
    // alone gets through clamp(abs(x) + 10, 0, 1).
    const reachedByOne: Build[] = [
      (a) => step(0.4, sin(a)),
      (a) => step(sin(a), 0.6),
      (a) => clamp(abs(sin(a)).add(10), 0, 1),
    ];
    for (const build of reachedByOne) {
      expect(accuracyOf(build, [0.5]).tolerances.glsl![0], `${build}`).toBe(Infinity);
    }
    expect(accuracyOf((a) => sin(a), [0.5]).tolerances.wgsl).not.toBeNull();
    expect(accuracyOf((a) => sin(a), [0.5]).unbounded).toBe(false);
    expect(accuracyOf((a) => exp(a), [0.5]).tolerances.glsl).not.toBeNull();
    expect(accuracyOf((a) => inverse(mat2(vec2(a, 1), vec2(2, 4))), [0.3]).tolerances.wgsl).not.toBeNull();
    expect(accuracyOf((a) => vec2(a, 1).normalize(), [0.3]).tolerances.glsl).not.toBeNull();
  });

  /**
   * @canon spec-a-call-whose-value-is-not-finite-has-no-bound
   */
  it("has no bound for a call whose value is not finite as a 32-bit float", () => {
    expect(unbounded((a) => log(a), [0])).toBe(true);
    expect(unbounded((a) => log(a), [-1])).toBe(true);
    expect(unbounded((a) => exp(a), [100])).toBe(true);
    expect(unbounded((a) => exp(a), [80])).toBe(false);
    expect(unbounded((a) => step(0.5, asin(a)), [2])).toBe(true);
  });

  /**
   * @canon spec-a-determinant-has-no-bound
   */
  it("has no bound for a program that computes a determinant", () => {
    expect(unbounded((a) => determinant(mat2(vec2(a, 1), vec2(2, 4))), [0.3])).toBe(true);
    const det = Fn(([m]: [Node<"mat2">]) => determinant(m)).setLayout({
      name: "det",
      type: "float",
      inputs: [{ name: "m", type: "mat2" }],
    });
    expect(unbounded((a) => det(mat2(vec2(a, 1), vec2(2, 4))).add(1), [0.3])).toBe(true);
    // A determinant the run never reaches does not count.
    const unreached: Build = (a) => select(a.greaterThan(100), determinant(mat2(vec2(a, 1), vec2(2, 4))), float(0));
    expect(unbounded(unreached, [0.3])).toBe(false);
    expect(unbounded((a) => vec2(a, 1).dot(vec2(2, 4)), [0.3])).toBe(false);
  });
});

describe("aggregate evaluation (vectors and matrices)", () => {
  /**
   * @canon spec-evaluation-reads-back-every-shape
   */
  it("round-trips a vec3 through the CPU (JS/WASM) backends", () => {
    const build = (a: any, b: any) => vec3(a, a, a).add(vec3(b, b, b));
    expect(evaluateJS(build, [1, 2])).toEqual(new Float64Array([3, 3, 3]));
    expect(evaluateWASM(build, [1, 2])).toEqual(new Float64Array([3, 3, 3]));
  });
  /**
   * @canon spec-evaluation-reads-back-every-shape
   */
  it("round-trips a mat3 through the CPU (JS/WASM) backends", () => {
    const build = (a: any) => mat3(a);
    expect(evaluateJS(build, [1])).toEqual(new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]));
    expect(evaluateWASM(build, [1])).toEqual(new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]));
  });

  describe.skipIf(EVALUATION_SKIPPED)("on the GPU backends", () => {
    /**
     * @canon spec-evaluation-reads-back-every-shape
     */
    it("round-trips a vec3 through GLSL and WGSL", async () => {
      const build = (a: any, b: any) => vec3(a, a, a).add(vec3(b, b, b));
      expect(await evaluateGLSL(build, [1, 2])).toEqual([3, 3, 3]);
      expect(await evaluateWGSL(build, [1, 2])).toEqual([3, 3, 3]);
    }, 60_000);
    /**
     * @canon spec-evaluation-reads-back-every-shape
     */
    it("round-trips a mat3 through GLSL and WGSL", async () => {
      const build = (a: any) => mat3(a);
      expect(await evaluateGLSL(build, [1])).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
      expect(await evaluateWGSL(build, [1])).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    }, 60_000);
  });
});

describe.skipIf(EVALUATION_SKIPPED)("WGSL evaluation harness", () => {
  /**
   * A failed compile is reported as an error, not silently returned as zero.
   *
   * @canon spec-evaluation-reports-a-shader-that-does-not-compile
   */
  it("reports a shader that does not compile instead of returning zero", async () => {
    await expect(runWGSL(`@compute @workgroup_size(1) fn main() { this is not wgsl }`)).rejects.toThrow(/WGSL/i);
  }, 60_000);

  /**
   * A type error rather than a syntax error: the parser accepts it and only
   * the validator objects, which is the case a parser-based check would miss.
   *
   * @canon spec-evaluation-reports-a-shader-that-does-not-compile
   */
  it("reports a shader that parses but does not type-check", async () => {
    await expect(
      runWGSL(`@group(0) @binding(0) var<storage, read_write> result: array<f32>;
@compute @workgroup_size(1)
fn main() { result[0] = refract(1.0, 2.0); }`),
    ).rejects.toThrow(/WGSL/i);
  }, 60_000);
  /**
   * @canon spec-evaluation-reads-back-every-shape
   */
  it("still reads back the value of a shader that does compile", async () => {
    await expect(
      runWGSL(`@group(0) @binding(0) var<storage, read_write> result: array<f32>;
@compute @workgroup_size(1)
fn main() { result[0] = 42.0; }`),
    ).resolves.toBeCloseTo(42, 4);
  }, 60_000);

  /**
   * Zero is the value a failed run used to produce, so it has to be readable
   * as a genuine result.
   *
   * @canon spec-evaluation-reads-back-every-shape
   */
  it("reads back a genuine zero", async () => {
    await expect(
      runWGSL(`@group(0) @binding(0) var<storage, read_write> result: array<f32>;
@compute @workgroup_size(1)
fn main() { result[0] = 0.0; }`),
    ).resolves.toBe(0);
  }, 60_000);
});
