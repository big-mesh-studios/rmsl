import { describe, it, expect, afterAll } from "vitest";
import {
  runWGSL,
  closeEvaluators,
  floatTolerance,
  transcendentalTolerance,
  callsTranscendental,
  valuesWithinTolerance,
  EVALUATION_SKIPPED,
  evaluateJS,
  evaluateGLSL,
  evaluateWGSL,
  evaluateWASM,
} from "./shader-eval";
import { vec3, mat3, atan, exp2, float, length, sin } from "../rmsl";
import { compileJSFn } from "../js";

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

describe("transcendental tolerance", () => {
  const ulp = (x: number) => Math.abs(x) * Math.pow(2, -23);
  const source = (build: (a: any, b: any) => any) =>
    compileJSFn(build, {
      name: "f",
      params: [
        { name: "a", type: "float" },
        { name: "b", type: "float" },
      ],
    });

  /**
   * The bounds WGSL gives `atan` and `sin`, in `fact-wgsl-bounds-the-error-of-each-float-built-in`.
   *
   * @canon spec-a-program-that-calls-a-transcendental-built-in-is-held-to-a-thousandth
   */
  it("is wider than WGSL allows atan and sin", () => {
    for (const magnitude of [0.1, 1, Math.PI / 2, 100]) {
      expect(transcendentalTolerance(magnitude), `magnitude ${magnitude}`).toBeGreaterThan(4096 * ulp(magnitude));
    }
    expect(transcendentalTolerance(0)).toBeGreaterThan(Math.pow(2, -11));
  });

  /**
   * The case of #344: lavapipe's `atan(1, 1)`, 55 units in the last place from the CPU's.
   *
   * @canon spec-a-program-that-calls-a-transcendental-built-in-is-held-to-a-thousandth
   */
  it("accepts lavapipe's atan(1, 1), which the float tolerance refuses", () => {
    expect(valuesWithinTolerance(0.7853949069976807, Math.PI / 4, true)).toBe(true);
    expect(valuesWithinTolerance(0.7853949069976807, Math.PI / 4)).toBe(false);
  });

  /**
   * `sin` emitted as `cos`, the nearest wrong built-in at 0.5, is off by 0.4.
   *
   * @canon spec-a-program-that-calls-a-transcendental-built-in-is-held-to-a-thousandth
   */
  it("stays tight enough to catch a wrong built-in", () => {
    expect(valuesWithinTolerance(Math.cos(0.5), Math.sin(0.5), true)).toBe(false);
    expect(transcendentalTolerance(1)).toBeLessThan(0.01);
  });

  /**
   * @canon spec-a-program-that-calls-a-transcendental-built-in-is-held-to-a-thousandth
   */
  it("applies to a program that calls a transcendental built-in, exp2 included", () => {
    expect(callsTranscendental(source((a, b) => atan(a, b)))).toBe(true);
    expect(callsTranscendental(source((a) => sin(a).add(1)))).toBe(true);
    expect(callsTranscendental(source((a) => exp2(a)))).toBe(true);
  });

  /**
   * `length` takes a square root, which WGSL bounds as tightly as a division.
   *
   * @canon spec-a-program-that-calls-a-transcendental-built-in-is-held-to-a-thousandth
   */
  it("leaves a program of exact operations to the float tolerance", () => {
    expect(callsTranscendental(source((a, b) => a.mul(b).add(float(1))))).toBe(false);
    expect(callsTranscendental(source((a, b) => length(vec3(a, b, a))))).toBe(false);
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
