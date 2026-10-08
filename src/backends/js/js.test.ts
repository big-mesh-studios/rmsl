import { describe, it, expect, afterAll, afterEach } from "vitest";
import {
  evaluateRecording,
  assertEvaluationsOfTheTestAgree,
  assertRecordedEvaluationsAgree,
  closeEvaluators,
  type CpuOnlyReason,
} from "../../testing/shader-eval";
import {
  compileJSRoutine,
  compileJSFn,
  compileJSFragment,
  type CpuTextureData,
  compileJSVertex,
  compileJSGrid,
} from "../../js";
import { compileWasmRoutine, compileWasmGrid } from "../../wasm";
import { compileJSProgram } from "./js";
import { compileWasmProgram } from "../wasm/wasm";
import {
  Fn,
  float,
  int,
  vec2,
  vec3,
  vec4,
  mat2,
  mat2x3,
  mat2x4,
  mat3x2,
  mat4,
  mat4x2,
  fragCoord,
  If,
  For,
  While,
  Switch,
  Loop,
  Break,
  Continue,
  Return,
  uniform,
  uniformArray,
  varying,
  attribute,
  instancedArray,
  invocationIndex,
  type UniformNode,
  outputStruct,
  builtinPosition,
  builtinFragDepth,
  Discard,
  ivec2,
  mul,
  add,
  sub,
  sin,
  mix,
  clamp,
  step,
  smoothstep,
  dot,
  cross,
  normalize,
  length,
  distance,
  reflect,
  refract,
  faceForward,
  atan,
  inverseSqrt,
  all,
  any,
  min,
  max,
  pow,
  textureLoad,
  type Node,
  texture,
} from "../../rmsl";

const approx = (actual: number, want: number) => expect(actual).toBeCloseTo(want, 9);

function slot(n: any): string {
  return n.value.slot;
}

type ScalarBuild = (...args: Node<"float">[]) => Node<"float">;

/**
 * Evaluate a scalar expression, and hold every backend to the answer.
 *
 * The value comes back from the CPU target, which needs no hardware and so runs
 * on every test run. The same program is recorded, and the `afterAll` below
 * replays it on both shading languages and requires them to agree — so an
 * assertion written here covers all three backends without saying so.
 *
 * `opts.cpuOnly` names a reason for a case that cannot run on a GPU. Passing a
 * reason rather than a flag keeps those exclusions countable, since the whole
 * point of this arrangement is that opting out is visible.
 */
function evalScalar(
  build: ScalarBuild,
  args: number[] = [],
  opts: { cpuOnly?: CpuOnlyReason } & Record<string, any> = {},
): number {
  const { cpuOnly, ...compileOpts } = opts;
  // A case passing compiler options wants that exact compilation, so it is run
  // directly rather than through the shared path, which compiles its own.
  if (Object.keys(compileOpts).length > 0) {
    const params = args.map((_, i) => ({ name: `a${i}`, type: "float" as const }));
    const fn = compileJSRoutine(build as any, { name: "main", params, ...compileOpts });
    const ctx: any = { params: Object.fromEntries(args.map((a, i) => [`a${i}`, a])) };
    const value = fn(ctx);
    if (typeof value === "number") return value;
    if (Array.isArray(value)) return value[0] as number;
    return value as unknown as number;
  }
  return evaluateRecording(build as any, args, cpuOnly) as number;
}

/**
 * Evaluate a matrix (or vector) expression and hold every backend to the
 * answer, the same way `evalScalar` does for a scalar root.
 */
function evalMatrix(build: (...args: Node<"float">[]) => any, args: number[] = []): Float64Array {
  return evaluateRecording(build as any, args) as Float64Array;
}

// Each test's programs are compared after it, so a disagreement fails the test that made the program.
afterEach(assertEvaluationsOfTheTestAgree, 120_000);

afterAll(async () => {
  await assertRecordedEvaluationsAgree();
  await closeEvaluators();
}, 120_000);

describe("JS backend: scalar arithmetic", () => {
  /**
   * @canon spec-arithmetic-compiles-to-the-operators-of-the-target
   */
  it("computes arithmetic", () => {
    expect(evalScalar((a, b) => a.add(b), [2, 3])).toBe(5);
    expect(evalScalar((a, b) => a.sub(b), [7, 3])).toBe(4);
    expect(evalScalar((a, b) => a.mul(b), [3, 4])).toBe(12);
    expect(evalScalar((a, b) => a.div(b), [8, 2])).toBe(4);
    expect(evalScalar((a) => a.negate(), [3])).toBe(-3);
  });
  /**
   * @canon spec-a-math-function-compiles-to-the-builtin-of-the-target
   */
  it("computes math builtins", () => {
    approx(
      evalScalar((a) => a.sqrt(), [9]),
      3,
    );
    expect(evalScalar((a) => a.abs(), [-4])).toBe(4);
    expect(evalScalar((a) => a.floor(), [2.7])).toBe(2);
    expect(evalScalar((a) => a.ceil(), [2.1])).toBe(3);
    approx(
      evalScalar((a) => a.sin(), [0.5]),
      Math.sin(0.5),
    );
    approx(
      evalScalar((a) => a.cos(), [0.5]),
      Math.cos(0.5),
    );
    approx(
      evalScalar((a, b) => a.pow(b), [2, 10]),
      1024,
    );
    approx(
      evalScalar((a) => a.cbrt(), [27]),
      3,
    );
    approx(
      evalScalar((a) => a.sinh(), [0.5]),
      Math.sinh(0.5),
    );
    expect(evalScalar((a) => a.round(), [2.6])).toBe(3);
    expect(evalScalar((a) => a.trunc(), [-2.7])).toBe(-2);
    expect(evalScalar((a) => a.saturate(), [2.5])).toBe(1);
    approx(
      evalScalar((a) => a.oneMinus(), [0.25]),
      0.75,
    );
    approx(
      evalScalar((a) => a.reciprocal(), [4]),
      0.25,
    );
  });
  /**
   * @canon spec-a-conversion-between-numeric-types-is-written-out
   */
  it("casts float to int and back", () => {
    expect(evalScalar((a) => a.toInt().toFloat(), [2.7])).toBe(2);
    expect(evalScalar((a) => a.toInt().toFloat(), [-2.7])).toBe(-2);
    expect(evalScalar((a) => a.toUint().toFloat(), [2.7])).toBe(2);
    expect(evalScalar((a) => a.toInt().toBool().toFloat(), [1.5])).toBe(1);
    expect(evalScalar((a) => a.toInt().toBool().toFloat(), [0])).toBe(0);
  });
  /**
   * @canon spec-a-conversion-between-numeric-types-is-written-out
   */
  it("casts a vector to a scalar through its first component", () => {
    expect(evalScalar((a, b) => vec3(a, b, b).toFloat(), [4.5, 9])).toBe(4.5);
    expect(evalScalar((a, b) => vec3(a, b, b).toInt().toFloat(), [2.7, 9])).toBe(2);
    expect(evalScalar((a, b) => vec2(a, b).toUint().toFloat(), [3.9, 9])).toBe(3);
  });
  /**
   * @canon spec-a-function-with-an-edge-takes-the-value-last
   */
  it("computes step, smoothstep, mix and clamp with operands in the right order", () => {
    expect(evalScalar((a, b) => b.step(a), [0.5, 2])).toBe(1);
    expect(evalScalar((a, b) => b.step(a), [2, 0.5])).toBe(0);
    expect(evalScalar((a, b) => a.mix(b, 0.25), [0, 4])).toBe(1);
    expect(evalScalar((a, b) => a.mix(b, 0.75), [0, 4])).toBe(3);
    approx(
      evalScalar((a) => a.smoothstep(0, 1), [0.5]),
      0.5,
    );
    expect(evalScalar((a) => a.clamp(0, 1), [2.5])).toBe(1);
    expect(evalScalar((a) => a.clamp(0, 1), [-2.5])).toBe(0);
  });
  /**
   * @canon spec-float-folding-gives-the-run-time-result
   */
  it("computes floored float modulus", () => {
    expect(evalScalar((a, b) => a.mod(b), [7.5, 2])).toBe(1.5);
    expect(evalScalar((a, b) => a.mod(b), [-7.5, 2])).toBe(0.5);
    expect(evalScalar((a, b) => a.mod(b), [7.5, -2])).toBe(-0.5);
    expect(evalScalar((a, b) => a.mod(b), [-1, 2])).toBe(1);
  });
  /**
   * @canon spec-float-folding-gives-the-run-time-result
   */
  it("folds constants to the same value it would compute at runtime", () => {
    const folded = evalScalar(() => float(7).div(float(2)), []);
    const runtime = evalScalar((a, b) => a.div(b), [7, 2]);
    expect(folded).toBe(3.5);
    expect(runtime).toBe(3.5);
  });
});

describe("JS backend: vector arithmetic", () => {
  /**
   * @canon spec-arithmetic-compiles-to-the-operators-of-the-target
   */
  it("adds, subtracts and scales vectors", () => {
    const f = compileJSRoutine((a: any, b: any) => a.add(b), {
      name: "main",
      params: [
        { name: "a", type: "vec3" },
        { name: "b", type: "vec3" },
      ],
    });
    expect(f({ params: { a: [1, 2, 3], b: [10, 20, 30] } })).toEqual(new Float64Array([11, 22, 33]));

    const g = compileJSRoutine((a: any) => a.mul(2), {
      name: "main",
      params: [{ name: "a", type: "vec3" }],
    });
    expect(g({ params: { a: [1, 2, 3] } })).toEqual(new Float64Array([2, 4, 6]));

    const h = compileJSRoutine((a: any) => a.sub(vec3(1, 1, 1)), {
      name: "main",
      params: [{ name: "a", type: "vec3" }],
    });
    expect(h({ params: { a: [5, 5, 5] } })).toEqual(new Float64Array([4, 4, 4]));
  });
  /**
   * @canon spec-a-scalar-fills-every-component-of-a-vector
   */
  it("broadcasts a lone scalar vector constructor across every component", () => {
    // GLSL/WGSL vec3(2.0) is (2.0, 2.0, 2.0), and the JS backend must match.
    const f = compileJSRoutine(() => vec3(2), { name: "main", params: [] });
    expect(f({})).toEqual(new Float64Array([2, 2, 2]));
    const g = compileJSRoutine(() => vec4(0.5), { name: "main", params: [] });
    expect(g({})).toEqual(new Float64Array([0.5, 0.5, 0.5, 0.5]));
    const v = compileJSRoutine(() => vec2(-1), { name: "main", params: [] });
    expect(v({})).toEqual(new Float64Array([-1, -1]));
  });
  /**
   * @canon spec-a-geometric-function-compiles-to-the-builtin-of-the-target
   */
  it("computes dot, cross, length, distance and normalize", () => {
    const dot = compileJSRoutine((a: any, b: any) => a.dot(b), {
      name: "main",
      params: [
        { name: "a", type: "vec3" },
        { name: "b", type: "vec3" },
      ],
    });
    expect(dot({ params: { a: [1, 2, 3], b: [4, 5, 6] } })).toBe(32);

    const cross = compileJSRoutine((a: any, b: any) => a.cross(b), {
      name: "main",
      params: [
        { name: "a", type: "vec3" },
        { name: "b", type: "vec3" },
      ],
    });
    expect(cross({ params: { a: [1, 0, 0], b: [0, 1, 0] } })).toEqual(new Float64Array([0, 0, 1]));

    const len = compileJSRoutine((a: any) => a.length(), {
      name: "main",
      params: [{ name: "a", type: "vec3" }],
    });
    approx(len({ params: { a: [3, 4, 0] } }) as number, 5);

    const dist = compileJSRoutine((a: any, b: any) => a.distance(b), {
      name: "main",
      params: [
        { name: "a", type: "vec2" },
        { name: "b", type: "vec2" },
      ],
    });
    approx(dist({ params: { a: [0, 0], b: [3, 4] } }) as number, 5);

    const norm = compileJSRoutine((a: any) => a.normalize(), {
      name: "main",
      params: [{ name: "a", type: "vec3" }],
    });
    const n = norm({ params: { a: [3, 0, 0] } }) as Float64Array;
    approx(n[0], 1);
    approx(n[1], 0);
    approx(n[2], 0);
  });
  /**
   * @canon spec-a-vector-comparison-gives-a-boolean-vector
   */
  it("computes vector comparisons to boolean vectors", () => {
    const f = compileJSRoutine((a: any, b: any) => a.lessThan(b), {
      name: "main",
      params: [
        { name: "a", type: "vec3" },
        { name: "b", type: "vec3" },
      ],
    });
    expect(f({ params: { a: [1, 5, 3], b: [2, 2, 2] } })).toEqual(new Int32Array([1, 0, 0]));
  });
  /**
   * @canon spec-a-conversion-between-numeric-types-is-written-out
   */
  it("constructs a numeric vector from a boolean one as 1 and 0", () => {
    // The components have to be numbers, not JavaScript booleans: arithmetic
    // coerces either way, but a comparison does not, and `false !== 0`.
    const mask = (a: Node<"float">) =>
      vec3(a, a, a)
        .lessThan(vec3(float(2)))
        .toVec3();
    expect(evalScalar((a) => mask(a).x, [3])).toBe(0);
    expect(evalScalar((a) => mask(a).x, [1])).toBe(1);
    expect(evalScalar((a) => mask(a).x.notEqual(float(0)).select(float(1), float(0)), [3])).toBe(0);
  });
  /**
   * @canon spec-a-conversion-between-numeric-types-is-written-out
   */
  it("truncates the components a float vector puts in an integer one", () => {
    expect(evalScalar((a, b) => vec3(a, b, b).toIVec3().x.toFloat(), [2.7, 9])).toBe(2);
    expect(evalScalar((a, b) => vec3(a, b, b).toIVec3().x.toFloat(), [-2.7, 9])).toBe(-2);
  });
  /**
   * @canon spec-a-boolean-vector-reduces-with-all-or-any
   */
  it("computes all/any on boolean vectors", () => {
    const f = compileJSRoutine((a: any) => a.greaterThan(vec3(0, 0, 0)).all(), {
      name: "main",
      params: [{ name: "a", type: "vec3" }],
    });
    expect(f({ params: { a: [1, 2, 3] } })).toBe(true);
    expect(f({ params: { a: [1, 0, 3] } })).toBe(false);
  });
  /**
   * @canon spec-a-geometric-function-compiles-to-the-builtin-of-the-target
   */
  it("reflects and refracts", () => {
    const f = compileJSRoutine((i: any, n: any) => i.reflect(n), {
      name: "main",
      params: [
        { name: "i", type: "vec3" },
        { name: "n", type: "vec3" },
      ],
    });
    // i = -n reflects back to +n
    const r = f({ params: { i: [0, -1, 0], n: [0, 1, 0] } }) as Float64Array;
    approx(r[0], 0);
    approx(r[1], 1);
    approx(r[2], 0);
  });
});

describe("JS backend: matrices", () => {
  /**
   * @canon spec-a-matrix-times-a-shorter-vector-promotes-it
   */
  it("multiplies mat4 by vec4 and vec3", () => {
    const m = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1];
    const f4 = compileJSRoutine((a: any, v: any) => a.mul(v), {
      name: "main",
      params: [
        { name: "a", type: "mat4" },
        { name: "v", type: "vec4" },
      ],
    });
    expect(f4({ params: { a: m, v: [1, 2, 3, 1] } })).toEqual(new Float64Array([6, 8, 10, 1]));

    const f3 = compileJSRoutine((a: any, v: any) => a.mul(v), {
      name: "main",
      params: [
        { name: "a", type: "mat4" },
        { name: "v", type: "vec3" },
      ],
    });
    expect(f3({ params: { a: m, v: [1, 2, 3] } })).toEqual(new Float64Array([6, 8, 10]));
  });
  /**
   * @canon spec-a-matrix-product-has-the-shape-of-the-product
   */
  it("multiplies matrices", () => {
    const id = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const f = compileJSRoutine((a: any, b: any) => a.mul(b), {
      name: "main",
      params: [
        { name: "a", type: "mat4" },
        { name: "b", type: "mat4" },
      ],
    });
    expect(f({ params: { a: id, b: id } })).toEqual(new Float64Array(id));
    const translate = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1];
    expect(f({ params: { a: translate, b: id } })).toEqual(new Float64Array(translate));
  });
  /**
   * @canon spec-a-matrix-product-has-the-shape-of-the-product
   */
  it("multiplies non-square matrices at the product shape", () => {
    // mat2x3 (2 cols x 3 rows): a = [[1,2,3],[4,5,6]], column-major.
    const a = [1, 2, 3, 4, 5, 6];
    // mat3x2 (3 cols x 2 rows): b = [[1,0],[0,1],[1,1]], column-major.
    const b = [1, 0, 0, 1, 1, 1];
    const f = compileJSRoutine((a: any, b: any) => a.mul(b), {
      name: "main",
      params: [
        { name: "a", type: "mat2x3" },
        { name: "b", type: "mat3x2" },
      ],
    });
    // a (3x2) * b (2x3) = product is mat3 (3x3), column-major.
    // A rows = [1,4],[2,5],[3,6]; B cols = [1,0],[0,1],[1,1].
    // col0 = A*b0 = [1,2,3]; col1 = A*b1 = [4,5,6]; col2 = A*(b0+b1) = [5,7,9].
    expect(f({ params: { a, b } })).toEqual(new Float64Array([1, 2, 3, 4, 5, 6, 5, 7, 9]));
  });
  /**
   * @canon spec-a-math-function-compiles-to-the-builtin-of-the-target
   */
  it("inverts, transposes and takes determinants", () => {
    const inv = compileJSRoutine((a: any) => a.inverse(), {
      name: "main",
      params: [{ name: "a", type: "mat2" }],
    });
    // mat2(2,1,3,4) = [[2,3],[1,4]]; inverse = [[0.8,-0.6],[-0.2,0.4]].
    const got = inv({ params: { a: [2, 1, 3, 4] } }) as Float64Array;
    got.forEach((v, i) => approx(v, [0.8, -0.2, -0.6, 0.4][i]));

    const det = compileJSRoutine((a: any) => a.determinant(), {
      name: "main",
      params: [{ name: "a", type: "mat2" }],
    });
    expect(det({ params: { a: [1, 0, 0, 1] } })).toBe(1);
    // mat2(a,b,c,d) is columns (a,b),(c,d); det = a*d - c*b.
    expect(det({ params: { a: [2, 0, 0, 3] } })).toBe(6);

    const tr = compileJSRoutine((a: any) => a.transpose(), {
      name: "main",
      params: [{ name: "a", type: "mat4" }],
    });
    const m = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    const expected = Array.from({ length: 16 }, (_, i) => m[(i % 4) * 4 + Math.floor(i / 4)]);
    expect(tr({ params: { a: m } })).toEqual(new Float64Array(expected));
  });
  /**
   * @canon spec-a-matrix-is-built-from-its-columns
   * @canon spec-a-scalar-matrix-is-a-diagonal
   */
  it("constructs matrices from columns and scalars", () => {
    const f = compileJSRoutine((c0: any, c1: any) => mat4(c0, c1, c1, c0), {
      name: "main",
      params: [
        { name: "c0", type: "vec4" },
        { name: "c1", type: "vec4" },
      ],
    });
    const r = f({ params: { c0: [1, 2, 3, 4], c1: [5, 6, 7, 8] } }) as Float64Array;
    expect(r).toEqual(new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 1, 2, 3, 4]));
  });
  /**
   * @canon spec-an-element-reads-a-component-by-index
   */
  it("reads matrix columns", () => {
    const f = compileJSRoutine((a: any) => a.element(1), {
      name: "main",
      params: [{ name: "a", type: "mat4" }],
    });
    const m = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    expect(f({ params: { a: m } })).toEqual(new Float64Array([5, 6, 7, 8]));
  });
});

describe("cross-backend: non-square matrix multiply", () => {
  /**
   * @canon spec-a-matrix-product-has-the-shape-of-the-product
   */
  it("multiplies a mat2x3 by a mat3x2 into a mat3", () => {
    const build = () => mat2x3(1, 2, 3, 4, 5, 6).mul(mat3x2(1, 0, 0, 1, 1, 1));
    expect(evalMatrix(build)).toEqual(new Float64Array([1, 2, 3, 4, 5, 6, 5, 7, 9]));
  });
  /**
   * @canon spec-a-matrix-product-has-the-shape-of-the-product
   */
  it("multiplies a mat3x2 by a mat2x3 into a mat2", () => {
    const build = () => mat3x2(1, 2, 3, 4, 5, 6).mul(mat2x3(1, 0, 0, 0, 1, 1));
    expect(evalMatrix(build)).toEqual(new Float64Array([1, 2, 8, 10]));
  });
  /**
   * @canon spec-a-matrix-product-has-the-shape-of-the-product
   */
  it("multiplies a mat2 by a mat3x2 into a mat3x2", () => {
    const build = () => mat2(2, 0, 0, 3).mul(mat3x2(1, 2, 3, 4, 5, 6));
    expect(evalMatrix(build)).toEqual(new Float64Array([2, 6, 6, 12, 10, 18]));
  });
  /**
   * @canon spec-a-matrix-product-has-the-shape-of-the-product
   */
  it("multiplies a mat4 by a mat2x4 into a mat2x4", () => {
    const build = () => mat4(1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4).mul(mat2x4(1, 2, 3, 4, 5, 6, 7, 8));
    expect(evalMatrix(build)).toEqual(new Float64Array([1, 4, 9, 16, 5, 12, 21, 32]));
  });
  /**
   * @canon spec-a-matrix-product-has-the-shape-of-the-product
   */
  it("multiplies a mat2x4 by a mat4x2 into a mat4", () => {
    const build = () => mat2x4(1, 2, 3, 4, 5, 6, 7, 8).mul(mat4x2(1, 0, 0, 1, 1, 1, 2, 0));
    expect(evalMatrix(build)).toEqual(new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 6, 8, 10, 12, 2, 4, 6, 8]));
  });
});

describe("JS backend: control flow", () => {
  /**
   * @canon spec-a-for-runs-its-body-and-update-while-its-condition-holds
   */
  it("runs a for loop the right number of times", () => {
    const sumTo = (n: Node<"float">) =>
      Fn(() => {
        const total = float(0).toVar();
        For(
          () => float(0).toVar(),
          (i) => i.lessThan(n),
          (i) => i.assign(i.add(1)),
          (i) => {
            total.assign(total.add(i));
          },
        );
        return total;
      })();
    expect(evalScalar(sumTo, [5])).toBe(10);
    expect(evalScalar(sumTo, [10])).toBe(45);
    expect(evalScalar(sumTo, [0])).toBe(0);
  });
  /**
   * @canon spec-a-for-runs-its-body-and-update-while-its-condition-holds
   */
  it("runs every statement of a loop update", () => {
    const tally = Fn(() => {
      const t = float(0).toVar();
      For(
        () => float(0).toVar(),
        (i) => i.lessThan(4),
        (i) => {
          t.assign(t.add(1));
          i.assign(i.add(1));
        },
        (i) => {
          t.assign(t.add(0));
        },
      );
      return t;
    })();
    const fn = compileJSRoutine(() => tally, { name: "main", params: [] });
    expect(fn({})).toBe(4);
  });
  /**
   * @canon spec-an-if-chain-takes-the-branch-its-conditions-select
   */
  it("takes the branch the condition selects", () => {
    const branch = (x: Node<"float">) =>
      Fn(() => {
        const out = float(0).toVar();
        If(x.greaterThan(1), () => {
          out.assign(float(10));
        }).Else(() => {
          out.assign(float(20));
        });
        return out;
      })();
    expect(evalScalar(branch, [2])).toBe(10);
    expect(evalScalar(branch, [0])).toBe(20);
  });

  /**
   * @canon spec-a-while-loop-stops-when-its-condition-fails
   */
  it("runs a while loop until its condition fails", () => {
    const countdown = (n: Node<"float">) =>
      Fn(() => {
        const left = n.toVar();
        const steps = float(0).toVar();
        While(left.greaterThan(0), () => {
          left.assign(left.sub(1));
          steps.assign(steps.add(1));
        });
        return steps;
      })();
    expect(evalScalar(countdown, [4])).toBe(4);
    expect(evalScalar(countdown, [0])).toBe(0);
  });
  /**
   * @canon spec-a-switch-runs-the-case-its-selector-matches
   */
  it("takes the branch Switch selects", () => {
    const classify = () =>
      Fn(() => {
        const out = float(0).toVar();
        Switch(int(1))
          .Case(0, () => {
            out.assign(float(10));
          })
          .Case(1, 2, () => {
            out.assign(float(20));
          })
          .Default(() => {
            out.assign(float(30));
          });
        return out;
      })();
    const fn = compileJSRoutine(() => classify(), { name: "main", params: [] });
    expect(fn({})).toBe(20);
  });
  /**
   * @canon spec-break-continue-return-and-discard-leave-where-tsl-leaves
   */
  it("honours break_ and continue_", () => {
    const sumUntilBreak = (limit: Node<"float">) =>
      Fn(() => {
        const total = float(0).toVar();
        For(
          () => float(0).toVar(),
          (i) => i.lessThan(100),
          (i) => i.assign(i.add(1)),
          (i) => {
            If(i.greaterThanEqual(limit), () => {
              Break();
            });
            total.assign(total.add(i));
          },
        );
        return total;
      })();
    expect(evalScalar(sumUntilBreak, [5])).toBe(10);
    expect(evalScalar(sumUntilBreak, [1])).toBe(0);

    const sumSkippingFirst = (n: Node<"float">) =>
      Fn(() => {
        const total = float(0).toVar();
        For(
          () => float(0).toVar(),
          (i) => i.lessThan(n),
          (i) => i.assign(i.add(1)),
          (i) => {
            If(i.lessThan(2), () => {
              Continue();
            });
            total.assign(total.add(i));
          },
        );
        return total;
      })();
    expect(evalScalar(sumSkippingFirst, [5])).toBe(9);
  });
  /**
   * @canon spec-an-if-chain-takes-the-branch-its-conditions-select
   */
  it("computes the same results through the lowercase aliases", () => {
    const branch = (x: Node<"float">) =>
      Fn(() => {
        const out = float(0).toVar();
        If(x.greaterThan(1), () => {
          out.assign(float(10));
        })
          .ElseIf(x.greaterThan(0), () => {
            out.assign(float(20));
          })
          .Else(() => {
            out.assign(float(30));
          });
        return out;
      })();
    expect(evalScalar(branch, [2])).toBe(10);
    expect(evalScalar(branch, [0.5])).toBe(20);
    expect(evalScalar(branch, [-1])).toBe(30);

    const sum = (n: Node<"float">) =>
      Fn(() => {
        const total = float(0).toVar();
        For(
          () => float(0).toVar(),
          (i) => i.lessThan(n),
          (i) => i.assign(i.add(1)),
          (i) => {
            total.assign(total.add(i));
          },
        );
        return total;
      })();
    expect(evalScalar(sum, [5])).toBe(10);

    const classify = () =>
      Fn(() => {
        const out = float(0).toVar();
        Switch(int(2))
          .Case(0, () => {
            out.assign(float(10));
          })
          .Case(1, 2, () => {
            out.assign(float(20));
          })
          .Default(() => {
            out.assign(float(30));
          });
        return out;
      })();
    const fn = compileJSRoutine(() => classify(), { name: "main", params: [] });
    expect(fn({})).toBe(20);
  });
});

describe("JS backend: shader I/O", () => {
  /**
   * @canon spec-a-cpu-program-reads-its-inputs-by-slot
   */
  it("reads uniforms", () => {
    let u!: any;
    const prog = Fn(() => {
      u = uniform("float");
      return u.mul(2);
    })();
    const fn = compileJSRoutine(() => prog, { name: "main", params: [] });
    expect(fn({ uniforms: { [u.name]: 21 } })).toBe(42);
  });
  /**
   * @canon spec-a-variable-holds-a-copy
   */
  it("keeps a uniform as it is when a variable copied from it is written", () => {
    const v = uniform("vec3");
    const prog = Fn(() => {
      const copy = v.toVar();
      copy.y.assign(float(9));
      return copy.y.add(v.y);
    })();
    const fn = compileJSRoutine(() => prog, { name: "main", params: [] });
    expect(fn({ uniforms: { [v.name]: [1, 2, 3] } })).toBe(11);
  });
  /**
   * @canon spec-a-js-routine-allocates-nothing-per-call
   */
  it("clears a stage output a call does not write, in the one object it returns them in", () => {
    const shade = varying("float");
    const write = uniform("int");
    const raw = compileJSProgram(
      () =>
        Fn(() => {
          If(write.equal(int(1)), () => {
            shade.assign(float(0.25));
          });
          builtinPosition().assign(vec4(0, 0, 0, 1));
        })(),
      { name: "main", params: [], stage: "vertex" },
    );
    const first = raw.runInPlace({ uniforms: { [write.name]: 1 } }) as any;
    expect(first.varyings[shade.name]).toBe(0.25);
    const second = raw.runInPlace({ uniforms: { [write.name]: 0 } }) as any;
    // The stage returns its outputs in one object, made once, as it keeps its slots.
    expect(second).toBe(first);
    expect(second.varyings[shade.name]).toBe(0);

    const depth = compileJSProgram(
      () =>
        Fn(() => {
          If(fragCoord().x.greaterThan(1), () => {
            builtinFragDepth().assign(float(0.25));
          });
          return vec4(1, 0, 0, 1);
        })(),
      { name: "main", params: [], stage: "fragment" },
    );
    expect((depth.runInPlace({ fragCoord: [2, 0] }) as any).fragDepth).toBe(0.25);
    expect((depth.runInPlace({ fragCoord: [0, 0] }) as any).fragDepth).toBeUndefined();
  });
  /**
   * @canon spec-a-variable-holds-a-copy
   */
  it("keeps a constant as it is when a varying assigned from it is written", () => {
    const shade = varying("vec3");
    const k = uniform("int");
    const stage = compileJSVertex(
      () =>
        Fn(() => {
          shade.assign(vec3(1, 2, 3));
          shade.element(k).assign(float(9));
          builtinPosition().assign(vec4(0, 0, 0, 1));
        })(),
      { name: "main", params: [] },
    );
    expect(stage({ uniforms: { [k.name]: 1 } }).varyings[shade.name]).toEqual(new Float64Array([1, 9, 3]));
    expect(stage({ uniforms: { [k.name]: 0 } }).varyings[shade.name]).toEqual(new Float64Array([9, 2, 3]));
  });
  /**
   * @canon spec-a-cpu-program-reads-its-inputs-by-slot
   */
  it("reads varyings and attributes", () => {
    let v!: any;
    let a!: any;
    const prog = Fn(() => {
      v = varying("vec3");
      a = attribute("float");
      return v.x.add(a);
    })();
    const fn = compileJSProgram(() => prog, { name: "main", params: [] });
    expect(fn.run({ varyings: { [v.name]: [3, 4, 5] }, attributes: { [a.name]: 1 } })).toBe(4);
  });
  /**
   * @canon spec-a-fragment-stage-returns-its-colour-and-outputs
   */
  it("returns the outputs and the fragment depth of a fragment stage", () => {
    let u!: any;
    const prog = Fn(() => {
      u = uniform("vec4");
      const d = builtinFragDepth();
      d.assign(float(0.5));
      return outputStruct(u.add(vec4(1, 1, 1, 0)));
    })();
    const stage = compileJSFragment(() => prog, { name: "main", params: [] });
    const r = stage({ uniforms: { [u.name]: [0, 0, 0, 1] } });
    expect(r).toEqual({ value: undefined, outputs: [new Float64Array([1, 1, 1, 1])], fragDepth: 0.5 });
  });
  /**
   * @canon spec-a-cpu-routine-returns-its-value
   */
  it("returns the bare value when nothing is written to outputs", () => {
    const prog = Fn(() => vec4(1, 2, 3, 4))();
    const fn = compileJSRoutine(() => prog, { name: "main", params: [] });
    expect(fn({})).toEqual(new Float64Array([1, 2, 3, 4]));
  });
  /**
   * @canon spec-a-cpu-program-reads-its-inputs-by-slot
   * @canon spec-a-uniform-array-is-read-by-element
   */
  it("reads uniform arrays", () => {
    let arr!: any;
    const prog = Fn(() => {
      arr = uniformArray("vec4", 4);
      return arr.element(2).x;
    })();
    const fn = compileJSRoutine(() => prog, { name: "main", params: [] });
    const values = [
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [7, 8, 9, 10],
      [0, 0, 0, 0],
    ];
    expect(fn({ uniforms: { [arr.name]: values } })).toBe(7);
  });
  /**
   * @canon spec-a-vertex-stage-returns-its-position-and-varyings
   * @canon spec-a-vertex-stage-writes-its-position
   */
  it("runs a vertex stage, writing position and varyings", () => {
    const prog = Fn(() => {
      const v = varying("vec3");
      v.assign(vec3(1, 2, 3));
      const p = builtinPosition();
      p.assign(vec4(0, 0, 0, 1));
      return p;
    })();
    const fn = compileJSVertex(() => prog, { name: "main", params: [] });
    const r = fn({}) as any;
    expect(r.position).toEqual(new Float64Array([0, 0, 0, 1]));
    expect(Object.values(r.varyings as Record<string, unknown>)).toEqual([new Float64Array([1, 2, 3])]);
  });
});

describe("JS backend: CPU-specific behaviour", () => {
  /**
   * @canon spec-a-cpu-routine-returns-its-value
   */
  it("discard returns null", () => {
    const prog = Fn(() => {
      If(float(1).greaterThan(0), () => Discard());
      return float(5);
    })();
    const fn = compileJSProgram(() => prog, { name: "main", params: [] });
    expect(fn.run({})).toBeNull();
  });
  /**
   * @canon spec-a-js-routine-allocates-nothing-per-call
   */
  it("hoisted scratch does not leak state across calls", () => {
    const prog = Fn(() => {
      const x = vec3(1, 2, 3).toVar();
      const y = vec3(10, 20, 30).toVar();
      If(float(0).greaterThan(1), () => {
        x.assign(y);
      });
      return x;
    })();
    const fn = compileJSRoutine(() => prog, { name: "main", params: [] });
    expect(fn({})).toEqual(new Float64Array([1, 2, 3]));
    expect(fn({})).toEqual(new Float64Array([1, 2, 3]));
  });
  /**
   * @canon spec-a-js-routine-allocates-nothing-per-call
   */
  it("reentrant mode computes the same results", () => {
    const sumTo = (n: Node<"float">) =>
      Fn(() => {
        const total = float(0).toVar();
        For(
          () => float(0).toVar(),
          (i) => i.lessThan(n),
          (i) => i.assign(i.add(1)),
          (i) => {
            total.assign(total.add(i));
          },
        );
        return total;
      })();
    const hoisted = compileJSRoutine(sumTo, { name: "sum", params: [{ name: "n", type: "float" }] });
    const perCall = compileJSRoutine(sumTo, { name: "sum", params: [{ name: "n", type: "float" }], reentrant: true });
    expect(hoisted({ params: { n: 7 } })).toBe(21);
    expect(perCall({ params: { n: 7 } })).toBe(21);
    expect(perCall({ params: { n: 3 } })).toBe(3);
  });
  /**
   * @canon exception-a-cpu-target-has-no-derivatives
   */
  it("compiles derivatives to zero on request", () => {
    const prog = Fn(() => {
      const x = vec2(1, 2).toVar();
      return x.fwidth();
    })();
    const fn = compileJSRoutine(() => prog, { name: "main", params: [], derivatives: "zero" });
    expect(fn({})).toEqual(new Float64Array([0, 0]));
  });
  /**
   * @canon exception-a-cpu-target-has-no-derivatives
   */
  it("refuses derivatives by default", () => {
    const prog = Fn(() => {
      const x = vec2(1, 2).toVar();
      return x.fwidth();
    })();
    expect(() => compileJSRoutine(() => prog, { name: "main", params: [] })).toThrow(/CPU target/);
  });
  /**
   * @canon spec-a-cpu-target-filters-as-the-texture-asks
   */
  it("samples textures with nearest-neighbour lookup", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("sampler2D");
      return texture(tex, vec2(0.5, 0.5));
    })();
    const fn = compileJSRoutine(() => prog, { name: "main", params: [] });
    // 2x2 RGBA; uv (0.5, 0.5) -> texel (1, 1).
    const data = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4];
    expect(fn({ textures: { [tex.name]: { data, width: 2, height: 2 } } })).toEqual(new Float64Array([4, 4, 4, 4]));
  });
  /**
   * @canon spec-a-byte-texture-reads-as-zero-to-one
   */
  it("reads a byte texture through a float sampler as 0 to 1", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("sampler2D");
      return texture(tex, vec2(0.5, 0.5));
    })();
    const fn = compileJSRoutine(() => prog, { name: "main", params: [] });
    // What a DataTexture holds: 8-bit channels. Both backends upload that as a
    // normalized format, so the shader reads 0..1 — and so must this.
    const data = new Uint8Array([0, 128, 255, 255]);
    expect(fn({ textures: { [tex.name]: { data, width: 1, height: 1 } } })).toEqual(
      new Float64Array([0, 128 / 255, 1, 1]),
    );
  });
  /**
   * @canon spec-a-byte-texture-reads-as-zero-to-one
   */
  it("leaves a float texture that already holds float data alone", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("sampler2D");
      return texture(tex, vec2(0.5, 0.5));
    })();
    const fn = compileJSRoutine(() => prog, { name: "main", params: [] });
    const data = new Float32Array([0, 0.5, 1, 1]);
    expect(fn({ textures: { [tex.name]: { data, width: 1, height: 1 } } })).toEqual(new Float64Array([0, 0.5, 1, 1]));
    // A plain array is a plain array, whatever is in it.
    expect(fn({ textures: { [tex.name]: { data: [0, 0.5, 1, 1], width: 1, height: 1 } } })).toEqual(
      new Float64Array([0, 0.5, 1, 1]),
    );
  });
  /**
   * @canon spec-a-byte-texture-reads-as-zero-to-one
   */
  it("keeps an integer texture's bytes as the texels they are", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("usampler2D");
      return texture(tex, ivec2(0, 0));
    })();
    const fn = compileJSRoutine(() => prog, { name: "main", params: [] });
    // An integer sampler fetches raw texels on a GPU too — there is no
    // normalized format under it to undo.
    const data = new Uint8Array([0, 128, 255, 255]);
    expect(fn({ textures: { [tex.name]: { data, width: 1, height: 1 } } })).toEqual(
      new Uint32Array([0, 128, 255, 255]),
    );
  });
  /**
   * @canon spec-a-texel-holds-the-channels-its-texture-stores
   */
  it("strides by the channels a texel holds, not by four", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("usampler2D");
      return texture(tex, ivec2(2, 0));
    })();
    const fn = compileJSRoutine(() => prog, { name: "main", params: [] });
    // Four single-channel texels in a row: the third is 30. Read as RGBA, the
    // same array is one texel and the fetch falls off the end of it.
    const data = new Uint8Array([10, 20, 30, 40]);
    expect(fn({ textures: { [tex.name]: { data, width: 4, height: 1, channels: 1 } } }))
      // green and blue read zero, alpha one, as a sampler reports the channels
      // a single-channel texture does not store
      .toEqual(new Uint32Array([30, 0, 0, 1]));
  });
  /**
   * @canon spec-a-texel-holds-the-channels-its-texture-stores
   */
  it("blends a single-channel texture without reading its neighbours' channels", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("sampler2D");
      return texture(tex, vec2(0.5, 0.5));
    })();
    const fn = compileJSRoutine(() => prog, { name: "main", params: [] });
    const texels = { data: [0, 100], width: 2, height: 1, channels: 1 as const };
    expect(fn({ textures: { [tex.name]: texels } })).toEqual(new Float64Array([100, 0, 0, 1]));
    expect(fn({ textures: { [tex.name]: { ...texels, magFilter: "linear" as const } } })).toEqual(
      new Float64Array([50, 0, 0, 1]),
    );
  });
  /**
   * @canon spec-a-byte-texture-reads-as-zero-to-one
   */
  it("normalizes a byte texture fetched with textureLoad too", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("sampler2D");
      return textureLoad(tex, ivec2(0, 0));
    })();
    const fn = compileJSRoutine(() => prog, { name: "main", params: [] });
    const data = new Uint8Array([0, 128, 255, 255]);
    expect(fn({ textures: { [tex.name]: { data, width: 1, height: 1 } } })).toEqual(
      new Float64Array([0, 128 / 255, 1, 1]),
    );
  });
  /**
   * @canon spec-a-cpu-target-filters-as-the-texture-asks
   */
  it("blends neighbouring texels when the texture asks for linear filtering", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("sampler2D");
      return texture(tex, vec2(0.5, 0.5));
    })();
    const fn = compileJSRoutine(() => prog, { name: "main", params: [] });
    // Two texels, 0 and 100, whose centres sit at 0.25 and 0.75. Sampling
    // halfway between them lands in the second texel outright without
    // filtering, and is half of each with it.
    const data = [0, 0, 0, 0, 100, 100, 100, 100];
    const texels = { data, width: 2, height: 1 };
    expect(fn({ textures: { [tex.name]: texels } })).toEqual(new Float64Array([100, 100, 100, 100]));
    expect(fn({ textures: { [tex.name]: { ...texels, magFilter: "linear" as const } } })).toEqual(
      new Float64Array([50, 50, 50, 50]),
    );
  });
  /**
   * @canon spec-a-cpu-target-wraps-as-the-texture-asks
   */
  it("wraps a coordinate past the edge the way the texture asks", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("sampler2D");
      return texture(tex, vec2(1.25, 0.5));
    })();
    const fn = compileJSRoutine(() => prog, { name: "main", params: [] });
    const texels = { data: [10, 10, 10, 10, 20, 20, 20, 20], width: 2, height: 1 };
    const red = (t: CpuTextureData): number => (fn({ textures: { [tex.name]: t } }) as Float64Array)[0];

    // A quarter past the right edge: the last texel stretched, the image
    // tiled back to the first, or tiled and flipped back to the last.
    expect(red(texels)).toBe(20);
    expect(red({ ...texels, wrapS: "repeat" as const })).toBe(10);
    expect(red({ ...texels, wrapS: "mirror" as const })).toBe(20);
  });
  /**
   * @canon spec-a-cpu-target-wraps-as-the-texture-asks
   */
  it("wraps behind the left edge too", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("sampler2D");
      return texture(tex, vec2(-0.25, 0.5));
    })();
    const fn = compileJSRoutine(() => prog, { name: "main", params: [] });
    const texels = { data: [10, 10, 10, 10, 20, 20, 20, 20], width: 2, height: 1 };
    const red = (t: CpuTextureData): number => (fn({ textures: { [tex.name]: t } }) as Float64Array)[0];

    expect(red(texels)).toBe(10);
    expect(red({ ...texels, wrapS: "repeat" as const })).toBe(20);
    expect(red({ ...texels, wrapS: "mirror" as const })).toBe(10);
  });
  /**
   * @canon spec-a-3d-texture-blends-across-its-depth
   */
  it("blends across the depth of a 3D texture", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("sampler3D");
      return texture(tex, vec3(0.5, 0.5, 0.5));
    })();
    const fn = compileJSRoutine(() => prog, { name: "main", params: [] });
    // Two slices, 0 and 100, sampled halfway between their centres.
    const texels: CpuTextureData = {
      data: [0, 0, 0, 0, 100, 100, 100, 100],
      width: 1,
      height: 1,
      depth: 2,
      magFilter: "linear",
    };
    expect(fn({ textures: { [tex.name]: texels } })).toEqual(new Float64Array([50, 50, 50, 50]));
  });
  /**
   * @canon spec-a-cube-map-is-sampled-on-the-face-its-direction-picks
   */
  it("samples the right face of a cube map by direction", () => {
    // 6 faces, one texel each, face order +X,-X,+Y,-Y,+Z,-Z: 10,20,30,40,50,60.
    const data = [10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30, 40, 40, 40, 40, 50, 50, 50, 50, 60, 60, 60, 60];
    const texels: CpuTextureData = { data, width: 1, height: 1 };

    const at = (dx: number, dy: number, dz: number): Float64Array => {
      let tex!: any;
      const build = () =>
        Fn(() => {
          tex = uniform("samplerCube");
          return texture(tex, vec3(dx, dy, dz));
        })();
      const fn = compileJSRoutine(build, { name: "main", params: [] });
      return fn({ textures: { [tex.name]: texels } }) as Float64Array;
    };

    expect(at(1, 0, 0)).toEqual(new Float64Array([10, 10, 10, 10]));
    expect(at(-1, 0, 0)).toEqual(new Float64Array([20, 20, 20, 20]));
    expect(at(0, 1, 0)).toEqual(new Float64Array([30, 30, 30, 30]));
    expect(at(0, -1, 0)).toEqual(new Float64Array([40, 40, 40, 40]));
    expect(at(0, 0, 1)).toEqual(new Float64Array([50, 50, 50, 50]));
    expect(at(0, 0, -1)).toEqual(new Float64Array([60, 60, 60, 60]));
  });
  /**
   * @canon spec-a-cube-map-is-sampled-on-the-face-its-direction-picks
   */
  it("blends within a cube face but never across its edge into a neighbour", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("samplerCube");
      // Direction close to the edge of the +Z face (u near 1): should stay
      // within +Z's own 2x1 texel row, not bleed toward another face.
      return texture(tex, vec3(0.9, 0, 1));
    })();
    const fn = compileJSRoutine(() => prog, { name: "main", params: [] });
    // +Z is face index 4: two texels side by side, 0 and 100.
    const data = new Array(6 * 2 * 1 * 4).fill(0);
    data[4 * 2 * 4 + 0] = 0;
    data[4 * 2 * 4 + 4] = 100;
    const texels: CpuTextureData = { data, width: 2, height: 1, magFilter: "linear" };
    const [r] = fn({ textures: { [tex.name]: texels } }) as Float64Array;
    expect(r).toBeGreaterThan(0);
    expect(r).toBeLessThanOrEqual(100);
  });
  /**
   * @canon spec-an-integer-texture-is-fetched-unfiltered
   */
  it("fetches integer textures at texel coordinates", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("isampler2D");
      return texture(tex, ivec2(1, 0));
    })();
    const fn = compileJSRoutine(() => prog, { name: "main", params: [] });
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    expect(fn({ textures: { [tex.name]: { data, width: 2, height: 2 } } })).toEqual(new Int32Array([5, 6, 7, 8]));
  });
  /**
   * @canon spec-a-cpu-routine-returns-its-value
   */
  it("supports multi-return functions, keeping only the last value as the result", () => {
    let side!: any;
    const prog = Fn(() => {
      side = float(1.0).toVar();
      let last = float(2.0).toVar();
      return [side, last];
    })();
    const fn = compileJSRoutine(() => prog as any, { name: "main", params: [] });
    expect(fn({})).toBe(2);
  });
  /**
   * @canon spec-the-js-target-compiles-a-function-of-a-context
   */
  it("compileJSFn emits a self-contained expression evaluating to the callable", () => {
    const src = compileJSFn((a: any, b: any) => a.add(b), {
      name: "main",
      params: [
        { name: "a", type: "float" },
        { name: "b", type: "float" },
      ],
    });
    expect(src).toContain("ctx.params");
    const fn = new Function(src)() as (ctx: any) => number;
    expect(fn({ params: { a: 1, b: 2 } })).toBe(3);
  });
});

describe("JS backend: screen-picking workflow", () => {
  /**
   * @canon spec-a-cpu-routine-answers-one-fragment-per-call
   */
  it("computes the world-space pick point of a ray-marched ground plane", () => {
    // A miniature of the picking flow: the fragment Fn computes where the ray
    // from the camera hits the y = 0 plane, written out as depth.
    const prog = Fn(() => {
      const ro = vec3(0, 2, 0).toVar(); // camera above the plane
      const rd = vec3(0, -1, 0).toVar(); // straight down
      const t = ro.y.negate().div(rd.y).toVar(); // distance to y = 0
      const hit = ro.add(rd.mul(t)).toVar();
      const d = builtinFragDepth();
      d.assign(t);
      return hit;
    })();
    const fn = compileJSProgram(() => prog, { name: "pick", params: [] });
    const r = fn.run({}) as any;
    // Ray hits y = 0 at t = 2, so the world point is (0, 0, 0).
    expect(r.value).toEqual(new Float64Array([0, 0, 0]));
    expect(r.fragDepth).toBe(2);
  });
  /**
   * @canon spec-a-cpu-routine-answers-one-fragment-per-call
   */
  it("reuses one compiled function across many pick calls", () => {
    const prog = Fn(() => {
      const ro = vec3(0, 1, 0).toVar();
      const rd = vec3(0, -1, 0).toVar();
      const t = ro.y.negate().div(rd.y).toVar();
      return ro.add(rd.mul(t));
    })();
    const fn = compileJSRoutine(() => prog, { name: "pick", params: [] });
    for (let i = 0; i < 100; i++) {
      expect(fn({})).toEqual(new Float64Array([0, 0, 0]));
    }
  });
});

describe("JS backend: TSL free functions", () => {
  /**
   * @canon spec-arithmetic-compiles-to-the-operators-of-the-target
   */
  it("computes arithmetic free functions", () => {
    expect(evalScalar((a, b) => add(a, b), [2, 3])).toBe(5);
    expect(evalScalar((a, b) => sub(a, b), [7, 3])).toBe(4);
    expect(evalScalar((a, b) => mul(a, b), [3, 4])).toBe(12);
    expect(evalScalar((a, b) => mul(a, b, a), [3, 4])).toBe(36);
  });
  /**
   * @canon spec-a-math-function-compiles-to-the-builtin-of-the-target
   */
  it("computes math free functions", () => {
    expect(
      approx(
        evalScalar((a) => sin(a), [Math.PI / 2]),
        1,
      ),
    );
    expect(evalScalar((a) => inverseSqrt(a), [4])).toBe(0.5);
    expect(
      approx(
        evalScalar((a, b) => atan(a, b), [1, 1]),
        Math.PI / 4,
      ),
    );
    expect(evalScalar((a, b) => min(a, b), [3, 1])).toBe(1);
    expect(evalScalar((a, b) => max(a, b), [3, 1])).toBe(3);
    expect(evalScalar((a, b) => pow(a, b), [2, 3])).toBe(8);
  });
  /**
   * @canon spec-a-function-with-an-edge-takes-the-value-last
   */
  it("computes interpolation free functions", () => {
    expect(evalScalar((a, b) => mix(a, b, 0.5), [0, 10])).toBe(5);
    expect(evalScalar((a, b) => clamp(a, 0, 1), [2])).toBe(1);
    expect(evalScalar((a, b) => step(0.5, a), [1])).toBe(1);
    expect(evalScalar((a, b) => smoothstep(0, 1, a), [0.5])).toBeCloseTo(0.5);
  });
  /**
   * @canon spec-a-geometric-function-compiles-to-the-builtin-of-the-target
   */
  it("computes vector free functions", () => {
    const fn = compileJSRoutine(
      () =>
        Fn(() => {
          const a = vec3(1, 2, 3).toVar();
          return dot(a, a).add(length(a)).toVar();
        })(),
      { name: "v2", params: [] },
    );
    expect(fn({})).toBeCloseTo(14 + Math.sqrt(14));
  });
  /**
   * @canon spec-a-geometric-function-compiles-to-the-builtin-of-the-target
   */
  it("computes cross/reflect/normalize/faceForward", () => {
    const fn = compileJSRoutine(
      () =>
        Fn(() => {
          const a = vec3(1, 0, 0).toVar();
          return cross(a, vec3(0, 1, 0))
            .add(normalize(vec3(0, 0, 2)))
            .toVar();
        })(),
      { name: "v3", params: [] },
    );
    const c = fn({}) as Float64Array;
    // cross((1,0,0),(0,1,0)) = (0,0,1), normalize(0,0,2) = (0,0,1).
    expect(c.map((x) => Math.abs(x))).toEqual(new Float64Array([0, 0, 2]));
    const ff = compileJSRoutine(
      () =>
        Fn(() => {
          const n = vec3(0, 1, 0).toVar();
          return faceForward(n, vec3(0, 1, 0), vec3(1, 0, 0)).toVar();
        })(),
      { name: "v4", params: [] },
    );
    // faceforward flips n because dot(nref, i) > 0; sign flips leave signed zero.
    expect((ff({}) as Float64Array).map((x) => (x === 0 ? 0 : x))).toEqual(new Float64Array([0, -1, 0]));
  });
  /**
   * @canon spec-a-boolean-vector-reduces-with-all-or-any
   */
  it("computes all/any reductions", () => {
    const fn = compileJSRoutine(
      () =>
        Fn(() => {
          const a = vec3(1, 2, 3).toVar();
          return all(a.greaterThan(0))
            .toInt()
            .add(any(a.lessThan(0)).toInt())
            .toVar();
        })(),
      { name: "r", params: [] },
    );
    expect(fn({})).toBe(1);
  });
});

describe("JS backend: TSL loop and return", () => {
  /**
   * @canon spec-loop-runs-its-body-count-times
   */
  it("Loop(count, (i) => ...) sums 0..3", () => {
    const fn = compileJSRoutine(
      () =>
        Fn(() => {
          let total = float(0).toVar();
          Loop(int(4), (i) => {
            total.assign(total.add(float(i)));
          });
          return total;
        })(),
      { name: "loop", params: [] },
    );
    expect(fn({})).toBe(6);
  });
  /**
   * @canon spec-break-continue-return-and-discard-leave-where-tsl-leaves
   */
  it("Return() exits the function early", () => {
    const fn = compileJSRoutine(
      () =>
        Fn(() => {
          const out = float(0).toVar();
          If(float(1).greaterThan(0), () => {
            Return();
          });
          out.assign(float(1));
          return out;
        })(),
      { name: "ret", params: [] },
    );
    // The `return;` fires before the trailing return, so the function is
    // undefined rather than 1.
    expect(fn({})).toBeUndefined();
  });
  /**
   * @canon spec-a-cpu-routine-returns-its-value
   */
  it("Discard() returns null", () => {
    const fn = compileJSProgram(
      () =>
        Fn(() => {
          const out = float(1).toVar();
          If(float(1).greaterThan(0), () => {
            Discard();
          });
          out.assign(float(2));
          return out;
        })(),
      { name: "disc", params: [] },
    );
    expect(fn.run({})).toBe(null);
  });
});

/**
 * Operands that are themselves expressions.
 *
 * Every other test here passes bare parameters, so an operand always arrives as
 * a single identifier and drops into any template unchanged. A compound operand
 * does not: written into `a - b * Math.floor(a / b)` without brackets, the
 * division binds to the last term of the sum rather than to the whole of it.
 *
 * The operations at risk are the ones written as a formula rather than as a
 * call, since a call's arguments are separated by commas and need no brackets.
 */
describe("JS backend: operands that are themselves expressions", () => {
  const floored = (x: number, y: number) => x - y * Math.floor(x / y);
  /**
   * @canon spec-an-operand-that-is-an-expression-keeps-its-grouping
   */
  it("computes modulo of a sum", () => {
    // (7 + 5) mod 4 is 0, not 7 + 5 - 4 * floor(7 + 5 / 4)
    expect(evalScalar((a, b, m) => a.add(b).mod(m), [7, 5, 4])).toBe(floored(7 + 5, 4));
  });
  /**
   * @canon spec-an-operand-that-is-an-expression-keeps-its-grouping
   */
  it("computes modulo by a sum", () => {
    expect(evalScalar((a, b, c) => a.mod(b.add(c)), [17, 3, 2])).toBe(floored(17, 3 + 2));
  });
  /**
   * @canon spec-an-operand-that-is-an-expression-keeps-its-grouping
   */
  it("computes whole-number division and modulo of a sum", () => {
    expect(evalScalar((a, b, m) => a.toInt().add(b.toInt()).div(m.toInt()).toFloat(), [7, 5, 4])).toBe(
      Math.trunc((7 + 5) / 4),
    );
    expect(evalScalar((a, b, m) => a.toInt().add(b.toInt()).mod(m.toInt()).toFloat(), [7, 5, 4])).toBe((7 + 5) % 4);
  });
  /**
   * @canon spec-an-operand-that-is-an-expression-keeps-its-grouping
   */
  it("mixes between sums", () => {
    // mix(3, 10, 0.25) is 4.75
    approx(
      evalScalar((a, b, c, d, t) => a.add(b).mix(c.add(d), t), [1, 2, 4, 6, 0.25]),
      3 + 0.25 * (10 - 3),
    );
  });
  /**
   * @canon spec-every-node-is-emitted-once
   */
  it("compiles a value every level of which is read twice once when it is copied into a variable", () => {
    // Each level reads the one below twice, so writing each level out where it
    // is read takes 2^14 copies of the bottom one.
    const build = (a: Node<"float">) =>
      Fn(() => {
        let x = vec3(a, a, a);
        for (let i = 0; i < 14; i++) x = x.add(x).mul(0.5);
        return x.toVar().x;
      })();
    const options = { name: "main", params: [{ name: "a", type: "float" as const }] };
    expect(compileJSFn(build, options).length).toBeLessThan(10_000);
    expect(compileJSRoutine(build, options)({ params: { a: 3 } })).toBe(3);
  });
  /**
   * @canon spec-an-operand-that-is-an-expression-keeps-its-grouping
   */
  it("smoothsteps across a sum edge", () => {
    // smoothstep(2, 6, 4): t is 0.5, so the result is 0.5
    approx(
      evalScalar((v, e0a, e0b, e1) => v.smoothstep(e0a.add(e0b), e1), [4, 1, 1, 6]),
      0.5,
    );
  });
});

describe("JS backend: .draw() — render a whole grid in one call", () => {
  /**
   * @canon spec-a-cpu-grid-evaluates-a-fragment-for-each-pixel
   */
  it("renders a scalar per pixel, fragCoord at pixel centers", () => {
    const build = () => Fn(() => fragCoord().x)();
    const fn = compileJSGrid(build as any, { name: "main", params: [] });
    const out = fn({}, 3, 2);
    expect(out.length).toBe(3 * 2);
    // Row-major, (y*width+x): x+0.5 regardless of row.
    expect(Array.from(out)).toEqual([0.5, 1.5, 2.5, 0.5, 1.5, 2.5]);
  });
  /**
   * @canon spec-a-cpu-grid-evaluates-a-fragment-for-each-pixel
   */
  it("renders both fragCoord axes packed into a vec4 per pixel, with no stage or output() involved", () => {
    const build = () => Fn(() => vec4(fragCoord().x, fragCoord().y, 0, 1))();
    const fn = compileJSGrid(build as any, { name: "main", params: [] });
    const out = fn({}, 2, 2);
    expect(out.length).toBe(2 * 2 * 4);
    expect(Array.from(out)).toEqual([
      0.5,
      0.5,
      0,
      1, // (0,0)
      1.5,
      0.5,
      0,
      1, // (1,0)
      0.5,
      1.5,
      0,
      1, // (0,1)
      1.5,
      1.5,
      0,
      1, // (1,1)
    ]);
  });
  /**
   * @canon spec-a-cpu-grid-evaluates-a-fragment-for-each-pixel
   */
  it("reads a uniform every pixel and reflects a changed uniform on the next call", () => {
    const scale = uniform("float");
    const build = () => Fn(() => fragCoord().x.mul(scale))();
    const fn = compileJSGrid(build as any, { name: "main", params: [] });
    expect(Array.from(fn({ uniforms: { [scale.name]: 2 } }, 2, 1))).toEqual([1, 3]);
    expect(Array.from(fn({ uniforms: { [scale.name]: 10 } }, 2, 1))).toEqual([5, 15]);
  });
  /**
   * @canon spec-a-cpu-grid-evaluates-a-fragment-for-each-pixel
   */
  it("picks dimensions per call, not at compile time", () => {
    const build = () => Fn(() => fragCoord().x)();
    const fn = compileJSGrid(build as any, { name: "main", params: [] });
    expect(Array.from(fn({}, 2, 1))).toEqual([0.5, 1.5]);
    expect(Array.from(fn({}, 4, 1))).toEqual([0.5, 1.5, 2.5, 3.5]);
    expect(Array.from(fn({}, 1, 1))).toEqual([0.5]);
  });
  /**
   * @canon spec-a-cpu-routine-answers-one-fragment-per-call
   */
  it("the same compiled function still works as a plain single-pixel call — draw() is a choice per call, not a compile mode", () => {
    const scale = uniform("float");
    const build = () => Fn(() => fragCoord().x.mul(scale))();
    const fn = compileJSProgram(build as any, { name: "main", params: [] });
    expect(fn.run({ uniforms: { [scale.name]: 2 }, fragCoord: [3, 0] })).toBe(6);
    expect(Array.from(fn.draw({ uniforms: { [scale.name]: 2 } }, 2, 1))).toEqual([1, 3]);
  });
  /**
   * @canon spec-a-cpu-grid-evaluates-a-fragment-for-each-pixel
   * @canon spec-wasm-and-js-give-the-same-float-bits
   */
  it("matches compileWasmRoutine's draw() output for the same program", () => {
    const build = () => Fn(() => fragCoord().x.add(fragCoord().y.mul(2)))();
    const jsFn = compileJSGrid(build as any, { name: "main", params: [] });
    const wasmFn = compileWasmGrid(build as any, { name: "main", params: [] });
    expect(Array.from(jsFn({}, 3, 3))).toEqual(Array.from(wasmFn({}, 3, 3)));
  });
  /**
   * @canon spec-a-cpu-program-reads-its-inputs-by-slot
   */
  it("declares its uniform inside the build function itself, not just before it", () => {
    // compileJSRoutine reads the root's result type for draw() by calling the build
    // function — it must do so exactly once. A build function that declares
    // its own uniform() (the idiom most tests in this file use, just always
    // with the uniform hoisted above the compileJSRoutine call rather than inside
    // the closure passed to it) has to see that same call reflected in the
    // compiled source; a second, throwaway invocation would declare a second,
    // differently-named uniform and leave `tex` pointing at the wrong one.
    let tex!: any;
    const build = () =>
      Fn(() => {
        tex = uniform("float");
        return fragCoord().x.mul(tex);
      })();
    const fn = compileJSProgram(build as any, { name: "main", params: [] });
    expect(fn.run({ uniforms: { [tex.name]: 2 }, fragCoord: [3, 0] })).toBe(6);
    expect(Array.from(fn.draw({ uniforms: { [tex.name]: 2 } }, 2, 1))).toEqual([1, 3]);
  });
  /**
   * @canon spec-a-cpu-compute-stage-runs-one-invocation-per-index
   */
  it("compiles storage()/invocationIndex() into a per-call array-indexed program", () => {
    // The same per-invocation model WGSL's compute path gives storage(): the
    // compiled fn is called once per element, with `index` naming which one
    // and `storages` holding the whole backing arrays for it to read/write.
    let dt!: UniformNode<"float">;
    const velNode = instancedArray(3, "float").toReadOnly();
    const posNode = instancedArray(3, "float");
    const build = () =>
      Fn(() => {
        dt = uniform("float");
        const i = invocationIndex();
        posNode.element(i).addAssign(velNode.element(i).mul(dt));
      })();
    const fn = compileJSProgram(build as any, { name: "step", params: [] });

    const pos = new Float32Array([0, 10, 20]);
    const vel = new Float32Array([1, 2, 3]);
    for (let i = 0; i < pos.length; i++) {
      fn.run({ storages: { [velNode.name]: vel, [posNode.name]: pos }, uniforms: { [dt.name]: 2 }, index: i });
    }
    expect(Array.from(pos)).toEqual([2, 14, 26]);
  });
});
