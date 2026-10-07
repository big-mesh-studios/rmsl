import { afterAll, describe, expect, it } from "vitest";
import {
  attribute,
  cross,
  distance,
  dot,
  length,
  inverse,
  transpose,
  bool,
  builtinFragDepth,
  builtinPosition,
  bvec3,
  Discard,
  EPSILON,
  float,
  Fn,
  fragCoord,
  HALF_PI,
  If,
  int,
  ivec2,
  ivec3,
  Loop,
  mat2,
  mat2x3,
  mat3,
  outputStruct,
  PI,
  select,
  time,
  TWO_PI,
  uint,
  uniform,
  uvec2,
  vec2,
  vec3,
  vec4,
  type Node,
} from "../rmsl";
import type { CompileCpuRoutine } from "../backends/cpu";
import { compileGlsl } from "../glsl";
import { compileWgsl } from "../wgsl";
import { compileJS, compileJSRoutine, createJsGrid, compileJSFragment, compileJSVertex } from "../js";
import { compileWasm, compileWasmRoutine, createWasmGrid, compileWasmFragment, compileWasmVertex } from "../wasm";
import { assertRecordedEvaluationsAgree, closeEvaluators, evaluateRecording } from "../testing/shader-eval";

afterAll(async () => {
  await assertRecordedEvaluationsAgree();
  await closeEvaluators();
}, 120_000);

const none = { name: "main", params: [] };

/** 1 where `condition` holds, 0 where it does not, as a float every target reads back. */
const asFloat = (condition: Node<"bool">) => select(condition, float(1), float(0)) as Node<"float">;

/** Compiles `build` as `stage` on every target, refusing or not as `refused` says. */
function expectOnEveryTarget(stage: "vertex" | "fragment", build: () => Node<any>, refused: RegExp | false) {
  const compilers: [string, () => unknown][] = [
    ["GLSL", () => compileGlsl[stage](build())],
    ["WGSL", () => compileWgsl[stage](build())],
    ["JS", () => (stage === "vertex" ? compileJSVertex(build, none) : compileJSFragment(build, none))],
    ["WASM", () => (stage === "vertex" ? compileWasmVertex(build, none) : compileWasmFragment(build, none))],
  ];
  for (const [name, compile] of compilers) {
    if (refused) expect(compile, name).toThrow(refused);
    else expect(compile, name).not.toThrow();
  }
}

describe("each leaf on every target it claims", () => {
  /**
   * @canon spec-a-bare-number-beside-a-float-is-a-float
   */
  it("computes a bare number beside a float as a float on every target", () => {
    expect(evaluateRecording((a) => a.add(2).div(4), [1])).toBe(0.75);
  });

  /**
   * @canon spec-a-bare-number-beside-an-integer-is-an-integer
   */
  it("computes a bare number beside an int as an int on every target", () => {
    expect(evaluateRecording((a) => a.toInt().add(2).div(4).toFloat(), [7])).toBe(2);
  });

  /**
   * @canon spec-round-takes-a-half-to-the-even-integer
   */
  it("rounds a value halfway between two integers to the even one on every target", () => {
    const round = (a: any) => a.round();
    for (const [half, even] of [
      [0.5, 0],
      [1.5, 2],
      [2.5, 2],
      [3.5, 4],
      [-1.5, -2],
      [-2.5, -2],
    ] as const) {
      expect(evaluateRecording(round, [half]), `round(${half})`).toBe(even);
    }
    expect(evaluateRecording(() => float(2.5).round())).toBe(2);
    expect(evaluateRecording((a) => vec3(a, a.add(1), a.add(2)).round(), [0.5])).toEqual(new Float64Array([0, 2, 2]));
  });

  /**
   * `radians` and `degrees` compile to the built-in of GLSL and WGSL that has
   * the name, and compute the same angle on every target.
   *
   * @canon spec-a-math-function-compiles-to-the-builtin-of-the-target
   */
  it("compiles radians and degrees to the built-ins of GLSL and WGSL", () => {
    const angle = uniform("float");
    const build = () => Fn(() => vec4(angle.radians(), angle.degrees(), 0, 1))();
    for (const code of [compileGlsl.fragment(build()), compileWgsl.fragment(build())]) {
      expect(code).toMatch(/\bradians\(/);
      expect(code).toMatch(/\bdegrees\(/);
    }
  });

  /**
   * @canon spec-a-math-function-compiles-to-the-builtin-of-the-target
   */
  it("converts an angle on every target, as a right operand and as a vector", () => {
    expect(evaluateRecording((a) => a.radians(), [180])).toBeCloseTo(Math.PI, 5);
    expect(evaluateRecording((a) => a.degrees(), [Math.PI])).toBeCloseTo(180, 4);
    expect(evaluateRecording((a) => float(10).div(a.radians()), [180])).toBeCloseTo(10 / Math.PI, 4);
    const vector = evaluateRecording((a) => vec3(a, a.mul(2), 0).radians(), [90]) as number[];
    [Math.PI / 2, Math.PI, 0].forEach((x, i) => expect(vector[i]).toBeCloseTo(x, 5));
    // The recorded evaluation skips a program WebAssembly refuses, so a vector of angles is run on it directly.
    const wasm = compileWasmRoutine((a: any) => Fn(() => vec3(a, a.mul(2), 0).degrees().y.toVar())(), {
      name: "main",
      params: [{ name: "a", type: "float" }],
    });
    expect(wasm({ params: { a: Math.PI / 2 } })).toBeCloseTo(180, 6);
  });

  /**
   * @canon spec-a-math-function-compiles-to-the-builtin-of-the-target
   */
  it("refuses the radians or degrees of an integer, which no target has", () => {
    expect(() => Fn(() => (int(90) as any).radians().toVar())()).toThrow(
      /radians\(\) takes a float or a float vector, not int/,
    );
    expect(() => Fn(() => (uvec2(1, 2) as any).degrees().toVar())()).toThrow(
      /degrees\(\) takes a float or a float vector, not uvec2/,
    );
  });

  /**
   * @canon spec-a-scalar-comparison-gives-a-bool
   */
  it("compares scalars on every target", () => {
    expect(
      evaluateRecording((a, b) => asFloat(a.lessThan(b)).add(asFloat(a.greaterThanEqual(b)).mul(10)), [1, 2]),
    ).toBe(1);
  });

  /**
   * @canon spec-a-vector-comparison-gives-a-boolean-vector
   */
  it("compares vectors component by component on every target", () => {
    const build = (a: Node<"float">) => {
      const c = vec3(a, 2, 3).lessThan(vec3(2, 2, 4));
      return asFloat(c.x).add(asFloat(c.y).mul(10)).add(asFloat(c.z).mul(100));
    };
    expect(evaluateRecording(build, [1])).toBe(101);
  });

  /**
   * @canon spec-a-scalar-compared-against-a-vector-is-broadcast
   */
  it("compares a vector against a scalar on every target", () => {
    expect(evaluateRecording((a) => asFloat(vec3(1, 2, 3).lessThan(a).all()), [4])).toBe(1);
    expect(evaluateRecording((a) => asFloat(vec3(1, 2, 3).lessThan(a).all()), [3])).toBe(0);
  });

  /**
   * @canon spec-an-assignment-computes-its-value-before-it-writes
   */
  it("reads an assignment's target as it was before the assignment on every target", () => {
    const crossed = (a: Node<"float">) =>
      Fn(() => {
        const v = vec3(a, 1.13, 0.1).toVar();
        v.assign(cross(v, vec3(0.11, 2, 0.57)));
        return v.x.add(v.y.mul(10)).add(v.z.mul(100));
      })();
    const swizzled = (a: Node<"float">) =>
      Fn(() => {
        const v = vec2(a, 2).toVar();
        v.assign(v.yx);
        return v.x.mul(10).add(v.y);
      })();
    const transposed = (a: Node<"float">) =>
      Fn(() => {
        const m = mat3(a, 2, 3, 4, 5, 6, 7, 8, 9).toVar();
        m.assign(transpose(m));
        return m
          .element(int(0))
          .y.mul(10)
          .add(m.element(int(1)).x);
      })();
    const inverted = (a: Node<"float">) =>
      Fn(() => {
        const m = mat2(a, 2, 3, 4).toVar();
        m.assign(inverse(m));
        return m.element(int(1)).y;
      })();
    const rebuilt = (a: Node<"float">) =>
      Fn(() => {
        const v = vec3(a, 2, 3).toVar();
        v.assign(vec3(v.z, v.x, v.y));
        return v.x.mul(100).add(v.y.mul(10)).add(v.z);
      })();
    const [x, y, z] = [1.13 * 0.57 - 0.1 * 2, 0.1 * 0.11 - 0.37 * 0.57, 0.37 * 2 - 1.13 * 0.11];
    expect(evaluateRecording(crossed, [0.37])).toBeCloseTo(x + y * 10 + z * 100, 12);
    expect(evaluateRecording(swizzled, [1])).toBe(21);
    expect(evaluateRecording(transposed, [1])).toBe(42);
    expect(evaluateRecording(inverted, [1])).toBe(-0.5);
    expect(evaluateRecording(rebuilt, [1])).toBe(312);
    // The WASM target compiles neither transpose nor inverse (#65, #220).
    for (const [build, a] of [
      [crossed, 0.37],
      [swizzled, 1],
      [rebuilt, 1],
    ] as const) {
      const params = [{ name: "a", type: "float" as const }];
      const wasm = compileWasmRoutine((p: any) => build(p), { name: "main", params })({ params: { a } });
      expect(wasm).toBe(compileJSRoutine((p: any) => build(p), { name: "main", params })({ params: { a } }));
    }
  });

  /**
   * @canon spec-length-distance-and-dot-of-a-scalar-treat-it-as-a-vector-of-one
   */
  it("gives length, distance and dot of a scalar as of a vector of one, on every target", () => {
    const build = (a: Node<"float">) =>
      length(a)
        .add(distance(a, float(3)).mul(10))
        .add(dot(a, float(2)).mul(100));
    expect(evaluateRecording(build, [-2])).toBe(2 + 50 - 400);
    const params = [{ name: "a", type: "float" as const }];
    const run = (compile: CompileCpuRoutine) =>
      compile((a: any) => Fn(() => build(a).toVar())(), { name: "main", params })({ params: { a: -2 } });
    expect(run(compileWasmRoutine as CompileCpuRoutine)).toBe(run(compileJSRoutine as CompileCpuRoutine));
  });

  /**
   * @canon spec-length-distance-and-dot-of-a-scalar-treat-it-as-a-vector-of-one
   */
  it("converts an integer scalar to float before length, distance and dot, on every target", () => {
    const build = (a: Node<"float">) => {
      const i = a.toInt();
      return length(i)
        .add(distance(i, int(3)).mul(10))
        .add(dot(i, int(2)).mul(100))
        .add(dot(uint(7), uint(5)).mul(1000));
    };
    expect(evaluateRecording(build, [-2])).toBe(2 + 50 - 400 + 35000);
    const params = [
      { name: "a", type: "int" as const },
      { name: "b", type: "uint" as const },
    ];
    const run = (compile: CompileCpuRoutine, float: "f64" | "f32") =>
      compile((a: any, b: any) => Fn(() => dot(a, a).add(length(b)).add(distance(a, b)).toVar())(), {
        name: "main",
        params,
        float,
      })({ params: { a: 70000, b: 4294967295 } });
    expect(run(compileJSRoutine as CompileCpuRoutine, "f64")).toBe(70000 * 70000 + 4294967295 + (4294967295 - 70000));
    for (const float of ["f64", "f32"] as const) {
      expect(run(compileWasmRoutine as CompileCpuRoutine, float)).toBe(
        run(compileJSRoutine as CompileCpuRoutine, float),
      );
    }
  });

  /**
   * @canon spec-a-boolean-vector-reduces-with-all-or-any
   */
  it("reduces a boolean vector with all and any on every target", () => {
    const build = (a: Node<"float">) => {
      const c = vec3(a, 2, 3).lessThan(vec3(2, 2, 4));
      return asFloat(c.all()).add(asFloat(c.any()).mul(10));
    };
    expect(evaluateRecording(build, [1])).toBe(10);
  });

  /**
   * @canon spec-not-negates-a-boolean-vector-component-wise
   */
  it("negates a boolean vector component by component on every target", () => {
    const build = (a: Node<"float">) => {
      const c = vec3(a, 2, 3)
        .lessThan(vec3(2, 2, 4))
        .not();
      return asFloat(c.x).add(asFloat(c.y).mul(10)).add(asFloat(c.z).mul(100));
    };
    expect(evaluateRecording(build, [1])).toBe(10);
  });

  /**
   * @canon spec-and-or-and-not-combine-bools
   */
  it("combines bools with and, or and not on every target", () => {
    const build = (a: Node<"float">) => {
      const p = a.greaterThan(0);
      const q = a.greaterThan(5);
      return asFloat(p.and(q))
        .add(asFloat(p.or(q)).mul(10))
        .add(asFloat(q.not()).mul(100));
    };
    expect(evaluateRecording(build, [1])).toBe(110);
  });

  /**
   * @canon spec-select-picks-one-of-two-values
   */
  it("selects between two values on every target", () => {
    const build = (a: Node<"float">) => select(a.greaterThan(0), a.mul(2), a.negate()) as Node<"float">;
    expect(evaluateRecording(build, [3])).toBe(6);
    expect(evaluateRecording(build, [-3])).toBe(3);
  });

  /**
   * @canon spec-a-compound-assignment-writes-the-result-back
   */
  it("writes the result of a compound assignment back on every target", () => {
    const build = (a: Node<"float">) =>
      Fn(() => {
        const v = a.toVar();
        v.addAssign(1);
        v.mulAssign(2);
        v.subAssign(1);
        v.divAssign(3);
        return v;
      })();
    expect(evaluateRecording(build, [4])).toBe(3);
  });

  /**
   * @canon spec-a-geometric-function-compiles-to-the-builtin-of-the-target
   */
  it("computes the geometric functions on every target", () => {
    const build = (a: Node<"float">) => {
      const v = vec3(a, 4, 0);
      const n = vec3(0, 1, 0);
      return v
        .length()
        .add(v.dot(n))
        .add(v.distance(vec3(0, 0, 0)))
        .add(v.normalize().x)
        .add(v.cross(n).z)
        .add(v.reflect(n).y)
        .add(n.faceForward(vec3(0, -1, 0), n).y);
    };
    // 5 + 4 + 5 + 0.6 + 3 + -4 + 1: faceForward keeps the normal, which faces against the incident vector
    expect(evaluateRecording(build, [3])).toBeCloseTo(14.6, 12);
  });

  /**
   * @canon spec-a-reducing-operation-has-a-scalar-type
   */
  it("reduces vectors to scalars with length, distance and dot on every target", () => {
    expect(
      evaluateRecording(
        (a) =>
          vec2(a, 4)
            .length()
            .add(vec2(a, 4).dot(vec2(1, 1))),
        [3],
      ),
    ).toBe(12);
  });

  /**
   * @canon spec-step-and-smoothstep-take-their-type-from-the-value
   */
  it("steps and smoothsteps a vector against scalar edges on every target", () => {
    const build = (a: Node<"float">) => {
      const s = vec2(a, 0.25).step(0.5);
      const t = vec2(a, 0.25).smoothstep(0, 1);
      return s.x.add(s.y.mul(10)).add(t.y.mul(100));
    };
    expect(evaluateRecording(build, [0.75])).toBeCloseTo(1 + 15.625, 12);
  });

  /**
   * @canon spec-an-operation-no-target-has-is-composed
   */
  it("computes the composed operations on every target", () => {
    const build = (a: Node<"float">) =>
      a
        .saturate()
        .add(a.oneMinus())
        .add(a.reciprocal())
        .add(a.pow2())
        .add(a.lengthSq())
        .add(asFloat(a.greaterThan(0).xor(a.greaterThan(1))));
    // 1 + -1 + 0.5 + 4 + 4 + 0
    expect(evaluateRecording(build, [2])).toBe(8.5);
  });

  /**
   * @canon spec-the-tsl-constants-are-float-literals
   */
  it("reads the TSL constants on every target", () => {
    const build = (a: Node<"float">) => a.mul(PI).add(TWO_PI).add(HALF_PI).add(EPSILON);
    expect(evaluateRecording(build, [1])).toBeCloseTo(3.5 * Math.PI + 1e-6, 12);
  });

  /**
   * @canon spec-a-literal-compiles-to-a-literal-of-its-type
   */
  it("computes with int, uint, bool and integer vector literals on every target", () => {
    const build = (a: Node<"float">) =>
      a
        .add(int(-3).toFloat())
        .add(uint(4).toFloat())
        .add(asFloat(bool(true)))
        .add(ivec2(5, 6).y.toFloat())
        .add(uvec2(7, 8).x.toFloat())
        .add(asFloat(bvec3(true, false, true).any()));
    expect(evaluateRecording(build, [1])).toBe(17);
  });

  /**
   * @canon spec-a-javascript-array-is-a-vector-of-its-length
   */
  it("reads a JavaScript array as a vector on every target", () => {
    expect(
      evaluateRecording(
        (a) =>
          vec3(a, 1, 1)
            .mul([1, 2, 3])
            .dot(vec3(1, 1, 1)),
        [2],
      ),
    ).toBe(7);
  });

  /**
   * @canon spec-a-scalar-fills-every-component-of-a-vector
   */
  it("fills every component of a vector from one scalar on every target", () => {
    expect(evaluateRecording((a) => vec3(a).dot(vec3(1, 10, 100)), [2])).toBe(222);
  });

  /**
   * @canon spec-parts-fill-a-vector-in-order
   */
  it("fills a vector from parts in order on every target", () => {
    expect(evaluateRecording((a) => vec4(vec2(a, 2), 3, 4).dot(vec4(1, 10, 100, 1000)), [1])).toBe(4321);
    expect(evaluateRecording((a) => vec2(vec3(a, 2, 3)).dot(vec2(1, 10)), [1])).toBe(21);
  });

  /**
   * @canon spec-a-matrix-is-built-from-its-columns
   */
  it("builds a matrix from its columns on every target", () => {
    const build = (a: Node<"float">) =>
      mat2x3(vec3(a, 2, 3), vec3(4, 5, 6))
        .element(int(1))
        .dot(vec3(1, 10, 100));
    expect(evaluateRecording(build, [1])).toBe(654);
  });

  /**
   * @canon spec-a-scalar-matrix-is-a-diagonal
   */
  it("builds a diagonal matrix from one scalar on every target", () => {
    const build = (a: Node<"float">) => {
      const m = mat3(a);
      return m
        .element(int(0))
        .x.add(m.element(int(0)).y.mul(10))
        .add(m.element(int(2)).z.mul(100));
    };
    expect(evaluateRecording(build, [2])).toBe(202);
  });

  /**
   * @canon spec-a-matrix-element-is-a-column
   */
  it("reads a matrix column as a vector on every target", () => {
    expect(evaluateRecording((a) => mat2(vec2(a, 2), vec2(3, 4)).element(int(1)), [1])).toEqual(
      new Float64Array([3, 4]),
    );
  });

  /**
   * @canon spec-a-bool-converts-to-one-or-zero-and-a-number-to-whether-it-is-nonzero
   */
  it("converts between bool and the numeric types the same way on every target", () => {
    const flag = (a: Node<"float">) => bool(a).select(10, 20) as Node<"float">;
    expect(evaluateRecording(flag, [0.5])).toBe(10);
    expect(evaluateRecording(flag, [-0.25])).toBe(10);
    expect(evaluateRecording(flag, [0])).toBe(20);
    expect(evaluateRecording((a) => bool(a.toInt()).select(10, 20) as Node<"float">, [3])).toBe(10);
    expect(evaluateRecording((a) => bool(a.toInt()).select(10, 20) as Node<"float">, [0])).toBe(20);
    expect(evaluateRecording((a) => bool(a.toUint()).select(10, 20) as Node<"float">, [0])).toBe(20);
    expect(evaluateRecording((a) => float(a.greaterThan(0)).add(1) as Node<"float">, [2])).toBe(2);
    expect(evaluateRecording((a) => float(a.greaterThan(0)).add(1) as Node<"float">, [-2])).toBe(1);
    expect(evaluateRecording((a) => int(a.greaterThan(0)).add(1).toFloat(), [2])).toBe(2);
    expect(evaluateRecording((a) => uint(a.lessThan(0)).add(1).toFloat(), [2])).toBe(1);
  });

  /**
   * @canon spec-a-matrix-constructor-takes-a-scalar-node-wherever-it-takes-a-number
   */
  it("builds a matrix from numbers and scalar nodes on every target", () => {
    expect(evaluateRecording((a) => mat2(a, 1, 2, 4).element(int(0)), [3])).toEqual(new Float64Array([3, 1]));
    expect(evaluateRecording((a) => mat3(1, a, 0, 0, 1, 0, 0, 0, a.add(1)).element(int(2)).z, [3])).toBe(4);
    expect(evaluateRecording((a) => mat2x3(a, 0, 0, 0, 1, a).element(int(1)), [5])).toEqual(
      new Float64Array([0, 1, 5]),
    );
  });

  /**
   * @canon spec-a-matrix-constructor-takes-a-scalar-node-wherever-it-takes-a-number
   */
  it("refuses a matrix built from the wrong number of values when one is a node", () => {
    expect(() => mat2(float(3), 1, 2)).toThrow(/takes 4 values/);
  });

  /**
   * @canon spec-a-transpose-swaps-the-shape
   */
  it("transposes a matrix that is not square on every target", () => {
    const build = (a: Node<"float">) =>
      mat2x3(vec3(a, 2, 3), vec3(4, 5, 6))
        .transpose()
        .element(int(2));
    expect(evaluateRecording(build, [1])).toEqual(new Float64Array([3, 6]));
  });

  /**
   * @canon spec-a-matrix-product-has-the-shape-of-the-product
   */
  it("multiplies matrices to the shape of the product on every target", () => {
    const build = (a: Node<"float">) =>
      mat2x3(vec3(a, 0, 0), vec3(0, 1, 0))
        .mul(vec2(2, 3))
        .dot(vec3(1, 10, 100));
    expect(evaluateRecording(build, [1])).toBe(32);
  });

  /**
   * @canon spec-a-matrix-times-a-shorter-vector-promotes-it
   */
  it("promotes a shorter vector by a last component of 1 on every target", () => {
    const build = (a: Node<"float">) =>
      mat3(vec3(1, 0, 0), vec3(0, 1, 0), vec3(a, 5, 1))
        .mul(vec2(1, 2))
        .dot(vec2(1, 10));
    expect(evaluateRecording(build, [4])).toBe(5 + 70);
  });

  /**
   * @canon spec-an-operand-that-is-an-expression-keeps-its-grouping
   */
  it("keeps the grouping of an operand that is an expression on every target", () => {
    expect(
      evaluateRecording(
        (a, b) =>
          a
            .add(b)
            .mod(a.sub(b))
            .mul(a.div(b.add(1))),
        [7, 2],
      ),
    ).toBe(4 * (7 / 3));
  });

  /**
   * @canon spec-an-operand-that-is-an-expression-keeps-its-grouping
   */
  it("keeps the grouping of a scalar fract and inverse square root used as operands on every target", () => {
    const build = (a: Node<"float">) => a.fract().mul(2).add(float(8).div(a.inverseSqrt()));
    expect(evaluateRecording(build, [2.75])).toBeCloseTo(0.75 * 2 + 8 * Math.sqrt(2.75), 10);
  });

  /**
   * @canon spec-every-node-is-emitted-once
   */
  it("runs a block that two values share once on every target", () => {
    const build = (a: Node<"float">) =>
      Fn(() => {
        const acc = a.toVar();
        If(acc.greaterThan(0), () => {
          acc.assign(acc.add(10));
        });
        return acc.add(acc.mul(2));
      })();
    expect(evaluateRecording(build, [1])).toBe(33);
  });

  /**
   * @canon spec-loop-runs-its-body-count-times
   */
  it("runs a Loop body count times on every target", () => {
    const build = (a: Node<"float">) =>
      Fn(() => {
        const total = a.toVar();
        Loop(int(4), (i) => {
          total.addAssign(i.toFloat());
        });
        return total;
      })();
    expect(evaluateRecording(build, [1])).toBe(7);
  });

  /**
   * @canon spec-an-fn-returns-what-its-body-returns
   */
  it("gives what a called Fn returns on every target", () => {
    const twice = Fn((x: Node<"float">) => x.mul(2));
    const empty = Fn(() => {});
    const build = (a: Node<"float">) =>
      Fn(() => {
        empty();
        return twice(twice(a));
      })();
    expect(evaluateRecording(build, [3])).toBe(12);
  });

  /**
   * @canon spec-var-is-to-var
   */
  it("makes a variable with var() on every target", () => {
    const build = (a: Node<"float">) =>
      Fn(() => {
        const v = a.var();
        v.addAssign(1);
        return v.add(a);
      })();
    expect(evaluateRecording(build, [2])).toBe(5);
  });

  /**
   * WASM refuses the inverse for another reason too: it compiles no inverse at
   * all yet (#65).
   *
   * @canon spec-only-a-square-matrix-is-inverted
   */
  it("refuses to invert a matrix that is not square on every target", () => {
    const build = () => Fn(() => vec4((uniform("mat2x3") as any).inverse().element(int(0)), 1).toVar())();
    expectOnEveryTarget("fragment", build, /square|unsupported/i);
  });

  /**
   * @canon spec-frag-coord-is-read-only-in-a-fragment-stage
   */
  it("refuses fragCoord in a vertex stage on every target", () => {
    expectOnEveryTarget("vertex", () => Fn(() => vec4(fragCoord(), 0, 1).toVar())(), /fragCoord|fragment|vertex/i);
  });

  /**
   * @canon spec-the-fragment-depth-is-written-only-in-a-fragment-stage
   */
  it("refuses the fragment depth in a vertex stage on every target", () => {
    expectOnEveryTarget(
      "vertex",
      () => Fn(() => vec4(builtinFragDepth(), 0, 0, 1).toVar())(),
      /builtinFragDepth|depth|fragment/i,
    );
  });

  /**
   * @canon spec-the-position-is-read-only-in-a-vertex-stage
   */
  it("refuses reading the position in a fragment stage on every target", () => {
    expectOnEveryTarget("fragment", () => Fn(() => vec4(builtinPosition().x, 0, 0, 1).toVar())(), /fragment|position/i);
  });

  /**
   * @canon spec-a-vertex-stage-without-a-position-is-refused
   */
  it("refuses a vertex stage without a position on every target", () => {
    expectOnEveryTarget("vertex", () => Fn(() => vec3(1, 2, 3).toVar())() as any, /vertex shader|position/i);
  });

  /**
   * The colour a fragment stage with no output writes, on every target: the
   * GLSL and WGSL source declares a `vec4` colour output, and JS and WASM
   * give the four channels.
   */
  function colourOnEveryTarget(build: () => Node<any>, expected: number[]) {
    expect(compileGlsl.fragment(build()), "GLSL").toContain("out vec4");
    expect(compileWgsl.fragment(build()), "WGSL").toMatch(/: vec4<f32>/);
    const js = compileJSFragment(build, { ...none })({}) as any;
    expect(js.value ?? js, "JS").toEqual(new Float64Array(expected));
    const wasm = compileWasmFragment(build, { ...none })({}) as any;
    expect(wasm.value ?? wasm, "WASM").toEqual(new Float64Array(expected));
  }

  /**
   * @canon spec-a-vec4-result-is-the-colour
   */
  it("writes a vec4 result as it is on every target", () => {
    colourOnEveryTarget(() => Fn(() => vec4(0.5, 0.25, 0.75, 0.125).toVar())(), [0.5, 0.25, 0.75, 0.125]);
  });

  /**
   * @canon spec-a-vec3-result-takes-an-opaque-alpha
   */
  it("writes a vec3 result with an alpha of one on every target", () => {
    colourOnEveryTarget(() => Fn(() => vec3(0.5, 0.25, 0.75).toVar())(), [0.5, 0.25, 0.75, 1]);
  });

  /**
   * @canon spec-a-vec2-result-takes-a-zero-blue-and-an-opaque-alpha
   */
  it("writes a vec2 result with a blue of zero and an alpha of one on every target", () => {
    colourOnEveryTarget(() => Fn(() => vec2(0.5, 0.25).toVar())(), [0.5, 0.25, 0, 1]);
  });

  /**
   * @canon spec-a-scalar-result-fills-every-channel
   */
  it("writes a scalar result into every channel on every target", () => {
    colourOnEveryTarget(() => Fn(() => float(0.5).toVar())(), [0.5, 0.5, 0.5, 0.5]);
    colourOnEveryTarget(() => Fn(() => int(2).toVar())(), [2, 2, 2, 2]);
    colourOnEveryTarget(() => Fn(() => bool(true).toVar())(), [1, 1, 1, 1]);
  });

  /**
   * @canon spec-a-vec3-result-takes-an-opaque-alpha
   */
  it("draws a vec3 result with an alpha of one in the JS and WASM rasterizers", () => {
    const position = attribute("vec3");
    const vertex = () => Fn(() => builtinPosition().assign(vec4(position, 1)))() as any;
    const fragment = () => Fn(() => vec3(1, 0, 0).toVar())() as any;
    const triangle = new Float64Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]);
    const inputs = { attributes: { [position.name]: triangle } };
    const options = { width: 1, height: 1 };
    const js = compileJS(vertex, fragment, { attributeTypes: { [position.name]: "vec3" } });
    expect(Array.from(js.draw(inputs, options))).toEqual([1, 0, 0, 1]);
    const wasm = compileWasm(vertex, fragment);
    expect(Array.from(wasm.draw(inputs, options))).toEqual([1, 0, 0, 1]);
  });

  /**
   * @canon spec-an-integer-or-boolean-vector-result-converts-to-a-float-vector-first
   */
  it("writes an integer or boolean vector result as the float vector it converts to on every target", () => {
    colourOnEveryTarget(() => Fn(() => ivec3(1, 2, 3).toVar())(), [1, 2, 3, 1]);
    colourOnEveryTarget(() => Fn(() => uvec2(4, 5).toVar())(), [4, 5, 0, 1]);
    colourOnEveryTarget(() => Fn(() => bvec3(true, false, true).toVar())(), [1, 0, 1, 1]);
  });

  /** The bytes a routine adapter puts on its canvas for one pixel of `draw`. */
  function shownBy(create: typeof createJsGrid, draw: Node<any>): number[] {
    const hadImageData = "ImageData" in globalThis;
    if (!hadImageData) {
      (globalThis as any).ImageData = class {
        data: Uint8ClampedArray;
        constructor(width: number, height: number) {
          this.data = new Uint8ClampedArray(width * height * 4);
        }
      };
    }
    try {
      let shown!: ImageData;
      const canvas = {
        width: 1,
        height: 1,
        getContext: () => ({ putImageData: (image: ImageData) => (shown = image) }),
      };
      const adapter = create({ draw, name: "shade" });
      adapter.attach(canvas as unknown as HTMLCanvasElement);
      adapter.draw();
      return Array.from(shown.data);
    } finally {
      if (!hadImageData) delete (globalThis as any).ImageData;
    }
  }

  /**
   * @canon spec-a-vec3-result-takes-an-opaque-alpha
   */
  it("shows a vec3 result with an opaque alpha on the JS and WASM routine adapters", () => {
    for (const create of [createJsGrid, createWasmGrid]) {
      expect(shownBy(create, Fn(() => vec3(1, 0.5, 0))())).toEqual([255, 128, 0, 255]);
    }
  });

  /**
   * @canon spec-a-vec2-result-takes-a-zero-blue-and-an-opaque-alpha
   */
  it("shows a vec2 result with a zero blue on the JS and WASM routine adapters", () => {
    for (const create of [createJsGrid, createWasmGrid]) {
      expect(shownBy(create, Fn(() => vec2(1, 0.5))())).toEqual([255, 128, 0, 255]);
    }
  });

  /**
   * @canon spec-a-scalar-result-fills-every-channel
   */
  it("shows a scalar result in every channel on the JS and WASM routine adapters", () => {
    for (const create of [createJsGrid, createWasmGrid]) {
      expect(shownBy(create, Fn(() => float(0.5))())).toEqual([128, 128, 128, 128]);
    }
  });

  /**
   * @canon spec-a-result-that-has-no-colour-is-refused
   */
  it("refuses a fragment result that has no colour on every target", () => {
    expectOnEveryTarget("fragment", () => Fn(() => mat2(1, 2, 3, 4).toVar())(), /fragment shader|colour/i);
    expectOnEveryTarget("fragment", () => Fn(() => mat2x3(1, 0, 0, 0, 1, 0).toVar())(), /fragment shader|colour/i);
  });

  /**
   * @canon spec-a-fragment-stage-may-write-no-colour
   */
  it("compiles a fragment stage that returns nothing on every target", () => {
    const build = () => Fn(() => {})() as any;
    expect(() => compileGlsl.fragment(build())).not.toThrow();
    expect(() => compileWgsl.fragment(build())).not.toThrow();
    expect(() => compileJSFragment(build, { ...none })).not.toThrow();
    expect(() => compileWasmFragment(build, { ...none })).not.toThrow();
    expect(() => compileWasmFragment(build, { ...none })({})).not.toThrow();
  });

  /**
   * The rasterizer takes a fragment stage that writes no colour, or only the
   * depth, covers the pixel with it and writes nothing there. A colour fragment
   * on the same triangle writes the pixel, which shows the triangle covers it,
   * and a rasterizer that copied a colour from the colourless stage would read
   * memory out of range and trap.
   *
   * @canon spec-a-cpu-rasterizer-draws-no-pixel-for-a-fragment-stage-that-writes-no-colour
   */
  it("draws a fragment stage that writes no colour on the WASM rasterizer, leaving the pixel as it was", () => {
    const pos = attribute("vec3");
    const vertex = () => Fn(() => builtinPosition().assign(vec4(pos, 1)))();
    const ctx = { attributes: { [pos.name]: new Float64Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]) } };
    const colour = () => Fn(() => vec4(1, 1, 1, 1))();
    const nothing = () => Fn(() => {})();
    const depthOnly = () =>
      Fn(() => {
        builtinFragDepth().assign(float(0.5));
      })();
    expect(Array.from(compileWasm(vertex as any, colour as any).draw(ctx, { width: 1, height: 1 }))).toEqual([
      1, 1, 1, 1,
    ]);
    for (const fragment of [nothing, depthOnly]) {
      const routine = compileWasm(vertex as any, fragment as any);
      expect(Array.from(routine.draw(ctx, { width: 1, height: 1 }))).toEqual([0, 0, 0, 0]);
    }
  });

  /**
   * @canon spec-an-output-struct-writes-each-member-at-its-position
   */
  it("refuses a fragment stage that returns an outputStruct on the WASM rasterizer, which draws a colour", () => {
    const pos = attribute("vec3");
    const vertex = () => Fn(() => builtinPosition().assign(vec4(pos, 1)))();
    const declared = () => Fn(() => outputStruct(vec4(1, 0, 0, 1)))();
    expect(() => compileWasm(vertex as any, declared as any)).toThrow(/cannot declare outputs/);
  });

  /**
   * @canon spec-a-fragment-stage-may-write-no-colour
   */
  it("compiles a fragment stage that only discards on both CPU targets", () => {
    const build = () =>
      Fn(() => {
        If(uniform("float").greaterThan(0), () => {
          Discard();
        });
      })() as any;
    expect(() => compileJSFragment(build, { ...none })).not.toThrow();
    expect(() => compileWasmFragment(build, { ...none })).not.toThrow();
    // An unset uniform reads zero, so the discard does not run; a routine that runs it still returns.
    expect(() => compileWasmFragment(build, { ...none })({})).not.toThrow();
    expect(() => compileWasmFragment(build, { ...none })({ uniforms: {} })).not.toThrow();
  });

  /**
   * @canon spec-an-unset-uniform-reads-zero
   */
  it("reads a uniform the host never set as zero on WASM", () => {
    const u = uniform("float");
    const build = () => Fn(() => u.add(1).toVar())();
    expect(compileWasmRoutine(build, none)({})).toBe(1);
  });

  /**
   * @canon spec-an-integer-reaches-the-host-as-the-integer-it-is
   */
  it("passes a uint above the largest int to and from JS and WASM unchanged", () => {
    const u = uniform("uint");
    const build = () => Fn(() => u.add(uint(1)).toVar())();
    const ctx = { uniforms: { [u.name]: 4000000000 } };
    expect(compileJSRoutine(build, none)(ctx)).toBe(4000000001);
    expect(compileWasmRoutine(build, none)(ctx)).toBe(4000000001);
  });

  /**
   * @canon spec-time-is-one-uniform-everywhere
   */
  it("names the clock _rmsl_time on every target", () => {
    const build = () => Fn(() => vec4(time(), 0, 0, 1).toVar())();
    expect(compileGlsl.fragment(build())).toContain("_rmsl_time");
    expect(compileWgsl.fragment(build())).toContain("_rmsl_time");
    expect(compileJSRoutine(build, none)({ uniforms: { _rmsl_time: 2 } })).toEqual(new Float64Array([2, 0, 0, 1]));
    const wasm = compileWasmRoutine(build, none)({ uniforms: { _rmsl_time: 2 } }) as any;
    expect(wasm.value ?? wasm).toEqual(new Float64Array([2, 0, 0, 1]));
  });
});
