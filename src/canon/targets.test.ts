import { afterAll, describe, expect, it } from "vitest";
import {
  bool,
  builtinFragDepth,
  builtinPosition,
  bvec3,
  EPSILON,
  float,
  Fn,
  fragCoord,
  HALF_PI,
  If,
  int,
  ivec2,
  Loop,
  mat2,
  mat2x3,
  mat3,
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
import { compileGlsl } from "../glsl";
import { compileWgsl } from "../wgsl";
import { compileJSRoutine } from "../js";
import { compileWasmRoutine } from "../wasm";
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
    ["JS", () => compileJSRoutine(build, { ...none, stage })],
    ["WASM", () => compileWasmRoutine(build, { ...none, stage })],
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
   * @canon spec-a-scalar-comparison-gives-a-bool
   */
  it("compares scalars on every target", () => {
    expect(evaluateRecording((a, b) => asFloat(a.lessThan(b)).add(asFloat(a.greaterThanEqual(b)).mul(10)), [1, 2])).toBe(1);
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
      const c = vec3(a, 2, 3).lessThan(vec3(2, 2, 4)).not();
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
      return asFloat(p.and(q)).add(asFloat(p.or(q)).mul(10)).add(asFloat(q.not()).mul(100));
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
    expect(evaluateRecording((a) => vec2(a, 4).length().add(vec2(a, 4).dot(vec2(1, 1))), [3])).toBe(12);
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
    expect(evaluateRecording((a) => vec3(a, 1, 1).mul([1, 2, 3]).dot(vec3(1, 1, 1)), [2])).toBe(7);
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
    const build = (a: Node<"float">) => mat2x3(vec3(a, 2, 3), vec3(4, 5, 6)).element(int(1)).dot(vec3(1, 10, 100));
    expect(evaluateRecording(build, [1])).toBe(654);
  });

  /**
   * @canon spec-a-scalar-matrix-is-a-diagonal
   */
  it("builds a diagonal matrix from one scalar on every target", () => {
    const build = (a: Node<"float">) => {
      const m = mat3(a);
      return m.element(int(0)).x.add(m.element(int(0)).y.mul(10)).add(m.element(int(2)).z.mul(100));
    };
    expect(evaluateRecording(build, [2])).toBe(202);
  });

  /**
   * @canon spec-a-matrix-element-is-a-column
   */
  it("reads a matrix column as a vector on every target", () => {
    expect(evaluateRecording((a) => mat2(vec2(a, 2), vec2(3, 4)).element(int(1)), [1])).toEqual([3, 4]);
  });

  /**
   * @canon spec-a-transpose-swaps-the-shape
   */
  it("transposes a matrix that is not square on every target", () => {
    const build = (a: Node<"float">) => mat2x3(vec3(a, 2, 3), vec3(4, 5, 6)).transpose().element(int(2));
    expect(evaluateRecording(build, [1])).toEqual([3, 6]);
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
    expect(evaluateRecording((a, b) => a.add(b).mod(a.sub(b)).mul(a.div(b.add(1))), [7, 2])).toBe(4 * (7 / 3));
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
    expectOnEveryTarget("vertex", () => Fn(() => vec4(builtinFragDepth(), 0, 0, 1).toVar())(), /builtinFragDepth|depth|fragment/i);
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
   * @canon spec-an-unset-uniform-reads-zero
   */
  it("reads a uniform the host never set as zero on WASM", () => {
    const u = uniform("float");
    const build = () => Fn(() => u.add(1).toVar())();
    expect(compileWasmRoutine(build, none).run({})).toBe(1);
  });

  /**
   * @canon spec-an-integer-reaches-the-host-as-the-integer-it-is
   */
  it("passes a uint above the largest int to and from JS and WASM unchanged", () => {
    const u = uniform("uint");
    const build = () => Fn(() => u.add(uint(1)).toVar())();
    const ctx = { uniforms: { [u.name]: 4000000000 } };
    expect(compileJSRoutine(build, none).run(ctx)).toBe(4000000001);
    expect(compileWasmRoutine(build, none).run(ctx)).toBe(4000000001);
  });

  /**
   * @canon spec-time-is-one-uniform-everywhere
   */
  it("names the clock _rmsl_time on every target", () => {
    const build = () => Fn(() => vec4(time(), 0, 0, 1).toVar())();
    expect(compileGlsl.fragment(build())).toContain("_rmsl_time");
    expect(compileWgsl.fragment(build())).toContain("_rmsl_time");
    expect(compileJSRoutine(build, none).run({ uniforms: { _rmsl_time: 2 } })).toEqual([2, 0, 0, 1]);
    const wasm = compileWasmRoutine(build, none).run({ uniforms: { _rmsl_time: 2 } }) as any;
    expect(wasm.value ?? wasm).toEqual([2, 0, 0, 1]);
  });
});
