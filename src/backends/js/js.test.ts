import { describe, it, expect, afterAll } from "vitest";
import {
  evaluateRecording,
  assertRecordedEvaluationsAgree,
  closeEvaluators,
  type CpuOnlyReason,
} from "../../testing/shader-eval";
import { compileJS, compileJSFn, type CpuTextureData } from "../../js";
import { compileWasmRoutine } from "../../wasm";
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
  storage,
  invocationIndex,
  type UniformNode,
  output,
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
    const fn = compileJS(build as any, { name: "main", params, ...compileOpts });
    const ctx: any = { params: Object.fromEntries(args.map((a, i) => [`a${i}`, a])) };
    const value = fn.invoke(ctx);
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
function evalMatrix(build: (...args: Node<"float">[]) => any, args: number[] = []): number[] {
  return evaluateRecording(build as any, args) as number[];
}

afterAll(async () => {
  await assertRecordedEvaluationsAgree();
  await closeEvaluators();
}, 120_000);

describe("JS backend: scalar arithmetic", () => {
  it("computes arithmetic", () => {
    expect(evalScalar((a, b) => a.add(b), [2, 3])).toBe(5);
    expect(evalScalar((a, b) => a.sub(b), [7, 3])).toBe(4);
    expect(evalScalar((a, b) => a.mul(b), [3, 4])).toBe(12);
    expect(evalScalar((a, b) => a.div(b), [8, 2])).toBe(4);
    expect(evalScalar((a) => a.negate(), [3])).toBe(-3);
  });

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

  it("casts float to int and back", () => {
    expect(evalScalar((a) => a.toInt().toFloat(), [2.7])).toBe(2);
    expect(evalScalar((a) => a.toInt().toFloat(), [-2.7])).toBe(-2);
    expect(evalScalar((a) => a.toUint().toFloat(), [2.7])).toBe(2);
    expect(evalScalar((a) => a.toInt().toBool().toFloat(), [1.5])).toBe(1);
    expect(evalScalar((a) => a.toInt().toBool().toFloat(), [0])).toBe(0);
  });

  it("casts a vector to a scalar through its first component", () => {
    expect(evalScalar((a, b) => vec3(a, b, b).toFloat(), [4.5, 9])).toBe(4.5);
    expect(evalScalar((a, b) => vec3(a, b, b).toInt().toFloat(), [2.7, 9])).toBe(2);
    expect(evalScalar((a, b) => vec2(a, b).toUint().toFloat(), [3.9, 9])).toBe(3);
  });

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

  it("computes floored float modulus", () => {
    expect(evalScalar((a, b) => a.mod(b), [7.5, 2])).toBe(1.5);
    expect(evalScalar((a, b) => a.mod(b), [-7.5, 2])).toBe(0.5);
    expect(evalScalar((a, b) => a.mod(b), [7.5, -2])).toBe(-0.5);
    expect(evalScalar((a, b) => a.mod(b), [-1, 2])).toBe(1);
  });

  it("folds constants to the same value it would compute at runtime", () => {
    const folded = evalScalar(() => float(7).div(float(2)), []);
    const runtime = evalScalar((a, b) => a.div(b), [7, 2]);
    expect(folded).toBe(3.5);
    expect(runtime).toBe(3.5);
  });
});

describe("JS backend: vector arithmetic", () => {
  it("adds, subtracts and scales vectors", () => {
    const f = compileJS((a: any, b: any) => a.add(b), {
      name: "main",
      params: [
        { name: "a", type: "vec3" },
        { name: "b", type: "vec3" },
      ],
    });
    expect(f.invoke({ params: { a: [1, 2, 3], b: [10, 20, 30] } })).toEqual([11, 22, 33]);

    const g = compileJS((a: any) => a.mul(2), {
      name: "main",
      params: [{ name: "a", type: "vec3" }],
    });
    expect(g.invoke({ params: { a: [1, 2, 3] } })).toEqual([2, 4, 6]);

    const h = compileJS((a: any) => a.sub(vec3(1, 1, 1)), {
      name: "main",
      params: [{ name: "a", type: "vec3" }],
    });
    expect(h.invoke({ params: { a: [5, 5, 5] } })).toEqual([4, 4, 4]);
  });

  it("broadcasts a lone scalar vector constructor across every component", () => {
    // GLSL/WGSL vec3(2.0) is (2.0, 2.0, 2.0), and the JS backend must match.
    const f = compileJS(() => vec3(2), { name: "main", params: [] });
    expect(f.invoke({})).toEqual([2, 2, 2]);
    const g = compileJS(() => vec4(0.5), { name: "main", params: [] });
    expect(g.invoke({})).toEqual([0.5, 0.5, 0.5, 0.5]);
    const v = compileJS(() => vec2(-1), { name: "main", params: [] });
    expect(v.invoke({})).toEqual([-1, -1]);
  });

  it("computes dot, cross, length, distance and normalize", () => {
    const dot = compileJS((a: any, b: any) => a.dot(b), {
      name: "main",
      params: [
        { name: "a", type: "vec3" },
        { name: "b", type: "vec3" },
      ],
    });
    expect(dot.invoke({ params: { a: [1, 2, 3], b: [4, 5, 6] } })).toBe(32);

    const cross = compileJS((a: any, b: any) => a.cross(b), {
      name: "main",
      params: [
        { name: "a", type: "vec3" },
        { name: "b", type: "vec3" },
      ],
    });
    expect(cross.invoke({ params: { a: [1, 0, 0], b: [0, 1, 0] } })).toEqual([0, 0, 1]);

    const len = compileJS((a: any) => a.length(), {
      name: "main",
      params: [{ name: "a", type: "vec3" }],
    });
    approx(len.invoke({ params: { a: [3, 4, 0] } }) as number, 5);

    const dist = compileJS((a: any, b: any) => a.distance(b), {
      name: "main",
      params: [
        { name: "a", type: "vec2" },
        { name: "b", type: "vec2" },
      ],
    });
    approx(dist.invoke({ params: { a: [0, 0], b: [3, 4] } }) as number, 5);

    const norm = compileJS((a: any) => a.normalize(), {
      name: "main",
      params: [{ name: "a", type: "vec3" }],
    });
    const n = norm.invoke({ params: { a: [3, 0, 0] } }) as number[];
    approx(n[0], 1);
    approx(n[1], 0);
    approx(n[2], 0);
  });

  it("computes vector comparisons to boolean vectors", () => {
    const f = compileJS((a: any, b: any) => a.lessThan(b), {
      name: "main",
      params: [
        { name: "a", type: "vec3" },
        { name: "b", type: "vec3" },
      ],
    });
    expect(f.invoke({ params: { a: [1, 5, 3], b: [2, 2, 2] } })).toEqual([true, false, false]);
  });

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

  it("truncates the components a float vector puts in an integer one", () => {
    expect(evalScalar((a, b) => vec3(a, b, b).toIVec3().x.toFloat(), [2.7, 9])).toBe(2);
    expect(evalScalar((a, b) => vec3(a, b, b).toIVec3().x.toFloat(), [-2.7, 9])).toBe(-2);
  });

  it("computes all/any on boolean vectors", () => {
    const f = compileJS((a: any) => a.greaterThan(vec3(0, 0, 0)).all(), {
      name: "main",
      params: [{ name: "a", type: "vec3" }],
    });
    expect(f.invoke({ params: { a: [1, 2, 3] } })).toBe(true);
    expect(f.invoke({ params: { a: [1, 0, 3] } })).toBe(false);
  });

  it("reflects and refracts", () => {
    const f = compileJS((i: any, n: any) => i.reflect(n), {
      name: "main",
      params: [
        { name: "i", type: "vec3" },
        { name: "n", type: "vec3" },
      ],
    });
    // i = -n reflects back to +n
    const r = f.invoke({ params: { i: [0, -1, 0], n: [0, 1, 0] } }) as number[];
    approx(r[0], 0);
    approx(r[1], 1);
    approx(r[2], 0);
  });
});

describe("JS backend: matrices", () => {
  it("multiplies mat4 by vec4 and vec3", () => {
    const m = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1];
    const f4 = compileJS((a: any, v: any) => a.mul(v), {
      name: "main",
      params: [
        { name: "a", type: "mat4" },
        { name: "v", type: "vec4" },
      ],
    });
    expect(f4.invoke({ params: { a: m, v: [1, 2, 3, 1] } })).toEqual([6, 8, 10, 1]);

    const f3 = compileJS((a: any, v: any) => a.mul(v), {
      name: "main",
      params: [
        { name: "a", type: "mat4" },
        { name: "v", type: "vec3" },
      ],
    });
    expect(f3.invoke({ params: { a: m, v: [1, 2, 3] } })).toEqual([6, 8, 10]);
  });

  it("multiplies matrices", () => {
    const id = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const f = compileJS((a: any, b: any) => a.mul(b), {
      name: "main",
      params: [
        { name: "a", type: "mat4" },
        { name: "b", type: "mat4" },
      ],
    });
    expect(f.invoke({ params: { a: id, b: id } })).toEqual(id);
    const translate = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1];
    expect(f.invoke({ params: { a: translate, b: id } })).toEqual(translate);
  });

  it("multiplies non-square matrices at the product shape", () => {
    // mat2x3 (2 cols x 3 rows): a = [[1,2,3],[4,5,6]], column-major.
    const a = [1, 2, 3, 4, 5, 6];
    // mat3x2 (3 cols x 2 rows): b = [[1,0],[0,1],[1,1]], column-major.
    const b = [1, 0, 0, 1, 1, 1];
    const f = compileJS((a: any, b: any) => a.mul(b), {
      name: "main",
      params: [
        { name: "a", type: "mat2x3" },
        { name: "b", type: "mat3x2" },
      ],
    });
    // a (3x2) * b (2x3) = product is mat3 (3x3), column-major.
    // A rows = [1,4],[2,5],[3,6]; B cols = [1,0],[0,1],[1,1].
    // col0 = A*b0 = [1,2,3]; col1 = A*b1 = [4,5,6]; col2 = A*(b0+b1) = [5,7,9].
    expect(f.invoke({ params: { a, b } })).toEqual([1, 2, 3, 4, 5, 6, 5, 7, 9]);
  });

  it("inverts, transposes and takes determinants", () => {
    const inv = compileJS((a: any) => a.inverse(), {
      name: "main",
      params: [{ name: "a", type: "mat2" }],
    });
    // mat2(2,1,3,4) = [[2,3],[1,4]]; inverse = [[0.8,-0.6],[-0.2,0.4]].
    const got = inv.invoke({ params: { a: [2, 1, 3, 4] } }) as number[];
    got.forEach((v, i) => approx(v, [0.8, -0.2, -0.6, 0.4][i]));

    const det = compileJS((a: any) => a.determinant(), {
      name: "main",
      params: [{ name: "a", type: "mat2" }],
    });
    expect(det.invoke({ params: { a: [1, 0, 0, 1] } })).toBe(1);
    // mat2(a,b,c,d) is columns (a,b),(c,d); det = a*d - c*b.
    expect(det.invoke({ params: { a: [2, 0, 0, 3] } })).toBe(6);

    const tr = compileJS((a: any) => a.transpose(), {
      name: "main",
      params: [{ name: "a", type: "mat4" }],
    });
    const m = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    const expected = Array.from({ length: 16 }, (_, i) => m[(i % 4) * 4 + Math.floor(i / 4)]);
    expect(tr.invoke({ params: { a: m } })).toEqual(expected);
  });

  it("constructs matrices from columns and scalars", () => {
    const f = compileJS((c0: any, c1: any) => mat4(c0, c1, c1, c0), {
      name: "main",
      params: [
        { name: "c0", type: "vec4" },
        { name: "c1", type: "vec4" },
      ],
    });
    const r = f.invoke({ params: { c0: [1, 2, 3, 4], c1: [5, 6, 7, 8] } }) as number[];
    expect(r).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 1, 2, 3, 4]);
  });

  it("reads matrix columns", () => {
    const f = compileJS((a: any) => a.element(1), {
      name: "main",
      params: [{ name: "a", type: "mat4" }],
    });
    const m = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    expect(f.invoke({ params: { a: m } })).toEqual([5, 6, 7, 8]);
  });
});

describe("cross-backend: non-square matrix multiply", () => {
  it("multiplies a mat2x3 by a mat3x2 into a mat3", () => {
    const build = () => mat2x3(1, 2, 3, 4, 5, 6).mul(mat3x2(1, 0, 0, 1, 1, 1));
    expect(evalMatrix(build)).toEqual([1, 2, 3, 4, 5, 6, 5, 7, 9]);
  });

  it("multiplies a mat3x2 by a mat2x3 into a mat2", () => {
    const build = () => mat3x2(1, 2, 3, 4, 5, 6).mul(mat2x3(1, 0, 0, 0, 1, 1));
    expect(evalMatrix(build)).toEqual([1, 2, 8, 10]);
  });

  it("multiplies a mat2 by a mat3x2 into a mat3x2", () => {
    const build = () => mat2(2, 0, 0, 3).mul(mat3x2(1, 2, 3, 4, 5, 6));
    expect(evalMatrix(build)).toEqual([2, 6, 6, 12, 10, 18]);
  });

  it("multiplies a mat4 by a mat2x4 into a mat2x4", () => {
    const build = () => mat4(1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4).mul(mat2x4(1, 2, 3, 4, 5, 6, 7, 8));
    expect(evalMatrix(build)).toEqual([1, 4, 9, 16, 5, 12, 21, 32]);
  });

  it("multiplies a mat2x4 by a mat4x2 into a mat4", () => {
    const build = () => mat2x4(1, 2, 3, 4, 5, 6, 7, 8).mul(mat4x2(1, 0, 0, 1, 1, 1, 2, 0));
    expect(evalMatrix(build)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 6, 8, 10, 12, 2, 4, 6, 8]);
  });
});

describe("JS backend: control flow", () => {
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
    const fn = compileJS(() => tally, { name: "main", params: [] });
    expect(fn.invoke({})).toBe(4);
  });

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

  it("takes the branch Switch selects", () => {
    const classify = () =>
      Fn(() => {
        const out = float(0).toVar();
        Switch(int(1), (s) => {
          s.Case(0, () => {
            out.assign(float(10));
          });
          s.Case([1, 2], () => {
            out.assign(float(20));
          });
          s.Default(() => {
            out.assign(float(30));
          });
        });
        return out;
      })();
    const fn = compileJS(() => classify(), { name: "main", params: [] });
    expect(fn.invoke({})).toBe(20);
  });

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
        Switch(int(2), (s) => {
          s.Case(0, () => {
            out.assign(float(10));
          });
          s.Case([1, 2], () => {
            out.assign(float(20));
          });
          s.Default(() => {
            out.assign(float(30));
          });
        });
        return out;
      })();
    const fn = compileJS(() => classify(), { name: "main", params: [] });
    expect(fn.invoke({})).toBe(20);
  });
});

describe("JS backend: shader I/O", () => {
  it("reads uniforms", () => {
    let u!: any;
    const prog = Fn(() => {
      u = uniform("float");
      return u.mul(2);
    })();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    expect(fn.invoke({ uniforms: { [u.name]: 21 } })).toBe(42);
  });

  it("reads varyings and attributes", () => {
    let v!: any;
    let a!: any;
    const prog = Fn(() => {
      v = varying("vec3");
      a = attribute("float");
      return v.x.add(a);
    })();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    expect(fn.invoke({ varyings: { [v.name]: [3, 4, 5] }, attributes: { [a.name]: 1 } })).toBe(4);
  });

  it("returns outputs and fragment depth in a result object", () => {
    let u!: any;
    const prog = Fn(() => {
      u = uniform("vec4");
      const out = output("vec4");
      out.assign(u.add(vec4(1, 1, 1, 0)));
      const d = builtinFragDepth();
      d.assign(float(0.5));
      return out;
    })();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    const r = fn.invoke({ uniforms: { [u.name]: [0, 0, 0, 1] } }) as any;
    expect(r.value).toEqual([1, 1, 1, 1]);
    expect(typeof r.outputs).toBe("object");
    expect(r.fragDepth).toBe(0.5);
  });

  it("returns the bare value when nothing is written to outputs", () => {
    const prog = Fn(() => vec4(1, 2, 3, 4))();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    expect(fn.invoke({})).toEqual([1, 2, 3, 4]);
  });

  it("reads uniform arrays", () => {
    let arr!: any;
    const prog = Fn(() => {
      arr = uniformArray("vec4", 4);
      return arr.element(2).x;
    })();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    const values = [
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [7, 8, 9, 10],
      [0, 0, 0, 0],
    ];
    expect(fn.invoke({ uniforms: { [arr.name]: values } })).toBe(7);
  });

  it("runs a vertex stage, writing position and varyings", () => {
    const prog = Fn(() => {
      const v = varying("vec3");
      v.assign(vec3(1, 2, 3));
      const p = builtinPosition();
      p.assign(vec4(0, 0, 0, 1));
      return p;
    })();
    const fn = compileJS(() => prog, { name: "main", params: [], stage: "vertex" });
    const r = fn.invoke({}) as any;
    expect(r.position).toEqual([0, 0, 0, 1]);
    expect(Object.values(r.varyings as Record<string, unknown>)).toEqual([[1, 2, 3]]);
  });
});

describe("JS backend: CPU-specific behaviour", () => {
  it("discard returns null", () => {
    const prog = Fn(() => {
      If(float(1).greaterThan(0), () => Discard());
      return float(5);
    })();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    expect(fn.invoke({})).toBeNull();
  });

  it("hoisted scratch does not leak state across calls", () => {
    const prog = Fn(() => {
      const x = vec3(1, 2, 3).toVar();
      const y = vec3(10, 20, 30).toVar();
      If(float(0).greaterThan(1), () => {
        x.assign(y);
      });
      return x;
    })();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    expect(fn.invoke({})).toEqual([1, 2, 3]);
    expect(fn.invoke({})).toEqual([1, 2, 3]);
  });

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
    const hoisted = compileJS(sumTo, { name: "sum", params: [{ name: "n", type: "float" }] });
    const perCall = compileJS(sumTo, { name: "sum", params: [{ name: "n", type: "float" }], reentrant: true });
    expect(hoisted.invoke({ params: { n: 7 } })).toBe(21);
    expect(perCall.invoke({ params: { n: 7 } })).toBe(21);
    expect(perCall.invoke({ params: { n: 3 } })).toBe(3);
  });

  it("compiles derivatives to zero on request", () => {
    const prog = Fn(() => {
      const x = vec2(1, 2).toVar();
      return x.fwidth();
    })();
    const fn = compileJS(() => prog, { name: "main", params: [], derivatives: "zero" });
    expect(fn.invoke({})).toEqual([0, 0]);
  });

  it("refuses derivatives by default", () => {
    const prog = Fn(() => {
      const x = vec2(1, 2).toVar();
      return x.fwidth();
    })();
    expect(() => compileJS(() => prog, { name: "main", params: [] })).toThrow(/CPU target/);
  });

  it("samples textures with nearest-neighbour lookup", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("sampler2D");
      return tex.texture(vec2(0.5, 0.5));
    })();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    // 2x2 RGBA; uv (0.5, 0.5) -> texel (1, 1).
    const data = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4];
    expect(fn.invoke({ textures: { [tex.name]: { data, width: 2, height: 2 } } })).toEqual([4, 4, 4, 4]);
  });

  it("reads a byte texture through a float sampler as 0 to 1", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("sampler2D");
      return tex.texture(vec2(0.5, 0.5));
    })();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    // What a DataTexture holds: 8-bit channels. Both backends upload that as a
    // normalized format, so the shader reads 0..1 — and so must this.
    const data = new Uint8Array([0, 128, 255, 255]);
    expect(fn.invoke({ textures: { [tex.name]: { data, width: 1, height: 1 } } })).toEqual([0, 128 / 255, 1, 1]);
  });

  it("leaves a float texture that already holds float data alone", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("sampler2D");
      return tex.texture(vec2(0.5, 0.5));
    })();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    const data = new Float32Array([0, 0.5, 1, 1]);
    expect(fn.invoke({ textures: { [tex.name]: { data, width: 1, height: 1 } } })).toEqual([0, 0.5, 1, 1]);
    // A plain array is a plain array, whatever is in it.
    expect(fn.invoke({ textures: { [tex.name]: { data: [0, 0.5, 1, 1], width: 1, height: 1 } } })).toEqual([0, 0.5, 1, 1]);
  });

  it("keeps an integer texture's bytes as the texels they are", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("usampler2D");
      return tex.texture(ivec2(0, 0));
    })();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    // An integer sampler fetches raw texels on a GPU too — there is no
    // normalized format under it to undo.
    const data = new Uint8Array([0, 128, 255, 255]);
    expect(fn.invoke({ textures: { [tex.name]: { data, width: 1, height: 1 } } })).toEqual([0, 128, 255, 255]);
  });

  it("strides by the channels a texel holds, not by four", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("usampler2D");
      return tex.texture(ivec2(2, 0));
    })();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    // Four single-channel texels in a row: the third is 30. Read as RGBA, the
    // same array is one texel and the fetch falls off the end of it.
    const data = new Uint8Array([10, 20, 30, 40]);
    expect(fn.invoke({ textures: { [tex.name]: { data, width: 4, height: 1, channels: 1 } } }))
      // green and blue read zero, alpha one, as a sampler reports the channels
      // a single-channel texture does not store
      .toEqual([30, 0, 0, 1]);
  });

  it("blends a single-channel texture without reading its neighbours' channels", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("sampler2D");
      return tex.texture(vec2(0.5, 0.5));
    })();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    const texture = { data: [0, 100], width: 2, height: 1, channels: 1 as const };
    expect(fn.invoke({ textures: { [tex.name]: texture } })).toEqual([100, 0, 0, 1]);
    expect(fn.invoke({ textures: { [tex.name]: { ...texture, magFilter: "linear" as const } } })).toEqual([50, 0, 0, 1]);
  });

  it("normalizes a byte texture fetched with textureLoad too", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("sampler2D");
      return textureLoad(tex, ivec2(0, 0));
    })();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    const data = new Uint8Array([0, 128, 255, 255]);
    expect(fn.invoke({ textures: { [tex.name]: { data, width: 1, height: 1 } } })).toEqual([0, 128 / 255, 1, 1]);
  });

  it("blends neighbouring texels when the texture asks for linear filtering", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("sampler2D");
      return tex.texture(vec2(0.5, 0.5));
    })();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    // Two texels, 0 and 100, whose centres sit at 0.25 and 0.75. Sampling
    // halfway between them lands in the second texel outright without
    // filtering, and is half of each with it.
    const data = [0, 0, 0, 0, 100, 100, 100, 100];
    const texture = { data, width: 2, height: 1 };
    expect(fn.invoke({ textures: { [tex.name]: texture } })).toEqual([100, 100, 100, 100]);
    expect(fn.invoke({ textures: { [tex.name]: { ...texture, magFilter: "linear" as const } } })).toEqual([50, 50, 50, 50]);
  });

  it("wraps a coordinate past the edge the way the texture asks", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("sampler2D");
      return tex.texture(vec2(1.25, 0.5));
    })();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    const texture = { data: [10, 10, 10, 10, 20, 20, 20, 20], width: 2, height: 1 };
    const red = (t: CpuTextureData): number => (fn.invoke({ textures: { [tex.name]: t } }) as number[])[0];

    // A quarter past the right edge: the last texel stretched, the image
    // tiled back to the first, or tiled and flipped back to the last.
    expect(red(texture)).toBe(20);
    expect(red({ ...texture, wrapS: "repeat" as const })).toBe(10);
    expect(red({ ...texture, wrapS: "mirror" as const })).toBe(20);
  });

  it("wraps behind the left edge too", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("sampler2D");
      return tex.texture(vec2(-0.25, 0.5));
    })();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    const texture = { data: [10, 10, 10, 10, 20, 20, 20, 20], width: 2, height: 1 };
    const red = (t: CpuTextureData): number => (fn.invoke({ textures: { [tex.name]: t } }) as number[])[0];

    expect(red(texture)).toBe(10);
    expect(red({ ...texture, wrapS: "repeat" as const })).toBe(20);
    expect(red({ ...texture, wrapS: "mirror" as const })).toBe(10);
  });

  it("blends across the depth of a 3D texture", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("sampler3D");
      return tex.texture(vec3(0.5, 0.5, 0.5));
    })();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    // Two slices, 0 and 100, sampled halfway between their centres.
    const texture: CpuTextureData = {
      data: [0, 0, 0, 0, 100, 100, 100, 100],
      width: 1,
      height: 1,
      depth: 2,
      magFilter: "linear",
    };
    expect(fn.invoke({ textures: { [tex.name]: texture } })).toEqual([50, 50, 50, 50]);
  });

  it("samples the right face of a cube map by direction", () => {
    // 6 faces, one texel each, face order +X,-X,+Y,-Y,+Z,-Z: 10,20,30,40,50,60.
    const data = [10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30, 40, 40, 40, 40, 50, 50, 50, 50, 60, 60, 60, 60];
    const texture: CpuTextureData = { data, width: 1, height: 1 };

    const at = (dx: number, dy: number, dz: number): number[] => {
      let tex!: any;
      const build = () =>
        Fn(() => {
          tex = uniform("samplerCube");
          return tex.texture(vec3(dx, dy, dz));
        })();
      const fn = compileJS(build, { name: "main", params: [] });
      return fn.invoke({ textures: { [tex.name]: texture } }) as number[];
    };

    expect(at(1, 0, 0)).toEqual([10, 10, 10, 10]);
    expect(at(-1, 0, 0)).toEqual([20, 20, 20, 20]);
    expect(at(0, 1, 0)).toEqual([30, 30, 30, 30]);
    expect(at(0, -1, 0)).toEqual([40, 40, 40, 40]);
    expect(at(0, 0, 1)).toEqual([50, 50, 50, 50]);
    expect(at(0, 0, -1)).toEqual([60, 60, 60, 60]);
  });

  it("blends within a cube face but never across its edge into a neighbour", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("samplerCube");
      // Direction close to the edge of the +Z face (u near 1): should stay
      // within +Z's own 2x1 texel row, not bleed toward another face.
      return tex.texture(vec3(0.9, 0, 1));
    })();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    // +Z is face index 4: two texels side by side, 0 and 100.
    const data = new Array(6 * 2 * 1 * 4).fill(0);
    data[4 * 2 * 4 + 0] = 0;
    data[4 * 2 * 4 + 4] = 100;
    const texture: CpuTextureData = { data, width: 2, height: 1, magFilter: "linear" };
    const [r] = fn.invoke({ textures: { [tex.name]: texture } }) as number[];
    expect(r).toBeGreaterThan(0);
    expect(r).toBeLessThanOrEqual(100);
  });

  it("fetches integer textures at texel coordinates", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("isampler2D");
      return tex.texture(ivec2(1, 0));
    })();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    expect(fn.invoke({ textures: { [tex.name]: { data, width: 2, height: 2 } } })).toEqual([5, 6, 7, 8]);
  });

  it("supports multi-return functions, keeping only the last value as the result", () => {
    let side!: any;
    const prog = Fn(() => {
      side = float(1.0).toVar();
      let last = float(2.0).toVar();
      return [side, last];
    })();
    const fn = compileJS(() => prog as any, { name: "main", params: [] });
    expect(fn.invoke({})).toBe(2);
  });

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
    const fn = compileJS(() => prog, { name: "pick", params: [] });
    const r = fn.invoke({}) as any;
    // Ray hits y = 0 at t = 2, so the world point is (0, 0, 0).
    expect(r.value).toEqual([0, 0, 0]);
    expect(r.fragDepth).toBe(2);
  });

  it("reuses one compiled function across many pick calls", () => {
    const prog = Fn(() => {
      const ro = vec3(0, 1, 0).toVar();
      const rd = vec3(0, -1, 0).toVar();
      const t = ro.y.negate().div(rd.y).toVar();
      return ro.add(rd.mul(t));
    })();
    const fn = compileJS(() => prog, { name: "pick", params: [] });
    for (let i = 0; i < 100; i++) {
      expect(fn.invoke({})).toEqual([0, 0, 0]);
    }
  });
});

describe("JS backend: TSL free functions", () => {
  it("computes arithmetic free functions", () => {
    expect(evalScalar((a, b) => add(a, b), [2, 3])).toBe(5);
    expect(evalScalar((a, b) => sub(a, b), [7, 3])).toBe(4);
    expect(evalScalar((a, b) => mul(a, b), [3, 4])).toBe(12);
    expect(evalScalar((a, b) => mul(a, b, a), [3, 4])).toBe(36);
  });

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

  it("computes interpolation free functions", () => {
    expect(evalScalar((a, b) => mix(a, b, 0.5), [0, 10])).toBe(5);
    expect(evalScalar((a, b) => clamp(a, 0, 1), [2])).toBe(1);
    expect(evalScalar((a, b) => step(0.5, a), [1])).toBe(1);
    expect(evalScalar((a, b) => smoothstep(0, 1, a), [0.5])).toBeCloseTo(0.5);
  });

  it("computes vector free functions", () => {
    const fn = compileJS(
      () =>
        Fn(() => {
          const a = vec3(1, 2, 3).toVar();
          return dot(a, a).add(length(a)).toVar();
        })(),
      { name: "v2", params: [] },
    );
    expect(fn.invoke({})).toBeCloseTo(14 + Math.sqrt(14));
  });

  it("computes cross/reflect/normalize/faceForward", () => {
    const fn = compileJS(
      () =>
        Fn(() => {
          const a = vec3(1, 0, 0).toVar();
          return cross(a, vec3(0, 1, 0))
            .add(normalize(vec3(0, 0, 2)))
            .toVar();
        })(),
      { name: "v3", params: [] },
    );
    const c = fn.invoke({}) as number[];
    // cross((1,0,0),(0,1,0)) = (0,0,1), normalize(0,0,2) = (0,0,1).
    expect(c.map((x) => Math.abs(x))).toEqual([0, 0, 2]);
    const ff = compileJS(
      () =>
        Fn(() => {
          const n = vec3(0, 1, 0).toVar();
          return faceForward(n, vec3(0, 1, 0), vec3(1, 0, 0)).toVar();
        })(),
      { name: "v4", params: [] },
    );
    // faceforward flips n because dot(nref, i) > 0; sign flips leave signed zero.
    expect((ff.invoke({}) as number[]).map((x) => (x === 0 ? 0 : x))).toEqual([0, -1, 0]);
  });

  it("computes all/any reductions", () => {
    const fn = compileJS(
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
    expect(fn.invoke({})).toBe(1);
  });
});

describe("JS backend: TSL loop and return", () => {
  it("Loop(count, (i) => ...) sums 0..3", () => {
    const fn = compileJS(
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
    expect(fn.invoke({})).toBe(6);
  });

  it("Return() exits the function early", () => {
    const fn = compileJS(
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
    expect(fn.invoke({})).toBeUndefined();
  });

  it("Discard() returns null", () => {
    const fn = compileJS(
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
    expect(fn.invoke({})).toBe(null);
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

  it("computes modulo of a sum", () => {
    // (7 + 5) mod 4 is 0, not 7 + 5 - 4 * floor(7 + 5 / 4)
    expect(evalScalar((a, b, m) => a.add(b).mod(m), [7, 5, 4])).toBe(floored(7 + 5, 4));
  });

  it("computes modulo by a sum", () => {
    expect(evalScalar((a, b, c) => a.mod(b.add(c)), [17, 3, 2])).toBe(floored(17, 3 + 2));
  });

  it("computes whole-number division and modulo of a sum", () => {
    expect(evalScalar((a, b, m) => a.toInt().add(b.toInt()).div(m.toInt()).toFloat(), [7, 5, 4])).toBe(
      Math.trunc((7 + 5) / 4),
    );
    expect(evalScalar((a, b, m) => a.toInt().add(b.toInt()).mod(m.toInt()).toFloat(), [7, 5, 4])).toBe((7 + 5) % 4);
  });

  it("mixes between sums", () => {
    // mix(3, 10, 0.25) is 4.75
    approx(
      evalScalar((a, b, c, d, t) => a.add(b).mix(c.add(d), t), [1, 2, 4, 6, 0.25]),
      3 + 0.25 * (10 - 3),
    );
  });

  it("smoothsteps across a sum edge", () => {
    // smoothstep(2, 6, 4): t is 0.5, so the result is 0.5
    approx(
      evalScalar((v, e0a, e0b, e1) => v.smoothstep(e0a.add(e0b), e1), [4, 1, 1, 6]),
      0.5,
    );
  });
});

describe("JS backend: .draw() — render a whole grid in one call", () => {
  it("renders a scalar per pixel, fragCoord at pixel centers", () => {
    const build = () => Fn(() => fragCoord().x)();
    const fn = compileJS(build as any, { name: "main", params: [] });
    const out = fn.batch({}, 3, 2);
    expect(out.length).toBe(3 * 2);
    // Row-major, (y*width+x): x+0.5 regardless of row.
    expect(Array.from(out)).toEqual([0.5, 1.5, 2.5, 0.5, 1.5, 2.5]);
  });

  it("renders both fragCoord axes packed into a vec4 per pixel, with no stage or output() involved", () => {
    const build = () => Fn(() => vec4(fragCoord().x, fragCoord().y, 0, 1))();
    const fn = compileJS(build as any, { name: "main", params: [] });
    const out = fn.batch({}, 2, 2);
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

  it("reads a uniform every pixel and reflects a changed uniform on the next call", () => {
    const scale = uniform("float");
    const build = () => Fn(() => fragCoord().x.mul(scale))();
    const fn = compileJS(build as any, { name: "main", params: [] });
    expect(Array.from(fn.batch({ uniforms: { [scale.name]: 2 } }, 2, 1))).toEqual([1, 3]);
    expect(Array.from(fn.batch({ uniforms: { [scale.name]: 10 } }, 2, 1))).toEqual([5, 15]);
  });

  it("picks dimensions per call, not at compile time", () => {
    const build = () => Fn(() => fragCoord().x)();
    const fn = compileJS(build as any, { name: "main", params: [] });
    expect(Array.from(fn.batch({}, 2, 1))).toEqual([0.5, 1.5]);
    expect(Array.from(fn.batch({}, 4, 1))).toEqual([0.5, 1.5, 2.5, 3.5]);
    expect(Array.from(fn.batch({}, 1, 1))).toEqual([0.5]);
  });

  it("the same compiled function still works as a plain single-pixel call — draw() is a choice per call, not a compile mode", () => {
    const scale = uniform("float");
    const build = () => Fn(() => fragCoord().x.mul(scale))();
    const fn = compileJS(build as any, { name: "main", params: [] });
    expect(fn.invoke({ uniforms: { [scale.name]: 2 }, fragCoord: [3, 0] })).toBe(6);
    expect(Array.from(fn.batch({ uniforms: { [scale.name]: 2 } }, 2, 1))).toEqual([1, 3]);
  });

  it("matches compileWasmRoutine's draw() output for the same program", () => {
    const build = () => Fn(() => fragCoord().x.add(fragCoord().y.mul(2)))();
    const jsFn = compileJS(build as any, { name: "main", params: [] });
    const wasmFn = compileWasmRoutine(build as any, { name: "main", params: [] });
    expect(Array.from(jsFn.batch({}, 3, 3))).toEqual(Array.from(wasmFn.batch({}, 3, 3)));
  });

  it("declares its uniform inside the build function itself, not just before it", () => {
    // compileJS reads the root's result type for draw() by calling the build
    // function — it must do so exactly once. A build function that declares
    // its own uniform() (the idiom most tests in this file use, just always
    // with the uniform hoisted above the compileJS call rather than inside
    // the closure passed to it) has to see that same call reflected in the
    // compiled source; a second, throwaway invocation would declare a second,
    // differently-named uniform and leave `tex` pointing at the wrong one.
    let tex!: any;
    const build = () =>
      Fn(() => {
        tex = uniform("float");
        return fragCoord().x.mul(tex);
      })();
    const fn = compileJS(build as any, { name: "main", params: [] });
    expect(fn.invoke({ uniforms: { [tex.name]: 2 }, fragCoord: [3, 0] })).toBe(6);
    expect(Array.from(fn.batch({ uniforms: { [tex.name]: 2 } }, 2, 1))).toEqual([1, 3]);
  });

  it("compiles storage()/invocationIndex() into a per-call array-indexed program", () => {
    // The same per-invocation model WGSL's compute path gives storage(): the
    // compiled fn is called once per element, with `index` naming which one
    // and `storages` holding the whole backing arrays for it to read/write.
    let dt!: UniformNode<"float">;
    const build = () =>
      Fn(() => {
        const vel = storage("vel", "float");
        const pos = storage("pos", "float", { access: "read_write" });
        dt = uniform("float");
        const i = invocationIndex();
        pos.element(i).addAssign(vel.element(i).mul(dt));
      })();
    const fn = compileJS(build as any, { name: "step", params: [] });

    const pos = new Float32Array([0, 10, 20]);
    const vel = new Float32Array([1, 2, 3]);
    for (let i = 0; i < pos.length; i++) {
      fn.invoke({ storages: { vel, pos }, uniforms: { [dt.name]: 2 }, index: i });
    }
    expect(Array.from(pos)).toEqual([2, 14, 26]);
  });
});
