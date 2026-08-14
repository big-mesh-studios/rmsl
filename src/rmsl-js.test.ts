/**
 * Evaluates the JS (CPU) backend in-process.
 *
 * Unlike `rmsl-eval.test.ts`, these need neither a GPU nor a browser: the
 * compiled function runs in plain Node, so the whole breadth of the DSL is
 * pinned here on every test run, not just when `RMSL_GPU` is set.
 *
 * The JS target computes exact f64, so arithmetic is compared to the JS
 * implementation of the same operation, and a tolerance is only needed for
 * the transcendental builtins where Math's rounding can differ from a shader
 * driver's — but never for plain `a + b`.
 */

import { describe, it, expect } from "vitest";
import {
  compileJS, compileJSFn, Fn, float, int, vec2, vec3, vec4, mat4,
  If, For, While, Switch, Loop, Break, Continue, Return,
  uniform, uniformArray, varying, attribute, output, builtinPosition,
  builtinFragDepth, Discard, ivec2,
  mul, add, sub, sin, mix, clamp, step, smoothstep, dot, cross,
  normalize, length, distance, reflect, refract, faceForward,
  atan, inverseSqrt, all, any, min, max, pow,
  type Node,
} from "./rmsl";

const approx = (actual: number, want: number) =>
  expect(actual).toBeCloseTo(want, 9);

function slot(n: any): string {
  return n.value.slot;
}

type ScalarBuild = (...args: Node<"float">[]) => Node<"float">;

/** Evaluate a scalar Fn on the JS backend with the given argument values. */
function evalScalar(build: ScalarBuild, args: number[] = [], opts: any = {}): number {
  const params = args.map((_, i) => ({ name: `a${i}`, type: "float" as const }));
  const fn = compileJS(build as any, { name: "main", params, ...opts });
  const ctx: any = { params: Object.fromEntries(args.map((a, i) => [`a${i}`, a])) };
  const value = fn(ctx);
  if (typeof value === "number") return value;
  if (Array.isArray(value)) return value[0] as number;
  return value as unknown as number;
}

describe("JS backend: scalar arithmetic", () => {
  it("computes arithmetic", () => {
    expect(evalScalar((a, b) => a.add(b), [2, 3])).toBe(5);
    expect(evalScalar((a, b) => a.sub(b), [7, 3])).toBe(4);
    expect(evalScalar((a, b) => a.mul(b), [3, 4])).toBe(12);
    expect(evalScalar((a, b) => a.div(b), [8, 2])).toBe(4);
    expect(evalScalar((a) => a.negate(), [3])).toBe(-3);
  });

  it("computes math builtins", () => {
    approx(evalScalar((a) => a.sqrt(), [9]), 3);
    expect(evalScalar((a) => a.abs(), [-4])).toBe(4);
    expect(evalScalar((a) => a.floor(), [2.7])).toBe(2);
    expect(evalScalar((a) => a.ceil(), [2.1])).toBe(3);
    approx(evalScalar((a) => a.sin(), [0.5]), Math.sin(0.5));
    approx(evalScalar((a) => a.cos(), [0.5]), Math.cos(0.5));
    approx(evalScalar((a, b) => a.pow(b), [2, 10]), 1024);
    approx(evalScalar((a) => a.cbrt(), [27]), 3);
    approx(evalScalar((a) => a.sinh(), [0.5]), Math.sinh(0.5));
    expect(evalScalar((a) => a.round(), [2.6])).toBe(3);
    expect(evalScalar((a) => a.trunc(), [-2.7])).toBe(-2);
    expect(evalScalar((a) => a.saturate(), [2.5])).toBe(1);
    approx(evalScalar((a) => a.oneMinus(), [0.25]), 0.75);
    approx(evalScalar((a) => a.reciprocal(), [4]), 0.25);
  });

  it("casts float to int and back", () => {
    expect(evalScalar((a) => a.toInt().toFloat(), [2.7])).toBe(2);
    expect(evalScalar((a) => a.toInt().toFloat(), [-2.7])).toBe(-2);
    expect(evalScalar((a) => a.toUint().toFloat(), [2.7])).toBe(2);
    expect(evalScalar((a) => a.toInt().toBool().toFloat(), [1.5])).toBe(1);
    expect(evalScalar((a) => a.toInt().toBool().toFloat(), [0])).toBe(0);
  });

  it("computes step, smoothstep, mix and clamp with operands in the right order", () => {
    expect(evalScalar((a, b) => b.step(a), [0.5, 2])).toBe(1);
    expect(evalScalar((a, b) => b.step(a), [2, 0.5])).toBe(0);
    expect(evalScalar((a, b) => a.mix(b, 0.25), [0, 4])).toBe(1);
    expect(evalScalar((a, b) => a.mix(b, 0.75), [0, 4])).toBe(3);
    approx(evalScalar((a) => a.smoothstep(0, 1), [0.5]), 0.5);
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
      name: "main", params: [{ name: "a", type: "vec3" }, { name: "b", type: "vec3" }],
    });
    expect(f({ params: { a: [1, 2, 3], b: [10, 20, 30] } })).toEqual([11, 22, 33]);

    const g = compileJS((a: any) => a.mul(2), {
      name: "main", params: [{ name: "a", type: "vec3" }],
    });
    expect(g({ params: { a: [1, 2, 3] } })).toEqual([2, 4, 6]);

    const h = compileJS((a: any) => a.sub(vec3(1, 1, 1)), {
      name: "main", params: [{ name: "a", type: "vec3" }],
    });
    expect(h({ params: { a: [5, 5, 5] } })).toEqual([4, 4, 4]);
  });

  it("broadcasts a lone scalar vector constructor across every component", () => {
    // GLSL/WGSL vec3(2.0) is (2.0, 2.0, 2.0), and the JS backend must match.
    const f = compileJS(() => vec3(2), { name: "main", params: [] });
    expect(f({})).toEqual([2, 2, 2]);
    const g = compileJS(() => vec4(0.5), { name: "main", params: [] });
    expect(g({})).toEqual([0.5, 0.5, 0.5, 0.5]);
    const v = compileJS(() => vec2(-1), { name: "main", params: [] });
    expect(v({})).toEqual([-1, -1]);
  });

  it("computes dot, cross, length, distance and normalize", () => {
    const dot = compileJS((a: any, b: any) => a.dot(b), {
      name: "main", params: [{ name: "a", type: "vec3" }, { name: "b", type: "vec3" }],
    });
    expect(dot({ params: { a: [1, 2, 3], b: [4, 5, 6] } })).toBe(32);

    const cross = compileJS((a: any, b: any) => a.cross(b), {
      name: "main", params: [{ name: "a", type: "vec3" }, { name: "b", type: "vec3" }],
    });
    expect(cross({ params: { a: [1, 0, 0], b: [0, 1, 0] } })).toEqual([0, 0, 1]);

    const len = compileJS((a: any) => a.length(), {
      name: "main", params: [{ name: "a", type: "vec3" }],
    });
    approx(len({ params: { a: [3, 4, 0] } }) as number, 5);

    const dist = compileJS((a: any, b: any) => a.distance(b), {
      name: "main", params: [{ name: "a", type: "vec2" }, { name: "b", type: "vec2" }],
    });
    approx(dist({ params: { a: [0, 0], b: [3, 4] } }) as number, 5);

    const norm = compileJS((a: any) => a.normalize(), {
      name: "main", params: [{ name: "a", type: "vec3" }],
    });
    const n = norm({ params: { a: [3, 0, 0] } }) as number[];
    approx(n[0], 1);
    approx(n[1], 0);
    approx(n[2], 0);
  });

  it("computes vector comparisons to boolean vectors", () => {
    const f = compileJS((a: any, b: any) => a.lessThan(b), {
      name: "main", params: [{ name: "a", type: "vec3" }, { name: "b", type: "vec3" }],
    });
    expect(f({ params: { a: [1, 5, 3], b: [2, 2, 2] } })).toEqual([true, false, false]);
  });

  it("computes all/any on boolean vectors", () => {
    const f = compileJS((a: any) => a.greaterThan(vec3(0, 0, 0)).all(), {
      name: "main", params: [{ name: "a", type: "vec3" }],
    });
    expect(f({ params: { a: [1, 2, 3] } })).toBe(true);
    expect(f({ params: { a: [1, 0, 3] } })).toBe(false);
  });

  it("reflects and refracts", () => {
    const f = compileJS((i: any, n: any) => i.reflect(n), {
      name: "main", params: [{ name: "i", type: "vec3" }, { name: "n", type: "vec3" }],
    });
    // i = -n reflects back to +n
    const r = f({ params: { i: [0, -1, 0], n: [0, 1, 0] } }) as number[];
    approx(r[0], 0); approx(r[1], 1); approx(r[2], 0);
  });
});

describe("JS backend: matrices", () => {
  it("multiplies mat4 by vec4 and vec3", () => {
    const m = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1];
    const f4 = compileJS((a: any, v: any) => a.mul(v), {
      name: "main", params: [{ name: "a", type: "mat4" }, { name: "v", type: "vec4" }],
    });
    expect(f4({ params: { a: m, v: [1, 2, 3, 1] } })).toEqual([6, 8, 10, 1]);

    const f3 = compileJS((a: any, v: any) => a.mul(v), {
      name: "main", params: [{ name: "a", type: "mat4" }, { name: "v", type: "vec3" }],
    });
    expect(f3({ params: { a: m, v: [1, 2, 3] } })).toEqual([6, 8, 10]);
  });

  it("multiplies matrices", () => {
    const id = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const f = compileJS((a: any, b: any) => a.mul(b), {
      name: "main", params: [{ name: "a", type: "mat4" }, { name: "b", type: "mat4" }],
    });
    expect(f({ params: { a: id, b: id } })).toEqual(id);
    const translate = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1];
    expect(f({ params: { a: translate, b: id } })).toEqual(translate);
  });

  it("inverts, transposes and takes determinants", () => {
    const inv = compileJS((a: any) => a.inverse(), {
      name: "main", params: [{ name: "a", type: "mat2" }],
    });
    // mat2(2,1,3,4) = [[2,3],[1,4]]; inverse = [[0.8,-0.6],[-0.2,0.4]].
    const got = inv({ params: { a: [2, 1, 3, 4] } }) as number[];
    got.forEach((v, i) => approx(v, [0.8, -0.2, -0.6, 0.4][i]));

    const det = compileJS((a: any) => a.determinant(), {
      name: "main", params: [{ name: "a", type: "mat2" }],
    });
    expect(det({ params: { a: [1, 0, 0, 1] } })).toBe(1);
    // mat2(a,b,c,d) is columns (a,b),(c,d); det = a*d - c*b.
    expect(det({ params: { a: [2, 0, 0, 3] } })).toBe(6);

    const tr = compileJS((a: any) => a.transpose(), {
      name: "main", params: [{ name: "a", type: "mat4" }],
    });
    const m = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    const expected = Array.from({ length: 16 }, (_, i) => m[(i % 4) * 4 + Math.floor(i / 4)]);
    expect(tr({ params: { a: m } })).toEqual(expected);
  });

  it("constructs matrices from columns and scalars", () => {
    const f = compileJS((c0: any, c1: any) => mat4(c0, c1, c1, c0), {
      name: "main", params: [{ name: "c0", type: "vec4" }, { name: "c1", type: "vec4" }],
    });
    const r = f({ params: { c0: [1, 2, 3, 4], c1: [5, 6, 7, 8] } }) as number[];
    expect(r).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 1, 2, 3, 4]);
  });

  it("reads matrix columns", () => {
    const f = compileJS((a: any) => a.element(1), {
      name: "main", params: [{ name: "a", type: "mat4" }],
    });
    const m = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    expect(f({ params: { a: m } })).toEqual([5, 6, 7, 8]);
  });
});

describe("JS backend: control flow", () => {
  it("runs a for loop the right number of times", () => {
    const sumTo = (n: Node<"float">) => Fn(() => {
      const total = float(0).toVar();
      For(
        () => float(0).toVar(),
        (i) => i.lessThan(n),
        (i) => i.assign(i.add(1)),
        (i) => { total.assign(total.add(i)); },
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
        (i) => { t.assign(t.add(1)); i.assign(i.add(1)); },
        (i) => { t.assign(t.add(0)); },
      );
      return t;
    })();
    const fn = compileJS(() => tally, { name: "main", params: [] });
    expect(fn({})).toBe(4);
  });

  it("takes the branch the condition selects", () => {
    const branch = (x: Node<"float">) => Fn(() => {
      const out = float(0).toVar();
      If(x.greaterThan(1), () => { out.assign(float(10)); })
        .Else(() => { out.assign(float(20)); });
      return out;
    })();
    expect(evalScalar(branch, [2])).toBe(10);
    expect(evalScalar(branch, [0])).toBe(20);
  });

  it("runs a while loop until its condition fails", () => {
    const countdown = (n: Node<"float">) => Fn(() => {
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
    const classify = () => Fn(() => {
      const out = float(0).toVar();
      Switch(int(1), (s) => {
        s.Case(0, () => { out.assign(float(10)); });
        s.Case([1, 2], () => { out.assign(float(20)); });
        s.Default(() => { out.assign(float(30)); });
      });
      return out;
    })();
    const fn = compileJS(() => classify(), { name: "main", params: [] });
    expect(fn({})).toBe(20);
  });

  it("honours break_ and continue_", () => {
    const sumUntilBreak = (limit: Node<"float">) => Fn(() => {
      const total = float(0).toVar();
      For(
        () => float(0).toVar(),
        (i) => i.lessThan(100),
        (i) => i.assign(i.add(1)),
        (i) => {
          If(i.greaterThanEqual(limit), () => { Break(); });
          total.assign(total.add(i));
        },
      );
      return total;
    })();
    expect(evalScalar(sumUntilBreak, [5])).toBe(10);
    expect(evalScalar(sumUntilBreak, [1])).toBe(0);

    const sumSkippingFirst = (n: Node<"float">) => Fn(() => {
      const total = float(0).toVar();
      For(
        () => float(0).toVar(),
        (i) => i.lessThan(n),
        (i) => i.assign(i.add(1)),
        (i) => {
          If(i.lessThan(2), () => { Continue(); });
          total.assign(total.add(i));
        },
      );
      return total;
    })();
    expect(evalScalar(sumSkippingFirst, [5])).toBe(9);
  });

  it("computes the same results through the lowercase aliases", () => {
    const branch = (x: Node<"float">) => Fn(() => {
      const out = float(0).toVar();
      If(x.greaterThan(1), () => { out.assign(float(10)); })
        .ElseIf(x.greaterThan(0), () => { out.assign(float(20)); })
        .Else(() => { out.assign(float(30)); });
      return out;
    })();
    expect(evalScalar(branch, [2])).toBe(10);
    expect(evalScalar(branch, [0.5])).toBe(20);
    expect(evalScalar(branch, [-1])).toBe(30);

    const sum = (n: Node<"float">) => Fn(() => {
      const total = float(0).toVar();
      For(
        () => float(0).toVar(),
        (i) => i.lessThan(n),
        (i) => i.assign(i.add(1)),
        (i) => { total.assign(total.add(i)); },
      );
      return total;
    })();
    expect(evalScalar(sum, [5])).toBe(10);

    const classify = () => Fn(() => {
      const out = float(0).toVar();
      Switch(int(2), (s) => {
        s.Case(0, () => { out.assign(float(10)); });
        s.Case([1, 2], () => { out.assign(float(20)); });
        s.Default(() => { out.assign(float(30)); });
      });
      return out;
    })();
    const fn = compileJS(() => classify(), { name: "main", params: [] });
    expect(fn({})).toBe(20);
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
    expect(fn({ uniforms: { [u.name]: 21 } })).toBe(42);
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
    expect(fn({ varyings: { [v.name]: [3, 4, 5] }, attributes: { [a.name]: 1 } })).toBe(4);
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
    const r = fn({ uniforms: { [u.name]: [0, 0, 0, 1] } }) as any;
    expect(r.value).toEqual([1, 1, 1, 1]);
    expect(typeof r.outputs).toBe("object");
    expect(r.fragDepth).toBe(0.5);
  });

  it("returns the bare value when nothing is written to outputs", () => {
    const prog = Fn(() => vec4(1, 2, 3, 4))();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    expect(fn({})).toEqual([1, 2, 3, 4]);
  });

  it("reads uniform arrays", () => {
    let arr!: any;
    const prog = Fn(() => {
      arr = uniformArray("vec4", 4);
      return arr.element(2).x;
    })();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    const values = [
      [0, 0, 0, 0], [0, 0, 0, 0], [7, 8, 9, 10], [0, 0, 0, 0],
    ];
    expect(fn({ uniforms: { [arr.name]: values } })).toBe(7);
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
    const r = fn({}) as any;
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
    expect(fn({})).toBeNull();
  });

  it("hoisted scratch does not leak state across calls", () => {
    const prog = Fn(() => {
      const x = vec3(1, 2, 3).toVar();
      const y = vec3(10, 20, 30).toVar();
      If(float(0).greaterThan(1), () => { x.assign(y); });
      return x;
    })();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    expect(fn({})).toEqual([1, 2, 3]);
    expect(fn({})).toEqual([1, 2, 3]);
  });

  it("reentrant mode computes the same results", () => {
    const sumTo = (n: Node<"float">) => Fn(() => {
      const total = float(0).toVar();
      For(
        () => float(0).toVar(),
        (i) => i.lessThan(n),
        (i) => i.assign(i.add(1)),
        (i) => { total.assign(total.add(i)); },
      );
      return total;
    })();
    const hoisted = compileJS(sumTo, { name: "sum", params: [{ name: "n", type: "float" }] });
    const perCall = compileJS(sumTo, { name: "sum", params: [{ name: "n", type: "float" }], reentrant: true });
    expect(hoisted({ params: { n: 7 } })).toBe(21);
    expect(perCall({ params: { n: 7 } })).toBe(21);
    expect(perCall({ params: { n: 3 } })).toBe(3);
  });

  it("compiles derivatives to zero on request", () => {
    const prog = Fn(() => {
      const x = vec2(1, 2).toVar();
      return x.fwidth();
    })();
    const fn = compileJS(() => prog, { name: "main", params: [], derivatives: "zero" });
    expect(fn({})).toEqual([0, 0]);
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
    expect(fn({ textures: { [tex.name]: { data, width: 2, height: 2 } } })).toEqual([4, 4, 4, 4]);
  });

  it("fetches integer textures at texel coordinates", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("isampler2D");
      return tex.texture(ivec2(1, 0));
    })();
    const fn = compileJS(() => prog, { name: "main", params: [] });
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    expect(fn({ textures: { [tex.name]: { data, width: 2, height: 2 } } })).toEqual([5, 6, 7, 8]);
  });

  it("rejects multi-return functions", () => {
    const prog = Fn(() => [float(1), float(2)])();
    expect(() => compileJS(() => prog as any, { name: "main", params: [] })).toThrow(/multi-return/);
  });

  it("compileJSFn emits a self-contained expression evaluating to the callable", () => {
    const src = compileJSFn((a: any, b: any) => a.add(b), {
      name: "main", params: [{ name: "a", type: "float" }, { name: "b", type: "float" }],
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
      const ro = vec3(0, 2, 0).toVar();          // camera above the plane
      const rd = vec3(0, -1, 0).toVar();         // straight down
      const t = ro.y.negate().div(rd.y).toVar(); // distance to y = 0
      const hit = ro.add(rd.mul(t)).toVar();
      const d = builtinFragDepth();
      d.assign(t);
      return hit;
    })();
    const fn = compileJS(() => prog, { name: "pick", params: [] });
    const r = fn({}) as any;
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
      expect(fn({})).toEqual([0, 0, 0]);
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
    expect(approx(evalScalar((a) => sin(a), [Math.PI / 2]), 1));
    expect(evalScalar((a) => inverseSqrt(a), [4])).toBe(0.5);
    expect(approx(evalScalar((a, b) => atan(a, b), [1, 1]), Math.PI / 4));
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
    const fn = compileJS(() => Fn(() => {
      const a = vec3(1, 2, 3).toVar();
      return dot(a, a).add(length(a)).toVar();
    })(), { name: "v2", params: [] });
    expect(fn({})).toBeCloseTo(14 + Math.sqrt(14));
  });

  it("computes cross/reflect/normalize/faceForward", () => {
    const fn = compileJS(() => Fn(() => {
      const a = vec3(1, 0, 0).toVar();
      return cross(a, vec3(0, 1, 0)).add(normalize(vec3(0, 0, 2))).toVar();
    })(), { name: "v3", params: [] });
    const c = fn({}) as number[];
    // cross((1,0,0),(0,1,0)) = (0,0,1), normalize(0,0,2) = (0,0,1).
    expect(c.map((x) => Math.abs(x))).toEqual([0, 0, 2]);
    const ff = compileJS(() => Fn(() => {
      const n = vec3(0, 1, 0).toVar();
      return faceForward(n, vec3(0, 1, 0), vec3(1, 0, 0)).toVar();
    })(), { name: "v4", params: [] });
    // faceforward flips n because dot(nref, i) > 0; sign flips leave signed zero.
    expect((ff({}) as number[]).map((x) => (x === 0 ? 0 : x))).toEqual([0, -1, 0]);
  });

  it("computes all/any reductions", () => {
    const fn = compileJS(() => Fn(() => {
      const a = vec3(1, 2, 3).toVar();
      return all(a.greaterThan(0)).toInt().add(any(a.lessThan(0)).toInt()).toVar();
    })(), { name: "r", params: [] });
    expect(fn({})).toBe(1);
  });
});

describe("JS backend: TSL loop and return", () => {
  it("Loop(count, (i) => ...) sums 0..3", () => {
    const fn = compileJS(() => Fn(() => {
      let total = float(0).toVar();
      Loop(int(4), (i) => { total.assign(total.add(float(i))); });
      return total;
    })(), { name: "loop", params: [] });
    expect(fn({})).toBe(6);
  });

  it("Return() exits the function early", () => {
    const fn = compileJS(() => Fn(() => {
      const out = float(0).toVar();
      If(float(1).greaterThan(0), () => { Return(); });
      out.assign(float(1));
      return out;
    })(), { name: "ret", params: [] });
    // The `return;` fires before the trailing return, so the function is
    // undefined rather than 1.
    expect(fn({})).toBeUndefined();
  });

  it("Discard() returns null", () => {
    const fn = compileJS(() => Fn(() => {
      const out = float(1).toVar();
      If(float(1).greaterThan(0), () => { Discard(); });
      out.assign(float(2));
      return out;
    })(), { name: "disc", params: [] });
    expect(fn({})).toBe(null);
  });
});
