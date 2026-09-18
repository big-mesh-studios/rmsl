import { describe, it, expect } from "vitest";
import { compileWasm, compileWasmFn, instantiateWasm } from "../../wasm";
import { compileJS } from "../../js";
import {
  Fn,
  If,
  For,
  While,
  Loop,
  Break,
  Continue,
  Return,
  Discard,
  float,
  int,
  uint,
  bool,
  uniform,
  uniformArray,
  storage,
  invocationIndex,
  type UniformNode,
  vec2,
  vec3,
  vec4,
  ivec3,
  uvec3,
  bvec3,
  mat2,
  mat2x3,
  mat3x2,
  mat3,
  mat4,
  sin,
  clamp,
  cross,
  length,
  normalize,
  distance,
  reflect,
  dFdx,
  dFdy,
  fwidth,
  attribute,
  varying,
  fragCoord,
  output,
  builtinPosition,
  builtinFragDepth,
  textureSize,
  textureLoad,
  ivec2,
  type Node,
  type ShaderType,
} from "../../rmsl";
import type { CompileWasmFnOptions } from "./wasm";

function run(
  build: (...args: any[]) => Node<ShaderType>,
  args: number[] = [],
  types: ShaderType[] = [],
): number | boolean {
  const params = args.map((_, i) => ({ name: `a${i}`, type: types[i] ?? ("float" as const) }));
  const fn = compileWasm(build, { name: "main", params });
  return fn.invoke({ params: Object.fromEntries(args.map((a, i) => [`a${i}`, a])) }) as number | boolean;
}

describe("WASM backend: scalar arithmetic", () => {
  it("computes arithmetic on function params", () => {
    expect(run((a, b) => a.add(b), [2, 3])).toBe(5);
    expect(run((a, b) => a.sub(b), [7, 3])).toBe(4);
    expect(run((a, b) => a.mul(b), [3, 4])).toBe(12);
    expect(run((a) => a.sqrt(), [16])).toBe(4);
  });

  it("reads a float uniform", () => {
    let u!: any;
    const build = () => {
      u = uniform("float");
      return u.mul(2);
    };
    const fn = compileWasm(build, { name: "main", params: [] });
    expect(fn.invoke({ uniforms: { [u.name]: 21 } })).toBe(42);
  });

  it("supports a multi-return function, keeping only the last value as the result", () => {
    let side!: any;
    const build = () => {
      side = float(1.0).toVar();
      let last = float(2.0).toVar();
      return [side, last];
    };
    const fn = compileWasm(Fn(build) as any, { name: "main", params: [] });
    expect(fn.invoke({})).toBe(2);
  });

  it("supports a plain non-scalar result, through the same memory-based path a stage program uses", () => {
    // A plain WASM function can only ever return one scalar, so an
    // aggregate root goes through `needsResult` mode automatically —
    // this is what lets a per-pixel `vec4` color work with `.draw()` (see
    // that describe block below) with no stage/output() involved at all.
    const fn = compileWasm(() => vec3(1, 2, 3) as any, { name: "main", params: [] });
    const result = fn.invoke({}) as any;
    expect(result.value).toEqual([1, 2, 3]);
  });

  it("reads a float uniform array element by a constant index", () => {
    let arr!: any;
    const build = () => {
      arr = uniformArray("float", 4);
      return arr.element(int(1));
    };
    const fn = compileWasm(build, { name: "main", params: [] });
    expect(fn.invoke({ uniforms: { [arr.name]: [10, 20, 30, 40] } })).toBe(20);
  });

  it("reads a bool uniform array element by a constant index", () => {
    let arr!: any;
    const build = () => {
      arr = uniformArray("bool", 4);
      return arr.element(int(1));
    };
    const fn = compileWasm(build, { name: "main", params: [] });
    expect(fn.invoke({ uniforms: { [arr.name]: [true, false, true, false] } })).toBe(false);
  });
});

describe("WASM backend: uniform arrays", () => {
  it("reads a vec4 uniform array element by a constant index", () => {
    let arr!: any;
    const build = () => {
      arr = uniformArray("vec4", 4);
      return arr.element(int(2)).x;
    };
    const fn = compileWasm(build, { name: "main", params: [] });
    const values = [
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [7, 8, 9, 10],
      [0, 0, 0, 0],
    ];
    expect(fn.invoke({ uniforms: { [arr.name]: values } })).toBe(7);
  });

  it("lets a uniform array element back a toVar()", () => {
    let arr!: any;
    const build = () => {
      arr = uniformArray("vec3", 3);
      return Fn(() => arr.element(int(1)).toVar().x)();
    };
    const fn = compileWasm(build, { name: "main", params: [] });
    expect(
      fn.invoke({
        uniforms: {
          [arr.name]: [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ],
        },
      }),
    ).toBe(4);
  });

  it("converts a plain-number (float) index", () => {
    let arr!: any;
    const build = () => {
      arr = uniformArray("float", 4);
      return arr.element(2.0);
    };
    const fn = compileWasm(build, { name: "main", params: [] });
    expect(fn.invoke({ uniforms: { [arr.name]: [10, 20, 30, 40] } })).toBe(30);
  });

  it("indexes a vec4 uniform array with a runtime index inside a loop", () => {
    let arr!: any;
    const build = () => {
      arr = uniformArray("vec4", 24);
      return Fn(() => {
        let total = vec4(0, 0, 0, 0).toVar();
        For(
          () => float(0).toVar(),
          (i) => i.lessThan(24),
          (i) => i.assign(i.add(1)),
          (i) => {
            total.assign(total.add(arr.element(i)));
          },
        );
        return total.x;
      })();
    };
    const fn = compileWasm(build, { name: "main", params: [] });
    const values = Array.from({ length: 24 }, (_, i) => [i, 0, 0, 0]);
    expect(fn.invoke({ uniforms: { [arr.name]: values } })).toBe(276); // sum of 0..23
  });

  it("supports a uniform array element as the function root", () => {
    let arr!: any;
    const build = () => {
      arr = uniformArray("vec3", 3);
      return arr.element(int(1));
    };
    const fn = compileWasm(build, { name: "main", params: [] });
    const result = fn.invoke({
      uniforms: {
        [arr.name]: [
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ],
      },
    }) as any;
    expect(result.value).toEqual([4, 5, 6]);
  });
});

describe("WASM backend: div, mod, min, max, sign, abs, round-trip", () => {
  it("divides and mods floats, floored", () => {
    expect(run((a, b) => a.div(b), [7, 2])).toBe(3.5);
    expect(run((a, b) => a.mod(b), [-1, 3])).toBe(2); // floored, not truncated
  });

  it("divides and mods ints, truncated toward zero", () => {
    expect(run((a, b) => a.div(b), [-7, 2], ["int", "int"])).toBe(-3);
    expect(run((a, b) => a.mod(b), [-7, 2], ["int", "int"])).toBe(-1);
  });

  it("divides uints unsigned", () => {
    // As a signed 32-bit read this bit pattern is negative; unsigned it's
    // the large positive value it's meant to be.
    expect(run((a, b) => a.div(b), [4000000000, 2], ["uint", "uint"])).toBe(2000000000);
  });

  it("takes min/max of floats and ints", () => {
    expect(run((a, b) => a.min(b), [3, 4])).toBe(3);
    expect(run((a, b) => a.max(b), [3, 4])).toBe(4);
    expect(run((a, b) => a.min(b), [-3, 4], ["int", "int"])).toBe(-3);
    expect(run((a, b) => a.max(b), [-3, 4], ["int", "int"])).toBe(4);
  });

  it("takes min/max of uints unsigned", () => {
    expect(run((a, b) => a.min(b), [4000000000, 2], ["uint", "uint"])).toBe(2);
    expect(run((a, b) => a.max(b), [4000000000, 2], ["uint", "uint"])).toBe(4000000000);
  });

  it("computes sign and abs for floats and ints", () => {
    expect(run((a) => a.sign(), [-5])).toBe(-1);
    expect(run((a) => a.sign(), [0])).toBe(0);
    expect(run((a) => a.sign(), [5])).toBe(1);
    expect(run((a) => a.abs(), [-5])).toBe(5);
    expect(run((a) => a.sign(), [-5], ["int"])).toBe(-1);
    expect(run((a) => a.abs(), [-5], ["int"])).toBe(5);
  });

  it("floors, ceils, truncs, fracts, and rounds", () => {
    expect(run((a) => a.floor(), [3.7])).toBe(3);
    expect(run((a) => a.ceil(), [3.2])).toBe(4);
    expect(run((a) => a.trunc(), [-3.7])).toBe(-3);
    expect(run((a) => a.fract(), [3.25])).toBeCloseTo(0.25, 9);
    expect(run((a) => a.round(), [2.5])).toBe(3);
    expect(run((a) => a.round(), [-2.5])).toBe(-2);
  });
});

describe("WASM backend: clamp, mix, step, smoothstep", () => {
  it("clamps floats and ints (min(max(x, lo), hi))", () => {
    expect(run((a) => a.clamp(0, 1), [0.5])).toBe(0.5);
    expect(run((a) => a.clamp(0, 1), [-0.5])).toBe(0);
    expect(run((a) => a.clamp(0, 1), [1.5])).toBe(1);
    expect(run((a) => a.clamp(0, 10), [15], ["int"])).toBe(10);
  });

  it("clamps uints unsigned", () => {
    // Reads negative under a signed comparison; unsigned it's correctly
    // above the clamp's own upper bound.
    expect(run((a) => a.clamp(0, 10), [4000000000], ["uint"])).toBe(10);
  });

  it("mixes (linear interpolation) at any blend factor, including outside 0..1", () => {
    expect(run((a, b) => a.mix(b, 0.25), [0, 4])).toBe(1);
    expect(run((a, b) => a.mix(b, 0.75), [0, 4])).toBe(3);
    expect(run((a, b) => a.mix(b, 0), [0, 4])).toBe(0);
    expect(run((a, b) => a.mix(b, 1), [0, 4])).toBe(4);
    expect(run((a, b) => a.mix(b, 2), [0, 4])).toBe(8); // extrapolates past b
  });

  it("steps: 0 below the edge, 1 at or above it", () => {
    expect(run((a) => a.step(0.5), [0.2])).toBe(0);
    expect(run((a) => a.step(0.5), [0.8])).toBe(1);
    expect(run((a) => a.step(0.5), [0.5])).toBe(1); // "x < edge", so equal is 1
  });

  it("smoothsteps: clamped, cubic-eased, matching compileJS bit for bit", () => {
    expect(run((a) => a.smoothstep(0, 1), [-0.5])).toBe(0);
    expect(run((a) => a.smoothstep(0, 1), [1.5])).toBe(1);
    expect(run((a) => a.smoothstep(0, 1), [0.5])).toBe(0.5);
    // Not a round number — checks the exact formula, not just the clamped ends.
    expect(run((a) => a.smoothstep(0, 1), [0.1])).toBeCloseTo(0.028, 9);
  });

  it("shares one local across nested smoothstep() calls without corrupting either", () => {
    // The inner smoothstep's own t-computation runs to completion (using
    // the shared local) before the outer one starts computing its own —
    // this would misbehave if the local were live across both instead.
    expect(run((a) => a.smoothstep(0, 1).smoothstep(0, 1), [0.5])).toBe(0.5);
  });

  it("clamps a vector componentwise, operands already broadcast to match by core.ts", () => {
    const build = () => Fn(() => vec3(0.5, -0.5, 1.5).clamp(vec3(0, 0, 0), vec3(1, 1, 1)))();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const result = fn.invoke({}) as any;
    expect(result.value).toEqual([0.5, 0, 1]);
  });

  it("mixes a vector with a scalar blend factor, broadcasting it to every component", () => {
    const build = () => Fn(() => vec3(0, 0, 0).mix(vec3(4, 8, 12), float(0.25)))();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    expect((fn.invoke({}) as any).value).toEqual([1, 2, 3]);
  });

  it("mixes a vector with a per-component vector blend factor", () => {
    // `.mix()`'s own TS signature only declares a scalar `t` — the runtime
    // (`core.ts`'s own doc comment on `UNIFORM_OPERAND_OPS`, and
    // `compileJS`'s `_v3mix`) supports a per-component one too, so this is
    // a real, if untyped, case worth covering.
    const build = () => Fn(() => (vec3(0, 0, 0).mix as any)(vec3(4, 8, 12), vec3(0.25, 0.5, 1)))();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    expect((fn.invoke({}) as any).value).toEqual([1, 4, 12]);
  });

  it("steps and smoothsteps a vector componentwise", () => {
    const stepBuild = () => Fn(() => vec3(0.2, 0.8, 0.5).step(vec3(0.5, 0.5, 0.5)))();
    expect((compileWasm(stepBuild as any, { name: "main", params: [] }).invoke({}) as any).value).toEqual([0, 1, 1]);

    const smoothBuild = () => Fn(() => vec3(-0.5, 0.5, 1.5).smoothstep(vec3(0, 0, 0), vec3(1, 1, 1)))();
    expect((compileWasm(smoothBuild as any, { name: "main", params: [] }).invoke({}) as any).value).toEqual([0, 0.5, 1]);
  });
});

describe("WASM backend: comparisons, logical, bitwise", () => {
  it("compares floats and ints", () => {
    expect(run((a, b) => a.lessThan(b), [1, 2])).toBe(true);
    expect(run((a, b) => a.lessThanEqual(b), [2, 2])).toBe(true);
    expect(run((a, b) => a.greaterThanEqual(b), [1, 2])).toBe(false);
    expect(run((a, b) => a.equal(b), [2, 2])).toBe(true);
    expect(run((a, b) => a.notEqual(b), [2, 2])).toBe(false);
  });

  it("compares uints unsigned", () => {
    // 4000000000 reads as negative under a signed comparison; unsigned it's
    // correctly greater than 2.
    expect(run((a, b) => a.lessThan(b), [4000000000, 2], ["uint", "uint"])).toBe(false);
    expect(run((a, b) => a.greaterThan(b), [4000000000, 2], ["uint", "uint"])).toBe(true);
  });

  it("combines booleans with and/or/not", () => {
    const build = (a: Node<"float">, b: Node<"float">) => a.greaterThan(0).and(b.greaterThan(0));
    expect(run(build, [1, 1])).toBe(true);
    expect(run(build, [1, -1])).toBe(false);
    expect(run((a: Node<"float">) => a.greaterThan(0).not(), [1])).toBe(false);
    expect(run((a: Node<"float">, b: Node<"float">) => a.greaterThan(0).or(b.greaterThan(0)), [-1, 1])).toBe(true);
  });

  it("does bitwise and shifts on ints", () => {
    expect(run((a, b) => a.bitAnd(b), [6, 3], ["int", "int"])).toBe(2);
    expect(run((a, b) => a.bitOr(b), [6, 3], ["int", "int"])).toBe(7);
    expect(run((a, b) => a.bitXor(b), [6, 3], ["int", "int"])).toBe(5);
    expect(run((a) => a.bitNot(), [6], ["int"])).toBe(-7);
    expect(run((a, b) => a.shiftLeft(b), [1, 4], ["int", "int"])).toBe(16);
  });

  it("shifts uints logically, ints arithmetically", () => {
    expect(run((a, b) => a.shiftRight(b), [-16, 2], ["int", "int"])).toBe(-4);
    expect(run((a, b) => a.shiftRight(b), [4000000000, 2], ["uint", "uint"])).toBe(1000000000);
  });
});

describe("WASM backend: transcendentals via host import", () => {
  it("calls Math functions through a WASM import", () => {
    expect(run((a) => a.sin(), [1.2345])).toBeCloseTo(Math.sin(1.2345), 9);
    expect(run((a) => a.cos(), [1.2345])).toBeCloseTo(Math.cos(1.2345), 9);
    expect(run((a) => a.exp(), [2])).toBeCloseTo(Math.exp(2), 9);
    expect(run((a) => a.log(), [2])).toBeCloseTo(Math.log(2), 9);
    expect(run((a, b) => a.pow(b), [2, 10])).toBe(1024);
    expect(run((a) => a.exp2(), [10])).toBe(1024);
  });

  it("agrees with compileJS on a transcendental", () => {
    // Built once: compileWasm and compileJS each call their `fn` argument
    // themselves, and a `build` that calls uniform() itself would mint a
    // fresh, differently-named uniform for each backend instead of sharing
    // one, so the graph is built up front and handed to both as `() => node`.
    const u = uniform("float");
    const node = sin(u).mul(u.cos());
    const wasmFn = compileWasm(() => node, { name: "main", params: [] });
    const jsFn = compileJS(() => node as any, { name: "main", params: [] });
    const ctx = { uniforms: { [u.name]: 0.6 } };
    expect(wasmFn.invoke(ctx)).toBeCloseTo(jsFn.invoke(ctx) as number, 9);
  });
});

describe("WASM backend: int/uint/bool values and casts", () => {
  it("returns int/uint/bool results, and reads int/uint/bool uniforms", () => {
    let iu!: any;
    const intResult = compileWasm(
      () => {
        iu = uniform("int");
        return iu.add(int(1));
      },
      { name: "main", params: [] },
    );
    expect(intResult.invoke({ uniforms: { [iu.name]: 5 } })).toBe(6);

    let uu!: any;
    const uintResult = compileWasm(
      () => {
        uu = uniform("uint");
        return uu.add(uint(1));
      },
      { name: "main", params: [] },
    );
    expect(uintResult.invoke({ uniforms: { [uu.name]: 5 } })).toBe(6);

    let bu!: any;
    const boolResult = compileWasm(
      () => {
        bu = uniform("bool");
        return bu.not();
      },
      { name: "main", params: [] },
    );
    expect(boolResult.invoke({ uniforms: { [bu.name]: true } })).toBe(false);
  });

  it("casts between float, int, uint, and bool", () => {
    expect(run((a) => a.toInt(), [3.9])).toBe(3); // truncates
    expect(run((a) => a.toFloat(), [3], ["int"])).toBe(3);
    expect(run((a) => a.toBool(), [0], ["float"])).toBe(false);
    expect(run((a) => a.toBool(), [5], ["float"])).toBe(true);
    expect(run((a) => a.toBool(), [0], ["int"])).toBe(false);
    expect(run((a) => a.toInt(), [1], ["bool"])).toBe(1);
  });

  it("agrees with compileJS across an int/uint/bool expression", () => {
    const build = (x: Node<"float">) => {
      const asInt = x.toInt();
      const doubled = asInt.mul(int(2));
      return doubled.greaterThan(int(4)).and(bool(true));
    };
    for (const x of [1, 3, -2]) {
      const wasmFn = compileWasm(build as any, { name: "main", params: [{ name: "x", type: "float" }] });
      const jsFn = compileJS(build as any, { name: "main", params: [{ name: "x", type: "float" }] });
      expect(wasmFn.invoke({ params: { x } })).toBe(jsFn.invoke({ params: { x } }));
    }
  });
});

describe("WASM backend: control flow", () => {
  it("takes the branch If/Else selects", () => {
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
    expect(run(branch, [2])).toBe(10);
    expect(run(branch, [0])).toBe(20);
  });
});

/**
 * Phase 4: control flow parity (see ROADMAP.md). `Loop`/`Switch` desugar to
 * `For`/`If` chains at build time (core.ts) and never reach the WASM
 * backend as their own node types, so only `for`/`while`/`break`/
 * `continue`/`return`/`discard` are new here.
 */
describe("WASM backend: loops", () => {
  it("sums a range with For", () => {
    const build = (n: Node<"float">) =>
      Fn(() => {
        const sum = float(0).toVar();
        For(
          () => int(0).toVar(),
          (i) => i.lessThan(n.toInt()),
          (i) => {
            i.assign(i.add(int(1)));
          },
          (i) => {
            sum.assign(sum.add(i.toFloat()));
          },
        );
        return sum;
      })();
    expect(run(build, [5])).toBe(10); // 0+1+2+3+4
  });

  it("sums the same range via the Loop sugar", () => {
    const build = (n: Node<"float">) =>
      Fn(() => {
        const sum = float(0).toVar();
        Loop(n.toInt(), (i) => {
          sum.assign(sum.add(i.toFloat()));
        });
        return sum;
      })();
    expect(run(build, [5])).toBe(10);
  });

  it("sums the same range with While", () => {
    const build = (n: Node<"float">) =>
      Fn(() => {
        const i = int(0).toVar();
        const sum = float(0).toVar();
        While(i.lessThan(n.toInt()), () => {
          sum.assign(sum.add(i.toFloat()));
          i.assign(i.add(int(1)));
        });
        return sum;
      })();
    expect(run(build, [5])).toBe(10);
  });

  it("breaks a loop directly, and from inside a nested If", () => {
    const directBreak = (n: Node<"float">) =>
      Fn(() => {
        const sum = float(0).toVar();
        For(
          () => int(0).toVar(),
          (i) => i.lessThan(n.toInt()),
          (i) => {
            i.assign(i.add(int(1)));
          },
          (i) => {
            Break();
            sum.assign(sum.add(i.toFloat()));
          },
        );
        return sum;
      })();
    expect(run(directBreak, [5])).toBe(0); // breaks before ever adding

    const ifBreak = (n: Node<"float">) =>
      Fn(() => {
        const sum = float(0).toVar();
        For(
          () => int(0).toVar(),
          (i) => i.lessThan(n.toInt()),
          (i) => {
            i.assign(i.add(int(1)));
          },
          (i) => {
            If(i.equal(int(3)), () => {
              Break();
            });
            sum.assign(sum.add(i.toFloat()));
          },
        );
        return sum;
      })();
    expect(run(ifBreak, [10])).toBe(3); // 0+1+2, stops before adding 3
  });

  it("continues a For loop, still running the update clause", () => {
    const build = (n: Node<"float">) =>
      Fn(() => {
        const sum = float(0).toVar();
        For(
          () => int(0).toVar(),
          (i) => i.lessThan(n.toInt()),
          (i) => {
            i.assign(i.add(int(1)));
          },
          (i) => {
            If(i.equal(int(2)), () => {
              Continue();
            });
            sum.assign(sum.add(i.toFloat()));
          },
        );
        return sum;
      })();
    // 0+1+3+4, skipping 2. If Continue skipped the update clause instead of
    // running it, `i` would never advance past 2 and this would hang.
    expect(run(build, [5])).toBe(8);
  });

  it("breaks only the innermost loop when loops are nested", () => {
    const build = (n: Node<"float">) =>
      Fn(() => {
        const count = float(0).toVar();
        For(
          () => int(0).toVar(),
          (i) => i.lessThan(n.toInt()),
          (i) => {
            i.assign(i.add(int(1)));
          },
          () => {
            For(
              () => int(0).toVar(),
              (j) => j.lessThan(int(10)),
              (j) => {
                j.assign(j.add(int(1)));
              },
              (j) => {
                If(j.equal(int(2)), () => {
                  Break();
                });
                count.assign(count.add(float(1)));
              },
            );
          },
        );
        return count;
      })();
    // The inner loop always runs exactly 2 iterations (j=0,1) before
    // breaking, regardless of the outer loop, so the total is n * 2 — an
    // outer-loop-breaking bug would instead give exactly 2.
    expect(run(build, [3])).toBe(6);
  });

  it("throws when Break/Continue appear outside a loop", () => {
    expect(() =>
      compileWasm(
        () =>
          Fn(() => {
            Break();
            return float(0);
          })(),
        { name: "main", params: [] },
      ),
    ).toThrow(/"Break" outside a loop/);
    expect(() =>
      compileWasm(
        () =>
          Fn(() => {
            Continue();
            return float(0);
          })(),
        { name: "main", params: [] },
      ),
    ).toThrow(/"Continue" outside a loop/);
  });
});

describe("WASM backend: Return and Discard", () => {
  it("returns early with a zero sentinel from inside an If", () => {
    const build = (x: Node<"float">) =>
      Fn(() => {
        const v = float(5).toVar();
        If(x.greaterThan(0), () => {
          Return();
        });
        v.assign(float(99));
        return v;
      })();
    expect(run(build, [1])).toBe(0); // Return fires: sentinel, v never reassigned
    expect(run(build, [-1])).toBe(99); // Return doesn't fire: normal path runs
  });

  it("compiles Discard to the same early-exit sentinel", () => {
    const build = () =>
      Fn(() => {
        Discard();
        return float(42);
      })();
    expect(run(build)).toBe(0);
  });
});

describe("WASM backend: vec3 dot", () => {
  it("computes a dot product through a vec3 uniform", () => {
    let dir!: any;
    let target!: any;
    const build = () => {
      dir = uniform("vec3");
      target = uniform("vec3");
      return dir.dot(target);
    };
    const fn = compileWasm(build, { name: "main", params: [] });
    expect(fn.invoke({ uniforms: { [dir.name]: [1, 2, 3], [target.name]: [4, 5, 6] } })).toBe(32);
  });

  it("drives a picking-style hit test (dot + If/Else)", () => {
    let dir!: any;
    let target!: any;
    let threshold!: any;
    const build = () =>
      Fn(() => {
        dir = uniform("vec3");
        target = uniform("vec3");
        threshold = uniform("float");
        const hit = float(0).toVar();
        If(dir.dot(target).greaterThan(threshold), () => {
          hit.assign(float(1));
        }).Else(() => {
          hit.assign(float(0));
        });
        return hit;
      })();
    const fn = compileWasm(build, { name: "main", params: [] });
    const ctx = (d: number[], t: number[], th: number) => ({
      uniforms: { [dir.name]: d, [target.name]: t, [threshold.name]: th },
    });
    expect(fn.invoke(ctx([1, 0, 0], [1, 0, 0], 0.5))).toBe(1); // parallel: dot=1
    expect(fn.invoke(ctx([1, 0, 0], [0, 1, 0], 0.5))).toBe(0); // perpendicular: dot=0
  });
});

/**
 * Phase 3: vectors/matrices as first-class values, backed by real WASM
 * linear memory (see ROADMAP.md). Every case below verifies memory layout
 * indirectly, by dotting a constructed/stored value against itself (or a
 * probe vector) and checking a hand-computed number — the same style the
 * Phase 1/2 vec3-dot tests above already use, generalized to every width
 * and to vars/swizzles/componentwise ops that go through the new
 * materialize-into-memory mechanism instead of the old vec3-only path.
 */
describe("WASM backend: vector/matrix construct and literals", () => {
  it("dots a vec2/vec3/vec4 literal against itself", () => {
    expect(run(() => vec2(3, 4).dot(vec2(3, 4)) as any)).toBe(25);
    expect(run(() => vec3(1, 2, 3).dot(vec3(1, 2, 3)) as any)).toBe(14);
    expect(run(() => vec4(1, 2, 3, 4).dot(vec4(1, 2, 3, 4)) as any)).toBe(30);
  });

  it("dots a mixed-arity construct (vec3-from-vec2, vec4-from-vec3) against itself", () => {
    expect(run(() => (vec3 as any)(vec2(1, 2), 3).dot(vec3(1, 2, 3)))).toBe(14);
    expect(run(() => vec4(vec3(1, 2, 3), 4).dot(vec4(1, 2, 3, 4)) as any)).toBe(30);
  });

  it("dots a matrix (Frobenius inner product) built from columns or a scalar diagonal", () => {
    const m = () => mat3(vec3(1, 2, 3), vec3(4, 5, 6), vec3(7, 8, 9));
    expect(run(() => (m() as any).dot(m() as any))).toBe(285); // sum of squares 1..9
    expect(run(() => (mat2(2) as any).dot(mat2(2) as any))).toBe(8); // diag(2,2) -> 4+4
    expect(run(() => (mat4(3) as any).dot(mat4(3) as any))).toBe(36); // diag(3,3,3,3) -> 4*9
  });
});

describe("WASM backend: toVar/assign on vectors", () => {
  it("reuses a toVar'd vector across an If/Else branch", () => {
    const branch = (x: Node<"float">) =>
      Fn(() => {
        const v = vec3(0, 0, 0).toVar();
        If(x.greaterThan(0), () => {
          v.assign(vec3(1, 2, 3));
        }).Else(() => {
          v.assign(vec3(4, 5, 6));
        });
        return v.dot(v) as any;
      })();
    expect(run(branch, [1])).toBe(14); // 1+4+9
    expect(run(branch, [-1])).toBe(77); // 16+25+36
  });

  it("assigns a multi-component swizzle target", () => {
    const build = () =>
      Fn(() => {
        const v = vec3(1, 2, 3).toVar();
        v.xy.assign(vec2(10, 20));
        return v.dot(v) as any;
      })();
    expect(run(build)).toBe(509); // 100+400+9
  });

  it("assigns a single-component swizzle target", () => {
    const build = () =>
      Fn(() => {
        const v = vec3(1, 2, 3).toVar();
        v.y.assign(float(99));
        return v.dot(v) as any;
      })();
    expect(run(build)).toBe(9811); // 1+9801+9
  });
});

describe("WASM backend: swizzle read", () => {
  it("reads single and multi-component swizzles, skipping components", () => {
    expect(run(() => vec4(1, 2, 3, 4).y as any)).toBe(2);
    expect(run(() => vec4(1, 2, 3, 4).yz.dot(vec2(1, 1)) as any)).toBe(5); // 2+3, skips x and w
    expect(run(() => Fn(() => vec3(1, 2, 3).toVar().z as any)())).toBe(3);
  });

  it("inlines a nested Fn call's seq result used mid-expression (scalar)", () => {
    // Unlike the case above, the nested Fn call here isn't the compiled
    // root — it's one operand of `.add`, so its "seq" node (statements +
    // final value) has to be lowered from inside expression position.
    const inner = Fn(() => float(2).toVar().mul(3));
    expect(run(() => inner().add(1) as any)).toBe(7);
  });

  it("inlines a nested Fn call's seq result used mid-expression (vector)", () => {
    const inner = Fn(() => vec3(10, 20, 30).toVar());
    expect(run(() => vec3(1, 2, 3).add(inner()).y as any)).toBe(22);
  });

  it("reads int/uint/bool vector components", () => {
    expect(run(() => ivec3(1, -2, 3).y as any)).toBe(-2);
    expect(run(() => uvec3(1, 2, 3).z as any)).toBe(3);
    expect(run(() => (bvec3(true, false, true) as any).y)).toBe(false);
  });
});

describe("WASM backend: componentwise vector arithmetic", () => {
  it("adds/subs/muls/divs vector-vector", () => {
    expect(
      run(
        () =>
          vec3(1, 2, 3)
            .add(vec3(4, 5, 6))
            .dot(vec3(1, 2, 3).add(vec3(4, 5, 6))) as any,
      ),
    ).toBe(155);
    expect(
      run(
        () =>
          vec3(2, 4, 6)
            .sub(vec3(1, 1, 1))
            .dot(vec3(1, 3, 5)) as any,
      ),
    ).toBe(35);
    expect(run(() => vec2(3, 4).mul(vec2(2, 2)).dot(vec2(6, 8)) as any)).toBe(100);
  });

  it("broadcasts a scalar across a vector for mul/div", () => {
    expect(
      run(
        () =>
          vec3(1, 2, 3)
            .mul(2)
            .dot(vec3(2, 4, 6)) as any,
      ),
    ).toBe(56);
    expect(
      run(
        () =>
          vec3(2, 4, 6)
            .div(2)
            .dot(vec3(1, 2, 3)) as any,
      ),
    ).toBe(14);
  });

  it("adds int vectors componentwise", () => {
    expect(run(() => ivec3(1, 2, 3).add(ivec3(10, 20, 30)).y as any)).toBe(22);
  });
});

describe("WASM backend: aggregate function params", () => {
  it("reads a vec3 function param via memory, not a WASM arg", () => {
    const fn = compileWasm((v: any) => v.dot(v), { name: "main", params: [{ name: "v", type: "vec3" }] });
    expect(fn.invoke({ params: { v: [1, 2, 3] } })).toBe(14);
  });
});

describe("WASM backend: cross, length, normalize, distance, reflect", () => {
  it("computes a cross product", () => {
    expect(run(() => cross(vec3(1, 0, 0), vec3(0, 1, 0)).dot(vec3(0, 0, 1)) as any)).toBe(1);
    expect(run(() => cross(vec3(1, 0, 0), vec3(0, 1, 0)).dot(vec3(1, 0, 0)) as any)).toBe(0);
  });

  it("throws for cross() on a non-vec3", () => {
    expect(() =>
      compileWasm(() => (cross(vec2(1, 0) as any, vec2(0, 1) as any) as any).dot(vec2(0, 1) as any), {
        name: "main",
        params: [],
      }),
    ).toThrow(/cross\(\) needs a vec3/);
  });

  it("computes length and distance", () => {
    expect(run(() => length(vec3(3, 4, 0)) as any)).toBe(5);
    expect(run(() => distance(vec3(0, 0, 0), vec3(3, 4, 0)) as any)).toBe(5);
  });

  it("normalizes a vector", () => {
    expect(run(() => normalize(vec3(3, 4, 0)).dot(normalize(vec3(3, 4, 0))) as any)).toBeCloseTo(1, 9);
    expect(run(() => normalize(vec3(3, 4, 0)).x as any)).toBeCloseTo(0.6, 9);
  });

  it("leaves a zero-length vector unchanged rather than dividing by zero", () => {
    expect(run(() => normalize(vec3(0, 0, 0)).x as any)).toBe(0);
  });

  it("reflects a vector off a normal", () => {
    // reflect(I, N) = I - 2*dot(N,I)*N; I=(1,-1,0), N=(0,1,0) -> (1,1,0)
    expect(run(() => reflect(vec3(1, -1, 0), vec3(0, 1, 0)).dot(vec3(1, 1, 0)) as any)).toBe(2);
    expect(run(() => reflect(vec3(1, -1, 0), vec3(0, 1, 0)).y as any)).toBe(1);
  });
});

describe("WASM backend: matrix×vector and matrix×matrix multiplication", () => {
  it("multiplies a square matrix by a full-width vector", () => {
    const identity = mat3(vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1));
    expect(run(() => (identity as any).mul(vec3(7, 8, 9)).dot(vec3(7, 8, 9)))).toBe(194);
  });

  it("multiplies a mat4 by a vec3, implying w=1 and dropping the w row", () => {
    // Translation matrix: columns (1,0,0,0),(0,1,0,0),(0,0,1,0),(10,20,30,1).
    const translate = mat4(vec4(1, 0, 0, 0), vec4(0, 1, 0, 0), vec4(0, 0, 1, 0), vec4(10, 20, 30, 1));
    const transformed = (translate as any).mul(vec3(1, 2, 3));
    expect(run(() => transformed.dot(vec3(11, 22, 33)))).toBe(1694); // (11,22,33) . itself
  });

  it("computes a real matrix product, not a componentwise one", () => {
    // A = columns (1,2),(3,4); B = columns (5,6),(7,8) — mat2's flat literal
    // value is already column-major, so these are just the flattened columns
    // in order (mat2's own constructor has no "columns of vector nodes"
    // overload the way mat3/mat4 do, only a flat-numbers literal).
    // Real A*B = columns (23,34),(31,46). A wrongly-componentwise A*B would
    // be columns (5,12),(21,32) instead.
    const a = mat2(1, 2, 3, 4);
    const b = mat2(5, 6, 7, 8);
    const product = (a as any).mul(b);
    // Extract the first column via matVecMul with the (1,0) basis vector.
    expect(run(() => (product as any).mul(vec2(1, 0)).dot(vec2(1, 1)))).toBe(57); // 23+34
  });

  it("multiplies non-square matrices at the product shape", () => {
    const a = mat2x3(1, 2, 3, 4, 5, 6);
    const b = mat3x2(1, 0, 0, 1, 1, 1);
    const fn = compileWasm(() => (a as any).mul(b), { name: "main", params: [] });
    const result = fn.invoke({}) as { value: number[] };
    // Same product pinned for the JS backend in js.test.ts.
    expect(result.value).toEqual([1, 2, 3, 4, 5, 6, 5, 7, 9]);
  });
});

describe("WASM backend: derivatives option", () => {
  it("throws by default, matching compileJS's own message shape", () => {
    expect(() => compileWasm(() => dFdx(float(1)), { name: "main", params: [] })).toThrow(
      /dFdx\(\) has no meaning on the CPU target/,
    );
  });

  it('evaluates dFdx/dFdy/fwidth as 0 when derivatives: "zero"', () => {
    const options: CompileWasmFnOptions = { name: "main", params: [], derivatives: "zero" };
    expect(compileWasm(() => dFdx(float(3)), options).invoke({})).toBe(0);
    expect(compileWasm(() => dFdy(float(3)), options).invoke({})).toBe(0);
    expect(compileWasm(() => fwidth(float(3)), options).invoke({})).toBe(0);
  });

  it('evaluates an aggregate dFdx as a zero vector when derivatives: "zero"', () => {
    const options: CompileWasmFnOptions = { name: "main", params: [], derivatives: "zero" };
    expect(compileWasm(() => dFdx(vec3(1, 2, 3)).dot(vec3(1, 1, 1)) as any, options).invoke({})).toBe(0);
  });
});

describe("WASM backend: reentrant option accepted as a no-op", () => {
  it("compiles and runs identically whether reentrant is set or not", () => {
    const build = (a: Node<"float">) => a.mul(2);
    const params = [{ name: "a", type: "float" as const }];
    const plain = compileWasm(build as any, { name: "main", params });
    const reentrant = compileWasm(build as any, { name: "main", params, reentrant: true });
    expect(plain.invoke({ params: { a: 21 } })).toBe(42);
    expect(reentrant.invoke({ params: { a: 21 } })).toBe(42);
  });
});

describe("WASM backend: input direction (attribute/varying/fragCoord)", () => {
  it("reads scalar and aggregate attributes", () => {
    const a = attribute("float");
    const b = attribute("vec3");
    const fn = compileWasm(() => a.add(b.dot(b)) as any, { name: "main", params: [] });
    expect(fn.invoke({ attributes: { [a.name]: 10, [b.name]: [1, 2, 3] } })).toBe(24); // 10 + (1+4+9)
  });

  it("reads a varying in the default (fragment) stage", () => {
    const v = varying("vec2");
    const fn = compileWasm(() => v.x.add(v.y) as any, { name: "main", params: [] });
    expect(fn.invoke({ varyings: { [v.name]: [3, 4] } })).toBe(7);
  });

  it("reads fragCoord, defaulting to [0, 0]", () => {
    const fn = compileWasm(() => fragCoord().x.add(fragCoord().y) as any, { name: "main", params: [] });
    expect(fn.invoke({ fragCoord: [5, 6] })).toBe(11);
    expect(fn.invoke({})).toBe(0);
  });

  it("throws for fragCoord() in a vertex stage", () => {
    expect(() => compileWasm(() => fragCoord().x as any, { name: "main", params: [], stage: "vertex" })).toThrow(
      /fragCoord\(\) can only be used in fragment shaders/,
    );
  });
});

describe("WASM backend: output direction (output/varying/builtinPosition/builtinFragDepth)", () => {
  it("writes to output(), scalar and aggregate, alongside a plain value", () => {
    const build = () =>
      Fn(() => {
        const colorOut = output("vec4");
        const idOut = output("float");
        colorOut.assign(vec4(1, 0, 0, 1));
        idOut.assign(float(7));
        return float(42);
      })();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const result = fn.invoke({}) as any;
    const values = Object.values(result.outputs as Record<string, unknown>);
    expect(values).toContainEqual([1, 0, 0, 1]);
    expect(values).toContainEqual(7);
    expect(result.value).toBe(42);
  });

  it("writes int and bool scalar outputs with their own kind", () => {
    const build = () =>
      Fn(() => {
        const nOut = output("int");
        const bOut = output("bool");
        nOut.assign(int(7));
        bOut.assign(bool(true));
        return float(1);
      })();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const result = fn.invoke({}) as any;
    const values = Object.values(result.outputs as Record<string, unknown>);
    expect(values).toContainEqual(7);
    expect(values).toContainEqual(true);
    expect(result.value).toBe(1);
  });

  it("writes a scalar int varying in a vertex stage with its own kind", () => {
    const build = () =>
      Fn(() => {
        const v = varying("int");
        v.assign(int(9));
        const p = builtinPosition();
        p.assign(vec4(0, 0, 0, 1));
        return p;
      })();
    const fn = compileWasm(build as any, { name: "main", params: [], stage: "vertex" });
    const result = fn.invoke({}) as any;
    expect(Object.values(result.varyings as Record<string, unknown>)).toEqual([9]);
  });

  it("writes position and a varying in a vertex stage, matching compileJS's own test", () => {
    const build = () =>
      Fn(() => {
        const v = varying("vec3");
        v.assign(vec3(1, 2, 3));
        const p = builtinPosition();
        p.assign(vec4(0, 0, 0, 1));
        return p;
      })();
    const fn = compileWasm(build as any, { name: "main", params: [], stage: "vertex" });
    const result = fn.invoke({}) as any;
    expect(result.position).toEqual([0, 0, 0, 1]);
    expect(Object.values(result.varyings as Record<string, unknown>)).toEqual([[1, 2, 3]]);
  });

  it("treats a plain vec4 result as the implicit position when builtinPosition() is never used", () => {
    const build = () => Fn(() => vec4(5, 6, 7, 8))();
    const fn = compileWasm(build as any, { name: "main", params: [], stage: "vertex" });
    const result = fn.invoke({}) as any;
    expect(result.position).toEqual([5, 6, 7, 8]);
    expect(result.value).toBeUndefined();
  });

  it("allows a vertex stage's result to be non-vec4 once position is written", () => {
    const build = () =>
      Fn(() => {
        builtinPosition().assign(vec4(1, 2, 3, 4));
        return float(0);
      })();
    const fn = compileWasm(build as any, { name: "main", params: [], stage: "vertex" });
    const result = fn.invoke({}) as any;
    expect(result.position).toEqual([1, 2, 3, 4]);
    expect(result.value).toBe(0);
  });

  it("throws when a vertex stage's result is neither vec4 nor paired with a written position", () => {
    const build = () => Fn(() => float(0))();
    expect(() => compileWasm(build as any, { name: "main", params: [], stage: "vertex" })).toThrow(
      /vertex shader has to produce a position/,
    );
  });

  it("writes builtinFragDepth in a fragment stage", () => {
    const build = () =>
      Fn(() => {
        builtinFragDepth().assign(float(0.25));
        return float(1);
      })();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const result = fn.invoke({}) as any;
    expect(result.fragDepth).toBe(0.25);
    expect(result.value).toBe(1);
  });

  it("throws for builtinFragDepth() in a vertex stage", () => {
    const build = () =>
      Fn(() => {
        builtinFragDepth().assign(float(0));
        return float(0);
      })();
    expect(() => compileWasm(build as any, { name: "main", params: [], stage: "vertex" })).toThrow(
      /builtinFragDepth\(\) can only be used in fragment shaders/,
    );
  });

  it("throws reading builtinPosition() from a fragment stage", () => {
    const build = () =>
      Fn(() => {
        builtinPosition().assign(vec4(0, 0, 0, 1));
        return builtinPosition().x;
      })();
    expect(() => compileWasm(build as any, { name: "main", params: [] })).toThrow(/fragment stage cannot read it/);
  });
});

describe("WASM backend: texture uniforms", () => {
  it("reads a sampler2D texture's dimensions via textureSize()", () => {
    const tex = uniform("sampler2D");
    const build = () =>
      Fn(() => {
        const s = textureSize(tex).toVar();
        return s.x.add(s.y);
      })();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const result = fn.invoke({ textures: { [tex.name]: { data: new Float32Array(4 * 3), width: 4, height: 3 } } });
    expect(result).toBe(7);
  });

  it("reads a sampler3D texture's dimensions via textureSize()", () => {
    const tex = uniform("sampler3D");
    const build = () =>
      Fn(() => {
        const s = textureSize(tex).toVar();
        return s.x.add(s.y).add(s.z);
      })();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const result = fn.invoke({
      textures: { [tex.name]: { data: new Float32Array(2 * 3 * 4), width: 2, height: 3, depth: 4 } },
    });
    expect(result).toBe(9);
  });

  it("grows WASM memory to fit a larger texture across calls without corrupting metadata", () => {
    const tex = uniform("sampler2D");
    const build = () => Fn(() => textureSize(tex).x)();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    expect(fn.invoke({ textures: { [tex.name]: { data: new Float32Array(4 * 4), width: 4, height: 4 } } })).toBe(4);
    // A much larger texture forces `memory.grow` — must not corrupt the
    // compile-time-fixed metadata address or throw.
    expect(fn.invoke({ textures: { [tex.name]: { data: new Float32Array(2000 * 2000), width: 2000, height: 2000 } } })).toBe(
      2000,
    );
  });

  it("keeps sampling correctly across repeated calls with the same texture object (the cached, skip-the-copy path)", () => {
    const tex = uniform("sampler2D");
    const build = () => Fn(() => textureLoad(tex, ivec2(1, 0)).x)();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const texture = { data: [10, 20], width: 2, height: 1, channels: 1 as const };
    // Same object reference every call — after the first, the wrapper skips
    // re-copying it entirely, so this also checks that skip never leaves a
    // call reading stale or uninitialized memory.
    for (let i = 0; i < 5; i++) {
      expect(fn.invoke({ textures: { [tex.name]: texture } })).toBe(20);
    }
  });

  it("picks up a different, same-size texture bound to the same slot", () => {
    const tex = uniform("sampler2D");
    const build = () => Fn(() => textureLoad(tex, ivec2(1, 0)).x)();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const first = { data: [10, 20], width: 2, height: 1, channels: 1 as const };
    const second = { data: [10, 99], width: 2, height: 1, channels: 1 as const };
    expect(fn.invoke({ textures: { [tex.name]: first } })).toBe(20);
    // A different object, same byte size — the layout doesn't need to
    // repack, but this slot's bytes must still be rewritten rather than
    // reusing `first`'s now-stale ones.
    expect(fn.invoke({ textures: { [tex.name]: second } })).toBe(99);
  });

  it("does not notice a texture's data mutated in place without swapping the object — a known, deliberate limitation of the reference-equality cache", () => {
    const tex = uniform("sampler2D");
    const build = () => Fn(() => textureLoad(tex, ivec2(1, 0)).x)();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const texture = { data: [10, 20], width: 2, height: 1, channels: 1 as const };
    expect(fn.invoke({ textures: { [tex.name]: texture } })).toBe(20);
    texture.data[1] = 55; // mutated in place — same object reference
    expect(fn.invoke({ textures: { [tex.name]: texture } })).toBe(20); // stale on purpose
  });

  it("supports textureSize() for a samplerCube uniform, matching compileJS's own lack of restriction there", () => {
    const tex = uniform("samplerCube");
    const build = () => Fn(() => (textureSize as any)(tex).x)();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    expect(fn.invoke({ textures: { [tex.name]: { data: new Float32Array(4 * 4 * 6), width: 4, height: 4 } } })).toBe(4);
  });
});

describe("WASM backend: textureLoad() — unfiltered texel fetch", () => {
  // A 2x2, 4-channel texture: (1,0)'s texel is [2,3,5,7], every other texel
  // is uniform filler so a wrong index reads something recognizably wrong.
  const checkerData = [1, 1, 1, 1, 2, 3, 5, 7, 9, 9, 9, 9, 9, 9, 9, 9];

  it("fetches an in-bounds texel's 4 channels", () => {
    const tex = uniform("sampler2D");
    const build = () =>
      Fn(() => {
        const v = textureLoad(tex, ivec2(1, 0)).toVar();
        return v.x.mul(1).add(v.y.mul(10)).add(v.z.mul(100)).add(v.w.mul(1000));
      })();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const result = fn.invoke({ textures: { [tex.name]: { data: checkerData, width: 2, height: 2 } } });
    expect(result).toBe(2 + 30 + 500 + 7000);
  });

  it("returns all zero for an out-of-bounds texel, including alpha", () => {
    const tex = uniform("sampler2D");
    const build = () =>
      Fn(() => {
        const v = textureLoad(tex, ivec2(5, 5)).toVar();
        return v.x.mul(1).add(v.y.mul(10)).add(v.z.mul(100)).add(v.w.mul(1000));
      })();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const result = fn.invoke({ textures: { [tex.name]: { data: checkerData, width: 2, height: 2 } } });
    expect(result).toBe(0);
  });

  it("returns all zero for a negative coordinate on an ivec2-coordinate fetch", () => {
    const tex = uniform("sampler2D");
    const build = () =>
      Fn(() => {
        const v = textureLoad(tex, ivec2(-1, 0)).toVar();
        return v.x.mul(1).add(v.y.mul(10)).add(v.z.mul(100)).add(v.w.mul(1000));
      })();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const result = fn.invoke({ textures: { [tex.name]: { data: checkerData, width: 2, height: 2 } } });
    expect(result).toBe(0);
  });

  it("defaults missing green/blue to 0 and missing alpha to 1 for a 1-channel texture", () => {
    const tex = uniform("sampler2D");
    const build = () =>
      Fn(() => {
        const v = textureLoad(tex, ivec2(1, 0)).toVar();
        return v.x.mul(1).add(v.y.mul(10)).add(v.z.mul(100)).add(v.w.mul(1000));
      })();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const result = fn.invoke({ textures: { [tex.name]: { data: [42, 84], width: 2, height: 1, channels: 1 } } });
    expect(result).toBe(84 + 0 + 0 + 1000);
  });

  it("divides a Uint8Array-backed texture's value by 255 (unorm)", () => {
    const tex = uniform("sampler2D");
    const build = () => Fn(() => textureLoad(tex, ivec2(0, 0)).x)();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const result = fn.invoke({ textures: { [tex.name]: { data: new Uint8Array([128]), width: 1, height: 1, channels: 1 } } });
    expect(result).toBeCloseTo(128 / 255, 10);
  });

  it("fetches a signed integer sampler's raw value with no division", () => {
    const tex = uniform("isampler2D");
    const build = () => Fn(() => textureLoad(tex, ivec2(1, 0)).x)();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const result = fn.invoke({ textures: { [tex.name]: { data: [100, -50], width: 2, height: 1, channels: 1 } } });
    expect(result).toBe(-50);
  });

  it("fetches an unsigned integer sampler's value above the int32 range correctly", () => {
    const tex = uniform("usampler2D") as any;
    const build = () => Fn(() => (textureLoad(tex, ivec2(0, 0) as any) as any).x)();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const result = fn.invoke({ textures: { [tex.name]: { data: [4000000000], width: 1, height: 1, channels: 1 } } });
    expect(result).toBe(4000000000);
  });

  it("fetches from a sampler3D texture", () => {
    const tex = uniform("sampler3D");
    const build = () => Fn(() => textureLoad(tex, ivec3(1, 0, 1)).x)();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    // width=2, height=1, depth=2, 1 channel: index (z*height+y)*width+x.
    // (1,0,1) -> (1*1+0)*2+1 = 3.
    const data = [10, 20, 30, 40];
    const result = fn.invoke({ textures: { [tex.name]: { data, width: 2, height: 1, depth: 2, channels: 1 } } });
    expect(result).toBe(40);
  });

  it("throws for a samplerCube uniform", () => {
    const tex = uniform("samplerCube") as any;
    const build = () => Fn(() => textureLoad(tex, ivec2(0, 0) as any).x)();
    expect(() => compileWasm(build as any, { name: "main", params: [] })).toThrow(/sampler2D\/sampler3D/);
  });
});

// Mirrors js.test.ts's own texture()/textureLod() cases directly —
// same inputs, same expected outputs — since both backends implement the
// exact same sampling semantics (js.ts's _tex2d/_tex3d/_wrap),
// just ported to different targets.
describe("WASM backend: texture()/textureLod() — filtered sampling", () => {
  function checksum4(v: any) {
    return v.x.add(v.y.mul(10)).add(v.z.mul(100)).add(v.w.mul(1000));
  }

  it("samples textures with nearest-neighbour lookup", () => {
    let tex: any;
    const build = () =>
      Fn(() => {
        tex = uniform("sampler2D");
        return checksum4(tex.texture(vec2(0.5, 0.5)).toVar());
      })();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    // 2x2 RGBA; uv (0.5, 0.5) -> texel (1, 1).
    const data = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4];
    expect(fn.invoke({ textures: { [tex.name]: { data, width: 2, height: 2 } } })).toBe(4 + 40 + 400 + 4000);
  });

  it("reads a byte texture through a float sampler as 0 to 1", () => {
    let tex: any;
    const build = () =>
      Fn(() => {
        tex = uniform("sampler2D");
        return checksum4(tex.texture(vec2(0.5, 0.5)).toVar());
      })();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const data = new Uint8Array([0, 128, 255, 255]);
    const result = fn.invoke({ textures: { [tex.name]: { data, width: 1, height: 1 } } }) as number;
    expect(result).toBeCloseTo(0 + (128 / 255) * 10 + 1 * 100 + 1 * 1000, 9);
  });

  it("leaves a float texture that already holds float data alone", () => {
    let tex: any;
    const build = () =>
      Fn(() => {
        tex = uniform("sampler2D");
        return checksum4(tex.texture(vec2(0.5, 0.5)).toVar());
      })();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const data = new Float32Array([0, 0.5, 1, 1]);
    const result = fn.invoke({ textures: { [tex.name]: { data, width: 1, height: 1 } } }) as number;
    expect(result).toBeCloseTo(0 + 0.5 * 10 + 1 * 100 + 1 * 1000, 9);
  });

  it("fetches integer textures at texel coordinates via texture()", () => {
    let tex: any;
    const build = () =>
      Fn(() => {
        tex = uniform("isampler2D");
        return checksum4(tex.texture(ivec2(1, 0)).toVar());
      })();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    expect(fn.invoke({ textures: { [tex.name]: { data, width: 2, height: 2 } } })).toBe(5 + 60 + 700 + 8000);
  });

  it("strides by the channels a texel holds, not by four", () => {
    let tex: any;
    const build = () =>
      Fn(() => {
        tex = uniform("usampler2D");
        return checksum4(tex.texture(ivec2(2, 0)).toVar());
      })();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const data = new Uint8Array([10, 20, 30, 40]);
    const result = fn.invoke({ textures: { [tex.name]: { data, width: 4, height: 1, channels: 1 } } });
    expect(result).toBe(30 + 0 + 0 + 1000);
  });

  it("blends a single-channel texture without reading its neighbours' channels", () => {
    let tex: any;
    const build = () =>
      Fn(() => {
        tex = uniform("sampler2D");
        return checksum4(tex.texture(vec2(0.5, 0.5)).toVar());
      })();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const texture = { data: [0, 100], width: 2, height: 1, channels: 1 as const };
    expect(fn.invoke({ textures: { [tex.name]: texture } })).toBe(100 + 0 + 0 + 1000);
    expect(fn.invoke({ textures: { [tex.name]: { ...texture, magFilter: "linear" as const } } })).toBe(50 + 0 + 0 + 1000);
  });

  it("normalizes a byte texture fetched with textureLod too", () => {
    let tex: any;
    const build = () =>
      Fn(() => {
        tex = uniform("sampler2D");
        return checksum4(tex.textureLod(vec2(0.5, 0.5), float(0)).toVar());
      })();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const data = new Uint8Array([0, 128, 255, 255]);
    const result = fn.invoke({ textures: { [tex.name]: { data, width: 1, height: 1 } } }) as number;
    expect(result).toBeCloseTo(0 + (128 / 255) * 10 + 1 * 100 + 1 * 1000, 9);
  });

  it("blends neighbouring texels when the texture asks for linear filtering", () => {
    let tex: any;
    const build = () =>
      Fn(() => {
        tex = uniform("sampler2D");
        return tex.texture(vec2(0.5, 0.5)).x;
      })();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    // Two texels, 0 and 100, whose centres sit at 0.25 and 0.75. Sampling
    // halfway between them lands in the second texel outright without
    // filtering, and is half of each with it.
    const data = [0, 0, 0, 0, 100, 100, 100, 100];
    const texture = { data, width: 2, height: 1 };
    expect(fn.invoke({ textures: { [tex.name]: texture } })).toBe(100);
    expect(fn.invoke({ textures: { [tex.name]: { ...texture, magFilter: "linear" as const } } })).toBe(50);
  });

  it("wraps a coordinate past the edge the way the texture asks", () => {
    let tex: any;
    const build = () =>
      Fn(() => {
        tex = uniform("sampler2D");
        return tex.texture(vec2(1.25, 0.5)).x;
      })();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const texture = { data: [10, 10, 10, 10, 20, 20, 20, 20], width: 2, height: 1 };
    const red = (t: any) => fn.invoke({ textures: { [tex.name]: t } });
    // A quarter past the right edge: the last texel stretched, the image
    // tiled back to the first, or tiled and flipped back to the last.
    expect(red(texture)).toBe(20);
    expect(red({ ...texture, wrapS: "repeat" as const })).toBe(10);
    expect(red({ ...texture, wrapS: "mirror" as const })).toBe(20);
  });

  it("wraps behind the left edge too", () => {
    let tex: any;
    const build = () =>
      Fn(() => {
        tex = uniform("sampler2D");
        return tex.texture(vec2(-0.25, 0.5)).x;
      })();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const texture = { data: [10, 10, 10, 10, 20, 20, 20, 20], width: 2, height: 1 };
    const red = (t: any) => fn.invoke({ textures: { [tex.name]: t } });
    expect(red(texture)).toBe(10);
    expect(red({ ...texture, wrapS: "repeat" as const })).toBe(20);
    expect(red({ ...texture, wrapS: "mirror" as const })).toBe(10);
  });

  it("blends across the depth of a 3D texture", () => {
    let tex: any;
    const build = () =>
      Fn(() => {
        tex = uniform("sampler3D");
        return tex.texture(vec3(0.5, 0.5, 0.5)).x;
      })();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    // Two slices, 0 and 100, sampled halfway between their centres.
    const texture = {
      data: [0, 0, 0, 0, 100, 100, 100, 100],
      width: 1,
      height: 1,
      depth: 2,
      magFilter: "linear" as const,
    };
    expect(fn.invoke({ textures: { [tex.name]: texture } })).toBe(50);
  });

  it("samples the right face of a cube map by direction", () => {
    // 6 faces, one texel each, face order +X,-X,+Y,-Y,+Z,-Z: 10,20,30,40,50,60.
    const data = [10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30, 40, 40, 40, 40, 50, 50, 50, 50, 60, 60, 60, 60];
    const texture = { data, width: 1, height: 1 };

    const at = (dx: number, dy: number, dz: number): number => {
      let tex!: any;
      const build = () =>
        Fn(() => {
          tex = uniform("samplerCube");
          return checksum4(tex.texture(vec3(dx, dy, dz)).toVar());
        })();
      const fn = compileWasm(build as any, { name: "main", params: [] });
      return fn.invoke({ textures: { [tex.name]: texture } }) as number;
    };

    const checksum = (v: number) => v + v * 10 + v * 100 + v * 1000;
    expect(at(1, 0, 0)).toBe(checksum(10));
    expect(at(-1, 0, 0)).toBe(checksum(20));
    expect(at(0, 1, 0)).toBe(checksum(30));
    expect(at(0, -1, 0)).toBe(checksum(40));
    expect(at(0, 0, 1)).toBe(checksum(50));
    expect(at(0, 0, -1)).toBe(checksum(60));
  });

  it("matches the JS backend's cube sample for an off-axis direction", () => {
    // 6 faces of 2x2 RGBA: 6*2*2*4 = 96 elements, each texel a distinct value.
    const data = Array.from({ length: 6 * 2 * 2 * 4 }, (_, i) => i / 10);
    const texture = { data, width: 2, height: 2, magFilter: "linear" as const };

    // Each compiler builds its own graph (uniform() picks a fresh slot name
    // per call), so each gets its own tex reference — sharing one `build`
    // closure's side effect between two separate compile calls would leave
    // the second compile's uniform name overwriting the first's.
    let wasmTex!: any;
    const wasmFn = compileWasm(
      () =>
        Fn(() => {
          wasmTex = uniform("samplerCube");
          return wasmTex.texture(vec3(0.3, 0.6, 0.9));
        })(),
      { name: "main", params: [] },
    );
    let jsTex!: any;
    const jsFn = compileJS(
      () =>
        Fn(() => {
          jsTex = uniform("samplerCube");
          return jsTex.texture(vec3(0.3, 0.6, 0.9));
        })(),
      { name: "main", params: [] },
    );
    // A vec4 root is an aggregate result: WASM wraps it as { value }, matching
    // compileWasm's documented shape; JS returns the bare array.
    const wasmResult = wasmFn.invoke({ textures: { [wasmTex.name]: texture } }) as { value: number[] };
    const jsResult = jsFn.invoke({ textures: { [jsTex.name]: texture } }) as number[];
    expect(wasmResult.value).toEqual(jsResult);
  });

  it("throws for isamplerCube/usamplerCube — not supported yet", () => {
    let tex: any;
    const build = () =>
      Fn(() => {
        tex = uniform("isamplerCube");
        return tex.texture(ivec3(0, 0, 0)).x;
      })();
    expect(() => compileWasm(build as any, { name: "main", params: [] })).toThrow(
      /isamplerCube\/usamplerCube aren't supported/,
    );
  });
});

describe("WASM backend: .draw() — render a whole grid in one call", () => {
  it("renders a scalar per pixel, fragCoord at pixel centers", () => {
    const build = () => Fn(() => fragCoord().x)();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const out = fn.batch({}, 3, 2);
    expect(out.length).toBe(3 * 2);
    // Row-major, (y*width+x): x+0.5 regardless of row.
    expect(Array.from(out)).toEqual([0.5, 1.5, 2.5, 0.5, 1.5, 2.5]);
  });

  it("renders both fragCoord axes packed into a vec4 per pixel, with no stage or output() involved", () => {
    const build = () => Fn(() => vec4(fragCoord().x, fragCoord().y, 0, 1))();
    const fn = compileWasm(build as any, { name: "main", params: [] });
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
    const fn = compileWasm(build as any, { name: "main", params: [] });
    expect(Array.from(fn.batch({ uniforms: { [scale.name]: 2 } }, 2, 1))).toEqual([1, 3]);
    expect(Array.from(fn.batch({ uniforms: { [scale.name]: 10 } }, 2, 1))).toEqual([5, 15]);
  });

  it("picks dimensions per call, not at compile time", () => {
    const build = () => Fn(() => fragCoord().x)();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    expect(Array.from(fn.batch({}, 2, 1))).toEqual([0.5, 1.5]);
    expect(Array.from(fn.batch({}, 4, 1))).toEqual([0.5, 1.5, 2.5, 3.5]);
    expect(Array.from(fn.batch({}, 1, 1))).toEqual([0.5]);
  });

  it("the same compiled function still works as a plain single-pixel call — draw() is a choice per call, not a compile mode", () => {
    const scale = uniform("float");
    const build = () => Fn(() => fragCoord().x.mul(scale))();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    expect(fn.invoke({ uniforms: { [scale.name]: 2 }, fragCoord: [3, 0] })).toBe(6);
    expect(Array.from(fn.batch({ uniforms: { [scale.name]: 2 } }, 2, 1))).toEqual([1, 3]);
  });

  it("grows memory for a large grid without corrupting earlier pixels", () => {
    const build = () => Fn(() => fragCoord().x.add(fragCoord().y))();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const small = fn.batch({}, 2, 1);
    expect(Array.from(small)).toEqual([1, 2]); // (0.5+0.5), (1.5+0.5)
    const width = 300,
      height = 300;
    const big = fn.batch({}, width, height);
    expect(big.length).toBe(width * height);
    expect(big[0]).toBe(1); // (0.5 + 0.5)
    expect(big[width * height - 1]).toBe(width - 0.5 + (height - 0.5));
  });

  it("throws when the function produces no value to render", () => {
    const build = () =>
      Fn(() => {
        output("float").assign(float(1));
      })();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    expect(() => fn.batch({}, 1, 1)).toThrow(/produces no value to render/);
  });

  it("samples a texture correctly during draw(), without colliding with the output buffer", () => {
    const tex = uniform("sampler2D") as any;
    const build = () => Fn(() => textureLoad(tex, ivec2(1, 0)).x.add(fragCoord().x))();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const texture = { data: [10, 99], width: 2, height: 1, channels: 1 as const };
    const out = fn.batch({ textures: { [tex.name]: texture } }, 3, 1);
    // texel(1,0) = 99, plus fragCoord().x per pixel (0.5, 1.5, 2.5).
    expect(Array.from(out)).toEqual([99.5, 100.5, 101.5]);
  });

  it("keeps the texture heap and the draw buffer correctly separated across memory growth in both", () => {
    const tex = uniform("sampler2D") as any;
    const build = () => Fn(() => textureLoad(tex, ivec2(0, 0)).x)();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const smallTex = { data: [7], width: 1, height: 1, channels: 1 as const };
    expect(Array.from(fn.batch({ textures: { [tex.name]: smallTex } }, 2, 2))).toEqual([7, 7, 7, 7]);
    // A much bigger texture and a much bigger draw grid together, forcing
    // both growable regions to grow in the same call.
    const width = 200,
      height = 200;
    const texSize = 500;
    const bigTex = {
      data: new Float64Array(texSize * texSize).fill(42),
      width: texSize,
      height: texSize,
      channels: 1 as const,
    };
    const out = fn.batch({ textures: { [tex.name]: bigTex } }, width, height);
    expect(out.length).toBe(width * height);
    expect(out[0]).toBe(42);
    expect(out[out.length - 1]).toBe(42);
  });

  it("interleaves a plain texture-sampling call with draw() using the same texture correctly", () => {
    const tex = uniform("sampler2D") as any;
    const build = () => Fn(() => textureLoad(tex, ivec2(0, 0)).x)();
    const fn = compileWasm(build as any, { name: "main", params: [] });
    const texture = { data: [55], width: 1, height: 1, channels: 1 as const };
    expect(fn.invoke({ textures: { [tex.name]: texture } })).toBe(55);
    expect(Array.from(fn.batch({ textures: { [tex.name]: texture } }, 2, 2))).toEqual([55, 55, 55, 55]);
    expect(fn.invoke({ textures: { [tex.name]: texture } })).toBe(55);
  });
});

describe("WASM backend: instantiateWasm — compile and instantiate as separate steps", () => {
  it("compileWasmFn + instantiateWasm behaves exactly like compileWasm", () => {
    const build = (a: any, b: any) => a.add(b).mul(2);
    const options = {
      name: "main",
      params: [
        { name: "a", type: "float" as const },
        { name: "b", type: "float" as const },
      ],
    };
    const compiled = compileWasmFn(build, options);
    const fn = instantiateWasm(compiled, options.name);
    expect(fn.invoke({ params: { a: 3, b: 4 } })).toBe(14);
  });

  it("instantiates the same compiled bytes more than once, independently", () => {
    const build = (a: any) => a.mul(a);
    const compiled = compileWasmFn(build, { name: "square", params: [{ name: "a", type: "float" }] });
    const first = instantiateWasm(compiled, "square");
    const second = instantiateWasm(compiled, "square");
    expect(first.invoke({ params: { a: 5 } })).toBe(25);
    expect(second.invoke({ params: { a: 6 } })).toBe(36);
    // Independent instances: calling one again doesn't reflect the other's call.
    expect(first.invoke({ params: { a: 5 } })).toBe(25);
  });

  it("draw() still works when instantiated separately from compilation", () => {
    const build = () => Fn(() => fragCoord().x)();
    const compiled = compileWasmFn(build as any, { name: "main", params: [] });
    const fn = instantiateWasm(compiled, "main");
    expect(Array.from(fn.batch({}, 2, 1))).toEqual([0.5, 1.5]);
  });

  it("compiles storage()/invocationIndex() into a per-call array-indexed program", () => {
    // Same per-invocation model as compileJS's storage support: one call per
    // element, `ctx.index` naming which one, `ctx.storages` the backing
    // arrays it reads/writes directly.
    let dt!: UniformNode<"float">;
    const build = () =>
      Fn(() => {
        const vel = storage("vel", "float");
        const pos = storage("pos", "float", { access: "read_write" });
        dt = uniform("float");
        const i = invocationIndex();
        pos.element(i).addAssign(vel.element(i).mul(dt));
      })();
    const fn = compileWasm(build as any, { name: "step", params: [] });

    const pos = new Float64Array([0, 10, 20]);
    const vel = new Float64Array([1, 2, 3]);
    for (let i = 0; i < pos.length; i++) {
      fn.invoke({ storages: { vel, pos }, uniforms: { [dt.name]: 2 }, index: i });
    }
    expect(Array.from(pos)).toEqual([2, 14, 26]);
  });
});

describe("WASM backend: memoryBase — several compiled modules sharing one WebAssembly.Memory", () => {
  it("keeps each module's compile-time addresses in its own, non-overlapping region", () => {
    const memory = new WebAssembly.Memory({ initial: 1 });
    const view = () => new DataView(memory.buffer);

    let vecA!: any;
    const a = compileWasm(
      () => {
        vecA = uniform("vec3");
        return vecA.x;
      },
      { name: "a", params: [], memory, memoryBase: 0 },
    );

    let vecB!: any;
    const b = compileWasm(
      () => {
        vecB = uniform("vec3");
        return vecB.x;
      },
      { name: "b", params: [], memory, memoryBase: 64 }, // clear of a's vec3 (3 * 8 = 24 bytes)
    );

    expect(a.invoke({ uniforms: { [vecA.name]: [1, 2, 3] } })).toBe(1);
    expect(view().getFloat64(0, true)).toBe(1); // a's vec3 landed at its own base, byte 0

    expect(b.invoke({ uniforms: { [vecB.name]: [4, 5, 6] } })).toBe(4);
    expect(view().getFloat64(64, true)).toBe(4); // b's vec3 landed at its own base, byte 64
    expect(view().getFloat64(0, true)).toBe(1); // a's region untouched by b's call
  });
});

describe("WASM backend: scalarsInMemory — scalar uniform/attribute/varying forced into memory", () => {
  it("compiles a scalar uniform to a zero-arg function and still reads its value", () => {
    let threshold!: any;
    const build = () => {
      threshold = uniform("float");
      return threshold.add(1);
    };
    const compiled = compileWasmFn(build, { name: "main", params: [], scalarsInMemory: true });
    expect(compiled.params.some((p) => p.kind === "uniform")).toBe(false);
    expect(compiled.params.some((p) => p.kind === "uniformMemory")).toBe(true);

    const fn = instantiateWasm(compiled, "main");
    expect(fn.invoke({ uniforms: { [threshold.name]: 4 } })).toBe(5);
    expect(fn.invoke({ uniforms: { [threshold.name]: 9 } })).toBe(10);
  });

  it("compiles a scalar attribute to a zero-arg function and still reads its value", () => {
    let a!: any;
    const build = () => {
      a = attribute("float");
      return a.mul(2);
    };
    const compiled = compileWasmFn(build, { name: "main", params: [], scalarsInMemory: true });
    expect(compiled.params.some((p) => p.kind === "attribute")).toBe(false);
    expect(compiled.params.some((p) => p.kind === "attributeMemory")).toBe(true);

    const fn = instantiateWasm(compiled, "main");
    expect(fn.invoke({ attributes: { [a.name]: 3 } })).toBe(6);
  });

  it("compiles a scalar fragment-stage varying to a zero-arg function and still reads its value", () => {
    let v!: any;
    const build = () => {
      v = varying("float");
      return v.sub(1);
    };
    const compiled = compileWasmFn(build, { name: "main", params: [], scalarsInMemory: true });
    expect(compiled.params.some((p) => p.kind === "varying")).toBe(false);
    expect(compiled.params.some((p) => p.kind === "varyingMemory")).toBe(true);

    const fn = instantiateWasm(compiled, "main");
    expect(fn.invoke({ varyings: { [v.name]: 10 } })).toBe(9);
  });

  it("leaves a scalar-only program at a real function argument by default", () => {
    let threshold!: any;
    const build = () => {
      threshold = uniform("float");
      return threshold.add(1);
    };
    const compiled = compileWasmFn(build, { name: "main", params: [] });
    expect(compiled.params.some((p) => p.kind === "uniform")).toBe(true);
    expect(compiled.params.some((p) => p.kind === "uniformMemory")).toBe(false);
  });
});
