/**
 * Evaluates the WASM (CPU) backend in-process.
 *
 * Unlike the JS backend in rmsl-js.test.ts, this one is not yet part of the
 * DSL's breadth — it covers the op set ROADMAP.md's Phase 1 and Phase 2
 * describe: scalar float/int/uint/bool arithmetic and casts, function
 * params, float/vec3 uniforms, `If`/`Else`, and vec3 `dot`. Cases here check
 * `compileWasm` against `compileJS` directly rather than through the shared
 * shader-eval recording, since a WASM case using an op this backend doesn't
 * support yet would fail the other backends' replay for the wrong reason.
 *
 * Comparisons involving `uint` are the one place this backend's coverage can
 * diverge from `compileJS` on purpose: the JS backend computes every
 * declared type as a plain JS number, so a uint holding a value above
 * 2^31-1 compares correctly as a large positive number, while a naive WASM
 * int comparison reading the same bit pattern as signed would see a negative
 * one. Phase 2 threads the unsigned opcode variant through for exactly this
 * reason — the tests below with a uint above that boundary are checking that
 * distinction, not just parroting compileJS.
 */

import { describe, it, expect } from "vitest";
import {
  compileWasm, compileJS, Fn, If, float, int, uint, bool, uniform, vec2, vec3, vec4,
  ivec3, uvec3, mat2, mat3, mat4, sin, clamp,
  type Node, type ShaderType,
} from "./rmsl";

function run(build: (...args: any[]) => Node<ShaderType>, args: number[] = [], types: ShaderType[] = []): number | boolean {
  const params = args.map((_, i) => ({ name: `a${i}`, type: types[i] ?? "float" as const }));
  const fn = compileWasm(build, { name: "main", params });
  return fn({ params: Object.fromEntries(args.map((a, i) => [`a${i}`, a])) });
}

describe("WASM backend: scalar arithmetic", () => {
  it("computes arithmetic on function params", () => {
    expect(run((a, b) => a.add(b), [2, 3])).toBe(5);
    expect(run((a, b) => a.sub(b), [7, 3])).toBe(4);
    expect(run((a, b) => a.mul(b), [3, 4])).toBe(12);
    expect(run(a => a.sqrt(), [16])).toBe(4);
  });

  it("reads a float uniform", () => {
    let u!: any;
    const build = () => { u = uniform("float"); return u.mul(2); };
    const fn = compileWasm(build, { name: "main", params: [] });
    expect(fn({ uniforms: { [u.name]: 21 } })).toBe(42);
  });

  it("rejects a multi-return function", () => {
    expect(() => compileWasm((() => [float(1), float(2)]) as any, { name: "main", params: [] }))
      .toThrow(/multi-return/);
  });

  it("rejects a non-scalar result", () => {
    expect(() => compileWasm(() => vec3(1, 2, 3) as any, { name: "main", params: [] }))
      .toThrow(/scalar result/);
  });

  it("rejects an op outside this backend's coverage so far", () => {
    const build = () => clamp(uniform("float"), float(0), float(1));
    expect(() => compileWasm(build as any, { name: "main", params: [] }))
      .toThrow(/unsupported node type/);
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
    expect(run(a => a.sign(), [-5])).toBe(-1);
    expect(run(a => a.sign(), [0])).toBe(0);
    expect(run(a => a.sign(), [5])).toBe(1);
    expect(run(a => a.abs(), [-5])).toBe(5);
    expect(run(a => a.sign(), [-5], ["int"])).toBe(-1);
    expect(run(a => a.abs(), [-5], ["int"])).toBe(5);
  });

  it("floors, ceils, truncs, fracts, and rounds", () => {
    expect(run(a => a.floor(), [3.7])).toBe(3);
    expect(run(a => a.ceil(), [3.2])).toBe(4);
    expect(run(a => a.trunc(), [-3.7])).toBe(-3);
    expect(run(a => a.fract(), [3.25])).toBeCloseTo(0.25, 9);
    expect(run(a => a.round(), [2.5])).toBe(3);
    expect(run(a => a.round(), [-2.5])).toBe(-2);
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
    expect(run(a => a.bitNot(), [6], ["int"])).toBe(-7);
    expect(run((a, b) => a.shiftLeft(b), [1, 4], ["int", "int"])).toBe(16);
  });

  it("shifts uints logically, ints arithmetically", () => {
    expect(run((a, b) => a.shiftRight(b), [-16, 2], ["int", "int"])).toBe(-4);
    expect(run((a, b) => a.shiftRight(b), [4000000000, 2], ["uint", "uint"])).toBe(1000000000);
  });
});

describe("WASM backend: transcendentals via host import", () => {
  it("calls Math functions through a WASM import", () => {
    expect(run(a => a.sin(), [1.2345])).toBeCloseTo(Math.sin(1.2345), 9);
    expect(run(a => a.cos(), [1.2345])).toBeCloseTo(Math.cos(1.2345), 9);
    expect(run(a => a.exp(), [2])).toBeCloseTo(Math.exp(2), 9);
    expect(run(a => a.log(), [2])).toBeCloseTo(Math.log(2), 9);
    expect(run((a, b) => a.pow(b), [2, 10])).toBe(1024);
    expect(run(a => a.exp2(), [10])).toBe(1024);
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
    expect(wasmFn(ctx)).toBeCloseTo(jsFn(ctx) as number, 9);
  });
});

describe("WASM backend: int/uint/bool values and casts", () => {
  it("returns int/uint/bool results, and reads int/uint/bool uniforms", () => {
    let iu!: any;
    const intResult = compileWasm(() => { iu = uniform("int"); return iu.add(int(1)); }, { name: "main", params: [] });
    expect(intResult({ uniforms: { [iu.name]: 5 } })).toBe(6);

    let uu!: any;
    const uintResult = compileWasm(() => { uu = uniform("uint"); return uu.add(uint(1)); }, { name: "main", params: [] });
    expect(uintResult({ uniforms: { [uu.name]: 5 } })).toBe(6);

    let bu!: any;
    const boolResult = compileWasm(() => { bu = uniform("bool"); return bu.not(); }, { name: "main", params: [] });
    expect(boolResult({ uniforms: { [bu.name]: true } })).toBe(false);
  });

  it("casts between float, int, uint, and bool", () => {
    expect(run(a => a.toInt(), [3.9])).toBe(3); // truncates
    expect(run(a => a.toFloat(), [3], ["int"])).toBe(3);
    expect(run(a => a.toBool(), [0], ["float"])).toBe(false);
    expect(run(a => a.toBool(), [5], ["float"])).toBe(true);
    expect(run(a => a.toBool(), [0], ["int"])).toBe(false);
    expect(run(a => a.toInt(), [1], ["bool"])).toBe(1);
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
      expect(wasmFn({ params: { x } })).toBe(jsFn({ params: { x } }));
    }
  });
});

describe("WASM backend: control flow", () => {
  it("takes the branch If/Else selects", () => {
    const branch = (x: Node<"float">) => Fn(() => {
      const out = float(0).toVar();
      If(x.greaterThan(1), () => { out.assign(float(10)); })
        .Else(() => { out.assign(float(20)); });
      return out;
    })();
    expect(run(branch, [2])).toBe(10);
    expect(run(branch, [0])).toBe(20);
  });
});

describe("WASM backend: vec3 dot", () => {
  it("computes a dot product through a vec3 uniform", () => {
    let dir!: any;
    let target!: any;
    const build = () => { dir = uniform("vec3"); target = uniform("vec3"); return dir.dot(target); };
    const fn = compileWasm(build, { name: "main", params: [] });
    expect(fn({ uniforms: { [dir.name]: [1, 2, 3], [target.name]: [4, 5, 6] } })).toBe(32);
  });

  it("drives a picking-style hit test (dot + If/Else)", () => {
    let dir!: any;
    let target!: any;
    let threshold!: any;
    const build = () => Fn(() => {
      dir = uniform("vec3");
      target = uniform("vec3");
      threshold = uniform("float");
      const hit = float(0).toVar();
      If(dir.dot(target).greaterThan(threshold), () => { hit.assign(float(1)); })
        .Else(() => { hit.assign(float(0)); });
      return hit;
    })();
    const fn = compileWasm(build, { name: "main", params: [] });
    const ctx = (d: number[], t: number[], th: number) => ({
      uniforms: { [dir.name]: d, [target.name]: t, [threshold.name]: th },
    });
    expect(fn(ctx([1, 0, 0], [1, 0, 0], 0.5))).toBe(1); // parallel: dot=1
    expect(fn(ctx([1, 0, 0], [0, 1, 0], 0.5))).toBe(0); // perpendicular: dot=0
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
    const branch = (x: Node<"float">) => Fn(() => {
      const v = vec3(0, 0, 0).toVar();
      If(x.greaterThan(0), () => { v.assign(vec3(1, 2, 3)); })
        .Else(() => { v.assign(vec3(4, 5, 6)); });
      return v.dot(v) as any;
    })();
    expect(run(branch, [1])).toBe(14); // 1+4+9
    expect(run(branch, [-1])).toBe(77); // 16+25+36
  });

  it("assigns a multi-component swizzle target", () => {
    const build = () => Fn(() => {
      const v = vec3(1, 2, 3).toVar();
      v.xy.assign(vec2(10, 20));
      return v.dot(v) as any;
    })();
    expect(run(build)).toBe(509); // 100+400+9
  });

  it("assigns a single-component swizzle target", () => {
    const build = () => Fn(() => {
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

  it("reads int/uint vector components", () => {
    // bvec's single-component swizzle is typed "float" not "bool" by
    // rmsl-core's swizzle() (it only special-cases ivec/uvec prefixes) —
    // a pre-existing core quirk, not something this backend introduces, so
    // bvec swizzle reads aren't exercised here.
    expect(run(() => ivec3(1, -2, 3).y as any)).toBe(-2);
    expect(run(() => uvec3(1, 2, 3).z as any)).toBe(3);
  });
});

describe("WASM backend: componentwise vector arithmetic", () => {
  it("adds/subs/muls/divs vector-vector", () => {
    expect(run(() => vec3(1, 2, 3).add(vec3(4, 5, 6)).dot(vec3(1, 2, 3).add(vec3(4, 5, 6))) as any)).toBe(155);
    expect(run(() => vec3(2, 4, 6).sub(vec3(1, 1, 1)).dot(vec3(1, 3, 5)) as any)).toBe(35);
    expect(run(() => vec2(3, 4).mul(vec2(2, 2)).dot(vec2(6, 8)) as any)).toBe(100);
  });

  it("broadcasts a scalar across a vector for mul/div", () => {
    expect(run(() => vec3(1, 2, 3).mul(2).dot(vec3(2, 4, 6)) as any)).toBe(56);
    expect(run(() => vec3(2, 4, 6).div(2).dot(vec3(1, 2, 3)) as any)).toBe(14);
  });

  it("adds int vectors componentwise", () => {
    expect(run(() => ivec3(1, 2, 3).add(ivec3(10, 20, 30)).y as any)).toBe(22);
  });
});

describe("WASM backend: aggregate function params", () => {
  it("reads a vec3 function param via memory, not a WASM arg", () => {
    const fn = compileWasm((v: any) => v.dot(v), { name: "main", params: [{ name: "v", type: "vec3" }] });
    expect(fn({ params: { v: [1, 2, 3] } })).toBe(14);
  });
});
