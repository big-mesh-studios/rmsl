/**
 * Evaluates the WASM (CPU) backend in-process.
 *
 * Unlike the JS backend in rmsl-js.test.ts, this one is not yet part of the
 * DSL's breadth — it covers only the narrow op set ROADMAP.md's Phase 1
 * promoted out of a throwaway prototype: float arithmetic, function params,
 * float/vec3 uniforms, `If`/`Else`, and vec3 `dot`. Cases here check
 * `compileWasm` against `compileJS` directly rather than through the shared
 * shader-eval recording, since a WASM case using an op this backend doesn't
 * support yet would fail the other backends' replay for the wrong reason.
 */

import { describe, it, expect } from "vitest";
import {
  compileWasm, compileJS, Fn, If, float, uniform, vec3, sin,
  type Node, type ShaderType,
} from "./rmsl";

function run(build: (...args: Node<"float">[]) => Node<ShaderType>, args: number[] = []): number {
  const params = args.map((_, i) => ({ name: `a${i}`, type: "float" as const }));
  const fn = compileWasm(build as any, { name: "main", params });
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

  it("rejects a non-float result", () => {
    expect(() => compileWasm(() => vec3(1, 2, 3) as any, { name: "main", params: [] }))
      .toThrow(/"float" result/);
  });

  it("rejects an op outside this backend's coverage so far", () => {
    const build = () => sin(uniform("float"));
    expect(() => compileWasm(build as any, { name: "main", params: [] }))
      .toThrow(/unsupported node type/);
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

  it("agrees with compileJS across both branches", () => {
    const branch = (x: Node<"float">) => Fn(() => {
      const out = float(0).toVar();
      If(x.greaterThan(1), () => { out.assign(float(10)); })
        .Else(() => { out.assign(float(20)); });
      return out;
    })();
    for (const x of [0, 2]) {
      const wasmFn = compileWasm((xx: any) => branch(xx), { name: "main", params: [{ name: "x", type: "float" }] });
      const jsFn = compileJS((xx: any) => branch(xx), { name: "main", params: [{ name: "x", type: "float" }] });
      expect(wasmFn({ params: { x } })).toBe(jsFn({ params: { x } }));
    }
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
