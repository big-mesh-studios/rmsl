import { bench, describe } from "vitest";
import { compileWasmRoutine, compileWasmFn } from "../wasm";
import { compileJSRoutine } from "../js";
import { Fn, If, uniform, float, sqrt, type Node } from "../rmsl";

describe("scalar arithmetic: sqrt(a*a + b*b + c*c)", () => {
  const build = (a: Node<"float">, b: Node<"float">, c: Node<"float">) => sqrt(a.mul(a).add(b.mul(b)).add(c.mul(c)));
  const params = [
    { name: "a", type: "float" as const },
    { name: "b", type: "float" as const },
    { name: "c", type: "float" as const },
  ];
  const wasmFn = compileWasmRoutine(build as any, { name: "main", params });
  const jsFn = compileJSRoutine(build as any, { name: "main", params });
  const ctx = { params: { a: 3, b: 4, c: 12 } };

  // Isolates compileWasmRoutine's ctx-marshalling wrapper (params array iteration,
  // property lookups by name) from the WASM call itself, by calling the
  // raw exported function directly with positional args.
  const { bytes } = compileWasmFn(build as any, { name: "main", params });
  const instance = new WebAssembly.Instance(new WebAssembly.Module(bytes.buffer as ArrayBuffer), { math: Math as any });
  const rawMain = instance.exports.main as (a: number, b: number, c: number) => number;

  bench("compileWasmRoutine", () => {
    wasmFn.run(ctx);
  });
  bench("compileWasmRoutine, raw exported function (no ctx wrapper)", () => {
    rawMain(3, 4, 12);
  });
  bench("compileJSRoutine", () => {
    jsFn.run(ctx);
  });
});

describe("vector dot + branch: If(dir.dot(target) > threshold)", () => {
  // Built once: compileWasmRoutine/compileJSRoutine each call the function handed to
  // them, so a build function that calls uniform()/Fn() itself would mint
  // a fresh graph (and fresh uniform names) per backend instead of sharing
  // one — the graph is built up front and handed to both as `() => node`.
  const dir = uniform("vec3");
  const target = uniform("vec3");
  const threshold = uniform("float");
  const node = Fn(() => {
    const out = float(0).toVar();
    If(dir.dot(target).greaterThan(threshold), () => {
      out.assign(float(1));
    }).Else(() => {
      out.assign(float(0));
    });
    return out;
  })();
  const wasmFn = compileWasmRoutine(() => node as any, { name: "main", params: [] });
  const jsFn = compileJSRoutine(() => node as any, { name: "main", params: [] });
  const ctx = { uniforms: { [dir.name]: [1, 0, 0], [target.name]: [0.9, 0.1, 0], [threshold.name]: 0.5 } };

  bench("compileWasmRoutine", () => {
    wasmFn.run(ctx);
  });
  bench("compileJSRoutine", () => {
    jsFn.run(ctx);
  });
});
