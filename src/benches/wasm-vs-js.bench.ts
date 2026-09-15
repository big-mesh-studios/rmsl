/**
 * WASM vs JS backend perf comparison, using vitest's benchmark mode (run
 * with `npx vitest bench src/wasm-vs-js.bench.ts`, not `vitest run` —
 * this is excluded from the normal test suite by vitest's default
 * benchmark-file pattern, so it never slows down `npm test`).
 *
 * Kept as a real, committed file — not a throwaway script — specifically so
 * a reported number can be reproduced later: check out the commit it's
 * cited against (ROADMAP.md's "Why" section cites this file by path and
 * commit SHA for each measurement) and run it again, or extract this exact
 * version with `git show <sha>:src/wasm-vs-js.bench.ts` and run it
 * against a different commit (its content only depends on `compileWasm`'s/
 * `compileJS`'s public call signature, which has been stable since Phase 1,
 * so the same file works unmodified against a pre-linear-memory commit too
 * — that's how the ROADMAP's linear-memory A/B was produced).
 *
 * Deliberately excludes any `for`/`while`/`Loop` scenario — those need
 * Phase 4, so putting one here would make this file fail to even load
 * against a pre-Phase-4 commit. See `wasm-loop.bench.ts` for that.
 */
import { bench, describe } from "vitest";
import { compileWasm, compileWasmFn } from "../wasm";
import { compileJS } from "../js";
import { Fn, If, uniform, float, sqrt, type Node } from "../rmsl";

describe("scalar arithmetic: sqrt(a*a + b*b + c*c)", () => {
  const build = (a: Node<"float">, b: Node<"float">, c: Node<"float">) => sqrt(a.mul(a).add(b.mul(b)).add(c.mul(c)));
  const params = [
    { name: "a", type: "float" as const },
    { name: "b", type: "float" as const },
    { name: "c", type: "float" as const },
  ];
  const wasmFn = compileWasm(build as any, { name: "main", params });
  const jsFn = compileJS(build as any, { name: "main", params });
  const ctx = { params: { a: 3, b: 4, c: 12 } };

  // Isolates compileWasm's ctx-marshalling wrapper (params array iteration,
  // property lookups by name) from the WASM call itself, by calling the
  // raw exported function directly with positional args.
  const { bytes } = compileWasmFn(build as any, { name: "main", params });
  const instance = new WebAssembly.Instance(new WebAssembly.Module(bytes.buffer as ArrayBuffer), { math: Math as any });
  const rawMain = instance.exports.main as (a: number, b: number, c: number) => number;

  bench("compileWasm", () => {
    wasmFn(ctx);
  });
  bench("compileWasm, raw exported function (no ctx wrapper)", () => {
    rawMain(3, 4, 12);
  });
  bench("compileJS", () => {
    jsFn(ctx);
  });
});

describe("vector dot + branch: If(dir.dot(target) > threshold)", () => {
  // Built once: compileWasm/compileJS each call the function handed to
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
  const wasmFn = compileWasm(() => node as any, { name: "main", params: [] });
  const jsFn = compileJS(() => node as any, { name: "main", params: [] });
  const ctx = { uniforms: { [dir.name]: [1, 0, 0], [target.name]: [0.9, 0.1, 0], [threshold.name]: 0.5 } };

  bench("compileWasm", () => {
    wasmFn(ctx);
  });
  bench("compileJS", () => {
    jsFn(ctx);
  });
});
