/**
 * Phase 7 (ROADMAP.md, "parity testing infrastructure"): pin down where a
 * loop's accumulated per-iteration work amortizes `compileWasm`'s fixed
 * per-call wrapper cost enough to cross over from a loss against
 * `compileJS` to a win, rather than only knowing it loses at zero loop
 * iterations (`rmsl-wasm-vs-js.bench.ts`'s plain scalar case) and wins at
 * one arbitrarily chosen loop length (64 iterations,
 * `rmsl-wasm-loop.bench.ts`).
 *
 * Same workload as `rmsl-wasm-loop.bench.ts` (`sum of sqrt(i)`), swept
 * across a range of iteration counts instead of fixed at 64, each compiled
 * once up front — the loop bound is baked into the compiled function
 * (`int(n)`), not a runtime parameter, so there is nothing here either
 * backend wouldn't also pay for a real fixed-length loop.
 *
 * Run with `npx vitest bench src/rmsl-wasm-crossover.bench.ts`. Kept
 * separate from `rmsl-wasm-loop.bench.ts` (rather than folding this sweep
 * into it) so that file's single, simple 64-iteration case stays the
 * quick thing to point at, and this sweep stays the thing to point at for
 * "where exactly is the crossover".
 */
import { bench, describe } from "vitest";
import { compileWasm, compileJS, Fn, For, float, int, sqrt } from "../rmsl";

function buildLoop(n: number) {
  return () =>
    Fn(() => {
      const sum = float(0).toVar();
      For(
        () => int(0).toVar(),
        (i) => i.lessThan(int(n)),
        (i) => {
          i.assign(i.add(int(1)));
        },
        (i) => {
          sum.assign(sum.add(sqrt(i.toFloat())));
        },
      );
      return sum;
    })();
}

for (const n of [1, 2, 4, 8, 16, 32, 64, 128]) {
  describe(`loop: sum of sqrt(i) for i in [0, ${n})`, () => {
    const build = buildLoop(n);
    const wasmFn = compileWasm(build as any, { name: "main", params: [] });
    const jsFn = compileJS(build as any, { name: "main", params: [] });
    const ctx = {};

    bench("compileWasm", () => {
      wasmFn(ctx);
    });
    bench("compileJS", () => {
      jsFn(ctx);
    });
  });
}
