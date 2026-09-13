/**
 * `.draw()` vs. calling `compileWasm` once per pixel for the same
 * `width x height` grid — the whole point of `.draw()` is skipping the
 * per-call marshalling cost (see `ROADMAP.md`'s "Why", the wrapper-cost
 * findings, and the loop-length crossover benchmark) by paying it once
 * for the entire image instead of once per pixel.
 *
 * Run with `npx vitest bench src/rmsl-wasm-draw.bench.ts`.
 */
import { bench, describe } from "vitest";
import { compileWasm, compileJS, Fn, uniform, fragCoord, sqrt } from "./rmsl";

const WIDTH = 128, HEIGHT = 128;

describe(`draw() vs one call per pixel: sqrt(fragCoord distance) over a ${WIDTH}x${HEIGHT} grid`, () => {
  const cx = uniform("float");
  const cy = uniform("float");
  const build = () => Fn(() => {
    const dx = fragCoord().x.sub(cx);
    const dy = fragCoord().y.sub(cy);
    return sqrt(dx.mul(dx).add(dy.mul(dy)));
  })();

  const wasmFn = compileWasm(build as any, { name: "main", params: [] });
  const jsFn = compileJS(build as any, { name: "main", params: [] });
  const ctx = { uniforms: { [cx.name]: WIDTH / 2, [cy.name]: HEIGHT / 2 } };

  bench("draw() — one call for the whole grid", () => {
    wasmFn.draw(ctx, WIDTH, HEIGHT);
  });

  bench("compileWasm, one call per pixel — same grid, same program", () => {
    for (let y = 0; y < HEIGHT; y++) {
      for (let x = 0; x < WIDTH; x++) {
        wasmFn({ ...ctx, fragCoord: [x + 0.5, y + 0.5] });
      }
    }
  });

  // The realistic alternative: this backend's own "called once per
  // pixel" niche (ROADMAP.md's "Why") already found compileJS beats a
  // per-call compileWasm here — draw()'s actual value proposition is
  // this comparison, not the one above.
  bench("compileJS, one call per pixel — same grid, same program", () => {
    for (let y = 0; y < HEIGHT; y++) {
      for (let x = 0; x < WIDTH; x++) {
        jsFn({ ...ctx, fragCoord: [x + 0.5, y + 0.5] });
      }
    }
  });
});
