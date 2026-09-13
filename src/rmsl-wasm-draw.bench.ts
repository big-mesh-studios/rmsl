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
import { compileWasm, compileJS, Fn, uniform, fragCoord, sqrt, ivec2, textureLoad, type JsTextureData } from "./rmsl";

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

// draw() combined with a texture uniform — the two growable regions
// (the texture heap and the draw output buffer) share the same memory,
// placed back to back fresh every call (see ROADMAP.md's texture-and-draw
// design note). Measures whether that placement math costs anything
// noticeable on top of draw()'s own win.
describe(`draw() with a sampled texture vs one call per pixel over a ${WIDTH}x${HEIGHT} grid`, () => {
  const tex = uniform("sampler2D") as any;
  const build = () => Fn(() => textureLoad(tex, ivec2(fragCoord().x.toInt(), fragCoord().y.toInt())).x)();
  const wasmFn = compileWasm(build as any, { name: "main", params: [] });
  const jsFn = compileJS(build as any, { name: "main", params: [] });
  const texture: JsTextureData = { data: new Float64Array(WIDTH * HEIGHT).fill(1), width: WIDTH, height: HEIGHT, channels: 1 };
  const ctx = { textures: { [tex.name]: texture } };

  bench("draw() — one call for the whole grid, sampling a texture", () => {
    wasmFn.draw(ctx, WIDTH, HEIGHT);
  });

  bench("compileJS, one call per pixel, sampling the same texture", () => {
    for (let y = 0; y < HEIGHT; y++) {
      for (let x = 0; x < WIDTH; x++) {
        jsFn({ ...ctx, fragCoord: [x + 0.5, y + 0.5] });
      }
    }
  });
});
