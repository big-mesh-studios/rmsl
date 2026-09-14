/**
 * `.draw()` vs. calling `compileWasm`/`compileJS` once per pixel for the
 * same `width x height` grid — the whole point of `.draw()` is skipping
 * the per-call marshalling cost (see `ROADMAP.md`'s "Why", the
 * wrapper-cost findings, and the loop-length crossover benchmark) by
 * paying it once for the entire image instead of once per pixel.
 *
 * The per-pixel loops reuse one `ctx` object and mutate its `fragCoord`
 * array in place, rather than allocating `{ ...ctx, fragCoord: [...] }`
 * fresh every pixel — that would pile up real garbage-collection pressure
 * that has nothing to do with either backend's own per-pixel cost, and
 * grows with the grid size, which would make a size sweep like this one
 * misleading rather than clarifying.
 *
 * Swept across two grid sizes (128x128 and 512x512, a 16x difference in
 * pixel count) to check the win holds at scale rather than being an
 * artifact of one arbitrarily chosen size — both a plain arithmetic
 * scenario and one sampling a texture (which showed a materially
 * different result at 128x128: a large win with no texture, close to
 * parity with one).
 *
 * Run with `npx vitest bench src/wasm-draw.bench.ts`.
 */
import { bench, describe } from "vitest";
import {
  compileWasm,
  compileJS,
  Fn,
  uniform,
  fragCoord,
  sqrt,
  ivec2,
  textureLoad,
  type CpuTextureData,
  type CpuShaderContext,
} from "../rmsl";

for (const SIZE of [128, 512]) {
  describe(`draw() vs one call per pixel: sqrt(fragCoord distance) over a ${SIZE}x${SIZE} grid`, () => {
    const cx = uniform("float");
    const cy = uniform("float");
    const build = () =>
      Fn(() => {
        const dx = fragCoord().x.sub(cx);
        const dy = fragCoord().y.sub(cy);
        return sqrt(dx.mul(dx).add(dy.mul(dy)));
      })();

    const wasmFn = compileWasm(build as any, { name: "main", params: [] });
    const jsFn = compileJS(build as any, { name: "main", params: [] });
    const drawCtx = { uniforms: { [cx.name]: SIZE / 2, [cy.name]: SIZE / 2 } };
    // Reused across every pixel in the loops below — only `fragCoord`'s
    // two numbers change, in place, so no per-pixel allocation at all.
    const perPixelCtx: CpuShaderContext = { uniforms: drawCtx.uniforms, fragCoord: [0, 0] };

    bench("draw() — one call for the whole grid", () => {
      wasmFn.draw(drawCtx, SIZE, SIZE);
    });

    bench("compileWasm, one call per pixel — same grid, same program", () => {
      for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
          (perPixelCtx.fragCoord as number[])[0] = x + 0.5;
          (perPixelCtx.fragCoord as number[])[1] = y + 0.5;
          wasmFn(perPixelCtx);
        }
      }
    });

    // The realistic alternative: this backend's own "called once per
    // pixel" niche (ROADMAP.md's "Why") already found compileJS beats a
    // per-call compileWasm here — draw()'s actual value proposition is
    // this comparison, not the one above.
    bench("compileJS, one call per pixel — same grid, same program", () => {
      for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
          (perPixelCtx.fragCoord as number[])[0] = x + 0.5;
          (perPixelCtx.fragCoord as number[])[1] = y + 0.5;
          jsFn(perPixelCtx);
        }
      }
    });
  });

  // draw() combined with a texture uniform — the two growable regions
  // (the texture heap and the draw output buffer) share the same memory,
  // placed back to back fresh every call (see ROADMAP.md's texture-and-draw
  // design note). Measures whether that placement math costs anything
  // noticeable on top of draw()'s own win, and whether the near-parity
  // result found at 128x128 holds, worsens, or improves at 512x512.
  describe(`draw() with a sampled texture vs one call per pixel over a ${SIZE}x${SIZE} grid`, () => {
    const tex = uniform("sampler2D") as any;
    const build = () => Fn(() => textureLoad(tex, ivec2(fragCoord().x.toInt(), fragCoord().y.toInt())).x)();
    const wasmFn = compileWasm(build as any, { name: "main", params: [] });
    const jsFn = compileJS(build as any, { name: "main", params: [] });
    const texture: CpuTextureData = {
      data: new Float64Array(SIZE * SIZE).fill(1),
      width: SIZE,
      height: SIZE,
      channels: 1,
    };
    const drawCtx = { textures: { [tex.name]: texture } };
    const perPixelCtx: CpuShaderContext = { textures: drawCtx.textures, fragCoord: [0, 0] };

    bench("draw() — one call for the whole grid, sampling a texture", () => {
      wasmFn.draw(drawCtx, SIZE, SIZE);
    });

    bench("compileJS, one call per pixel, sampling the same texture", () => {
      for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
          (perPixelCtx.fragCoord as number[])[0] = x + 0.5;
          (perPixelCtx.fragCoord as number[])[1] = y + 0.5;
          jsFn(perPixelCtx);
        }
      }
    });
  });
}
