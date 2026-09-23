import { bench, describe } from "vitest";
import { compileWasmRoutine } from "../wasm";
import { compileJSRoutine, type CpuTextureData, type CpuShaderContext } from "../js";
import { Fn, uniform, fragCoord, sqrt, ivec2, textureLoad } from "../rmsl";

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

    const wasmFn = compileWasmRoutine(build as any, { name: "main", params: [] });
    const jsFn = compileJSRoutine(build as any, { name: "main", params: [] });
    const drawCtx = { uniforms: { [cx.name]: SIZE / 2, [cy.name]: SIZE / 2 } };
    // Reused across every pixel in the loops below — only `fragCoord`'s
    // two numbers change, in place, so no per-pixel allocation at all.
    const perPixelCtx: CpuShaderContext = { uniforms: drawCtx.uniforms, fragCoord: [0, 0] };

    bench("draw() — one call for the whole grid", () => {
      wasmFn.draw(drawCtx, SIZE, SIZE);
    });

    bench("compileWasmRoutine, one call per pixel — same grid, same program", () => {
      for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
          (perPixelCtx.fragCoord as number[])[0] = x + 0.5;
          (perPixelCtx.fragCoord as number[])[1] = y + 0.5;
          wasmFn.run(perPixelCtx);
        }
      }
    });

    // The realistic alternative: this backend's own "called once per
    // pixel" niche (ROADMAP.md's "Why") already found compileJSRoutine beats a
    // per-call compileWasmRoutine here — draw()'s actual value proposition is
    // this comparison, not the one above.
    bench("compileJSRoutine, one call per pixel — same grid, same program", () => {
      for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
          (perPixelCtx.fragCoord as number[])[0] = x + 0.5;
          (perPixelCtx.fragCoord as number[])[1] = y + 0.5;
          jsFn.run(perPixelCtx);
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
    const wasmFn = compileWasmRoutine(build as any, { name: "main", params: [] });
    const jsFn = compileJSRoutine(build as any, { name: "main", params: [] });
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

    bench("compileJSRoutine, one call per pixel, sampling the same texture", () => {
      for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
          (perPixelCtx.fragCoord as number[])[0] = x + 0.5;
          (perPixelCtx.fragCoord as number[])[1] = y + 0.5;
          jsFn.run(perPixelCtx);
        }
      }
    });
  });
}
