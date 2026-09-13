/**
 * WASM vs JS backend perf comparison for Phase 6's texture sampling.
 *
 * Unlike every other scenario in `rmsl-wasm-vs-js.bench.ts`, a texture
 * uniform's data isn't copied once and left alone — `compileWasm`'s wrapper
 * repacks and rewrites the whole texture into its linear-memory heap on
 * every single call (see `ROADMAP.md`, "Texture data lives in linear
 * memory, not behind a host call"), since the bound texture can change
 * call to call and there is no cheap way to detect "same texture as last
 * time" from a plain `ArrayLike`. `compileJS` pays nothing per call for
 * this — it reads directly out of `ctx.textures[slot]`. So this is
 * expected to be the one place `compileWasm` pays a real, structural
 * per-call cost `compileJS` doesn't, independent of loop length — worth
 * measuring on its own rather than assuming it away.
 *
 * Three scenarios, cheapest to most expensive sampling math, all against
 * the same 8x8 RGBA texture: `textureSize()` (metadata only, no per-texel
 * math or heap indexing at all), `textureLoad()` (unfiltered, one texel),
 * and `texture()` with bilinear filtering (four texel reads and three
 * lerps). If the per-call copy cost dominates, all three should show a
 * similar WASM/JS ratio; if it doesn't, the ratio should widen as the
 * sampling math gets more expensive, the same way `rmsl-wasm-vs-js.bench.ts`'s
 * scalar case shows the wrapper cost separately from call-boundary cost.
 *
 * Run with `npx vitest bench src/rmsl-wasm-texture.bench.ts`.
 */
import { bench, describe } from "vitest";
import { compileWasm, compileJS, Fn, uniform, vec2, ivec2, textureSize, textureLoad, type JsTextureData } from "./rmsl";

const size = 8;
const data = new Float32Array(size * size * 4);
for (let i = 0; i < data.length; i++) data[i] = (i % 97) / 97;
const texture: JsTextureData = { data, width: size, height: size, magFilter: "linear" };

describe("textureSize(): metadata round trip only, no sampling math", () => {
  const tex = uniform("sampler2D") as any;
  const build = () => Fn(() => textureSize(tex).x.toFloat())();
  const wasmFn = compileWasm(build as any, { name: "main", params: [] });
  const jsFn = compileJS(build as any, { name: "main", params: [] });
  const ctx = { textures: { [tex.name]: texture } };

  bench("compileWasm", () => { wasmFn(ctx); });
  bench("compileJS", () => { jsFn(ctx); });
});

describe("textureLoad(): one unfiltered texel", () => {
  const tex = uniform("sampler2D") as any;
  const build = () => Fn(() => textureLoad(tex, ivec2(3, 5)).x)();
  const wasmFn = compileWasm(build as any, { name: "main", params: [] });
  const jsFn = compileJS(build as any, { name: "main", params: [] });
  const ctx = { textures: { [tex.name]: texture } };

  bench("compileWasm", () => { wasmFn(ctx); });
  bench("compileJS", () => { jsFn(ctx); });
});

describe("texture(): bilinear-filtered sample", () => {
  const tex = uniform("sampler2D") as any;
  const build = () => Fn(() => tex.texture(vec2(0.3, 0.7)).x)();
  const wasmFn = compileWasm(build as any, { name: "main", params: [] });
  const jsFn = compileJS(build as any, { name: "main", params: [] });
  const ctx = { textures: { [tex.name]: texture } };

  bench("compileWasm", () => { wasmFn(ctx); });
  bench("compileJS", () => { jsFn(ctx); });
});
