import { bench, describe } from "vitest";
import { compileWasmRoutine } from "../wasm";
import { compileJSRoutine, type CpuTextureData } from "../js";
import { Fn, uniform, vec2, ivec2, textureSize, textureLoad } from "../rmsl";

const size = 8;
const data = new Float32Array(size * size * 4);
for (let i = 0; i < data.length; i++) data[i] = (i % 97) / 97;
const linearTexture: CpuTextureData = { data, width: size, height: size, magFilter: "linear" };
const nearestTexture: CpuTextureData = { data, width: size, height: size, magFilter: "nearest" };
const texture = linearTexture; // kept for the two scenarios below that don't care which filter mode

describe("textureSize(): metadata round trip only, no sampling math", () => {
  const tex = uniform("sampler2D") as any;
  const build = () => Fn(() => textureSize(tex).x.toFloat())();
  const wasmFn = compileWasmRoutine(build as any, { name: "main", params: [] });
  const jsFn = compileJSRoutine(build as any, { name: "main", params: [] });
  const ctx = { textures: { [tex.name]: texture } };

  bench("compileWasmRoutine", () => {
    wasmFn.run(ctx);
  });
  bench("compileJSRoutine", () => {
    jsFn.run(ctx);
  });
});

describe("textureLoad(): one unfiltered texel", () => {
  const tex = uniform("sampler2D") as any;
  const build = () => Fn(() => textureLoad(tex, ivec2(3, 5)).x)();
  const wasmFn = compileWasmRoutine(build as any, { name: "main", params: [] });
  const jsFn = compileJSRoutine(build as any, { name: "main", params: [] });
  const ctx = { textures: { [tex.name]: texture } };

  bench("compileWasmRoutine", () => {
    wasmFn.run(ctx);
  });
  bench("compileJSRoutine", () => {
    jsFn.run(ctx);
  });
});

describe("texture(): nearest-filtered sample", () => {
  const tex = uniform("sampler2D") as any;
  const build = () => Fn(() => tex.texture(vec2(0.3, 0.7)).x)();
  const wasmFn = compileWasmRoutine(build as any, { name: "main", params: [] });
  const jsFn = compileJSRoutine(build as any, { name: "main", params: [] });
  const ctx = { textures: { [tex.name]: nearestTexture } };

  bench("compileWasmRoutine", () => {
    wasmFn.run(ctx);
  });
  bench("compileJSRoutine", () => {
    jsFn.run(ctx);
  });
});

describe("texture(): bilinear-filtered sample", () => {
  const tex = uniform("sampler2D") as any;
  const build = () => Fn(() => tex.texture(vec2(0.3, 0.7)).x)();
  const wasmFn = compileWasmRoutine(build as any, { name: "main", params: [] });
  const jsFn = compileJSRoutine(build as any, { name: "main", params: [] });
  const ctx = { textures: { [tex.name]: linearTexture } };

  bench("compileWasmRoutine", () => {
    wasmFn.run(ctx);
  });
  bench("compileJSRoutine", () => {
    jsFn.run(ctx);
  });
});
