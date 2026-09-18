import { describe, it, expect } from "vitest";
import { compileWasmFn } from "../../wasm";
import { instantiateRasterizer } from "./rasterizer";
import { builtinPosition, Fn, varying, vec3, vec4 } from "../../rmsl";

describe("WASM backend: generic rasterizer module (step 1 — linking skeleton)", () => {
  it("calls an imported vertex then an imported fragment module, sharing one memory", () => {
    const memory = new WebAssembly.Memory({ initial: 1 });
    const view = new DataView(memory.buffer);

    const vertexBuild = () =>
      Fn(() => {
        const v = varying("vec3");
        v.assign(vec3(1, 2, 3));
        builtinPosition().assign(vec4(10, 20, 30, 1));
      })();
    const vertexCompiled = compileWasmFn(vertexBuild as any, {
      name: "main",
      params: [],
      stage: "vertex",
      memory,
      memoryBase: 0,
    });

    const fragmentBuild = () => Fn(() => vec4(9, 8, 7, 6))();
    const fragmentCompiled = compileWasmFn(fragmentBuild as any, {
      name: "main",
      params: [],
      memoryBase: 512, // clear of the vertex module's own layout
      memory,
    });

    const vertexInstance = new WebAssembly.Instance(new WebAssembly.Module(vertexCompiled.bytes.buffer as ArrayBuffer), {
      math: Math as unknown as WebAssembly.ModuleImports,
      env: { memory },
    });
    const fragmentInstance = new WebAssembly.Instance(
      new WebAssembly.Module(fragmentCompiled.bytes.buffer as ArrayBuffer),
      { math: Math as unknown as WebAssembly.ModuleImports, env: { memory } },
    );

    const { rasterize } = instantiateRasterizer(
      vertexInstance.exports.main as () => void,
      fragmentInstance.exports.main as () => void,
      memory,
    );
    rasterize();

    const positionParam = vertexCompiled.params.find((p) => p.kind === "positionMemory")!;
    expect(positionParam).toBeDefined();
    const position = [0, 1, 2, 3].map((i) => view.getFloat64(positionParam.address + i * 8, true));
    expect(position).toEqual([10, 20, 30, 1]);

    const valueParam = fragmentCompiled.params.find((p) => p.kind === "valueMemory")!;
    expect(valueParam).toBeDefined();
    const value = [0, 1, 2, 3].map((i) => view.getFloat64(valueParam.address + i * 8, true));
    expect(value).toEqual([9, 8, 7, 6]);
  });
});
