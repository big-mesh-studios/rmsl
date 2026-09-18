import { describe, it, expect } from "vitest";
import { compileWasmFn } from "../../wasm";
import { instantiateRasterizer } from "./rasterizer";
import { attribute, builtinPosition, Fn, vec4 } from "../../rmsl";

describe("WASM backend: generic rasterizer module (step 1 — linking skeleton)", () => {
  it("calls an imported vertex then an imported fragment module, sharing one memory", () => {
    const memory = new WebAssembly.Memory({ initial: 1 });
    const view = new DataView(memory.buffer);

    const vertexBuild = () =>
      Fn(() => {
        const pos = attribute("vec3");
        builtinPosition().assign(vec4(pos.x, pos.y, pos.z, 1));
      })();
    const vertexCompiled = compileWasmFn(vertexBuild as any, {
      name: "main",
      params: [],
      stage: "vertex",
      memory,
      memoryBase: 0,
    });

    const fragmentBuild = () => Fn(() => vec4(0, 0, 0, 0))();
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

    const attrSrcBase = 1024;
    const positionsOutBase = 2048;
    const vertices = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ];
    vertices.forEach((v, i) => v.forEach((c, j) => view.setFloat64(attrSrcBase + i * 24 + j * 8, c, true)));

    const attrDestAddress = vertexCompiled.params.find((p) => p.kind === "attributeMemory")!.address;
    const positionAddress = vertexCompiled.params.find((p) => p.kind === "positionMemory")!.address;

    rasterize(vertices.length, attrSrcBase, 24, attrDestAddress, positionAddress, positionsOutBase);

    for (let i = 0; i < vertices.length; i++) {
      const written = [0, 1, 2, 3].map((k) => view.getFloat64(positionsOutBase + i * 32 + k * 8, true));
      expect(written).toEqual([...vertices[i], 1]);
    }
  });
});
