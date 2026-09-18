import { describe, it, expect } from "vitest";
import { compileWasmFn } from "../../wasm";
import { compileJS } from "../../js";
import { rasterizeTriangles } from "../cpu-rasterizer";
import { instantiateRasterizer } from "./rasterizer";
import { attribute, builtinPosition, Fn, vec4, type AttributeNode } from "../../rmsl";

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

    rasterize(vertices.length, attrSrcBase, 24, attrDestAddress, positionAddress, positionsOutBase, 0, 0, 0, 0);

    for (let i = 0; i < vertices.length; i++) {
      const written = [0, 1, 2, 3].map((k) => view.getFloat64(positionsOutBase + i * 32 + k * 8, true));
      expect(written).toEqual([...vertices[i], 1]);
    }
  });
});

describe("WASM backend: generic rasterizer module (step 2 — triangle setup and edge functions)", () => {
  it("matches rasterizeTriangles' own (compileJS-driven) output for one triangle", () => {
    let posAttr!: AttributeNode<"vec3">;
    const vertexBuild = () =>
      Fn(() => {
        posAttr = attribute("vec3");
        builtinPosition().assign(vec4(posAttr.x, posAttr.y, posAttr.z, 1));
      })();
    const fragmentBuild = () => Fn(() => vec4(5, 6, 7, 8))();

    const width = 4;
    const height = 4;
    const triangle = [
      [-1, -1, 0],
      [1, -1, 0],
      [-1, 1, 0],
    ];
    const attrData = new Float64Array(triangle.flat());

    // Oracle: the same vertex/fragment programs run through compileJS and the
    // already-tested rasterizeTriangles, to check the WASM loop below against
    // an independent implementation of the same math rather than hand-derived
    // pixel coordinates.
    const jsVertex = compileJS(vertexBuild as any, { name: "vertex", params: [], stage: "vertex" });
    const attrSlot = posAttr.name;
    const jsFragment = compileJS(fragmentBuild as any, { name: "fragment", params: [] });
    const expected = rasterizeTriangles(jsVertex, jsFragment, {
      attributes: { [attrSlot]: attrData },
      attributeTypes: { [attrSlot]: "vec3" },
      width,
      height,
      componentCount: 4,
    });

    const memory = new WebAssembly.Memory({ initial: 1 });
    const view = new DataView(memory.buffer);

    const vertexCompiled = compileWasmFn(vertexBuild as any, {
      name: "main",
      params: [],
      stage: "vertex",
      memory,
      memoryBase: 0,
    });
    const fragmentCompiled = compileWasmFn(fragmentBuild as any, {
      name: "main",
      params: [],
      memoryBase: 512,
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
    const outputBase = 4096;
    attrData.forEach((c, i) => view.setFloat64(attrSrcBase + i * 8, c, true));

    const attrDestAddress = vertexCompiled.params.find((p) => p.kind === "attributeMemory")!.address;
    const positionAddress = vertexCompiled.params.find((p) => p.kind === "positionMemory")!.address;
    const fragmentValueAddress = fragmentCompiled.params.find((p) => p.kind === "valueMemory")!.address;

    rasterize(
      triangle.length,
      attrSrcBase,
      24,
      attrDestAddress,
      positionAddress,
      positionsOutBase,
      width,
      height,
      fragmentValueAddress,
      outputBase,
    );

    const actual = new Float64Array(view.buffer, outputBase, width * height * 4);
    expect(Array.from(actual)).toEqual(Array.from(expected));
    // Sanity check the oracle itself found real coverage, not an empty triangle.
    expect(Array.from(expected).some((v) => v !== 0)).toBe(true);
  });
});
