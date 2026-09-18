import { describe, it, expect } from "vitest";
import { compileWasmFn } from "../../wasm";
import { compileJS } from "../../js";
import { rasterizeTriangles } from "../cpu-rasterizer";
import { instantiateRasterizer, writeAttributeDescriptors, writeVaryingDescriptors } from "./rasterizer";
import { attribute, builtinPosition, Fn, varying, vec4, type AttributeNode } from "../../rmsl";

describe("WASM backend: generic rasterizer module — linking skeleton", () => {
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
    const attrDescBase = 900;
    writeAttributeDescriptors(view, attrDescBase, [{ srcOffset: 0, destAddress: attrDestAddress, sizeBytes: 24 }]);

    rasterize(
      vertices.length,
      attrSrcBase,
      24,
      attrDescBase,
      1,
      positionAddress,
      positionsOutBase,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
    );

    for (let i = 0; i < vertices.length; i++) {
      const written = [0, 1, 2, 3].map((k) => view.getFloat64(positionsOutBase + i * 32 + k * 8, true));
      expect(written).toEqual([...vertices[i], 1]);
    }
  });
});

describe("WASM backend: generic rasterizer module — triangle setup and edge functions", () => {
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
    const attrDescBase = 900;
    writeAttributeDescriptors(view, attrDescBase, [{ srcOffset: 0, destAddress: attrDestAddress, sizeBytes: 24 }]);

    rasterize(
      triangle.length,
      attrSrcBase,
      24,
      attrDescBase,
      1,
      positionAddress,
      positionsOutBase,
      width,
      height,
      fragmentValueAddress,
      outputBase,
      0,
      0,
      0,
      0,
    );

    const actual = new Float64Array(view.buffer, outputBase, width * height * 4);
    expect(Array.from(actual)).toEqual(Array.from(expected));
    // Sanity check the oracle itself found real coverage, not an empty triangle.
    expect(Array.from(expected).some((v) => v !== 0)).toBe(true);
  });
});

describe("WASM backend: generic rasterizer module — perspective-correct varying interpolation", () => {
  it("matches rasterizeTriangles' own output for a per-vertex vec3 color varying", () => {
    // One attribute slot (this rasterizer's v1 scope): the varying is
    // derived from the position attribute itself, still exercising real
    // per-vertex-varying interpolation since each vertex's position differs.
    let posAttr!: AttributeNode<"vec3">;
    const colorVarying = varying("vec3");
    const vertexBuild = () =>
      Fn(() => {
        posAttr = attribute("vec3");
        colorVarying.assign(posAttr.mul(0.5).add(0.5));
        builtinPosition().assign(vec4(posAttr.x, posAttr.y, posAttr.z, 1));
      })();
    const fragmentBuild = () => Fn(() => vec4(colorVarying.x, colorVarying.y, colorVarying.z, 1))();

    const width = 4;
    const height = 4;
    const positions = [
      [-1, -1, 0],
      [1, -1, 0],
      [-1, 1, 0],
    ];

    const jsVertex = compileJS(vertexBuild as any, { name: "vertex", params: [], stage: "vertex" });
    const posSlot = posAttr.name;
    const jsFragment = compileJS(fragmentBuild as any, { name: "fragment", params: [] });
    const expected = rasterizeTriangles(jsVertex, jsFragment, {
      attributes: { [posSlot]: new Float64Array(positions.flat()) },
      attributeTypes: { [posSlot]: "vec3" },
      width,
      height,
      componentCount: 4,
    });
    expect(Array.from(expected).some((v) => v !== 0)).toBe(true);

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
      memoryBase: 1024,
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

    const attrSrcBase = 2048;
    const positionsOutBase = 4096;
    const varyingsOutBase = 8192;
    const outputBase = 16384;
    const attrData = new Float64Array(positions.flat());
    attrData.forEach((c, i) => view.setFloat64(attrSrcBase + i * 8, c, true));

    const attrDestAddress = vertexCompiled.params.find((p) => p.kind === "attributeMemory")!.address;
    const positionAddress = vertexCompiled.params.find((p) => p.kind === "positionMemory")!.address;
    const fragmentValueAddress = fragmentCompiled.params.find((p) => p.kind === "valueMemory")!.address;
    const vertexVaryingAddress = vertexCompiled.params.find((p) => p.kind === "varyingOutputMemory")!.address;
    const fragmentVaryingAddress = fragmentCompiled.params.find((p) => p.kind === "varyingMemory")!.address;
    const attrDescBase = 1536;
    writeAttributeDescriptors(view, attrDescBase, [{ srcOffset: 0, destAddress: attrDestAddress, sizeBytes: 24 }]);
    const varyingDescBase = 1600;
    writeVaryingDescriptors(view, varyingDescBase, [
      { recordOffset: 0, vertexSrcAddress: vertexVaryingAddress, fragmentDestAddress: fragmentVaryingAddress, sizeBytes: 24 },
    ]);

    rasterize(
      3,
      attrSrcBase,
      24,
      attrDescBase,
      1,
      positionAddress,
      positionsOutBase,
      width,
      height,
      fragmentValueAddress,
      outputBase,
      24,
      varyingDescBase,
      1,
      varyingsOutBase,
    );

    const actual = new Float64Array(view.buffer, outputBase, width * height * 4);
    expect(Array.from(actual)).toEqual(Array.from(expected));
  });
});

describe("WASM backend: generic rasterizer module — multiple attribute slots", () => {
  it("matches rasterizeTriangles' own output for two independently-addressed attributes", () => {
    let posAttr!: AttributeNode<"vec3">;
    let colorAttr!: AttributeNode<"vec3">;
    const colorVarying = varying("vec3");
    const vertexBuild = () =>
      Fn(() => {
        posAttr = attribute("vec3");
        colorAttr = attribute("vec3");
        colorVarying.assign(colorAttr);
        builtinPosition().assign(vec4(posAttr.x, posAttr.y, posAttr.z, 1));
      })();
    const fragmentBuild = () => Fn(() => vec4(colorVarying.x, colorVarying.y, colorVarying.z, 1))();

    const width = 4;
    const height = 4;
    const positions = [
      [-1, -1, 0],
      [1, -1, 0],
      [-1, 1, 0],
    ];
    const colors = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];

    const jsVertex = compileJS(vertexBuild as any, { name: "vertex", params: [], stage: "vertex" });
    const posSlot = posAttr.name;
    const colorSlot = colorAttr.name;
    const jsFragment = compileJS(fragmentBuild as any, { name: "fragment", params: [] });
    const expected = rasterizeTriangles(jsVertex, jsFragment, {
      attributes: { [posSlot]: new Float64Array(positions.flat()), [colorSlot]: new Float64Array(colors.flat()) },
      attributeTypes: { [posSlot]: "vec3", [colorSlot]: "vec3" },
      width,
      height,
      componentCount: 4,
    });
    expect(Array.from(expected).some((v) => v !== 0)).toBe(true);

    const memory = new WebAssembly.Memory({ initial: 1 });
    const view = new DataView(memory.buffer);

    const vertexCompiled = compileWasmFn(vertexBuild as any, {
      name: "main",
      params: [],
      stage: "vertex",
      memory,
      memoryBase: 0,
    });
    // vertexBuild() ran again for this compile, so posAttr/colorAttr now
    // hold this compile's own (freshly re-generated) attribute nodes.
    const wasmPosSlot = posAttr.name;
    const wasmColorSlot = colorAttr.name;
    const fragmentCompiled = compileWasmFn(fragmentBuild as any, {
      name: "main",
      params: [],
      memoryBase: 1024,
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

    const attrSrcBase = 2048;
    const positionsOutBase = 4096;
    const varyingsOutBase = 8192;
    const outputBase = 16384;
    const attrDescBase = 1536;
    for (let i = 0; i < 3; i++) {
      positions[i].forEach((c, j) => view.setFloat64(attrSrcBase + i * 48 + j * 8, c, true));
      colors[i].forEach((c, j) => view.setFloat64(attrSrcBase + i * 48 + 24 + j * 8, c, true));
    }

    const attributeMemoryParams = vertexCompiled.params.filter((p) => p.kind === "attributeMemory");
    const posDestAddress = attributeMemoryParams.find((p) => p.slot === wasmPosSlot)!.address;
    const colorDestAddress = attributeMemoryParams.find((p) => p.slot === wasmColorSlot)!.address;
    writeAttributeDescriptors(view, attrDescBase, [
      { srcOffset: 0, destAddress: posDestAddress, sizeBytes: 24 },
      { srcOffset: 24, destAddress: colorDestAddress, sizeBytes: 24 },
    ]);

    const positionAddress = vertexCompiled.params.find((p) => p.kind === "positionMemory")!.address;
    const fragmentValueAddress = fragmentCompiled.params.find((p) => p.kind === "valueMemory")!.address;
    const vertexVaryingAddress = vertexCompiled.params.find((p) => p.kind === "varyingOutputMemory")!.address;
    const fragmentVaryingAddress = fragmentCompiled.params.find((p) => p.kind === "varyingMemory")!.address;
    const varyingDescBase = 1728;
    writeVaryingDescriptors(view, varyingDescBase, [
      { recordOffset: 0, vertexSrcAddress: vertexVaryingAddress, fragmentDestAddress: fragmentVaryingAddress, sizeBytes: 24 },
    ]);

    rasterize(
      3,
      attrSrcBase,
      48,
      attrDescBase,
      2,
      positionAddress,
      positionsOutBase,
      width,
      height,
      fragmentValueAddress,
      outputBase,
      24,
      varyingDescBase,
      1,
      varyingsOutBase,
    );

    const actual = new Float64Array(view.buffer, outputBase, width * height * 4);
    expect(Array.from(actual)).toEqual(Array.from(expected));
  });
});

describe("WASM backend: generic rasterizer module — multiple varying slots", () => {
  it("matches rasterizeTriangles' own output for two independently-addressed varyings", () => {
    // A scalar (non-aggregate) varying becomes a real WASM function
    // parameter even in an otherwise zero-arg program, which this
    // rasterizer's v1 scope doesn't support — both varyings here stay
    // vec3 to keep the program zero-arg/zero-return.
    let posAttr!: AttributeNode<"vec3">;
    let colorAttr!: AttributeNode<"vec3">;
    const colorVarying = varying("vec3");
    const normalVarying = varying("vec3");
    const vertexBuild = () =>
      Fn(() => {
        posAttr = attribute("vec3");
        colorAttr = attribute("vec3");
        colorVarying.assign(colorAttr);
        normalVarying.assign(posAttr.mul(0.5).add(0.5));
        builtinPosition().assign(vec4(posAttr.x, posAttr.y, posAttr.z, 1));
      })();
    const fragmentBuild = () => Fn(() => vec4(colorVarying.x, colorVarying.y, normalVarying.z, 1))();

    const width = 4;
    const height = 4;
    const positions = [
      [-1, -1, 0],
      [1, -1, 0],
      [-1, 1, 0],
    ];
    const colors = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];

    const jsVertex = compileJS(vertexBuild as any, { name: "vertex", params: [], stage: "vertex" });
    const posSlot = posAttr.name;
    const colorSlot = colorAttr.name;
    const jsFragment = compileJS(fragmentBuild as any, { name: "fragment", params: [] });
    const expected = rasterizeTriangles(jsVertex, jsFragment, {
      attributes: { [posSlot]: new Float64Array(positions.flat()), [colorSlot]: new Float64Array(colors.flat()) },
      attributeTypes: { [posSlot]: "vec3", [colorSlot]: "vec3" },
      width,
      height,
      componentCount: 4,
    });
    expect(Array.from(expected).some((v) => v !== 0)).toBe(true);

    const memory = new WebAssembly.Memory({ initial: 1 });
    const view = new DataView(memory.buffer);

    const vertexCompiled = compileWasmFn(vertexBuild as any, {
      name: "main",
      params: [],
      stage: "vertex",
      memory,
      memoryBase: 0,
    });
    const wasmPosSlot = posAttr.name;
    const wasmColorSlot = colorAttr.name;
    const fragmentCompiled = compileWasmFn(fragmentBuild as any, {
      name: "main",
      params: [],
      memoryBase: 1024,
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

    const attrSrcBase = 2048;
    const positionsOutBase = 4096;
    const varyingsOutBase = 8192;
    const outputBase = 16384;
    const attrDescBase = 1536;
    const varyingDescBase = 1728;
    for (let i = 0; i < 3; i++) {
      positions[i].forEach((c, j) => view.setFloat64(attrSrcBase + i * 48 + j * 8, c, true));
      colors[i].forEach((c, j) => view.setFloat64(attrSrcBase + i * 48 + 24 + j * 8, c, true));
    }

    const attributeMemoryParams = vertexCompiled.params.filter((p) => p.kind === "attributeMemory");
    const posDestAddress = attributeMemoryParams.find((p) => p.slot === wasmPosSlot)!.address;
    const colorDestAddress = attributeMemoryParams.find((p) => p.slot === wasmColorSlot)!.address;
    writeAttributeDescriptors(view, attrDescBase, [
      { srcOffset: 0, destAddress: posDestAddress, sizeBytes: 24 },
      { srcOffset: 24, destAddress: colorDestAddress, sizeBytes: 24 },
    ]);

    const positionAddress = vertexCompiled.params.find((p) => p.kind === "positionMemory")!.address;
    const fragmentValueAddress = fragmentCompiled.params.find((p) => p.kind === "valueMemory")!.address;
    const vertexVaryingParams = vertexCompiled.params.filter((p) => p.kind === "varyingOutputMemory");
    const fragmentVaryingParams = fragmentCompiled.params.filter((p) => p.kind === "varyingMemory");
    const colorVertexVaryingAddress = vertexVaryingParams.find((p) => p.slot === colorVarying.name)!.address;
    const normalVertexVaryingAddress = vertexVaryingParams.find((p) => p.slot === normalVarying.name)!.address;
    const colorFragmentVaryingAddress = fragmentVaryingParams.find((p) => p.slot === colorVarying.name)!.address;
    const normalFragmentVaryingAddress = fragmentVaryingParams.find((p) => p.slot === normalVarying.name)!.address;
    writeVaryingDescriptors(view, varyingDescBase, [
      {
        recordOffset: 0,
        vertexSrcAddress: colorVertexVaryingAddress,
        fragmentDestAddress: colorFragmentVaryingAddress,
        sizeBytes: 24,
      },
      {
        recordOffset: 24,
        vertexSrcAddress: normalVertexVaryingAddress,
        fragmentDestAddress: normalFragmentVaryingAddress,
        sizeBytes: 24,
      },
    ]);

    rasterize(
      3,
      attrSrcBase,
      48,
      attrDescBase,
      2,
      positionAddress,
      positionsOutBase,
      width,
      height,
      fragmentValueAddress,
      outputBase,
      48,
      varyingDescBase,
      2,
      varyingsOutBase,
    );

    const actual = new Float64Array(view.buffer, outputBase, width * height * 4);
    expect(Array.from(actual)).toEqual(Array.from(expected));
  });
});
