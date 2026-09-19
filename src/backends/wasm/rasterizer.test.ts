import { describe, it, expect } from "vitest";
import { compileWasmFn } from "../../wasm";
import { compileJS, compileJSRoutine } from "../../js";
import { instantiateRasterizer, writeAttributeDescriptors, writeVaryingDescriptors } from "./rasterizer";
import { attribute, builtinPosition, Fn, uniform, varying, vec4, type AttributeNode } from "../../rmsl";

/** Clears a depth buffer so the first triangle over any pixel always passes the depth test. */
function clearDepthBuffer(view: DataView, base: number, pixelCount: number): void {
  for (let i = 0; i < pixelCount; i++) view.setFloat64(base + i * 8, Number.POSITIVE_INFINITY, true);
}

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

    const vertexInstance = new WebAssembly.Instance(
      new WebAssembly.Module(vertexCompiled.bytes.buffer as ArrayBuffer),
      {
        math: Math as unknown as WebAssembly.ModuleImports,
        env: { memory },
      },
    );
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
      4096,
      8192,
      12288,
      16384,
    );

    for (let i = 0; i < vertices.length; i++) {
      const written = [0, 1, 2, 3].map((k) => view.getFloat64(positionsOutBase + i * 32 + k * 8, true));
      expect(written).toEqual([...vertices[i], 1]);
    }
  });
});

describe("WASM backend: generic rasterizer module — triangle setup and edge functions", () => {
  it("matches compileJS's own (JsRasterRoutine-driven) output for one triangle", () => {
    // Declared once and referenced by both compiles below (not
    // re-generated per compile) so compileJS's own attributeTypes option,
    // which needs the slot name up front, stays valid.
    const posAttr = attribute("vec3");
    const vertexBuild = () =>
      Fn(() => {
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
    const attrSlot = posAttr.name;

    // Oracle: the same vertex/fragment programs run through compileJS's own
    // JsRasterRoutine, to check the WASM loop below against an independent
    // implementation of the same math rather than hand-derived pixel
    // coordinates.
    const jsRasterizer = compileJS(vertexBuild as any, fragmentBuild as any, {
      attributeTypes: { [attrSlot]: "vec3" },
    });
    const expected = jsRasterizer.draw(
      { attributes: { [attrSlot]: attrData } },
      { width, height, clear: true, clearDepth: true },
    );

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

    const vertexInstance = new WebAssembly.Instance(
      new WebAssembly.Module(vertexCompiled.bytes.buffer as ArrayBuffer),
      {
        math: Math as unknown as WebAssembly.ModuleImports,
        env: { memory },
      },
    );
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

    const depthBufferBase = 20480;
    clearDepthBuffer(view, depthBufferBase, width * height);

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
      8192,
      12288,
      16384,
      depthBufferBase,
    );

    const actual = new Float64Array(view.buffer, outputBase, width * height * 4);
    expect(Array.from(actual)).toEqual(Array.from(expected));
    // Sanity check the oracle itself found real coverage, not an empty triangle.
    expect(Array.from(expected).some((v) => v !== 0)).toBe(true);
  });
});

describe("WASM backend: generic rasterizer module — perspective-correct varying interpolation", () => {
  it("matches compileJS's own output for a per-vertex vec3 color varying", () => {
    // One attribute slot (this rasterizer's v1 scope): the varying is
    // derived from the position attribute itself, still exercising real
    // per-vertex-varying interpolation since each vertex's position differs.
    // posAttr is declared once and shared across both compiles below (not
    // re-generated per compile) so compileJS's attributeTypes option,
    // which needs the slot name up front, stays valid.
    const posAttr = attribute("vec3");
    const colorVarying = varying("vec3");
    const vertexBuild = () =>
      Fn(() => {
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
    const posSlot = posAttr.name;

    const jsRasterizer = compileJS(vertexBuild as any, fragmentBuild as any, {
      attributeTypes: { [posSlot]: "vec3" },
    });
    const expected = jsRasterizer.draw(
      { attributes: { [posSlot]: new Float64Array(positions.flat()) } },
      { width, height, clear: true, clearDepth: true },
    );
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

    const vertexInstance = new WebAssembly.Instance(
      new WebAssembly.Module(vertexCompiled.bytes.buffer as ArrayBuffer),
      {
        math: Math as unknown as WebAssembly.ModuleImports,
        env: { memory },
      },
    );
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
      {
        recordOffset: 0,
        vertexSrcAddress: vertexVaryingAddress,
        fragmentDestAddress: fragmentVaryingAddress,
        sizeBytes: 24,
      },
    ]);

    const depthBufferBase = 40960;
    clearDepthBuffer(view, depthBufferBase, width * height);

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
      24576,
      28672,
      32768,
      depthBufferBase,
    );

    const actual = new Float64Array(view.buffer, outputBase, width * height * 4);
    expect(Array.from(actual)).toEqual(Array.from(expected));
  });
});

describe("WASM backend: generic rasterizer module — multiple attribute slots", () => {
  it("matches compileJS's own output for two independently-addressed attributes", () => {
    // Declared once and shared across both compiles below (not
    // re-generated per compile) so compileJS's attributeTypes option,
    // which needs each slot name up front, stays valid.
    const posAttr = attribute("vec3");
    const colorAttr = attribute("vec3");
    const colorVarying = varying("vec3");
    const vertexBuild = () =>
      Fn(() => {
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
    const posSlot = posAttr.name;
    const colorSlot = colorAttr.name;

    const jsRasterizer = compileJS(vertexBuild as any, fragmentBuild as any, {
      attributeTypes: { [posSlot]: "vec3", [colorSlot]: "vec3" },
    });
    const expected = jsRasterizer.draw(
      { attributes: { [posSlot]: new Float64Array(positions.flat()), [colorSlot]: new Float64Array(colors.flat()) } },
      { width, height, clear: true, clearDepth: true },
    );
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
    const wasmPosSlot = posSlot;
    const wasmColorSlot = colorSlot;
    const fragmentCompiled = compileWasmFn(fragmentBuild as any, {
      name: "main",
      params: [],
      memoryBase: 1024,
      memory,
    });

    const vertexInstance = new WebAssembly.Instance(
      new WebAssembly.Module(vertexCompiled.bytes.buffer as ArrayBuffer),
      {
        math: Math as unknown as WebAssembly.ModuleImports,
        env: { memory },
      },
    );
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
      {
        recordOffset: 0,
        vertexSrcAddress: vertexVaryingAddress,
        fragmentDestAddress: fragmentVaryingAddress,
        sizeBytes: 24,
      },
    ]);

    const depthBufferBase = 40960;
    clearDepthBuffer(view, depthBufferBase, width * height);

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
      24576,
      28672,
      32768,
      depthBufferBase,
    );

    const actual = new Float64Array(view.buffer, outputBase, width * height * 4);
    expect(Array.from(actual)).toEqual(Array.from(expected));
  });
});

describe("WASM backend: generic rasterizer module — multiple varying slots", () => {
  it("matches compileJS's own output for two independently-addressed varyings", () => {
    // A scalar (non-aggregate) varying becomes a real WASM function
    // parameter even in an otherwise zero-arg program, which this
    // rasterizer's v1 scope doesn't support — both varyings here stay
    // vec3 to keep the program zero-arg/zero-return.
    // posAttr/colorAttr are declared once and shared across both compiles
    // below (not re-generated per compile) so compileJS's attributeTypes
    // option, which needs each slot name up front, stays valid.
    const posAttr = attribute("vec3");
    const colorAttr = attribute("vec3");
    const colorVarying = varying("vec3");
    const normalVarying = varying("vec3");
    const vertexBuild = () =>
      Fn(() => {
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
    const posSlot = posAttr.name;
    const colorSlot = colorAttr.name;

    const jsRasterizer = compileJS(vertexBuild as any, fragmentBuild as any, {
      attributeTypes: { [posSlot]: "vec3", [colorSlot]: "vec3" },
    });
    const expected = jsRasterizer.draw(
      { attributes: { [posSlot]: new Float64Array(positions.flat()), [colorSlot]: new Float64Array(colors.flat()) } },
      { width, height, clear: true, clearDepth: true },
    );
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
    const wasmPosSlot = posSlot;
    const wasmColorSlot = colorSlot;
    const fragmentCompiled = compileWasmFn(fragmentBuild as any, {
      name: "main",
      params: [],
      memoryBase: 1024,
      memory,
    });

    const vertexInstance = new WebAssembly.Instance(
      new WebAssembly.Module(vertexCompiled.bytes.buffer as ArrayBuffer),
      {
        math: Math as unknown as WebAssembly.ModuleImports,
        env: { memory },
      },
    );
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

    const depthBufferBase = 40960;
    clearDepthBuffer(view, depthBufferBase, width * height);

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
      24576,
      28672,
      32768,
      depthBufferBase,
    );

    const actual = new Float64Array(view.buffer, outputBase, width * height * 4);
    expect(Array.from(actual)).toEqual(Array.from(expected));
  });
});

describe("WASM backend: generic rasterizer module — scalarsInMemory for a scalar uniform", () => {
  it("matches compileJS's own output for a fragment program with a scalar uniform", () => {
    // posAttr declared once and shared across both compiles below (not
    // re-generated per compile) so compileJS's attributeTypes option,
    // which needs the slot name up front, stays valid.
    const posAttr = attribute("vec3");
    const brightness = uniform("float");
    const vertexBuild = () =>
      Fn(() => {
        builtinPosition().assign(vec4(posAttr.x, posAttr.y, posAttr.z, 1));
      })();
    const fragmentBuild = () => Fn(() => vec4(brightness, brightness, brightness, 1))();

    const width = 4;
    const height = 4;
    const positions = [
      [-1, -1, 0],
      [1, -1, 0],
      [-1, 1, 0],
    ];
    const posSlot = posAttr.name;

    const jsRasterizer = compileJS(vertexBuild as any, fragmentBuild as any, {
      attributeTypes: { [posSlot]: "vec3" },
    });
    const expected = jsRasterizer.draw(
      { attributes: { [posSlot]: new Float64Array(positions.flat()) }, uniforms: { [brightness.name]: 0.5 } },
      { width, height, clear: true, clearDepth: true },
    );
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
    const wasmPosSlot = posSlot;
    // The fragment program's only input is a scalar uniform, which would
    // otherwise compile to a real function argument — LinkError against
    // the rasterizer's fixed zero-arg import without scalarsInMemory.
    const fragmentCompiled = compileWasmFn(fragmentBuild as any, {
      name: "main",
      params: [],
      memoryBase: 1024,
      memory,
      scalarsInMemory: true,
    });
    const brightnessAddress = fragmentCompiled.params.find((p) => p.kind === "uniformMemory")!.address;
    new DataView(memory.buffer).setFloat64(brightnessAddress, 0.5, true);

    const vertexInstance = new WebAssembly.Instance(
      new WebAssembly.Module(vertexCompiled.bytes.buffer as ArrayBuffer),
      {
        math: Math as unknown as WebAssembly.ModuleImports,
        env: { memory },
      },
    );
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
    const outputBase = 16384;
    const attrDescBase = 1536;
    const attrData = new Float64Array(positions.flat());
    attrData.forEach((c, i) => view.setFloat64(attrSrcBase + i * 8, c, true));

    const attrDestAddress = vertexCompiled.params.find((p) => p.kind === "attributeMemory")!.address;
    writeAttributeDescriptors(view, attrDescBase, [{ srcOffset: 0, destAddress: attrDestAddress, sizeBytes: 24 }]);
    const positionAddress = vertexCompiled.params.find((p) => p.kind === "positionMemory")!.address;
    const fragmentValueAddress = fragmentCompiled.params.find((p) => p.kind === "valueMemory")!.address;
    const depthBufferBase = 40960;
    clearDepthBuffer(view, depthBufferBase, width * height);

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
      0,
      0,
      0,
      0,
      24576,
      28672,
      32768,
      depthBufferBase,
    );

    // rasterize() re-reads brightness fresh from its own memory address
    // every fragment call (no host round-trip needed), so writing it once
    // up front — matching what a real caller would do before a draw — is
    // enough for every covered pixel.
    const actual = new Float64Array(view.buffer, outputBase, width * height * 4);
    expect(Array.from(actual)).toEqual(Array.from(expected));
  });
});

describe("WASM backend: generic rasterizer module — near-plane clipping", () => {
  it("clips a triangle with one vertex behind the eye into two triangles, matching an independent JS clip+raster reference", () => {
    let posAttr!: AttributeNode<"vec3">;
    const vertexBuild = () =>
      Fn(() => {
        posAttr = attribute("vec3");
        builtinPosition().assign(vec4(posAttr.x, posAttr.y, posAttr.z, posAttr.z));
      })();
    const fragmentBuild = () => Fn(() => vec4(9, 8, 7, 6))();

    const width = 8;
    const height = 8;
    // w = z per the vertex program above: vertex 0 sits behind the eye (w < 0).
    const positions = [
      [-1, -1, -1],
      [1, -1, 2],
      [-1, 1, 2],
    ];

    const jsVertex = compileJSRoutine(vertexBuild as any, { name: "vertex", params: [], stage: "vertex" });
    const clipSpacePositions = positions.map(
      (p) => (jsVertex.invoke({ attributes: { [posAttr.name]: p } }) as { position: number[] }).position,
    );

    const EPS = 1e-5;
    const clipAgainstW = (poly: number[][]): number[][] => {
      const out: number[][] = [];
      for (let i = 0; i < poly.length; i++) {
        const a = poly[i];
        const b = poly[(i + 1) % poly.length];
        const aIn = a[3] > EPS;
        const bIn = b[3] > EPS;
        if (aIn) out.push(a);
        if (aIn !== bIn) {
          const t = (EPS - a[3]) / (b[3] - a[3]);
          out.push(a.map((av, k) => av + (b[k] - av) * t));
        }
      }
      return out;
    };
    const fanTriangulate = (poly: number[][]): [number[], number[], number[]][] => {
      const tris: [number[], number[], number[]][] = [];
      for (let i = 1; i + 1 < poly.length; i++) tris.push([poly[0], poly[i], poly[i + 1]]);
      return tris;
    };
    const rasterizeTriangleInto = (out: Float64Array, p0: number[], p1: number[], p2: number[], color: number[]) => {
      const screen = (p: number[]) => [((p[0] / p[3]) * 0.5 + 0.5) * width, (1 - ((p[1] / p[3]) * 0.5 + 0.5)) * height];
      const s0 = screen(p0);
      const s1 = screen(p1);
      const s2 = screen(p2);
      const area = (s1[0] - s0[0]) * (s2[1] - s0[1]) - (s1[1] - s0[1]) * (s2[0] - s0[0]);
      if (area === 0) return;
      const minX = Math.max(0, Math.floor(Math.min(s0[0], s1[0], s2[0])));
      const maxX = Math.min(width - 1, Math.ceil(Math.max(s0[0], s1[0], s2[0])));
      const minY = Math.max(0, Math.floor(Math.min(s0[1], s1[1], s2[1])));
      const maxY = Math.min(height - 1, Math.ceil(Math.max(s0[1], s1[1], s2[1])));
      for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
          const px = x + 0.5;
          const py = y + 0.5;
          const e0 = (s1[0] - px) * (s2[1] - py) - (s1[1] - py) * (s2[0] - px);
          const e1 = (s2[0] - px) * (s0[1] - py) - (s2[1] - py) * (s0[0] - px);
          const e2 = (s0[0] - px) * (s1[1] - py) - (s0[1] - py) * (s1[0] - px);
          const inside = (e0 >= 0 && e1 >= 0 && e2 >= 0) || (e0 <= 0 && e1 <= 0 && e2 <= 0);
          if (!inside) continue;
          const base = (y * width + x) * 4;
          for (let c = 0; c < 4; c++) out[base + c] = color[c];
        }
      }
    };

    const expected = new Float64Array(width * height * 4);
    const clippedPoly = clipAgainstW(clipSpacePositions);
    expect(clippedPoly.length).toBe(4); // one vertex out, two in -> a quad
    for (const [a, b, c] of fanTriangulate(clippedPoly)) rasterizeTriangleInto(expected, a, b, c, [9, 8, 7, 6]);
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

    const vertexInstance = new WebAssembly.Instance(
      new WebAssembly.Module(vertexCompiled.bytes.buffer as ArrayBuffer),
      {
        math: Math as unknown as WebAssembly.ModuleImports,
        env: { memory },
      },
    );
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
    const outputBase = 16384;
    const attrDescBase = 1536;
    const clipScratchBase = 24576;
    const clippedPositionsOutBase = 28672;
    const clippedVaryingsOutBase = 32768;
    positions.flat().forEach((c, i) => view.setFloat64(attrSrcBase + i * 8, c, true));

    const attrDestAddress = vertexCompiled.params.find((p) => p.kind === "attributeMemory")!.address;
    writeAttributeDescriptors(view, attrDescBase, [{ srcOffset: 0, destAddress: attrDestAddress, sizeBytes: 24 }]);
    const positionAddress = vertexCompiled.params.find((p) => p.kind === "positionMemory")!.address;
    const fragmentValueAddress = fragmentCompiled.params.find((p) => p.kind === "valueMemory")!.address;
    const depthBufferBase = 40960;
    clearDepthBuffer(view, depthBufferBase, width * height);

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
      0,
      0,
      0,
      0,
      clipScratchBase,
      clippedPositionsOutBase,
      clippedVaryingsOutBase,
      depthBufferBase,
    );

    const actual = new Float64Array(view.buffer, outputBase, width * height * 4);
    expect(Array.from(actual)).toEqual(Array.from(expected));
  });

  it("culls a triangle entirely behind the eye", () => {
    let posAttr!: AttributeNode<"vec3">;
    const vertexBuild = () =>
      Fn(() => {
        posAttr = attribute("vec3");
        builtinPosition().assign(vec4(posAttr.x, posAttr.y, posAttr.z, posAttr.z));
      })();
    const fragmentBuild = () => Fn(() => vec4(9, 8, 7, 6))();

    const width = 8;
    const height = 8;
    // Every vertex has w = z < 0: the whole triangle is behind the eye.
    const positions = [
      [-1, -1, -1],
      [1, -1, -2],
      [-1, 1, -2],
    ];

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

    const vertexInstance = new WebAssembly.Instance(
      new WebAssembly.Module(vertexCompiled.bytes.buffer as ArrayBuffer),
      {
        math: Math as unknown as WebAssembly.ModuleImports,
        env: { memory },
      },
    );
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
    const outputBase = 16384;
    const attrDescBase = 1536;
    positions.flat().forEach((c, i) => view.setFloat64(attrSrcBase + i * 8, c, true));

    const attrDestAddress = vertexCompiled.params.find((p) => p.kind === "attributeMemory")!.address;
    writeAttributeDescriptors(view, attrDescBase, [{ srcOffset: 0, destAddress: attrDestAddress, sizeBytes: 24 }]);
    const positionAddress = vertexCompiled.params.find((p) => p.kind === "positionMemory")!.address;
    const fragmentValueAddress = fragmentCompiled.params.find((p) => p.kind === "valueMemory")!.address;
    const depthBufferBase = 40960;
    clearDepthBuffer(view, depthBufferBase, width * height);

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
      0,
      0,
      0,
      0,
      24576,
      28672,
      32768,
      depthBufferBase,
    );

    const actual = new Float64Array(view.buffer, outputBase, width * height * 4);
    expect(Array.from(actual).every((v) => v === 0)).toBe(true);
  });
});

describe("WASM backend: generic rasterizer module — depth test", () => {
  it("keeps the closer triangle's color regardless of draw order", () => {
    let posAttr!: AttributeNode<"vec3">;
    const vertexBuild = () =>
      Fn(() => {
        posAttr = attribute("vec3");
        builtinPosition().assign(vec4(posAttr.x, posAttr.y, posAttr.z, 1));
      })();
    const redFragmentBuild = () => Fn(() => vec4(1, 0, 0, 1))();
    const blueFragmentBuild = () => Fn(() => vec4(0, 0, 1, 1))();

    const width = 4;
    const height = 4;
    // A triangle large enough to fully enclose the [-1, 1] NDC viewport
    // (a single triangle can cover at most half a square along a diagonal
    // otherwise) at a given depth (z).
    const triangleAt = (z: number) => [
      [-10, -10, z],
      [10, -10, z],
      [0, 10, z],
    ];
    const farTriangle = triangleAt(0.5);
    const nearTriangle = triangleAt(-0.5); // smaller z = closer, per this rasterizer's depth convention

    function draw(order: "far-then-near" | "near-then-far"): Float64Array {
      const memory = new WebAssembly.Memory({ initial: 1 });
      const view = new DataView(memory.buffer);

      const vertexCompiled = compileWasmFn(vertexBuild as any, {
        name: "main",
        params: [],
        stage: "vertex",
        memory,
        memoryBase: 0,
      });
      const redCompiled = compileWasmFn(redFragmentBuild as any, {
        name: "main",
        params: [],
        memoryBase: 1024,
        memory,
      });
      const blueCompiled = compileWasmFn(blueFragmentBuild as any, {
        name: "main",
        params: [],
        memoryBase: 2048,
        memory,
      });

      const vertexInstance = new WebAssembly.Instance(
        new WebAssembly.Module(vertexCompiled.bytes.buffer as ArrayBuffer),
        {
          math: Math as unknown as WebAssembly.ModuleImports,
          env: { memory },
        },
      );
      const redInstance = new WebAssembly.Instance(new WebAssembly.Module(redCompiled.bytes.buffer as ArrayBuffer), {
        math: Math as unknown as WebAssembly.ModuleImports,
        env: { memory },
      });
      const blueInstance = new WebAssembly.Instance(new WebAssembly.Module(blueCompiled.bytes.buffer as ArrayBuffer), {
        math: Math as unknown as WebAssembly.ModuleImports,
        env: { memory },
      });
      const { rasterize: rasterizeRed } = instantiateRasterizer(
        vertexInstance.exports.main as () => void,
        redInstance.exports.main as () => void,
        memory,
      );
      const { rasterize: rasterizeBlue } = instantiateRasterizer(
        vertexInstance.exports.main as () => void,
        blueInstance.exports.main as () => void,
        memory,
      );

      const attrSrcBase = 4096;
      const positionsOutBase = 8192;
      const outputBase = 16384;
      const attrDescBase = 3072;
      const clipScratchBase = 24576;
      const clippedPositionsOutBase = 28672;
      const clippedVaryingsOutBase = 32768;
      const depthBufferBase = 40960;

      const attrDestAddress = vertexCompiled.params.find((p) => p.kind === "attributeMemory")!.address;
      writeAttributeDescriptors(view, attrDescBase, [{ srcOffset: 0, destAddress: attrDestAddress, sizeBytes: 24 }]);
      const positionAddress = vertexCompiled.params.find((p) => p.kind === "positionMemory")!.address;
      const redValueAddress = redCompiled.params.find((p) => p.kind === "valueMemory")!.address;
      const blueValueAddress = blueCompiled.params.find((p) => p.kind === "valueMemory")!.address;
      clearDepthBuffer(view, depthBufferBase, width * height);

      const drawOne = (triangle: number[][], rasterize: (...args: number[]) => void, fragmentValueAddress: number) => {
        triangle.flat().forEach((c, i) => view.setFloat64(attrSrcBase + i * 8, c, true));
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
          0,
          0,
          0,
          0,
          clipScratchBase,
          clippedPositionsOutBase,
          clippedVaryingsOutBase,
          depthBufferBase,
        );
      };

      if (order === "far-then-near") {
        drawOne(farTriangle, rasterizeRed, redValueAddress);
        drawOne(nearTriangle, rasterizeBlue, blueValueAddress);
      } else {
        drawOne(nearTriangle, rasterizeBlue, blueValueAddress);
        drawOne(farTriangle, rasterizeRed, redValueAddress);
      }

      return new Float64Array(view.buffer, outputBase, width * height * 4).slice();
    }

    const farThenNear = draw("far-then-near");
    const nearThenFar = draw("near-then-far");

    // The near (blue) triangle must win in both draw orders.
    for (let i = 0; i < width * height; i++) {
      expect(Array.from(farThenNear.slice(i * 4, i * 4 + 4))).toEqual([0, 0, 1, 1]);
    }
    expect(Array.from(nearThenFar)).toEqual(Array.from(farThenNear));
  });
});
