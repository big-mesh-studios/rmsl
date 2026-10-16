import { describe, it, expect } from "vitest";
import { compileWasmFn } from "../../wasm";
import { compileJS, compileJSRoutine } from "../../js";
import { LINES_BYTES, instantiateRasterizer, writeAttributeDescriptors, writeVaryingDescriptors } from "./rasterizer";
import { attribute, builtinPosition, Fn, uniform, varying, vec4, type AttributeNode } from "../../rmsl";

/** Clears a depth buffer so the first triangle over any pixel always passes the depth test. */
function clearDepthBuffer(view: DataView, base: number, pixelCount: number): void {
  for (let i = 0; i < pixelCount; i++) view.setFloat64(base + i * 8, Number.POSITIVE_INFINITY, true);
}

describe("WASM backend: generic rasterizer module — linking skeleton", () => {
  /**
   * @canon spec-the-wasm-rasterizer-links-its-stages-in-one-memory
   */
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
      false,
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
  /**
   * @canon spec-the-wasm-rasterizer-draws-what-the-js-rasterizer-draws
   */
  it("matches compileJS's own (JsRasterRoutine-driven) output for one triangle", () => {
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
    const jsRasterizer = compileJS(vertexBuild as any, fragmentBuild as any);
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
      false,
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
      1,
    );

    const actual = new Float64Array(view.buffer, outputBase, width * height * 4);
    expect(Array.from(actual)).toEqual(Array.from(expected));
    // Sanity check the oracle itself found real coverage, not an empty triangle.
    expect(Array.from(expected).some((v) => v !== 0)).toBe(true);
  });
});

describe("WASM backend: generic rasterizer module — perspective-correct varying interpolation", () => {
  /**
   * @canon spec-the-wasm-rasterizer-draws-what-the-js-rasterizer-draws
   */
  it("matches compileJS's own output for a per-vertex vec3 color varying", () => {
    // One attribute slot (this rasterizer's v1 scope): the varying is
    // derived from the position attribute itself, still exercising real
    // per-vertex-varying interpolation since each vertex's position differs.
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

    const jsRasterizer = compileJS(vertexBuild as any, fragmentBuild as any);
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
      false,
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
      1,
    );

    const actual = new Float64Array(view.buffer, outputBase, width * height * 4);
    expect(Array.from(actual)).toEqual(Array.from(expected));
  });
});

describe("WASM backend: generic rasterizer module — multiple attribute slots", () => {
  /**
   * @canon spec-the-wasm-rasterizer-draws-what-the-js-rasterizer-draws
   */
  it("matches compileJS's own output for two independently-addressed attributes", () => {
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

    const jsRasterizer = compileJS(vertexBuild as any, fragmentBuild as any);
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
      false,
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
      1,
    );

    const actual = new Float64Array(view.buffer, outputBase, width * height * 4);
    expect(Array.from(actual)).toEqual(Array.from(expected));
  });
});

describe("WASM backend: generic rasterizer module — multiple varying slots", () => {
  /**
   * @canon spec-the-wasm-rasterizer-draws-what-the-js-rasterizer-draws
   */
  it("matches compileJS's own output for two independently-addressed varyings", () => {
    // A scalar (non-aggregate) varying becomes a real WASM function
    // parameter even in an otherwise zero-arg program, which this
    // rasterizer's v1 scope doesn't support — both varyings here stay
    // vec3 to keep the program zero-arg/zero-return.
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

    const jsRasterizer = compileJS(vertexBuild as any, fragmentBuild as any);
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
      false,
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
      1,
    );

    const actual = new Float64Array(view.buffer, outputBase, width * height * 4);
    expect(Array.from(actual)).toEqual(Array.from(expected));
  });
});

describe("WASM backend: generic rasterizer module — scalarsInMemory for a scalar uniform", () => {
  /**
   * @canon spec-the-wasm-rasterizer-draws-what-the-js-rasterizer-draws
   */
  it("matches compileJS's own output for a fragment program with a scalar uniform", () => {
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

    const jsRasterizer = compileJS(vertexBuild as any, fragmentBuild as any);
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
      false,
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
      1,
    );

    // rasterize() re-reads brightness fresh from its own memory address
    // every fragment call (no host round-trip needed), so writing it once
    // up front — matching what a real caller would do before a draw — is
    // enough for every covered pixel.
    const actual = new Float64Array(view.buffer, outputBase, width * height * 4);
    expect(Array.from(actual)).toEqual(Array.from(expected));
  });
});

/** The clip volume's planes, each kept where it is at least 0: w + x, w - x, w + y, w - y, z and w - z. */
const CLIP_PLANES = [
  (p: ArrayLike<number>) => p[3] + p[0],
  (p: ArrayLike<number>) => p[3] - p[0],
  (p: ArrayLike<number>) => p[3] + p[1],
  (p: ArrayLike<number>) => p[3] - p[1],
  (p: ArrayLike<number>) => p[2],
  (p: ArrayLike<number>) => p[3] - p[2],
];

/** An independent clip of a polygon to the clip volume, plane by plane, each cut run from the vertex inside. */
function clipToVolume(poly: number[][]): number[][] {
  for (const distance of CLIP_PLANES) {
    const out: number[][] = [];
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i]!;
      const b = poly[(i + 1) % poly.length]!;
      const da = distance(a);
      const db = distance(b);
      if (da >= 0) out.push(a);
      if (da >= 0 !== db >= 0) {
        const [from, to, t] = da >= 0 ? [a, b, da / (da - db)] : [b, a, db / (db - da)];
        out.push(from.map((v, k) => v + (to[k]! - v) * t));
      }
    }
    poly = out;
  }
  return poly;
}

/**
 * An independent rasterization of a triangle of clip positions in one colour,
 * with no clip: a pixel centre is drawn where the point of the triangle it
 * shows lies in front of the eye, at a depth from 0 to 1, and inside each
 * edge, the line through two vertices in homogeneous `(x, y, w)`, a centre on
 * an edge by the top-left rule. Its numbers are exact for positions of a few
 * binary digits.
 */
function rasterizeExactly(positions: number[][], width: number, height: number, color: number[]): Float64Array {
  const out = new Float64Array(width * height * 4);
  const lines = [0, 1, 2].map((e) => {
    const i = positions[(e + 1) % 3]!;
    const j = positions[(e + 2) % 3]!;
    return [i[1]! * j[3]! - i[3]! * j[1]!, i[3]! * j[0]! - i[0]! * j[3]!, i[0]! * j[1]! - i[1]! * j[0]!];
  });
  const [a0, b0, c0] = lines[0]!;
  const p0 = positions[0]!;
  const sign = Math.sign(a0! * p0[0]! + b0! * p0[1]! + c0! * p0[3]!);
  // A centre on an edge is the triangle's when the edge runs up the screen, or right along it, by WebGPU's rule.
  const owns = lines.map(([a, b]) => a! * sign > 0 || (a! * sign === 0 && -b! * sign > 0));
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const X = (2 * x + 1) / width - 1;
      const Y = 1 - (2 * y + 1) / height;
      const weights = lines.map(([a, b, c]) => (a! * X + b! * Y + c!) * sign);
      if (!weights.every((weight, e) => weight > 0 || (weight === 0 && owns[e]))) continue;
      const sum = weights[0]! + weights[1]! + weights[2]!;
      const z = weights.reduce((total, weight, e) => total + weight * positions[e]![2]!, 0);
      const w = weights.reduce((total, weight, e) => total + weight * positions[e]![3]!, 0);
      if (!(sum > 0) || z / w < 0 || z / w > 1) continue;
      out.set(color, (y * width + x) * 4);
    }
  }
  return out;
}

/** The bytes after the clip scratch, which a draw must leave as they were. */
const CANARY_BYTES = 64;

/**
 * Draws one triangle of clip positions in one colour through the rasterizer
 * module alone, with no varyings, and gives its image and the canary bytes
 * after its 18 scratch slots and the lines of the triangle being drawn.
 */
function drawOnModule(positions: number[][], width: number, height: number, color: number[]) {
  const vertexBuild = () => Fn(() => builtinPosition().assign(attribute("vec4")))();
  const fragmentBuild = () => Fn(() => vec4(color[0]!, color[1]!, color[2]!, color[3]!))();
  const memory = new WebAssembly.Memory({ initial: 1 });
  const view = new DataView(memory.buffer);
  const vertexCompiled = compileWasmFn(vertexBuild as any, {
    name: "main",
    params: [],
    stage: "vertex",
    memory,
    memoryBase: 0,
  });
  const fragmentCompiled = compileWasmFn(fragmentBuild as any, { name: "main", params: [], memoryBase: 1024, memory });
  const imports = { math: Math as unknown as WebAssembly.ModuleImports, env: { memory } };
  const vertexInstance = new WebAssembly.Instance(
    new WebAssembly.Module(vertexCompiled.bytes.buffer as ArrayBuffer),
    imports,
  );
  const fragmentInstance = new WebAssembly.Instance(
    new WebAssembly.Module(fragmentCompiled.bytes.buffer as ArrayBuffer),
    imports,
  );
  const { rasterize } = instantiateRasterizer(
    vertexInstance.exports.main as () => void,
    fragmentInstance.exports.main as () => void,
    memory,
    false,
  );

  const attrDescBase = 1536;
  const attrSrcBase = 2048;
  const positionsOutBase = 4096;
  const outputBase = 16384;
  const clipScratchBase = 24576;
  // 18 vertex slots of 32 bytes with no varyings, then the lines of the triangle being drawn.
  const scratchEnd = clipScratchBase + 18 * 32 + LINES_BYTES;
  const depthBufferBase = 40960;
  positions.flat().forEach((c, i) => view.setFloat64(attrSrcBase + i * 8, c, true));
  const attrDestAddress = vertexCompiled.params.find((p) => p.kind === "attributeMemory")!.address;
  writeAttributeDescriptors(view, attrDescBase, [{ srcOffset: 0, destAddress: attrDestAddress, sizeBytes: 32 }]);
  const positionAddress = vertexCompiled.params.find((p) => p.kind === "positionMemory")!.address;
  const fragmentValueAddress = fragmentCompiled.params.find((p) => p.kind === "valueMemory")!.address;
  clearDepthBuffer(view, depthBufferBase, width * height);
  new Uint8Array(memory.buffer, scratchEnd, CANARY_BYTES).fill(0xa5);

  rasterize(
    positions.length,
    attrSrcBase,
    32,
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
    depthBufferBase,
    1,
  );

  return {
    image: Array.from(new Float64Array(memory.buffer, outputBase, width * height * 4)),
    canaries: [...new Uint8Array(memory.buffer, scratchEnd, CANARY_BYTES)],
  };
}

describe("WASM backend: generic rasterizer module — clipping to the clip volume", () => {
  /**
   * @canon spec-a-rasterizer-clips-a-triangle-to-the-clip-volume
   */
  it("draws a triangle with one vertex behind the eye as its part inside the clip volume, matching an independent rasterization with no clip", () => {
    const positions = [
      [-1, -1, -1, -1],
      [1, -1, 2, 2],
      [-1, 1, 2, 2],
    ];
    const polygon = clipToVolume(positions);
    expect(polygon.length).toBe(5); // two planes cut off the vertex behind the eye: a pentagon
    const expected = Array.from(rasterizeExactly(positions, 8, 8, [9, 8, 7, 6]));
    expect(expected.some((v) => v !== 0)).toBe(true);
    expect(drawOnModule(positions, 8, 8, [9, 8, 7, 6]).image).toEqual(expected);
  });

  /**
   * Each of the six planes cuts this triangle, which leaves a polygon of 9
   * vertices, the most a triangle can: a scratch half full, whose polygon
   * bounds the pixels the triangle's own lines test.
   *
   * @canon spec-a-rasterizer-clips-a-triangle-to-the-clip-volume
   */
  it("clips a triangle every plane cuts into 9 vertices, writing nothing past its scratch", () => {
    const positions = [
      [2.5, 2.25, 1, 2],
      [0.75, -2, -2, 3.75],
      [-4, -2, 3.75, -0.5],
    ];
    const polygon = clipToVolume(positions);
    expect(polygon.length).toBe(9);
    const expected = Array.from(rasterizeExactly(positions, 16, 16, [9, 8, 7, 6]));
    expect(expected.some((v) => v !== 0)).toBe(true);
    const { image, canaries } = drawOnModule(positions, 16, 16, [9, 8, 7, 6]);
    expect(image).toEqual(expected);
    expect(canaries.every((byte) => byte === 0xa5)).toBe(true);
  });
  /**
   * @canon spec-a-rasterizer-clips-a-triangle-to-the-clip-volume
   */
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
      false,
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
      1,
    );

    const actual = new Float64Array(view.buffer, outputBase, width * height * 4);
    expect(Array.from(actual).every((v) => v === 0)).toBe(true);
  });
});

describe("WASM backend: generic rasterizer module — depth test", () => {
  /**
   * @canon spec-a-rasterizer-keeps-the-closer-fragment
   */
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
    const nearTriangle = triangleAt(0.25); // smaller z = closer, per this rasterizer's depth convention

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
        false,
      );
      const { rasterize: rasterizeBlue } = instantiateRasterizer(
        vertexInstance.exports.main as () => void,
        blueInstance.exports.main as () => void,
        memory,
        false,
      );

      const attrSrcBase = 4096;
      const positionsOutBase = 8192;
      const outputBase = 16384;
      const attrDescBase = 3072;
      const clipScratchBase = 24576;
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
          depthBufferBase,
          1,
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
