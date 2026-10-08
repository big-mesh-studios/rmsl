import { afterAll, describe, expect, it } from "vitest";
import { GPU_ENABLED, releaseGpu } from "../testing/gpu";
import { GL_STATE, READ_PIXEL, runInGpuPage } from "../testing/browser";

/**
 * A full-screen triangle, a 4×4 canvas, and a uniform colour every adapter
 * draws it in. Pixel (x, y) counts from the top left.
 */
const SCENE = `
import { Fn, attribute, builtinPosition, float, fragCoord, uint, uniform, varying, vec2, vec4, vertexIndex } from "../rmsl";
import { createGlsl } from "../glsl";
import { createJs, createJsGrid } from "../js";
import { createWasm, createWasmGrid } from "../wasm";
${READ_PIXEL}
${GL_STATE}
const position = attribute("vec3");
const colour = uniform("vec4");
const vertex = () => Fn(() => { builtinPosition().assign(vec4(position, 1)); })();
const fragment = () => Fn(() => colour)();
// A vertex stage that also reads an attribute the host never sets, which moves the triangle out of view unless it holds 0.
const offset = attribute("vec4");
const offsetVertex = () => Fn(() => { builtinPosition().assign(vec4(position.add(offset.xyz.mul(8)), 1)); })();
const TRIANGLE = Float32Array.of(-1, -1, 0, 3, -1, 0, -1, 3, 0);
const canvas = () => {
  const c = document.createElement("canvas");
  c.width = 4;
  c.height = 4;
  return c;
};
const drawWith = (adapter) => {
  const target = canvas();
  adapter.attach(target);
  adapter.setAttribute(position, TRIANGLE);
  adapter.setUniform(colour, [0, 1, 0, 1]);
  adapter.draw({ count: 3 });
  return readPixel(target, 1, 2);
};
const image = uniform("sampler2D");
const texturedFragment = () => Fn(() => image.texture(vec2(0.75, 0.25)))();
const drawTextured = (adapter) => {
  const target = canvas();
  adapter.attach(target);
  adapter.setAttribute(position, TRIANGLE);
  // Row 0 is red then green, row 1 blue then white, so (0.75, 0.25) lands on the green texel.
  adapter.setTexture(image, {
    data: Uint8Array.of(255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255),
    width: 2,
    height: 2,
  });
  adapter.draw({ count: 3 });
  return readPixel(target, 1, 2);
};
// A triangle over the canvas whose vertices write their index to an integer varying.
const flatIndex = () => {
  const k = varying("int");
  const vertex = Fn(() => {
    const v = vertexIndex();
    k.assign(v.toInt());
    return vec4(v.equal(uint(1)).select(float(3), float(-1)), v.equal(uint(2)).select(float(3), float(-1)), 0, 1);
  })();
  const target = canvas();
  const adapter = createGlsl(vertex, Fn(() => vec4(k.toFloat().div(2), 0, 0, 1))());
  adapter.attach(target);
  adapter.draw({ count: 3 });
  return readPixel(target, 1, 2);
};
// Two textures of one shape, the second set over the first: the draw shows the
// second, written into the GL texture the first made.
const drawRetextured = (adapter) => {
  const target = canvas();
  const gl = target.getContext("webgl2");
  let created = 0;
  const createTexture = gl.createTexture.bind(gl);
  gl.createTexture = () => (created++, createTexture());
  adapter.attach(target);
  adapter.setAttribute(position, TRIANGLE);
  const texels = (r, g, b) => ({ data: Uint8Array.of(r, g, b, 255, r, g, b, 255, r, g, b, 255, r, g, b, 255), width: 2, height: 2 });
  adapter.setTexture(image, texels(0, 255, 0));
  adapter.setTexture(image, texels(255, 0, 0));
  adapter.draw({ count: 3 });
  return { pixel: readPixel(target, 1, 2), created };
};
// The textured triangle drawn over state the page set, and over a fresh context.
const drawnOver = (dirty) => {
  const target = canvas();
  const gl = target.getContext("webgl2");
  if (dirty) dirtyGlState(gl);
  const adapter = createGlsl(offsetVertex(), texturedFragment());
  adapter.attach(target);
  adapter.setAttribute(position, TRIANGLE);
  adapter.setTexture(image, {
    data: Uint8Array.of(255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255),
    width: 2,
    height: 2,
  });
  adapter.draw({ count: 3, clearColor: [0.2, 0.4, 0.6, 1] });
  return [readPixel(target, 1, 2), readPixel(target, 3, 3), readPixel(target, 0, 0)];
};
// Every piece of state that a call of a GLSL adapter left changed, over state the page set.
const glslStateChanged = (preserveState) => {
  const target = canvas();
  const gl = target.getContext("webgl2");
  dirtyGlState(gl);
  const before = glState(gl);
  const changed = new Set();
  const check = () => changedGlState(before, glState(gl)).forEach((name) => changed.add(name));
  const adapter = createGlsl(offsetVertex(), Fn(() => image.texture(vec2(0.75, 0.25)).mul(colour))(), { preserveState });
  adapter.attach(target);
  check();
  adapter.setAttribute(position, TRIANGLE);
  check();
  adapter.setUniform(colour, [0, 1, 0, 1]);
  check();
  adapter.setTexture(image, { data: Uint8Array.of(255, 0, 0), width: 1, height: 1, channels: 3 });
  check();
  adapter.draw({ count: 3 });
  check();
  return [...changed].sort();
};
const routine = () => vec4(fragCoord().x.div(4), 0, 0, 1);
const drawRoutine = (adapter) => {
  const target = canvas();
  adapter.attach(target);
  adapter.draw();
  return readPixel(target, 3, 0);
};
globalThis.__rmslAdapterDraw = {
  glsl: () => drawWith(createGlsl(vertex(), fragment())),
  js: () => drawWith(createJs(vertex, fragment, { attributeTypes: { [position.name]: "vec3" } })),
  wasm: () => drawWith(createWasm(vertex, fragment, { attributeTypes: { [position.name]: "vec3" } })),
  glslTexture: () => drawTextured(createGlsl(vertex(), texturedFragment())),
  glslRetexture: () => drawRetextured(createGlsl(vertex(), texturedFragment())),
  jsTexture: () => drawTextured(createJs(vertex, texturedFragment, { attributeTypes: { [position.name]: "vec3" } })),
  wasmTexture: () => drawTextured(createWasm(vertex, texturedFragment, { attributeTypes: { [position.name]: "vec3" } })),
  jsRoutine: () => drawRoutine(createJsGrid({ draw: routine() })),
  wasmRoutine: () => drawRoutine(createWasmGrid({ draw: routine() })),
  glslFlatIndex: () => flatIndex(),
  glslStateKept: () => glslStateChanged(false),
  glslDrawsOver: () => ({ dirty: drawnOver(true), clean: drawnOver(false) }),
  glslStatePreserved: () => glslStateChanged(true),
};
`;

afterAll(async () => {
  await releaseGpu();
}, 120_000);

/**
 * A full-screen triangle on a 4×4 canvas that `createGlsl` colours from a
 * storage buffer, by entry. Each entry returns the pixel it read, or the error
 * it hit.
 */
const STORAGE_SCENE = `
import { Fn, attribute, builtinPosition, instancedArray, int, vec3, vec4 } from "../rmsl";
import { createGlsl } from "../glsl";
${READ_PIXEL}
const position = attribute("vec3");
const TRIANGLE = Float32Array.of(-1, -1, 0, 3, -1, 0, -1, 3, 0);
const vertex = () => Fn(() => { builtinPosition().assign(vec4(position, 1)); })();
/** Draws \`fragment\` with createGlsl over \`vertexStage\`, after \`set\` gives the adapter its values. */
const draw = (fragment, set = () => {}, vertexStage = vertex()) => {
  try {
    const target = document.createElement("canvas");
    target.width = 4;
    target.height = 4;
    const adapter = createGlsl(vertexStage, fragment);
    adapter.attach(target);
    adapter.setAttribute(position, TRIANGLE);
    set(adapter);
    adapter.draw({ count: 3 });
    return readPixel(target, 1, 2);
  } catch (error) {
    return { error: error.message };
  }
};
globalThis.__rmslStorageDraw = {
  vec3: () => {
    const colours = instancedArray(Float32Array.of(1, 0, 0, 0, 1, 0), "vec3");
    return draw(Fn(() => vec4(colours.element(int(1)), 1))());
  },
  wrapped: () => {
    const values = instancedArray(Float32Array.from({ length: 40 }, (_, i) => i / 64), "float");
    return draw(Fn(() => vec4(values.element(int(37)), 0, 0, 1))());
  },
  ivec2: () => {
    const values = instancedArray(Int32Array.of(0, 0, -3, 255), "ivec2");
    return draw(Fn(() => vec4(0, values.element(int(1)).y.toFloat().div(255), 0, 1))());
  },
  uint: () => {
    const values = instancedArray(Uint32Array.of(7, 4294967295), "uint");
    return draw(Fn(() => vec4(0, values.element(int(1)).equal(4294967295).select(1, 0), 0, 1))());
  },
  mat3: () => {
    const matrices = instancedArray(Float32Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0), "mat3");
    return draw(Fn(() => vec4(matrices.element(int(1)).mul(vec3(0, 0, 1)), 1))());
  },
  refilled: () => {
    const colours = instancedArray(2, "vec4");
    return draw(Fn(() => colours.element(int(1)))(), (adapter) =>
      adapter.setAttribute(colours.name, Float32Array.of(1, 0, 0, 1, 0, 1, 0, 1)),
    );
  },
  vertexStage: () => {
    const shift = instancedArray(Float32Array.of(0, 0, 8, 8), "vec2");
    const shifted = Fn(() => { builtinPosition().assign(vec4(position.xy.add(shift.element(int(0))), 0, 1)); })();
    return draw(Fn(() => vec4(0, 1, 0, 1))(), undefined, shifted);
  },
};
`;

describe.skipIf(!GPU_ENABLED)("createGlsl reading storage in a render stage", () => {
  /** Draws the entry \`name\` of \`STORAGE_SCENE\` and reads back its pixel. */
  const drawn = async (name: string) =>
    runInGpuPage(
      `${STORAGE_SCENE}\nglobalThis.__rmslStorageDrawRun = async () => globalThis.__rmslStorageDraw.${name}();`,
      "__rmslStorageDrawRun",
      new URL(".", import.meta.url).pathname,
    );
  const GREEN = { r: 0, g: 255, b: 0, a: 255 };

  /**
   * @canon spec-a-storage-texel-holds-one-element-or-one-column
   */
  it.each(["vec3", "ivec2", "uint", "mat3"])(
    "reads a %s element of a storage buffer with createGlsl",
    async (entry) => {
      expect(await drawn(entry)).toEqual(GREEN);
    },
    120_000,
  );

  /**
   * 40 texels make a texture 8 wide, so element 37 sits at column 5 of row 4.
   *
   * @canon spec-a-storage-texture-is-a-power-of-two-wide
   */
  it("reads an element past the first row of a storage texture with createGlsl", async () => {
    expect(await drawn("wrapped")).toEqual({ r: 147, g: 0, b: 0, a: 255 });
  }, 120_000);

  /**
   * @canon spec-the-glsl-adapter-uploads-each-storage-buffer-it-reads
   */
  it("reads storage from a vertex stage with createGlsl", async () => {
    expect(await drawn("vertexStage")).toEqual(GREEN);
  }, 120_000);

  /**
   * @canon spec-the-glsl-adapter-uploads-each-storage-buffer-it-reads
   */
  it("reads what setAttribute put in a storage buffer with createGlsl", async () => {
    expect(await drawn("refilled")).toEqual(GREEN);
  }, 120_000);
});

describe.skipIf(!GPU_ENABLED)("adapters drawing into a canvas in a browser", () => {
  /** Draws with one adapter in the browser and reads back its pixel. */
  const drawn = async (adapter: string) =>
    runInGpuPage(
      `${SCENE}\nglobalThis.__rmslAdapterDrawRun = async () => globalThis.__rmslAdapterDraw.${adapter}();`,
      "__rmslAdapterDrawRun",
      new URL(".", import.meta.url).pathname,
    );

  /**
   * @canon spec-an-adapter-draws-one-frame-for-each-call
   */
  it("draws a triangle into its canvas with createGlsl", async () => {
    expect(await drawn("glsl")).toEqual({ r: 0, g: 255, b: 0, a: 255 });
  }, 120_000);

  /**
   * @canon spec-an-adapter-draws-one-frame-for-each-call
   */
  it("draws a triangle into its canvas with createJs", async () => {
    expect(await drawn("js")).toEqual({ r: 0, g: 255, b: 0, a: 255 });
  }, 120_000);

  /**
   * @canon spec-an-adapter-draws-one-frame-for-each-call
   */
  it("draws a triangle into its canvas with createWasm", async () => {
    expect(await drawn("wasm")).toEqual({ r: 0, g: 255, b: 0, a: 255 });
  }, 120_000);

  /**
   * @canon spec-an-adapter-draws-one-frame-for-each-call
   */
  it("draws a fragCoord program over its canvas with createJsGrid", async () => {
    expect(await drawn("jsRoutine")).toEqual({ r: Math.round((3.5 / 4) * 255), g: 0, b: 0, a: 255 });
  }, 120_000);

  /**
   * The vertices write 0, 1 and 2, drawn as red of half that.
   *
   * @canon exception-a-glsl-flat-varying-takes-the-last-vertex
   */
  it("reads an integer varying as the triangle's last vertex wrote it on GLSL", async () => {
    expect(await drawn("glslFlatIndex")).toEqual({ r: 255, g: 0, b: 0, a: 255 });
  }, 120_000);

  /**
   * @canon spec-a-webgl-call-sets-the-state-it-reads
   * @canon spec-a-webgl-draw-gives-an-attribute-with-no-data-a-fresh-value
   */
  it("draws over state the page set as it draws on a fresh context with createGlsl", async () => {
    const { dirty, clean } = await drawn("glslDrawsOver");
    expect(dirty).toEqual(clean);
  }, 120_000);

  /**
   * @canon spec-a-webgl-call-leaves-the-state-it-set
   */
  it("leaves the unpack alignment and its program as it set them with createGlsl", async () => {
    expect(await drawn("glslStateKept")).toEqual(expect.arrayContaining(["UNPACK_ALIGNMENT", "CURRENT_PROGRAM"]));
  }, 120_000);

  /**
   * @canon spec-a-webgl-call-sets-the-state-it-reads
   */
  it("turns dithering off over a page that turned it on with createGlsl", async () => {
    expect(await drawn("glslStateKept")).toContain("DITHER");
  }, 120_000);

  /**
   * @canon spec-a-glsl-adapter-asked-to-preserve-state-puts-it-back
   */
  it("puts back every piece of state it changed with preserveState with createGlsl", async () => {
    expect(await drawn("glslStatePreserved")).toEqual([]);
  }, 120_000);

  /**
   * @canon spec-an-adapter-takes-the-texture-a-sampler-reads-from-the-host
   */
  it("samples the texture set with setTexture with createGlsl", async () => {
    expect(await drawn("glslTexture")).toEqual({ r: 0, g: 255, b: 0, a: 255 });
  }, 120_000);

  /**
   * @canon spec-an-adapter-writes-a-texture-of-the-same-shape-in-place
   */
  it("writes a texture of the same shape into the texture it has with createGlsl", async () => {
    const result = await drawn("glslRetexture");
    expect(result.pixel).toEqual({ r: 255, g: 0, b: 0, a: 255 });
    expect(result.created).toBe(1);
  }, 120_000);

  /**
   * @canon spec-an-adapter-takes-the-texture-a-sampler-reads-from-the-host
   */
  it("samples the texture set with setTexture with createJs", async () => {
    expect(await drawn("jsTexture")).toEqual({ r: 0, g: 255, b: 0, a: 255 });
  }, 120_000);

  /**
   * @canon spec-an-adapter-takes-the-texture-a-sampler-reads-from-the-host
   */
  it("samples the texture set with setTexture with createWasm", async () => {
    expect(await drawn("wasmTexture")).toEqual({ r: 0, g: 255, b: 0, a: 255 });
  }, 120_000);

  /**
   * @canon bug-create-wasm-grid-fails-under-its-default-name
   */
  it.fails(
    "draws a fragCoord program over its canvas with createWasmGrid",
    async () => {
      expect(await drawn("wasmRoutine")).toEqual({ r: Math.round((3.5 / 4) * 255), g: 0, b: 0, a: 255 });
    },
    120_000,
  );
});
