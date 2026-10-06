import { afterAll, describe, expect, it } from "vitest";
import { GPU_ENABLED, releaseGpu } from "../testing/gpu";
import { READ_PIXEL, runInGpuPage } from "../testing/browser";

/**
 * A full-screen triangle, a 4×4 canvas, and a uniform colour every adapter
 * draws it in. Pixel (x, y) counts from the top left.
 */
const SCENE = `
import { Fn, attribute, builtinPosition, fragCoord, uniform, vec2, vec4 } from "../rmsl";
import { createGlsl } from "../glsl";
import { createJs, createJsGrid } from "../js";
import { createWasm, createWasmGrid } from "../wasm";
${READ_PIXEL}
const position = attribute("vec3");
const colour = uniform("vec4");
const vertex = () => Fn(() => { builtinPosition().assign(vec4(position, 1)); })();
const fragment = () => Fn(() => colour)();
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
  jsTexture: () => drawTextured(createJs(vertex, texturedFragment, { attributeTypes: { [position.name]: "vec3" } })),
  wasmTexture: () => drawTextured(createWasm(vertex, texturedFragment, { attributeTypes: { [position.name]: "vec3" } })),
  jsRoutine: () => drawRoutine(createJsGrid({ draw: routine() })),
  wasmRoutine: () => drawRoutine(createWasmGrid({ draw: routine() })),
};
`;

afterAll(async () => {
  await releaseGpu();
}, 120_000);

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
   * @canon spec-an-adapter-takes-the-texture-a-sampler-reads-from-the-host
   */
  it("samples the texture set with setTexture with createGlsl", async () => {
    expect(await drawn("glslTexture")).toEqual({ r: 0, g: 255, b: 0, a: 255 });
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
