import { afterAll, describe, expect, it } from "vitest";
import { attribute, builtinPosition, Fn, uniform, vec4 } from "../rmsl";
import { compileJS } from "../js";
import { compileWasm } from "../wasm";
import { GPU_ENABLED, releaseGpu } from "../testing/gpu";
import { READ_PIXEL, runInGpuPage } from "../testing/browser";

/** A triangle at depth `z` that covers the whole viewport. */
const screen = (z = 0) => [-1, -1, z, 3, -1, z, -1, 3, z];

/** A triangle that covers no pixel of the viewport. */
const offscreen = [5, 5, 0, 6, 5, 0, 5, 6, 0];

/** A rasterizer of a flat colour that takes its triangles and colour as given, on a 2×2 viewport. */
function flat(target: "JS" | "WASM") {
  const position = attribute("vec3");
  const color = uniform("vec4");
  const vertex = () => Fn(() => builtinPosition().assign(vec4(position, 1)))();
  const fragment = () => Fn(() => color)();
  const routine =
    target === "JS"
      ? compileJS(vertex as any, fragment as any, { attributeTypes: { [position.name]: "vec3" } })
      : compileWasm(vertex as any, fragment as any);
  return (triangles: number[], rgba: number[], options: Record<string, unknown> = {}) =>
    Array.from(
      routine.draw(
        { attributes: { [position.name]: new Float64Array(triangles) }, uniforms: { [color.name]: rgba } },
        { width: 2, height: 2, ...options },
      ),
    );
}

const pixels = (rgba: number[]) => Array.from({ length: 4 }, () => rgba).flat();

describe.each(["JS", "WASM"] as const)("a CPU rasterizer on %s clears before it draws", (target) => {
  const RED = [1, 0, 0, 1];
  const BLUE = [0, 0, 1, 1];

  /**
   * @canon spec-a-draw-clears-the-colour-of-its-target-to-its-clear-colour
   */
  it("replaces the pixels of an earlier draw with the clear colour", () => {
    const draw = flat(target);
    draw(screen(), RED);
    expect(draw(offscreen, RED, { clearColor: BLUE })).toEqual(pixels(BLUE));
  });

  /**
   * @canon spec-a-glsl-js-and-wasm-draw-clears-to-transparent-black
   */
  it("clears to transparent black when the draw gives no clear colour", () => {
    const draw = flat(target);
    draw(screen(), RED);
    expect(draw(offscreen, RED)).toEqual(pixels([0, 0, 0, 0]));
  });

  /**
   * @canon spec-a-draw-keeps-what-is-under-it-when-it-asks-not-to-clear
   */
  it("keeps the pixels an earlier draw left when the draw passes clear: false", () => {
    const draw = flat(target);
    draw(screen(), RED);
    expect(draw(offscreen, RED, { clear: false })).toEqual(pixels(RED));
  });

  /**
   * @canon spec-a-cpu-draw-clears-its-depth-first-unless-it-asks-not-to
   */
  it("lets a farther draw show after a nearer one, since each draw clears depth", () => {
    const draw = flat(target);
    draw(screen(0.2), RED);
    expect(draw(screen(0.6), BLUE, { clear: false })).toEqual(pixels(BLUE));
  });

  /**
   * @canon spec-a-cpu-draw-clears-its-depth-first-unless-it-asks-not-to
   */
  it("hides a farther draw behind a nearer one when the draw passes clearDepth: false", () => {
    const draw = flat(target);
    draw(screen(0.2), RED);
    expect(draw(screen(0.6), BLUE, { clear: false, clearDepth: false })).toEqual(pixels(RED));
  });
});

/**
 * Draws a full-screen green triangle with `createGlsl`, then a second draw of
 * no vertices whose options an entry chooses, and reads a pixel.
 */
const GLSL_CLEARING = `
import { Fn, attribute, builtinPosition, vec4 } from "../rmsl";
import { createGlsl } from "../glsl";
${READ_PIXEL}
const TRIANGLE = Float32Array.of(-1, -1, 0, 3, -1, 0, -1, 3, 0);
const second = (options) => {
  const position = attribute("vec3");
  const target = document.createElement("canvas");
  target.width = 4;
  target.height = 4;
  const adapter = createGlsl(
    Fn(() => { builtinPosition().assign(vec4(position, 1)); })(),
    Fn(() => vec4(0, 1, 0, 1))(),
  );
  adapter.attach(target);
  adapter.setAttribute(position, TRIANGLE);
  adapter.draw({ count: 3 });
  adapter.draw({ count: 0, ...options });
  return readPixel(target, 1, 2);
};
globalThis.__rmslClearing = {
  given: async () => second({ clearColor: [0, 0, 1, 1] }),
  byDefault: async () => second({}),
  declined: async () => second({ clear: false }),
};
`;

const glsl = (name: "given" | "byDefault" | "declined") =>
  runInGpuPage(
    `${GLSL_CLEARING}\nglobalThis.__rmslClearingRun = async () => globalThis.__rmslClearing.${name}();`,
    "__rmslClearingRun",
    new URL(".", import.meta.url).pathname,
  );

describe.skipIf(!GPU_ENABLED)("createGlsl clears before it draws", () => {
  /**
   * @canon spec-a-draw-clears-the-colour-of-its-target-to-its-clear-colour
   */
  it("replaces the pixels of an earlier draw with the clear colour", async () => {
    expect(await glsl("given")).toEqual({ r: 0, g: 0, b: 255, a: 255 });
  }, 120_000);

  /**
   * @canon spec-a-glsl-js-and-wasm-draw-clears-to-transparent-black
   */
  it("clears to transparent black when the draw gives no clear colour", async () => {
    expect(await glsl("byDefault")).toEqual({ r: 0, g: 0, b: 0, a: 0 });
  }, 120_000);

  /**
   * @canon spec-a-draw-keeps-what-is-under-it-when-it-asks-not-to-clear
   */
  it("keeps the pixels an earlier draw left when the draw passes clear: false", async () => {
    expect(await glsl("declined")).toEqual({ r: 0, g: 255, b: 0, a: 255 });
  }, 120_000);
});

afterAll(async () => {
  await releaseGpu();
}, 120_000);
