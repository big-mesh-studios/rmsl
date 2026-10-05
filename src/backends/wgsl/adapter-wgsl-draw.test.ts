import { describe, it, expect, afterAll } from "vitest";
import { releaseGpu, webgpuAvailable } from "../../testing/gpu";
import { READ_PIXEL, runInWebGpuPage } from "../../testing/browser";

const WEBGPU = await webgpuAvailable();

/**
 * Helpers every entry shares: a quad over a clip-space rectangle, placed by
 * `vertexIndex()`, and a 4×4 canvas. Pixel (x, y) counts from the top left.
 */
const QUAD = `
import { Fn, float, instancedArray, invocationIndex, uint, uniform, vec2, vec4, vertexIndex } from "../../rmsl";
import { createWgsl, createWgslContext } from "../../wgsl";
${READ_PIXEL}
const quad = (x0, y0, x1, y1) => Fn(() => {
  const v = vertexIndex();
  const right = v.equal(uint(1)).or(v.equal(uint(4))).or(v.equal(uint(5)));
  const top = v.equal(uint(2)).or(v.equal(uint(3))).or(v.equal(uint(5)));
  return vec4(right.select(float(x1), float(x0)), top.select(float(y1), float(y0)), 0, 1);
})();
const canvas = () => {
  const c = document.createElement("canvas");
  c.width = 4;
  c.height = 4;
  return c;
};
`;

const ENTRY_CONTEXT = `
${QUAD}
globalThis.__rmslAdapterContextRun = async () => {
  const context = await createWgslContext();
  const shade = instancedArray(1, "float");
  context.compute(Fn(() => { shade.element(invocationIndex()).assign(float(0.5)); })().compute(1));
  const adapter = createWgsl({ context, vertex: quad(-1, -1, 1, 1), fragment: Fn(() => vec4(shade.element(0), 0, 0, 1))() });
  const target = canvas();
  await adapter.attach(target);
  adapter.draw({ count: 6 });
  await context.device.queue.onSubmittedWorkDone();
  return readPixel(target, 2, 2);
};
`;

const ENTRY_COMPOSE = `
${QUAD}
globalThis.__rmslAdapterComposeRun = async () => {
  const context = await createWgslContext();
  const target = canvas();
  const unread = uniform("float");
  const left = createWgsl({ context, vertex: quad(-1, -1, 0, 1), fragment: Fn(() => vec4(1, 0, 0, 1))() });
  const corner = createWgsl({ context, vertex: quad(0, 0, 1, 1), fragment: Fn(() => vec4(0, 1, 0, 1))() });
  await left.attach(target);
  await corner.attach(target);
  left.setUniform(unread, 1);
  left.draw({ count: 6, clearColor: [0, 0, 1, 1] });
  corner.draw({ count: 6, clear: false });
  await context.device.queue.onSubmittedWorkDone();
  return { left: readPixel(target, 0, 2), corner: readPixel(target, 3, 0), cleared: readPixel(target, 3, 3) };
};
`;

const ENTRY_OWN_BUFFERS = `
${QUAD}
globalThis.__rmslAdapterOwnBuffersRun = async () => {
  const shade = instancedArray(1, "float");
  const adapter = createWgsl({ vertex: quad(-1, -1, 1, 1), fragment: Fn(() => vec4(0, shade.element(0), 0, 1))() });
  const target = canvas();
  await adapter.attach(target);
  adapter.setAttribute(shade.name, Float32Array.of(0.25));
  adapter.draw({ count: 6 });
  await adapter.device().queue.onSubmittedWorkDone();
  return readPixel(target, 2, 2);
};
`;

function run(source: string, entryPoint: string) {
  return runInWebGpuPage(source, entryPoint, new URL(".", import.meta.url).pathname);
}

describe.skipIf(!WEBGPU)("createWgsl drawing storage buffers on a real adapter", () => {
  /**
   * @canon spec-a-wgsl-buffer-feeds-a-draw-without-a-copy
   */
  it("draws what a compute context's program wrote", async () => {
    const pixel = await run(ENTRY_CONTEXT, "__rmslAdapterContextRun");
    expect(pixel.r).toBeGreaterThan(120);
    expect(pixel.r).toBeLessThan(136);
  }, 60_000);
  /**
   * @canon spec-several-adapters-draw-on-one-canvas
   */
  it("composes adapters on one canvas, clearing to the clear colour, and ignores an unread uniform", async () => {
    const { left, corner, cleared } = await run(ENTRY_COMPOSE, "__rmslAdapterComposeRun");
    expect(left).toMatchObject({ r: 255, g: 0, b: 0 });
    expect(corner).toMatchObject({ r: 0, g: 255, b: 0 });
    expect(cleared).toMatchObject({ r: 0, g: 0, b: 255 });
  }, 60_000);
  /**
   * @canon spec-a-wgsl-buffer-feeds-a-draw-without-a-copy
   */
  it("fills its own storage buffers through setAttribute without a context", async () => {
    const pixel = await run(ENTRY_OWN_BUFFERS, "__rmslAdapterOwnBuffersRun");
    expect(pixel.g).toBeGreaterThan(56);
    expect(pixel.g).toBeLessThan(72);
  }, 60_000);
});

afterAll(async () => {
  await releaseGpu();
});
