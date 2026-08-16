// End-to-end smoke test for the WebGL renderer: bundle the scene library with
// esbuild, draw a lit mesh in a real Chromium page, and read pixels back.
//
// Runs with the other layers that need a graphics device, and is skipped only
// where RMSL_SKIP_GPU says the machine has none.

import { describe, it, expect, afterAll } from "vitest";
import { build } from "esbuild";
import { gpuPage, GPU_ENABLED, releaseGpu } from "../testing/gpu";

const ENTRY = `
import { WebGLRenderer, Scene, Mesh, PerspectiveCamera, BoxGeometry,
  MeshStandardMaterial, AmbientLight, DirectionalLight } from "./index";
globalThis.__rmslRun = () => {
  const canvas = document.createElement("canvas");
  canvas.width = 32;
  canvas.height = 32;
  const renderer = new WebGLRenderer(canvas, { antialias: false });
  renderer.setClearColor(0x000000);
  const scene = new Scene();
  scene.add(new AmbientLight(0xffffff, 0.3));
  const sun = new DirectionalLight(0xffffff, 1.5);
  sun.position.set(2, 4, 3);
  scene.add(sun);
  const mesh = new Mesh(new BoxGeometry(), new MeshStandardMaterial({ color: 0xff0000, roughness: 0.5 }));
  scene.add(mesh);
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 4);
  camera.lookAt(0, 0, 0);
  renderer.render(scene, camera);
  const pixels = new Uint8Array(4);
  const gl = renderer.gl;
  gl.readPixels(16, 16, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  return { r: pixels[0], g: pixels[1], b: pixels[2] };
};
`;

const ENTRY_INT = `
import { WebGLRenderer, Scene, Mesh, PerspectiveCamera, PlaneGeometry,
  MeshBasicMaterial, DataTexture } from "./index";
import { uvec2 } from "../rmsl";
globalThis.__rmslIntRun = () => {
  const canvas = document.createElement("canvas");
  canvas.width = 16;
  canvas.height = 16;
  const renderer = new WebGLRenderer(canvas, { antialias: false });
  renderer.setClearColor(0x000000);
  const scene = new Scene();
  const material = new MeshBasicMaterial();
  material.fragmentNode = (b) => {
    const tex = b.sampler("data", "usampler2D",
      () => new DataTexture(new Uint8Array([0, 0, 255, 255]), 1, 1));
    return tex.texture(uvec2(0, 0)).toVec4();
  };
  const mesh = new Mesh(new PlaneGeometry(2, 2), material);
  scene.add(mesh);
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);
  renderer.render(scene, camera);
  const pixels = new Uint8Array(4);
  const gl = renderer.gl;
  gl.readPixels(8, 8, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  return { r: pixels[0], g: pixels[1], b: pixels[2] };
};
`;

const ENTRY_LINES = `
import { WebGLRenderer, Scene, PerspectiveCamera,
  LineSegments2, LineSegmentsGeometry, Line2NodeMaterial } from "./index";
globalThis.__rmslLineRun = () => {
  const canvas = document.createElement("canvas");
  canvas.width = 32;
  canvas.height = 32;
  const renderer = new WebGLRenderer(canvas, { antialias: false });
  renderer.setClearColor(0x000000);
  const scene = new Scene();
  const geometry = new LineSegmentsGeometry();
  geometry.setPositions([-0.8, 0, 0, 0.8, 0, 0]);
  const line = new LineSegments2(geometry, new Line2NodeMaterial({ color: 0xff0000, linewidth: 4 }));
  scene.add(line);
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 4);
  camera.lookAt(0, 0, 0);
  renderer.render(scene, camera);
  const gl = renderer.gl;
  const center = new Uint8Array(4);
  gl.readPixels(16, 16, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, center);
  const corner = new Uint8Array(4);
  gl.readPixels(2, 2, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, corner);
  return { center: [center[0], center[1], center[2]], corner: [corner[0], corner[1], corner[2]] };
};
`;

// Three samplers whose textures are all uploaded during the same draw, each
// written to its own colour channel. A sampler reading a texture other than its
// own shows up as a channel holding another channel's value.
const ENTRY_SAMPLERS = `
import { WebGLRenderer, Scene, Mesh, PerspectiveCamera, PlaneGeometry,
  MeshBasicMaterial, DataTexture } from "./index";
import { float, uvec2, vec2, vec4 } from "../rmsl";
globalThis.__rmslSamplersRun = () => {
  const canvas = document.createElement("canvas");
  canvas.width = 16;
  canvas.height = 16;
  const renderer = new WebGLRenderer(canvas, { antialias: false });
  renderer.setClearColor(0x000000);
  const scene = new Scene();
  const first = new DataTexture(new Uint8Array([20, 0, 0, 0]), 1, 1);
  const second = new DataTexture(new Uint8Array([120, 0, 0, 0]), 1, 1);
  const third = new DataTexture(new Uint8Array([0, 0, 220, 255]), 1, 1);
  const material = new MeshBasicMaterial();
  material.fragmentNode = (b) => {
    const a = b.sampler("first", "usampler2D", () => first);
    const c = b.sampler("second", "usampler2D", () => second);
    const d = b.sampler("third", "sampler2D", () => third);
    return vec4(
      a.texture(uvec2(0, 0)).r.toFloat().div(float(255)),
      c.texture(uvec2(0, 0)).r.toFloat().div(float(255)),
      d.texture(vec2(0.5, 0.5)).b,
      float(1),
    );
  };
  const mesh = new Mesh(new PlaneGeometry(2, 2), material);
  scene.add(mesh);
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);
  renderer.render(scene, camera);
  const pixels = new Uint8Array(4);
  const gl = renderer.gl;
  gl.readPixels(8, 8, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  return { r: pixels[0], g: pixels[1], b: pixels[2], error: gl.getError() };
};
`;

async function bundleEntry(source: string): Promise<string> {
  const result = await build({
    stdin: {
      contents: source,
      resolveDir: new URL(".", import.meta.url).pathname,
      loader: "ts",
    },
    bundle: true,
    write: false,
    format: "iife",
    platform: "browser",
    logLevel: "silent",
  });
  return result.outputFiles[0].text;
}

describe.skipIf(!GPU_ENABLED)("WebGLRenderer", () => {
  it("renders a lit mesh to non-background pixels", async () => {
    const page = await gpuPage();
    const code = await bundleEntry(ENTRY);
    const pixel = await page.evaluate(async (source: string) => {
      // eslint-disable-next-line no-new-func
      const fn = new Function(source);
      fn();
      return (globalThis as any).__rmslRun();
    }, code);

    // The lit red box against a black background must have written red.
    expect(pixel.r).toBeGreaterThan(50);
    expect(pixel.b).toBeLessThan(60);
  }, 60_000);

  it("renders a usampler2D texture to the color target", async () => {
    const page = await gpuPage();
    const code = await bundleEntry(ENTRY_INT);
    const pixel = await page.evaluate(async (source: string) => {
      // eslint-disable-next-line no-new-func
      const fn = new Function(source);
      fn();
      return (globalThis as any).__rmslIntRun();
    }, code);

    // The 1×1 unsigned texture holds (0, 0, 255, 255); reading its texel and
    // widening to a float color writes solid blue, not black.
    expect(pixel.b).toBeGreaterThan(200);
    expect(pixel.r).toBeLessThan(60);
  }, 60_000);

  it("renders a wide line across the canvas via instanced draws", async () => {
    const page = await gpuPage();
    const code = await bundleEntry(ENTRY_LINES);
    const pixel = await page.evaluate(async (source: string) => {
      // eslint-disable-next-line no-new-func
      const fn = new Function(source);
      fn();
      return (globalThis as any).__rmslLineRun();
    }, code);

    // The horizontal red line crosses the center but not the corners.
    expect(pixel.center[0]).toBeGreaterThan(100);
    expect(pixel.center[1]).toBeLessThan(60);
    expect(pixel.corner[0]).toBeLessThan(60);
  }, 60_000);

  it("gives each sampler its own texture when several upload in one draw", async () => {
    const page = await gpuPage();
    const code = await bundleEntry(ENTRY_SAMPLERS);
    const pixel = await page.evaluate(async (source: string) => {
      // eslint-disable-next-line no-new-func
      const fn = new Function(source);
      fn();
      return (globalThis as any).__rmslSamplersRun();
    }, code);

    // Each channel carries the texture its own sampler was given: 20 from the
    // first, 120 from the second, 220 from the third. A sampler left pointing
    // at a neighbour's texture puts that neighbour's value in the channel, and
    // an unsigned sampler left pointing at a float texture is an invalid draw
    // that writes nothing at all.
    expect(pixel.error).toBe(0);
    expect(pixel.r).toBeGreaterThan(10);
    expect(pixel.r).toBeLessThan(40);
    expect(pixel.g).toBeGreaterThan(100);
    expect(pixel.g).toBeLessThan(140);
    expect(pixel.b).toBeGreaterThan(200);
  }, 60_000);
});

afterAll(async () => {
  await releaseGpu();
});
