// End-to-end smoke test for the WebGL renderer: bundle the scene library with
// esbuild, draw a lit mesh in a real Chromium page, and read pixels back.
//
// Opt-in via RMSL_GPU=1 like the other GPU layers — skipped otherwise.

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
});

afterAll(async () => {
  await releaseGpu();
});
