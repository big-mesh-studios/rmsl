import { describe, it, expect, afterAll } from "vitest";
import { build } from "esbuild";
import { webgpuPage, webgpuAvailable, releaseGpu } from "../testing/gpu";

const WEBGPU = await webgpuAvailable();

// Every entry reads its pixels through a 2D canvas rather than copying the
// WebGPU texture: the drawing surface is `bgra8unorm` here and `rgba8unorm`
// elsewhere, and `getImageData` is in the same channel order either way.
const READBACK = `
const readPixel = (canvas, x, y) => {
  const flat = document.createElement("canvas");
  flat.width = canvas.width;
  flat.height = canvas.height;
  const context = flat.getContext("2d");
  context.drawImage(canvas, 0, 0);
  const [r, g, b, a] = context.getImageData(x, y, 1, 1).data;
  return { r, g, b, a };
};
`;

const ENTRY_LIT = `
import { WebGPURenderer, Scene, Mesh, PerspectiveCamera, BoxGeometry,
  MeshStandardMaterial, AmbientLight, DirectionalLight } from "./index";
${READBACK}
globalThis.__rmslGpuLitRun = async () => {
  const canvas = document.createElement("canvas");
  canvas.width = 32;
  canvas.height = 32;
  const renderer = await WebGPURenderer.init(canvas);
  renderer.setClearColor(0x000000);
  const scene = new Scene();
  scene.add(new AmbientLight(0xffffff, 0.3));
  const sun = new DirectionalLight(0xffffff, 1.5);
  sun.position.set(2, 4, 3);
  scene.add(sun);
  scene.add(new Mesh(new BoxGeometry(), new MeshStandardMaterial({ color: 0xff0000, roughness: 0.5 })));
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 4);
  camera.lookAt(0, 0, 0);
  renderer.render(scene, camera);
  await renderer.device.queue.onSubmittedWorkDone();
  return readPixel(canvas, 16, 16);
};
`;

// The same two texels, coordinates and answers as the WebGL renderer's test in
// `renderer.test.ts` and the CPU one in `src/test/program-usage.test.ts`.
const ENTRY_SAMPLER_STATE = `
import { WebGPURenderer, Scene, Mesh, PerspectiveCamera, PlaneGeometry,
  MeshBasicMaterial, DataTexture, NearestFilter, RepeatWrapping } from "./index";
import { vec2 } from "../rmsl";
${READBACK}
globalThis.__rmslGpuSamplerRun = async () => {
  const canvas = document.createElement("canvas");
  canvas.width = 16;
  canvas.height = 16;
  const renderer = await WebGPURenderer.init(canvas);
  renderer.setClearColor(0x000000);
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);

  const texels = () => new Uint8Array([255, 0, 0, 255, 0, 0, 255, 255]);
  const draw = async (texture, x) => {
    const scene = new Scene();
    const material = new MeshBasicMaterial();
    material.fragmentNode = (b) => b.sampler("map", () => texture).texture(vec2(x, 0.5));
    scene.add(new Mesh(new PlaneGeometry(2, 2), material));
    renderer.render(scene, camera);
    await renderer.device.queue.onSubmittedWorkDone();
    return readPixel(canvas, 8, 8);
  };

  const clampedTexture = new DataTexture(texels(), 2, 1);
  clampedTexture.magFilter = NearestFilter;
  const clamped = await draw(clampedTexture, 1.25);
  const nearest = await draw(clampedTexture, 0.5);

  const tiled = new DataTexture(texels(), 2, 1);
  tiled.magFilter = NearestFilter;
  tiled.wrapS = RepeatWrapping;
  const repeated = await draw(tiled, 1.25);

  const linear = await draw(new DataTexture(texels(), 2, 1), 0.5);
  return { clamped, repeated, nearest, linear };
};
`;

// A texture rewritten between renders has to reach the GPU on the second one:
// the pipeline it is bound to is already built and cached by then.
const ENTRY_UPDATE = `
import { WebGPURenderer, Scene, Mesh, PerspectiveCamera, PlaneGeometry,
  MeshBasicMaterial, DataTexture } from "./index";
import { vec2 } from "../rmsl";
${READBACK}
globalThis.__rmslGpuUpdateRun = async () => {
  const canvas = document.createElement("canvas");
  canvas.width = 16;
  canvas.height = 16;
  const renderer = await WebGPURenderer.init(canvas);
  renderer.setClearColor(0x000000);
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);

  const texture = new DataTexture(new Uint8Array([255, 0, 0, 255]), 1, 1);
  const scene = new Scene();
  const material = new MeshBasicMaterial();
  material.fragmentNode = (b) => b.sampler("map", () => texture).texture(vec2(0.5, 0.5));
  scene.add(new Mesh(new PlaneGeometry(2, 2), material));

  const draw = async () => {
    renderer.render(scene, camera);
    await renderer.device.queue.onSubmittedWorkDone();
    return readPixel(canvas, 8, 8);
  };

  const before = await draw();
  texture.image = new Uint8Array([0, 0, 255, 255]);
  texture.needsUpdate = true;
  const after = await draw();
  return { before, after };
};
`;

// Several samplers in one draw: the compiler numbers texture bindings by slot
// name and sampler bindings in the order the graph samples them, and the
// renderer has to number them the same way or a draw reads the wrong image.
const ENTRY_SAMPLERS = `
import { WebGPURenderer, Scene, Mesh, PerspectiveCamera, PlaneGeometry,
  MeshBasicMaterial, DataTexture, NearestFilter } from "./index";
import { float, uvec2, vec2, vec4 } from "../rmsl";
${READBACK}
globalThis.__rmslGpuSamplersRun = async () => {
  const canvas = document.createElement("canvas");
  canvas.width = 16;
  canvas.height = 16;
  const renderer = await WebGPURenderer.init(canvas);
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
  scene.add(new Mesh(new PlaneGeometry(2, 2), material));
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);
  renderer.render(scene, camera);
  await renderer.device.queue.onSubmittedWorkDone();
  return readPixel(canvas, 8, 8);
};
`;

// Instancing: two boxes drawn from one geometry, each with its own transform
// and colour. The instance transform is a mat4 attribute, which WGSL takes as
// four vec4 columns — an instance drawn at the wrong place, or every instance
// drawn at the same place, is what a mistake there looks like.
const ENTRY_INSTANCED = `
import { WebGPURenderer, Scene, InstancedMesh, PerspectiveCamera, BoxGeometry,
  MeshBasicMaterial, Matrix4, Color } from "./index";
${READBACK}
globalThis.__rmslGpuInstancedRun = async () => {
  const canvas = document.createElement("canvas");
  canvas.width = 32;
  canvas.height = 32;
  const renderer = await WebGPURenderer.init(canvas);
  renderer.setClearColor(0x000000);
  const scene = new Scene();
  const mesh = new InstancedMesh(new BoxGeometry(), new MeshBasicMaterial(), 2);
  mesh.setMatrixAt(0, new Matrix4().makeTranslation(-1.1, 0, 0));
  mesh.setMatrixAt(1, new Matrix4().makeTranslation(1.1, 0, 0));
  mesh.instanceMatrix.needsUpdate = true;
  mesh.setColorAt(0, new Color().setRGB(1, 0, 0));
  mesh.setColorAt(1, new Color().setRGB(0, 0, 1));
  mesh.instanceColor.needsUpdate = true;
  scene.add(mesh);
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 4);
  camera.lookAt(0, 0, 0);
  renderer.render(scene, camera);
  await renderer.device.queue.onSubmittedWorkDone();
  return { left: readPixel(canvas, 8, 16), right: readPixel(canvas, 24, 16) };
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

/** Bundle an entry, run it in the WebGPU page, and hand back what it returned. */
async function runInBrowser(source: string, entryPoint: string): Promise<any> {
  const page = await webgpuPage();
  const code = await bundleEntry(source);
  return await page.evaluate(
    async ([bundle, name]: [string, string]) => {
      // eslint-disable-next-line no-new-func
      new Function(bundle)();
      return await (globalThis as any)[name]();
    },
    [code, entryPoint] as [string, string],
  );
}

describe.skipIf(!WEBGPU)("WebGPURenderer on a real adapter", () => {
  it("renders a lit mesh to non-background pixels", async () => {
    const pixel = await runInBrowser(ENTRY_LIT, "__rmslGpuLitRun");
    expect(pixel.r).toBeGreaterThan(50);
    expect(pixel.b).toBeLessThan(60);
  }, 60_000);

  it("wraps and filters a texture the way the texture asks", async () => {
    const result = await runInBrowser(ENTRY_SAMPLER_STATE, "__rmslGpuSamplerRun");

    // Past the right edge: the last texel stretched, then the image tiled.
    expect(result.clamped.b).toBeGreaterThan(200);
    expect(result.clamped.r).toBeLessThan(50);
    expect(result.repeated.r).toBeGreaterThan(200);
    expect(result.repeated.b).toBeLessThan(50);

    // Between the two texel centres: one texel under NearestFilter, a blend of
    // both under the default LinearFilter.
    expect(Math.max(result.nearest.r, result.nearest.b)).toBeGreaterThan(200);
    expect(Math.min(result.nearest.r, result.nearest.b)).toBeLessThan(50);
    expect(result.linear.r).toBeGreaterThan(80);
    expect(result.linear.b).toBeGreaterThan(80);
  }, 60_000);

  it("gives each sampler its own texture when several read in one draw", async () => {
    const pixel = await runInBrowser(ENTRY_SAMPLERS, "__rmslGpuSamplersRun");
    // Each channel comes from a different texture: 20, 120 and 220 of 255.
    expect(pixel.r).toBeGreaterThan(10);
    expect(pixel.r).toBeLessThan(40);
    expect(pixel.g).toBeGreaterThan(100);
    expect(pixel.g).toBeLessThan(140);
    expect(pixel.b).toBeGreaterThan(190);
  }, 60_000);

  it("draws each instance with its own transform and colour", async () => {
    const result = await runInBrowser(ENTRY_INSTANCED, "__rmslGpuInstancedRun");
    // The left box is red and the right one blue, so each instance was placed
    // by its own matrix and tinted by its own colour.
    expect(result.left.r).toBeGreaterThan(100);
    expect(result.left.b).toBeLessThan(60);
    expect(result.right.b).toBeGreaterThan(100);
    expect(result.right.r).toBeLessThan(60);
  }, 60_000);

  it("shows a texture rewritten between renders", async () => {
    const result = await runInBrowser(ENTRY_UPDATE, "__rmslGpuUpdateRun");
    expect(result.before.r).toBeGreaterThan(200);
    expect(result.after.b).toBeGreaterThan(200);
    expect(result.after.r).toBeLessThan(50);
  }, 60_000);
});

afterAll(async () => {
  await releaseGpu();
});
