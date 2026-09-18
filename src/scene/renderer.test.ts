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

// A 1×1 single-channel unsigned texture, declared with the three.js-style
// RedIntegerFormat + UnsignedByteType pair, must upload as R8UI: reading its
// texel through a usampler2D yields the stored value in .r and zero elsewhere.
const ENTRY_R8UI = `
import { WebGLRenderer, Scene, Mesh, PerspectiveCamera, PlaneGeometry,
  MeshBasicMaterial, DataTexture, RedIntegerFormat, UnsignedByteType } from "./index";
import { uvec2 } from "../rmsl";
globalThis.__rmslR8UIRun = () => {
  const canvas = document.createElement("canvas");
  canvas.width = 16;
  canvas.height = 16;
  const renderer = new WebGLRenderer(canvas, { antialias: false });
  renderer.setClearColor(0x000000);
  const scene = new Scene();
  const material = new MeshBasicMaterial();
  material.fragmentNode = (b) => {
    const tex = b.sampler("data", "usampler2D",
      () => new DataTexture(new Uint8Array([255]), 1, 1, 1, RedIntegerFormat, UnsignedByteType));
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

// Two instances of one box, offset left and right and tinted red and blue via
// instanceColor: a single InstancedMesh must draw both, each with its own
// transform and colour, in one instanced draw.
const ENTRY_INSTANCED = `
import { WebGLRenderer, Scene, PerspectiveCamera, InstancedMesh,
  BoxGeometry, MeshBasicMaterial, Matrix4, Color } from "./index";
globalThis.__rmslInstancedRun = () => {
  const canvas = document.createElement("canvas");
  canvas.width = 32;
  canvas.height = 32;
  const renderer = new WebGLRenderer(canvas, { antialias: false });
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
  const gl = renderer.gl;
  const left = new Uint8Array(4);
  gl.readPixels(8, 16, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, left);
  const right = new Uint8Array(4);
  gl.readPixels(24, 16, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, right);
  return { left: [left[0], left[1], left[2]], right: [right[0], right[1], right[2]] };
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

// A lit mesh rendered with a `mediump` renderer and a `lowp` material pressing
// the material override: if either stage failed to compile or link at the
// lowered precision, the render would throw or `getError()` would carry the
// error, and the box would stay black.
const ENTRY_PRECISION = `
import { WebGLRenderer, Scene, Mesh, PerspectiveCamera, PlaneGeometry,
  MeshBasicMaterial, AmbientLight, DirectionalLight } from "./index";
globalThis.__rmslPrecisionRun = () => {
  const canvas = document.createElement("canvas");
  canvas.width = 32;
  canvas.height = 32;
  const renderer = new WebGLRenderer(canvas, { antialias: false, precision: "mediump" });
  renderer.setClearColor(0x000000);
  const scene = new Scene();
  scene.add(new AmbientLight(0xffffff, 0.3));
  const sun = new DirectionalLight(0xffffff, 1.5);
  sun.position.set(2, 4, 3);
  scene.add(sun);
  const mesh = new Mesh(new PlaneGeometry(2, 2), new MeshBasicMaterial({
    color: 0xff0000, precision: "lowp",
  }));
  scene.add(mesh);
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);
  renderer.render(scene, camera);
  const pixels = new Uint8Array(4);
  const gl = renderer.gl;
  gl.readPixels(16, 16, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  return { r: pixels[0], g: pixels[1], b: pixels[2], error: gl.getError() };
};
`;

// Disposing a texture must free the GL texture the renderer made for it, and
// leave the texture itself usable: the second render re-creates and re-uploads
// it, so the blue texel still reaches the color target.
const ENTRY_TEXTURE_DISPOSE = `
import { WebGLRenderer, Scene, Mesh, PerspectiveCamera, PlaneGeometry,
  MeshBasicMaterial, DataTexture } from "./index";
import { vec2 } from "../rmsl";
globalThis.__rmslTextureDisposeRun = () => {
  const canvas = document.createElement("canvas");
  canvas.width = 16;
  canvas.height = 16;
  const renderer = new WebGLRenderer(canvas, { antialias: false });
  renderer.setClearColor(0x000000);
  const scene = new Scene();
  const texture = new DataTexture(new Uint8Array([0, 0, 220, 255]), 1, 1);
  const material = new MeshBasicMaterial();
  material.fragmentNode = (b) => b.sampler("map", () => texture).texture(vec2(0.5, 0.5));
  const mesh = new Mesh(new PlaneGeometry(2, 2), material);
  scene.add(mesh);
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);
  renderer.render(scene, camera);

  const gl = renderer.gl;
  const glTexture = renderer.textures.get(texture);
  const liveBefore = gl.isTexture(glTexture);
  texture.dispose();
  const liveAfter = gl.isTexture(glTexture);
  const trackedAfter = renderer.textures.size;

  renderer.render(scene, camera);
  const pixels = new Uint8Array(4);
  gl.readPixels(8, 8, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  return {
    liveBefore, liveAfter, trackedAfter,
    trackedAgain: renderer.textures.size,
    b: pixels[2], error: gl.getError(),
  };
};
`;

// A texture's filtering and wrapping reach the driver: a 2x1 red/blue texture
// sampled past its right edge clamps to the last texel, and tiles once the
// texture asks to repeat. Sampling between the two texel centres blends them
// under LinearFilter and picks one under NearestFilter.
const ENTRY_SAMPLER_STATE = `
import { WebGLRenderer, Scene, Mesh, PerspectiveCamera, PlaneGeometry,
  MeshBasicMaterial, DataTexture, NearestFilter, RepeatWrapping } from "./index";
import { vec2 } from "../rmsl";
globalThis.__rmslSamplerStateRun = () => {
  const canvas = document.createElement("canvas");
  canvas.width = 16;
  canvas.height = 16;
  const renderer = new WebGLRenderer(canvas, { antialias: false });
  renderer.setClearColor(0x000000);
  const gl = renderer.gl;
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);

  const read = (scene, coordinate) => {
    renderer.render(scene, camera);
    const pixels = new Uint8Array(4);
    gl.readPixels(8, 8, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    return { r: pixels[0], b: pixels[2], error: gl.getError() };
  };

  const sceneAt = (texture, x) => {
    const scene = new Scene();
    const material = new MeshBasicMaterial();
    material.fragmentNode = (b) => b.sampler("map", () => texture).texture(vec2(x, 0.5));
    scene.add(new Mesh(new PlaneGeometry(2, 2), material));
    return scene;
  };

  // Two texels: red on the left, blue on the right.
  const texture = new DataTexture(new Uint8Array([255, 0, 0, 255, 0, 0, 255, 255]), 2, 1);
  texture.magFilter = NearestFilter;
  texture.minFilter = NearestFilter;

  const clamped = read(sceneAt(texture, 1.25), 1.25);
  texture.wrapS = RepeatWrapping;
  texture.needsUpdate = true;
  const repeated = read(sceneAt(texture, 1.25), 1.25);

  const nearest = read(sceneAt(texture, 0.5), 0.5);
  const smooth = new DataTexture(new Uint8Array([255, 0, 0, 255, 0, 0, 255, 255]), 2, 1);
  const linear = read(sceneAt(smooth, 0.5), 0.5);

  return { clamped, repeated, nearest, linear };
};
`;

/**
 * A draw redirects into an offscreen render target, whose pixels are read back
 * without ever touching the canvas's drawing buffer; the next un-targeted
 * render is back on the canvas.
 */
const ENTRY_TARGET = `
import { WebGLRenderer, Scene, Mesh, PerspectiveCamera, PlaneGeometry,
  MeshBasicMaterial, WebGLRenderTarget } from "./index";
globalThis.__rmslTargetRun = () => {
  const canvas = document.createElement("canvas");
  canvas.width = 32;
  canvas.height = 32;
  const renderer = new WebGLRenderer(canvas, { antialias: false });
  renderer.setClearColor(0x000000);
  const scene = new Scene();
  scene.add(new Mesh(new PlaneGeometry(2, 2),
    new MeshBasicMaterial({ color: 0xff0000 })));
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);

  const target = new WebGLRenderTarget(8, 8);
  renderer.render(scene, camera, target);
  const offscreen = renderer.readPixels(target);
  const gl = renderer.gl;
  const canvasPixels = new Uint8Array(4);
  gl.readPixels(16, 16, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, canvasPixels);

  renderer.render(scene, camera);
  gl.readPixels(16, 16, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, canvasPixels);

  const center = (4 + 4 * 8) * 4;
  return {
    targetR: offscreen[center],
    canvasR: canvasPixels[0],
    error: gl.getError(),
  };
};
`;

/**
 * The async, PBO-backed readback returns the same pixels as `readPixels` for
 * the same render, without stalling the pipeline to get them.
 */
const ENTRY_TARGET_ASYNC = `
import { WebGLRenderer, Scene, Mesh, PerspectiveCamera, PlaneGeometry,
  MeshBasicMaterial, WebGLRenderTarget } from "./index";
globalThis.__rmslTargetAsyncRun = async () => {
  const canvas = document.createElement("canvas");
  canvas.width = 32;
  canvas.height = 32;
  const renderer = new WebGLRenderer(canvas, { antialias: false });
  renderer.setClearColor(0x000000);
  const scene = new Scene();
  scene.add(new Mesh(new PlaneGeometry(2, 2),
    new MeshBasicMaterial({ color: 0xff0000 })));
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);

  const target = new WebGLRenderTarget(8, 8);
  renderer.render(scene, camera, target);
  const offscreen = await renderer.readPixelsAsync(target);
  const gl = renderer.gl;

  const center = (4 + 4 * 8) * 4;
  return {
    targetR: offscreen[center],
    error: gl.getError(),
  };
};
`;

/**
 * Two meshes share one uploaded geometry and draw adjacent slices of it via
 * their draw ranges: left half red, right half blue. A draw-range not honored
 * draws the whole geometry for each — the blue (drawn last) covers the red
 * half, and the left pixel reads blue instead of red.
 */
const ENTRY_RANGE = `
import { WebGLRenderer, Scene, Mesh, PerspectiveCamera, BufferGeometry,
  BufferAttribute, MeshBasicMaterial, Side } from "./index";
globalThis.__rmslRangeRun = () => {
  const canvas = document.createElement("canvas");
  canvas.width = 32;
  canvas.height = 32;
  const renderer = new WebGLRenderer(canvas, { antialias: false });
  renderer.setClearColor(0x000000);
  const scene = new Scene();

  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(new Float32Array([
    -1, -1, 0,  -1, 1, 0,  0, 1, 0,  0, -1, 0,
    0, -1, 0,  0, 1, 0,  1, 1, 0,  1, -1, 0,
  ]), 3));
  geometry.setAttribute("normal", new BufferAttribute(new Float32Array([
    0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
    0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
  ]), 3));
  geometry.setAttribute("uv", new BufferAttribute(new Float32Array([
    0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0,
  ]), 2));
  geometry.setIndex(new BufferAttribute(new Uint16Array([
    0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7,
  ]), 1));

  const left = new Mesh(geometry, new MeshBasicMaterial({ color: 0xff0000, side: Side.DoubleSide }));
  left.drawRange = { start: 0, count: 6 };
  const right = new Mesh(geometry, new MeshBasicMaterial({ color: 0x0000ff, side: Side.DoubleSide }));
  right.drawRange = { start: 6, count: 6 };
  scene.add(left, right);

  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 2);
  camera.lookAt(0, 0, 0);
  renderer.render(scene, camera);

  const pixels = new Uint8Array(4);
  const gl = renderer.gl;
  gl.readPixels(8, 16, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  const leftPix = { r: pixels[0], b: pixels[2] };
  gl.readPixels(24, 16, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  const rightPix = { r: pixels[0], b: pixels[2] };
  return { leftPix, rightPix, error: gl.getError() };
};
`;

// Disposing a geometry must free the vertex and index buffers the renderer made
// for it, and leave the geometry itself usable: the second render uploads its
// attributes into fresh buffers, so the blue quad still reaches the color
// target.
const ENTRY_GEOMETRY_DISPOSE = `
import { WebGLRenderer, Scene, Mesh, PerspectiveCamera, PlaneGeometry,
  MeshBasicMaterial } from "./index";
globalThis.__rmslGeometryDisposeRun = () => {
  const canvas = document.createElement("canvas");
  canvas.width = 16;
  canvas.height = 16;
  const renderer = new WebGLRenderer(canvas, { antialias: false });
  renderer.setClearColor(0x000000);
  const scene = new Scene();
  const geometry = new PlaneGeometry(2, 2);
  const mesh = new Mesh(geometry, new MeshBasicMaterial({ color: 0x0000dc }));
  scene.add(mesh);
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);
  renderer.render(scene, camera);

  const gl = renderer.gl;
  const uploaded = renderer.geometryBuffers.get(geometry);
  const held = [...uploaded.attributes.values(), uploaded.index];
  const liveBefore = held.every((buffer) => gl.isBuffer(buffer));
  geometry.dispose();
  const liveAfter = held.some((buffer) => gl.isBuffer(buffer));
  const trackedAfter = renderer.geometryBuffers.size;

  renderer.render(scene, camera);
  const pixels = new Uint8Array(4);
  gl.readPixels(8, 8, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  return {
    liveBefore, liveAfter, trackedAfter,
    trackedAgain: renderer.geometryBuffers.size,
    b: pixels[2], error: gl.getError(),
  };
};
`;

// Two quads whose colour arrives in a narrower attribute than a float: the left
// from a normalized Uint8Array bound as unorm8x4, the right from half floats
// held in a Uint16Array that says so with `format`. Both encode the same
// colour, so both must read back as it. Bound as floats — what every attribute
// used to be — each would read four times its own bytes per vertex and draw
// nothing recognisable; bound unnormalized, the byte 255 would arrive as 255.0
// rather than 1.0 and clamp the left quad's green to full.
const ENTRY_PACKED = `
import { WebGLRenderer, Scene, Mesh, PerspectiveCamera, BufferGeometry,
  BufferAttribute, MeshBasicMaterial, Side } from "./index";
import { vec4 } from "../rmsl";
globalThis.__rmslPackedRun = () => {
  const canvas = document.createElement("canvas");
  canvas.width = 32;
  canvas.height = 32;
  const renderer = new WebGLRenderer(canvas, { antialias: false });
  renderer.setClearColor(0x000000);
  const scene = new Scene();

  const quad = (x0, x1) => new Float32Array([
    x0, -1, 0,  x1, -1, 0,  x1, 1, 0,  x0, 1, 0,
  ]);
  const indices = () => new BufferAttribute(new Uint16Array([0, 1, 2, 0, 2, 3]), 1);

  // Left: one byte a channel, scaled into 0..1 on the way in.
  const byteGeometry = new BufferGeometry();
  byteGeometry.setAttribute("position", new BufferAttribute(quad(-1.6, -0.2), 3));
  byteGeometry.setAttribute("tint", new BufferAttribute(new Uint8Array([
    255, 64, 0, 255,  255, 64, 0, 255,  255, 64, 0, 255,  255, 64, 0, 255,
  ]), 4, true));
  byteGeometry.setIndex(indices());
  const byteMaterial = new MeshBasicMaterial({ side: Side.DoubleSide });
  byteMaterial.vertexNode = (b) => {
    b.varying("tint", "vec4").assign(b.attribute("tint", "vec4"));
    return b.projectionMatrix.mul(b.viewMatrix.mul(b.modelMatrix.mul(vec4(b.position, 1))));
  };
  byteMaterial.fragmentNode = (b) => b.varying("tint", "vec4");
  scene.add(new Mesh(byteGeometry, byteMaterial));

  // Right: 1.0 and 0.25 as half floats, which only \`format\` can distinguish
  // from a Uint16Array of normalized integers.
  const halfGeometry = new BufferGeometry();
  halfGeometry.setAttribute("position", new BufferAttribute(quad(0.2, 1.6), 3));
  const halves = new BufferAttribute(new Uint16Array([
    0x3c00, 0x3400,  0x3c00, 0x3400,  0x3c00, 0x3400,  0x3c00, 0x3400,
  ]), 2);
  halves.format = "float16x2";
  halfGeometry.setAttribute("shade", halves);
  halfGeometry.setIndex(indices());
  const halfMaterial = new MeshBasicMaterial({ side: Side.DoubleSide });
  halfMaterial.vertexNode = (b) => {
    b.varying("shade", "vec2").assign(b.attribute("shade", "vec2"));
    return b.projectionMatrix.mul(b.viewMatrix.mul(b.modelMatrix.mul(vec4(b.position, 1))));
  };
  halfMaterial.fragmentNode = (b) => {
    const shade = b.varying("shade", "vec2");
    return vec4(shade.x, shade.y, 0, 1);
  };
  scene.add(new Mesh(halfGeometry, halfMaterial));

  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 4);
  camera.lookAt(0, 0, 0);
  renderer.render(scene, camera);

  const gl = renderer.gl;
  const pixels = new Uint8Array(4);
  gl.readPixels(8, 16, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  const bytes = [pixels[0], pixels[1], pixels[2]];
  gl.readPixels(24, 16, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  const half = [pixels[0], pixels[1], pixels[2]];
  return { bytes, half, error: gl.getError() };
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

  it("reads a packed attribute at its own width, not as floats", async () => {
    const page = await gpuPage();
    const code = await bundleEntry(ENTRY_PACKED);
    const result = await page.evaluate(async (source: string) => {
      // eslint-disable-next-line no-new-func
      const fn = new Function(source);
      fn();
      return (globalThis as any).__rmslPackedRun();
    }, code);

    expect(result.error).toBe(0);
    // 255 and 64 as bytes are 1.0 and ~0.251 in the shader, which come back as
    // the bytes they started from. Read unnormalized both would clamp to full.
    expect(result.bytes[0]).toBeGreaterThan(250);
    expect(result.bytes[1]).toBeGreaterThan(56);
    expect(result.bytes[1]).toBeLessThan(72);
    expect(result.bytes[2]).toBeLessThan(6);
    // The half floats encode the same colour by a different route.
    expect(result.half[0]).toBeGreaterThan(250);
    expect(result.half[1]).toBeGreaterThan(56);
    expect(result.half[1]).toBeLessThan(72);
    expect(result.half[2]).toBeLessThan(6);
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

  it("renders an R8UI DataTexture from its single stored byte", async () => {
    const page = await gpuPage();
    const code = await bundleEntry(ENTRY_R8UI);
    const pixel = await page.evaluate(async (source: string) => {
      // eslint-disable-next-line no-new-func
      const fn = new Function(source);
      fn();
      return (globalThis as any).__rmslR8UIRun();
    }, code);

    // The single-channel texel holds 255 in .r, so the BGRA read must come
    // back solid red — proving it uploaded as R8UI, not a 4-byte RGBA8UI.
    expect(pixel.r).toBeGreaterThan(200);
    expect(pixel.g).toBeLessThan(60);
    expect(pixel.b).toBeLessThan(60);
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

  it("draws each InstancedMesh instance with its own transform and color", async () => {
    const page = await gpuPage();
    const code = await bundleEntry(ENTRY_INSTANCED);
    const pixel = await page.evaluate(async (source: string) => {
      // eslint-disable-next-line no-new-func
      const fn = new Function(source);
      fn();
      return (globalThis as any).__rmslInstancedRun();
    }, code);

    // The left box is red where the right box is blue; the shared material is
    // tinted per instance rather than drawing the same box twice.
    expect(pixel.left[0]).toBeGreaterThan(100);
    expect(pixel.left[2]).toBeLessThan(60);
    expect(pixel.right[2]).toBeGreaterThan(100);
    expect(pixel.right[0]).toBeLessThan(60);
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

  it("frees a disposed texture and re-uploads it on the next render", async () => {
    const page = await gpuPage();
    const code = await bundleEntry(ENTRY_TEXTURE_DISPOSE);
    const result = await page.evaluate(async (source: string) => {
      // eslint-disable-next-line no-new-func
      const fn = new Function(source);
      fn();
      return (globalThis as any).__rmslTextureDisposeRun();
    }, code);

    expect(result.liveBefore).toBe(true);
    expect(result.liveAfter).toBe(false);
    expect(result.trackedAfter).toBe(0);
    // The next render made a new GL texture for the same Texture object.
    expect(result.trackedAgain).toBe(1);
    expect(result.b).toBeGreaterThan(150);
    expect(result.error).toBe(0);
  }, 60_000);

  it("wraps and filters a texture the way the texture asks", async () => {
    const page = await gpuPage();
    const code = await bundleEntry(ENTRY_SAMPLER_STATE);
    const result = await page.evaluate(async (source: string) => {
      // eslint-disable-next-line no-new-func
      const fn = new Function(source);
      fn();
      return (globalThis as any).__rmslSamplerStateRun();
    }, code);

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
    expect(result.linear.error).toBe(0);
  }, 60_000);

  it("renders with lowered precision from the renderer and a material override", async () => {
    const page = await gpuPage();
    const code = await bundleEntry(ENTRY_PRECISION);
    const pixel = await page.evaluate(async (source: string) => {
      // eslint-disable-next-line no-new-func
      const fn = new Function(source);
      fn();
      return (globalThis as any).__rmslPrecisionRun();
    }, code);

    // The mediump renderer + lowp material shaders linked and drew the red
    // box with no WebGL error; a failed link/compile would have thrown.
    expect(pixel.error).toBe(0);
    expect(pixel.r).toBeGreaterThan(150);
    expect(pixel.g).toBeLessThan(60);
    expect(pixel.b).toBeLessThan(60);
  }, 60_000);

  it("renders into a render target and reads its pixels back", async () => {
    const page = await gpuPage();
    const code = await bundleEntry(ENTRY_TARGET);
    const pixel = await page.evaluate(async (source: string) => {
      // eslint-disable-next-line no-new-func
      const fn = new Function(source);
      fn();
      return (globalThis as any).__rmslTargetRun();
    }, code);

    // The draw went to the offscreen target, not the canvas, and read back red;
    // the following un-targeted render puts it on the canvas again.
    expect(pixel.error).toBe(0);
    expect(pixel.targetR).toBeGreaterThan(150);
    expect(pixel.canvasR).toBeGreaterThan(150);
  }, 60_000);

  it("reads a render target's pixels back asynchronously without stalling", async () => {
    const page = await gpuPage();
    const code = await bundleEntry(ENTRY_TARGET_ASYNC);
    const pixel = await page.evaluate(async (source: string) => {
      // eslint-disable-next-line no-new-func
      const fn = new Function(source);
      fn();
      return (globalThis as any).__rmslTargetAsyncRun();
    }, code);

    expect(pixel.error).toBe(0);
    expect(pixel.targetR).toBeGreaterThan(150);
  }, 60_000);

  it("frees a disposed geometry's buffers and re-uploads them on the next render", async () => {
    const page = await gpuPage();
    const code = await bundleEntry(ENTRY_GEOMETRY_DISPOSE);
    const result = await page.evaluate(async (source: string) => {
      // eslint-disable-next-line no-new-func
      const fn = new Function(source);
      fn();
      return (globalThis as any).__rmslGeometryDisposeRun();
    }, code);

    expect(result.liveBefore).toBe(true);
    expect(result.liveAfter).toBe(false);
    expect(result.trackedAfter).toBe(0);
    // The next render made fresh buffers for the same geometry object.
    expect(result.trackedAgain).toBe(1);
    expect(result.b).toBeGreaterThan(150);
    expect(result.error).toBe(0);
  }, 60_000);

  it("draws only the slice a mesh's drawRange selects from a shared geometry", async () => {
    const page = await gpuPage();
    const code = await bundleEntry(ENTRY_RANGE);
    const result = await page.evaluate(async (source: string) => {
      // eslint-disable-next-line no-new-func
      const fn = new Function(source);
      fn();
      return (globalThis as any).__rmslRangeRun();
    }, code);

    // Left red, right blue: the blue mesh drawn afterwards stays off the red
    // half only because its range starts after the left quad's indices.
    expect(result.error).toBe(0);
    expect(result.leftPix.r).toBeGreaterThan(150);
    expect(result.leftPix.b).toBeLessThan(60);
    expect(result.rightPix.b).toBeGreaterThan(150);
    expect(result.rightPix.r).toBeLessThan(60);
  }, 60_000);
});

afterAll(async () => {
  await releaseGpu();
});
