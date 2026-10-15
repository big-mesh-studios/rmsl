/**
 * Question: how many samples do three.js's renderers draw with, by default and
 * with `antialias: true`, and what does its `WebGLRenderer` do on a canvas that
 * already has a WebGL 2 context?
 *
 * `WebGLRenderer` on a fresh canvas, with no options, `antialias: false` and
 * `antialias: true`: the context's `antialias`, its drawing buffer's
 * `SAMPLES`, and the alphas of row 0 after drawing an opaque red rectangle
 * whose edges cross the centres of columns 0 and 2 of a 4×4 canvas cleared to
 * transparent. Then on a canvas whose context was made first by
 * `getContext("webgl2", { antialias })`, each way round. Every
 * `console.warn` and `console.error` during the case is kept, and whether the
 * constructor threw. `WebGPURenderer` from `three/webgpu`, after `init`: the
 * backend it chose and its `samples`, with no options and with
 * `antialias: true`.
 *
 * Runs three.js 0.186.0 from `probes/single-sample/three` (`npm ci` there
 * first). Measures three.js and the platform, no rmsl.
 */
import { PROBES, report, runInGpuPage, runInWebGpuPage } from "./harness";

const ENTRY = `
import * as THREE from "three";
import { WebGPURenderer } from "three/webgpu";
const logged = [];
for (const level of ["warn", "error"]) {
  const original = console[level];
  console[level] = (...args) => { logged.push(level + ": " + args.join(" ")); original.apply(console, args); };
}
const drawn = (renderer) => {
  renderer.setSize(4, 4, false);
  renderer.setClearColor(0x000000, 0);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute([-0.75, -2, 0, 0.25, -2, 0, -0.75, 2, 0, -0.75, 2, 0, 0.25, -2, 0, 0.25, 2, 0], 3));
  const scene = new THREE.Scene();
  scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0xff0000, side: THREE.DoubleSide })));
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  camera.position.z = 1;
  renderer.render(scene, camera);
  const gl = renderer.getContext();
  const pixels = new Uint8Array(16);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.readPixels(0, 0, 4, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  return {
    antialias: gl.getContextAttributes().antialias,
    samples: gl.getParameter(gl.SAMPLES),
    alphas: [3, 7, 11, 15].map((i) => pixels[i]).join(" "),
    error: gl.getError(),
  };
};
const webgl = (options, first) => {
  logged.length = 0;
  const canvas = document.createElement("canvas");
  if (first) canvas.getContext("webgl2", first);
  try {
    return { ...drawn(new THREE.WebGLRenderer({ canvas, ...options })), logged: [...logged] };
  } catch (error) {
    return { threw: String(error), logged: [...logged] };
  }
};
const webgpu = async (options) => {
  const renderer = new WebGPURenderer({ canvas: document.createElement("canvas"), ...options });
  await renderer.init();
  return { backend: renderer.backend.isWebGPUBackend ? "webgpu" : "webgl", samples: renderer.samples };
};
globalThis.__probe = async () => ({
  revision: THREE.REVISION,
  webglRenderer: {
    byDefault: webgl({}),
    antialiasFalse: webgl({ antialias: false }),
    antialiasTrue: webgl({ antialias: true }),
    antialiasTrueOnSingleSampledContext: webgl({ antialias: true }, { antialias: false }),
    byDefaultOnMultisampledContext: webgl({}, { antialias: true }),
  },
  webgpuRenderer: navigator.gpu
    ? { byDefault: await webgpu({}), antialiasTrue: await webgpu({ antialias: true }) }
    : "no navigator.gpu",
});
`;

await report("three-renderer-samples", async () => ({
  webglPage: await runInGpuPage(ENTRY, "__probe", PROBES + "three"),
  webgpuPage: await runInWebGpuPage(ENTRY, "__probe", PROBES + "three"),
}));
