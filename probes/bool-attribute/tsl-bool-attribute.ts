/**
 * Runs three.js's TSL with a `bool` geometry attribute, `attribute("flag", "bool")`,
 * read in the vertex stage and, apart, in the fragment stage, on the WebGPU and
 * the WebGL 2 backend of `WebGPURenderer`, for three.js 0.186.1 and a pinned
 * commit of its dev branch (r187dev). It records the shaders TSL generates, the
 * errors the device or the console reports, and the pixel the draw leaves, for
 * each case below: an array kind and a value that tells a test against zero
 * from a truncation, a boolean vector, and a geometry that lacks the attribute.
 * The vertex stage writes clip w 1 where the attribute reads true, and 0, which
 * draws nothing, where it reads false.
 *
 * The WebGPU run uses the browser the repository's `webgpuPage` launches, the
 * WebGL 2 run the SwiftShader browser its `gpuPage` launches. The probe fetches
 * the three.js builds at the pinned versions below, and serves them itself.
 *
 * Run from the repository root: `node probes/bool-attribute/tsl-bool-attribute.ts`.
 * It writes `tsl-bool-attribute.out.json` beside itself.
 */
import { writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { chromium } from "playwright";

const THREE = {
  "0.186.1": "https://cdn.jsdelivr.net/npm/three@0.186.1/build/",
  "r187dev@58c219bc1e166f27846f25f888a8953452d7cb5e":
    "https://raw.githubusercontent.com/mrdoob/three.js/58c219bc1e166f27846f25f888a8953452d7cb5e/build/",
};
const BUILDS = ["three.webgpu.js", "three.tsl.js", "three.core.js"];
const files = new Map<string, string>();
for (const [version, base] of Object.entries(THREE)) {
  for (const file of BUILDS) {
    const response = await fetch(base + file);
    if (!response.ok) throw new Error(`${base + file}: ${response.status}`);
    files.set(`/${version}/${file}`, await response.text());
  }
}

const page = (version: string) => `<!doctype html><title>probe</title>
<script type="importmap">{"imports":{"three":"/${version}/three.webgpu.js","three/webgpu":"/${version}/three.webgpu.js","three/tsl":"/${version}/three.tsl.js"}}</script>
<script type="module">
import * as T from "three/webgpu";
import { attribute, vec3, vec4, float, positionGeometry } from "three/tsl";
window.run = async (forceWebGL, arrayName, stage, value, type, component) => {
  const shaders = [], errors = [], logs = [];
  const log = (kind) => { const f = console[kind]; console[kind] = (...a) => { logs.push(kind + ": " + a.map(String).join(" ").slice(0, 400)); f.apply(console, a); }; };
  log("warn"); log("error");
  if (navigator.gpu) {
    const create = GPUDevice.prototype.createShaderModule;
    GPUDevice.prototype.createShaderModule = function (d) { shaders.push(d.code); return create.call(this, d); };
    const request = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function (d) { const device = await request.call(this, d); device.addEventListener("uncapturederror", (e) => errors.push(e.error.message.split("\\n").slice(0, 2).join(" | "))); return device; };
  }
  const source = WebGL2RenderingContext.prototype.shaderSource;
  WebGL2RenderingContext.prototype.shaderSource = function (s, code) { shaders.push(code); return source.call(this, s, code); };
  const canvas = document.createElement("canvas"); canvas.width = 4; canvas.height = 4;
  const renderer = new T.WebGPURenderer({ canvas, forceWebGL, antialias: false });
  await renderer.init();
  renderer.setClearColor(0x000000, 0);
  const geometry = new T.BufferGeometry();
  geometry.setAttribute("position", new T.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));
  const width = type === "bool" ? 1 : Number(type.slice(-1));
  if (arrayName !== null) {
    const Arr = { Float32Array, Int32Array, Uint32Array, Uint8Array }[arrayName];
    geometry.setAttribute("flag", new T.BufferAttribute(new Arr(Array(3 * width).fill(value)), width));
  }
  const node = attribute("flag", type);
  const flag = component === null ? node : node[component];
  const material = new T.MeshBasicNodeMaterial();
  if (stage === "vertex") {
    material.vertexNode = vec4(positionGeometry.xy, 0, flag.select(float(1), float(0)));
    material.colorNode = vec4(0, 1, 0, 1);
  } else {
    material.vertexNode = vec4(positionGeometry, 1);
    material.colorNode = vec4(0, flag.select(float(1), float(0)), 0, 1);
  }
  const scene = new T.Scene();
  scene.add(new T.Mesh(geometry, material));
  try { renderer.render(scene, new T.OrthographicCamera()); } catch (e) { errors.push("render threw: " + e.message); }
  if (!forceWebGL) await renderer.backend.device.queue.onSubmittedWorkDone();
  await new Promise((r) => setTimeout(r, 200));
  const flat = document.createElement("canvas"); flat.width = 4; flat.height = 4;
  const c = flat.getContext("2d"); c.drawImage(canvas, 0, 0);
  const [r, g, b, a] = c.getImageData(1, 2, 1, 1).data;
  const pick = (code) => code.split("\\n").filter((l) => /flag|nodeVarying|bool/.test(l)).map((l) => l.trim());
  const adapter = forceWebGL ? null : (renderer.backend.adapter?.info ?? (await navigator.gpu.requestAdapter())?.info);
  return { revision: T.REVISION, adapter: adapter ? { vendor: adapter.vendor, architecture: adapter.architecture, description: adapter.description } : null, shaderLines: [...new Set(shaders.flatMap(pick))], errors, logs: [...new Set(logs)].slice(0, 6), pixel: { r, g, b, a } };
};
window.ready = true;
</script>`;

const server = createServer((request, response) => {
  const path = decodeURIComponent(request.url ?? "/");
  const file = files.get(path);
  if (file !== undefined) {
    response.writeHead(200, { "Content-Type": "text/javascript" });
    response.end(file);
    return;
  }
  response.writeHead(200, { "Content-Type": "text/html" });
  response.end(page(path.slice(1)));
});
await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
const origin = `http://127.0.0.1:${(server.address() as { port: number }).port}/`;

const browsers = {
  webgpu: await chromium.launch({ channel: "chromium" }),
  webgl2: await chromium.launch({ args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"] }),
};
/** Each case: the stage that reads the attribute, its array (null for none), the value at every component, its type, and the component read. */
const CASES: [string, string | null, number, string, string | null][] = [
  ["vertex", "Float32Array", 1, "bool", null],
  ["vertex", "Float32Array", 0, "bool", null],
  ["vertex", "Float32Array", 0.5, "bool", null],
  ["vertex", "Float32Array", NaN, "bool", null],
  ["vertex", "Int32Array", -1, "bool", null],
  ["vertex", "Int32Array", 0, "bool", null],
  ["vertex", "Uint32Array", 2 ** 32 - 1, "bool", null],
  ["vertex", "Uint8Array", 1, "bool", null],
  ["vertex", "Int32Array", -1, "bvec2", "y"],
  ["vertex", "Float32Array", 0.5, "bvec2", "y"],
  ["vertex", null, 0, "bool", null],
  ["vertex", null, 0, "bvec4", "w"],
  ["vertex", null, 0, "bvec4", "x"],
  ["fragment", "Float32Array", 1, "bool", null],
  ["fragment", "Int32Array", 1, "bool", null],
  ["fragment", "Uint32Array", 1, "bool", null],
];
const results: Record<string, unknown> = {};
for (const version of Object.keys(THREE)) {
  for (const backend of ["webgpu", "webgl2"] as const) {
    for (const [stage, array, value, type, component] of CASES) {
      const tab = await browsers[backend].newPage();
      await tab.goto(origin + encodeURIComponent(version), { waitUntil: "domcontentloaded", timeout: 120_000 });
      await tab.waitForFunction(() => (window as any).ready === true, null, { timeout: 120_000 });
      const name = `${version} ${backend} ${stage} ${type}${component ? "." + component : ""} ${array ?? "no data"}${array ? " of " + value : ""}`;
      results[name] = await tab.evaluate(([forceWebGL, ...rest]) => (window as any).run(forceWebGL, ...rest), [
        backend === "webgl2",
        array,
        stage,
        value,
        type,
        component,
      ] as const);
      await tab.close();
    }
  }
}
const out = {
  environment: {
    node: process.version,
    platform: `${process.platform} ${process.arch}`,
    chromium: { webgpu: browsers.webgpu.version(), webgl2: browsers.webgl2.version() },
    three: THREE,
  },
  results,
};
writeFileSync(new URL("./tsl-bool-attribute.out.json", import.meta.url), JSON.stringify(out, null, 2) + "\n");
console.log(JSON.stringify(out, null, 2));
await browsers.webgpu.close();
await browsers.webgl2.close();
server.close();
