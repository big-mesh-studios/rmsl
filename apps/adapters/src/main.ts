// Two shapes, four adapters. createGlsl (WebGL) and createWgsl (WebGPU)
// draw a rotating quad directly, through their vertex/fragment stages —
// the same "return a value" contract every backend's per-element work
// ultimately reduces to. createJs and createWasm can't rasterize a
// triangle, but their compiled result already has a second capability
// besides storage()-based compute: `.draw()`, a fragCoord() program
// evaluated once per pixel over the whole canvas (see cpu.ts's
// CpuRenderer). createJs/createWasm's own `draw` wraps exactly that, so
// the CPU backends here draw a full-screen color gradient instead of a
// quad — a genuinely different shape, not a lesser stand-in for the
// vertex-based one.
import { attribute, cos, Fn, fragCoord, sin, uniform, varying, vec3, vec4 } from "@random-mesh/rmsl";
import { createGlsl } from "@random-mesh/rmsl/glsl";
import { createJs } from "@random-mesh/rmsl/js";
import { createWasm } from "@random-mesh/rmsl/wasm";
import { createWgsl } from "@random-mesh/rmsl/wgsl";

const glCanvas = document.createElement("canvas");
const gpuCanvas = document.createElement("canvas");
const cpuCanvas = document.createElement("canvas");
for (const c of [glCanvas, gpuCanvas, cpuCanvas]) {
  c.width = 512;
  c.height = 512;
  c.style.width = "512px";
  c.style.height = "512px";
}
document.body.append(glCanvas, gpuCanvas, cpuCanvas);

const backendSelect = document.getElementById("backend") as HTMLSelectElement;
const statusEl = document.getElementById("status")!;

// === Draw programs (createGlsl / createWgsl) — rotate and color a quad
// entirely inside the vertex/fragment stage. ===
const pos = attribute("vec2");
const time = uniform("float");
const vColor = varying("vec3");

const vertexRoot = Fn(() => {
  const c = cos(time);
  const s = sin(time);
  const x = pos.x.mul(c).sub(pos.y.mul(s));
  const y = pos.x.mul(s).add(pos.y.mul(c));
  vColor.assign(vec3(pos.x.mul(0.5).add(0.5), pos.y.mul(0.5).add(0.5), sin(time).mul(0.5).add(0.5)));
  return vec4(x, y, 0.0, 1.0);
})();

const fragmentRoot = Fn(() => vec4(vColor, 1.0))();

const TRIANGLE_STRIP_QUAD = new Float32Array([-0.6, -0.6, 0.6, -0.6, -0.6, 0.6, 0.6, -0.6, -0.6, 0.6, 0.6, 0.6]);

const glAdapter = createGlsl(vertexRoot, fragmentRoot);
glAdapter.attach(glCanvas);
glAdapter.setAttribute(pos.name, TRIANGLE_STRIP_QUAD);

const wgpuAdapter = createWgsl({ vertex: vertexRoot, fragment: fragmentRoot });
let wgpuReady = false;
wgpuAdapter
  .attach(gpuCanvas)
  .then(() => {
    wgpuAdapter.setAttribute(pos.name, TRIANGLE_STRIP_QUAD);
    wgpuReady = true;
  })
  .catch(() => {
    const opt = backendSelect.querySelector('option[value="wgsl"]') as HTMLOptionElement;
    opt.disabled = true;
    opt.textContent += " (unavailable)";
  });

// === Draw programs (createJs / createWasm) — a full-screen color
// gradient, one fragCoord() evaluation per pixel. No attribute, no
// vertex stage: a CPU adapter's draw has nothing to rasterize with. ===
const resolution = uniform("vec2");
const cpuTime = uniform("float");

const cpuDrawRoot = Fn(() => {
  const uv = fragCoord().div(resolution);
  return vec4(uv.x, uv.y, sin(cpuTime).mul(0.5).add(0.5), 1.0);
})();

const jsAdapter = createJs({ draw: cpuDrawRoot, drawName: "cpuDraw" });
jsAdapter.attach(cpuCanvas);

const wasmAdapter = createWasm({ draw: cpuDrawRoot, drawName: "cpuDraw" });
wasmAdapter.attach(cpuCanvas);

const startTime = performance.now();
const fpsEl = document.getElementById("fps")!;

// A raw per-frame FPS jumps around too much to read — averaged over a
// short window and refreshed a few times a second instead, same as
// apps/ecs's stats readout.
const FPS_WINDOW_MS = 500;
let fpsWindowStart = performance.now();
let fpsWindowFrameCount = 0;

function frame(now: number) {
  const t = (now - startTime) / 1000;
  const requested = backendSelect.value;
  const backend = requested === "wgsl" && !wgpuReady ? "glsl" : requested;

  if (backend === "glsl") {
    glAdapter.setUniform(time, t);
    glAdapter.draw({ mode: "triangles" });
  } else if (backend === "wgsl") {
    wgpuAdapter.setUniform(time, t);
    wgpuAdapter.draw();
  } else {
    const adapter = backend === "js" ? jsAdapter : wasmAdapter;
    adapter.setUniform(resolution, [cpuCanvas.width, cpuCanvas.height]);
    adapter.setUniform(cpuTime, t);
    adapter.draw();
  }

  glCanvas.style.display = backend === "glsl" ? "block" : "none";
  gpuCanvas.style.display = backend === "wgsl" ? "block" : "none";
  cpuCanvas.style.display = backend === "js" || backend === "wasm" ? "block" : "none";
  statusEl.textContent =
    backend === "glsl"
      ? "drawing via createGlsl"
      : backend === "wgsl"
        ? "drawing via createWgsl"
        : `drawing via create${backend === "js" ? "Js" : "Wasm"}`;

  fpsWindowFrameCount++;
  if (now - fpsWindowStart >= FPS_WINDOW_MS) {
    const fps = fpsWindowFrameCount / ((now - fpsWindowStart) / 1000);
    fpsEl.textContent = `${fps.toFixed(0)} fps`;
    fpsWindowStart = now;
    fpsWindowFrameCount = 0;
  }

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
