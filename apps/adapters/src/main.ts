// One shape, four adapters — createGlsl (WebGL) and createWgsl (WebGPU)
// draw it directly; createJs and createWasm can't draw at all, so for
// those two the quad's vertices are rotated by a storage()/invocationIndex()
// compute pass first, and the result is fed into a second, "static" GLSL
// draw pipeline as an ordinary attribute — a compute adapter's output
// chained straight into a draw adapter's input, no readback ceremony
// beyond the plain compute(out) call every CPU adapter already has.
import {
  attribute,
  cos,
  Fn,
  invocationIndex,
  sin,
  storage,
  uniform,
  varying,
  vec3,
  vec4,
} from "@random-mesh/rmsl";
import { createGlsl } from "@random-mesh/rmsl/glsl";
import { createJs, type CpuAdapter } from "@random-mesh/rmsl/js";
import { createWasm } from "@random-mesh/rmsl/wasm";
import { createWgsl } from "@random-mesh/rmsl/wgsl";

const glCanvas = document.createElement("canvas");
const gpuCanvas = document.createElement("canvas");
for (const c of [glCanvas, gpuCanvas]) {
  c.width = 512;
  c.height = 512;
  c.style.width = "512px";
  c.style.height = "512px";
}
document.body.append(glCanvas, gpuCanvas);

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

// === Compute programs (createJs / createWasm) — rotate the same quad's
// vertices on the CPU instead, one storage()/invocationIndex() call per
// vertex. ===
const inX = storage("in_x", "float", { access: "read" });
const inY = storage("in_y", "float", { access: "read" });
const outX = storage("out_x", "float", { access: "read_write" });
const outY = storage("out_y", "float", { access: "read_write" });
const rotTime = uniform("float");

const computeRoot = Fn(() => {
  const i = invocationIndex();
  const c = cos(rotTime);
  const s = sin(rotTime);
  outX.element(i).assign(inX.element(i).mul(c).sub(inY.element(i).mul(s)));
  outY.element(i).assign(inX.element(i).mul(s).add(inY.element(i).mul(c)));
  return outX.element(i);
})();

const jsCompute = createJs(computeRoot, { name: "rotateJs" });
const wasmCompute = createWasm(computeRoot, { name: "rotateWasm" });

const QUAD_X = new Float32Array([-0.6, 0.6, -0.6, 0.6, -0.6, 0.6]);
const QUAD_Y = new Float32Array([-0.6, -0.6, 0.6, -0.6, 0.6, 0.6]);
// out_x/out_y are write-only (never read), but the CPU adapters still need
// setAttribute called on them at least once — that's what allocates the
// backing array the compiled step writes into. compute() is then called
// with no `out`, so it mutates these two arrays in place rather than
// copying every registered slot (in_x/in_y included) into a same-shaped
// `out` object.
const computedX = new Float32Array(6);
const computedY = new Float32Array(6);
const rotatedQuad = new Float32Array(12);
let computeSetUp: CpuAdapter | null = null;

function computeRotatedQuad(adapter: CpuAdapter, t: number): Float32Array {
  if (computeSetUp !== adapter) {
    adapter.setAttribute(outX.name, computedX);
    adapter.setAttribute(outY.name, computedY);
    computeSetUp = adapter;
  }
  adapter.setAttribute(inX.name, QUAD_X);
  adapter.setAttribute(inY.name, QUAD_Y);
  adapter.setUniform(rotTime.name, t);
  adapter.compute();
  for (let i = 0; i < 6; i++) {
    rotatedQuad[i * 2] = computedX[i];
    rotatedQuad[i * 2 + 1] = computedY[i];
  }
  return rotatedQuad;
}

// A second, "static" GLSL draw pipeline for the compute backends — it
// just colors and draws whatever positions it's given, no rotation of its
// own, since the compute pass already did that. Shares glCanvas with
// `glAdapter`: attach() reuses the canvas's existing WebGL2 context rather
// than creating a second one.
const staticPos = attribute("vec2");
const colorTime = uniform("float");
const staticVColor = varying("vec3");

const staticVertexRoot = Fn(() => {
  staticVColor.assign(
    vec3(staticPos.x.mul(0.5).add(0.5), staticPos.y.mul(0.5).add(0.5), sin(colorTime).mul(0.5).add(0.5)),
  );
  return vec4(staticPos, 0.0, 1.0);
})();
const staticFragmentRoot = Fn(() => vec4(staticVColor, 1.0))();

const staticDrawAdapter = createGlsl(staticVertexRoot, staticFragmentRoot);
staticDrawAdapter.attach(glCanvas);

const startTime = performance.now();

function frame() {
  const t = (performance.now() - startTime) / 1000;
  const requested = backendSelect.value;
  const backend = requested === "wgsl" && !wgpuReady ? "glsl" : requested;

  if (backend === "glsl") {
    glAdapter.setUniform(time.name, t);
    glAdapter.draw({ mode: "triangles" });
  } else if (backend === "wgsl") {
    wgpuAdapter.setUniform(time.name, t);
    wgpuAdapter.draw();
  } else {
    const rotated = computeRotatedQuad(backend === "js" ? jsCompute : wasmCompute, t);
    staticDrawAdapter.setAttribute(staticPos.name, rotated);
    staticDrawAdapter.setUniform(colorTime.name, t);
    staticDrawAdapter.draw({ mode: "triangles" });
  }

  glCanvas.style.display = backend === "wgsl" ? "none" : "block";
  gpuCanvas.style.display = backend === "wgsl" ? "block" : "none";
  statusEl.textContent =
    backend === "glsl"
      ? "drawing via createGlsl"
      : backend === "wgsl"
        ? "drawing via createWgsl"
        : `computing via create${backend === "js" ? "Js" : "Wasm"}, drawing via createGlsl`;

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
