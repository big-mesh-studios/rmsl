// One RMSL vertex/fragment graph, drawn through two different adapters —
// createGlsl (WebGL) and createWgsl (WebGPU) — to show that swapping the
// render backend is just swapping which adapter setUniform/setAttribute/draw
// get called on, not rewriting the program or the ceremony around it.
import { attribute, cos, Fn, sin, uniform, varying, vec3, vec4 } from "@random-mesh/rmsl";
import { createGlsl } from "@random-mesh/rmsl/glsl";
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

const startTime = performance.now();

function frame() {
  const t = (performance.now() - startTime) / 1000;
  const backend = backendSelect.value === "wgsl" && wgpuReady ? "wgsl" : "glsl";

  if (backend === "glsl") {
    glAdapter.setUniform(time.name, t);
    glAdapter.draw({ mode: "triangles" });
  } else {
    wgpuAdapter.setUniform(time.name, t);
    wgpuAdapter.draw!();
  }

  glCanvas.style.display = backend === "glsl" ? "block" : "none";
  gpuCanvas.style.display = backend === "wgsl" ? "block" : "none";
  statusEl.textContent = backend === "glsl" ? "drawing via createGlsl" : "drawing via createWgsl";

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
