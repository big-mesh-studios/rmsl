import { compileGlsl } from "@random-mesh/rmsl/glsl";
import { compileJS } from "@random-mesh/rmsl/js";
import { compileWasmRoutine } from "@random-mesh/rmsl/wasm";
import { compileWgsl, wgslUniformLayout } from "@random-mesh/rmsl/wgsl";
import {
  calcMandelbrot,
  calcMandelbrotCpu,
  quadPos,
  u_maxIter,
  u_palette,
  u_pan_hi,
  u_pan_lo,
  u_resolution,
  u_rowOffset,
  u_scale_hi,
  u_scale_lo,
  u_useHighPrecision,
  vertexMain,
} from "./mandelbrotShader";
import { WasmWorkerPool } from "./wasmWorkerPool";

// === Compile RMSL shaders to GLSL, WGSL, JS and WASM ===
// The same RMSL source (mandelbrotColorAt) drives all four: GLSL for
// WebGL, WGSL for WebGPU, and the CPU-target Fn for compileJS/compileWasmRoutine's
// .draw() — one call per pixel, packed into a flat RGBA buffer.
const vsGLSL = compileGlsl.vertex(vertexMain());
const fsGLSL = compileGlsl.fragment(calcMandelbrot());
const jsRenderer = compileJS(() => calcMandelbrotCpu(), { name: "mandelbrotJS", params: [] });
const wasmRenderer = compileWasmRoutine(() => calcMandelbrotCpu(), { name: "mandelbrotWasm", params: [] });

type RendererMode = "webgpu" | "webgl" | "js" | "wasm" | "wasm-pool";
let mode: RendererMode = "webgl";

// A worker-pool wasm renderer that splits one frame's rows across several
// wasm instances sharing one SharedArrayBuffer-backed WebAssembly.Memory —
// see wasmWorkerPool.ts. Created lazily since it needs cross-origin
// isolation (COOP/COEP) to be available at all.
let wasmPool: WasmWorkerPool | null = null;
let wasmPoolBusy = false;
function getWasmPool(): WasmWorkerPool {
  if (!wasmPool) {
    const cores = typeof navigator.hardwareConcurrency === "number" ? navigator.hardwareConcurrency : 4;
    wasmPool = new WasmWorkerPool(Math.max(1, Math.min(8, cores - 1)));
  }
  return wasmPool;
}

// Helper to split a double (f64 number) into two single precision floats (f32)
function splitFloat(v: number): [number, number] {
  const hi = Math.fround(v);
  const lo = Math.fround(v - hi);
  return [hi, lo];
}

// Every uniform the shader reads, keyed by its generated slot name — the one
// thing every backend needs, computed once and handed to whichever
// renderer is active. `w`/`h` is the resolution being rendered at, which is
// each canvas' own backing-store size (see `applyResolution`) — the same
// `renderScale` fraction of the window for all four renderers.
function computeUniformValues(w: number, h: number): Record<string, number | number[]> {
  const scale = zoom / Math.min(w, h);
  const [panXHi, panXLo] = splitFloat(panX);
  const [panYHi, panYLo] = splitFloat(panY);
  const [scaleHi, scaleLo] = splitFloat(scale);
  return {
    [u_resolution.name]: [w, h],
    [u_maxIter.name]: maxIter,
    [u_useHighPrecision.name]: useHighPrecision ? 1 : 0,
    [u_pan_hi.name]: [panXHi, panYHi],
    [u_pan_lo.name]: [panXLo, panYLo],
    [u_scale_hi.name]: [scaleHi, scaleHi],
    [u_scale_lo.name]: [scaleLo, scaleLo],
    [u_palette.name]: palette,
    [u_rowOffset.name]: 0, // non-zero only inside a worker-pool draw() call — see wasmWorkerPool.ts
  };
}

function clamp255(v: number): number {
  return Math.max(0, Math.min(255, Math.round(v)));
}

// === App state ===
let panX = -0.75;
let panY = 0.0;
let zoom = 3.2; // initial view span
let useHighPrecision = false;
let maxIter = 256;
let palette = 0;
// Fraction of the window's own pixel resolution every renderer draws at —
// one shared knob, since each of the four otherwise picked its own
// (WebGL/WebGPU at 100%, JS/WASM at some ad hoc fraction to stay
// interactive), which made "how big a frame is this backend actually
// drawing" impossible to compare or reason about across them.
let renderScale = 0.5;
let lastFrameMs: number | null = null;

// Quad geometry (-1..1)
const quadVerts = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);

// === WebGL2 setup ===
const canvas = document.getElementById("c") as HTMLCanvasElement;
const gl = canvas.getContext("webgl2");

if (!gl) {
  document.body.innerHTML = "<h1 style='color:white;padding:20px'>WebGL2 not supported</h1>";
  throw new Error("WebGL2 not supported");
}

// === CPU (JS/WASM) display: a visible 2D canvas the compiled .draw()
// buffer is blitted into directly — a canvas can only ever have one kind
// of context, so the GPU and CPU paths each need their own element even
// though they show the same view. Its *backing store* (width/height) is
// the shared render resolution, same as the other two canvases; its CSS
// size fills the viewport regardless, so the browser does the upscaling
// for free, the same way it already does for WebGL/WebGPU.
const cpuCanvas = document.getElementById("cCpu") as HTMLCanvasElement;
const cpuCtx = cpuCanvas.getContext("2d")!;

// One canvas' worth of pointer/wheel input drives whichever renderer is
// active, regardless of which canvas is currently visible underneath it.
const interact = document.getElementById("interact") as HTMLDivElement;

// === WebGPU setup (async, best-effort) ===
// A canvas can only ever have one kind of context, so WebGPU gets its own
// element too — three canvases in total, one per GPU-ish target, switched
// by visibility. Set up in the background rather than with a top-level
// await, so WebGL/JS/WASM are usable immediately without waiting on GPU
// adapter/device negotiation.
const webgpuCanvas = document.getElementById("cWebgpu") as HTMLCanvasElement;

type WebGPUState = {
  device: GPUDevice;
  context: GPUCanvasContext;
  pipeline: GPURenderPipeline;
  vertexBuffer: GPUBuffer;
  uniformBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  layout: ReturnType<typeof wgslUniformLayout>;
};
let webgpu: WebGPUState | null = null;
let webgpuError: string | null = null;

// Every uniform the WGSL fragment stage reads, sorted the same way
// `compileWgsl`'s own uniform-struct layout sorts them (by generated slot
// name) — `wgslUniformLayout` has to see the identical list in the identical
// order to compute the byte offsets the compiled struct actually uses.
const wgslDeclaredUniforms = [
  { slot: u_resolution.name, type: "vec2<f32>" },
  { slot: u_maxIter.name, type: "i32" },
  { slot: u_useHighPrecision.name, type: "i32" },
  { slot: u_pan_hi.name, type: "vec2<f32>" },
  { slot: u_pan_lo.name, type: "vec2<f32>" },
  { slot: u_scale_hi.name, type: "vec2<f32>" },
  { slot: u_scale_lo.name, type: "vec2<f32>" },
  { slot: u_palette.name, type: "i32" },
].sort((a, b) => a.slot.localeCompare(b.slot));

/** Byte-pack `values` into `layout`'s struct, per each member's own WGSL type. */
function packUniformBytes(
  layout: ReturnType<typeof wgslUniformLayout>,
  values: Record<string, number | number[]>,
): ArrayBuffer {
  const bytes = new ArrayBuffer(layout.size);
  const view = new DataView(bytes);
  for (const member of layout.members) {
    const value = values[member.name];
    if (value === undefined) continue;
    const components = Array.isArray(value) ? value : [value];
    for (let i = 0; i < components.length; i++) {
      const offset = member.offset + i * 4;
      if (member.type === "i32") view.setInt32(offset, components[i], true);
      else if (member.type === "u32") view.setUint32(offset, components[i], true);
      else view.setFloat32(offset, components[i], true);
    }
  }
  return bytes;
}

async function setupWebGPU(): Promise<void> {
  if (!navigator.gpu) {
    webgpuError = "WebGPU not supported";
    updateUI();
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    webgpuError = "No WebGPU adapter";
    updateUI();
    return;
  }
  const device = await adapter.requestDevice();
  const context = webgpuCanvas.getContext("webgpu");
  if (!context) {
    webgpuError = "WebGPU context unavailable";
    updateUI();
    return;
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "premultiplied" });

  const layout = wgslUniformLayout(wgslDeclaredUniforms);
  const bufferSize = Math.max(16, Math.ceil(layout.size / 16) * 16);

  const vsWGSL = compileWgsl.vertex(vertexMain());
  const fsWGSL = compileWgsl.fragment(calcMandelbrot(), { uniforms: wgslDeclaredUniforms });

  const vertexModule = device.createShaderModule({ code: vsWGSL });
  const fragmentModule = device.createShaderModule({ code: fsWGSL });

  const vertexBuffer = device.createBuffer({
    size: quadVerts.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, quadVerts);

  const uniformBuffer = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
  });
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: vertexModule,
      entryPoint: "main",
      buffers: [
        {
          arrayStride: 8,
          attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
        },
      ],
    },
    fragment: {
      module: fragmentModule,
      entryPoint: "main",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-strip" },
  });

  webgpu = { device, context, pipeline, vertexBuffer, uniformBuffer, bindGroup, layout };
  updateUI();
  requestRender();
}

setupWebGPU().catch((err) => {
  webgpuError = err instanceof Error ? err.message : String(err);
  updateUI();
});

function compileShader(src: string, type: number): WebGLShader {
  const s = gl!.createShader(type)!;
  gl!.shaderSource(s, src);
  gl!.compileShader(s);
  if (!gl!.getShaderParameter(s, gl!.COMPILE_STATUS)) {
    const err = gl!.getShaderInfoLog(s);
    console.error("Shader compile error:", err, "\nSource:\n", src);
    throw new Error("Shader compile error: " + err);
  }
  return s;
}

const vs = compileShader(vsGLSL, gl.VERTEX_SHADER);
const fs = compileShader(fsGLSL, gl.FRAGMENT_SHADER);

const program = gl.createProgram()!;
gl.attachShader(program, vs);
gl.attachShader(program, fs);
gl.linkProgram(program);

if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
  const err = gl.getProgramInfoLog(program);
  console.error("Program link error:", err);
  throw new Error("Program link error: " + err);
}

gl.useProgram(program);

// Set up full-screen quad VAO/VBO
const vao = gl.createVertexArray();
gl.bindVertexArray(vao);
const vbo = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);

const attrLoc = gl.getAttribLocation(program, quadPos.name);
gl.enableVertexAttribArray(attrLoc);
gl.vertexAttribPointer(attrLoc, 2, gl.FLOAT, false, 0, 0);

// Get uniform locations
const locRes = gl.getUniformLocation(program, u_resolution.name);
const locMaxIter = gl.getUniformLocation(program, u_maxIter.name);
const locPrec = gl.getUniformLocation(program, u_useHighPrecision.name);
const locPanHi = gl.getUniformLocation(program, u_pan_hi.name);
const locPanLo = gl.getUniformLocation(program, u_pan_lo.name);
const locScaleHi = gl.getUniformLocation(program, u_scale_hi.name);
const locScaleLo = gl.getUniformLocation(program, u_scale_lo.name);
const locPalette = gl.getUniformLocation(program, u_palette.name);

// === UI elements ===
const precisionBtn = document.getElementById("precisionBtn") as HTMLButtonElement;
const iterInput = document.getElementById("iterInput") as HTMLInputElement;
const iterVal = document.getElementById("iterVal") as HTMLElement;
const scaleInput = document.getElementById("scaleInput") as HTMLInputElement;
const scaleVal = document.getElementById("scaleVal") as HTMLElement;
const paletteSelect = document.getElementById("paletteSelect") as HTMLSelectElement;
const resetBtn = document.getElementById("resetBtn") as HTMLButtonElement;
const hudZoom = document.getElementById("hudZoom") as HTMLElement;
const hudPan = document.getElementById("hudPan") as HTMLElement;
const hudFrame = document.getElementById("hudFrame") as HTMLElement;
const hudRes = document.getElementById("hudRes") as HTMLElement;
const hudWorkersItem = document.getElementById("hudWorkersItem") as HTMLElement;
const hudWorkers = document.getElementById("hudWorkers") as HTMLElement;
const rendererBadge = document.getElementById("rendererBadge") as HTMLElement;
const modeButtons = Array.from(document.querySelectorAll<HTMLButtonElement>(".mode-btn"));

// === Dirty-flag render scheduling ===
let needsRender = false;
let rafId: number | null = null;

function requestRender() {
  needsRender = true;
  if (rafId === null) {
    rafId = requestAnimationFrame(render);
  }
}

const RENDERER_BADGE: Record<RendererMode, string> = {
  webgpu: "RMSL WGSL",
  webgl: "RMSL GLSL",
  js: "RMSL JS (CPU)",
  wasm: "RMSL WASM (CPU)",
  "wasm-pool": "RMSL WASM (Workers)",
};

function updateUI() {
  precisionBtn.classList.toggle("active", useHighPrecision);
  precisionBtn.innerHTML = useHighPrecision
    ? "High Precision: ON <span>(2x Float32)</span>"
    : "High Precision: OFF <span>(1x Float32)</span>";
  iterVal.textContent = maxIter.toString();
  scaleVal.textContent = `${Math.round(renderScale * 100)}%`;

  const scale = zoom / Math.min(canvas.width, canvas.height);
  const zoomFactor = 1.0 / scale;
  hudZoom.textContent = zoomFactor > 1e4 ? zoomFactor.toExponential(2) + "x" : zoomFactor.toFixed(1) + "x";
  hudPan.textContent = `(${panX.toFixed(6)}, ${panY.toFixed(6)})`;
  hudRes.textContent = `${canvas.width}×${canvas.height}`;
  hudFrame.textContent = lastFrameMs === null ? "-" : `${lastFrameMs.toFixed(1)} ms`;

  rendererBadge.textContent = RENDERER_BADGE[mode] + (mode === "webgpu" && webgpuError ? ` — ${webgpuError}` : "");
  for (const btn of modeButtons) {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  }
  const webgpuBtn = modeButtons.find((b) => b.dataset.mode === "webgpu");
  if (webgpuBtn) webgpuBtn.disabled = webgpuError !== null;

  canvas.classList.toggle("hidden", mode !== "webgl");
  cpuCanvas.classList.toggle("hidden", mode !== "js" && mode !== "wasm" && mode !== "wasm-pool");
  webgpuCanvas.classList.toggle("hidden", mode !== "webgpu");

  hudWorkersItem.style.display = mode === "wasm-pool" ? "flex" : "none";
  if (mode === "wasm-pool") hudWorkers.textContent = String(getWasmPool().workerCount || "-");

  requestRender();
}

for (const btn of modeButtons) {
  btn.addEventListener("click", () => {
    const next = btn.dataset.mode as RendererMode;
    if (next === mode) return;
    mode = next;
    updateUI();
  });
}

precisionBtn.addEventListener("click", () => {
  useHighPrecision = !useHighPrecision;
  updateUI();
  requestRender();
});

iterInput.addEventListener("input", (e) => {
  maxIter = parseInt((e.target as HTMLInputElement).value, 10);
  updateUI();
});

scaleInput.addEventListener("input", (e) => {
  renderScale = parseInt((e.target as HTMLInputElement).value, 10) / 100;
  applyResolution();
  updateUI();
});

paletteSelect.addEventListener("change", (e) => {
  palette = parseInt((e.target as HTMLSelectElement).value, 10);
  requestRender();
});

resetBtn.addEventListener("click", () => {
  panX = -0.75;
  panY = 0.0;
  zoom = 3.2;
  updateUI();
});

// Every pointer/wheel coordinate below arrives in CSS pixels (clientX/clientY,
// getBoundingClientRect()), but zoom/pan math happens in render-pixel units
// (canvas.width/height, which is the CSS size times renderScale — see
// applyResolution). Converting once here keeps every call site in the same
// units as the canvas it's actually computing an offset into.
function cssToRenderPx(px: number): number {
  return px * renderScale;
}

// === Pointer & Multi-touch Pinch-Zoom-Pan State ===
const activePointers = new Map<number, { clientX: number; clientY: number }>();
let isPanning = false;
let isPinching = false;
let panStart: { panX: number; panY: number; px: number; py: number } | null = null;
let lastPinchDist = 0;
let lastPinchCenterX = 0;
let lastPinchCenterY = 0;

interact.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  activePointers.set(e.pointerId, { clientX: e.clientX, clientY: e.clientY });

  try {
    interact.setPointerCapture(e.pointerId);
  } catch (err) {
    console.warn("setPointerCapture failed:", err);
  }

  if (activePointers.size === 1) {
    isPanning = true;
    isPinching = false;
    panStart = { panX, panY, px: e.clientX, py: e.clientY };
  } else if (activePointers.size === 2) {
    isPanning = false;
    isPinching = true;
    panStart = null;
    const pointers = [...activePointers.values()];
    lastPinchDist = Math.hypot(pointers[0].clientX - pointers[1].clientX, pointers[0].clientY - pointers[1].clientY);

    const rect = interact.getBoundingClientRect();
    const screenX = cssToRenderPx((pointers[0].clientX + pointers[1].clientX) / 2 - rect.left);
    const screenY = cssToRenderPx((pointers[0].clientY + pointers[1].clientY) / 2 - rect.top);

    lastPinchCenterX = screenX - canvas.width / 2;
    lastPinchCenterY = canvas.height / 2 - screenY;
  }
});

interact.addEventListener("pointermove", (e) => {
  if (!activePointers.has(e.pointerId)) return;
  e.preventDefault();

  activePointers.set(e.pointerId, { clientX: e.clientX, clientY: e.clientY });

  const minDim = Math.min(canvas.width, canvas.height);

  if (isPinching && activePointers.size === 2) {
    const pointers = [...activePointers.values()];
    const dist = Math.hypot(pointers[0].clientX - pointers[1].clientX, pointers[0].clientY - pointers[1].clientY);

    const rect = interact.getBoundingClientRect();
    const screenX = cssToRenderPx((pointers[0].clientX + pointers[1].clientX) / 2 - rect.left);
    const screenY = cssToRenderPx((pointers[0].clientY + pointers[1].clientY) / 2 - rect.top);

    // Current pixel offset relative to canvas center
    const dxCurr = screenX - canvas.width / 2;
    const dyCurr = canvas.height / 2 - screenY; // WebGL +Y is UP

    if (lastPinchDist > 0 && dist > 0) {
      const oldScale = zoom / minDim;

      // Complex number that was under the fingers at the start of frame
      const mX = panX + lastPinchCenterX * oldScale;
      const mY = panY + lastPinchCenterY * oldScale;

      // Update zoom level
      zoom *= lastPinchDist / dist;
      const newScale = zoom / minDim;

      // Update pan so (mX, mY) follows the fingers to dxCurr, dyCurr
      panX = mX - dxCurr * newScale;
      panY = mY - dyCurr * newScale;

      updateUI();
    }

    lastPinchDist = dist;
    lastPinchCenterX = dxCurr;
    lastPinchCenterY = dyCurr;
    return;
  }

  if (isPanning && panStart) {
    const scale = zoom / minDim;
    const dx = cssToRenderPx(e.clientX - panStart.px);
    const dy = cssToRenderPx(e.clientY - panStart.py);

    panX = panStart.panX - dx * scale;
    panY = panStart.panY + dy * scale; // WebGL +Y is UP
    updateUI();
  }
});

function endPointer(pointerId: number) {
  activePointers.delete(pointerId);
  try {
    interact.releasePointerCapture(pointerId);
  } catch {}

  if (activePointers.size < 2) {
    lastPinchDist = 0;
    isPinching = false;
  }
  if (activePointers.size === 1) {
    const p = [...activePointers.values()][0];
    isPanning = true;
    panStart = { panX, panY, px: p.clientX, py: p.clientY };
  } else if (activePointers.size === 0) {
    isPanning = false;
    panStart = null;
  }
}

interact.addEventListener("pointerup", (e) => {
  endPointer(e.pointerId);
});

interact.addEventListener("pointercancel", (e) => {
  endPointer(e.pointerId);
});

interact.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    const rect = interact.getBoundingClientRect();
    const screenX = cssToRenderPx(e.clientX - rect.left);
    const screenY = cssToRenderPx(e.clientY - rect.top);

    const dx = screenX - canvas.width / 2;
    const dy = canvas.height / 2 - screenY;

    const minDim = Math.min(canvas.width, canvas.height);
    const oldScale = zoom / minDim;

    const mX = panX + dx * oldScale;
    const mY = panY + dy * oldScale;

    const factor = Math.pow(1.0015, e.deltaY);
    zoom *= factor;

    const newScale = zoom / minDim;
    panX = mX - dx * newScale;
    panY = mY - dy * newScale;

    updateUI();
  },
  { passive: false },
);

// Every canvas' *backing store* is sized to renderScale fraction of the
// window's own pixels; its CSS size (100vw/100vh, set once in the
// stylesheet) never changes, so the browser stretches whatever resolution
// was actually rendered up to fill the screen — the same trick the CPU
// targets used to do for themselves with an extra offscreen canvas, now
// applied uniformly so all four renderers answer to the one slider.
function applyResolution() {
  const w = Math.max(1, Math.round(window.innerWidth * renderScale));
  const h = Math.max(1, Math.round(window.innerHeight * renderScale));
  for (const cv of [canvas, cpuCanvas, webgpuCanvas]) {
    if (cv.width !== w || cv.height !== h) {
      cv.width = w;
      cv.height = h;
    }
  }
}

function resize() {
  applyResolution();
  updateUI();
}

window.addEventListener("resize", resize);
resize();

// === Render (on-demand, driven by dirty flag) ===
function render() {
  rafId = null;
  if (!needsRender) return;
  needsRender = false;

  if (mode === "webgl") renderWebGL();
  else if (mode === "webgpu") renderWebGPU();
  else if (mode === "wasm-pool") renderWasmPool();
  else renderCpu(mode);
}

function renderWebGL() {
  const start = performance.now();
  const w = canvas.width;
  const h = canvas.height;
  gl!.viewport(0, 0, w, h);
  gl!.clear(gl!.COLOR_BUFFER_BIT);

  const v = computeUniformValues(w, h);
  gl!.uniform2f(locRes, ...(v[u_resolution.name] as [number, number]));
  gl!.uniform1i(locMaxIter, v[u_maxIter.name] as number);
  gl!.uniform1i(locPrec, v[u_useHighPrecision.name] as number);
  gl!.uniform2f(locPanHi, ...(v[u_pan_hi.name] as [number, number]));
  gl!.uniform2f(locPanLo, ...(v[u_pan_lo.name] as [number, number]));
  gl!.uniform2f(locScaleHi, ...(v[u_scale_hi.name] as [number, number]));
  gl!.uniform2f(locScaleLo, ...(v[u_scale_lo.name] as [number, number]));
  gl!.uniform1i(locPalette, v[u_palette.name] as number);

  gl!.bindVertexArray(vao);
  gl!.drawArrays(gl!.TRIANGLE_STRIP, 0, 4);
  // drawArrays returns as soon as the GPU commands are *queued*, not once
  // they finish — finish() blocks the CPU until the GPU actually catches up,
  // trading a little stalling for a frame time that means something.
  gl!.finish();

  lastFrameMs = performance.now() - start;
  hudFrame.textContent = `${lastFrameMs.toFixed(1)} ms`;
}

function renderWebGPU() {
  if (!webgpu) return;
  const start = performance.now();
  const { device, context, pipeline, vertexBuffer, uniformBuffer, bindGroup, layout } = webgpu;
  const w = webgpuCanvas.width;
  const h = webgpuCanvas.height;

  device.queue.writeBuffer(uniformBuffer, 0, packUniformBytes(layout, computeUniformValues(w, h)));

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.setVertexBuffer(0, vertexBuffer);
  pass.draw(4);
  pass.end();
  device.queue.submit([encoder.finish()]);

  // submit() returns immediately too — onSubmittedWorkDone() resolves once
  // the GPU has actually finished, which is the WebGPU way to get the same
  // "wait for it to really be done" signal gl.finish() gives synchronously.
  // Reported a frame or so late rather than blocking the main thread on it.
  device.queue.onSubmittedWorkDone().then(() => {
    lastFrameMs = performance.now() - start;
    hudFrame.textContent = `${lastFrameMs.toFixed(1)} ms`;
  });
}

function renderCpu(cpuMode: "js" | "wasm") {
  const start = performance.now();
  const renderer = cpuMode === "js" ? jsRenderer : wasmRenderer;
  const w = cpuCanvas.width;
  const h = cpuCanvas.height;

  const buffer = renderer.draw({ uniforms: computeUniformValues(w, h) }, w, h);

  const image = cpuCtx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    image.data[i * 4 + 0] = clamp255((buffer[i * 4 + 0] as number) * 255);
    image.data[i * 4 + 1] = clamp255((buffer[i * 4 + 1] as number) * 255);
    image.data[i * 4 + 2] = clamp255((buffer[i * 4 + 2] as number) * 255);
    image.data[i * 4 + 3] = clamp255((buffer[i * 4 + 3] as number) * 255);
  }
  cpuCtx.putImageData(image, 0, 0);

  lastFrameMs = performance.now() - start;
  hudFrame.textContent = `${lastFrameMs.toFixed(1)} ms`;
}

// A frame in flight blocks starting another — if input arrives mid-frame,
// this just re-marks `needsRender` (already how the dirty flag works) and
// the next `render()` tick picks it up once the current frame resolves.
async function renderWasmPool() {
  if (wasmPoolBusy) {
    needsRender = true;
    if (rafId === null) rafId = requestAnimationFrame(render);
    return;
  }
  wasmPoolBusy = true;
  const start = performance.now();
  const w = cpuCanvas.width;
  const h = cpuCanvas.height;

  try {
    const buffer = await getWasmPool().render(computeUniformValues(w, h), w, h);

    const image = cpuCtx.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      image.data[i * 4 + 0] = clamp255((buffer[i * 4 + 0] as number) * 255);
      image.data[i * 4 + 1] = clamp255((buffer[i * 4 + 1] as number) * 255);
      image.data[i * 4 + 2] = clamp255((buffer[i * 4 + 2] as number) * 255);
      image.data[i * 4 + 3] = clamp255((buffer[i * 4 + 3] as number) * 255);
    }
    cpuCtx.putImageData(image, 0, 0);

    lastFrameMs = performance.now() - start;
    hudFrame.textContent = `${lastFrameMs.toFixed(1)} ms`;
    hudWorkers.textContent = String(getWasmPool().workerCount || "-");
  } finally {
    wasmPoolBusy = false;
    if (needsRender) requestRender();
  }
}

// Initial render
requestRender();
