import { compileJS, compileWasm, createCpu } from "@random-mesh/rmsl";
import { createWgsl } from "@random-mesh/rmsl/wgsl";
import { createGpuRenderer } from "./gpu-renderer";
import { createEcsSystem } from "./system";

const canvas = document.createElement("canvas");
const gpuCanvas = document.createElement("canvas");
for (const c of [canvas, gpuCanvas]) {
  c.width = window.innerWidth;
  c.height = window.innerHeight;
  c.style.position = "fixed";
  c.style.inset = "0";
  c.style.zIndex = "-1";
  document.body.appendChild(c);
}
// Not `.hidden` — the page's own `canvas { display: block; }` rule (an
// author style) overrides the UA stylesheet's `[hidden] { display: none }`,
// so toggling the attribute would have no visual effect here.
gpuCanvas.style.display = "none";
const ctx2d = canvas.getContext("2d")!;

window.addEventListener("resize", () => {
  for (const c of [canvas, gpuCanvas]) {
    c.width = window.innerWidth;
    c.height = window.innerHeight;
  }
});

const backendSelect = document.getElementById("backend") as HTMLSelectElement;
const entityCountInput = document.getElementById("entityCount") as HTMLInputElement;
const statsEl = document.getElementById("stats")!;

// === Shared entity state (SoA), read/written by js and wasm directly.
// wgsl only syncs against this once, on entry (see uploadStorages below) —
// each backend runs at its own pace, not kept in lockstep every frame.
// Reseeded whenever the entity count input changes; the WGSL adapter notices
// the buffer length change on the next setAttribute and resizes itself. ===
let N = Number(entityCountInput.value);
let posX = new Float32Array(N);
let posY = new Float32Array(N);
let velX = new Float32Array(N);
let velY = new Float32Array(N);

function seed(n: number) {
  N = n;
  posX = new Float32Array(N);
  posY = new Float32Array(N);
  velX = new Float32Array(N);
  velY = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    posX[i] = Math.random() * canvas.width;
    posY[i] = Math.random() * canvas.height;
    velX[i] = (Math.random() - 0.5) * 200;
    velY[i] = (Math.random() - 0.5) * 200;
  }
}
seed(N);

// === One rmsl Fn, adapted three ways ===
const system = createEcsSystem();
const { slots } = system;

const jsAdapter = createCpu(compileJS(() => system.program.root, { name: "ecsSystem", params: [] }));
const wasmAdapter = createCpu(compileWasm(() => system.program.root, { name: "ecsSystem", params: [] }));

// Re-set every frame: cheap (a handful of object-field assignments), and
// it means a fresh posX/posY/velX/velY from seed() (entity count changed)
// is picked up without a separate "resync" path.
function stepCPU(adapter: typeof jsAdapter, dt: number) {
  adapter.setAttribute(slots.posX, posX);
  adapter.setAttribute(slots.posY, posY);
  adapter.setAttribute(slots.velX, velX);
  adapter.setAttribute(slots.velY, velY);
  adapter.setUniform(slots.width, canvas.width);
  adapter.setUniform(slots.height, canvas.height);
  adapter.setUniform(slots.dt, dt);
  adapter.compute!();
}

// === WGSL compute backend, via the adapter ===
// This demo measures each backend in its own optimal state, not seamless
// mid-run continuity — so while wgsl is active, `compute()` is called with
// no `out`: it dispatches and stays fully GPU-resident, and gpuRenderer
// reads the same buffers directly (`buffer(slot)`), with no per-frame
// upload or readback at all. The CPU-side posX/posY/velX/velY only get
// synced once, when wgsl is first selected — switching away leaves them at
// whatever they held when you switched in, which is fine here.
const wgslAdapter = createWgsl({ compute: system.program.root });
let wgslReady = false;
let gpuRenderer: ReturnType<typeof createGpuRenderer> | null = null;

function currentStorages() {
  return { [slots.posX]: posX, [slots.posY]: posY, [slots.velX]: velX, [slots.velY]: velY };
}

function uploadStorages() {
  const storages = currentStorages();
  for (const slot in storages) wgslAdapter.setAttribute(slot, storages[slot]);
}

function rebuildGpuRenderer() {
  const device = wgslAdapter.device();
  if (!device) return;
  gpuRenderer?.destroy();
  gpuRenderer = createGpuRenderer(device, gpuCanvas, wgslAdapter.buffer(slots.posX)!, wgslAdapter.buffer(slots.posY)!);
}

wgslAdapter
  .attach()
  .then(() => {
    uploadStorages();
    rebuildGpuRenderer();
    wgslReady = true;
  })
  .catch(() => {
    const opt = backendSelect.querySelector('option[value="wgsl"]') as HTMLOptionElement;
    opt.disabled = true;
    opt.textContent += " (unavailable)";
  });

async function stepWGSL(dt: number) {
  wgslAdapter.setUniform(slots.width, canvas.width);
  wgslAdapter.setUniform(slots.height, canvas.height);
  wgslAdapter.setUniform(slots.dt, dt);
  await wgslAdapter.compute!();
}

entityCountInput.addEventListener("change", () => {
  const n = Math.max(1, Math.min(200000, Math.floor(Number(entityCountInput.value) || 1)));
  entityCountInput.value = String(n);
  seed(n);
  if (wgslReady) {
    uploadStorages();
    rebuildGpuRenderer();
  }
});

// === Render loop ===
function draw() {
  ctx2d.fillStyle = "#101318";
  ctx2d.fillRect(0, 0, canvas.width, canvas.height);
  ctx2d.fillStyle = "#7fd4ff";
  for (let i = 0; i < N; i++) {
    ctx2d.fillRect(posX[i] - 1, posY[i] - 1, 2, 2);
  }
}

let lastTime = performance.now();
let running = true;

// A single frame's step time is too noisy (GC pauses, the WGSL readback
// stall, display refresh jitter) to compare backends by eye, so the readout
// is a plain average over a time window rather than a per-frame or EMA
// value — updated a few times a second instead of every frame.
const STATS_WINDOW_MS = 500;
let windowStart = performance.now();
let windowStepMsTotal = 0;
let windowFrameCount = 0;
let lastBackend = backendSelect.value;

function resetStatsWindow(now: number) {
  windowStart = now;
  windowStepMsTotal = 0;
  windowFrameCount = 0;
}

async function frame(now: number) {
  if (!running) return;
  const dt = Math.min((now - lastTime) / 1000, 1 / 30);
  lastTime = now;

  const t0 = performance.now();
  const backend = backendSelect.value;
  try {
    if (backend === "js") {
      stepCPU(jsAdapter, dt);
      draw();
    } else if (backend === "wasm") {
      stepCPU(wasmAdapter, dt);
      draw();
    } else if (backend === "wgsl" && wgslReady) {
      if (lastBackend !== "wgsl") uploadStorages();
      await stepWGSL(dt);
      gpuRenderer!.render(N, canvas.width, canvas.height);
    }
  } catch (err) {
    // A bad frame (a GPU hiccup, or a step racing an entity-count change
    // mid-flight) shouldn't kill the loop forever — requestAnimationFrame
    // below is what keeps it alive, and this is the one place standing
    // between a thrown/rejected step and that call never happening.
    console.error("[ecs] frame error", err);
  }
  const stepMs = performance.now() - t0;

  canvas.style.display = backend === "wgsl" ? "none" : "block";
  gpuCanvas.style.display = backend === "wgsl" ? "block" : "none";

  // A backend change makes the window's average meaningless, so it starts over.
  if (backend !== lastBackend) {
    lastBackend = backend;
    resetStatsWindow(now);
  }
  windowStepMsTotal += stepMs;
  windowFrameCount++;
  if (now - windowStart >= STATS_WINDOW_MS) {
    const avgStepMs = windowStepMsTotal / windowFrameCount;
    const avgFps = windowFrameCount / ((now - windowStart) / 1000);
    statsEl.textContent = `${N} entities | ${avgFps.toFixed(0)} fps | step ${avgStepMs.toFixed(2)} ms`;
    resetStatsWindow(now);
  }

  requestAnimationFrame(frame);
}

backendSelect.addEventListener("change", () => {
  if (backendSelect.value === "wgsl" && !wgslReady) {
    backendSelect.value = "js";
  }
});

requestAnimationFrame(frame);
