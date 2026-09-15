import { compileJS, compileWasm } from "@random-mesh/rmsl";
import { compile, type WgslResource } from "@random-mesh/rmsl/wgsl";
import { createEcsSystem } from "./system";

const canvas = document.createElement("canvas");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
canvas.style.position = "fixed";
canvas.style.inset = "0";
canvas.style.zIndex = "-1";
document.body.appendChild(canvas);
const ctx2d = canvas.getContext("2d")!;

window.addEventListener("resize", () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
});

const backendSelect = document.getElementById("backend") as HTMLSelectElement;
const entityCountInput = document.getElementById("entityCount") as HTMLInputElement;
const statsEl = document.getElementById("stats")!;

// === Shared entity state (SoA) — the source of truth every backend reads
// from and writes back to, so switching backends mid-run is seamless.
// Reseeded (and the WGSL buffers rebuilt to match) whenever the entity count
// input changes. ===
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

// === One rmsl Fn, compiled three ways ===
const system = createEcsSystem();
const { slots } = system;

const jsStep = compileJS(() => system.program.root, { name: "ecsSystem", params: [] });
const wasmStep = compileWasm(() => system.program.root, { name: "ecsSystem", params: [] });

// storage()/invocationIndex() give js/wasm the same per-invocation model WGSL
// gets: one call per entity, `index` naming which one, `storages` the whole
// backing arrays it reads/writes directly in place — no separate output
// buffers or res.outputs indirection needed.
function stepCPU(fn: typeof jsStep, dt: number) {
  const storages = { [slots.posX]: posX, [slots.posY]: posY, [slots.velX]: velX, [slots.velY]: velY };
  const uniforms = { [slots.width]: canvas.width, [slots.height]: canvas.height, [slots.dt]: dt };
  for (let i = 0; i < N; i++) {
    fn({ storages, uniforms, index: i } as any);
  }
}

// === WGSL compute backend ===
type WgslBackend = {
  step(dt: number): Promise<void>;
  destroy(): void;
};
type WgslDevice = {
  device: GPUDevice;
  pipeline: GPUComputePipeline;
  resources: WgslResource[];
};

async function setupWgslDevice(): Promise<WgslDevice | null> {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) return null;
  const device = await adapter.requestDevice();

  const program = compile({ stage: "compute", workgroupSize: 64 }, system.program.root);
  const module = device.createShaderModule({ code: program.code });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: program.entryPoint },
  });

  return { device, pipeline, resources: program.resources };
}

// Entity buffers are sized to N, so a count change tears these down and
// rebuilds them — the device and pipeline above are reused as-is.
function createWgslBuffers({ device, pipeline, resources }: WgslDevice, n: number): WgslBackend {
  const BYTES = n * 4;
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

  const storageResources = resources.filter((r): r is Extract<WgslResource, { kind: "storage" }> => r.kind === "storage");
  const uniformResources = resources.filter((r): r is Extract<WgslResource, { kind: "uniform" }> => r.kind === "uniform");

  // read_write storage: one buffer per slot, updated in place — no separate
  // in/out pair and no GPU-side feed-forward copy needed, since every
  // invocation only ever touches its own element.
  const storageBuffers = new Map(storageResources.map((r) => [r.name, device.createBuffer({ size: BYTES, usage })]));
  const arraysBySlot = (): Record<string, Float32Array<ArrayBuffer>> => ({
    [slots.posX]: posX,
    [slots.posY]: posY,
    [slots.velX]: velX,
    [slots.velY]: velY,
  });

  const bindGroup1 = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(1),
    entries: storageResources.map((r) => ({ binding: r.binding, resource: { buffer: storageBuffers.get(r.name)! } })),
  });

  const uniformSize = Math.max(16, ...uniformResources.map((r) => r.offset + r.size));
  const uniformBuffer = device.createBuffer({
    size: uniformSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const bindGroup0 = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });
  const uniformScratch = new Float32Array(uniformSize / 4);
  const uniformOffsetOf = (slot: string) => uniformResources.find((r) => r.name === slot)!.offset / 4;

  const staging = device.createBuffer({
    size: BYTES,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // Seed every storage buffer from whatever the CPU arrays currently hold, so
  // a switch into this backend continues the same simulation.
  function upload() {
    const arrays = arraysBySlot();
    for (const [slot, buffer] of storageBuffers) {
      device.queue.writeBuffer(buffer, 0, arrays[slot]);
    }
  }
  upload();

  const workgroups = Math.ceil(n / 64);

  async function step(dt: number) {
    uniformScratch[uniformOffsetOf(slots.width)] = canvas.width;
    uniformScratch[uniformOffsetOf(slots.height)] = canvas.height;
    uniformScratch[uniformOffsetOf(slots.dt)] = dt;
    device.queue.writeBuffer(uniformBuffer, 0, uniformScratch);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup0);
    pass.setBindGroup(1, bindGroup1);
    pass.dispatchWorkgroups(workgroups);
    pass.end();
    device.queue.submit([encoder.finish()]);

    // Read every storage buffer back so the shared CPU arrays stay the
    // render/backend source of truth, at the cost of a CPU-GPU sync per
    // buffer per frame.
    const arrays = arraysBySlot();
    for (const [slot, buffer] of storageBuffers) {
      const readEncoder = device.createCommandEncoder();
      readEncoder.copyBufferToBuffer(buffer, 0, staging, 0, BYTES);
      device.queue.submit([readEncoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      arrays[slot].set(new Float32Array(staging.getMappedRange()));
      staging.unmap();
    }
  }

  function destroy() {
    for (const buffer of storageBuffers.values()) buffer.destroy();
    uniformBuffer.destroy();
    staging.destroy();
  }

  return { step, destroy };
}

let wgslDevice: WgslDevice | null = null;
let wgslBackend: WgslBackend | null = null;
let wgslReady = false;
setupWgslDevice().then((gpu) => {
  wgslDevice = gpu;
  wgslReady = true;
  if (!gpu) {
    const opt = backendSelect.querySelector('option[value="wgsl"]') as HTMLOptionElement;
    opt.disabled = true;
    opt.textContent += " (unavailable)";
    return;
  }
  wgslBackend = createWgslBuffers(gpu, N);
});

entityCountInput.addEventListener("change", () => {
  const n = Math.max(1, Math.min(200000, Math.floor(Number(entityCountInput.value) || 1)));
  entityCountInput.value = String(n);
  seed(n);
  if (wgslDevice) {
    wgslBackend?.destroy();
    wgslBackend = createWgslBuffers(wgslDevice, n);
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
  if (backend === "js") {
    stepCPU(jsStep, dt);
  } else if (backend === "wasm") {
    stepCPU(wasmStep, dt);
  } else if (backend === "wgsl" && wgslBackend) {
    await wgslBackend.step(dt);
  }
  const stepMs = performance.now() - t0;

  draw();

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
  if (backendSelect.value === "wgsl" && wgslReady && !wgslBackend) {
    backendSelect.value = "js";
  }
});

requestAnimationFrame(frame);
