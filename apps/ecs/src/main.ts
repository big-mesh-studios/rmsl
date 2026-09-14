import { compileJS, compileWasm, compileWGSL, wgslUniformLayout } from "@random-mesh/rmsl";
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

function stepCPU(fn: typeof jsStep, dt: number) {
  for (let i = 0; i < N; i++) {
    const res = fn({
      attributes: {
        [slots.posX]: posX[i],
        [slots.posY]: posY[i],
        [slots.velX]: velX[i],
        [slots.velY]: velY[i],
      },
      uniforms: {
        [slots.width]: canvas.width,
        [slots.height]: canvas.height,
        [slots.dt]: dt,
      },
    }) as any;
    posX[i] = res.outputs[slots.outPosX];
    posY[i] = res.outputs[slots.outPosY];
    velX[i] = res.outputs[slots.outVelX];
    velY[i] = res.outputs[slots.outVelY];
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
};

async function setupWgslDevice(): Promise<WgslDevice | null> {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) return null;
  const device = await adapter.requestDevice();

  const wgsl = compileWGSL.compute(system.program.root);
  const module = device.createShaderModule({ code: wgsl });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  return { device, pipeline };
}

// Entity buffers are sized to N, so a count change tears these down and
// rebuilds them — the device and pipeline above are reused as-is.
function createWgslBuffers({ device, pipeline }: WgslDevice, n: number): WgslBackend {
  const BYTES = n * 4;
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const inPosX = device.createBuffer({ size: BYTES, usage });
  const inPosY = device.createBuffer({ size: BYTES, usage });
  const inVelX = device.createBuffer({ size: BYTES, usage });
  const inVelY = device.createBuffer({ size: BYTES, usage });
  const outPosX = device.createBuffer({ size: BYTES, usage });
  const outPosY = device.createBuffer({ size: BYTES, usage });
  const outVelX = device.createBuffer({ size: BYTES, usage });
  const outVelY = device.createBuffer({ size: BYTES, usage });

  // Attributes are bound in the order they were created (posX, posY, velX,
  // velY), then outputs in their creation order — the same order
  // compileWGSL.compute assigns @group(1) bindings in.
  const bindGroup1 = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: inPosX } },
      { binding: 1, resource: { buffer: inPosY } },
      { binding: 2, resource: { buffer: inVelX } },
      { binding: 3, resource: { buffer: inVelY } },
      { binding: 4, resource: { buffer: outPosX } },
      { binding: 5, resource: { buffer: outPosY } },
      { binding: 6, resource: { buffer: outVelX } },
      { binding: 7, resource: { buffer: outVelY } },
    ],
  });

  const layout = wgslUniformLayout([
    { slot: slots.width, type: "f32" },
    { slot: slots.height, type: "f32" },
    { slot: slots.dt, type: "f32" },
  ]);
  const uniformBuffer = device.createBuffer({
    size: Math.max(16, layout.size),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const bindGroup0 = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });
  const uniformScratch = new Float32Array(uniformBuffer.size / 4);
  const offsetOf = (slot: string) => layout.members.find((m) => m.name === slot)!.offset / 4;

  const staging = device.createBuffer({
    size: BYTES * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // Seed the input buffers from whatever the CPU arrays currently hold, so a
  // switch into this backend continues the same simulation.
  function upload() {
    device.queue.writeBuffer(inPosX, 0, posX);
    device.queue.writeBuffer(inPosY, 0, posY);
    device.queue.writeBuffer(inVelX, 0, velX);
    device.queue.writeBuffer(inVelY, 0, velY);
  }
  upload();

  const workgroups = Math.ceil(n / 64);

  async function step(dt: number) {
    uniformScratch[offsetOf(slots.width)] = canvas.width;
    uniformScratch[offsetOf(slots.height)] = canvas.height;
    uniformScratch[offsetOf(slots.dt)] = dt;
    device.queue.writeBuffer(uniformBuffer, 0, uniformScratch);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup0);
    pass.setBindGroup(1, bindGroup1);
    pass.dispatchWorkgroups(workgroups);
    pass.end();

    // Feed this frame's outputs back in as next frame's inputs, GPU-side.
    encoder.copyBufferToBuffer(outPosX, 0, inPosX, 0, BYTES);
    encoder.copyBufferToBuffer(outPosY, 0, inPosY, 0, BYTES);
    encoder.copyBufferToBuffer(outVelX, 0, inVelX, 0, BYTES);
    encoder.copyBufferToBuffer(outVelY, 0, inVelY, 0, BYTES);

    // ...and read them back so the shared CPU arrays stay the render/backend
    // source of truth, at the cost of a CPU-GPU sync every frame.
    encoder.copyBufferToBuffer(outPosX, 0, staging, 0, BYTES);
    encoder.copyBufferToBuffer(outPosY, 0, staging, BYTES, BYTES);
    encoder.copyBufferToBuffer(outVelX, 0, staging, BYTES * 2, BYTES);
    encoder.copyBufferToBuffer(outVelY, 0, staging, BYTES * 3, BYTES);
    device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const mapped = new Float32Array(staging.getMappedRange());
    posX.set(mapped.subarray(0, n));
    posY.set(mapped.subarray(n, n * 2));
    velX.set(mapped.subarray(n * 2, n * 3));
    velY.set(mapped.subarray(n * 3, n * 4));
    staging.unmap();
  }

  function destroy() {
    for (const buf of [inPosX, inPosY, inVelX, inVelY, outPosX, outPosY, outVelX, outVelY, uniformBuffer, staging]) {
      buf.destroy();
    }
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
