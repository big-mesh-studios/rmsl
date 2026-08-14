import { compileWGSL } from "@random-mesh/rmsl";
import {
  vertexMain, calcColourAndDepth,
  quadPos, cameraProjectionMatrix, cameraViewMatrix,
  cameraProjectionMatrixInverse, cameraWorldMatrix, cameraPosition,
  quadVerts,
  mat4Perspective, mat4LookAt, mat4Inverse,
} from "../../shared/shader";

// === Compile shaders ===
let vsWGSL = compileWGSL.vertex(vertexMain());
let fsWGSL = compileWGSL.fragment(calcColourAndDepth());

// === Orbital camera state ===
let theta = 0;
let phi = 0.6;
let radius = 10;
let isDragging = false;
let lastMX = 0;
let lastMY = 0;

function getViewMatrix(): Float32Array<ArrayBuffer> {
  let eyeX = radius * Math.sin(theta) * Math.cos(phi);
  let eyeY = radius * Math.sin(phi);
  let eyeZ = radius * Math.cos(theta) * Math.cos(phi);
  return mat4LookAt(eyeX, eyeY, eyeZ, 0, 0, 0, 0, 1, 0);
}

function getCameraPosition(): [number, number, number] {
  return [
    radius * Math.sin(theta) * Math.cos(phi),
    radius * Math.sin(phi),
    radius * Math.cos(theta) * Math.cos(phi),
  ];
}

// === WebGPU setup ===
let canvas = document.createElement("canvas");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
canvas.style.position = "fixed";
canvas.style.top = "0";
canvas.style.left = "0";
canvas.style.zIndex = "0";
canvas.style.touchAction = "none";
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
document.body.appendChild(canvas);

let adapter = await navigator.gpu?.requestAdapter();
if (!adapter) {
  document.body.innerHTML = "<h1>WebGPU not supported</h1>";
  throw new Error("WebGPU not supported");
}

let device = await adapter.requestDevice();
let context = canvas.getContext("webgpu");
if (!context) {
  document.body.innerHTML = "<h1>WebGPU context not supported</h1>";
  throw new Error("WebGPU context not supported");
}

let format = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format, alphaMode: "premultiplied" });

let depthTexture = device.createTexture({
  size: [canvas.width, canvas.height],
  format: "depth24plus",
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});

let vertexModule = device.createShaderModule({ code: vsWGSL });
let fragmentModule = device.createShaderModule({ code: fsWGSL });

let vertexBuffer = device.createBuffer({
  size: quadVerts.byteLength,
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, 0, quadVerts);

let uniformSizes = [64, 64, 64, 64, 16];
let uniformBuffers = uniformSizes.map(size =>
  device.createBuffer({
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
);

let bindGroupLayout = device.createBindGroupLayout({
  entries: uniformBuffers.map((_, i) => ({
    binding: i,
    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
    buffer: { type: "uniform" },
  })),
});

let bindGroup = device.createBindGroup({
  layout: bindGroupLayout,
  entries: uniformBuffers.map((buf, i) => ({
    binding: i,
    resource: { buffer: buf },
  })),
});

let pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

let pipeline = device.createRenderPipeline({
  layout: pipelineLayout,
  vertex: {
    module: vertexModule,
    entryPoint: "main",
    buffers: [{
      arrayStride: 8,
      attributes: [{
        shaderLocation: 0,
        offset: 0,
        format: "float32x2",
      }],
    }],
  },
  fragment: {
    module: fragmentModule,
    entryPoint: "main",
    targets: [{ format }],
  },
  primitive: {
    topology: "triangle-strip",
  },
  depthStencil: {
    format: "depth24plus",
    depthWriteEnabled: true,
    depthCompare: "always",
  },
});

// === Pointer / wheel events ===
canvas.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  isDragging = true;
  lastMX = e.clientX;
  lastMY = e.clientY;
  try {
    canvas.setPointerCapture(e.pointerId);
  } catch (err) {
    console.warn("setPointerCapture failed:", err);
  }
});
canvas.addEventListener("pointermove", (e) => {
  e.preventDefault();
  if (!isDragging) return;
  let dx = e.clientX - lastMX;
  let dy = e.clientY - lastMY;
  theta -= dx * 0.005;
  phi = Math.max(-1.5, Math.min(1.5, phi - dy * 0.005));
  lastMX = e.clientX;
  lastMY = e.clientY;
});
canvas.addEventListener("pointerup", (e) => {
  isDragging = false;
  try {
    canvas.releasePointerCapture(e.pointerId);
  } catch (err) {
    console.warn("releasePointerCapture failed:", err);
  }
});
canvas.addEventListener("pointercancel", (e) => {
  isDragging = false;
  try {
    canvas.releasePointerCapture(e.pointerId);
  } catch (err) {
    console.warn("releasePointerCapture failed:", err);
  }
});
canvas.addEventListener("pointerleave", (e) => {
  isDragging = false;
  try {
    canvas.releasePointerCapture(e.pointerId);
  } catch (err) {
    console.warn("releasePointerCapture failed:", err);
  }
});
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  radius *= 1 + e.deltaY * 0.001;
  radius = Math.max(0.5, Math.min(500, radius));
});

// === Resize ===
window.addEventListener("resize", () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  context!.configure({ device, format, alphaMode: "premultiplied" });
  depthTexture.destroy();
  depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
});

// === Render loop ===
function render() {
  let w = canvas.width;
  let h = canvas.height;

  let aspect = w / h;
  let near = 0.1;
  let far = 1000;
  let fov = 0.6;
  let proj = mat4Perspective(fov, aspect, near, far);
  let view = getViewMatrix();
  let projInv = mat4Inverse(proj);
  let world = mat4Inverse(view);
  let camPos = getCameraPosition();

  device.queue.writeBuffer(uniformBuffers[0], 0, proj);
  device.queue.writeBuffer(uniformBuffers[1], 0, view);
  device.queue.writeBuffer(uniformBuffers[2], 0, projInv);
  device.queue.writeBuffer(uniformBuffers[3], 0, world);
  device.queue.writeBuffer(uniformBuffers[4], 0, new Float32Array(camPos));

  let commandEncoder = device.createCommandEncoder();
  let renderPass = commandEncoder.beginRenderPass({
    colorAttachments: [{
      view: context!.getCurrentTexture().createView(),
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      loadOp: "clear",
      storeOp: "store",
    }],
    depthStencilAttachment: {
      view: depthTexture.createView(),
      depthClearValue: 1.0,
      depthLoadOp: "clear",
      depthStoreOp: "store",
    },
  });

  renderPass.setPipeline(pipeline);
  renderPass.setBindGroup(0, bindGroup);
  renderPass.setVertexBuffer(0, vertexBuffer);
  renderPass.draw(4);
  renderPass.end();

  device.queue.submit([commandEncoder.finish()]);

  requestAnimationFrame(render);
}

render();
