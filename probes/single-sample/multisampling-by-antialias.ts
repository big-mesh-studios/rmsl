/**
 * Question: with each `antialias` attribute, how many samples does a WebGL 2
 * context's drawing buffer get, and does a pixel an edge crosses take a part of
 * the colour? And does a WebGPU pipeline with no `multisample` state draw with
 * one sample?
 *
 * WebGL 2: a canvas of 4×4 and one of 5×5 gets a context made with no
 * attributes, `{}`, `antialias: true` and `antialias: false`. Each draws an
 * opaque red rectangle from x 0.5 to 2.5 pixels, the full height, over a
 * transparent clear, so its edges cross the centres of columns 0 and 2. The
 * result is the `antialias` the context reports, its `SAMPLES`, the link
 * status, the alpha of each pixel of row 0 and `getError` after the read.
 *
 * WebGPU: the same rectangle into a 4×4 and a 5×5 `rgba8unorm` texture, with
 * no `multisample` state and, as a control that the read can see a partial
 * colour, with `count: 4` resolved into the texture. The result is the alpha
 * of each pixel of row 0 and the messages of a validation error scope.
 *
 * Measures the platform only: plain WebGL 2 and WebGPU, no rmsl.
 */
import { PROBES, report, runInGpuPage, runInWebGpuPage } from "./harness";

const ENTRY = `
const rectangle = (size) => {
  const x = (p) => (p / size) * 2 - 1;
  return Float32Array.of(x(0.5), -1, x(2.5), -1, x(0.5), 1, x(0.5), 1, x(2.5), -1, x(2.5), 1);
};
const onWebGl = (attributes, size) => {
  const canvas = Object.assign(document.createElement("canvas"), { width: size, height: size });
  const gl = attributes === undefined ? canvas.getContext("webgl2") : canvas.getContext("webgl2", attributes);
  const program = gl.createProgram();
  for (const [type, source] of [
    [gl.VERTEX_SHADER, "#version 300 es\\nin vec2 p; void main() { gl_Position = vec4(p, 0.0, 1.0); }"],
    [gl.FRAGMENT_SHADER, "#version 300 es\\nprecision highp float; out vec4 o; void main() { o = vec4(1.0, 0.0, 0.0, 1.0); }"],
  ]) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    gl.attachShader(program, shader);
  }
  gl.linkProgram(program);
  gl.useProgram(program);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, rectangle(size), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.viewport(0, 0, size, size);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  const pixels = new Uint8Array(size * 4);
  gl.readPixels(0, 0, size, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  return {
    antialias: gl.getContextAttributes().antialias,
    samples: gl.getParameter(gl.SAMPLES),
    linked: gl.getProgramParameter(program, gl.LINK_STATUS),
    alphas: Array.from({ length: size }, (_, i) => pixels[i * 4 + 3]).join(" "),
    error: gl.getError(),
  };
};
const onWebGpu = async (device, count, size) => {
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code: \`
    @vertex fn vs(@location(0) p: vec2f) -> @builtin(position) vec4f { return vec4f(p, 0.0, 1.0); }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0, 0.0, 0.0, 1.0); }\` });
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vs", buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }] },
    fragment: { module, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] },
    ...(count ? { multisample: { count } } : {}),
  });
  const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC;
  const texture = device.createTexture({ size: [size, size], format: "rgba8unorm", usage });
  const msaa = count ? device.createTexture({ size: [size, size], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT, sampleCount: count }) : null;
  const vertices = rectangle(size);
  const buffer = device.createBuffer({ size: vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, vertices);
  const read = device.createBuffer({ size: 256 * size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({ colorAttachments: [{
    view: (msaa ?? texture).createView(), resolveTarget: msaa ? texture.createView() : undefined,
    clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" }] });
  pass.setPipeline(pipeline);
  pass.setVertexBuffer(0, buffer);
  pass.draw(6);
  pass.end();
  encoder.copyTextureToBuffer({ texture }, { buffer: read, bytesPerRow: 256 }, [size, size]);
  device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const pixels = new Uint8Array(read.getMappedRange().slice(0));
  const error = await device.popErrorScope();
  return { alphas: Array.from({ length: size }, (_, i) => pixels[i * 4 + 3]).join(" "), errors: error ? [error.message] : [] };
};
globalThis.__probe = async () => {
  const webgl = {};
  for (const size of [4, 5])
    for (const [name, attributes] of [["none", undefined], ["empty", {}], ["antialiasTrue", { antialias: true }], ["antialiasFalse", { antialias: false }]])
      webgl[name + size] = onWebGl(attributes, size);
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) return { webgl, webgpu: "no adapter" };
  const device = await adapter.requestDevice();
  const webgpu = {};
  for (const size of [4, 5]) {
    webgpu["noMultisample" + size] = await onWebGpu(device, 0, size);
    webgpu["count4" + size] = await onWebGpu(device, 4, size);
  }
  return { webgl, webgpu };
};
`;

const dir = PROBES;
await report("multisampling-by-antialias", async () => ({
  webglPage: await runInGpuPage(ENTRY, "__probe", dir),
  webgpuPage: await runInWebGpuPage(ENTRY, "__probe", dir),
}));
