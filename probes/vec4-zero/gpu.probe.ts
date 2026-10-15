/**
 * Probe: what Chromium's WebGPU, and Chromium's WebGL 2 on SwiftShader and on the
 * machine's GPU, draw for a triangle through vec4(0) and for triangles whose
 * vertices lie in one plane with the eye. It draws through raw WebGPU and WebGL 2
 * calls, never through rmsl, on a 16×16 target with one sample.
 *
 * - toTheCentre: (-1,-1,.5,1), (1,-1,.5,1), vec4(0).
 * - oppositeEdge: (-1,.0625,.5,1), (1,.0625,.5,1), vec4(0); the edge opposite vec4(0)
 *   lies on the centres of row 7.
 * - edgeThroughOrigin: (-1,-1,.5,1), (1,-1,.5,1), (1,1,-.5,-1); the third vertex is
 *   minus the first, so the edge between them passes through vec4(0), and no vertex
 *   lies there.
 * - inPlaneFront: 200 triangles of grid coordinates whose third vertex is the sum
 *   of the other two, every w above 0: each lies in one plane with the eye.
 * - inPlaneBehind: 200 such triangles whose third vertex is the first minus twice
 *   the second, with a w below 0.
 *
 * Each vertex has a colour of its own, red, green, blue. A pattern row has one
 * character a pixel: R, G or B for that colour alone (each byte within 2 of it), #
 * for a mix, . for none lit (alpha 0; a lit pixel has alpha 255, so a lit black
 * pixel is told apart). Errors: a WebGPU validation error scope, gl.getError.
 *
 * Run: pnpm vitest run --config probes/vec4-zero/vitest.config.ts
 * Writes probes/vec4-zero/gpu.probe.json.
 */
import { writeFileSync } from "node:fs";
import { afterAll, it } from "vitest";
import { runInGpuPage, runInWebGpuPage } from "../../src/testing/browser";
import { releaseGpu, webgpuPage, gpuPage } from "../../src/testing/gpu";

const COMMON = `
import { random } from "../../src/testing/rasterizer-reference";
const A = [-1, -1, 0.5, 1], B = [1, -1, 0.5, 1];
const CASES = {
  toTheCentre: [A, B, [0, 0, 0, 0]],
  oppositeEdge: [[-1, 0.0625, 0.5, 1], [1, 0.0625, 0.5, 1], [0, 0, 0, 0]],
  edgeThroughOrigin: [A, B, [1, 1, -0.5, -1]],
};
const inPlane = (behind) => {
  const next = random(behind ? 52 : 51);
  const grid = (r) => Math.round((next() * 2 - 1) * r * 16) / 16;
  const out = [];
  while (out.length < 200) {
    const w0 = Math.abs(grid(1)) + 0.25, w1 = Math.abs(grid(1)) + 0.25;
    const p0 = [grid(w0), grid(w0), w0 / 2, w0], p1 = [grid(w1), grid(w1), w1 / 2, w1];
    const p2 = behind ? p0.map((c, i) => c - 2 * p1[i]) : p0.map((c, i) => c + p1[i]);
    if (behind && !(p2[3] < 0)) continue;
    out.push([p0, p1, p2]);
  }
  return out;
};
const COLOURS = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const pattern = (data, size) => {
  const rows = [];
  let lit = 0;
  for (let y = 0; y < size; y++) {
    let row = "";
    for (let x = 0; x < size; x++) {
      const [r, g, b, a] = data.slice((y * size + x) * 4, (y * size + x) * 4 + 4);
      if (a === 0) { row += "."; continue; }
      lit++;
      const near = (v, t) => Math.abs(v - t) <= 2;
      row += near(r, 255) && near(g, 0) && near(b, 0) ? "R" : near(r, 0) && near(g, 255) && near(b, 0) ? "G" : near(r, 0) && near(g, 0) && near(b, 255) ? "B" : "#";
    }
    rows.push(row);
  }
  return { lit, rows };
};
const summarize = async (draw) => {
  const cases = {};
  for (const [name, vs] of Object.entries(CASES)) cases[name] = pattern(await draw(vs.flat()), 16);
  for (const [name, behind] of [["inPlaneFront", false], ["inPlaneBehind", true]]) {
    let trianglesLit = 0, pixelsLit = 0;
    for (const t of inPlane(behind)) {
      const { lit } = pattern(await draw(t.flat()), 16);
      if (lit) { trianglesLit++; pixelsLit += lit; }
    }
    cases[name] = { triangles: 200, trianglesLit, pixelsLit };
  }
  return cases;
};
`;

const WEBGPU = `
${COMMON}
globalThis.__probe = async () => {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code: \`
    struct O { @builtin(position) p: vec4f, @location(0) c: vec3f };
    @vertex fn vs(@location(0) p: vec4f, @location(1) c: vec3f) -> O { return O(p, c); }
    @fragment fn fs(@location(0) c: vec3f) -> @location(0) vec4f { return vec4f(c, 1.0); }
  \` });
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vs", buffers: [
      { arrayStride: 16, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x4" }] },
      { arrayStride: 12, attributes: [{ shaderLocation: 1, offset: 0, format: "float32x3" }] } ] },
    fragment: { module, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] },
    primitive: { topology: "triangle-list" },
  });
  const draw = async (positions) => {
    const size = 16;
    const texture = device.createTexture({ size: [size, size], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const bytesPerRow = 256;
    const read = device.createBuffer({ size: bytesPerRow * size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const p = device.createBuffer({ size: positions.length * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    const c = device.createBuffer({ size: 36, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(p, 0, Float32Array.from(positions));
    device.queue.writeBuffer(c, 0, Float32Array.from(COLOURS));
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({ colorAttachments: [{ view: texture.createView(), loadOp: "clear", clearValue: [0, 0, 0, 0], storeOp: "store" }] });
    pass.setPipeline(pipeline);
    pass.setVertexBuffer(0, p);
    pass.setVertexBuffer(1, c);
    pass.draw(3);
    pass.end();
    encoder.copyTextureToBuffer({ texture }, { buffer: read, bytesPerRow }, [size, size]);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const raw = new Uint8Array(read.getMappedRange());
    const data = new Uint8Array(size * size * 4);
    for (let y = 0; y < size; y++) data.set(raw.subarray(y * bytesPerRow, y * bytesPerRow + size * 4), y * size * 4);
    read.unmap();
    texture.destroy(); read.destroy(); p.destroy(); c.destroy();
    return data;
  };
  const cases = await summarize(draw);
  const error = await device.popErrorScope();
  return { cases, error: error ? error.message : null };
};
`;

const WEBGL = `
${COMMON}
globalThis.__probe = async () => {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 16;
  const gl = canvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: true });
  const shader = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
  const program = gl.createProgram();
  gl.attachShader(program, shader(gl.VERTEX_SHADER, "#version 300 es\\nin vec4 p; in vec3 c; out vec3 v; void main() { v = c; gl_Position = p; }"));
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER, "#version 300 es\\nprecision highp float; in vec3 v; out vec4 o; void main() { o = vec4(v, 1.0); }"));
  gl.linkProgram(program);
  gl.useProgram(program);
  const buffer = (data, loc, n) => {
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, Float32Array.from(data), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, n, gl.FLOAT, false, 0, 0);
  };
  const draw = async (positions) => {
    buffer(positions, gl.getAttribLocation(program, "p"), 4);
    buffer(COLOURS, gl.getAttribLocation(program, "c"), 3);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const data = new Uint8Array(16 * 16 * 4);
    gl.readPixels(0, 0, 16, 16, gl.RGBA, gl.UNSIGNED_BYTE, data);
    // readPixels starts at the bottom row
    const flipped = new Uint8Array(data.length);
    for (let y = 0; y < 16; y++) flipped.set(data.subarray((15 - y) * 64, (16 - y) * 64), y * 64);
    return flipped;
  };
  const cases = await summarize(draw);
  const debug = gl.getExtension("WEBGL_debug_renderer_info");
  return { cases, error: gl.getError(), samples: gl.getParameter(gl.SAMPLES), renderer: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER) };
};
`;

afterAll(releaseGpu);

it("draws triangles through vec4(0) and in one plane with the eye on WebGPU and on WebGL 2", async () => {
  const dir = new URL(".", import.meta.url).pathname;
  const environment = await (
    await webgpuPage()
  ).evaluate(async () => {
    const info = (await (navigator as any).gpu.requestAdapter()).info ?? {};
    return { userAgent: navigator.userAgent, vendor: info.vendor, architecture: info.architecture };
  });
  const webgpu = await runInWebGpuPage(WEBGPU, "__probe", dir);
  const webglUserAgent = await (await gpuPage()).evaluate(() => navigator.userAgent);
  const webglSwiftShader = await runInGpuPage(WEBGL, "__probe", dir);
  // The same WebGL 2 draw in the WebGPU browser, which renders on the machine's GPU.
  const webglOnGpu = await runInWebGpuPage(WEBGL, "__probe", dir);
  writeFileSync(
    new URL("gpu.probe.json", import.meta.url),
    JSON.stringify(
      {
        environment: { webgpu: environment, webgl: webglUserAgent, node: process.version },
        webgpu,
        webglSwiftShader,
        webglOnGpu,
      },
      null,
      2,
    ) + "\n",
  );
});
