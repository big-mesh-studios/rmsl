/**
 * Question: which pixel centres lying exactly on an edge of a triangle does
 * each API shade, and does it give a centre on an edge two triangles share to
 * exactly one of them?
 *
 * Each shape is a list of triangles, its corners in pixels from the top left
 * of a 4×4 target, and every edge that crosses the target passes through
 * pixel centres: a rectangle with vertical edges, one with horizontal edges,
 * a triangle with a left and a top edge, one with a right and a bottom edge,
 * pairs of triangles split on edges of slope 1, -1, 1/2 and 2, and pairs of
 * rectangles split on a vertical and on a horizontal edge. Each shape is drawn
 * alone, opaque red over a transparent clear, in both windings, with culling
 * off. The result is the alpha of each pixel, rows from the top, and the
 * pairs of shapes that cover the target together with every pixel's alphas
 * added, so 255 everywhere means each centre on their shared edge was shaded
 * exactly once.
 *
 * WebGL 2 draws with `antialias: false` into the canvas and reads it with
 * `readPixels`, rows flipped to count from the top, and reports `SAMPLES` and
 * `getError`. WebGPU draws into an `rgba8unorm` texture with no `multisample`
 * state, under a validation error scope.
 *
 * Measures the platform only: plain WebGL 2 and WebGPU, no rmsl.
 */
import { PROBES, report, runInGpuPage, runInWebGpuPage } from "./harness";

const ENTRY = `
const rectangle = (l, t, r, b) => [[[l, t], [r, t], [l, b]], [[l, b], [r, t], [r, b]]];
const SHAPES = {
  columns: rectangle(0.5, -1, 2.5, 5),
  rows: rectangle(-1, 0.5, 5, 2.5),
  topLeft: [[[0.5, 0.5], [0.5, 9], [9, 0.5]]],
  bottomRight: [[[2.5, 2.5], [2.5, -9], [-9, 2.5]]],
  slopeOneAbove: [[[-2, -2], [6, -2], [6, 6]]],
  slopeOneBelow: [[[-2, -2], [6, 6], [-2, 6]]],
  slopeMinusOneAbove: [[[-2, 6], [6, -2], [-2, -2]]],
  slopeMinusOneBelow: [[[-2, 6], [6, -2], [6, 6]]],
  slopeHalfAbove: [[[-4.5, -2], [11.5, 6], [11.5, -2]]],
  slopeHalfBelow: [[[-4.5, -2], [11.5, 6], [-4.5, 6]]],
  slopeTwoRight: [[[-0.75, -2], [3.25, 6], [20, -2]]],
  slopeTwoLeft: [[[-0.75, -2], [3.25, 6], [-20, 6]]],
  splitColumnLeft: rectangle(-1, -1, 1.5, 5),
  splitColumnRight: rectangle(1.5, -1, 5, 5),
  splitRowAbove: rectangle(-1, -1, 5, 1.5),
  splitRowBelow: rectangle(-1, 1.5, 5, 5),
};
const PAIRS = [
  ["slopeOneAbove", "slopeOneBelow"], ["slopeMinusOneAbove", "slopeMinusOneBelow"],
  ["slopeHalfAbove", "slopeHalfBelow"], ["slopeTwoRight", "slopeTwoLeft"],
  ["splitColumnLeft", "splitColumnRight"], ["splitRowAbove", "splitRowBelow"],
];
const reversed = (triangles) => triangles.map(([a, b, c]) => [c, b, a]);
const clip = ([x, y]) => [(x / 4) * 2 - 1, 1 - (y / 4) * 2];
const vertices = (triangles) => Float32Array.from(triangles.flat().flatMap(clip));
const pattern = (alphaAt) => Array.from({ length: 4 }, (_, y) => Array.from({ length: 4 }, (_, x) => alphaAt(x, y)).join(" ")).join("/");
const pairs = (shapes) => Object.fromEntries(PAIRS.map(([a, b]) => {
  const one = shapes[a][0].split(/[ /]/).map(Number), two = shapes[b][0].split(/[ /]/).map(Number);
  return [a + "+" + b, one.map((v, i) => v + two[i]).join(" ")];
}));
const onWebGl = () => {
  const canvas = Object.assign(document.createElement("canvas"), { width: 4, height: 4 });
  const gl = canvas.getContext("webgl2", { antialias: false });
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
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.viewport(0, 0, 4, 4);
  const draw = (triangles) => {
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bufferData(gl.ARRAY_BUFFER, vertices(triangles), gl.STATIC_DRAW);
    gl.drawArrays(gl.TRIANGLES, 0, triangles.length * 3);
    const pixels = new Uint8Array(64);
    gl.readPixels(0, 0, 4, 4, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    return pattern((x, y) => pixels[((3 - y) * 4 + x) * 4 + 3]);
  };
  const shapes = Object.fromEntries(Object.entries(SHAPES).map(([n, t]) => [n, [draw(t), draw(reversed(t))]]));
  const info = gl.getExtension("WEBGL_debug_renderer_info");
  return {
    renderer: gl.getParameter(info ? info.UNMASKED_RENDERER_WEBGL : gl.RENDERER),
    samples: gl.getParameter(gl.SAMPLES), linked: gl.getProgramParameter(program, gl.LINK_STATUS),
    error: gl.getError(), shapes, pairs: pairs(shapes),
  };
};
const onWebGpu = async () => {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) return "no adapter";
  const device = await adapter.requestDevice();
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code: \`
    @vertex fn vs(@location(0) p: vec2f) -> @builtin(position) vec4f { return vec4f(p, 0.0, 1.0); }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0, 0.0, 0.0, 1.0); }\` });
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vs", buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }] },
    fragment: { module, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] },
    primitive: { cullMode: "none" },
  });
  const draw = async (triangles) => {
    const texture = device.createTexture({ size: [4, 4], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const data = vertices(triangles);
    const buffer = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffer, 0, data);
    const read = device.createBuffer({ size: 256 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({ colorAttachments: [{ view: texture.createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" }] });
    pass.setPipeline(pipeline);
    pass.setVertexBuffer(0, buffer);
    pass.draw(triangles.length * 3);
    pass.end();
    encoder.copyTextureToBuffer({ texture }, { buffer: read, bytesPerRow: 256 }, [4, 4]);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const pixels = new Uint8Array(read.getMappedRange().slice(0));
    return pattern((x, y) => pixels[y * 256 + x * 4 + 3]);
  };
  const shapes = {};
  for (const [n, t] of Object.entries(SHAPES)) shapes[n] = [await draw(t), await draw(reversed(t))];
  const error = await device.popErrorScope();
  return { adapter: { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture }, errors: error ? [error.message] : [], shapes, pairs: pairs(shapes) };
};
globalThis.__probe = async () => ({ webgl: onWebGl(), webgpu: await onWebGpu() });
`;

await report("pixel-centre-on-an-edge", async () => ({
  webglPage: await runInGpuPage(ENTRY, "__probe", PROBES),
  webgpuPage: await runInWebGpuPage(ENTRY, "__probe", PROBES),
}));
