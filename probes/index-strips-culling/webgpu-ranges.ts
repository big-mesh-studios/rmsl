/**
 * How Chromium's WebGPU takes a `firstIndex`, `indexCount`, `firstVertex` or
 * `vertexCount` that is negative, fractional or not finite.
 *
 * Run with `node probes/index-strips-culling/webgpu-ranges.ts`. It draws a list
 * of six vertices, two triangles with ids 10 to 15, through raw WebGPU in the
 * WebGPU page of the test harness, and prints for each call what it threw, or
 * which triangles drew. A pixel shows the id of the first vertex of the triangle that covers it: pixel 0 is triangle 0, 1, 2; pixel 3 is triangle 3, 4, 5; pixel 2 is triangle 2, 3, 4.
 */
import { webgpuPage, releaseGpu } from "../../src/testing/gpu.ts";

const SHADER = /* wgsl */ `
struct VOut { @builtin(position) p: vec4f, @location(0) @interpolate(flat) id: u32 }
@vertex fn vs(@location(0) pos: vec4f, @location(1) id: u32) -> VOut { return VOut(pos, id); }
@fragment fn fs(v: VOut) -> @location(0) vec4f { return vec4f(f32(v.id) / 255.0, 0, 0, 1); }`;

async function inPage(shader: string) {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter!.requestDevice();
  const module = device.createShaderModule({ code: shader });
  // Triangle 0, 1, 2 on the left half and 3, 4, 5 on the right half of a 4 by 1 target.
  const positions = [-1, -1, 0, 1, 0, -1, 0, 1, -1, 3, 0, 1, 0, -1, 0, 1, 1, -1, 0, 1, 0, 3, 0, 1];
  const pos = device.createBuffer({ size: 96, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(pos, 0, new Float32Array(positions));
  const ids = device.createBuffer({ size: 24, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(ids, 0, Uint32Array.of(10, 11, 12, 13, 14, 15));
  const index = device.createBuffer({ size: 12, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(index, 0, Uint16Array.of(0, 1, 2, 3, 4, 5));
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module,
      entryPoint: "vs",
      buffers: [
        { arrayStride: 16, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x4" }] },
        { arrayStride: 4, attributes: [{ shaderLocation: 1, offset: 0, format: "uint32" }] },
      ],
    },
    fragment: { module, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] },
    primitive: { topology: "triangle-list" },
  });
  const calls: [string, boolean, number, number][] = [
    ["drawIndexed", true, 3, 2.9],
    ["drawIndexed", true, 3.9, 0],
    ["drawIndexed", true, 3, -0.5],
    ["drawIndexed", true, 3, -1],
    ["drawIndexed", true, -3, 0],
    ["drawIndexed", true, 3, NaN],
    ["drawIndexed", true, Infinity, 0],
    ["draw", false, 3, 2.9],
    ["draw", false, 3.9, 0],
    ["draw", false, 3, -1],
    ["draw", false, NaN, 0],
  ];
  const results: string[] = [];
  for (const [name, indexed, count, first] of calls) {
    device.pushErrorScope("validation");
    const texture = device.createTexture({
      size: [4, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: texture.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 0] }],
    });
    pass.setPipeline(pipeline);
    pass.setVertexBuffer(0, pos);
    pass.setVertexBuffer(1, ids);
    pass.setIndexBuffer(index, "uint16");
    let thrown = "";
    try {
      if (indexed) pass.drawIndexed(count, 1, first, 0, 0);
      else pass.draw(count, 1, first, 0);
    } catch (error) {
      thrown = `${(error as Error).name}: ${(error as Error).message}`;
    }
    pass.end();
    device.queue.submit([encoder.finish()]);
    const validation = await device.popErrorScope();
    const read = device.createBuffer({ size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const copy = device.createCommandEncoder();
    copy.copyTextureToBuffer({ texture }, { buffer: read, bytesPerRow: 256 }, [4, 1]);
    device.queue.submit([copy.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8Array(read.getMappedRange());
    const shown = [0, 1, 2, 3].map((x) => (bytes[x * 4 + 3] ? String(bytes[x * 4]) : "nothing")).join(" | ");
    read.unmap();
    results.push(
      `${name}(count ${count}, first ${first}): ${thrown || "no exception"}; validation ${validation?.message ?? "none"}; pixels 0 to 3: ${shown}`,
    );
  }
  return { userAgent: navigator.userAgent, results };
}

const page = await webgpuPage();
const out = await page.evaluate(
  ([source, shader]: [string, string]) =>
    // eslint-disable-next-line no-new-func
    new Function("return " + source)()(shader),
  [inPage.toString(), SHADER] as [string, string],
);
console.log(`# WebGPU — ${out.userAgent}`);
for (const line of out.results) console.log(line);
await releaseGpu();
