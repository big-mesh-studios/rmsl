/**
 * What Chromium's WebGPU draws for triangle strips, indexed draws, primitive restart and culling.
 *
 * Run with `node probes/index-strips-culling/webgpu.ts`. It draws every case of `cases.ts` through
 * raw WebGPU in the WebGPU page of the test harness, and prints, for each probe pixel, the flat id,
 * the vertex index and the facing of the fragment that drew it, or `nothing`. A validation or
 * internal error of a draw is printed beside it. A fan has no WebGPU topology and is skipped.
 * It draws every case twice: on the page's default adapter, and on Chromium's fallback adapter
 * in a second browser, a second implementation of WebGPU.
 */
import { webgpuPage, releaseGpu } from "../../src/testing/gpu.ts";
import { cases, decode, histogram, HEIGHT, WIDTH, type DrawCase } from "./cases.ts";

const SHADER = /* wgsl */ `
struct VOut {
  @builtin(position) p: vec4f,
  @location(0) @interpolate(flat) id: u32,
  @location(1) @interpolate(flat) vi: u32,
}
@vertex fn vs(@location(0) pos: vec4f, @location(1) id: u32, @builtin(vertex_index) vi: u32) -> VOut {
  return VOut(pos, id, vi);
}
@fragment fn fs(v: VOut, @builtin(front_facing) ff: bool) -> @location(0) vec4f {
  return vec4f(f32(v.id) / 255.0, f32(v.vi % 256u) / 255.0, select(128.0, 255.0, ff) / 255.0, 1.0);
}`;

async function inPage(drawCases: DrawCase[], shader: string, width: number, height: number, fallback: boolean) {
  const adapter = await navigator.gpu.requestAdapter({ forceFallbackAdapter: fallback });
  const device = await adapter!.requestDevice();
  const info = adapter!.info;
  const module = device.createShaderModule({ code: shader });
  const results: { name: string; error: string | null; pixels: Record<string, number[]>; image: number[] }[] = [];
  for (const c of drawCases) {
    device.pushErrorScope("validation");
    device.pushErrorScope("internal");
    const n = c.positions.length / 4;
    const pos = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(pos, 0, new Float32Array(c.positions));
    const ids = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(ids, 0, new Uint32Array(c.ids ?? Array.from({ length: n }, (_, k) => 10 + k)));
    const strip = c.topology === "triangle-strip";
    const indexFormat = c.index?.kind as GPUIndexFormat | undefined;
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
      primitive: {
        topology: c.topology as GPUPrimitiveTopology,
        cullMode: c.cullMode,
        frontFace: c.frontFace,
        ...(strip && indexFormat ? { stripIndexFormat: indexFormat } : {}),
      },
    });
    const texture = device.createTexture({
      size: [width, height],
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
    if (c.index) {
      const array = c.index.kind === "uint16" ? new Uint16Array(c.index.values) : new Uint32Array(c.index.values);
      const size = Math.ceil(array.byteLength / 4) * 4;
      const indexBuffer = device.createBuffer({ size, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
      const padded = new Uint8Array(size);
      padded.set(new Uint8Array(array.buffer));
      device.queue.writeBuffer(indexBuffer, 0, padded);
      pass.setIndexBuffer(indexBuffer, c.index.kind as GPUIndexFormat);
      pass.drawIndexed(c.count, 1, c.firstIndex ?? 0, c.baseVertex ?? 0, 0);
    } else {
      pass.draw(c.count, 1, c.firstVertex ?? 0, 0);
    }
    pass.end();
    const bytesPerRow = 256;
    const read = device.createBuffer({
      size: bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyTextureToBuffer({ texture }, { buffer: read, bytesPerRow }, [width, height]);
    device.queue.submit([encoder.finish()]);
    const internal = await device.popErrorScope();
    const validation = await device.popErrorScope();
    const error = validation?.message ?? internal?.message ?? null;
    const pixels: Record<string, number[]> = {};
    let image: number[] = [];
    if (!error) {
      await read.mapAsync(GPUMapMode.READ);
      const bytes = new Uint8Array(read.getMappedRange());
      for (const p of c.probes) {
        const o = p.y * bytesPerRow + p.x * 4;
        pixels[p.name] = [bytes[o]!, bytes[o + 1]!, bytes[o + 2]!, bytes[o + 3]!];
      }
      image = Array.from(bytes);
      read.unmap();
    }
    results.push({ name: c.name, error, pixels, image });
  }
  return {
    adapter: {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description,
      isFallbackAdapter: (info as { isFallbackAdapter?: boolean }).isFallbackAdapter,
    },
    userAgent: navigator.userAgent,
    results,
  };
}

const drawCases = cases().filter((c) => c.topology !== "triangle-fan" && c.index?.kind !== "uint8");

/** Draws every case in `page`, on its default adapter or on its fallback adapter, and prints what each probe pixel shows. */
async function run(page: any, fallback: boolean): Promise<void> {
  const out = await page.evaluate(
    ([source, list, shader, width, height, onFallback]: [string, DrawCase[], string, number, number, boolean]) =>
      // eslint-disable-next-line no-new-func
      new Function("return " + source)()(list, shader, width, height, onFallback),
    [inPage.toString(), drawCases, SHADER, WIDTH, HEIGHT, fallback] as [
      string,
      DrawCase[],
      string,
      number,
      number,
      boolean,
    ],
  );
  console.log(`# WebGPU, ${fallback ? "fallback" : "default"} adapter — ${out.userAgent}`);
  console.log(`# adapter: ${JSON.stringify(out.adapter)}`);
  for (const r of out.results) {
    console.log(`\n## ${r.name}${r.error ? `\nerror: ${r.error}` : ""}`);
    for (const [name, rgba] of Object.entries(r.pixels as Record<string, number[]>))
      console.log(`  ${name}: ${decode(rgba)}`);
    if (r.image.length) console.log(`  whole image: ${JSON.stringify(histogram(r.image, 256))}`);
  }
}

const page = await webgpuPage();
await run(page, false);

// The fallback adapter is a second WebGPU implementation, which Chromium offers only with these switches.
const { chromium } = await import("playwright");
const browser = await chromium.launch({
  channel: "chromium",
  args: ["--enable-unsafe-swiftshader", "--enable-unsafe-webgpu"],
});
const fallbackPage = await browser.newPage();
await fallbackPage.goto(page.url());
console.log("");
await run(fallbackPage, true);
await browser.close();
await releaseGpu();
