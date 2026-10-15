/**
 * Probe: which pixel centres exactly on an edge Chromium's WebGPU lights, for
 * triangles whose vertices all lie in front of the eye and for triangles with a
 * vertex behind it, on targets of power-of-two and other sizes. It draws through
 * raw WebGPU calls, never through rmsl, and compares each image with two models
 * computed beside it:
 *
 * - exact: the top-left rule and the depth range applied to the triangle as its
 *   vertices place it, computed exactly in rationals;
 * - rounded: the top-left rule applied to the vertices' projections computed in
 *   64-bit floats, x / w then to pixels, as rmsl's CPU rasterizers decided an
 *   unclipped triangle before this change (only for triangles in front of the eye).
 *
 * The coordinates are multiples of 1/16, exact in 32 bits, so many centres lie
 * exactly on an edge. Each pixel where WebGPU and the exact rule differ is
 * counted as off an edge unless its centre lies exactly on a line through two
 * of the triangle's vertices. A pixel is lit when its alpha byte is not 0; the clear is
 * alpha 0 and the fragment writes alpha 1. Errors: a validation error scope.
 *
 * Run: pnpm vitest run --config probes/edge-rule/vitest.config.ts
 * Writes probes/edge-rule/webgpu-edge.probe.json.
 */
import { writeFileSync } from "node:fs";
import { afterAll, it } from "vitest";
import { runInWebGpuPage } from "../../src/testing/browser";
import { releaseGpu, webgpuPage } from "../../src/testing/gpu";

const PAGE = `
import { exactCoverage, exactlyOnLine } from "../../src/testing/rasterizer-reference";
import { triangles } from "./triangles";

function rounded(t, width, height) {
  const s = [0, 1, 2].map((k) => [
    (t[4 * k] / t[4 * k + 3] * 0.5 + 0.5) * width,
    (1 - (t[4 * k + 1] / t[4 * k + 3] * 0.5 + 0.5)) * height,
  ]);
  const area = (s[1][0] - s[0][0]) * (s[2][1] - s[0][1]) - (s[1][1] - s[0][1]) * (s[2][0] - s[0][0]);
  const out = new Uint8Array(width * height);
  if (area === 0) return out;
  const wind = area > 0 ? 1 : -1;
  const owns = (dx, dy) => dy < 0 || (dy === 0 && dx > 0);
  const o = [0, 1, 2].map((e) => {
    const a = s[(e + 1) % 3], b = s[(e + 2) % 3];
    return owns((b[0] - a[0]) * wind, (b[1] - a[1]) * wind);
  });
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const px = x + 0.5, py = y + 0.5;
      let inside = true;
      for (let e = 0; e < 3; e++) {
        const a = s[(e + 1) % 3], b = s[(e + 2) % 3];
        const f = ((a[0] - px) * (b[1] - py) - (a[1] - py) * (b[0] - px)) * wind;
        if (f < 0 || (f === 0 && !o[e])) inside = false;
      }
      if (!inside) continue;
      out[y * width + x] = 1;
    }
  return out;
}

globalThis.__probe = async () => {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code: \`
    @vertex fn vs(@location(0) p: vec4f) -> @builtin(position) vec4f { return p; }
    @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
  \` });
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vs", buffers: [{ arrayStride: 16, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x4" }] }] },
    fragment: { module, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] },
    primitive: { topology: "triangle-list" },
  });
  const draw = async (positions, width, height) => {
    const texture = device.createTexture({ size: [width, height], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const bytesPerRow = Math.ceil(width * 4 / 256) * 256;
    const read = device.createBuffer({ size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const p = device.createBuffer({ size: positions.length * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(p, 0, Float32Array.from(positions));
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({ colorAttachments: [{ view: texture.createView(), loadOp: "clear", clearValue: [0, 0, 0, 0], storeOp: "store" }] });
    pass.setPipeline(pipeline);
    pass.setVertexBuffer(0, p);
    pass.draw(3);
    pass.end();
    encoder.copyTextureToBuffer({ texture }, { buffer: read, bytesPerRow }, [width, height]);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const raw = new Uint8Array(read.getMappedRange());
    const lit = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) lit[y * width + x] = raw[y * bytesPerRow + x * 4 + 3] !== 0 ? 1 : 0;
    read.unmap();
    texture.destroy(); read.destroy(); p.destroy();
    return lit;
  };
  const sizes = [[16, 16], [15, 15], [30, 15], [24, 12]];
  const runs = {};
  for (const set of ["front", "behind"]) {
    for (const [width, height] of sizes) {
      let drawn = 0, offEdge = 0, exactTriangles = 0, exactPixels = 0, roundedTriangles = 0, roundedPixels = 0, roundedExactTriangles = 0;
      const examples = [];
      for (const t of triangles(set)) {
        const exact = exactCoverage(t, width, height);
        if (!exact) continue;
        drawn++;
        const gpu = await draw(t, width, height);
        const model = set === "front" ? rounded(t, width, height) : null;
        const v = [0, 1, 2].map((k) => t.slice(4 * k, 4 * k + 4));
        const edges = [exactlyOnLine(v[0], v[1], width, height), exactlyOnLine(v[1], v[2], width, height), exactlyOnLine(v[2], v[0], width, height)];
        let de = 0, dr = 0, dre = 0;
        for (let i = 0; i < width * height; i++) {
          if (gpu[i] !== exact[i]) { de++; if (!(edges[0][i] || edges[1][i] || edges[2][i])) offEdge++; }
          if (model && gpu[i] !== model[i]) dr++;
          if (model && model[i] !== exact[i]) dre++;
        }
        if (de) { exactTriangles++; exactPixels += de; if (examples.length < 5) examples.push({ triangle: t, pixels: de }); }
        if (dr) { roundedTriangles++; roundedPixels += dr; }
        if (dre) roundedExactTriangles++;
      }
      runs[set + " " + width + "x" + height] = {
        drawn,
        gpuAgainstExact: { trianglesDiffering: exactTriangles, pixelsDiffering: exactPixels, pixelsDifferingOffAnEdge: offEdge, examples },
        ...(set === "front" ? { gpuAgainstRounded: { trianglesDiffering: roundedTriangles, pixelsDiffering: roundedPixels }, roundedAgainstExact: { trianglesDiffering: roundedExactTriangles } } : {}),
      };
    }
  }
  const error = await device.popErrorScope();
  return { runs, error: error ? error.message : null };
};
`;

afterAll(releaseGpu);

it("compares the pixel centres Chromium's WebGPU lights on an edge with the exact and the rounded rule", async () => {
  const dir = new URL(".", import.meta.url).pathname;
  const environment = await (
    await webgpuPage()
  ).evaluate(async () => {
    const info = (await (navigator as any).gpu.requestAdapter()).info ?? {};
    return { userAgent: navigator.userAgent, vendor: info.vendor, architecture: info.architecture };
  });
  const result = await runInWebGpuPage(PAGE, "__probe", dir);
  writeFileSync(
    new URL("webgpu-edge.probe.json", import.meta.url),
    JSON.stringify({ environment: { webgpu: environment, node: process.version }, ...result }, null, 2) + "\n",
  );
});
