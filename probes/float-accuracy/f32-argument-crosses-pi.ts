/**
 * Question: can the argument a GPU computes for `sin` lie outside [-π, π]
 * where the same expression computed in 64 bits and rounded once to 32 bits
 * lies inside it?
 *
 * Computes `a + b` for a = 3.0004 and b = 0.1411925918 in WGSL on Dawn, through
 * @kmamal/gpu and no rmsl: once from a storage buffer the host fills with a
 * `Float32Array`, and once from a function called with the two numbers as
 * literal arguments. It reads back the sum and `sin` of it, and compares the
 * sum with π, with `Math.fround(a + b)` and with the sum of the two inputs
 * rounded to 32 bits. A control pair, b = 0.1411925, does not cross. Errors: a
 * validation error scope.
 *
 * Run: node probes/float-accuracy/f32-argument-crosses-pi.ts > probes/float-accuracy/results/f32-argument-crosses-pi.json
 */
import { readFileSync } from "node:fs";
import { cpus } from "node:os";
import gpu from "@kmamal/gpu";

const pairs = [
  { name: "crossing", a: 3.0004, b: 0.1411925918 },
  { name: "control", a: 3.0004, b: 0.1411925 },
];

const instance = gpu.create([]);
const adapter = await instance.requestAdapter();
if (!adapter) throw new Error("no WebGPU adapter");
const device = await adapter.requestDevice();

/** Runs `body` over an input buffer of `inputs` and reads back `count` floats it writes to `out`. */
async function run(body: string, inputs: Float32Array, count: number) {
  const STORAGE = 0x80,
    COPY_SRC = 0x4,
    COPY_DST = 0x8,
    MAP_READ = 0x1;
  device.pushErrorScope("validation");
  const module = device.createShaderModule({
    code: `@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
${body}`,
  });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const input = device.createBuffer({ size: inputs.byteLength, usage: STORAGE | COPY_DST });
  device.queue.writeBuffer(input, 0, inputs);
  const output = device.createBuffer({ size: 4 * count, usage: STORAGE | COPY_SRC });
  const readback = device.createBuffer({ size: 4 * count, usage: MAP_READ | COPY_DST });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(
    0,
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: output } },
      ],
    }),
  );
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, 4 * count);
  device.queue.submit([encoder.finish()]);
  const error = await device.popErrorScope();
  await readback.mapAsync(MAP_READ);
  const values = Array.from(new Float32Array(readback.getMappedRange().slice(0)));
  readback.unmap();
  return { values, error: error ? error.message : null };
}

const results = [];
for (const { name, a, b } of pairs) {
  const fromStorage = await run(
    `@compute @workgroup_size(1) fn main() {
  let sum = inputs[0] + inputs[1];
  out[0] = sum;
  out[1] = sin(sum);
}`,
    Float32Array.of(a, b),
    2,
  );
  const fromLiterals = await run(
    `fn add(a0: f32, a1: f32) -> f32 { return a0 + a1; }
@compute @workgroup_size(1) fn main() {
  let sum = add(${a}, ${b}) + inputs[0] * 0.0;
  out[0] = sum;
  out[1] = sin(sum);
}`,
    Float32Array.of(0),
    2,
  );
  const roundedOnce = Math.fround(a + b);
  const roundedEachStep = Math.fround(Math.fround(a) + Math.fround(b));
  results.push({
    name,
    a,
    b,
    roundedOnce,
    roundedEachStep,
    fromStorage,
    fromLiterals,
    checks: {
      roundedOnceIsInside: Math.abs(roundedOnce) <= Math.PI,
      storageSumIsOutside: Math.abs(fromStorage.values[0]!) > Math.PI,
      literalSumIsOutside: Math.abs(fromLiterals.values[0]!) > Math.PI,
      storageSumIsRoundedEachStep: fromStorage.values[0] === roundedEachStep,
      literalSumIsRoundedEachStep: fromLiterals.values[0] === roundedEachStep,
    },
  });
}

const info = adapter.info ?? (await adapter.requestAdapterInfo?.());
const kmamal = JSON.parse(
  readFileSync(new URL("../../node_modules/@kmamal/gpu/package.json", import.meta.url), "utf8"),
);
console.log(
  JSON.stringify(
    {
      probe: "f32-argument-crosses-pi",
      environment: {
        node: process.version,
        platform: `${process.platform} ${process.arch}`,
        cpu: cpus()[0]?.model ?? null,
        "@kmamal/gpu": kmamal.version,
        adapter: info?.description ?? null,
      },
      pi: Math.PI,
      results,
    },
    null,
    2,
  ),
);
process.exit(0);
