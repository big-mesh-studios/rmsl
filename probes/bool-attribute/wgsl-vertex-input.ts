/**
 * Asks Dawn, through @kmamal/gpu, whether a WGSL shader module may declare a
 * `bool` or boolean vector as a vertex input or as a value passed from the
 * vertex to the fragment stage, beside the `u32` input that could carry one.
 *
 * Run from the repository root: `node probes/bool-attribute/wgsl-vertex-input.ts`.
 * It writes `wgsl-vertex-input.out.json` beside itself.
 */
import { readFileSync, writeFileSync } from "node:fs";
import gpu from "@kmamal/gpu";

const cases: Record<string, string> = {
  "vertex input bool": `@vertex fn main(@location(0) f: bool) -> @builtin(position) vec4f { return vec4f(select(0.0, 1.0, f)); }`,
  "vertex input vec2<bool>": `@vertex fn main(@location(0) f: vec2<bool>) -> @builtin(position) vec4f { return vec4f(select(0.0, 1.0, f.x)); }`,
  "vertex input vec4<bool>": `@vertex fn main(@location(0) f: vec4<bool>) -> @builtin(position) vec4f { return vec4f(select(0.0, 1.0, f.x)); }`,
  "vertex input u32 compared with zero": `@vertex fn main(@location(0) f: u32) -> @builtin(position) vec4f { return vec4f(select(0.0, 1.0, f != 0u)); }`,
  "vertex input vec2<u32> compared with zero": `@vertex fn main(@location(0) f: vec2<u32>) -> @builtin(position) vec4f { return vec4f(select(0.0, 1.0, (f != vec2<u32>(0u)).x)); }`,
  "vertex output bool": `struct O { @builtin(position) p: vec4f, @location(0) b: bool };
@vertex fn main() -> O { var o: O; o.b = true; return o; }`,
  "fragment input bool": `@fragment fn main(@location(0) @interpolate(flat) b: bool) -> @location(0) vec4f { return vec4f(select(0.0, 1.0, b)); }`,
  "vertex output u32 flat": `struct O { @builtin(position) p: vec4f, @location(0) @interpolate(flat) b: u32 };
@vertex fn main() -> O { var o: O; o.b = 1u; return o; }`,
};

const instance = gpu.create([]);
const adapter = await instance.requestAdapter();
if (!adapter) throw new Error("no WebGPU adapter");
const device = await adapter.requestDevice();
const results: Record<string, string> = {};
for (const [name, code] of Object.entries(cases)) {
  device.pushErrorScope("validation");
  device.createShaderModule({ code });
  const error = await device.popErrorScope();
  results[name] = error
    ? (error.message
        .split("\n")
        .find((l: string) => l.includes("error:"))
        ?.trim() ?? error.message)
    : "valid";
}
const pkg = JSON.parse(readFileSync(new URL("../../node_modules/@kmamal/gpu/package.json", import.meta.url), "utf8"));
const out = {
  environment: {
    node: process.version,
    platform: `${process.platform} ${process.arch}`,
    "@kmamal/gpu": pkg.version,
    dawnCommit: pkg.config.dawn.commit,
    adapter: adapter.info?.description ?? null,
  },
  results,
};
writeFileSync(new URL("./wgsl-vertex-input.out.json", import.meta.url), JSON.stringify(out, null, 2) + "\n");
console.log(JSON.stringify(out, null, 2));
device.destroy();
gpu.destroy?.(instance);
