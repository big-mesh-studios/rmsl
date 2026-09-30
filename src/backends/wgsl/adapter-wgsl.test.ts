import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { Fn, invocationIndex, storage, uniform } from "../../rmsl";
import { createWgslCompute } from "./adapter-wgsl";
import { GPU_ENABLED } from "../../testing/gpu";

let dawn: any;
let instance: any;

/**
 * The adapter reaches for the browser's WebGPU globals, so the Dawn binding
 * stands in for them here: `navigator.gpu` and the usage-flag namespaces.
 */
beforeAll(async () => {
  if (!GPU_ENABLED) return;
  dawn = await import("@kmamal/gpu");
  instance = dawn.create([]);
  // Node defines `navigator` as a getter-only global, so it has to be redefined, not assigned.
  Object.defineProperty(globalThis, "navigator", { value: { gpu: instance }, configurable: true });
  Object.assign(globalThis, { GPUBufferUsage: dawn.GPUBufferUsage, GPUMapMode: dawn.GPUMapMode });
});

afterAll(() => {
  if (instance) dawn.destroy(instance);
});

describe.skipIf(!GPU_ENABLED)("createWgslCompute with integer data", () => {
  it("reads an int storage back as the integers it holds", async () => {
    const program = Fn(() => {
      const values = storage("values", "int", { access: "read_write" });
      const i = invocationIndex();
      values.element(i).assign(values.element(i).add(1));
    })();
    const adapter = createWgslCompute(program);
    await adapter.attach();
    adapter.setAttribute("values", Int32Array.from([16777217, -5]));
    const out = { values: new Int32Array(2) };
    await adapter.compute(out);
    expect(Array.from(out.values)).toEqual([16777218, -4]);
    adapter.destroy();
  });

  it("reads a uint storage above INT_MAX back unsigned", async () => {
    const program = Fn(() => {
      const values = storage("values", "uint", { access: "read_write" });
      const i = invocationIndex();
      values.element(i).assign(values.element(i).sub(1));
    })();
    const adapter = createWgslCompute(program);
    await adapter.attach();
    adapter.setAttribute("values", Uint32Array.from([4294967295]));
    const out = { values: new Uint32Array(1) };
    await adapter.compute(out);
    expect(Array.from(out.values)).toEqual([4294967294]);
    adapter.destroy();
  });

  it("passes an int uniform as an integer", async () => {
    let offset!: ReturnType<typeof uniform<"int">>;
    const program = Fn(() => {
      const values = storage("values", "int", { access: "read_write" });
      offset = uniform("int");
      const i = invocationIndex();
      values.element(i).assign(values.element(i).add(offset));
    })();
    const adapter = createWgslCompute(program);
    await adapter.attach();
    adapter.setAttribute("values", Int32Array.from([10, -10]));
    adapter.setUniform(offset, -3);
    const out = { values: new Int32Array(2) };
    await adapter.compute(out);
    expect(Array.from(out.values)).toEqual([7, -13]);
    adapter.destroy();
  });

  it("passes a uint uniform above INT_MAX as an unsigned integer", async () => {
    let offset!: ReturnType<typeof uniform<"uint">>;
    const program = Fn(() => {
      const values = storage("values", "uint", { access: "read_write" });
      offset = uniform("uint");
      const i = invocationIndex();
      values.element(i).assign(values.element(i).add(offset));
    })();
    const adapter = createWgslCompute(program);
    await adapter.attach();
    adapter.setAttribute("values", Uint32Array.from([1]));
    adapter.setUniform(offset, 4000000000);
    const out = { values: new Uint32Array(1) };
    await adapter.compute(out);
    expect(Array.from(out.values)).toEqual([4000000001]);
    adapter.destroy();
  });
});
