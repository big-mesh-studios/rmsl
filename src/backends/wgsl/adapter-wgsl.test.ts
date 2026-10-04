import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { Fn, instancedArray, invocationIndex, uint, uniform } from "../../rmsl";
import { createWgslCompute } from "./adapter-wgsl";
import { GPU_ENABLED, installWebGpuGlobals } from "../../testing/gpu";

let uninstall: (() => void) | undefined;

beforeAll(async () => {
  if (GPU_ENABLED) uninstall = await installWebGpuGlobals();
});

afterAll(() => uninstall?.());

describe.skipIf(!GPU_ENABLED)("createWgslCompute with integer data", () => {
  /**
   * @canon spec-an-integer-reaches-the-host-as-the-integer-it-is
   */
  it("reads an int storage back as the integers it holds", async () => {
    const values = instancedArray(2, "int");
    const program = Fn(() => {
      const i = invocationIndex();
      values.element(i).assign(values.element(i).add(1));
    })();
    const adapter = createWgslCompute(program);
    await adapter.attach();
    adapter.setAttribute(values.name, Int32Array.from([16777217, -5]));
    const out = { [values.name]: new Int32Array(2) };
    await adapter.compute(out);
    expect(Array.from(out[values.name])).toEqual([16777218, -4]);
    adapter.destroy();
  });
  /**
   * @canon spec-an-integer-reaches-the-host-as-the-integer-it-is
   */
  it("reads a uint storage above INT_MAX back unsigned", async () => {
    const values = instancedArray(1, "uint");
    const program = Fn(() => {
      const i = invocationIndex();
      values.element(i).assign(values.element(i).sub(1));
    })();
    const adapter = createWgslCompute(program);
    await adapter.attach();
    adapter.setAttribute(values.name, Uint32Array.from([4294967295]));
    const out = { [values.name]: new Uint32Array(1) };
    await adapter.compute(out);
    expect(Array.from(out[values.name])).toEqual([4294967294]);
    adapter.destroy();
  });
  /**
   * @canon spec-an-integer-reaches-the-host-as-the-integer-it-is
   */
  it("passes an int uniform as an integer", async () => {
    const values = instancedArray(2, "int");
    let offset!: ReturnType<typeof uniform<"int">>;
    const program = Fn(() => {
      offset = uniform("int");
      const i = invocationIndex();
      values.element(i).assign(values.element(i).add(offset));
    })();
    const adapter = createWgslCompute(program);
    await adapter.attach();
    adapter.setAttribute(values.name, Int32Array.from([10, -10]));
    adapter.setUniform(offset, -3);
    const out = { [values.name]: new Int32Array(2) };
    await adapter.compute(out);
    expect(Array.from(out[values.name])).toEqual([7, -13]);
    adapter.destroy();
  });
  /**
   * @canon spec-an-integer-reaches-the-host-as-the-integer-it-is
   */
  it("passes a uint uniform above INT_MAX as an unsigned integer", async () => {
    const values = instancedArray(1, "uint");
    let offset!: ReturnType<typeof uniform<"uint">>;
    const program = Fn(() => {
      offset = uniform("uint");
      const i = invocationIndex();
      values.element(i).assign(values.element(i).add(offset));
    })();
    const adapter = createWgslCompute(program);
    await adapter.attach();
    adapter.setAttribute(values.name, Uint32Array.from([1]));
    adapter.setUniform(offset, 4000000000);
    const out = { [values.name]: new Uint32Array(1) };
    await adapter.compute(out);
    expect(Array.from(out[values.name])).toEqual([4000000001]);
    adapter.destroy();
  });
});

describe.skipIf(!GPU_ENABLED)("createWgslCompute reading back into out", () => {
  const a = instancedArray(2, "int");
  const b = instancedArray(2, "int");
  const program = () =>
    Fn(() => {
      const i = invocationIndex();
      a.element(i).assign(a.element(i).add(1));
      b.element(i).assign(b.element(i).add(2));
    })();

  /**
   * @canon spec-compute-copies-back-only-the-slots-out-names
   * @canon spec-a-wgsl-adapter-attaches-and-computes-through-a-promise
   */
  it("reads back only the slots out names", async () => {
    const adapter = createWgslCompute(program());
    await adapter.attach();
    adapter.setAttribute(a.name, Int32Array.from([1, 2]));
    adapter.setAttribute(b.name, Int32Array.from([10, 20]));
    const out = { [b.name]: new Int32Array(2) };
    await adapter.compute(out);
    expect(Array.from(out[b.name]!)).toEqual([12, 22]);
    expect(Object.keys(out)).toEqual([b.name]);
    adapter.destroy();
  });
  /**
   * @canon spec-compute-copies-back-only-the-slots-out-names
   */
  it("rejects a slot the program has no storage for", async () => {
    const adapter = createWgslCompute(program());
    await adapter.attach();
    adapter.setAttribute(a.name, Int32Array.from([1, 2]));
    adapter.setAttribute(b.name, Int32Array.from([10, 20]));
    await expect(adapter.compute({ c: new Int32Array(2) })).rejects.toThrow(/"c".*no storage slot/);
    adapter.destroy();
  });
});

describe.skipIf(!GPU_ENABLED)("createWgslCompute limits", () => {
  /**
   * An adapter binding 8, so the program stays small whatever the hardware binds.
   *
   * @canon spec-a-program-uses-as-many-storage-buffers-as-the-hardware-binds
   */
  it("rejects a program using more storage buffers than one shader stage can bind, without requesting a device", async () => {
    const gpu = navigator.gpu;
    let devicesRequested = 0;
    const adapter8 = {
      limits: { maxStorageBuffersPerShaderStage: 8 },
      requestDevice: () => {
        devicesRequested++;
        return gpu.requestAdapter().then((real) => real!.requestDevice());
      },
    };
    Object.assign(navigator, { gpu: { requestAdapter: async () => adapter8 } });
    try {
      const inputs = Array.from({ length: 9 }, () => instancedArray(1, "uint"));
      const program = Fn(() => {
        inputs[0]!.element(0).assign(inputs.slice(1).reduce((total, input) => total.add(input.element(0)), uint(0)));
      })();
      await expect(createWgslCompute(program).attach()).rejects.toThrow(
        /createWgslCompute: a compute program uses 9 storage buffers, more than the 8/,
      );
      expect(devicesRequested).toBe(0);
    } finally {
      Object.assign(navigator, { gpu });
    }
  });
});
