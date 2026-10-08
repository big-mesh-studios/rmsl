import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { Fn, instancedArray, int, invocationIndex, uint, uniform, vec4 } from "../../rmsl";
import { createWgslCompute } from "./adapter-wgsl";
import { GPU_ENABLED, installWebGpuGlobals } from "../../testing/gpu";

let uninstall: (() => void) | undefined;

beforeAll(async () => {
  if (GPU_ENABLED) uninstall = await installWebGpuGlobals();
});

afterAll(() => uninstall?.());

describe.skipIf(!GPU_ENABLED)("createWgslCompute reading back", () => {
  /**
   * Each dispatch adds 1, so the first call reads 1 and the second 2.
   *
   * @canon spec-overlapping-compute-calls-each-read-back-their-own-result
   */
  it("fills the out of each of two overlapping compute calls", async () => {
    const values = instancedArray(4, "float");
    const program = Fn(() => {
      const i = invocationIndex();
      values.element(i).assign(values.element(i).add(1));
    })();
    const adapter = createWgslCompute(program);
    await adapter.attach();
    adapter.setAttribute(values.name, new Float32Array(4));
    const first = { [values.name]: new Float32Array(4) };
    const second = { [values.name]: new Float32Array(4) };
    await Promise.all([adapter.compute(first), adapter.compute(second)]);
    expect(Array.from(first[values.name])).toEqual([1, 1, 1, 1]);
    expect(Array.from(second[values.name])).toEqual([2, 2, 2, 2]);
    adapter.destroy();
  });
});

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
   * @canon spec-compute-copies-back-only-the-named-slots
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
   * @canon spec-compute-refuses-a-slot-with-no-storage
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

describe.skipIf(!GPU_ENABLED)("createWgslCompute dispatching a count", () => {
  // A workgroup of 4, so a count of 2 dispatches one workgroup and a count
  // of 8 two: the marks a dispatch leaves are then the invocations it ran.
  const marks = instancedArray(8, "int");
  // Every storage the adapter binds has to be one the program reads, so
  // `touch` is what keeps a buffer's own alive for the dispatch.
  const marksProgram = (touch: () => void) =>
    Fn(() => {
      touch();
      marks.element(invocationIndex()).assign(int(1));
    })();

  /**
   * The count a caller names is the dispatch: a count of zero dispatches no
   * workgroup at all, and a count below one workgroup runs only that one.
   *
   * @canon spec-a-compute-call-takes-the-count-the-caller-names
   */
  it("runs the count the caller names", async () => {
    const adapter = createWgslCompute(
      marksProgram(() => {}),
      { workgroupSize: 4 },
    );
    await adapter.attach();
    const out = { [marks.name]: new Int32Array(8) };
    adapter.setAttribute(marks.name, new Int32Array(8));
    await adapter.compute(out, 0);
    expect(Array.from(out[marks.name]!)).toEqual(new Array(8).fill(0));
    await adapter.compute(out, 2);
    expect(Array.from(out[marks.name]!.slice(0, 4))).toEqual([1, 1, 1, 1]);
    adapter.destroy();
  });

  /**
   * Given no count, the dispatch covers the first storage buffer the host
   * passed, so a longer buffer passed after it does not widen the dispatch.
   *
   * @canon spec-a-compute-call-takes-its-count-from-the-first-storage-buffer
   */
  it("runs one invocation per element of the first buffer", async () => {
    const first = instancedArray(2, "int");
    const second = instancedArray(8, "int");
    const adapter = createWgslCompute(
      marksProgram(() => {
        first.element(int(0)).assign(int(0));
        second.element(int(0)).assign(int(0));
      }),
      { workgroupSize: 4 },
    );
    await adapter.attach();
    adapter.setAttribute(first.name, new Int32Array(2));
    adapter.setAttribute(second.name, new Int32Array(8));
    const out = { [marks.name]: new Int32Array(8) };
    adapter.setAttribute(marks.name, new Int32Array(8));
    await adapter.compute(out);
    expect(Array.from(out[marks.name]!)).toEqual([1, 1, 1, 1, 0, 0, 0, 0]);
    adapter.destroy();
  });

  /**
   * A buffer of vectors holds fewer elements than it has components, so the
   * first buffer of four-component elements counts a quarter as many
   * invocations as its array has components.
   *
   * @canon spec-a-vector-storage-buffer-counts-its-elements
   */
  it("counts a vector buffer in elements, not components", async () => {
    const vectors = instancedArray(2, "vec4");
    const adapter = createWgslCompute(
      marksProgram(() => {
        vectors.element(int(0)).assign(vec4(0));
      }),
      { workgroupSize: 4 },
    );
    await adapter.attach();
    adapter.setAttribute(vectors.name, new Float32Array(8));
    const out = { [marks.name]: new Int32Array(8) };
    adapter.setAttribute(marks.name, new Int32Array(8));
    await adapter.compute(out);
    expect(Array.from(out[marks.name]!)).toEqual([1, 1, 1, 1, 0, 0, 0, 0]);
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
