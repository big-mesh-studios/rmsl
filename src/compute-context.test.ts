import { describe, it, expect, beforeAll, afterAll } from "vitest";
import {
  Fn,
  instancedArray,
  int,
  invocationIndex,
  Loop,
  storage,
  uint,
  uniform,
  uniformArray,
  vec2,
  vec3,
  vec4,
  type StorageNode,
  texture,
} from "./rmsl";
import { createWgslContext } from "./wgsl";
import { createWasmContext } from "./wasm";
import { StorageBufferAttribute, type ComputeNode, type UniformArrayNode, type UniformNode } from "./rmsl";
import { GPU_ENABLED, installWebGpuGlobals } from "./testing/gpu";

/** What these tests need from a context, whichever backend runs it. */
type Context = {
  compute(nodes: ComputeNode | ComputeNode[]): void;
  setUniform(uniform: UniformNode<any> | UniformArrayNode<any>, value: number | number[] | number[][]): void;
  write(attribute: StorageBufferAttribute, data: Uint32Array | Int32Array | Float32Array, offset?: number): void;
  read(attribute: StorageBufferAttribute): Promise<number[]>;
  destroy(): void;
};

const backends: { name: string; enabled: boolean; create(): Promise<Context> }[] = [
  {
    name: "WGSL",
    enabled: GPU_ENABLED,
    async create() {
      const context = await createWgslContext();
      return {
        compute: (nodes) => context.compute(nodes),
        setUniform: (u, v) => context.setUniform(u as any, v as any),
        write: (attribute, data, offset) => context.write(attribute, data, offset),
        async read(attribute) {
          const bytes = await context.getArrayBufferAsync(attribute);
          return Array.from(new attribute.arrayClass(bytes, 0, attribute.count * attribute.itemSize));
        },
        destroy: () => context.destroy(),
      };
    },
  },
  {
    name: "WASM",
    enabled: true,
    async create() {
      const context = createWasmContext();
      return {
        compute: (nodes) => context.compute(nodes),
        setUniform: (u, v) => context.setUniform(u as any, v as any),
        write: (attribute, data, offset) => context.write(attribute, data, offset),
        async read(attribute) {
          return Array.from(
            new attribute.arrayClass(context.getArrayBuffer(attribute), 0, attribute.count * attribute.itemSize),
          );
        },
        destroy: () => {},
      };
    },
  },
];

let uninstall: (() => void) | undefined;
beforeAll(async () => {
  if (GPU_ENABLED) uninstall = await installWebGpuGlobals();
});
afterAll(() => uninstall?.());

/**
 * One inclusive prefix-sum pass (Hillis–Steele): each element adds the one
 * `stride` places before it. A scan over n elements runs this for strides
 * 1, 2, 4, … past n, alternating between two buffers.
 */
function scanPass(src: StorageNode<"uint">, dst: StorageNode<"uint">, stride: number, count: number): ComputeNode {
  return Fn(() => {
    const i = invocationIndex();
    const hasLeft = i.greaterThanEqual(uint(stride));
    const left = src.element(hasLeft.select(i.sub(uint(stride)), uint(0)));
    dst.element(i).assign(src.element(i).add(hasLeft.select(left, uint(0))));
  })().compute(count);
}

for (const backend of backends) {
  describe.skipIf(!backend.enabled)(`${backend.name} compute context`, () => {
    /**
     * @canon spec-a-compute-context-runs-programs-over-shared-buffers
     */
    it("runs a batch in order, over buffers of different sizes", async () => {
      const context = await backend.create();
      const values = instancedArray(8, "uint");
      const total = instancedArray(1, "uint");
      const fill = Fn(() => {
        const i = invocationIndex();
        values.element(i).assign(i.mul(3));
      })().compute(8);
      const sum = Fn(() => {
        Loop(8, ({ i }) => {
          total.element(0).assign(total.element(0).add(values.element(i)));
        });
      })().compute(1);

      context.compute([fill, sum]);

      expect(await context.read(values.attribute)).toEqual([0, 3, 6, 9, 12, 15, 18, 21]);
      expect(await context.read(total.attribute)).toEqual([84]);
      context.destroy();
    });
    /**
     * @canon spec-a-dispatch-skips-the-indices-past-its-count
     */
    it("bounds each program by its own count", async () => {
      const context = await backend.create();
      const values = instancedArray(8, "uint");
      const mark = Fn(() => {
        values.element(invocationIndex()).assign(uint(1));
      })().compute(5);

      context.compute(mark);

      expect(await context.read(values.attribute)).toEqual([1, 1, 1, 1, 1, 0, 0, 0]);
      context.destroy();
    });
    /**
     * @canon spec-a-context-keeps-a-uniform-until-it-is-set-again
     */
    it("passes a uniform to the programs that read it", async () => {
      const context = await backend.create();
      const values = instancedArray(Int32Array.of(10, -10), "int");
      const offset = uniform("int");
      const add = Fn(() => {
        const i = invocationIndex();
        values.element(i).assign(values.element(i).add(offset));
      })().compute(2);

      context.setUniform(offset, -3);
      context.compute(add);

      expect(await context.read(values.attribute)).toEqual([7, -13]);
      context.destroy();
    });
    /**
     * @canon spec-a-context-keeps-a-uniform-until-it-is-set-again
     */
    it("keeps a uniform's value from when it was set, until it is set again", async () => {
      const context = await backend.create();
      const out = instancedArray(2, "float");
      const scalars = uniformArray("float", 2);
      const read = Fn(() => {
        out.element(0).assign(scalars.element(0));
        out.element(1).assign(scalars.element(1));
      })().compute(1);

      const values = [1, 2];
      context.setUniform(scalars, values);
      context.compute(read);
      values[0] = 5;
      context.compute(read);
      expect(await context.read(out.attribute)).toEqual([1, 2]);

      context.setUniform(scalars, values);
      context.compute(read);
      expect(await context.read(out.attribute)).toEqual([5, 2]);
      context.destroy();
    });
    /**
     * @canon spec-a-context-keeps-a-uniform-until-it-is-set-again
     */
    it("passes uniform arrays and matrices, laid out as each backend reads them", async () => {
      const context = await backend.create();
      const out = instancedArray(7, "float");
      const scalars = uniformArray("float", 3);
      const pairs = uniformArray("vec2", 2);
      const matrix = uniform("mat3");
      const read = Fn(() => {
        out.element(0).assign(scalars.element(0));
        out.element(1).assign(scalars.element(1));
        out.element(2).assign(scalars.element(2));
        out.element(3).assign(pairs.element(1).x);
        out.element(4).assign(pairs.element(1).y);
        out.element(5).assign(matrix.mul(vec3(0, 1, 0)).z);
        out.element(6).assign(matrix.mul(vec3(0, 0, 1)).x);
      })().compute(1);

      context.setUniform(scalars, [1, 2, 3]);
      context.setUniform(pairs, [
        [4, 5],
        [6, 7],
      ]);
      context.setUniform(matrix, [10, 11, 12, 13, 14, 15, 16, 17, 18]);
      context.compute(read);

      expect(await context.read(out.attribute)).toEqual([1, 2, 3, 6, 7, 15, 16]);
      context.destroy();
    });
    /**
     * @canon spec-an-unset-uniform-reads-zero
     */
    it("reads an unset uniform as zero", async () => {
      const context = await backend.create();
      const out = instancedArray(3, "float");
      const k = uniform("float");
      const ks = uniformArray("float", 2);
      const v = uniform("vec2");
      const read = Fn(() => {
        out.element(0).assign(k.add(1));
        out.element(1).assign(ks.element(1).add(1));
        out.element(2).assign(v.y.add(1));
      })().compute(1);

      context.compute(read);

      expect(await context.read(out.attribute)).toEqual([1, 1, 1]);
      context.destroy();
    });
    /**
     * @canon spec-a-swizzle-write-writes-the-components-it-names
     */
    it("writes single components and swizzles of a storage element", async () => {
      const context = await backend.create();
      const out = instancedArray(2, "vec4");
      const write = Fn(() => {
        const i = invocationIndex();
        const value = i.toFloat().add(1);
        out.element(i).x.assign(value);
        out.element(i).wy.assign(vec2(value.mul(10), value.mul(100)));
        out.element(i).element(2).assign(value.mul(1000));
      })().compute(2);

      context.compute(write);

      expect(await context.read(out.attribute)).toEqual([1, 100, 1000, 10, 2, 200, 2000, 20]);
      context.destroy();
    });
    /**
     * @canon spec-an-element-write-writes-at-its-index
     */
    it("writes a component of a storage element by a computed index", async () => {
      const context = await backend.create();
      const out = instancedArray(2, "vec4");
      const write = Fn(() => {
        const i = invocationIndex();
        out.element(i).element(i.toInt().add(1)).assign(i.toFloat().add(1));
      })().compute(2);

      context.compute(write);

      expect(await context.read(out.attribute)).toEqual([0, 1, 0, 0, 0, 0, 2, 0]);
      context.destroy();
    });
    /**
     * @canon spec-an-element-write-writes-at-its-index
     */
    it("writes and reads a column of a storage element by a computed index", async () => {
      const context = await backend.create();
      const out = instancedArray(2, "mat2");
      const columns = instancedArray(2, "vec2");
      const write = Fn(() => {
        const i = invocationIndex();
        out
          .element(i)
          .element(i.toInt())
          .assign(vec2(i.toFloat().add(1), i.toFloat().add(10)));
        columns.element(i).assign(out.element(i).element(i.toInt()).yx);
      })().compute(2);

      context.compute(write);

      expect(await context.read(out.attribute)).toEqual([1, 10, 0, 0, 0, 0, 2, 11]);
      expect(await context.read(columns.attribute)).toEqual([10, 1, 11, 2]);
      context.destroy();
    });
    /**
     * @canon spec-an-element-write-writes-at-its-index
     */
    it("writes columns of storage elements of every unpadded matrix size", async () => {
      const context = await backend.create();
      const wide = instancedArray(2, "mat4");
      const narrow = instancedArray(2, "mat3x2");
      const write = Fn(() => {
        const i = invocationIndex();
        const value = i.toFloat().add(1);
        wide
          .element(i)
          .element(i.toInt().add(2))
          .assign(vec4(value, value.add(1), value.add(2), value.add(3)));
        narrow
          .element(i)
          .element(2)
          .assign(vec2(value, value.add(10)));
      })().compute(2);

      context.compute(write);

      const wideExpected = new Array(32).fill(0);
      wideExpected.splice(8, 4, 1, 2, 3, 4);
      wideExpected.splice(28, 4, 2, 3, 4, 5);
      expect(await context.read(wide.attribute)).toEqual(wideExpected);
      expect(await context.read(narrow.attribute)).toEqual([0, 0, 0, 0, 1, 11, 0, 0, 0, 0, 2, 12]);
      context.destroy();
    });
    /**
     * @canon exception-a-vec3-storage-element-is-padded-only-on-the-gpu
     */
    it("reads and writes elements with columns of three, which WGSL pads to four", async () => {
      const context = await backend.create();
      const vectors = instancedArray(Float32Array.of(1, 2, 3, 4, 5, 6), "vec3");
      const matrices = instancedArray(2, "mat2x3");
      const square = instancedArray(
        Float32Array.from({ length: 9 }, (_, k) => k + 1),
        "mat3",
      );
      const columns = instancedArray(2, "vec3");
      const program = Fn(() => {
        const i = invocationIndex();
        vectors.element(i).assign(vectors.element(i).zxy.mul(10));
        matrices.element(i).element(1).y.assign(vectors.element(i).x);
        columns.element(i).assign(square.element(uint(0)).element(i.toInt().add(1)));
      })().compute(2);

      context.write(matrices.attribute, Float32Array.of(7, 8, 9, 6), 1);
      context.compute(program);

      expect(await context.read(vectors.attribute)).toEqual([30, 10, 20, 60, 40, 50]);
      expect(await context.read(matrices.attribute)).toEqual([0, 0, 0, 0, 30, 0, 7, 8, 9, 6, 60, 0]);
      expect(await context.read(columns.attribute)).toEqual([4, 5, 6, 7, 8, 9]);
      context.destroy();
    });
    /**
     * @canon spec-a-swizzle-write-writes-the-components-it-names
     * @canon spec-an-element-write-writes-at-its-index
     */
    it("writes a component of a storage element's column, by a swizzle or an index", async () => {
      const context = await backend.create();
      const out = instancedArray(2, "mat2");
      const write = Fn(() => {
        const i = invocationIndex();
        out.element(i).element(1).y.assign(i.toFloat().add(1));
        out.element(i).element(int(0)).element(i.toInt()).assign(i.toFloat().add(10));
      })().compute(2);

      context.compute(write);

      expect(await context.read(out.attribute)).toEqual([10, 0, 0, 1, 0, 11, 0, 2]);
      context.destroy();
    });
    /**
     * @canon spec-a-context-writes-a-buffer-and-reads-it-back
     */
    it("writes a buffer and reads it back", async () => {
      const context = await backend.create();
      const values = instancedArray(4, "uint");
      context.write(values.attribute, Uint32Array.of(4294967295, 0, 7, 16777217));
      expect(await context.read(values.attribute)).toEqual([4294967295, 0, 7, 16777217]);
      context.destroy();
    });
    /**
     * @canon spec-a-context-writes-a-buffer-and-reads-it-back
     */
    it("writes values converted to the buffer's type, and refuses a write past its end", async () => {
      const context = await backend.create();
      const values = instancedArray(3, "uint");
      context.write(values.attribute, Float32Array.of(1, 2), 1);
      expect(await context.read(values.attribute)).toEqual([0, 1, 2]);
      expect(() => context.write(values.attribute, Uint32Array.of(1, 2), 2)).toThrow(/past the end/);
      context.destroy();
    });
    /**
     * @canon spec-a-compute-context-runs-programs-over-shared-buffers
     */
    it("computes a prefix sum with one program per pass", async () => {
      const context = await backend.create();
      const n = 1000;
      const input = Uint32Array.from({ length: n }, (_, i) => (i * 7919) % 13);
      const buffers = [instancedArray(input, "uint"), instancedArray(n, "uint")];
      const passes: ComputeNode[] = [];
      let from = 0;
      for (let stride = 1; stride < n; stride *= 2, from = 1 - from) {
        passes.push(scanPass(buffers[from]!, buffers[1 - from]!, stride, n));
      }

      context.compute(passes);

      let running = 0;
      const expected = Array.from(input, (v) => (running += v));
      expect(await context.read(buffers[from]!.attribute)).toEqual(expected);
      context.destroy();
    });
  });
}

describe.skipIf(!GPU_ENABLED)("WGSL compute context buffers", () => {
  /**
   * @canon spec-a-wgsl-buffer-feeds-a-draw-without-a-copy
   */
  it("can be bound as vertex data by a render pipeline on the same device", async () => {
    const context = await createWgslContext();
    const positions = instancedArray(4, "vec2");
    const buffer = context.buffer(positions.attribute);
    expect(buffer.usage & GPUBufferUsage.VERTEX).toBe(GPUBufferUsage.VERTEX);
    expect(buffer.size).toBe(4 * 2 * 4);
    context.destroy();
  });
  /**
   * @canon exception-a-vec3-storage-element-is-padded-only-on-the-gpu
   */
  it("gives a vec3 element 16 bytes in its buffer, as WGSL lays it out", async () => {
    const context = await createWgslContext();
    const positions = instancedArray(4, "vec3");
    expect(context.buffer(positions.attribute).size).toBe(4 * 16);
    context.destroy();
  });
  /**
   * @canon spec-a-context-runs-no-program-it-cannot-bind
   */
  it("rejects a program over a buffer laid out before a storage node named its type", async () => {
    const context = await createWgslContext();
    const attribute = new StorageBufferAttribute(2, 6);
    context.write(attribute, Float32Array.of(1, 2));
    const matrices = storage(attribute, "mat2x3");
    const read = Fn(() => {
      matrices.element(0).toVar();
    })().compute(1);
    expect(() => context.compute(read)).toThrow(/laid out before a storage node named its type/);
    context.destroy();
  });
  /**
   * @canon spec-a-context-writes-a-buffer-and-reads-it-back
   */
  it("reads back an empty buffer as no bytes, as the WASM context does", async () => {
    const context = await createWgslContext();
    const empty = instancedArray(0, "uint");
    expect((await context.getArrayBufferAsync(empty.attribute)).byteLength).toBe(0);
    expect(createWasmContext().getArrayBuffer(empty.attribute).byteLength).toBe(0);
    context.destroy();
  });
});

describe.skipIf(!GPU_ENABLED)("WGSL compute context limits", () => {
  /**
   * @canon spec-a-context-runs-no-program-it-cannot-bind
   */
  it("rejects a program that samples a texture, which the context doesn't bind", async () => {
    const context = await createWgslContext();
    const out = instancedArray(1, "float");
    const image = uniform("sampler2D");
    const sample = Fn(() => {
      out.element(0).assign(texture(image, vec2(0.5, 0.5)).x);
    })().compute(1);
    expect(() => context.compute(sample)).toThrow(/sample textures/);
    context.destroy();
  });
  /**
   * @canon spec-a-program-uses-as-many-storage-buffers-as-the-hardware-binds
   */
  it("runs a program using more storage buffers than WebGPU's default, where the hardware binds more", async () => {
    const context = await createWgslContext();
    if (context.device.limits.maxStorageBuffersPerShaderStage <= 8) return context.destroy();
    const inputs = Array.from({ length: 8 }, (_, k) => instancedArray(Uint32Array.of(k + 1), "uint"));
    const out = instancedArray(1, "uint");
    const sum = Fn(() => {
      out.element(0).assign(inputs.reduce((total, input) => total.add(input.element(0)), uint(0)));
    })().compute(1);
    context.compute(sum);
    expect(Array.from(new Uint32Array(await context.getArrayBufferAsync(out.attribute)))).toEqual([36]);
    context.destroy();
  });

  /**
   * A device with WebGPU's default limit of 8, so the program stays small whatever the hardware binds.
   *
   * @canon spec-a-program-uses-as-many-storage-buffers-as-the-hardware-binds
   */
  it("rejects a program using more storage buffers than one shader stage can bind", async () => {
    const device = await (await navigator.gpu.requestAdapter())!.requestDevice();
    const context = await createWgslContext({ device });
    const inputs = Array.from({ length: 8 }, () => instancedArray(1, "uint"));
    const out = instancedArray(1, "uint");
    const sum = Fn(() => {
      out.element(0).assign(inputs.reduce((total, input) => total.add(input.element(0)), uint(0)));
    })().compute(1);
    expect(() => context.compute(sum)).toThrow(
      /createWgslContext: a compute program uses 9 storage buffers, more than the 8/,
    );
    context.destroy();
    device.destroy();
  });
  /**
   * @canon spec-a-context-runs-no-program-it-cannot-bind
   */
  it("rejects a workgroup size past the device's limit", async () => {
    const context = await createWgslContext();
    const out = instancedArray(1, "uint");
    const limit = context.device.limits.maxComputeInvocationsPerWorkgroup;
    const wide = Fn(() => {
      out.element(0).assign(uint(1));
    })().compute(1, limit * 2);
    expect(() => context.compute(wide)).toThrow(/workgroup size/);
    context.destroy();
  });
  /**
   * @canon spec-a-context-runs-no-program-it-cannot-bind
   */
  it("rejects a dispatch with more workgroups than the device allows", async () => {
    const context = await createWgslContext();
    const out = instancedArray(1, "uint");
    const limit = context.device.limits.maxComputeWorkgroupsPerDimension;
    const wide = Fn(() => {
      out.element(0).assign(uint(1));
    })().compute((limit + 1) * 64);
    expect(() => context.compute(wide)).toThrow(/workgroups/);
    expect(await context.getArrayBufferAsync(out.attribute)).toEqual(new Uint32Array(1).buffer);
    context.destroy();
  });
});
