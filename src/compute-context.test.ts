import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { Fn, instancedArray, invocationIndex, Loop, uint, uniform, type StorageNode } from "./rmsl";
import { createWgslContext } from "./wgsl";
import type { ComputeNode, StorageBufferAttribute, UniformNode } from "./rmsl";
import { GPU_ENABLED, installWebGpuGlobals } from "./testing/gpu";

/** What these tests need from a context, whichever backend runs it. */
type Context = {
  compute(nodes: ComputeNode | ComputeNode[]): void;
  setUniform(uniform: UniformNode<any>, value: number): void;
  write(attribute: StorageBufferAttribute, data: Uint32Array | Int32Array): void;
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
        setUniform: (u, v) => context.setUniform(u, v),
        write: (attribute, data) => context.write(attribute, data),
        async read(attribute) {
          const bytes = await context.getArrayBufferAsync(attribute);
          return Array.from(new attribute.arrayClass(bytes, 0, attribute.count));
        },
        destroy: () => context.destroy(),
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
    it("runs a batch in order, over buffers of different sizes", async () => {
      const context = await backend.create();
      const values = instancedArray(8, "uint");
      const total = instancedArray(1, "uint");
      const fill = Fn(() => {
        const i = invocationIndex();
        values.element(i).assign(i.mul(3));
      })().compute(8);
      const sum = Fn(() => {
        Loop(8, (i) => {
          total.element(0).assign(total.element(0).add(values.element(i)));
        });
      })().compute(1);

      context.compute([fill, sum]);

      expect(await context.read(values.attribute)).toEqual([0, 3, 6, 9, 12, 15, 18, 21]);
      expect(await context.read(total.attribute)).toEqual([84]);
      context.destroy();
    });

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

    it("writes a buffer and reads it back", async () => {
      const context = await backend.create();
      const values = instancedArray(4, "uint");
      context.write(values.attribute, Uint32Array.of(4294967295, 0, 7, 16777217));
      expect(await context.read(values.attribute)).toEqual([4294967295, 0, 7, 16777217]);
      context.destroy();
    });

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
