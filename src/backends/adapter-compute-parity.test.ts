import { describe, expect, it } from "vitest";
import {
  Fn,
  If,
  float,
  instancedArray,
  int,
  invocationIndex,
  storage,
  uint,
  uniform,
  vec2,
  vec4,
  type Node,
  type ShaderType,
  type UniformNode,
} from "../rmsl";
import { createJsCompute } from "../js";
import { createWasmCompute } from "../wasm";

describe("createJsCompute/createWasmCompute over a multi-root array", () => {
  /**
   * @canon spec-every-root-of-a-program-keeps-its-effects
   */
  it("applies every root's statements, not just the last one", () => {
    let force!: UniformNode<"float">;
    let dt!: UniformNode<"float">;
    const vel = instancedArray(3, "float");
    const pos = instancedArray(3, "float");

    // rootA writes velocity from a uniform, independently of rootB — the
    // shape a compute demo uses to compose several systems over shared
    // storage (one Fn per system, dispatched together as one array).
    const rootA = Fn(() => {
      force = uniform("float");
      const i = invocationIndex();
      vel.element(i).assign(force);
    })();

    // rootB reads the velocity rootA wrote, through its own read-only node
    // over the same buffer, and integrates it into position.
    const rootB = Fn(() => {
      const velIn = storage(vel.attribute, "float").toReadOnly();
      dt = uniform("float");
      const i = invocationIndex();
      pos.element(i).addAssign(velIn.element(i).mul(dt));
    })();

    const roots = [rootA, rootB];

    function run(adapter: ReturnType<typeof createJsCompute> | ReturnType<typeof createWasmCompute>) {
      const posData = new Float32Array([0, 10, 20]);
      adapter.setAttribute(pos.name, posData);
      adapter.setAttribute(vel.name, new Float32Array([0, 0, 0]));
      adapter.setUniform(force.name, 5);
      adapter.setUniform(dt.name, 2);
      adapter.compute();
      return Array.from(posData);
    }

    // If a backend only compiled the last root (rootB), rootA's velocity
    // write would never happen and pos would stay at its initial values —
    // exactly the bug this pins: rootA's uniform must reach the output.
    const want = [10, 20, 30]; // pos + force * dt = pos + 5 * 2

    expect(run(createJsCompute(roots, { name: "step" }))).toEqual(want);
    expect(run(createWasmCompute(roots, { name: "step" }))).toEqual(want);
  });
  /**
   * @canon spec-every-root-of-a-program-keeps-its-effects
   */
  it("a single root still works the same way through the array-accepting entry point", () => {
    let dt!: UniformNode<"float">;
    const vel = instancedArray(3, "float").toReadOnly();
    const pos = instancedArray(3, "float");
    const root = Fn(() => {
      dt = uniform("float");
      const i = invocationIndex();
      pos.element(i).addAssign(vel.element(i).mul(dt));
    })();

    function run(adapter: ReturnType<typeof createJsCompute> | ReturnType<typeof createWasmCompute>) {
      const posData = new Float32Array([0, 10, 20]);
      adapter.setAttribute(pos.name, posData);
      adapter.setAttribute(vel.name, new Float32Array([1, 2, 3]));
      adapter.setUniform(dt.name, 2);
      adapter.compute();
      return Array.from(posData);
    }

    const want = [2, 14, 26];
    expect(run(createJsCompute(root, { name: "step" }))).toEqual(want);
    expect(run(createWasmCompute(root, { name: "step" }))).toEqual(want);
  });
});

describe("createJsCompute/createWasmCompute reading and writing elements other than their own", () => {
  type ComputeAdapter = ReturnType<typeof createJsCompute> | ReturnType<typeof createWasmCompute>;

  function runBoth(root: Node<ShaderType>, storages: () => Record<string, Float32Array | Int32Array>) {
    return [createJsCompute, createWasmCompute].map((create) => {
      const adapter: ComputeAdapter = create(root, { name: "step" });
      const arrays = storages();
      for (const slot in arrays) adapter.setAttribute(slot, arrays[slot]);
      adapter.compute();
      return Object.fromEntries(Object.entries(arrays).map(([slot, array]) => [slot, Array.from(array)]));
    });
  }
  /**
   * @canon spec-a-swizzle-write-writes-the-components-it-names
   */
  it("writes single components and swizzles of a storage element", () => {
    const out = instancedArray(2, "vec4");
    const root = Fn(() => {
      const i = invocationIndex();
      const value = i.toFloat().add(1);
      out.element(i).x.assign(value);
      out.element(i).wy.assign(vec2(value.mul(10), value.mul(100)));
      out.element(i).element(2).assign(value.mul(1000));
    })();

    const [js, wasm] = runBoth(root, () => ({ [out.name]: new Float32Array(8) }));
    expect(js[out.name]).toEqual([1, 100, 1000, 10, 2, 200, 2000, 20]);
    expect(wasm).toEqual(js);
  });
  /**
   * @canon spec-an-element-write-writes-at-its-index
   */
  it("writes a component of a storage element by a computed index", () => {
    const out = instancedArray(2, "vec4");
    const root = Fn(() => {
      const i = invocationIndex();
      out.element(i).element(i.toInt().add(1)).assign(i.toFloat().add(1));
    })();

    const [js, wasm] = runBoth(root, () => ({ [out.name]: new Float32Array(8) }));
    expect(js[out.name]).toEqual([0, 1, 0, 0, 0, 0, 2, 0]);
    expect(wasm).toEqual(js);
  });
  /**
   * @canon spec-an-element-write-writes-at-its-index
   */
  it("writes and reads a column of a storage element by a computed index", () => {
    const out = instancedArray(2, "mat2");
    const columns = instancedArray(2, "vec2");
    const root = Fn(() => {
      const i = invocationIndex();
      out
        .element(i)
        .element(i.toInt())
        .assign(vec2(i.toFloat().add(1), i.toFloat().add(10)));
      columns.element(i).assign(out.element(i).element(i.toInt()).yx);
    })();

    const [js, wasm] = runBoth(root, () => ({
      [out.name]: new Float32Array(8),
      [columns.name]: new Float32Array(4),
    }));
    expect(js[out.name]).toEqual([1, 10, 0, 0, 0, 0, 2, 11]);
    expect(js[columns.name]).toEqual([10, 1, 11, 2]);
    expect(wasm).toEqual(js);
  });
  /**
   * @canon spec-a-swizzle-write-writes-the-components-it-names
   * @canon spec-an-element-write-writes-at-its-index
   */
  it("writes a component of a storage element's column, by a swizzle or an index", () => {
    const out = instancedArray(2, "mat2");
    const root = Fn(() => {
      const i = invocationIndex();
      out.element(i).element(1).y.assign(i.toFloat().add(1));
      out.element(i).element(int(0)).element(i.toInt()).assign(i.toFloat().add(10));
    })();

    const [js, wasm] = runBoth(root, () => ({ [out.name]: new Float32Array(8) }));
    expect(js[out.name]).toEqual([10, 0, 0, 1, 0, 11, 0, 2]);
    expect(wasm).toEqual(js);
  });
  /**
   * @canon spec-a-cpu-target-reaches-the-last-element-out-of-range
   */
  it("keeps a write by a column or component index computed outside a storage matrix inside it", () => {
    const out = instancedArray(2, "mat2");
    const root = Fn(() => {
      const i = invocationIndex();
      out.element(i).element(i.toInt().add(8)).y.assign(i.toFloat().add(1));
      out.element(i).element(int(0)).element(i.toInt().sub(5)).assign(i.toFloat().add(10));
    })();

    const [js, wasm] = runBoth(root, () => ({ [out.name]: new Float32Array(8) }));
    expect(js[out.name]).toEqual([0, 10, 0, 1, 0, 11, 0, 2]);
    expect(wasm).toEqual(js);
  });
  /**
   * @canon spec-a-cpu-target-reaches-the-last-element-out-of-range
   */
  it("reads a component and a column of a storage element by an index computed outside it", () => {
    const vectors = instancedArray(2, "vec4").toReadOnly();
    const matrices = instancedArray(2, "mat2").toReadOnly();
    const out = instancedArray(2, "vec4");
    const root = Fn(() => {
      const i = invocationIndex();
      const column = matrices.element(i).element(i.toInt().add(5));
      out
        .element(i)
        .assign(vec4(vectors.element(i).element(i.toInt().add(7)), vectors.element(i).y, column.x, column.y));
    })();

    const [js, wasm] = runBoth(root, () => ({
      [vectors.name]: Float32Array.of(1, 2, 3, 4, 5, 6, 7, 8),
      [matrices.name]: Float32Array.of(10, 11, 12, 13, 20, 21, 22, 23),
      [out.name]: new Float32Array(8),
    }));
    expect(js[out.name]).toEqual([4, 2, 12, 13, 8, 6, 22, 23]);
    expect(wasm).toEqual(js);
  });
  /**
   * @canon spec-a-cpu-compute-stage-runs-one-invocation-per-index
   */
  it("gathers from a neighbouring element", () => {
    const src = instancedArray(4, "float").toReadOnly();
    const dst = instancedArray(4, "float");
    const root = Fn(() => {
      const i = invocationIndex();
      dst.element(i).assign(src.element(i.add(1).mod(4)));
    })();

    const [js, wasm] = runBoth(root, () => ({
      [src.name]: new Float32Array([10, 20, 30, 40]),
      [dst.name]: new Float32Array(4),
    }));
    expect(js[dst.name]).toEqual([20, 30, 40, 10]);
    expect(wasm).toEqual(js);
  });
  /**
   * @canon spec-a-cpu-compute-stage-runs-one-invocation-per-index
   */
  it("scatters to another element", () => {
    const src = instancedArray(4, "int").toReadOnly();
    const dst = instancedArray(4, "int");
    const root = Fn(() => {
      const i = invocationIndex();
      dst.element(uint(3).sub(i)).assign(src.element(i).mul(2));
    })();

    const [js, wasm] = runBoth(root, () => ({
      [src.name]: new Int32Array([1, 2, 3, 4]),
      [dst.name]: new Int32Array(4),
    }));
    expect(js[dst.name]).toEqual([8, 6, 4, 2]);
    expect(wasm).toEqual(js);
  });
  /**
   * @canon spec-a-cpu-target-runs-invocations-in-index-order
   */
  it("sees an earlier invocation's write to the same buffer", () => {
    // Invocations run in index order on both CPU backends, so a prefix sum
    // written in place is well defined there, though not on a GPU.
    const acc = instancedArray(4, "float");
    const root = Fn(() => {
      const i = invocationIndex();
      If(i.greaterThan(uint(0)), () => {
        acc.element(i).addAssign(acc.element(i.sub(1)));
      });
    })();

    const [js, wasm] = runBoth(root, () => ({ [acc.name]: new Float32Array([1, 2, 3, 4]) }));
    expect(js[acc.name]).toEqual([1, 3, 6, 10]);
    expect(wasm).toEqual(js);
  });
  /**
   * @canon spec-a-vector-written-to-a-storage-element-is-copied-into-it
   */
  it("keeps the vector each invocation writes to its own element", () => {
    const buf = instancedArray(3, "vec2");
    const root = Fn(() => {
      buf.element(invocationIndex()).assign(vec2(invocationIndex().toFloat(), 1));
    })();

    const [js, wasm] = runBoth(root, () => ({ [buf.name]: new Float32Array(6) }));
    expect(js[buf.name]).toEqual([0, 1, 1, 1, 2, 1]);
    expect(wasm).toEqual(js);
  });
  /**
   * @canon spec-a-vector-written-to-a-storage-element-is-copied-into-it
   */
  it("leaves a literal vector alone when a component of the element it was written to changes", () => {
    const buf = instancedArray(3, "vec2");
    const root = Fn(() => {
      const i = invocationIndex();
      buf.element(i).assign(vec2(5, 6));
      If(i.equal(uint(0)), () => {
        buf.element(i).x.assign(float(7));
      });
    })();

    const [js, wasm] = runBoth(root, () => ({ [buf.name]: new Float32Array(6) }));
    expect(js[buf.name]).toEqual([7, 6, 5, 6, 5, 6]);
    expect(wasm).toEqual(js);
  });
  /**
   * @canon spec-a-float-index-of-a-storage-element-drops-its-fraction
   */
  it("reads and writes the element at the index without its fraction", () => {
    const buf = instancedArray(3, "vec2");
    const scalars = instancedArray(3, "float");
    const root = Fn(() => {
      If(invocationIndex().equal(uint(0)), () => {
        buf.element(float(1.5)).assign(buf.element(float(2.75)).add(vec2(10, 20)));
        scalars.element(float(1.5)).assign(scalars.element(float(2.75)).add(1));
      });
    })();

    const [js, wasm] = runBoth(root, () => ({
      [buf.name]: Float32Array.of(0, 0, 0, 0, 5, 6),
      [scalars.name]: Float32Array.of(0, 0, 7),
    }));
    expect(js[buf.name]).toEqual([0, 0, 15, 26, 5, 6]);
    expect(js[scalars.name]).toEqual([0, 8, 7]);
    expect(wasm).toEqual(js);
  });

  /**
   * @canon spec-a-float-index-of-a-storage-element-drops-its-fraction
   */
  it("reaches element 0 by a negative float index", () => {
    const buf = instancedArray(2, "vec2");
    const scalars = instancedArray(2, "float");
    const root = Fn(() => {
      If(invocationIndex().equal(uint(0)), () => {
        buf.element(float(-1.5)).assign(buf.element(float(-0.5)).add(vec2(10, 20)));
        scalars.element(float(-1.5)).assign(scalars.element(float(-3)).add(1));
      });
    })();

    const [js, wasm] = runBoth(root, () => ({
      [buf.name]: Float32Array.of(1, 2, 3, 4),
      [scalars.name]: Float32Array.of(7, 8),
    }));
    expect(js[buf.name]).toEqual([11, 22, 3, 4]);
    expect(js[scalars.name]).toEqual([8, 8]);
    expect(wasm).toEqual(js);
  });

  /**
   * @canon spec-a-float-index-of-a-storage-element-drops-its-fraction
   */
  it("reaches element 0 by a NaN float index, and no element past the end by a huge one", () => {
    const buf = instancedArray(2, "vec2");
    const zeros = instancedArray(1, "float").toReadOnly();
    const root = Fn(() => {
      If(invocationIndex().equal(uint(0)), () => {
        const zero = zeros.element(uint(0));
        const nan = zero.div(zero);
        buf.element(nan).assign(buf.element(nan).add(vec2(10, 20)));
        buf.element(float(1e10)).assign(vec2(7, 7));
      });
    })();

    const [js, wasm] = runBoth(root, () => ({
      [buf.name]: Float32Array.of(1, 2, 3, 4),
      [zeros.name]: new Float32Array(1),
    }));
    expect(js[buf.name]).toEqual([11, 22, 3, 4]);
    expect(wasm).toEqual(js);
  });
});

describe("createJsCompute/createWasmCompute reading back into out", () => {
  const a = instancedArray(2, "float");
  const b = instancedArray(2, "float");
  const program = () =>
    Fn(() => {
      const i = invocationIndex();
      a.element(i).assign(a.element(i).add(1));
      b.element(i).assign(b.element(i).add(2));
    })();

  for (const [name, create] of [
    ["JS", createJsCompute],
    ["WASM", createWasmCompute],
  ] as const) {
    /**
     * @canon spec-compute-copies-back-only-the-named-slots
     * @canon spec-a-cpu-adapter-computes-synchronously
     */
    it(`${name}: reads back only the slots out names`, () => {
      const adapter = create(program(), { name: "step" });
      adapter.setAttribute(a.name, new Float32Array([1, 2]));
      adapter.setAttribute(b.name, new Float32Array([10, 20]));
      const out = { [b.name]: new Float32Array(2) };
      adapter.compute(out);
      expect(Array.from(out[b.name]!)).toEqual([12, 22]);
    });
    /**
     * @canon spec-compute-refuses-a-slot-with-no-storage
     */
    it(`${name}: rejects a slot the program has no storage for`, () => {
      const adapter = create(program(), { name: "step" });
      adapter.setAttribute(a.name, new Float32Array([1, 2]));
      adapter.setAttribute(b.name, new Float32Array([10, 20]));
      expect(() => adapter.compute({ c: new Float32Array(2) })).toThrow(/"c".*no storage slot/);
    });
  }
});

describe("createJsCompute/createWasmCompute dispatching a count", () => {
  // Each program marks the element its own index names, so the marks a
  // dispatch leaves are the invocations that ran.
  const marks = instancedArray(8, "float");
  const marksProgram = () => Fn(() => marks.element(invocationIndex()).assign(float(1)))();

  for (const [name, create] of [
    ["JS", createJsCompute],
    ["WASM", createWasmCompute],
  ] as const) {
    /**
     * The count a caller names is the dispatch, so a buffer longer than the
     * count leaves its tail as the host passed it, and a count of zero runs
     * nothing at all.
     *
     * @canon spec-a-compute-call-takes-the-count-the-caller-names
     */
    it(`${name}: runs the count the caller names`, () => {
      const adapter = create(marksProgram(), { name: "step" });
      const data = new Float32Array(5);
      adapter.setAttribute(marks.name, data);
      adapter.compute(undefined, 3);
      expect(Array.from(data)).toEqual([1, 1, 1, 0, 0]);

      const none = new Float32Array(5);
      adapter.setAttribute(marks.name, none);
      adapter.compute(undefined, 0);
      expect(Array.from(none)).toEqual([0, 0, 0, 0, 0]);
    });

    /**
     * Given no count, the dispatch covers the first storage buffer the host
     * passed, so a longer buffer passed after it does not widen the dispatch.
     *
     * @canon spec-a-compute-call-takes-its-count-from-the-first-storage-buffer
     */
    it(`${name}: runs one invocation per element of the first buffer`, () => {
      const first = instancedArray(3, "float");
      const second = instancedArray(5, "float");
      const adapter = create(marksProgram(), { name: "step" });
      adapter.setAttribute(first.name, new Float32Array(3));
      adapter.setAttribute(second.name, new Float32Array(5));
      const data = new Float32Array(8);
      adapter.setAttribute(marks.name, data);
      adapter.compute();
      expect(Array.from(data)).toEqual([1, 1, 1, 0, 0, 0, 0, 0]);
    });

    /**
     * A buffer of vectors holds fewer elements than it has components, so the
     * first buffer of four-component elements counts a quarter as many
     * invocations as its array has components.
     *
     * @canon spec-a-vector-storage-buffer-counts-its-elements
     */
    it(`${name}: counts a vector buffer in elements, not components`, () => {
      const vectors = instancedArray(2, "vec4");
      const adapter = create(marksProgram(), { name: "step" });
      adapter.setAttribute(vectors, new Float32Array(8));
      const data = new Float32Array(8);
      adapter.setAttribute(marks.name, data);
      adapter.compute();
      expect(Array.from(data)).toEqual([1, 1, 0, 0, 0, 0, 0, 0]);
    });
  }
});

describe("createJsCompute/createWasmCompute dispatching every frame", () => {
  const buf = instancedArray(256, "vec4");
  const scale = uniform("float");
  const program = () =>
    Fn(() => {
      const i = invocationIndex();
      buf.element(i).assign(
        buf
          .element(i)
          .mul(scale)
          .add(vec4(1, 2, 3, 4)),
      );
    })();

  /** Heap growth over `count` calls of `step`, after a warm-up. A collection during the loop only shrinks it. */
  function heapGrowth(step: () => void, count: number): number {
    for (let i = 0; i < count; i++) step();
    const before = process.memoryUsage().heapUsed;
    for (let i = 0; i < count; i++) step();
    return process.memoryUsage().heapUsed - before;
  }

  for (const [name, create] of [
    ["JS", createJsCompute],
    ["WASM", createWasmCompute],
  ] as const) {
    /**
     * @canon spec-a-cpu-compute-dispatch-allocates-nothing
     */
    it(`${name}: allocates nothing per dispatch`, () => {
      const adapter = create(program(), { name: "step" });
      const data = new Float32Array(256 * 4);
      const out = { [buf.name]: new Float32Array(256 * 4) };
      adapter.setAttribute(buf.name, data);
      adapter.setUniform(scale, 0.5);
      const count = 20000;
      expect(heapGrowth(() => adapter.compute(), count)).toBeLessThan(count * 4);
      expect(heapGrowth(() => adapter.compute(out), count)).toBeLessThan(count * 4);
    });
  }
});
