import { describe, expect, it } from "vitest";
import {
  Fn,
  If,
  instancedArray,
  int,
  invocationIndex,
  storage,
  uint,
  uniform,
  vec2,
  type Node,
  type ShaderType,
  type UniformNode,
} from "../rmsl";
import { createJsCompute } from "../js";
import { createWasmCompute } from "../wasm";

describe("createJsCompute/createWasmCompute over a multi-root array", () => {
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

  it("writes single components and swizzles of a storage element", () => {
    const out = instancedArray(2, "vec4");
    const root = Fn(() => {
      const i = invocationIndex();
      const value = i.toFloat().add(1);
      out.element(i).x.assign(value);
      out.element(i).wy.assign(vec2(value.mul(10), value.mul(100)));
      out.element(i).element(2).assign(value.mul(1000));
    })();

    // Both adapters hold a vector storage buffer as one array per element.
    const results = [createJsCompute, createWasmCompute].map((create) => {
      const adapter = create(root, { name: "step" });
      const data = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ];
      adapter.setAttribute(out.name, data as any);
      adapter.compute();
      return data;
    });

    expect(results[0]).toEqual([
      [1, 100, 1000, 10],
      [2, 200, 2000, 20],
    ]);
    expect(results[1]).toEqual(results[0]);
  });

  it("writes a component of a storage element by a computed index", () => {
    const out = instancedArray(2, "vec4");
    const root = Fn(() => {
      const i = invocationIndex();
      out.element(i).element(i.toInt().add(1)).assign(i.toFloat().add(1));
    })();

    const results = [createJsCompute, createWasmCompute].map((create) => {
      const adapter = create(root, { name: "step" });
      const data = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ];
      adapter.setAttribute(out.name, data as any);
      adapter.compute();
      return data;
    });

    expect(results[0]).toEqual([
      [0, 1, 0, 0],
      [0, 0, 2, 0],
    ]);
    expect(results[1]).toEqual(results[0]);
  });

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

    const results = [createJsCompute, createWasmCompute].map((create) => {
      const adapter = create(root, { name: "step" });
      const data = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ];
      const read = [
        [0, 0],
        [0, 0],
      ];
      adapter.setAttribute(out.name, data as any);
      adapter.setAttribute(columns.name, read as any);
      adapter.compute();
      return [data, read];
    });

    expect(results[0]).toEqual([
      [
        [1, 10, 0, 0],
        [0, 0, 2, 11],
      ],
      [
        [10, 1],
        [11, 2],
      ],
    ]);
    expect(results[1]).toEqual(results[0]);
  });

  it("writes a component of a storage element's column, by a swizzle or an index", () => {
    const out = instancedArray(2, "mat2");
    const root = Fn(() => {
      const i = invocationIndex();
      out.element(i).element(1).y.assign(i.toFloat().add(1));
      out.element(i).element(int(0)).element(i.toInt()).assign(i.toFloat().add(10));
    })();

    const results = [createJsCompute, createWasmCompute].map((create) => {
      const adapter = create(root, { name: "step" });
      const data = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ];
      adapter.setAttribute(out.name, data as any);
      adapter.compute();
      return data;
    });

    expect(results[0]).toEqual([
      [10, 0, 0, 1],
      [0, 11, 0, 2],
    ]);
    expect(results[1]).toEqual(results[0]);
  });

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
    it(`${name}: reads back only the slots out names`, () => {
      const adapter = create(program(), { name: "step" });
      adapter.setAttribute(a.name, new Float32Array([1, 2]));
      adapter.setAttribute(b.name, new Float32Array([10, 20]));
      const out = { [b.name]: new Float32Array(2) };
      adapter.compute(out);
      expect(Array.from(out[b.name]!)).toEqual([12, 22]);
    });

    it(`${name}: rejects a slot the program has no storage for`, () => {
      const adapter = create(program(), { name: "step" });
      adapter.setAttribute(a.name, new Float32Array([1, 2]));
      adapter.setAttribute(b.name, new Float32Array([10, 20]));
      expect(() => adapter.compute({ c: new Float32Array(2) })).toThrow(/"c".*no storage slot/);
    });
  }
});
