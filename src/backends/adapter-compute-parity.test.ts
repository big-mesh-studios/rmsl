import { describe, expect, it } from "vitest";
import { Fn, If, invocationIndex, storage, uint, uniform, type Node, type ShaderType, type UniformNode } from "../rmsl";
import { createJsCompute } from "../js";
import { createWasmCompute } from "../wasm";

describe("createJsCompute/createWasmCompute over a multi-root array", () => {
  it("applies every root's statements, not just the last one", () => {
    let force!: UniformNode<"float">;
    let dt!: UniformNode<"float">;

    // rootA writes velocity from a uniform, independently of rootB — the
    // shape a compute demo uses to compose several systems over shared
    // storage (one Fn per system, dispatched together as one array).
    const rootA = Fn(() => {
      const vel = storage("vel", "float", { access: "read_write" });
      force = uniform("float");
      const i = invocationIndex();
      vel.element(i).assign(force);
      return vel.element(i);
    })();

    // rootB reads the velocity rootA wrote and integrates it into position.
    const rootB = Fn(() => {
      const vel = storage("vel", "float");
      const pos = storage("pos", "float", { access: "read_write" });
      dt = uniform("float");
      const i = invocationIndex();
      pos.element(i).addAssign(vel.element(i).mul(dt));
      return pos.element(i);
    })();

    const roots = [rootA, rootB];

    function run(adapter: ReturnType<typeof createJsCompute> | ReturnType<typeof createWasmCompute>) {
      const pos = new Float32Array([0, 10, 20]);
      const vel = new Float32Array([0, 0, 0]);
      adapter.setAttribute("pos", pos);
      adapter.setAttribute("vel", vel);
      adapter.setUniform(force.name, 5);
      adapter.setUniform(dt.name, 2);
      adapter.compute();
      return Array.from(pos);
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
    const root = Fn(() => {
      const vel = storage("vel", "float");
      const pos = storage("pos", "float", { access: "read_write" });
      dt = uniform("float");
      const i = invocationIndex();
      pos.element(i).addAssign(vel.element(i).mul(dt));
      return pos.element(i);
    })();

    function run(adapter: ReturnType<typeof createJsCompute> | ReturnType<typeof createWasmCompute>) {
      const pos = new Float32Array([0, 10, 20]);
      const vel = new Float32Array([1, 2, 3]);
      adapter.setAttribute("pos", pos);
      adapter.setAttribute("vel", vel);
      adapter.setUniform(dt.name, 2);
      adapter.compute();
      return Array.from(pos);
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

  it("gathers from a neighbouring element", () => {
    const root = Fn(() => {
      const src = storage("src", "float");
      const dst = storage("dst", "float", { access: "read_write" });
      const i = invocationIndex();
      dst.element(i).assign(src.element(i.add(1).mod(4)));
    })();

    const [js, wasm] = runBoth(root, () => ({
      src: new Float32Array([10, 20, 30, 40]),
      dst: new Float32Array(4),
    }));
    expect(js.dst).toEqual([20, 30, 40, 10]);
    expect(wasm).toEqual(js);
  });

  it("scatters to another element", () => {
    const root = Fn(() => {
      const src = storage("src", "int");
      const dst = storage("dst", "int", { access: "read_write" });
      const i = invocationIndex();
      dst.element(uint(3).sub(i)).assign(src.element(i).mul(2));
    })();

    const [js, wasm] = runBoth(root, () => ({
      src: new Int32Array([1, 2, 3, 4]),
      dst: new Int32Array(4),
    }));
    expect(js.dst).toEqual([8, 6, 4, 2]);
    expect(wasm).toEqual(js);
  });

  it("sees an earlier invocation's write to the same buffer", () => {
    // Invocations run in index order on both CPU backends, so a prefix sum
    // written in place is well defined there, though not on a GPU.
    const root = Fn(() => {
      const acc = storage("acc", "float", { access: "read_write" });
      const i = invocationIndex();
      If(i.greaterThan(uint(0)), () => {
        acc.element(i).addAssign(acc.element(i.sub(1)));
      });
    })();

    const [js, wasm] = runBoth(root, () => ({ acc: new Float32Array([1, 2, 3, 4]) }));
    expect(js.acc).toEqual([1, 3, 6, 10]);
    expect(wasm).toEqual(js);
  });
});
