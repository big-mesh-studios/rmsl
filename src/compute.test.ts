import { describe, it, expect } from "vitest";
import { Fn, invocationIndex, instancedArray, compute, ComputeNode } from "./rmsl";
import { compile } from "./wgsl";

describe("compute()", () => {
  const values = instancedArray(16, "uint");
  const increment = Fn(() => {
    const i = invocationIndex();
    values.element(i).assign(values.element(i).add(1));
  });

  it("carries a dispatch count and a workgroup size, as a method or a function", () => {
    const node = increment().compute(10, 32);
    expect(node).toBeInstanceOf(ComputeNode);
    expect(node.count).toBe(10);
    expect(node.workgroupSize).toBe(32);
    expect(compute(increment(), 5).workgroupSize).toBe(64);
  });

  it("compiles a bounds check against its count uniform, not a buffer length", () => {
    const node = increment().compute(10, 32);
    const program = compile({ stage: "compute" }, node);
    expect(program.workgroupSize).toBe(32);
    expect(program.code).toContain("@workgroup_size(32)");
    expect(program.code).not.toContain("arrayLength");
    expect(program.resources).toContainEqual(expect.objectContaining({ kind: "uniform", name: node.countNode.name }));
  });

  it("is recognized by its flag, so a node from another copy of the package compiles the same", () => {
    const node = increment().compute(10, 32);
    const foreign = { ...node } as ComputeNode;
    expect(foreign).not.toBeInstanceOf(ComputeNode);
    expect(compile({ stage: "compute" }, foreign).code).toBe(compile({ stage: "compute" }, node).code);
  });
});
