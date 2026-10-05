import { describe, it, expect } from "vitest";
import { Fn, instancedArray, int, invocationIndex } from "../../rmsl";
import { compileWgsl } from "./wgsl";

describe("compileWgsl.compute", () => {
  /**
   * @canon spec-wgsl-compiles-a-compute-node-to-a-compute-entry-point
   */
  it("emits a @compute entry point reading and writing storage buffers", () => {
    const src = instancedArray(4, "float");
    const dst = instancedArray(4, "float");
    const prog = Fn(() => {
      const i = invocationIndex();
      dst.element(i).assign(src.element(i).add(1));
    })();
    const wgsl = compileWgsl.compute(prog);

    expect(wgsl).toContain("@compute @workgroup_size(64)");
    expect(wgsl).toContain("@builtin(global_invocation_id)");
    expect(wgsl).toMatch(/@group\(1\) @binding\(0\) var<storage, read_write> _rmsl_s0: array<f32>;/);
    expect(wgsl).toMatch(/@group\(1\) @binding\(1\) var<storage, read_write> _rmsl_s1: array<f32>;/);
    expect(wgsl).toContain("arrayLength(&_rmsl_s0)");
  });

  /**
   * @canon spec-wgsl-compiles-a-compute-node-to-a-compute-entry-point
   */
  it("indexes buffers by the invocation id instead of a vertex/fragment struct", () => {
    const src = instancedArray(4, "vec2");
    const dst = instancedArray(4, "vec2");
    const prog = Fn(() => {
      const i = invocationIndex();
      dst.element(i).assign(src.element(i));
    })();
    const wgsl = compileWgsl.compute(prog);

    expect(wgsl).not.toContain("VertexInput");
    expect(wgsl).not.toContain("FragmentOutput");
    expect(wgsl).toContain("let _rmsl_index = _rmsl_globalId.x;");
    expect(wgsl).toMatch(/_rmsl_s1\[_rmsl_globalId\.x\] = _rmsl_s0\[_rmsl_globalId\.x\];/);
    // A compute dispatch reads one element per invocation, never a vertex.
    expect(wgsl).not.toContain("@location(");
  });
});
