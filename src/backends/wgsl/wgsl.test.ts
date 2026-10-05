import { describe, it, expect } from "vitest";
import { Fn, instancedArray, int, invocationIndex } from "../../rmsl";
import { compileWgsl } from "./wgsl";

describe("compileWgsl.compute", () => {
  /**
   * @canon spec-wgsl-compiles-a-compute-node-to-a-compute-entry-point
   */
  it("emits a @compute entry point reading and writing storage buffers", () => {
    const src = instancedArray(4, "float").toReadOnly();
    const dst = instancedArray(4, "float");
    const prog = Fn(() => {
      const i = invocationIndex();
      dst.element(i).assign(src.element(i).add(1));
    })();
    const wgsl = compileWgsl.compute(prog);

    expect(wgsl).toContain("@compute @workgroup_size(64)");
    expect(wgsl).toContain("@builtin(global_invocation_id)");
    const [, srcName] = wgsl.match(/var<storage, read> (\w+): array<f32>;/) ?? [];
    const [, dstName] = wgsl.match(/var<storage, read_write> (\w+): array<f32>;/) ?? [];
    expect(srcName).toBeDefined();
    expect(dstName).toBeDefined();
    expect(wgsl).toContain(`${dstName}[_rmsl_globalId.x] = ${srcName}[_rmsl_globalId.x] + 1f;`);
    expect(wgsl).toContain(`arrayLength(&${srcName})`);
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
    expect(wgsl).toMatch(/(_rmsl_s\d)\[_rmsl_globalId\.x\] = (?!\1)_rmsl_s\d\[_rmsl_globalId\.x\];/);
    // A compute dispatch reads one element per invocation, never a vertex.
    expect(wgsl).not.toContain("@location(");
  });
});
