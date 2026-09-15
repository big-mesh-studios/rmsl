/**
 * Minimal coverage for the WGSL `@compute` stage: it only needs to prove the
 * generated source has the shape a compute pipeline expects, not exercise the
 * whole DSL the way the vertex/fragment stages already are elsewhere.
 */
import { describe, it, expect } from "vitest";
import { attribute, Fn, output } from "../rmsl";
import { compileWgsl } from "./wgsl";

describe("compileWgsl.compute", () => {
  it("emits a @compute entry point reading and writing storage buffers", () => {
    const prog = Fn(() => {
      const pos = attribute("float");
      const out = output("float");
      out.assign(pos.add(1));
      return out;
    })();
    const wgsl = compileWgsl.compute(prog);

    expect(wgsl).toContain("@compute @workgroup_size(64)");
    expect(wgsl).toContain("@builtin(global_invocation_id)");
    expect(wgsl).toMatch(/var<storage, read> _rmsl_a\d+: array<f32>;/);
    expect(wgsl).toMatch(/var<storage, read_write> _rmsl_o\d+: array<f32>;/);
    expect(wgsl).toContain("arrayLength(&_rmsl_a");
  });

  it("indexes buffers by the invocation id instead of a vertex/fragment struct", () => {
    const prog = Fn(() => {
      const pos = attribute("vec2");
      const out = output("vec2");
      out.assign(pos);
      return out;
    })();
    const wgsl = compileWgsl.compute(prog);

    expect(wgsl).not.toContain("VertexInput");
    expect(wgsl).not.toContain("FragmentOutput");
    expect(wgsl).toContain("let _rmsl_index = _rmsl_globalId.x;");
    expect(wgsl).toMatch(/_rmsl_o\d+\[_rmsl_index\] = _rmsl_a\d+\[_rmsl_index\];/);
  });
});
