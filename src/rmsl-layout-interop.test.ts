/**
 * Stage 2 of docs/design-shared-layout-ir.md: does the shared allocator
 * (src/rmsl-layout.ts) actually let a WASM computation place its uniforms
 * at the exact byte offsets a real WGSL uniform buffer would use for the
 * same members, rather than just sharing the placement *algorithm* the way
 * stage 1 already proved?
 *
 * The honest result: **offsets can match exactly, but that alone is not
 * safe, full zero-copy interop.** Phase 3's WASM backend stores `float` as
 * f64 (8 bytes) — a deliberate choice (ROADMAP.md, "`float` is f64,
 * `int`/`uint`/`bool` are real `i32`") — while `wgslUniformLayout`'s
 * offsets assume `f32` (4 bytes) per component. Same offsets, different
 * byte *widths* — and the second test below shows this isn't just "reads
 * back wrong," it's "two GPU-adjacent uniforms can end up overlapping in
 * this backend's wider storage and corrupt each other." Closing that gap
 * (storing as f32 for a GPU-bound uniform) is a separate, bigger change
 * this doesn't attempt — see `GpuUniformLayout`'s doc comment in
 * rmsl-wasm.ts.
 */
import { describe, it, expect } from "vitest";
import { Fn, uniform, wgslUniformLayout } from "./rmsl";
import { wgslType } from "./rmsl-wgsl";
import { compileWasm, compileWasmFn, type CompileWasmFnOptions } from "./rmsl-wasm";

describe("stage 2: WASM uniforms placed at WGSL-computed offsets", () => {
  it("places two differently-aligned aggregate uniforms at wgslUniformLayout's exact offsets", () => {
    const dir = uniform("vec3");
    const scale = uniform("vec2");

    // Real production layout computation — exactly what a WGSL uniform
    // struct for these two members would use, reordering vec3 (align 16)
    // ahead of vec2 (align 8) even though vec2 is declared first here.
    const layout = wgslUniformLayout([
      { slot: scale.name, type: wgslType("vec2") },
      { slot: dir.name, type: wgslType("vec3") },
    ]);
    const offsetOf = (slot: string) => layout.members.find(m => m.name === slot)!.offset;

    const options: CompileWasmFnOptions = {
      name: "main",
      params: [],
      gpuUniformLayout: {
        offsets: { [dir.name]: offsetOf(dir.name), [scale.name]: offsetOf(scale.name) },
        totalSize: layout.size,
      },
    };
    const build = () => Fn(() => dir.dot(dir).mul(scale.x).add(scale.y))();

    // The core claim that does hold: this backend's own uniform addresses
    // are exactly the ones a real WGSL uniform buffer would use for the
    // same members — not independently re-derived, not merely equal by
    // coincidence for this one case (vec3 landing before vec2 despite
    // being declared second is exactly what a naive "don't reorder"
    // implementation would get wrong).
    const { params } = compileWasmFn(build as any, options);
    const dirAddr = (params.find(p => p.kind === "uniformMemory" && p.slot === dir.name) as any).address;
    const scaleAddr = (params.find(p => p.kind === "uniformMemory" && p.slot === scale.name) as any).address;
    expect(dirAddr).toBe(offsetOf(dir.name));
    expect(scaleAddr).toBe(offsetOf(scale.name));
    expect(dirAddr).toBeLessThan(scaleAddr); // vec3 really did get reordered ahead of vec2
  });

  it("demonstrates the remaining gap: GPU-spaced offsets let this backend's wider f64 writes corrupt an adjacent uniform", () => {
    const dir = uniform("vec3");
    const scale = uniform("vec2");
    const layout = wgslUniformLayout([
      { slot: dir.name, type: wgslType("vec3") },
      { slot: scale.name, type: wgslType("vec2") },
    ]);
    const dirOffset = layout.members.find(m => m.name === dir.name)!.offset;
    const scaleOffset = layout.members.find(m => m.name === scale.name)!.offset;
    // WGSL's vec3<f32> occupies 12 bytes; this backend's vec3 (3 x f64)
    // occupies 24 — spilling 12 bytes past where the GPU layout expected
    // the *next* member (scale) to start.
    expect(scaleOffset - dirOffset).toBeLessThan(24);

    const options: CompileWasmFnOptions = {
      name: "main", params: [],
      gpuUniformLayout: {
        offsets: { [dir.name]: dirOffset, [scale.name]: scaleOffset },
        totalSize: layout.size,
      },
    };
    const build = () => Fn(() => dir.dot(dir).mul(scale.x).add(scale.y))();
    const fn = compileWasm(build as any, options);
    const result = fn({ uniforms: { [dir.name]: [1, 2, 3], [scale.name]: [10, 20] } });
    // Correct would be dot([1,2,3],[1,2,3])*10 + 20 = 14*10+20 = 160 — but
    // writing dir's third f64 component overlaps scale's first f64
    // component at this spacing, so scale.x is corrupted before it's ever
    // read, and the result is neither 160 nor any other value derived from
    // dir/scale's real inputs.
    expect(result).not.toBe(160);
  });
});
