/**
 * Stage 2 of docs/design-shared-layout-ir.md: does the shared allocator
 * (src/rmsl-layout.ts) actually let a WASM computation place its uniforms
 * at the exact byte offsets a real WGSL uniform buffer would use for the
 * same members, rather than just sharing the placement *algorithm* the way
 * stage 1 already proved?
 *
 * An earlier version of this fix used the caller's raw GPU offset as this
 * backend's *only* address for the uniform, which matched offsets exactly
 * but corrupted adjacent uniforms: this backend's `float` is f64 (8 bytes)
 * while `wgslUniformLayout`'s offsets assume `f32` (4 bytes), so two
 * GPU-adjacent members spaced 4 bytes apart overlapped in this backend's
 * wider writes. The fix (`rmsl-wasm.ts`'s `GpuUniformLayout` doc comment)
 * gives such a uniform *two* addresses: the caller's raw, narrow one
 * (never touched by this backend's arithmetic directly) and an ordinary
 * packed scratch address like any other uniform gets, with a promotion
 * step bridging them. What's left, and is real rather than a bug: reading
 * a GPU-placed uniform is only as precise as `f32` allows, same as any
 * real GPU uniform buffer sharing those bytes would be.
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

    // The core claim: this backend's own uniform addresses are exactly the
    // ones a real WGSL uniform buffer would use for the same members — not
    // independently re-derived, not merely equal by coincidence for this
    // one case (vec3 landing before vec2 despite being declared second is
    // exactly what a naive "don't reorder" implementation would get wrong).
    const { params } = compileWasmFn(build as any, options);
    const dirAddr = (params.find(p => p.kind === "uniformMemory" && p.slot === dir.name) as any).address;
    const scaleAddr = (params.find(p => p.kind === "uniformMemory" && p.slot === scale.name) as any).address;
    expect(dirAddr).toBe(offsetOf(dir.name));
    expect(scaleAddr).toBe(offsetOf(scale.name));
    expect(dirAddr).toBeLessThan(scaleAddr); // vec3 really did get reordered ahead of vec2
  });

  it("no longer corrupts an adjacent uniform at tight GPU spacing", () => {
    const dir = uniform("vec3");
    const scale = uniform("vec2");
    const layout = wgslUniformLayout([
      { slot: dir.name, type: wgslType("vec3") },
      { slot: scale.name, type: wgslType("vec2") },
    ]);
    const dirOffset = layout.members.find(m => m.name === dir.name)!.offset;
    const scaleOffset = layout.members.find(m => m.name === scale.name)!.offset;
    // WGSL's vec3<f32> occupies 12 bytes — tighter than this backend's own
    // vec3 (3 x f64 = 24 bytes) would need if it used this offset as its
    // only address, which is exactly the case the old, broken version got
    // wrong (see this file's own history).
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
    expect(result).toBe((1 + 4 + 9) * 10 + 20); // dot(dir,dir)*scale.x + scale.y = 160
  });

  it("is only as precise as f32 for a GPU-placed uniform — real, not a bug", () => {
    const scale = uniform("vec2");
    const layout = wgslUniformLayout([{ slot: scale.name, type: wgslType("vec2") }]);
    const options: CompileWasmFnOptions = {
      name: "main", params: [],
      gpuUniformLayout: { offsets: { [scale.name]: layout.members[0].offset }, totalSize: layout.size },
    };
    // 0.1 has no exact f32 (or f64) representation; Math.fround is JS's own
    // "round this f64 to the nearest f32" — the same rounding
    // `writeAggregateToMemory`'s `setFloat32` applies when this uniform's
    // value crosses into its narrow, GPU-shaped storage.
    const fn = compileWasm(() => Fn(() => scale.x)() as any, options);
    const result = fn({ uniforms: { [scale.name]: [0.1, 0] } });
    expect(result).toBe(Math.fround(0.1));
    expect(result).not.toBe(0.1); // the real, inherent cost: an ordinary (non-GPU) uniform would keep full f64 precision here
  });
});
