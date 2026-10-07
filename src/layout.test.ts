import { describe, it, expect } from "vitest";
import { Fn, uniform, uniformArray, int } from "./rmsl";
import { wgslUniformLayout } from "./wgsl";
import { wgslType } from "./backends/wgsl/wgsl";
import { compileWasmRoutine, compileWasmFn, type CompileWasmFnOptions } from "./backends/wasm/wasm";
import { compileJSRoutine } from "./js";

describe("stage 2: WASM uniforms placed at WGSL-computed offsets", () => {
  /**
   * @canon spec-a-wasm-routine-reads-uniforms-from-the-wgsl-layout
   */
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
    const offsetOf = (slot: string) => layout.members.find((m) => m.name === slot)!.offset;

    const options: CompileWasmFnOptions = {
      name: "main",
      params: [],
      float: "f32",
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
    const dirAddr = (params.find((p) => p.kind === "uniformMemory" && p.slot === dir.name) as any).address;
    const scaleAddr = (params.find((p) => p.kind === "uniformMemory" && p.slot === scale.name) as any).address;
    expect(dirAddr).toBe(offsetOf(dir.name));
    expect(scaleAddr).toBe(offsetOf(scale.name));
    expect(dirAddr).toBeLessThan(scaleAddr); // vec3 really did get reordered ahead of vec2
  });

  /**
   * @canon spec-a-wasm-routine-reads-uniforms-from-the-wgsl-layout
   */
  it("reads a scalar uniform at its offset in the layout under scalarsInMemory", () => {
    const s = uniform("float");
    const routine = compileWasmRoutine(() => Fn(() => s.add(0).toVar())(), {
      name: "main",
      params: [],
      scalarsInMemory: true,
      float: "f32",
      gpuUniformLayout: { offsets: { [s.name]: 0 }, totalSize: 16 },
    });
    expect(routine({ uniforms: { [s.name]: 0.5 } })).toBe(0.5);
  });

  /**
   * @canon spec-a-wasm-routine-reads-uniforms-from-the-wgsl-layout
   */
  it("no longer corrupts an adjacent uniform at tight GPU spacing", () => {
    const dir = uniform("vec3");
    const scale = uniform("vec2");
    const layout = wgslUniformLayout([
      { slot: dir.name, type: wgslType("vec3") },
      { slot: scale.name, type: wgslType("vec2") },
    ]);
    const dirOffset = layout.members.find((m) => m.name === dir.name)!.offset;
    const scaleOffset = layout.members.find((m) => m.name === scale.name)!.offset;
    // WGSL's vec3<f32> occupies 12 bytes — tighter than this backend's own
    // vec3 (3 x f64 = 24 bytes) would need if it used this offset as its
    // only address, which is exactly the case the old, broken version got
    // wrong (see this file's own history).
    expect(scaleOffset - dirOffset).toBeLessThan(24);

    const options: CompileWasmFnOptions = {
      name: "main",
      params: [],
      float: "f32",
      gpuUniformLayout: {
        offsets: { [dir.name]: dirOffset, [scale.name]: scaleOffset },
        totalSize: layout.size,
      },
    };
    const build = () => Fn(() => dir.dot(dir).mul(scale.x).add(scale.y))();
    const fn = compileWasmRoutine(build as any, options);
    const result = fn({ uniforms: { [dir.name]: [1, 2, 3], [scale.name]: [10, 20] } });
    expect(result).toBe((1 + 4 + 9) * 10 + 20); // dot(dir,dir)*scale.x + scale.y = 160
  });

  /**
   * @canon spec-a-cpu-target-at-f32-rounds-every-float-value-it-computes
   * @canon spec-wasm-and-js-give-the-same-float-bits
   */
  it("reads a GPU-placed uniform as JS reads it at f32", () => {
    const scale = uniform("vec2");
    const layout = wgslUniformLayout([{ slot: scale.name, type: wgslType("vec2") }]);
    const options: CompileWasmFnOptions = {
      name: "main",
      params: [],
      float: "f32",
      gpuUniformLayout: { offsets: { [scale.name]: layout.members[0].offset }, totalSize: layout.size },
    };
    const build = () => Fn(() => scale.x)() as any;
    const ctx = { uniforms: { [scale.name]: [0.1, 0] } };
    const wasm = compileWasmRoutine(build, options)(ctx);
    expect(wasm).toBe(Math.fround(0.1));
    expect(wasm).toBe(compileJSRoutine(build, { name: "main", params: [], float: "f32" })(ctx));
  });

  /**
   * @canon spec-a-wasm-routine-reads-uniforms-from-the-wgsl-layout
   */
  it("reads a scalar uniform placed by gpuUniformLayout from the layout without scalarsInMemory", () => {
    const s = uniform("float");
    const routine = compileWasmRoutine(() => Fn(() => s.add(0).toVar())(), {
      name: "main",
      params: [],
      float: "f32",
      gpuUniformLayout: { offsets: { [s.name]: 0 }, totalSize: 16 },
    });
    expect(routine({ uniforms: { [s.name]: 0.1 } })).toBe(Math.fround(0.1));
  });

  /**
   * The options are cast, as a caller that bypasses the types does, since the
   * types refuse a layout without `float: "f32"` before the compile can.
   *
   * @canon spec-a-gpu-uniform-layout-needs-32-bit-floats
   */
  it("refuses gpuUniformLayout on a compile at 64 bits", () => {
    const s = uniform("float");
    const options = { name: "main", params: [], gpuUniformLayout: { offsets: { [s.name]: 0 }, totalSize: 16 } } as any;
    expect(() => compileWasmRoutine(() => Fn(() => s.add(0).toVar())(), options)).toThrow(/float: "f32"/);
    expect(() => compileWasmRoutine(() => Fn(() => s.add(0).toVar())(), { ...options, float: "f64" })).toThrow(
      /drop gpuUniformLayout/,
    );
    const layout = { offsets: { [s.name]: 0 }, totalSize: 16 };
    const build = () => Fn(() => s.add(0).toVar())();
    // @ts-expect-error a layout needs float: "f32"
    expect(() => compileWasmRoutine(build, { name: "main", params: [], gpuUniformLayout: layout })).toThrow();
    const at64 = { name: "main", params: [], float: "f64" as const, gpuUniformLayout: layout };
    expect(() =>
      // @ts-expect-error a layout needs float: "f32", not "f64"
      compileWasmRoutine(build, at64),
    ).toThrow();
  });

  /**
   * @canon spec-a-wasm-routine-reads-uniforms-from-the-wgsl-layout
   */
  it("places a uniform array at wgslUniformLayout's offset and stride, f32-accurate", () => {
    const arr = uniformArray("vec4", 2);
    const layout = wgslUniformLayout([{ slot: arr.name, type: wgslType("vec4"), length: 2 }]);
    const member = layout.members.find((m) => m.name === arr.name)!;
    expect(member.stride).toBe(16);

    const options: CompileWasmFnOptions = {
      name: "main",
      params: [],
      float: "f32",
      gpuUniformLayout: {
        offsets: { [arr.name]: member.offset },
        strides: { [arr.name]: member.stride! },
        totalSize: layout.size,
      },
    };
    const fn = compileWasmRoutine(() => Fn(() => arr.element(int(1)).x)() as any, options);
    // 0.1 is not exact in f32; the GPU path stores f32, so it reads back fround(0.1).
    expect(
      fn({
        uniforms: {
          [arr.name]: [
            [0, 0, 0, 0],
            [0.1, 0, 0, 0],
          ],
        },
      }),
    ).toBe(Math.fround(0.1));
    expect(Math.fround(0.1)).not.toBe(0.1);
  });
});
