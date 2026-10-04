import { describe, expect, it } from "vitest";
import {
  attribute,
  builtinFragDepth,
  builtinPosition,
  Discard,
  float,
  Fn,
  fragCoord,
  If,
  instancedArray,
  int,
  ivec2,
  textureLoad,
  uniform,
  uniformArray,
  varying,
  vec2,
  vec3,
  vec4,
} from "../rmsl";
import { compileWasm, compileWasmFn, compileWasmRoutine, createWasmCompute } from "../wasm";

const none = { name: "main", params: [] };
const param = { name: "main", params: [{ name: "a", type: "float" as const }] };

/** A triangle at depth `z` that covers the whole viewport. */
const screen = (z = 0) => new Float64Array([-1, -1, z, 3, -1, z, -1, 3, z]);

/** A vertex stage that places `pos` as given, and a fragment stage that draws `color`. */
function flat() {
  const pos = attribute("vec3");
  const color = uniform("vec4");
  const vertex = () => Fn(() => builtinPosition().assign(vec4(pos, 1)))();
  const fragment = () => Fn(() => color)();
  return { pos, color, routine: compileWasm(vertex as any, fragment as any) };
}

describe("known WASM bugs, each failing until its fix", () => {
  /**
   * `compileWasm` starts the vertex stage's texture heap where the fragment
   * stage's layout starts, so the fragment inputs written on the next draw
   * overwrite the vertex stage's texture.
   *
   * @canon bug-wasm-places-a-vertex-texture-over-the-fragment-layout
   */
  it.fails("keeps a vertex-stage texture intact across draws on WASM", () => {
    const pos = attribute("vec3");
    const tex = uniform("sampler2D");
    const shade = varying("float");
    const tint = uniform("vec4");
    const vertex = () =>
      Fn(() => {
        shade.assign(textureLoad(tex, ivec2(0, 0)).x);
        builtinPosition().assign(vec4(pos, 1));
      })();
    const fragment = () => Fn(() => vec4(shade, tint.x, tint.y, 1))();
    const routine = compileWasm(vertex as any, fragment as any);
    const ctx = {
      attributes: { [pos.name]: screen() },
      uniforms: { [tint.name]: [0.25, 0.5, 0, 0] },
      textures: { [tex.name]: { data: new Float32Array(64).fill(9), width: 4, height: 4 } },
    };
    expect(Array.from(routine.draw(ctx, { width: 1, height: 1 }))).toEqual([9, 0.25, 0.5, 1]);
    expect(Array.from(routine.draw(ctx, { width: 1, height: 1 }))).toEqual([9, 0.25, 0.5, 1]);
  });

  /**
   * `compileWasm` fixes the depth buffer's address at the first draw while
   * every other region moves with the draw's size, so a later, larger draw
   * lays its vertices and colours over the depth buffer.
   *
   * @canon bug-wasm-rasterizer-keeps-its-depth-buffer-where-a-larger-draw-writes
   */
  it.fails("draws the same after a smaller draw as on a fresh routine on WASM", () => {
    const fresh = flat();
    const used = flat();
    const four = new Float64Array(Array.from({ length: 4 }, () => Array.from(screen())).flat());
    const options = { width: 2, height: 2, clear: true, clearDepth: true };
    const expected = fresh.routine.draw(
      { attributes: { [fresh.pos.name]: four }, uniforms: { [fresh.color.name]: [0, 0, 1, 1] } },
      options,
    );
    used.routine.draw(
      { attributes: { [used.pos.name]: screen() }, uniforms: { [used.color.name]: [1, 0, 0, 1] } },
      { width: 1, height: 1 },
    );
    const got = used.routine.draw(
      { attributes: { [used.pos.name]: four }, uniforms: { [used.color.name]: [0, 0, 1, 1] } },
      options,
    );
    expect(Array.from(got)).toEqual(Array.from(expected));
  });

  /**
   * `compileWasm` copies an integer attribute in as an f64, where the vertex
   * stage reads an i32.
   *
   * @canon bug-wasm-rasterizer-writes-an-integer-attribute-as-a-float
   */
  it.fails("passes an int attribute to the vertex stage on WASM", () => {
    const pos = attribute("vec3");
    const k = attribute("int");
    const shade = varying("float");
    const vertex = () =>
      Fn(() => {
        shade.assign(k.toFloat());
        builtinPosition().assign(vec4(pos, 1));
      })();
    const fragment = () => Fn(() => vec4(shade, 0, 0, 1))();
    const routine = compileWasm(vertex as any, fragment as any);
    const got = routine.draw(
      { attributes: { [pos.name]: screen(), [k.name]: Int32Array.of(5, 5, 5) } },
      { width: 1, height: 1 },
    );
    expect(Array.from(got)).toEqual([5, 0, 0, 1]);
  });

  /**
   * The WASM rasterizer interpolates an integer varying as an f64, though the
   * stages write and read it as an i32.
   *
   * @canon bug-wasm-rasterizer-interpolates-an-integer-varying-as-a-float
   */
  it.fails("passes an int varying from the vertex to the fragment stage on WASM", () => {
    const pos = attribute("vec3");
    const k = varying("int");
    const vertex = () =>
      Fn(() => {
        k.assign(int(5));
        builtinPosition().assign(vec4(pos, 1));
      })();
    const fragment = () => Fn(() => vec4(k.toFloat(), 0, 0, 1))();
    const routine = compileWasm(vertex as any, fragment as any);
    expect(Array.from(routine.draw({ attributes: { [pos.name]: screen() } }, { width: 1, height: 1 }))).toEqual([
      5, 0, 0, 1,
    ]);
  });

  /**
   * Under `scalarsInMemory`, the host writes a scalar uniform at its offset in
   * `gpuUniformLayout`, while the program reads it from an address of its own.
   *
   * @canon bug-wasm-reads-a-gpu-placed-scalar-uniform-from-the-wrong-address
   */
  it.fails("reads a scalar uniform placed by gpuUniformLayout with scalarsInMemory on WASM", () => {
    const s = uniform("float");
    const routine = compileWasmRoutine(() => Fn(() => s.add(0).toVar())(), {
      ...none,
      scalarsInMemory: true,
      gpuUniformLayout: { offsets: { [s.name]: 0 }, totalSize: 16 },
    });
    expect(routine.run({ uniforms: { [s.name]: 0.5 } })).toBe(0.5);
  });

  /**
   * Without `scalarsInMemory`, a scalar uniform placed by `gpuUniformLayout`
   * arrives as a 64-bit argument and never reads from the layout.
   *
   * @canon bug-wasm-passes-a-gpu-placed-scalar-uniform-around-the-layout
   */
  it.fails("holds a scalar uniform placed by gpuUniformLayout as an f32 on WASM", () => {
    const s = uniform("float");
    const routine = compileWasmRoutine(() => Fn(() => s.add(0).toVar())(), {
      ...none,
      gpuUniformLayout: { offsets: { [s.name]: 0 }, totalSize: 16 },
    });
    expect(routine.run({ uniforms: { [s.name]: 0.1 } })).toBe(Math.fround(0.1));
  });

  /**
   * A uniform array element at a run-time index out of range reads whatever
   * memory lies there, and traps below zero.
   *
   * @canon bug-wasm-reads-a-uniform-array-element-out-of-range-from-foreign-memory
   */
  it.fails("reads the last element of a uniform array for an index out of range on WASM", () => {
    const arr = uniformArray("float", 2);
    const routine = compileWasmRoutine((i: any) => Fn(() => arr.element(i).add(0).toVar())(), {
      name: "main",
      params: [{ name: "i", type: "int" }],
    });
    const uniforms = { [arr.name]: [3, 4] };
    expect(routine.run({ params: { i: 2 }, uniforms })).toBe(4);
    expect(routine.run({ params: { i: -1 }, uniforms })).toBe(4);
  });

  /**
   * Sampling truncates a coordinate far past the edge to an i32 before it
   * wraps, which traps.
   *
   * @canon bug-wasm-traps-on-a-sampling-coordinate-far-past-the-edge
   */
  it.fails("clamps a sampling coordinate far past the edge on WASM", () => {
    const tex = uniform("sampler2D");
    const routine = compileWasmRoutine((a: any) => Fn(() => tex.texture(vec2(a, 0.5)).x.toVar())(), param);
    const texture = { data: [1, 2], width: 2, height: 1, channels: 1 as const };
    expect(routine.run({ params: { a: 1e12 }, textures: { [tex.name]: texture } })).toBe(2);
  });

  /**
   * The rasterizer truncates a triangle's bounding box to i32 before it
   * clamps it to the viewport, which traps for a triangle far off screen.
   *
   * @canon bug-wasm-rasterizer-traps-on-a-triangle-far-off-screen
   */
  it.fails("draws nothing for a triangle far off screen on WASM", () => {
    const { pos, routine } = flat();
    const far = new Float64Array([1e12, 0, 0, 2e12, 0, 0, 1e12, 1, 0]);
    expect(Array.from(routine.draw({ attributes: { [pos.name]: far } }, { width: 1, height: 1 }))).toEqual([
      0, 0, 0, 0,
    ]);
  });

  /**
   * A WASM routine returns `0`, or a result holding a zero value, for a
   * discarded fragment.
   *
   * @canon bug-wasm-returns-a-value-for-a-discarded-fragment
   */
  it.fails("returns null for a discarded fragment on WASM", () => {
    const build = () =>
      Fn(() => {
        Discard();
        return float(1);
      })();
    expect(compileWasmRoutine(build, { ...none, stage: "fragment" }).run({})).toBeNull();
  });

  /**
   * The WASM rasterizer writes a colour for a discarded fragment: the colour
   * the fragment stage last left in its memory.
   *
   * @canon bug-the-cpu-rasterizers-paint-a-discarded-fragment
   */
  it.fails("leaves the pixel of a discarded fragment cleared on WASM", () => {
    const pos = attribute("vec3");
    const side = varying("float");
    const vertex = () =>
      Fn(() => {
        side.assign(pos.x);
        builtinPosition().assign(vec4(pos, 1));
      })();
    const fragment = () =>
      Fn(() => {
        If(side.greaterThan(0), () => Discard());
        return vec4(1, 0, 0, 1);
      })();
    const routine = compileWasm(vertex as any, fragment as any);
    const got = routine.draw({ attributes: { [pos.name]: screen() } }, { width: 2, height: 1, clear: true });
    expect(Array.from(got)).toEqual([1, 0, 0, 1, 0, 0, 0, 0]);
  });

  /**
   * The WASM rasterizer writes the depth of a fragment before it runs the
   * fragment stage, so a discarded fragment still occludes later ones.
   *
   * @canon bug-the-cpu-rasterizers-write-the-depth-of-a-discarded-fragment
   */
  it.fails("lets a discarded fragment leave the depth buffer as it was on WASM", () => {
    const pos = attribute("vec3");
    const drop = uniform("float");
    const color = uniform("vec4");
    const vertex = () => Fn(() => builtinPosition().assign(vec4(pos, 1)))();
    const fragment = () =>
      Fn(() => {
        If(drop.greaterThan(0.5), () => Discard());
        return color;
      })();
    const routine = compileWasm(vertex as any, fragment as any);
    const draw = (z: number, dropped: number, rgba: number[], clearDepth = false) =>
      routine.draw(
        { attributes: { [pos.name]: screen(z) }, uniforms: { [drop.name]: dropped, [color.name]: rgba } },
        { width: 1, height: 1, clear: true, clearDepth },
      );
    draw(-0.5, 1, [1, 0, 0, 1], true);
    expect(Array.from(draw(0.5, 0, [0, 0, 1, 1]))).toEqual([0, 0, 1, 1]);
  });

  /**
   * The WASM rasterizer tests and stores the interpolated depth, ignoring the
   * depth the fragment stage writes.
   *
   * @canon bug-wasm-rasterizer-ignores-the-fragment-depth
   */
  it.fails("tests the depth the fragment stage writes on WASM", () => {
    const pos = attribute("vec3");
    const depth = uniform("float");
    const color = uniform("vec4");
    const vertex = () => Fn(() => builtinPosition().assign(vec4(pos, 1)))();
    const fragment = () =>
      Fn(() => {
        builtinFragDepth().assign(depth);
        return color;
      })();
    const routine = compileWasm(vertex as any, fragment as any);
    const draw = (z: number, d: number, rgba: number[], clearDepth = false) =>
      routine.draw(
        { attributes: { [pos.name]: screen(z) }, uniforms: { [depth.name]: d, [color.name]: rgba } },
        { width: 1, height: 1, clear: true, clearDepth },
      );
    draw(-0.5, 0.9, [1, 0, 0, 1], true);
    expect(Array.from(draw(0.5, 0, [0, 0, 1, 1]))).toEqual([0, 0, 1, 1]);
  });

  /**
   * The WASM rasterizer never writes `fragCoord()`, so every fragment reads
   * it as `[0, 0]`, where the JS rasterizer passes the pixel's centre.
   *
   * @canon bug-wasm-rasterizer-gives-every-fragment-coordinate-zero
   */
  it.fails("gives each fragment the centre of its pixel as fragCoord on WASM", () => {
    const pos = attribute("vec3");
    const vertex = () => Fn(() => builtinPosition().assign(vec4(pos, 1)))();
    const fragment = () => Fn(() => vec4(fragCoord(), 0, 1))();
    const routine = compileWasm(vertex as any, fragment as any);
    const got = routine.draw({ attributes: { [pos.name]: screen() } }, { width: 2, height: 1 });
    expect(Array.from(got)).toEqual([0.5, 0.5, 0, 1, 1.5, 0.5, 0, 1]);
  });

  /**
   * `compileWasm` clears its colour buffer only when a draw passes `clear`,
   * so a pixel the next draw leaves uncovered keeps the earlier colour.
   *
   * @canon bug-the-cpu-rasterizers-keep-the-colour-of-an-earlier-draw
   */
  it.fails("starts each draw from a cleared colour buffer on WASM", () => {
    const { pos, color, routine } = flat();
    routine.draw(
      { attributes: { [pos.name]: screen() }, uniforms: { [color.name]: [1, 0, 0, 1] } },
      { width: 1, height: 1 },
    );
    const offscreen = new Float64Array([5, 5, 0, 6, 5, 0, 5, 6, 0]);
    const got = routine.draw(
      { attributes: { [pos.name]: offscreen }, uniforms: { [color.name]: [0, 0, 1, 1] } },
      { width: 1, height: 1 },
    );
    expect(Array.from(got)).toEqual([0, 0, 0, 0]);
  });

  /**
   * `compileWasm` makes its own memory without `sharedMemory` or
   * `maxMemoryPages`, which the modules it compiled declared, so linking fails.
   *
   * @canon bug-wasm-rasterizer-makes-its-memory-without-the-shared-flag
   */
  it.fails("compiles a vertex and fragment pair with sharedMemory and no memory on WASM", () => {
    const pos = attribute("vec3");
    const vertex = () => Fn(() => builtinPosition().assign(vec4(pos, 1)))();
    const fragment = () => Fn(() => vec4(1, 0, 0, 1))();
    expect(() => compileWasm(vertex as any, fragment as any, { sharedMemory: true })).not.toThrow();
  });

  /**
   * `createWasmCompute` names its routine `compute` by default, the name of
   * the module's own dispatch export, so a program with storage fails to compile.
   *
   * @canon bug-wasm-compute-names-its-routine-as-its-dispatch-loop
   */
  it.fails("creates a compute adapter under its default name on WASM", () => {
    const buf = instancedArray(2, "float");
    expect(() => createWasmCompute(Fn(() => buf.element(int(0)).assign(float(3)))())).not.toThrow();
  });

  /**
   * A program with neither `storage()` nor `invocationIndex()` gets no
   * dispatch export, and `compute` calls its `main` from the host once per
   * invocation.
   *
   * @canon bug-wasm-loops-a-program-without-storage-from-the-host
   */
  it.fails("compiles a dispatch loop for a program that reads no storage on WASM", () => {
    const compiled = compileWasmFn(() => Fn(() => float(1).toVar())(), none);
    expect(compiled.compute).toBe(true);
  });

  /**
   * An operation that reads a scalar operand once per component, or twice
   * as `fract` and `sign` do, emits it each time, so the statements of an
   * inline `Fn` it returns run more than once.
   *
   * @canon bug-wasm-runs-an-inline-fn-once-per-read-of-its-value
   */
  it.fails("runs the statements of an inline Fn once when an operation reads its value twice on WASM", () => {
    const counter = instancedArray(1, "float");
    const counted = () =>
      Fn(() => {
        counter.element(int(0)).addAssign(1);
        return float(0.25);
      })() as any;
    const runs = (build: () => any) => {
      const data = new Float64Array(1);
      compileWasmRoutine(build, none).run({ storages: { [counter.name]: data } });
      return data[0];
    };
    expect(runs(() => counted().fract())).toBe(1);
    expect(runs(() => vec3(1, 2, 3).mul(counted()))).toBe(1);
  });

  /**
   * A routine leaves a vector or matrix uniform the call does not set as the
   * last call wrote it, where an unset scalar uniform reads zero.
   *
   * @canon bug-wasm-keeps-an-aggregate-uniform-the-call-leaves-out
   */
  it.fails("reads a vector uniform the call leaves out as zero on WASM", () => {
    const u = uniform("vec2");
    const routine = compileWasmRoutine(() => Fn(() => u.x.add(0).toVar())(), none);
    routine.run({ uniforms: { [u.name]: [3, 0] } });
    expect(routine.run({ uniforms: {} })).toBe(0);
  });

  /**
   * An integer texture sampled at a float coordinate reads zero, where the
   * GPU targets and JS truncate the coordinate to a texel.
   *
   * @canon bug-wasm-reads-an-integer-texture-at-a-float-coordinate-as-zero
   */
  it.fails("fetches the texel a float coordinate truncates to from an integer texture on WASM", () => {
    const tex = uniform("isampler2D") as any;
    const routine = compileWasmRoutine(() => Fn(() => tex.texture(vec2(0.75, 0.25)).x.toVar())(), none);
    const texture = { data: [10, 20, 30, 40], width: 2, height: 2, channels: 1 as const };
    expect(routine.run({ textures: { [tex.name]: texture } })).toBe(10);
  });

  /**
   * A scalar varying the host leaves out reads as `NaN`.
   *
   * @canon bug-wasm-reads-an-unset-scalar-input-as-nan
   */
  it.fails("reads a scalar varying the host leaves out as zero on WASM", () => {
    const v = varying("float");
    const routine = compileWasmRoutine(() => Fn(() => v.add(1).toVar())(), { ...none, stage: "fragment" });
    expect(routine.run({})).toBe(1);
  });

  /**
   * A vector varying the host leaves out throws a `TypeError` while the
   * routine writes it into memory.
   *
   * @canon bug-wasm-throws-on-an-unset-aggregate-input
   */
  it.fails("reads a vector varying the host leaves out as zero on WASM", () => {
    const v = varying("vec2");
    const routine = compileWasmRoutine(() => Fn(() => v.x.add(1).toVar())(), { ...none, stage: "fragment" });
    expect(routine.run({})).toBe(1);
  });
});
