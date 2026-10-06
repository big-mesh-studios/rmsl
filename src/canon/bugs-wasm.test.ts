import { describe, expect, it } from "vitest";
import {
  attribute,
  builtinFragDepth,
  builtinPosition,
  Discard,
  float,
  floor,
  Fn,
  For,
  fragCoord,
  If,
  instancedArray,
  int,
  invocationIndex,
  ivec2,
  pow,
  sin,
  textureLoad,
  uniform,
  uniformArray,
  varying,
  vec2,
  vec3,
  vec4,
} from "../rmsl";
import { compileWasm, compileWasmFn, compileWasmRoutine, createWasmCompute, createWasmRoutine } from "../wasm";

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

  /**
   * `draw` of a WASM routine returns a view of its own memory, so the next
   * `draw` overwrites the pixels an earlier one returned.
   *
   * @canon bug-wasm-routine-draws-into-a-buffer-the-next-draw-overwrites
   */
  it.fails("keeps the pixels a WASM routine drew when it draws again", () => {
    const routine = compileWasmRoutine((a: any) => Fn(() => vec2(a, a.add(1)).toVar())(), param);
    const first = routine.draw({ params: { a: 0.5 } }, 1, 1);
    routine.draw({ params: { a: 100.5 } }, 1, 1);
    expect(Array.from(first)).toEqual([0.5, 1.5]);
  });

  /**
   * An element past the end of a shorter array the call passes keeps the
   * value an earlier call wrote there.
   *
   * @canon bug-wasm-keeps-a-uniform-array-element-the-call-leaves-out
   */
  it.fails("reads a uniform array element the call leaves out as zero on WASM", () => {
    const arr = uniformArray("float", 3);
    const routine = compileWasmRoutine(() => Fn(() => arr.element(int(2)).add(0).toVar())(), none);
    routine.run({ uniforms: { [arr.name]: [1, 2, 3] } });
    expect(routine.run({ uniforms: { [arr.name]: [1, 2] } })).toBe(0);
  });

  /**
   * The WASM rasterizer shades a pixel centre on an edge two triangles share
   * with both, so the triangle drawn last wins it.
   *
   * @canon bug-wasm-rasterizer-shades-a-shared-edge-twice
   */
  it.fails("gives a pixel on a shared edge to one triangle whatever their order on WASM", () => {
    const { pos, color, routine } = flat();
    const upper = new Float64Array([-1, 1, 0, 1, -1, 0, 1, 1, 0]);
    const lower = new Float64Array([-1, 1, 0, -1, -1, 0, 1, -1, 0]);
    const draw = (triangle: Float64Array, rgba: number[], clear: boolean) =>
      Array.from(
        routine.draw(
          { attributes: { [pos.name]: triangle }, uniforms: { [color.name]: rgba } },
          { width: 3, height: 3, clear, clearDepth: clear },
        ),
      );
    draw(upper, [1, 0, 0, 1], true);
    const upperFirst = draw(lower, [0, 0, 1, 1], false);
    draw(lower, [0, 0, 1, 1], true);
    const lowerFirst = draw(upper, [1, 0, 0, 1], false);
    expect(upperFirst).toEqual(lowerFirst);
  });

  /**
   * `createWasmCompute` reads a vector storage buffer as one array per
   * element, so the flat typed array `setAttribute` takes ends up as `NaN`.
   *
   * @canon bug-the-cpu-compute-adapters-take-a-vector-storage-element-as-an-array
   */
  it.fails("writes a vector storage buffer given as a flat typed array on WASM", () => {
    const buf = instancedArray(2, "vec2");
    const adapter = createWasmCompute(Fn(() => buf.element(invocationIndex()).assign(vec2(3, 4)))(), { name: "step" });
    const data = new Float32Array(4);
    adapter.setAttribute(buf.name, data);
    adapter.compute();
    expect(Array.from(data)).toEqual([3, 4, 3, 4]);
  });

  /**
   * `assign` passes a bare number on as it is, and the WASM target throws on
   * it as a node of no type.
   *
   * @canon bug-assign-leaves-a-bare-number-untyped
   */
  it.fails("assigns a bare number to a float component on WASM", () => {
    const routine = compileWasmRoutine(
      () =>
        Fn(() => {
          const v = vec2(1, 2).toVar();
          (v.x as any).assign(7);
          return v.x;
        })(),
      none,
    );
    expect(routine.run({})).toBe(7);
  });

  /**
   * The WASM target compiles no component-wise math function of a vector,
   * such as `pow`, `sin` or `floor`, and throws that the node is unsupported.
   *
   * @canon bug-wasm-compiles-no-component-wise-math-on-a-vector
   */
  it.fails("computes pow, sin and floor of a vector on WASM", () => {
    const first = (build: (a: any) => any) => compileWasmRoutine((a: any) => Fn(() => build(a).x.toVar())(), param);
    expect(first((a) => pow(vec3(a, 2, 3), vec3(2, 2, 2))).run({ params: { a: 4 } })).toBe(16);
    expect(first((a) => sin(vec3(a, 2, 3))).run({ params: { a: 0 } })).toBe(0);
    expect(first((a) => floor(vec3(a, 2, 3))).run({ params: { a: 1.5 } })).toBe(1);
  });
});
