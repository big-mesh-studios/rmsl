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
  varying,
  vec2,
  vec3,
  vec4,
} from "../rmsl";
import { compileWasm, compileWasmFn, compileWasmRoutine, createWasmCompute, createWasmGrid } from "../wasm";

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
   * The WASM rasterizer writes a colour for a discarded fragment: the colour
   * the fragment stage last left in its memory.
   *
   * @canon bug-the-wasm-rasterizer-paints-a-discarded-fragment
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
   * The WASM target compiles no component-wise math function of a vector,
   * such as `pow`, `sin` or `floor`, and throws that the node is unsupported.
   *
   * @canon bug-wasm-compiles-no-component-wise-math-on-a-vector
   */
  it.fails("computes pow, sin and floor of a vector on WASM", () => {
    const first = (build: (a: any) => any) => compileWasmRoutine((a: any) => Fn(() => build(a).x.toVar())(), param);
    expect(first((a) => pow(vec3(a, 2, 3), vec3(2, 2, 2)))({ params: { a: 4 } })).toBe(16);
    expect(first((a) => sin(vec3(a, 2, 3)))({ params: { a: 0 } })).toBe(0);
    expect(first((a) => floor(vec3(a, 2, 3)))({ params: { a: 1.5 } })).toBe(1);
  });
});
