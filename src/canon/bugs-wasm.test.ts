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
  uniform,
  varying,
  vec4,
} from "../rmsl";
import { compileWasm, compileWasmFn, createWasmCompute } from "../wasm";

const none = { name: "main", params: [] };

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
});
