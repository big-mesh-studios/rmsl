import { describe, expect, it } from "vitest";
import {
  Fn,
  equal,
  float,
  instancedArray,
  int,
  sub,
  Switch,
  uint,
  uniform,
  uniformRaw,
  vec4,
  type Node,
} from "../rmsl";
import { compileGlsl } from "../glsl";
import { compileWgsl } from "../wgsl";
import { compileJSRoutine } from "../js";
import { compileWasmRoutine } from "../wasm";
import { evaluateJS, evaluateWASM } from "../testing/shader-eval";
import { deserialize, serialize, type SerializedGraph } from "../serialize";

const param = { name: "main", params: [{ name: "a", type: "float" as const }] };
const none = { name: "main", params: [] };

/** `graph` after a real JSON round-trip, rebuilt. */
const roundTrip = (graph: SerializedGraph) => deserialize(JSON.parse(JSON.stringify(graph)));

describe("known bugs of the core, each failing until its fix", () => {
  /**
   * Folding a float operation whose result is not finite writes JavaScript's
   * spelling of it as the literal, `Infinity.0` on GLSL and `NaNf` on WGSL,
   * which no driver accepts.
   *
   * @canon bug-folding-a-non-finite-float-writes-infinity-or-nan-as-a-literal
   */
  it.fails("folds a division by zero and the root of a negative number to source a driver accepts", () => {
    expect(compileGlsl.fragment(Fn(() => vec4(float(1).div(0)))())).not.toMatch(/Infinity|NaN/);
    expect(compileWgsl.fragment(Fn(() => vec4(float(-1).sqrt()))())).not.toMatch(/Infinity|NaN/);
  });

  /**
   * `int` and `uint` given a number outside their range wrap it modulo 2^32,
   * so `int(3e9)` is -1294967296 and `uint(5e9)` is 705032704.
   *
   * @canon bug-an-integer-literal-out-of-range-wraps
   */
  it.fails("clamps a number outside the range of int or uint given to its constructor", () => {
    expect(evaluateJS(() => Fn(() => int(3e9).toFloat())())).toBe(2147483520);
    expect(evaluateJS(() => Fn(() => uint(5e9).toFloat())())).toBe(4294967040);
  });

  /**
   * A literal that is NaN or infinite becomes `null` in JSON, and comes back
   * as a float with no value, so a restored `u + Infinity` computes `u`.
   *
   * @canon bug-a-non-finite-literal-does-not-survive-json
   */
  it.fails("restores a literal that is not finite", () => {
    const build = () => Fn(() => uniformRaw("gain", "float").add(float(Infinity)).toVar())();
    const restored = roundTrip(serialize(build()));
    const run = compileJSRoutine(() => restored as any, none);
    expect(run({ uniforms: { gain: 1 } })).toBe(Infinity);
  });

  /**
   * A buffer's contents that are NaN or infinite become `null` in JSON, and
   * the restored buffer holds 0 in their place.
   *
   * @canon bug-a-non-finite-buffer-value-does-not-survive-json
   */
  it.fails("restores the contents of a buffer that are not finite", () => {
    const values = instancedArray(Float32Array.of(NaN, Infinity), "float");
    const restored = roundTrip(serialize(Fn(() => values.element(int(0)).toVar())())) as any;
    const storageNode = (n: any): any => (n.type === "storage" ? n : (n.params ?? []).map(storageNode).find(Boolean));
    expect(Array.from(storageNode(restored).attribute.array)).toEqual([NaN, Infinity]);
  });

  /**
   * `deserialize` accepts a node type no node has, and a uniform with neither
   * a slot nor a local name, and rebuilds a node from each.
   *
   * @canon bug-deserialize-accepts-unknown-and-unnamed-nodes
   */
  it.fails("refuses an unknown node type and a uniform without a name", () => {
    const graph = (node: object) => ({ nodes: [node], buffers: [], roots: 0 }) as unknown as SerializedGraph;
    expect(() => deserialize(graph({ _t: "float", type: "frobnicate" }))).toThrow();
    expect(() => deserialize(graph({ _t: "float", type: "uniform", value: { shaderType: "float" } }))).toThrow();
  });

  /**
   * An operation on an `int` and a `float` operand compiles, converting one
   * of them: WGSL truncates the float to `i32`, GLSL widens the int to
   * `float`, so the targets disagree.
   *
   * @canon bug-an-int-and-a-float-operand-compile-with-a-hidden-conversion
   */
  it.fails("refuses an operation on an int and a float operand on GLSL and WGSL", () => {
    const count = uniform("int");
    const scale = uniform("float");
    const build = () => Fn(() => vec4((count as any).add(scale), 0, 0, 1))();
    expect(() => compileGlsl.fragment(build())).toThrow();
    expect(() => compileWgsl.fragment(build())).toThrow();
  });
});
