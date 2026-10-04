import { describe, expect, it } from "vitest";
import {
  Fn,
  equal,
  float,
  instancedArray,
  int,
  output,
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
   * `round` of a half rounds up on JS and WASM and in folding,
   * where WGSL rounds it to the even neighbour, so `round(2.5)` gives 3, not 2.
   *
   * @canon bug-the-cpu-targets-round-a-half-up
   */
  it.fails("rounds a half to the even neighbour on the CPU targets and in folding, as WGSL does", () => {
    const build = (a: Node<"float">) => Fn(() => a.round())();
    expect(evaluateJS(build, [2.5])).toBe(2);
    expect(evaluateWASM(build, [2.5])).toBe(2);
    expect(evaluateJS(() => Fn(() => float(2.5).round())())).toBe(2);
  });

  /**
   * A bare number given as the first operand of a free function beside an
   * integer stays a float, so `sub(7, i).div(2)` divides as floats on JS and
   * fails to validate on WASM.
   *
   * @canon bug-a-bare-number-before-an-integer-stays-a-float
   */
  it.fails("makes a bare number given before an integer an integer", () => {
    const build = (a: any) => Fn(() => sub(7, a.toInt()).div(2).toVar())();
    expect(compileJSRoutine(build, param).run({ params: { a: 0 } })).toBe(3);
    expect(compileWasmRoutine(build, param).run({ params: { a: 0 } })).toBe(3);
  });

  /**
   * `equal(1, i)` compares a float literal with an integer, which the WASM
   * target refuses to validate.
   *
   * @canon bug-a-bare-number-before-an-integer-stays-a-float
   */
  it.fails("compares a bare number with an integer on WASM", () => {
    const build = (a: any) => Fn(() => equal(1, a.toInt()).select(float(1), float(0)).toVar())();
    expect(compileWasmRoutine(build, param).run({ params: { a: 1 } })).toBe(1);
  });

  /**
   * `select` types a bare-number branch as a float beside an integer branch,
   * so the node takes the type of whichever branch comes first: JS divides it
   * as a float, and WASM fails to validate it.
   *
   * @canon bug-select-types-a-bare-number-branch-as-a-float
   */
  it.fails("gives select with an integer branch and a bare-number branch the integer type", () => {
    const first = (a: any) => Fn(() => a.greaterThan(0).select(a.toInt(), 0).toVar())();
    const second = (a: any) => Fn(() => a.greaterThan(0).select(0, a.toInt()).div(2).toVar())();
    expect(compileJSRoutine(second, param).run({ params: { a: -3 } })).toBe(-1);
    expect(compileWasmRoutine(first, param).run({ params: { a: 1.5 } })).toBe(1);
  });

  /**
   * A `Switch` with no `Case` and no `Default` leaves no statement in its
   * block, and the WASM target crashes on the missing node.
   *
   * @canon bug-wasm-crashes-on-an-empty-switch
   */
  it.fails("compiles a Switch with no Case and no Default on WASM", () => {
    const build = () =>
      Fn(() => {
        const v = float(0).toVar();
        Switch(int(uniform("float")), () => {});
        return vec4(v);
      })();
    expect(() => compileWasmRoutine(build, { ...none, stage: "fragment" })).not.toThrow();
  });

  /**
   * A `Case` given an empty array of values builds an `if` with no
   * condition, and the compiler crashes on it rather than naming the cause.
   *
   * @canon bug-a-case-with-no-values-crashes-the-compiler
   */
  it.fails("refuses a Case with no values, naming it", () => {
    const program = Fn(() => {
      const v = float(0).toVar();
      Switch(int(uniform("float")), (s) => {
        s.Case([], () => {
          v.assign(float(1));
        });
      });
      return vec4(v);
    });
    expect(() => compileGlsl.fragment(program())).toThrow(/Case/);
  });

  /**
   * A JavaScript array whose length no vector has becomes the float of its
   * first element, silently, so `vec4(0).add([1, 2, 3, 4, 5])` adds 1.
   *
   * @canon bug-an-array-of-a-length-no-vector-has-becomes-its-first-element
   */
  it.fails("refuses a JavaScript array whose length no vector has", () => {
    const build = (a: any) =>
      Fn(() =>
        vec4(a)
          .add([1, 2, 3, 4, 5] as any)
          .toVar(),
      )();
    expect(() => compileJSRoutine(build, param)).toThrow();
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
   * Two `Fn`s that each take the variable name `color`, compiled as the roots
   * of one program, both declare `color`, which GLSL and WGSL refuse.
   *
   * @canon bug-roots-of-one-program-declare-one-variable-name-twice
   */
  it.fails("numbers a variable name another root of the program took", () => {
    const build = () => {
      const u = uniform("float");
      return [Fn(() => u.add(1).toVar("color"))(), Fn(() => u.add(2).toVar("color"))()];
    };
    const glsl = compileGlsl.fragment(build() as any);
    expect(glsl).toContain("float color = ");
    expect(glsl).toContain("float color1 = ");
  });

  /**
   * `serialize` keeps the generated name of a stage output, `_rmsl_oN`, so
   * two graphs restored from it write one output.
   *
   * @canon bug-serialize-keeps-the-generated-name-of-an-output
   */
  it.fails("gives each restored output a name of its own", () => {
    const color = output("vec4");
    const graph = serialize(
      Fn(() => {
        color.assign(vec4(1, 0, 0, 1));
      })(),
    );
    const glsl = compileGlsl.fragment([roundTrip(graph), roundTrip(graph)] as any);
    expect(glsl.match(/ out vec4 /g)).toHaveLength(2);
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
    expect(run.run({ uniforms: { gain: 1 } })).toBe(Infinity);
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
});
