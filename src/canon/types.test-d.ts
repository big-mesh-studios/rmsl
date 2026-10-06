import { describe, expectTypeOf, it } from "vitest";
import { Fn, float, int, vec2, vec3, bool } from "../rmsl";
import { compileJSRoutine } from "../js";
import { compileWasmRoutine } from "../wasm";

describe("refusals the types make", () => {
  /**
   * @canon spec-operands-of-different-widths-are-refused
   */
  it("refuses an operation on vectors of different widths", () => {
    // @ts-expect-error a vec2 and a vec3 have no dot product
    vec2(1, 2).dot(vec3(1, 2, 3));
    // @ts-expect-error nor a minimum
    vec2(1, 2).min(vec3(1, 2, 3));
  });
});

const none = { name: "main", params: [] };

describe("what a routine returns", () => {
  /**
   * @canon spec-a-routine-is-typed-by-the-value-it-returns
   */
  it("is typed by the value its program returns, on JS", () => {
    expectTypeOf(compileJSRoutine(() => Fn(() => float(1))(), none).run({})).toEqualTypeOf<number>();
    expectTypeOf(compileJSRoutine(() => Fn(() => int(1))(), none).run({})).toEqualTypeOf<number>();
    expectTypeOf(compileJSRoutine(() => Fn(() => bool(true))(), none).run({})).toEqualTypeOf<boolean>();
    expectTypeOf(compileJSRoutine(() => Fn(() => vec3(1, 2, 3))(), none).run({})).toEqualTypeOf<number[]>();
  });

  /**
   * @canon spec-a-routine-is-typed-by-the-value-it-returns
   */
  it("is typed by the value its program returns, on WASM", () => {
    expectTypeOf(compileWasmRoutine(() => Fn(() => float(1))(), none).run({})).toEqualTypeOf<number>();
    expectTypeOf(compileWasmRoutine(() => Fn(() => bool(true))(), none).run({})).toEqualTypeOf<boolean>();
    expectTypeOf(compileWasmRoutine(() => Fn(() => vec2(1, 2))(), none).run({})).toEqualTypeOf<number[]>();
  });
});
