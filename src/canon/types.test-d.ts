import { describe, expectTypeOf, it } from "vitest";
import { Discard, Fn, float, int, uint, vec2, vec3, vec4, bool, outputStruct } from "../rmsl";
import { compileJSFragment, compileJSGrid, compileJSRoutine } from "../js";
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
    // @ts-expect-error nor a sum
    vec2(1, 2).add(vec3(1, 2, 3));
    // @ts-expect-error nor a product
    vec3(1, 2, 3).mul(vec2(1, 2));
  });
});

const none = { name: "main", params: [] };

describe("what a routine returns", () => {
  /**
   * @canon spec-a-routine-is-typed-by-the-value-it-returns
   */
  it("is typed by the value its program returns, on JS", () => {
    expectTypeOf(compileJSRoutine(() => Fn(() => float(1))(), none)({})).toEqualTypeOf<number>();
    expectTypeOf(compileJSRoutine(() => Fn(() => int(1))(), none)({})).toEqualTypeOf<number>();
    expectTypeOf(compileJSRoutine(() => Fn(() => bool(true))(), none)({})).toEqualTypeOf<boolean>();
    expectTypeOf(compileJSRoutine(() => Fn(() => vec3(1, 2, 3))(), none)({})).toEqualTypeOf<number[]>();
  });

  /**
   * @canon spec-a-routine-is-typed-by-the-value-it-returns
   */
  it("is typed by the value its program returns, on WASM", () => {
    expectTypeOf(compileWasmRoutine(() => Fn(() => float(1))(), none)({})).toEqualTypeOf<number>();
    expectTypeOf(compileWasmRoutine(() => Fn(() => bool(true))(), none)({})).toEqualTypeOf<boolean>();
    expectTypeOf(compileWasmRoutine(() => Fn(() => vec2(1, 2))(), none)({})).toEqualTypeOf<number[]>();
  });
});

describe("what a fragment stage returns", () => {
  /**
   * @canon spec-an-output-struct-writes-each-member-at-its-position
   */
  it("types the outputs of an outputStruct by position, with no colour", () => {
    const stage = compileJSFragment(() => Fn(() => outputStruct(float(7), vec3(1, 2, 3)))(), none);
    const result = stage({});
    expectTypeOf(result).toEqualTypeOf<{ value: undefined; outputs: [number, number[]]; fragDepth?: number } | null>();
  });

  /**
   * @canon spec-a-fragment-stage-returns-its-colour-and-outputs
   */
  it("types a stage that returns nothing as having no colour and no outputs", () => {
    const stage = compileJSFragment(
      () =>
        Fn(() => {
          Discard();
        })(),
      none,
    );
    expectTypeOf(stage({})).toEqualTypeOf<{ value: undefined; outputs: []; fragDepth?: number } | null>();
  });

  /**
   * @canon spec-a-fragment-stage-returns-its-colour-and-outputs
   */
  it("types the colour of a stage that returns one, with no outputs", () => {
    const stage = compileJSFragment(() => Fn(() => vec4(1, 2, 3, 4))(), none);
    expectTypeOf(stage({})).toEqualTypeOf<{ value: number[]; outputs: []; fragDepth?: number } | null>();
  });
});

describe("what a grid fills", () => {
  /**
   * @canon spec-a-grid-fills-a-float64-array-for-a-float-result
   * @canon spec-a-grid-fills-an-int32-array-for-an-int-result
   * @canon spec-a-grid-fills-a-uint32-array-for-a-uint-result
   * @canon spec-a-grid-writes-a-bool-result-as-one-or-zero-in-an-int32-array
   */
  it("is typed by the type of the result", () => {
    expectTypeOf(compileJSGrid(() => Fn(() => float(1))(), none)({}, 1, 1)).toEqualTypeOf<Float64Array>();
    expectTypeOf(compileJSGrid(() => Fn(() => vec3(1, 2, 3))(), none)({}, 1, 1)).toEqualTypeOf<Float64Array>();
    expectTypeOf(compileJSGrid(() => Fn(() => int(1))(), none)({}, 1, 1)).toEqualTypeOf<Int32Array>();
    expectTypeOf(compileJSGrid(() => Fn(() => bool(true))(), none)({}, 1, 1)).toEqualTypeOf<Int32Array>();
    expectTypeOf(compileJSGrid(() => Fn(() => uint(1))(), none)({}, 1, 1)).toEqualTypeOf<Uint32Array>();
  });
});
