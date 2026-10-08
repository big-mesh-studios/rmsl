import { describe, expectTypeOf, it } from "vitest";
import {
  Discard,
  Fn,
  float,
  int,
  uint,
  vec2,
  vec3,
  vec4,
  bool,
  ivec2,
  uvec3,
  bvec4,
  mat3,
  outputStruct,
  attribute,
  dot,
  length,
  distance,
  screenSize,
  type Node,
  type UniformNode,
} from "../rmsl";
import { compileJSFragment, compileJSGrid, compileJSRoutine, compileJSVertex, createJsRoutine } from "../js";
import {
  compileWasmFragment,
  compileWasmGrid,
  compileWasmRoutine,
  compileWasmVertex,
  createWasmRoutine,
} from "../wasm";

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

  /**
   * @canon spec-length-distance-and-dot-take-only-floats
   */
  it("refuses length, distance and dot of an integer or a boolean", () => {
    expectTypeOf(dot(float(2), float(3))).toEqualTypeOf<Node<"float">>();
    expectTypeOf(length(vec3(1, 2, 3))).toEqualTypeOf<Node<"float">>();
    // @ts-expect-error an int has no length
    length(int(3));
    // @ts-expect-error nor two uints a distance
    distance(uint(3), uint(4));
    // @ts-expect-error nor two ivec2s a dot product
    dot(ivec2(1, 2), ivec2(3, 4));
    // @ts-expect-error nor a bool a length
    length(bool(true));
    // @ts-expect-error nor a vector and a scalar a dot product
    dot(vec3(1, 2, 3), 2);
    // @ts-expect-error nor a scalar and a vector a distance
    distance(float(2), vec2(1, 2));
    // @ts-expect-error nor a matrix a length
    length(mat3(1));
    expectTypeOf(dot(vec3(1, 2, 3), [1, 2, 3])).toEqualTypeOf<Node<"float">>();
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
    expectTypeOf(compileJSRoutine(() => Fn(() => vec3(1, 2, 3))(), none)({})).toEqualTypeOf<Float64Array>();
    expectTypeOf(compileJSRoutine(() => Fn(() => mat3(1))(), none)({})).toEqualTypeOf<Float64Array>();
    expectTypeOf(compileJSRoutine(() => Fn(() => ivec2(1, 2))(), none)({})).toEqualTypeOf<Int32Array>();
    expectTypeOf(compileJSRoutine(() => Fn(() => uvec3(1, 2, 3))(), none)({})).toEqualTypeOf<Uint32Array>();
    expectTypeOf(compileJSRoutine(() => Fn(() => bvec4(true))(), none)({})).toEqualTypeOf<Int32Array>();
  });

  /**
   * @canon spec-a-routine-is-typed-by-the-value-it-returns
   */
  it("names a Float32Array for a float vector at float: f32, and keeps the integer kinds, on JS", () => {
    const f32 = { ...none, float: "f32" as const };
    expectTypeOf(compileJSRoutine(() => Fn(() => vec3(1, 2, 3))(), f32)({})).toEqualTypeOf<Float32Array>();
    expectTypeOf(compileJSRoutine(() => Fn(() => mat3(1))(), f32)({})).toEqualTypeOf<Float32Array>();
    expectTypeOf(compileJSRoutine(() => Fn(() => float(1))(), f32)({})).toEqualTypeOf<number>();
    expectTypeOf(compileJSRoutine(() => Fn(() => ivec2(1, 2))(), f32)({})).toEqualTypeOf<Int32Array>();
    expectTypeOf(
      compileJSRoutine(() => Fn(() => vec3(1, 2, 3))(), { ...none, float: "f64" })({}),
    ).toEqualTypeOf<Float64Array>();
  });

  /**
   * @canon spec-a-routine-is-typed-by-the-value-it-returns
   */
  it("is typed by the value its program returns, on WASM", () => {
    expectTypeOf(compileWasmRoutine(() => Fn(() => float(1))(), none)({})).toEqualTypeOf<number>();
    expectTypeOf(compileWasmRoutine(() => Fn(() => bool(true))(), none)({})).toEqualTypeOf<boolean>();
    expectTypeOf(compileWasmRoutine(() => Fn(() => vec2(1, 2))(), none)({})).toEqualTypeOf<Float64Array>();
    expectTypeOf(compileWasmRoutine(() => Fn(() => uvec3(1, 2, 3))(), none)({})).toEqualTypeOf<Uint32Array>();
    expectTypeOf(
      compileWasmRoutine(() => Fn(() => vec2(1, 2))(), { ...none, float: "f32" })({}),
    ).toEqualTypeOf<Float32Array>();
  });

  /**
   * @canon spec-a-routine-is-typed-by-the-value-it-returns
   */
  it("carries the width into the run of an adapter, on JS and WASM", () => {
    expectTypeOf(createJsRoutine(vec3(1, 2, 3)).run()).toEqualTypeOf<Float64Array>();
    expectTypeOf(createJsRoutine(vec3(1, 2, 3), { float: "f32" }).run()).toEqualTypeOf<Float32Array>();
    expectTypeOf(createWasmRoutine(vec3(1, 2, 3), { float: "f32" }).run()).toEqualTypeOf<Float32Array>();
  });
});

describe("what a fragment stage returns", () => {
  /**
   * @canon spec-an-output-struct-writes-each-member-at-its-position
   */
  it("types the outputs of an outputStruct by position, with no colour", () => {
    const stage = compileJSFragment(() => Fn(() => outputStruct(float(7), vec3(1, 2, 3)))(), none);
    const result = stage({});
    expectTypeOf(result).toEqualTypeOf<{
      value: undefined;
      outputs: [number, Float64Array];
      fragDepth?: number;
    } | null>();
    const at32 = compileWasmFragment(() => Fn(() => outputStruct(float(7), vec3(1, 2, 3)))(), {
      ...none,
      float: "f32",
    });
    expectTypeOf(at32({})).toEqualTypeOf<{
      value: undefined;
      outputs: [number, Float32Array];
      fragDepth?: number;
    } | null>();
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
    expectTypeOf(stage({})).toEqualTypeOf<{ value: Float64Array; outputs: []; fragDepth?: number } | null>();
    const at32 = compileJSFragment(() => Fn(() => vec4(1, 2, 3, 4))(), { ...none, float: "f32" });
    expectTypeOf(at32({})).toEqualTypeOf<{ value: Float32Array; outputs: []; fragDepth?: number } | null>();
  });
});

describe("what a vertex stage returns", () => {
  /**
   * @canon spec-a-vertex-stage-returns-its-position-and-varyings
   */
  it("types the position in the typed array of the width, on JS and WASM", () => {
    const place = attribute("vec3");
    const build = () => Fn(() => vec4(place, 1))();
    expectTypeOf(compileJSVertex(build, none)({}).position).toEqualTypeOf<Float64Array>();
    expectTypeOf(compileJSVertex(build, { ...none, float: "f32" })({}).position).toEqualTypeOf<Float32Array>();
    expectTypeOf(compileWasmVertex(build, { ...none, float: "f32" })({}).position).toEqualTypeOf<Float32Array>();
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
    expectTypeOf(
      compileJSGrid(() => Fn(() => vec3(1, 2, 3))(), { ...none, float: "f32" })({}, 1, 1),
    ).toEqualTypeOf<Float32Array>();
    expectTypeOf(
      compileWasmGrid(() => Fn(() => float(1))(), { ...none, float: "f32" })({}, 1, 1),
    ).toEqualTypeOf<Float32Array>();
    expectTypeOf(
      compileJSGrid(() => Fn(() => int(1))(), { ...none, float: "f32" })({}, 1, 1),
    ).toEqualTypeOf<Int32Array>();
  });
});

describe("the screen size", () => {
  /**
   * @canon spec-screen-size-is-one-uniform-everywhere
   */
  it("is typed a uniform, whose name the host binds", () => {
    expectTypeOf(screenSize()).toEqualTypeOf<UniformNode<"vec2">>();
    expectTypeOf(screenSize().name).toEqualTypeOf<string>();
  });
});
