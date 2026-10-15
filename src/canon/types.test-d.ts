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
  uvec2,
  bvec4,
  mat3,
  outputStruct,
  attribute,
  instancedArray,
  dot,
  length,
  distance,
  sin,
  floor,
  pow,
  mix,
  step,
  smoothstep,
  normalize,
  cross,
  bitAnd,
  bitNot,
  shiftLeft,
  ivec3,
  abs,
  sign,
  negate,
  screenSize,
  type Node,
  type UniformNode,
  type StorageBufferAttribute,
  type SetUniform,
  type SetUniformByNode,
  type UniformValue,
  uniform,
  uniformArray,
} from "../rmsl";
import {
  compileJSFragment,
  compileJSGrid,
  compileJSRoutine,
  compileJSVertex,
  createJsRoutine,
  type JsAdapter,
  type JsComputeAdapter,
} from "../js";
import type { CpuAdapter, CpuRoutineAdapter } from "../backends/adapter-cpu";
import type { GlslAdapter } from "../glsl";
import type { WgslAdapter, WgslComputeAdapter, WgslContext } from "../wgsl";
import type { WasmContext } from "../wasm";
import type { CpuShaderContext } from "../backends/cpu";
import {
  Color,
  Matrix3,
  Matrix4,
  PerspectiveCamera,
  Quaternion,
  Vector2,
  Vector3,
  Vector4,
  type Matrix3Tuple,
  type Matrix4Tuple,
} from "../scene";
import {
  compileWasmFragment,
  compileWasmGrid,
  compileWasmRoutine,
  compileWasmVertex,
  createWasmRoutine,
  type WasmAdapter,
  type WasmComputeAdapter,
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
   * @canon spec-an-integer-and-a-float-operand-are-refused
   */
  it("refuses an operation on an integer and a float", () => {
    // @ts-expect-error an int and a float have no sum
    int(1).add(float(2));
    // @ts-expect-error nor a float and a uint a product
    float(1).mul(uint(2));
    // @ts-expect-error nor an ivec2 and a vec2 a minimum
    ivec2(1, 2).min(vec2(1, 2));
    // @ts-expect-error nor an int and a float an order
    int(1).lessThan(float(2));
    expectTypeOf(int(1).add(2)).toEqualTypeOf<Node<"int">>();
    expectTypeOf(float(1).add(2)).toEqualTypeOf<Node<"float">>();
  });

  /**
   * @canon spec-a-signed-and-an-unsigned-integer-operand-are-refused
   */
  it("refuses an operation on a signed and an unsigned integer", () => {
    // @ts-expect-error an int and a uint have no sum
    int(1).add(uint(2));
    // @ts-expect-error nor a uint and an int a remainder
    uint(1).mod(int(2));
    // @ts-expect-error nor an int and a uint an order
    int(1).lessThan(uint(2));
    // @ts-expect-error nor an ivec2 and a uvec2 a product
    ivec2(1, 2).mul(uvec2(1, 2));
    // @ts-expect-error nor a uvec2 and an int a maximum
    uvec2(1, 2).max(int(1));
    expectTypeOf(int(1).add(2)).toEqualTypeOf<Node<"int">>();
    expectTypeOf(uint(1).add(2)).toEqualTypeOf<Node<"uint">>();
    expectTypeOf(uvec2(1, 2).add(2)).toEqualTypeOf<Node<"uvec2">>();
    // @ts-expect-error nor an int and a uint a bitwise and
    int(1).bitAnd(uint(2));
    // @ts-expect-error nor a uvec2 and an ivec2 a bitwise or
    uvec2(1, 2).bitOr(ivec2(1, 2));
    // @ts-expect-error nor an int between uint bounds a clamp
    int(1).clamp(uint(0), uint(2));
    // @ts-expect-error nor an int and a uint a choice
    bool(true).select(int(1), uint(2));
  });

  /**
   * @canon spec-an-integer-and-a-float-operand-are-refused
   */
  it("refuses an integer beside a float in a function of several operands", () => {
    // @ts-expect-error a float has no power of an int
    float(2).pow(int(3));
    // @ts-expect-error nor a float a clamp between int bounds
    float(2).clamp(int(0), int(1));
    // @ts-expect-error nor a vec3 a blend by an int weight
    vec3(1, 2, 3).mix(vec3(1, 2, 3), int(1));
    // @ts-expect-error nor an int and a float a bitwise and
    int(1).bitAnd(float(2));
    // @ts-expect-error nor an int a shift by a float
    int(1).shiftLeft(float(2));
    // @ts-expect-error nor an int and a float a choice
    bool(true).select(int(1), float(2));
  });

  /**
   * @canon spec-a-shift-amount-may-be-of-either-integer-kind
   */
  it("takes a shift amount of either integer kind", () => {
    expectTypeOf(int(1).shiftLeft(uint(2))).toEqualTypeOf<Node<"int">>();
    expectTypeOf(uint(1).shiftRight(int(2))).toEqualTypeOf<Node<"uint">>();
    expectTypeOf(ivec2(1, 2).shiftLeft(uvec2(1, 2))).toEqualTypeOf<Node<"ivec2">>();
    expectTypeOf(uvec2(1, 2).shiftRight(int(1))).toEqualTypeOf<Node<"uvec2">>();
  });

  /**
   * @canon spec-a-math-function-of-floats-refuses-an-integer-operand
   */
  it("refuses an integer operand of a function defined only for floats", () => {
    // @ts-expect-error an int has no sine
    sin(int(1));
    // @ts-expect-error nor a uint a floor
    floor(uint(1));
    // @ts-expect-error nor an int a power
    pow(int(2), 3);
    // @ts-expect-error nor two ints a blend
    mix(int(1), int(2), 0.5);
    // @ts-expect-error nor an int a step
    step(0, int(1));
    // @ts-expect-error nor an int a smooth step
    smoothstep(0, 1, int(1));
    // @ts-expect-error nor an ivec3 a direction
    normalize(ivec3(1, 2, 3));
    // @ts-expect-error nor two ivec3s a cross product
    cross(ivec3(1, 2, 3), ivec3(1, 2, 3));
    // @ts-expect-error nor an int a method of floats
    int(1).sin();
    sin(1);
    pow(float(2), 3);
    mix(vec3(1, 2, 3), [0, 0, 0], 0.5);
  });

  /**
   * @canon spec-an-integer-operation-refuses-a-float-operand
   */
  it("refuses a float operand of an integer operation", () => {
    // @ts-expect-error a float has no bitwise and
    bitAnd(float(1), 2);
    // @ts-expect-error nor a vec2 a bitwise not
    bitNot(vec2(1, 2));
    // @ts-expect-error nor an int a shift by a float
    shiftLeft(int(1), float(2));
    // @ts-expect-error nor a float a method of integers
    float(1).bitAnd(1);
    bitAnd(int(1), 2);
    shiftLeft(uint(1), int(2));
  });

  /**
   * @canon spec-a-signed-operation-refuses-an-unsigned-operand
   */
  it("refuses an unsigned operand of abs, sign and negate", () => {
    // @ts-expect-error a uint has no absolute value
    abs(uint(1));
    // @ts-expect-error nor a uvec2 a sign
    sign(uvec2(1, 2));
    // @ts-expect-error nor a uint a negation
    negate(uint(1));
    abs(int(-1));
    sign(vec3(1, 2, 3));
    negate(mat3(1));
    negate(ivec2(1, 2));
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

describe("the array setAttribute takes", () => {
  const ids = attribute("ivec2");
  const flags = attribute("uint");
  const position = attribute("vec3");
  let glsl!: GlslAdapter;
  let wgsl!: WgslAdapter;
  let js!: JsAdapter;
  let wasm!: WasmAdapter;

  /**
   * @canon spec-an-adapter-takes-only-the-array-of-a-numeric-attributes-kind
   */
  it("takes only the array of an attribute node's kind on a GPU adapter", () => {
    for (const gpu of [glsl, wgsl]) {
      gpu.setAttribute(ids, new Int32Array(2));
      gpu.setAttribute(flags, new Uint32Array(1));
      gpu.setAttribute(position, new Float32Array(3));
      // @ts-expect-error an ivec2 attribute takes an Int32Array
      gpu.setAttribute(ids, new Float32Array(2));
      // @ts-expect-error an ivec2 attribute takes no narrower array either
      gpu.setAttribute(ids, new Int16Array(2));
      // @ts-expect-error a uint attribute takes a Uint32Array
      gpu.setAttribute(flags, new Int32Array(1));
      // @ts-expect-error a vec3 attribute takes a Float32Array
      gpu.setAttribute(position, new Uint8Array(3));
      // @ts-expect-error a GPU adapter reads no Float64Array
      gpu.setAttribute(position, new Float64Array(3));
    }
  });

  /**
   * @canon spec-the-type-of-set-attribute-takes-an-int32-or-uint32-array-for-a-bool-attribute
   */
  it("takes an Int32Array or a Uint32Array for a bool attribute node on every adapter", () => {
    const flag = attribute("bool");
    const flags = attribute("bvec3");
    glsl.setAttribute(flag, new Int32Array(3));
    glsl.setAttribute(flag, new Uint32Array(3));
    glsl.setAttribute(flags, new Int32Array(9));
    glsl.setAttribute(flags, new Uint32Array(9));
    // @ts-expect-error a bool attribute takes no float array
    glsl.setAttribute(flag, new Float32Array(3));
    // @ts-expect-error a bvec3 attribute takes no float array
    glsl.setAttribute(flags, new Float32Array(9));
    wgsl.setAttribute(flag, new Int32Array(3));
    wgsl.setAttribute(flag, new Uint32Array(3));
    wgsl.setAttribute(flags, new Int32Array(9));
    wgsl.setAttribute(flags, new Uint32Array(9));
    // @ts-expect-error a bool attribute takes no float array
    wgsl.setAttribute(flag, new Float32Array(3));
    // @ts-expect-error a bvec3 attribute takes no float array
    wgsl.setAttribute(flags, new Float32Array(9));
    js.setAttribute(flag, new Int32Array(3));
    js.setAttribute(flag, new Uint32Array(3));
    js.setAttribute(flags, new Int32Array(9));
    js.setAttribute(flags, new Uint32Array(9));
    // @ts-expect-error a bool attribute takes no float array
    js.setAttribute(flag, new Float32Array(3));
    // @ts-expect-error a bvec3 attribute takes no float array
    js.setAttribute(flags, new Float64Array(9));
    wasm.setAttribute(flag, new Int32Array(3));
    wasm.setAttribute(flag, new Uint32Array(3));
    wasm.setAttribute(flags, new Int32Array(9));
    wasm.setAttribute(flags, new Uint32Array(9));
    // @ts-expect-error a bool attribute takes no float array
    wasm.setAttribute(flag, new Float32Array(3));
    // @ts-expect-error a bvec3 attribute takes no float array
    wasm.setAttribute(flags, new Float64Array(9));
  });

  /**
   * @canon spec-an-adapter-takes-only-the-array-of-a-numeric-attributes-kind
   */
  it("takes a Float64Array for a float attribute node on a CPU adapter", () => {
    for (const cpu of [js, wasm]) {
      cpu.setAttribute(position, new Float32Array(3));
      cpu.setAttribute(position, new Float64Array(3));
      cpu.setAttribute(ids, new Int32Array(2));
      // @ts-expect-error an ivec2 attribute takes an Int32Array
      cpu.setAttribute(ids, new Float64Array(2));
    }
  });

  /**
   * @canon spec-an-adapter-checks-an-attribute-named-by-its-slot-where-set-attribute-is-called
   */
  it("takes any typed array by slot name", () => {
    glsl.setAttribute(ids.name, new Float32Array(2));
    js.setAttribute(ids.name, new Uint8Array(2));
  });

  /**
   * @canon spec-a-storage-buffer-takes-only-an-array-of-its-element-kind
   */
  it("takes only the array of a storage node's element kind on every adapter", () => {
    const counts = instancedArray(2, "int");
    const weights = instancedArray(2, "vec2");
    let wgslCompute!: WgslComputeAdapter;
    let jsCompute!: JsComputeAdapter;
    let wasmCompute!: WasmComputeAdapter;
    let grid!: CpuAdapter;
    for (const adapter of [glsl, wgsl, wgslCompute]) {
      adapter.setAttribute(counts, new Int32Array(2));
      adapter.setAttribute(counts.toReadOnly(), new Int32Array(2));
      adapter.setAttribute(weights, new Float32Array(4));
      // @ts-expect-error an int storage buffer takes an Int32Array
      adapter.setAttribute(counts, new Float32Array(2));
      // @ts-expect-error a GPU target reads no Float64Array
      adapter.setAttribute(weights, new Float64Array(4));
    }
    for (const adapter of [js, wasm, jsCompute, wasmCompute, grid]) {
      adapter.setAttribute(weights, new Float64Array(4));
      // @ts-expect-error an int storage buffer takes an Int32Array
      adapter.setAttribute(counts.toReadOnly(), new Uint32Array(2));
    }
  });
});

describe("the arrays a CPU program and a context take for a storage buffer", () => {
  /**
   * @canon spec-the-type-of-a-programs-storages-admits-only-typed-arrays
   */
  it("takes only typed arrays in the storages of a CPU program's context", () => {
    const ctx = (storages: CpuShaderContext["storages"]) => storages;
    ctx({ a: new Float64Array(2), b: new Float32Array(2), c: new Int32Array(2), d: new Uint32Array(2) });
    // @ts-expect-error a plain array holds numbers of no one kind
    ctx({ a: [1, 2] });
  });

  /**
   * @canon spec-a-context-write-takes-only-the-array-of-its-attributes-kind
   */
  it("takes only typed arrays of a component kind in a context's write", () => {
    const write = (context: WasmContext, wgsl: WgslContext, attribute: StorageBufferAttribute) => {
      context.write(attribute, new Float64Array(2));
      wgsl.write(attribute, new Uint32Array(2));
      // @ts-expect-error a plain array holds numbers of no one kind
      context.write(attribute, [1, 2]);
      // @ts-expect-error a WGSL context holds no 64-bit floats
      wgsl.write(attribute, new Float64Array(2));
      // @ts-expect-error an 8-bit array is no component kind of a storage buffer
      wgsl.write(attribute, new Uint8Array(2));
    };
    void write;
  });
});

describe("the value setUniform takes for a boolean uniform", () => {
  let glsl!: GlslAdapter;

  /**
   * @canon spec-a-bool-host-value-reads-true-for-a-nonzero-integer
   */
  it("takes true, false, 1 or 0", () => {
    const flag = uniform("bool");
    glsl.setUniform(flag, true);
    glsl.setUniform(flag, false);
    glsl.setUniform(flag, 1);
    glsl.setUniform(flag, 0);
    // @ts-expect-error a bool takes 1 or 0, not any number
    glsl.setUniform(flag, 2);
  });

  /**
   * @canon spec-the-types-hold-a-uniform-value-to-its-shape
   */
  it("takes for a boolean vector an array of them, or an integer array", () => {
    const pair = uniform("bvec2");
    glsl.setUniform(pair, [true, 0]);
    glsl.setUniform(pair, Int32Array.of(1, 0));
    glsl.setUniform(pair, Uint32Array.of(1, 0));
    // @ts-expect-error a float array holds no boolean exactly
    glsl.setUniform(pair, Float32Array.of(1, 0));
    // @ts-expect-error a bvec2 takes two components
    glsl.setUniform(pair, [true]);
  });
});

describe("the value setUniform takes for a vector or matrix uniform", () => {
  let glsl!: GlslAdapter;

  /**
   * @canon spec-the-types-hold-a-uniform-value-to-its-shape
   */
  it("derives the value of each type from its width or its columns and rows", () => {
    type Floats = Float32Array | Float64Array;
    type Bools = boolean | 0 | 1;
    expectTypeOf<UniformValue<"float">>().toEqualTypeOf<number>();
    expectTypeOf<UniformValue<"int">>().toEqualTypeOf<number>();
    expectTypeOf<UniformValue<"uint">>().toEqualTypeOf<number>();
    expectTypeOf<UniformValue<"bool">>().toEqualTypeOf<Bools>();
    expectTypeOf<UniformValue<"vec2">>().toEqualTypeOf<[number, number] | Floats>();
    expectTypeOf<UniformValue<"vec4">>().toEqualTypeOf<[number, number, number, number] | Floats>();
    expectTypeOf<UniformValue<"ivec3">>().toEqualTypeOf<[number, number, number] | Int32Array>();
    expectTypeOf<UniformValue<"uvec2">>().toEqualTypeOf<[number, number] | Uint32Array>();
    expectTypeOf<UniformValue<"bvec3">>().toEqualTypeOf<[Bools, Bools, Bools] | Int32Array | Uint32Array>();
    expectTypeOf<UniformValue<"mat2">>().toEqualTypeOf<[number, number, number, number] | Floats>();
    expectTypeOf<UniformValue<"mat3x2">>().toEqualTypeOf<[number, number, number, number, number, number] | Floats>();
    expectTypeOf<Exclude<UniformValue<"mat4">, Floats>["length"]>().toEqualTypeOf<16>();
    expectTypeOf<UniformValue<"sampler2D">>().toBeNever();
    expectTypeOf<UniformValue<"void">>().toBeNever();
  });

  /**
   * @canon spec-the-types-hold-a-uniform-value-to-its-shape
   */
  it("takes exactly the components of the type, flat for a matrix, and any shape by slot", () => {
    const tint = uniform("vec3");
    const turn = uniform("mat2");
    const frame = uniform("mat3x2");
    const ids = uniform("ivec2");
    glsl.setUniform(tint, [1, 0, 0]);
    glsl.setUniform(tint, Float32Array.of(1, 0, 0));
    // @ts-expect-error a vec3 takes three components
    glsl.setUniform(tint, [1, 0]);
    glsl.setUniform(turn, [1, 0, 0, 1]);
    glsl.setUniform(turn, Float32Array.of(1, 0, 0, 1));
    // @ts-expect-error a mat2 takes four components
    glsl.setUniform(turn, [1, 0, 0]);
    // @ts-expect-error a matrix takes its columns in one flat array
    glsl.setUniform(turn, [
      [1, 0],
      [0, 1],
    ]);
    glsl.setUniform(frame, [1, 0, 0, 1, 0, 0]);
    // @ts-expect-error a mat3x2 takes six components
    glsl.setUniform(frame, [1, 0, 0, 1]);
    const wide = uniform("mat4x3");
    glsl.setUniform(wide, [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
    // @ts-expect-error a mat4x3 takes twelve components, four columns of three
    glsl.setUniform(wide, [1, 0, 0, 0, 1, 0, 0, 0, 1]);
    glsl.setUniform(ids, Int32Array.of(1, 2));
    // @ts-expect-error an ivec2 takes an Int32Array, not a float array
    glsl.setUniform(ids, Float32Array.of(1, 2));
    glsl.setUniform(tint.name, Float32Array.of(1, 0, 0));
    glsl.setUniform("colours", [
      [1, 0, 0, 1],
      [0, 1, 0, 1],
    ]);
  });
});

describe("the type of setUniform", () => {
  /**
   * @canon spec-every-adapter-shares-one-type-of-setuniform
   */
  it("is SetUniform on every adapter, and SetUniformByNode on a context", () => {
    expectTypeOf<GlslAdapter["setUniform"]>().toEqualTypeOf<SetUniform>();
    expectTypeOf<WgslAdapter["setUniform"]>().toEqualTypeOf<SetUniform>();
    expectTypeOf<WgslComputeAdapter["setUniform"]>().toEqualTypeOf<SetUniform>();
    expectTypeOf<JsAdapter["setUniform"]>().toEqualTypeOf<SetUniform>();
    expectTypeOf<WasmAdapter["setUniform"]>().toEqualTypeOf<SetUniform>();
    expectTypeOf<JsComputeAdapter["setUniform"]>().toEqualTypeOf<SetUniform>();
    expectTypeOf<WasmComputeAdapter["setUniform"]>().toEqualTypeOf<SetUniform>();
    expectTypeOf<CpuAdapter["setUniform"]>().toEqualTypeOf<SetUniform>();
    expectTypeOf<CpuRoutineAdapter["setUniform"]>().toEqualTypeOf<SetUniform>();
    expectTypeOf<WgslContext["setUniform"]>().toEqualTypeOf<SetUniformByNode>();
    expectTypeOf<WasmContext["setUniform"]>().toEqualTypeOf<SetUniformByNode>();
  });
});

describe("the arrays of the scene math classes", () => {
  let glsl!: GlslAdapter;

  /**
   * @canon spec-the-math-classes-type-their-arrays-as-tuples
   */
  it("set a uniform of their type with no cast, and a uniform of another shape is still refused", () => {
    const camera = new PerspectiveCamera();
    const mvp = uniform("mat4");
    const normalMatrix = uniform("mat3");
    const tint = uniform("vec3");
    glsl.setUniform(mvp, camera.projectionMatrix.elements);
    glsl.setUniform(mvp, new Matrix4().toArray());
    glsl.setUniform(normalMatrix, new Matrix3().elements);
    glsl.setUniform(normalMatrix, new Matrix3().toArray());
    glsl.setUniform(tint, new Color(1, 0.5, 0).toArray());
    glsl.setUniform(tint, new Vector3(1, 0.5, 0).toArray());
    glsl.setUniform(uniform("vec2"), new Vector2(1, 0.5).toArray());
    glsl.setUniform(uniform("vec4"), new Vector4(1, 0.5, 0, 1).toArray());
    glsl.setUniform(uniform("vec4"), new Quaternion().toArray());
    // @ts-expect-error a vec3 takes three components
    glsl.setUniform(tint, [1, 0.5]);
    // @ts-expect-error a mat4 takes sixteen components, not a mat3's nine
    glsl.setUniform(mvp, new Matrix3().elements);
    // @ts-expect-error a vec3 takes three components, not a vec2's two
    glsl.setUniform(tint, new Vector2().toArray());
  });

  /**
   * @canon spec-the-math-classes-type-their-arrays-as-tuples
   */
  it("type elements and toArray() as tuples of their length, and toArray(array, offset) as the array", () => {
    expectTypeOf(new Matrix4().elements).toEqualTypeOf<Matrix4Tuple>();
    expectTypeOf(new Matrix3().elements).toEqualTypeOf<Matrix3Tuple>();
    expectTypeOf(new Vector3().toArray()).toEqualTypeOf<[number, number, number]>();
    expectTypeOf(new Color().toArray()).toEqualTypeOf<[number, number, number]>();
    expectTypeOf(new Quaternion().toArray()).toEqualTypeOf<[number, number, number, number]>();
    const out = new Float32Array(6);
    expectTypeOf(new Vector3().toArray(out, 3)).toEqualTypeOf(out);
    expectTypeOf(new Matrix4().toArray([] as number[], 16)).toEqualTypeOf<number[]>();
  });
});

describe("a uniform set to nothing", () => {
  let glsl!: GlslAdapter;
  let wgsl!: WgslAdapter;
  let wgslCompute!: WgslComputeAdapter;
  let wgslContext!: WgslContext;
  let wasmContext!: WasmContext;
  let js!: JsAdapter;
  let wasm!: WasmAdapter;
  let jsCompute!: JsComputeAdapter;
  let wasmCompute!: WasmComputeAdapter;
  let cpu!: CpuAdapter;

  /**
   * @canon spec-the-types-admit-a-uniform-set-to-nothing
   */
  it("takes null or undefined for a uniform by node, a uniform array or a slot name, on every adapter", () => {
    const tint = uniform("vec3");
    const items = uniformArray("float", 2);
    for (const adapter of [glsl, wgsl, wgslCompute, wgslContext, wasmContext, js, wasm, jsCompute, wasmCompute, cpu]) {
      adapter.setUniform(tint, null);
      adapter.setUniform(tint, undefined);
      adapter.setUniform(items, null);
      adapter.setUniform(items, undefined);
    }
    for (const adapter of [glsl, wgsl, wgslCompute, js, wasm, jsCompute, wasmCompute, cpu]) {
      adapter.setUniform(tint.name, null);
      adapter.setUniform(tint.name, undefined);
    }
    const routine = createJsRoutine(tint);
    routine.setUniform(tint, null);
    routine.setUniform(items, undefined);
    routine.setUniform(tint.name, null);
  });
});

describe("the parameters run takes", () => {
  /**
   * @canon spec-the-types-hold-a-routine-parameter-to-its-shape
   */
  it("takes each parameter named where the routine was made in the shape of its type", () => {
    const params = [
      { name: "p", type: "vec2" },
      { name: "n", type: "int" },
    ] as const;
    for (const routine of [createJsRoutine(float(1), { params }), createWasmRoutine(float(1), { params })]) {
      routine.run({ p: [1, 2], n: 3 });
      routine.run({ p: Float32Array.of(1, 2) });
      routine.run({ p: null });
      routine.run();
      // @ts-expect-error a vec2 takes two components
      routine.run({ p: [1, 2, 3] });
      // @ts-expect-error an int takes one value
      routine.run({ n: [1] });
      // @ts-expect-error the routine has no parameter q
      routine.run({ q: 1 });
    }
  });

  /**
   * @canon spec-the-types-hold-a-routine-parameter-to-its-shape
   */
  it("takes any value a parameter can hold, a typed array included, when the names are not known", () => {
    const params: { name: string; type: "vec2" | "float" }[] = [{ name: "p", type: "vec2" }];
    const routine = createJsRoutine(float(1), { params });
    routine.run({ p: Float32Array.of(1, 2) });
    routine.run({ p: [1, 2], q: 1, flag: true });
    // @ts-expect-error a parameter holds numbers
    routine.run({ p: "1" });
  });
});
