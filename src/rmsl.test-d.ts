import { describe, it, expectTypeOf } from "vitest";
import {
  Fn,
  float,
  vec2,
  vec3,
  vec4,
  int,
  uint,
  bool,
  ivec2,
  builtinFragDepth,
  builtinPosition,
  ivec3,
  ivec4,
  uvec2,
  uvec3,
  uvec4,
  uniform,
  mat3,
  mat4,
  mul,
  dot,
  length,
  distance,
  all,
  any,
  determinant,
  instancedArray,
  invocationIndex,
  attribute,
  varying,
  type Node,
  type ShaderType,
  type Var,
  type VaryingNode,
} from "./rmsl";
import { compileGlsl, type GlslAdapter } from "./glsl";
import { compileWgsl, createWgslCompute } from "./wgsl";
import { createJsCompute } from "./js";
import { createWasmCompute } from "./wasm";

describe("comparison result types", () => {
  /**
   * Only a scalar reduces to a single boolean; a comparison is component-wise,
   * so a vector yields one boolean per component.
   *
   * @canon spec-a-scalar-comparison-gives-a-bool
   * @canon spec-a-vector-comparison-gives-a-boolean-vector
   */
  it("reduces to bool for scalars and to a boolean vector otherwise", () => {
    expectTypeOf(float(1).lessThan(float(2))).toEqualTypeOf<Node<"bool">>();
    expectTypeOf(vec2(1, 2).lessThan(vec2(3, 4))).toEqualTypeOf<Node<"bvec2">>();
    expectTypeOf(vec3(1, 2, 3).lessThan(vec3(4, 5, 6))).toEqualTypeOf<Node<"bvec3">>();
    expectTypeOf(vec4(1, 2, 3, 4).lessThan(vec4(5, 6, 7, 8))).toEqualTypeOf<Node<"bvec4">>();
    expectTypeOf(int(1).lessThan(int(2))).toEqualTypeOf<Node<"bool">>();
  });

  /**
   * A vector against a scalar broadcasts, which is what the caller means.
   *
   * @canon spec-a-scalar-compared-against-a-vector-is-broadcast
   */
  it("allows a vector compared against a scalar", () => {
    expectTypeOf(vec3(1, 2, 3).lessThan(uniform("float"))).toEqualTypeOf<Node<"bvec3">>();
    expectTypeOf(vec3(1, 2, 3).greaterThan(0.5)).toEqualTypeOf<Node<"bvec3">>();
  });

  /**
   * The compiler widens the scalar, so the result is a boolean vector of the
   * vector's width, whichever side the scalar is on.
   *
   * @canon spec-a-scalar-compared-against-a-vector-is-broadcast
   */
  it("allows a scalar compared against a vector, with a boolean vector result", () => {
    expectTypeOf(float(1).lessThan(vec3(1, 2, 3))).toEqualTypeOf<Node<"bvec3">>();
    expectTypeOf(uniform("float").greaterThanEqual(vec4(1, 2, 3, 4))).toEqualTypeOf<Node<"bvec4">>();
    expectTypeOf(float(1).equal(2)).toEqualTypeOf<Node<"bool">>();
  });
});

describe("boolean vector reduction", () => {
  /**
   * @canon spec-a-boolean-vector-reduces-with-all-or-any
   * @canon spec-not-negates-a-boolean-vector-component-wise
   */
  it("reduces to a single bool, and negates component-wise", () => {
    const compared = vec3(1, 2, 3).lessThan(vec3(4, 5, 6));
    expectTypeOf(compared.all()).toEqualTypeOf<Node<"bool">>();
    expectTypeOf(compared.any()).toEqualTypeOf<Node<"bool">>();
    expectTypeOf(compared.not()).toEqualTypeOf<Node<"bvec3">>();
  });
});

describe("operations whose result is not their operand's type", () => {
  /**
   * These reduce a vector to a scalar.
   *
   * @canon spec-a-reducing-operation-has-a-scalar-type
   */
  it("types length, dot and distance as float", () => {
    expectTypeOf(vec3(1, 2, 3).length()).toEqualTypeOf<Node<"float">>();
    expectTypeOf(vec3(1, 2, 3).dot(vec3(4, 5, 6))).toEqualTypeOf<Node<"float">>();
    expectTypeOf(vec2(3, 4).distance(vec2(0, 0))).toEqualTypeOf<Node<"float">>();
  });

  /**
   * A matrix column, not a matrix.
   *
   * @canon spec-a-matrix-element-is-a-column
   */
  it("types a matrix element as the column vector it is", () => {
    expectTypeOf(uniform("mat4").element(0)).toEqualTypeOf<Node<"vec4">>();
    expectTypeOf(uniform("mat3").element(0)).toEqualTypeOf<Node<"vec3">>();
  });
});

describe("operations whose value operand is not the first", () => {
  /**
   * GLSL takes the value last in step(edge, x), so the result follows the
   * value rather than the edge — vec3.step(0.5) is a vec3, not a float.
   *
   * @canon spec-step-and-smoothstep-take-their-type-from-the-value
   */
  it("types step and smoothstep from the value", () => {
    expectTypeOf(vec3(1, 2, 3).step(0.5)).toEqualTypeOf<Node<"vec3">>();
    expectTypeOf(vec3(1, 2, 3).smoothstep(0, 1)).toEqualTypeOf<Node<"vec3">>();
    expectTypeOf(float(1).step(0.5)).toEqualTypeOf<Node<"float">>();
  });
});

describe("what a vertex stage accepts", () => {
  /**
   * Its result becomes the position, so anything that cannot be one is refused
   * where it is written rather than when the compiler runs.
   *
   * @canon spec-a-vertex-stage-writes-its-position
   */
  it("takes a vec4 result", () => {
    expectTypeOf(compileGlsl.vertex(Fn(() => vec4(1, 2, 3, 4).toVar())())).toEqualTypeOf<string>();
  });

  /**
   * The other way to satisfy it: assign the position and return nothing. A
   * body that returns nothing has type void, which is why void is admitted.
   *
   * @canon spec-a-vertex-stage-may-return-several-values-ending-in-its-position
   */
  it("takes a program that returns nothing", () => {
    expectTypeOf(
      compileWgsl.vertex(
        Fn(() => {
          vec4(1, 2, 3, 4).toVar();
        })(),
      ),
    ).toEqualTypeOf<string>();
  });

  /**
   * A body with no return, and one that assigns the position but likewise
   * returns nothing, both give the call itself — not just what a compiler
   * accepts — the type Node<"void">: the call still produces a node holding
   * the body's statements.
   *
   * @canon spec-an-fn-has-the-type-of-what-it-returns
   */
  it("types a program that returns nothing as a void node", () => {
    expectTypeOf(
      Fn(() => {
        float(1).toVar();
      })(),
    ).toEqualTypeOf<Node<"void">>();

    expectTypeOf(
      Fn(() => {
        builtinPosition().assign(vec4(1, 2, 3, 4));
      })(),
    ).toEqualTypeOf<Node<"void">>();
  });

  /**
   * A body typed `any` (a helper declared to return `any`, say) keeps that
   * type rather than being taken for a body that returns nothing.
   *
   * @canon spec-an-fn-has-the-type-of-what-it-returns
   */
  it("leaves a body typed any as any", () => {
    expectTypeOf(Fn(() => vec4(0).toVar() as any)()).toBeAny();
  });

  /**
   * Several values can be returned at once, and the last becomes the position.
   * The values before it are whatever the shader needed on the way there.
   *
   * @canon spec-a-vertex-stage-may-return-several-values-ending-in-its-position
   */
  it("takes several values, of which the last is the position", () => {
    expectTypeOf(compileGlsl.vertex(Fn(() => [float(1).toVar(), vec4(0, 0, 0, 1).toVar()])())).toEqualTypeOf<string>();
  });

  /**
   * @canon spec-a-vertex-stage-without-a-position-is-refused
   */
  it("refuses several values that do not end in a position", () => {
    // @ts-expect-error the last of these is a float
    compileGlsl.vertex(Fn(() => [float(1).toVar(), float(2).toVar()])());
    // @ts-expect-error a position has to come last, not first
    compileWgsl.vertex(Fn(() => [vec4(0, 0, 0, 1).toVar(), float(1).toVar()])());
  });

  /**
   * @canon spec-a-vertex-stage-without-a-position-is-refused
   */
  it("refuses a result that cannot become a position", () => {
    // @ts-expect-error a vec3 is not a position
    compileGlsl.vertex(Fn(() => vec3(1, 2, 3).toVar())());
    // @ts-expect-error a float is not a position
    compileWgsl.vertex(Fn(() => float(1).toVar())());
  });

  /**
   * A fragment stage has no such requirement: a shader with no colour output is
   * legal, so any result is allowed through.
   *
   * @canon spec-a-vertex-stage-may-return-several-values-ending-in-its-position
   */
  it("puts no such requirement on a fragment stage", () => {
    expectTypeOf(compileGlsl.fragment(Fn(() => float(1).toVar())())).toEqualTypeOf<string>();
  });
});

describe("compute programs", () => {
  /**
   * A compute program writes its results into storage, so it has nothing to
   * return; every compute entry point takes it as it is.
   *
   * @canon spec-a-compute-program-returns-nothing
   */
  it("take a program that returns nothing", () => {
    const step = Fn(() => {
      const pos = instancedArray(1, "float");
      pos.element(invocationIndex()).addAssign(1);
    })();
    createJsCompute(step);
    createWasmCompute(step);
    createWgslCompute(step);
  });
});

describe("adapters", () => {
  /**
   * @canon spec-a-glsl-adapter-attaches-and-draws-synchronously
   */
  it("attach and draw a GLSL adapter without a promise", () => {
    expectTypeOf<ReturnType<GlslAdapter["attach"]>>().toEqualTypeOf<void>();
    expectTypeOf<ReturnType<GlslAdapter["draw"]>>().toEqualTypeOf<void>();
  });

  /**
   * @canon spec-an-adapter-has-no-method-for-a-capability-its-target-lacks
   */
  it("give a GLSL adapter no compute to call", () => {
    const adapter = {} as GlslAdapter;
    // @ts-expect-error WebGL 2 has no compute stage
    adapter.compute();
  });
});

describe("compileGlsl precision options", () => {
  /**
   * The options mirror three.js's `precision` setting (`"highp" | "mediump" |
   * "lowp"`); every compileGlsl call takes them, and a value outside the union
   * is refused at the type level.
   *
   * @canon spec-every-glsl-call-shape-takes-a-precision
   */
  it("accepts a precision option on every call shape", () => {
    const root = Fn(() => vec4(1, 2, 3, 4).toVar())();
    expectTypeOf(compileGlsl(root, { precision: "mediump" })).toEqualTypeOf<string>();
    expectTypeOf(compileGlsl.fragment(root, { precision: "lowp" })).toEqualTypeOf<string>();
    expectTypeOf(compileGlsl.vertex(root, { precision: "highp" })).toEqualTypeOf<string>();
    expectTypeOf(compileGlsl(root)).toEqualTypeOf<string>();
  });

  /**
   * @canon spec-glsl-refuses-an-unknown-precision
   */
  it("refuses an unknown precision value", () => {
    const root = Fn(() => vec4(1, 2, 3, 4).toVar())();
    // @ts-expect-error "high" is not a precision
    compileGlsl(root, { precision: "high" });
    // @ts-expect-error a number is not a precision
    compileGlsl.vertex(root, { precision: 1 });
  });
});

describe("matrix operations", () => {
  /**
   * Every matrix type carries these at runtime, but only the two square ones
   * were declared to, so the rest had to be reached through a cast — which
   * switches off checking for the whole expression rather than just the method.
   *
   * @canon spec-every-matrix-type-has-the-operations-the-compiler-implements
   */
  it("gives every matrix type the operations the compiler implements", () => {
    expectTypeOf(uniform("mat2").transpose()).toEqualTypeOf<Node<"mat2">>();
    expectTypeOf(uniform("mat2").inverse()).toEqualTypeOf<Node<"mat2">>();
    expectTypeOf(uniform("mat4").inverse()).toEqualTypeOf<Node<"mat4">>();
  });

  /**
   * A matCxR has C columns of R rows, so one of its columns is a vecR.
   *
   * @canon spec-a-matrix-element-is-a-column
   */
  it("types a column by the matrix's row count", () => {
    expectTypeOf(uniform("mat2").element(0)).toEqualTypeOf<Node<"vec2">>();
    expectTypeOf(uniform("mat2x3").element(0)).toEqualTypeOf<Node<"vec3">>();
    expectTypeOf(uniform("mat3x2").element(0)).toEqualTypeOf<Node<"vec2">>();
    expectTypeOf(uniform("mat4x3").element(0)).toEqualTypeOf<Node<"vec3">>();
  });

  /**
   * Transposing swaps the two, so a matCxR becomes a matRxC.
   *
   * @canon spec-a-transpose-swaps-the-shape
   */
  it("swaps the shape when transposing a non-square matrix", () => {
    expectTypeOf(uniform("mat2x3").transpose()).toEqualTypeOf<Node<"mat3x2">>();
    expectTypeOf(uniform("mat4x2").transpose()).toEqualTypeOf<Node<"mat2x4">>();
  });

  /**
   * Multiplying takes one component per column and gives one per row.
   *
   * @canon spec-a-matrix-product-has-the-shape-of-the-product
   */
  it("types a matrix times a vector by the matrix's shape", () => {
    expectTypeOf(uniform("mat2x3").mul(vec2(1, 2))).toEqualTypeOf<Node<"vec3">>();
    expectTypeOf(uniform("mat3x2").mul(vec3(1, 2, 3))).toEqualTypeOf<Node<"vec2">>();
    expectTypeOf(uniform("mat4").mul(vec4(1, 2, 3, 4))).toEqualTypeOf<Node<"vec4">>();
  });

  /**
   * Only a square matrix has an inverse, and the compiler refuses the rest.
   *
   * @canon spec-only-a-square-matrix-is-inverted
   */
  it("offers no inverse on a non-square matrix", () => {
    // @ts-expect-error a matrix that is not square cannot be inverted
    uniform("mat2x3").inverse();
  });
});

describe("declared variables", () => {
  /**
   * A uniform carries its type's operations directly, alongside its name.
   *
   * @canon spec-a-variable-is-declared-under-the-name-given
   */
  it("carries both a name and the operations of its type", () => {
    expectTypeOf(uniform("vec3").name).toEqualTypeOf<string>();
    expectTypeOf(uniform("vec3").x).toEqualTypeOf<Node<"float">>();
    expectTypeOf(uniform("vec3").normalize()).toEqualTypeOf<Node<"vec3">>();
    expectTypeOf(uniform("mat4").mul(vec4(1, 2, 3, 4))).toEqualTypeOf<Node<"vec4">>();
  });
});

describe("texture sampling", () => {
  /**
   * A sampler samples at a coordinate of its own dimension and always returns
   * a vec4. sampler3D takes a volume coordinate, like a cube map.
   *
   * @canon spec-a-float-texture-is-sampled-through-a-sampler
   */
  it("types a sampler3D's sample as a vec4", () => {
    expectTypeOf(uniform("sampler3D").texture(vec3(1, 2, 3))).toEqualTypeOf<Node<"vec4">>();
    expectTypeOf(uniform("sampler3D").textureLod(vec3(1, 2, 3), float(0))).toEqualTypeOf<Node<"vec4">>();
    expectTypeOf(uniform("sampler2D").texture(vec2(1, 2))).toEqualTypeOf<Node<"vec4">>();
  });
});

describe("integer vectors", () => {
  /**
   * @canon spec-a-literal-compiles-to-a-literal-of-its-type
   */
  it("constructs with the right type", () => {
    expectTypeOf(ivec2(1, 2)).toEqualTypeOf<Node<"ivec2">>();
    expectTypeOf(ivec3(1, 2, 3)).toEqualTypeOf<Node<"ivec3">>();
    expectTypeOf(ivec4(1, 2, 3, 4)).toEqualTypeOf<Node<"ivec4">>();
    expectTypeOf(uvec2(1, 2)).toEqualTypeOf<Node<"uvec2">>();
    expectTypeOf(uvec3(1, 2, 3)).toEqualTypeOf<Node<"uvec3">>();
    expectTypeOf(uvec4(1, 2, 3, 4)).toEqualTypeOf<Node<"uvec4">>();
  });

  /**
   * Comparisons stay component-wise, so an integer vector yields a boolean
   * vector like a float one does.
   *
   * @canon spec-a-vector-comparison-gives-a-boolean-vector
   */
  it("types comparisons as boolean vectors", () => {
    expectTypeOf(ivec3(1, 2, 3).lessThan(ivec3(4, 5, 6))).toEqualTypeOf<Node<"bvec3">>();
    expectTypeOf(uvec4(1, 2, 3, 4).equal(uvec4(1, 2, 3, 4))).toEqualTypeOf<Node<"bvec4">>();
    expectTypeOf(ivec2(1, 2).greaterThan(0)).toEqualTypeOf<Node<"bvec2">>();
  });

  /**
   * A single component of an integer vector is that integer scalar, not a float.
   *
   * @canon spec-a-swizzle-reads-the-components-it-names
   */
  it("types swizzle components as the integer scalar", () => {
    expectTypeOf(ivec3(1, 2, 3).x).toEqualTypeOf<Node<"int">>();
    expectTypeOf(ivec4(1, 2, 3, 4).xy).toEqualTypeOf<Node<"ivec2">>();
    expectTypeOf(ivec4(1, 2, 3, 4).rgb).toEqualTypeOf<Node<"ivec3">>();
    expectTypeOf(uvec3(1, 2, 3).z).toEqualTypeOf<Node<"uint">>();
    expectTypeOf(uvec4(1, 2, 3, 4).xy).toEqualTypeOf<Node<"uvec2">>();
  });

  /**
   * @canon spec-an-element-reads-a-component-by-index
   */
  it("types element() as the integer scalar", () => {
    expectTypeOf(ivec3(1, 2, 3).element(0)).toEqualTypeOf<Node<"int">>();
    expectTypeOf(uvec3(1, 2, 3).element(0)).toEqualTypeOf<Node<"uint">>();
    expectTypeOf(vec3(1, 2, 3).element(0)).toEqualTypeOf<Node<"float">>();
  });

  /**
   * @canon spec-arithmetic-compiles-to-the-operators-of-the-target
   */
  it("types integer arithmetic as integer vectors", () => {
    expectTypeOf(ivec2(1, 2).add(ivec2(3, 4))).toEqualTypeOf<Node<"ivec2">>();
    expectTypeOf(ivec2(1, 2).add(3)).toEqualTypeOf<Node<"ivec2">>();
    expectTypeOf(uvec3(1, 2, 3).bitAnd(uvec3(1, 2, 3))).toEqualTypeOf<Node<"uvec3">>();
  });
});

describe("integer samplers", () => {
  /**
   * Integer textures are not filterable, so sampling returns the signed or
   * unsigned integer vector the fetch produces rather than a float vec4.
   *
   * @canon spec-an-integer-texture-is-fetched-unfiltered
   */
  it("types isampler and usampler samples as integer vectors", () => {
    expectTypeOf(uniform("isampler2D").texture(ivec2(1, 2))).toEqualTypeOf<Node<"ivec4">>();
    expectTypeOf(uniform("isampler3D").texture(ivec3(1, 2, 3))).toEqualTypeOf<Node<"ivec4">>();
    expectTypeOf(uniform("isamplerCube").textureLod(ivec3(1, 2, 3), int(0))).toEqualTypeOf<Node<"ivec4">>();
    expectTypeOf(uniform("usampler2D").texture(uvec2(1, 2))).toEqualTypeOf<Node<"uvec4">>();
    expectTypeOf(uniform("usampler3D").texture(uvec3(1, 2, 3))).toEqualTypeOf<Node<"uvec4">>();
  });

  /**
   * @canon spec-an-integer-texture-is-fetched-unfiltered
   */
  it("refuses float coordinates and a mismatched vector width", () => {
    // Integer textures are fetched at integer texel coordinates — the
    // texelFetch/textureLoad both backends emit take an integer vector, so a
    // float one is rejected rather than silently truncated.
    // @ts-expect-error integer samplers fetch at integer texel coordinates
    uniform("isampler2D").texture(vec2(1, 2));
    // @ts-expect-error a 2D sampler takes an ivec2, not an ivec3
    uniform("isampler2D").texture(ivec3(1, 2, 3));
    // @ts-expect-error a 2D sampler takes a uvec2, not a uvec3
    uniform("usampler2D").texture(uvec3(1, 2, 3));
  });
});

describe("casts and conversions", () => {
  /**
   * @canon spec-a-conversion-between-numeric-types-is-written-out
   */
  it("types the scalar constructors", () => {
    expectTypeOf(uint(5)).toEqualTypeOf<Node<"uint">>();
    expectTypeOf(uint(float(2.5))).toEqualTypeOf<Node<"uint">>();
    expectTypeOf(int(float(2.5))).toEqualTypeOf<Node<"int">>();
    expectTypeOf(bool(int(1))).toEqualTypeOf<Node<"bool">>();
  });

  /**
   * @canon spec-a-conversion-between-numeric-types-is-written-out
   */
  it("types the chained conversions", () => {
    expectTypeOf(float(2.5).toInt()).toEqualTypeOf<Node<"int">>();
    expectTypeOf(int(2).toFloat()).toEqualTypeOf<Node<"float">>();
    expectTypeOf(float(2.5).toUint()).toEqualTypeOf<Node<"uint">>();
    expectTypeOf(int(1).toBool()).toEqualTypeOf<Node<"bool">>();
    expectTypeOf(ivec3(1, 2, 3).toVec3()).toEqualTypeOf<Node<"vec3">>();
    expectTypeOf(vec3(1, 2, 3).toIVec3()).toEqualTypeOf<Node<"ivec3">>();
    expectTypeOf(uvec4(1, 2, 3, 4).toVec4()).toEqualTypeOf<Node<"vec4">>();
    expectTypeOf(vec4(1, 2, 3, 4).toBVec4()).toEqualTypeOf<Node<"bvec4">>();
    expectTypeOf(float(1).toMat4()).toEqualTypeOf<Node<"mat4">>();
    expectTypeOf(float(1).convert("uvec3")).toEqualTypeOf<Node<"uvec3">>();
  });
});

describe("TSL free-function API", () => {
  /**
   * @canon spec-a-reducing-operation-has-a-scalar-type
   */
  it("types the reducing free functions", () => {
    expectTypeOf(dot(vec3(1, 2, 3), vec3(4, 5, 6))).toEqualTypeOf<Node<"float">>();
    expectTypeOf(length(vec3(1, 2, 3))).toEqualTypeOf<Node<"float">>();
    expectTypeOf(distance(vec2(0, 0), vec2(1, 1))).toEqualTypeOf<Node<"float">>();
    expectTypeOf(all(vec3(1, 2, 3).greaterThan(0))).toEqualTypeOf<Node<"bool">>();
    expectTypeOf(any(vec3(1, 2, 3).lessThan(0))).toEqualTypeOf<Node<"bool">>();
    expectTypeOf(determinant(uniform("mat4"))).toEqualTypeOf<Node<"float">>();
  });

  /**
   * @canon spec-a-javascript-array-is-a-vector-of-its-length
   */
  it("accepts raw numbers and arrays", () => {
    expectTypeOf(mul(2, 3)).toMatchTypeOf<Node<ShaderType>>();
    expectTypeOf(dot(vec3(1, 0, 0), [1, 2, 3])).toEqualTypeOf<Node<"float">>();
  });

  /**
   * @canon spec-a-matrix-product-has-the-shape-of-the-product
   */
  it("types the matrix multiply overloads", () => {
    expectTypeOf(uniform("mat4").mul(vec4(1, 2, 3, 4))).toEqualTypeOf<Node<"vec4">>();
    expectTypeOf(uniform("mat4").mul(vec3(1, 2, 3))).toEqualTypeOf<Node<"vec3">>();
    expectTypeOf(uniform("mat3").mul(vec2(1, 2))).toEqualTypeOf<Node<"vec2">>();
    expectTypeOf(uniform("mat4").mul(uniform("mat4"))).toEqualTypeOf<Node<"mat4">>();
  });

  /**
   * @canon spec-a-matrix-product-has-the-shape-of-the-product
   */
  it("types a matrix times a matrix by the column/row product", () => {
    expectTypeOf(uniform("mat2x3").mul(uniform("mat3x2"))).toEqualTypeOf<Node<"mat3">>();
    expectTypeOf(uniform("mat3x2").mul(uniform("mat2x3"))).toEqualTypeOf<Node<"mat2">>();
    expectTypeOf(uniform("mat2").mul(uniform("mat3x2"))).toEqualTypeOf<Node<"mat3x2">>();
    expectTypeOf(uniform("mat4").mul(uniform("mat2x4"))).toEqualTypeOf<Node<"mat2x4">>();
    expectTypeOf(uniform("mat2x4").mul(uniform("mat4x2"))).toEqualTypeOf<Node<"mat4">>();
    expectTypeOf(uniform("mat4").mul(uniform("mat4"))).toEqualTypeOf<Node<"mat4">>();
  });

  /**
   * @canon spec-a-matrix-product-whose-shapes-do-not-meet-is-refused
   */
  it("rejects a matrix product whose shapes do not meet", () => {
    expectTypeOf(uniform("mat2x3").mul(uniform("mat2x4"))).toEqualTypeOf<never>();
  });

  /**
   * @canon spec-a-swizzle-reads-the-components-it-names
   */
  it("types stpq swizzles by the source type", () => {
    expectTypeOf(vec4(1, 2, 3, 4).stpq).toEqualTypeOf<Node<"vec4">>();
    expectTypeOf(vec4(1, 2, 3, 4).st).toEqualTypeOf<Node<"vec2">>();
    expectTypeOf(vec3(1, 2, 3).p).toEqualTypeOf<Node<"float">>();
    expectTypeOf(ivec3(1, 2, 3).stp).toEqualTypeOf<Node<"ivec3">>();
    expectTypeOf(uvec2(1, 2).st).toEqualTypeOf<Node<"uvec2">>();
  });
});

describe("scalar-broadcast result types", () => {
  /**
   * `1 - vec3` (oneMinus) and `1 / vec3` (reciprocal) must stay vec3: the
   * operand defining the result is the widest, not the first. Before the fix
   * these declared float, which made the JS target multiply an array by a
   * scalar (NaN) and mis-typed any intermediate variable.
   *
   * @canon spec-an-operation-no-target-has-is-composed
   */
  it("types oneMinus and reciprocal from a vector receiver", () => {
    expectTypeOf(vec3(1, 2, 3).oneMinus()).toEqualTypeOf<Node<"vec3">>();
    expectTypeOf(vec3(1, 2, 3).reciprocal()).toEqualTypeOf<Node<"vec3">>();
    expectTypeOf(vec4(1, 2, 3, 4).oneMinus()).toEqualTypeOf<Node<"vec4">>();
    expectTypeOf(float(0.5).oneMinus()).toEqualTypeOf<Node<"float">>();
    expectTypeOf(float(0.5).reciprocal()).toEqualTypeOf<Node<"float">>();
  });

  /**
   * @canon spec-step-and-smoothstep-take-their-type-from-the-value
   */
  it("keeps a float step/smoothstep value operand type", () => {
    expectTypeOf(vec3(1, 2, 3).step(float(0.5))).toEqualTypeOf<Node<"vec3">>();
    expectTypeOf(vec3(1, 2, 3).smoothstep(float(0), float(1))).toEqualTypeOf<Node<"vec3">>();
  });
});

describe("what can be assigned to", () => {
  /**
   * @canon spec-a-var-can-be-assigned
   */
  it("offers assign on a variable, a stage output and a storage element", () => {
    expectTypeOf(vec4(1, 2, 3, 4).toVar()).toEqualTypeOf<Var<"vec4">>();
    expectTypeOf(vec3(1, 2, 3).var()).toEqualTypeOf<Var<"vec3">>();
    expectTypeOf(builtinPosition()).toEqualTypeOf<Var<"vec4">>();
    expectTypeOf(builtinFragDepth()).toEqualTypeOf<Var<"float">>();
    expectTypeOf(varying("vec2")).toEqualTypeOf<VaryingNode<"vec2">>();
    expectTypeOf(varying("vec2")).toMatchTypeOf<Var<"vec2">>();
    expectTypeOf(instancedArray(4, "vec4").element(invocationIndex())).toEqualTypeOf<Var<"vec4">>();
  });

  /**
   * @canon spec-a-var-can-be-assigned
   */
  it("offers it through a component, a column and a swizzle naming each component once", () => {
    const v = vec4(1, 2, 3, 4).toVar();
    expectTypeOf(v.x).toEqualTypeOf<Var<"float">>();
    expectTypeOf(v.element(int(2))).toEqualTypeOf<Var<"float">>();
    expectTypeOf(v.wzy).toHaveProperty("assign");
    expectTypeOf(v.ba).toHaveProperty("assign");
    expectTypeOf(v.yzx.xy).toHaveProperty("assign");
    const m = mat3(1, 2, 3, 4, 5, 6, 7, 8, 9).toVar();
    expectTypeOf(m.element(int(1))).toEqualTypeOf<Var<"vec3">>();
    expectTypeOf(m.element(int(1)).zx).toHaveProperty("assign");
  });

  /**
   * @canon spec-a-uniform-cannot-be-assigned
   * @canon spec-an-attribute-cannot-be-assigned
   * @canon spec-a-whole-storage-buffer-cannot-be-assigned
   * @canon spec-a-computed-value-cannot-be-assigned
   * @canon spec-a-swizzle-that-repeats-a-component-cannot-be-assigned
   */
  it("refuses a uniform, an attribute, a whole storage buffer, a computed value and a repeated swizzle", () => {
    const v = vec4(1, 2, 3, 4).toVar();
    // @ts-expect-error a uniform is read-only
    uniform("vec4").assign(vec4(0, 0, 0, 0));
    // @ts-expect-error an attribute is read-only
    attribute("vec3").x.assign(float(0));
    // @ts-expect-error the result of an operation is a value, not a variable
    v.add(1).x.assign(float(0));
    // @ts-expect-error a whole storage buffer can't be assigned, only its elements
    instancedArray(4, "vec4").assign(vec4(0, 0, 0, 0));
    // @ts-expect-error a literal is a value, not a variable
    float(1).assign(float(2));
    // @ts-expect-error a swizzle naming a component twice can't be written
    v.xx.assign(vec2(0, 0));
    // @ts-expect-error a component of a repeated swizzle can't be written either
    v.xxy.z.assign(float(0));
    const column = mat3(1, 2, 3, 4, 5, 6, 7, 8, 9).element(int(0));
    // @ts-expect-error neither can a column of a matrix that isn't a variable
    column.assign(vec3(0, 0, 0));
  });

  /**
   * @canon spec-a-swizzle-reads-the-components-it-names
   */
  it("reads the same swizzles from a variable as from any node", () => {
    const v = vec4(1, 2, 3, 4).toVar();
    expectTypeOf(v.xx).toEqualTypeOf<Node<"vec2">>();
    expectTypeOf(v.wzyx.add(1)).toEqualTypeOf<Node<"vec4">>();
  });

  /**
   * @canon spec-a-var-can-be-assigned
   */
  it("accepts a writable node wherever a node is expected", () => {
    const take = (n: Node<"vec4">) => n;
    take(vec4(1, 2, 3, 4).toVar());
    take(instancedArray(4, "vec4").element(int(0)));
  });

  /**
   * @canon spec-a-read-only-storage-element-cannot-be-assigned
   */
  it("makes a read-only storage node's elements read-only", () => {
    const values = instancedArray(4, "vec4").toReadOnly();
    expectTypeOf(values.element(int(0))).toEqualTypeOf<Node<"vec4">>();
    // @ts-expect-error an element of a read-only storage node can't be assigned
    values.element(int(0)).assign(vec4(0, 0, 0, 0));
  });

  /**
   * @canon spec-a-storage-node-is-read-write-until-to-read-only
   */
  it("types a storage node indexed by a number or an int as its element", () => {
    expectTypeOf(instancedArray(4, "vec4").element(0)).toEqualTypeOf<Var<"vec4">>();
    expectTypeOf(instancedArray(4, "vec4").element(int(0))).toEqualTypeOf<Var<"vec4">>();
    expectTypeOf(instancedArray(4, "mat3").element(0)).toEqualTypeOf<Var<"mat3">>();
  });
});

describe("the width of a float's arithmetic", () => {
  /**
   * @canon spec-an-arithmetic-result-has-the-width-of-the-wider-operand
   */
  it("gives a float beside a vector the vector's width", () => {
    expectTypeOf(float(2).mul(vec4(1, 1, 1, 1))).toEqualTypeOf<Node<"vec4">>();
    expectTypeOf(uniform("float").add(vec2(1, 2))).toEqualTypeOf<Node<"vec2">>();
    expectTypeOf(float(2).div(3)).toEqualTypeOf<Node<"float">>();
  });
});
