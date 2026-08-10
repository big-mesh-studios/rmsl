/**
 * Type-level tests.
 *
 * These catch mismatches between a signature's return type and the node's
 * actual type, which are invisible to a runtime assertion.
 *
 * Run by `pnpm test:types`, and also checked by `pnpm type-check`.
 */

import { describe, it, expectTypeOf } from "vitest";
import {
  Fn, float, vec2, vec3, vec4, int, uint, boolean, ivec2, ivec3, ivec4,
  uvec2, uvec3, uvec4, uniform, mat3, mat4,
  compileGLSL, compileWGSL, type Node,
} from "./rmsl";

describe("comparison result types", () => {
  // Only a scalar reduces to a single boolean; a comparison is component-wise,
  // so a vector yields one boolean per component.
  it("reduces to bool for scalars and to a boolean vector otherwise", () => {
    expectTypeOf(float(1).lessThan(float(2))).toEqualTypeOf<Node<"bool">>();
    expectTypeOf(vec2(1, 2).lessThan(vec2(3, 4))).toEqualTypeOf<Node<"bvec2">>();
    expectTypeOf(vec3(1, 2, 3).lessThan(vec3(4, 5, 6))).toEqualTypeOf<Node<"bvec3">>();
    expectTypeOf(vec4(1, 2, 3, 4).lessThan(vec4(5, 6, 7, 8))).toEqualTypeOf<Node<"bvec4">>();
    expectTypeOf(int(1).lessThan(int(2))).toEqualTypeOf<Node<"bool">>();
  });

  // A vector against a scalar broadcasts, which is what the caller means.
  it("allows a vector compared against a scalar", () => {
    expectTypeOf(vec3(1, 2, 3).lessThan(uniform("float"))).toEqualTypeOf<Node<"bvec3">>();
    expectTypeOf(vec3(1, 2, 3).greaterThan(0.5)).toEqualTypeOf<Node<"bvec3">>();
  });

  // The reverse has no single answer. The compiler widens the scalar and
  // produces a boolean vector, so a signature promising `bool` would be lying.
  it("rejects a scalar compared against a vector", () => {
    // @ts-expect-error a float has no single comparison against a vec3
    float(1).lessThan(vec3(1, 2, 3));
    // @ts-expect-error a float has no single comparison against a vec4
    uniform("float").greaterThanEqual(vec4(1, 2, 3, 4));
  });
});

describe("boolean vector reduction", () => {
  it("reduces to a single bool, and negates component-wise", () => {
    const compared = vec3(1, 2, 3).lessThan(vec3(4, 5, 6));
    expectTypeOf(compared.all()).toEqualTypeOf<Node<"bool">>();
    expectTypeOf(compared.any()).toEqualTypeOf<Node<"bool">>();
    expectTypeOf(compared.not()).toEqualTypeOf<Node<"bvec3">>();
  });
});

describe("operations whose result is not their operand's type", () => {
  // These reduce a vector to a scalar.
  it("types length, dot and distance as float", () => {
    expectTypeOf(vec3(1, 2, 3).length()).toEqualTypeOf<Node<"float">>();
    expectTypeOf(vec3(1, 2, 3).dot(vec3(4, 5, 6))).toEqualTypeOf<Node<"float">>();
    expectTypeOf(vec2(3, 4).distance(vec2(0, 0))).toEqualTypeOf<Node<"float">>();
  });

  // A matrix column, not a matrix.
  it("types a matrix element as the column vector it is", () => {
    expectTypeOf(uniform("mat4").element(0)).toEqualTypeOf<Node<"vec4">>();
    expectTypeOf(uniform("mat3").element(0)).toEqualTypeOf<Node<"vec3">>();
  });
});

describe("operations whose value operand is not the first", () => {
  // GLSL takes the value last in step(edge, x), so the result follows the
  // value rather than the edge — vec3.step(0.5) is a vec3, not a float.
  it("types step and smoothstep from the value", () => {
    expectTypeOf(vec3(1, 2, 3).step(0.5)).toEqualTypeOf<Node<"vec3">>();
    expectTypeOf(vec3(1, 2, 3).smoothstep(0, 1)).toEqualTypeOf<Node<"vec3">>();
    expectTypeOf(float(1).step(0.5)).toEqualTypeOf<Node<"float">>();
  });
});

describe("what a vertex stage accepts", () => {
  // Its result becomes the position, so anything that cannot be one is refused
  // where it is written rather than when the compiler runs.
  it("takes a vec4 result", () => {
    expectTypeOf(compileGLSL.vertex(Fn(() => vec4(1, 2, 3, 4).toVar())()))
      .toEqualTypeOf<string>();
  });

  // The other way to satisfy it: assign the position and return nothing. A
  // body that returns nothing has type void, which is why void is admitted.
  it("takes a program that returns nothing", () => {
    expectTypeOf(compileWGSL.vertex(Fn(() => { vec4(1, 2, 3, 4).toVar(); })()))
      .toEqualTypeOf<string>();
  });

  // Several values can be returned at once, and the last becomes the position.
  // The values before it are whatever the shader needed on the way there.
  it("takes several values, of which the last is the position", () => {
    expectTypeOf(compileGLSL.vertex(Fn(() => [
      float(1).toVar(),
      vec4(0, 0, 0, 1).toVar(),
    ])())).toEqualTypeOf<string>();
  });

  it("refuses several values that do not end in a position", () => {
    // @ts-expect-error the last of these is a float
    compileGLSL.vertex(Fn(() => [float(1).toVar(), float(2).toVar()])());
    // @ts-expect-error a position has to come last, not first
    compileWGSL.vertex(Fn(() => [vec4(0, 0, 0, 1).toVar(), float(1).toVar()])());
  });

  it("refuses a result that cannot become a position", () => {
    // @ts-expect-error a vec3 is not a position
    compileGLSL.vertex(Fn(() => vec3(1, 2, 3).toVar())());
    // @ts-expect-error a float is not a position
    compileWGSL.vertex(Fn(() => float(1).toVar())());
  });

  // A fragment stage has no such requirement: a shader with no colour output is
  // legal, so any result is allowed through.
  it("puts no such requirement on a fragment stage", () => {
    expectTypeOf(compileGLSL.fragment(Fn(() => float(1).toVar())()))
      .toEqualTypeOf<string>();
  });
});

describe("matrix operations", () => {
  // Every matrix type carries these at runtime, but only the two square ones
  // were declared to, so the rest had to be reached through a cast — which
  // switches off checking for the whole expression rather than just the method.
  it("gives every matrix type the operations the compiler implements", () => {
    expectTypeOf(uniform("mat2").transpose()).toEqualTypeOf<Node<"mat2">>();
    expectTypeOf(uniform("mat2").inverse()).toEqualTypeOf<Node<"mat2">>();
    expectTypeOf(uniform("mat4").inverse()).toEqualTypeOf<Node<"mat4">>();
  });

  // A matCxR has C columns of R rows, so one of its columns is a vecR.
  it("types a column by the matrix's row count", () => {
    expectTypeOf(uniform("mat2").element(0)).toEqualTypeOf<Node<"vec2">>();
    expectTypeOf(uniform("mat2x3").element(0)).toEqualTypeOf<Node<"vec3">>();
    expectTypeOf(uniform("mat3x2").element(0)).toEqualTypeOf<Node<"vec2">>();
    expectTypeOf(uniform("mat4x3").element(0)).toEqualTypeOf<Node<"vec3">>();
  });

  // Transposing swaps the two, so a matCxR becomes a matRxC.
  it("swaps the shape when transposing a non-square matrix", () => {
    expectTypeOf(uniform("mat2x3").transpose()).toEqualTypeOf<Node<"mat3x2">>();
    expectTypeOf(uniform("mat4x2").transpose()).toEqualTypeOf<Node<"mat2x4">>();
  });

  // Multiplying takes one component per column and gives one per row.
  it("types a matrix times a vector by the matrix's shape", () => {
    expectTypeOf(uniform("mat2x3").mult(vec2(1, 2))).toEqualTypeOf<Node<"vec3">>();
    expectTypeOf(uniform("mat3x2").mult(vec3(1, 2, 3))).toEqualTypeOf<Node<"vec2">>();
    expectTypeOf(uniform("mat4").mult(vec4(1, 2, 3, 4))).toEqualTypeOf<Node<"vec4">>();
  });

  // Only a square matrix has an inverse, and the compiler refuses the rest.
  it("offers no inverse on a non-square matrix", () => {
    // @ts-expect-error a matrix that is not square cannot be inverted
    uniform("mat2x3").inverse();
  });
});

describe("declared variables", () => {
  // A uniform carries its type's operations directly, alongside its name.
  it("carries both a name and the operations of its type", () => {
    expectTypeOf(uniform("vec3").name).toEqualTypeOf<string>();
    expectTypeOf(uniform("vec3").x).toEqualTypeOf<Node<"float">>();
    expectTypeOf(uniform("vec3").normalize()).toEqualTypeOf<Node<"vec3">>();
    expectTypeOf(uniform("mat4").mult(vec4(1, 2, 3, 4))).toEqualTypeOf<Node<"vec4">>();
  });
});

describe("texture sampling", () => {
  // A sampler samples at a coordinate of its own dimension and always returns
  // a vec4. sampler3D takes a volume coordinate, like a cube map.
  it("types a sampler3D's sample as a vec4", () => {
    expectTypeOf(uniform("sampler3D").texture(vec3(1, 2, 3))).toEqualTypeOf<Node<"vec4">>();
    expectTypeOf(uniform("sampler3D").textureLod(vec3(1, 2, 3), float(0))).toEqualTypeOf<Node<"vec4">>();
    expectTypeOf(uniform("sampler2D").texture(vec2(1, 2))).toEqualTypeOf<Node<"vec4">>();
  });
});

describe("integer vectors", () => {
  it("constructs with the right type", () => {
    expectTypeOf(ivec2(1, 2)).toEqualTypeOf<Node<"ivec2">>();
    expectTypeOf(ivec3(1, 2, 3)).toEqualTypeOf<Node<"ivec3">>();
    expectTypeOf(ivec4(1, 2, 3, 4)).toEqualTypeOf<Node<"ivec4">>();
    expectTypeOf(uvec2(1, 2)).toEqualTypeOf<Node<"uvec2">>();
    expectTypeOf(uvec3(1, 2, 3)).toEqualTypeOf<Node<"uvec3">>();
    expectTypeOf(uvec4(1, 2, 3, 4)).toEqualTypeOf<Node<"uvec4">>();
  });

  // Comparisons stay component-wise, so an integer vector yields a boolean
  // vector like a float one does.
  it("types comparisons as boolean vectors", () => {
    expectTypeOf(ivec3(1, 2, 3).lessThan(ivec3(4, 5, 6))).toEqualTypeOf<Node<"bvec3">>();
    expectTypeOf(uvec4(1, 2, 3, 4).equal(uvec4(1, 2, 3, 4))).toEqualTypeOf<Node<"bvec4">>();
    expectTypeOf(ivec2(1, 2).greaterThan(0)).toEqualTypeOf<Node<"bvec2">>();
  });

  // A single component of an integer vector is that integer scalar, not a float.
  it("types swizzle components as the integer scalar", () => {
    expectTypeOf(ivec3(1, 2, 3).x).toEqualTypeOf<Node<"int">>();
    expectTypeOf(ivec4(1, 2, 3, 4).xy).toEqualTypeOf<Node<"ivec2">>();
    expectTypeOf(ivec4(1, 2, 3, 4).rgb).toEqualTypeOf<Node<"ivec3">>();
    expectTypeOf(uvec3(1, 2, 3).z).toEqualTypeOf<Node<"uint">>();
    expectTypeOf(uvec4(1, 2, 3, 4).xy).toEqualTypeOf<Node<"uvec2">>();
  });

  it("types element() as the integer scalar", () => {
    expectTypeOf(ivec3(1, 2, 3).element(0)).toEqualTypeOf<Node<"int">>();
    expectTypeOf(uvec3(1, 2, 3).element(0)).toEqualTypeOf<Node<"uint">>();
    expectTypeOf(vec3(1, 2, 3).element(0)).toEqualTypeOf<Node<"float">>();
  });

  it("types integer arithmetic as integer vectors", () => {
    expectTypeOf(ivec2(1, 2).add(ivec2(3, 4))).toEqualTypeOf<Node<"ivec2">>();
    expectTypeOf(ivec2(1, 2).add(3)).toEqualTypeOf<Node<"ivec2">>();
    expectTypeOf(uvec3(1, 2, 3).bitAnd(uvec3(1, 2, 3))).toEqualTypeOf<Node<"uvec3">>();
  });
});

describe("integer samplers", () => {
  // Integer textures are not filterable, so sampling returns the signed or
  // unsigned integer vector the fetch produces rather than a float vec4.
  it("types isampler and usampler samples as integer vectors", () => {
    expectTypeOf(uniform("isampler2D").texture(ivec2(1, 2))).toEqualTypeOf<Node<"ivec4">>();
    expectTypeOf(uniform("isampler3D").texture(ivec3(1, 2, 3))).toEqualTypeOf<Node<"ivec4">>();
    expectTypeOf(uniform("isamplerCube").textureLod(ivec3(1, 2, 3), int(0))).toEqualTypeOf<Node<"ivec4">>();
    expectTypeOf(uniform("usampler2D").texture(uvec2(1, 2))).toEqualTypeOf<Node<"uvec4">>();
    expectTypeOf(uniform("usampler3D").texture(uvec3(1, 2, 3))).toEqualTypeOf<Node<"uvec4">>();
  });
});

describe("casts and conversions", () => {
  it("types the scalar constructors", () => {
    expectTypeOf(uint(5)).toEqualTypeOf<Node<"uint">>();
    expectTypeOf(uint(float(2.5))).toEqualTypeOf<Node<"uint">>();
    expectTypeOf(int(float(2.5))).toEqualTypeOf<Node<"int">>();
    expectTypeOf(boolean(int(1))).toEqualTypeOf<Node<"bool">>();
  });

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
