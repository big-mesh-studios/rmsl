/**
 * Type-level tests for the post-processing primitives and effects.
 *
 * Run by `pnpm test:types`, and also checked by `pnpm type-check`.
 */

import { describe, it, expectTypeOf } from "vitest";
import {
  float, vec2, vec3, vec4, uniform,
  select, luminance, rand, interleavedGradientNoise,
  textureSize, textureLoad, fragCoord, uv, screenCoordinate, time,
  type Node,
} from "../rmsl";
import { sepia, bleach, dotScreen, rgbShift, gaussianBlur, bloom, circle, type PassGraph, type HighPassFn } from "./index";

describe("effects primitives", () => {
  it("select follows the branch type", () => {
    // The .select method is generic over the branch type.
    expectTypeOf(boolNode().select(float(1), float(2))).toEqualTypeOf<Node<"float">>();
    expectTypeOf(boolNode().select(vec2(1, 2), vec2(3, 4))).toEqualTypeOf<Node<"vec2">>();
    expectTypeOf(boolNode().select(vec3(1), vec3(0))).toEqualTypeOf<Node<"vec3">>();
  });

  it("luminance reduces a colour to a float", () => {
    expectTypeOf(luminance(vec3(0.2, 0.4, 0.6))).toEqualTypeOf<Node<"float">>();
    expectTypeOf(luminance(vec4(0.2, 0.4, 0.6, 1))).toEqualTypeOf<Node<"float">>();
  });

  it("rand and interleavedGradientNoise return floats", () => {
    expectTypeOf(rand(vec2(1, 2))).toEqualTypeOf<Node<"float">>();
    expectTypeOf(interleavedGradientNoise(vec2(1, 2))).toEqualTypeOf<Node<"float">>();
  });

  it("textureSize is uvec2 for 2D and uvec3 for 3D samplers", () => {
    expectTypeOf(textureSize(uniform("sampler2D"))).toEqualTypeOf<Node<"uvec2">>();
    expectTypeOf(textureSize(uniform("sampler3D"))).toEqualTypeOf<Node<"uvec3">>();
  });

  it("textureLoad returns a vec4 for a float sampler", () => {
    expectTypeOf(textureLoad(uniform("sampler2D"), vec2(1, 2).toIVec2())).toEqualTypeOf<Node<"vec4">>();
    expectTypeOf(textureLoad(uniform("sampler3D"), vec3(1, 2, 3).toIVec3())).toEqualTypeOf<Node<"vec4">>();
  });

  it("screen-space accessors are vec2, time is a float uniform", () => {
    expectTypeOf(fragCoord()).toEqualTypeOf<Node<"vec2">>();
    expectTypeOf(screenCoordinate()).toEqualTypeOf<Node<"vec2">>();
    expectTypeOf(uv()).toEqualTypeOf<Node<"vec2">>();
    // time() returns a shared uniform node, which carries `.name` on top of the
    // float operations.
    expectTypeOf(time()).toMatchTypeOf<Node<"float">>();
  });
});

describe("effect signatures", () => {
  it("sepia and bleach take and return vec4", () => {
    expectTypeOf(sepia(vec4(1))).toEqualTypeOf<Node<"vec4">>();
    expectTypeOf(bleach(vec4(1))).toEqualTypeOf<Node<"vec4">>();
  });

  it("dotScreen takes a color and returns a color", () => {
    expectTypeOf(dotScreen(vec4(1))).toEqualTypeOf<Node<"vec4">>();
  });

  it("rgbShift takes a sampler and returns a color", () => {
    expectTypeOf(rgbShift(uniform("sampler2D"))).toEqualTypeOf<Node<"vec4">>();
  });

  it("circle returns a float mask", () => {
    expectTypeOf(circle()).toEqualTypeOf<Node<"float">>();
  });

  it("gaussianBlur and bloom return a PassGraph", () => {
    expectTypeOf(gaussianBlur(uniform("sampler2D"))).toEqualTypeOf<PassGraph>();
    expectTypeOf(bloom(uniform("sampler2D"))).toEqualTypeOf<PassGraph>();
  });

  it("a custom high-pass filter types its inputs", () => {
    const anamorphic: HighPassFn = (input, threshold, smoothWidth) =>
      vec4(luminance(input.rgb).mul(threshold).add(smoothWidth), 1);
    const graph = bloom(uniform("sampler2D"), { highPassFn: anamorphic });
    expectTypeOf(graph.passes[0].color).toEqualTypeOf<Node<"vec4">>();
  });
});

function boolNode(): Node<"bool"> {
  return float(1).greaterThan(0.5);
}
