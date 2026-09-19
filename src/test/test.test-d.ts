import { describe, it, expectTypeOf } from "vitest";
import { float, vec2, vec3, vec4, uniform, varying } from "../rmsl";
import { DataTexture, RedIntegerFormat } from "../scene";
import { evaluate, render, runner, type ShaderValue, type EvaluationResult } from "./index";

describe("bound values", () => {
  it("takes the value a node's type carries", () => {
    const scale = uniform("float");
    const tint = uniform("vec3");
    const flag = uniform("bool");
    evaluate(() => tint.mul(scale), {
      uniforms: [
        [scale, 2],
        [tint, [1, 0, 0]],
        [flag, true],
      ],
    });
  });

  it("rejects a value of the wrong shape", () => {
    const tint = uniform("vec3");
    const scale = uniform("float");
    // @ts-expect-error a vec3 is bound to an array, not a number
    evaluate(() => tint, { uniforms: [[tint, 1]] });
    // @ts-expect-error a float is bound to a number, not an array
    evaluate(() => scale.mul(1), { uniforms: [[scale, [1, 2, 3]]] });
  });

  it("rejects a texture bound as a plain uniform", () => {
    const map = uniform("sampler2D");
    // @ts-expect-error a sampler takes texture data, and takes it under `textures`
    evaluate(() => map.texture(vec2(0, 0)), { uniforms: [[map, [1, 2, 3, 4]]] });
    evaluate(() => map.texture(vec2(0, 0)), {
      textures: [[map, { data: [1, 1, 1, 1], width: 1, height: 1 }]],
    });
  });

  it("takes a scene texture as it stands, and its own data too", () => {
    const map = uniform("usampler3D");
    const volume = new DataTexture(new Uint8Array([1, 2]), 2, 1, 1, RedIntegerFormat);
    evaluate(() => map.texture(vec3(0, 0, 0).toUVec3()), {
      textures: [[map, volume]],
    });
    evaluate(() => map.texture(vec3(0, 0, 0).toUVec3()), {
      textures: [[map, { data: [1, 2], width: 2, height: 1, channels: 1 }]],
    });
  });

  it("rejects pixels nothing could read", () => {
    const map = uniform("sampler2D");
    evaluate(() => map.texture(vec2(0, 0)), {
      // @ts-expect-error a path is not an image; nothing here would fetch it
      textures: [[map, { image: "pixels.png", width: 2, height: 2 }]],
    });
    evaluate(() => map.texture(vec2(0, 0)), {
      // @ts-expect-error a texture with no pixels at all is not texture data
      textures: [[map, { width: 2, height: 2 }]],
    });
  });

  it("binds varyings by node too", () => {
    const coordinate = varying("vec2");
    render(() => vec4(coordinate, 0, 1), {
      width: 2,
      height: 2,
      inputs: ({ u, v }) => ({ varyings: [[coordinate, [u, v]]] }),
    });
  });
});

describe("results", () => {
  it("carries the value type of the graph", () => {
    expectTypeOf(evaluate(() => float(1))).toEqualTypeOf<EvaluationResult<"float">>();
    expectTypeOf(evaluate(() => float(1)).value).toEqualTypeOf<number | null>();
    expectTypeOf(evaluate(() => vec4(1, 0, 0, 1)).value).toEqualTypeOf<number[] | null>();
    expectTypeOf(runner(() => vec3(1, 2, 3))().value).toEqualTypeOf<number[] | null>();
  });

  it("maps shader types to what they are on the CPU", () => {
    expectTypeOf<ShaderValue<"float">>().toEqualTypeOf<number>();
    expectTypeOf<ShaderValue<"int">>().toEqualTypeOf<number>();
    expectTypeOf<ShaderValue<"bool">>().toEqualTypeOf<boolean>();
    expectTypeOf<ShaderValue<"bvec3">>().toEqualTypeOf<boolean[]>();
    expectTypeOf<ShaderValue<"vec4">>().toEqualTypeOf<number[]>();
    expectTypeOf<ShaderValue<"mat4">>().toEqualTypeOf<number[]>();
  });
});
