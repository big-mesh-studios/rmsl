import { describe, it, expect } from "vitest";
import {
  Fn,
  If,
  Discard,
  float,
  vec2,
  vec3,
  vec4,
  ivec2,
  uvec3,
  uniform,
  varying,
  attribute,
  output,
  builtinPosition,
  builtinFragDepth,
  fragCoord,
  mix,
  step,
  uv,
  length,
  smoothstep,
} from "../rmsl";
import { DataTexture } from "../scene/textures/DataTexture";
import { LinearFilter, NearestMipmapNearestFilter, RedIntegerFormat } from "../scene/textures/constants";
import { evaluate, render, runner, uniformsIn, approx, assertClose, closeTo, tolerance } from "./index";

describe("evaluate", () => {
  it("returns the value a graph computes", () => {
    const result = evaluate(() => float(2).mul(3).add(1));
    expect(result.value).toBe(7);
    expect(result.discarded).toBe(false);
  });

  it("binds a uniform by its node, not by its slot name", () => {
    const tint = uniform("vec3");
    const graph = vec4(tint.mul(0.5), 1);
    expect(evaluate(() => graph, { uniforms: [[tint, [1, 0.5, 0]]] }).value).toEqual([0.5, 0.25, 0, 1]);
  });

  it("binds varyings and attributes the same way", () => {
    const position = attribute("vec3");
    const weight = varying("float");
    const graph = position.mul(weight);
    const result = evaluate(() => graph, {
      attributes: [[position, [1, 2, 3]]],
      varyings: [[weight, 2]],
    });
    expect(result.value).toEqual([2, 4, 6]);
  });

  it("places the fragment where fragCoord() says", () => {
    const graph = fragCoord().x;
    expect(evaluate(() => graph, { fragCoord: [12.5, 0.5] }).value).toBe(12.5);
  });

  it("samples a texture bound to its sampler", () => {
    const map = uniform("sampler2D");
    const graph = map.texture(vec2(0.5, 0.5));
    const data = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4];
    const result = evaluate(() => graph, { textures: [[map, { data, width: 2, height: 2 }]] });
    expect(result.value).toEqual([4, 4, 4, 4]);
  });

  it("reads an 8-bit texture as the 0-1 a float sampler gives a shader", () => {
    const map = uniform("sampler2D");
    const graph = map.texture(vec2(0.5, 0.5));
    const data = new Uint8Array([0, 128, 255, 255]);
    expect(evaluate(() => graph, { textures: [[map, { data, width: 1, height: 1 }]] }).value).toEqual([
      0,
      128 / 255,
      1,
      1,
    ]);
  });

  it("takes a texture in the shape a scene DataTexture already has", () => {
    const map = uniform("isampler2D");
    const graph = map.texture(ivec2(1, 0));
    // What `new DataTexture(data, 2, 2)` looks like: pixels under `image`.
    const image = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    const result = evaluate(() => graph, { textures: [[map, { image, width: 2, height: 2 }]] });
    expect(result.value).toEqual([5, 6, 7, 8]);
  });

  it("reads a single-channel DataTexture a texel at a time", () => {
    const map = uniform("usampler3D");
    const graph = map.texture(uvec3(1, 1, 0));
    // A voxel volume, one byte a voxel: the format is what says where one texel
    // ends and the next begins, and green, blue and alpha read as a sampler
    // reports the channels the texture does not store.
    const image = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const volume = new DataTexture(image, 2, 2, 2, RedIntegerFormat);
    const result = evaluate(() => graph, { textures: [[map, volume]] });
    expect(result.value).toEqual([4, 0, 0, 1]);
  });

  it("samples a scene texture the way the renderers read it", () => {
    const map = uniform("sampler2D");
    const graph = map.texture(vec2(0.5, 0.5));
    const texture = new DataTexture(new Uint8Array([0, 0, 0, 0, 255, 255, 255, 255]), 2, 1);
    // Two texels, black and white, sampled halfway between their centres. What
    // that returns is the texture's own filtering — and a mipmapped spelling
    // counts as its base filter, which is `samplerState`'s rule, not a second
    // one written here.
    texture.magFilter = LinearFilter;
    expect(evaluate(() => graph, { textures: [[map, texture]] }).value).toEqual([0.5, 0.5, 0.5, 0.5]);
    texture.magFilter = NearestMipmapNearestFilter;
    expect(evaluate(() => graph, { textures: [[map, texture]] }).value).toEqual([1, 1, 1, 1]);
  });

  it("reports a discarded fragment rather than a bare null", () => {
    const graph = Fn(() => {
      If(float(1).greaterThan(0), () => {
        Discard();
      });
      return float(5);
    })();
    const result = evaluate(() => graph);
    expect(result.discarded).toBe(true);
    expect(result.value).toBe(null);
  });

  it("carries the fragment depth and the written outputs", () => {
    const graph = Fn(() => {
      const depth = builtinFragDepth();
      depth.assign(float(0.25));
      const color = output("vec4");
      color.assign(vec4(1, 0, 0, 1));
      return color;
    })();
    const result = evaluate(() => graph);
    expect(result.fragDepth).toBe(0.25);
    expect(Object.values(result.outputs)).toEqual([[1, 0, 0, 1]]);
  });

  it("runs a vertex stage, with its position and varyings", () => {
    const graph = Fn(() => {
      const colour = varying("vec3");
      colour.assign(vec3(1, 2, 3));
      const position = builtinPosition();
      position.assign(vec4(0, 0, 0, 1));
      return position;
    })();
    const result = evaluate(() => graph, { stage: "vertex" });
    expect(result.position).toEqual([0, 0, 0, 1]);
    expect(Object.values(result.varyings)).toEqual([[1, 2, 3]]);
  });

  it("reads derivatives as zero, and says so on request", () => {
    const graph = float(3).fwidth();
    expect(evaluate(() => graph).value).toBe(0);
    expect(() => evaluate(() => graph, { derivatives: "throw" })).toThrow(/CPU target/);
  });
});

describe("runner", () => {
  it("compiles once and evaluates many times", () => {
    const threshold = uniform("float");
    const graph = step(threshold, fragCoord().x);
    const run = runner(() => graph);

    const below = run({ uniforms: [[threshold, 10]], fragCoord: [4.5, 0.5] });
    const above = run({ uniforms: [[threshold, 10]], fragCoord: [14.5, 0.5] });
    expect([below.value, above.value]).toEqual([0, 1]);
  });

  it("holds defaults that a call can override", () => {
    const tint = uniform("float");
    const run = runner(() => tint.mul(2), { uniforms: [[tint, 1]] });
    expect(run().value).toBe(2);
    expect(run({ uniforms: [[tint, 3]] }).value).toBe(6);
  });

  it("keeps the generated source for when a result needs explaining", () => {
    const run = runner(() => float(1).add(2));
    expect(run.source).toContain("function rmslTestShader");
  });

  it("is accepted wherever a graph is", () => {
    const run = runner(() => fragCoord().x.div(4));
    expect(evaluate(run, { fragCoord: [2.5, 0.5] }).value).toBe(0.625);
    expect(render(run, { width: 4, height: 1 }).at(3, 0)[0]).toBe(0.875);
  });
});

describe("render", () => {
  const tint = uniform("vec3");
  const ramp = vec4(mix(vec3(0, 0, 0), tint, fragCoord().x.div(8)), 1);

  it("evaluates every fragment of the grid", () => {
    const image = render(() => ramp, { width: 8, height: 2, uniforms: [[tint, [1, 1, 1]]] });
    expect(image.width).toBe(8);
    expect(image.pixels).toHaveLength(8 * 2 * 4);
    expect(image.at(0, 0)).toEqual([0.0625, 0.0625, 0.0625, 1]);
    expect(image.at(7, 1)).toEqual([0.9375, 0.9375, 0.9375, 1]);
  });

  it("makes a shape assertable — this ramp only ever brightens", () => {
    const image = render(() => ramp, { width: 16, height: 1, uniforms: [[tint, [1, 1, 1]]] });
    for (let x = 1; x < image.width; x++) {
      expect(image.at(x, 0)[0]).toBeGreaterThan(image.at(x - 1, 0)[0]);
    }
  });

  it("feeds each fragment its own inputs", () => {
    const coordinate = varying("vec2");
    const image = render(() => vec4(coordinate, 0, 1), {
      width: 4,
      height: 4,
      inputs: ({ u, v }) => ({ varyings: [[coordinate, [u, v]]] }),
    });
    expect(image.at(0, 0)).toEqual([0.125, 0.125, 0, 1]);
    expect(image.at(3, 3)).toEqual([0.875, 0.875, 0, 1]);
  });

  it("draws a picture that reads in a terminal", () => {
    const half = step(float(4), fragCoord().x);
    const image = render(() => vec4(vec3(half), 1), { width: 8, height: 2 });
    expect(image.toAscii({ ramp: ".#" })).toBe("....####\n....####");
  });

  it("draws the disc the documentation shows", () => {
    const centred = fragCoord().sub(vec2(12, 6)).div(vec2(12, 6));
    const disc = float(1).sub(smoothstep(float(0.7), float(1), length(centred)));
    const image = render(() => vec4(vec3(disc), 1), { width: 24, height: 12 });
    expect(image.toAscii()).toMatchInlineSnapshot(`
      "         ..::..         
           .-+*#%%%%#*+-.     
         :=#@@@@@@@@@@@@#=:   
        -*%@@@@@@@@@@@@@@%*-  
       :+%@@@@@@@@@@@@@@@@%+: 
       -#@@@@@@@@@@@@@@@@@@#- 
       -#@@@@@@@@@@@@@@@@@@#- 
       :+%@@@@@@@@@@@@@@@@%+: 
        -*%@@@@@@@@@@@@@@%*-  
         :=#@@@@@@@@@@@@#=:   
           .-+*#%%%%#*+-.     
               ..::..         "
    `);
  });

  it("names the slot when a graph reads an input nothing bound", () => {
    const fade = varying("float");
    // No program, so there is no name but the generated slot — which is still
    // better than the NaN this used to shade with.
    expect(() => evaluate(() => vec4(vec3(1), fade))).toThrow(
      /fragment stage reads a varying nothing bound.*\[node, value\]/s,
    );
  });

  it("draws a shape that lives in alpha", () => {
    // A particle: white everywhere, its shape carried entirely by alpha. Over
    // black that is the disc it draws; on the colour channels alone it is a
    // flat block, which is what `{ alpha: false }` asks for.
    const coverage = step(float(4), fragCoord().x);
    const image = render(() => vec4(vec3(1), coverage), { width: 8, height: 1 });
    expect(image.toAscii({ ramp: ".#" })).toBe("....####");
    expect(image.toAscii({ ramp: ".#", alpha: false })).toBe("########");
  });

  it("prints from the bottom row up, and the other way round on request", () => {
    const graph = vec4(vec3(step(float(1), fragCoord().y)), 1);
    const image = render(() => graph, { width: 2, height: 2 });
    expect(image.toAscii({ ramp: ".#" })).toBe("##\n..");
    expect(image.toAscii({ ramp: ".#", flipY: true })).toBe("..\n##");
  });

  it("converts to clamped bytes, in either row order", () => {
    const graph = vec4(vec3(fragCoord().y.mul(1.5)), 1);
    const image = render(() => graph, { width: 1, height: 2 });
    // The bottom row is 0.75 of full brightness; the top row is over 1 and
    // clamps. Byte order follows the rows, so flipping swaps which comes first.
    expect([...image.toUint8()].slice(0, 4)).toEqual([191, 191, 191, 255]);
    expect([...image.toUint8({ flipY: true })].slice(0, 4)).toEqual([255, 255, 255, 255]);
  });

  it("marks the fragments that discarded, and leaves them black", () => {
    const graph = Fn(() => {
      If(fragCoord().x.greaterThan(2), () => {
        Discard();
      });
      return vec4(1, 1, 1, 1);
    })();
    const image = render(() => graph, { width: 4, height: 1 });
    expect(image.isDiscarded(0, 0)).toBe(false);
    expect(image.isDiscarded(3, 0)).toBe(true);
    expect(image.at(3, 0)).toEqual([0, 0, 0, 0]);
    expect(image.resultAt(3, 0).value).toBe(null);
  });

  it("lights all three channels from a scalar shader", () => {
    const image = render(() => fragCoord().x.div(2), { width: 1, height: 1 });
    expect(image.at(0, 0)).toEqual([0.25, 0.25, 0.25, 1]);
  });

  it("refuses a coordinate outside the image, and an empty one", () => {
    const image = render(() => float(1), { width: 2, height: 2 });
    expect(() => image.at(2, 0)).toThrow(/outside the 2x2 image/);
    expect(() => image.at(0, -1)).toThrow(/outside the 2x2 image/);
    expect(() => render(() => float(1), { width: 0, height: 4 })).toThrow(/at least 1/);
  });
});

describe("uniformsIn", () => {
  it("hands back the uniforms a graph made for itself", () => {
    // `uv()` mints a screen-size uniform inside the graph: the caller never
    // sees the node, so this is the only way to bind it.
    const graph = vec4(uv(), 0, 1);
    const [resolution] = uniformsIn(graph, "vec2");
    const image = render(() => graph, {
      width: 4,
      height: 4,
      uniforms: [[resolution, [4, 4]]],
    });
    expect(image.at(0, 0)).toEqual([0.125, 0.125, 0, 1]);
    expect(image.at(3, 3)).toEqual([0.875, 0.875, 0, 1]);
  });

  it("lists each uniform once, in the order the graph reaches them", () => {
    const first = uniform("float");
    const second = uniform("vec2");
    const graph = vec4(second, first, first);
    expect(uniformsIn(graph).map((node) => node.name)).toEqual([second.name, first.name]);
  });
});

describe("comparing values", () => {
  it("allows the last bit of a float to differ", () => {
    expect(closeTo(0.1 + 0.2, 0.3)).toBe(true);
    expect(closeTo(0.3001, 0.3)).toBe(false);
    expect(closeTo([1, 2, 3], [1, 2, 3.0000001])).toBe(true);
  });

  it("scales what it allows with the size of the number", () => {
    expect(tolerance(0)).toBe(1e-6);
    expect(tolerance(1000)).toBe(1e-3);
    expect(closeTo(1000.0005, 1000)).toBe(true);
    expect(closeTo(1.0005, 1)).toBe(false);
  });

  it("takes a tolerance of its own", () => {
    expect(closeTo(0.55, 0.5, { tolerance: 0.1 })).toBe(true);
    expect(closeTo(0.55, 0.5)).toBe(false);
  });

  it("compares lengths, booleans and NaN", () => {
    expect(closeTo([1, 2], [1, 2, 3])).toBe(false);
    expect(closeTo(true, true)).toBe(true);
    expect(closeTo(true, false)).toBe(false);
    expect(closeTo(NaN, NaN)).toBe(true);
    expect(closeTo(NaN, 0)).toBe(false);
  });

  it("reads as a predicate for a matcher that takes one", () => {
    const result = evaluate(() => vec4(1, 0, 0, 1));
    expect(result.value).toSatisfy(approx([1, 0, 0, 1]));
  });

  it("names the component that differs when it throws", () => {
    expect(() => assertClose([1, 0, 0, 1], [1, 0, 1, 1], { message: "tint" })).toThrow(
      /tint: component 2: expected 1, got 0/,
    );
    expect(() => assertClose([1, 0, 0, 1], [1, 0, 1, 1])).toThrow(/expected: \[1, 0, 1, 1\]/);
    expect(() => assertClose(1, 1)).not.toThrow();
  });
});
