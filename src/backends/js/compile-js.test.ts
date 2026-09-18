import { describe, expect, it } from "vitest";
import { attribute, builtinPosition, Fn, uniform, varying, vec4 } from "../../rmsl";
import { compileJS } from "./rasterizer";

describe("JS backend: compileJS (vertex+fragment rasterizer pipeline)", () => {
  it("draws a triangle covering the screen, interpolating a varying color", () => {
    const posAttr = attribute("vec3");
    const colorVarying = varying("vec3");

    const vertexFn = () =>
      Fn(() => {
        colorVarying.assign(posAttr.mul(0.5).add(0.5));
        builtinPosition().assign(vec4(posAttr.x, posAttr.y, posAttr.z, 1));
      })();
    const fragmentFn = () => Fn(() => vec4(colorVarying.x, colorVarying.y, colorVarying.z, 1))();

    const routine = compileJS(vertexFn as any, fragmentFn as any, { attributeTypes: { [posAttr.name]: "vec3" } });

    const width = 4;
    const height = 4;
    // one triangle overscaled past the clip-space square so every pixel is covered
    const positions = [
      [-1, -1, 0],
      [3, -1, 0],
      [-1, 3, 0],
    ];

    const result = routine.draw(
      { attributes: { [posAttr.name]: new Float64Array(positions.flat()) } },
      3,
      width,
      height,
    );
    expect(result.length).toBe(width * height * 4);
    expect(Array.from(result).every((v) => v >= 0 && v <= 1)).toBe(true);
    expect(Array.from(result).some((v) => v !== 0)).toBe(true);
  });

  it("threads a uniform into the fragment stage", () => {
    const posAttr = attribute("vec3");
    const colorUniform = uniform("vec3");

    const vertexFn = () => Fn(() => builtinPosition().assign(vec4(posAttr.x, posAttr.y, posAttr.z, 1)))();
    const fragmentFn = () => Fn(() => vec4(colorUniform.x, colorUniform.y, colorUniform.z, 1))();

    const routine = compileJS(vertexFn as any, fragmentFn as any, { attributeTypes: { [posAttr.name]: "vec3" } });
    const positions = new Float64Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]);

    const result = routine.draw(
      { attributes: { [posAttr.name]: positions }, uniforms: { [colorUniform.name]: [0, 1, 0] } },
      3,
      2,
      2,
    );
    expect(Array.from(result.slice(0, 4))).toEqual([0, 1, 0, 1]);
  });

  it("clears the output buffer between draw() calls instead of leaving stale pixels", () => {
    const posAttr = attribute("vec3");
    const colorUniform = uniform("vec3");

    const vertexFn = () => Fn(() => builtinPosition().assign(vec4(posAttr.x, posAttr.y, posAttr.z, 1)))();
    const fragmentFn = () => Fn(() => vec4(colorUniform.x, colorUniform.y, colorUniform.z, 1))();

    const routine = compileJS(vertexFn as any, fragmentFn as any, { attributeTypes: { [posAttr.name]: "vec3" } });
    const width = 4;
    const height = 4;

    // draw 1: a big red triangle covering the whole screen
    const bigTriangle = new Float64Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]);
    const first = routine.draw(
      { attributes: { [posAttr.name]: bigTriangle }, uniforms: { [colorUniform.name]: [1, 0, 0] } },
      3,
      width,
      height,
    );
    expect(Array.from(first).every((_, i) => i % 4 !== 3 || first[i] === 1)).toBe(true); // every pixel's alpha is 1: fully covered

    // draw 2: a tiny blue triangle covering exactly one corner pixel (NDC 0.5 units, one quarter of a 4-wide axis),
    // with `clear: true` — the "one draw, one frame" case this option exists for
    const tinyTriangle = new Float64Array([-1, -1, 0, -0.5, -1, 0, -1, -0.5, 0]);
    const second = routine.draw(
      { attributes: { [posAttr.name]: tinyTriangle }, uniforms: { [colorUniform.name]: [0, 0, 1] } },
      3,
      width,
      height,
      undefined,
      true,
    );

    // an uncovered pixel reads as cleared (0,0,0,0), not the stale red `first` left at that same address
    expect(Array.from(second.slice(-4))).toEqual([0, 0, 0, 0]);
    // some pixel is genuinely covered by the tiny triangle
    expect(Array.from(second).some((v, i) => i % 4 === 2 && v === 1)).toBe(true);
  });

  it("keeps a persistent depth buffer across draw() calls until clearDepth()", () => {
    const posAttr = attribute("vec3");
    const colorUniform = uniform("vec3");

    const vertexFn = () => Fn(() => builtinPosition().assign(vec4(posAttr.x, posAttr.y, posAttr.z, 1)))();
    const fragmentFn = () => Fn(() => vec4(colorUniform.x, colorUniform.y, colorUniform.z, 1))();

    const routine = compileJS(vertexFn as any, fragmentFn as any, { attributeTypes: { [posAttr.name]: "vec3" } });
    const width = 2;
    const height = 2;
    const triangleAt = (z: number) => new Float64Array([-1, -1, z, 3, -1, z, -1, 3, z]);
    const draw = (z: number, color: number[]) =>
      routine.draw(
        { attributes: { [posAttr.name]: triangleAt(z) }, uniforms: { [colorUniform.name]: color } },
        3,
        width,
        height,
      );

    // near, red — passes the depth test against the freshly (auto-)cleared buffer
    expect(Array.from(draw(-0.5, [1, 0, 0]).slice(0, 3))).toEqual([1, 0, 0]);

    // far, blue, same routine, no clearDepth(): occluded by the persisted near depth
    expect(Array.from(draw(0.5, [0, 0, 1]).slice(0, 3))).toEqual([1, 0, 0]);

    // clearDepth(), then the same far draw now passes
    routine.clearDepth();
    expect(Array.from(draw(0.5, [0, 0, 1]).slice(0, 3))).toEqual([0, 0, 1]);
  });
});
