import { describe, expect, it } from "vitest";
import { attribute, builtinPosition, float, Fn, int, varying, vec3, vec4 } from "./rmsl";
import { compileGlsl } from "./glsl";
import { compileWgsl } from "./wgsl";
import { compileJSRoutine } from "./js";
import { compileWasmRoutine } from "./wasm";
import { compileJS } from "./backends/js/rasterizer";
import { compileWasm } from "./backends/wasm/rasterizer";

describe("writing part of a stage output", () => {
  /**
   * @canon spec-a-swizzle-write-writes-the-components-it-names
   * @canon spec-an-element-write-writes-at-its-index
   */
  it("writes a varying and the position through a swizzle and an index, on the JS and WASM rasterizers", () => {
    const position = attribute("vec3");
    const color = varying("vec3");
    const vertexFn = () =>
      Fn(() => {
        color.assign(vec3(0, 0, 0));
        color.x.assign(float(1));
        color.element(int(2)).assign(float(0.5));
        const clip = builtinPosition();
        clip.assign(vec4(0, 0, 0, 0));
        clip.xy.assign(position.xy);
        clip.element(int(3)).assign(float(1));
      })();
    const fragmentFn = () => Fn(() => vec4(color, 1))();
    // one triangle overscaled past the clip-space square so every pixel is covered
    const positions = new Float64Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]);

    const routines = [
      compileJS(vertexFn as any, fragmentFn as any, { attributeTypes: { [position.name]: "vec3" } }),
      compileWasm(vertexFn as any, fragmentFn as any),
    ];
    for (const routine of routines) {
      const pixels = routine.draw({ attributes: { [position.name]: positions } }, { count: 3, width: 2, height: 2 });
      expect(Array.from(pixels.slice(0, 4)).map((v) => Math.round(v * 1000) / 1000)).toEqual([1, 0, 0.5, 1]);
    }
  });

  /**
   * @canon spec-a-swizzle-write-writes-the-components-it-names
   * @canon spec-an-element-write-writes-at-its-index
   */
  it("compiles a fragment output written through a swizzle and an index, on every backend", () => {
    const fragment = () =>
      Fn(() => {
        const color = vec4(0, 0, 0, 1).toVar();
        color.zx.assign(color.xz.add(1));
        color.element(int(1)).assign(float(0.5));
        return color;
      })();
    expect(() => compileGlsl.fragment(fragment())).not.toThrow();
    expect(() => compileWgsl.fragment(fragment())).not.toThrow();
    expect(() => compileJSRoutine(fragment as any, { name: "main", params: [], stage: "fragment" })).not.toThrow();
    expect(() => compileWasmRoutine(fragment as any, { name: "main", params: [], stage: "fragment" })).not.toThrow();
  });
});
