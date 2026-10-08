import { describe, expect, it } from "vitest";
import { Fn, float, instanceIndex, int, Loop, mat3, vec4, vertexIndex } from "../rmsl";
import { compileGlsl } from "../glsl";
import { compileJSRoutine, compileJSVertex } from "../js";
import { compileWasmVertex } from "../wasm";

const param = { name: "main", params: [{ name: "a", type: "float" as const }] };
const none = { name: "main", params: [] };

describe("known bugs, each failing until its fix", () => {
  /**
   * @canon bug-the-cpu-targets-compile-no-index-accessors
   */
  it.fails("compiles vertexIndex and instanceIndex on the CPU targets", () => {
    const build = () => Fn(() => vec4(vertexIndex().toFloat(), instanceIndex().toFloat(), 0, 1))();
    expect(() => compileJSVertex(build, { ...none })).not.toThrow();
    expect(() => compileWasmVertex(build, { ...none })).not.toThrow();
  });

  /**
   * @canon bug-a-bool-count-compiles-into-a-comparison-no-driver-accepts
   */
  it.fails("refuses a bool given to Loop as its count", () => {
    const program = Fn(() => {
      const m = float(0).toVar();
      Loop(m.lessThan(10) as any, () => {
        m.assign(m.add(1));
      });
      return m;
    });
    expect(() => compileGlsl(program())).toThrow();
  });

  /**
   * @canon bug-a-write-by-index-through-a-swizzle-differs-by-target
   */
  it.fails("writes by index through a swizzle of a matrix column on JS", () => {
    const build = (a: any) =>
      Fn(() => {
        const m = mat3(1, 2, 3, 4, 5, 6, 7, 8, 9).toVar();
        m.element(int(1)).yx.element(a.toInt()).assign(float(0));
        return m.element(1);
      })();
    expect(compileJSRoutine(build, param)({ params: { a: 0 } })).toEqual([4, 0, 6]);
  });
});
