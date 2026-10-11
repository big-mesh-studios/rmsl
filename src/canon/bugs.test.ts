import { describe, expect, it } from "vitest";
import { Fn, float, int, Loop, mat3, vec4 } from "../rmsl";
import { compileGlsl } from "../glsl";
import { compileJSRoutine, compileJSVertex } from "../js";
import { compileWasmVertex } from "../wasm";

const param = { name: "main", params: [{ name: "a", type: "float" as const }] };
const none = { name: "main", params: [] };

describe("known bugs, each failing until its fix", () => {
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
});
