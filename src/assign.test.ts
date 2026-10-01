import { describe, it, expect } from "vitest";
import { attribute, float, Fn, int, mat3, uniform, vec3, type Node, type ShaderType } from "./rmsl";
import { compileGlslFn } from "./glsl";
import { compileWgslFn } from "./wgsl";
import { compileJSFn } from "./js";
import { compileWasmFn } from "./wasm";

const compilers = { compileGlslFn, compileWgslFn, compileJSFn, compileWasmFn };

/** Expects compiling `write` to throw `message` on every backend. */
function expectRefusedEverywhere(write: () => void, message: RegExp) {
  const root = Fn(() => {
    write();
    return float(0);
  })();
  for (const [name, compile] of Object.entries(compilers)) {
    expect(() => compile(() => root as Node<ShaderType>, { name: "main", params: [] }), name).toThrow(message);
  }
}

describe("an assignment's target", () => {
  it("refuses a uniform, whole or in part, on every backend", () => {
    const v = uniform("vec3");
    const m = uniform("mat3");
    const refused =
      /\[RMSL\] can't assign to a uniform: only a variable, a storage element or a stage output can be assigned/;
    expectRefusedEverywhere(() => v.assign(vec3(1, 2, 3)), refused);
    expectRefusedEverywhere(() => v.y.assign(float(9)), refused);
    expectRefusedEverywhere(() => v.element(int(1)).assign(float(9)), refused);
    expectRefusedEverywhere(() => m.element(int(0)).assign(vec3(9, 9, 9)), refused);
    expectRefusedEverywhere(() => m.element(int(0)).y.assign(float(9)), refused);
  });

  it("refuses an attribute, whole or in part, on every backend", () => {
    const position = attribute("vec3");
    const refused = /\[RMSL\] can't assign to an attribute: /;
    expectRefusedEverywhere(() => position.assign(vec3(1, 2, 3)), refused);
    expectRefusedEverywhere(() => position.x.assign(float(9)), refused);
  });

  it("refuses a computed value, whole or in part, on every backend", () => {
    const refused = /\[RMSL\] can't assign to a computed value: .* copy the value into a variable with toVar\(\) first/;
    expectRefusedEverywhere(() => {
      const w = vec3(1, 2, 3).toVar();
      w.add(1).y.assign(float(9));
    }, refused);
    expectRefusedEverywhere(() => {
      const w = vec3(1, 2, 3).toVar();
      w.mul(2).element(int(0)).assign(float(9));
    }, refused);
    expectRefusedEverywhere(() => {
      const n = mat3(1, 2, 3, 4, 5, 6, 7, 8, 9).toVar();
      n.mul(n)
        .element(int(0))
        .assign(vec3(0, 0, 0));
    }, refused);
    expectRefusedEverywhere(() => {
      const n = mat3(1, 2, 3, 4, 5, 6, 7, 8, 9).toVar();
      n.mul(n).element(int(0)).y.assign(float(0));
    }, refused);
  });
});
