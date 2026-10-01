import { describe, it, expect } from "vitest";
import {
  attribute,
  builtinFragDepth,
  builtinPosition,
  float,
  Fn,
  instancedArray,
  int,
  invocationIndex,
  mat3,
  uniform,
  uniformArray,
  uint,
  varying,
  vec2,
  vec3,
  vec4,
  type Node,
  type ShaderType,
} from "./rmsl";
import { compileGlsl, compileGlslFn } from "./glsl";
import { compileWgsl, compileWgslFn } from "./wgsl";
import { compileJSFn, compileJSRoutine } from "./js";
import { compileWasmFn, compileWasmRoutine } from "./wasm";

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

/** Expects compiling `write` as a `stage` stage to throw `message` on every backend. */
function expectRefusedInStage(stage: "vertex" | "fragment", write: () => void, message: RegExp) {
  const build = () =>
    Fn(() => {
      write();
      return stage === "vertex" ? vec4(0, 0, 0, 1) : vec4(1, 1, 1, 1);
    })();
  expect(() => compileGlsl[stage](build()), "GLSL").toThrow(message);
  expect(() => compileWgsl[stage](build()), "WGSL").toThrow(message);
  expect(() => compileJSRoutine(build as any, { name: "main", params: [], stage }), "JS").toThrow(message);
  expect(() => compileWasmRoutine(build as any, { name: "main", params: [], stage }), "WASM").toThrow(message);
}

describe("an assignment's target", () => {
  it("refuses a stage output outside the stage that writes it, on every backend", () => {
    const color = varying("vec3");
    expectRefusedInStage(
      "fragment",
      () => color.x.assign(float(1)),
      /\[RMSL\] can't assign to a varying in a fragment stage; only a vertex stage writes it/,
    );
    expectRefusedInStage(
      "fragment",
      () => builtinPosition().assign(vec4(0, 0, 0, 1)),
      /\[RMSL\] can't assign to the position in a fragment stage; only a vertex stage writes it/,
    );
    expectRefusedInStage(
      "vertex",
      () => builtinFragDepth().assign(float(0.5)),
      /\[RMSL\] can't assign to the fragment depth in a vertex stage; only a fragment stage writes it/,
    );
  });

  it("refuses a parameter of the compiled function, on every backend", () => {
    for (const [name, compile] of Object.entries(compilers)) {
      expect(
        () =>
          compile(
            (x: Node<"float">) =>
              Fn(() => {
                (x as any).assign(float(2));
                return x;
              })(),
            { name: "main", params: [{ name: "x", type: "float" }] },
          ),
        name,
      ).toThrow(
        /\[RMSL\] can't assign to "x", a parameter of the compiled function, whose value belongs to the caller/,
      );
    }
  });

  it("refuses a uniform, whole or in part, on every backend", () => {
    const v = uniform("vec3");
    const m = uniform("mat3");
    const refused =
      /\[RMSL\] can't assign to a uniform: only a variable, a storage element or a stage output can be assigned/;
    // @ts-expect-error: refused at run time, and by its type
    expectRefusedEverywhere(() => v.assign(vec3(1, 2, 3)), refused);
    // @ts-expect-error: refused at run time, and by its type
    expectRefusedEverywhere(() => v.y.assign(float(9)), refused);
    // @ts-expect-error: refused at run time, and by its type
    expectRefusedEverywhere(() => v.element(int(1)).assign(float(9)), refused);
    // @ts-expect-error: refused at run time, and by its type
    expectRefusedEverywhere(() => m.element(int(0)).assign(vec3(9, 9, 9)), refused);
    // @ts-expect-error: refused at run time, and by its type
    expectRefusedEverywhere(() => m.element(int(0)).y.assign(float(9)), refused);
    // @ts-expect-error: refused at run time, and by its type
    expectRefusedEverywhere(() => m.assign(mat3(1, 0, 0, 0, 1, 0, 0, 0, 1)), refused);
    const vs = uniformArray("vec3", 4);
    // @ts-expect-error: refused at run time, and by its type
    expectRefusedEverywhere(() => vs.element(int(1)).x.assign(float(9)), refused);
  });

  it("names a whole uniform array a uniform, and a built-in input as one, on every backend", () => {
    const vs = uniformArray("vec3", 4);
    expectRefusedEverywhere(() => (vs as any).assign(vec3(1, 2, 3)), /\[RMSL\] can't assign to a uniform: /);
    // @ts-expect-error: refused at run time, and by its type
    expectRefusedEverywhere(() => invocationIndex().assign(uint(0)), /\[RMSL\] can't assign to a built-in input: /);
  });

  it("refuses an attribute, whole or in part, on every backend", () => {
    const position = attribute("vec3");
    const refused = /\[RMSL\] can't assign to an attribute: /;
    // @ts-expect-error: refused at run time, and by its type
    expectRefusedEverywhere(() => position.assign(vec3(1, 2, 3)), refused);
    // @ts-expect-error: refused at run time, and by its type
    expectRefusedEverywhere(() => position.x.assign(float(9)), refused);
  });

  it("refuses a whole storage buffer, rather than one of its elements, on every backend", () => {
    const values = instancedArray(4, "vec3");
    expectRefusedEverywhere(
      // @ts-expect-error: refused at run time, and by its type
      () => values.assign(vec3(1, 2, 3)),
      /\[RMSL\] can't assign to a whole storage buffer; assign to one of its elements with \.element\(i\)/,
    );
  });

  it("refuses a swizzle that names a component more than once, on every backend", () => {
    const refused = (pattern: string) =>
      new RegExp(`\\[RMSL\\] can't assign through the swizzle \\.${pattern}, which names a component more than once`);
    expectRefusedEverywhere(() => {
      const v = vec3(1, 2, 3).toVar();
      // @ts-expect-error: refused at run time, and by its type
      v.xx.assign(vec2(1, 2));
    }, refused("xx"));
    expectRefusedEverywhere(() => {
      const v = vec3(1, 2, 3).toVar();
      // @ts-expect-error: refused at run time, and by its type
      v.xy.xx.assign(vec2(1, 2));
    }, refused("xx"));
    expectRefusedEverywhere(() => {
      const v = vec3(1, 2, 3).toVar();
      // @ts-expect-error: refused at run time, and by its type
      v.xxy.z.assign(float(1));
    }, refused("xxy"));
  });

  it("accepts a swizzle that names each component once, through another swizzle", () => {
    const write = Fn(() => {
      const v = vec3(1, 2, 3).toVar();
      v.yzx.xy.assign(vec2(7, 8));
      v.zx.assign(vec2(4, 5));
      return v;
    })();
    for (const [name, compile] of Object.entries(compilers)) {
      expect(() => compile(() => write as Node<ShaderType>, { name: "main", params: [] }), name).not.toThrow();
    }
  });

  it("refuses a computed value, whole or in part, on every backend", () => {
    const refused = /\[RMSL\] can't assign to a computed value: .* copy the value into a variable with toVar\(\) first/;
    expectRefusedEverywhere(() => {
      const w = vec3(1, 2, 3).toVar();
      // @ts-expect-error: refused at run time, and by its type
      w.add(1).y.assign(float(9));
    }, refused);
    expectRefusedEverywhere(() => {
      const w = vec3(1, 2, 3).toVar();
      // @ts-expect-error: refused at run time, and by its type
      w.mul(2).element(int(0)).assign(float(9));
    }, refused);
    expectRefusedEverywhere(() => {
      const n = mat3(1, 2, 3, 4, 5, 6, 7, 8, 9).toVar();
      n.mul(n)
        .element(int(0))
        // @ts-expect-error: refused at run time, and by its type
        .assign(vec3(0, 0, 0));
    }, refused);
    expectRefusedEverywhere(() => {
      const n = mat3(1, 2, 3, 4, 5, 6, 7, 8, 9).toVar();
      // @ts-expect-error: refused at run time, and by its type
      n.mul(n).element(int(0)).y.assign(float(0));
    }, refused);
  });
});
