import { NodeError, ProgramError } from "./errors";
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
import { compileJSFn, compileJSCompute, compileJSFragment, compileJSVertex } from "./js";
import { compileWasmFn, compileWasmCompute, compileWasmFragment, compileWasmVertex } from "./wasm";

const compilers = { compileGlslFn, compileWgslFn, compileJSFn, compileWasmFn };

/** Expects building `write` to throw a `NodeError` of `code`, with a message that matches `message` where one is given. */
function expectRefusedAsBuilt(write: () => void, code: string, message?: RegExp) {
  const build = () =>
    Fn(() => {
      write();
      return float(0);
    })();
  expect(build).toThrowRmsl(NodeError, code, message);
}

/** Expects compiling `write` as a `stage` stage to throw a `NodeError` of `assign-in-wrong-stage` on every backend. */
function expectRefusedInStage(stage: "vertex" | "fragment", write: () => void) {
  const build = () =>
    Fn(() => {
      write();
      return stage === "vertex" ? vec4(0, 0, 0, 1) : vec4(1, 1, 1, 1);
    })();
  const code = "assign-in-wrong-stage";
  expect(() => compileGlsl[stage](build()), "GLSL").toThrowRmsl(ProgramError, code);
  expect(() => compileWgsl[stage](build()), "WGSL").toThrowRmsl(ProgramError, code);
  const none = { name: "main", params: [] };
  const [js, wasm]: ((fn: any, options: typeof none) => unknown)[] =
    stage === "vertex" ? [compileJSVertex, compileWasmVertex] : [compileJSFragment, compileWasmFragment];
  expect(() => js(build as any, none), "JS").toThrowRmsl(ProgramError, code);
  expect(() => wasm(build as any, none), "WASM").toThrowRmsl(ProgramError, code);
}

describe("an assignment's target", () => {
  /**
   * @canon spec-a-stage-output-is-refused-in-the-other-render-stage
   */
  it("refuses a stage output in the other render stage, on every backend", () => {
    const color = varying("vec3");
    expectRefusedInStage("fragment", () => color.x.assign(float(1)));
    expectRefusedInStage("fragment", () => builtinPosition().assign(vec4(0, 0, 0, 1)));
    expectRefusedInStage("vertex", () => builtinFragDepth().assign(float(0.5)));
  });

  /**
   * A compute program that writes a storage buffer compiles. Assigning a
   * varying, the position or the fragment depth is refused on each target
   * that has a compute stage, which is not GLSL.
   *
   * @canon spec-a-compute-program-cannot-assign-a-stage-output
   */
  it("refuses a stage output assigned by a compute program", () => {
    const buf = instancedArray(4, "float");
    const program = (write: () => void) => () =>
      Fn(() => {
        buf.element(invocationIndex()).assign(float(1));
        write();
      })();
    const compilers = [
      (build: () => Node<any>) => compileWgsl.compute(build()),
      (build: () => Node<any>) => compileJSCompute(build as any, { name: "main", params: [] }),
      (build: () => Node<any>) => compileWasmCompute(build as any, { name: "main", params: [] }),
    ];
    const refused = [
      () => varying("vec3").x.assign(float(1)),
      () => builtinPosition().assign(vec4(0, 0, 0, 1)),
      () => builtinFragDepth().assign(float(0.5)),
    ];
    for (const compile of compilers) {
      expect(() => compile(program(() => {}))).not.toThrow();
      for (const write of refused)
        expect(() => compile(program(write))).toThrowRmsl(ProgramError, "assign-in-wrong-stage");
    }
  });

  /**
   * @canon spec-a-parameter-of-the-compiled-function-cannot-be-assigned
   */
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
      ).toThrowRmsl(NodeError, "assign-to-parameter");
    }
  });

  /**
   * @canon spec-a-uniform-cannot-be-assigned
   */
  it("refuses a uniform, whole or in part, as the graph is built", () => {
    const v = uniform("vec3");
    const m = uniform("mat3");
    const refused = "assign-to-read-only";
    // @ts-expect-error: refused at run time, and by its type
    expectRefusedAsBuilt(() => v.assign(vec3(1, 2, 3)), refused);
    // @ts-expect-error: refused at run time, and by its type
    expectRefusedAsBuilt(() => v.y.assign(float(9)), refused);
    // @ts-expect-error: refused at run time, and by its type
    expectRefusedAsBuilt(() => v.element(int(1)).assign(float(9)), refused);
    // @ts-expect-error: refused at run time, and by its type
    expectRefusedAsBuilt(() => m.element(int(0)).assign(vec3(9, 9, 9)), refused);
    // @ts-expect-error: refused at run time, and by its type
    expectRefusedAsBuilt(() => m.element(int(0)).y.assign(float(9)), refused);
    // @ts-expect-error: refused at run time, and by its type
    expectRefusedAsBuilt(() => m.assign(mat3(1, 0, 0, 0, 1, 0, 0, 0, 1)), refused);
    const vs = uniformArray("vec3", 4);
    // @ts-expect-error: refused at run time, and by its type
    expectRefusedAsBuilt(() => vs.element(int(1)).x.assign(float(9)), refused);
  });

  /**
   * @canon spec-a-uniform-cannot-be-assigned
   * @canon spec-a-built-in-input-cannot-be-assigned
   */
  it("names a whole uniform array a uniform, and a built-in input as one, as the graph is built", () => {
    const vs = uniformArray("vec3", 4);
    expectRefusedAsBuilt(() => (vs as any).assign(vec3(1, 2, 3)), "assign-to-read-only");
    // @ts-expect-error: refused at run time, and by its type
    expectRefusedAsBuilt(() => invocationIndex().assign(uint(0)), "assign-to-read-only");
  });

  /**
   * @canon spec-an-attribute-cannot-be-assigned
   */
  it("refuses an attribute, whole or in part, as the graph is built", () => {
    const position = attribute("vec3");
    const refused = "assign-to-read-only";
    // @ts-expect-error: refused at run time, and by its type
    expectRefusedAsBuilt(() => position.assign(vec3(1, 2, 3)), refused);
    // @ts-expect-error: refused at run time, and by its type
    expectRefusedAsBuilt(() => position.x.assign(float(9)), refused);
  });

  /**
   * @canon spec-a-whole-storage-buffer-cannot-be-assigned
   */
  it("refuses a whole storage buffer, rather than one of its elements, as the graph is built", () => {
    const values = instancedArray(4, "vec3");
    expectRefusedAsBuilt(
      // @ts-expect-error: refused at run time, and by its type
      () => values.assign(vec3(1, 2, 3)),
      "assign-to-whole-storage",
      /can't assign to a whole storage buffer; assign to one of its elements with \.element\(i\)/,
    );
  });

  /**
   * @canon spec-a-swizzle-that-repeats-a-component-cannot-be-assigned
   */
  it("refuses a swizzle that names a component more than once, as the graph is built", () => {
    expectRefusedAsBuilt(() => {
      const v = vec3(1, 2, 3).toVar();
      // @ts-expect-error: refused at run time, and by its type
      v.xx.assign(vec2(1, 2));
    }, "swizzle-repeats-component");
    expectRefusedAsBuilt(() => {
      const v = vec3(1, 2, 3).toVar();
      // @ts-expect-error: refused at run time, and by its type
      v.xy.xx.assign(vec2(1, 2));
    }, "swizzle-repeats-component");
    expectRefusedAsBuilt(() => {
      const v = vec3(1, 2, 3).toVar();
      // @ts-expect-error: refused at run time, and by its type
      v.xxy.z.assign(float(1));
    }, "swizzle-repeats-component");
    expectRefusedAsBuilt(() => {
      const v = vec3(1, 2, 3).toVar();
      // @ts-expect-error: refused at run time, and by its type
      v.xx.element(int(1)).assign(float(1));
    }, "swizzle-repeats-component");
    expectRefusedAsBuilt(() => {
      const v = vec3(1, 2, 3).toVar();
      // @ts-expect-error: refused at run time, and by its type
      v.xxy.element(int(uniform("float"))).assign(float(1));
    }, "swizzle-repeats-component");
  });

  /**
   * @canon spec-a-swizzle-that-names-each-component-once-can-be-assigned
   */
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

  /**
   * @canon spec-a-computed-value-cannot-be-assigned
   */
  it("refuses a computed value, whole or in part, as the graph is built", () => {
    const refused = /can't assign to a computed value: .* copy the value into a variable with toVar\(\) first/;
    expectRefusedAsBuilt(
      () => {
        const w = vec3(1, 2, 3).toVar();
        // @ts-expect-error: refused at run time, and by its type
        w.add(1).y.assign(float(9));
      },
      "assign-to-read-only",
      refused,
    );
    expectRefusedAsBuilt(
      () => {
        const w = vec3(1, 2, 3).toVar();
        // @ts-expect-error: refused at run time, and by its type
        w.mul(2).element(int(0)).assign(float(9));
      },
      "assign-to-read-only",
      refused,
    );
    expectRefusedAsBuilt(
      () => {
        const n = mat3(1, 2, 3, 4, 5, 6, 7, 8, 9).toVar();
        n.mul(n)
          .element(int(0))
          // @ts-expect-error: refused at run time, and by its type
          .assign(vec3(0, 0, 0));
      },
      "assign-to-read-only",
      refused,
    );
    expectRefusedAsBuilt(
      () => {
        const n = mat3(1, 2, 3, 4, 5, 6, 7, 8, 9).toVar();
        // @ts-expect-error: refused at run time, and by its type
        n.mul(n).element(int(0)).y.assign(float(0));
      },
      "assign-to-read-only",
      refused,
    );
  });
});
