import { afterAll, describe, expect, it } from "vitest";
import {
  Break,
  builtinFragDepth,
  builtinPosition,
  float,
  fragCoord,
  Fn,
  instancedArray,
  int,
  mat2x3,
  mat2x4,
  mat3,
  output,
  uniform,
  uint,
  varying,
  vec2,
  vec3,
  vec4,
  type Node,
} from "../rmsl";
import { compileJSRoutine } from "../js";
import { compileWasmRoutine } from "../wasm";
import {
  assertRecordedShadersValid,
  recordingGLSL as compileGlsl,
  recordingWGSL as compileWgsl,
} from "../testing/shader-validity";

afterAll(async () => {
  await assertRecordedShadersValid();
}, 120_000);

const cpuCompilers = [
  (build: () => Node<any>) => compileJSRoutine(build, { name: "main", params: [] }),
  (build: () => Node<any>) => compileWasmRoutine(build, { name: "main", params: [] }),
];

describe("a mistake is refused before the program runs", () => {
  /**
   * A product whose shapes meet and the inverse of a square matrix compile
   * beside each other, while a product whose shapes do not meet and the
   * inverse of a matrix that is not square are each refused.
   *
   * @canon spec-a-matrix-product-whose-shapes-do-not-meet-is-refused
   * @canon spec-only-a-square-matrix-is-inverted
   */
  it("compiles the matrix operations the shapes allow and refuses the others", () => {
    const allowed = Fn(() =>
      uniform("mat2x3")
        .mul(uniform("mat3x2"))
        .inverse()
        .element(int(0))
        .toVar(),
    );
    expect(compileGlsl(allowed())).toContain("inverse(");
    expect(compileWgsl(allowed())).toContain("_rmsl_inverse3");
    expect(() => (mat2x3(1, 0, 0, 1, 0, 0) as any).mul(mat2x4(1, 0, 0, 1, 0, 0, 0, 0))).toThrow();
    const notSquare = () => Fn(() => (uniform("mat2x3") as any).inverse().element(int(0)).toVar())();
    expect(() => compileGlsl(notSquare())).toThrow(/square/i);
    expect(() => compileWgsl(notSquare())).toThrow(/square/i);
  });

  /**
   * Bare numbers beside an `int`, a `uint` and a `float` in one program each
   * take their neighbour's type, and the numbers those types cannot hold are
   * refused.
   *
   * @canon spec-a-bare-number-beside-an-integer-is-an-integer
   * @canon spec-a-bare-number-beside-a-float-is-a-float
   * @canon spec-a-fraction-beside-an-integer-is-refused
   * @canon spec-a-negative-number-for-an-unsigned-type-is-refused
   */
  it("gives each bare number its neighbour's type in one program, and refuses what the type cannot hold", () => {
    const prog = Fn(() =>
      uniform("int")
        .add(2)
        .toFloat()
        .add(uniform("uint").mul(3).toFloat())
        .add(uniform("float").mul(4))
        .toVar(),
    );
    const wgsl = compileWgsl(prog());
    expect(wgsl).toContain("2i");
    expect(wgsl).toContain("3u");
    expect(wgsl).toContain("4f");
    expect(compileGlsl(prog())).not.toMatch(/float\(2\)|float\(3\)/);
    expect(() => Fn(() => uniform("int").add(2.5).toVar())()).toThrow(/whole number|integer/i);
    expect(() => Fn(() => uniform("uint").add(-1).toVar())()).toThrow(/negative|unsigned/i);
    expect(() => uint(-1)).toThrow(/unsigned/);
  });

  /**
   * A vertex stage that writes its position and a varying, and a fragment
   * stage that writes its depth and reads its coordinate, compile together.
   * Moving any of those built-ins to the other stage is refused.
   *
   * @canon spec-the-fragment-depth-is-written-only-in-a-fragment-stage
   * @canon spec-the-position-is-read-only-in-a-vertex-stage
   * @canon spec-a-vertex-stage-writes-its-position
   */
  it("compiles each stage's built-ins in its own stage and refuses them in the other", () => {
    const tint = varying("vec2");
    const vertex = () =>
      Fn(() => {
        tint.assign(vec2(1, 0));
        builtinPosition().assign(vec4(0, 0, 0, 1));
      })();
    const fragment = () =>
      Fn(() => {
        builtinFragDepth().assign(float(0.5));
        const out = output("vec4");
        out.assign(vec4(tint.add(fragCoord()), 0, 1));
        return out;
      })();
    for (const compile of [compileGlsl, compileWgsl]) {
      expect(compile.vertex(vertex())).toContain(tint.name);
      expect(compile.fragment(fragment())).toContain(tint.name);
      expect(() => compile.vertex(fragment() as any)).toThrow();
      expect(() => compile.fragment(Fn(() => vec4(builtinPosition().x, 0, 0, 1).toVar())())).toThrow(/fragment/i);
    }
  });

  /**
   * A `Break` outside a loop and a `cross` of vectors that are not `vec3` are
   * each refused on the CPU targets, beside a program that does both the way
   * a target can run them. WASM also refuses a whole storage buffer read as a
   * value.
   *
   * @canon spec-break-or-continue-outside-a-loop-is-refused
   * @canon spec-cross-of-a-vector-that-is-not-a-vec3-is-refused
   * @canon spec-a-whole-storage-buffer-cannot-be-read
   */
  it("refuses each operation no target can run, beside the same program written the way they run", () => {
    const values = instancedArray(4, "float");
    const runnable = () =>
      Fn(() => vec3(1, 0, 0).cross(vec3(0, 1, 0)).x.add(values.element(int(0))).toVar())();
    for (const compile of cpuCompilers) {
      expect(() => compile(runnable)).not.toThrow();
      expect(() =>
        compile(() =>
          Fn(() => {
            Break();
            return float(1);
          })(),
        ),
      ).toThrow();
      expect(() => compile(() => Fn(() => (vec2(1, 0) as any).cross(vec2(0, 1)).toVar())())).toThrow();
    }
    expect(() => cpuCompilers[1]!(() => Fn(() => (values as any).add(1).toVar())())).toThrow(/read as a whole/);
  });
});
