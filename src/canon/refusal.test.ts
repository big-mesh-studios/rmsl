import { afterAll, describe, expect, it } from "vitest";
import {
  attribute,
  Break,
  builtinFragDepth,
  builtinPosition,
  float,
  For,
  fragCoord,
  Fn,
  If,
  instancedArray,
  int,
  mat2x3,
  mat2x4,
  mat3,
  output,
  uniform,
  Switch,
  uint,
  varying,
  vec2,
  While,
  vec3,
  vec4,
  type Node,
} from "../rmsl";
import { compileJSRoutine } from "../js";
import { compileWasmRoutine } from "../wasm";
import {
  assertRecordedShadersValid,
  recordShaderSource,
  recordingGLSL as compileGlsl,
  recordingWGSL as compileWgsl,
} from "../testing/shader-validity";
// The real compiler: the recording stand-ins cover vertex and fragment only,
// so a compute program's source is recorded by hand to reach Dawn.
import { compileWgsl as realCompileWgsl } from "../backends/wgsl/wgsl";

const computeWgsl = (root: Node<any>) => recordShaderSource("wgsl", "compute", realCompileWgsl.compute(root));

afterAll(async () => {
  await assertRecordedShadersValid();
}, 120_000);

const cpuCompilers = [
  (build: () => Node<any>) => compileJSRoutine(build, { name: "main", params: [] }),
  (build: () => Node<any>) => compileWasmRoutine(build, { name: "main", params: [] }),
];

/** The same two, compiling for the compute stage, where a storage program goes. */
const cpuComputeCompilers = [
  (build: () => Node<any>) => compileJSRoutine(build, { name: "main", params: [], stage: "compute" }),
  (build: () => Node<any>) => compileWasmRoutine(build, { name: "main", params: [], stage: "compute" }),
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

  /**
   * A `For` whose update holds a block, such as an `If`, is refused on every
   * target, because the update slot of a GLSL, WGSL or JavaScript `for` takes
   * none. A `For` whose update is a plain statement compiles on each.
   *
   * @canon spec-a-for-update-that-holds-a-block-is-refused
   */
  it("refuses a For whose update holds a block on every target", () => {
    const loop = (update: (i: any) => any) => () =>
      Fn(() => {
        const sum = float(0).toVar();
        For(
          () => int(0).toVar(),
          (i) => i.lessThan(3),
          update,
          () => sum.addAssign(1),
        );
        return sum;
      })();
    const plain = loop((i) => i.addAssign(1));
    const refusal = /update cannot contain a block/;
    expect(() => compileGlsl(plain())).not.toThrow();
    expect(() => compileWgsl(plain())).not.toThrow();
    for (const compile of cpuCompilers) expect(() => compile(plain)).not.toThrow();
    // A block directly in the update, in a loop of its own, and behind a nested Fn.
    const blocks = [
      loop((i) => If(i.greaterThan(-1), () => i.addAssign(1))),
      loop((i) => While(i.lessThan(1), () => i.addAssign(1))),
      loop((i) =>
        Fn(() => {
          If(i.greaterThan(-1), () => i.addAssign(1));
        })(),
      ),
    ];
    for (const block of blocks) {
      expect(() => compileGlsl(block())).toThrow(refusal);
      expect(() => compileWgsl(block())).toThrow(refusal);
      for (const compile of cpuCompilers) expect(() => compile(block)).toThrow(refusal);
    }
  });

  /**
   * A `Case` given no values can match no selector, so it is refused, naming
   * `Case`.
   *
   * @canon spec-a-case-with-no-values-is-refused
   */
  it("refuses a Case with no values, naming it", () => {
    const program = () =>
      Fn(() => {
        const v = float(0).toVar();
        Switch(int(uniform("float")), (s) => {
          s.Case([], () => {
            v.assign(float(1));
          });
        });
        return vec4(v);
      })();
    expect(program).toThrow(/Case\(\) needs at least one value/);
  });

  /**
   * A `Switch` with no `Case` and no `Default` has nothing to run, and compiles
   * on every target.
   *
   * @canon spec-a-switch-runs-the-case-its-selector-matches
   */
  it("compiles a Switch with no Case and no Default on every target, and runs the statements after it", () => {
    const build = () =>
      Fn(() => {
        const v = float(0).toVar();
        Switch(int(uniform("float")), () => {});
        v.assign(float(2));
        return vec4(v);
      })();
    expect(() => compileGlsl.fragment(build())).not.toThrow();
    expect(() => compileWgsl.fragment(build())).not.toThrow();
    for (const compile of cpuCompilers) expect(() => compile(build)).not.toThrow();
    for (const compile of cpuCompilers) {
      const result: any = compile(build).run({});
      expect(Array.from(Array.isArray(result) ? result : result.value)[0]).toBe(2);
    }
  });

  /**
   * A compute program that reads a buffer through `storage()` compiles on
   * every target. Spelling the same buffer as an `attribute()` is refused on
   * each, because a compute dispatch has no vertices to read one for. GLSL has
   * no compute stage at all, so the three targets here are all of them.
   *
   * @canon spec-a-compute-program-cannot-read-an-attribute
   */
  it("refuses an attribute read by a compute program, where the same buffer through storage() compiles", () => {
    const buf = instancedArray(4, "float");
    const viaStorage = () => Fn(() => buf.element(int(0)).add(1).toVar())();
    const viaAttribute = () => Fn(() => attribute("float").add(1).toVar())();

    expect(computeWgsl(viaStorage())).toContain("@compute");
    expect(() => computeWgsl(viaAttribute())).toThrow(/cannot read an attribute/);
    for (const compile of cpuComputeCompilers) {
      expect(() => compile(viaStorage)).not.toThrow();
      expect(() => compile(viaAttribute)).toThrow(/cannot read an attribute/);
    }
  });

  /**
   * A compute program writes into a storage buffer and returns nothing, which
   * compiles on every target. Assigning to an `output()` is refused on each,
   * because an output is a fragment stage's result and a compute entry point
   * returns nothing to hold one. GLSL has no compute stage, so the three
   * targets here are all of them.
   *
   * @canon spec-a-compute-program-cannot-write-an-output
   */
  it("refuses an output assigned by a compute program, where writing a storage buffer compiles", () => {
    const buf = instancedArray(4, "float");
    const viaStorage = () => Fn(() => buf.element(int(0)).assign(float(2)))();
    const viaOutput = () => Fn(() => output("float").assign(float(2)))();

    expect(computeWgsl(viaStorage())).toContain("@compute");
    expect(() => computeWgsl(viaOutput())).toThrow(/cannot write an output/);
    for (const compile of cpuComputeCompilers) {
      expect(() => compile(viaStorage)).not.toThrow();
      expect(() => compile(viaOutput)).toThrow(/cannot write an output/);
    }
  });

  /**
   * A compute program that reads a uniform compiles on every target. Reading a
   * `varying()` is refused on each, because a compute dispatch has no vertex
   * stage to pass one from. GLSL has no compute stage, so the three targets
   * here are all of them.
   *
   * @canon spec-a-compute-program-cannot-read-a-varying
   */
  it("refuses a varying read by a compute program, where a uniform compiles", () => {
    const buf = instancedArray(4, "float");
    const viaUniform = () => Fn(() => buf.element(int(0)).assign(uniform("float")))();
    const viaVarying = () => Fn(() => buf.element(int(0)).assign(varying("float")))();

    expect(computeWgsl(viaUniform())).toContain("@compute");
    expect(() => computeWgsl(viaVarying())).toThrow(/cannot read a varying/);
    for (const compile of cpuComputeCompilers) {
      expect(() => compile(viaUniform)).not.toThrow();
      expect(() => compile(viaVarying)).toThrow(/cannot read a varying/);
    }
  });
});
