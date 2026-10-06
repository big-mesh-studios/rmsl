import { afterAll, describe, expect, it } from "vitest";
import {
  Fn,
  If,
  float,
  int,
  mat2,
  mat3,
  uniform,
  uniformArray,
  varying,
  vec2,
  vec3,
  vec4,
  type Node,
} from "../rmsl";
import { compileJSRoutine } from "../js";
import { compileWasmRoutine } from "../wasm";
import {
  assertRecordedEvaluationsAgree,
  closeEvaluators,
  evaluateIntegerJS,
  evaluateIntegerWASM,
  evaluateIntegerWGSL,
  evaluateGLSL,
  evaluateJS,
  evaluateWGSL,
  evaluateRecording,
  evaluateWASM,
  GPU_EVALUATION_SKIPPED,
  recordedEvaluationSummary,
} from "../testing/shader-eval";
import {
  assertRecordedShadersValid,
  recordingGLSL as compileGlsl,
  recordingWGSL as compileWgsl,
} from "../testing/shader-validity";

afterAll(async () => {
  await assertRecordedShadersValid();
  await assertRecordedEvaluationsAgree();
  await closeEvaluators();
}, 120_000);

const INT_MAX = 2147483647;

describe("one program means the same on every target", () => {
  /**
   * A floored float modulus and a truncating integer division, once on
   * literals the compiler folds and once on values that arrive at run time,
   * give one result.
   *
   * @canon spec-float-folding-gives-the-run-time-result
   * @canon spec-integer-folding-gives-the-run-time-result
   */
  it("folds literals to what the same program computes at run time", () => {
    const folded = evaluateRecording(() => int(7).div(int(-2)).toFloat().add(float(-7).mod(float(3))));
    const computed = evaluateRecording((a, b, c, d) => a.toInt().div(b.toInt()).toFloat().add(c.mod(d)), [7, -2, -7, 3]);
    expect(folded).toBe(-1);
    expect(computed).toBe(folded);
  });

  /**
   * A `mat2` cut down from a `mat3` that arrives at run time keeps the leading
   * rows of its leading columns on GLSL and WGSL. JS and WASM depart from it,
   * as their bugs say.
   *
   * @canon spec-a-matrix-built-from-a-larger-matrix-keeps-its-leading-rows-and-columns
   */
  it.skipIf(GPU_EVALUATION_SKIPPED)("narrows a matrix to its leading rows and columns on the GPU targets", async () => {
    const build = (a: Node<"float">) =>
      mat2(mat3(vec3(a, 2, 3), vec3(4, 5, 6), vec3(7, 8, 9)))
        .element(int(1))
        .y;
    expect(await evaluateGLSL(build, [1])).toBe(5);
    expect(await evaluateWGSL(build, [1])).toBe(5);
  });

  /**
   * Converting a float past the range of an `int` clamps it on WGSL.
   *
   * @canon spec-a-float-converted-to-an-integer-clamps-to-its-range
   */
  it.skipIf(GPU_EVALUATION_SKIPPED)("clamps a float past the int range on WGSL", async () => {
    expect(await evaluateWGSL((a) => a.toInt().toFloat(), [3e9])).toBe(2147483520);
    expect(await evaluateWGSL((a) => a.toInt().toFloat(), [-3e9])).toBe(-2147483648);
  });

  /**
   * More value uniforms than WGSL has uniform buffers, a uniform array, a
   * boolean uniform and a texture, all in one program.
   *
   * @canon spec-wgsl-packs-every-value-uniform-into-one-binding
   * @canon spec-a-texture-keeps-a-binding-of-its-own
   * @canon spec-a-bool-uniform-travels-as-an-unsigned-integer
   * @canon spec-a-uniform-array-is-read-by-element
   */
  it("declares many uniforms of every kind in one program on every target", () => {
    const build = () =>
      Fn(() => {
        let sum = float(0).toVar();
        for (let i = 0; i < 16; i++) sum.assign(sum.add(uniform("float")));
        const flags = uniformArray("bool", 4);
        const flag = uniform("bool");
        const tex = uniform("sampler2D");
        If(flags.element(int(1)).and(flag), () => {
          sum.assign(sum.add(1));
        });
        return tex.texture(vec2(0.5, 0.5)).mul(sum);
      })();
    const wgsl = compileWgsl(build());
    expect(wgsl.split("\n").filter((l) => l.includes("var<uniform>"))).toHaveLength(1);
    expect(wgsl).toMatch(/var \S+: texture_2d<f32>;/);
    expect(wgsl).not.toContain("array<bool");
    const glsl = compileGlsl(build());
    expect(glsl.match(/uniform float /g) ?? []).toHaveLength(16);
  });

  /**
   * A vertex stage that writes a varying and a fragment stage that reads it
   * and returns its colour, compiled for both GPU targets.
   *
   * @canon spec-a-varying-passes-from-the-vertex-to-the-fragment-stage
   * @canon spec-a-vec4-result-is-the-colour
   */
  it("passes a varying to a fragment stage that writes its colour", () => {
    const tint = varying("vec3");
    const vertex = Fn(() => {
      tint.assign(vec3(1, 0.5, 0));
      return vec4(0, 0, 0, 1);
    });
    const fragment = Fn(() => vec4(tint, 1).toVar());
    for (const compile of [compileGlsl, compileWgsl]) {
      const v = compile.vertex(vertex());
      const f = compile.fragment(fragment());
      expect(v).toContain(tint.name);
      expect(f).toContain(tint.name);
      expect(f).toContain("_rmsl_fragColor");
    }
  });

  /**
   * The value a write computes runs before its index is read, and a column
   * index before a component index, in one program.
   *
   * @canon spec-the-index-of-a-write-is-read-after-the-value-is-computed
   * @canon spec-a-column-index-runs-before-a-component-index
   */
  it("runs the statements of values and indices in the order the program writes them", () => {
    const build = () =>
      Fn(() => {
        const v = mat3(0).toVar();
        const j = int(0).toVar();
        const value = Fn(() => {
          j.assign(j.add(1));
          return float(5);
        })();
        v.element(j).x.assign(value);
        const c = int(0).toVar();
        const next = Fn(() => {
          c.assign(c.add(1));
          return c.toVar();
        });
        const column = next();
        const row = next();
        v.element(column).element(row).assign(float(7));
        return v.element(1).x.mul(10).add(v.element(1).z.mul(100)).add(v.element(2).y);
      })();
    expect(evaluateRecording(build)).toBe(750);
    expect(evaluateWASM(build)).toBe(750);
  });

  /**
   * A single-channel byte texture asking for linear filtering and a repeating
   * wrap, sampled past its edge, reads the same on JS and WASM: normalized,
   * blended, wrapped, and one channel wide.
   *
   * @canon spec-a-cpu-target-filters-as-the-texture-asks
   * @canon spec-a-cpu-target-wraps-as-the-texture-asks
   * @canon spec-a-byte-texture-reads-as-zero-to-one
   * @canon spec-a-texel-holds-the-channels-its-texture-stores
   */
  it("filters, wraps and normalizes a single-channel byte texture alike on the CPU targets", () => {
    let tex!: any;
    const prog = Fn(() => {
      tex = uniform("sampler2D");
      return tex.texture(vec2(1.5, 0.5));
    })();
    const texture = {
      data: new Uint8Array([0, 255]),
      width: 2,
      height: 1,
      channels: 1 as const,
      magFilter: "linear" as const,
      wrapS: "repeat" as const,
    };
    const ctx = { textures: { [tex.name]: texture } };
    const js = compileJSRoutine(() => prog, { name: "main", params: [] }).run(ctx);
    const wasm = compileWasmRoutine(() => prog, { name: "main", params: [] }).run(ctx);
    expect(js).toEqual([0.5, 0, 0, 1]);
    expect((wasm as any).value ?? wasm).toEqual(js);
  });

  /**
   * An overflow, a division by a run-time zero and a shift by more than the
   * bit width, chained in one expression.
   *
   * @canon spec-js-integer-arithmetic-follows-wgsl
   * @canon spec-wasm-integer-arithmetic-follows-wgsl
   * @canon spec-wgsl-gives-the-defined-integer-result
   */
  it("wraps, divides by zero and shifts past the width in one expression", async () => {
    const build = (a: Node<"int">, b: Node<"int">, c: Node<"int">) =>
      a.add(int(1)).div(b).shiftRight(c) as Node<"int">;
    const args = [INT_MAX, 0, 33];
    const want = -1073741824;
    expect(evaluateIntegerJS(build as any, "int", args)).toBe(want);
    expect(evaluateIntegerWASM(build as any, "int", args)).toBe(want);
    if (!GPU_EVALUATION_SKIPPED) expect(await evaluateIntegerWGSL(build as any, "int", args)).toBe(want);
  });

  /**
   * Arithmetic, a transcendental function and a floored modulus in one float
   * expression: WASM gives the JS result exactly, and the GPU targets are held
   * to it within the f32 tolerance.
   *
   * @canon spec-wasm-and-js-give-the-same-float-bits
   * @canon exception-a-cpu-target-computes-floats-in-64-bits
   */
  it("gives one float result for arithmetic and transcendentals together", () => {
    const build = (a: Node<"float">, b: Node<"float">) => a.sin().mul(b).add(a.div(b)).add(b.mod(a)).sqrt();
    const js = evaluateRecording(build, [0.7, 2.3]);
    expect(evaluateWASM(build, [0.7, 2.3])).toBe(js);
    expect(js).toBeCloseTo(Math.sqrt(Math.sin(0.7) * 2.3 + 0.7 / 2.3 + (2.3 - 0.7 * Math.floor(2.3 / 0.7))), 12);
  });

  /**
   * A program recorded for evaluation and a shader recorded for validation in
   * one test: both reach the harness, which compares the targets after it.
   *
   * @canon spec-evaluation-reads-back-every-shape
   */
  it("records what it evaluates and what it compiles for the harness to check", () => {
    const before = recordedEvaluationSummary().total;
    expect(evaluateRecording((a) => a.mul(2), [4])).toBe(8);
    expect(recordedEvaluationSummary().total).toBe(before + 1);
    expect(evaluateJS((a) => a.mul(2), [4])).toBe(8);
    expect(compileWgsl(Fn(() => float(1).toVar())())).toContain("fn main");
  });
});
