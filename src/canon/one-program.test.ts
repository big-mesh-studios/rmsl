import { afterAll, describe, expect, it } from "vitest";
import {
  Fn,
  If,
  float,
  int,
  mat2,
  mat3,
  output,
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
  evaluateJS,
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
   * @canon spec-folding-gives-the-run-time-result
   */
  it("folds literals to what the same program computes at run time", () => {
    const folded = evaluateRecording(() => int(7).div(int(-2)).toFloat().add(float(-7).mod(float(3))));
    const computed = evaluateRecording((a, b, c, d) => a.toInt().div(b.toInt()).toFloat().add(c.mod(d)), [7, -2, -7, 3]);
    expect(folded).toBe(-1);
    expect(computed).toBe(folded);
  });

  /**
   * A matrix inverse, a floored vector modulus and a matrix cut down from a
   * larger one each need a helper on WGSL. Used together, each still computes
   * its own part.
   *
   * Fails until the JS target narrows a matrix by its columns (#64) and the
   * WASM target compiles both inverse and narrowing (#65). It is evaluated
   * here without recording, so the GPU targets, which already agree, do not
   * fail the file meanwhile.
   *
   * @canon spec-a-target-without-a-builtin-gets-a-helper
   */
  it.fails("computes an inverse, a modulus and a narrowed matrix in one program", () => {
    const build = (a: Node<"float">) =>
      mat2(vec2(a, 1), vec2(2, 4))
        .inverse()
        .element(int(0))
        .x.add(vec2(a, 5).mod(3).y)
        .add(mat2(mat3(1, 2, 3, 4, 5, 6, 7, 8, 9)).element(int(1)).y);
    expect(evaluateJS(build, [3])).toBeCloseTo(7.4, 12);
    expect(evaluateWASM(build, [3])).toBe(evaluateJS(build, [3]));
  });

  /**
   * More value uniforms than WGSL has uniform buffers, a uniform array, a
   * boolean uniform and a texture, all in one program.
   *
   * @canon spec-a-program-declares-any-number-of-uniforms-on-every-target
   */
  it("declares many uniforms of every kind in one program on every target", () => {
    const build = () =>
      Fn(() => {
        const out = output("vec4");
        let sum = float(0).toVar();
        for (let i = 0; i < 16; i++) sum.assign(sum.add(uniform("float")));
        const flags = uniformArray("bool", 4);
        const flag = uniform("bool");
        const tex = uniform("sampler2D");
        If(flags.element(int(1)).and(flag), () => {
          sum.assign(sum.add(1));
        });
        out.assign(tex.texture(vec2(0.5, 0.5)).mul(sum));
        return out;
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
   * @canon spec-a-stage-passes-its-values-on-every-target
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
   * @canon spec-a-program-runs-its-statements-in-the-order-it-writes-them
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
   * @canon spec-a-cpu-target-samples-a-texture-as-a-gpu-sampler-does
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
   * @canon spec-integer-arithmetic-follows-wgsl
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
   * @canon spec-float-arithmetic-gives-one-result
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
   * @canon spec-the-test-suite-holds-every-target-to-the-program
   */
  it("records what it evaluates and what it compiles for the harness to check", () => {
    const before = recordedEvaluationSummary().total;
    expect(evaluateRecording((a) => a.mul(2), [4])).toBe(8);
    expect(recordedEvaluationSummary().total).toBe(before + 1);
    expect(evaluateJS((a) => a.mul(2), [4])).toBe(8);
    expect(compileWgsl(Fn(() => float(1).toVar())())).toContain("fn main");
  });
});
