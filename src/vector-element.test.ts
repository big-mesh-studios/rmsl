import { describe, it, expect, afterAll } from "vitest";
import { float, Fn, int, ivec3, vec4, type Node } from "./rmsl";
import {
  assertRecordedEvaluationsAgree,
  closeEvaluators,
  evaluateRecording,
  evaluateWASM,
} from "./testing/shader-eval";

afterAll(async () => {
  await assertRecordedEvaluationsAgree();
  await closeEvaluators();
}, 120_000);

describe("a vector's component by index", () => {
  /**
   * @canon spec-an-element-reads-a-component-by-index
   */
  it("reads by a constant index, as on every backend", () => {
    expect(evaluateRecording(() => vec4(1, 2, 3, 4).element(int(2)))).toBe(3);
    expect(evaluateRecording(() => vec4(1, 2, 3, 4).element(0))).toBe(1);
    expect(evaluateRecording(() => ivec3(7, 8, 9).element(int(1)).toFloat())).toBe(8);
  });

  /**
   * @canon spec-an-element-reads-a-component-by-index
   */
  it("reads by an index computed at run time", () => {
    expect(evaluateRecording((a) => vec4(1, 2, 3, 4).element(a.toInt()), [3])).toBe(4);
    expect(evaluateRecording((a) => vec4(1, 2, 3, 4).mul(2).element(a.toInt()), [1])).toBe(4);
  });

  /**
   * @canon spec-an-element-write-writes-at-its-index
   */
  it("writes into a variable by a constant or a computed index", () => {
    const write = Fn((a: Node<"float">) => {
      const v = vec4(1, 2, 3, 4).toVar();
      v.element(int(0)).assign(float(10));
      v.element(a.toInt()).assign(float(20));
      return v;
    });
    expect(evaluateRecording((a) => write(a), [2])).toEqual([10, 2, 20, 4]);
  });

  /**
   * The recording harness skips WASM when it reports a construct unsupported, so these pin it there directly.
   *
   * @canon spec-an-element-reads-a-component-by-index
   * @canon spec-an-element-write-writes-at-its-index
   */
  it("compiles on WASM, rather than being reported unsupported", () => {
    expect(evaluateWASM(() => vec4(1, 2, 3, 4).element(int(2)))).toBe(3);
    expect(evaluateWASM((a) => vec4(1, 2, 3, 4).element(a.toInt()), [3])).toBe(4);
    const write = Fn((a: Node<"float">) => {
      const v = vec4(1, 2, 3, 4).toVar();
      v.element(a.toInt()).assign(float(20));
      return v;
    });
    expect(evaluateWASM((a) => write(a), [1])).toEqual([1, 20, 3, 4]);
  });

  /**
   * @canon spec-a-run-time-index-past-the-end-reaches-the-last-element
   */
  it("clamps an index computed past the vector's end, or below zero, to its last component on WASM", () => {
    expect(evaluateWASM((a) => vec4(1, 2, 3, 4).element(a.toInt()), [9])).toBe(4);
    expect(evaluateWASM((a) => vec4(1, 2, 3, 4).element(a.toInt()), [-1])).toBe(4);
    const write = Fn((a: Node<"float">) => {
      const v = vec4(1, 2, 3, 4).toVar();
      const w = vec4(5, 6, 7, 8).toVar();
      v.element(a.toInt()).assign(float(20));
      return v.add(w);
    });
    expect(evaluateWASM((a) => write(a), [9])).toEqual([6, 8, 10, 28]);
    expect(evaluateWASM((a) => write(a), [-1])).toEqual([6, 8, 10, 28]);
  });

  /**
   * @canon spec-a-constant-index-outside-a-vector-or-matrix-is-refused
   */
  it("rejects a constant index outside the vector on WASM, as WGSL and GLSL do", () => {
    expect(() => evaluateWASM(() => vec4(1, 2, 3, 4).element(int(4)))).toThrow(
      /\[RMSL\] compileWasmFn: index 4 is outside a vec4's components 0 to 3/,
    );
    const write = Fn(() => {
      const v = vec4(1, 2, 3, 4).toVar();
      v.element(int(-1)).assign(float(20));
      return v;
    });
    expect(() => evaluateWASM(() => write())).toThrow(/index -1 is outside a vec4's components 0 to 3/);
  });
});
