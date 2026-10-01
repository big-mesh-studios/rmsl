import { describe, it, expect, afterAll } from "vitest";
import { Fn, int, mat2x3, mat3, vec3, type Node } from "./rmsl";
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

const m = () => mat3(1, 2, 3, 4, 5, 6, 7, 8, 9);

describe("a matrix's column by index", () => {
  it("reads by a constant index, as on every backend", () => {
    expect(evaluateRecording(() => m().element(int(1)))).toEqual([4, 5, 6]);
    expect(evaluateRecording(() => m().element(2).y)).toBe(8);
    expect(evaluateRecording(() => mat2x3(1, 2, 3, 4, 5, 6).element(int(1)))).toEqual([4, 5, 6]);
  });

  it("reads by an index computed at run time", () => {
    expect(evaluateRecording((a) => m().element(a.toInt()), [2])).toEqual([7, 8, 9]);
    const column = Fn((a: Node<"float">) => m().toVar().element(a.toInt()).z);
    expect(evaluateRecording((a) => column(a), [0])).toBe(3);
  });

  it("writes into a variable by a constant or a computed index", () => {
    const write = Fn((a: Node<"float">) => {
      const v = m().toVar();
      v.element(int(0)).assign(vec3(10, 11, 12));
      v.element(a.toInt()).assign(vec3(20, 21, 22));
      return v.element(0).add(v.element(1)).add(v.element(2));
    });
    expect(evaluateRecording((a) => write(a), [2])).toEqual([34, 37, 40]);
  });

  // The recording harness skips WASM when it reports a construct unsupported, so these pin it there directly.
  it("compiles on WASM, rather than being reported unsupported", () => {
    expect(evaluateWASM(() => m().element(int(1)))).toEqual([4, 5, 6]);
    expect(evaluateWASM((a) => m().element(a.toInt()), [2])).toEqual([7, 8, 9]);
    const write = Fn((a: Node<"float">) => {
      const v = m().toVar();
      v.element(a.toInt()).assign(vec3(20, 21, 22));
      return v.element(0).add(v.element(1)).add(v.element(2));
    });
    expect(evaluateWASM((a) => write(a), [1])).toEqual([28, 31, 34]);
  });

  it("clamps an index computed past the matrix's end, or below zero, to its last column on WASM", () => {
    expect(evaluateWASM((a) => m().element(a.toInt()), [9])).toEqual([7, 8, 9]);
    expect(evaluateWASM((a) => m().element(a.toInt()), [-1])).toEqual([7, 8, 9]);
    const write = Fn((a: Node<"float">) => {
      const v = m().toVar();
      const w = vec3(100, 100, 100).toVar();
      v.element(a.toInt()).assign(vec3(0, 0, 0));
      return v.element(2).add(w);
    });
    expect(evaluateWASM((a) => write(a), [9])).toEqual([100, 100, 100]);
    expect(evaluateWASM((a) => write(a), [-1])).toEqual([100, 100, 100]);
  });

  it("rejects a constant index outside the matrix on WASM, as WGSL and GLSL do", () => {
    expect(() => evaluateWASM(() => m().element(int(3)))).toThrow(
      /\[RMSL\] compileWasmFn: index 3 is outside a mat3's columns 0 to 2/,
    );
    const write = Fn(() => {
      const v = m().toVar();
      v.element(int(-1)).assign(vec3(0, 0, 0));
      return v.element(0);
    });
    expect(() => evaluateWASM(() => write())).toThrow(/index -1 is outside a mat3's columns 0 to 2/);
  });
});
