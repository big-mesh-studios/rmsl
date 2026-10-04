import { describe, it, expect, afterAll } from "vitest";
import { float, Fn, int, mat2x3, mat3, vec2, vec3, type Node } from "./rmsl";
import {
  assertRecordedEvaluationsAgree,
  closeEvaluators,
  evaluateJS,
  evaluateRecording,
  evaluateWASM,
} from "./testing/shader-eval";

afterAll(async () => {
  await assertRecordedEvaluationsAgree();
  await closeEvaluators();
}, 120_000);

const m = () => mat3(1, 2, 3, 4, 5, 6, 7, 8, 9);

describe("a matrix's column by index", () => {
  /**
   * @canon spec-an-element-reads-a-component-by-index
   */
  it("reads by a constant index, as on every backend", () => {
    expect(evaluateRecording(() => m().element(int(1)))).toEqual([4, 5, 6]);
    expect(evaluateRecording(() => m().element(2).y)).toBe(8);
    expect(evaluateRecording(() => mat2x3(1, 2, 3, 4, 5, 6).element(int(1)))).toEqual([4, 5, 6]);
  });

  /**
   * @canon spec-an-element-reads-a-component-by-index
   */
  it("reads by an index computed at run time", () => {
    expect(evaluateRecording((a) => m().element(a.toInt()), [2])).toEqual([7, 8, 9]);
    const column = Fn((a: Node<"float">) => m().toVar().element(a.toInt()).z);
    expect(evaluateRecording((a) => column(a), [0])).toBe(3);
  });

  /**
   * @canon spec-an-element-write-writes-at-its-index
   */
  it("writes into a variable by a constant or a computed index", () => {
    const write = Fn((a: Node<"float">) => {
      const v = m().toVar();
      v.element(int(0)).assign(vec3(10, 11, 12));
      v.element(a.toInt()).assign(vec3(20, 21, 22));
      return v.element(0).add(v.element(1)).add(v.element(2));
    });
    expect(evaluateRecording((a) => write(a), [2])).toEqual([34, 37, 40]);
  });

  /**
   * @canon spec-an-element-write-writes-at-its-index
   */
  it("writes components of a column, by a swizzle or an index", () => {
    const write = Fn((a: Node<"float">) => {
      const v = m().toVar();
      v.element(a.toInt()).y.assign(float(50));
      v.element(int(2)).element(a.toInt()).assign(float(60));
      v.element(int(0)).zx.assign(vec2(70, 80));
      return v.element(0).add(v.element(1)).add(v.element(2));
    });
    expect(evaluateRecording((a) => write(a), [1])).toEqual([91, 112, 85]);
    expect(evaluateWASM((a) => write(a), [1])).toEqual([91, 112, 85]);
  });

  /**
   * The recording harness skips WASM when it reports a construct unsupported, so these pin it there directly.
   *
   * @canon spec-an-element-reads-a-component-by-index
   * @canon spec-an-element-write-writes-at-its-index
   */
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

  /**
   * @canon spec-a-run-time-index-past-the-end-reaches-the-last-element
   */
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

  /**
   * @canon spec-a-run-time-index-past-the-end-reaches-the-last-element
   */
  it("keeps a write by an index computed outside the matrix inside it, alike on JS and WASM", () => {
    const write = Fn((a: Node<"float">) => {
      const v = m().toVar();
      v.element(a.toInt()).assign(vec3(0, 0, 0));
      v.element(int(1)).element(a.toInt()).assign(float(50));
      return v.element(0).add(v.element(1)).add(v.element(2));
    });
    expect(evaluateJS((a) => write(a), [9])).toEqual([5, 7, 53]);
    expect(evaluateJS((a) => write(a), [-1])).toEqual([5, 7, 53]);
    expect(evaluateWASM((a) => write(a), [9])).toEqual([5, 7, 53]);
    expect(evaluateWASM((a) => write(a), [-1])).toEqual([5, 7, 53]);
  });

  /**
   * @canon spec-the-index-of-a-write-is-read-after-the-value-is-computed
   */
  it("reads a column index at the write, after the statements of the value written", () => {
    const write = Fn(() => {
      const v = mat3(0).toVar();
      const j = int(0).toVar();
      const value = Fn(() => {
        j.assign(j.add(1));
        return float(5);
      })();
      v.element(j).x.assign(value);
      return v.element(0).x.add(v.element(1).x.mul(10));
    });
    expect(evaluateRecording(() => write())).toBe(50);
    expect(evaluateWASM(() => write())).toBe(50);
  });

  /**
   * @canon spec-a-column-index-runs-before-a-component-index
   */
  it("runs the statements of a column index before those of a component index", () => {
    const write = Fn(() => {
      const v = mat3(0).toVar();
      const c = int(0).toVar();
      const next = Fn(() => {
        c.assign(c.add(1));
        return c.toVar();
      });
      const column = next();
      const row = next();
      v.element(column).element(row).assign(float(7));
      return v.element(1).z.mul(10).add(v.element(2).y);
    });
    expect(evaluateRecording(() => write())).toBe(70);
    expect(evaluateWASM(() => write())).toBe(70);
  });

  /**
   * @canon spec-a-constant-index-outside-a-vector-or-matrix-is-refused
   */
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
