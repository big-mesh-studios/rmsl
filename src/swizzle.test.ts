import { describe, it, expect, afterAll } from "vitest";
import { bvec4, vec2, vec4 } from "./rmsl";
import { compileWgsl } from "./wgsl";
import { assertRecordedEvaluationsAgree, closeEvaluators, evaluateRecording } from "./testing/shader-eval";

afterAll(async () => {
  await assertRecordedEvaluationsAgree();
  await closeEvaluators();
}, 120_000);

describe("swizzles", () => {
  /**
   * @canon spec-a-swizzle-reads-the-components-it-names
   */
  it("read any components in any order, with repeats, as on every backend", () => {
    expect(evaluateRecording(() => vec4(1, 2, 3, 4).wzyx)).toEqual([4, 3, 2, 1]);
    expect(evaluateRecording(() => vec4(1, 2, 3, 4).zz)).toEqual([3, 3]);
    expect(evaluateRecording(() => vec4(1, 2, 3, 4).agr)).toEqual([4, 2, 1]);
    expect(evaluateRecording(() => vec2(5, 6).ts)).toEqual([6, 5]);
    expect(evaluateRecording(() => vec2(5, 6).xxyy)).toEqual([5, 5, 6, 6]);
  });

  /**
   * @canon spec-a-swizzle-reads-the-components-it-names
   */
  it("reach boolean vectors too", () => {
    const wgsl = compileWgsl.fragment(vec4(bvec4(true, false, true, false).wx.select(vec2(1, 1), vec2(0, 0)), 0, 1));
    expect(wgsl).toContain(".wx");
  });
});
