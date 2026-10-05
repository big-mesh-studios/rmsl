import { describe, it, expect, afterAll } from "vitest";
import { int, ivec2, uint } from "../rmsl";
import { GPU_EVALUATION_SKIPPED } from "./shader-eval";
import { releaseGpu } from "./gpu";
import { sweepGLSLValidity, sweepJS, sweepWASM, sweepWGSL, type SweepCase } from "./integer-sweep";

afterAll(async () => {
  await releaseGpu();
}, 120_000);

/**
 * Each case is right or wrong on purpose. A harness that passes the wrong
 * ones, or fails the right ones, would make every sweep built on it
 * meaningless — and one that reads unwritten output as zeros passes every
 * case whose answer is zero.
 */
const right: SweepCase[] = [
  {
    label: "constant",
    type: "uint",
    width: 1,
    paramTypes: [],
    build: () => uint(5).add(uint(1)),
    runs: [{ args: [], want: [6] }],
  },
  {
    label: "run-time",
    type: "int",
    width: 1,
    paramTypes: ["int", "int"],
    build: (a, b) => a.sub(b),
    runs: [
      { args: [7, 2], want: [5] },
      { args: [-2147483648, 1], want: [2147483647] },
    ],
  },
  {
    label: "vector",
    type: "int",
    width: 2,
    paramTypes: ["int"],
    build: (a) => ivec2(a, int(3)),
    runs: [{ args: [-4], want: [-4, 3] }],
  },
];

const wrong: SweepCase[] = [
  {
    label: "wrong constant",
    type: "uint",
    width: 1,
    paramTypes: [],
    build: () => uint(5),
    runs: [{ args: [], want: [0] }],
  },
  {
    label: "wrong second lane",
    type: "int",
    width: 2,
    paramTypes: ["int"],
    build: (a) => ivec2(a, int(3)),
    runs: [{ args: [1], want: [1, 4] }],
  },
];

describe("integer sweep harness", () => {
  /**
   * @canon spec-the-integer-sweep-tells-right-from-wrong
   */
  it("JS passes right cases and reports wrong ones", () => {
    expect(sweepJS(right)).toEqual([]);
    expect(sweepJS(wrong).map((m) => m.label)).toEqual(["wrong constant", "wrong second lane"]);
  });
  /**
   * @canon spec-the-integer-sweep-tells-right-from-wrong
   */
  it("WASM passes right cases and reports wrong ones", () => {
    expect(sweepWASM(right)).toEqual([]);
    expect(sweepWASM(wrong).map((m) => m.label)).toEqual(["wrong constant", "wrong second lane"]);
  });
  /**
   * @canon spec-the-integer-sweep-tells-right-from-wrong
   */
  it.skipIf(GPU_EVALUATION_SKIPPED)("WGSL passes right cases and reports wrong ones", async () => {
    expect(await sweepWGSL(right)).toEqual([]);
    expect((await sweepWGSL(wrong)).map((m) => m.label)).toEqual(["wrong constant", "wrong second lane"]);
  });
  /**
   * @canon spec-the-integer-sweep-tells-right-from-wrong
   */
  it.skipIf(GPU_EVALUATION_SKIPPED)("WGSL runs a batch in which no case takes arguments", async () => {
    // With no case reading the argument buffer, its binding is dropped from
    // the pipeline; the dispatch must still run rather than read back zeros.
    expect(await sweepWGSL([right[0]!])).toEqual([]);
  });
  /**
   * @canon spec-the-integer-sweep-tells-right-from-wrong
   */
  it.skipIf(GPU_EVALUATION_SKIPPED)("GLSL accepts valid programs", async () => {
    expect(await sweepGLSLValidity(right)).toEqual([]);
  });
});
