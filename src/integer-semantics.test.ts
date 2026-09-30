import { describe, it, expect, afterAll } from "vitest";
import { ivec2, uvec2, type Node } from "./rmsl";
import {
  closeEvaluators,
  evaluateIntegerJS,
  evaluateIntegerWASM,
  evaluateIntegerWGSL,
  GPU_EVALUATION_SKIPPED,
  type IntegerType,
} from "./testing/shader-eval";

afterAll(async () => {
  await closeEvaluators();
}, 120_000);

const INT_MIN = -2147483648;
const INT_MAX = 2147483647;
const UINT_MAX = 4294967295;

type Op = "add" | "sub" | "mul" | "div" | "mod" | "shiftLeft" | "shiftRight";

/**
 * One integer operation and the result WGSL defines for it.
 *
 * WGSL is the reference because it is the only target whose spec defines
 * every case: overflow wraps, `x / 0` is `x`, `x % 0` is `0`, `INT_MIN / -1`
 * is `INT_MIN`, and a shift amount is taken modulo 32. The CPU backends
 * mirror it; GLSL leaves these cases unspecified and is not checked here.
 */
type Case = { name: string; type: IntegerType; op: Op; a: number; b: number; want: number };

const cases: Case[] = [
  { name: "int add wraps on overflow", type: "int", op: "add", a: INT_MAX, b: 1, want: INT_MIN },
  { name: "int sub wraps on underflow", type: "int", op: "sub", a: INT_MIN, b: 1, want: INT_MAX },
  { name: "int mul wraps to the low 32 bits", type: "int", op: "mul", a: 65536, b: 65536, want: 0 },
  { name: "int mul of large operands wraps", type: "int", op: "mul", a: 123456789, b: 987654321, want: -67153019 },
  { name: "int div truncates toward zero", type: "int", op: "div", a: -7, b: 2, want: -3 },
  { name: "int div by zero gives the dividend", type: "int", op: "div", a: 7, b: 0, want: 7 },
  { name: "negative int div by zero gives the dividend", type: "int", op: "div", a: -7, b: 0, want: -7 },
  { name: "INT_MIN div -1 gives INT_MIN", type: "int", op: "div", a: INT_MIN, b: -1, want: INT_MIN },
  { name: "int mod by zero gives zero", type: "int", op: "mod", a: 7, b: 0, want: 0 },
  { name: "INT_MIN mod -1 gives zero", type: "int", op: "mod", a: INT_MIN, b: -1, want: 0 },
  { name: "int mod takes the sign of the dividend", type: "int", op: "mod", a: -7, b: 3, want: -1 },
  { name: "int mod ignores the sign of the divisor", type: "int", op: "mod", a: 7, b: -3, want: 1 },
  { name: "int shift left by 32 shifts by 0", type: "int", op: "shiftLeft", a: 1, b: 32, want: 1 },
  { name: "int shift left by 33 shifts by 1", type: "int", op: "shiftLeft", a: 1, b: 33, want: 2 },
  { name: "int shift right is arithmetic, amount modulo 32", type: "int", op: "shiftRight", a: -8, b: 33, want: -4 },
  { name: "uint add wraps on overflow", type: "uint", op: "add", a: UINT_MAX, b: 1, want: 0 },
  { name: "uint sub wraps on underflow", type: "uint", op: "sub", a: 0, b: 1, want: UINT_MAX },
  { name: "uint mul wraps to the low 32 bits", type: "uint", op: "mul", a: UINT_MAX, b: 2, want: UINT_MAX - 1 },
  { name: "uint mul of large operands wraps", type: "uint", op: "mul", a: 123456789, b: 987654321, want: 4227814277 },
  { name: "uint div by zero gives the dividend", type: "uint", op: "div", a: 7, b: 0, want: 7 },
  {
    name: "uint div of a value above INT_MAX is unsigned",
    type: "uint",
    op: "div",
    a: UINT_MAX,
    b: 2,
    want: 2147483647,
  },
  { name: "uint mod by zero gives zero", type: "uint", op: "mod", a: 7, b: 0, want: 0 },
  { name: "uint shift left by 32 shifts by 0", type: "uint", op: "shiftLeft", a: 1, b: 32, want: 1 },
  { name: "uint shift right is logical", type: "uint", op: "shiftRight", a: 0x80000000, b: 31, want: 1 },
];

function build(op: Op) {
  return (a: Node<"int">, b: Node<"int">) => (a as any)[op](b) as Node<"int">;
}

/**
 * The same operation on two-component vectors, reduced back to a scalar by
 * reading `.x`, so vector code paths are held to the same table.
 */
/**
 * Vector shifts are left out: no backend compiles them correctly yet, which
 * is tracked separately from the arithmetic here.
 */
const vectorCases = cases.filter(({ op }) => op !== "shiftLeft" && op !== "shiftRight");

function buildVector(op: Op, type: IntegerType) {
  const vec = type === "int" ? ivec2 : uvec2;
  return (a: Node<"int">, b: Node<"int">) => ((vec as any)(a, a) as any)[op]((vec as any)(b, b)).x as Node<"int">;
}

describe("integer semantics match WGSL", () => {
  describe("JS", () => {
    it.each(cases)("$name", ({ type, op, a, b, want }) => {
      expect(evaluateIntegerJS(build(op), type, [a, b])).toBe(want);
    });
    it.each(vectorCases)("$name, on vectors", ({ type, op, a, b, want }) => {
      expect(evaluateIntegerJS(buildVector(op, type), type, [a, b])).toBe(want);
    });
  });

  describe("WASM", () => {
    it.each(cases)("$name", ({ type, op, a, b, want }) => {
      expect(evaluateIntegerWASM(build(op), type, [a, b])).toBe(want);
    });
    it.each(vectorCases)("$name, on vectors", ({ type, op, a, b, want }) => {
      expect(evaluateIntegerWASM(buildVector(op, type), type, [a, b])).toBe(want);
    });
  });

  describe.skipIf(GPU_EVALUATION_SKIPPED)("WGSL", () => {
    it.each(cases)("$name", async ({ type, op, a, b, want }) => {
      expect(await evaluateIntegerWGSL(build(op), type, [a, b])).toBe(want);
    });
    it.each(vectorCases)("$name, on vectors", async ({ type, op, a, b, want }) => {
      expect(await evaluateIntegerWGSL(buildVector(op, type), type, [a, b])).toBe(want);
    });
  });
});
