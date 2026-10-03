import { describe, it, expect, afterAll } from "vitest";
import { int, ivec2, uint, uvec2, type Node } from "./rmsl";
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

type Op = "add" | "sub" | "mul" | "div" | "mod" | "shiftLeft" | "shiftRight" | "bitAnd" | "bitOr" | "bitXor" | "bitNot";

/**
 * One integer operation and the result WGSL defines for it. `b` is unused by
 * the one unary operation, `bitNot`.
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
  { name: "int bitAnd keeps the common bits", type: "int", op: "bitAnd", a: -1, b: 0x0f0f, want: 0x0f0f },
  { name: "int bitOr combines the bits", type: "int", op: "bitOr", a: 0x0f00, b: 0x00f0, want: 0x0ff0 },
  { name: "int bitXor flips the bits", type: "int", op: "bitXor", a: -1, b: 1, want: -2 },
  { name: "int bitNot flips every bit", type: "int", op: "bitNot", a: 5, b: 0, want: -6 },
  { name: "uint add wraps on overflow", type: "uint", op: "add", a: UINT_MAX, b: 1, want: 0 },
  { name: "uint sub wraps on underflow", type: "uint", op: "sub", a: 0, b: 1, want: UINT_MAX },
  { name: "uint mul wraps to the low 32 bits", type: "uint", op: "mul", a: UINT_MAX, b: 2, want: UINT_MAX - 1 },
  { name: "uint mul of large operands wraps", type: "uint", op: "mul", a: 123456789, b: 987654321, want: 4227814277 },
  { name: "uint div by zero gives the dividend", type: "uint", op: "div", a: 7, b: 0, want: 7 },
  { name: "uint div above INT_MAX is unsigned", type: "uint", op: "div", a: UINT_MAX, b: 2, want: INT_MAX },
  { name: "uint mod by zero gives zero", type: "uint", op: "mod", a: 7, b: 0, want: 0 },
  { name: "uint shift left by 32 shifts by 0", type: "uint", op: "shiftLeft", a: 1, b: 32, want: 1 },
  { name: "uint shift right is logical", type: "uint", op: "shiftRight", a: 0x80000000, b: 31, want: 1 },
  {
    name: "uint bitAnd keeps the high bit unsigned",
    type: "uint",
    op: "bitAnd",
    a: 0x80000001,
    b: 0x80000000,
    want: 0x80000000,
  },
  { name: "uint bitXor stays unsigned", type: "uint", op: "bitXor", a: UINT_MAX, b: 1, want: UINT_MAX - 1 },
  { name: "uint bitNot of zero is UINT_MAX", type: "uint", op: "bitNot", a: 0, b: 0, want: UINT_MAX },
];

function apply(op: Op, a: any, b: any): any {
  return op === "bitNot" ? a.bitNot() : a[op](b);
}

/**
 * The shapes each case runs in. The vector ones read `.x` back out, so a
 * vector code path is held to the same scalar table; `vector by scalar`
 * applies a scalar right operand to a vector, as the typed API allows.
 */
const shapes = {
  scalar: (op: Op) => (a: Node<"int">, b: Node<"int">) => apply(op, a, b) as Node<"int">,
  vector: (op: Op, type: IntegerType) => {
    const vec: any = type === "int" ? ivec2 : uvec2;
    return (a: Node<"int">, b: Node<"int">) => apply(op, vec(a, a), vec(b, b)).x as Node<"int">;
  },
  "vector by scalar": (op: Op, type: IntegerType) => {
    const vec: any = type === "int" ? ivec2 : uvec2;
    return (a: Node<"int">, b: Node<"int">) => apply(op, vec(a, a), b).x as Node<"int">;
  },
};

/**
 * Shapes with literal operands, which the compilers fold or pass to the
 * target as constants instead of computing at run time. `literal right
 * operand` keeps `a` a run-time value (the only argument) and makes `b` a
 * literal, as in `a.div(int(0))`.
 */
const literalShapes = {
  constant: (c: Case) => {
    const literal: any = c.type === "int" ? int : uint;
    return () => apply(c.op, literal(c.a), literal(c.b)) as Node<"int">;
  },
  "constant vector": (c: Case) => {
    const vec: any = c.type === "int" ? ivec2 : uvec2;
    return () => apply(c.op, vec(c.a, c.a), vec(c.b, c.b)).x as Node<"int">;
  },
  "literal right operand": (c: Case) => {
    const literal: any = c.type === "int" ? int : uint;
    return (a: Node<"int">) => apply(c.op, a, literal(c.b)) as Node<"int">;
  },
  "vector by literal vector": (c: Case) => {
    const vec: any = c.type === "int" ? ivec2 : uvec2;
    return (a: Node<"int">) => apply(c.op, vec(a, a), vec(c.b, c.b)).x as Node<"int">;
  },
};

const runs = [
  ...Object.entries(shapes).flatMap(([shape, make]) =>
    cases.map((c) => ({ ...c, shape, build: make(c.op, c.type), args: [c.a, c.b] })),
  ),
  ...Object.entries(literalShapes).flatMap(([shape, make]) =>
    cases.map((c) => ({ ...c, shape, build: make(c), args: shape.startsWith("constant") ? [] : [c.a] })),
  ),
];

describe("integer semantics match WGSL", () => {
  /**
   * @canon spec-js-integer-arithmetic-follows-wgsl
   */
  it.each(runs)("JS: $name ($shape)", ({ type, args, want, build }) => {
    expect(evaluateIntegerJS(build, type, args)).toBe(want);
  });

  /**
   * @canon spec-wasm-integer-arithmetic-follows-wgsl
   */
  it.each(runs)("WASM: $name ($shape)", ({ type, args, want, build }) => {
    expect(evaluateIntegerWASM(build, type, args)).toBe(want);
  });

  /**
   * @canon spec-wgsl-integer-arithmetic-keeps-its-defined-result
   */
  it.skipIf(GPU_EVALUATION_SKIPPED).each(runs)("WGSL: $name ($shape)", async ({ type, args, want, build }) => {
    expect(await evaluateIntegerWGSL(build, type, args)).toBe(want);
  });
});
