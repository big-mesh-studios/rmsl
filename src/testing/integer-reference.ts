/**
 * WGSL's integer semantics, written independently of every backend so the
 * sweep has something to hold them to that none of them share code with.
 *
 * Arithmetic goes through `BigInt` and is wrapped once at the end, instead of
 * the `Math.imul`/`| 0` idioms the JS backend uses: a reference that repeats
 * the implementation's tricks repeats its mistakes too.
 */

export type IntegerType = "int" | "uint";

export const INT_MIN = -2147483648;
export const INT_MAX = 2147483647;
export const UINT_MAX = 4294967295;

/** `x` reduced to a 32-bit value of `type`, the way WGSL wraps overflow. */
export function wrap(type: IntegerType, x: bigint | number): number {
  const big = typeof x === "bigint" ? x : BigInt(x);
  return Number(type === "int" ? BigInt.asIntN(32, big) : BigInt.asUintN(32, big));
}

export type BinaryOp =
  "add" | "sub" | "mul" | "div" | "mod" | "shiftLeft" | "shiftRight" | "bitAnd" | "bitOr" | "bitXor" | "min" | "max";

export type Comparison = "lessThan" | "greaterThan" | "lessThanEqual" | "greaterThanEqual" | "equal" | "notEqual";

export type UnaryOp = "negate" | "abs" | "bitNot";

export const BINARY_OPS: readonly BinaryOp[] = [
  "add",
  "sub",
  "mul",
  "div",
  "mod",
  "shiftLeft",
  "shiftRight",
  "bitAnd",
  "bitOr",
  "bitXor",
  "min",
  "max",
];

export const COMPARISONS: readonly Comparison[] = [
  "lessThan",
  "greaterThan",
  "lessThanEqual",
  "greaterThanEqual",
  "equal",
  "notEqual",
];

/** `uint` has no `negate` or `abs`, as in WGSL. */
export function unaryOps(type: IntegerType): readonly UnaryOp[] {
  return type === "int" ? ["negate", "abs", "bitNot"] : ["bitNot"];
}

/**
 * The shift amount WGSL uses: the low 5 bits of the right operand, read as
 * unsigned. WGSL only defines that for a run-time amount; a constant one of
 * 32 or more is a shader-creation error, which the compilers avoid by folding
 * or masking it first, so both have to land on this value.
 */
function shiftAmount(b: number): bigint {
  return BigInt(b & 31);
}

export function binary(type: IntegerType, op: BinaryOp, a: number, b: number): number {
  const x = BigInt(a);
  const y = BigInt(b);
  switch (op) {
    case "add":
      return wrap(type, x + y);
    case "sub":
      return wrap(type, x - y);
    case "mul":
      return wrap(type, x * y);
    case "div":
      if (b === 0) return a;
      if (type === "int" && a === INT_MIN && b === -1) return a;
      // BigInt division truncates toward zero, as WGSL's does.
      return wrap(type, x / y);
    case "mod":
      if (b === 0) return 0;
      if (type === "int" && a === INT_MIN && b === -1) return 0;
      // BigInt's remainder takes the sign of the dividend, as WGSL's does.
      return wrap(type, x % y);
    case "shiftLeft":
      return wrap(type, x << shiftAmount(b));
    case "shiftRight":
      // `x` is already signed for int and non-negative for uint, so BigInt's
      // `>>` is arithmetic and logical respectively, as each type needs.
      return wrap(type, x >> shiftAmount(b));
    case "bitAnd":
      return wrap(type, x & y);
    case "bitOr":
      return wrap(type, x | y);
    case "bitXor":
      return wrap(type, x ^ y);
    case "min":
      return a < b ? a : b;
    case "max":
      return a > b ? a : b;
  }
}

export function compare(op: Comparison, a: number, b: number): boolean {
  switch (op) {
    case "lessThan":
      return a < b;
    case "greaterThan":
      return a > b;
    case "lessThanEqual":
      return a <= b;
    case "greaterThanEqual":
      return a >= b;
    case "equal":
      return a === b;
    case "notEqual":
      return a !== b;
  }
}

export function unary(type: IntegerType, op: UnaryOp, a: number): number {
  switch (op) {
    case "negate":
      return wrap(type, -BigInt(a));
    case "abs":
      // abs(INT_MIN) is INT_MIN: the positive value does not fit, and wraps.
      return wrap(type, a < 0 ? -BigInt(a) : BigInt(a));
    case "bitNot":
      return wrap(type, ~BigInt(a));
  }
}

/** `clamp(e, low, high)`, which WGSL defines as `min(max(e, low), high)` for integers. */
export function clamp(a: number, low: number, high: number): number {
  const raised = a > low ? a : low;
  return raised < high ? raised : high;
}

/** `i32(u32)` and `u32(i32)` keep the bits, so a conversion is a wrap to the other type. */
export function convert(to: IntegerType, a: number): number {
  return wrap(to, a);
}
