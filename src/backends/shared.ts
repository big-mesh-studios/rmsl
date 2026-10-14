import { ProgramError, UsageError, type BackendOrigin, type Origin } from "../errors";
import { safeName } from "./reserved-names";
import { withArticle } from "../article";
import {
  assertCallFits,
  assertFnIsClosed,
  assertNodeNames,
  assertForUpdateNode,
  type FnDefinition,
  BaseNode,
  LOOP_JUMP_STATEMENTS,
  MATRIX_DIMENSIONS,
  Node,
  NodeImpl,
  ShaderType,
  StorageAccess,
  StorageNode,
  StorageBufferAttribute,
  TYPE_WIDTH,
  node,
  operandRefusal,
  someNode,
  swizzleWritePlace,
  var_,
  varying,
  vec4,
} from "../core";
import { componentCountOf, componentKindOf } from "./cpu";
import { isInlineFn } from "./share";
/**
 * What compiling one node yields: statements to emit, how to refer to it, and
 * its operator precedence (higher = tighter binding, for bracket reduction).
 *
 * `prec` is omitted for atoms (literals, variables, function calls) — callers
 * default it to `PREC_ATOM` so they never need wrapping.
 */
export interface CompiledNode {
  decls: string[];
  body: string[];
  expr: string;
  prec?: number;
  /** JS target: the {@link CompileCtx.jsEpoch} a value reading a filled slot was compiled at. */
  jsEpoch?: number;
  /** JS target: the block a copy of a host input was made in, which a write leaves valid. */
  jsBlock?: number;
}

/**
 * Precedence values for bracket reduction. Higher number = tighter binding.
 *
 * GLSL and WGSL share the same relative ordering, so one table covers both.
 */
export const PRECEDENCE: Record<string, number> = {
  // The ternary is looser than every binary operator — the branch runs across
  // the whole conditional — so it binds loosest of all.
  select: 5,
  or: 10,
  and: 20,
  bitOr: 30,
  bitXor: 40,
  bitAnd: 50,
  equal: 60,
  notEqual: 60,
  lessThan: 60,
  greaterThan: 60,
  lessThanEqual: 60,
  greaterThanEqual: 60,
  shiftLeft: 70,
  shiftRight: 70,
  add: 80,
  sub: 80,
  mul: 90,
  div: 90,
  mod: 90,
};

/** Precedence for unary operators (negate, not). Tighter than all binary ops. */
export const PREC_UNARY = 100;

/** Precedence for atoms — never needs wrapping. */
export const PREC_ATOM = 200;

/**
 * Wrap a child expression in parens when its precedence is lower than (or equal
 * to) the parent operator's, otherwise the child would be parsed differently.
 */
export function wrapExpr(childPrec: number | undefined, parentPrec: number, expr: string): string {
  return (childPrec ?? PREC_ATOM) <= parentPrec ? `(${expr})` : expr;
}

export interface CompileCtx extends BackendOrigin {
  nextId: number;
  shaderStage: "vertex" | "fragment" | "compute";
  /**
   * The uniforms the program reads, by name: two nodes with one name are one
   * uniform. `length` is set only for uniform arrays, and gives their element count.
   * WGSL also records each uniform's node and its `order` of creation. GLSL
   * sets `precision` on a uniform that carries a qualifier of its own.
   */
  uniforms: Map<
    string,
    {
      type: string;
      slot: string;
      length?: number;
      order?: number;
      node?: BaseNode<ShaderType>;
      precision?: "lowp" | "mediump" | "highp";
    }
  >;
  /** The storage buffers the program reads, by slot; `order` is the creation order of the buffer. */
  storages?: Map<
    string,
    {
      name: string;
      type: string;
      access: "read" | "write" | "read_write";
      wgslName: string;
      order: number;
    }
  >;
  /**
   * The shader type of one element of each storage buffer the program reads,
   * by slot. Every target fills it: WGSL needs it to lay a buffer out, and a
   * compute adapter needs it to count the elements of the array the host
   * passes for a slot.
   */
  storageTypes?: Map<string, string>;
  /** The attributes the program reads, by name; `id` is the first one's, which orders them as they were made. */
  attributes: Map<string, { id: number; type: string; shaderType: string; slot: string }>;
  /** The varyings the program reads or writes, by name. */
  varyings: Map<string, { id: number; type: string; slot: string }>;
  outputs: Map<number, { type: string; slot: string; location: number }>;
  wgslSamplers: Map<string, { textureSlot: string; samplerSlot: string }>;
  varDefs: Map<string, string>;
  /**
   * What each node already compiled to, keyed by the node itself.
   *
   * The graph is a directed acyclic graph, not a tree: `Fn` returning an array
   * gives every element the whole block scope, so one node is reachable from
   * several roots — as the same object, not a copy. Compiling it once per root
   * repeats whatever it does, which for a declaration is a redefinition and for
   * an assignment or a loop is the work happening twice.
   *
   * So a node is compiled the first time it is reached and its statements are
   * emitted there. Later arrivals get its expression alone, since the
   * statements producing that expression are already in the output.
   */
  memo: Map<BaseNode<ShaderType>, CompiledNode>;
  /**
   * Names of WGSL helper functions the shader needs, emitted ahead of the entry
   * point. GLSL provides some builtins that WGSL does not, so they are written
   * out on demand rather than always.
   */
  wgslHelpers: Set<string>;
  /** GLSL helper functions the shader needs, by name, emitted ahead of `main`. */
  glslHelpers?: Map<string, string>;
  /**
   * Whether the program assigned the position itself. A vertex stage that has
   * done so needs no implicit write, and its result is free to be anything.
   */
  positionWritten: boolean;
  inFn: boolean;
  fragDepthUsed: boolean;
  /** Whether a fragment stage assigns its depth, where `fragDepthUsed` holds for a read too. */
  fragDepthWritten?: boolean;
  /** Whether the shader reads the fragment's screen position. */
  fragCoordUsed: boolean;
  /** What an early `Return()` compiles to: the function's zero value, in the form its last return takes. */
  returnStatement?: string;
  /**
   * The declarations of the variables the update of a GLSL `for` makes, while
   * that update compiles. The update is an expression, so each variable is
   * declared ahead of the loop and assigned in the update.
   */
  forUpdateDeclarations?: string[];
  /** Whether a vertex stage reads `vertexIndex()` / `instanceIndex()`. */
  vertexIndexUsed?: boolean;
  instanceIndexUsed?: boolean;
  /** Fn parameter names, which the JS target reads from `ctx.params`. */
  jsParams: Set<string>;
  /** Names of JS helper functions the compiled function needs. */
  jsHelpers: Set<string>;
  /** The constant vectors and matrices a JS function reads, by the name each is declared under once. */
  jsConstants?: Map<string, string>;
  /** The kinds of input a JS function reads by name, `uniforms` or `varyings`, each read into a local once a call. */
  jsInputKinds?: Set<string>;
  /**
   * Slot a vector/matrix-typed expression should be written into, when the JS
   * target is lowering an assignment. `null` means a plain expression.
   */
  outTarget: string | null;
  /**
   * JS target: how many writes and block boundaries compiling has passed. A
   * value computed into a slot holds what it was there, so its memo entry is
   * only reused while this is the count it was compiled at.
   */
  jsEpoch: number;
  /** JS target: the blocks being compiled, outermost first; a copy of a host input is valid inside its own. */
  jsBlocks?: number[];
  /**
   * JS target: inside a loop, the copies of host inputs to make before the
   * outermost one, and the block that loop is in, where those copies hold.
   */
  jsLoopCopies?: { lines: string[]; block: number };
  /** JS target: whether the value being compiled reads a slot that statements filled. */
  jsReadsSlot: boolean;
  /** What derivative ops (dFdx/dFdy/fwidth) compile to on the CPU. */
  derivatives: "throw" | "zero";
  /** Per-call variable bindings instead of hoisted per-program scratch. */
  reentrant: boolean;
  /** JS target: whether the program computes every float in 32 bits, as `float: "f32"` asks. */
  jsFloat32?: boolean;
  /** JS target: the node being compiled as the target of an assignment, which is written, not read. */
  jsTarget?: unknown;
  /**
   * GLSL and WGSL: the functions the program declares for its `Fn`s with a
   * layout, shared by the context of the program and the context of each body.
   */
  userFns?: UserFns;
  /** JS target: the labels of the `scope`s being compiled, innermost last; a `Return()` leaves the innermost. */
  jsScopes?: string[];
}

/** The functions a program declares for its `Fn`s with a layout. */
export interface UserFns {
  /** The name each function is declared under, by its body. */
  names: Map<BaseNode<ShaderType>, string>;
  /** The names taken in the program. */
  taken: Set<string>;
  /** The text of each declared function, a function ahead of any function that calls it. */
  declarations: string[];
}

// === Constant folding ===
export function isLeafLiteral(n: BaseNode<ShaderType>): boolean {
  return (n.type === "float" || n.type === "int" || n.type === "uint" || n.type === "bool") && !n.params;
}

/**
 * An integer operation on literal operands, computed the way every backend
 * computes it at run time: wrapped to 32 bits, with WGSL's results for
 * division and remainder by zero and for `INT_MIN / -1`, shift amounts taken
 * modulo 32, and `clamp` as `min(max(e, low), high)`. `undefined` for an
 * operation this doesn't fold.
 */
function foldInteger(op: string, t: "int" | "uint", [a = 0, b = 0, c = 0]: number[]): number | undefined {
  let wrap = t === "int" ? (x: number) => x | 0 : (x: number) => x >>> 0;
  switch (op) {
    case "add":
      return wrap(a + b);
    case "sub":
      return wrap(a - b);
    case "mul":
      return wrap(Math.imul(a, b));
    case "div":
      return b === 0 || (t === "int" && a === -2147483648 && b === -1) ? a : wrap(Math.trunc(a / b));
    case "mod":
      return b === 0 ? 0 : wrap(a % b);
    case "negate":
      return wrap(-a);
    case "abs":
      return wrap(Math.abs(a));
    case "bitAnd":
      return wrap(a & b);
    case "bitOr":
      return wrap(a | b);
    case "bitXor":
      return wrap(a ^ b);
    case "bitNot":
      return wrap(~a);
    case "shiftLeft":
      return wrap(a << b);
    case "shiftRight":
      return t === "int" ? a >> b : a >>> b;
    case "min":
      return Math.min(a, b);
    case "max":
      return Math.max(a, b);
    case "clamp":
      return Math.min(Math.max(a, b), c);
  }
  return undefined;
}

const INTEGER_COMPARISONS: Record<string, (a: number, b: number) => boolean> = {
  lessThan: (a, b) => a < b,
  greaterThan: (a, b) => a > b,
  lessThanEqual: (a, b) => a <= b,
  greaterThanEqual: (a, b) => a >= b,
  equal: (a, b) => a === b,
  notEqual: (a, b) => a !== b,
};

/** A component of a literal that may be a splat: one value stands for every component. */
function componentAt(components: number[], i: number): number {
  return components[components.length > 1 ? i : 0]!;
}

/** Whether `t` is `int`, `uint`, or a vector of either. */
export function isIntegerType(t: string): boolean {
  let kind = componentKindOf(t);
  return kind === "int" || kind === "uint";
}

function isUnsignedType(t: string): boolean {
  return componentKindOf(t) === "uint";
}

/**
 * The components of an integer literal — a scalar, a vector literal, or a
 * vector constructor whose parts are all constant — or `null` when `n` isn't
 * one. A part may be a constant expression, and may have the other
 * signedness: its bits are kept, as the conversion does at run time. One
 * scalar part fills every component.
 */
function integerLiteralComponents(origin: Origin, n: BaseNode<ShaderType>): number[] | null {
  let t = n._t as string;
  if (!isIntegerType(t)) return null;
  if (isLeafLiteral(n)) return [n.value as number];
  if (n.type === t && Array.isArray(n.value)) return n.value as number[];
  let width = TYPE_WIDTH[t] ?? 1;
  if (width === 1 || n.type !== "construct" || !n.params?.length) return null;
  let parts: number[] = [];
  for (let param of n.params) {
    let constant = integerConstant(origin, param);
    let components = constant && integerLiteralComponents(origin, constant);
    if (!components) return null;
    parts.push(...components);
  }
  if (n.params.length === 1 && parts.length === 1) parts = Array(width).fill(parts[0]);
  if (parts.length !== width) return null;
  let wrap = isUnsignedType(t) ? (x: number) => x >>> 0 : (x: number) => x | 0;
  return parts.map(wrap);
}

const integerConstants = new WeakMap<object, BaseNode<ShaderType> | null>();

/**
 * `n` as an integer literal when it is a constant integer expression, or
 * `null`. Folding happens from the leaves up, so `a.div(int(2).sub(int(2)))`
 * sees its divisor as the literal `0` rather than as an unfolded subtraction.
 * Cached per node: a node is asked for by each of its parents.
 */
function integerConstant(origin: Origin, n: BaseNode<ShaderType>): BaseNode<ShaderType> | null {
  if (!n || typeof n !== "object" || !isIntegerType(n._t as string)) return null;
  if (integerLiteralComponents(origin, n)) return n;
  let cached = integerConstants.get(n);
  if (cached !== undefined) return cached;
  integerConstants.set(n, null);
  let folded = tryFold(origin, n);
  let result = folded && folded !== n ? integerConstant(origin, folded) : null;
  integerConstants.set(n, result);
  return result;
}

function integerLiteral(t: string, components: number[]): BaseNode<ShaderType> {
  return mkNode({ _t: t, type: t, value: (TYPE_WIDTH[t] ?? 1) > 1 ? components : components[0] });
}

/**
 * The result of a comparison between integer constants, one boolean per
 * component, or `null` when `n` isn't one.
 */
function integerComparison(origin: Origin, n: BaseNode<ShaderType>): boolean[] | null {
  let compare = INTEGER_COMPARISONS[n.type];
  let [lhs, rhs] = n.params ?? [];
  if (!compare || !lhs || !rhs) return null;
  let a = integerConstant(origin, lhs);
  let b = integerConstant(origin, rhs);
  let ca = a && integerLiteralComponents(origin, a);
  let cb = b && integerLiteralComponents(origin, b);
  if (!ca || !cb) return null;
  return Array.from({ length: Math.max(ca.length, cb.length) }, (_, i) =>
    compare(componentAt(ca, i), componentAt(cb, i)),
  );
}

/**
 * A select whose condition is a constant: the chosen branch, or — for a
 * condition that differs between components — the components of two
 * constant branches picked one by one. WGSL evaluates a constant condition
 * itself, so a branch it picks is as constant as a literal would be.
 */
function foldSelect(origin: Origin, n: BaseNode<ShaderType>): BaseNode<ShaderType> | null {
  let [cond, ifTrue, ifFalse] = n.params ?? [];
  if (!cond || !ifTrue || !ifFalse) return null;
  let picks = isLeafLiteral(cond) ? [Boolean(cond.value)] : integerComparison(origin, cond);
  if (!picks) return null;
  if (picks.every((p) => p === picks![0])) return picks[0] ? ifTrue : ifFalse;
  let a = integerLiteralComponents(origin, ifTrue);
  let b = integerLiteralComponents(origin, ifFalse);
  if (!a || !b) return null;
  return integerLiteral(
    n._t as string,
    picks.map((p, i) => (p ? componentAt(a!, i) : componentAt(b!, i))),
  );
}

/**
 * Integer operations on literals, folded component by component, and a
 * literal divisor or shift amount rewritten to the value the run-time rules
 * already use: a zero divisor becomes 1 (`x / 1` is `x`, `x % 1` is `0`) and a
 * shift amount keeps its low 5 bits. WGSL rejects the unrewritten constants at
 * shader creation, even when the other operand is a run-time value.
 */
function foldIntegerOperands(origin: Origin, n: BaseNode<ShaderType>): BaseNode<ShaderType> | null {
  let t = n._t as string;
  let [lhs, rhs] = n.params ?? [];
  if (!isIntegerType(t) || !lhs) return null;
  let kind: "int" | "uint" = isUnsignedType(t) ? "uint" : "int";
  let operands = n.params!.map((param) => integerLiteralComponents(origin, param));
  if (operands.every((c) => c !== null)) {
    let folded = Array.from({ length: TYPE_WIDTH[t] ?? 1 }, (_, i) =>
      foldInteger(
        n.type,
        kind,
        operands.map((c) => componentAt(c!, i)),
      ),
    );
    if (folded.every((v) => v !== undefined)) return integerLiteral(t, folded as number[]);
  }
  let b = rhs ? integerLiteralComponents(origin, rhs) : null;
  if (!rhs || !b) return null;
  let rewrite =
    n.type === "div" || n.type === "mod"
      ? (v: number) => (v === 0 ? 1 : v)
      : n.type === "shiftLeft" || n.type === "shiftRight"
        ? (v: number) => v & 31
        : null;
  if (!rewrite || b.every((v) => rewrite!(v) === v)) return null;
  return mkNode({ _t: t, type: n.type, params: [lhs, integerLiteral(rhs._t as string, b.map(rewrite))] });
}

/**
 * `n` folded to the literal it computes, or null. With `float32`, a float
 * literal operand and a float result are rounded to 32 bits, as a CPU target at
 * `float: "f32"` computes them at run time.
 */
export function tryFold(origin: Origin, n: BaseNode<ShaderType>, float32 = false): BaseNode<ShaderType> | null {
  let folded = float32 ? foldNode32(origin, n) : foldNode(origin, n);
  if (
    folded &&
    folded._t === "float" &&
    folded.type === "float" &&
    n.type !== "float" &&
    n.type !== "select" &&
    !Number.isFinite(Math.fround(folded.value as number))
  ) {
    throw new ProgramError(
      "constant-not-finite",
      `${n.type}() of literal operands is ${folded.value}, which a 32-bit float cannot hold and no target can write as a constant. ` +
        `Compute it from a value that is not a literal, such as a uniform.`,
      origin,
    );
  }
  return folded;
}

function foldNode32(origin: Origin, n: BaseNode<ShaderType>): BaseNode<ShaderType> | null {
  let params = n.params?.map((p) => (p && isLeafLiteral(p) ? roundLiteral32(p) : p));
  let rounded = params?.some((p, i) => p !== n.params![i])
    ? mkNode({ _t: n._t as string, type: n.type, params, value: n.value })
    : n;
  let folded = foldNode(origin, rounded);
  return folded && isLeafLiteral(folded) ? roundLiteral32(folded) : folded;
}

/** A float literal, scalar, vector or matrix, with each value rounded to 32 bits; any other node as it is. */
function roundLiteral32(n: BaseNode<ShaderType>): BaseNode<ShaderType> {
  let t = n._t as string;
  if (t !== "float" && !/^(vec|mat)/.test(t)) return n;
  let value = Array.isArray(n.value) ? (n.value as number[]).map(Math.fround) : Math.fround(n.value as number);
  return mkNode({ _t: t, type: n.type, value });
}

function foldNode(origin: Origin, n: BaseNode<ShaderType>): BaseNode<ShaderType> | null {
  // Integer operands are folded first, so a constant subexpression counts as
  // the literal it is when this node is folded or its divisor is rewritten.
  let operands = n.params?.map((p) => integerConstant(origin, p) ?? p);
  if (operands?.some((p, i) => p !== n.params![i])) {
    n = mkNode({ _t: n._t as string, type: n.type, params: operands, value: n.value });
  }
  // A select with a constant condition collapses to the chosen branch,
  // whatever the branches are — the guard below only admits scalar literals,
  // so this is checked before it.
  if (n.type === "select") return foldSelect(origin, n);
  let integerOperands = foldIntegerOperands(origin, n);
  if (integerOperands) return integerOperands;
  let params = n.params ?? [];
  if (!params.every(isLeafLiteral)) return null;
  let p0 = params[0]?.value;
  let p1 = params[1]?.value;
  let t = n._t;
  if (t === "float" || t === "int" || t === "uint") {
    let a = p0 as number;
    let b = p1 as number;
    switch (n.type) {
      case "add":
        return mkNode({ _t: t, type: t, value: a + b });
      case "sub":
        return mkNode({ _t: t, type: t, value: a - b });
      case "mul":
        return mkNode({ _t: t, type: t, value: a * b });
      case "div":
        return mkNode({ _t: t, type: t, value: a / b });
      case "negate":
        return mkNode({ _t: t, type: t, value: -a });
      // JavaScript's % truncates toward zero. The float operation is floored,
      // following GLSL's mod(), so folding it with % would give a literal that
      // disagrees with what the same expression computes when its operands are
      // not constants.
      case "mod":
        return mkNode({ _t: t, type: t, value: a - b * Math.floor(a / b) });
      case "sin":
        return mkNode({ _t: t, type: t, value: Math.sin(a) });
      case "cos":
        return mkNode({ _t: t, type: t, value: Math.cos(a) });
      case "tan":
        return mkNode({ _t: t, type: t, value: Math.tan(a) });
      case "asin":
        return mkNode({ _t: t, type: t, value: Math.asin(a) });
      case "acos":
        return mkNode({ _t: t, type: t, value: Math.acos(a) });
      case "atan":
        return mkNode({ _t: t, type: t, value: Math.atan(a) });
      case "sinh":
        return mkNode({ _t: t, type: t, value: Math.sinh(a) });
      case "cosh":
        return mkNode({ _t: t, type: t, value: Math.cosh(a) });
      case "tanh":
        return mkNode({ _t: t, type: t, value: Math.tanh(a) });
      case "asinh":
        return mkNode({ _t: t, type: t, value: Math.asinh(a) });
      case "acosh":
        return mkNode({ _t: t, type: t, value: Math.acosh(a) });
      case "atanh":
        return mkNode({ _t: t, type: t, value: Math.atanh(a) });
      case "abs":
        return mkNode({ _t: t, type: t, value: Math.abs(a) });
      case "sign":
        return mkNode({ _t: t, type: t, value: Math.sign(a) });
      case "floor":
        return mkNode({ _t: t, type: t, value: Math.floor(a) });
      case "ceil":
        return mkNode({ _t: t, type: t, value: Math.ceil(a) });
      case "round":
        return mkNode({ _t: t, type: t, value: roundHalfToEven(a) });
      case "trunc":
        return mkNode({ _t: t, type: t, value: Math.trunc(a) });
      case "radians":
        return mkNode({ _t: t, type: t, value: a * RADIANS_PER_DEGREE });
      case "degrees":
        return mkNode({ _t: t, type: t, value: a * DEGREES_PER_RADIAN });
      case "fract":
        return mkNode({ _t: t, type: t, value: a - Math.floor(a) });
      case "sqrt":
        return mkNode({ _t: t, type: t, value: Math.sqrt(a) });
      case "inverseSqrt":
        return mkNode({ _t: t, type: t, value: 1 / Math.sqrt(a) });
      case "atan2":
        return mkNode({ _t: t, type: t, value: Math.atan2(a, b) });
      case "exp":
        return mkNode({ _t: t, type: t, value: Math.exp(a) });
      case "log":
        return mkNode({ _t: t, type: t, value: Math.log(a) });
      case "exp2":
        return mkNode({ _t: t, type: t, value: Math.pow(2, a) });
      case "log2":
        return mkNode({ _t: t, type: t, value: Math.log2(a) });
      case "pow":
        return mkNode({ _t: t, type: t, value: Math.pow(a, b) });
      case "min":
        return mkNode({ _t: t, type: t, value: Math.min(a, b) });
      case "max":
        return mkNode({ _t: t, type: t, value: Math.max(a, b) });
      case "dot":
        return mkNode({ _t: t, type: t, value: a * b });
    }
  }
  return null;
}

export function mkNode(config: {
  _t?: string;
  type: string;
  params?: BaseNode<ShaderType>[];
  value?: unknown;
}): BaseNode<ShaderType> {
  return new NodeImpl({
    _t: config._t ?? config.type,
    type: config.type,
    params: config.params,
    value: config.value,
  }) as BaseNode<ShaderType>;
}

/** Drop a trailing semicolon, for the slots that take an expression. */
export function withoutSemicolon(statement: string): string {
  return statement.endsWith(";") ? statement.slice(0, -1) : statement;
}

/** The lines opening a loop's body that run a compiled condition, then break when its test fails. */
export function loopGuard(cond: CompiledNode): string[] {
  return [...cond.body, `if (!(${cond.expr})) { break; }`];
}

/**
 * How a loop tests a compiled condition: the expression its header holds, and
 * the lines opening its body. A condition that runs statements before its test
 * can't sit in a header, so the header holds `true`, and the body opens with
 * the guard.
 */
export function loopTest(cond: CompiledNode): { header: string; guard: string[] } {
  if (cond.body.length === 0) return { header: cond.expr, guard: [] };
  return { header: "true", guard: loopGuard(cond) };
}

/**
 * Reject a vertex stage whose result cannot reach the position output.
 *
 * That output is a vec4 and writing it is not optional, so a vertex shader
 * returning anything else is unambiguously a mistake: a value was produced and
 * has nowhere to go. Skipping the write instead would link cleanly and draw
 * nothing, which is the silent-corruption failure mode the unhandled-node case
 * throws to avoid.
 *
 * A fragment stage may return nothing, since a shader with no colour output is
 * legal. Its result becomes the colour through {@link fragmentColour}, so only a
 * type with no conversion to one is refused here.
 */
export function assertStageResult(
  origin: Origin,
  shaderStage: "vertex" | "fragment" | "compute",
  lastType: string | undefined,
  positionWritten: boolean,
  declaresOutput = false,
): void {
  if (shaderStage === "fragment") {
    assertFragmentResult(origin, lastType, declaresOutput);
    return;
  }
  if (shaderStage !== "vertex") return;
  // The program set the position itself, so its result has nowhere it needs to
  // go and can be anything, including nothing.
  if (positionWritten) return;
  // Otherwise the result becomes the position, and has to be able to.
  if (lastType === "vec4") return;
  // Whether a stage produced a value is a question about its type, not the text
  // it compiled to. A vertex shader returning zero or returning nothing both
  // fail the same check.
  throw new ProgramError(
    "vertex-without-position",
    `A vertex shader has to produce a position. This one ` +
      (lastType === undefined || lastType === "void"
        ? `returns nothing and never assigns builtinPosition(). Return a vec4, or ` +
          `assign builtinPosition() yourself.`
        : `returns ${lastType}, which cannot become one. Wrap it — for example ` + `vec4(value, 1.0).`),
    origin,
  );
}

function assertFragmentResult(origin: Origin, lastType: string | undefined, declaresOutput: boolean): void {
  if (declaresOutput) return;
  if (lastType === undefined || lastType === "void" || lastType === "vec4") return;
  throw new ProgramError(
    "fragment-result-not-a-colour",
    `A fragment shader that declares no output writes its result as the colour. ` +
      `This one returns ${lastType}, which has no colour to become. Return a vec4, or declare an output.`,
    origin,
  );
}

/**
 * The roots of a fragment stage with an `outputStruct` it returns written out:
 * member `i` becomes an assignment to an output at location `i`, in slot
 * `_rmsl_out<i>`. Only the last value of a stage can be one.
 */
function lowerOutputStruct<T extends BaseNode<ShaderType>>(origin: Origin, roots: readonly T[]): T[] {
  const nothing = () => node({ _t: "void", type: "void" }) as BaseNode<ShaderType>;
  /** The assignments of the members, which end in nothing: the stage returns no value. */
  const written = (struct: BaseNode<ShaderType>): BaseNode<ShaderType>[] => [
    ...struct.params!.map((member, i) => {
      const target = node({
        _t: member._t,
        type: "output",
        value: { id: i, slot: `_rmsl_out${i}`, shaderType: member._t, location: i },
      });
      return node({
        _t: "void",
        type: "assign",
        params: [target as BaseNode<ShaderType>, member],
      }) as BaseNode<ShaderType>;
    }),
    nothing(),
  ];
  const lower = (n: BaseNode<ShaderType>): BaseNode<ShaderType> => {
    if (n.type === "outputStruct") return node({ _t: "void", type: "seq", params: written(n) });
    if (n.type !== "seq") return n;
    const params = n.params!;
    const last = params[params.length - 1]!;
    // The assignments join the statements before them, where every backend runs them.
    if (last.type === "outputStruct")
      return node({ _t: "void", type: "seq", params: [...params.slice(0, -1), ...written(last)] });
    const lowered = lower(last);
    return lowered === last ? n : node({ _t: lowered._t, type: "seq", params: [...params.slice(0, -1), lowered] });
  };
  const lowered = roots.map((root, i) => (i === roots.length - 1 ? (lower(root) as T) : root));
  if (someNode(lowered, (n) => n.type === "outputStruct")) {
    throw new ProgramError(
      "output-struct-misplaced",
      "outputStruct is the value a fragment stage returns. It cannot be an operand or a statement.",
      origin,
    );
  }
  return lowered;
}

/** Why a `Break()` or a `Continue()` with no loop around it is refused, for each jump. */
export const LOOP_JUMP_OUTSIDE_A_LOOP_MESSAGE: Record<"break" | "continue", string> = {
  break: '"Break" outside a loop: no loop is around it for it to leave. Call it inside a loop body.',
  continue: '"Continue" outside a loop: no loop is around it to start again. Call it inside a loop body.',
};

/**
 * Whether slot `i` of `statement` is a block of the program, whose statements
 * run where the statement puts them: an item of a `seq`, a branch of an `If`,
 * or the init of a `For`. A loop's other slots run inside it. Any other slot
 * holds a value, and a value with statements of its own is an inline `Fn`.
 */
const isBlockSlot = (statement: any, i: number) =>
  statement.type === "seq" || (statement.type === "if" && i > 0) || (statement.type === "for" && i === 0);

/** Refuses a storage node read as a whole value: a whole buffer has no value a program computes with. */
function refuseWholeStorageRead(origin: Origin, storage: any): never {
  throw new ProgramError(
    "storage-read-as-a-whole",
    `storage "${storage.value.slot}" is read as a whole; read one element with .element(i).`,
    origin,
  );
}

/**
 * Checks every node of `roots`, in one walk, for the refusals every compiler
 * shares: an operation on operands it does not take, see {@link operandRefusal},
 * a storage node read as a whole rather than through `.element(i)`
 * (an assignment to one has its own refusal, in `assertAssignable`), a node of
 * a `for` update that the update slot of a `for` cannot take, and a write by
 * index through a swizzle that its read would refuse. It refuses a `Break()`
 * or a `Continue()` outside every loop, the init of a `For` included, which
 * runs before its loop starts. A jump in an inline `Fn` acts where the `Fn`
 * runs, which the sharing pass decides, so `placeInlineFns` checks it. It
 * gives each such write made a write into the vector the swizzle reads, see
 * {@link swizzleWritePlace}, and whether the program holds an `outputStruct`.
 */
function checkRoots(
  origin: Origin,
  roots: readonly unknown[],
  stageInputsInFns: boolean,
): {
  swizzleWrites: Map<any, any>;
  holdsOutputStruct: boolean;
  holdsReturn: boolean;
  holdsStageWrite: boolean;
  holdsVarying: boolean;
} {
  const swizzleWrites = new Map<any, any>();
  const checkedFns = new Set<unknown>();
  const checkFn = (definition: FnDefinition): void => {
    if (checkedFns.has(definition.body)) return;
    checkedFns.add(definition.body);
    assertFnIsClosed(definition);
    assertFnBody(origin, definition, stageInputsInFns);
    // A function may call `Return()`: it ends the function, not the stage.
    const stageHeldReturn = holdsReturn;
    visit(definition.body, false, true);
    holdsReturn = stageHeldReturn;
  };
  let holdsOutputStruct = false;
  let holdsReturn = false;
  let holdsStageWrite = false;
  let holdsVarying = false;
  for (const root of roots) if ((root as any)?.type === "storage") refuseWholeStorageRead(origin, root);
  // A node is visited once in each place: in a loop update, elsewhere in a
  // loop or an inline Fn, and among the statements of the program outside every loop.
  const visited = [new Set<unknown>(), new Set<unknown>(), new Set<unknown>()];
  const visit = (node: any, inUpdate: boolean, outsideLoops: boolean): void => {
    if (!node || typeof node !== "object") return;
    const seen = visited[inUpdate ? 2 : +outsideLoops]!;
    if (seen.has(node)) return;
    seen.add(node);
    if (Array.isArray(node)) return node.forEach((n) => visit(n, inUpdate, outsideLoops));
    if (inUpdate) assertForUpdateNode(node);
    if (outsideLoops && LOOP_JUMP_STATEMENTS.has(node.type))
      throw new ProgramError(
        "jump-outside-a-loop",
        LOOP_JUMP_OUTSIDE_A_LOOP_MESSAGE[node.type as "break" | "continue"],
        origin,
      );
    const refusal = operandRefusal(node.type, node.params?.map((param: any) => param?._t) ?? []);
    if (refusal) throw new ProgramError(refusal.code, refusal.message, origin);
    if (node.type === "outputStruct") holdsOutputStruct = true;
    assertNodeNames(node);
    if (node.type === "call") {
      assertCallFits(node);
      checkFn(node.value);
    }
    if (node.type === "return") holdsReturn = true;
    if (STAGE_WRITES.has(node.type)) holdsStageWrite = true;
    if (node.type === "varying") holdsVarying = true;
    if (node.type === "assign" && isSwizzleWrite(node.params[0])) {
      const target = node.params[0];
      assertNoRepeatedSwizzle(origin, target);
      assertConstantIndexInRange(origin, target.params[0], target.params[1]);
      swizzleWrites.set(
        node,
        Object.assign(Object.create(Object.getPrototypeOf(node)), node, {
          params: [swizzleWritePlace(target), ...node.params.slice(1)],
        }),
      );
    }
    node.params?.forEach((param: any, i: number) => {
      const asBuffer = i === 0 && (node.type === "storageElement" || node.type === "assign");
      if (param?.type === "storage" && !asBuffer) refuseWholeStorageRead(origin, param);
      const outside =
        outsideLoops &&
        !(node.type === "while" || (node.type === "for" && i > 0)) &&
        (isBlockSlot(node, i) || !isInlineFn(param));
      visit(param, inUpdate || (node.type === "for" && i === 2), outside);
    });
  };
  visit(roots, false, true);
  return { swizzleWrites, holdsOutputStruct, holdsReturn, holdsStageWrite, holdsVarying };
}

/** The node types a stage writes its result through, rather than through the value it returns. */
const STAGE_WRITES = new Set(["output", "outputStruct", "builtinPosition", "builtinFragDepth"]);

/** What the shared walk of {@link prepareRoots} found in a program, for a compiler that asks. */
export type FoundInRoots = {
  /** Whether the program holds a `Return()`. */
  holdsReturn?: boolean;
  /**
   * Whether the program writes what only a stage's result object holds: an
   * output, the position, the depth, or a vertex stage's varying. It holds
   * every write compiling reaches, and a write in a branch that folds away too.
   */
  writesStageResult?: boolean;
};

/**
 * The roots of a stage, ready to compile, once every compiler's shared
 * refusals have checked them, see {@link checkRoots}, and each write by index
 * through a swizzle is made a write into the vector the swizzle reads. A
 * fragment stage has its `outputStruct` written out and its implicit colour
 * converted, see {@link fragmentColour}. Any other stage, and a program with
 * no stage, refuses an `outputStruct`, which is a fragment stage's result.
 * `stageInputsInFns` is true for a target that holds a stage's inputs and
 * varyings as globals, where a function with a layout may reach them.
 */
export function prepareRoots<T extends Node<ShaderType>>(
  origin: Origin,
  stage: string | undefined,
  roots: readonly T[],
  found?: FoundInRoots,
  stageInputsInFns = false,
): T[] {
  const { swizzleWrites, holdsOutputStruct, holdsReturn, holdsStageWrite, holdsVarying } = checkRoots(
    origin,
    roots,
    stageInputsInFns,
  );
  if (found) {
    found.holdsReturn = holdsReturn;
    found.writesStageResult = holdsStageWrite || (stage === "vertex" && holdsVarying);
  }
  if ((stage === "vertex" || stage === "fragment") && holdsReturn) {
    throw new ProgramError(
      "return-ends-the-stage",
      `Return() would end the ${stage} stage, leaving the outputs it writes after it unwritten on every target. ` +
        `Write the rest of the stage in an If or Else instead` +
        (stage === "fragment" ? ", or drop the fragment with Discard()." : "."),
      origin,
    );
  }
  roots = substituteNodes(roots, swizzleWrites);
  if (stage === "fragment") {
    assertReadsNoAttribute(origin, roots);
    return fragmentColour(lowerOutputStruct(origin, roots as readonly BaseNode<ShaderType>[]) as unknown as T[]);
  }
  if (holdsOutputStruct) {
    throw new ProgramError(
      "output-struct-outside-fragment",
      `outputStruct is the value a fragment stage returns, and this program is compiled ` +
        (stage === undefined ? "with no stage" : `as a ${stage} stage`) +
        `. Compile it as a fragment stage.`,
      origin,
    );
  }
  return [...roots];
}

/**
 * Throws when a fragment stage compiled alone reads an attribute, which only a
 * vertex stage is given. {@link linkStages} turns the read into a varying
 * first, for a call that takes both stages.
 */
function assertReadsNoAttribute(origin: Origin, roots: unknown): void {
  let read: any;
  someNode(roots, (n) => (n.type === "attribute" ? ((read = n), true) : false));
  if (read) {
    throw new ProgramError(
      "fragment-reads-attribute",
      `the fragment stage reads the attribute "${read.value.slot}", which only a vertex stage is given. ` +
        `Compile it together with its vertex stage, in createGlsl, createWgsl, compileJS or compileWasm, or pass the value on in a varying.`,
      origin,
    );
  }
}

/**
 * The two stages of a program with each attribute the fragment stage reads
 * passed on in a varying, as TSL does: the fragment stage reads the varying,
 * and the vertex stage writes it from the attribute. Nodes of one slot share
 * one varying. The stages the caller holds are left as they are. `carriers`
 * holds each varying under the slot of the attribute it carries.
 */
export function linkStages<V, F>(
  origin: Origin,
  vertexRoot: V,
  fragmentRoot: F,
  stageInputsInFns = false,
): { vertex: V; fragment: F; carriers: Map<string, any> } {
  const reads: any[] = [];
  // A target that refuses a stage input in a function leaves a read there to the refusal, which names the attribute.
  const visited = new Set<unknown>();
  const find = (n: any): void => {
    if (!n || typeof n !== "object" || visited.has(n)) return;
    visited.add(n);
    if (Array.isArray(n)) return n.forEach(find);
    if (n.type === "attribute") reads.push(n);
    if (Array.isArray(n.params)) n.params.forEach(find);
    if (stageInputsInFns && n.type === "call") find(n.value.body);
  };
  find(fragmentRoot);
  // Reads of one slot become one varying, so a conflict between them has to show before they merge.
  if (reads.length > 0) assertOneDeclarationPerName(origin, fragmentRoot);
  const replaced = new Map<any, any>();
  const varyings = new Map<string, any>();
  const assignments: BaseNode<ShaderType>[] = [];
  for (const n of reads) {
    const { slot, shaderType } = n.value;
    let carrier = varyings.get(slot);
    if (!carrier) {
      carrier = varying(shaderType);
      // A refusal of the varying names the attribute it carries, which is what the application wrote.
      carrier.value.carries = slot;
      varyings.set(slot, carrier);
      assignments.push(node({ _t: "void", type: "assign", params: [carrier, n] }) as BaseNode<ShaderType>);
    }
    replaced.set(n, carrier);
  }
  if (replaced.size === 0) return { vertex: vertexRoot, fragment: fragmentRoot, carriers: new Map() };
  const last = Array.isArray(vertexRoot) ? vertexRoot[vertexRoot.length - 1] : vertexRoot;
  if (!last || typeof last !== "object") {
    throw new ProgramError(
      "fragment-attribute-without-vertex-root",
      `the fragment stage reads the attribute "${reads[0].value.slot}", but the vertex stage gives no root to pass it on from. ` +
        `A vertex stage returns its position.`,
      origin,
    );
  }
  const roots: any[] = Array.isArray(vertexRoot) ? [...vertexRoot] : [vertexRoot];
  roots[roots.length - 1] = node({
    _t: last._t,
    type: "seq",
    params: last.type === "seq" ? [...assignments, ...last.params] : [...assignments, last],
  });
  const vertex = Array.isArray(vertexRoot) ? roots : roots[0];
  return {
    vertex: vertex as V,
    fragment: substituteNodes(fragmentRoot, replaced, stageInputsInFns),
    carriers: varyings,
  };
}

/**
 * The roots of a fragment stage with the implicit colour converted to a `vec4`
 * the way TSL converts it to its render target's type. A `vec3` gains an alpha
 * of 1, a `vec2` a blue of 0 and an alpha of 1, and a scalar fills every
 * channel. A stage that declares an output, or returns anything else, comes
 * back as it is.
 */
export function fragmentColour<T extends Node<ShaderType>>(roots: readonly T[]): T[] {
  const last = roots[roots.length - 1] as BaseNode<ShaderType> | undefined;
  if (!last || someNode(roots, (n) => n.type === "output")) return [...roots];
  let tail = (last.type === "seq" ? last.params![last.params!.length - 1] : last) as Node<any>;
  // An integer or boolean value reaches a colour as the float it converts to.
  if (/^[iub]vec[234]$/.test(tail._t)) tail = tail.convert(`vec${tail._t.slice(-1)}`);
  let colour: Node<"vec4">;
  switch (tail._t) {
    case "vec4":
      colour = tail;
      break;
    case "vec3":
      colour = vec4(tail, 1);
      break;
    case "vec2":
      colour = vec4(tail, 0, 1);
      break;
    case "float":
      colour = vec4(tail);
      break;
    case "int":
    case "uint":
    case "bool":
      colour = vec4(tail.toFloat());
      break;
    default:
      return [...roots];
  }
  const converted =
    last.type === "seq"
      ? node({ _t: "vec4", type: "seq", params: [...last.params!.slice(0, -1), colour as BaseNode<ShaderType>] })
      : colour;
  return [...roots.slice(0, -1), converted as unknown as T];
}

/** Which component each accessor letter names, in all three spellings. */
export const COMPONENT_INDEX: Record<string, number> = {
  x: 0,
  y: 1,
  z: 2,
  w: 3,
  r: 0,
  g: 1,
  b: 2,
  a: 3,
  s: 0,
  t: 1,
  p: 2,
  q: 3,
};

/**
 * Resolve a chain of swizzles down to the node underneath it.
 *
 * A swizzle can't be assigned to itself. `a.xyz.xy = e` has to become a
 * write to `a`, whatever writable node `a` is — and which components of `a`
 * that is takes composing the patterns: the outer pattern indexes into the
 * inner one, so `a.yzw.xy` selects the first two of y, z, w, which is `a.yz`.
 */
export function resolveSwizzleTarget(target: any): { base: BaseNode<ShaderType>; pattern: string } {
  let pattern = target.value as string;
  let base = target.params![0];
  while (base?.type === "swizzle") {
    let inner = base.value as string;
    pattern = [...pattern].map((c) => inner[COMPONENT_INDEX[c]]).join("");
    base = base.params![0];
  }
  return { base, pattern };
}

/**
 * A constant index outside the components of a vector, the columns of a
 * matrix or the elements of a uniform array is refused by every target, so the
 * same program is not refused on one and compiled to a read of nothing on
 * another. A constant index is a literal, or an operation of literals that
 * folds to one. Asked wherever an element is compiled, which also reaches a
 * graph that `deserialize` rebuilt. Returns the index when it is inside, and
 * `undefined` when it is not constant.
 */
export function assertConstantIndexInRange(
  origin: Origin,
  target: { _t?: string; type?: string; value?: unknown },
  index: BaseNode<ShaderType>,
): number | undefined {
  const constant = isLeafLiteral(index) ? index : tryFold(origin, index);
  if (!constant || !isLeafLiteral(constant)) return undefined;
  const type = target._t as string;
  const array = target.type === "uniformArray" ? (target.value as { length: number }).length : undefined;
  const shape = MATRIX_DIMENSIONS[type];
  const count = array ?? (shape ? shape[0] : componentCountOf(type));
  const what = array !== undefined ? `${type}[${array}]` : type;
  const items = array !== undefined ? "elements" : shape ? "columns" : "components";
  const k = Math.trunc(Number(constant.value));
  if (!(k >= 0 && k < count)) {
    throw new ProgramError(
      "index-out-of-range",
      `index ${k} is outside ${withArticle(what)}'s ${items} 0 to ${count - 1}`,
      origin,
    );
  }
  return k;
}

/** `Math.round` takes a half toward +Infinity. WGSL takes it to the even neighbour, so folding does too. */
export function roundHalfToEven(x: number): number {
  const r = Math.round(x);
  return x - Math.floor(x) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** The factors `radians` and `degrees` multiply by, on every target that computes them itself. */
export const RADIANS_PER_DEGREE = Math.PI / 180;
export const DEGREES_PER_RADIAN = 180 / Math.PI;

/**
 * Only a square matrix has an inverse, and neither language offers an overload
 * for the rest. Asked in both backends, so the two cannot come to different
 * answers about the same program — one emitting a call no driver accepts while
 * the other refuses it.
 *
 * Returns the matrix's size, which the WGSL side needs to pick its helper.
 */
export function assertSquareMatrix(origin: Origin, operandType: string | undefined): number {
  let shape = MATRIX_DIMENSIONS[operandType as string];
  if (shape === undefined || shape[0] !== shape[1]) {
    throw new ProgramError(
      "inverse-of-non-square",
      `inverse() needs a square matrix, but this one is ` + `${operandType ?? "untyped"}.`,
      origin,
    );
  }
  return shape[0];
}

/**
 * A compute dispatch has no vertex or fragment stage, so what those stages
 * pass in and out has nothing to attach to. Throws when `stage` is compute.
 */
export function assertNotInAComputeStage(
  origin: Origin,
  stage: "vertex" | "fragment" | "compute",
  refused: { action: string; because: string; instead: string },
): void {
  if (stage !== "compute") return;
  throw new ProgramError(
    "compute-cannot-use-stage-io",
    `A compute program cannot ${refused.action}, because ${refused.because}. ${refused.instead}`,
    origin,
  );
}

/** The refusals of a compute program, one for each stage input or output it cannot use. */
export const COMPUTE_REFUSES = {
  attribute: {
    action: "read an attribute",
    because: "a compute dispatch has no vertices to read one for",
    instead: "Read the buffer with storage(attribute, type) instead.",
  },
  varying: {
    action: "read a varying",
    because: "a compute dispatch has no vertex stage to pass one from",
    instead: "Pass the value in through a uniform or a storage buffer instead.",
  },
} as const;

/**
 * The position is the vertex stage's output. A fragment stage cannot read it:
 * GLSL's gl_Position is write-only there and WGSL has no such value at all.
 * Emitting it anyway produced an identifier neither backend declares.
 */
export function assertPositionIsReadable(ctx: CompileCtx): void {
  if (ctx.shaderStage === "vertex") return;
  throw new ProgramError(
    "position-read-in-fragment",
    "builtinPosition() is the vertex stage's output position, and a " +
      "fragment stage cannot read it. Pass the value you need through a " +
      "varying() instead.",
    ctx,
  );
}

/**
 * What a vertex stage may be handed.
 *
 * Its result becomes the position, so a vec4 is the ordinary case, and anything
 * else is refused here rather than at run time.
 *
 * `Node<"void">` is the other way to satisfy a vertex stage: assign
 * builtinPosition() yourself and return nothing. Calling an `Fn` whose body
 * returns nothing gives that type, so the two cases are exactly the two the
 * signature admits. Whether an assignment actually happened is not something
 * a signature can see, so that half stays a run-time check.
 *
 * Several values may be returned at once, of which the last becomes the
 * position — so that is the one constrained, and the values before it are
 * whatever the shader needed on the way there. Saying so requires knowing which
 * value is last, which is why `Fn` infers an array return as a tuple.
 */
export type VertexRoot = Node<"vec4"> | readonly [...Node<ShaderType>[], Node<"vec4">] | Node<"void">;

export type CompileFnOptions = {
  name: string;
  params: Array<{ name: string; type: ShaderType }>;
  /**
   * Set by the compile functions of a routine and of a grid: the program has
   * no stage, so what only a stage has is refused. A grid has `fragCoord()` and
   * `Discard()`, which a routine has not.
   *
   * @internal
   */
  kind?: "routine" | "grid";
};

/** The inputs only a stage has, by the node that reads them, with the stage that has them. */
const STAGE_INPUTS: Record<string, { what: string; stage: string }> = {
  fragCoord: { what: "fragCoord()", stage: "fragment stage, or a grid" },
  discard: { what: "Discard()", stage: "fragment stage, or a grid" },
  builtinFragDepth: { what: "builtinFragDepth()", stage: "fragment stage" },
  invocationIndex: { what: "invocationIndex()", stage: "compute stage" },
  vertexIndex: { what: "vertexIndex()", stage: "vertex stage" },
  instanceIndex: { what: "instanceIndex()", stage: "vertex stage" },
  builtinPosition: { what: "builtinPosition()", stage: "vertex stage" },
  attribute: { what: "an attribute", stage: "vertex stage" },
  varying: { what: "a varying", stage: "vertex or fragment stage" },
};

/** What a grid has that a routine has not: the pixel it evaluates, and a fragment that drops it. */
const GRID_INPUTS = new Set(["fragCoord", "discard"]);

/** Throws when a program with no stage, a routine or a grid, reads an input that only a stage has. */
export function assertReadsNoStageInput(origin: Origin, roots: unknown, kind: "routine" | "grid"): void {
  someNode(roots, (node) => {
    const input = STAGE_INPUTS[node.type];
    if (!input || (kind === "grid" && GRID_INPUTS.has(node.type))) return false;
    throw new ProgramError(
      "stage-input-without-stage",
      `${input.what} is an input of a ${input.stage}, and a ${kind} has no stage to give it. ` +
        `Compile the program as a ${input.stage} instead.`,
      origin,
    );
  });
}

/** A storage node as the program's graph holds it: a node of its element type. */
export type StorageGraphNode = StorageNode<ShaderType> & BaseNode<ShaderType>;

/**
 * One entry for each storage buffer reachable from the roots, in the order the
 * program created the buffers: a node over the buffer, and the widest access
 * any node over it needs.
 */
export function storageBuffersOf(roots: unknown): { node: StorageGraphNode; access: StorageAccess }[] {
  const buffers = new Map<string, { node: StorageGraphNode; access: StorageAccess }>();
  someNode(roots, (node) => {
    if (node.type !== "storage") return;
    const buffer = buffers.get(node.value.slot);
    if (!buffer) buffers.set(node.value.slot, { node: node as StorageGraphNode, access: node.value.access });
    else if (buffer.access !== node.value.access) buffer.access = "read_write";
  });
  return [...buffers.values()].sort((a, b) => a.node.attribute.id - b.node.attribute.id);
}

/** The type of a uniform, and the length of a uniform array, as its node's value holds them. */
export type UniformDeclaration = { readonly shaderType: string; readonly length?: number };

/** One storage node for each buffer reachable from the roots, in the order the program created the buffers. */
export function storageNodesOf(roots: unknown): StorageGraphNode[] {
  return storageBuffersOf(roots).map((buffer) => buffer.node);
}

/** What a uniform, attribute or varying node declares, in the words of an error about its name. */
function describeNamed(node: any): string | undefined {
  const value = node.value;
  const precision = value?.precision ? ` of precision "${value.precision}"` : "";
  if (node.type === "uniform") return `${withArticle(value.shaderType)} uniform${precision}`;
  if (node.type === "uniformArray") return `a uniform array of ${value.length} ${value.shaderType}${precision}`;
  if (node.type === "attribute") return `${withArticle(value.shaderType)} attribute`;
  if (node.type === "varying") return `${withArticle(value.shaderType)} varying`;
  return undefined;
}

/**
 * Throws when two uniforms, attributes or varyings reachable from the roots
 * share a name but not what they declare. One name is one declaration on
 * every backend, so nodes made separately with one name read the same value.
 */
export function assertOneDeclarationPerName(origin: Origin, roots: unknown): void {
  const declared = new Map<string, string>();
  someNode(roots, (node) => {
    const what = describeNamed(node);
    if (what === undefined) return;
    const name = node.value.slot as string;
    const earlier = declared.get(name);
    if (earlier === undefined) declared.set(name, what);
    else if (earlier !== what) {
      throw new ProgramError(
        "name-declared-twice",
        `"${name}" names ${earlier} and ${what}: one name is one declaration. Give them different names`,
        origin,
      );
    }
  });
}

/**
 * Numbers the variables of the roots that share a name, so each declares a name
 * of its own. `toVar("color")` keeps one registry of names for each top-level `Fn`, so
 * two `Fn`s compiled as the roots of one program can each take `color`, and the
 * program would declare it twice. The first variable keeps its name and each
 * later one gets the next free number appended, as `toVar` does within one `Fn`.
 * A variable named like a function with a layout is numbered too, as a variable
 * of that name would hide the function where it is called, and a variable named
 * like a keyword or built-in is renamed, see {@link safeName}.
 *
 * A variable is one node, so a renamed copy of it takes the place of every read.
 * The roots come back as copies where a variable was renamed, and the graph the
 * caller holds keeps the names it was given.
 */
export function numberClashingVariables<T>(roots: T): T {
  const byName = new Map<string, any[]>();
  const functions = new Set<string>();
  someNode(roots, (node) => {
    if (node.type === "call" && node.value.name !== undefined) functions.add(node.value.name);
    if (node.type !== "var" || node.value.parameter) return;
    // A name GLSL or WGSL cannot declare is a hint, and the variable declares the name that stands for it.
    const name = safeName(node.value.varName);
    const group = byName.get(name);
    if (group === undefined) byName.set(name, [node]);
    else if (!group.includes(node)) group.push(node);
  });
  const taken = new Set([...byName.keys(), ...functions]);
  const renamed = new Map<any, any>();
  for (const [name, variables] of byName) {
    // A function keeps the name it is declared and called by, so a variable of that name is renamed.
    variables.forEach((variable, i) => {
      let varName = name;
      if (i > 0 || functions.has(name)) {
        let number = 1;
        while (taken.has(`${name}${number}`)) number++;
        varName = `${name}${number}`;
        taken.add(varName);
      }
      if (varName === variable.value.varName) return;
      renamed.set(
        variable,
        Object.assign(Object.create(Object.getPrototypeOf(variable)), variable, {
          value: { ...variable.value, varName },
        }),
      );
    });
  }
  return substituteNodes(roots, renamed);
}

/** The node types an assignment can write, through any swizzle, component or column of them. */
const ASSIGNABLE = new Set(["var", "storageElement", "output", "varying", "builtinPosition", "builtinFragDepth"]);

/** The node types an assignment passes through on the way to the node it writes. */
const ASSIGNMENT_PATH = new Set(["swizzle", "vectorElement", "matrixElement"]);

/** What to call a node that can't be assigned, by its type, in the error refusing it. */
const READ_ONLY_NAMES: Record<string, string> = {
  uniform: "a uniform",
  uniformArray: "a uniform",
  uniformArrayElement: "a uniform",
  attribute: "an attribute",
  fragCoord: "a built-in input",
  invocationIndex: "a built-in input",
  vertexIndex: "a built-in input",
  instanceIndex: "a built-in input",
};

/** The stage outputs only one stage writes, which stage, and what to call them in an error. */
const WRITTEN_BY_ONE_STAGE: Record<string, { stage: string; name: string }> = {
  varying: { stage: "vertex", name: "a varying" },
  builtinPosition: { stage: "vertex", name: "the position" },
  builtinFragDepth: { stage: "fragment", name: "the fragment depth" },
};

/** The node types of a stage's inputs, and its varyings, which a function reaches on a target that has them as globals. */
const FN_REACHABLE_STAGE_VALUES = new Set([
  "attribute",
  "varying",
  "fragCoord",
  "instanceIndex",
  "invocationIndex",
  "vertexIndex",
]);

/** The node types that belong to a stage: its inputs and its outputs, which WGSL passes to and from its entry point. */
const STAGE_BOUND: Record<string, string> = {
  attribute: "an attribute",
  varying: "a varying",
  output: "an output",
  outputStruct: "an output",
  builtinPosition: "the position",
  builtinFragDepth: "the fragment depth",
  fragCoord: "fragCoord()",
  instanceIndex: "instanceIndex()",
  invocationIndex: "invocationIndex()",
  vertexIndex: "vertexIndex()",
};

/**
 * Throws if the body of a function with a layout holds a `Return` although the
 * function gives a value, or reads or writes something that belongs to a stage.
 * A call inside it is checked with its own body.
 */
function assertFnBody(origin: Origin, definition: FnDefinition, stageInputsInFns: boolean): void {
  const seen = new Set<unknown>();
  const walk = (n: any): void => {
    if (!n || typeof n !== "object" || seen.has(n)) return;
    seen.add(n);
    if (Array.isArray(n)) return n.forEach(walk);
    if (n.type === "return" && definition.type !== "void") {
      throw new ProgramError(
        "return-without-value",
        `${definition.name === undefined ? "An Fn with a layout" : `The Fn "${definition.name}"`} gives a ${definition.type}, and its body calls Return(), which gives no value. ` +
          `Return a value at the end, or give the layout the type "void".`,
        origin,
      );
    }
    const stage =
      stageInputsInFns && FN_REACHABLE_STAGE_VALUES.has(n.type)
        ? undefined
        : n.type === "varying" && n.value.carries !== undefined
          ? `the attribute "${n.value.carries}"`
          : STAGE_BOUND[n.type];
    if (stage !== undefined) {
      throw new ProgramError(
        "layout-fn-uses-stage-value",
        `${definition.name === undefined ? "An Fn with a layout" : `The Fn "${definition.name}"`} uses ${stage}, which belongs to the stage and not to a function. ` +
          `Pass its value in as an input.`,
        origin,
      );
    }
    n.params?.forEach(walk);
  };
  walk(definition.body);
}

/**
 * The name a function of a program is declared under, with `emit` called the
 * first time to write its lines. A name taken gets `_` and the next free number
 * appended, and a function without a name gets a generated one.
 */
export function declareFn(ctx: CompileCtx, definition: FnDefinition, emit: (name: string) => string): string {
  const fns: UserFns = (ctx.userFns ??= { names: new Map(), taken: new Set(), declarations: [] });
  const known = fns.names.get(definition.body);
  if (known !== undefined) return known;
  const base = definition.name ?? `${RESERVED_NAME}fn`;
  let name: string = base;
  if (definition.name === undefined || fns.taken.has(name)) {
    let number = 1;
    while (fns.taken.has(`${base}_${number}`)) number++;
    name = `${base}_${number}`;
  }
  fns.taken.add(name);
  fns.names.set(definition.body, name);
  fns.declarations.push(emit(name));
  return name;
}

const RESERVED_NAME = "_rmsl_";

/**
 * A context for compiling the body of a function: the maps that collect what
 * the program declares are the program's own, and what a statement remembers is
 * the body's own.
 */
export function bodyContext(ctx: CompileCtx): CompileCtx {
  ctx.storageTypes ??= new Map();
  ctx.userFns ??= { names: new Map(), taken: new Set(), declarations: [] };
  ctx.glslHelpers ??= new Map();
  return { ...ctx, memo: new Map(), varDefs: new Map(), inFn: true };
}

/**
 * The roots with every call of an `Fn` with a layout written out where it is
 * called, for a target that has no use for a function of its own. The inputs
 * are bound to variables the call makes, and the variables of the body are
 * renamed for each call, so two calls of one function do not share them.
 *
 * A call that is a statement, or the value of a `let`, puts the statements of
 * the body into the block it stands in. A body that calls `Return()` goes in a
 * `scope`, which `Return()` leaves. Any other call is a `seq` that holds the
 * statements ahead of its value. The graph the caller holds is left as it is.
 */
export function inlineCalls<T>(roots: T): T {
  let instance = 0;
  const lowered = new Map<any, any>();
  const copyOf = (original: any, params: any[]) =>
    Object.assign(Object.create(Object.getPrototypeOf(original)), original, { params });

  /** Whether the body returns early, not counting the bodies of the calls inside it. */
  const returns = (body: any): boolean => {
    const seen = new Set<unknown>();
    const walk = (n: any): boolean => {
      if (!n || typeof n !== "object" || seen.has(n)) return false;
      seen.add(n);
      return n.type === "return" || (Array.isArray(n.params) && n.params.some(walk));
    };
    return walk(body);
  };

  /** The call written out: its statements, and the node that holds its value. */
  const expand = (call: any): { statements: any[]; value: any } => {
    const id = instance++;
    const definition = call.value as FnDefinition;
    const rename = (name: string) => `${RESERVED_NAME}c${id}_${name}`;
    const variables = new Map<string, any>();
    const variable = (n: any) => {
      let v = variables.get(n.value.varName);
      if (v === undefined) {
        v = var_(rename(n.value.varName), n.value.varType);
        variables.set(n.value.varName, v);
      }
      return v;
    };
    const copies = new Map<any, any>();
    const copy = (n: any): any => {
      if (!n || typeof n !== "object") return n;
      if (copies.has(n)) return copies.get(n);
      let result = n;
      if (n.type === "var") result = variable(n);
      else if (Array.isArray(n.params)) result = copyOf(n, n.params.map(copy));
      copies.set(n, result);
      return result;
    };
    const bindings = definition.inputs.map((input, i) => {
      const parameter = variable({ value: { varName: input.name, varType: input.type } });
      return node({ _t: "void", type: "let", params: [parameter, lower(call.params[i])] }) as any;
    });
    const body = lower(copy(definition.body));
    const statements = body.params.slice(0, -1);
    const value = body.params[body.params.length - 1];
    if (!returns(definition.body)) return { statements: [...bindings, ...statements], value };
    const scope = node({ _t: "void", type: "scope", params: [node({ _t: "void", type: "seq", params: statements })] });
    return { statements: [...bindings, scope], value };
  };

  /** A block's statements, with the calls that are statements written out inside it. */
  const block = (statements: any[]): any[] =>
    statements.flatMap((statement) => {
      if (statement?.type === "call") return expand(statement).statements;
      if (statement?.type === "let" && statement.params[1]?.type === "call") {
        const { statements: before, value } = expand(statement.params[1]);
        return [...before, copyOf(statement, [lower(statement.params[0]), value])];
      }
      return [lower(statement)];
    });

  function lower(n: any): any {
    if (!n || typeof n !== "object") return n;
    if (Array.isArray(n)) return n.map(lower);
    const known = lowered.get(n);
    if (known !== undefined) return known;
    let result = n;
    if (n.type === "call") {
      const { statements, value } = expand(n);
      result = node({ _t: n._t, type: "seq", params: [...statements, value] });
    } else if (n.type === "seq") {
      const params = block(n.params ?? []);
      const unchanged = params.length === n.params.length && params.every((p, i) => p === n.params[i]);
      result = unchanged ? n : copyOf(n, params);
    } else if (Array.isArray(n.params)) {
      const params = n.params.map(lower);
      result = params.every((p: any, i: number) => p === n.params[i]) ? n : copyOf(n, params);
    }
    lowered.set(n, result);
    return result;
  }

  return lower(roots);
}

/**
 * A parameter of a function a backend compiles with its `compile…Fn`, as the
 * body sees it. Its value belongs to the caller, so it can't be assigned.
 */
export function parameterNode(name: string, type: string): any {
  const node = var_(name, type);
  (node as any).value.parameter = true;
  return node;
}

/** The node an assignment to `target` writes, below any swizzle, component or column. */
function assignmentRoot(target: any): any {
  while (ASSIGNMENT_PATH.has(target?.type)) target = target.params[0];
  return target;
}

/** Throws when a swizzle an assignment to `target` writes through names a component more than once. */
function assertNoRepeatedSwizzle(origin: Origin, target: any): void {
  for (let node = target; ASSIGNMENT_PATH.has(node?.type); node = node.params[0]) {
    if (node.type === "swizzle" && new Set(node.value).size !== node.value.length) {
      throw new ProgramError(
        "swizzle-repeats-component",
        `can't assign through the swizzle .${node.value}, which names a component more than once; name each component once`,
        origin,
      );
    }
  }
}

/** Whether an assignment to `target` writes by index through a swizzle. */
function isSwizzleWrite(target: any): boolean {
  return target?.type === "vectorElement" && target.params[0]?.type === "swizzle";
}

/**
 * `roots` with each node `replaced` maps to put in its place. A node that
 * holds a replaced node is copied, once, so a node several others hold stays
 * one node. The graph the caller holds is left as it is.
 */
function substituteNodes<T>(roots: T, replaced: Map<any, any>, intoCalls = false): T {
  if (replaced.size === 0) return roots;
  const copies = new Map<any, any>();
  const definitions = new Map<any, any>();
  const substitute = (node: any): any => {
    if (Array.isArray(node)) return node.map(substitute);
    if (!node || typeof node !== "object") return node;
    const known = replaced.get(node) ?? copies.get(node);
    if (known !== undefined) return known;
    if (!Array.isArray(node.params)) return node;
    const params = node.params.map(substitute);
    // A call holds the body of its function, which every call of it shares, so the body is replaced once.
    let value = node.value;
    if (intoCalls && node.type === "call") {
      const body = substitute(node.value.body);
      if (body !== node.value.body) {
        value = definitions.get(node.value) ?? { ...node.value, body };
        definitions.set(node.value, value);
      }
    }
    const result =
      value === node.value && params.every((p: any, i: number) => p === node.params[i])
        ? node
        : Object.assign(Object.create(Object.getPrototypeOf(node)), node, { params, value });
    copies.set(node, result);
    return result;
  };
  return substitute(roots);
}

/**
 * Throws unless an assignment to `target`, compiled for `stage`, writes
 * something writable: a variable, a storage element or a stage output,
 * directly or through any swizzle, component or column of it, where no
 * swizzle names a component more than once. Called by every backend, so a
 * uniform, an attribute or a computed value is refused the same way on each,
 * and so is an element of a storage node made read-only, a stage output
 * written outside the stage that writes it, and a parameter of the compiled
 * function. With `storageWritesNeedCompute`, as on a GPU target, a storage
 * element is written only from a compute stage.
 */
export function assertAssignable(
  origin: Origin,
  target: any,
  stage: "vertex" | "fragment" | "compute",
  storageWritesNeedCompute = false,
): void {
  assertNoRepeatedSwizzle(origin, target);
  const root = assignmentRoot(target);
  if (root?.type === "storage") {
    throw new ProgramError(
      "assign-to-whole-storage",
      "can't assign to a whole storage buffer; assign to one of its elements with .element(i)",
      origin,
    );
  }
  if (!ASSIGNABLE.has(root?.type)) {
    const what = READ_ONLY_NAMES[root?.type] ?? "a computed value";
    throw new ProgramError(
      "assign-to-read-only",
      `can't assign to ${what}: only a variable, a storage element or a stage output can be assigned; copy the value into a variable with toVar() first`,
      origin,
    );
  }
  if (root.type === "var" && root.value.parameter) {
    throw new ProgramError(
      "assign-to-parameter",
      `can't assign to "${root.value.varName}", a parameter of the compiled function, whose value belongs to the caller; copy it into a variable with toVar() first`,
      origin,
    );
  }
  const writer = WRITTEN_BY_ONE_STAGE[root.type];
  if (writer !== undefined && writer.stage !== stage) {
    throw new ProgramError(
      "assign-in-wrong-stage",
      `can't assign to ${writer.name} in a ${stage} stage; only a ${writer.stage} stage writes it`,
      origin,
    );
  }
  if (root.type === "storageElement" && root.params[0].value.access === "read") {
    throw new ProgramError(
      "assign-to-read-only-storage",
      "can't assign to an element of a storage node made read-only with toReadOnly()",
      origin,
    );
  }
  if (storageWritesNeedCompute && root.type === "storageElement" && stage !== "compute") {
    throw new ProgramError(
      "storage-written-outside-compute",
      `storage buffers are read-only in a ${stage} shader; write them from a compute program`,
      origin,
    );
  }
}

/** The storage element an assignment to `target` writes, through any swizzle, component or column, or undefined. */
export function assignedStorageElement(target: any): any {
  const root = assignmentRoot(target);
  return root?.type === "storageElement" ? root : undefined;
}

/**
 * Throws unless `length` values from element `offset` on fit in the
 * attribute's buffer, so a context's `write()` fails the same way on every backend.
 */
export function assertWriteFits(
  origin: Origin,
  attribute: StorageBufferAttribute,
  length: number,
  offset: number,
): void {
  const capacity = attribute.count * attribute.itemSize;
  if (offset * attribute.itemSize + length > capacity) {
    throw new UsageError(
      "write-past-attribute",
      `writing ${length} values from element ${offset} runs past the end of an attribute of ${capacity} values`,
      origin,
    );
  }
}
