import {
  BaseNode,
  MATRIX_DIMENSIONS,
  Node,
  NodeImpl,
  ShaderType,
  StorageBufferAttribute,
  TYPE_WIDTH,
  node,
  someNode,
  var_,
  vec4,
} from "../core";
import { componentCountOf, componentKindOf } from "./cpu";
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

export interface CompileCtx {
  nextId: number;
  shaderStage: "vertex" | "fragment" | "compute";
  /**
   * The uniforms the program reads, by name: two nodes with one name are one
   * uniform. `length` is set only for uniform arrays, and gives their element count.
   */
  uniforms: Map<string, { type: string; slot: string; length?: number }>;
  storages?: Map<
    string,
    {
      name: string;
      type: string;
      access: "read" | "write" | "read_write";
      wgslName: string;
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
  attributes: Map<string, { id: number; type: string; slot: string }>;
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
  /**
   * Whether the program assigned the position itself. A vertex stage that has
   * done so needs no implicit write, and its result is free to be anything.
   */
  positionWritten: boolean;
  inFn: boolean;
  fragDepthUsed: boolean;
  /** Whether the shader reads the fragment's screen position. */
  fragCoordUsed: boolean;
  /** Whether a vertex stage reads `vertexIndex()` / `instanceIndex()`. */
  vertexIndexUsed?: boolean;
  instanceIndexUsed?: boolean;
  /** Fn parameter names, which the JS target reads from `ctx.params`. */
  jsParams: Set<string>;
  /** Names of JS helper functions the compiled function needs. */
  jsHelpers: Set<string>;
  /** The constant vectors and matrices a JS function reads, by the name each is declared under once. */
  jsConstants?: Map<string, string>;
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
  /** JS target: whether the value being compiled reads a slot that statements filled. */
  jsReadsSlot: boolean;
  /** What derivative ops (dFdx/dFdy/fwidth) compile to on the CPU. */
  derivatives: "throw" | "zero";
  /** Per-call variable bindings instead of hoisted per-program scratch. */
  reentrant: boolean;
  /** Whether the program writes outputs/position/fragDepth via a result object. */
  jsNeedsRes: boolean;
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
function integerLiteralComponents(n: BaseNode<ShaderType>): number[] | null {
  let t = n._t as string;
  if (!isIntegerType(t)) return null;
  if (isLeafLiteral(n)) return [n.value as number];
  if (n.type === t && Array.isArray(n.value)) return n.value as number[];
  let width = TYPE_WIDTH[t] ?? 1;
  if (width === 1 || n.type !== "construct" || !n.params?.length) return null;
  let parts: number[] = [];
  for (let param of n.params) {
    let constant = integerConstant(param);
    let components = constant && integerLiteralComponents(constant);
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
function integerConstant(n: BaseNode<ShaderType>): BaseNode<ShaderType> | null {
  if (!n || typeof n !== "object" || !isIntegerType(n._t as string)) return null;
  if (integerLiteralComponents(n)) return n;
  let cached = integerConstants.get(n);
  if (cached !== undefined) return cached;
  integerConstants.set(n, null);
  let folded = tryFold(n);
  let result = folded && folded !== n ? integerConstant(folded) : null;
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
function integerComparison(n: BaseNode<ShaderType>): boolean[] | null {
  let compare = INTEGER_COMPARISONS[n.type];
  let [lhs, rhs] = n.params ?? [];
  if (!compare || !lhs || !rhs) return null;
  let a = integerConstant(lhs);
  let b = integerConstant(rhs);
  let ca = a && integerLiteralComponents(a);
  let cb = b && integerLiteralComponents(b);
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
function foldSelect(n: BaseNode<ShaderType>): BaseNode<ShaderType> | null {
  let [cond, ifTrue, ifFalse] = n.params ?? [];
  if (!cond || !ifTrue || !ifFalse) return null;
  let picks = isLeafLiteral(cond) ? [Boolean(cond.value)] : integerComparison(cond);
  if (!picks) return null;
  if (picks.every((p) => p === picks![0])) return picks[0] ? ifTrue : ifFalse;
  let a = integerLiteralComponents(ifTrue);
  let b = integerLiteralComponents(ifFalse);
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
function foldIntegerOperands(n: BaseNode<ShaderType>): BaseNode<ShaderType> | null {
  let t = n._t as string;
  let [lhs, rhs] = n.params ?? [];
  if (!isIntegerType(t) || !lhs) return null;
  let kind: "int" | "uint" = isUnsignedType(t) ? "uint" : "int";
  let operands = n.params!.map(integerLiteralComponents);
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
  let b = rhs ? integerLiteralComponents(rhs) : null;
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

export function tryFold(n: BaseNode<ShaderType>): BaseNode<ShaderType> | null {
  // Integer operands are folded first, so a constant subexpression counts as
  // the literal it is when this node is folded or its divisor is rewritten.
  let operands = n.params?.map((p) => integerConstant(p) ?? p);
  if (operands?.some((p, i) => p !== n.params![i])) {
    n = mkNode({ _t: n._t as string, type: n.type, params: operands, value: n.value });
  }
  // A select with a constant condition collapses to the chosen branch,
  // whatever the branches are — the guard below only admits scalar literals,
  // so this is checked before it.
  if (n.type === "select") return foldSelect(n);
  let integerOperands = foldIntegerOperands(n);
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

/**
 * Render a for-loop's update clause.
 *
 * The clause is authored as statements — `(i) => i.assign(i.add(1))` — so it
 * arrives with its work in `body` and only a bare variable reference in `expr`.
 * Emitting `expr` alone drops the increment and produces an infinite loop.
 *
 * GLSL's update slot accepts a comma expression, so every statement survives.
 * WGSL's grammar allows exactly one update statement, so callers there keep
 * the last.
 */
export const FOR_UPDATE_BLOCK_MESSAGE =
  "[RMSL] A for-loop's update cannot contain a block. Move the branch into the loop body, or write the loop with While.";

export function forUpdateStatements(update: CompiledNode): string[] {
  // A nested block cannot go in either language's update slot: GLSL's takes an
  // expression, and accepting one in WGSL alone would make a program that runs
  // on one backend and not the other.
  if (update.body.some((line) => line.includes("{"))) {
    throw new Error(FOR_UPDATE_BLOCK_MESSAGE);
  }
  return update.body;
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
  shaderStage: "vertex" | "fragment" | "compute",
  lastType: string | undefined,
  positionWritten: boolean,
  declaresOutput = false,
): void {
  if (shaderStage === "fragment") {
    assertFragmentResult(lastType, declaresOutput);
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
  throw new Error(
    `[RMSL] A vertex shader has to produce a position. This one ` +
      (lastType === undefined || lastType === "void"
        ? `returns nothing and never assigns builtinPosition(). Return a vec4, or ` +
          `assign builtinPosition() yourself.`
        : `returns ${lastType}, which cannot become one. Wrap it — for example ` + `vec4(value, 1.0).`),
  );
}

function assertFragmentResult(lastType: string | undefined, declaresOutput: boolean): void {
  if (declaresOutput) return;
  if (lastType === undefined || lastType === "void" || lastType === "vec4") return;
  throw new Error(
    `[RMSL] A fragment shader that declares no output writes its result as the colour. ` +
      `This one returns ${lastType}, which has no colour to become. Return a vec4, or declare an output.`,
  );
}

/**
 * The roots of a fragment stage with an `outputStruct` it returns written out:
 * member `i` becomes an assignment to an output at location `i`, in slot
 * `_rmsl_out<i>`. Only the last value of a stage can be one.
 */
function lowerOutputStruct<T extends BaseNode<ShaderType>>(roots: readonly T[]): T[] {
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
    throw new Error(
      "[RMSL] outputStruct is the value a fragment stage returns. It cannot be an operand or a statement.",
    );
  }
  return lowered;
}

/**
 * The roots of a stage, ready to compile. A fragment stage has its
 * `outputStruct` written out and its implicit colour converted, see
 * {@link fragmentColour}. Any other stage, and a program with no stage,
 * refuses an `outputStruct`, which is a fragment stage's result.
 */
export function prepareRoots<T extends Node<ShaderType>>(stage: string | undefined, roots: readonly T[]): T[] {
  if (stage === "fragment")
    return fragmentColour(lowerOutputStruct(roots as readonly BaseNode<ShaderType>[]) as unknown as T[]);
  if (someNode(roots, (n) => n.type === "outputStruct")) {
    throw new Error(
      `[RMSL] outputStruct is the value a fragment stage returns, and this program is compiled ` +
        (stage === undefined ? "with no stage" : `as a ${stage} stage`) +
        `. Compile it as a fragment stage.`,
    );
  }
  return [...roots];
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
 * A literal index outside the components of a vector or the columns of a matrix
 * is refused by every target, so the same program is not refused on one and
 * compiled to a read of nothing on another. Asked wherever an element is
 * compiled, which also reaches a graph that `deserialize` rebuilt. Returns the
 * index when it is inside, and `undefined` when it is not a literal.
 */
export function assertLiteralIndexInRange(target: { _t?: string }, index: BaseNode<ShaderType>): number | undefined {
  if (!isLeafLiteral(index)) return undefined;
  const type = target._t as string;
  const shape = MATRIX_DIMENSIONS[type];
  const count = shape ? shape[0] : componentCountOf(type);
  const k = Math.trunc(Number(index.value));
  if (!(k >= 0 && k < count)) {
    throw new Error(`[RMSL] index ${k} is outside a ${type}'s ${shape ? "columns" : "components"} 0 to ${count - 1}`);
  }
  return k;
}

/** `Math.round` takes a half toward +Infinity. WGSL takes it to the even neighbour, so folding does too. */
function roundHalfToEven(x: number): number {
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
export function assertSquareMatrix(operandType: string | undefined): number {
  let shape = MATRIX_DIMENSIONS[operandType as string];
  if (shape === undefined || shape[0] !== shape[1]) {
    throw new Error(`[RMSL] inverse() needs a square matrix, but this one is ` + `${operandType ?? "untyped"}.`);
  }
  return shape[0];
}

/**
 * A compute dispatch has no vertex or fragment stage, so what those stages
 * pass in and out has nothing to attach to. Throws when `stage` is compute.
 */
export function assertNotInAComputeStage(
  stage: "vertex" | "fragment" | "compute",
  refused: { action: string; because: string; instead: string },
): void {
  if (stage !== "compute") return;
  throw new Error(`[RMSL] A compute program cannot ${refused.action}, because ${refused.because}. ${refused.instead}`);
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
  throw new Error(
    "[RMSL] builtinPosition() is the vertex stage's output position, and a " +
      "fragment stage cannot read it. Pass the value you need through a " +
      "varying() instead.",
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
export function assertReadsNoStageInput(roots: unknown, kind: "routine" | "grid"): void {
  someNode(roots, (node) => {
    const input = STAGE_INPUTS[node.type];
    if (!input || (kind === "grid" && GRID_INPUTS.has(node.type))) return false;
    throw new Error(
      `[RMSL] ${input.what} is an input of a ${input.stage}, and a ${kind} has no stage to give it. ` +
        `Compile the program as a ${input.stage} instead.`,
    );
  });
}

/** Every storage attribute reachable from the roots, keyed by the slot name its nodes compile to. */
export function storageAttributes(roots: unknown): Map<string, StorageBufferAttribute> {
  const attributes = new Map<string, StorageBufferAttribute>();
  someNode(roots, (node) => {
    if (node.type === "storage") attributes.set(node.value.slot, node.value.attribute);
  });
  return attributes;
}

/** What a uniform, attribute or varying node declares, in the words of an error about its name. */
function describeNamed(node: any): string | undefined {
  const value = node.value;
  const article = /^[aeiou]/.test(value?.shaderType ?? "") ? "an" : "a";
  if (node.type === "uniform") return `${article} ${value.shaderType} uniform`;
  if (node.type === "uniformArray") return `a uniform array of ${value.length} ${value.shaderType}`;
  if (node.type === "attribute") return `${article} ${value.shaderType} attribute`;
  if (node.type === "varying") return `${article} ${value.shaderType} varying`;
  return undefined;
}

/**
 * Throws when two uniforms, attributes or varyings reachable from the roots
 * share a name but not what they declare. One name is one declaration on
 * every backend, so nodes made separately with one name read the same value.
 */
export function assertOneDeclarationPerName(roots: unknown): void {
  const declared = new Map<string, string>();
  someNode(roots, (node) => {
    const what = describeNamed(node);
    if (what === undefined) return;
    const name = node.value.slot as string;
    const earlier = declared.get(name);
    if (earlier === undefined) declared.set(name, what);
    else if (earlier !== what) {
      throw new Error(
        `[RMSL] "${name}" names ${earlier} and ${what}: one name is one declaration. Give them different names`,
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
 *
 * A variable is one node, so a renamed copy of it takes the place of every read.
 * The roots come back as copies where a variable was renamed, and the graph the
 * caller holds keeps the names it was given.
 */
export function numberClashingVariables<T>(roots: T): T {
  const byName = new Map<string, any[]>();
  someNode(roots, (node) => {
    if (node.type !== "var" || node.value.parameter) return;
    const group = byName.get(node.value.varName);
    if (group === undefined) byName.set(node.value.varName, [node]);
    else if (!group.includes(node)) group.push(node);
  });
  const taken = new Set(byName.keys());
  const renamed = new Map<any, any>();
  for (const [name, variables] of byName) {
    for (const variable of variables.slice(1)) {
      let number = 1;
      while (taken.has(`${name}${number}`)) number++;
      const varName = `${name}${number}`;
      taken.add(varName);
      renamed.set(
        variable,
        Object.assign(Object.create(Object.getPrototypeOf(variable)), variable, {
          value: { ...variable.value, varName },
        }),
      );
    }
  }
  if (renamed.size === 0) return roots;
  const copies = new Map<any, any>();
  const substitute = (node: any): any => {
    if (Array.isArray(node)) return node.map(substitute);
    if (!node || typeof node !== "object") return node;
    const known = renamed.get(node) ?? copies.get(node);
    if (known !== undefined) return known;
    if (!Array.isArray(node.params)) return node;
    const params = node.params.map(substitute);
    const result = params.every((p: any, i: number) => p === node.params[i])
      ? node
      : Object.assign(Object.create(Object.getPrototypeOf(node)), node, { params });
    copies.set(node, result);
    return result;
  };
  return substitute(roots);
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
};

/** The stage outputs only one stage writes, which stage, and what to call them in an error. */
const WRITTEN_BY_ONE_STAGE: Record<string, { stage: string; name: string }> = {
  varying: { stage: "vertex", name: "a varying" },
  builtinPosition: { stage: "vertex", name: "the position" },
  builtinFragDepth: { stage: "fragment", name: "the fragment depth" },
};

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

/**
 * Throws unless an assignment to `target`, compiled for `stage`, writes
 * something writable: a variable, a storage element or a stage output,
 * directly or through any swizzle, component or column of it, where no
 * swizzle names a component more than once. Called by every backend, so a
 * uniform, an attribute or a computed value is refused the same way on each,
 * and so is an element of a storage node made read-only, a stage output
 * written outside the stage that writes it, and a parameter of the compiled
 * function.
 */
export function assertAssignable(target: any, stage: "vertex" | "fragment" | "compute"): void {
  for (let node = target; ASSIGNMENT_PATH.has(node?.type); node = node.params[0]) {
    if (node.type === "swizzle" && new Set(node.value).size !== node.value.length) {
      throw new Error(
        `[RMSL] can't assign through the swizzle .${node.value}, which names a component more than once; name each component once`,
      );
    }
  }
  const root = assignmentRoot(target);
  if (root?.type === "storage") {
    throw new Error("[RMSL] can't assign to a whole storage buffer; assign to one of its elements with .element(i)");
  }
  if (!ASSIGNABLE.has(root?.type)) {
    const what = READ_ONLY_NAMES[root?.type] ?? "a computed value";
    throw new Error(
      `[RMSL] can't assign to ${what}: only a variable, a storage element or a stage output can be assigned; copy the value into a variable with toVar() first`,
    );
  }
  if (root.type === "var" && root.value.parameter) {
    throw new Error(
      `[RMSL] can't assign to "${root.value.varName}", a parameter of the compiled function, whose value belongs to the caller; copy it into a variable with toVar() first`,
    );
  }
  const writer = WRITTEN_BY_ONE_STAGE[root.type];
  if (writer !== undefined && writer.stage !== stage) {
    throw new Error(
      `[RMSL] can't assign to ${writer.name} in a ${stage} stage; only a ${writer.stage} stage writes it`,
    );
  }
  if (root.type === "storageElement" && root.params[0].value.access === "read") {
    throw new Error("[RMSL] can't assign to an element of a storage node made read-only with toReadOnly()");
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
export function assertWriteFits(attribute: StorageBufferAttribute, length: number, offset: number): void {
  const capacity = attribute.count * attribute.itemSize;
  if (offset * attribute.itemSize + length > capacity) {
    throw new Error(
      `[RMSL] writing ${length} values from element ${offset} runs past the end of an attribute of ${capacity} values`,
    );
  }
}
