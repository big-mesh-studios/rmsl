// === Compiler internals shared by all four backends ===
// CompileCtx/CompiledNode (the context and per-node result every backend's
// node walker threads through), precedence tables, constant folding, and the
// handful of assertions/helpers more than one backend calls.
import { BaseNode, MATRIX_DIMENSIONS, Node, NodeImpl, ShaderType } from "../core";
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
  /** `length` is set only for uniform arrays, and gives their element count. */
  uniforms: Map<number, { type: string; slot: string; length?: number }>;
  attributes: Map<number, { type: string; slot: string }>;
  varyings: Map<number, { id: number; type: string; slot: string }>;
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
  /** Fn parameter names, which the JS target reads from `ctx.params`. */
  jsParams: Set<string>;
  /** Names of JS helper functions the compiled function needs. */
  jsHelpers: Set<string>;
  /**
   * Slot a vector/matrix-typed expression should be written into, when the JS
   * target is lowering an assignment. `null` means a plain expression.
   */
  outTarget: string | null;
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

export function tryFold(n: BaseNode<ShaderType>): BaseNode<ShaderType> | null {
  // A select with a literal condition collapses to the chosen branch, whatever
  // the branches are — the guard below only admits scalar literals, so this is
  // checked before it.
  if (n.type === "select") {
    let cond = n.params?.[0];
    if (cond && isLeafLiteral(cond)) return (cond.value ? n.params![1] : n.params![2]) ?? null;
  }
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
        return mkNode({ _t: t, type: t, value: t === "int" || t === "uint" ? (a + b) | 0 : a + b });
      case "sub":
        return mkNode({ _t: t, type: t, value: t === "int" || t === "uint" ? (a - b) | 0 : a - b });
      case "mul":
        return mkNode({ _t: t, type: t, value: t === "int" || t === "uint" ? (a * b) | 0 : a * b });
      case "div":
        return mkNode({ _t: t, type: t, value: t === "int" || t === "uint" ? (a / b) | 0 : a / b });
      case "negate":
        return mkNode({ _t: t, type: t, value: t === "int" || t === "uint" ? -a | 0 : -a });
      // JavaScript's % truncates toward zero. The float operation is floored,
      // following GLSL's mod(), so folding it with % would give a literal that
      // disagrees with what the same expression computes when its operands are
      // not constants. The integer path keeps % because that is what both
      // backends emit for integers.
      case "mod":
        return mkNode({
          _t: t,
          type: t,
          value: t === "int" || t === "uint" ? (a % b) | 0 : a - b * Math.floor(a / b),
        });
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
        return mkNode({ _t: t, type: t, value: Math.round(a) });
      case "trunc":
        return mkNode({ _t: t, type: t, value: Math.trunc(a) });
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
export function forUpdateStatements(update: CompiledNode): string[] {
  // A nested block cannot go in either language's update slot: GLSL's takes an
  // expression, and accepting one in WGSL alone would make a program that runs
  // on one backend and not the other.
  if (update.body.some((line) => line.includes("{"))) {
    throw new Error(
      "[RMSL] A for-loop's update cannot contain a block. Move the branch into " +
        "the loop body, or write the loop with While.",
    );
  }
  return update.body;
}

/** Drop a trailing semicolon, for the slots that take an expression. */
export function withoutSemicolon(statement: string): string {
  return statement.endsWith(";") ? statement.slice(0, -1) : statement;
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
 * A fragment stage is deliberately not checked. A shader with no colour output
 * is legal, so "no result" there is a choice rather than a mistake.
 */
export function assertStageResult(
  shaderStage: "vertex" | "fragment" | "compute",
  lastType: string | undefined,
  positionWritten: boolean,
): void {
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
 * Resolve a chain of swizzles down to the variable underneath it.
 *
 * Only a variable can be assigned to. `a.xyz` is a value, so `a.xyz.xy = e`
 * has to become a write to `a` — and which components of `a` that is takes
 * composing the patterns: the outer pattern indexes into the inner one, so
 * `a.yzw.xy` selects the first two of y, z, w, which is `a.yz`.
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
 * Void is the other way to satisfy a vertex stage: assign builtinPosition()
 * yourself and return nothing. A function whose body returns nothing has that
 * type, so the two cases are exactly the two the signature admits. Whether an
 * assignment actually happened is not something a signature can see, so that
 * half stays a run-time check.
 *
 * Several values may be returned at once, of which the last becomes the
 * position — so that is the one constrained, and the values before it are
 * whatever the shader needed on the way there. Saying so requires knowing which
 * value is last, which is why `Fn` infers an array return as a tuple.
 */
export type VertexRoot = Node<"vec4"> | readonly [...Node<ShaderType>[], Node<"vec4">] | void;

export type CompileFnOptions = {
  name: string;
  params: Array<{ name: string; type: ShaderType }>;
};
