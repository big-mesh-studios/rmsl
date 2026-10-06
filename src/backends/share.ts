import { node, var_ } from "../core";

/** The prefix of the variable a shared node gets, under the `_rmsl_` prefix the compiler keeps for its own names. */
const SHARED_NAME_PREFIX = "_rmsl_gen_";

/** Node types that are statements, whose parameters are blocks or expressions the pass places by kind. */
const STATEMENTS = new Set(["seq", "let", "assign", "if", "for", "while", "break", "continue", "return", "discard"]);

/**
 * Node types read where they are, however often. A swizzle or an element reads
 * a node that is already a name, or an address, and costs nothing to read again.
 */
const READ_WHERE_IT_IS = new Set([
  "seq",
  "var",
  "swizzle",
  "vectorElement",
  "matrixElement",
  "storageElement",
  "uniformArrayElement",
]);

/** The node types whose value a statement can change, so a stored copy of a node that reads them can go stale. */
const MUTABLE_READS = new Set([
  "var",
  "storageElement",
  "storage",
  "output",
  "varying",
  "builtinPosition",
  "builtinFragDepth",
]);

/** Node types that are literals, which a target folds. */
const LITERALS = new Set(["float", "int", "uint", "bool"]);

const isStatement = (n: any) => STATEMENTS.has(n.type);

/** Whether `n` is a literal, or an expression of literals alone, which a target folds into one. */
function isConstant(n: any, memo: Map<any, boolean>): boolean {
  const known = memo.get(n);
  if (known !== undefined) return known;
  const result =
    Array.isArray(n.params) && n.params.length > 0
      ? n.params.every((p: any) => isConstant(p, memo))
      : LITERALS.has(n.type) || (n.type === n._t && Array.isArray(n.value));
  memo.set(n, result);
  return result;
}

/** Whether an expression holds an inline `Fn` result with statements, whose statements run where it is read. */
function holdsStatements(expression: any, memo: Map<any, boolean>): boolean {
  const known = memo.get(expression);
  if (known !== undefined) return known;
  memo.set(expression, false);
  const result =
    (expression.type === "seq" && (expression.params ?? []).some(isStatement)) ||
    (Array.isArray(expression.params) && expression.params.some((p: any) => holdsStatements(p, memo)));
  memo.set(expression, result);
  return result;
}

/**
 * Gives every node that an operation reads more than once a variable of its
 * own, so the program computes it once, where it first runs. A later read in
 * that block or a block inside it reads the variable.
 *
 * The program is the statements the roots hold. The pass counts the reads of
 * each node over them, then walks them in the order they run. The first read of
 * a shared node inserts `let _rmsl_gen_N = <the node>` ahead of the statement
 * that reads it, and the read takes the variable.
 *
 * A variable is read only where the value it holds is still the value the node
 * computes: not outside the block that holds it, and not after a statement that
 * could have changed what the node reads. A loop runs its body again, so a
 * variable made before a loop is read inside it only when the node reads
 * nothing a statement can change. The header of a loop is left as written,
 * since it runs again on every iteration, and so is a statement whose
 * expression holds an inline `Fn` with statements of its own.
 *
 * The statements and expressions the pass changes are copied, and the graph the
 * caller holds is left as it is. A root that is an expression gets a block of
 * its own.
 */
export function shareNodes<T>(roots: T): T {
  const reads = new Map<any, number>();
  const expanded = new Set<any>();
  const statementsMemo = new Map<any, boolean>();
  const constantsMemo = new Map<any, boolean>();
  /** Whether `n` is a node that gets a variable when it is read more than once. */
  const isShareable = (n: any): boolean =>
    Array.isArray(n.params) &&
    n.params.length > 0 &&
    n._t !== "void" &&
    !READ_WHERE_IT_IS.has(n.type) &&
    !isConstant(n, constantsMemo);

  const read = (expression: any) => {
    if (!expression || typeof expression !== "object") return;
    reads.set(expression, (reads.get(expression) ?? 0) + 1);
    if (expanded.has(expression)) return;
    expanded.add(expression);
    if (Array.isArray(expression.params)) for (const p of expression.params) read(p);
  };

  /** Counts the reads of the expressions of a statement, and of the statements inside it. */
  const count = (statement: any): void => {
    if (!statement || typeof statement !== "object" || expanded.has(statement)) return;
    switch (statement.type) {
      case "seq":
        expanded.add(statement);
        for (const p of statement.params ?? []) {
          if (isStatement(p)) count(p);
          else if (!holdsStatements(p, statementsMemo)) read(p);
        }
        return;
      case "let":
      case "assign":
        expanded.add(statement);
        // The value of a `let` is where its variable is made, so it is not a read of that node.
        if (!holdsStatements(statement.params[1], statementsMemo)) {
          if (statement.type === "assign") read(statement.params[1]);
          else if (Array.isArray(statement.params[1].params)) for (const p of statement.params[1].params) read(p);
        }
        return;
      case "if":
        expanded.add(statement);
        if (!holdsStatements(statement.params[0], statementsMemo)) read(statement.params[0]);
        count(statement.params[1]);
        if (statement.params[2]) count(statement.params[2]);
        return;
      case "for":
        expanded.add(statement);
        count(statement.params[3]);
        return;
      case "while":
        expanded.add(statement);
        count(statement.params[1]);
        return;
      default:
        expanded.add(statement);
    }
  };

  let next = 0;
  let epoch = 0;
  let blockCount = 0;
  const blocks: number[] = [];
  const pure = new Map<any, boolean>();
  /** The variable that holds each shared node, where it was made, and when. */
  const held = new Map<any, { variable: any; block: number; epoch: number; pure: boolean }>();

  const isPure = (n: any): boolean => {
    const known = pure.get(n);
    if (known !== undefined) return known;
    pure.set(n, true);
    const result = !(MUTABLE_READS.has(n.type) && !n.value?.parameter) && (n.params ?? []).every(isPure);
    pure.set(n, result);
    return result;
  };

  const copyOf = (original: any, params: any[]) =>
    Object.assign(Object.create(Object.getPrototypeOf(original)), original, { params });

  /** `expression` with each shared node replaced by its variable, the `let`s that make them pushed on `lets`. */
  const share = (expression: any, lets: any[], copies: Map<any, any>): any => {
    if (!expression || typeof expression !== "object" || !Array.isArray(expression.params)) return expression;
    const copied = copies.get(expression);
    if (copied !== undefined) return copied;
    const shared = (reads.get(expression) ?? 0) > 1 && isShareable(expression);
    const stored = shared ? held.get(expression) : undefined;
    if (stored && blocks.includes(stored.block) && (stored.pure || stored.epoch === epoch)) {
      copies.set(expression, stored.variable);
      return stored.variable;
    }
    let params = expression.params;
    for (let i = 0; i < params.length; i++) {
      const rewrittenParam = share(params[i], lets, copies);
      if (rewrittenParam !== params[i]) {
        if (params === expression.params) params = params.slice();
        params[i] = rewrittenParam;
      }
    }
    const rebuilt = params === expression.params ? expression : copyOf(expression, params);
    let result = rebuilt;
    if (shared) {
      const variable = var_(`${SHARED_NAME_PREFIX}${next++}`, expression._t);
      lets.push(node({ _t: "void", type: "let", params: [variable, rebuilt] }));
      held.set(expression, { variable, block: blocks[blocks.length - 1]!, epoch, pure: isPure(expression) });
      result = variable;
    }
    copies.set(expression, result);
    return result;
  };

  /** A statement's expression with its shared nodes replaced, and the `let`s that make them added to `out`. */
  const rewrite = (expression: any, out: any[]): any => {
    if (holdsStatements(expression, statementsMemo)) return expression;
    return share(expression, out, new Map());
  };

  const sameParams = (a: any[], b: any[]) => a.length === b.length && a.every((p, i) => p === b[i]);

  /** What each statement and block became, so one that several roots hold stays one node. */
  const rewritten = new Map<any, any>();

  /** A copy of the block with the `let`s its statements need, which holds the statements it rewrote. */
  const block = (seq: any): any => {
    const known = rewritten.get(seq);
    if (known !== undefined) return known;
    blocks.push(blockCount++);
    const out: any[] = [];
    for (const p of seq.params ?? []) {
      if (isStatement(p)) statement(p, out);
      else out.push(rewrite(p, out));
    }
    blocks.pop();
    const result = sameParams(out, seq.params ?? []) ? seq : copyOf(seq, out);
    rewritten.set(seq, result);
    return result;
  };

  /** Pushes the statement, rewritten, onto `out`, after any `let` it needs. */
  const statement = (s: any, out: any[]): void => {
    const known = rewritten.get(s);
    if (known !== undefined) {
      epoch++;
      out.push(known);
      return;
    }
    let result = s;
    switch (s.type) {
      case "let": {
        const value = s.params[1];
        if (!holdsStatements(value, statementsMemo) && Array.isArray(value.params)) {
          // The value of a `let` is where its variable is made: only what it reads is shared.
          const params = value.params.map((p: any) => rewrite(p, out));
          if (!sameParams(params, value.params)) result = copyOf(s, [s.params[0], copyOf(value, params)]);
        }
        epoch++;
        break;
      }
      case "assign": {
        const value = rewrite(s.params[1], out);
        if (value !== s.params[1]) result = copyOf(s, [s.params[0], value]);
        epoch++;
        break;
      }
      case "if": {
        const params = [rewrite(s.params[0], out)];
        for (const branch of s.params.slice(1)) params.push(branch ? enclosed(branch) : branch);
        if (!sameParams(params, s.params)) result = copyOf(s, params);
        epoch++;
        break;
      }
      case "for":
      case "while": {
        epoch++;
        const params = s.params.slice();
        params[params.length - 1] = enclosed(params[params.length - 1]);
        if (!sameParams(params, s.params)) result = copyOf(s, params);
        epoch++;
        break;
      }
      case "seq":
        result = block(s);
        break;
    }
    rewritten.set(s, result);
    out.push(result);
  };

  /** A block, or the statement standing for one, that runs under its own condition. */
  const enclosed = (body: any): any => {
    if (body.type === "seq") return block(body);
    blocks.push(blockCount++);
    const out: any[] = [];
    statement(body, out);
    blocks.pop();
    return out.length === 1 ? out[0] : node({ _t: "void", type: "seq", params: out });
  };

  /** A root that is an expression gets a block of its own, to hold the variables it needs. */
  const normalize = (root: any): any => {
    if (Array.isArray(root)) return root.map(normalize);
    if (!root || typeof root !== "object" || root.type === "seq") return { root, block: root };
    return { root, block: node({ _t: root._t, type: "seq", params: [root] }) };
  };

  const normalized = normalize(roots);
  const blocksOf = (n: any): any[] =>
    Array.isArray(n) ? n.flatMap(blocksOf) : n.block?.type === "seq" ? [n.block] : [];
  for (const b of blocksOf(normalized)) count(b);

  /** What a caller gets for each root: the root, or its rewritten block when the pass changed it. */
  const result = (n: any): any => {
    if (Array.isArray(n)) return n.map(result);
    const { root, block: own } = n;
    if (own?.type !== "seq") return root;
    const changed = block(own);
    if (own === root) return changed;
    return changed.params.length === 1 ? root : changed;
  };
  return result(normalized) as T;
}
