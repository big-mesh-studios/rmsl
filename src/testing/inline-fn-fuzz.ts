import { Break, Continue, Fn, float, For, If, instancedArray, int, Loop, Return, uniform, While } from "../rmsl";
import { INLINE_FN_JUMP_IN_A_DEEPER_LOOP_MESSAGE, INLINE_FN_JUMP_WITH_NO_LOOP_MESSAGE } from "../backends/share";
import { createWgslCompute } from "../wgsl";

/** An expression of a generated program: a float. */
export type Expr =
  | { k: "lit"; v: number }
  | { k: "read"; fn: number; value: number }
  | { k: "add"; a: Expr; b: Expr }
  | { k: "double"; e: Expr; times: number };

/** A condition of an `If`: uniform `u` above a half, plus `e` times zero, which reads `e` without changing the test. */
export type Cond = { u: number; e?: Expr };

/** A statement of a generated program. Each `out` adds its expression to a storage slot of its own. */
export type Stmt =
  | { k: "out"; slot: number; e: Expr }
  | { k: "if"; cond: Cond; then: Stmt[]; else?: Stmt[] }
  | { k: "loop"; n: number; body: Stmt[] }
  | { k: "for"; n: number; init?: Stmt; cond?: Stmt; update?: Stmt; body: Stmt[] }
  | { k: "while"; form: WhileForm; n: number; cond?: Expr; condOut?: Stmt; body: Stmt[] };

/**
 * The form of a loop whose condition is a bool: `While` given a node, `While`
 * given a function that writes `out` before its test, or `Loop` given a bool.
 */
export type WhileForm = "node" | "function" | "loopBool";

/** A jump an inline `Fn` makes at the end of its body, on the run whose count is `at`. */
export type Jump = { kind: "break" | "continue" | "return"; at: number };

/**
 * An inline `Fn`, called outside the program's body. It adds one to its own
 * counter, keeps the count as `r`, runs `body`, makes `jump` when the count is
 * `jump.at`, and returns `r + extra[0]`, and `r * 10 + extra[1]` when it
 * returns two values.
 */
export type FnDef = { body: Stmt[]; values: 1 | 2; extra: (Expr | undefined)[]; jump?: Jump };

/** A generated program: inline `Fn`s, each reading only earlier ones, and roots run in order. */
export type Program = { seed: number; fns: FnDef[]; roots: Stmt[][]; slots: number };

/** What a program does: refused with one of `messages`, or the counts of each `Fn` and the totals of each slot. */
export type Outcome = { refused: string[] } | { counters: number[]; outs: number[] };

/** A deterministic generator of numbers in [0, 1) from `seed`. */
export function random(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * A random program of inline `Fn`s from `seed`. Any root reads any `Fn`, and an
 * `Fn` reads any made before it. An `Fn` read in a `For` update holds no block
 * and no jump, as the update of a `For` takes none.
 */
export function generate(seed: number): Program {
  const next = random(seed);
  const pick = <T>(items: readonly T[]): T => items[Math.floor(next() * items.length)]!;
  const chance = (p: number) => next() < p;
  const rootCount = chance(0.5) ? 1 : chance(0.6) ? 2 : 3;
  const fnCount = 1 + Math.floor(next() * 5);
  const fns: FnDef[] = [];
  /** Whether `Fn` `i` holds no block and no jump, through every `Fn` it reads. */
  const simple: boolean[] = [];
  let slots = 0;

  /** An expression of `readable` `Fn`s, `depth` levels at most. */
  const expr = (readable: number[], depth: number): Expr => {
    const roll = next();
    if (readable.length > 0 && (roll < 0.45 || depth === 0)) {
      const fn = pick(readable);
      return { k: "read", fn, value: fns[fn]!.values === 2 && chance(0.5) ? 1 : 0 };
    }
    if (depth === 0 || roll < 0.6) return { k: "lit", v: 1 + Math.floor(next() * 3) };
    if (roll < 0.8) return { k: "add", a: expr(readable, depth - 1), b: expr(readable, depth - 1) };
    return { k: "double", e: expr(readable, depth - 1), times: 1 + Math.floor(next() * 3) };
  };
  const out = (readable: number[]): Stmt => ({ k: "out", slot: slots++, e: expr(readable, 2) });
  /** A block of statements `depth` levels of blocks deep at most; `plain` keeps it to writes alone. */
  const block = (readable: number[], depth: number, plain = false): Stmt[] => {
    const length = 1 + Math.floor(next() * 3);
    return Array.from({ length }, () => statement(readable, depth, plain));
  };
  const statement = (readable: number[], depth: number, plain: boolean): Stmt => {
    const roll = next();
    if (plain || depth === 0 || roll < 0.4) return out(readable);
    const n = 1 + Math.floor(next() * 3);
    if (roll < 0.6) {
      const cond: Cond = { u: Math.floor(next() * 2), e: chance(0.3) ? expr(readable, 1) : undefined };
      const then = block(readable, depth - 1);
      return chance(0.5) ? { k: "if", cond, then, else: block(readable, depth - 1) } : { k: "if", cond, then };
    }
    if (roll < 0.72) return { k: "loop", n, body: block(readable, depth - 1) };
    if (roll < 0.88) {
      const updatable = readable.filter((i) => simple[i]);
      return {
        k: "for",
        n,
        init: chance(0.3) ? out(readable) : undefined,
        cond: chance(0.4) ? out(readable) : undefined,
        update: chance(0.3) ? out(updatable) : undefined,
        body: block(readable, depth - 1),
      };
    }
    const form = pick(["node", "function", "loopBool"] as const);
    const reads = chance(0.6);
    return {
      k: "while",
      form,
      n,
      cond: reads && form !== "function" ? expr(readable, 1) : undefined,
      condOut: reads && form === "function" ? out(readable) : undefined,
      body: block(readable, depth - 1),
    };
  };

  for (let i = 0; i < fnCount; i++) {
    const readable = Array.from({ length: i }, (_, j) => j);
    const plain = chance(0.4);
    const body = chance(0.7) ? block(readable, 1, plain) : [];
    const values: 1 | 2 = chance(0.3) ? 2 : 1;
    const extra = Array.from({ length: values }, () =>
      chance(0.25) && readable.length > 0 ? expr(readable, 1) : undefined,
    );
    const jump: Jump | undefined = chance(0.2)
      ? { kind: pick(["break", "continue", "return"] as const), at: 1 + Math.floor(next() * 3) }
      : undefined;
    fns.push({ body, values, extra, jump });
    simple.push(plain && !jump && [...readsOf({ body, values, extra })].every((j) => simple[j]));
  }
  const everyFn = Array.from({ length: fnCount }, (_, i) => i);
  const roots = Array.from({ length: rootCount }, () => block(everyFn, 3));
  return { seed, fns, roots, slots };
}

/** Every `Fn` that `fn` reads in its body or its values. */
function readsOf(fn: Pick<FnDef, "body" | "values" | "extra">): Set<number> {
  const found = new Set<number>();
  const inExpr = (e: Expr | undefined): void => {
    if (!e) return;
    if (e.k === "read") found.add(e.fn);
    else if (e.k === "add") (inExpr(e.a), inExpr(e.b));
    else if (e.k === "double") inExpr(e.e);
  };
  const inStmt = (s: Stmt | undefined): void => {
    if (!s) return;
    if (s.k === "out") inExpr(s.e);
    else if (s.k === "if") (inExpr(s.cond.e), s.then.forEach(inStmt), s.else?.forEach(inStmt));
    else if (s.k === "loop") s.body.forEach(inStmt);
    else if (s.k === "for") [s.init, s.cond, s.update, ...s.body].forEach(inStmt);
    else (inExpr(s.cond), inStmt(s.condOut), s.body.forEach(inStmt));
  };
  fn.body.forEach(inStmt);
  fn.extra.forEach(inExpr);
  return found;
}

/** The storage and uniforms a built program reads and writes. */
export type Built = {
  build: () => any[];
  counter: { name: string };
  out: { name: string };
  uniforms: ReturnType<typeof uniform<"float">>[];
};

/** `program` as rmsl: a build function for any target, and the buffers and uniforms it uses. */
export function build(program: Program): Built {
  const counter = instancedArray(Math.max(1, program.fns.length), "float");
  const out = instancedArray(Math.max(1, program.slots), "float");
  const uniforms = [uniform("float"), uniform("float")];
  const make = () => {
    const results: any[] = [];
    const value = (fn: number, index: number) => {
      const result = results[fn];
      return Array.isArray(result) ? result[index] : result;
    };
    const expression = (e: Expr): any => {
      if (e.k === "lit") return float(e.v);
      if (e.k === "read") return value(e.fn, e.value);
      if (e.k === "add") return expression(e.a).add(expression(e.b));
      let x = expression(e.e);
      for (let t = 0; t < e.times; t++) x = x.add(x);
      return x;
    };
    const statements = (list: Stmt[]) => list.forEach(statement);
    const statement = (s: Stmt): void => {
      if (s.k === "out") {
        out.element(int(s.slot)).addAssign(expression(s.e));
      } else if (s.k === "if") {
        const u = uniforms[s.cond.u]!;
        const test = (s.cond.e ? u.add(expression(s.cond.e).mul(0)) : u).greaterThan(0.5);
        const chain = If(test, () => statements(s.then));
        if (s.else) chain.Else(() => statements(s.else!));
      } else if (s.k === "loop") {
        Loop(s.n, () => statements(s.body));
      } else if (s.k === "for") {
        For(
          () => {
            if (s.init) statement(s.init);
            return float(0).toVar();
          },
          (n) => {
            if (s.cond) statement(s.cond);
            return n.lessThan(s.n);
          },
          (n) => {
            if (s.update) statement(s.update);
            n.addAssign(1);
          },
          () => statements(s.body),
        );
      } else {
        const v = float(0).toVar();
        const test = () => v.lessThan(s.cond ? float(s.n).add(expression(s.cond).mul(0)) : float(s.n));
        const body = () => {
          v.addAssign(1);
          statements(s.body);
        };
        if (s.form === "node") While(test(), body);
        else if (s.form === "loopBool") Loop(test(), body);
        else
          While(() => {
            if (s.condOut) statement(s.condOut);
            return v.lessThan(s.n);
          }, body);
      }
    };
    program.fns.forEach((fn, i) => {
      results[i] = Fn(() => {
        counter.element(int(i)).addAssign(1);
        const r = counter.element(int(i)).toVar();
        statements(fn.body);
        if (fn.jump) {
          const jump = { break: Break, continue: Continue, return: Return }[fn.jump.kind];
          If(r.equal(fn.jump.at), () => jump());
        }
        const plus = (base: any, extra: Expr | undefined) => (extra ? base.add(expression(extra)) : base);
        const first = plus(r, fn.extra[0]);
        return fn.values === 2 ? [first, plus(r.mul(10), fn.extra[1])] : first;
      })();
    });
    return program.roots.map((root) => Fn(() => statements(root))());
  };
  return { build: make, counter, out, uniforms };
}

/**
 * A place in a block of the model. `pos` orders the places of one block: an
 * item of its own is `[k]`, and statement `k` of an `Fn` whose statements were
 * made statements of the block, before the place `p`, is `[...p, -1, order, k]`.
 */
type Frame = { block: Block; pos: number[] };

/** A block of the model: one that runs again on every iteration of a loop when `repeats`. */
type Block = { repeats: boolean };

/** A read of an `Fn` value: inside the body of `within`, or of the program when -1, at `path` from there. */
type Read = { fn: number; value: number; within: number; path: Frame[]; times: number };

/** Orders two places of one block. A place before an item sorts ahead of the item itself. */
function comparePos(a: number[], b: number[]): number {
  for (let i = 0; i < Math.min(a.length, b.length); i++) if (a[i] !== b[i]) return a[i]! - b[i]!;
  return b.length - a.length;
}

const samePos = (a: number[], b: number[]) => a.length === b.length && a.every((x, i) => x === b[i]);

/** The frames of the place that runs before every read at `paths`: the innermost block holding them all, at the first place there. */
function placeBefore(paths: Frame[][]): Frame[] {
  const place: Frame[] = [];
  for (let depth = 0; ; depth++) {
    const frames = paths.map((path) => path[depth]);
    const block = frames[0]?.block;
    if (frames.some((frame) => !frame || frame.block !== block)) return place;
    const least = frames.map((f) => f!.pos).reduce((a, b) => (comparePos(a, b) <= 0 ? a : b));
    place.push({ block: block!, pos: least });
    if (frames.some((frame) => !samePos(frame!.pos, least))) return place;
  }
}

/** A jump out of the code the model runs. */
class Jumped {
  constructor(readonly kind: "break" | "continue" | "return") {}
}

/** Thrown when a loop of the model runs on, which the generator never means to make. */
export class Runaway extends Error {}

/**
 * What `program` does under the rule of the canon, for each uniform value in
 * `uniforms`: an inline `Fn` runs once each time the innermost block that holds
 * every read of its value runs, at the start of the first statement there that
 * reads it. The roots are the statements of one block, the program. A loop
 * condition and update are blocks that run on every iteration. An `Fn` read
 * more than once has its statements made statements of that block; one read
 * once runs where it is read. A `Break()` or `Continue()` acts on the loop
 * around the place it runs, and is refused when there is none, or when a read
 * sits in a loop inside that one.
 */
export function reference(program: Program, uniforms: number[][]): Outcome[] {
  const { fns } = program;
  const PROGRAM: Block = { repeats: false };
  const fnBlocks = fns.map((): Block => ({ repeats: false }));
  /** The index in an `Fn`'s block of its first statement after the count, of its jump, and of its values. */
  const bodyStart = 1;
  const jumpIndex = (i: number) => fns[i]!.body.length + 1;
  const valueIndex = (i: number) => fns[i]!.body.length + 2;
  const blocksOf = new Map<any, Record<string, Block>>();
  const blocks = (s: any): Record<string, Block> => {
    let known = blocksOf.get(s);
    if (!known) {
      known = {
        then: { repeats: false },
        else: { repeats: false },
        init: { repeats: false },
        cond: { repeats: true },
        update: { repeats: true },
        body: { repeats: s.k !== "if" },
      };
      blocksOf.set(s, known);
    }
    return known;
  };
  const rootBlocks = program.roots.map((): Block => ({ repeats: false }));

  // Every read, with its path from the program or from the body of the Fn that holds it.
  const reads: Read[] = [];
  /** The reads of the value expressions, by `Fn` and value, which count only when that value is read. */
  const valueReads: Read[][][] = fns.map((fn) => fn.extra.map(() => []));
  const inExpr = (e: Expr | undefined, within: number, path: Frame[], into: Read[], times = 1): void => {
    if (!e) return;
    if (e.k === "read") into.push({ fn: e.fn, value: e.value, within, path, times });
    else if (e.k === "add") (inExpr(e.a, within, path, into, times), inExpr(e.b, within, path, into, times));
    else if (e.k === "double") inExpr(e.e, within, path, into, times * 2 ** e.times);
  };
  const inBlock = (list: Stmt[], block: Block, within: number, path: Frame[], offset = 0) =>
    list.forEach((s, k) => inStmt(s, within, [...path, { block, pos: [k + offset] }]));
  const inStmt = (s: Stmt, within: number, here: Frame[]): void => {
    if (s.k === "out") return inExpr(s.e, within, here, reads);
    const b = blocks(s);
    if (s.k === "if") {
      inExpr(s.cond.e, within, here, reads);
      inBlock(s.then, b.then!, within, here);
      if (s.else) inBlock(s.else, b.else!, within, here);
    } else if (s.k === "loop") inBlock(s.body, b.body!, within, here);
    else if (s.k === "for") {
      if (s.init) inStmt(s.init, within, [...here, { block: b.init!, pos: [0] }]);
      if (s.cond) inStmt(s.cond, within, [...here, { block: b.cond!, pos: [0] }]);
      if (s.update) inStmt(s.update, within, [...here, { block: b.update!, pos: [0] }]);
      inBlock(s.body, b.body!, within, here);
    } else {
      inExpr(s.cond, within, [...here, { block: b.cond!, pos: [0] }], reads);
      if (s.condOut) inStmt(s.condOut, within, [...here, { block: b.cond!, pos: [0] }]);
      inBlock(s.body, b.body!, within, here, 1);
    }
  };
  program.roots.forEach((root, r) => inBlock(root, rootBlocks[r]!, -1, [{ block: PROGRAM, pos: [r] }]));
  fns.forEach((fn, i) => {
    inBlock(fn.body, fnBlocks[i]!, i, [], bodyStart);
    fn.extra.forEach((e, v) => inExpr(e, i, [{ block: fnBlocks[i]!, pos: [valueIndex(i)] }], valueReads[i]![v]!));
  });

  // The reads that count: of the program, and of the body of an Fn that runs, or of a value of one that is read.
  const counted: Read[][] = fns.map(() => []);
  const valuesRead = fns.map(() => new Set<number>());
  for (const read of reads) if (read.within === -1) counted[read.fn]!.push(read);
  for (let i = fns.length - 1; i >= 0; i--) {
    for (const read of counted[i]!) valuesRead[i]!.add(read.value);
    if (counted[i]!.length === 0) continue;
    for (const read of reads) if (read.within === i) counted[read.fn]!.push(read);
    for (const v of valuesRead[i]!) for (const read of valueReads[i]![v]!) counted[read.fn]!.push(read);
  }
  const runs = counted.map((list) => list.length > 0);
  const weight = counted.map((list) => list.reduce((sum, read) => sum + read.times, 0));
  /** Whether an `Fn` is read more than once, so its statements become statements of the block where it runs. */
  const spliced = weight.map((w) => w >= 2);

  // The order the program first reads each Fn in, which orders Fns placed at one place.
  const order: number[] = fns.map(() => Infinity);
  let met = 0;
  const firstReads = (within: number, list: Read[]) => {
    for (const read of list) {
      if (order[read.fn] !== Infinity) continue;
      order[read.fn] = met++;
      firstReads(
        read.fn,
        reads.filter((r) => r.within === read.fn),
      );
      for (const v of valuesRead[read.fn]!) firstReads(read.fn, valueReads[read.fn]![v]!);
    }
  };
  firstReads(
    -1,
    reads.filter((r) => r.within === -1),
  );

  // Where each Fn runs, from the outermost in: an Fn is read only in the program and in later Fns.
  const location: (Frame[] | undefined)[] = fns.map(() => undefined);
  /** The frames from the program to a read. */
  const fullPath = (read: Read): Frame[] => {
    if (read.within === -1) return read.path;
    const j = read.within;
    const [first, ...rest] = read.path;
    const at = location[j]!;
    if (!spliced[j]) return [...at, first!, ...rest];
    const last = at[at.length - 1]!;
    return [...at.slice(0, -1), { block: last.block, pos: [...last.pos, -1, order[j]!, ...first!.pos] }, ...rest];
  };
  for (let i = fns.length - 1; i >= 0; i--) {
    if (runs[i]) location[i] = placeBefore(counted[i]!.map(fullPath));
  }

  // The refusals: an Fn that jumps on a loop, with no loop around where it runs, or a read in a deeper one.
  const loops = (frames: Frame[]) => frames.filter((f) => f.block.repeats).length;
  const runsInside = (h: number, i: number) =>
    location[h]!.some((f) => f.block === fnBlocks[i] || f.pos.some((x, t) => x === -1 && f.pos[t + 1] === order[i]));
  const loopJump = (i: number, seen = new Set<number>()): Jump["kind"][] => {
    if (seen.has(i)) return [];
    seen.add(i);
    const own = fns[i]!.jump && fns[i]!.jump!.kind !== "return" ? [fns[i]!.jump!.kind] : [];
    const inner = counted
      .flatMap((list) => list)
      .filter((r) => r.within === i && loops(r.path) === 0 && runsInside(r.fn, i))
      .flatMap((r) => loopJump(r.fn, seen));
    return [...new Set([...own, ...inner])];
  };
  const refused = new Set<string>();
  fns.forEach((_, i) => {
    if (!runs[i]) return;
    const around = loops(location[i]!);
    for (const kind of loopJump(i) as ("break" | "continue")[]) {
      if (around === 0) refused.add(INLINE_FN_JUMP_WITH_NO_LOOP_MESSAGE[kind]);
      else if (counted[i]!.some((read) => loops(fullPath(read)) > around)) {
        refused.add(INLINE_FN_JUMP_IN_A_DEEPER_LOOP_MESSAGE[kind]);
      }
    }
  });
  if (refused.size > 0) return uniforms.map(() => ({ refused: [...refused] }));

  // The Fns whose statements run at each place of each block.
  const placedIn = new Map<Block, { fn: number; pos: number[] }[]>();
  fns.forEach((_, i) => {
    if (!runs[i] || !spliced[i]) return;
    const at = location[i]!;
    const last = at[at.length - 1]!;
    let list = placedIn.get(last.block);
    if (!list) placedIn.set(last.block, (list = []));
    list.push({ fn: i, pos: last.pos });
  });

  return uniforms.map((u) => {
    const counters = fns.map(() => 0);
    const outs = Array.from({ length: program.slots }, () => 0);
    const r = fns.map(() => 0);
    const values: number[][] = fns.map(() => []);
    let steps = 0;
    const step = () => {
      if (++steps > 100_000) throw new Runaway(`program ${program.seed} runs on`);
    };

    /** Runs the entries of `block`: its items, each with the statements of the Fns placed before it. */
    const runBlock = (block: Block, items: (() => void)[]) => {
      const entries: { pos: number[]; run: () => void }[] = items.map((run, k) => ({ pos: [k], run }));
      const add = (fn: number, pos: number[]) => {
        const base = [...pos, -1, order[fn]!];
        fnItems(fn).forEach((run, k) => entries.push({ pos: [...base, k], run }));
      };
      for (const { fn, pos } of placedIn.get(block) ?? []) add(fn, pos);
      entries.sort((a, b) => comparePos(a.pos, b.pos));
      for (const entry of entries) entry.run();
    };
    /** The items of the body of `Fn` `fn`: its count, its statements, its jump and its values. */
    const fnItems = (fn: number): (() => void)[] => {
      const def = fns[fn]!;
      return [
        () => {
          counters[fn]!++;
          r[fn] = counters[fn]!;
        },
        ...def.body.map((s) => () => stmt(s)),
        () => {
          if (def.jump && r[fn] === def.jump.at) throw new Jumped(def.jump.kind);
        },
        () => {
          for (const v of valuesRead[fn]!) {
            const base = v === 0 ? r[fn]! : r[fn]! * 10;
            values[fn]![v] = base + (def.extra[v] ? evaluate(def.extra[v]!) : 0);
          }
        },
      ];
    };
    const evaluate = (e: Expr): number => {
      if (e.k === "lit") return e.v;
      if (e.k === "add") return evaluate(e.a) + evaluate(e.b);
      if (e.k === "double") return evaluate(e.e) * 2 ** e.times;
      if (!spliced[e.fn]) runBlock(fnBlocks[e.fn]!, fnItems(e.fn));
      return values[e.fn]![e.value]!;
    };
    const list = (body: Stmt[], offset: (() => void)[] = []) => [...offset, ...body.map((s) => () => stmt(s))];
    /** Runs one iteration's block of a loop, telling whether a `Break()` ended the loop. */
    const iteration = (run: () => void, onContinue: () => boolean = () => false): boolean => {
      try {
        run();
        return false;
      } catch (jump) {
        if (!(jump instanceof Jumped) || jump.kind === "return") throw jump;
        if (jump.kind === "break") return true;
        return onContinue();
      }
    };
    const stmt = (s: Stmt): void => {
      step();
      if (s.k === "out") {
        outs[s.slot]! += evaluate(s.e);
        return;
      }
      const b = blocks(s);
      if (s.k === "if") {
        const test = u[s.cond.u]! + (s.cond.e ? evaluate(s.cond.e) * 0 : 0) > 0.5;
        if (test) runBlock(b.then!, list(s.then));
        else if (s.else) runBlock(b.else!, list(s.else));
      } else if (s.k === "loop") {
        for (let t = 0; t < s.n; t++) {
          step();
          if (iteration(() => runBlock(b.body!, list(s.body)))) break;
        }
      } else if (s.k === "for") {
        let n = 0;
        runBlock(b.init!, s.init ? [() => stmt(s.init!)] : [() => {}]);
        const update = () => runBlock(b.update!, [...(s.update ? [() => stmt(s.update!)] : []), () => n++]);
        for (;;) {
          step();
          let held = false;
          let continued = false;
          const ended = iteration(
            () =>
              runBlock(b.cond!, [
                () => {
                  if (s.cond) stmt(s.cond);
                  held = n < s.n;
                },
              ]),
            () => ((continued = true), false),
          );
          if (ended) break;
          // A Continue() in the condition runs the update, then tests again.
          if (continued) {
            update();
            continue;
          }
          if (!held) break;
          if (iteration(() => runBlock(b.body!, list(s.body)))) break;
          update();
        }
      } else {
        let v = 0;
        for (;;) {
          step();
          let held = false;
          let continued = false;
          const ended = iteration(
            () =>
              runBlock(b.cond!, [
                () => {
                  if (s.cond) evaluate(s.cond);
                  if (s.condOut) stmt(s.condOut);
                  held = v < s.n;
                },
              ]),
            () => ((continued = true), false),
          );
          if (ended) break;
          if (continued) continue;
          if (!held) break;
          if (iteration(() => runBlock(b.body!, list(s.body, [() => v++])))) break;
        }
      }
    };
    try {
      runBlock(
        PROGRAM,
        program.roots.map((root, k) => () => runBlock(rootBlocks[k]!, list(root))),
      );
    } catch (jump) {
      if (!(jump instanceof Jumped) || jump.kind !== "return") throw jump;
    }
    return { counters, outs };
  });
}

/** `program` as the rmsl source it builds, for a report that a reader can run again. */
export function programSource(program: Program): string {
  const lines: string[] = [];
  const value = (fn: number, v: number) => (program.fns[fn]!.values === 2 ? `f${fn}[${v}]` : `f${fn}`);
  const expr = (e: Expr): string => {
    if (e.k === "lit") return `float(${e.v})`;
    if (e.k === "read") return value(e.fn, e.value);
    if (e.k === "add") return `${expr(e.a)}.add(${expr(e.b)})`;
    return `double(${expr(e.e)}, ${e.times})`;
  };
  const write = (indent: string, s: Stmt): void => {
    if (s.k === "out") return void lines.push(`${indent}out[${s.slot}] += ${expr(s.e)};`);
    if (s.k === "if") {
      const test = s.cond.e ? `u${s.cond.u}.add(${expr(s.cond.e)}.mul(0))` : `u${s.cond.u}`;
      lines.push(`${indent}If(${test}.greaterThan(0.5), () => {`);
      s.then.forEach((t) => write(indent + "  ", t));
      if (s.else) {
        lines.push(`${indent}}).Else(() => {`);
        s.else.forEach((t) => write(indent + "  ", t));
      }
      return void lines.push(`${indent}});`);
    }
    if (s.k === "loop") {
      lines.push(`${indent}Loop(${s.n}, () => {`);
      s.body.forEach((t) => write(indent + "  ", t));
      return void lines.push(`${indent}});`);
    }
    if (s.k === "for") {
      const part = (p: Stmt | undefined, rest: string) => {
        if (!p) return rest;
        const inner: string[] = [];
        const saved = lines.length;
        write("", p);
        inner.push(...lines.splice(saved));
        return `{ ${inner.join(" ")} ${rest} }`;
      };
      lines.push(
        `${indent}For(() => ${part(s.init, "return float(0).toVar();")}, (n) => ${part(s.cond, `return n.lessThan(${s.n});`)}, ` +
          `(n) => ${part(s.update, "n.addAssign(1);")}, () => {`,
      );
      s.body.forEach((t) => write(indent + "  ", t));
      return void lines.push(`${indent}});`);
    }
    const test = s.cond ? `v.lessThan(float(${s.n}).add(${expr(s.cond)}.mul(0)))` : `v.lessThan(${s.n})`;
    lines.push(`${indent}const v = float(0).toVar();`);
    if (s.form === "function") {
      const saved = lines.length;
      if (s.condOut) write("", s.condOut);
      const inner = lines.splice(saved).join(" ");
      lines.push(`${indent}While(() => { ${inner} return ${test}; }, () => {`);
    } else lines.push(`${indent}${s.form === "node" ? "While" : "Loop"}(${test}, () => {`);
    lines.push(`${indent}  v.addAssign(1);`);
    s.body.forEach((t) => write(indent + "  ", t));
    lines.push(`${indent}});`);
  };
  lines.push(`// seed ${program.seed}; double(x, t) is x = x.add(x) t times; out and counter are storage.`);
  program.fns.forEach((fn, i) => {
    lines.push(`const f${i} = Fn(() => {`);
    lines.push(`  counter[${i}] += 1;`);
    lines.push(`  const r = counter[${i}].toVar();`);
    fn.body.forEach((s) => write("  ", s));
    if (fn.jump)
      lines.push(`  If(r.equal(${fn.jump.at}), () => ${fn.jump.kind[0]!.toUpperCase()}${fn.jump.kind.slice(1)}());`);
    const plus = (base: string, extra: Expr | undefined) => (extra ? `${base}.add(${expr(extra)})` : base);
    const first = plus("r", fn.extra[0]);
    lines.push(fn.values === 2 ? `  return [${first}, ${plus("r.mul(10)", fn.extra[1])}];` : `  return ${first};`);
    lines.push(`})();`);
  });
  lines.push(`return [`);
  program.roots.forEach((root) => {
    lines.push(`  Fn(() => {`);
    root.forEach((s) => write("    ", s));
    lines.push(`  })(),`);
  });
  lines.push(`];`);
  return lines.join("\n");
}

/** The uniform values each program runs with: both branches of each `If`. */
export const UNIFORM_CASES = [
  [0, 0],
  [1, 0],
  [0, 1],
  [1, 1],
];

/** A compiler of a routine for a CPU target. */
type CompileRoutine = (build: () => any, options: { name: string; params: [] }) => (ctx: any) => unknown;

/**
 * What `program` does on a CPU target, for each of `uniforms`: the message it
 * is refused with, or the counts and the totals it writes.
 */
export function runCpu(program: Program, compile: CompileRoutine, uniforms: number[][]): Outcome[] {
  const built = build(program);
  let routine: (ctx: any) => unknown;
  try {
    routine = compile(built.build, { name: "main", params: [] });
  } catch (error) {
    return uniforms.map(() => ({ refused: [String((error as Error).message)] }));
  }
  return uniforms.map((u) => {
    const counters = new Float64Array(Math.max(1, program.fns.length));
    const outs = new Float64Array(Math.max(1, program.slots));
    routine({
      storages: { [built.counter.name]: counters, [built.out.name]: outs },
      uniforms: Object.fromEntries(built.uniforms.map((uniform, i) => [uniform.name, u[i]])),
    });
    return {
      counters: Array.from(counters).slice(0, program.fns.length),
      outs: Array.from(outs).slice(0, program.slots),
    };
  });
}

/** What `program` does on WGSL compute, one invocation for each of `uniforms`. */
export async function runWgslCompute(program: Program, uniforms: number[][]): Promise<Outcome[]> {
  const built = build(program);
  // A workgroup of one, so a dispatch of one runs one invocation.
  const adapter = createWgslCompute(built.build(), { workgroupSize: 1 });
  await adapter.attach();
  try {
    const outcomes: Outcome[] = [];
    for (const u of uniforms) {
      built.uniforms.forEach((uniform, i) => adapter.setUniform(uniform, u[i]!));
      const storages = {
        [built.counter.name]: new Float32Array(Math.max(1, program.fns.length)),
        [built.out.name]: new Float32Array(Math.max(1, program.slots)),
      };
      // A buffer the program never reads or writes has no slot, and stays zero. Each run starts from zeros.
      const used = Object.fromEntries(Object.entries(storages).filter(([slot]) => adapter.buffer(slot)));
      for (const [slot, data] of Object.entries(used)) adapter.setAttribute(slot, data);
      await adapter.compute(used, 1);
      outcomes.push({
        counters: Array.from(storages[built.counter.name]!).slice(0, program.fns.length),
        outs: Array.from(storages[built.out.name]!).slice(0, program.slots),
      });
    }
    return outcomes;
  } finally {
    adapter.destroy();
  }
}

/**
 * Whether `actual` is what `expected` says: refused with one of its messages,
 * or with equal counts and totals.
 */
export function agrees(expected: Outcome, actual: Outcome): boolean {
  if ("refused" in expected) return "refused" in actual && expected.refused.includes(actual.refused[0]!);
  if ("refused" in actual) return false;
  const same = (a: number[], b: number[]) => a.length === b.length && a.every((x, i) => Object.is(x, b[i]));
  return same(expected.counters, actual.counters) && same(expected.outs, actual.outs);
}
