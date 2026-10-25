import { ProgramError, type Origin } from "../errors";
import { LOOP_JUMP_STATEMENTS, LOOP_STATEMENTS, node, var_ } from "../core";

/** The prefix of the variable a shared node gets, under the `_rmsl_` prefix the compiler keeps for its own names. */
export const SHARED_NAME_PREFIX = "_rmsl_gen_";

/** Node types that are statements, whose parameters are blocks or expressions the pass places by kind. */
const STATEMENTS = new Set([
  "seq",
  "let",
  "assign",
  "if",
  "for",
  "while",
  "scope",
  "break",
  "continue",
  "return",
  "discard",
]);

/**
 * Node types read where they are, however often. A swizzle or an element reads
 * a node that is already a name, and costs nothing to read again. An element
 * of a storage buffer is a load, or a fetch on GLSL, and is not among them.
 */
const READ_WHERE_IT_IS = new Set([
  "seq",
  "var",
  "swizzle",
  "vectorElement",
  "matrixElement",
  "member",
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

/** A statement that acts on the loop around it. */
type LoopJump = "break" | "continue";

/**
 * Why a `Break()` or a `Continue()` in an inline `Fn` is refused when no loop
 * is around the place the `Fn` runs, for each jump.
 */
export const INLINE_FN_JUMP_WITH_NO_LOOP_MESSAGE: Record<LoopJump, string> = {
  break: inlineFnJumpWithNoLoop("Break()"),
  continue: inlineFnJumpWithNoLoop("Continue()"),
};

/**
 * Why a `Break()` or a `Continue()` in an inline `Fn` is refused when a read
 * sits in a loop inside the one around the place the `Fn` runs, for each jump.
 */
export const INLINE_FN_JUMP_IN_A_DEEPER_LOOP_MESSAGE: Record<LoopJump, string> = {
  break: inlineFnJumpInADeeperLoop("Break()"),
  continue: inlineFnJumpInADeeperLoop("Continue()"),
};

/** The message of {@link INLINE_FN_JUMP_WITH_NO_LOOP_MESSAGE} for `jump`, as the user writes it. */
function inlineFnJumpWithNoLoop(jump: string) {
  return (
    `${jump} in an Fn called outside the program's body runs where no loop is around it: ` +
    `at the start of the first statement that reads its value, in the innermost block that holds every read. ` +
    `Read its value only inside the loop it acts on.`
  );
}

/** The message of {@link INLINE_FN_JUMP_IN_A_DEEPER_LOOP_MESSAGE} for `jump`, as the user writes it. */
function inlineFnJumpInADeeperLoop(jump: string) {
  return (
    `${jump} in an Fn called outside the program's body acts on the loop around the first ` +
    `statement that reads its value, but a read sits in a loop inside that one. ` +
    `Call the Fn inside the loop it acts on.`
  );
}

/** Node types that are literals, which a target folds. */
const LITERALS = new Set(["float", "int", "uint", "bool"]);

const isStatement = (n: any) => STATEMENTS.has(n.type) || (n.type === "call" && n._t === "void");

/** Whether `n` is the result of an inline `Fn` with statements of its own. */
export const isInlineFn = (n: any) => n?.type === "seq" && (n.params ?? []).some(isStatement);

/**
 * A read's place: a block, and the index there of the item that holds the
 * read. `pos`, when given, orders places of the block finer than `index`:
 * statement `k` of a result placed before the place `p` is `[...p, -1, order, k]`.
 */
type Frame = { block: any; index: number; pos?: number[] };

/** The order of `frame` among the places of its block. */
const posOf = (frame: Frame) => frame.pos ?? [frame.index];

/** Orders two places of one block. A place before an item sorts ahead of the item. */
function comparePos(a: number[], b: number[]): number {
  for (let i = 0; i < Math.min(a.length, b.length); i++) if (a[i] !== b[i]) return a[i]! - b[i]!;
  return b.length - a.length;
}

/**
 * The frames to the place that runs before every read at `paths`, each the
 * frames from one block in, all from the same: the nearest block that holds
 * every read, at the first place there that holds one, last.
 */
function placeAmong(paths: Frame[][]): Frame[] {
  const frames: Frame[] = [];
  for (let depth = 0; ; depth++) {
    const here = paths.map((path) => path[depth]);
    const block = here[0]?.block;
    if (here.some((frame) => !frame || frame.block !== block)) return frames;
    const least = here.reduce((a, b) => (comparePos(posOf(a!), posOf(b!)) <= 0 ? a : b))!;
    frames.push(least);
    if (here.some((frame) => comparePos(posOf(frame!), posOf(least)) !== 0)) return frames;
  }
}

/** The items of a block: a seq's own, or a loop or branch body that is one statement. */
const itemsOf = (block: any): any[] => (block.type === "seq" ? (block.params ?? []) : [block]);

/** `original` with `params` in place of its own, the graph the caller holds left as it is. */
const copyOf = (original: any, params: any[]) =>
  Object.assign(Object.create(Object.getPrototypeOf(original)), original, { params });

/**
 * The place that runs before every read at `paths`, each the frames from the
 * outermost block in: the nearest block that holds every read, at the first
 * item there that holds one. `undefined` when no block holds them all, as for
 * reads in two roots.
 */
function placeBefore(paths: Frame[][]): Frame | undefined {
  let place: Frame | undefined;
  for (let depth = 0; ; depth++) {
    const frames = paths.map((path) => path[depth]);
    const block = frames[0]?.block;
    if (frames.some((frame) => !frame || frame.block !== block)) return place;
    const index = frames.reduce((least, frame) => Math.min(least, frame!.index), Infinity);
    place = { block, index };
    if (frames.some((frame) => frame!.index !== index)) return place;
  }
}

/**
 * The key a loop or branch body that is one statement, not a block, stands
 * under as a block, so a statement can be placed before it.
 */
function bodyKeys() {
  const keys = new Map<any, any[]>();
  return (statement: any, slot: number) => {
    let own = keys.get(statement);
    if (!own) keys.set(statement, (own = []));
    return (own[slot] ??= { statement, slot });
  };
}

/** The slots of `s` walked as blocks of their own, its condition aside when it is a loop. */
const bodySlots = (s: any): number[] =>
  s.type === "if" ? [1, 2] : s.type === "for" ? [0, 2, 3] : s.type === "while" ? [1] : [];

/** The slot of `s` that holds its condition when it is a loop, tested before every iteration. */
const conditionSlot = (s: any): number | undefined => (s.type === "for" ? 1 : s.type === "while" ? 0 : undefined);

/**
 * A read of an inline `Fn` result: the result whose statements hold it, or
 * `undefined` for a block of the program, and the frames from that block in.
 */
type Read = { within: any; path: Frame[] };

/**
 * The frames from an anchor to a read, as a chain of segments that each point
 * at the one before, so a read inside a result read once shares the frames to
 * that result instead of copying them. `loops` counts the loop bodies among
 * every frame of the chain.
 */
type Chain = { parent: Chain | undefined; frames: Frame[]; loops: number };

/** A read with its frames from its anchor in: the nearest result around it read more than once, or the program. */
type Lifted = { anchor: any; chain: Chain };

/** The frames of `chain`, from its anchor in. */
function framesOf(chain: Chain): Frame[] {
  const segments: Frame[][] = [];
  for (let c: Chain | undefined = chain; c; c = c.parent) segments.push(c.frames);
  return segments.reverse().flat();
}

/**
 * Makes the statements of each inline `Fn` result that the program reads more
 * than once statements of the program, run once before every read: in the
 * nearest block that holds every read, before the first item there that holds
 * one. Each read takes the value the `Fn` returns, which the sharing pass then
 * treats as any other node. A result read once stays where it is read, so its
 * statements keep their place among those of the operation that reads it.
 * A call of an `Fn` that returns several values gives a result for each, all
 * holding the same statements, so a read of any of them is a read of the call,
 * whose statements run once.
 *
 * The roots are the statements of one block, the program, run in order. A
 * result that several roots read, and no block inside one of them holds every
 * read of, runs before the first root that reads it.
 *
 * A result is a block of its own, so one read only inside another runs in
 * the nearest block there that holds every read, and never outside it. A
 * result whose reads sit both inside another result and outside it is placed
 * from the program, through the place that other result runs: a statement of
 * a placed result sits there, before the item it was placed before, so the
 * result can run before one of those statements.
 *
 * A `Break()` or `Continue()` that a result holds outside its own loops, in any
 * of the values of its call, acts on the loop around the place it runs. The init
 * of a `For` runs before that loop, so a jump there is outside it. Such a result
 * is refused when no loop is around that place, or when a read sits in a loop
 * inside the one around it.
 *
 * The statements of each result are walked once, however often it is read,
 * and an expression is walked once at each place, however many paths reach it
 * there. A read inside a result is placed from the frames inside it, and a
 * read inside a result read once shares the frames to that result. So nested
 * results cost work in proportion to the levels, whether each reads the next
 * once or twice.
 *
 * Returns each of `blocks`, rewritten where a result was placed.
 */
function placeInlineFns(origin: Origin, blocks: any[]): any[] {
  /** The block that holds the roots, each an item of it. */
  const program = {};
  const holdsMemo = new Map<any, boolean>();
  /** Whether `n` is or holds an inline `Fn` result, statements and blocks included. */
  const holds = (n: any): boolean => {
    if (!n || typeof n !== "object" || !Array.isArray(n.params)) return false;
    const known = holdsMemo.get(n);
    if (known !== undefined) return known;
    holdsMemo.set(n, false);
    const result = isInlineFn(n) || n.params.some(holds);
    holdsMemo.set(n, result);
    return result;
  };
  const loopJumpMemo = new Map<any, LoopJump | undefined>();
  /** The type of a `Break()` or `Continue()` that `n` holds outside every loop it holds, a `For` init included. */
  const loopJump = (n: any): LoopJump | undefined => {
    if (!n || typeof n !== "object") return undefined;
    if (LOOP_JUMP_STATEMENTS.has(n.type)) return n.type;
    if (n.type === "for") return loopJump(n.params?.[0]);
    if (LOOP_STATEMENTS.has(n.type) || !Array.isArray(n.params)) return undefined;
    if (loopJumpMemo.has(n)) return loopJumpMemo.get(n);
    loopJumpMemo.set(n, undefined);
    let result: LoopJump | undefined;
    for (const p of n.params) if ((result ??= loopJump(p))) break;
    loopJumpMemo.set(n, result);
    return result;
  };

  /** The call each result the program reads stands for. */
  const callOf = new Map<any, any>();
  /** The call whose results start with each statement. */
  const callsByFirst = new Map<any, any>();
  /** The results of each call that the program reads. */
  const resultsOf = new Map<any, any[]>();
  /**
   * The result that stands for the call that made result `n`. A call of an
   * `Fn` that returns several values gives a result for each, all holding the
   * same statements, so a read of any of them is a read of that call.
   */
  const callFor = (n: any): any => {
    const known = callOf.get(n);
    if (known) return known;
    let call = callsByFirst.get(n.params[0]);
    if (!call) callsByFirst.set(n.params[0], (call = n));
    callOf.set(n, call);
    let results = resultsOf.get(call);
    if (!results) resultsOf.set(call, (results = []));
    results.push(n);
    return call;
  };
  /** The jump of every value of `call`, its statements included. */
  const callJump = (call: any): LoopJump | undefined => {
    for (const result of resultsOf.get(call) ?? [call]) {
      const jump = loopJump(result);
      if (jump) return jump;
    }
    return undefined;
  };
  const reads = new Map<any, Read[]>();
  const otherValuesWalked = new Set<any>();
  const walked = new Set<any>();
  /** The blocks that run on every iteration of a loop, a loop condition among them. */
  const loopBodies = new Set<any>();
  const key = bodyKeys();
  /** The calls each expression holds, by the path it was walked at, for a later path to the same place. */
  const scanned = new Map<Frame[], Map<any, any[]>>();

  /**
   * Records each result `n` holds as read at `path` inside `within`, and
   * returns their calls. A result is a block: the results its statements
   * read are read inside it, at frames from its own block. An expression
   * that several paths reach at one place is walked once, and each later
   * path reads its calls again.
   */
  const scan = (n: any, within: any, path: Frame[]): any[] => {
    if (!holds(n)) return [];
    let here = scanned.get(path);
    if (!here) scanned.set(path, (here = new Map()));
    const seen = here.get(n);
    if (seen) {
      for (const call of seen) reads.get(call)!.push({ within, path });
      return seen;
    }
    let calls: any[];
    if (!isInlineFn(n)) calls = [...new Set(n.params.flatMap((p: any) => scan(p, within, path)))];
    else {
      const call = callFor(n);
      let known = reads.get(call);
      if (!known) {
        reads.set(call, (known = []));
        walkBlock(n, n, [], n);
      }
      if (call !== n && !otherValuesWalked.has(n)) {
        otherValuesWalked.add(n);
        // The value of another result of the call, read as the last item of the call's block.
        const last = n.params.length - 1;
        scan(n.params[last], call, [{ block: call, index: last }]);
      }
      known.push({ within, path });
      calls = [call];
    }
    here.set(n, calls);
    return calls;
  };
  const walkBlock = (block: any, frameBlock: any, path: Frame[], within: any) =>
    itemsOf(block).forEach((item, index) => {
      const here = [...path, { block: frameBlock, index }];
      if (isStatement(item)) walkStatement(item, here, within);
      else scan(item, within, here);
    });
  /** Walks slot `slot` of `statement` as a block, one that runs on every iteration when `repeats`. */
  const walkBody = (statement: any, slot: number, path: Frame[], within: any, repeats = false) => {
    const body = statement.params[slot];
    if (!body) return;
    const frameBlock = body.type === "seq" ? body : key(statement, slot);
    if (repeats) loopBodies.add(frameBlock);
    walkBlock(body, frameBlock, path, within);
  };
  /**
   * Walks slot `slot` of `statement`, a loop condition, as a block of one item
   * that runs before every test, read where it is. So a condition that is a
   * result counts as a read of it.
   */
  const walkCondition = (statement: any, slot: number, here: Frame[], within: any) => {
    const condition = key(statement, slot);
    loopBodies.add(condition);
    scan(statement.params[slot], within, [...here, { block: condition, index: 0 }]);
  };
  /** Walks statement `s` at `here`; one that several roots hold, outside every result, once. */
  const walkStatement = (s: any, here: Frame[], within: any) => {
    if (within === undefined) {
      if (walked.has(s)) return;
      walked.add(s);
    }
    const condition = conditionSlot(s);
    if (condition !== undefined) {
      // A For's init comes before the condition and runs once; the update and the body run on every iteration.
      for (const slot of bodySlots(s)) if (slot < condition) walkBody(s, slot, here, within);
      walkCondition(s, condition, here, within);
      for (const slot of bodySlots(s)) if (slot > condition) walkBody(s, slot, here, within, true);
      return;
    }
    if (s.type === "seq") return walkBlock(s, s, here, within);
    if (s.type === "if") scan(s.params[0], within, here);
    else for (const p of s.params ?? []) scan(p, within, here);
    for (const slot of bodySlots(s)) walkBody(s, slot, here, within);
  };
  blocks.forEach((block, index) => {
    if (block?.type === "seq") walkBlock(block, block, [{ block: program, index }], undefined);
  });

  /** `frames` after `parent`, the loop bodies among them counted. */
  const extend = (parent: Chain | undefined, frames: Frame[]): Chain => ({
    parent,
    frames,
    loops: (parent?.loops ?? 0) + frames.filter((frame) => loopBodies.has(frame.block)).length,
  });
  /**
   * Whether `read` sits in the value of a call the program reads more than
   * once. That value is read at each read of the call, so a result read there
   * once runs with the statements of the call, once, as one read more than once does.
   */
  const inValueOfPlacedCall = (read: Read) =>
    read.within !== undefined &&
    reads.get(read.within)!.length >= 2 &&
    read.path[0]!.index === read.within.params.length - 1;
  const anchors = new Map<any, { anchor: any; chain: Chain | undefined }>();
  /**
   * Where the statements of result `fn` sit: the nearest result around them
   * that is placed, or `undefined` for the program, and the frames from there
   * to `fn`'s own block. A result read once runs where it is read.
   */
  const anchorOf = (fn: any): { anchor: any; chain: Chain | undefined } => {
    let known = anchors.get(fn);
    if (known) return known;
    const only = reads.get(fn)!;
    if (only.length !== 1 || inValueOfPlacedCall(only[0]!)) known = { anchor: fn, chain: undefined };
    else {
      const outer = only[0]!.within === undefined ? { anchor: undefined, chain: undefined } : anchorOf(only[0]!.within);
      known = { anchor: outer.anchor, chain: extend(outer.chain, only[0]!.path) };
    }
    anchors.set(fn, known);
    return known;
  };
  /** A read with its frames from its anchor in. */
  const lift = (read: Read): Lifted => {
    if (read.within === undefined) return { anchor: undefined, chain: extend(undefined, read.path) };
    const { anchor, chain } = anchorOf(read.within);
    return { anchor, chain: extend(chain, read.path) };
  };

  /** The calls in the order the walk first met them, which orders the results placed at one place. */
  const met = [...reads.keys()];
  const order = new Map(met.map((call, i) => [call, i]));
  /** Each placed result's anchor, and the frames from there to its place, the place last. */
  const sites = new Map<any, { anchor: any; frames: Frame[] }>();
  const fullSites = new Map<any, Frame[]>();
  /**
   * `frames`, from inside result `anchor` in, as frames from the program in.
   * A frame of the anchor's own block becomes a place in the block it was
   * placed in: statement `k` of it there sits at `[...place, -1, order, k]`.
   */
  const fromProgram = (anchor: any, frames: Frame[]): Frame[] => {
    if (anchor === undefined) return frames;
    const site = fullSiteOf(anchor);
    const last = site[site.length - 1]!;
    const [first, ...rest] = frames;
    return [
      ...site.slice(0, -1),
      { block: last.block, index: last.index, pos: [...posOf(last), first!.index] },
      ...rest,
    ];
  };
  /** The frames from the program to the block placed result `fn` runs in, its statements' place last. */
  const fullSiteOf = (fn: any): Frame[] => {
    let known = fullSites.get(fn);
    if (known) return known;
    const { anchor, frames } = place(fn);
    const full = fromProgram(anchor, frames);
    const last = full[full.length - 1]!;
    known = [...full.slice(0, -1), { block: last.block, index: last.index, pos: [...posOf(last), -1, order.get(fn)!] }];
    fullSites.set(fn, known);
    return known;
  };
  /** The block and the item that `frame`, a place, runs before: a statement of a placed result's own, or an item. */
  const placeOf = (frame: Frame): Frame => {
    const pos = posOf(frame);
    if (pos.length > 1) return { block: met[pos[pos.length - 2]!], index: pos[pos.length - 1]! };
    return { block: frame.block, index: pos[0]! };
  };

  const placed = new Map<any, Frame>();
  /** Places result `fn`, read more than once, and returns its site. */
  const place = (fn: any): { anchor: any; frames: Frame[] } => {
    const known = sites.get(fn);
    if (known) return known;
    const own = reads.get(fn)!.map(lift);
    // Reads inside different results are compared from the program, through the places those results run.
    const anchor = own.every((read) => read.anchor === own[0]!.anchor) ? own[0]!.anchor : undefined;
    const paths = own.map((read) =>
      read.anchor === anchor ? framesOf(read.chain) : fromProgram(read.anchor, framesOf(read.chain)),
    );
    const site = { anchor, frames: placeAmong(paths) };
    sites.set(fn, site);
    return site;
  };
  for (const [fn, list] of reads) if (list.length >= 2 || inValueOfPlacedCall(list[0]!)) place(fn);
  for (const call of met) {
    const site = sites.get(call);
    if (site) placed.set(call, placeOf(site.frames[site.frames.length - 1]!));
  }
  /** How many loops hold the place at the end of `frames`. */
  const loopsOn = (frames: Frame[]) => frames.filter((frame) => loopBodies.has(frame.block)).length;
  /** The frames from the program to where result `fn` runs: its place, or its one read. */
  const runsAt = (fn: any): Frame[] => {
    const site = sites.get(fn);
    if (site) return fromProgram(site.anchor, site.frames);
    const read = lift(reads.get(fn)![0]!);
    return fromProgram(read.anchor, framesOf(read.chain));
  };
  /** How many loops of result `f`'s own hold the place at the end of `frames`, or `undefined` when it is not inside `f`. */
  const loopsInside = (frames: Frame[], f: any): number | undefined => {
    const own = order.get(f);
    const entry = frames.findIndex(
      (frame) => frame.block === f || posOf(frame).some((x, i, pos) => x === -1 && pos[i + 1] === own),
    );
    return entry < 0 ? undefined : loopsOn(frames.slice(entry + 1));
  };
  /** The `Break()` or `Continue()` that call `fn`'s values hold outside their loops, results they read aside. */
  const ownJump = (fn: any): LoopJump | undefined => {
    const seen = new Set<any>();
    const walk = (n: any): LoopJump | undefined => {
      if (!n || typeof n !== "object" || seen.has(n)) return undefined;
      seen.add(n);
      if (LOOP_JUMP_STATEMENTS.has(n.type)) return n.type;
      if (n.type === "for") return walk(n.params?.[0]);
      if (LOOP_STATEMENTS.has(n.type) || !Array.isArray(n.params)) return undefined;
      const call = callOf.get(n);
      if (call !== undefined && call !== fn) return undefined;
      for (const p of n.params) {
        const jump = walk(p);
        if (jump) return jump;
      }
      return undefined;
    };
    for (const result of resultsOf.get(fn) ?? [fn]) {
      const jump = walk(result);
      if (jump) return jump;
    }
    return undefined;
  };
  // A jump acts on the loop around where it runs, so a result that runs among
  // the statements of another, outside that one's loops, jumps where it runs.
  const jumping = met.filter((fn) => callJump(fn));
  const jumps = new Map<any, LoopJump>();
  for (const fn of jumping) {
    const jump = ownJump(fn);
    if (jump) jumps.set(fn, jump);
  }
  for (let grew = true; grew;) {
    grew = false;
    for (const fn of jumping) {
      if (jumps.has(fn)) continue;
      const inner = jumping.find((h) => h !== fn && jumps.has(h) && loopsInside(runsAt(h), fn) === 0);
      if (inner === undefined) continue;
      jumps.set(fn, jumps.get(inner)!);
      grew = true;
    }
  }
  for (const [fn, jump] of jumps) {
    const loopsAround = loopsOn(runsAt(fn));
    if (loopsAround === 0)
      throw new ProgramError("jump-in-inline-fn-outside-a-loop", INLINE_FN_JUMP_WITH_NO_LOOP_MESSAGE[jump], origin);
    const paths = reads.get(fn)!.map((read) => {
      const { anchor, chain } = lift(read);
      return fromProgram(anchor, framesOf(chain));
    });
    if (paths.some((path) => loopsOn(path) > loopsAround)) {
      throw new ProgramError(
        "jump-in-inline-fn-in-a-deeper-loop",
        INLINE_FN_JUMP_IN_A_DEEPER_LOOP_MESSAGE[jump],
        origin,
      );
    }
  }
  if (placed.size === 0) return blocks;

  /** The results to run before each item of each block, in the order the program first reads them. */
  const before = new Map<any, Map<number, any[]>>();
  for (const [fn, { block, index }] of placed) {
    let at = before.get(block);
    if (!at) before.set(block, (at = new Map()));
    at.set(index, [...(at.get(index) ?? []), fn]);
  }
  const copies = new Map<any, any>();
  /**
   * The statements of the results placed before item `index` of
   * `frameBlock`, rewritten, with those of the results placed inside them.
   */
  const placedAt = (frameBlock: any, index: number): any[] =>
    (before.get(frameBlock)?.get(index) ?? []).flatMap((fn: any) => items(fn, fn).slice(0, -1));
  /** The items of `block`, rewritten, with the statements of each result placed before an item ahead of it. */
  const items = (block: any, frameBlock: any): any[] =>
    itemsOf(block).flatMap((item, index) => [...placedAt(frameBlock, index), rewrite(item)]);
  /**
   * A slot of `statement` walked as a block, rewritten: a block of its own,
   * ending in what it held, when statements are placed before what it holds.
   */
  const body = (statement: any, slot: number) => {
    const b = statement.params[slot];
    if (!b || b.type === "seq" || !before.has(key(statement, slot))) return rewrite(b);
    return node({ _t: b._t, type: "seq", params: items(b, key(statement, slot)) });
  };
  /** Condition `slot` of `statement`, a loop, rewritten, the statements of the results placed before it ahead. */
  const condition = (statement: any, slot: number) => {
    const tested = rewrite(statement.params[slot]);
    const ahead = placedAt(key(statement, slot), 0);
    if (ahead.length === 0) return tested;
    const own = tested.type === "seq" ? tested.params : [tested];
    return node({ _t: tested._t, type: "seq", params: [...ahead, ...own] });
  };
  const rewrite = (n: any): any => {
    if (!n || typeof n !== "object" || !Array.isArray(n.params)) return n;
    const known = copies.get(n);
    if (known !== undefined) return known;
    let result: any;
    // A block is keyed by itself, and a result by its call, whose block the walk framed it in.
    const frameBlock = callOf.get(n) ?? n;
    if (placed.has(frameBlock)) result = rewrite(n.params[n.params.length - 1]);
    else if (!holds(n) && !before.has(frameBlock)) result = n;
    else {
      const slots = bodySlots(n);
      const params =
        n.type === "seq" && before.has(frameBlock)
          ? items(n, frameBlock)
          : n.params.map((p: any, i: number) =>
              i === conditionSlot(n) ? condition(n, i) : slots.includes(i) ? body(n, i) : rewrite(p),
            );
      result =
        params.length === n.params.length && params.every((p: any, i: number) => p === n.params[i])
          ? n
          : copyOf(n, params);
    }
    copies.set(n, result);
    return result;
  };
  // A result that only the program as a whole holds every read of runs before the first root that reads it.
  return blocks.map((block, index) => {
    const ahead = placedAt(program, index);
    const rewritten = rewrite(block);
    return ahead.length === 0 ? rewritten : copyOf(rewritten, [...ahead, ...rewritten.params]);
  });
}

/** Whether `n` is a literal, or an expression of literals alone, which a target folds into one. */
function isConstant(n: any, memo: Map<any, boolean>): boolean {
  const known = memo.get(n);
  if (known !== undefined) return known;
  const result =
    n.type === "call"
      ? false
      : Array.isArray(n.params) && n.params.length > 0
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
 * own, so the program computes it once. The variable is made before every
 * read: in the nearest block that holds every read, before the first statement
 * there that holds one. A later read in that block or a block inside it reads
 * the variable.
 *
 * The program is the statements the roots hold. An inline `Fn` result read
 * more than once first has its statements made statements of the program, so
 * they run once, before every read. The pass then counts the reads of each
 * node over the statements, and where each read sits. It walks the statements
 * in the order they run, and inserts `let _rmsl_gen_N = <the node>` at the
 * place before every read. A read takes the variable.
 *
 * A variable is read only where the value it holds is still the value the node
 * computes: not outside the block that holds it, and not after a statement that
 * could have changed what the node reads. Such a read computes the node again,
 * into a variable made before its own statement. A loop runs its body again, so a
 * variable made before a loop is read inside it only when the node reads
 * nothing a statement can change. The init, condition and update of a loop
 * share no node, since the condition and the update run again on every
 * iteration. The statements of an inline `Fn` result placed in one of them
 * run there. A statement whose expression holds an inline `Fn` read once is left as
 * written, so its statements keep their place among those of the operation
 * that reads it.
 *
 * The statements and expressions the pass changes are copied, and the graph the
 * caller holds is left as it is. A root that is an expression gets a block of
 * its own.
 */
export function shareNodes<T>(origin: Origin, roots: T): T {
  const reads = new Map<any, number>();
  /** Where each node is read: for each read, the frames from the outermost block in. */
  const readAt = new Map<any, Frame[][]>();
  const expanded = new Set<any>();
  const statementsMemo = new Map<any, boolean>();
  const constantsMemo = new Map<any, boolean>();
  const key = bodyKeys();
  /** Whether `n` is a node that gets a variable when it is read more than once. */
  const isShareable = (n: any): boolean =>
    Array.isArray(n.params) &&
    (n.params.length > 0 || n.type === "call") &&
    n._t !== "void" &&
    !READ_WHERE_IT_IS.has(n.type) &&
    !isConstant(n, constantsMemo);

  const read = (expression: any, path: Frame[]) => {
    if (!expression || typeof expression !== "object") return;
    reads.set(expression, (reads.get(expression) ?? 0) + 1);
    let paths = readAt.get(expression);
    if (!paths) readAt.set(expression, (paths = []));
    paths.push(path);
    if (expanded.has(expression)) return;
    expanded.add(expression);
    if (Array.isArray(expression.params)) for (const p of expression.params) read(p, path);
  };

  /** Counts the reads of each item of a block, `frameBlock` the block it stands as. */
  const countBlock = (block: any, frameBlock: any, path: Frame[]) => {
    expanded.add(block);
    itemsOf(block).forEach((p, index) => {
      const here = [...path, { block: frameBlock, index }];
      if (isStatement(p)) count(p, here);
      else if (!holdsStatements(p, statementsMemo)) read(p, here);
    });
  };
  const countBody = (statement: any, slot: number, path: Frame[]) => {
    const body = statement.params[slot];
    if (body) countBlock(body, body.type === "seq" ? body : key(statement, slot), path);
  };

  /** Counts the reads of the expressions of a statement at `path`, and of the statements inside it. */
  const count = (statement: any, path: Frame[]): void => {
    if (!statement || typeof statement !== "object" || expanded.has(statement)) return;
    switch (statement.type) {
      case "seq":
        countBlock(statement, statement, path);
        return;
      case "let":
      case "assign":
        expanded.add(statement);
        // The value of a `let` is where its variable is made, so it is not a read of that node.
        if (!holdsStatements(statement.params[1], statementsMemo)) {
          if (statement.type === "assign") read(statement.params[1], path);
          else if (Array.isArray(statement.params[1].params)) for (const p of statement.params[1].params) read(p, path);
        }
        return;
      case "if":
        expanded.add(statement);
        if (!holdsStatements(statement.params[0], statementsMemo)) read(statement.params[0], path);
        countBody(statement, 1, path);
        countBody(statement, 2, path);
        return;
      case "for":
        expanded.add(statement);
        countBody(statement, 3, path);
        return;
      case "while":
        expanded.add(statement);
        countBody(statement, 1, path);
        return;
      case "scope":
        expanded.add(statement);
        countBody(statement, 0, path);
        return;
      case "call":
        expanded.add(statement);
        for (const p of statement.params ?? []) if (!holdsStatements(p, statementsMemo)) read(p, path);
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

  /** The shared nodes to make before each item of each block, in the order the program first reads them. */
  const placed = new Map<any, Map<number, any[]>>();

  /** Makes the variables of the shared nodes placed before item `index` of `frameBlock`, the `let`s added to `out`. */
  const makePlaced = (frameBlock: any, index: number, out: any[]) => {
    for (const n of placed.get(frameBlock)?.get(index) ?? []) share(n, out, new Map());
  };

  /** What each statement and block became, so one that several roots hold stays one node. */
  const rewritten = new Map<any, any>();

  /** A copy of the block with the `let`s its statements need, which holds the statements it rewrote. */
  const block = (seq: any): any => {
    const known = rewritten.get(seq);
    if (known !== undefined) return known;
    blocks.push(blockCount++);
    const out: any[] = [];
    (seq.params ?? []).forEach((p: any, index: number) => {
      makePlaced(seq, index, out);
      if (isStatement(p)) statement(p, out);
      else out.push(rewrite(p, out));
    });
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
        s.params.slice(1).forEach((branch: any, i: number) => params.push(branch ? enclosed(s, i + 1) : branch));
        if (!sameParams(params, s.params)) result = copyOf(s, params);
        epoch++;
        break;
      }
      case "for":
      case "while": {
        epoch++;
        const params = s.params.slice();
        params[params.length - 1] = enclosed(s, params.length - 1);
        if (!sameParams(params, s.params)) result = copyOf(s, params);
        epoch++;
        break;
      }
      case "scope": {
        const body = enclosed(s, 0);
        if (body !== s.params[0]) result = copyOf(s, [body]);
        epoch++;
        break;
      }
      case "call": {
        const params = s.params.map((p: any) => rewrite(p, out));
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

  /** Body `slot` of `parent`, a block or the statement standing for one, that runs under its own condition. */
  const enclosed = (parent: any, slot: number): any => {
    const body = parent.params[slot];
    if (body.type === "seq") return block(body);
    blocks.push(blockCount++);
    const out: any[] = [];
    makePlaced(key(parent, slot), 0, out);
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

  const blocksOf = (n: any): any[] =>
    Array.isArray(n) ? n.flatMap(blocksOf) : n.block?.type === "seq" ? [n.block] : [];
  const normalized = normalize(roots);
  const own = blocksOf(normalized);
  const lowered = new Map(placeInlineFns(origin, own).map((b, i) => [own[i], b]));
  /** A root with its block as the inline `Fn` results placed in it left it. */
  const withPlaced = (n: any): any => {
    if (Array.isArray(n)) return n.map(withPlaced);
    const changed = lowered.get(n.block);
    if (changed === undefined || changed === n.block) return n;
    if (n.root === n.block) return { root: changed, block: changed };
    return { root: changed.params.length === 1 ? changed.params[0] : n.root, block: changed };
  };
  const program = withPlaced(normalized);
  for (const b of blocksOf(program)) countBlock(b, b, []);
  for (const [n, count] of reads) {
    if (count < 2 || !isShareable(n)) continue;
    const place = placeBefore(readAt.get(n)!);
    if (!place) continue;
    let at = placed.get(place.block);
    if (!at) placed.set(place.block, (at = new Map()));
    at.set(place.index, [...(at.get(place.index) ?? []), n]);
  }

  /** What a caller gets for each root: the root, or its rewritten block when the pass changed it. */
  const result = (n: any): any => {
    if (Array.isArray(n)) return n.map(result);
    const { root, block: own } = n;
    if (own?.type !== "seq") return root;
    const changed = block(own);
    if (own === root) return changed;
    return changed.params.length === 1 ? root : changed;
  };
  return result(program) as T;
}
