import { describe, it, expect, afterAll } from "vitest";
import { int, uint, ivec2, uvec2, type Node } from "./rmsl";
import { GPU_EVALUATION_SKIPPED } from "./testing/shader-eval";
import { releaseGpu } from "./testing/gpu";
import {
  BINARY_OPS,
  COMPARISONS,
  INT_MAX,
  INT_MIN,
  UINT_MAX,
  binary,
  clamp,
  compare,
  convert,
  unary,
  unaryOps,
  wrap,
  type BinaryOp,
  type Comparison,
  type IntegerType,
  type UnaryOp,
} from "./testing/integer-reference";
import {
  describeMismatches,
  sweepGLSLValidity,
  sweepJS,
  sweepWASM,
  sweepWGSL,
  type Mismatch,
  type SweepCase,
} from "./testing/integer-sweep";

/**
 * Every integer operation, alone and composed with every other, on every
 * backend, against a reference model of WGSL's semantics.
 *
 * `integer-semantics.test.ts` checks a table of hand-picked cases, each an
 * operation on two operands that are both run-time values or both literals.
 * Bugs lived in what that table never built: an operation whose operand is
 * another operation (precedence), a literal operand that only becomes a
 * literal once folded (constant folding order), a vector made from one
 * scalar (splats), and the edge values of one operation fed into another.
 * This sweep builds those systematically instead of by hand:
 *
 * - every operation on every combination of operand kinds (run-time value,
 *   literal, constant expression, splat) over every pair of edge values;
 * - every operation with every other operation as its left or right operand,
 *   over every combination of operand kinds;
 * - each of those as scalars, as vectors, and as vectors with a scalar right
 *   operand;
 * - conversions between `int` and `uint`, and vectors built from the other
 *   signedness.
 *
 * JS, WASM and WGSL results are held to the reference exactly. GLSL leaves the
 * edge cases undefined, so its programs are only required to compile.
 */

afterAll(async () => {
  await releaseGpu();
}, 120_000);

/**
 * How much of the sweep runs, from `RMSL_INTEGER_SWEEP`:
 * - unset: every composition, in every shape, with every leaf kind, but a
 *   sample of which kinds go together — about a minute with the GPU layers;
 * - `full`: every combination of leaf kinds — several minutes;
 * - `skip`: none of it, for the fast and mutation runs, which it would
 *   otherwise slow from under a second to half a minute.
 */
const SWEEP = process.env.RMSL_INTEGER_SWEEP;
const FULL_SWEEP = SWEEP === "full";

if (SWEEP === "skip") {
  process.stderr.write(
    `\n[integer-sweep] SKIPPED — no integer operation was checked against the reference on any backend.\n`,
  );
}

// === Values ===

/** Values the run-time operands take: each operation's edge cases and a few ordinary values. */
const EDGE: Record<IntegerType, number[]> = {
  int: [
    0,
    1,
    -1,
    2,
    -2,
    3,
    7,
    -7,
    31,
    32,
    33,
    64,
    65535,
    65536,
    123456789,
    -987654321,
    INT_MAX,
    INT_MIN,
    INT_MIN + 1,
    INT_MAX - 1,
  ],
  uint: [
    0,
    1,
    2,
    3,
    7,
    31,
    32,
    33,
    64,
    65535,
    65536,
    123456789,
    3000000000,
    INT_MAX,
    0x80000000,
    0x80000001,
    UINT_MAX,
    UINT_MAX - 1,
  ],
};

/**
 * Values a literal operand takes. Each one is compiled into its own program
 * rather than passed at run time, so the set is smaller: the values that are
 * special to some operation, as divisor, shift amount or overflow boundary.
 */
const SPECIAL: Record<IntegerType, number[]> = {
  int: [0, 1, -1, 2, 7, -7, 32, 33, INT_MAX, INT_MIN],
  uint: [0, 1, 2, 7, 32, 33, INT_MAX, 0x80000000, UINT_MAX],
};

/** Values each run-time operand of a composition takes; three operands make this set cubed. */
const COMPOSITION_EDGE: Record<IntegerType, number[]> = {
  int: [0, 1, -1, 2, -7, 32, 33, INT_MAX, INT_MIN],
  uint: [0, 1, 2, 7, 32, 33, INT_MAX, 0x80000000, UINT_MAX],
};

/**
 * The second lane of a vector operand. It differs from the first, so a
 * backend that computes one lane and copies it, or reads the wrong lane,
 * gets a different answer.
 */
function secondLane(type: IntegerType, value: number, slot: number): number {
  const values = EDGE[type];
  const at = values.indexOf(value);
  return values[(Math.max(at, 0) * 7 + 3 + slot * 5) % values.length]!;
}

// === Expressions ===

type Leaf = { slot: number };
type Op =
  | { kind: "binary"; op: BinaryOp }
  | { kind: "compare"; op: Comparison }
  | { kind: "unary"; op: UnaryOp }
  | { kind: "clamp" };
type Expr = Leaf | { op: Op; args: Expr[] };

/**
 * How a leaf enters the program:
 * - `run`: a function parameter, so its value is only known at run time;
 * - `literal`: `int(v)`, or `ivec2(v, w)` for a vector;
 * - `folded`: a constant expression that folds to the value, so the compiler
 *   only sees a literal after folding it;
 * - `splat`: a vector made from one scalar literal, `ivec2(int(v))` or `ivec2(v)`.
 */
type LeafKind = "run" | "literal" | "folded" | "splat";

type Shape = "scalar" | "vector" | "vector by scalar";
const SHAPES: Shape[] = ["scalar", "vector", "vector by scalar"];

const isLeaf = (e: Expr): e is Leaf => "slot" in e;

function arity(op: Op): number {
  return op.kind === "unary" ? 1 : op.kind === "clamp" ? 3 : 2;
}

function opName(op: Op): string {
  return op.kind === "clamp" ? "clamp" : op.op;
}

function show(e: Expr): string {
  if (isLeaf(e)) return `x${e.slot}`;
  const [first, ...rest] = e.args.map(show);
  if (e.op.kind === "compare") return `${first}.${e.op.op}(${rest[0]}).select(${first}, ${rest[0]})`;
  return `${first}.${opName(e.op)}(${rest.join(", ")})`;
}

/**
 * Whether each leaf is a vector. In `vector by scalar`, every operand after
 * an operation's first is a scalar, so `ivec2.add(int)` and
 * `ivec2.clamp(int, int)` are exercised as well as the all-vector forms.
 */
function leafContexts(e: Expr, shape: Shape, vector = shape !== "scalar", out: boolean[] = []): boolean[] {
  if (isLeaf(e)) {
    out[e.slot] = vector;
    return out;
  }
  e.args.forEach((arg, i) => leafContexts(arg, shape, i === 0 ? vector : vector && shape === "vector", out));
  return out;
}

type Binding = { kind: LeafKind; lanes: [number, number] };

/** The reference result for one lane. A scalar leaf has the same value in both lanes. */
function evaluate(type: IntegerType, e: Expr, bindings: Binding[], lane: 0 | 1): number {
  if (isLeaf(e)) return bindings[e.slot]!.lanes[lane];
  const args = e.args.map((a) => evaluate(type, a, bindings, lane));
  switch (e.op.kind) {
    case "binary":
      return binary(type, e.op.op, args[0]!, args[1]!);
    case "compare":
      return compare(e.op.op, args[0]!, args[1]!) ? args[0]! : args[1]!;
    case "unary":
      return unary(type, e.op.op, args[0]!);
    case "clamp":
      return clamp(args[0]!, args[1]!, args[2]!);
  }
}

const scalarOf = (type: IntegerType): any => (type === "int" ? int : uint);
const vectorOf = (type: IntegerType): any => (type === "int" ? ivec2 : uvec2);

/** An offset for folded leaves: large enough that the parts overflow on their own for the edge values. */
const FOLD_OFFSET = 0x6b2f_1e37;

function buildLeaf(type: IntegerType, slot: number, vector: boolean, binding: Binding, param: () => Node<any>): any {
  const scalar = scalarOf(type);
  const vec = vectorOf(type);
  const [v, w] = binding.lanes;
  const k = FOLD_OFFSET;
  switch (binding.kind) {
    case "run":
      return vector ? vec(param(), param()) : param();
    case "literal":
      return vector ? vec(v, w) : scalar(v);
    case "folded":
      // Alternate between two folds, and for vectors between folding inside
      // the constructor and folding two vector literals.
      if (!vector) {
        return slot % 2 === 0
          ? scalar(wrap(type, v - k)).add(scalar(k))
          : scalar(wrap(type, v ^ k)).bitXor(scalar(wrap(type, k)));
      }
      return slot % 2 === 0
        ? vec(wrap(type, v - k), wrap(type, w - k)).add(vec(k, k))
        : vec(scalar(wrap(type, v ^ k)).bitXor(scalar(k)), scalar(wrap(type, w ^ k)).bitXor(scalar(k)));
    case "splat":
      return slot % 2 === 0 ? vec(scalar(v)) : vec(v);
  }
}

function buildExpr(
  type: IntegerType,
  e: Expr,
  shape: Shape,
  contexts: boolean[],
  bindings: Binding[],
  params: Node<any>[],
) {
  let next = 0;
  const leaves = contexts.map((vector, slot) => buildLeaf(type, slot, vector, bindings[slot]!, () => params[next++]!));
  const vec = vectorOf(type);
  const go = (e: Expr, vector: boolean): any => {
    if (isLeaf(e)) return leaves[e.slot];
    const args = e.args.map((a, i) => go(a, i === 0 ? vector : vector && shape === "vector"));
    const [first, ...rest] = args;
    switch (e.op.kind) {
      case "binary":
        return first[e.op.op](rest[0]);
      case "compare": {
        // A scalar `ifFalse` beside a vector `ifTrue` is splatted, which is
        // itself a vector built from an expression.
        const ifFalse = vector && shape === "vector by scalar" ? vec(rest[0]) : rest[0];
        return first[e.op.op](rest[0]).select(first, ifFalse);
      }
      case "unary":
        return first[e.op.op]();
      case "clamp":
        return first.clamp(rest[0], rest[1]);
    }
  };
  return go(e, shape !== "scalar");
}

function slotsOf(e: Expr, out = new Set<number>()): Set<number> {
  if (isLeaf(e)) out.add(e.slot);
  else e.args.forEach((a) => slotsOf(a, out));
  return out;
}

/** Every list that picks one entry from each list in `lists`. */
function product<T>(lists: T[][]): T[][] {
  return lists.reduce<T[][]>((acc, list) => acc.flatMap((prefix) => list.map((x) => [...prefix, x])), [[]]);
}

/** A small deterministic hash, to pick sample values without a random seed. */
function hash(...parts: number[]): number {
  let h = 2166136261;
  for (const p of parts) h = Math.imul(h ^ p, 16777619) >>> 0;
  return h;
}

type SweepOptions = {
  /** Values each run-time leaf takes; every combination of them is one run. */
  runValues: number[];
  /**
   * Values a literal leaf takes. `all` compiles a program for every
   * combination; `sample` compiles one per combination of leaf kinds, with
   * values picked from the set by hash.
   */
  literalValues: number[];
  literalCoverage: "all" | "sample";
  /**
   * Which combinations of leaf kinds get a program. `all` is every one;
   * `sample` is the combinations where every leaf has the same kind, plus
   * {@link SAMPLED_MIXED_KINDS} mixed ones picked by hash, so every kind
   * still reaches every position across the sweep.
   */
  kindCoverage: "all" | "sample";
};

/** Mixed combinations of leaf kinds each expression gets under `kindCoverage: "sample"`. */
const SAMPLED_MIXED_KINDS = 2;

/**
 * The cases for one expression: one program per combination of leaf kinds
 * (and, with `all`, per combination of literal values), each run over every
 * combination of run-time values.
 */
function casesFor(type: IntegerType, e: Expr, shape: Shape, options: SweepOptions, seed: number): SweepCase[] {
  const contexts = leafContexts(e, shape);
  const slots = [...slotsOf(e)].sort((a, b) => a - b);
  const width: 1 | 2 = shape === "scalar" ? 1 : 2;
  const kindChoices = slots.map((slot) =>
    contexts[slot]
      ? (["run", "literal", "folded", "splat"] as LeafKind[])
      : (["run", "literal", "folded"] as LeafKind[]),
  );
  const cases: SweepCase[] = [];

  const combos = product(kindChoices);
  const uniform = combos.filter((kinds) => kinds.every((k) => k === kinds[0]));
  const mixed = combos.filter((kinds) => kinds.some((k) => k !== kinds[0]));
  const chosen =
    options.kindCoverage === "all"
      ? combos
      : [
          ...uniform,
          ...Array.from(
            { length: Math.min(SAMPLED_MIXED_KINDS, mixed.length) },
            (_, i) => mixed.splice(hash(seed, i) % mixed.length, 1)[0]!,
          ),
        ];

  chosen.forEach((kinds) => {
    const combo = combos.indexOf(kinds);
    const baked = slots.filter((_, i) => kinds[i] !== "run");
    const running = slots.filter((_, i) => kinds[i] === "run");
    const bakedAssignments =
      options.literalCoverage === "all"
        ? product(baked.map(() => options.literalValues))
        : [baked.map((slot) => options.literalValues[hash(seed, combo, slot) % options.literalValues.length]!)];

    for (const bakedValues of bakedAssignments) {
      const bindingFor = (runValues: number[]): Binding[] => {
        const bindings: Binding[] = [];
        slots.forEach((slot, i) => {
          const kind = kinds[i]!;
          const v = kind === "run" ? runValues[running.indexOf(slot)]! : bakedValues[baked.indexOf(slot)]!;
          const w = contexts[slot] && kind !== "splat" ? secondLane(type, v, slot) : v;
          bindings[slot] = { kind, lanes: [v, w] };
        });
        return bindings;
      };
      const template = bindingFor(running.map(() => 0));
      const paramTypes = running.flatMap((slot) => (contexts[slot] ? [type, type] : [type]));
      const kindLabel = slots.map((slot, i) => `x${slot}=${kinds[i]}`).join(" ");
      const bakedLabel = baked.map((slot, i) => `x${slot}=${bakedValues[i]}`).join(" ");

      cases.push({
        label: `${type} ${shape}: ${show(e)} [${kindLabel}${bakedLabel ? `; ${bakedLabel}` : ""}]`,
        type,
        width,
        paramTypes,
        build: (...params) => buildExpr(type, e, shape, contexts, template, params),
        runs: product(running.map(() => options.runValues)).map((runValues) => {
          const bindings = bindingFor(runValues);
          const want =
            width === 1 ? [evaluate(type, e, bindings, 0)] : [0, 1].map((l) => evaluate(type, e, bindings, l as 0 | 1));
          const args = running.flatMap((slot) => (contexts[slot] ? bindings[slot]!.lanes : [bindings[slot]!.lanes[0]]));
          return { args, want };
        }),
      });
    }
  });
  return cases;
}

function opsFor(type: IntegerType): Op[] {
  return [
    ...BINARY_OPS.map((op) => ({ kind: "binary", op }) as Op),
    ...COMPARISONS.map((op) => ({ kind: "compare", op }) as Op),
    ...unaryOps(type).map((op) => ({ kind: "unary", op }) as Op),
  ];
}

/** `op` applied to fresh leaves numbered from `first`. */
function applied(op: Op, first: number): Expr {
  return { op, args: Array.from({ length: arity(op) }, (_, i) => ({ slot: first + i })) };
}

/** Every operation applied to leaves, including `clamp`, which takes three. */
function singleOperations(type: IntegerType): Expr[] {
  return [...opsFor(type), { kind: "clamp" } as Op].map((op) => applied(op, 0));
}

/**
 * Every operation with every other operation as one of its operands: the
 * inner operation is the first operand in one expression and the last in
 * another, since precedence and folding bugs differ between the two sides.
 */
function compositions(type: IntegerType): Expr[] {
  const ops = opsFor(type);
  const out: Expr[] = [];
  for (const outer of ops) {
    for (const inner of ops) {
      const inside = applied(inner, 0);
      const next = arity(inner);
      if (arity(outer) === 1) {
        out.push({ op: outer, args: [inside] });
        continue;
      }
      out.push({ op: outer, args: [inside, { slot: next }] });
      out.push({ op: outer, args: [{ slot: next }, inside] });
    }
  }
  return out;
}

// === Groups ===

/**
 * A named set of cases. `cases` builds them on first use and `release` drops
 * them: a group holds up to a few million runs, so each is freed once its
 * tests are done rather than all being held until the file ends.
 */
type Group = { name: string; cases: () => SweepCase[]; release: () => void };

function group(name: string, make: () => SweepCase[]): Group {
  let cases: SweepCase[] | undefined;
  return { name, cases: () => (cases ??= make()), release: () => (cases = undefined) };
}

const TYPES: IntegerType[] = ["int", "uint"];

const groups: Group[] = [
  ...TYPES.flatMap((type) =>
    SHAPES.map((shape) =>
      group(`single operations, ${type} ${shape}`, () =>
        singleOperations(type).flatMap((e, i) =>
          casesFor(
            type,
            e,
            shape,
            {
              // clamp has three operands; every combination of the full edge
              // set would be thousands of programs for one operation.
              runValues: arity((e as { op: Op }).op) === 3 ? COMPOSITION_EDGE[type] : EDGE[type],
              literalValues: arity((e as { op: Op }).op) === 3 ? SPECIAL[type].slice(0, 5) : SPECIAL[type],
              literalCoverage: "all",
              kindCoverage: "all",
            },
            i,
          ),
        ),
      ),
    ),
  ),
  ...TYPES.flatMap((type) =>
    SHAPES.map((shape) =>
      group(`compositions, ${type} ${shape}`, () =>
        compositions(type).flatMap((e, i) =>
          casesFor(
            type,
            e,
            shape,
            {
              runValues: COMPOSITION_EDGE[type],
              literalValues: SPECIAL[type],
              literalCoverage: "sample",
              kindCoverage: FULL_SWEEP ? "all" : "sample",
            },
            i,
          ),
        ),
      ),
    ),
  ),
  ...TYPES.map((from) => group(`conversions from ${from}`, () => conversionCases(from))),
];

/**
 * `toInt`/`toUint` on each kind of operand, and a vector of the other
 * signedness built from it, one or two components at a time. Conversions
 * keep the bits, as WGSL's `i32(u32)` and `u32(i32)` do.
 */
function conversionCases(from: IntegerType): SweepCase[] {
  const to: IntegerType = from === "int" ? "uint" : "int";
  const scalar = scalarOf(from);
  const target = vectorOf(to);
  const convertMethod = to === "int" ? "toInt" : "toUint";
  const k = FOLD_OFFSET;
  const leaves: { kind: string; make: (v: number, p?: Node<any>) => any }[] = [
    { kind: "run", make: (_v, p) => p },
    { kind: "literal", make: (v) => scalar(v) },
    { kind: "folded", make: (v) => scalar(wrap(from, v - k)).add(scalar(k)) },
  ];
  const forms: { name: string; width: 1 | 2; build: (x: any) => any }[] = [
    { name: `x.${convertMethod}()`, width: 1, build: (x) => x[convertMethod]() },
    { name: `x.${convertMethod}().add(1)`, width: 1, build: (x) => x[convertMethod]().add(scalarOf(to)(1)) },
    { name: `${to === "int" ? "ivec2" : "uvec2"}(x)`, width: 2, build: (x) => target(x) },
    { name: `${to === "int" ? "ivec2" : "uvec2"}(x, x)`, width: 2, build: (x) => target(x, x) },
  ];
  const cases: SweepCase[] = [];
  for (const form of forms) {
    for (const leaf of leaves) {
      const want = (v: number) => {
        const converted = convert(to, v);
        const first = form.name.endsWith(".add(1)") ? wrap(to, converted + 1) : converted;
        return form.width === 1 ? [first] : [first, first];
      };
      if (leaf.kind === "run") {
        cases.push({
          label: `${from} → ${to}: ${form.name} [x=run]`,
          type: to,
          width: form.width,
          paramTypes: [from],
          build: (p) => form.build(leaf.make(0, p)),
          runs: EDGE[from].map((v) => ({ args: [v], want: want(v) })),
        });
        continue;
      }
      for (const v of EDGE[from]) {
        cases.push({
          label: `${from} → ${to}: ${form.name} [x=${leaf.kind} ${v}]`,
          type: to,
          width: form.width,
          paramTypes: [],
          build: () => form.build(leaf.make(v)),
          runs: [{ args: [], want: want(v) }],
        });
      }
    }
  }
  return cases;
}

// === Tests ===

/**
 * Dawn on Metal divides a constant `u32` numerator from `0xFFFFFF80` to
 * `0xFFFFFFFE` by a run-time divisor wrongly: `4294967291u / a` gives
 * `4294967294` for `a = 1`, and `%` is as far off. `0xFFFFFF80` is where a u32
 * starts rounding to 2^32 as an f32, so the driver evidently divides through a
 * float; the same values as run-time operands, or `0xFFFFFFFF`, divide
 * correctly. The generated WGSL is right, so a wrong result from exactly
 * that shape is the platform's and is not counted against the compiler.
 */
function isMetalUintDivisionBug(m: Mismatch): boolean {
  if (m.backend !== "WGSL" || typeof m.got === "string" || !m.source) return false;
  // A scalar numerator, `4294967291u / a`, or a vector one, `vec2<u32>(4294967291u, 2u) / a`.
  const numerators = [...m.source.matchAll(/(\d+u|vec\d<u32>\([^()]*\)) [/%] /g)].map(([, n]) => n!);
  return numerators.some((n) =>
    [...n.matchAll(/(\d+)u/g)].some(([, v]) => Number(v) >= 0xffffff80 && Number(v) <= 0xfffffffe),
  );
}

/** Fail with a readable list, after setting aside results the platform gets wrong. */
function expectNoMismatches(mismatches: Mismatch[]) {
  const counted = mismatches.filter((m) => !isMetalUintDivisionBug(m));
  expect(counted.length, describeMismatches(counted)).toBe(0);
}

describe.skipIf(SWEEP === "skip")("integer sweep", () => {
  for (const group of groups) {
    describe(group.name, () => {
      afterAll(() => group.release());

      it("JS matches the reference", { timeout: 600_000 }, () => {
        const mismatches = sweepJS(group.cases());
        expectNoMismatches(mismatches);
      });

      it("WASM matches the reference", { timeout: 600_000 }, () => {
        const mismatches = sweepWASM(group.cases());
        expectNoMismatches(mismatches);
      });

      it.skipIf(GPU_EVALUATION_SKIPPED)(
        "WGSL matches the reference",
        async () => {
          const mismatches = await sweepWGSL(group.cases());
          expectNoMismatches(mismatches);
        },
        600_000,
      );

      it.skipIf(GPU_EVALUATION_SKIPPED)(
        "GLSL compiles",
        async () => {
          const mismatches = await sweepGLSLValidity(group.cases());
          expectNoMismatches(mismatches);
        },
        600_000,
      );
    });
  }
});
