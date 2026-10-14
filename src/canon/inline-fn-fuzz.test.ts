import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { compileGlsl } from "../glsl";
import { compileWgsl } from "../wgsl";
import { compileJSRoutine } from "../js";
import { compileWasmRoutine } from "../wasm";
import { GPU_ENABLED, installWebGpuGlobals, readsStorageFromBeforeTheLoop, releaseGpu } from "../testing/gpu";
import {
  agrees,
  build,
  generate,
  programSource,
  reference,
  runCpu,
  runWgslCompute,
  UNIFORM_CASES,
  type Outcome,
  type Program,
} from "../testing/inline-fn-fuzz";

/**
 * Random programs of inline `Fn`s, each checked against a reference model of
 * where the canon says each `Fn` runs. A program reads `Fn`s that count their
 * runs in storage, from several roots, inside other `Fn`s, in branches, after
 * them, in loop bodies, in the init, condition and update of a `For`, in the
 * condition of a `While` or a `Loop`, through expressions that read a node many
 * times, through each value of an `Fn` that returns two, and some `Fn`s break,
 * continue or return. The model gives the counts and totals each program
 * writes, or the refusal it gets.
 *
 * How many programs run, from `RMSL_INLINE_FN_FUZZ`:
 * - unset: seeds 1 to 300, about a second;
 * - `long`: seeds 1 to 5000;
 * - a number: seeds 1 to that number.
 * `RMSL_INLINE_FN_FUZZ_SEED` runs the one seed it names. A failure prints the
 * seed and the program, so the run reproduces it.
 */
const FUZZ = process.env.RMSL_INLINE_FN_FUZZ;
const COUNT = FUZZ === undefined ? 300 : FUZZ === "long" ? 5000 : Number(FUZZ);
if (!Number.isInteger(COUNT) || COUNT < 1) {
  throw new Error(`RMSL_INLINE_FN_FUZZ is "${FUZZ}"; it takes "long" or a count of programs, or is left unset.`);
}
const ONE_SEED = process.env.RMSL_INLINE_FN_FUZZ_SEED;
const SEEDS = ONE_SEED !== undefined ? [Number(ONE_SEED)] : Array.from({ length: COUNT }, (_, i) => i + 1);

const none = { name: "main", params: [] } as const;

/** A report of `program` disagreeing with the model on `target`, that names the seed and shows the program. */
function report(program: Program, target: string, uniforms: number[] | undefined, want: Outcome, got: Outcome) {
  return (
    `seed ${program.seed} on ${target}` +
    (uniforms ? ` with uniforms ${uniforms.join(", ")}` : "") +
    `\n  model:  ${JSON.stringify(want)}\n  target: ${JSON.stringify(got)}\n` +
    `  rerun it with RMSL_INLINE_FN_FUZZ_SEED=${program.seed}\n${programSource(program)}`
  );
}

/** The message a compile throws, or `undefined` when it compiles. */
function refusal(compile: () => unknown): string | undefined {
  try {
    compile();
    return undefined;
  } catch (error) {
    return (error as Error).message;
  }
}

/** Whether `program` holds an `Fn` that returns, which a GLSL fragment stage refuses for a reason of its own. */
const returns = (program: Program) => program.fns.some((fn) => fn.jump?.kind === "return");

describe("random programs of inline Fns", () => {
  /**
   * JavaScript and WASM run every program and give the counts and totals the
   * model gives, for each pair of branches the uniforms pick. A program the
   * model refuses is refused on every target, with the message the model
   * names: on GLSL only when no `Fn` returns, since a fragment stage refuses a
   * `Return()` for a reason of its own.
   *
   * @canon spec-random-inline-fn-programs-run-as-the-rule-says-on-the-cpu-targets
   */
  it(
    `run each inline Fn as the rule says in ${SEEDS.length} programs on JS and WASM, and refuse alike on every target`,
    {
      timeout: 3_600_000,
    },
    () => {
      const failures: string[] = [];
      for (const seed of SEEDS) {
        const program = generate(seed);
        const want = reference(program, UNIFORM_CASES);
        for (const [target, compile] of [
          ["JS", compileJSRoutine],
          ["WASM", compileWasmRoutine],
        ] as const) {
          let got: Outcome[];
          try {
            got = runCpu(program, compile as any, UNIFORM_CASES);
          } catch (error) {
            failures.push(report(program, target, undefined, want[0]!, { refused: [String(error)] }));
            continue;
          }
          const wrong = got.findIndex((outcome, i) => !agrees(want[i]!, outcome));
          if (wrong >= 0) failures.push(report(program, target, UNIFORM_CASES[wrong], want[wrong]!, got[wrong]!));
        }
        const roots = () => build(program).build();
        const gpu: [string, () => unknown][] = [["WGSL", () => compileWgsl.compute(roots())]];
        if ("refused" in want[0]! && !returns(program)) gpu.push(["GLSL", () => compileGlsl.fragment(roots())]);
        for (const [target, compile] of gpu) {
          const message = refusal(compile);
          const got: Outcome = message === undefined ? { counters: [], outs: [] } : { refused: [message] };
          const compiled = "refused" in want[0]! ? agrees(want[0]!, got) : message === undefined;
          if (!compiled) failures.push(report(program, target, undefined, want[0]!, got));
        }
      }
      if (failures.length > 0) {
        throw new Error(`${failures.length} disagreements with the model:\n\n${failures.slice(0, 5).join("\n\n")}`);
      }
    },
  );
});

describe.skipIf(!GPU_ENABLED)("random programs of inline Fns on WGSL compute on a Dawn device", () => {
  let uninstall: (() => void) | undefined;
  beforeAll(async () => {
    uninstall = await installWebGpuGlobals();
  });
  afterAll(async () => {
    uninstall?.();
    await releaseGpu();
  }, 120_000);

  /** How many programs the GPU runs: a sample, since each builds a pipeline. */
  const SAMPLE = FUZZ === undefined ? 40 : 400;

  /**
   * A sample of the programs the model runs, each on WGSL compute, for each
   * pair of branches. Where the probe finds that the driver reads storage
   * from before the loop, the sample leaves out a program that holds a loop.
   * It also leaves out a program whose totals an f32 does not hold exactly.
   *
   * @canon spec-a-sample-of-random-inline-fn-programs-runs-as-the-rule-says-on-wgsl-compute
   * @canon exception-the-wgsl-sample-of-random-inline-fn-programs-leaves-out-loops-where-storage-reads-go-stale
   */
  it(
    `run each inline Fn as the rule says in a sample of ${SAMPLE} programs on WGSL compute`,
    {
      timeout: 3_600_000,
    },
    async () => {
      const stale = await readsStorageFromBeforeTheLoop();
      const failures: string[] = [];
      let ran = 0;
      for (const seed of SEEDS) {
        if (ran >= SAMPLE) break;
        const program = generate(seed);
        if (stale && holdsALoop(program)) continue;
        const want = reference(program, UNIFORM_CASES);
        if ("refused" in want[0]!) continue;
        if (want.some((w) => "outs" in w && [...w.outs, ...w.counters].some((x) => Math.abs(x) >= 2 ** 24))) continue;
        ran++;
        const got = await runWgslCompute(program, UNIFORM_CASES);
        const wrong = got.findIndex((outcome, i) => !agrees(want[i]!, outcome));
        if (wrong >= 0) failures.push(report(program, "WGSL", UNIFORM_CASES[wrong], want[wrong]!, got[wrong]!));
      }
      expect(ran).toBeGreaterThan(0);
      if (failures.length > 0) {
        throw new Error(`${failures.length} disagreements with the model:\n\n${failures.slice(0, 5).join("\n\n")}`);
      }
    },
  );
});

/** Whether `program` holds a loop, where a driver that reads storage from before the loop can count wrongly. */
function holdsALoop(program: Program): boolean {
  const inBlock = (list: Program["roots"][number]): boolean =>
    list.some((s) => {
      if (s.k === "if") return inBlock(s.then) || (s.else !== undefined && inBlock(s.else));
      return s.k !== "out";
    });
  return program.roots.some(inBlock) || program.fns.some((fn) => inBlock(fn.body));
}

describe("the inline Fn fuzz", () => {
  /**
   * The model counts the runs of a program written by hand, and the check
   * rejects an outcome that differs from it in one count or one total.
   *
   * @canon spec-the-inline-fn-fuzz-tells-right-from-wrong
   */
  it("gives the runs of a program written by hand, and rejects a wrong count", () => {
    // f0 is read in both branches of an If, so it runs once, before the If.
    const program: Program = {
      seed: 0,
      slots: 2,
      fns: [{ body: [], values: 1, extra: [undefined] }],
      roots: [
        [
          {
            k: "if",
            cond: { u: 0 },
            then: [{ k: "out", slot: 0, e: { k: "read", fn: 0, value: 0 } }],
            else: [{ k: "out", slot: 1, e: { k: "read", fn: 0, value: 0 } }],
          },
        ],
      ],
    };
    const [taken, other] = reference(program, [
      [1, 0],
      [0, 0],
    ]);
    expect(taken).toEqual({ counters: [1], outs: [1, 0] });
    expect(other).toEqual({ counters: [1], outs: [0, 1] });
    expect(agrees(taken!, { counters: [2], outs: [1, 0] })).toBe(false);
    expect(agrees(taken!, { counters: [1], outs: [1, 1] })).toBe(false);
    expect(agrees(taken!, { counters: [1], outs: [1, 0] })).toBe(true);
  });

  /**
   * @canon spec-the-inline-fn-fuzz-tells-right-from-wrong
   */
  it("builds the same program from the same seed, and shows it with its seed", () => {
    expect(generate(42)).toEqual(generate(42));
    expect(generate(42)).not.toEqual(generate(43));
    expect(programSource(generate(42))).toContain("seed 42");
  });

  /**
   * A refusal agrees only with a refusal the model names.
   *
   * @canon spec-the-inline-fn-fuzz-tells-right-from-wrong
   */
  it("rejects a refusal the model does not name, and a run where it expects a refusal", () => {
    const want: Outcome = { refused: ["one"] };
    expect(agrees(want, { refused: ["one"] })).toBe(true);
    expect(agrees(want, { refused: ["two"] })).toBe(false);
    expect(agrees(want, { counters: [], outs: [] })).toBe(false);
  });
});
