/**
 * Checks that every shader the test suite generates is actually valid, in both
 * backends.
 *
 * The suite's assertions are substring matches — `expect(glsl).toContain(...)`
 * — which cannot tell a correct shader from a broken one that happens to
 * contain the expected text. Three real compiler bugs shipped through that gap:
 * `refract(I, N)` still contains "refract(", `lessThan(float, float)` still
 * contains "lessThan(", and a for-loop missing its increment is valid syntax.
 *
 * So the generated source is handed to real compilers instead of being pattern
 * matched. A parser is not enough — it accepts every one of those three, since
 * they are type errors rather than syntax errors:
 *
 *              1e-7.0   refract(I,N)   lessThan(f,f)   mat2x3 = 0.0
 *   parser     reject      accept         accept          accept
 *   real       reject      reject         reject          reject
 *
 * GLSL goes to Chromium's WebGL2 compiler, WGSL to Dawn. Both report type
 * errors, and both are the implementations the output actually has to run on.
 *
 * Wiring: a test file aliases the compilers through `recordGLSL`/`recordWGSL`,
 * so no individual test changes, then awaits `assertRecordedShadersValid()` in
 * `afterAll`.
 */

import { expect } from "vitest";
import {
  compileGLSL,
  compileWGSL,
  type Node,
  type ShaderType,
  type VertexRoot,
} from "../rmsl";

// This file only ever runs under vitest, in Node. Declared here rather than
// depending on @types/node, which the package itself has no use for.
// Written to rather than console.warn: vitest intercepts console output and
// does not surface it here, so a warning sent that way is not seen at all.
declare const process: {
  env: Record<string, string | undefined>;
  stderr: { write(message: string): void };
};

export type ShaderLang = "glsl" | "wgsl";
export type ShaderStage = "vertex" | "fragment";

export interface Recorded {
  test: string;
  lang: ShaderLang;
  stage: ShaderStage;
  /**
   * Ties the two languages' entries for one program together. A test may
   * compile several programs at the same stage, so test and stage alone do not
   * identify a pair.
   */
  pair: number;
  /** null when the compiler threw rather than producing source. */
  src: string | null;
  compileError?: string;
}

const recorded: Recorded[] = [];
let nextPair = 0;

/**
 * Shaders that are known to be invalid, with the defect that produces them.
 *
 * Keyed `lang:test name`. Entries are asserted to still fail — fixing a bug
 * without removing its entry is itself reported, so the list cannot outlive
 * the problems it documents.
 */
export const KNOWN_INVALID: Record<string, string> = {
  // FXAA's whole algorithm is data-dependent: it decides whether a pixel sits
  // on an edge from sampled luminance, then walks that edge. WebGPU requires
  // texture sampling to happen in uniform control flow, so Dawn rejects every
  // sample that ends up inside the skip-pixel guard or an edge branch. A
  // branchless (select-based) port would satisfy it, but that is a rewrite of
  // the algorithm, not a codegen fix.
  "wgsl:texture-based effects > fxaa": "WGSL forbids textureSample in the non-uniform control flow FXAA's edge detection requires",
};

/**
 * Compile a program to *both* backends, whichever one the test asked for.
 *
 * Compiling both here makes parity structural — a new test gets it for free —
 * while the test's own assertions still run against the language it chose.
 */
function recordBoth(root: any, stage: ShaderStage, want: ShaderLang): string {
  const test = expect.getState().currentTestName ?? "<unknown test>";
  const pair = nextPair++;
  const entries: Recorded[] = [];

  for (const [lang, compile] of [
    ["glsl", compileGLSL] as const,
    ["wgsl", compileWGSL] as const,
  ]) {
    const entry: Recorded = { test, lang, stage, pair, src: null };
    try {
      entry.src =
        stage === "vertex" ? compile.vertex(root) : compile.fragment(root);
    } catch (error) {
      // A backend refusing to compile is itself a signal, so it is recorded
      // rather than swallowed.
      entry.compileError =
        error instanceof Error ? error.message : String(error);
    }
    entries.push(entry);
    recorded.push(entry);
  }

  // The caller's own compilation is one of these, so it is returned rather than
  // run a third time.
  const wanted = entries.find((e) => e.lang === want)!;
  if (wanted.compileError) throw new Error(wanted.compileError);
  return wanted.src!;
}

/**
 * Submit a hand-assembled shader for validation.
 *
 * `compileGLSLFn` / `compileWGSLFn` emit a single function rather than a whole
 * shader, so they cannot be compiled on their own. A test can wrap the emitted
 * function in a minimal shader and pass it here, which checks the thing that
 * actually matters about them: that what they produce compiles when embedded.
 */
export function recordShaderSource(
  lang: ShaderLang,
  stage: ShaderStage,
  src: string,
): string {
  recorded.push({
    test: expect.getState().currentTestName ?? "<unknown test>",
    lang,
    stage,
    pair: nextPair++, // no counterpart; this is a single-language submission
    src,
  });
  return src;
}

/**
 * Shape of `compileGLSL` / `compileWGSL`: callable, with vertex and fragment.
 *
 * Mirrors the real signatures rather than taking anything. These stand-ins are
 * what the tests import, so typing them loosely would mean no test in that file
 * was checked on what it hands a compiler — the aliasing that buys every test a
 * real driver would have cost it the type checker.
 */
interface Compiler {
  (root: Node<ShaderType> | readonly Node<ShaderType>[]): string;
  vertex(root: VertexRoot): string;
  fragment(root: Node<ShaderType> | readonly Node<ShaderType>[]): string;
}

function wrap(lang: ShaderLang): Compiler {
  return Object.assign(
    (root: Node<ShaderType> | readonly Node<ShaderType>[]) =>
      recordBoth(root, "fragment", lang),
    {
      vertex: (root: VertexRoot) => recordBoth(root as any, "vertex", lang),
      fragment: (root: Node<ShaderType> | readonly Node<ShaderType>[]) =>
        recordBoth(root, "fragment", lang),
    },
  );
}

/**
 * Drop-in replacements for `compileGLSL` / `compileWGSL`. Each returns the
 * language it is named for, and records both.
 */
export const recordingGLSL = wrap("glsl");
export const recordingWGSL = wrap("wgsl");

/** Compile every recorded GLSL shader, in the shared browser page. */
async function validateGLSL(items: Recorded[]): Promise<(string | null)[]> {
  const { compileGLSLInPage } = await import("./gpu");
  return compileGLSLInPage(items.map((i) => ({ src: i.src!, stage: i.stage })));
}

/** Compile every recorded WGSL shader through Dawn. */
async function validateWGSL(items: Recorded[]): Promise<(string | null)[]> {
  if (items.length === 0) return [];
  const { gpuDevice } = await import("./gpu");
  const device = await gpuDevice();

  // pushErrorScope, createShaderModule and popErrorScope are all synchronous
  // stack operations — only the *result* of a pop is asynchronous. So every
  // module is created and its scope popped up front, and the results are
  // awaited together.
  const pending = items.map((item) => {
    device.pushErrorScope("validation");
    device.createShaderModule({ code: item.src! });
    return device.popErrorScope();
  });

  const errors = await Promise.all(pending);
  return errors.map((error) =>
    error
      ? (error.message
          .split("\n")
          .find((l: string) => l.includes("error:"))
          ?.trim() ?? error.message.split("\n")[0])
      : null,
  );
}

/**
 * Validate everything recorded so far and throw a single report if the set of
 * invalid shaders differs from `KNOWN_INVALID` in either direction.
 */
export async function assertRecordedShadersValid(): Promise<void> {
  // Validation runs unless a machine says it cannot, via RMSL_SKIP_GPU — for
  // one without a graphics device or a browser to install, such as Android
  // under Termux. Skipping is announced rather than silent, so a run without
  // shader checking cannot be mistaken for one with it.
  if (process.env.RMSL_SKIP_GPU || process.env.RMSL_SKIP_SHADER_VALIDATION) {
    process.stderr.write(
      `\n[shader-validity] SKIPPED — ${recorded.length} shaders were recorded` +
        ` but never handed to a real compiler.\n`,
    );
    return;
  }

  // An entry whose compiler threw has no source to hand a validator; it is
  // already accounted for below.
  const glsl = recorded.filter((r) => r.lang === "glsl" && r.src !== null);
  const wgsl = recorded.filter((r) => r.lang === "wgsl" && r.src !== null);

  const [glslErrors, wgslErrors] = await Promise.all([
    validateGLSL(glsl),
    validateWGSL(wgsl),
  ]);

  // Validation runs once, from an afterAll hook, so what it opened is released
  // here rather than left for the process to reclaim.
  const { releaseGpu } = await import("./gpu");
  await releaseGpu();

  const report = validationReport(recorded, glslErrors, wgslErrors);
  if (report) throw new Error(report);
}

/**
 * Turn a validation run into a report, or null when there is nothing to say.
 *
 * Kept separate from the GPU work so the reporting can be tested without a
 * browser or a device — this is the part that decides whether a run passes, so
 * a mistake here is invisible in exactly the way the harness exists to prevent.
 */
export function validationReport(
  recorded: Recorded[],
  glslErrors: (string | null)[],
  wgslErrors: (string | null)[],
  known: Record<string, string> = KNOWN_INVALID,
): string | null {
  // Nothing recorded means nothing was checked, which is not the same as
  // everything being fine. A run filtered down to tests that compile nothing,
  // or a test file that lost the aliased import, would otherwise finish green
  // having verified none of the shaders it appears to cover.
  if (recorded.length === 0) {
    return (
      "Validated no shaders at all. Either the run was filtered down to" +
      " tests that compile nothing, or a test file is importing compileGLSL" +
      " / compileWGSL directly instead of the recording stand-ins in" +
      " src/testing/shader-validity.ts. Set RMSL_SKIP_SHADER_VALIDATION=1 if" +
      " skipping validation is what you meant."
    );
  }

  const glsl = recorded.filter((r) => r.lang === "glsl" && r.src !== null);
  const wgsl = recorded.filter((r) => r.lang === "wgsl" && r.src !== null);

  // Every failing shader is kept, not just the last one per test. A single test
  // routinely compiles dozens — the breadth tests loop over every builtin — and
  // overwriting by test name reported one defect while hiding the rest.
  const failed = new Map<string, string[]>();
  const fail = (key: string, message: string) => {
    const existing = failed.get(key);
    if (existing) existing.push(message);
    else failed.set(key, [message]);
  };

  for (const entry of recorded) {
    if (!entry.compileError) continue;
    // Both backends refusing a program is consistent, and some tests assert
    // exactly that — builtinFragDepth() in a vertex stage, for one. Only one
    // backend refusing what the other accepts is a gap worth reporting.
    const counterpart = recorded.find(
      (r) => r.pair === entry.pair && r.lang !== entry.lang,
    );
    if (counterpart?.compileError) continue;
    fail(
      `${entry.lang}:${entry.test}`,
      `did not compile — ${entry.compileError}`,
    );
  }
  // Compared against null rather than tested for truthiness: an empty string is
  // a failure whose driver said nothing, and treating it as valid is how a lost
  // context could mark every remaining shader as fine.
  glsl.forEach((item, i) => {
    if (glslErrors[i] != null) fail(`glsl:${item.test}`, glslErrors[i]!);
  });
  wgsl.forEach((item, i) => {
    if (wgslErrors[i] != null) fail(`wgsl:${item.test}`, wgslErrors[i]!);
  });

  const unexpected = [...failed].filter(([key]) => !(key in known));
  // An entry is only "fixed" when the shader it names was actually recorded and
  // validated in this run. KNOWN_INVALID is shared across every test file, and
  // each file only records its own shaders — an entry for a shader this file
  // never compiled would otherwise read as "now compiles" every single run.
  const recordedKeys = new Set(recorded.map((r) => `${r.lang}:${r.test}`));
  const fixed = Object.keys(known).filter(
    (key) => recordedKeys.has(key) && !failed.has(key),
  );

  if (unexpected.length === 0 && fixed.length === 0) return null;

  const report: string[] = [
    `Validated ${recorded.length} generated shaders from` +
      ` ${new Set(recorded.map((r) => r.test)).size} tests` +
      ` (${glsl.length} GLSL, ${wgsl.length} WGSL).`,
  ];

  if (unexpected.length > 0) {
    const count = unexpected.reduce((n, [, errors]) => n + errors.length, 0);
    report.push("", `${count} shader(s) newly invalid:`);
    for (const [key, errors] of unexpected) {
      report.push(`  ${describe(key)}`);
      for (const error of errors) report.push(`      ${error}`);
    }
  }

  if (fixed.length > 0) {
    report.push(
      "",
      `${fixed.length} entr(y/ies) in KNOWN_INVALID now compile — delete them from`,
      "src/testing/shader-validity.ts so the list keeps meaning something:",
    );
    for (const key of fixed) report.push(`  ${describe(key)}  [key: ${key}]`);
  }

  return report.join("\n");
}

/**
 * Render a `lang:test` key for reading.
 *
 * Every test's program is compiled to both backends, so a test named for one
 * language routinely fails in the other — "wgsl:… compiles If/Else to GLSL"
 * looks like a contradiction until you know that. Naming the backend as the
 * output being reported, separately from the test, removes the collision.
 */
function describe(key: string): string {
  const separator = key.indexOf(":");
  const lang = key.slice(0, separator).toUpperCase();
  const test = key.slice(separator + 1);
  return `${test}\n      -> invalid ${lang} output`;
}
