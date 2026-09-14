/**
 * Runs a compiled expression and reports the number it produces.
 *
 * An expression is compiled to a function, called on both backends, and the
 * result compared against the same arithmetic in JS. Running both also makes
 * the backends checkable against each other: one RMSL program must produce one
 * number, and a divergence is a bug in whichever side disagrees with JS.
 *
 * GLSL renders to an RGBA32F texture and reads the red channel; WGSL dispatches
 * a compute shader and reads a storage buffer. Both return exact f32, so the
 * only tolerance needed is for f32 against JS's f64.
 */

import { expect } from "vitest";
import { compileGLSLFn, compileWGSLFn, compileJSFn, compileWasm, var_, type Node } from "../rmsl";
import { MATRIX_DIMENSIONS, TYPE_WIDTH } from "../rmsl-core";

// Written to rather than console.warn: vitest intercepts console output and
// does not surface it here, so a warning sent that way is not seen at all.
declare const process: {
  env: Record<string, string | undefined>;
  stderr: { write(message: string): void };
};

/**
 * Difference allowed between two results of the same magnitude.
 *
 * A 32-bit float carries 24 bits of mantissa, so neighbouring representable
 * values at |x| are about |x| * 2^-23 apart — at 1024 that is 1.2e-4, larger
 * than any flat tolerance worth using. Two independent implementations are not
 * obliged to agree bit for bit: `pow` is commonly evaluated as
 * exp2(y * log2(x)) and lands a unit or two either side, so a fixed allowance
 * either fails on correct backends at large magnitudes or waves through real
 * mistakes at small ones.
 *
 * Scaling with magnitude gives roughly eight units in the last place, with a
 * floor for values near zero where the relative gap collapses.
 */
export function floatTolerance(magnitude: number): number {
  return Math.max(1e-6, Math.abs(magnitude) * 1e-6);
}

/** Every root type a recorded program is allowed to return — scalar, vector or matrix. */
export type EvaluableRoot =
  | Node<"float">
  | Node<"vec2">
  | Node<"vec3">
  | Node<"vec4">
  | Node<"mat2">
  | Node<"mat2x3">
  | Node<"mat2x4">
  | Node<"mat3">
  | Node<"mat3x2">
  | Node<"mat3x4">
  | Node<"mat4">
  | Node<"mat4x2">
  | Node<"mat4x3">;

type Build = (...args: Node<"float">[]) => EvaluableRoot;

function params(count: number) {
  return Array.from({ length: count }, (_, i) => ({ name: `a${i}`, type: "float" as const }));
}

function componentCountOf(t: string): number {
  const width = TYPE_WIDTH[t];
  if (width !== undefined) return width;
  const shape = MATRIX_DIMENSIONS[t];
  if (shape !== undefined) return shape[0] * shape[1];
  return 1;
}

function callExpr(args: number[]) {
  // Emitted as literals. Whether the driver folds them is immaterial: if the
  // wrong operator was emitted the answer is wrong either way.
  return `rmsl_eval(${args.map((a) => (Number.isInteger(a) ? a.toFixed(1) : String(a))).join(", ")})`;
}

/**
 * The root node's `_t`, found by calling `build` with placeholder float vars
 * rather than real arguments — purely to read the type off the resulting
 * expression graph, since the fully-compiled function no longer exposes it.
 */
function rootType(build: Build, argCount: number): string {
  const probes = Array.from({ length: argCount }, (_, i) => var_(`a${i}`, "float"));
  const root = build(...(probes as Node<"float">[]));
  return (root as unknown as { _t: string })._t;
}

/** `r[col][row]` for a matrix element, `r[i]` for a vector component (GLSL and WGSL alike). */
function elementIndices(type: string, i: number): { col: number; row: number } | null {
  const shape = MATRIX_DIMENSIONS[type];
  if (shape === undefined) return null;
  const rows = shape[1];
  return { col: Math.floor(i / rows), row: i % rows };
}

/**
 * Read `n` scalar components out of a compiled GLSL function by rendering
 * into a `ceil(n/4)`-wide RGBA32F row and picking the four components of each
 * pixel with an unrolled `gl_FragCoord.y` dispatch — matrix subscripts must
 * be constant in GLSL, so the dispatch itself has to be what varies, not the
 * index expressions inside it.
 */
async function evaluateGLSLElements(fn: string, args: number[], type: string, n: number): Promise<Float32Array> {
  const width = Math.ceil(n / 4);
  const element = (i: number) => {
    if (type === "float") return "r";
    const idx = elementIndices(type, i);
    return idx === null ? `r[${i}]` : `r[${idx.col}][${idx.row}]`;
  };
  const rowLines = Array.from({ length: width }, (_, row) => {
    const base = row * 4;
    const comps = Array.from({ length: 4 }, (_, k) => (base + k < n ? element(base + k) : "0.0"));
    return `  if (int(gl_FragCoord.y) == ${row}) result = vec4(${comps.join(", ")});`;
  });
  const source = `#version 300 es
precision highp float;
${fn}
layout(location=0) out vec4 result;
void main() {
  ${type} r = ${callExpr(args)};
${rowLines.join("\n")}
}`;

  const { gpuPage } = await import("./gpu");
  const page = await gpuPage();
  const out = await page.evaluate(
    ({ fragment, width }: { fragment: string; width: number }) => {
      // A fresh context per call. The page is what is expensive to stand up —
      // opening and navigating one cost around 130ms — and keeping a context
      // alive across calls turned out to be unreliable: SwiftShader drops it
      // now and again, and every later call in the run then fails.
      const gl = document.createElement("canvas").getContext("webgl2")!;
      if (!gl.getExtension("EXT_color_buffer_float")) {
        throw new Error("EXT_color_buffer_float unavailable; cannot read float output");
      }
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 1, width, 0, gl.RGBA, gl.FLOAT, null);
      const framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      const vertices = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vertices);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);

      // Compiled inline rather than through a helper: the bundler renames
      // functions and injects a `__name` shim that does not exist in the page.
      const program = gl.createProgram()!;
      for (const [src, type] of [
        [`#version 300 es\nin vec2 p; void main(){ gl_Position = vec4(p,0.,1.); }`, gl.VERTEX_SHADER],
        [fragment, gl.FRAGMENT_SHADER],
      ] as [string, number][]) {
        const shader = gl.createShader(type)!;
        gl.shaderSource(shader, src);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
          throw new Error(gl.getShaderInfoLog(shader) || "shader failed to compile");
        }
        gl.attachShader(program, shader);
      }
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(program) ?? "program failed to link");
      }
      gl.useProgram(program);

      const location = gl.getAttribLocation(program, "p");
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0);

      gl.viewport(0, 0, 1, width);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      const out = new Float32Array(4 * width);
      gl.readPixels(0, 0, 1, width, gl.RGBA, gl.FLOAT, out);

      return Array.from(out);
    },
    { fragment: source, width },
  );
  return new Float32Array(out);
}

/** Compile, run and read back the GLSL backend's result — a scalar, vector or matrix. */
export async function evaluateGLSL(build: Build, args: number[] = []): Promise<number | number[]> {
  const fn = compileGLSLFn(build, { name: "rmsl_eval", params: params(args.length) });
  const type = rootType(build, args.length);
  const n = componentCountOf(type);
  const out = await evaluateGLSLElements(fn, args, type, n);
  return n === 1 ? out[0]! : Array.from(out.subarray(0, n));
}

/**
 * Read `n` scalar components out of a compiled WGSL function by writing them
 * into a storage buffer at constant indices — WGSL matrix subscripts, like
 * GLSL's, must be constant, so every element gets its own unrolled store.
 */
async function evaluateWGSLElements(fn: string, args: number[], type: string, n: number): Promise<Float32Array> {
  const stores = Array.from({ length: n }, (_, i) => {
    if (type === "float") return `  result[${i}] = r;`;
    const idx = elementIndices(type, i);
    return idx === null ? `  result[${i}] = r[${i}];` : `  result[${i}] = r[${idx.col}][${idx.row}];`;
  });
  const code = `${fn}
@group(0) @binding(0) var<storage, read_write> result: array<f32>;
@compute @workgroup_size(1)
fn main() {
  let r = ${callExpr(args)};
${stores.join("\n")}
}`;
  return runWGSLElements(code, n);
}

/** Compile, run and read back the WGSL backend's result — a scalar, vector or matrix. */
export async function evaluateWGSL(build: Build, args: number[] = []): Promise<number | number[]> {
  const fn = compileWGSLFn(build, { name: "rmsl_eval", params: params(args.length) });
  const type = rootType(build, args.length);
  const n = componentCountOf(type);
  const out = await evaluateWGSLElements(fn, args, type, n);
  return n === 1 ? out[0]! : Array.from(out.subarray(0, n));
}

/**
 * Run a compute shader that writes `floatCount` floats to `result[0..)`, and
 * read them all back.
 *
 * Separate from `evaluateWGSL` so the execution path can be exercised with
 * source the compiler would never produce, which is the only way to check that
 * a shader failing to compile is actually reported.
 */
export async function runWGSLElements(code: string, floatCount: number): Promise<Float32Array> {
  const { gpuDevice } = await import("./gpu");
  const gpu = await gpuDevice();

  // A shader that fails to compile is reported as an uncaptured device error
  // rather than an exception: the pipeline, the dispatch and the copy that
  // follow are all quietly invalid.
  gpu.pushErrorScope("validation");
  const module = gpu.createShaderModule({ code });
  const pipeline = gpu.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  // Popped now but awaited later: asking the device for the answer here would
  // block on a round trip before any work is even submitted. The dispatch that
  // follows is harmless if the module turned out to be invalid — it produces a
  // buffer of zeroes, which is exactly why the result cannot be trusted until
  // this has been checked.
  const compileFailure = gpu.popErrorScope();

  const STORAGE = 0x80,
    COPY_SRC = 0x4,
    MAP_READ = 0x1,
    COPY_DST = 0x8;
  const byteSize = 4 * floatCount;
  const storage = gpu.createBuffer({ size: byteSize, usage: STORAGE | COPY_SRC });
  const readback = gpu.createBuffer({ size: byteSize, usage: MAP_READ | COPY_DST });
  try {
    const encoder = gpu.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      gpu.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: storage } }],
      }),
    );
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(storage, 0, readback, 0, byteSize);
    gpu.queue.submit([encoder.finish()]);
    const [failure] = await Promise.all([compileFailure, readback.mapAsync(MAP_READ)]);
    if (failure) {
      const detail =
        failure.message.split("\n").find((l: string) => l.includes("error:")) ?? failure.message.split("\n")[0];
      throw new Error(`WGSL shader failed to compile: ${detail.trim()}`);
    }
    return new Float32Array(readback.getMappedRange().slice(0));
  } finally {
    readback.destroy?.();
    storage.destroy?.();
  }
}

/**
 * Run a compute shader that writes one float to `result[0]`, and read it back.
 *
 * Kept for callers that exercise source the compiler would never produce
 * (checking that a failed compile is reported) with a plain single-float
 * result.
 */
export async function runWGSL(code: string): Promise<number> {
  return (await runWGSLElements(code, 1))[0]!;
}

/**
 * Run an expression on the JS backend — in-process, no GPU, no browser.
 *
 * The result is exact JS f64 rather than the f32 the GPU backends return, so
 * it is the natural arbiter when the two GPUs disagree: whatever the shaders
 * compute, the CPU must agree with the caller's arithmetic.
 */
export function evaluateJS(build: Build, args: number[] = []): number | number[] {
  const fn = compileJSFn(build, { name: "rmsl_eval", params: params(args.length) });
  const callable = new Function(fn)() as (ctx: { params: Record<string, number> }) => number | number[];
  const ctx = { params: Object.fromEntries(args.map((a, i) => [`a${i}`, a])) };
  const value = callable(ctx);
  if (typeof value === "number" || Array.isArray(value)) return value;
  return value as unknown as number;
}

/**
 * Run an expression on the WASM backend — in-process, no GPU, no browser,
 * same as `evaluateJS`.
 *
 * This backend's `float` is f64, matching `compileJS`'s plain JS-number
 * arithmetic bit for bit (`ROADMAP.md`, "`float` is f64") — including the
 * transcendental functions, which both backends call through the literal
 * same `Math` object. So unlike the GLSL/WGSL comparison, nothing here
 * should ever need `floatTolerance`: a real difference is a bug, not
 * rounding.
 */
export function evaluateWASM(build: Build, args: number[] = []): number | number[] {
  const fn = compileWasm(build, { name: "rmsl_eval", params: params(args.length) });
  const ctx = { params: Object.fromEntries(args.map((a, i) => [`a${i}`, a])) };
  const result = fn(ctx);
  // Scalar mode returns the raw number; an aggregate root is instead read
  // back as an output slot, wrapped in a `{ value }` shader-result object.
  if (typeof result === "number") return result;
  const value = (result as { value?: number | number[] }).value;
  if (typeof value === "number" || Array.isArray(value)) return value;
  return result as unknown as number;
}

/**
 * Whether a `compileWasm`/`compileWasmFn` failure means "not supported by
 * this backend yet" rather than a real bug.
 *
 * Every deliberate "can't compile this (yet)" throw in `rmsl-wasm.ts` — for
 * an unsupported node type, a non-square matrix multiply, a multi-return
 * function, and so on — is constructed with this exact prefix (confirmed:
 * every `throw new Error(...)` in that file uses it, whether the case is a
 * known coverage gap or an internal-misuse check). A genuine WASM engine
 * trap (`WebAssembly.RuntimeError`, thrown by the VM itself when a compiled
 * module actually executes a trapping instruction) or a `CompileError`/
 * `LinkError` (malformed bytecode — a real codegen bug) never carries this
 * prefix, so this check only ever recognizes "doesn't compile", never
 * "crashed" or "computed the wrong answer".
 */
function isWasmUnsupported(error: unknown): boolean {
  return error instanceof Error && error.message.startsWith("[RMSL] compileWasmFn");
}

/**
 * Run an expression on both backends.
 *
 * Comparing the two against each other is the part text assertions cannot do:
 * one RMSL program has one meaning, so any disagreement is a defect in
 * whichever backend differs from the arithmetic the caller expected.
 */
export async function evaluateBoth(
  build: Build,
  args: number[] = [],
): Promise<{ glsl: number | number[]; wgsl: number | number[] }> {
  const [glsl, wgsl] = await Promise.all([evaluateGLSL(build, args), evaluateWGSL(build, args)]);
  return { glsl, wgsl };
}

/**
 * Run an expression on all three backends, for the cross-check a single pair
 * cannot do alone: when GLSL and WGSL disagree, the JS result decides which is
 * wrong.
 */
export async function evaluateAll(
  build: Build,
  args: number[] = [],
): Promise<{ glsl: number | number[]; wgsl: number | number[]; js: number | number[] }> {
  const [glsl, wgsl] = await Promise.all([evaluateGLSL(build, args), evaluateWGSL(build, args)]);
  return { glsl, wgsl, js: evaluateJS(build, args) };
}

/** Release the browser and the graphics device held open across evaluations. */
export async function closeEvaluators(): Promise<void> {
  const { releaseGpu } = await import("./gpu");
  await releaseGpu();
}

// === Recording evaluation ===
//
// The problem this solves is that depth used to be a property of which file a
// test was written in. Evaluating on a GPU is asynchronous and needs hardware,
// so the tests that ran on every backend were gated behind `RMSL_GPU` and
// stayed few, while the breadth of the language accumulated in a file that ran
// only on the CPU because that one needed nothing. Neither covered an operand
// that was itself an expression, and five operations computed the wrong answer
// on the CPU target for a year.
//
// So the same arrangement `shader-validity.ts` uses for compilation is used
// here for evaluation: the CPU result is computed immediately and returned, and
// the program is recorded so the GPU backends can be checked against it in an
// `afterAll`. A test calls one synchronous function and is covered everywhere,
// without knowing that is what it is doing.

/** One recorded program, with what the CPU target computed for it. */
interface RecordedEvaluation {
  test: string;
  build: Build;
  args: number[];
  js: number | number[];
  /** Set when the case deliberately does not run on the GPU backends. */
  cpuOnly?: string;
}

function asArray(v: number | number[]): number[] {
  return Array.isArray(v) ? v : [v];
}

function formatValue(v: number | number[]): string {
  return Array.isArray(v) ? `[${v.join(", ")}]` : String(v);
}

/** Exact equality, elementwise for an aggregate. */
function valuesExactlyEqual(a: number | number[], b: number | number[]): boolean {
  const av = asArray(a);
  const bv = asArray(b);
  return av.length === bv.length && av.every((x, i) => x === bv[i]);
}

/** Within `floatTolerance` of each other, elementwise for an aggregate. */
function valuesWithinTolerance(a: number | number[], b: number | number[]): boolean {
  const av = asArray(a);
  const bv = asArray(b);
  return av.length === bv.length && av.every((x, i) => Math.abs(x - bv[i]!) < floatTolerance(x));
}

const recordedEvaluations: RecordedEvaluation[] = [];

/**
 * Why a case runs on the CPU target alone.
 *
 * Taking a reason rather than a flag keeps the exclusions listable: every case
 * that opted out says here why, so the set can be read and argued with rather
 * than growing quietly whenever a case is inconvenient.
 */
export type CpuOnlyReason = "derivatives" | "reentrant" | "texture" | "js-only-api" | "exceeds-float32";

/**
 * Evaluate on the CPU target and record the program for the GPU backends.
 *
 * Returns the CPU result, which is exact f64 and so is the value a caller's
 * assertion pins. The GPU backends are compared against that same value later,
 * which is what makes one assertion cover three backends.
 */
export function evaluateRecording(build: Build, args: number[] = [], cpuOnly?: CpuOnlyReason): number | number[] {
  const js = evaluateJS(build, args);
  recordedEvaluations.push({
    test: currentTestName(),
    build,
    args,
    js,
    cpuOnly,
  });
  return js;
}

function currentTestName(): string {
  return expect.getState().currentTestName ?? "<unknown test>";
}

/**
 * Replay every recorded program on the GPU backends and report disagreements.
 *
 * Compared against the CPU result rather than against a separately written
 * expectation, because the caller already pinned that result with an assertion
 * of its own. So a mismatch here means the backends disagree about a program
 * whose answer is already known to be right.
 */
export async function assertRecordedEvaluationsAgree(): Promise<void> {
  const runnable = recordedEvaluations.filter((r) => r.cpuOnly === undefined);
  // Recording nothing is not the same as everything agreeing. A file that
  // stopped going through the shared helper would otherwise finish green having
  // checked one backend of three, which is the arrangement this replaced.
  if (recordedEvaluations.length === 0) {
    throw new Error(
      `Evaluated no programs at all. Either the run was filtered down to tests that evaluate nothing, or a test file stopped calling the shared evaluation helper in src/testing/shader-eval.ts. Set RMSL_SKIP_SHADER_EVALUATION=1 if skipping evaluation is what you meant.`,
    );
  }

  const failures: string[] = [];

  // WASM needs neither a browser nor a graphics device, so — unlike GLSL/WGSL
  // below — it always runs, even under RMSL_SKIP_GPU/RMSL_SKIP_SHADER_EVALUATION
  // (those exist specifically to skip hardware-dependent work). A case this
  // backend doesn't compile yet is a countable, visible skip, never a silent
  // one — see `isWasmUnsupported`'s doc comment for why that's safe to do
  // without also hiding a real bug.
  let wasmUnsupported = 0;
  for (const item of runnable) {
    try {
      const wasm = evaluateWASM(item.build, item.args);
      // Exact equality, not floatTolerance — see `evaluateWASM`'s doc comment.
      if (!valuesExactlyEqual(wasm, item.js)) {
        failures.push(`  ${item.test}\n      WASM computed ${formatValue(wasm)}, CPU computed ${formatValue(item.js)}`);
      }
    } catch (error) {
      if (!isWasmUnsupported(error)) throw error;
      wasmUnsupported++;
    }
  }
  if (wasmUnsupported > 0) {
    process.stderr.write(
      `\n[shader-eval] WASM: ${wasmUnsupported} of ${runnable.length} recorded programs are not supported by this backend yet (skipped, not failed) — see ROADMAP.md.\n`,
    );
  }

  if (GPU_EVALUATION_SKIPPED) {
    process.stderr.write(
      `\n[shader-eval] SKIPPED — ${runnable.length} programs ran on the CPU and WASM targets only; neither shading language was evaluated.\n`,
    );
  } else if (runnable.length > 0) {
    for (const item of runnable) {
      let glsl: number | number[];
      let wgsl: number | number[];
      try {
        [glsl, wgsl] = await Promise.all([evaluateGLSL(item.build, item.args), evaluateWGSL(item.build, item.args)]);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        failures.push(`  ${item.test}\n      did not evaluate — ${message}`);
        continue;
      }
      if (!valuesWithinTolerance(glsl, item.js)) {
        failures.push(`  ${item.test}\n      GLSL computed ${formatValue(glsl)}, CPU computed ${formatValue(item.js)}`);
      }
      if (!valuesWithinTolerance(wgsl, item.js)) {
        failures.push(`  ${item.test}\n      WGSL computed ${formatValue(wgsl)}, CPU computed ${formatValue(item.js)}`);
      }
    }
  }

  if (failures.length > 0) {
    throw new Error(
      `Evaluated ${runnable.length} recorded programs; ${failures.length} disagreed with the CPU target:\n\n${failures.join("\n")}`,
    );
  }
}

/** How many programs were recorded, and how many opted out of the GPU backends. */
export function recordedEvaluationSummary(): { total: number; cpuOnly: number } {
  return {
    total: recordedEvaluations.length,
    cpuOnly: recordedEvaluations.filter((r) => r.cpuOnly !== undefined).length,
  };
}

/**
 * Whether to skip evaluating on the two shading languages.
 *
 * Evaluation needs a graphics device and validation needs a browser, so both
 * are worth turning off in a mutation run. `RMSL_SKIP_GPU` turns off every
 * layer that needs hardware, and each layer also has a flag of its own for
 * turning off just that one.
 *
 * Skipping is announced, and says how many programs went unchecked. The CPU
 * target is not covered by any of this — it needs nothing, so it always runs.
 */
export const GPU_EVALUATION_SKIPPED = !!process.env.RMSL_SKIP_GPU || !!process.env.RMSL_SKIP_SHADER_EVALUATION;

/**
 * Kept under its former name for the tests that evaluate eagerly, which have to
 * skip outright because they await a GPU in the body of the test itself.
 *
 * The recording path above does not use it: the CPU target needs no hardware,
 * so it runs whatever this says, and only the comparison against the two
 * shading languages waits on a device.
 */
export const EVALUATION_SKIPPED = GPU_EVALUATION_SKIPPED;
