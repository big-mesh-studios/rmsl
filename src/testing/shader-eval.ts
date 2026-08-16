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
import {
  compileGLSLFn, compileWGSLFn, compileJSFn, type Node,
} from "../rmsl";

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

type Build = (...args: Node<"float">[]) => Node<"float">;

function params(count: number) {
  return Array.from({ length: count }, (_, i) => ({ name: `a${i}`, type: "float" as const }));
}

function callExpr(args: number[]) {
  // Emitted as literals. Whether the driver folds them is immaterial: if the
  // wrong operator was emitted the answer is wrong either way.
  return `rmsl_eval(${args.map(a => (Number.isInteger(a) ? a.toFixed(1) : String(a))).join(", ")})`;
}

/** Compile, run and read back one float from the GLSL backend. */
export async function evaluateGLSL(build: Build, args: number[] = []): Promise<number> {
  const fn = compileGLSLFn(build, { name: "rmsl_eval", params: params(args.length) });
  const source = `#version 300 es
precision highp float;
${fn}
layout(location=0) out vec4 result;
void main() { result = vec4(${callExpr(args)}, 0.0, 0.0, 1.0); }`;

  const { gpuPage } = await import("./gpu");
  const page = await gpuPage();
  {
    return await page.evaluate((fragment: string) => {
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
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 1, 1, 0, gl.RGBA, gl.FLOAT, null);
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

      gl.viewport(0, 0, 1, 1);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      const out = new Float32Array(4);
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, out);

      return out[0];
    }, source);
  }
}

/** Compile, run and read back one float from the WGSL backend. */
export async function evaluateWGSL(build: Build, args: number[] = []): Promise<number> {
  const fn = compileWGSLFn(build, { name: "rmsl_eval", params: params(args.length) });
  return runWGSL(`${fn}
@group(0) @binding(0) var<storage, read_write> result: array<f32>;
@compute @workgroup_size(1)
fn main() { result[0] = ${callExpr(args)}; }`);
}

/**
 * Run a compute shader that writes one float to `result[0]`, and read it back.
 *
 * Separate from `evaluateWGSL` so the execution path can be exercised with
 * source the compiler would never produce, which is the only way to check that
 * a shader failing to compile is actually reported.
 */
export async function runWGSL(code: string): Promise<number> {
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

  const STORAGE = 0x80, COPY_SRC = 0x4, MAP_READ = 0x1, COPY_DST = 0x8;
  const storage = gpu.createBuffer({ size: 4, usage: STORAGE | COPY_SRC });
  const readback = gpu.createBuffer({ size: 4, usage: MAP_READ | COPY_DST });
  try {
    const encoder = gpu.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, gpu.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: storage } }],
    }));
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(storage, 0, readback, 0, 4);
    gpu.queue.submit([encoder.finish()]);
    const [failure] = await Promise.all([compileFailure, readback.mapAsync(MAP_READ)]);
    if (failure) {
      const detail = failure.message.split("\n").find((l: string) => l.includes("error:"))
        ?? failure.message.split("\n")[0];
      throw new Error(`WGSL shader failed to compile: ${detail.trim()}`);
    }
    return new Float32Array(readback.getMappedRange())[0];
  } finally {
    readback.destroy?.();
    storage.destroy?.();
  }
}

/**
 * Run an expression on the JS backend — in-process, no GPU, no browser.
 *
 * The result is exact JS f64 rather than the f32 the GPU backends return, so
 * it is the natural arbiter when the two GPUs disagree: whatever the shaders
 * compute, the CPU must agree with the caller's arithmetic.
 */
export function evaluateJS(build: Build, args: number[] = []): number {
  const fn = compileJSFn(build, { name: "rmsl_eval", params: params(args.length) });
  const callable = new Function(fn)() as (ctx: { params: Record<string, number> }) => number;
  const ctx = { params: Object.fromEntries(args.map((a, i) => [`a${i}`, a])) };
  const value = callable(ctx);
  if (typeof value === "number") return value;
  // A vector result: the first component, the same channel the GPU paths read.
  if (Array.isArray(value)) return value[0] as number;
  return value as unknown as number;
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
): Promise<{ glsl: number; wgsl: number }> {
  const [glsl, wgsl] = await Promise.all([
    evaluateGLSL(build, args),
    evaluateWGSL(build, args),
  ]);
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
): Promise<{ glsl: number; wgsl: number; js: number }> {
  const [glsl, wgsl] = await Promise.all([
    evaluateGLSL(build, args),
    evaluateWGSL(build, args),
  ]);
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
  js: number;
  /** Set when the case deliberately does not run on the GPU backends. */
  cpuOnly?: string;
}

const recordedEvaluations: RecordedEvaluation[] = [];

/**
 * Why a case runs on the CPU target alone.
 *
 * Taking a reason rather than a flag keeps the exclusions listable: every case
 * that opted out says here why, so the set can be read and argued with rather
 * than growing quietly whenever a case is inconvenient.
 */
export type CpuOnlyReason =
  | "derivatives"
  | "reentrant"
  | "texture"
  | "js-only-api"
  | "exceeds-float32";

/**
 * Evaluate on the CPU target and record the program for the GPU backends.
 *
 * Returns the CPU result, which is exact f64 and so is the value a caller's
 * assertion pins. The GPU backends are compared against that same value later,
 * which is what makes one assertion cover three backends.
 */
export function evaluateRecording(
  build: Build,
  args: number[] = [],
  cpuOnly?: CpuOnlyReason,
): number {
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
  const runnable = recordedEvaluations.filter(r => r.cpuOnly === undefined);
  if (GPU_EVALUATION_SKIPPED) {
    process.stderr.write(
      `\n[shader-eval] SKIPPED — ${runnable.length} programs ran on the CPU target only; neither shading language was evaluated.\n`,
    );
    return;
  }
  // Recording nothing is not the same as everything agreeing. A file that
  // stopped going through the shared helper would otherwise finish green having
  // checked one backend of three, which is the arrangement this replaced.
  if (recordedEvaluations.length === 0) {
    throw new Error(
      `Evaluated no programs at all. Either the run was filtered down to tests that evaluate nothing, or a test file stopped calling the shared evaluation helper in src/testing/shader-eval.ts. Set RMSL_SKIP_SHADER_EVALUATION=1 if skipping evaluation is what you meant.`,
    );
  }
  if (runnable.length === 0) return;

  const failures: string[] = [];
  for (const item of runnable) {
    const tolerance = floatTolerance(item.js);
    let glsl: number;
    let wgsl: number;
    try {
      [glsl, wgsl] = await Promise.all([
        evaluateGLSL(item.build, item.args),
        evaluateWGSL(item.build, item.args),
      ]);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      failures.push(`  ${item.test}\n      did not evaluate — ${message}`);
      continue;
    }
    if (Math.abs(glsl - item.js) >= tolerance) {
      failures.push(`  ${item.test}\n      GLSL computed ${glsl}, CPU computed ${item.js}`);
    }
    if (Math.abs(wgsl - item.js) >= tolerance) {
      failures.push(`  ${item.test}\n      WGSL computed ${wgsl}, CPU computed ${item.js}`);
    }
  }

  if (failures.length > 0) {
    throw new Error(
      `Evaluated ${runnable.length} recorded programs on both shading languages; ${failures.length} disagreed with the CPU target:\n\n${failures.join("\n")}`,
    );
  }
}

/** How many programs were recorded, and how many opted out of the GPU backends. */
export function recordedEvaluationSummary(): { total: number; cpuOnly: number } {
  return {
    total: recordedEvaluations.length,
    cpuOnly: recordedEvaluations.filter(r => r.cpuOnly !== undefined).length,
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
export const GPU_EVALUATION_SKIPPED =
  !!process.env.RMSL_SKIP_GPU || !!process.env.RMSL_SKIP_SHADER_EVALUATION;

/**
 * Kept under its former name for the tests that evaluate eagerly, which have to
 * skip outright because they await a GPU in the body of the test itself.
 *
 * The recording path above does not use it: the CPU target needs no hardware,
 * so it runs whatever this says, and only the comparison against the two
 * shading languages waits on a device.
 */
export const EVALUATION_SKIPPED = GPU_EVALUATION_SKIPPED;
