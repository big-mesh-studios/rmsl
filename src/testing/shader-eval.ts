import { expect } from "vitest";
import { var_, type Node } from "../rmsl";
import { compileGlslFn } from "../glsl";
import { compileWgslFn } from "../wgsl";
import { compileJSFn } from "../js";
import { compileWasmRoutine } from "../wasm";
import { MATRIX_DIMENSIONS, TYPE_WIDTH } from "../core";
import { isVector } from "../backends/cpu";

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

/**
 * Compile, run and read back the GLSL backend's result — a scalar, vector
 * or matrix. Renders to an RGBA32F texture and reads the red channel.
 */
export async function evaluateGLSL(build: Build, args: number[] = []): Promise<number | number[]> {
  const fn = compileGlslFn(build, { name: "rmsl_eval", params: params(args.length) });
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

/**
 * Compile, run and read back the WGSL backend's result — a scalar, vector
 * or matrix. Dispatches a compute shader and reads a storage buffer.
 */
export async function evaluateWGSL(build: Build, args: number[] = []): Promise<number | number[]> {
  const fn = compileWgslFn(build, { name: "rmsl_eval", params: params(args.length) });
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
export function evaluateJS(build: Build, args: number[] = []): number | Float64Array {
  const fn = compileJSFn(build, { name: "rmsl_eval", params: params(args.length) });
  const callable = new Function(fn)() as (ctx: { params: Record<string, number> }) => number | Float64Array;
  const ctx = { params: Object.fromEntries(args.map((a, i) => [`a${i}`, a])) };
  return callable(ctx);
}

/** What an evaluation gives: a scalar, or the components of a vector or matrix, as a plain or a typed array. */
export type EvalValue = number | number[] | Float64Array;

/**
 * Run an expression on the WASM backend — in-process, no GPU, no browser,
 * same as `evaluateJS`.
 *
 * This backend's `float` is f64, matching `compileJSRoutine`'s plain JS-number
 * arithmetic bit for bit (`ROADMAP.md`, "`float` is f64") — including the
 * transcendental functions, which both backends call through the literal
 * same `Math` object. So unlike the GLSL/WGSL comparison, nothing here
 * should ever need `floatTolerance`: a real difference is a bug, not
 * rounding.
 */
export function evaluateWASM(build: Build, args: number[] = []): number | number[] {
  const fn = compileWasmRoutine(build, { name: "rmsl_eval", params: params(args.length) });
  const ctx = { params: Object.fromEntries(args.map((a, i) => [`a${i}`, a])) };
  const result = fn(ctx);
  // A scalar root returns the number, and an aggregate root the array.
  return result as number | number[];
}

/** A scalar integer type, the only kind the integer evaluators below take and return. */
export type IntegerType = "int" | "uint";

type IntegerBuild = (...args: Node<any>[]) => Node<"int"> | Node<"uint">;

function integerParams(type: IntegerType, count: number) {
  return Array.from({ length: count }, (_, i) => ({ name: `a${i}`, type }));
}

/**
 * Run an integer expression on the WGSL backend and read the result back as
 * the integer it is, not through an `f32`.
 *
 * The arguments reach the shader through a storage buffer rather than as
 * literals: WGSL rejects a constant expression that divides by zero or
 * overflows at shader-creation time, so a literal argument would test the
 * compiler's constant folding, not the runtime arithmetic.
 */
export async function evaluateIntegerWGSL(build: IntegerBuild, type: IntegerType, args: number[]): Promise<number> {
  const fn = compileWgslFn(build, { name: "rmsl_eval", params: integerParams(type, args.length) });
  const scalar = type === "int" ? "i32" : "u32";
  const call = `rmsl_eval(${args.map((_, i) => `args[${i}]`).join(", ")})`;
  // An unused binding would be stripped from the pipeline's layout, so it is only declared when read.
  const argDeclaration = args.length > 0 ? `@group(0) @binding(1) var<storage, read> args: array<${scalar}>;` : "";
  const code = `${fn}
@group(0) @binding(0) var<storage, read_write> result: array<${scalar}>;
${argDeclaration}
@compute @workgroup_size(1)
fn main() {
  result[0] = ${call};
}`;

  const { gpuDevice } = await import("./gpu");
  const gpu = await gpuDevice();
  gpu.pushErrorScope("validation");
  const module = gpu.createShaderModule({ code });
  const pipeline = gpu.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const compileFailure = gpu.popErrorScope();

  const STORAGE = 0x80,
    COPY_SRC = 0x4,
    MAP_READ = 0x1,
    COPY_DST = 0x8;
  const input = type === "int" ? Int32Array.from(args) : Uint32Array.from(args);
  const result = gpu.createBuffer({ size: 4, usage: STORAGE | COPY_SRC });
  const argBuffer = gpu.createBuffer({ size: Math.max(4, input.byteLength), usage: STORAGE | COPY_DST });
  const readback = gpu.createBuffer({ size: 4, usage: MAP_READ | COPY_DST });
  try {
    gpu.queue.writeBuffer(argBuffer, 0, input);
    const encoder = gpu.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      gpu.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: result } },
          ...(args.length > 0 ? [{ binding: 1, resource: { buffer: argBuffer } }] : []),
        ],
      }),
    );
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(result, 0, readback, 0, 4);
    gpu.queue.submit([encoder.finish()]);
    const [failure] = await Promise.all([compileFailure, readback.mapAsync(MAP_READ)]);
    if (failure) throw new Error(`WGSL shader failed to compile: ${failure.message}`);
    const bytes = readback.getMappedRange().slice(0);
    return type === "int" ? new Int32Array(bytes)[0]! : new Uint32Array(bytes)[0]!;
  } finally {
    readback.destroy?.();
    argBuffer.destroy?.();
    result.destroy?.();
  }
}

/**
 * Run an integer expression on the GLSL backend and read the result back as
 * the integer it is, from an integer render target rather than through a float.
 *
 * The arguments reach the shader as uniforms rather than literals, so the
 * driver computes the operation at run time instead of folding it.
 */
export async function evaluateIntegerGLSL(build: IntegerBuild, type: IntegerType, args: number[]): Promise<number> {
  const fn = compileGlslFn(build, { name: "rmsl_eval", params: integerParams(type, args.length) });
  const scalar = type === "int" ? "int" : "uint";
  const uniforms = args.map((_, i) => `uniform ${scalar} rmsl_arg${i};`).join("\n");
  const call = `rmsl_eval(${args.map((_, i) => `rmsl_arg${i}`).join(", ")})`;
  const fragment = `#version 300 es
precision highp float;
precision highp int;
${uniforms}
${fn}
layout(location=0) out highp ${type === "int" ? "ivec4" : "uvec4"} result;
void main() {
  result = ${type === "int" ? "ivec4" : "uvec4"}(${call}, 0, 0, 0);
}`;

  const { gpuPage } = await import("./gpu");
  const page = await gpuPage();
  const out = await page.evaluate(
    ({ fragment, args, signed }: { fragment: string; args: number[]; signed: boolean }) => {
      // A fresh context per call, as the float evaluator makes one.
      const gl = document.createElement("canvas").getContext("webgl2")!;
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        signed ? gl.RGBA32I : gl.RGBA32UI,
        1,
        1,
        0,
        gl.RGBA_INTEGER,
        signed ? gl.INT : gl.UNSIGNED_INT,
        null,
      );
      const framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      const vertices = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vertices);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);

      // Compiled inline rather than through a helper: the bundler renames
      // functions and injects a `__name` shim that does not exist in the page.
      const program = gl.createProgram()!;
      for (const [src, kind] of [
        [`#version 300 es\nin vec2 p; void main(){ gl_Position = vec4(p,0.,1.); }`, gl.VERTEX_SHADER],
        [fragment, gl.FRAGMENT_SHADER],
      ] as [string, number][]) {
        const shader = gl.createShader(kind)!;
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
      args.forEach((a, i) => {
        const location = gl.getUniformLocation(program, `rmsl_arg${i}`);
        if (signed) gl.uniform1i(location, a);
        else gl.uniform1ui(location, a >>> 0);
      });

      const location = gl.getAttribLocation(program, "p");
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0);
      gl.viewport(0, 0, 1, 1);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      const out = signed ? new Int32Array(4) : new Uint32Array(4);
      gl.readPixels(0, 0, 1, 1, gl.RGBA_INTEGER, signed ? gl.INT : gl.UNSIGNED_INT, out);
      return out[0]!;
    },
    { fragment, args, signed: type === "int" },
  );
  return out;
}

/** Run an integer expression on the JS backend. */
export function evaluateIntegerJS(build: IntegerBuild, type: IntegerType, args: number[]): number {
  const fn = compileJSFn(build, { name: "rmsl_eval", params: integerParams(type, args.length) });
  const callable = new Function(fn)() as (ctx: { params: Record<string, number> }) => number;
  return callable({ params: Object.fromEntries(args.map((a, i) => [`a${i}`, a])) });
}

/** Run an integer expression on the WASM backend. */
export function evaluateIntegerWASM(build: IntegerBuild, type: IntegerType, args: number[]): number {
  const fn = compileWasmRoutine(build, { name: "rmsl_eval", params: integerParams(type, args.length) });
  return fn({ params: Object.fromEntries(args.map((a, i) => [`a${i}`, a])) }) as number;
}

/**
 * Whether a `compileWasmRoutine`/`compileWasmFn` failure means "not supported by
 * this backend yet" rather than a real bug.
 *
 * Every deliberate "can't compile this (yet)" throw in `wasm.ts` — for
 * an unsupported node type, an integer cube-map sampler, and so on — is
 * constructed with this exact prefix (confirmed:
 * every `throw new Error(...)` in that file uses it, whether the case is a
 * known coverage gap or an internal-misuse check). A genuine WASM engine
 * trap (`WebAssembly.RuntimeError`, thrown by the VM itself when a compiled
 * module actually executes a trapping instruction) or a `CompileError`/
 * `LinkError` (malformed bytecode — a real codegen bug) never carries this
 * prefix, so this check only ever recognizes "doesn't compile", never
 * "crashed" or "computed the wrong answer".
 */
/**
 * The tests whose recorded programs the WASM target does not compile yet, each
 * with the issue that tracks it. Keyed by the test's full name, as
 * `KNOWN_INVALID` is, so a refusal is a listed gap rather than a quiet skip.
 */
export const KNOWN_WASM_REFUSALS: Record<string, string> = {
  "each leaf on every target it claims > compares a vector against a scalar on every target": "all: #219",
  "each leaf on every target it claims > reduces a boolean vector with all and any on every target": "all, any: #219",
  "each leaf on every target it claims > computes with int, uint, bool and integer vector literals on every target":
    "any: #219",
  "each leaf on every target it claims > transposes a matrix that is not square on every target": "transpose: #220",
  "each leaf on every target it claims > reads an assignment's target as it was before the assignment on every target":
    "inverse: #65, transpose: #220",
  "each leaf on every target it claims > computes the geometric functions on every target": "faceForward: #130",
  "each leaf on every target it claims > negates a boolean vector component by component on every target":
    "not of a vector: #130",
  "each leaf on every target it claims > rounds a value halfway between two integers to the even one on every target":
    "round of a vector: #130",
  "units of the core > widens the scalar argument of pow to the vector on every target": "pow of a vector: #130",
};

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
): Promise<{ glsl: number | number[]; wgsl: number | number[]; js: EvalValue }> {
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
  js: EvalValue;
  /** Set when the case deliberately does not run on the GPU backends. */
  cpuOnly?: string;
}

function asArray(v: EvalValue): ArrayLike<number> {
  return isVector(v) ? v : [v];
}

function formatValue(v: EvalValue): string {
  return isVector(v) ? `[${Array.from(v).join(", ")}]` : String(v);
}

/** The same bits, elementwise for an aggregate: a NaN equals a NaN, and `-0` differs from `0`. */
function valuesExactlyEqual(a: EvalValue, b: EvalValue): boolean {
  const av = asArray(a);
  const bv = asArray(b);
  if (av.length !== bv.length) return false;
  for (let i = 0; i < av.length; i++) if (!Object.is(av[i], bv[i])) return false;
  return true;
}

/**
 * Within `floatTolerance` of each other, elementwise for an aggregate. Two
 * equal values agree whatever they are, two infinities of one sign and two
 * NaNs included, where their distance is NaN.
 */
function valuesWithinTolerance(a: EvalValue, b: EvalValue): boolean {
  const av = asArray(a);
  const bv = asArray(b);
  if (av.length !== bv.length) return false;
  for (let i = 0; i < av.length; i++) {
    const x = av[i]!;
    const y = bv[i]!;
    if (x === y || (Number.isNaN(x) && Number.isNaN(y))) continue;
    if (!(Math.abs(x - y) < floatTolerance(x))) return false;
  }
  return true;
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
export function evaluateRecording(build: Build, args: number[] = [], cpuOnly?: CpuOnlyReason): EvalValue {
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

/** How many of `recordedEvaluations` have been compared, from the start: those recorded since are the current test's. */
let compared = 0;

/**
 * Compare the programs recorded since the last comparison, which are the
 * current test's, on every target. Called from a file's `afterEach`, so a
 * disagreement fails the test whose program caused it, and a tool that maps
 * tests to what they check sees the comparison in that test.
 */
export async function assertEvaluationsOfTheTestAgree(): Promise<void> {
  const items = recordedEvaluations.slice(compared);
  compared = recordedEvaluations.length;
  await compareEvaluations(items);
}

/**
 * Compare every recorded program not compared yet on every target, and refuse
 * a file that recorded none. Called from a file's `afterAll`, after its
 * `afterEach` has compared each test's own programs.
 *
 * Compared against the CPU result rather than against a separately written
 * expectation, because the caller already pinned that result with an assertion
 * of its own. So a mismatch here means the backends disagree about a program
 * whose answer is already known to be right.
 */
export async function assertRecordedEvaluationsAgree(): Promise<void> {
  // Recording nothing is not the same as everything agreeing. A file that
  // stopped going through the shared helper would otherwise finish green having
  // checked one backend of three, which is the arrangement this replaced.
  if (recordedEvaluations.length === 0) {
    throw new Error(
      `Evaluated no programs at all. Either the run was filtered down to tests that evaluate nothing, or a test file stopped calling the shared evaluation helper in src/testing/shader-eval.ts. Set RMSL_SKIP_SHADER_EVALUATION=1 if skipping evaluation is what you meant.`,
    );
  }
  const items = recordedEvaluations.slice(compared);
  compared = recordedEvaluations.length;
  await compareEvaluations(items);
  if (GPU_EVALUATION_SKIPPED) {
    const runnable = recordedEvaluations.filter((r) => r.cpuOnly === undefined).length;
    process.stderr.write(
      `\n[shader-eval] SKIPPED — ${runnable} programs ran on the CPU and WASM targets only; neither shading language was evaluated.\n`,
    );
  }
}

/** Compare `items` on every target, and throw naming each one a target disagrees about. */
async function compareEvaluations(items: readonly RecordedEvaluation[]): Promise<void> {
  const runnable = items.filter((r) => r.cpuOnly === undefined);
  const failures: string[] = [];

  // WASM needs neither a browser nor a graphics device, so — unlike GLSL/WGSL
  // below — it always runs, even under RMSL_SKIP_GPU/RMSL_SKIP_SHADER_EVALUATION
  // (those exist specifically to skip hardware-dependent work). A program it
  // refuses fails the run, unless KNOWN_WASM_REFUSALS names the issue that
  // tracks it; a listed program that compiles fails the run too.
  const refused = new Set<string>();
  for (const item of runnable) {
    try {
      const wasm = evaluateWASM(item.build, item.args);
      // Exact equality, not floatTolerance — see `evaluateWASM`'s doc comment.
      if (!valuesExactlyEqual(wasm, item.js)) {
        failures.push(`  ${item.test}\n      WASM computed ${formatValue(wasm)}, CPU computed ${formatValue(item.js)}`);
      }
    } catch (error) {
      if (!isWasmUnsupported(error)) throw error;
      refused.add(item.test);
      if (KNOWN_WASM_REFUSALS[item.test] === undefined) {
        failures.push(`  ${item.test}\n      WASM refused it — ${(error as Error).message}`);
      }
    }
  }
  for (const test of new Set(runnable.map((item) => item.test))) {
    if (KNOWN_WASM_REFUSALS[test] !== undefined && !refused.has(test)) {
      failures.push(`  ${test}\n      WASM compiles every program of it now — delete it from KNOWN_WASM_REFUSALS`);
    }
  }

  if (!GPU_EVALUATION_SKIPPED && runnable.length > 0) {
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
