import type { Node } from "../rmsl";
import { compileGlslFn } from "../glsl";
import { compileWgslFn } from "../wgsl";
import { compileJSFn } from "../js";
import { compileWasmRoutine } from "../wasm";
import type { IntegerType } from "./integer-reference";

/**
 * One program of an integer sweep, and every set of arguments it runs with.
 *
 * `build` returns a scalar or a two-component vector of `type`; `want` holds
 * one number per component. A program is compiled once and called for each
 * run, so a sweep over run-time values costs one compile, not one per value.
 */
export type SweepCase = {
  label: string;
  type: IntegerType;
  width: 1 | 2;
  paramTypes: IntegerType[];
  build: (...params: Node<any>[]) => Node<any>;
  runs: { args: number[]; want: number[] }[];
};

/**
 * A run whose result differs from `want`, or a program a backend could not
 * compile or run. `source` is the generated code, where the backend has one
 * the caller may need to inspect.
 */
export type Mismatch = {
  backend: string;
  label: string;
  args: number[];
  want: number[];
  got: number[] | string;
  source?: string;
};

const paramsOf = (c: SweepCase) => c.paramTypes.map((type, i) => ({ name: `a${i}`, type }));
const argsObject = (args: number[]) => ({ params: Object.fromEntries(args.map((a, i) => [`a${i}`, a])) });

function toComponents(value: unknown): number[] {
  if (typeof value === "number") return [value];
  if (Array.isArray(value) || ArrayBuffer.isView(value)) return Array.from(value as ArrayLike<number>);
  if (value && typeof value === "object" && "value" in value) return toComponents((value as { value: unknown }).value);
  return [NaN];
}

function sameComponents(want: number[], got: number[]): boolean {
  return want.length <= got.length && want.every((w, i) => Object.is(w, got[i]));
}

/** The first line of an error's message. A `GPUError` has one but is not an `Error`. */
function message(error: unknown): string {
  const text =
    error && typeof error === "object" && "message" in error
      ? String((error as { message: unknown }).message)
      : String(error);
  return text.split("\n")[0]!;
}

/**
 * Compile each case with `compile` and check every run against it. A compile
 * failure is reported once per case, against its first run.
 */
function sweepCpu(
  backend: string,
  cases: SweepCase[],
  compile: (c: SweepCase) => (args: number[]) => unknown,
): Mismatch[] {
  const mismatches: Mismatch[] = [];
  for (const c of cases) {
    let call: (args: number[]) => unknown;
    try {
      call = compile(c);
    } catch (error) {
      const run = c.runs[0]!;
      mismatches.push({ backend, label: c.label, args: run.args, want: run.want, got: `compile: ${message(error)}` });
      continue;
    }
    for (const run of c.runs) {
      let got: number[] | string;
      try {
        got = toComponents(call(run.args)).slice(0, c.width);
      } catch (error) {
        got = `run: ${message(error)}`;
      }
      if (typeof got === "string" || !sameComponents(run.want, got)) {
        mismatches.push({ backend, label: c.label, args: run.args, want: run.want, got });
      }
    }
  }
  return mismatches;
}

export function sweepJS(cases: SweepCase[]): Mismatch[] {
  const cache = new Map<string, (ctx: unknown) => unknown>();
  return sweepCpu("JS", cases, (c) => {
    const source = compileJSFn(c.build as any, { name: "rmsl_eval", params: paramsOf(c) });
    let fn = cache.get(source);
    if (!fn) {
      fn = new Function(source)() as (ctx: unknown) => unknown;
      cache.set(source, fn);
    }
    const callable = fn;
    return (args) => callable(argsObject(args));
  });
}

export function sweepWASM(cases: SweepCase[]): Mismatch[] {
  return sweepCpu("WASM", cases, (c) => {
    const routine = compileWasmRoutine(c.build as any, { name: "rmsl_eval", params: paramsOf(c) });
    return (args) => routine.run(argsObject(args) as any);
  });
}

// === WGSL ===

const WGSL_SCALAR = { int: "i32", uint: "u32" } as const;

/** Most functions run from one module, and most invocations in one dispatch. */
const WGSL_FUNCTIONS_PER_MODULE = 1000;
const WGSL_INVOCATIONS_PER_DISPATCH = 1 << 20;
/** Argument slots each invocation has in the argument buffer. */
const WGSL_ARG_STRIDE = 8;

type WgslProgram = { body: string; failure?: string; cases: SweepCase[] };

/**
 * A case's function under a placeholder name, so two cases that compile to
 * the same code share one function in the module.
 */
function wgslBody(c: SweepCase): string {
  return compileWgslFn(c.build as any, { name: "rmsl_case", params: paramsOf(c) });
}

/**
 * Validate each program in a module of its own and record why on the ones
 * that fail. One bad function would fail a whole batched module, and a
 * single-function module takes Dawn well under a millisecond.
 */
async function recordWgslFailures(gpu: any, programs: WgslProgram[]): Promise<void> {
  const failures = programs.map((p) => {
    gpu.pushErrorScope("validation");
    gpu.createShaderModule({ code: p.body });
    return gpu.popErrorScope();
  });
  (await Promise.all(failures)).forEach((failure, i) => {
    if (failure) programs[i]!.failure = message(failure);
  });
}

type WgslInvocation = { program: number; c: SweepCase; run: SweepCase["runs"][number] };

async function dispatchWgsl(gpu: any, programs: WgslProgram[], invocations: WgslInvocation[]): Promise<Uint32Array> {
  const functions = programs.map((p, i) => p.body.replace("fn rmsl_case(", `fn c${i}(`));
  const branches = programs.map((p, i) => {
    const c = p.cases[0]!;
    const args = c.paramTypes.map((t, k) => (t === "int" ? `bitcast<i32>(args[base + ${k}u])` : `args[base + ${k}u]`));
    const call = `c${i}(${args.join(", ")})`;
    const store =
      c.width === 1
        ? `out[w * 2u] = bitcast<u32>(r);`
        : `out[w * 2u] = bitcast<u32>(r.x); out[w * 2u + 1u] = bitcast<u32>(r.y);`;
    return `    case ${i}u: { let r = ${call}; ${store} }`;
  });
  const code = `${functions.join("\n")}
@group(0) @binding(0) var<storage, read> programs: array<u32>;
@group(0) @binding(1) var<storage, read> args: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let w = id.x + id.y * 65535u * 64u;
  if (w >= arrayLength(&programs)) { return; }
  let base = w * ${WGSL_ARG_STRIDE}u;
  // An unread binding is dropped from the pipeline's layout, and the bind
  // group naming it would then fail; this keeps it read when no case is.
  _ = args[base];
  switch programs[w] {
${branches.join("\n")}
    default: {}
  }
}`;

  const STORAGE = 0x80,
    COPY_SRC = 0x4,
    COPY_DST = 0x8,
    MAP_READ = 0x1;
  // Everything from here to the submit is in one scope: a failure anywhere in
  // it leaves the output unwritten, which would otherwise read as all zeros.
  gpu.pushErrorScope("validation");
  const module = gpu.createShaderModule({ code });
  const pipeline = gpu.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });

  const programIndex = Uint32Array.from(invocations, (inv) => inv.program);
  const argData = new Uint32Array(invocations.length * WGSL_ARG_STRIDE);
  invocations.forEach((inv, w) => inv.run.args.forEach((a, k) => (argData[w * WGSL_ARG_STRIDE + k] = a >>> 0)));
  const outBytes = invocations.length * 2 * 4;

  const programBuffer = gpu.createBuffer({ size: programIndex.byteLength, usage: STORAGE | COPY_DST });
  const argBuffer = gpu.createBuffer({ size: argData.byteLength, usage: STORAGE | COPY_DST });
  const outBuffer = gpu.createBuffer({ size: outBytes, usage: STORAGE | COPY_SRC });
  const readback = gpu.createBuffer({ size: outBytes, usage: MAP_READ | COPY_DST });
  try {
    gpu.queue.writeBuffer(programBuffer, 0, programIndex);
    gpu.queue.writeBuffer(argBuffer, 0, argData);
    const encoder = gpu.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      gpu.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: programBuffer } },
          { binding: 1, resource: { buffer: argBuffer } },
          { binding: 2, resource: { buffer: outBuffer } },
        ],
      }),
    );
    const groups = Math.ceil(invocations.length / 64);
    pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535));
    pass.end();
    encoder.copyBufferToBuffer(outBuffer, 0, readback, 0, outBytes);
    gpu.queue.submit([encoder.finish()]);
    const failure = await gpu.popErrorScope();
    if (failure) throw new Error(`the sweep's WGSL dispatch failed: ${message(failure)}`);
    await readback.mapAsync(MAP_READ);
    return new Uint32Array(readback.getMappedRange().slice(0));
  } finally {
    readback.destroy?.();
    outBuffer.destroy?.();
    argBuffer.destroy?.();
    programBuffer.destroy?.();
  }
}

/**
 * Run every case on Dawn. Cases are compiled many to a module and run many
 * to a dispatch, one invocation per run: a module per case would spend the
 * sweep's whole budget on pipeline creation.
 */
export async function sweepWGSL(cases: SweepCase[]): Promise<Mismatch[]> {
  const { gpuDevice } = await import("./gpu");
  const gpu = await gpuDevice();
  const mismatches: Mismatch[] = [];

  const programs = new Map<string, WgslProgram>();
  for (const c of cases) {
    let body: string;
    try {
      body = wgslBody(c);
    } catch (error) {
      const run = c.runs[0]!;
      mismatches.push({
        backend: "WGSL",
        label: c.label,
        args: run.args,
        want: run.want,
        got: `compile: ${message(error)}`,
      });
      continue;
    }
    // The body includes the signature, so equal bodies take the same arguments and return the same type.
    const program = programs.get(body);
    if (program) program.cases.push(c);
    else programs.set(body, { body, cases: [c] });
  }

  const all = [...programs.values()];
  for (let start = 0; start < all.length; start += WGSL_FUNCTIONS_PER_MODULE) {
    const chunk = all.slice(start, start + WGSL_FUNCTIONS_PER_MODULE);
    await recordWgslFailures(gpu, chunk);
    const runnable: WgslProgram[] = [];
    for (const p of chunk) {
      if (p.failure === undefined) {
        runnable.push(p);
        continue;
      }
      for (const c of p.cases) {
        const run = c.runs[0]!;
        mismatches.push({
          backend: "WGSL",
          label: c.label,
          args: run.args,
          want: run.want,
          got: `compile: ${p.failure}`,
        });
      }
    }

    const invocations: WgslInvocation[] = runnable.flatMap((p, program) =>
      p.cases.flatMap((c) => c.runs.map((run) => ({ program, c, run }))),
    );
    for (let at = 0; at < invocations.length; at += WGSL_INVOCATIONS_PER_DISPATCH) {
      const batch = invocations.slice(at, at + WGSL_INVOCATIONS_PER_DISPATCH);
      const out = await dispatchWgsl(gpu, runnable, batch);
      batch.forEach(({ program, c, run }, w) => {
        const got = Array.from({ length: c.width }, (_, k) => {
          const bits = out[w * 2 + k]!;
          return c.type === "int" ? bits | 0 : bits;
        });
        if (!sameComponents(run.want, got))
          mismatches.push({
            backend: "WGSL",
            label: c.label,
            args: run.args,
            want: run.want,
            got,
            source: runnable[program]!.body,
          });
      });
    }
  }
  return mismatches;
}

// === GLSL ===

const GLSL_FUNCTIONS_PER_SHADER = 500;

/**
 * Hand every case's GLSL to Chromium's compiler. Only whether it compiles is
 * checked: GLSL leaves overflow, division by zero and large shifts undefined,
 * so there is no result to hold it to — but a precedence mistake that makes
 * the source invalid is still a bug there.
 */
export async function sweepGLSLValidity(cases: SweepCase[]): Promise<Mismatch[]> {
  const { compileGLSLInPage } = await import("./gpu");
  const mismatches: Mismatch[] = [];
  const programs = new Map<string, SweepCase[]>();
  for (const c of cases) {
    let body: string;
    try {
      body = compileGlslFn(c.build as any, { name: "rmsl_case", params: paramsOf(c) });
    } catch (error) {
      const run = c.runs[0]!;
      mismatches.push({
        backend: "GLSL",
        label: c.label,
        args: run.args,
        want: run.want,
        got: `compile: ${message(error)}`,
      });
      continue;
    }
    const list = programs.get(body);
    if (list) list.push(c);
    else programs.set(body, [c]);
  }

  const shader = (bodies: string[]) =>
    `#version 300 es
precision highp float;
precision highp int;
${bodies.map((b, i) => b.replace(/\brmsl_case\(/, `c${i}(`)).join("\n")}
out vec4 color;
void main() { color = vec4(0.0); }`;

  const report = (cases: SweepCase[], failure: string) => {
    for (const c of cases) {
      const run = c.runs[0]!;
      mismatches.push({ backend: "GLSL", label: c.label, args: run.args, want: run.want, got: `compile: ${failure}` });
    }
  };

  // Compiled many to a shader; a shader that fails has its functions compiled
  // one to a shader, in one round trip, to find which of them fail.
  const all = [...programs.entries()];
  for (let start = 0; start < all.length; start += GLSL_FUNCTIONS_PER_SHADER) {
    const chunk = all.slice(start, start + GLSL_FUNCTIONS_PER_SHADER);
    const [failure] = await compileGLSLInPage([{ src: shader(chunk.map(([body]) => body)), stage: "fragment" }]);
    if (failure === null) continue;
    const failures = await compileGLSLInPage(chunk.map(([body]) => ({ src: shader([body]), stage: "fragment" })));
    failures.forEach((f, i) => f !== null && report(chunk[i]![1], f));
  }
  return mismatches;
}

/** The first `limit` mismatches, one per line, for a failure message a person can read. */
export function describeMismatches(mismatches: Mismatch[], limit = 40): string {
  const lines = mismatches
    .slice(0, limit)
    .map(
      (m) =>
        `${m.backend}: ${m.label} args=[${m.args.join(", ")}] want=[${m.want.join(", ")}] got=${
          typeof m.got === "string" ? m.got : `[${m.got.join(", ")}]`
        }`,
    );
  if (mismatches.length > limit) lines.push(`… and ${mismatches.length - limit} more`);
  return lines.join("\n");
}
