/// <reference types="vite/client" />
import { afterAll, afterEach, describe, expect, it, vi } from "vitest";
import { build } from "esbuild";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Fn, bool, outputStruct, uniform, varying, vec2, vec3, vec4, builtinPosition, type Node } from "../rmsl";
import { compileGlsl } from "../glsl";
import { compileWgsl } from "../wgsl";
import { fxaa, gaussianBlur, getGaussianCoefficients, rgbShift, transition } from "../effects";
import { fromProgram, render, runner } from "../test";
import { compileWat, precompileJS, precompileShaders, precompileWasm } from "../vite/vite";
import {
  assertEvaluationsOfTheTestAgree,
  assertRecordedEvaluationsAgree,
  closeEvaluators,
  evaluateRecording,
  recordedEvaluationSummary,
} from "../testing/shader-eval";
import { assertRecordedShadersValid, recordingGLSL, recordingWGSL } from "../testing/shader-validity";
import { GPU_ENABLED } from "../testing/gpu";
import { sweepWGSL } from "../testing/integer-sweep";
import cpuFnsSource from "../vite/fixtures/cpu-fns.ts?raw";
import wasmFnsSource from "../vite/fixtures/wasm-fns.ts?raw";

// Each test's programs are compared after it, so a disagreement fails the test that made the program.
afterEach(assertEvaluationsOfTheTestAgree, 120_000);

afterAll(async () => {
  await assertRecordedShadersValid();
  await assertRecordedEvaluationsAgree();
  await closeEvaluators();
}, 120_000);

const FIXTURES = new URL("../vite/fixtures/", import.meta.url).pathname;

type TransformResult = { code: string; map: null } | null;
type Transform = { transform: { call(context: unknown, code: string, id: string): Promise<TransformResult> } };
type Load = { load(id: string): Promise<string | null> };

const CPU_FNS_PATH = new URL("../vite/fixtures/cpu-fns.ts", import.meta.url).pathname;
const WASM_FNS_PATH = new URL("../vite/fixtures/wasm-fns.ts", import.meta.url).pathname;
const RASTERIZER_WAT = new URL("../backends/wasm/rasterizer.wat", import.meta.url).pathname;

/** A Rollup plugin context with just the `emitFile` precompileWasm calls. */
function pluginContext() {
  const emitted: { source: Uint8Array }[] = [];
  return {
    context: { emitFile: (asset: { source: Uint8Array }) => `ref${emitted.push(asset) - 1}` },
    emitted,
  };
}

/** The uniform declarations of a compiled GLSL shader. */
const glslUniforms = (code: string) => code.split("\n").filter((line) => line.startsWith("uniform "));

/** The fields of the uniform block of a compiled WGSL shader. */
const wgslUniformFields = (code: string) =>
  (code.split("struct _RmslUniforms {")[1] ?? "").split("};")[0]!.trim().split("\n").filter(Boolean);

/**
 * The evaluation harness and the rmsl it compiles with, loaded afresh so what
 * a test records stays out of every other test. `skipGpu` leaves the CPU and
 * WASM targets alone in the comparison.
 */
async function freshHarness(skipGpu: boolean) {
  if (skipGpu) vi.stubEnv("RMSL_SKIP_SHADER_EVALUATION", "1");
  vi.resetModules();
  const harness = await import("../testing/shader-eval");
  const rmsl = await import("../rmsl");
  const gpu = await import("../testing/gpu");
  return { harness, rmsl, gpu };
}

afterEach(() => {
  vi.unstubAllEnvs();
  vi.doUnmock("playwright");
  vi.doUnmock("../wasm");
});

describe("the harness checks what it recorded", () => {
  /**
   * @canon spec-evaluation-counts-equal-results-as-agreeing
   */
  it("passes a program that gives NaN on both CPU targets", async () => {
    const { harness } = await freshHarness(true);
    harness.evaluateRecording((a) => a.div(a), [0]);
    await expect(harness.assertRecordedEvaluationsAgree()).resolves.toBeUndefined();
  });

  /**
   * @canon spec-wasm-and-js-give-the-same-float-bits
   */
  it("reports a WASM zero whose sign differs from the JS one", async () => {
    const { harness } = await freshHarness(true);
    // The build reads the sign when a target compiles it, so JS and WASM
    // compile programs that differ only in the sign of their zero.
    let sign = -1;
    const build = (a: Node<"float">) => a.mul(sign);
    expect(Object.is(harness.evaluateRecording(build, [0]), -0)).toBe(true);
    sign = 1;
    expect(Object.is(harness.evaluateWASM(build, [0]), 0)).toBe(true);
    await expect(harness.assertRecordedEvaluationsAgree()).rejects.toThrow(/WASM computed 0, CPU computed 0/);
  });

  /**
   * @canon spec-evaluation-counts-equal-results-as-agreeing
   */
  it.skipIf(!GPU_ENABLED)(
    "passes a program that gives the same infinity on every target",
    async () => {
      const { harness, rmsl, gpu } = await freshHarness(false);
      try {
        harness.evaluateRecording((a) => rmsl.float(1).div(a), [0]);
        await expect(harness.assertRecordedEvaluationsAgree()).resolves.toBeUndefined();
      } finally {
        await gpu.releaseGpu();
      }
    },
    120_000,
  );

  /**
   * @canon spec-a-test-is-held-to-its-own-programs
   */
  it("compares the programs recorded since the last comparison, and only those", async () => {
    const { harness } = await freshHarness(true);
    // The build reads the sign when a target compiles it, so JS and WASM compile programs that disagree.
    let sign = -1;
    harness.evaluateRecording((a) => a.mul(sign), [0]);
    sign = 1;
    await expect(harness.assertEvaluationsOfTheTestAgree()).rejects.toThrow(/WASM computed 0, CPU computed 0/);
    harness.evaluateRecording((a) => a.add(1), [3]);
    await expect(harness.assertEvaluationsOfTheTestAgree()).resolves.toBeUndefined();
    await expect(harness.assertRecordedEvaluationsAgree()).resolves.toBeUndefined();
  });

  /**
   * @canon spec-a-program-wasm-refuses-names-its-issue
   */
  it("fails a run in which WASM refused a recorded program", async () => {
    // A WASM compiler that refuses every program stands in for a gap in it.
    vi.doMock("../wasm", async (original) => ({
      ...(await original<typeof import("../wasm")>()),
      compileWasmRoutine: () => {
        throw new Error('[RMSL] compileWasmFn: unsupported node type in expression position: "stand-in"');
      },
    }));
    const { harness } = await freshHarness(true);
    const build = (a: Node<"float">) => a.add(1);
    expect(() => harness.evaluateWASM(build, [3])).toThrow(/compileWasmFn/);
    harness.evaluateRecording(build, [3]);
    await expect(harness.assertRecordedEvaluationsAgree()).rejects.toThrow(/WASM/);
  });

  /**
   * @canon spec-a-program-wasm-refuses-names-its-issue
   */
  it("fails a run in which a program listed as refused by WASM compiles", async () => {
    const { harness } = await freshHarness(true);
    harness.KNOWN_WASM_REFUSALS[expect.getState().currentTestName!] = "a stand-in issue";
    harness.evaluateRecording((a) => a.add(1), [3]);
    await expect(harness.assertRecordedEvaluationsAgree()).rejects.toThrow(/delete it from KNOWN_WASM_REFUSALS/);
  });

  /**
   * @canon spec-the-integer-sweep-tells-right-from-wrong
   */
  it.skipIf(!GPU_ENABLED)("passes a right case of nine arguments on WGSL", async () => {
    const mismatches = await sweepWGSL([
      {
        label: "nine arguments",
        type: "int",
        width: 1,
        paramTypes: Array(9).fill("int"),
        build: (...args: Node<"int">[]) => args.reduce((sum, a) => sum.add(a)),
        runs: [
          { args: [0, 0, 0, 0, 0, 0, 0, 0, 1], want: [1] },
          { args: [10, 0, 0, 0, 0, 0, 0, 0, 2], want: [12] },
        ],
      },
    ]);
    expect(mismatches).toEqual([]);
  });

  /**
   * @canon spec-releasing-the-harness-never-throws
   */
  it("releases the harness's resources after a browser failed to launch", async () => {
    vi.doMock("playwright", () => ({ chromium: { launch: () => Promise.reject(new Error("no browser")) } }));
    vi.resetModules();
    const gpu = await import("../testing/gpu");
    await expect(gpu.gpuPage()).rejects.toThrow("no browser");
    await expect(gpu.releaseGpu()).resolves.toBeUndefined();
  });

  /**
   * Runs before any test of this file records a program.
   *
   * @canon spec-evaluation-fails-when-it-recorded-nothing
   */
  it("fails the replay when nothing was recorded", async () => {
    expect(recordedEvaluationSummary().total).toBe(0);
    await expect(assertRecordedEvaluationsAgree()).rejects.toThrow(/Evaluated no programs at all/);
  });

  /**
   * @canon spec-evaluate-recording-returns-the-cpu-result-at-once
   */
  it("returns the CPU result at once and keeps the program for the replay", () => {
    const before = recordedEvaluationSummary().total;
    const result = evaluateRecording((a) => a.mul(3), [2]);
    expect(result).toBe(6);
    expect(recordedEvaluationSummary().total).toBe(before + 1);
  });

  /**
   * @canon spec-a-program-kept-off-the-gpu-names-its-reason
   */
  it("keeps a program off the GPU targets only under a named reason", () => {
    const before = recordedEvaluationSummary().cpuOnly;
    expect(evaluateRecording((a) => a.add(1), [1], "js-only-api")).toBe(2);
    expect(recordedEvaluationSummary().cpuOnly).toBe(before + 1);
    // @ts-expect-error A reason outside the list does not type-check.
    const unlisted: Parameters<typeof evaluateRecording>[2] = "slow";
    expect(unlisted).toBe("slow");
  });
});

describe("effects", () => {
  /**
   * @canon spec-a-single-pass-effect-gives-a-colour-node
   */
  it("cross-fades without a mix texture when useTexture is 0", () => {
    expect(() => transition(uniform("sampler2D"), uniform("sampler2D"), null, 0.5, 0.1, 0)).not.toThrow();
  });

  /**
   * @canon spec-a-pass-keys-an-input-by-the-pass-that-makes-it
   */
  it("keys the vertical pass's input by the horizontal pass", () => {
    const graph = gaussianBlur(uniform("sampler2D"));
    expect(Object.keys(graph.passes[1]!.inputs)).toEqual(["gaussianBlur.horizontal"]);
  });

  /**
   * FXAA samples where its control flow depends on what it sampled before, so
   * it reads the base level explicitly, which WGSL allows there. The recorded
   * shaders are compiled by the GPU compilers after the file.
   *
   * @canon spec-a-single-pass-effect-gives-a-colour-node
   */
  it("compiles fxaa on GLSL and WGSL", () => {
    expect(recordingGLSL.fragment(fxaa(uniform("sampler2D")))).toContain("textureLod(");
    expect(recordingWGSL.fragment(fxaa(uniform("sampler2D")))).toContain("textureSampleLevel(");
  });

  /**
   * @canon spec-a-number-given-to-an-effect-compiles-as-a-literal
   */
  it.each([
    ["GLSL", (color: ReturnType<typeof rgbShift>) => recordingGLSL.fragment(color)],
    ["WGSL", (color: ReturnType<typeof rgbShift>) => recordingWGSL.fragment(color)],
  ])("compiles a number given to rgbShift as a literal on %s", (_, compile) => {
    const code = compile(rgbShift(uniform("sampler2D"), 0.25, 0));
    expect(code).toMatch(/\* 0\.25/);
  });

  /**
   * @canon spec-a-node-given-to-an-effect-is-read-as-given
   */
  it.each([
    ["GLSL", (color: ReturnType<typeof rgbShift>) => recordingGLSL.fragment(color)],
    ["WGSL", (color: ReturnType<typeof rgbShift>) => recordingWGSL.fragment(color)],
  ])("reads a uniform given to rgbShift on %s", (_, compile) => {
    const amount = uniform("float");
    const code = compile(rgbShift(uniform("sampler2D"), amount, 0));
    expect(code).toMatch(new RegExp(`\\* (_rmsl_uniforms\\.)?${amount.name}\\b`));
  });

  /**
   * The blur weights and FXAA's edge steps are TSL constant tables. They sit
   * in the code, and the only uniforms left are the sampler and the screen
   * size `uv()` reads.
   *
   * @canon spec-an-effect-writes-its-constant-tables-into-its-code
   */
  it.each([
    ["gaussianBlur", () => gaussianBlur(uniform("sampler2D")).passes[0]!.color],
    ["fxaa", () => fxaa(uniform("sampler2D"))],
  ])("declares no uniform for %s's constant table", (_, build) => {
    const glsl = glslUniforms(compileGlsl.fragment(build()));
    expect(glsl).toHaveLength(2);
    expect(glsl.filter((line) => /^uniform sampler2D /.test(line))).toHaveLength(1);
    expect(glsl.filter((line) => /^uniform vec2 /.test(line))).toHaveLength(1);
    const wgsl = wgslUniformFields(compileWgsl.fragment(build()));
    expect(wgsl).toHaveLength(1);
    expect(wgsl[0]).toMatch(/: vec2<f32>,$/);
  });

  /**
   * @canon spec-an-effect-writes-its-constant-tables-into-its-code
   */
  it("writes the gaussian weights into the blur's code", () => {
    const code = compileGlsl.fragment(gaussianBlur(uniform("sampler2D")).passes[0]!.color);
    const centre = getGaussianCoefficients(3 + 2 * 4)[0]!;
    expect(code).toContain(String(centre).slice(0, 8));
  });
});

describe("./test", () => {
  /**
   * @canon spec-the-test-library-loads-no-scene-graph
   */
  it("loads no scene graph from the test library", async () => {
    const result = await build({
      entryPoints: [new URL("../test/index.ts", import.meta.url).pathname],
      bundle: true,
      write: false,
      metafile: true,
      format: "esm",
      logLevel: "silent",
    });
    const inputs = Object.keys(result.metafile.inputs);
    expect(inputs.filter((path) => /src\/scene\/(core|objects|lights)\//.test(path))).toEqual([]);
  });

  /**
   * A variable a graph declares lives outside the routine unless the runner
   * is reentrant.
   *
   * @canon spec-a-runner-shares-its-scratch-unless-asked
   */
  it("declares its scratch outside the routine by default, and inside it when reentrant", () => {
    const u = uniform("float");
    const graph = () =>
      Fn(() => {
        const v = vec3(u, u.mul(2), 1).toVar();
        v.addAssign(vec3(1));
        return v;
      })();
    const [outside] = runner(graph).source.split("return function");
    const [before, inside] = runner(graph, { reentrant: true }).source.split("return function");
    expect(outside).toMatch(/^let _rmsl_\w+ = new Float64Array\(_rmsl_slots, \d+, 3\);/m);
    expect(before).not.toMatch(/^let _rmsl_/m);
    expect(inside).toMatch(/var _rmsl_\w+ = new Float64Array\(_rmsl_slots, \d+, 3\);/);
  });

  /**
   * @canon spec-render-draws-a-bool-as-black-or-white
   */
  it.each([
    [true, [1, 1, 1, 1]],
    [false, [0, 0, 0, 1]],
  ])("draws %s as an opaque grey level", (value, rgba) => {
    expect(render(() => bool(value), { width: 1, height: 1 }).at(0, 0)).toEqual(rgba);
  });

  /**
   * @canon spec-render-draws-a-short-vector-opaque
   */
  it("draws a vec3 with an alpha of one", () => {
    expect(render(() => vec3(0.25, 0.5, 0.75), { width: 1, height: 1 }).at(0, 0)).toEqual([0.25, 0.5, 0.75, 1]);
  });

  /** A program whose only uniform is the one `name` gives, read by `root`. */
  const program = (name: string, type: "mat4" | "mat3" | "vec3" | "vec2", value?: () => never) => {
    const node = uniform(type) as any;
    const roots = {
      mat4: () => node.mul(vec4(1, 2, 3, 1)),
      mat3: () => vec4(node.mul(vec3(1, 2, 3)), 1),
      vec3: () => vec4(node, 1),
      vec2: () => vec4(node, 0, 1),
    };
    return { fragmentRoot: roots[type]() as never, uniforms: [{ name, node, value }] };
  };

  /**
   * @canon spec-a-program-under-test-reads-a-renderer-matrix-as-the-identity
   */
  it.each([
    ["modelMatrix", "mat4"],
    ["viewMatrix", "mat4"],
    ["projectionMatrix", "mat4"],
    ["normalMatrix", "mat3"],
  ] as const)("reads %s as the identity", (name, type) => {
    const shade = fromProgram(program(name, type));
    expect(shade.unbound).toEqual([]);
    expect(shade().value).toEqual(new Float64Array([1, 2, 3, 1]));
  });

  /**
   * @canon spec-a-program-under-test-sees-its-camera-at-the-origin
   */
  it("reads cameraPosition as the origin", () => {
    expect(fromProgram(program("cameraPosition", "vec3"))().value).toEqual(new Float64Array([0, 0, 0, 1]));
  });

  /**
   * @canon spec-a-program-under-test-sees-a-resolution-of-one-pixel
   */
  it("reads resolution as one pixel unless the options give one", () => {
    expect(fromProgram(program("resolution", "vec2"))().value).toEqual(new Float64Array([1, 1, 0, 1]));
    expect(fromProgram(program("resolution", "vec2"), { resolution: [4, 2] })().value).toEqual(
      new Float64Array([4, 2, 0, 1]),
    );
  });

  /**
   * @canon spec-a-program-under-test-falls-back-when-a-value-function-throws
   */
  it("falls back to the default when a uniform's value function throws", () => {
    const throws = () => {
      throw new Error("no camera");
    };
    const shade = fromProgram(program("cameraPosition", "vec3", throws as () => never));
    expect(shade.unbound).toEqual([]);
    expect(shade().value).toEqual(new Float64Array([0, 0, 0, 1]));
  });

  /**
   * @canon spec-a-slot-two-names-share-reads-back-under-the-first
   */
  it("hands back a varying two names point at under the first name", () => {
    const shared = varying("vec2");
    const vertexRoot = Fn(() => {
      shared.assign(vec2(0.25, 0.75));
      const position = builtinPosition();
      position.assign(vec4(0, 0, 0, 1));
      return position;
    })();
    const project = fromProgram(
      {
        fragmentRoot: vec4(shared, 0, 1),
        vertexRoot: vertexRoot as never,
        varyings: [
          { name: "uv", node: shared },
          { name: "texCoord", node: shared },
        ],
      },
      { stage: "vertex" },
    );
    const written = project().varyings;
    expect(Object.keys(written)).toEqual(["uv"]);
    expect(written.uv).toEqual(new Float64Array([0.25, 0.75]));
  });

  /**
   * @canon spec-render-draws-the-only-member-of-an-output-struct
   */
  it("draws the one member of an outputStruct", () => {
    const graph = Fn(() => outputStruct(vec4(1, 0.5, 0.25, 1)))();
    expect(render(() => graph, { width: 1, height: 1, stage: "fragment" }).at(0, 0)).toEqual([1, 0.5, 0.25, 1]);
  });
});

describe("the Vite plugins", () => {
  /**
   * @canon spec-a-plugin-compiles-a-module-again-when-an-import-changes
   */
  it("compiles a module again when a module it imports changes", async () => {
    const dir = mkdtempSync(join(tmpdir(), "rmsl-vite-"));
    try {
      const entry = join(dir, "entry.ts");
      const source = `import value from "./value";\nexport default { value };\n`;
      writeFileSync(join(dir, "value.ts"), "export default 1;\n");
      const plugin = precompileShaders({ include: entry }) as unknown as Transform;
      expect((await plugin.transform.call({}, source, entry))!.code).toBe('export default {"value":1};');
      writeFileSync(join(dir, "value.ts"), "export default 2;\n");
      expect((await plugin.transform.call({}, source, entry))!.code).toBe('export default {"value":2};');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  /**
   * @canon spec-precompile-wasm-emits-each-program-as-an-asset
   */
  it("emits the assets of a cached module again in every build", async () => {
    const id = `${FIXTURES}wasm-fns.ts`;
    const plugin = precompileWasm({ include: "fixtures/wasm-fns.ts" }) as unknown as Transform;
    const first = pluginContext();
    await plugin.transform.call(first.context, wasmFnsSource, id);
    const second = pluginContext();
    await plugin.transform.call(second.context, wasmFnsSource, id);
    expect(second.emitted).toHaveLength(first.emitted.length);
  });

  /**
   * @canon spec-a-plugin-fails-the-build-on-a-module-it-cannot-compile
   */
  it("fails the build on a default export holding a function", async () => {
    const id = `${FIXTURES}fn-member.ts`;
    const plugin = precompileShaders({ include: "fn-member.ts" }) as unknown as Transform;
    await expect(plugin.transform.call({}, "export default { f: () => 1 };\n", id)).rejects.toThrow();
  });

  /**
   * @canon spec-a-plugin-fails-the-build-on-a-module-it-cannot-compile
   */
  it("fails the build on a program named by no identifier", async () => {
    const id = `${FIXTURES}bad-name.ts`;
    const plugin = precompileJS({ include: "bad-name.ts" }) as unknown as Transform;
    const source = `export const __RMSL_JS_CODE = { "my-fn": "return () => 1;" };\n`;
    await expect(plugin.transform.call({}, source, id)).rejects.toThrow();
    const reserved = `export const __RMSL_JS_CODE = { default: "return () => 1;" };\n`;
    await expect(plugin.transform.call({}, reserved, id)).rejects.toThrow(/"default", which is not/);
  });

  /**
   * @canon spec-a-plugin-fails-the-build-on-a-module-it-cannot-compile
   */
  it("fails the build on a default export holding a negative zero", async () => {
    const id = `${FIXTURES}negative-zero.ts`;
    const plugin = precompileShaders({ include: "negative-zero.ts" }) as unknown as Transform;
    await expect(plugin.transform.call({}, "export default { offset: -0 };\n", id)).rejects.toThrow(
      /-0 at default\.offset/,
    );
  });

  const plugins = [
    ["precompileShaders", precompileShaders],
    ["precompileJS", precompileJS],
    ["precompileWasm", precompileWasm],
  ] as const;

  /**
   * @canon spec-a-plugin-with-no-include-rewrites-nothing
   */
  it.each(plugins)("%s rewrites nothing when it is given no include", async (_, make) => {
    const plugin = make() as unknown as Transform;
    expect(await plugin.transform.call(pluginContext().context, cpuFnsSource, CPU_FNS_PATH)).toBeNull();
  });

  /**
   * @canon spec-a-plugin-leaves-a-module-its-exclude-matches
   */
  it.each(plugins)("%s leaves a module its exclude matches, though its include matches too", async (_, make) => {
    const plugin = make({ include: "fixtures/cpu-fns.ts", exclude: /cpu-fns/g }) as unknown as Transform;
    const { context } = pluginContext();
    expect(await plugin.transform.call(context, cpuFnsSource, CPU_FNS_PATH)).toBeNull();
    expect(await plugin.transform.call(context, cpuFnsSource, CPU_FNS_PATH)).toBeNull();
  });

  /**
   * @canon spec-compile-wat-loads-every-wat-module
   */
  it("compileWat turns a .wat module into its bytes with no include", async () => {
    const plugin = compileWat() as unknown as Load;
    const code = await plugin.load(RASTERIZER_WAT);
    expect(code).toMatch(/^export default new Uint8Array\(\[0,97,115,109,1,0,0,0,/);
    expect(await plugin.load(CPU_FNS_PATH)).toBeNull();
  });

  /**
   * @canon spec-a-wat-module-exports-its-shared-variant-or-undefined
   */
  it("exports a shared variant of a .wat module that imports a memory, and undefined for one that does not", async () => {
    const plugin = compileWat() as unknown as Load;
    const dir = mkdtempSync(join(tmpdir(), "rmsl-wat-"));
    try {
      const plain = join(dir, "plain.wat");
      writeFileSync(plain, "(module)");
      const load = async (path: string) =>
        (await import(`data:text/javascript,${encodeURIComponent((await plugin.load(path))!)}`)) as Record<string, any>;
      const withoutMemory = await load(plain);
      expect(Object.keys(withoutMemory)).toContain("shared");
      expect(withoutMemory.shared).toBeUndefined();
      expect((await load(RASTERIZER_WAT)).shared).toBeInstanceOf(Uint8Array);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  /**
   * The memory import is spread over lines, names its memory and gives a
   * maximum; the shared variant links against a shared memory.
   *
   * @canon spec-a-wat-module-exports-its-shared-variant-or-undefined
   */
  it("exports a shared variant of a .wat module whose memory import has a name, a maximum and line breaks", async () => {
    const plugin = compileWat() as unknown as Load;
    const dir = mkdtempSync(join(tmpdir(), "rmsl-wat-"));
    try {
      const path = join(dir, "named.wat");
      writeFileSync(path, '(module\n  (import "env" "memory"\n    (memory $mem 1 4)))');
      const { shared } = (await import(
        `data:text/javascript,${encodeURIComponent((await plugin.load(path))!)}`
      )) as Record<string, any>;
      const memory = new WebAssembly.Memory({ initial: 1, maximum: 65536, shared: true });
      expect(() => new WebAssembly.Instance(new WebAssembly.Module(shared), { env: { memory } })).not.toThrow();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  /**
   * The memory is imported inline in its own declaration, a form the loader
   * does not rewrite.
   *
   * @canon spec-compile-wat-refuses-a-memory-import-it-cannot-make-shared
   */
  it("refuses a .wat module whose memory import it cannot make shared", async () => {
    const plugin = compileWat() as unknown as Load;
    const dir = mkdtempSync(join(tmpdir(), "rmsl-wat-"));
    try {
      const path = join(dir, "inline.wat");
      writeFileSync(path, '(module (memory (import "env" "memory") 1))');
      await expect(plugin.load(path)).rejects.toThrow(/inline\.wat/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  /**
   * @canon spec-a-precompiled-wasm-program-is-ready-when-its-module-loads
   */
  it("instantiates each WASM program before its module finishes loading", async () => {
    const plugin = precompileWasm({ include: WASM_FNS_PATH }) as unknown as Transform;
    const { context, emitted } = pluginContext();
    let code = (await plugin.transform.call(context, wasmFnsSource, WASM_FNS_PATH))!.code;
    expect(code).toMatch(/^const _rmslBytes_brightness = await fetch\(/m);
    emitted.forEach((asset, i) => {
      const url = `data:application/wasm;base64,${Buffer.from(asset.source).toString("base64")}`;
      code = code.replaceAll(`import.meta.ROLLUP_FILE_URL_ref${i}`, JSON.stringify(url));
    });
    code = code.replace(
      '"@random-mesh/rmsl/wasm"',
      JSON.stringify(new URL("../../dist/wasm.js", import.meta.url).href),
    );
    const mod = (await import(`data:text/javascript,${encodeURIComponent(code)}`)) as Record<string, any>;
    const result = mod.brightness({ uniforms: { _rmsl_u0: [1, 2, 3] } });
    expect(result).not.toBeInstanceOf(Promise);
    expect(result).toEqual(new Float64Array([0.5, 1, 1.5]));
  });
});
