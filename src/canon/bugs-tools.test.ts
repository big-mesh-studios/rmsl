/// <reference types="vite/client" />
import { afterAll, afterEach, describe, expect, it, vi } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { build } from "esbuild";
import { uniform, type Node } from "../rmsl";
import { compileWgsl } from "../wgsl";
import { fxaa, gaussianBlur, transition } from "../effects";
import { precompileJS, precompileShaders, precompileWasm } from "../vite/vite";
import { GPU_ENABLED, gpuDevice, releaseGpu } from "../testing/gpu";
import { sweepWGSL } from "../testing/integer-sweep";
import wasmFnsSource from "../vite/fixtures/wasm-fns.ts?raw";

type TransformResult = { code: string; map: null } | null;
type Transform = { transform: { call(context: unknown, code: string, id: string): Promise<TransformResult> } };

const FIXTURES = new URL("../vite/fixtures/", import.meta.url).pathname;

/** A Rollup plugin context with just the `emitFile` precompileWasm calls. */
function pluginContext() {
  const emitted: unknown[] = [];
  return { context: { emitFile: (asset: unknown) => `ref${emitted.push(asset) - 1}` }, emitted };
}

/** What Dawn says about a WGSL shader, or null when it compiles. */
async function wgslError(code: string): Promise<string | null> {
  const device = await gpuDevice();
  device.pushErrorScope("validation");
  device.createShaderModule({ code });
  const error = await device.popErrorScope();
  return error ? error.message : null;
}

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
});

afterAll(async () => {
  await releaseGpu();
}, 120_000);

describe("known bugs of the tools, each failing until its fix", () => {
  /**
   * FXAA samples its texture inside the branch that skips a pixel off an edge,
   * which depends on sampled luminance, so Dawn refuses the WGSL: "textureSample
   * must only be called from uniform control flow".
   *
   * @canon bug-fxaa-samples-in-non-uniform-control-flow-on-wgsl
   */
  it.skipIf(!GPU_ENABLED).fails("compiles fxaa on WGSL", async () => {
    expect(await wgslError(compileWgsl.fragment(fxaa(uniform("sampler2D"))))).toBeNull();
  });

  /**
   * The vertical pass of `gaussianBlur` keys the horizontal pass's render
   * target as `input`, where `bloom` keys an internal link by the pass that
   * produces it.
   *
   * @canon bug-gaussian-blur-keys-its-internal-link-as-input
   */
  it.fails("keys the vertical pass's input by the horizontal pass", () => {
    const graph = gaussianBlur(uniform("sampler2D"));
    expect(Object.keys(graph.passes[1]!.inputs)).toEqual(["gaussianBlur.horizontal"]);
  });

  /**
   * `transition` builds the branch that samples the mix texture whatever
   * `useTexture` is, so a `null` mix texture throws a `TypeError`.
   *
   * @canon bug-transition-reads-a-null-mix-texture
   */
  it.fails("cross-fades without a mix texture when useTexture is 0", () => {
    expect(() => transition(uniform("sampler2D"), uniform("sampler2D"), null, 0.5, 0.1, 0)).not.toThrow();
  });

  /**
   * The plugins cache a result by the source of the module alone, so a module
   * whose import changed gives the result it gave before the change.
   *
   * @canon bug-a-plugin-serves-a-stale-result-after-an-import-changes
   */
  it.fails("compiles a module again when a module it imports changes", async () => {
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
   * `precompileWasm` emits its assets only on the first transform of a source,
   * so a later build answers from its cache with a reference to an asset that
   * build never emitted.
   *
   * @canon bug-precompile-wasm-emits-no-asset-on-a-cached-transform
   */
  it.fails("emits the assets of a cached module again in every build", async () => {
    const id = `${FIXTURES}wasm-fns.ts`;
    const plugin = precompileWasm({ include: "fixtures/wasm-fns.ts" }) as unknown as Transform;
    const first = pluginContext();
    await plugin.transform.call(first.context, wasmFnsSource, id);
    const second = pluginContext();
    await plugin.transform.call(second.context, wasmFnsSource, id);
    expect(second.emitted).toHaveLength(first.emitted.length);
  });

  /**
   * `precompileShaders` checks only that `JSON.stringify` gives a string, which
   * it does after dropping a function, so a default export holding one ships
   * without it.
   *
   * @canon bug-precompile-shaders-drops-a-function-silently
   */
  it.fails("fails the build on a default export holding a function", async () => {
    const id = `${FIXTURES}fn-member.ts`;
    const plugin = precompileShaders({ include: "fn-member.ts" }) as unknown as Transform;
    await expect(plugin.transform.call({}, "export default { f: () => 1 };\n", id)).rejects.toThrow();
  });

  /**
   * `precompileJS` writes each key of the code map as `export const <key>`
   * without checking it is an identifier, so `my-fn` gives a module that does
   * not parse.
   *
   * @canon bug-precompile-js-writes-a-key-that-is-no-identifier
   */
  it.fails("fails the build on a program named by no identifier", async () => {
    const id = `${FIXTURES}bad-name.ts`;
    const plugin = precompileJS({ include: "bad-name.ts" }) as unknown as Transform;
    const source = `export const __RMSL_JS_CODE = { "my-fn": "return () => 1;" };\n`;
    await expect(plugin.transform.call({}, source, id)).rejects.toThrow();
  });

  /**
   * The harness compares a WASM result with the JS one by `===`, which a NaN
   * never meets, so two targets that both give NaN are reported as disagreeing.
   *
   * @canon bug-the-harness-reports-nan-on-both-targets-as-a-disagreement
   */
  it.fails("passes a program that gives NaN on both CPU targets", async () => {
    const { harness } = await freshHarness(true);
    harness.evaluateRecording((a) => a.div(a), [0]);
    await expect(harness.assertRecordedEvaluationsAgree()).resolves.toBeUndefined();
  });

  /**
   * The harness compares a WASM result with the JS one by `===`, which holds
   * between `-0` and `0`, so a WASM result whose sign of zero differs passes.
   *
   * @canon bug-the-harness-reads-negative-zero-as-zero
   */
  it.fails("reports a WASM zero whose sign differs from the JS one", async () => {
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
   * The harness compares a GPU result with the JS one by the distance between
   * them, which is NaN between two equal infinities, so a program that gives
   * the same infinity everywhere is reported as disagreeing.
   *
   * @canon bug-the-harness-reports-an-infinity-on-every-target-as-a-disagreement
   */
  it.skipIf(!GPU_ENABLED).fails(
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
   * The harness counts a program WASM refuses as a skip and passes the run,
   * so a program evaluated on no WASM at all passes as if it agreed.
   *
   * @canon bug-the-harness-passes-a-program-wasm-refuses
   */
  it.fails("fails a run in which WASM refused a recorded program", async () => {
    const { harness, rmsl } = await freshHarness(true);
    const build = (a: Node<"float">) => rmsl.mat2(rmsl.vec2(a, 1), rmsl.vec2(2, 4)).inverse().element(rmsl.int(0)).x;
    expect(() => harness.evaluateWASM(build, [3])).toThrow(/compileWasmFn/);
    harness.evaluateRecording(build, [3]);
    await expect(harness.assertRecordedEvaluationsAgree()).rejects.toThrow(/WASM/);
  });

  /**
   * The WGSL sweep gives each invocation eight argument slots and never checks
   * a case fits them, so a ninth argument is read from the next invocation's.
   *
   * @canon bug-the-wgsl-sweep-reads-a-ninth-argument-from-the-next-run
   */
  it.skipIf(!GPU_ENABLED).fails("passes a right case of nine arguments on WGSL", async () => {
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
   * `releaseGpu` awaits each browser and page it opened, so a browser that
   * failed to launch makes it throw that failure again.
   *
   * @canon bug-release-gpu-throws-when-a-launch-failed
   */
  it.fails("releases the harness's resources after a browser failed to launch", async () => {
    vi.doMock("playwright", () => ({ chromium: { launch: () => Promise.reject(new Error("no browser")) } }));
    vi.resetModules();
    const gpu = await import("../testing/gpu");
    await expect(gpu.gpuPage()).rejects.toThrow("no browser");
    await expect(gpu.releaseGpu()).resolves.toBeUndefined();
  });

  /**
   * `./test` imports the sampling helpers of the scene renderers, which import
   * the scene's lights and through them its objects, so a test of a plain
   * shader loads the scene graph.
   *
   * @canon bug-the-test-library-loads-the-scene-graph
   */
  it.fails("loads no scene graph from the test library", async () => {
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
});
