/// <reference types="vite/client" />
import { afterAll, describe, expect, it } from "vitest";
import { Fn, bool, uniform, varying, vec2, vec3, vec4, builtinPosition } from "../rmsl";
import { compileGlsl } from "../glsl";
import { compileWgsl } from "../wgsl";
import { fxaa, gaussianBlur, getGaussianCoefficients, rgbShift } from "../effects";
import { fromProgram, render, runner } from "../test";
import { compileWat, precompileJS, precompileShaders, precompileWasm } from "../vite/vite";
import {
  assertRecordedEvaluationsAgree,
  closeEvaluators,
  evaluateRecording,
  recordedEvaluationSummary,
} from "../testing/shader-eval";
import { assertRecordedShadersValid, recordingGLSL, recordingWGSL } from "../testing/shader-validity";
import cpuFnsSource from "../vite/fixtures/cpu-fns.ts?raw";
import wasmFnsSource from "../vite/fixtures/wasm-fns.ts?raw";

afterAll(async () => {
  await assertRecordedShadersValid();
  await assertRecordedEvaluationsAgree();
  await closeEvaluators();
}, 120_000);

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

describe("the harness checks what it recorded", () => {
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
    expect(outside).toMatch(/^let _rmsl_\w+ = \[0, 0, 0\];/m);
    expect(before).not.toMatch(/^let _rmsl_/m);
    expect(inside).toMatch(/var _rmsl_\w+ = \[0, 0, 0\];/);
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
    expect(shade().value).toEqual([1, 2, 3, 1]);
  });

  /**
   * @canon spec-a-program-under-test-sees-its-camera-at-the-origin
   */
  it("reads cameraPosition as the origin", () => {
    expect(fromProgram(program("cameraPosition", "vec3"))().value).toEqual([0, 0, 0, 1]);
  });

  /**
   * @canon spec-a-program-under-test-sees-a-resolution-of-one-pixel
   */
  it("reads resolution as one pixel unless the options give one", () => {
    expect(fromProgram(program("resolution", "vec2"))().value).toEqual([1, 1, 0, 1]);
    expect(fromProgram(program("resolution", "vec2"), { resolution: [4, 2] })().value).toEqual([4, 2, 0, 1]);
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
    expect(shade().value).toEqual([0, 0, 0, 1]);
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
    expect(written.uv).toEqual([0.25, 0.75]);
  });
});

describe("the Vite plugins", () => {
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
    const result = mod.brightness.run({ uniforms: { _rmsl_u0: [1, 2, 3] } });
    expect(result).not.toBeInstanceOf(Promise);
    expect(result).toEqual({ value: [0.5, 1, 1.5] });
  });
});
