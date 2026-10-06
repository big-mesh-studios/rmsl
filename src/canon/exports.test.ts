import { afterAll, describe, expect, it } from "vitest";
import {
  difference,
  float,
  Fn,
  PI2,
  premultiplyAlpha,
  screenUV,
  uniform,
  unpremultiplyAlpha,
  vec2,
  vec3,
  vec4,
  type Node,
} from "../rmsl";
import { compileJSRoutine, compileJSGrid } from "../js";
import { uniformsIn } from "../test";
import { barrelMask, barrelUV, colorBleeding, getGaussianCoefficients, premultipliedGaussianBlur, scanlines, vignette } from "../effects";
import {
  Blending,
  Builder,
  ceilPowerOfTwo,
  ClampToEdgeWrapping,
  damp,
  floorPowerOfTwo,
  euclideanModulo,
  inverseLerp,
  lambertDiffuse,
  LinearFilter,
  mapLinear,
  MirroredRepeatWrapping,
  NearestFilter,
  pingpong,
  pointLightAttenuation,
  randFloat,
  randInt,
  RepeatWrapping,
  rendererUniformValue,
  resolveSlot,
  seededRandom,
  smootherstep,
  standardLight,
  toBufferView,
} from "../scene";
import { assertRecordedEvaluationsAgree, closeEvaluators, evaluateRecording } from "../testing/shader-eval";
import {
  assertRecordedShadersValid,
  recordingGLSL as compileGlsl,
  recordingWGSL as compileWgsl,
} from "../testing/shader-validity";

afterAll(async () => {
  await assertRecordedShadersValid();
  await assertRecordedEvaluationsAgree();
  await closeEvaluators();
}, 120_000);

const none = { name: "main", params: [] };

/** Compiles a fragment stage that returns `color` on both GPU targets. */
function compileOnBothGpus(color: () => Node<"vec4">) {
  expect(compileGlsl.fragment(Fn(() => color().toVar())())).toContain("main");
  expect(compileWgsl.fragment(Fn(() => color().toVar())())).toContain("main");
}

describe("exports no other test reaches", () => {
  /**
   * @canon spec-a-math-function-compiles-to-the-builtin-of-the-target
   */
  it("computes the inverse hyperbolic functions on every target", () => {
    const build = (a: Node<"float">) => a.asinh().add(a.add(1).acosh()).add(a.mul(0.5).atanh());
    expect(evaluateRecording(build, [0.5])).toBeCloseTo(Math.asinh(0.5) + Math.acosh(1.5) + Math.atanh(0.25), 12);
  });

  /**
   * @canon spec-an-operation-no-target-has-is-composed
   */
  it("computes difference and premultiplied alpha on every target", () => {
    expect(evaluateRecording((a, b) => difference(a, b), [2, 5])).toBe(3);
    expect(evaluateRecording((a) => premultiplyAlpha(vec4(1, a, 3, 0.5)), [2])).toEqual([0.5, 1, 1.5, 0.5]);
    expect(evaluateRecording((a) => unpremultiplyAlpha(vec4(0.5, a, 1.5, 0.5)), [1])).toEqual([1, 2, 3, 0.5]);
    expect(evaluateRecording((a) => unpremultiplyAlpha(vec4(a, 1, 1, 0)), [1])).toEqual([0, 0, 0, 0]);
  });

  /**
   * @canon spec-the-tsl-constants-are-float-literals
   */
  it("reads PI2 as two pi on every target", () => {
    expect(evaluateRecording((a) => a.mul(PI2), [1])).toBeCloseTo(2 * Math.PI, 12);
  });

  /**
   * @canon spec-the-screen-accessors-follow-tsl
   */
  it("divides the fragment coordinate by the screen size in screenUV", () => {
    compileOnBothGpus(() => vec4(screenUV(), 0, 1));
    const uv = screenUV();
    const [size] = uniformsIn(uv);
    const run = compileJSGrid(() => Fn(() => uv.toVar())(), none);
    expect(Array.from(run({ uniforms: { [size!.name]: [2, 1] } }, 2, 1))).toEqual([0.25, 0.5, 0.75, 0.5]);
  });

  /**
   * @canon spec-a-single-pass-effect-gives-a-colour-node
   */
  it("compiles the CRT building blocks on both GPU targets", () => {
    const tex = uniform("sampler2D");
    compileOnBothGpus(() => vec4(barrelUV(0.1), barrelMask(barrelUV(0.1)), 1));
    compileOnBothGpus(() => vec4(colorBleeding(tex, 0.002), 1));
    compileOnBothGpus(() => vec4(scanlines(vec3(1, 1, 1)), 1) as Node<"vec4">);
    compileOnBothGpus(() => vec4(vignette(vec3(1, 1, 1)), 1) as Node<"vec4">);
  });

  /**
   * @canon spec-gaussian-blur-weights-follow-tsl
   */
  it("gives the Gaussian weights TSL gives", () => {
    const sigma = 1;
    expect(getGaussianCoefficients(3)).toEqual([0, 1, 2].map((i) => (0.39894 * Math.exp((-0.5 * i * i) / (sigma * sigma))) / sigma));
  });

  /**
   * @canon spec-an-effect-with-several-passes-is-a-pass-graph
   */
  it("gives a pass graph for a premultiplied Gaussian blur", () => {
    const graph = premultipliedGaussianBlur(uniform("sampler2D"));
    expect(graph.passes.length).toBeGreaterThan(0);
    expect(graph.passes.map((p) => p.name)).toContain(graph.output);
  });

  /**
   * @canon spec-the-math-classes-follow-three-js
   */
  it("computes the MathUtils helpers as three.js does", () => {
    expect(euclideanModulo(-1, 3)).toBe(2);
    expect(inverseLerp(2, 4, 3)).toBe(0.5);
    expect(mapLinear(5, 0, 10, 0, 100)).toBe(50);
    expect(smootherstep(0.5, 0, 1)).toBe(0.5);
    expect(pingpong(1.5, 1)).toBe(0.5);
    expect(damp(0, 1, 0, 1)).toBe(0);
    expect(ceilPowerOfTwo(5)).toBe(8);
    expect(floorPowerOfTwo(5)).toBe(4);
    expect(seededRandom(1)).toBe(seededRandom(1));
    for (let i = 0; i < 20; i++) {
      const n = randInt(2, 4);
      expect(Number.isInteger(n) && n >= 2 && n <= 4).toBe(true);
      const f = randFloat(2, 4);
      expect(f >= 2 && f < 4).toBe(true);
    }
  });

  /**
   * @canon spec-the-three-js-constants-carry-three-js-values
   */
  it("gives the wrapping, filtering and blending constants three.js's numbers", () => {
    expect([RepeatWrapping, ClampToEdgeWrapping, MirroredRepeatWrapping]).toEqual([1000, 1001, 1002]);
    expect([NearestFilter, LinearFilter]).toEqual([1003, 1006]);
    expect([Blending.NoBlending, Blending.NormalBlending, Blending.AdditiveBlending]).toEqual([0, 1, 2]);
  });

  /**
   * @canon spec-the-lighting-terms-compile-on-both-gpu-targets
   */
  it("compiles the lighting terms on both GPU targets", () => {
    const n = vec3(0, 1, 0);
    const l = vec3(0, 1, 0);
    const c = vec3(1, 1, 1);
    compileOnBothGpus(() => vec4(lambertDiffuse(c, n, l, c), 1));
    compileOnBothGpus(() => vec4(standardLight(c, n, vec3(0, 0, 1), l, c, float(0.5), float(0), vec3(0.04, 0.04, 0.04)), 1));
    compileOnBothGpus(() => vec4(pointLightAttenuation(vec3(0, 2, 0), vec3(0, 0, 0), float(10), float(2)), 0, 0, 1));
  });

  /**
   * @canon spec-a-renderer-supplies-the-camera-and-object-uniforms
   */
  it("gives a program the renderer's resolution, and nothing for a name it does not know", () => {
    expect(rendererUniformValue("resolution", 640, 480)).toEqual([640, 480]);
    expect(rendererUniformValue("unknown", 640, 480)).toEqual([]);
  });

  /**
   * @canon spec-a-material-slot-takes-a-node-or-a-builder
   */
  it("resolves a slot given as a node or as a function of the builder", () => {
    const node = vec3(1, 2, 3);
    const b = new Builder();
    expect(resolveSlot(node, b)).toBe(node);
    expect(resolveSlot(() => node, b)).toBe(node);
    expect(resolveSlot(undefined, b)).toBeUndefined();
  });

  /**
   * @canon spec-a-geometry-builds-the-vertices-three-js-builds
   */
  it("uploads an index as 16-bit until it passes 65535", () => {
    expect(toBufferView([0, 1, 2], true)).toBeInstanceOf(Uint16Array);
    expect(toBufferView([0, 70000], true)).toBeInstanceOf(Uint32Array);
    expect(toBufferView([0.5, 1])).toBeInstanceOf(Float32Array);
  });
});
