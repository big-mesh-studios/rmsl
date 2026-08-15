import { describe, it, expect, afterAll } from "vitest";
import {
  Fn, float, int, uint, vec2, vec3, vec4, bool,
  uniform, If, Loop, select, luminance, rand, interleavedGradientNoise,
  textureSize, textureLoad, fragCoord, uv, screenCoordinate, time,
  compileGLSLFn, compileWGSLFn,
} from "../rmsl";
import {
  recordingGLSL as compileGLSL,
  recordingWGSL as compileWGSL,
  assertRecordedShadersValid,
} from "../testing/shader-validity";
import {
  sepia, bleach, dotScreen, rgbShift, lut3D, sobel, chromaticAberration,
  transition, film, crt, motionBlur, circle, sharpen, fxaa,
  boxBlur, hashBlur, radialBlur, gaussianBlur, bloom, luminosityHighPass,
} from "./index";

afterAll(async () => {
  await assertRecordedShadersValid();
}, 120_000);

describe("core primitives for effects", () => {
  it("compiles select to a ternary in GLSL and select() in WGSL", () => {
    const prog = Fn(() => {
      const c = uniform("float");
      return c.greaterThan(0.5).select(float(1.0), float(0.0)).toVar();
    });
    const glsl = compileGLSL(prog());
    expect(glsl).toMatch(/\? /);
    const wgsl = compileWGSL(prog());
    expect(wgsl).toContain("select(");
  });

  it("folds a select with a literal condition", () => {
    const prog = Fn(() => select(bool(true), float(1.0), float(2.0)).toVar());
    const glsl = compileGLSL(prog());
    expect(glsl).toContain("1.0");
    expect(glsl).not.toContain("?");
  });

  it("parenthesizes a select nested in a higher-precedence operation", () => {
    // `a * (c ? x : y)` must not collapse to `(a * c) ? x : y`.
    const prog = Fn(() => {
      const x = uniform("float").toVar();
      const sel = x.greaterThan(0.5).select(float(1.0), float(0.0));
      return x.mul(sel).toVar();
    });
    const glsl = compileGLSL(prog());
    expect(glsl).toContain("?");
    // The ternary must sit inside parentheses in the multiply.
    expect(glsl).toMatch(/\* \([^()]*\? [^()]*: [^()]*\)/);
    const wgsl = compileWGSL(prog());
    expect(wgsl).toContain("select(");
  });

  it("compiles luminance, rand and interleavedGradientNoise", () => {
    const prog = Fn(() => {
      const color = vec3(0.2, 0.4, 0.6);
      const u = vec2(0.25, 0.75);
      return vec4(luminance(color), rand(u), interleavedGradientNoise(u), 1.0).toVar();
    });
    const glsl = compileGLSL(prog());
    expect(glsl).toContain("sin");
    const wgsl = compileWGSL(prog());
    expect(wgsl).toContain("sin");
  });

  it("compiles textureSize and textureLoad on a float sampler", () => {
    const prog = Fn(() => {
      const tex = uniform("sampler2D");
      const p = int(4);
      return textureLoad(tex, vec2(1, 2).toIVec2()).toVar();
    });
    const glsl = compileGLSL(prog());
    expect(glsl).toContain("texelFetch");
    const wgsl = compileWGSL(prog());
    expect(wgsl).toContain("textureLoad(");
  });

  it("reads the fragment coordinate", () => {
    const prog = Fn(() => fragCoord().div(vec2(800, 600)).toVar());
    const glsl = compileGLSL(prog());
    expect(glsl).toContain("gl_FragCoord.xy");
    const wgsl = compileWGSL(prog());
    expect(wgsl).toContain("@builtin(position)");
  });

  it("declares a shared time uniform", () => {
    const prog = Fn(() => time().mul(1.0).toVar());
    const glsl = compileGLSL(prog());
    expect(glsl).toContain("uniform float");
  });
});

describe("color effects", () => {
  it("sepia", () => {
    const color = vec4(0.5, 0.3, 0.2, 1.0);
    const glsl = compileGLSL(sepia(color));
    expect(glsl).toContain("dot(");
  });

  it("bleach bypass", () => {
    const color = vec4(0.5, 0.3, 0.2, 1.0);
    const glsl = compileGLSL(bleach(color));
    expect(glsl).toContain("mix(");
  });

  it("dotScreen", () => {
    const color = vec4(0.5, 0.3, 0.2, 1.0);
    const glsl = compileGLSL(dotScreen(color));
    expect(glsl).toContain("gl_FragCoord.xy");
    const wgsl = compileWGSL(dotScreen(color));
    expect(wgsl).toContain("@builtin(position)");
  });

  it("circle", () => {
    const glsl = compileGLSL(vec4(circle(1.0, 0.5, uniform("vec2")), 0, 0, 1));
    expect(glsl).toContain("smoothstep");
  });
});

describe("texture-based effects", () => {
  const tex = () => uniform("sampler2D");

  it("rgbShift", () => {
    const glsl = compileGLSL(rgbShift(tex()));
    expect(glsl).toContain("texture(");
    expect(glsl).toContain("uniform sampler2D");
  });

  it("chromaticAberration", () => {
    const glsl = compileGLSL(chromaticAberration(tex()));
    expect(glsl).toContain("texture(");
  });

  it("film", () => {
    const glsl = compileGLSL(film(vec4(0.5, 0.3, 0.2, 1.0)));
    expect(glsl).toContain("sin(");
  });

  it("transition with a mix texture", () => {
    const a = tex();
    const b = tex();
    const mixTex = tex();
    const glsl = compileGLSL(transition(a, b, mixTex, 0.5, 0.1, 1));
    expect(glsl).toContain("if (");
    expect(glsl).toContain("else");
  });

  it("motionBlur", () => {
    const velocity = vec2(0.01, 0.0);
    const glsl = compileGLSL(motionBlur(tex(), velocity, 8));
    expect(glsl).toContain("for (");
    const wgsl = compileWGSL(motionBlur(tex(), velocity, 8));
    expect(wgsl).toContain("for (");
  });

  it("sharpen (RCAS)", () => {
    const glsl = compileGLSL(sharpen(tex()));
    expect(glsl).toContain("texelFetch");
    const wgsl = compileWGSL(sharpen(tex()));
    expect(wgsl).toContain("textureLoad(");
  });

  it("fxaa", () => {
    const glsl = compileGLSL(fxaa(tex()));
    expect(glsl).toContain("texture(");
    expect(glsl).toContain("break;");
  });

  it("sobel", () => {
    const glsl = compileGLSL(sobel(tex()));
    expect(glsl).toContain("texture(");
    expect(glsl).toContain("sqrt(");
  });
});

describe("blur effects", () => {
  const tex = () => uniform("sampler2D");

  it("boxBlur", () => {
    const glsl = compileGLSL(boxBlur(tex(), { size: 1 }));
    expect(glsl).toContain("for (");
    const wgsl = compileWGSL(boxBlur(tex(), { size: 1 }));
    expect(wgsl).toContain("for (");
  });

  it("hashBlur", () => {
    const glsl = compileGLSL(hashBlur(tex()));
    expect(glsl).toContain("sin");
    const wgsl = compileWGSL(hashBlur(tex()));
    expect(wgsl).toContain("sin");
  });

  it("radialBlur", () => {
    const glsl = compileGLSL(radialBlur(tex()));
    expect(glsl).toContain("for (");
    const wgsl = compileWGSL(radialBlur(tex()));
    expect(wgsl).toContain("for (");
  });
});

describe("lut3D", () => {
  it("samples a 3D lookup texture", () => {
    const input = vec4(0.5, 0.3, 0.2, 1.0);
    const lut = uniform("sampler3D");
    const glsl = compileGLSL(lut3D(input, lut, 16));
    expect(glsl).toContain("uniform sampler3D");
    expect(glsl).toContain("texture(");
    const wgsl = compileWGSL(lut3D(input, lut, 16));
    expect(wgsl).toContain("texture_3d");
  });
});

describe("crt", () => {
  it("composes the CRT effects", () => {
    const glsl = compileGLSL(crt(uniform("sampler2D")));
    expect(glsl).toContain("texture(");
    expect(glsl).toContain("sin(");
    expect(glsl).toContain("smoothstep");
    const wgsl = compileWGSL(crt(uniform("sampler2D")));
    expect(wgsl).toContain("sin(");
  });
});

describe("gaussianBlur pass graph", () => {
  it("produces a two-pass graph that compiles on both backends", () => {
    const graph = gaussianBlur(uniform("sampler2D"), [1, 1], 4);
    expect(graph.passes).toHaveLength(2);
    expect(graph.output).toBe("gaussianBlur.vertical");
    for (const pass of graph.passes) {
      const glsl = compileGLSL(pass.color);
      expect(glsl).toContain("texture(");
      const wgsl = compileWGSL(pass.color);
      expect(wgsl).toContain("textureSample");
    }
  });

  it("exposes each pass's input sampler for binding", () => {
    const input = uniform("sampler2D");
    const graph = gaussianBlur(input);
    expect(graph.passes[0].inputs.input).toBe(input);
    // The vertical pass reads the horizontal pass's render target.
    expect(graph.passes[1].inputs.input).not.toBe(input);
  });
});

describe("bloom", () => {
  it("produces the faithful 12-pass graph", () => {
    const graph = bloom(uniform("sampler2D"));
    expect(graph.passes).toHaveLength(12);
    expect(graph.output).toBe("bloom.composite");
    expect(graph.passes[0].name).toBe("bloom.highpass");
    expect(graph.passes[0].scale).toBe(0.5);
    // Mip 0 does not halve relative to the high pass; later horizontal passes do.
    expect(graph.passes[1].scale).toBe(1);   // mip0.horizontal
    expect(graph.passes[3].scale).toBe(0.5); // mip1.horizontal
    expect(graph.passes[11].name).toBe("bloom.composite");
    // The composite reads the five vertical mips.
    expect(Object.keys(graph.passes[11].inputs)).toEqual([
      "bloom.mip0.vertical", "bloom.mip1.vertical", "bloom.mip2.vertical",
      "bloom.mip3.vertical", "bloom.mip4.vertical",
    ]);
  });

  it("compiles every pass to GLSL and WGSL", () => {
    const graph = bloom(uniform("sampler2D"));
    for (const pass of graph.passes) {
      const glsl = compileGLSL(pass.color);
      expect(glsl).toContain("texture(");
      const wgsl = compileWGSL(pass.color);
      expect(wgsl).toContain("textureSample(");
    }
  });

  it("high pass applies a luminance threshold", () => {
    const glsl = compileGLSL(bloom(uniform("sampler2D")).passes[0].color);
    expect(glsl).toContain("smoothstep(");
    expect(glsl).toContain("0.2126");
  });

  it("the composite sums five tinted mips scaled by strength", () => {
    const graph = bloom(uniform("sampler2D"), { strength: 0.7 });
    const glsl = compileGLSL(graph.passes[11].color);
    expect(glsl).toMatch(/0\.8/); // factor for mip 1
    expect(glsl).toMatch(/0\.7/); // folded strength
    // Five texture samples in the composite.
    expect(glsl.match(/texture\(/g)).toHaveLength(5);
  });

  it("accepts a custom high-pass filter", () => {
    const graph = bloom(uniform("sampler2D"), {
      highPassFn: (input, threshold, smoothWidth) =>
        vec4(luminance(input.rgb).mul(threshold).add(smoothWidth), 1),
    });
    const glsl = compileGLSL(graph.passes[0].color);
    expect(glsl).toContain("0.2126"); // luminance coefficients of the custom filter
  });

  it("luminosityHighPass is available standalone", () => {
    const input = uniform("sampler2D").texture(uv());
    const glsl = compileGLSL(luminosityHighPass(input, 0.3, 0.01));
    expect(glsl).toContain("smoothstep(");
  });
});

describe("compileGLSLFn/WGSLFn embedding", () => {
  it("emits a standalone sepia function", () => {
    const glslFn = compileGLSLFn((color) => sepia(color), {
      name: "sepia",
      params: [{ name: "color", type: "vec4" }],
    });
    expect(glslFn).toContain("vec4 sepia(vec4 color)");
    expect(glslFn).toContain("color");
  });

  it("emits a standalone WGSL sepia function", () => {
    const wgslFn = compileWGSLFn((color) => sepia(color), {
      name: "sepia",
      params: [{ name: "color", type: "vec4" }],
    });
    expect(wgslFn).toContain("fn sepia");
  });
});
