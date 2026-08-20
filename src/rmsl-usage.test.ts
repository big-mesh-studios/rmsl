import { describe, it, expect, afterAll } from "vitest";
import {
  Fn, float, vec2, vec3, vec4, int, uint, bool,
  ivec2, ivec3, ivec4, uvec2, uvec3, uvec4, bvec2, bvec3, bvec4,
  mat2, mat2x3, mat2x4, mat3, mat3x2, mat3x4, mat4, mat4x2, mat4x3,
  If, For, While, Switch, Loop, Discard, Break, Continue, Return,
  uniform, attribute, varying, output, builtinPosition, builtinFragDepth,
  uniformRaw, attributeRaw, varyingRaw,
  isUniformNode, isAttributeNode, isVaryingNode,
  compileGLSLFn, compileWGSLFn, compileJSFn, wgslUniformLayout, uniformArray,
  add, sub, mul, div, mod, min, max, pow, pow2, pow3, pow4,
  abs, sign, floor, ceil, fract, round, trunc, radians, degrees,
  sqrt, inverseSqrt, inversesqrt, exp, log, exp2, log2, cbrt,
  negate, oneMinus, reciprocal, saturate, lengthSq, difference,
  sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh,
  normalize, length, distance, dot, cross, reflect, refract, faceForward,
  mix, clamp, step, smoothstep, all, any,
  and, or, not, xor, equal, notEqual, lessThan, greaterThan,
  lessThanEqual, greaterThanEqual,
  bitAnd, bitOr, bitXor, bitNot, shiftLeft, shiftRight,
  transpose, determinant, inverse, element,
  PI, TWO_PI, HALF_PI, EPSILON, INFINITY,
} from "./rmsl";
// Recording stand-ins for the real compilers: each returns the language it is
// named for and additionally compiles the program to the other, so every shader
// the tests generate is checked by a real GLSL and WGSL implementation in the
// afterAll below. Aliasing at the import means the tests need no changes.
import {
  recordingGLSL as compileGLSL,
  recordingWGSL as compileWGSL,
  recordShaderSource,
  assertRecordedShadersValid,
} from "./testing/shader-validity";

afterAll(async () => {
  await assertRecordedShadersValid();
}, 120_000);


describe("RMSL", () => {
  it("compiles a simple float expression to GLSL", () => {
    let prog = Fn(() => {
      let x = float(1.5).toVar();
      let y = float(2.0).toVar();
      let z = x.add(y).toVar();
      return z;
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("#version 300 es");
    expect(glsl).toContain("void main(void)");
    expect(glsl).toContain("float _rmsl_");
  });

  it("compiles a simple float expression to WGSL", () => {
    let prog = Fn(() => float(3.14).toVar());
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("fn main()");
    expect(wgsl).toContain("3.14f");
  });

  it("supports uniform declarations", () => {
    let prog = Fn(() => {
      let uTime = uniform("float");
      let uColor = uniform("vec3");
      return uTime.add(uColor.x);
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("uniform float");
    expect(glsl).toContain("uniform vec3");
  });

  it("uniform nodes have .name property", () => {
    let uTime = uniform("float");
    let uColor = uniform("vec3");
    let uMVP = uniform("mat4");
    expect(uTime.name).toMatch(/^_rmsl_u\d+$/);
    expect(uColor.name).toMatch(/^_rmsl_u\d+$/);
    expect(uMVP.name).toMatch(/^_rmsl_u\d+$/);
  });

  it("attribute nodes have .name property", () => {
    let pos = attribute("vec3");
    let tex = attribute("vec2");
    expect(pos.name).toBe("_rmsl_a0");
    expect(tex.name).toBe("_rmsl_a1");
  });

  it("varying nodes have .name property", () => {
    let v = varying("vec3");
    expect(v.name).toBe("_rmsl_v0");
  });

  it("type guards work for uniform nodes", () => {
    let u = uniform("float");
    expect(isUniformNode(u)).toBe(true);
    expect(isAttributeNode(u)).toBe(false);
    expect(isVaryingNode(u)).toBe(false);
  });

  it("type guards work for attribute nodes", () => {
    let a = attribute("vec3");
    expect(isAttributeNode(a)).toBe(true);
    expect(isUniformNode(a)).toBe(false);
    expect(isVaryingNode(a)).toBe(false);
  });

  it("type guards work for varying nodes", () => {
    let v = varying("vec3");
    expect(isVaryingNode(v)).toBe(true);
    expect(isUniformNode(v)).toBe(false);
    expect(isAttributeNode(v)).toBe(false);
  });

  it("compiles vec3 swizzles in GLSL", () => {
    let prog = Fn(() => {
      let a = vec3(1, 2, 3).toVar();
      return a.x.toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("vec3(1, 2, 3)");
    expect(glsl).toContain(".x");
  });

  it("compiles vec3 swizzles in WGSL", () => {
    let prog = Fn(() => vec3(4, 5, 6).xyz.toVar());
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("vec3<f32>(4, 5, 6)");
    expect(wgsl).toContain(".xyz");
  });

  it("compiles nested arithmetic", () => {
    let prog = Fn(() => {
      let a = float(10).toVar();
      let b = float(3).toVar();
      return a.div(b).add(float(1)).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("/");
    expect(glsl).toContain("+ 1");
  });

  it("supports Fn with multiple return values", () => {
    let prog = Fn(() => {
      let a = float(1.0).toVar();
      let b = float(2.0).toVar();
      return [a, b];
    });
    let [a, b] = prog();
    let glsl = compileGLSL([a, b]);
    expect(glsl).toContain("float _rmsl_");
  });

  it("compiles math built-ins", () => {
    let prog = Fn(() => {
      let x = float(0.5).toVar();
      return x.sin().add(x.cos()).add(x.abs()).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("sin(");
    expect(glsl).toContain("cos(");
    expect(glsl).toContain("abs(");
  });

  it("compiles If/Else to GLSL", () => {
    let prog = Fn(() => {
      let x = float(1.0).toVar();
      let cond = uniform("bool");
      If(cond, () => {
        x.assign(float(100.0));
      }).Else(() => {
        x.assign(float(200.0));
      });
      return x;
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("if (");
    expect(glsl).toContain("else {");
  });

  // === Phase 1.1: Comparison ops ===
  it("compiles float lessThan to GLSL", () => {
    let prog = Fn(() => float(1.0).lessThan(float(2.0)).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("1.0 < 2.0");
  });

  it("compiles float lessThan to WGSL", () => {
    let prog = Fn(() => float(1.0).lessThan(float(2.0)).toVar());
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("1f < 2f");
  });

  // Comparing vectors gives one boolean per component, so the result is a
  // bvecN rather than a bool. GLSL spells it with a function; WGSL uses the
  // operator directly. Both yield a boolean vector.
  it("compiles vec3 lessThan to GLSL (vector path)", () => {
    let prog = Fn(() => vec3(1,2,3).lessThan(vec3(4,5,6)).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("lessThan(");
    expect(glsl).toContain("bvec3 ");
  });

  it("compiles vec3 lessThan to WGSL (vector path)", () => {
    let prog = Fn(() => vec3(1,2,3).lessThan(vec3(4,5,6)).toVar());
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("vec3<bool>");
  });

  it("compiles vec3 equal/notEqual to GLSL", () => {
    let prog = Fn(() => vec3(1,2,3).equal(vec3(1,2,3)).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("equal(");
    expect(glsl).toContain("bvec3 ");
  });

  it("reduces a boolean vector with all()/any() in GLSL", () => {
    let prog = Fn(() => [
      vec3(1,2,3).lessThan(vec3(4,5,6)).all().toVar(),
      vec3(1,2,3).greaterThan(vec3(4,5,6)).any().toVar(),
    ]);
    let [a, b] = prog();
    let glsl = compileGLSL([a, b]);
    expect(glsl).toContain("all(lessThan(");
    expect(glsl).toContain("any(greaterThan(");
    expect(glsl).toContain("bool ");
  });

  it("reduces a boolean vector with all()/any() in WGSL", () => {
    let prog = Fn(() => vec3(1,2,3).lessThan(vec3(4,5,6)).all().toVar());
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("all(");
    expect(wgsl).toContain(": bool");
  });

  // GLSL's `!` is scalar-only, so negating a boolean vector needs not().
  it("negates a boolean vector component-wise in both backends", () => {
    let prog = Fn(() => vec3(1,2,3).lessThan(vec3(4,5,6)).not().toVar());
    expect(compileGLSL(prog())).toContain("not(lessThan(");
    expect(compileWGSL(prog())).toContain("!(");
  });

  // refract(I, N, eta) takes three arguments.
  it("compiles refract with all three arguments to GLSL", () => {
    let prog = Fn(() => vec3(1,0,0).refract(vec3(0,1,0), 0.5).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("refract(vec3(1, 0, 0), vec3(0, 1, 0), 0.5)");
  });

  it("compiles refract with all three arguments to WGSL", () => {
    let prog = Fn(() => vec3(1,0,0).refract(vec3(0,1,0), 0.5).toVar());
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("refract(vec3<f32>(1, 0, 0), vec3<f32>(0, 1, 0), 0.5f)");
  });

  // The update clause arrives as statements, so its work sits in `body` while
  // `expr` holds only a bare variable reference.
  it("emits the for-loop update clause in GLSL", () => {
    let prog = Fn(() => {
      let total = float(0).toVar();
      For(
        () => float(0).toVar(),
        (i) => i.lessThan(4),
        (i) => i.assign(i.add(1)),
        (i) => { total.assign(total.add(i)); },
      );
      return total;
    });
    let glsl = compileGLSL(prog());
    let header = glsl.split("\n").find(l => l.includes("for ("))!;
    expect(header).toMatch(/;\s*(\S+) = \1 \+ 1\.0\) \{$/);
  });

  it("emits the for-loop update clause in WGSL", () => {
    let prog = Fn(() => {
      let total = float(0).toVar();
      For(
        () => float(0).toVar(),
        (i) => i.lessThan(4),
        (i) => i.assign(i.add(1)),
        (i) => { total.assign(total.add(i)); },
      );
      return total;
    });
    let wgsl = compileWGSL(prog());
    let header = wgsl.split("\n").find(l => l.includes("for ("))!;
    expect(header).toMatch(/;\s*(\S+) = \1 \+ 1f\) \{$/);
  });

  // dot/length/distance reduce a vector to a scalar, so their result type is
  // float, not the operand type.
  it("types length/distance/dot as float, not the operand type", () => {
    let prog = Fn(() => {
      let a = vec2(3, 4).toVar();
      return [a.length().toVar(), a.distance(vec2(0, 0)).toVar(), a.dot(vec2(1, 1)).toVar()];
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("float");
    expect(glsl).not.toMatch(/vec2 \S+ = length\(/);
    expect(glsl).not.toMatch(/vec2 \S+ = distance\(/);
    expect(glsl).not.toMatch(/vec2 \S+ = dot\(/);
  });

  it("compares a distance against a float with an operator, not lessThan()", () => {
    let prog = Fn(() => vec2(3, 4).distance(vec2(0, 0)).lessThan(5.0).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("< 5.0");
    expect(glsl).not.toContain("lessThan(");
  });

  // === Phase 1.2: Texture sampling ===
  it("compiles texture sampling to GLSL", () => {
    let prog = Fn(() => {
      let tex = uniform("sampler2D");
      return tex.texture(vec2(0.5, 0.5)).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("uniform sampler2D");
    expect(glsl).toContain("texture(");
  });

  it("compiles texture sampling to WGSL", () => {
    let prog = Fn(() => {
      let tex = uniform("sampler2D");
      return tex.texture(vec2(0.5, 0.5)).toVar();
    });
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("texture_2d<f32>");
    expect(wgsl).toContain("textureSample(");
  });

  it("compiles textureLod to GLSL", () => {
    let prog = Fn(() => {
      let tex = uniform("sampler2D");
      return tex.textureLod(vec2(0.5, 0.5), float(0.0)).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("textureLod(");
  });

  // A 3D texture samples with a volume coordinate — a vec3, like a cube map —
  // and maps to sampler3D / texture_3d<f32>.
  it("compiles sampler3D sampling to GLSL", () => {
    let prog = Fn(() => {
      let tex = uniform("sampler3D");
      return tex.texture(vec3(0.5, 0.5, 0.5)).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("uniform sampler3D");
    expect(glsl).toContain("texture(");
  });

  it("compiles sampler3D textureLod to GLSL", () => {
    let prog = Fn(() => {
      let tex = uniform("sampler3D");
      return tex.textureLod(vec3(0.5, 0.5, 0.5), float(0.0)).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("uniform sampler3D");
    expect(glsl).toContain("textureLod(");
  });

  it("compiles sampler3D sampling to WGSL", () => {
    let prog = Fn(() => {
      let tex = uniform("sampler3D");
      return tex.texture(vec3(0.5, 0.5, 0.5)).toVar();
    });
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("texture_3d<f32>");
    expect(wgsl).toContain("textureSample(");
  });

  // === Phase 1.3: Matrix * vec3 ===
  it("compiles mat4 * vec3 with implicit promotion", () => {
    let prog = Fn(() => {
      let m = mat4(1);
      let v = vec3(1, 2, 3);
      return m.mul(v).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("vec4(");
    expect(glsl).toContain(".xyz");
  });

  // === Phase 1.4: Int/Uint comparison ===
  it("compiles int lessThan", () => {
    let prog = Fn(() => int(1).lessThan(int(2)).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("1 < 2");
  });

  // A plain JS number has no shader type of its own. A uniform is used because
  // constants fold away.
  it("gives a plain number the operand's integer type", () => {
    let prog = Fn(() => uniform("int").mod(2).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toMatch(/int \S+ = \S+ % 2;/);
    expect(glsl).not.toContain("float(");
    expect(compileWGSL(prog())).toMatch(/var \S+: i32 = \S+ % 2i;/);
  });

  it("keeps float literals float", () => {
    let prog = Fn(() => uniform("float").add(2).toVar());
    expect(compileGLSL(prog())).toContain("+ 2.0");
    expect(compileWGSL(prog())).toContain("+ 2f");
  });

  // === Phase 2: Constructor overloads ===
  it("vec3 scalar promotion", () => {
    let prog = Fn(() => vec3(1.0).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("vec3(1");
  });

  it("vec3 scalar promotion WGSL", () => {
    let prog = Fn(() => vec3(1.0).toVar());
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("vec3<f32>(1");
  });

  it("vec4 from vec3 + scalar", () => {
    let prog = Fn(() => vec4(vec3(1,2,3), 1.0).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("vec4(vec3(1, 2, 3), 1");
  });

  it("vec3 from vec4 truncation", () => {
    let prog = Fn(() => vec3(vec4(1,2,3,4)).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("vec3(vec4(");
  });

  it("float from int type casting", () => {
    let prog = Fn(() => float(int(5)).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("float(5)");
  });

  it("int from float type casting", () => {
    let prog = Fn(() => int(float(3.14)).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("int(3");
  });

  it("mat4 scalar constructor", () => {
    let prog = Fn(() => mat4(1.0).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("mat4(1");
  });

  it("mat4 scalar constructor WGSL", () => {
    let prog = Fn(() => mat4(1.0).toVar());
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("mat4x4<f32>(1");
  });

  // === Phase 3: Compiler Quality ===
  it("compileGLSL.vertex emits gl_Position", () => {
    let prog = Fn(() => {
      let mvp = uniform("mat4");
      let pos = attribute("vec3");
      // gl_Position is a vec4, so the stage result has to be one too.
      return vec4(mvp.mul(pos), 1.0);
    });
    let glsl = compileGLSL.vertex(prog());
    expect(glsl).toContain("gl_Position");
    expect(glsl).toContain("#version 300 es");
  });

  it("compileGLSL.fragment emits output", () => {
    let prog = Fn(() => float(1.0).toVar());
    let glsl = compileGLSL.fragment(prog());
    expect(glsl).toContain("#version 300 es");
  });

  it("compileWGSL.vertex emits @vertex and VertexOutput", () => {
    let prog = Fn(() => vec4(1,2,3,4));
    let wgsl = compileWGSL.vertex(prog());
    expect(wgsl).toContain("@vertex");
    expect(wgsl).toContain("VertexOutput");
    expect(wgsl).toContain("@builtin(position)");
  });

  it("compileWGSL.fragment emits @fragment and FragmentOutput", () => {
    // The struct only appears when there is something to return, so the stage
    // result has to be a vec4 for the implicit colour output to be emitted.
    let prog = Fn(() => vec4(1.0, 0.0, 0.0, 1.0).toVar());
    let wgsl = compileWGSL.fragment(prog());
    expect(wgsl).toContain("@fragment");
    expect(wgsl).toContain("FragmentOutput");
    expect(wgsl).toContain("_rmsl_fragColor");
  });

  it("output() creates output variable in GLSL fragment", () => {
    let prog = Fn(() => {
      let outColor = output("vec4");
      outColor.assign(vec4(1,0,0,1));
      return outColor;
    });
    let glsl = compileGLSL.fragment(prog());
    expect(glsl).toContain("layout(location=0)");
    expect(glsl).toContain("out vec4");
  });

  // GLSL ES 3.00 removed gl_FragColor, so a fragment shader with nowhere to
  // write its colour needs an explicit output declaration.
  it("declares an implicit colour output when a fragment shader has none", () => {
    let build = () => Fn(() => vec4(1, 0, 0, 1).toVar());

    let glsl = compileGLSL.fragment(build()());
    expect(glsl).toContain("layout(location=0) out vec4 _rmsl_fragColor;");
    expect(glsl).toMatch(/_rmsl_fragColor = \S+;/);

    let wgsl = compileWGSL.fragment(build()());
    expect(wgsl).toContain("_rmsl_fragColor");
  });

  // Declared outputs are assigned by the program.
  it("leaves an explicitly written output alone", () => {
    let prog = Fn(() => {
      let outColor = output("vec4");
      outColor.assign(vec4(1, 0, 0, 1));
      return vec4(0, 1, 0, 1);
    });
    let glsl = compileGLSL.fragment(prog());
    expect(glsl.match(/_rmsl_o\d+ = /g) ?? []).toHaveLength(1);
    expect(glsl).toContain("vec4(1, 0, 0, 1)");
  });

  it("does not assign a non-vec4 result to a declared output", () => {
    let prog = Fn(() => {
      let outColor = output("vec4");
      outColor.assign(vec4(1, 0, 0, 1));
      return float(2).toVar();
    });
    let glsl = compileGLSL.fragment(prog());
    expect(glsl.match(/_rmsl_o\d+ = /g) ?? []).toHaveLength(1);
  });

  // Everything a vertex stage passes onward shares one set of numbered slots.
  it("numbers everything a vertex stage passes on without collision", () => {
    let build = () => Fn(() => {
      let v = varying("vec2");
      v.assign(vec2(1, 2));
      let o = output("vec4");
      o.assign(vec4(1, 0, 0, 1));
      return vec4(0, 0, 0, 1);
    });

    let wgsl = compileWGSL.vertex(build()());
    let slots = [...wgsl.matchAll(/@location\((\d+)\)/g)].map(m => m[1]);
    expect(slots.length, "two values are passed on").toBe(2);
    expect(new Set(slots).size, "each slot used once").toBe(slots.length);

    // A location qualifier belongs on a fragment output. GLSL ES 3.00 does not
    // allow one on a vertex output, whatever the number.
    expect(compileGLSL.vertex(build()())).not.toMatch(/layout\(location=\d+\) out/);
  });

  it("output() creates output in WGSL fragment", () => {
    let prog = Fn(() => {
      let outColor = output("vec4");
      outColor.assign(vec4(1,0,0,1));
      return outColor;
    });
    let wgsl = compileWGSL.fragment(prog());
    expect(wgsl).toMatch(/@location\(\d+\)/);
  });

  it("builtinPosition() maps to gl_Position in GLSL vertex", () => {
    let prog = Fn(() => {
      let pos = builtinPosition();
      return pos;
    });
    let glsl = compileGLSL.vertex(prog());
    expect(glsl).toContain("gl_Position");
  });

  it("builtinPosition() maps to position in WGSL vertex", () => {
    let prog = Fn(() => {
      let pos = builtinPosition();
      return pos;
    });
    let wgsl = compileWGSL.vertex(prog());
    expect(wgsl).toContain("position");
  });

  it("builtinFragDepth() maps to gl_FragDepth in GLSL fragment", () => {
    let prog = Fn(() => {
      let fd = builtinFragDepth();
      fd.assign(float(0.5));
      return float(1.0);
    });
    let glsl = compileGLSL.fragment(prog());
    expect(glsl).toContain("gl_FragDepth");
    expect(glsl).toContain("gl_FragDepth = 0.5");
  });

  it("builtinFragDepth() maps to @builtin(frag_depth) in WGSL fragment", () => {
    let prog = Fn(() => {
      let fd = builtinFragDepth();
      fd.assign(float(0.5));
      return float(1.0);
    });
    let wgsl = compileWGSL.fragment(prog());
    expect(wgsl).toContain("@builtin(frag_depth)");
    expect(wgsl).toContain("_rmsl_fragDepth = 0.5");
  });

  it("builtinFragDepth() throws in vertex shader", () => {
    // A valid position, so the depth builtin is the only thing left to object
    // to — otherwise this would be refused for two reasons at once.
    let prog = Fn(() => vec4(builtinFragDepth(), 0, 0, 1).toVar());
    expect(() => compileGLSL.vertex(prog())).toThrow("builtinFragDepth");
    expect(() => compileWGSL.vertex(prog())).toThrow("builtinFragDepth");
  });

  it("varying is out in vertex, in in fragment GLSL", () => {
    let prog = Fn(() => {
      let v = varying("vec3");
      // A vertex shader's result is its position, so it has to be a vec4.
      return vec4(v.x, v.y, v.z, 1.0);
    });
    let vertexGLSL = compileGLSL.vertex(prog());
    let fragmentGLSL = compileGLSL.fragment(prog());
    expect(vertexGLSL).toContain("out vec3");
    expect(fragmentGLSL).toContain("in vec3");
  });

  // Skipping the write instead would link cleanly and draw nothing.
  it("rejects a vertex shader whose result is not a vec4", () => {
    let prog = Fn(() => vec3(1, 2, 3).toVar());
    // @ts-expect-error a vec3 cannot become a position
    expect(() => compileGLSL.vertex(prog())).toThrow(/vertex shader/);
    // @ts-expect-error a vec3 cannot become a position
    expect(() => compileWGSL.vertex(prog())).toThrow(/vertex shader/);
  });

  // A vertex shader must produce a position, and zero is not a valid position.
  it("rejects a vertex result of zero, like any other non-position", () => {
    // @ts-expect-error a float cannot become a position, zero included
    expect(() => compileGLSL.vertex(Fn(() => float(0))()))
      .toThrow(/vertex shader/);
    // @ts-expect-error a float cannot become a position, zero included
    expect(() => compileWGSL.vertex(Fn(() => float(0))()))
      .toThrow(/vertex shader/);
  });

  it("rejects a vertex result that folds to zero", () => {
    // @ts-expect-error a folded zero is still a float
    expect(() => compileGLSL.vertex(Fn(() => float(2).sub(float(2)))()))
      .toThrow(/vertex shader/);
  });

  // Writing the position explicitly is the documented way to set it.
  it("allows a vertex shader that writes the position itself", () => {
    let build = () => Fn(() => {
      builtinPosition().assign(vec4(1, 2, 3, 4));
    });
    let glsl = compileGLSL.vertex(build()());
    expect(glsl.match(/gl_Position = /g) ?? [], "written once").toHaveLength(1);
    expect(glsl).toContain("vec4(1, 2, 3, 4)");

    let wgsl = compileWGSL.vertex(build()());
    expect(wgsl.match(/result\.position = /g) ?? [], "written once").toHaveLength(1);
  });

  // With no explicit write, the result still has to be able to become one.
  it("still requires a position when the shader writes none", () => {
    // @ts-expect-error a float cannot become a position
    expect(() => compileGLSL.vertex(Fn(() => float(1))()))
      .toThrow(/vertex shader/);
  });

  // A vertex shader that returns nothing and never writes a position fails
  // validation.
  it("rejects a vertex shader that produces no position at all", () => {
    let prog = Fn(() => { float(1).toVar(); });
    expect(() => compileGLSL.vertex(prog())).toThrow(/vertex shader/);
    expect(() => compileWGSL.vertex(prog())).toThrow(/vertex shader/);
  });

  // A fragment shader with no colour output is legal, so this stays permitted.
  it("allows a fragment shader with no output", () => {
    let prog = Fn(() => float(3.14).toVar());
    expect(() => compileGLSL.fragment(prog())).not.toThrow();
    expect(compileGLSL.fragment(prog())).toContain("3.14");
  });

  it("For loop init hoists declaration into for header in GLSL", () => {
    let prog = Fn(() => {
      For(
        () => int(0).toVar(),
        (i) => i.lessThan(int(10)),
        (i) => {},
        (i) => {},
      );
      return float(1.0);
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("for (int _rmsl_");
  });

  // === Phase 4: WGSL Polish ===
  it("WGSL vertex has VertexInput struct with attributes", () => {
    let prog = Fn(() => {
      let pos = attribute("vec3");
      let mvp = uniform("mat4");
      // @builtin(position) is a vec4, so the stage result has to be one too.
      return vec4(mvp.mul(pos), 1.0);
    });
    let wgsl = compileWGSL.vertex(prog());
    expect(wgsl).toContain("struct VertexInput");
    expect(wgsl).toContain("@location(0)");
    expect(wgsl).toMatch(/input\._rmsl_a\d/);
  });

  it("WGSL uniforms use @group(0), textures @group(1), samplers @group(2)", () => {
    let prog = Fn(() => {
      let u = uniform("float");
      let tex = uniform("sampler2D");
      return tex.texture(vec2(0, 0)).add(u);
    });
    let wgsl = compileWGSL.fragment(prog());
    expect(wgsl).toContain("@group(0) @binding(0) var<uniform>");
    expect(wgsl).toContain("@group(1) @binding(0) var ");
    expect(wgsl).toContain("@group(2) @binding(0) var ");
  });

  it("WGSL gives a sampler3D a texture binding and a companion sampler", () => {
    let prog = Fn(() => {
      let tex = uniform("sampler3D");
      return tex.texture(vec3(0, 0, 0));
    });
    let wgsl = compileWGSL.fragment(prog());
    expect(wgsl).toMatch(/@group\(1\) @binding\(0\) var \S+: texture_3d<f32>;/);
    expect(wgsl).toMatch(/@group\(2\) @binding\(0\) var \S+_s: sampler;/);
    expect(wgsl).toContain("textureSample(");
  });

  it("plain number literals are float in WGSL", () => {
    let prog = Fn(() => {
      let x = float(5).toVar();
      return x.mod(2).toVar();
    });
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("2f");
    expect(wgsl).not.toContain("f32(");
  });

  // === Phase 5: Node System Gaps ===
  it("swizzle write via assign works in GLSL", () => {
    let prog = Fn(() => {
      let a = vec3(1, 2, 3).toVar();
      let b = vec3(4, 5, 6).toVar();
      a.xy.assign(b.xy);
      return a;
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain(".xy = ");
  });

  it("swizzle write via assign works in WGSL", () => {
    let prog = Fn(() => {
      let a = vec3(1, 2, 3).toVar();
      let b = vec3(4, 5, 6).toVar();
      a.xy.assign(b.xy);
      return a;
    });
    let wgsl = compileWGSL(prog());
    // WGSL cannot assign to a multi-component swizzle, so the write is split
    // into one assignment per component.
    expect(wgsl).toContain(".x = ");
    expect(wgsl).toContain(".y = ");
    expect(wgsl).not.toContain(".xy = ");
  });

  it("break_ compiles in GLSL", () => {
    let prog = Fn(() => {
      For(
        () => int(0).toVar(),
        (i) => i.lessThan(int(10)),
        (i) => {},
        (i) => {
          Break();
        },
      );
      return float(1.0);
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("break;");
  });

  it("continue_ compiles in WGSL", () => {
    let prog = Fn(() => {
      For(
        () => int(0).toVar(),
        (i) => i.lessThan(int(10)),
        (i) => {},
        (i) => {
          Continue();
        },
      );
      return float(1.0);
    });
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("continue;");
  });

  it("mat2 constructor compiles to GLSL", () => {
    let prog = Fn(() => mat2(1, 0, 0, 1).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("mat2(1, 0, 0, 1)");
  });

  it("mat2x3 constructor compiles to GLSL", () => {
    let prog = Fn(() => mat2x3(1,0,0,0,1,0).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("mat2x3(");
  });

  // === Phase 6: Infrastructure — Comprehensive coverage ===

  // -- All FloatMathOps (constant-folded with literals) --
  it.each([
    "sin", "cos", "tan", "asin", "acos", "atan",
    "abs", "sign", "floor", "ceil", "fract",
    "sqrt", "inversesqrt", "exp", "log", "exp2", "log2",
  ])("float.%s() compiles to GLSL (constant-folded)", (op) => {
    let prog = Fn(() => (float(0.5) as any)[op]().toVar());
    let glsl = compileGLSL(prog());
    // With constant folding, the result is a literal, not a function call
    expect(glsl).not.toContain(op);
    expect(glsl).toMatch(/_rmsl_\d+ = /);
  });

  it("float.pow() compiles to GLSL (constant-folded)", () => {
    let glsl = compileGLSL(Fn(() => float(2.0).pow(float(3.0)).toVar())());
    expect(glsl).toContain("= 8");
  });

  it("float.min()/max()/mod() compile to GLSL (constant-folded)", () => {
    let glsl1 = compileGLSL(Fn(() => float(5).min(float(3)).toVar())());
    expect(glsl1).toContain("= 3");
    let glsl2 = compileGLSL(Fn(() => float(5).max(float(3)).toVar())());
    expect(glsl2).toContain("= 5");
    let glsl3 = compileGLSL(Fn(() => float(5).mod(float(3)).toVar())());
    expect(glsl3).toContain("= 2");
  });

  // -- VecCommonOps --
  it.each([
    "dot", "length", "normalize", "distance",
  ])("vec3.%s() compiles to GLSL", (op) => {
    let prog = Fn(() => {
      let a = vec3(1,2,3).toVar();
      return (a as any)[op](op === "dot" || op === "distance" ? a : undefined).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain(op);
  });

  // Neither language compares a vector against a scalar, so the scalar has to
  // be broadcast: GLSL has no lessThan(vec3, float) and WGSL no
  // `operator < (vec3<f32>, f32)`. The signatures accept the mix.
  it("broadcasts a scalar compared against a vector", () => {
    let prog = Fn(() => vec3(1, 2, 3).lessThan(uniform("float")).all().toVar());
    expect(compileGLSL(prog())).toMatch(/lessThan\(vec3\(1, 2, 3\), vec3\(\S+\)\)/);
    expect(compileWGSL(prog())).toMatch(/vec3<f32>\(1, 2, 3\) < vec3<f32>\(\S+\)/);
  });

  // A scalar compared against a vector is a type error — see
  // src/rmsl.test-d.ts, where that belongs. This direction is the allowed one:
  // the scalar broadcasts, which is what the caller means and what both
  // languages need.
  it("allows a vector compared against a scalar", () => {
    let prog = Fn(() => vec3(1, 2, 3).lessThan(uniform("float")).all().toVar());
    expect(compileGLSL(prog())).toContain("lessThan(");
  });

  // step/smoothstep take the value last, so the result type follows that
  // operand rather than the edge — `vec3.step(0.5)` is a vec3, not a float.
  it("types step/smoothstep from the value, not the edge", () => {
    let stepped = Fn(() => uniform("vec3").step(0.5).toVar());
    let smoothed = Fn(() => uniform("vec3").smoothstep(0.0, 1.0).toVar());
    expect(compileGLSL(stepped())).toMatch(/vec3 \S+ = step\(/);
    expect(compileGLSL(smoothed())).toMatch(/vec3 \S+ = smoothstep\(/);
    expect(compileWGSL(stepped())).toMatch(/var \S+: vec3<f32> = step\(/);
  });

  it("vec3.reflect/refract/clamp/mix/step/smoothstep compile to GLSL", () => {
    let v = vec3(1,2,3);
    let n = vec3(0,1,0);
    expect(compileGLSL(Fn(() => v.reflect(n).toVar())())).toContain("reflect");
    expect(compileGLSL(Fn(() => v.refract(n, float(0.5)).toVar())())).toContain("refract");
    let zero = vec3(0,0,0);
    let one = vec3(1,1,1);
    expect(compileGLSL(Fn(() => v.clamp(zero, one).toVar())())).toContain("clamp");
    expect(compileGLSL(Fn(() => v.mix(one, float(0.5)).toVar())())).toContain("mix");
    expect(compileGLSL(Fn(() => v.step(one).toVar())())).toContain("step");
    expect(compileGLSL(Fn(() => v.smoothstep(one, zero).toVar())())).toContain("smoothstep");
  });

  it("mix/clamp/smoothstep emit 3 args in GLSL", () => {
    let glsl_mix = compileGLSL(Fn(() => vec3(1,0,0).mix(vec3(0,1,0), float(0.5)).toVar())());
    expect(glsl_mix).toContain("mix(");
    expect(glsl_mix).not.toContain("mix(vec3(1, 0, 0), vec3(0, 1, 0))");
    let glsl_clamp = compileGLSL(Fn(() => float(0.5).clamp(float(0), float(1)).toVar())());
    expect(glsl_clamp).toContain("clamp(");
    expect(glsl_clamp).toContain("0.0, 1.0");
    let glsl_ss = compileGLSL(Fn(() => float(0.5).smoothstep(float(0), float(1)).toVar())());
    expect(glsl_ss).toMatch(/smoothstep\(0\.0, 1\.0, 0\.5/);
  });

  it("Fn return type is not void for float expression", () => {
    let prog = Fn(() => {
      let x = float(1.5).toVar();
      return x;
    });
    let result = prog();
    expect((result as any)._t).toBe("float");
  });

  it("vec3.cross compiles to GLSL", () => {
    let glsl = compileGLSL(Fn(() => vec3(1,0,0).cross(vec3(0,1,0)).toVar())());
    expect(glsl).toContain("cross");
  });

  // -- MatOps --
  it("mat3.inverse/transpose compiles to GLSL", () => {
    let m = mat3(1,0,0,0,1,0,0,0,1);
    expect(compileGLSL(Fn(() => m.inverse().toVar())())).toContain("inverse");
    expect(compileGLSL(Fn(() => m.transpose().toVar())())).toContain("transpose");
  });

  it("mat4.inverse/transpose/mul compiles to GLSL", () => {
    let m = mat4(1);
    let v = vec4(1,2,3,4);
    expect(compileGLSL(Fn(() => m.inverse().toVar())())).toContain("inverse");
    expect(compileGLSL(Fn(() => m.transpose().toVar())())).toContain("transpose");
    expect(compileGLSL(Fn(() => m.mul(v).toVar())())).toContain("*");
  });

  // -- IntOps --
  it.each([
    ["add", 8], ["sub", 2], ["mul", 15], ["div", 1], ["mod", 2],
  ])("int.%s() compiles to GLSL (constant-folded)", (op, expected) => {
    let prog = Fn(() => (int(5) as any)[op](int(3)).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain(String(expected));
  });

  it.each([
    ["bitAnd", "&"], ["bitOr", "|"], ["bitXor", "^"], ["shiftLeft", "<<"], ["shiftRight", ">>"],
  ])("int.%s() compiles to GLSL (bitwise)", (op, expected) => {
    let prog = Fn(() => (int(5) as any)[op](int(3)).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain(expected as string);
  });

  // -- BoolOps --
  it("bool.and/or/not compile to GLSL", () => {
    let t = bool(true);
    let f = bool(false);
    expect(compileGLSL(Fn(() => t.and(f).toVar())())).toContain("&&");
    expect(compileGLSL(Fn(() => t.or(f).toVar())())).toContain("||");
    expect(compileGLSL(Fn(() => t.not().toVar())())).toContain("!");
  });

  // -- Type literals --
  it("int/uint/bool/vec2 literals compile to GLSL", () => {
    expect(compileGLSL(Fn(() => int(42).toVar())())).toContain("42");
    expect(compileGLSL(Fn(() => vec2(1,2).toVar())())).toContain("vec2(1, 2)");
  });

  it("int/uint/bool/vec2 literals compile to WGSL", () => {
    expect(compileWGSL(Fn(() => int(42).toVar())())).toContain("42i");
    expect(compileWGSL(Fn(() => vec2(1,2).toVar())())).toContain("vec2<f32>(1, 2)");
  });

  // -- Control flow: While, discard --
  it("While loop compiles to GLSL", () => {
    let prog = Fn(() => {
      let i = int(0).toVar();
      While(i.lessThan(int(5)), () => {
        i.assign(i.add(int(1)));
      });
      return float(1.0);
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("while (");
  });

  it("discard compiles to GLSL", () => {
    let prog = Fn(() => {
      If(bool(true), () => { Discard(); });
      return float(1.0);
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("discard;");
  });

  it("discard compiles to WGSL", () => {
    let prog = Fn(() => {
      If(bool(true), () => { Discard(); });
      return float(1.0);
    });
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("discard;");
  });

  // -- Edge cases --
  it("empty Fn does not throw", () => {
    expect(() => { Fn(() => {}); }).not.toThrow();
  });

  it("single-statement Fn compiles", () => {
    expect(() => compileGLSL(Fn(() => float(1.0))())).not.toThrow();
  });

  it("nested Fn compiles", () => {
    let inner = Fn(() => float(2.0));
    let innerVal = inner();
    let outer = Fn(() => innerVal.add(float(1.0)).toVar());
    expect(() => compileGLSL(outer())).not.toThrow();
  });

  // -- Error cases --
  it("assign outside Fn throws", () => {
    let x = float(1.0);
    expect(() => x.assign(float(2.0))).toThrow("assign must be called inside");
  });

  it("toVar outside Fn throws", () => {
    let x = float(1.0);
    expect(() => x.toVar()).toThrow("toVar must be called inside");
  });

  it("If outside Fn throws", () => {
    expect(() => If(bool(true), () => {})).toThrow("must be called inside");
  });

  // === Standalone function compilers ===
  //
  // compileGLSLFn/compileWGSLFn and uniformRaw emit a function rather than a
  // whole shader, so each result is embedded in a minimal shader and validated.

  it("compileGLSLFn emits a named function with typed params", () => {
    let glsl = compileGLSLFn((a: any, b: any) => a.add(b).sin(), {
      name: "myFunc",
      params: [{ name: "a", type: "float" }, { name: "b", type: "float" }],
    });
    expect(glsl).toContain("float myFunc(float a, float b)");
    expect(glsl).toContain("return sin(a + b);");

    recordShaderSource("glsl", "fragment", `#version 300 es
precision highp float;
layout(location=0) out vec4 outColor;
${glsl}
void main(void) { outColor = vec4(myFunc(1.0, 2.0)); }`);
  });

  it("compileWGSLFn emits a named function with typed params", () => {
    let wgsl = compileWGSLFn((a: any, b: any) => a.add(b).sin(), {
      name: "myFunc",
      params: [{ name: "a", type: "float" }, { name: "b", type: "float" }],
    });
    expect(wgsl).toContain("fn myFunc(a: f32, b: f32) -> f32");
    expect(wgsl).toContain("return sin(a + b);");

    recordShaderSource("wgsl", "fragment", `${wgsl}
@fragment fn main() -> @location(0) vec4<f32> {
  return vec4<f32>(myFunc(1.0, 2.0));
}`);
  });

  // uniformRaw names its own slot, unlike uniform() which generates one.
  it("uniformRaw declares a custom-named uniform alongside the function", () => {
    let glsl = compileGLSLFn((v: any) => v.mul(uniformRaw("uScale", "float")), {
      name: "scale",
      params: [{ name: "v", type: "float" }],
    });
    expect(glsl).toContain("uniform float uScale;");
    expect(glsl).toContain("float scale(float v)");
    expect(uniformRaw("uOther", "vec3").name).toBe("uOther");

    recordShaderSource("glsl", "fragment", `#version 300 es
precision highp float;
layout(location=0) out vec4 outColor;
${glsl}
void main(void) { outColor = vec4(scale(2.0)); }`);
  });

  // attributeRaw / varyingRaw name their own slots, as uniformRaw does.
  it("attributeRaw declares a custom-named attribute", () => {
    let tex = attributeRaw("tex", "vec2");
    expect(tex.name).toBe("tex");
    let prog = Fn(() => vec4(tex.x, tex.y, 0, 1));
    expect(compileGLSL.vertex(prog())).toContain("in vec2 tex;");
    expect(compileWGSL.vertex(prog())).toContain("tex: vec2<f32>");
  });

  it("varyingRaw emits its name in both stages", () => {
    let v = varyingRaw("myNormal", "vec3");
    expect(v.name).toBe("myNormal");
    let vertex = Fn(() => {
      v.assign(vec3(0, 1, 0));
      return vec4(0, 0, 0, 1);
    });
    let fragment = Fn(() => vec4(v, 1));
    expect(compileGLSL.vertex(vertex())).toContain("out vec3 myNormal;");
    expect(compileGLSL.fragment(fragment())).toContain("in vec3 myNormal;");
  });

  it("two raw varyings get distinct locations both stages agree on", () => {
    let a = varyingRaw("aVal", "vec3");
    let b = varyingRaw("bVal", "vec4");
    let vertex = Fn(() => {
      a.assign(vec3(1, 0, 0));
      b.assign(vec4(0, 1, 0, 1));
      return vec4(0, 0, 0, 1);
    });
    let fragment = Fn(() => vec4(a, 1).add(b).toVar());
    const locations = (src: string) =>
      Object.fromEntries(
        [...src.matchAll(/@location\((\d+)\) (\w+):/g)].map((m) => [m[2], Number(m[1])]),
      );
    let vloc = locations(compileWGSL.vertex(vertex()));
    let floc = locations(compileWGSL.fragment(fragment()));
    expect(vloc.aVal).not.toBe(vloc.bVal);
    expect(vloc.aVal).toBe(floc.aVal);
    expect(vloc.bVal).toBe(floc.bVal);
  });

  it("raw attribute and varying feed the JS backend by name", () => {
    let vertex = Fn(() => {
      let uv = attributeRaw("tex", "vec2");
      let n = varyingRaw("myNormal", "vec3");
      n.assign(vec3(uv.x, uv.y, 1));
      return vec4(0, 0, 0, 1);
    });
    let src = compileJSFn(() => vertex(), { name: "main", params: [], stage: "vertex" });
    expect(src).toContain('ctx.attributes["tex"]');
    expect(src).toContain('res.varyings["myNormal"]');
  });

  // Refused by the signature as well as at runtime. Written as a directive
  // rather than a cast so both are asserted: a cast would silence the whole
  // expression, and would keep passing if the signature ever stopped refusing.
  it("compileGLSLFn rejects a multi-return function", () => {
    expect(() =>
      // @ts-expect-error a function compiled on its own returns a single value
      compileGLSLFn(() => [float(1), float(2)], { name: "bad", params: [] }),
    ).toThrow(/multi-return/);
  });

  // WGSL has no inverse() builtin, so the compiler writes one out on demand.
  it("compileWGSLFn emits the helpers the function body calls", () => {
    let wgsl = compileWGSLFn((m: any) => m.inverse(), {
      name: "invert",
      params: [{ name: "m", type: "mat4" }],
    });
    expect(wgsl).toContain("_rmsl_inverse4(m)");
    expect(wgsl).toContain("fn _rmsl_inverse4(");

    recordShaderSource("wgsl", "fragment", `${wgsl}
@fragment fn main() -> @location(0) vec4<f32> {
  return invert(mat4x4<f32>(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f))[0];
}`);
  });

  it("compileWGSLFn emits the mat3 helper for a mat3 operand", () => {
    let wgsl = compileWGSLFn((m: any) => m.inverse(), {
      name: "invert3",
      params: [{ name: "m", type: "mat3" }],
    });
    expect(wgsl).toContain("fn _rmsl_inverse3(");
    expect(wgsl).not.toContain("fn _rmsl_inverse4(");
  });

  // === Breadth coverage ===
  //
  // These are exported API, and going through the recording compilers means
  // both backends are checked by a real implementation as well as by the
  // assertion.

  // Driven by a uniform because constant folding collapses these on literals.
  it("compiles every unary math builtin", () => {
    let ops = [
      "sin", "cos", "tan", "asin", "acos", "atan", "abs", "sign",
      "floor", "ceil", "fract", "sqrt", "inversesqrt",
      "exp", "log", "exp2", "log2",
    ] as const;
    for (let op of ops) {
      let prog = Fn(() => (uniform("float") as any)[op]().toVar());
      expect(compileGLSL(prog()), op).toContain(op + "(");
      // Names differ between backends (inversesqrt vs inverseSqrt), so WGSL is
      // left to the validator rather than matched on text.
      expect(() => compileWGSL(prog()), op).not.toThrow();
    }
  });

  it("compiles every matrix type, including the non-square ones", () => {
    let cases: [string, (...v: number[]) => any, number][] = [
      ["mat2", mat2, 4], ["mat2x3", mat2x3, 6], ["mat2x4", mat2x4, 8],
      ["mat3x2", mat3x2, 6], ["mat3", mat3, 9], ["mat3x4", mat3x4, 12],
      ["mat4x2", mat4x2, 8], ["mat4x3", mat4x3, 12], ["mat4", mat4, 16],
    ];
    for (let [name, ctor, count] of cases) {
      let values = Array.from({ length: count }, (_, i) => i);
      let prog = Fn(() => ctor(...values).toVar());
      expect(compileGLSL(prog()), name).toContain(name + "(");
      expect(compileWGSL(prog()), name).toContain("<f32>(");
    }
  });

  // GLSL reads a lone scalar as a diagonal, so WGSL — which has no such
  // overload — has to write every component out. The expansion has to know the
  // shape: a matCxR is C columns of R rows.
  it("expands a scalar matrix constructor to a full diagonal in WGSL", () => {
    let cases: [string, any, string][] = [
      ["mat2", mat2, "mat2x2<f32>(2f, 0f, 0f, 2f)"],
      ["mat3", mat3, "mat3x3<f32>(2f, 0f, 0f, 0f, 2f, 0f, 0f, 0f, 2f)"],
      ["mat2x3", mat2x3, "mat2x3<f32>(2f, 0f, 0f, 0f, 2f, 0f)"],
      ["mat3x2", mat3x2, "mat3x2<f32>(2f, 0f, 0f, 2f, 0f, 0f)"],
      ["mat4x2", mat4x2, "mat4x2<f32>(2f, 0f, 0f, 2f, 0f, 0f, 0f, 0f)"],
    ];
    for (let [name, ctor, want] of cases) {
      let prog = Fn(() => ctor(2.0).toVar());
      expect(compileWGSL(prog()), name).toContain(want);
      // GLSL keeps the scalar form, which is what it means natively.
      expect(compileGLSL(prog()), name).toContain(name + "(2.0)");
    }
  });

  // A lone argument is only a diagonal when it is a scalar. Building a matrix
  // from another matrix is a copy/truncate.
  it("keeps a matrix-from-matrix constructor as a single argument", () => {
    // WGSL uniforms are members of one struct, so a reference is qualified —
    // the assertion is that there is one argument, whatever it is spelled.
    let prog3 = Fn(() => mat3(uniform("mat3")).toVar());
    expect(compileWGSL(prog3())).toMatch(/mat3x3<f32>\(_rmsl_uniforms\._rmsl_u\d+\)/);
    expect(compileGLSL(prog3())).toMatch(/mat3\(_rmsl_u\d+\)/);

    let prog4 = Fn(() => mat4(uniform("mat4")).toVar());
    expect(compileWGSL(prog4())).toMatch(/mat4x4<f32>\(_rmsl_uniforms\._rmsl_u\d+\)/);
  });

  it("compiles every swizzle accessor", () => {
    let single = ["x", "y", "z", "w", "r", "g", "b", "a"];
    let pairs = ["xy", "xz", "xw", "yz", "yw", "zw"];
    let triples = ["xyz", "xyw", "xzw", "yzw", "rgb"];
    for (let s of [...single, ...pairs, ...triples, "rgba"]) {
      let prog = Fn(() => (uniform("vec4") as any)[s].toVar());
      expect(compileGLSL(prog()), s).toContain("." + s);
      expect(compileWGSL(prog()), s).toContain("." + s);
    }
  });

  // Raw JS arrays are accepted wherever a node is, and their length picks the
  // type.
  it("wraps raw arrays by length", () => {
    // Widths have to match the receiver — neither language multiplies a vec4 by
    // a vec3.
    let cases: [() => any, string, string][] = [
      [() => vec2(1, 1).mul([1, 2]), "vec2", "vec2<f32>"],
      [() => vec3(1, 1, 1).mul([1, 2, 3]), "vec3", "vec3<f32>"],
      [() => vec4(1, 1, 1, 1).mul([1, 2, 3, 4]), "vec4", "vec4<f32>"],
    ];
    for (let [build, glslType, wgslType] of cases) {
      let prog = Fn(() => build().toVar());
      expect(compileGLSL(prog()), glslType).toContain(glslType + "(");
      expect(compileWGSL(prog()), wgslType).toContain(wgslType + "(");
    }
  });

  // === Shared nodes ===
  //
  // The graph is a DAG, not a tree: `Fn` returning an array gives every element
  // the whole block scope, so a node reachable from two roots is traversed
  // twice. Emitting it twice duplicates its side effects, and suppressing only
  // its declaration leaves the second copy referring to a name that no longer
  // exists. Each shared node has to be emitted exactly once.

  it("emits a shared If body once, not once per root", () => {
    let build = () => Fn(() => {
      let acc = float(0).toVar();
      If(bool(true), () => { acc.assign(acc.add(10)); });
      return [acc, acc.mul(2)];
    });
    let glsl = compileGLSL(build()());
    expect(glsl.match(/if \(true\)/g) ?? []).toHaveLength(1);
    expect(glsl.match(/\+ 10\.0/g) ?? []).toHaveLength(1);

    let wgsl = compileWGSL(build()());
    expect(wgsl.match(/if \(true\)/g) ?? []).toHaveLength(1);
    expect(wgsl.match(/\+ 10f/g) ?? []).toHaveLength(1);
  });

  // A toVar() inside a shared block declares its variable in that block. Emit
  // the block twice while declaring only once and the second copy reads a name
  // scoped to the first — `_rmsl_2` used outside the `if` that declared it.
  it("keeps a variable declared inside a shared block in scope", () => {
    let build = () => Fn(() => {
      let a = float(1).toVar();
      If(bool(true), () => { let b = a.add(2).toVar(); a.assign(b); });
      return [a, a.mul(2)];
    });
    let glsl = compileGLSL(build()());
    expect(glsl.match(/if \(true\)/g) ?? []).toHaveLength(1);
    // The declaration and its only use are both inside the single block.
    let inner = glsl.slice(glsl.indexOf("if (true)"));
    let declared = inner.match(/float (_rmsl_\d+) = _rmsl_\d+ \+ 2\.0;/);
    expect(declared, "inner declaration emitted").not.toBeNull();
    expect(inner.match(new RegExp(declared![1], "g")) ?? []).toHaveLength(2);

    expect(compileWGSL(build()()).match(/if \(true\)/g) ?? []).toHaveLength(1);
  });

  // The loop counter is declared by the for header. Emitting the loop twice
  // while suppressing the declaration produced `for (_rmsl_4; ...)`, whose
  // counter belongs to the first loop's scope.
  it("emits a shared For loop once, with its init intact", () => {
    let build = () => Fn(() => {
      let total = float(0).toVar();
      For(
        () => float(0).toVar(),
        (i) => i.lessThan(4),
        (i) => i.assign(i.add(1)),
        (i) => { total.assign(total.add(i)); },
      );
      return [total, total.mul(2)];
    });
    let glsl = compileGLSL(build()());
    expect(glsl.match(/for \(/g) ?? []).toHaveLength(1);
    expect(glsl).toMatch(/for \(float _rmsl_\d+ = 0\.0;/);

    let wgsl = compileWGSL(build()());
    expect(wgsl.match(/for \(/g) ?? []).toHaveLength(1);
    expect(wgsl).toMatch(/for \(var _rmsl_\d+: f32 = 0f;/);
  });

  // A floored modulus has to be written out for WGSL, whose % truncates. Doing
  // that inline repeats both operands twice each, so the cost of an expensive
  // operand is paid four times over. A helper is a plain expression, so it fits
  // anywhere the operator did — including a loop header, where a bound
  // temporary could not go.
  it("emits a modulus helper rather than repeating its operands", () => {
    let prog = Fn(() => uniform("float").mod(2).toVar());
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("fn _rmsl_mod_float(");

    let main = wgsl.slice(wgsl.indexOf("@fragment"));
    expect(main).toMatch(/_rmsl_mod_float\(\S+, 2f\)/);
    expect(main.match(/_rmsl_u\d+/g) ?? [], "operand used once").toHaveLength(1);
  });

  it("uses the width-matched modulus helper for vectors", () => {
    let prog = Fn(() => uniform("vec3").mod(2).toVar());
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("fn _rmsl_mod_vec3(");
    expect(wgsl).toContain("_rmsl_mod_vec3(");
    // GLSL keeps its own builtin, which needs no helper.
    expect(compileGLSL(prog())).toContain("mod(");
  });

  // A loop's update clause can do more than one thing. GLSL joins them with
  // the comma operator; WGSL's grammar allows a single statement there.
  function twoStepLoop(counterFirst: boolean) {
    return Fn(() => {
      let other = float(0).toVar();
      let total = float(0).toVar();
      For(
        () => float(0).toVar(),
        (i) => i.lessThan(4),
        (i) => {
          if (counterFirst) { i.assign(i.add(1)); other.assign(other.add(1)); }
          else { other.assign(other.add(1)); i.assign(i.add(1)); }
        },
        (i) => { total.assign(total.add(i)); },
      );
      return total;
    });
  }

  it("keeps every statement of a loop update in both backends", () => {
    for (let counterFirst of [false, true]) {
      let label = counterFirst ? "counter first" : "counter last";

      let glsl = compileGLSL(twoStepLoop(counterFirst)());
      let glslHeader = glsl.split("\n").find(l => l.includes("for ("))!;
      expect(glslHeader, label).toMatch(/\+ 1\.0,.*\+ 1\.0/);

      // WGSL takes one statement in the header, so more than one moves into a
      // continuing block — which runs after the body, including after continue.
      let wgsl = compileWGSL(twoStepLoop(counterFirst)());
      expect(wgsl.match(/\+ 1f/g) ?? [], label).toHaveLength(2);
      expect(wgsl, label).toContain("continuing");
    }
  });

  // The ordinary single-statement loop keeps the plain for-header it had.
  it("leaves a single-statement loop update in the header", () => {
    let prog = Fn(() => {
      let total = float(0).toVar();
      For(
        () => float(0).toVar(),
        (i) => i.lessThan(4),
        (i) => i.assign(i.add(1)),
        (i) => { total.assign(total.add(i)); },
      );
      return total;
    });
    let wgsl = compileWGSL(prog());
    expect(wgsl).toMatch(/for \(var \S+: f32 = 0f; .*; \S+ = \S+ \+ 1f\)/);
    expect(wgsl).not.toContain("continuing");
  });

  // A plain JavaScript number has no shader type of its own, so it takes the
  // type of the operand it sits beside.
  it("gives a compared literal the operand's integer type", () => {
    let unsigned = Fn(() => uniform("uint").lessThan(2).toVar());
    expect(compileWGSL(unsigned())).toMatch(/\S+ < 2u/);
    expect(compileGLSL(unsigned())).not.toContain("float(");

    let signed = Fn(() => uniform("int").greaterThan(3).toVar());
    expect(compileWGSL(signed())).toMatch(/\S+ > 3i/);
    expect(compileGLSL(signed())).not.toContain("float(");
  });

  // A literal that is not a whole number cannot take an integer type.
  it("refuses a fractional literal beside an integer operand", () => {
    expect(() => Fn(() => uniform("int").add(2.5).toVar())())
      .toThrow(/whole number|integer/i);
  });

  // Negative literals stay signed: an unsigned type has no negative values,
  // and WGSL has no unary minus for one either.
  it("refuses a negative literal beside an unsigned operand", () => {
    expect(() => Fn(() => uniform("uint").add(-1).toVar())())
      .toThrow(/negative|unsigned/i);
  });

  // WGSL makes a single component assignable but not a multi-component
  // swizzle, so the write is split per component.
  it("resolves a nested swizzle down to the variable it writes to", () => {
    let prog = Fn(() => {
      let a = vec4(1, 2, 3, 4).toVar();
      a.xyz.xy.assign(vec2(9, 9));
      return a;
    });
    let wgsl = compileWGSL(prog());
    expect(wgsl).not.toMatch(/\.xyz\.[xyzw] = /);
    expect(wgsl).toMatch(/_rmsl_\d+\.x = /);
    expect(wgsl).toMatch(/_rmsl_\d+\.y = /);
  });

  // gl_Position is written by the vertex stage and is not readable in the
  // fragment stage; WGSL has no free-standing `position` there either, and
  // never emitted a parameter for one. Both produced an undeclared identifier.
  it("refuses to read the position from a fragment stage", () => {
    let prog = Fn(() => vec4(builtinPosition().x, 0, 0, 1).toVar());
    expect(() => compileGLSL.fragment(prog())).toThrow(/fragment/i);
    expect(() => compileWGSL.fragment(prog())).toThrow(/fragment/i);
  });

  it("still reads the position in a vertex stage", () => {
    let prog = Fn(() => vec4(builtinPosition().x, 0, 0, 1).toVar());
    expect(compileGLSL.vertex(prog())).toContain("gl_Position");
    expect(compileWGSL.vertex(prog())).toContain("result.position");
  });

  // WGSL's uniform address space takes host-shareable types only, and neither
  // a bool nor a boolean vector is one.
  it("carries a boolean vector uniform through a host-shareable type", () => {
    let prog = Fn(() => uniform("bvec3").all().toVar());
    let wgsl = compileWGSL(prog());
    expect(wgsl).not.toContain("var<uniform> _rmsl_u0: vec3<bool>");
    expect(wgsl).toContain("vec3<u32>");
    // GLSL has no such restriction and declares it directly.
    expect(compileGLSL(prog())).toContain("uniform bvec3");
  });

  // A matrix column is a vector. The node inherited its operand's type, so it
  // claimed to be a matrix around an expression that produces a vector, and the
  // declaration it generated did not compile in either backend. The signature
  // has always said Node<"vec4">; only the runtime type disagreed.
  it("types a matrix element as the column vector it is", () => {
    let prog4 = Fn(() => uniform("mat4").element(0).toVar());
    expect(compileGLSL(prog4())).toMatch(/vec4 \S+ = \S+\[\S+\];/);
    expect(compileWGSL(prog4())).toMatch(/var \S+: vec4<f32> = \S+\[\S+\];/);

    let prog3 = Fn(() => uniform("mat3").element(1).toVar());
    expect(compileGLSL(prog3())).toMatch(/vec3 \S+ = \S+\[\S+\];/);
    expect(compileWGSL(prog3())).toMatch(/var \S+: vec3<f32> = \S+\[\S+\];/);
  });

  // The index is an integer, so it has to be emitted as one. Typing it from
  // the matrix operand left it a float and produced m[int(0.0)].
  it("indexes a matrix with an integer", () => {
    let prog = Fn(() => uniform("mat4").element(0).toVar());
    expect(compileGLSL(prog())).not.toContain("int(0.0)");
    expect(compileWGSL(prog())).not.toContain("i32(0f)");
  });

  // Transposing swaps columns for rows, so the result type changes for a
  // matrix that is not square. Only the square types were reachable before, and
  // they keep their own type, so the runtime never had to get this right.
  it("types a non-square transpose with its shape swapped", () => {
    let prog = Fn(() => uniform("mat2x3").transpose().toVar());
    expect(compileGLSL(prog())).toMatch(/mat3x2 \S+ = transpose\(/);
    expect(compileWGSL(prog())).toMatch(/var \S+: mat3x2<f32> = transpose\(/);
  });

  it("multiplies a non-square matrix by a vector of its column count", () => {
    let prog = Fn(() => uniform("mat2x3").mul(vec2(1, 2)).toVar());
    expect(compileGLSL(prog())).toMatch(/vec3 \S+ = /);
    expect(compileWGSL(prog())).toMatch(/var \S+: vec3<f32> = /);
  });

  // WGSL has no inverse() builtin, so one is written out per matrix size. The
  // helper was chosen by testing for mat3 and falling back to mat4, so a mat2
  // was inverted by the four-by-four helper and the call matched nothing.
  //
  // Only a square matrix has an inverse, and neither language offers an
  // overload for the rest. WGSL says so; GLSL emitted the call anyway, and a
  // shader that no driver accepts is worse than one that fails to build here.
  it("refuses to invert a matrix that is not square, in both backends", () => {
    let prog = Fn(() => (uniform("mat2x3") as any).inverse().toVar());
    expect(() => compileGLSL(prog())).toThrow(/square/i);
    expect(() => compileWGSL(prog())).toThrow(/square/i);
  });

  it("uses the inverse helper matching the matrix size", () => {
    let wgsl2 = compileWGSL(Fn(() => uniform("mat2").inverse().toVar())());
    expect(wgsl2).toContain("fn _rmsl_inverse2(");
    expect(wgsl2).not.toContain("_rmsl_inverse4");

    expect(compileWGSL(Fn(() => uniform("mat3").inverse().toVar())()))
      .toContain("fn _rmsl_inverse3(");
    expect(compileWGSL(Fn(() => uniform("mat4").inverse().toVar())()))
      .toContain("fn _rmsl_inverse4(");
  });

  // Folding runs on literal operands; the integer branch truncates with `| 0`.
  it("folds integer arithmetic with truncation", () => {
    expect(compileGLSL(Fn(() => int(7).div(int(2)).toVar())())).toContain("= 3;");
    expect(compileGLSL(Fn(() => int(7).add(int(2)).toVar())())).toContain("= 9;");
    expect(compileGLSL(Fn(() => int(7).sub(int(2)).toVar())())).toContain("= 5;");
    expect(compileGLSL(Fn(() => int(7).mul(int(2)).toVar())())).toContain("= 14;");
    // The float path keeps the fraction the integer path drops.
    expect(compileGLSL(Fn(() => float(7).div(float(2)).toVar())())).toContain("3.5");
  });

  // === WGSL uniform packing ===
  //
  // WGSL allows 12 uniform *buffers* per stage — the spec minimum and what real
  // devices report — so a binding per uniform stops working at the thirteenth.
  // They are packed into one struct instead, which is one binding regardless of
  // count.

  it("packs every value uniform into a single binding", () => {
    let prog = Fn(() => {
      // Twenty is comfortably past the twelve-buffer limit.
      let sum = float(0).toVar();
      for (let i = 0; i < 20; i++) sum.assign(sum.add(uniform("float")));
      return sum;
    });
    let wgsl = compileWGSL(prog());

    let bindings = wgsl.split("\n").filter(l => l.includes("var<uniform>"));
    expect(bindings, "one binding no matter how many uniforms").toHaveLength(1);
    expect(wgsl).toContain("struct _RmslUniforms {");
    // Every reference goes through the struct.
    expect(wgsl).toContain("_rmsl_uniforms._rmsl_u");
  });

  it("keeps textures out of the uniform struct", () => {
    let prog = Fn(() => {
      let tex = uniform("sampler2D");
      let scale = uniform("float");
      let out = output("vec4");
      out.assign(tex.texture(vec2(0.5, 0.5)).mul(scale));
      return out;
    });
    let wgsl = compileWGSL(prog());

    // A texture is an opaque handle: it cannot be a uniform variable nor a
    // member of a uniform struct, so it keeps a binding of its own.
    expect(wgsl).toMatch(/@group\(1\) @binding\(0\) var \S+: texture_2d<f32>;/);
    expect(wgsl).toContain("@group(0) @binding(0) var<uniform> _rmsl_uniforms");
    // The scalar is in the struct; the texture is referenced bare.
    expect(wgsl).toContain("_rmsl_uniforms._rmsl_u");
    expect(wgsl).not.toMatch(/_rmsl_uniforms\.\S*: texture/);
  });

  it("leaves GLSL uniforms as individual declarations", () => {
    let prog = Fn(() => uniform("float").add(uniform("vec2").x).toVar());
    let glsl = compileGLSL(prog());
    // GLSL has no such limit, so nothing is packed.
    expect(glsl).toContain("uniform float");
    expect(glsl).toContain("uniform vec2");
    expect(glsl).not.toContain("_rmsl_uniforms");
  });

  // The host has to write the buffer, and declaration order is not layout
  // order: members are sorted by descending alignment so WGSL inserts no
  // padding between them. Offsets are returned because there is no other way
  // for a caller to know them.
  it("reports uniform offsets following WGSL alignment rules", () => {
    let layout = wgslUniformLayout([
      { slot: "a", type: "f32" },
      { slot: "b", type: "vec2<f32>" },
      { slot: "c", type: "vec3<f32>" },
      { slot: "d", type: "vec4<f32>" },
    ]);

    expect(layout.members.map(m => m.name), "ordered by descending alignment")
      .toEqual(["c", "d", "b", "a"]);
    expect(layout.members.map(m => m.offset)).toEqual([0, 16, 32, 40]);
    // vec3 occupies 12 bytes but aligns to 16, so d starts at 16 not 12.
    expect(layout.members.find(m => m.name === "c")!.size).toBe(12);
    // Struct is rounded up to its largest member's alignment.
    expect(layout.size).toBe(48);
  });

  // === Uniform arrays ===
  //
  // One slot however long, where N separate uniforms cost N slots — which is
  // what made a 24-brick scene need 32 of them. It also lets the shader loop
  // over the elements instead of unrolling a test per value.

  it("declares a uniform array once and indexes it", () => {
    let prog = Fn(() => {
      let items = uniformArray("vec4", 24);
      let total = vec4(0, 0, 0, 0).toVar();
      For(
        () => float(0).toVar(),
        (i) => i.lessThan(24),
        (i) => i.assign(i.add(1)),
        (i) => { total.assign(total.add(items.element(i))); },
      );
      let out = output("vec4");
      out.assign(total);
      return out;
    });

    let glsl = compileGLSL(prog());
    expect(glsl).toMatch(/uniform vec4 _rmsl_u\d+\[24\];/);
    // GLSL indexes with an int, so a float loop counter is converted.
    expect(glsl).toMatch(/_rmsl_u\d+\[int\(\S+\)\]/);

    let wgsl = compileWGSL(prog());
    expect(wgsl).toMatch(/_rmsl_u\d+: array<vec4<f32>, 24>,/);
    expect(wgsl).toMatch(/_rmsl_uniforms\._rmsl_u\d+\[i32\(\S+\)\]/);
    // Still a single binding, which is the whole point.
    expect(wgsl.split("\n").filter(l => l.includes("var<uniform>"))).toHaveLength(1);
  });

  it("indexes a uniform array by a constant", () => {
    let prog = Fn(() => {
      let items = uniformArray("vec4", 4);
      return items.element(2).add(items.element(0)).toVar();
    });
    // A plain number is a float node, so the index is converted rather than
    // emitted bare.
    expect(compileGLSL(prog())).toMatch(/_rmsl_u\d+\[int\(2\.0\)\]/);
    expect(compileWGSL(prog())).toMatch(/_rmsl_u\d+\[i32\(2f\)\]/);
  });

  // Element types too narrow to align are stored widened and read back out of
  // the leading components, so the padding never reaches the caller. The same
  // approach TSL takes.
  it("pads array elements WGSL cannot align, transparently", () => {
    let cases: [string, string, string][] = [
      // declared      stored in WGSL        read back with
      ["float", "array<vec4<f32>, 4>", ".x"],
      ["vec2", "array<vec4<f32>, 4>", ".xy"],
      // A vec3 aligns to 16 already, so it is stored as itself.
      ["vec3", "array<vec3<f32>, 4>", ""],
      ["vec4", "array<vec4<f32>, 4>", ""],
    ];
    for (let [type, stored, read] of cases) {
      let prog = Fn(() => (uniformArray(type as any, 4).element(1) as any).toVar());
      let wgsl = compileWGSL(prog());
      expect(wgsl, type).toContain(stored);
      expect(wgsl, type).toMatch(new RegExp(`\\[i32\\([^)]*\\)\\]${read.replace(".", "\\.")}`));
      // GLSL has no such rule, so nothing is padded there.
      expect(compileGLSL(prog()), type).toMatch(/uniform \w+ _rmsl_u\d+\[4\];/);
    }
  });

  // The location came from a module-wide counter, so the fifth output declared
  // anywhere landed at location 4 even in a shader that had only one — past
  // MAX_DRAW_BUFFERS soon enough for an app with a few materials.
  it("numbers output locations per shader", () => {
    for (let i = 0; i < 6; i++) {
      let prog = Fn(() => {
        let o = output("vec4");
        o.assign(vec4(1, 0, 0, 1));
        return o;
      });
      expect(compileGLSL(prog()), `shader ${i}`).toContain("layout(location=0) out");
      expect(compileWGSL(prog()), `shader ${i}`).toMatch(/@location\(0\) _rmsl_o\d+/);
    }
  });

  it("still numbers several outputs in one shader in order", () => {
    let prog = Fn(() => {
      let a = output("vec4");
      let b = output("vec4");
      a.assign(vec4(1, 0, 0, 1));
      b.assign(vec4(0, 1, 0, 1));
      return b;
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("layout(location=0) out");
    expect(glsl).toContain("layout(location=1) out");
  });

  // WGSL's uniform address space takes host-shareable types only, so a bool
  // travels as an unsigned integer and is compared back on read. A single bool
  // uniform already did that; an array of them did not, and reached WGSL
  // declared as an array of bool, which it refuses.
  it("carries an array of booleans through a host-shareable type", () => {
    let prog = Fn(() => uniformArray("bool", 4).element(int(1)).toVar());
    let wgsl = compileWGSL(prog());
    expect(wgsl).not.toContain("array<bool");
    expect(wgsl).toContain("u32");
    // The caller asked for a bool and gets one back.
    expect(wgsl).toMatch(/!= 0u\)/);
    expect(wgsl).toMatch(/var \S+: bool =/);

    // GLSL has no such restriction and declares the array directly.
    expect(compileGLSL(prog())).toMatch(/uniform bool \S+\[4\];/);
  });

  it("carries an array of boolean vectors the same way", () => {
    let prog = Fn(() => uniformArray("bvec3", 2).element(int(0)).all().toVar());
    let wgsl = compileWGSL(prog());
    expect(wgsl).not.toContain("array<vec3<bool>");
    expect(wgsl).toMatch(/var \S+: bool =/);
  });

  // A texture cannot live in a uniform block — it is not host-shareable — so
  // the whole-shader path gives it a binding of its own. The standalone
  // function path packed everything it had, textures included.
  it("keeps a texture out of the uniform struct in a standalone function", () => {
    let wgsl = compileWGSLFn(() => uniform("sampler2D").texture(vec2(0.5, 0.5)), {
      name: "sample", params: [],
    });
    expect(wgsl).toMatch(/var \S+: texture_2d<f32>;/);
    expect(wgsl).not.toMatch(/struct \S+ \{[^}]*texture_2d/s);

    recordShaderSource("wgsl", "fragment", `${wgsl}
@fragment fn main() -> @location(0) vec4<f32> { return sample(); }`);
  });

  it("keeps a sampler3D out of the uniform struct in a standalone function", () => {
    let wgsl = compileWGSLFn(() => uniform("sampler3D").texture(vec3(0.5, 0.5, 0.5)), {
      name: "sample3D", params: [],
    });
    expect(wgsl).toMatch(/var \S+: texture_3d<f32>;/);
    expect(wgsl).not.toMatch(/struct \S+ \{[^}]*texture_3d/s);

    recordShaderSource("wgsl", "fragment", `${wgsl}
@fragment fn main() -> @location(0) vec4<f32> { return sample3D(); }`);
  });

  // Neither backend has a way to express this that the other shares: WGSL has
  // no plain array of textures in the uniform address space, and emitting one
  // produced a reference to a struct that does not hold it.
  it("refuses an array of textures", () => {
    expect(() => uniformArray("sampler2D", 2)).toThrow(/texture|sampler/i);
    expect(() => uniformArray("sampler3D", 2)).toThrow(/texture|sampler/i);
    expect(() => uniformArray("samplerCube", 2)).toThrow(/texture|sampler/i);
  });

  it("rejects a nonsensical array length", () => {
    expect(() => uniformArray("vec4", 0)).toThrow(/positive integer/);
    expect(() => uniformArray("vec4", -3)).toThrow(/positive integer/);
    expect(() => uniformArray("vec4", 2.5)).toThrow(/positive integer/);
  });

  // The stride is what a caller cannot guess. It happens to equal the element
  // size for vec4, but the layout reports it rather than leaving the host to
  // assume the two are the same.
  it("reports the stride of an array member", () => {
    let layout = wgslUniformLayout([
      { slot: "vectors", type: "vec4<f32>", length: 2 },
    ]);

    let vectors = layout.members.find(m => m.name === "vectors")!;
    expect(vectors.stride, "a vec4 is already 16").toBe(16);
    expect(vectors.size).toBe(32);

    expect(layout.size).toBe(32);
  });

  // The caller writes the buffer from these numbers, so a wrong one is not a
  // shader that fails to build — it is a shader that reads whatever happens to
  // be at that address. None of these fail loudly, which is why each is pinned.
  it("sizes a matrix that is not square", () => {
    // A matCxR is C columns of vecR, and a column is padded to its alignment,
    // so a mat2x3 is two columns of sixteen bytes rather than two of twelve.
    let layout = wgslUniformLayout([
      { slot: "m", type: "mat2x3<f32>" },
      { slot: "v", type: "vec4<f32>" },
    ]);
    let m = layout.members.find(x => x.name === "m")!;
    expect(m.size, "two columns of sixteen").toBe(32);
    expect(layout.size).toBe(48);
  });

  it("refuses a type it has no layout for, rather than guessing", () => {
    expect(() => wgslUniformLayout([{ slot: "x", type: "mat9x9<f32>" }]))
      .toThrow(/layout/i);
  });

  // An array's alignment is raised to sixteen whatever it holds, and the struct
  // takes its alignment from its members — so the struct's own size has to
  // account for that, or the buffer is short.
  it("rounds the struct up to an array member's alignment", () => {
    let layout = wgslUniformLayout([
      { slot: "a", type: "f32", length: 3 },
      { slot: "b", type: "f32" },
    ]);
    expect(layout.members.find(x => x.name === "a")!.size, "three slots of 16").toBe(48);
    expect(layout.size, "rounded up to sixteen").toBe(64);
  });

  // Members that align the same are kept in the order they were declared. The
  // tie was broken on the generated slot name, which carries a counter that
  // climbs for the life of the process — so "…u10" sorted before "…u9" and the
  // same program compiled twice put its values at different addresses.
  it("orders equally aligned members by declaration, not by name", () => {
    let byName = wgslUniformLayout([
      { slot: "_rmsl_u9", type: "f32" },
      { slot: "_rmsl_u10", type: "f32" },
    ]);
    expect(byName.members.map(m => m.name)).toEqual(["_rmsl_u9", "_rmsl_u10"]);
    expect(byName.members.map(m => m.offset)).toEqual([0, 4]);
  });

  it("gives a boolean array the footprint of what it travels as", () => {
    let layout = wgslUniformLayout([{ slot: "flags", type: "bool", length: 4 }]);
    let flags = layout.members.find(m => m.name === "flags")!;
    expect(flags.stride, "widened to a four-component vector").toBe(16);
    expect(flags.size).toBe(64);
  });

  it("gives a single uniform offset zero", () => {
    let layout = wgslUniformLayout([{ slot: "only", type: "vec4<f32>" }]);
    expect(layout.members).toEqual([
      { name: "only", type: "vec4<f32>", offset: 0, size: 16 },
    ]);
    expect(layout.size).toBe(16);
  });
});

describe("integer vectors", () => {
  it("compiles ivec and uvec constructors to GLSL", () => {
    let prog = Fn(() => ivec3(1, 2, 3).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("ivec3(1, 2, 3)");
  });

  it("compiles ivec and uvec constructors to WGSL", () => {
    let prog = Fn(() => uvec2(4, 5).toVar());
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("vec2<u32>(4u, 5u)");
  });

  it("adds an integer scalar to an integer vector", () => {
    let prog = Fn(() => ivec2(1, 2).add(3).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("ivec2(1, 2) + 3");
  });

  it("compiles integer vector comparison to a boolean vector", () => {
    let prog = Fn(() => ivec3(1, 2, 3).lessThan(ivec3(4, 5, 6)).all().toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("lessThan(");
    expect(glsl).toContain("all(");
  });

  it("compiles an integer vector swizzle to the integer scalar", () => {
    let prog = Fn(() => uvec3(1, 2, 3).y.toVar());
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("vec3<u32>(1u, 2u, 3u)");
    expect(wgsl).toContain(".y");
  });

  it("rejects a non-whole number beside an integer vector", () => {
    let prog = () => ivec2(1, 2).add(0.5).toVar();
    expect(() => prog()).toThrow(/not a whole number/);
  });

  it("rejects a negative literal beside an unsigned vector", () => {
    expect(() => uvec2(1, 2).add(-1).toVar()).toThrow(/unsigned/);
  });

  it("indexes an integer vector by element", () => {
    let prog = Fn(() => {
      let v = ivec3(7, 8, 9).toVar();
      return v.element(1).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("[1]");
  });
});

describe("integer samplers", () => {
  it("compiles isampler3D sampling to a GLSL texelFetch", () => {
    let prog = Fn(() => {
      let tex = uniform("isampler3D");
      return tex.texture(ivec3(1, 2, 3)).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("uniform isampler3D");
    expect(glsl).toContain("texelFetch(");
  });

  it("compiles usampler3D sampling to a GLSL texelFetch with an lod", () => {
    let prog = Fn(() => {
      let tex = uniform("usampler3D");
      return tex.textureLod(uvec3(1, 2, 3), int(0)).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("uniform usampler3D");
    expect(glsl).toContain("texelFetch(");
  });

  it("compiles isampler2D to a WGSL textureLoad with no sampler", () => {
    let prog = Fn(() => {
      let tex = uniform("isampler2D");
      return tex.texture(ivec2(1, 2)).toVar();
    });
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("texture_2d<i32>");
    expect(wgsl).toContain("textureLoad(");
    expect(wgsl).not.toContain("_s: sampler;");
  });

  it("declares high precision for integer samplers in GLSL", () => {
    let prog = Fn(() => {
      let tex = uniform("usampler2D");
      return tex.texture(uvec2(1, 2)).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("precision highp usampler2D;");
  });

  it("refuses to hold an integer sampler in a uniform array", () => {
    expect(() => uniformArray("usampler3D", 4)).toThrow(/cannot hold a texture/);
  });
});

describe("GLSL precision", () => {
  // The compileGLSL options mirror three.js's `precision` setting: the float
  // and every declared sampler precision follow it, defaulting to "highp".
  // WGSL has no precision qualifiers, so nothing here applies to it.

  it("defaults to highp", () => {
    let prog = Fn(() => float(1).toVar());
    expect(compileGLSL(prog())).toContain("precision highp float;");
  });

  it("emits mediump float precision", () => {
    let prog = Fn(() => float(1).toVar());
    let glsl = compileGLSL(prog(), { precision: "mediump" });
    expect(glsl).toContain("precision mediump float;");
    expect(glsl).not.toContain("precision highp float;");
  });

  it("emits lowp precision for an explicit fragment stage", () => {
    let prog = Fn(() => float(1).toVar());
    let glsl = compileGLSL.fragment(prog(), { precision: "lowp" });
    expect(glsl).toContain("precision lowp float;");
  });

  it("emits the configured precision for vertex shaders", () => {
    let prog = Fn(() => vec4(0, 0, 0, 1).toVar());
    let glsl = compileGLSL.vertex(prog(), { precision: "mediump" });
    expect(glsl).toContain("precision mediump float;");
    expect(glsl).not.toContain("precision highp float;");
  });

  it("applies the configured precision to sampler declarations", () => {
    let prog = Fn(() => uniform("sampler2D").texture(vec2(0, 0)).toVar());
    let glsl = compileGLSL(prog(), { precision: "lowp" });
    expect(glsl).toContain("precision lowp float;");
    expect(glsl).toContain("precision lowp sampler2D;");
    expect(glsl).not.toContain("precision highp");
  });

  it("rejects an unknown precision", () => {
    let prog = Fn(() => float(1).toVar());
    expect(() => compileGLSL(prog(), { precision: "high" as any })).toThrow(
      /precision/,
    );
  });
});

describe("casts and conversions", () => {
  it("casts a float to an int in GLSL and WGSL", () => {
    let prog = Fn(() => {
      let x = float(2.9).toVar();
      return x.toInt().toVar();
    });
    expect(compileGLSL(prog())).toContain("int(");
    expect(compileWGSL(prog())).toContain("i32(");
  });

  it("constructs a uint from a float", () => {
    let prog = Fn(() => uint(float(3.0)).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("uint(");
  });

  it("casts a bool from an int", () => {
    let prog = Fn(() => bool(int(1)).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("bool(");
  });

  it("casts between float and integer vectors", () => {
    let prog = Fn(() => ivec3(1, 2, 3).toVec3().toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("vec3(");
    let back = Fn(() => vec3(1, 2, 3).toIVec3().toVar());
    let wgsl = compileWGSL(back());
    expect(wgsl).toContain("vec3<i32>(");
  });

  it("converts a value with convert()", () => {
    let prog = Fn(() => vec3(1, 2, 3).convert("uvec3").toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("uvec3(");
  });

  it("rejects a negative uint literal", () => {
    expect(() => uint(-1)).toThrow(/unsigned/);
  });

  it("mixes signed and unsigned operands by converting explicitly", () => {
    let prog = Fn(() => int(5).add(uniform("uint").toInt()).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("int(");
  });
});

describe("new math builtins", () => {
  it("compiles round, trunc and the hyperbolic functions", () => {
    let prog = Fn(() => {
      let x = float(0.5).toVar();
      return x.round().add(x.trunc()).add(x.sinh()).add(x.tanh()).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("round(");
    expect(glsl).toContain("trunc(");
    expect(glsl).toContain("sinh(");
    expect(glsl).toContain("tanh(");
  });

  it("emits WGSL's derivative spellings", () => {
    let prog = Fn(() => float(0.5).dFdx().toVar());
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("dpdx(");
    let glsl = compileGLSL(Fn(() => float(0.5).dFdy().toVar())());
    expect(glsl).toContain("dFdy(");
  });

  it("composes saturate, oneMinus, reciprocal and pow2", () => {
    let prog = Fn(() => {
      let x = float(0.5).toVar();
      return x.saturate().add(x.oneMinus()).add(x.reciprocal()).add(x.pow2()).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("clamp(");
    expect(glsl).toContain("1.0 - ");
    expect(glsl).toContain(" / ");
  });

  it("computes a matrix determinant", () => {
    let prog = Fn(() => mat2(1, 2, 3, 4).determinant().toVar());
    expect(compileGLSL(prog())).toContain("determinant(");
    expect(compileWGSL(Fn(() => mat3().determinant().toVar())())).toContain("determinant(");
  });

  it("computes a bitwise not on an integer", () => {
    let prog = Fn(() => int(5).bitNot().toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("~");
    let wgsl = compileWGSL(Fn(() => ivec2(1, 2).bitNot().toVar())());
    expect(wgsl).toContain("~");
  });

  it("computes a logical xor", () => {
    let prog = Fn(() => bool(true).xor(bool(false)).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("&&");
    expect(glsl).toContain("||");
  });

  it("folds a lengthSq on a scalar to its squared value", () => {
    let prog = Fn(() => float(3).lengthSq().toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("9");
  });
});

describe("Switch", () => {
  it("branches on an integer selector as an if/else chain", () => {
    let prog = Fn(() => {
      let out = float(0).toVar();
      Switch(int(1), (s) => {
        s.Case(0, () => { out.assign(float(10)); });
        s.Case([1, 2], () => { out.assign(float(20)); });
        s.Default(() => { out.assign(float(30)); });
      });
      return out;
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("if (");
    expect(glsl).toContain("==");
    expect(glsl).toContain("||");
    expect(glsl).toContain("else {");
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("==");
  });
});

describe("TSL control flow", () => {
  it("If / ElseIf / Else emit an if/else chain", () => {
    let prog = Fn(() => {
      let out = float(0).toVar();
      If(float(1).greaterThan(0), () => { out.assign(float(10)); })
        .ElseIf(float(1).lessThan(0), () => { out.assign(float(20)); })
        .Else(() => { out.assign(float(30)); });
      return out;
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("if (");
    expect(glsl).toContain("else {");
  });

  it("For runs a counted loop", () => {
    let prog = Fn(() => {
      let total = float(0).toVar();
      For(
        () => float(0).toVar(),
        (i) => i.lessThan(4),
        (i) => i.assign(i.add(1)),
        (i) => { total.assign(total.add(i)); },
      );
      return total;
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("for (");
  });

  it("While runs a conditional loop", () => {
    let prog = Fn(() => {
      let n = float(3).toVar();
      While(n.greaterThan(0), () => { n.assign(n.sub(1)); });
      return n;
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("while (");
  });

  it("Loop(count, (i) => ...) iterates like TSL's counting loop", () => {
    let prog = Fn(() => {
      let total = float(0).toVar();
      Loop(int(4), (i) => { total.assign(total.add(float(i))); });
      return total;
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("for (");
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("for (");
  });

  it("Switch / Case / Default branch like Switch", () => {
    let prog = Fn(() => {
      let out = float(0).toVar();
      Switch(int(2), (s) => {
        s.Case([1, 2], () => { out.assign(float(20)); });
        s.Default(() => { out.assign(float(30)); });
      });
      return out;
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("==");
    expect(glsl).toContain("||");
    expect(glsl).toContain("else {");
  });

  it("Break / Continue emit their statements", () => {
    let prog = Fn(() => {
      let total = float(0).toVar();
      For(
        () => int(0).toVar(),
        (i) => i.lessThan(10),
        (i) => i.assign(i.add(int(1))),
        (i) => {
          If(i.greaterThanEqual(8), () => { Break(); });
          If(i.lessThan(2), () => { Continue(); });
          total.assign(total.add(1));
        },
      );
      return total;
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("break;");
    expect(glsl).toContain("continue;");
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("break;");
    expect(wgsl).toContain("continue;");
  });

  it("Return emits an early return", () => {
    let prog = Fn(() => {
      let out = float(0).toVar();
      If(float(1).greaterThan(0), () => { Return(); });
      out.assign(float(1));
      return out;
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("return;");
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("return;");
  });

  it("Discard emits the fragment discard", () => {
    let prog = Fn(() => {
      let out = float(0).toVar();
      If(float(1).greaterThan(0), () => { Discard(); });
      out.assign(float(1));
      return out;
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("discard;");
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("discard;");
  });
});

describe("TSL free-function API", () => {
  it("compiles add/sub/mul/div/mod free functions", () => {
    let prog = Fn(() => {
      let u = uniform("float");
      return add(mul(u, float(2)), sub(u, div(u, float(2)))).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("*");
    expect(glsl).toContain("+");
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("*");
  });

  it("compiles min/max/pow and the powers", () => {
    let prog = Fn(() => {
      let u = uniform("float");
      return max(min(u, float(5)), pow(u, float(3))).add(pow2(u)).add(pow3(u)).add(pow4(u)).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("max(");
    expect(glsl).toContain("min(");
    expect(glsl).toContain("pow(");
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("max(");
  });

  it("compiles unary math free functions", () => {
    let prog = Fn(() => {
      let u = uniform("float");
      return abs(sin(u)).add(cos(u)).add(floor(u)).add(sqrt(u)).add(inverseSqrt(u)).add(inversesqrt(u)).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("sin(");
    expect(glsl).toContain("inversesqrt(");
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("inverseSqrt(");
  });

  it("compiles step/smoothstep/clamp/mix with TSL argument order", () => {
    let prog = Fn(() => {
      let x = float(0.5).toVar();
      return step(float(0.25), x).add(smoothstep(float(0), float(1), x)).add(clamp(x, float(0), float(1))).add(mix(float(0), float(1), x)).toVar();
    });
    let glsl = compileGLSL(prog());
    // step(edge, x): the edge is the first emitted argument.
    expect(glsl).toContain("step(0.25, ");
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("step(");
    expect(wgsl).toContain("smoothstep(");
  });

  it("compiles vector free functions", () => {
    let prog = Fn(() => {
      let v = vec3(1, 2, 3).toVar();
      return normalize(v).mul(dot(v, v)).add(cross(v, vec3(0, 1, 0))).add(distance(v, vec3(0, 0, 0))).add(length(v)).add(lengthSq(v)).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("normalize(");
    expect(glsl).toContain("dot(");
    expect(glsl).toContain("cross(");
    expect(glsl).toContain("distance(");
    expect(glsl).toContain("length(");
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("normalize(");
  });

  it("compiles reflect/refract/faceForward", () => {
    let prog = Fn(() => {
      let v = vec3(1, 0, 0).toVar();
      return reflect(v, vec3(0, 1, 0)).add(refract(v, vec3(0, 1, 0), 0.9)).add(faceForward(v, vec3(0, 1, 0), vec3(1, 0, 0))).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("reflect(");
    expect(glsl).toContain("refract(");
    expect(glsl).toContain("faceforward(");
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("faceForward(");
  });

  it("compiles atan(y, x) as atan2", () => {
    let prog = Fn(() => {
      let u = uniform("float");
      return atan(u, float(2)).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("atan(");
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("atan2(");
  });

  it("compiles comparison and logic free functions", () => {
    let prog = Fn(() => {
      let a = float(1).toVar();
      let b = float(2).toVar();
      return equal(a, b).toInt().add(lessThan(a, b).toInt()).add(not(greaterThanEqual(a, b)).toInt()).add(and(notEqual(a, b), or(a.lessThan(b), xor(a.lessThan(b), a.greaterThan(b)))).toInt()).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("==");
    expect(glsl).toContain("<");
    expect(glsl).toContain("&&");
    expect(glsl).toContain("||");
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("==");
  });

  it("compiles all/any reductions", () => {
    let prog = Fn(() => {
      let v = vec3(1, 2, 3).toVar();
      return all(v.greaterThan(0)).toInt().add(any(v.lessThan(0)).toInt()).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("all(");
    expect(glsl).toContain("any(");
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("all(");
  });

  it("compiles matrix free functions", () => {
    let prog = Fn(() => determinant(mat4(1)).add(determinant(transpose(mat3(1)))).add(determinant(inverse(mat4(1)))).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("determinant(");
    expect(glsl).toContain("transpose(");
    expect(glsl).toContain("inverse(");
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("transpose(");
  });

  it("accepts raw numbers and arrays as operands", () => {
    let prog = Fn(() => mul(2, 3).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("6.0");
    let prog2 = Fn(() => dot(vec3(1, 0, 0), [1, 2, 3]).toVar());
    let glsl2 = compileGLSL(prog2());
    expect(glsl2).toContain("dot(");
    expect(glsl2).toContain("1, 0, 0");
  });
});

describe("TSL additions", () => {
  it("exports the TSL constants as float nodes", () => {
    let prog = Fn(() => {
      let u = uniform("float");
      return u.add(PI).add(TWO_PI).add(HALF_PI).add(EPSILON).add(INFINITY).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("3.141592653589793");
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("6.283185307179586");
  });

  it("constructs boolean vectors", () => {
    let prog = Fn(() => bvec2(true, false).all().toInt().add(bvec3(true, true, false).any().toInt()).add(bvec4(false, false, false, false).any().toInt()).toVar());
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("bvec2(");
    expect(glsl).toContain("bvec3(");
    expect(glsl).toContain("bvec4(");
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("vec2<bool>(");
  });

  it("exposes stpq swizzles", () => {
    let prog = Fn(() => {
      let v = vec4(1, 2, 3, 4).toVar();
      // stpq is the texture-coordinate spelling of xyzw: st is a vec2, p a
      // scalar, so the sum has to rebuild a vec4 rather than add across widths.
      return v.stpq.add(vec4(v.st, v.p, 1)).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain(".stpq");
    expect(glsl).toContain(".st");
    let wgsl = compileWGSL(prog());
    // WGSL only knows the xyzw and rgba spellings, so stpq is mapped to xyzw.
    expect(wgsl).toContain(".xyzw");
    expect(wgsl).toContain(".xy");
  });

  it("supports compound assignments", () => {
    let prog = Fn(() => {
      let a = float(1).toVar();
      a.addAssign(1);
      a.mulAssign(2);
      a.subAssign(1);
      a.divAssign(2);
      a.modAssign(3);
      return a;
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("=");
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("=");
  });

  it("aliases .var() for toVar()", () => {
    let prog = Fn(() => {
      let a = float(1).var();
      return a;
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("float");
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("var ");
  });

  it("multiplies a matrix by a position vector", () => {
    let prog = Fn(() => {
      let m = mat4(1).toVar();
      let v = vec3(1, 2, 3).toVar();
      return m.mul(v).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("vec4(");
    expect(glsl).toContain(".xyz");
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("vec4<f32>(");
  });

  it("multiplies a mat3 by a vec2", () => {
    let prog = Fn(() => {
      let m = mat3(1).toVar();
      let v = vec2(1, 2).toVar();
      return m.mul(v).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("vec3(");
    expect(glsl).toContain(".xy");
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("vec3<f32>(");
  });
});

describe("JS target", () => {
  it("emits a callable expression reading inputs from ctx", () => {
    let src = compileJSFn((a: any, b: any) => a.add(b), {
      name: "add2", params: [{ name: "a", type: "float" }, { name: "b", type: "float" }],
    });
    expect(src).toContain("function add2(ctx)");
    expect(src).toContain("ctx.params");
    expect(src).toContain("return function add2");
  });

  it("hoists internal variables into a scratch block", () => {
    let prog = Fn(() => {
      let x = vec3(1, 2, 3).toVar();
      let y = x.add(vec3(1, 1, 1)).toVar();
      return y;
    });
    let src = compileJSFn(() => prog(), { name: "main", params: [] });
    // The scratch block lives outside the callable, one zeroed array per slot.
    expect(src).toMatch(/let _rmsl_\d+ = \[0, 0, 0\];/);
    // Assignments write into the hoisted slots rather than declaring them.
    expect(src).not.toContain("let _rmsl_0 = [1, 2, 3]");
  });

  it("emits out-parameter vector helpers for assignments", () => {
    let prog = Fn(() => {
      let x = vec3(1, 2, 3).toVar();
      x.assign(x.add(vec3(1, 1, 1)));
      return x;
    });
    let src = compileJSFn(() => prog(), { name: "main", params: [] });
    expect(src).toContain("function _v3add(a, b, out)");
    // The assignment passes the hoisted slot as the output.
    expect(src).toMatch(/_v3add\([^)]*,\s*_rmsl_\d+\);/);
  });

  it("uses Math builtins for scalar math", () => {
    let src = compileJSFn((a: any) => a.sin(), {
      name: "main", params: [{ name: "a", type: "float" }],
    });
    expect(src).toContain("Math.sin");
  });

  it("reads uniforms from ctx.uniforms by slot", () => {
    let u!: any;
    let prog = Fn(() => {
      u = uniform("float");
      return u.mul(2);
    });
    let src = compileJSFn(() => prog(), { name: "main", params: [] });
    expect(src).toContain(`ctx.uniforms["${u.name}"]`);
  });

  it("returns a result object when the program writes outputs or depth", () => {
    let prog = Fn(() => {
      let out = output("vec4");
      out.assign(vec4(1, 1, 1, 1));
      let d = builtinFragDepth();
      d.assign(float(0.5));
      return out;
    });
    let src = compileJSFn(() => prog(), { name: "main", params: [] });
    expect(src).toContain("var res = { outputs: {}, varyings: {} };");
    expect(src).toContain("res.outputs");
    expect(src).toContain("res.fragDepth");
    expect(src).toContain("res.value");
  });

  it("emits return null for discard", () => {
    let prog = Fn(() => {
      Discard();
      return float(1);
    });
    let src = compileJSFn(() => prog(), { name: "main", params: [] });
    expect(src).toContain("return null;");
  });

  it("reentrant mode declares variables inside the callable", () => {
    let prog = Fn(() => {
      let x = vec3(1, 2, 3).toVar();
      return x;
    });
    let src = compileJSFn(() => prog(), { name: "main", params: [], reentrant: true });
    expect(src).toMatch(/var _rmsl_\d+ = \[0, 0, 0\];/);
  });

  it("throws on derivative ops unless told to zero them", () => {
    let prog = Fn(() => {
      let x = vec2(1, 2).toVar();
      return x.fwidth();
    });
    expect(() => compileJSFn(() => prog(), { name: "main", params: [] }))
      .toThrow(/CPU target/);
    let src = compileJSFn(() => prog(), { name: "main", params: [], derivatives: "zero" });
    expect(src).toContain("0");
  });
});

describe("named toVar variables", () => {
  it("emits a user-provided name in GLSL", () => {
    let prog = Fn(() => {
      let color = vec3(1, 2, 3).toVar("color");
      return color.add(vec3(1)).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("vec3 color = vec3(1, 2, 3);");
  });

  it("emits a user-provided name in WGSL", () => {
    let prog = Fn(() => {
      let color = vec3(1, 2, 3).toVar("color");
      return color.add(vec3(1)).toVar();
    });
    let wgsl = compileWGSL(prog());
    expect(wgsl).toContain("var color: vec3<f32> = vec3<f32>(1, 2, 3);");
  });

  it("appends a number when the name is already taken", () => {
    let prog = Fn(() => {
      let a = float(1).toVar("x");
      let b = float(2).toVar("x");
      let c = float(3).toVar("x");
      return a.add(b).add(c).toVar();
    });
    let glsl = compileGLSL(prog());
    expect(glsl).toContain("float x = 1.0;");
    expect(glsl).toContain("float x1 = 2.0;");
    expect(glsl).toContain("float x2 = 3.0;");
  });

  it("the var() alias takes a name too", () => {
    let prog = Fn(() => {
      let uv = vec2(1, 2).var("uv");
      return uv;
    });
    expect(compileGLSL(prog())).toContain("vec2 uv = vec2(1, 2);");
    expect(compileWGSL(prog())).toContain("var uv: vec2<f32>");
  });

  it("rejects a name that is not a valid identifier", () => {
    let bad = () => Fn(() => float(1).toVar("my var"))();
    expect(() => compileGLSL(bad())).toThrow(/valid identifier/);
  });

  it("rejects a leading digit", () => {
    let bad = () => Fn(() => float(1).toVar("2x"))();
    expect(() => compileGLSL(bad())).toThrow(/valid identifier/);
  });

  it("rejects the reserved _rmsl_ prefix", () => {
    let bad = () => Fn(() => float(1).toVar("_rmsl_0"))();
    expect(() => compileGLSL(bad())).toThrow(/reserved/);
  });

  it("gives the same name to separate compiles", () => {
    let build = () => Fn(() => float(1).toVar("color"))();
    expect(compileGLSL(build())).toContain("float color = 1.0;");
    expect(compileGLSL(build())).toContain("float color = 1.0;");
  });

  it("emits a named variable into the JS scratch block", () => {
    let prog = Fn(() => {
      let x = vec3(1, 2, 3).toVar("local");
      return x;
    });
    let src = compileJSFn(() => prog(), { name: "main", params: [] });
    expect(src).toMatch(/let local = \[0, 0, 0\];/);
  });
});
