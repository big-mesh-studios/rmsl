import { afterAll, describe, expect, it } from "vitest";
import * as rmsl from "../rmsl";
import {
  attribute,
  float,
  Fn,
  If,
  instancedArray,
  int,
  Loop,
  mix,
  select,
  smoothstep,
  textureSize,
  uniform,
  uniformArray,
  vec3,
  vec4,
  type Node,
} from "../rmsl";
import { compileGlslFn } from "../glsl";
import { compileWgslFn } from "../wgsl";
import { compileJSFn, compileJSRoutine } from "../js";
import { compileWasmFn, compileWasmRoutine } from "../wasm";
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

/** The source of `build` compiled as a fragment stage on GLSL, WGSL and JS. */
function sources(build: () => Node<"vec4">) {
  return {
    glsl: compileGlsl.fragment(build()),
    wgsl: compileWgsl.fragment(build()),
    js: compileJSFn(build, none),
  };
}

describe("units of the core", () => {
  /**
   * GLSL accepts a scalar beside a vector in these built-ins, so a vector
   * constructor in its source shows the widening happened before the target.
   *
   * @canon spec-a-scalar-argument-beside-a-vector-is-widened-to-it
   */
  it.each([
    ["step", (v: Node<"vec3">) => rmsl.step(0.25, v), "step(vec3(0.25), ", 1],
    ["smoothstep", (v: Node<"vec3">) => rmsl.smoothstep(0, 1, v), "smoothstep(vec3(0.0), vec3(1.0), ", 0.352],
    ["clamp", (v: Node<"vec3">) => v.clamp(0.25, 0.5), ", vec3(0.25), vec3(0.5))", 0.4],
    ["min", (v: Node<"vec3">) => v.min(0.25), ", vec3(0.25))", 0.25],
    ["max", (v: Node<"vec3">) => v.max(0.25), ", vec3(0.25))", 0.4],
    ["pow", (v: Node<"vec3">) => v.pow(2), ", vec3(2.0))", 0.16],
    ["mod", (v: Node<"vec3">) => v.mod(0.25), ", vec3(0.25))", 0.15],
  ])("widens the scalar argument of %s to the vector on every target", (_, apply, glsl, expected) => {
    const u = uniform("vec3");
    expect(compileGlsl.fragment(vec4(apply(u), 1))).toContain(glsl);
    expect(compileWgsl.fragment(vec4(apply(u), 1))).toMatch(/vec3<f32>\((0\.25|0|1|2)f\)/);
    expect(compileJSFn(() => vec4(apply(u), 1), none)).toMatch(/\[(0\.25|0|1|2), \1, \1\]/);
    expect(evaluateRecording((a) => apply(vec3(a, 0.3, 0.6)).x, [0.4])).toBeCloseTo(expected, 12);
  });

  /**
   * @canon spec-the-weight-of-mix-stays-a-scalar
   */
  it("passes the weight of mix to every target as a scalar", () => {
    const u = uniform("vec3");
    const { glsl, wgsl, js } = sources(() => vec4(mix(u, vec3(1), 0.25), 1));
    expect(glsl).toMatch(/mix\(_rmsl_u\d+, vec3\(1\.0\), 0\.25\)/);
    expect(wgsl).toMatch(/mix\(_rmsl_uniforms\._rmsl_u\d+, vec3<f32>\(1f\), 0\.25f\)/);
    expect(js).toMatch(/_v3mix\(ctx\.uniforms\["_rmsl_u\d+"\], \[1, 1, 1\], 0\.25\)/);
    expect(evaluateRecording((a) => mix(vec3(a), vec3(1), 0.25).x, [-3])).toBe(-2);
  });

  /**
   * @canon spec-cbrt-is-composed-of-sign-abs-and-pow
   */
  it("composes cbrt of sign, abs and pow on every target", () => {
    const u = uniform("float");
    const { glsl, wgsl, js } = sources(() => vec4(u.cbrt(), 0, 0, 1));
    expect(glsl).toMatch(/sign\(_rmsl_u\d+\) \* pow\(abs\(_rmsl_u\d+\), 0\.333/);
    expect(wgsl).toMatch(/sign\(_rmsl_uniforms\._rmsl_u\d+\) \* pow\(abs\(_rmsl_uniforms\._rmsl_u\d+\), 0\.333/);
    expect(js).toContain("Math.sign(");
    expect(js).not.toContain("Math.cbrt");
    expect(evaluateRecording((a) => a.cbrt(), [-8])).toBe(-2);
  });

  /**
   * Two roots that each leave a variable unnamed declare two variables, and
   * building one `Fn` twice names its variable twice over.
   *
   * @canon spec-an-unnamed-variable-gets-a-name-no-other-variable-has
   */
  it("gives every unnamed variable a name of its own across Fns and builds", () => {
    const u = uniform("float");
    const first = Fn(() => u.add(1).toVar());
    const second = Fn(() => u.add(2).toVar());
    const declared = (src: string) => [...src.matchAll(/(?:float|var) (_rmsl_\d+)/g)].map((m) => m[1]);
    for (const src of [compileGlsl.fragment([first(), second()]), compileWgsl.fragment([first(), second()])]) {
      const names = declared(src);
      expect(names).toHaveLength(2);
      expect(new Set(names).size).toBe(2);
    }
    expect(declared(compileGlsl.fragment(first()))).not.toEqual(declared(compileGlsl.fragment(first())));
  });

  /**
   * The attributes are read in the other order from the one they were made in.
   *
   * @canon spec-wgsl-numbers-attributes-in-the-order-of-their-creation
   */
  it("gives the attribute made first location 0 on WGSL, whatever reads it first", () => {
    const made = attribute("vec2");
    const second = attribute("vec3");
    const wgsl = compileWgsl.vertex(vec4(second.add(vec3(made.x, made.y, 1)), 1));
    const slot = (node: Node<any>) => (node as any).value.slot as string;
    expect(wgsl).toContain(`@location(0) ${slot(made)}: vec2<f32>`);
    expect(wgsl).toContain(`@location(1) ${slot(second)}: vec3<f32>`);
  });

  /**
   * @canon spec-a-divisor-that-folds-to-zero-divides-like-a-literal-zero
   */
  it("divides by a subexpression that folds to zero as by a literal zero on WGSL", () => {
    const u = uniform("int");
    const wgsl = compileWgsl.fragment(vec4(float(u.div(int(2).sub(int(2)))), 0, 0, 1));
    expect(wgsl).toMatch(/_rmsl_u\d+ \/ 1i/);
    expect(
      evaluateRecording(
        (a) =>
          a
            .toInt()
            .div(int(2).sub(int(2)))
            .toFloat(),
        [7],
      ),
    ).toBe(7);
  });

  /**
   * @canon spec-a-select-on-a-comparison-of-integer-literals-folds-to-its-branch
   */
  it("folds a select whose condition compares integer literals", () => {
    const u = uniform("float");
    const { glsl, wgsl, js } = sources(() => vec4(select(int(1).lessThan(int(2)), u, float(6)), 0, 0, 1));
    expect(glsl).not.toContain("?");
    expect(glsl).toMatch(/vec4\(_rmsl_u\d+, 0\.0, 0\.0, 1\.0\)/);
    expect(wgsl).not.toContain("select(");
    expect(wgsl).toMatch(/vec4<f32>\(_rmsl_uniforms\._rmsl_u\d+, 0f, 0f, 1f\)/);
    expect(js).not.toContain("?");
    expect(evaluateRecording((a) => select(int(1).lessThan(int(2)), a, float(6)), [3])).toBe(3);
  });

  /**
   * @canon spec-the-main-entry-exports-no-compiler
   */
  it("exports the graph and serialization from the main entry, and no compiler", () => {
    const names = Object.keys(rmsl);
    expect(names).toEqual(expect.arrayContaining(["Fn", "serialize", "deserialize"]));
    expect(names.filter((name) => /^(compile|create|instantiate)/.test(name))).toEqual([]);
  });
});

describe("a uniform follows TSL", () => {
  /**
   * @canon spec-a-uniform-takes-the-type-it-names
   */
  it("declares a uniform of the type it names on GLSL and WGSL", () => {
    const tint = uniform("vec3");
    const build = () => vec4(tint, 1);
    expect(compileGlsl.fragment(build())).toMatch(new RegExp(`uniform (highp )?vec3 ${tint.name};`));
    expect(compileWgsl.fragment(build())).toMatch(new RegExp(`${tint.name}: vec3<f32>`));
  });

  /**
   * TSL's `uniform(0.5)` and `uniformArray([1, 2, 3], "float")` hold their
   * values; rmsl's take a type and read what the host passes.
   *
   * @canon exception-a-uniform-holds-no-value
   */
  it("takes a type, and reads the value the host passes at each call", () => {
    const scale = uniform("float");
    const weights = uniformArray("float", 3);
    const run = compileJSRoutine(() => Fn(() => scale.mul(weights.element(int(2))).toVar())(), none);
    expect(run.run({ uniforms: { [scale.name]: 2, [weights.name]: [1, 2, 3] } })).toBe(6);
    expect(run.run({ uniforms: { [scale.name]: 3, [weights.name]: [1, 2, 4] } })).toBe(12);
  });
});

describe("the variable names of a program", () => {
  /**
   * @canon spec-a-taken-variable-name-gets-a-number
   */
  it.each([
    ["GLSL", "float color = ", "float color1 = "],
    ["WGSL", "var color: f32", "var color1: f32"],
  ] as const)("numbers a name another root of the program took on %s", (target, first, second) => {
    const build = () => {
      const u = uniform("float");
      return [Fn(() => u.add(1).toVar("color"))(), Fn(() => u.add(2).toVar("color"))()];
    };
    const source = (target === "GLSL" ? compileGlsl : compileWgsl).fragment(build() as any);
    expect(source).toContain(first);
    expect(source).toContain(second);
  });

  /**
   * @canon spec-a-taken-variable-name-gets-a-number
   */
  it("keeps the name a root took when it is compiled alone after a program that renamed it, on GLSL and WGSL", () => {
    const u = uniform("float");
    const first = Fn(() => u.add(1).toVar("color"))();
    const second = Fn(() => u.add(2).toVar("color"))();
    expect(compileGlsl.fragment([first, second] as any)).toContain("float color1 = ");
    expect(compileGlsl.fragment(second as any)).toContain("float color = ");
    expect(compileWgsl.fragment([first, second] as any)).toContain("var color1: f32");
    expect(compileWgsl.fragment(second as any)).toContain("var color: f32");
  });

  /**
   * @canon spec-a-taken-variable-name-gets-a-number
   */
  it.each([
    ["JS", compileJSRoutine],
    ["WASM", compileWasmRoutine],
  ] as const)("keeps each root's own variable when two roots take one name on %s", (_target, compile) => {
    const out = instancedArray(2, "float");
    const build = () => [
      Fn(() => {
        const color = float(1).toVar("color");
        out.element(int(0)).assign(color.add(1));
      })(),
      Fn(() => {
        const color = float(10).toVar("color");
        out.element(int(1)).assign(color.add(1));
      })(),
    ];
    const data = new Float64Array(2);
    compile(build as any, none).run({ storages: { [out.name]: data } });
    expect(Array.from(data)).toEqual([2, 11]);
  });
});

describe("the type of the size of a texture", () => {
  /**
   * `textureSize` is typed `uvec2`, and GLSL returns an `ivec2`, so a variable of
   * the node's type refused it.
   *
   * @canon spec-a-conversion-between-numeric-types-is-written-out
   */
  it("converts the size of a texture to the unsigned type the node has on GLSL", () => {
    const glsl = compileGlsl.fragment(Fn(() => vec4(textureSize(uniform("sampler2D")).x.toFloat(), 0, 0, 1))() as any);
    expect(glsl).toContain("uvec2(textureSize(");
  });
});

describe("what a program does with a node it reads more than once", () => {
  /** A declaration of the variable a shared node gets. */
  const sharedLocal = /\b(?:float|vec\d|int|uint|bool|let|var|const) _rmsl_gen_\d+\b/;
  const param = { name: "main", params: [{ name: "a", type: "float" as const }] };

  /** A value whose every level reads the level below twice. */
  const nested = (levels: number) => (a: Node<"float">) =>
    Fn(() => {
      let x = vec3(a, a, a);
      for (let i = 0; i < levels; i++) x = x.add(x).mul(0.5);
      return x.x;
    })();

  /**
   * Twelve levels would give about 64 times the output of six if each level wrote
   * the one below out twice, and about 2 times if the output grows with the levels.
   *
   * @canon spec-output-grows-in-proportion-to-the-levels-of-nested-reads
   */
  it.each([
    ["GLSL", (build: any) => compileGlslFn(build, param).length],
    ["WGSL", (build: any) => compileWgslFn(build, param).length],
    ["JS", (build: any) => compileJSFn(build, param).length],
    ["WASM", (build: any) => compileWasmFn(build, param).bytes.length],
  ] as const)("compiles output that grows in proportion to the levels of nested reads on %s", (_target, size) => {
    expect(size(nested(12)) / size(nested(6))).toBeLessThan(4);
  });

  /**
   * @canon spec-output-grows-in-proportion-to-the-levels-of-nested-reads
   */
  it.each([
    ["fract", (x: any) => x.fract()],
    ["mod", (x: any) => x.mod(float(0.7))],
    ["mix", (x: any) => mix(x, float(1), float(0.5))],
    ["smoothstep", (x: any) => smoothstep(x, float(4), float(2))],
  ] as const)(
    "compiles a chain of %s that each level reads once in output that grows in proportion to the levels on JS",
    (_op, step) => {
      const chain = (levels: number) => (a: Node<"float">) =>
        Fn(() => {
          let x: any = a;
          for (let i = 0; i < levels; i++) x = step(x);
          return x;
        })();
      expect(compileJSFn(chain(12), param).length / compileJSFn(chain(6), param).length).toBeLessThan(4);
    },
  );

  /**
   * @canon spec-a-node-that-is-already-a-name-is-read-where-it-is
   */
  it("gives a node that is already a name no variable of its own on GLSL, WGSL and JS", () => {
    const build = () => {
      const u = uniform("float");
      const v = uniform("vec3");
      return vec4(
        u
          .mul(u)
          .add(v.x.mul(v.x))
          .add(float(2).mul(float(2))),
        0,
        0,
        1,
      );
    };
    const { glsl, wgsl, js } = sources(build);
    expect(glsl).not.toMatch(sharedLocal);
    expect(wgsl).not.toMatch(sharedLocal);
    expect(js).not.toMatch(sharedLocal);
  });

  /**
   * @canon spec-a-shared-value-gets-a-generated-name
   */
  it("names the variable of a shared node _rmsl_gen_ and a number on GLSL, WGSL and JS", () => {
    const build = () => {
      const u = uniform("float");
      const shared = u.sin().add(u.cos());
      return vec4(shared.mul(shared), 0, 0, 1);
    };
    const { glsl, wgsl, js } = sources(build);
    expect(glsl).toMatch(sharedLocal);
    expect(wgsl).toMatch(sharedLocal);
    expect(js).toContain("_rmsl_gen_0");
  });

  /**
   * @canon spec-a-shared-value-is-computed-again-in-a-block-it-is-not-visible-in
   */
  it("computes a value read in both branches of an If in each, on every target", () => {
    const build = (a: Node<"float">) =>
      Fn(() => {
        const shared = a.sin().add(a.cos());
        const r = float(0).toVar();
        If(a.greaterThan(0), () => {
          r.assign(shared);
        }).Else(() => {
          r.assign(shared.mul(10));
        });
        return r;
      })();
    expect(evaluateRecording(build, [1])).toBeCloseTo(Math.sin(1) + Math.cos(1), 10);
    expect(evaluateRecording(build, [-1])).toBeCloseTo((Math.sin(-1) + Math.cos(-1)) * 10, 10);
  });

  /**
   * @canon spec-a-shared-value-is-computed-again-after-what-it-reads-changed
   */
  it("computes a shared value again after a statement changed a variable it reads, on every target", () => {
    const build = (a: Node<"float">) =>
      Fn(() => {
        const x = a.toVar();
        const shared = x.mul(2).sin();
        const before = shared.add(1).toVar();
        x.assign(x.add(1));
        const after = shared.add(1).toVar();
        return before.add(after.mul(10));
      })();
    const want = (a: number) => Math.sin(2 * a) + 1 + (Math.sin(2 * (a + 1)) + 1) * 10;
    expect(evaluateRecording(build, [0.3])).toBeCloseTo(want(0.3), 10);
  });

  /**
   * @canon spec-an-expression-of-constants-is-folded-not-shared
   */
  it("gives an expression of constants no variable, however often it is read, on GLSL, WGSL and JS", () => {
    const build = () => {
      const u = uniform("float");
      const k = float(2).mul(3);
      return vec4(u.mul(k).add(k), 0, 0, 1);
    };
    const { glsl, wgsl, js } = sources(build);
    expect(glsl).not.toMatch(sharedLocal);
    expect(wgsl).not.toMatch(sharedLocal);
    expect(js).not.toContain("_rmsl_gen_");
  });

  /**
   * @canon spec-a-shared-value-is-computed-again-inside-a-loop-that-changes-what-it-reads
   */
  it("computes a shared value made before a loop again inside it when the loop changes what it reads, on every target", () => {
    const build = (a: Node<"float">) =>
      Fn(() => {
        const x = a.toVar();
        const shared = x.mul(x);
        const first = shared.add(1).toVar();
        const total = float(0).toVar();
        Loop(2, () => {
          x.assign(x.add(1));
          total.assign(total.add(shared.add(2)));
        });
        return first.mul(0.1).add(total.mul(0.1));
      })();
    const want = (a: number) => {
      let x = a;
      const first = x * x + 1;
      let total = 0;
      for (let i = 0; i < 2; i++) {
        x += 1;
        total += x * x + 2;
      }
      return first * 0.1 + total * 0.1;
    };
    expect(evaluateRecording(build, [0.3])).toBeCloseTo(want(0.3), 10);
  });
});
