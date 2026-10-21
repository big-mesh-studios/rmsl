import { NodeError } from "../errors";
import { afterAll, afterEach, describe, expect, it } from "vitest";
import {
  attribute,
  builtinFragDepth,
  builtinPosition,
  fragCoord,
  varying,
  Fn,
  For,
  If,
  Return,
  float,
  instancedArray,
  int,
  uniform,
  vec3,
  vec4,
  type Node,
} from "../rmsl";
import { compileWgsl as compileWgslStage, compileWgslFn } from "../wgsl";
import { compileGlsl as compileGlslStage, compileGlslFn } from "../glsl";
import { deserialize, serialize } from "../serialize";
import { compileJSFn, compileJSFragment, compileJSRoutine, compileJSVertex } from "../js";
import { compileWasmFn, compileWasmFragment, compileWasmRoutine, compileWasmVertex } from "../wasm";
import {
  assertRecordedShadersValid,
  recordingGLSL as compileGlsl,
  expectCompileRejection,
  recordingWGSL as compileWgsl,
} from "../testing/shader-validity";
import {
  assertEvaluationsOfTheTestAgree,
  assertRecordedEvaluationsAgree,
  closeEvaluators,
  evaluateRecording,
} from "../testing/shader-eval";

afterEach(assertEvaluationsOfTheTestAgree, 120_000);

afterAll(async () => {
  await assertRecordedEvaluationsAgree();
  await assertRecordedShadersValid();
  await closeEvaluators();
}, 120_000);

const none = { name: "main", params: [] };

/** The CPU targets, each compiling a program to a routine its tests call with inputs. */
const routines = [
  ["JS", (build: () => Node<any>) => compileJSRoutine(build, none) as (inputs: object) => unknown],
  ["WASM", (build: () => Node<any>) => compileWasmRoutine(build, none) as (inputs: object) => unknown],
] as const;

/** A function of two vectors, a clamped dot product, with the layout it is declared by. */
const makeShade = (name?: string) =>
  Fn(([n, l]: [Node<"vec3">, Node<"vec3">]) => n.dot(l).max(0)).setLayout({
    name,
    type: "float",
    inputs: [
      { name: "n", type: "vec3" },
      { name: "l", type: "vec3" },
    ],
  });

/** A fragment stage that lights with `shade` from two directions. */
const lit = (shade: (...args: any[]) => any) => () => {
  const a = uniform("vec3");
  const b = uniform("vec3");
  return Fn(() => {
    const first = shade(a, b);
    const second = shade(b, a);
    return vec4(first.add(second), 0, 0, 1);
  })();
};

/** The number of functions a WebAssembly module declares, read from its function section. */
function functionCount(bytes: Uint8Array): number {
  let at = 8;
  const leb = () => {
    let value = 0;
    for (let shift = 0; ; shift += 7) {
      const byte = bytes[at++]!;
      value |= (byte & 0x7f) << shift;
      if (!(byte & 0x80)) return value;
    }
  };
  while (at < bytes.length) {
    const id = bytes[at++]!;
    const size = leb();
    if (id === 3) return leb();
    at += size;
  }
  return 0;
}

describe("an Fn with a layout is a function of its own", () => {
  /**
   * @canon spec-a-layout-fn-is-declared-once-and-called-at-each-use
   */
  it("declares the function once and calls it twice on GLSL", () => {
    const code = compileGlsl.fragment(lit(makeShade("shade"))());
    expect(code.match(/float shade\(vec3 n, vec3 l\)/g)).toHaveLength(1);
    expect(code.match(/= shade\(/g)).toHaveLength(2);
  });

  /**
   * @canon spec-a-layout-fn-is-declared-once-and-called-at-each-use
   */
  it("declares the function once and calls it twice on WGSL", () => {
    const code = compileWgsl.fragment(lit(makeShade("shade"))());
    expect(code.match(/fn shade\(n: vec3<f32>, l: vec3<f32>\) -> f32/g)).toHaveLength(1);
    expect(code.match(/= shade\(/g)).toHaveLength(2);
  });

  /**
   * @canon spec-a-cpu-target-runs-the-body-of-a-layout-fn-where-it-is-called
   */
  it("declares no function for it in the JavaScript source", () => {
    const source = compileJSFragment(lit(makeShade("shade")), none) as unknown as { source?: string };
    const text = compileJSFn(() => makeShade("shade")(vec3(1, 0, 0), vec3(1, 0, 0)), none);
    expect(text).not.toMatch(/function shade/);
    expect(text.match(/function /g)).toHaveLength(1);
    expect(source).toBeDefined();
  });

  /**
   * @canon spec-a-cpu-target-runs-the-body-of-a-layout-fn-where-it-is-called
   */
  it("declares no function for it in the WebAssembly module", () => {
    const shade = makeShade("shade");
    const inline = Fn(([n, l]: [Node<"vec3">, Node<"vec3">]) => n.dot(l).max(0));
    const program = (f: (...args: any[]) => any) => () => Fn(() => f(vec3(1, 2, 3), vec3(0, 1, 0)).toVar())();
    expect(functionCount(compileWasmFn(program(shade), none) as unknown as Uint8Array)).toBe(
      functionCount(
        compileWasmFn(
          program((n: any, l: any) => inline([n, l] as any)),
          none,
        ) as unknown as Uint8Array,
      ),
    );
  });

  /**
   * @canon spec-a-layout-fn-computes-the-same-on-every-target
   */
  it("computes a call twice with different arguments on every target", () => {
    const mad = Fn(([x, y]: [Node<"float">, Node<"float">]) => x.mul(y).add(1)).setLayout({
      name: "mad",
      type: "float",
      inputs: [
        { name: "x", type: "float" },
        { name: "y", type: "float" },
      ],
    });
    expect(evaluateRecording((a, b) => Fn(() => mad(a, b).add(mad(b, a.add(1))))(), [2, 3])).toBe(7 + 10);
  });

  /**
   * @canon spec-a-layout-fn-computes-the-same-on-every-target
   */
  it("computes a vector argument, a vector result and a local variable of the body on every target", () => {
    const scaled = Fn(([v, k]: [Node<"vec3">, Node<"float">]) => {
      const doubled = v.mul(2).toVar("doubled");
      return doubled.mul(k);
    }).setLayout({
      name: "scaled",
      type: "vec3",
      inputs: [
        { name: "v", type: "vec3" },
        { name: "k", type: "float" },
      ],
    });
    expect(
      evaluateRecording((a) => Fn(() => scaled(vec3(a, 1, 2), float(3)).add(scaled(vec3(1, 1, 1), a)))(), [1]),
    ).toEqual(new Float64Array([6 + 2, 6 + 2, 12 + 2]));
  });

  /**
   * @canon spec-a-layout-fn-takes-its-inputs-as-an-array-and-by-name
   */
  it("destructures the inputs as an array and as an object, and takes the call by position and by name", () => {
    const layout = {
      name: "pair",
      type: "float" as const,
      inputs: [
        { name: "a", type: "float" as const },
        { name: "b", type: "float" as const },
      ],
    };
    const asArray = Fn(([a, b]: [Node<"float">, Node<"float">]) => a.sub(b)).setLayout(layout);
    const asObject = Fn(({ a, b }: { a: Node<"float">; b: Node<"float"> }) => a.sub(b)).setLayout(layout);
    expect(evaluateRecording((x, y) => Fn(() => asArray(x, y))(), [5, 2])).toBe(3);
    expect(evaluateRecording((x, y) => Fn(() => (asObject as any)(x, y))(), [5, 2])).toBe(3);
    const byName = asArray as unknown as (inputs: { a: Node<"float">; b: Node<"float"> }) => Node<"float">;
    expect(evaluateRecording((x, y) => Fn(() => byName({ a: x, b: y }))(), [5, 2])).toBe(3);
  });

  /**
   * @canon spec-a-layout-fn-takes-its-inputs-as-an-array-and-by-name
   */
  it("takes the layout as the second argument of Fn", () => {
    const twice = Fn(([x]: [Node<"float">]) => x.mul(2), {
      name: "twice",
      type: "float",
      inputs: [{ name: "x", type: "float" }],
    });
    expect(compileGlsl.fragment(Fn(() => vec4(twice(uniform("float")), 0, 0, 1))())).toMatch(/float twice\(float x\)/);
  });

  /**
   * @canon spec-a-layout-fn-without-a-name-gets-one
   */
  it("declares a function with no name under a generated one", () => {
    const code = compileGlsl.fragment(lit(makeShade())());
    expect(code).toMatch(/float (_rmsl_fn_\d+)\(vec3 n, vec3 l\)/);
    const wgsl = compileWgsl.fragment(lit(makeShade())());
    expect(wgsl).toMatch(/fn (_rmsl_fn_\d+)\(n: vec3<f32>, l: vec3<f32>\) -> f32/);
  });

  /**
   * @canon spec-a-taken-function-name-gets-a-number
   */
  it("declares a second function of a taken name with a number", () => {
    const first = makeShade("dup");
    const second = Fn(([x]: [Node<"float">]) => x.mul(3)).setLayout({
      name: "dup",
      type: "float",
      inputs: [{ name: "x", type: "float" }],
    });
    const build = () => {
      const u = uniform("float");
      return Fn(() => vec4(first(vec3(u), vec3(1, 1, 1)).add(second(u)), 0, 0, 1))();
    };
    for (const code of [compileGlsl.fragment(build()), compileWgsl.fragment(build())]) {
      expect(code).toMatch(/\bdup\(/);
      expect(code).toMatch(/\bdup_1\(/);
    }
  });

  /**
   * @canon spec-return-leaves-a-void-layout-fn
   */
  it("leaves only the function at a Return, on the CPU targets", () => {
    const out = instancedArray(2, "float");
    const early = Fn(([x]: [Node<"float">]) => {
      If(x.greaterThan(1), () => {
        Return();
      });
      out.element(int(0)).assign(x);
    }).setLayout({ name: "early", type: "void", inputs: [{ name: "x", type: "float" }] });
    const x = uniform("float");
    const build = () =>
      Fn(() => {
        early(x);
        out.element(int(1)).assign(x.add(10));
      })();
    for (const [name, compile] of routines) {
      const run = (value: number) => {
        const data = new Float64Array(2).fill(-1);
        compile(build)({ storages: { [out.name]: data }, uniforms: { [x.name]: value } });
        return [...data];
      };
      expect(run(0.5), name).toEqual([0.5, 10.5]);
      expect(run(3), name).toEqual([-1, 13]);
    }
  });

  /**
   * @canon spec-return-leaves-a-void-layout-fn
   */
  it("writes the Return into the function on GLSL and WGSL", () => {
    const early = Fn(([x]: [Node<"float">]) => {
      const seen = float(0).toVar("seen");
      If(x.greaterThan(1), () => {
        Return();
      });
      seen.addAssign(x);
    }).setLayout({ name: "early", type: "void", inputs: [{ name: "x", type: "float" }] });
    const build = () =>
      Fn(() => {
        early(uniform("float"));
        return vec4(0, 0, 0, 1);
      })();
    const glsl = compileGlsl.fragment(build());
    expect(glsl).toMatch(/void early\(float x\) \{[^}]*\{\s*return;\s*\}/);
    expect(glsl.slice(glsl.indexOf("void main"))).not.toMatch(/return;/);
    const wgsl = compileWgsl.fragment(build());
    expect(wgsl).toMatch(/fn early\(x: f32\) \{[^}]*\{\s*return;\s*\}/);
    expect(wgsl.slice(wgsl.indexOf("@fragment"))).not.toMatch(/return;/);
  });

  /**
   * @canon spec-a-layout-fn-may-call-a-layout-fn
   */
  it("declares the callee ahead of the caller and computes the same on every target", () => {
    const inner = Fn(([x]: [Node<"float">]) => x.add(1)).setLayout({
      name: "inner",
      type: "float",
      inputs: [{ name: "x", type: "float" }],
    });
    const outer = Fn(([x]: [Node<"float">]) => inner(x).mul(inner(x.mul(2)))).setLayout({
      name: "outer",
      type: "float",
      inputs: [{ name: "x", type: "float" }],
    });
    const build = () => Fn(() => vec4(outer(uniform("float")), 0, 0, 1))();
    const glsl = compileGlsl.fragment(build());
    expect(glsl.indexOf("float inner(")).toBeGreaterThan(-1);
    expect(glsl.indexOf("float inner(")).toBeLessThan(glsl.indexOf("float outer("));
    compileWgsl.fragment(build());
    expect(evaluateRecording((a) => Fn(() => outer(a))(), [3])).toBe(4 * 7);
  });

  /**
   * @canon spec-a-layout-fn-reads-a-uniform
   */
  it("reads a uniform inside the function on every target", () => {
    const gain = uniform("float");
    const scaled = Fn(([x]: [Node<"float">]) => x.mul(gain)).setLayout({
      name: "scaled",
      type: "float",
      inputs: [{ name: "x", type: "float" }],
    });
    const build = () => Fn(() => vec4(scaled(float(2)), 0, 0, 1))();
    const glsl = compileGlsl.fragment(build());
    expect(glsl).toMatch(new RegExp(`float scaled\\(float x\\) \\{\\s*return x \\* ${gain.name};`));
    const wgsl = compileWgsl.fragment(build());
    expect(wgsl).toMatch(new RegExp(`fn scaled\\(x: f32\\) -> f32 \\{\\s*return x \\* _rmsl_uniforms\\.${gain.name};`));
    for (const [name, compile] of routines) {
      expect(compile(() => Fn(() => scaled(float(2)).toVar())())({ uniforms: { [gain.name]: 4 } }), name).toBe(8);
    }
  });

  /**
   * @canon spec-a-layout-fn-writes-a-storage-element
   */
  it("writes a storage element inside the function on every target that has storage", () => {
    const out = instancedArray(1, "float");
    const store = Fn(([x]: [Node<"float">]) => {
      out.element(int(0)).assign(x.mul(2));
    }).setLayout({ name: "store", type: "void", inputs: [{ name: "x", type: "float" }] });
    const build = () =>
      Fn(() => {
        store(float(21));
      })();
    expect(compileWgslStage.compute(build())).toMatch(/fn store\(x: f32\) \{/);
    for (const [name, compile] of routines) {
      const data = new Float64Array(1);
      compile(build)({ storages: { [out.name]: data } });
      expect(data[0], name).toBe(42);
    }
  });

  /**
   * @canon spec-a-graph-with-a-layout-fn-survives-json
   */
  it("compiles a graph rebuilt from JSON to the same code, with one function for the calls that shared it", () => {
    const build = lit(makeShade("shade"));
    const original = build();
    const restored = deserialize(JSON.parse(JSON.stringify(serialize(original)))) as Node<"vec4">;
    const normalize = (code: string) => code.replace(/_rmsl_[uav]?\d+/g, "_rmsl_x");
    expect(normalize(compileGlsl.fragment(restored))).toBe(normalize(compileGlsl.fragment(original)));
    expect(normalize(compileWgsl.fragment(restored))).toBe(normalize(compileWgsl.fragment(original)));
    expect(compileGlsl.fragment(restored).match(/float shade\(/g)).toHaveLength(1);
    const run = (root: Node<"vec4">) =>
      compileJSFragment(() => root, none) as unknown as (ctx: object) => { value: number[] };
    expect(run(restored)).toBeDefined();
    expect(() => compileWasmFragment(() => restored, none)).not.toThrow();
  });
});

describe("a mistake in a function with a layout is refused", () => {
  const layout = (type: "float" | "void" = "float") => ({
    name: "f",
    type,
    inputs: [{ name: "x", type: "float" as const }],
  });

  /**
   * @canon spec-a-call-with-the-wrong-number-of-arguments-is-refused
   */
  it("refuses a call with too few or too many arguments", () => {
    const f = Fn(([x]: [Node<"float">]) => x).setLayout(layout()) as (...args: any[]) => any;
    expect(() => Fn(() => f())()).toThrowRmsl(NodeError, "argument-count");
    expect(() => Fn(() => f(float(1), float(2)))()).toThrowRmsl(NodeError, "argument-count");
  });

  /**
   * @canon spec-a-call-with-an-argument-of-the-wrong-type-is-refused
   */
  it("refuses an argument of another type, and takes a number for a float", () => {
    const f = Fn(([x]: [Node<"float">]) => x).setLayout(layout()) as (...args: any[]) => any;
    expect(() => Fn(() => f(vec3(1, 2, 3)))()).toThrowRmsl(NodeError, "argument-of-another-type");
    expect(() => Fn(() => f(int(1)))()).toThrowRmsl(NodeError, "argument-of-another-type");
    expect(() => Fn(() => f(2))()).not.toThrow();
  });

  /**
   * @canon spec-a-body-that-returns-another-type-than-its-layout-is-refused
   */
  it("refuses a body that returns another type, or nothing, or a value for void", () => {
    const wrong = Fn(([x]: [Node<"float">]) => vec3(x)).setLayout(layout()) as (...args: any[]) => any;
    expect(() => Fn(() => wrong(1))()).toThrowRmsl(NodeError, "result-of-another-type");
    const nothing = Fn(([x]: [Node<"float">]) => {
      x.toVar();
    }).setLayout(layout()) as (...args: any[]) => any;
    expect(() => Fn(() => nothing(1))()).toThrowRmsl(NodeError, "result-of-another-type");
    const value = Fn(([x]: [Node<"float">]) => x).setLayout(layout("void")) as (...args: any[]) => any;
    expect(() => Fn(() => value(1))()).toThrowRmsl(NodeError, "result-of-another-type");
  });

  /**
   * @canon spec-a-layout-fn-that-reads-a-variable-of-its-caller-is-refused
   */
  it("refuses a body that reads a variable of its caller, and names it", () => {
    expect(() =>
      Fn(() => {
        const total = float(0).toVar("total");
        const read = Fn(() => total.mul(2)).setLayout({ name: "read", type: "float", inputs: [] });
        return read();
      })(),
    ).toThrowRmsl(
      NodeError,
      "layout-fn-uses-outer-variable",
      /Fn "read" uses the variable "total", which is declared outside it/,
    );
  });

  /**
   * @canon spec-a-layout-fn-that-writes-a-variable-of-its-caller-is-refused
   */
  it("refuses a body that assigns a variable of its caller, and names it", () => {
    expect(() =>
      Fn(() => {
        const total = float(0).toVar("total");
        const accumulate = Fn(([x]: [Node<"float">]) => {
          total.addAssign(x);
        }).setLayout({ name: "accumulate", type: "void", inputs: [{ name: "x", type: "float" }] });
        (accumulate as (...args: any[]) => any)(1);
      })(),
    ).toThrowRmsl(
      NodeError,
      "layout-fn-uses-outer-variable",
      /Fn "accumulate" uses the variable "total", which is declared outside it/,
    );
  });

  /**
   * The names a body may use are its inputs, its own variables, and the globals.
   *
   * @canon spec-a-layout-fn-that-reads-a-variable-of-its-caller-is-refused
   */
  it("accepts a body that uses only its inputs, its own variables and the globals", () => {
    const gain = uniform("float");
    const ok = Fn(([x]: [Node<"float">]) => {
      const sum = float(0).toVar("sum");
      For(
        () => int(0).toVar(),
        (i) => i.lessThan(3),
        (i) => i.addAssign(1),
        () => sum.addAssign(x.mul(gain)),
      );
      return sum;
    }).setLayout(layout()) as (...args: any[]) => any;
    expect(() => Fn(() => ok(1))()).not.toThrow();
  });

  /**
   * @canon spec-a-compiler-refuses-a-layout-fn-that-names-a-variable-of-its-caller
   */
  it("refuses it again on every target when a graph is rebuilt from JSON with a body that names a variable of the caller", () => {
    const program = Fn(() => {
      const caller = float(0).toVar("caller");
      const ok = Fn(([x]: [Node<"float">]) => x.mul(2)).setLayout(layout());
      return vec4(ok(caller), 0, 0, 1);
    })();
    const graph = serialize(program);
    const call = graph.nodes.find((n) => n.type === "call")!;
    const body = graph.nodes[(call.value as { body: number }).body]!;
    const caller = graph.nodes.findIndex((n) => n.type === "var" && (n.value as any).varName === "caller");
    body.params![body.params!.length - 1] = caller;
    const restored = () => deserialize(JSON.parse(JSON.stringify(graph))) as Node<"vec4">;
    const refusal = /uses the variable "caller", which is declared outside it/;
    expect(() => compileGlsl.fragment(restored())).toThrowRmsl(NodeError, "layout-fn-uses-outer-variable", refusal);
    expect(() => compileWgsl.fragment(restored())).toThrowRmsl(NodeError, "layout-fn-uses-outer-variable", refusal);
    expect(() => compileJSFragment(restored, none)).toThrowRmsl(NodeError, "layout-fn-uses-outer-variable", refusal);
    expect(() => compileWasmFragment(restored, none)).toThrowRmsl(NodeError, "layout-fn-uses-outer-variable", refusal);
  });

  /**
   * @canon spec-return-in-a-layout-fn-that-gives-a-value-is-refused
   */
  it("refuses a Return in a function that gives a value, on every target", () => {
    const early = Fn(([x]: [Node<"float">]) => {
      If(x.greaterThan(1), () => {
        Return();
      });
      return x;
    }).setLayout(layout());
    const build = () => Fn(() => vec4(early(uniform("float")), 0, 0, 1))();
    const refusal = /Fn "f" gives a float, and its body calls Return\(\)/;
    expect(() => compileGlsl.fragment(build())).toThrowRmsl(NodeError, "return-without-value", refusal);
    expect(() => compileWgsl.fragment(build())).toThrowRmsl(NodeError, "return-without-value", refusal);
    expect(() => compileJSFragment(build, none)).toThrowRmsl(NodeError, "return-without-value", refusal);
    expect(() => compileWasmFragment(build, none)).toThrowRmsl(NodeError, "return-without-value", refusal);
  });

  /**
   * @canon spec-a-layout-fn-writes-no-output-of-its-stage
   */
  it.each([
    ["the fragment depth", "fragment", () => builtinFragDepth().assign(float(0.5)), /uses the fragment depth/],
    ["the position", "vertex", () => builtinPosition().assign(vec4(0, 0, 0, 1)), /uses the position/],
  ] as const)("refuses a body that writes %s, on every target", (_, stage, write, refusal) => {
    const f = Fn(() => {
      write();
      return float(1);
    }).setLayout({ name: "f", type: "float", inputs: [] });
    const build = () => Fn(() => vec4(f(), 0, 0, 1))();
    const [glsl, wgsl, js, wasm] =
      stage === "fragment"
        ? [compileGlsl.fragment, compileWgsl.fragment, compileJSFragment, compileWasmFragment]
        : [compileGlsl.vertex, compileWgsl.vertex, compileJSVertex, compileWasmVertex];
    expect(() => (glsl as any)(build())).toThrowRmsl(NodeError, "layout-fn-uses-stage-value", refusal);
    expect(() => (wgsl as any)(build())).toThrowRmsl(NodeError, "layout-fn-uses-stage-value", refusal);
    expect(() => (js as any)(build, none)).toThrowRmsl(NodeError, "layout-fn-uses-stage-value", refusal);
    expect(() => (wasm as any)(build, none)).toThrowRmsl(NodeError, "layout-fn-uses-stage-value", refusal);
  });

  /** A stage input a layout `Fn` reads, built into a function and a fragment stage that calls it. */
  const readsStageInput = (name: string, read: () => Node<"float">) => {
    const f = Fn(() => read()).setLayout({ name, type: "float", inputs: [] });
    return () => Fn(() => vec4(0, f().add(f()), 0, 1))();
  };
  const aVarying = varying("float");
  const reads = {
    varying: readsStageInput("readsVarying", () => aVarying.mul(2)),
    builtIn: readsStageInput("readsFragCoord", () => fragCoord().x.mul(0).add(0.25)),
  };

  /**
   * @canon spec-a-layout-fn-reads-a-varying
   */
  it("reads a varying from a function, on GLSL, JS and WASM", () => {
    const build = reads.varying;
    expect(compileGlslStage.fragment(build())).toContain("readsVarying");
    // The function reads 0.25 twice, doubled each time.
    for (const compile of [compileJSFragment, compileWasmFragment]) {
      const stage = (compile as any)(build, none) as (ctx: object) => { value: ArrayLike<number> } | undefined;
      expect(Array.from(stage({ varyings: { [aVarying.name]: 0.25 } })!.value)).toEqual([0, 1, 0, 1]);
    }
  });

  /**
   * @canon spec-a-layout-fn-reads-a-built-in-input
   */
  it("reads fragCoord() from a function, on GLSL, JS and WASM", () => {
    const build = reads.builtIn;
    expect(compileGlslStage.fragment(build())).toContain("readsFragCoord");
    for (const compile of [compileJSFragment, compileWasmFragment]) {
      const stage = (compile as any)(build, none) as (ctx: object) => { value: ArrayLike<number> } | undefined;
      expect(Array.from(stage({ fragCoord: [3, 4] })!.value)).toEqual([0, 0.5, 0, 1]);
    }
  });

  /** A vertex stage whose layout `Fn` assigns the varying and returns 1. */
  const assignsVarying = () => {
    const written = varying("float");
    const f = Fn(() => {
      written.assign(float(0.5));
      return float(1);
    }).setLayout({ name: "assignsVarying", type: "float", inputs: [] });
    return { written, build: () => Fn(() => vec4(f(), 0, 0, 1))() };
  };

  /**
   * @canon spec-a-layout-fn-assigns-a-varying
   */
  it("assigns a varying from a function, on GLSL, JS and WASM", () => {
    const { written, build } = assignsVarying();
    expect(compileGlslStage.vertex(build())).toMatch(new RegExp(`${written.name} = 0\\.5`));
    for (const compile of [compileJSVertex, compileWasmVertex]) {
      const stage = (compile as any)(build, none) as (ctx: object) => {
        varyings: Record<string, ArrayLike<number> | number>;
      };
      expect(stage({}).varyings[written.name]).toBe(0.5);
    }
  });

  /**
   * @canon exception-wgsl-refuses-a-layout-fn-that-reaches-a-value-of-its-stage
   */
  it.each([
    ["an attribute", readsStageInput("f", () => attribute("float").add(0)), /uses an attribute/],
    ["a varying", reads.varying, /uses a varying/],
    ["fragCoord()", reads.builtIn, /uses fragCoord\(\)/],
  ] as const)("refuses a body that reads %s on WGSL", (_, build, refusal) => {
    expect(expectCompileRejection(() => compileWgsl.fragment(build()))).toMatch(refusal);
  });

  /**
   * @canon exception-wgsl-refuses-a-layout-fn-that-reaches-a-value-of-its-stage
   */
  it("refuses a body that assigns a varying on WGSL", () => {
    const { build } = assignsVarying();
    expect(expectCompileRejection(() => compileWgsl.vertex(build()))).toMatch(/uses a varying/);
  });

  /**
   * @canon spec-a-malformed-layout-is-refused
   */
  it.each([
    ["no type", { name: "f", inputs: [] }, /has the type undefined/],
    ["a type that is a sampler", { name: "f", type: "sampler2D", inputs: [] }, /has the type "sampler2D"/],
    ["inputs that are no list", { name: "f", type: "float", inputs: 3 }, /has no inputs list/],
    ["an input with no type", { name: "f", type: "float", inputs: [{ name: "x" }] }, /input "x" of the type undefined/],
    [
      "an input with no name",
      { name: "f", type: "float", inputs: [{ type: "float" }] },
      /input with the name undefined/,
    ],
    [
      "two inputs of one name",
      {
        name: "f",
        type: "float",
        inputs: [
          { name: "x", type: "float" },
          { name: "x", type: "float" },
        ],
      },
      /two inputs named "x"/,
    ],
    ["a name that is no identifier", { name: "1f", type: "float", inputs: [] }, /no identifier/],
    ["a name with the reserved prefix", { name: "_rmsl_f", type: "float", inputs: [] }, /starts with "_rmsl_"/],
  ] as const)("refuses a layout with %s", (_, malformed, refusal) => {
    expect(() => Fn(() => float(1)).setLayout(malformed as any)).toThrowRmsl(NodeError, "invalid-layout", refusal);
    expect(() => Fn(() => float(1), malformed as any)).toThrowRmsl(NodeError, "invalid-layout", refusal);
  });

  /**
   * @canon spec-a-malformed-layout-is-refused
   */
  it.each<[string, (call: any) => void]>([
    ["a function name that is code", (call) => void (call.value.name = "f(){} void evil")],
    ["an input name that is code", (call) => void (call.value.inputs[0].name = "x) { } void evil(float y")],
    ["an input name that is a keyword", (call) => void (call.value.inputs[0].name = "in")],
    ["an input type that is code", (call) => void (call.value.inputs[0].type = "float x) {} void evil(float")],
  ])("refuses a graph rebuilt from JSON with %s, on every target", (_, tamper) => {
    const ok = Fn(([x]: [Node<"float">]) => x.mul(2)).setLayout(layout());
    const graph = serialize(Fn(() => vec4(ok(uniform("float")), 0, 0, 1))());
    tamper(graph.nodes.find((n) => n.type === "call"));
    const restored = () => deserialize(JSON.parse(JSON.stringify(graph))) as Node<"vec4">;
    expect(() => compileGlsl.fragment(restored())).toThrowRmsl(NodeError, "invalid-layout");
    expect(() => compileWgsl.fragment(restored())).toThrowRmsl(NodeError, "invalid-layout");
    expect(() => compileJSFragment(restored, none)).toThrowRmsl(NodeError, "invalid-layout");
    expect(() => compileWasmFragment(restored, none)).toThrowRmsl(NodeError, "invalid-layout");
  });

  /**
   * @canon spec-a-taken-function-name-gets-a-number
   */
  it("numbers a layout function named like the function a compile wraps the program in", () => {
    const scale = Fn(([x]: [Node<"float">]) => x.mul(2)).setLayout({
      name: "scale",
      type: "float",
      inputs: [{ name: "x", type: "float" }],
    });
    const build = (inputs: Node<"float">) => Fn(() => scale(inputs))();
    const options = { name: "scale", params: [{ name: "v", type: "float" as const }], returnType: "float" as const };
    expect(compileGlslFn(build as any, options as any)).toMatch(
      /float scale_1\(float x\)[\s\S]*float scale\(float v\)/,
    );
    expect(compileWgslFn(build as any, options as any)).toMatch(/fn scale_1\(x: f32\)[\s\S]*fn scale\(v: f32\)/);
  });

  /**
   * @canon spec-a-layout-fn-is-declared-once-and-called-at-each-use
   */
  it("evaluates a shared argument of a void call once, as the argument of an assignment is", () => {
    const out = instancedArray(2, "float");
    const store = Fn(([x]: [Node<"float">]) => {
      out.element(int(0)).assign(x);
    }).setLayout({ name: "store", type: "void", inputs: [{ name: "x", type: "float" }] });
    const u = uniform("float");
    const wgsl = compileWgslStage.compute(
      Fn(() => {
        const shared = u.sin();
        store(shared.add(shared));
        store(shared);
      })(),
    );
    expect(wgsl.match(/sin\(/g)).toHaveLength(1);
  });

  /**
   * @canon spec-a-layout-fn-is-declared-once-and-called-at-each-use
   */
  it("calls a function once for a result that is read twice outside an Fn", () => {
    const twice = Fn(([x]: [Node<"float">]) => x.mul(2)).setLayout({
      name: "twice",
      type: "float",
      inputs: [{ name: "x", type: "float" }],
    });
    const result = twice(float(2));
    const build = () => Fn(() => vec4(result.add(result), 0, 0, 1))();
    expect(compileGlsl.fragment(build()).match(/twice\(2\.0\)/g)).toHaveLength(1);
    expect(compileWgsl.fragment(build()).match(/twice\(2f\)/g)).toHaveLength(1);
  });

  /**
   * @canon spec-a-call-with-an-argument-of-the-wrong-type-is-refused
   */
  it("refuses a node of another type than its input, and says how to convert it", () => {
    const next = Fn(([n]: [Node<"int">]) => n.add(1)).setLayout({
      name: "next",
      type: "int",
      inputs: [{ name: "n", type: "int" }],
    }) as (...args: any[]) => any;
    expect(() => next(float(3))).toThrowRmsl(NodeError, "argument-of-another-type", /int/);
    expect(() => next(int(3))).not.toThrow();
  });

  /**
   * @canon spec-a-number-for-a-layout-fn-input-takes-the-input-type
   */
  it("passes a number as the type of its input, on every target", () => {
    const next = Fn(([n]: [Node<"int">]) => n.add(1)).setLayout({
      name: "next",
      type: "int",
      inputs: [{ name: "n", type: "int" }],
    }) as (...args: any[]) => any;
    const scale = Fn(([v, k]: [Node<"vec3">, Node<"uint">]) => v.mul(k.toFloat())).setLayout({
      name: "scale",
      type: "vec3",
      inputs: [
        { name: "v", type: "vec3" },
        { name: "k", type: "uint" },
      ],
    }) as (...args: any[]) => any;
    const build = () => Fn(() => vec4(scale(2, 3), next(3).toFloat()))();
    expect(compileGlsl.fragment(build())).toMatch(/next\(3\)/);
    expect(compileWgsl.fragment(build())).toMatch(/next\(3i\)/);
    expect(compileWgsl.fragment(build())).toMatch(/scale\(vec3<f32>\(2f\), 3u\)/);
    expect(Array.from(evaluateRecording(build as any) as ArrayLike<number>)).toEqual([6, 6, 6, 4]);
    const flag = Fn(([b]: [Node<"bool">]) => b.not()).setLayout({
      name: "flag",
      type: "bool",
      inputs: [{ name: "b", type: "bool" }],
    }) as (...args: any[]) => any;
    expect(() => flag(1)).toThrowRmsl(NodeError, "argument-of-another-type", /"b".*bool/);
  });

  /**
   * @canon spec-a-fraction-or-a-number-out-of-range-for-an-integer-input-is-refused
   */
  it("refuses a fraction or a number out of range for an integer input, naming the input", () => {
    const pick = Fn(([n, m]: [Node<"int">, Node<"uint">]) => n.add(m.toInt())).setLayout({
      name: "pick",
      type: "int",
      inputs: [
        { name: "n", type: "int" },
        { name: "m", type: "uint" },
      ],
    }) as (...args: any[]) => any;
    expect(() => pick(2.5, 1)).toThrowRmsl(NodeError, "argument-of-another-type", /"n".*int.*2\.5/);
    expect(() => pick(3e9, 1)).toThrowRmsl(NodeError, "argument-of-another-type", /"n".*int/);
    expect(() => pick(1, -1)).toThrowRmsl(NodeError, "argument-of-another-type", /"m".*uint/);
  });

  /**
   * @canon spec-a-taken-function-name-gets-a-number
   */
  it("numbers a variable named like a function with a layout, so the call is not hidden", () => {
    const shade = Fn(([x]: [Node<"float">]) => x.mul(2)).setLayout({
      name: "shade",
      type: "float",
      inputs: [{ name: "x", type: "float" }],
    });
    const build = () =>
      Fn(() => {
        const mine = float(1).toVar("shade");
        return vec4(shade(mine), 0, 0, 1);
      })();
    for (const code of [compileGlsl.fragment(build()), compileWgsl.fragment(build())]) {
      expect(code).toMatch(/\bshade\(/);
      expect(code).toMatch(/\bshade1\b/);
      expect(code).not.toMatch(/float shade =|var shade:/);
    }
  });

  /**
   * @canon spec-a-taken-function-name-gets-a-number
   */
  it("declares a function and its inputs under names GLSL and WGSL can, whatever the layout calls them", () => {
    const f = Fn(([a, b]: [Node<"float">, Node<"float">]) => a.mul(2).add(b)).setLayout({
      name: "float",
      type: "float",
      inputs: [
        { name: "in", type: "float" },
        { name: "gl_b", type: "float" },
      ],
    });
    const build = () => Fn(() => vec4(f(uniform("float"), uniform("float")), 0, 0, 1))();
    for (const code of [compileGlsl.fragment(build()), compileWgsl.fragment(build())]) {
      expect(code).toMatch(/\bfloat_\(/);
      expect(code).not.toMatch(/float float\b|fn float\b|\bin\b[:,)]|float in\b|(?<![\w])gl_b/);
    }
    const byName = Fn(({ in: a, gl_b: b }: { in: Node<"float">; gl_b: Node<"float"> }) => a.sub(b)).setLayout({
      name: "diff",
      type: "float",
      inputs: [
        { name: "in", type: "float" },
        { name: "gl_b", type: "float" },
      ],
    });
    expect(() => (byName as any)({ in: float(3), gl_b: float(1) })).not.toThrow();
  });
});
