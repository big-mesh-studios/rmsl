import { afterAll, afterEach, describe, expect, it } from "vitest";
import {
  attribute,
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
import { compileWgsl as compileWgslStage } from "../wgsl";
import { deserialize, serialize } from "../serialize";
import { compileJSFn, compileJSFragment, compileJSRoutine } from "../js";
import { compileWasmFn, compileWasmFragment, compileWasmRoutine } from "../wasm";
import {
  assertRecordedShadersValid,
  recordingGLSL as compileGlsl,
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
    expect(() => Fn(() => f())()).toThrow(/"f" takes 1 input, and the call passes 0/);
    expect(() => Fn(() => f(float(1), float(2)))()).toThrow(/"f" takes 1 input, and the call passes 2/);
  });

  /**
   * @canon spec-a-call-with-an-argument-of-the-wrong-type-is-refused
   */
  it("refuses an argument of another type, and takes a number for a float", () => {
    const f = Fn(([x]: [Node<"float">]) => x).setLayout(layout()) as (...args: any[]) => any;
    expect(() => Fn(() => f(vec3(1, 2, 3)))()).toThrow(
      /input "x" of the Fn "f" is a float, and the call passes a vec3/,
    );
    expect(() => Fn(() => f(int(1)))()).toThrow(/passes a int/);
    expect(() => Fn(() => f(2))()).not.toThrow();
  });

  /**
   * @canon spec-a-body-that-returns-another-type-than-its-layout-is-refused
   */
  it("refuses a body that returns another type, or nothing, or a value for void", () => {
    const wrong = Fn(([x]: [Node<"float">]) => vec3(x)).setLayout(layout()) as (...args: any[]) => any;
    expect(() => Fn(() => wrong(1))()).toThrow(/has the type float, and its body returns a vec3/);
    const nothing = Fn(([x]: [Node<"float">]) => {
      x.toVar();
    }).setLayout(layout()) as (...args: any[]) => any;
    expect(() => Fn(() => nothing(1))()).toThrow(/has the type float, and its body returns nothing/);
    const value = Fn(([x]: [Node<"float">]) => x).setLayout(layout("void")) as (...args: any[]) => any;
    expect(() => Fn(() => value(1))()).toThrow(/has the type void, and its body returns a float/);
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
    ).toThrow(/Fn "read" uses the variable "total", which is declared outside it/);
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
    ).toThrow(/Fn "accumulate" uses the variable "total", which is declared outside it/);
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
    expect(() => compileGlsl.fragment(restored())).toThrow(refusal);
    expect(() => compileWgsl.fragment(restored())).toThrow(refusal);
    expect(() => compileJSFragment(restored, none)).toThrow(refusal);
    expect(() => compileWasmFragment(restored, none)).toThrow(refusal);
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
    expect(() => compileGlsl.fragment(build())).toThrow(refusal);
    expect(() => compileWgsl.fragment(build())).toThrow(refusal);
    expect(() => compileJSFragment(build, none)).toThrow(refusal);
    expect(() => compileWasmFragment(build, none)).toThrow(refusal);
  });

  /**
   * @canon spec-a-layout-fn-uses-no-input-or-output-of-its-stage
   */
  it.each([
    ["an attribute", () => attribute("vec3"), /uses an attribute/],
    ["a varying", () => varying("float"), /uses a varying/],
    ["fragCoord()", () => fragCoord(), /uses fragCoord\(\)/],
  ] as const)("refuses a body that uses %s, on every target", (_, stageBound, refusal) => {
    const use: any = stageBound();
    const f = Fn(() => float(use.x ?? use)).setLayout({ name: "f", type: "float", inputs: [] });
    const build = () => Fn(() => vec4(f(), 0, 0, 1))();
    expect(() => compileGlsl.fragment(build())).toThrow(refusal);
    expect(() => compileWgsl.fragment(build())).toThrow(refusal);
    expect(() => compileJSFragment(build, none)).toThrow(refusal);
    expect(() => compileWasmFragment(build, none)).toThrow(refusal);
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
    [
      "a name that is a GLSL keyword",
      { name: "double", type: "float", inputs: [] },
      /"double", which is a keyword or a built-in/,
    ],
    [
      "a name that is a WGSL built-in",
      { name: "saturate", type: "float", inputs: [] },
      /"saturate", which is a keyword or a built-in/,
    ],
  ] as const)("refuses a layout with %s", (_, malformed, refusal) => {
    expect(() => Fn(() => float(1)).setLayout(malformed as any)).toThrow(refusal);
    expect(() => Fn(() => float(1), malformed as any)).toThrow(refusal);
  });
});
