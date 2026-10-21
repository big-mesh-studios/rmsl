import { afterAll, afterEach, describe, expect, it } from "vitest";
import {
  Fn,
  If,
  attribute,
  bool,
  select,
  attributeArray,
  float,
  instancedArray,
  int,
  struct,
  uniform,
  uniformArray,
  varying,
  vec3,
  vec4,
  type Node,
} from "../rmsl";
import { compileJSFn } from "../js";
import { compileWasmFn } from "../wasm";
import { deserialize, serialize } from "../serialize";
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

/** A node of a BVH: its bounds and the index of its left child. */
const BVHNode = struct({ min: "vec3", max: "vec3", left: "int" }, "BVHNode");

/** An expectation on the components of a result, as a plain array. */
const expectValues = (result: unknown) => expect(Array.from(result as ArrayLike<number>));

/** A fragment stage whose colour is `value`, built inside an `Fn`. */
const fragment = (value: () => Node<"vec4">) => Fn(value)();

describe("a struct groups named members as TSL's struct does", () => {
  /**
   * @canon spec-a-struct-is-built-from-its-members-in-order
   */
  it("gives the members the values in the order of the layout", () => {
    expectValues(
      evaluateRecording(
        (x) => {
          const node = BVHNode(vec3(x, 2, 3), vec3(4, 5, 6), int(7));
          return vec4(node.get("min").x, node.get("max").y, node.get("left").toFloat(), 0);
        },
        [1],
      ),
    ).toEqual([1, 5, 7, 0]);
  });

  /**
   * @canon spec-a-struct-is-built-from-its-members-by-name
   */
  it("gives each member the value of its name", () => {
    expectValues(
      evaluateRecording(
        (x) => {
          const node = BVHNode({ left: int(7), max: vec3(4, 5, 6), min: vec3(x, 2, 3) });
          return vec4(node.get("min").x, node.get("max").y, node.get("left").toFloat(), 0);
        },
        [1],
      ),
    ).toEqual([1, 5, 7, 0]);
  });

  /**
   * @canon spec-a-member-a-struct-is-not-given-is-zero
   */
  it("gives zero to each member it is not given", () => {
    expectValues(
      evaluateRecording(
        (x) => {
          const given = BVHNode(vec3(x));
          const empty = BVHNode();
          return vec4(given.get("min").x, given.get("max").y, given.get("left").toFloat(), empty.get("min").z);
        },
        [1],
      ),
    ).toEqual([1, 0, 0, 0]);
  });

  /**
   * @canon spec-a-number-for-a-member-is-a-value-of-its-type
   */
  it("builds a number or an array as a value of the member's type", () => {
    expectValues(
      evaluateRecording(() => {
        const node = BVHNode(2, [4, 5, 6], 7);
        return vec4(node.get("min").y, node.get("max").z, node.get("left").toFloat(), 0);
      }),
    ).toEqual([2, 6, 7, 0]);
  });

  /**
   * @canon exception-a-member-is-built-from-an-array-and-by-name
   */
  it("builds an array, and a number given by name, as the member's type", () => {
    expectValues(
      evaluateRecording(() => {
        const node = BVHNode({ min: [1, 2, 3], max: 4, left: 5 });
        return vec4(node.get("min").z, node.get("max").x, node.get("left").toFloat(), 0);
      }),
    ).toEqual([3, 4, 5, 0]);
  });

  /**
   * @canon spec-a-member-may-take-the-name-of-a-built-in
   */
  it("takes a member named like a built-in, and refuses one named like a keyword", () => {
    const Named = struct({ length: "float", texture: "float" }, "NamedLikeBuiltIns");
    expectValues(
      evaluateRecording(() => {
        const value = Named(float(1), float(2));
        return vec4(value.get("length"), value.get("texture"), 0, 0);
      }),
    ).toEqual([1, 2, 0, 0]);
    expect(() => struct({ for: "float" }, "NamedLikeAKeyword")).toThrow(/"for"/);
    expect(() => struct({ type: "float" }, "NamedLikeAWgslWord")).toThrow(/"type"/);
    expect(() => struct({ gl_x: "float" }, "NamedWithGl")).toThrow(/"gl_x"/);
  });

  /**
   * @canon spec-a-struct-name-is-one-a-target-can-declare
   */
  it("refuses a struct name no target can declare", () => {
    for (const name of ["float", "max", "type", "gl_S", "_rmsl_S", "S__T", "VertexOutput", "FragmentOutput"]) {
      expect(() => struct({ a: "float" }, name), name).toThrow(new RegExp(`"${name}"`));
    }
  });

  /**
   * @canon spec-a-name-a-struct-has-is-numbered-for-a-variable-or-a-function
   */
  it("numbers a variable and a function named like a struct", () => {
    const Light = struct({ a: "float" }, "Light");
    const shine = Fn(([x]: [Node<"float">]) => x.mul(2)).setLayout({
      name: "Light",
      type: "float",
      inputs: [{ name: "x", type: "float" }],
    });
    const build = () =>
      Fn(() => {
        const light = Light(float(1)).toVar("Light");
        const other = Light(float(2)).toVar();
        return vec4(shine(light.get("a")), other.get("a"), 0, 1);
      })();
    const glsl = compileGlsl.fragment(build());
    expect(glsl).toMatch(/Light Light1 = /);
    expect(glsl).toMatch(/float Light_1\(float x\)/);
    expect(compileWgsl.fragment(build())).toMatch(/var Light1: Light = /);
    expectValues(evaluateRecording(() => build())).toEqual([2, 2, 0, 1]);
  });

  /**
   * @canon spec-get-reads-a-member
   */
  it("reads a member by its name", () => {
    expectValues(
      evaluateRecording(
        (x) =>
          Fn(() => {
            const node = BVHNode(vec3(1), vec3(x), int(0)).toVar();
            return node.get("max").add(node.get("min"));
          })(),
        [3],
      ),
    ).toEqual([4, 4, 4]);
  });

  /**
   * @canon spec-a-member-of-a-struct-variable-is-assigned
   */
  it("writes one member, and one component of it, and leaves the rest", () => {
    expectValues(
      evaluateRecording(
        (x) =>
          Fn(() => {
            const node = BVHNode(vec3(1), vec3(2), int(3)).toVar();
            node.get("min").assign(vec3(x));
            node.get("max").y.assign(float(9));
            return vec4(node.get("max"), node.get("min").x);
          })(),
        [5],
      ),
    ).toEqual([2, 9, 2, 5]);
  });

  /**
   * @canon spec-a-struct-variable-is-assigned-whole
   */
  it("writes every member, from a value read in full before the first write", () => {
    const Pair = struct({ a: "float", b: "float" }, "Pair");
    expectValues(
      evaluateRecording(
        (x) =>
          Fn(() => {
            const pair = Pair(x, float(2)).toVar();
            pair.assign(Pair(pair.get("b"), pair.get("a")));
            const other = Pair().toVar();
            If(x.greaterThan(0), () => {
              other.assign(pair);
            });
            return vec4(pair.get("a"), pair.get("b"), other.get("a"), other.get("b"));
          })(),
        [1],
      ),
    ).toEqual([2, 1, 2, 1]);
  });

  /**
   * @canon spec-a-member-may-be-a-struct
   */
  it("reads and writes a member of a member that is a struct", () => {
    const Inner = struct({ a: "vec3" }, "Inner");
    const Outer = struct({ inner: "Inner", b: "float" }, "Outer");
    expectValues(
      evaluateRecording(
        (x) =>
          Fn(() => {
            const outer = Outer(Inner(vec3(x)), float(2)).toVar();
            outer.get("inner").get("a").z.assign(7);
            return vec4(outer.get("inner").get("a"), outer.get("b"));
          })(),
        [1],
      ),
    ).toEqual([1, 1, 7, 2]);
  });

  /**
   * @canon spec-a-layout-fn-returns-a-struct
   */
  it("returns a struct from an Fn with a layout", () => {
    const getNode = Fn(([i]: [Node<"float">]) => BVHNode(vec3(i), vec3(i.add(1)), i.toInt())).setLayout({
      name: "getNode",
      type: "BVHNode",
      inputs: [{ name: "i", type: "float" }],
    });
    const build = (x: Node<"float">) =>
      Fn(() => {
        const node = getNode(x);
        return vec4(node.get("min").x, node.get("max").x, node.get("left").toFloat(), 0);
      })();
    expect(compileGlsl.fragment(build(uniform("float")))).toMatch(/BVHNode getNode\(float i\)/);
    expect(compileWgsl.fragment(build(uniform("float")))).toMatch(/fn getNode\(i: f32\) -> BVHNode/);
    expectValues(evaluateRecording(build, [2])).toEqual([2, 3, 2, 0]);
  });

  /**
   * @canon spec-a-layout-fn-returns-a-struct
   */
  it("reads a struct an Fn with a layout returns outside every Fn, a member of a member included", () => {
    const Inner = struct({ a: "vec3" }, "Inner");
    const Outer = struct({ inner: "Inner", b: "float" }, "Outer");
    const makeOuter = Fn(([x]: [Node<"float">]) => Outer(Inner(vec3(x)), x.add(1))).setLayout({
      name: "makeOuter",
      type: "Outer",
      inputs: [{ name: "x", type: "float" }],
    });
    expectValues(
      evaluateRecording((x) => vec4(makeOuter(x).get("inner").get("a"), makeOuter(x).get("b")), [2]),
    ).toEqual([2, 2, 2, 3]);
  });

  /**
   * @canon spec-a-layout-fn-takes-a-struct-input
   */
  it("takes a struct as an input of an Fn with a layout", () => {
    const size = Fn(([node]: [any]) => node.get("max").sub(node.get("min"))).setLayout({
      name: "size",
      type: "vec3",
      inputs: [{ name: "node", type: "BVHNode" }],
    });
    const build = (x: Node<"float">) => Fn(() => vec4(size(BVHNode(vec3(1), vec3(x), int(0))), 1))();
    expect(compileGlsl.fragment(build(uniform("float")))).toMatch(/vec3 size\(BVHNode node\)/);
    expect(compileWgsl.fragment(build(uniform("float")))).toMatch(/fn size\(node: BVHNode\) -> vec3<f32>/);
    expectValues(evaluateRecording(build, [4])).toEqual([3, 3, 3, 1]);
  });

  /**
   * @canon spec-a-struct-without-a-name-gets-one
   */
  it("names a struct given no name StructType and a number", () => {
    const Unnamed = struct({ a: "float" });
    const code = compileGlsl.fragment(fragment(() => vec4(Unnamed(float(1)).toVar().get("a"), 0, 0, 1)));
    expect(code).toMatch(/struct StructType\d+ \{/);
  });

  /**
   * @canon spec-a-gpu-target-declares-each-struct-once
   */
  it("declares each struct once on GLSL, the member struct first", () => {
    const Inner = struct({ a: "vec3" }, "Inner");
    const Outer = struct({ inner: "Inner", b: "float" }, "Outer");
    const code = compileGlsl.fragment(
      fragment(() => {
        const a = Outer(Inner(vec3(1)), float(2)).toVar();
        const b = Outer(Inner(vec3(3)), float(4)).toVar();
        return vec4(a.get("inner").get("a").add(b.get("inner").get("a")), a.get("b"));
      }),
    );
    expect(code.match(/struct Inner \{/g)).toHaveLength(1);
    expect(code.match(/struct Outer \{/g)).toHaveLength(1);
    expect(code.indexOf("struct Inner {")).toBeLessThan(code.indexOf("struct Outer {"));
  });

  /**
   * @canon spec-a-gpu-target-declares-each-struct-once
   */
  it("declares each struct once on WGSL, the member struct first", () => {
    const Inner = struct({ a: "vec3" }, "Inner");
    const Outer = struct({ inner: "Inner", b: "float" }, "Outer");
    const code = compileWgsl.fragment(
      fragment(() => {
        const a = Outer(Inner(vec3(1)), float(2)).toVar();
        const b = Outer(Inner(vec3(3)), float(4)).toVar();
        return vec4(a.get("inner").get("a").add(b.get("inner").get("a")), a.get("b"));
      }),
    );
    expect(code.match(/struct Inner \{/g)).toHaveLength(1);
    expect(code.match(/struct Outer \{/g)).toHaveLength(1);
    expect(code.indexOf("struct Inner {")).toBeLessThan(code.indexOf("struct Outer {"));
  });

  /**
   * @canon spec-a-cpu-target-holds-each-member-in-a-variable-of-its-own
   */
  it("declares no struct in the JavaScript source, and a variable for each member", () => {
    const source = compileJSFn(
      () =>
        Fn(() => {
          const node = BVHNode(vec3(1), vec3(2), int(3)).toVar("node");
          return vec4(node.get("min"), node.get("left").toFloat());
        })(),
      none,
    );
    expect(source).not.toMatch(/BVHNode/);
    expect(source).toMatch(/node_min/);
    expect(source).toMatch(/node_left/);
  });

  /**
   * @canon spec-a-cpu-target-holds-each-member-in-a-variable-of-its-own
   */
  it("compiles a struct variable to WebAssembly", () => {
    const compiled = compileWasmFn(
      () =>
        Fn(() => {
          const node = BVHNode(vec3(1), vec3(2), int(3)).toVar("node");
          return vec4(node.get("min"), node.get("left").toFloat());
        })(),
      none,
    );
    expect(WebAssembly.validate(compiled.bytes as BufferSource)).toBe(true);
  });

  /**
   * @canon spec-a-struct-keeps-its-type-through-json
   */
  it("keeps a struct through serialize and deserialize", () => {
    const Inner = struct({ a: "vec3" }, "Inner");
    const Outer = struct({ inner: "Inner", b: "float" }, "Outer");
    const build = () =>
      Fn(() => {
        const outer = Outer(Inner(vec3(1, 2, 3)), float(4)).toVar();
        outer.get("b").assign(float(5));
        return vec4(outer.get("inner").get("a"), outer.get("b"));
      })();
    const restored = deserialize(JSON.parse(JSON.stringify(serialize(build())))) as Node<"vec4">;
    const normalize = (code: string) => code.replace(/_rmsl_[uav]?\d+/g, "_rmsl_x");
    expect(normalize(compileGlsl.fragment(restored))).toBe(normalize(compileGlsl.fragment(build())));
    expect(normalize(compileWgsl.fragment(restored))).toBe(normalize(compileWgsl.fragment(build())));
    expect(normalize(compileJSFn(() => restored, none))).toBe(normalize(compileJSFn(build, none)));
  });

  /**
   * @canon spec-a-restored-struct-without-a-name-takes-a-name-of-its-own
   */
  it("gives a restored struct without a name a name of its own", () => {
    const Unnamed = struct({ q: "float" });
    const build = () => Fn(() => vec4(Unnamed(float(3)).toVar().get("q"), 0, 0, 1))();
    const taken = struct({ z: "vec2" }).structName;
    const saved = JSON.stringify(serialize(build())).replaceAll(Unnamed.structName, taken);
    const restored = deserialize(JSON.parse(saved)) as Node<"vec4">;
    const code = compileGlsl.fragment(restored);
    expect(code).not.toMatch(new RegExp(`struct ${taken} `));
    expect(code).toMatch(/struct StructType\d+ \{\s*float q;/);
  });

  /**
   * @canon spec-a-restored-struct-of-a-taken-name-with-other-members-is-refused
   */
  it("refuses a restored struct whose name a struct of other members has", () => {
    const Saved = struct({ a: "float" }, "Saved");
    const graph = serialize(Fn(() => vec4(Saved(float(1)).toVar().get("a"), 0, 0, 1))());
    struct({ b: "vec3" }, "Saved");
    expect(() => deserialize(graph)).toThrow(/Saved.*b: vec3.*a: float/);
  });

  /**
   * @canon spec-an-assignment-of-another-type-to-a-struct-is-refused
   */
  it("refuses an assignment of another type to a struct, and of a struct to another type", () => {
    expect(() =>
      Fn(() => {
        BVHNode()
          .toVar()
          .assign(vec3(1) as any);
      })(),
    ).toThrow(/BVHNode/);
    expect(() =>
      Fn(() => {
        vec3(1)
          .toVar()
          .assign(BVHNode() as any);
      })(),
    ).toThrow(/BVHNode/);
  });

  /**
   * @canon spec-a-struct-is-no-result-of-a-program
   */
  it("refuses a program whose result is a struct on every target", () => {
    const build = () => Fn(() => BVHNode(vec3(1)).toVar())() as any;
    expect(() => compileGlsl.fragment(build())).toThrow(/BVHNode.*member/);
    expect(() => compileWgsl.fragment(build())).toThrow(/BVHNode.*member/);
    expect(() => compileJSFn(build, none)).toThrow(/BVHNode.*member/);
    expect(() => compileWasmFn(build, none)).toThrow(/BVHNode.*member/);
  });

  /**
   * @canon spec-a-struct-is-no-input-of-a-program
   */
  it("refuses a struct type for an input where it is made, and in a rebuilt graph on every target", () => {
    expect(() => uniform("BVHNode" as any)).toThrow(/uniform.*BVHNode/);
    expect(() => uniformArray("BVHNode" as any, 2)).toThrow(/BVHNode/);
    expect(() => attribute("BVHNode" as any)).toThrow(/BVHNode/);
    expect(() => varying("BVHNode" as any)).toThrow(/BVHNode/);
    const graph = serialize(Fn(() => vec4(uniform("float"), 0, 0, 1))());
    const read = graph.nodes.find((n) => n.type === "uniform")!;
    read._t = "BVHNode";
    (read.value as { shaderType: string }).shaderType = "BVHNode";
    const restored = () =>
      deserialize({
        ...graph,
        structs: [
          {
            name: "BVHNode",
            members: [
              { name: "min", type: "vec3" },
              { name: "max", type: "vec3" },
              { name: "left", type: "int" },
            ],
          },
        ],
      }) as Node<"vec4">;
    expect(() => compileGlsl.fragment(restored())).toThrow(/BVHNode/);
    expect(() => compileWgsl.fragment(restored())).toThrow(/BVHNode/);
    expect(() => compileJSFn(restored, none)).toThrow(/BVHNode/);
    expect(() => compileWasmFn(restored, none)).toThrow(/BVHNode/);
  });

  /**
   * @canon exception-a-storage-buffer-of-structs-is-refused
   */
  it("refuses a storage buffer of a struct element type", () => {
    expect(() => attributeArray(4, "BVHNode" as any)).toThrow(/attributeArray.*BVHNode/);
    expect(() => instancedArray(4, "BVHNode" as any)).toThrow(/instancedArray.*BVHNode/);
  });

  /**
   * @canon spec-an-argument-of-another-type-for-a-struct-input-is-refused
   */
  it("refuses an argument of another type for a struct input", () => {
    const size = Fn(([node]: [any]) => node.get("max").sub(node.get("min"))).setLayout({
      name: "size",
      type: "vec3",
      inputs: [{ name: "node", type: "BVHNode" }],
    });
    expect(() => Fn(() => size(vec3(1) as any))()).toThrow(/BVHNode/);
  });

  /**
   * @canon exception-get-refuses-a-member-the-struct-lacks
   */
  it("refuses a member the struct lacks, naming the members it has", () => {
    expect(() => (BVHNode() as any).get("mid")).toThrow(/"mid".*min, max, left/);
  });

  /**
   * @canon exception-a-member-value-of-another-type-is-refused
   */
  it("refuses a node of another type than its member's", () => {
    expect(() => BVHNode(float(1) as any)).toThrow(/min.*vec3.*float/);
    expect(() => BVHNode(vec3(1), vec3(2), float(3) as any)).toThrow(/left.*int.*toInt/);
  });

  /**
   * @canon exception-a-program-with-two-structs-of-one-name-is-refused
   */
  it("makes a struct again under its name, and refuses a program that holds a value of each", () => {
    const First = struct({ a: "float" }, "Remade");
    const first = First(float(1));
    const Second = struct({ a: "vec3" }, "Remade");
    const both = () =>
      Fn(() =>
        vec4(
          Second(vec3(first.get("a")))
            .toVar()
            .get("a"),
          1,
        ),
      )();
    expect(() => compileGlsl.fragment(both())).toThrow(/Remade.*a: float.*a: vec3/);
    expect(() => compileWgsl.fragment(both())).toThrow(/Remade/);
    expect(() => compileJSFn(both, none)).toThrow(/Remade/);
    expect(() => compileWasmFn(both, none)).toThrow(/Remade/);
    expect(compileGlsl.fragment(Fn(() => vec4(Second(vec3(1)).toVar().get("a"), 1))())).toMatch(
      /struct Remade \{\s*vec3 a;/,
    );
  });

  /**
   * @canon exception-a-struct-is-refused-as-an-operand
   */
  it("refuses a struct given to an operation", () => {
    const Pair = struct({ a: "float", b: "float" }, "Pair");
    expect(() => (Pair() as any).equal(Pair())).toThrow(/Pair/);
    expect(() => float(1).add(Pair() as any)).toThrow(/Pair/);
    expect(() => select(bool(true), Pair() as any, Pair() as any)).toThrow(/Pair/);
  });
});
