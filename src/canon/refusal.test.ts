import { NodeError, ProgramError } from "../errors";
import { afterAll, describe, expect, it } from "vitest";
import {
  attribute,
  bool,
  Break,
  Continue,
  builtinFragDepth,
  builtinPosition,
  Discard,
  float,
  For,
  Return,
  fragCoord,
  Fn,
  If,
  instancedArray,
  invocationIndex,
  int,
  mat2x3,
  mat2x4,
  mat2,
  mat3,
  outputStruct,
  uniform,
  uniformArray,
  Switch,
  uint,
  varying,
  vec2,
  While,
  vec3,
  vec4,
  type Node,
  type Var,
  serialize,
  deserialize,
  texture,
  textureLoad,
  ivec2,
} from "../rmsl";
import { compileJSCompute, compileJSFragment, compileJSGrid, compileJSRoutine, compileJSVertex } from "../js";
import {
  compileWasmCompute,
  compileWasmFragment,
  compileWasmGrid,
  compileWasmRoutine,
  compileWasmVertex,
} from "../wasm";
import {
  assertRecordedShadersValid,
  recordShaderSource,
  recordingGLSL as compileGlsl,
  recordingWGSL as compileWgsl,
} from "../testing/shader-validity";
// The real compiler: the recording stand-ins cover vertex and fragment only,
// so a compute program's source is recorded by hand to reach Dawn.
import { compileWgsl as realCompileWgsl } from "../backends/wgsl/wgsl";
import type { CompileCpuRoutine } from "../backends/cpu";

const computeWgsl = (root: Node<any>) => recordShaderSource("wgsl", "compute", realCompileWgsl.compute(root));

afterAll(async () => {
  await assertRecordedShadersValid();
}, 120_000);

const cpuCompilers = [
  (build: () => Node<any>) => compileJSRoutine(build, { name: "main", params: [] }),
  (build: () => Node<any>) => compileWasmRoutine(build, { name: "main", params: [] }),
];

/** The same two, compiling for the compute stage, where a storage program goes. */
const cpuComputeCompilers = [
  (build: () => Node<any>) => compileJSCompute(build, { name: "main", params: [] }),
  (build: () => Node<any>) => compileWasmCompute(build, { name: "main", params: [] }),
];

describe("a mistake is refused before the program runs", () => {
  /**
   * A product whose shapes meet and the inverse of a square matrix compile
   * beside each other, while a product whose shapes do not meet and the
   * inverse of a matrix that is not square are each refused.
   *
   * @canon spec-a-matrix-product-whose-shapes-do-not-meet-is-refused
   * @canon spec-only-a-square-matrix-is-inverted
   */
  it("compiles the matrix operations the shapes allow and refuses the others", () => {
    const allowed = Fn(() => uniform("mat2x3").mul(uniform("mat3x2")).inverse().element(int(0)).toVar());
    expect(compileGlsl(allowed())).toContain("inverse(");
    expect(compileWgsl(allowed())).toContain("_rmsl_inverse3");
    expect(() => (mat2x3(1, 0, 0, 1, 0, 0) as any).mul(mat2x4(1, 0, 0, 1, 0, 0, 0, 0))).toThrowRmsl(
      NodeError,
      "matrix-product-shape",
    );
    const notSquare = () => Fn(() => (uniform("mat2x3") as any).inverse().element(int(0)).toVar())();
    expect(() => compileGlsl(notSquare())).toThrowRmsl(NodeError, "inverse-of-non-square");
    expect(() => compileWgsl(notSquare())).toThrowRmsl(NodeError, "inverse-of-non-square");
  });

  /**
   * Bare numbers beside an `int`, a `uint` and a `float` in one program each
   * take their neighbour's type, and the numbers those types cannot hold are
   * refused.
   *
   * @canon spec-a-bare-number-beside-an-integer-is-an-integer
   * @canon spec-a-bare-number-beside-a-float-is-a-float
   * @canon spec-a-fraction-beside-an-integer-is-refused
   * @canon spec-a-negative-number-for-an-unsigned-type-is-refused
   */
  it("gives each bare number its neighbour's type in one program, and refuses what the type cannot hold", () => {
    const prog = Fn(() =>
      uniform("int").add(2).toFloat().add(uniform("uint").mul(3).toFloat()).add(uniform("float").mul(4)).toVar(),
    );
    const wgsl = compileWgsl(prog());
    expect(wgsl).toContain("2i");
    expect(wgsl).toContain("3u");
    expect(wgsl).toContain("4f");
    expect(compileGlsl(prog())).not.toMatch(/float\(2\)|float\(3\)/);
    expect(() => Fn(() => uniform("int").add(2.5).toVar())()).toThrowRmsl(NodeError, "literal-not-whole");
    expect(() => Fn(() => uniform("uint").add(-1).toVar())()).toThrowRmsl(NodeError, "literal-negative-for-unsigned");
    expect(() => uint(-1)).toThrowRmsl(NodeError, "literal-negative-for-unsigned");
  });

  /**
   * A vertex stage that writes its position and a varying, and a fragment
   * stage that writes its depth and reads its coordinate, compile together.
   * Moving any of those built-ins to the other stage is refused.
   *
   * @canon spec-the-fragment-depth-is-written-only-in-a-fragment-stage
   * @canon spec-the-position-is-read-only-in-a-vertex-stage
   * @canon spec-a-vertex-stage-writes-its-position
   */
  it("compiles each stage's built-ins in its own stage and refuses them in the other", () => {
    const tint = varying("vec2");
    const vertex = () =>
      Fn(() => {
        tint.assign(vec2(1, 0));
        builtinPosition().assign(vec4(0, 0, 0, 1));
      })();
    const fragment = () =>
      Fn(() => {
        builtinFragDepth().assign(float(0.5));
        return vec4(tint.add(fragCoord()), 0, 1);
      })();
    for (const compile of [compileGlsl, compileWgsl]) {
      expect(compile.vertex(vertex())).toContain(tint.name);
      expect(compile.fragment(fragment())).toContain(tint.name);
      expect(() => compile.vertex(fragment() as any)).toThrowRmsl(ProgramError, "assign-in-wrong-stage");
      expect(() => compile.fragment(Fn(() => vec4(builtinPosition().x, 0, 0, 1).toVar())())).toThrowRmsl(
        ProgramError,
        "position-read-in-fragment",
        /fragment/i,
      );
    }
  });

  /**
   * A `Break` outside a loop and a `cross` of vectors that are not `vec3` are
   * each refused on the CPU targets, beside a program that does both the way
   * a target can run them. Every target refuses a whole storage buffer read as
   * a value, beside the element of it the runnable program reads.
   *
   * @canon spec-break-or-continue-outside-a-loop-is-refused
   * @canon spec-cross-of-a-vector-that-is-not-a-vec3-is-refused
   * @canon spec-a-whole-storage-buffer-cannot-be-read
   */
  it("refuses each operation no target can run, beside the same program written the way they run", () => {
    const values = instancedArray(4, "float");
    const runnable = () =>
      Fn(() =>
        vec3(1, 0, 0)
          .cross(vec3(0, 1, 0))
          .x.add(values.element(int(0)))
          .toVar(),
      )();
    for (const compile of cpuCompilers) {
      expect(() => compile(runnable)).not.toThrow();
      expect(() =>
        compile(() =>
          Fn(() => {
            Break();
            return float(1);
          })(),
        ),
      ).toThrowRmsl(ProgramError, "jump-outside-a-loop");
      expect(() => compile(() => Fn(() => (vec2(1, 0) as any).cross(vec2(0, 1)).toVar())())).toThrowRmsl(
        NodeError,
        "cross-needs-vec3",
      );
      expect(() => compile(() => Fn(() => (values as any).add(1).toVar())())).toThrowRmsl(
        NodeError,
        "storage-read-as-a-whole",
      );
    }
    const wholeRead = () => Fn(() => vec4((values as any).add(1), 0, 0, 1).toVar())();
    expect(() => compileWgsl.fragment(wholeRead())).toThrowRmsl(NodeError, "storage-read-as-a-whole");
    expect(() => compileGlsl.fragment(wholeRead())).toThrowRmsl(NodeError, "storage-read-as-a-whole");
  });

  /**
   * A `For` whose update holds a block, such as an `If`, is refused as it is
   * built, so on every target, because the update slot of a GLSL, WGSL or
   * JavaScript `for` takes none. A `For` whose update is a plain statement compiles on each.
   *
   * @canon spec-a-for-update-that-holds-a-block-is-refused
   */
  it("refuses a For whose update holds a block on every target", () => {
    const loop = (update: (i: any) => any) => () =>
      Fn(() => {
        const sum = float(0).toVar();
        For(
          () => int(0).toVar(),
          (i) => i.lessThan(3),
          update,
          () => sum.addAssign(1),
        );
        return sum;
      })();
    const plain = loop((i) => i.addAssign(1));
    expect(() => compileGlsl(plain())).not.toThrow();
    expect(() => compileWgsl(plain())).not.toThrow();
    for (const compile of cpuCompilers) expect(() => compile(plain)).not.toThrow();
    // A block directly in the update, in a loop of its own, and behind a nested Fn.
    const blocks = [
      loop((i) => If(i.greaterThan(-1), () => i.addAssign(1))),
      loop((i) => While(i.lessThan(1), () => i.addAssign(1))),
      loop((i) =>
        Fn(() => {
          If(i.greaterThan(-1), () => i.addAssign(1));
        })(),
      ),
    ];
    for (const block of blocks) {
      // Refused as the program is built, before any target compiles it.
      expect(block).toThrowRmsl(NodeError, "block-in-for-update");
      expect(() => compileGlsl(block())).toThrowRmsl(NodeError, "block-in-for-update");
      expect(() => compileWgsl(block())).toThrowRmsl(NodeError, "block-in-for-update");
      for (const compile of cpuCompilers) expect(() => compile(block)).toThrowRmsl(NodeError, "block-in-for-update");
    }
  });

  /**
   * A `For` whose update holds a block in a graph `deserialize` rebuilt never
   * met the `For` builder, and every target refuses it as it compiles.
   *
   * @canon spec-a-for-update-that-holds-a-block-is-refused
   */
  it("refuses a deserialized For whose update holds a block on every target", () => {
    const build = () =>
      Fn(() => {
        const sum = float(0).toVar();
        // The If comes first: a deserialized node names only children before it.
        If(sum.greaterThan(1), () => sum.addAssign(1));
        For(
          () => int(0).toVar(),
          (i) => i.lessThan(3),
          (i) => i.addAssign(1),
          () => sum.addAssign(1),
        );
        return sum;
      })();
    const graph = serialize(build());
    const loop = graph.nodes.find((n) => n.type === "for")!;
    loop.params![2] = graph.nodes.findIndex((n) => n.type === "if");
    expect(() => compileGlsl(deserialize(graph) as Node<"float">)).toThrowRmsl(NodeError, "block-in-for-update");
    expect(() => compileWgsl(deserialize(graph) as Node<"float">)).toThrowRmsl(NodeError, "block-in-for-update");
    for (const compile of cpuCompilers)
      expect(() => compile(() => deserialize(graph) as Node<"float">)).toThrowRmsl(NodeError, "block-in-for-update");
  });

  /**
   * A graph `deserialize` rebuilt never met the builders that refuse an
   * operand of the wrong kind, so every target refuses it as it compiles. The
   * data is edited by hand so that a node reads an operand no builder lets it.
   */
  const rewired = (build: () => Node<"vec4">, type: string, from: string) => {
    const graph = serialize(build());
    const node = graph.nodes.find((n) => n.type === type)!;
    node.params![node.params!.length - 1] = graph.nodes.findIndex((n) => n.type === "uniform" && n._t === from);
    return graph;
  };
  const refusedOnEveryTarget = (graph: ReturnType<typeof serialize>, code: string, refusal: RegExp) => {
    const none = { name: "main", params: [] };
    expect(() => compileGlsl.fragment(deserialize(graph) as Node<"vec4">)).toThrowRmsl(NodeError, code, refusal);
    expect(() => compileWgsl.fragment(deserialize(graph) as Node<"vec4">)).toThrowRmsl(NodeError, code, refusal);
    expect(() => compileJSFragment(() => deserialize(graph) as Node<"vec4">, none)).toThrowRmsl(
      NodeError,
      code,
      refusal,
    );
    expect(() => compileWasmFragment(() => deserialize(graph) as Node<"vec4">, none)).toThrowRmsl(
      NodeError,
      code,
      refusal,
    );
  };

  /**
   * @canon spec-a-signed-and-an-unsigned-integer-operand-are-refused
   */
  it("refuses a deserialized sum of an int and a uint on every target", () => {
    const count = uniform("int");
    const mask = uniform("uint");
    const build = () => Fn(() => vec4(count.add(count).toFloat(), mask.toFloat(), 0, 1))();
    refusedOnEveryTarget(
      rewired(build, "add", "uint"),
      "operands-of-different-kinds",
      /add\(\) takes operands of one kind, not int and uint/,
    );
  });

  /**
   * @canon spec-an-integer-and-a-float-operand-are-refused
   */
  it("refuses a deserialized minimum of an int and a float on every target", () => {
    const count = uniform("int");
    const scale = uniform("float");
    const build = () => Fn(() => vec4(count.min(count).toFloat(), scale, 0, 1))();
    refusedOnEveryTarget(
      rewired(build, "min", "float"),
      "operands-of-different-kinds",
      /min\(\) takes operands of one kind, not int and float/,
    );
  });

  /**
   * @canon spec-operands-of-different-widths-are-refused
   */
  it("refuses a deserialized sum of vectors of different widths on every target", () => {
    const short = uniform("vec2");
    const long = uniform("vec3");
    const build = () => Fn(() => vec4(short.add(short).x, long.x, 0, 1))();
    refusedOnEveryTarget(rewired(build, "add", "vec3"), "vectors-of-different-widths", /one width/);
  });

  /**
   * @canon spec-a-math-function-of-floats-refuses-an-integer-operand
   */
  it("refuses a deserialized power of two ints on every target", () => {
    const count = uniform("int");
    const scale = uniform("float");
    const build = () => Fn(() => vec4(scale.pow(scale), count.toFloat(), 0, 1))();
    const graph = rewired(build, "pow", "int");
    graph.nodes.find((n) => n.type === "pow")!.params![0] = graph.nodes.findIndex((n) => n._t === "int");
    refusedOnEveryTarget(graph, "float-only-operand", /pow\(\) takes a float or a float vector, not int/);
  });

  /**
   * @canon spec-an-integer-operation-refuses-a-float-operand
   */
  it("refuses a deserialized bitwise and of floats on every target", () => {
    const count = uniform("int");
    const scale = uniform("float");
    const build = () => Fn(() => vec4(count.bitAnd(count).toFloat(), scale, 0, 1))();
    const graph = rewired(build, "bitAnd", "float");
    graph.nodes.find((n) => n.type === "bitAnd")!.params![0] = graph.nodes.findIndex((n) => n._t === "float");
    refusedOnEveryTarget(graph, "integer-only-operand", /bitAnd\(\) takes an integer or an integer vector, not float/);
  });

  /**
   * @canon spec-a-signed-operation-refuses-an-unsigned-operand
   */
  it("refuses a deserialized absolute value of a uint on every target", () => {
    const count = uniform("int");
    const mask = uniform("uint");
    const build = () => Fn(() => vec4(count.abs().toFloat(), mask.toFloat(), 0, 1))();
    refusedOnEveryTarget(
      rewired(build, "abs", "uint"),
      "signed-only-operand",
      /abs\(\) takes a float or a signed integer, scalar or vector, not uint/,
    );
  });

  /**
   * A `For` whose update leaves the loop or the function is refused as it is
   * built, and as every target compiles it, a graph `deserialize` rebuilt
   * included: the update slot of a GLSL, WGSL or JavaScript `for` takes no
   * `break`, `continue`, `discard` or `return`.
   *
   * @canon spec-a-for-update-that-jumps-is-refused
   */
  it.each([
    ["Break", Break],
    ["Continue", Continue],
    ["Discard", Discard],
    ["Return", Return],
  ] as const)("refuses a For whose update holds a %s on every target", (_, jump) => {
    const none = { name: "main", params: [] };
    const compilers: [string, (build: () => Node<any>) => unknown][] = [
      ["GLSL", (build) => compileGlsl.fragment(build())],
      ["WGSL", (build) => compileWgsl.fragment(build())],
      ["JS", (build) => compileJSFragment(build, none)],
      ["WASM", (build) => compileWasmFragment(build, none)],
    ];
    const loop = (update: (i: any) => void) => () =>
      Fn(() => {
        const sum = float(0).toVar();
        // A loop of its own before the For, whose jump a rebuilt update can name.
        While(sum.lessThan(1), () => {
          sum.addAssign(1);
          jump();
        });
        For(
          () => int(0).toVar(),
          (i) => i.lessThan(3),
          update,
          () => sum.addAssign(1),
        );
        return vec4(sum);
      })();
    expect(
      loop((i) => {
        i.addAssign(1);
        jump();
      }),
    ).toThrowRmsl(NodeError, "jump-in-for-update");
    const graph = serialize(loop((i) => i.addAssign(1))());
    const update = graph.nodes[graph.nodes.find((n) => n.type === "for")!.params![2]!]!;
    update.params!.push(graph.nodes.findIndex((n) => n.type === jump.name.toLowerCase()));
    for (const [name, compile] of compilers) {
      expect(() => compile(() => deserialize(graph) as Node<"vec4">), name).toThrowRmsl(
        NodeError,
        "jump-in-for-update",
      );
    }
  });

  /**
   * A `For` whose update writes a storage element, a scalar one and a
   * component of a vector one, past the end of the buffer too, compiles on
   * every target with storage buffers, and the CPU targets write the same.
   *
   * @canon spec-a-for-update-that-holds-a-block-is-refused
   */
  it("compiles a For whose update writes a storage element on every target with storage buffers", () => {
    const scalars = instancedArray(4, "float");
    const triples = instancedArray(4, "vec3");
    const build = () =>
      Fn(() => {
        For(
          () => int(0).toVar(),
          (i) => i.lessThan(6),
          (i) => {
            triples.element(i).x.assign(float(1));
            scalars.element(i).assign(i.toFloat());
            i.assign(i.add(1));
          },
          () => {},
        );
      })();
    expect(computeWgsl(build())).toContain("continuing");
    for (const compile of [compileJSCompute, compileWasmCompute]) {
      const storages = { [scalars.name]: Float64Array.of(9, 9, 9, 9), [triples.name]: new Float64Array(12) };
      compile(build, { name: "main", params: [] })({ storages }, 1);
      expect(Array.from(storages[scalars.name]!)).toEqual([0, 1, 2, 3]);
      expect(Array.from(storages[triples.name]!)).toEqual([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]);
    }
  });

  /**
   * A vector given as a whole matrix, other than a `vec4` to `mat2`, and a
   * column with a length other than the matrix's rows are refused as the
   * program builds them, so on every target alike.
   *
   * @canon spec-a-matrix-is-built-from-its-columns
   */
  it("refuses a matrix built from a vector alone or from columns of the wrong length", () => {
    const v = (type: "vec2" | "vec3" | "vec4") => uniform(type);
    const refused: [string, () => unknown][] = [
      ["mat2(vec2)", () => mat2(v("vec2"))],
      ["mat2(vec3)", () => mat2(v("vec3"))],
      ["mat3(vec4)", () => mat3(v("vec4"))],
      ["mat2x3(vec4)", () => mat2x3(v("vec4"))],
      ["mat2(vec3, vec3)", () => mat2(v("vec3"), v("vec3"))],
      ["mat3(vec4, vec4, vec4)", () => mat3(v("vec4"), v("vec4"), v("vec4"))],
      ["mat2x3(vec2, vec2)", () => mat2x3(v("vec2"), v("vec2"))],
      ["mat2(float, float)", () => mat2(uniform("float"), uniform("float"))],
    ];
    for (const [name, build] of refused) expect(build, name).toThrowRmsl(NodeError, "constructor-arity");
    expect(() => mat2(v("vec4"))).not.toThrow();
    expect(() => mat2x3(v("vec3"), v("vec3"))).not.toThrow();
  });

  /**
   * A `Case` given no values can match no selector, so it is refused, naming
   * `Case`.
   *
   * @canon spec-a-case-with-no-values-is-refused
   */
  it("refuses a Case with no values, naming it", () => {
    const program = () =>
      Fn(() => {
        const v = float(0).toVar();
        Switch(int(uniform("float"))).Case(() => {
          v.assign(float(1));
        });
        return vec4(v);
      })();
    expect(program).toThrowRmsl(NodeError, "case-without-values", /Case\(\) needs at least one value/);
  });

  /**
   * A `Case` added from a block other than its `Switch`'s, or after its
   * `Default`, could not reach the program, so it is refused, naming it. So is
   * a `Case` or `Default` added after a statement that follows the case before
   * it, which the chain would run before that statement.
   *
   * @canon spec-a-case-is-added-in-the-block-of-its-switch
   */
  it("refuses a Case added from another block or after the Default, naming it", () => {
    const fromAnotherBlock = () =>
      Fn(() => {
        const v = float(0).toVar();
        const s = Switch(int(uniform("float")));
        If(bool(true), () => {
          s.Case(0, () => v.assign(float(1)));
        });
        return vec4(v);
      })();
    const afterDefault = () =>
      Fn(() => {
        const v = float(0).toVar();
        Switch(int(uniform("float")))
          .Default(() => v.assign(float(2)))
          .Case(0, () => v.assign(float(1)));
        return vec4(v);
      })();
    let kept: ReturnType<typeof Switch> | undefined;
    const afterTheFunction = () => {
      Fn(() => {
        kept = Switch(int(uniform("float")));
        return vec4(0);
      })();
      kept!.Case(0, () => {});
    };
    expect(fromAnotherBlock).toThrowRmsl(
      NodeError,
      "case-outside-switch",
      /Case\(\) must be called from the block that holds its Switch\(\)/,
    );
    expect(afterDefault).toThrowRmsl(NodeError, "case-after-default", /Case\(\) after Default\(\)/);
    expect(afterTheFunction).toThrowRmsl(
      NodeError,
      "case-outside-switch",
      /Case\(\) must be called from the block that holds its Switch\(\)/,
    );
    const afterAStatement = (add: (s: ReturnType<typeof Switch>, w: Var<"float">, v: Var<"float">) => void) => () =>
      Fn(() => {
        const v = float(0).toVar();
        const s = Switch(int(uniform("float"))).Case(0, () => v.assign(float(1)));
        const w = float(5).toVar();
        add(s, w, v);
        return vec4(v);
      })();
    expect(afterAStatement((s, w, v) => s.Case(1, () => v.assign(w)))).toThrowRmsl(
      NodeError,
      "case-after-statement",
      /Case\(\) after a statement that follows the case before it/,
    );
    expect(afterAStatement((s, w, v) => s.Default(() => v.assign(w)))).toThrowRmsl(
      NodeError,
      "case-after-statement",
      /Default\(\) after a statement that follows the case before it/,
    );
  });

  /**
   * A `Switch` with no `Case` and no `Default` has nothing to run, and compiles
   * on every target.
   *
   * @canon spec-a-switch-runs-the-case-its-selector-matches
   */
  it("compiles a Switch with no Case and no Default on every target, and runs the statements after it", () => {
    const build = () =>
      Fn(() => {
        const v = float(0).toVar();
        Switch(int(uniform("float")));
        v.assign(float(2));
        return vec4(v);
      })();
    expect(() => compileGlsl.fragment(build())).not.toThrow();
    expect(() => compileWgsl.fragment(build())).not.toThrow();
    for (const compile of cpuCompilers) expect(() => compile(build)).not.toThrow();
    for (const compile of cpuCompilers) {
      const result: any = compile(build)({});
      expect(Array.from(Array.isArray(result) || ArrayBuffer.isView(result) ? result : result.value)[0]).toBe(2);
    }
  });

  /**
   * An array of 2, 3 or 4 numbers is a vector, and one of a length no vector
   * has is refused, naming the length and the operation it was given to, where
   * it used to become the float of its first element.
   *
   * @canon spec-a-javascript-array-is-a-vector-of-its-length
   */
  it("refuses a JavaScript array whose length no vector has", () => {
    const build = (length: number) => () =>
      Fn(() =>
        vec4(0)
          .add(Array.from({ length }, (_, i) => i + 1) as any)
          .toVar(),
      )();
    expect(() => build(4)()).not.toThrow();
    for (const length of [0, 1, 5, 7]) {
      expect(build(length), `length ${length}`).toThrowRmsl(
        NodeError,
        "array-is-no-vector",
        new RegExp(`array of length ${length} given to add\\(\\) is no vector`),
      );
    }
  });

  /**
   * A constant index outside the elements of a uniform array is refused by
   * every target, a literal or an operation of literals that folds to one, and
   * an index inside them compiles.
   *
   * @canon spec-a-constant-index-outside-a-uniform-array-is-refused
   */
  it("refuses a constant index outside a uniform array on every target", () => {
    const items = uniformArray("float", 3);
    const read = (index: () => any) => () => Fn(() => vec4(items.element(index()), 0, 0, 1).toVar())();
    const compilers: Array<[string, (build: () => any) => unknown]> = [
      ["GLSL", (build) => compileGlsl.fragment(build())],
      ["WGSL", (build) => compileWgsl.fragment(build())],
      ["JS", (build) => cpuCompilers[0]!(build)],
      ["WASM", (build) => cpuCompilers[1]!(build)],
    ];
    for (const [name, compile] of compilers) {
      expect(() => compile(read(() => int(2))), `${name} read 2`).not.toThrow();
      expect(() => compile(read(() => int(3))), `${name} read 3`).toThrowRmsl(NodeError, "index-out-of-range");
      expect(() => compile(read(() => int(-1))), `${name} read -1`).toThrowRmsl(NodeError, "index-out-of-range");
      expect(() => compile(read(() => int(1).add(int(4)))), `${name} read 1 + 4`).toThrowRmsl(
        NodeError,
        "index-out-of-range",
      );
    }
  });

  /**
   * A constant index outside the components of a vector or the columns of a
   * matrix is refused by every target when it compiles the element, for a read
   * and for a write, and for the component of a column too, whether it is a
   * literal or an operation of literals that folds to one. So is a write by
   * index through a swizzle at a constant index outside the swizzle, in a graph
   * `deserialize` rebuilt too. An index
   * inside them compiles on every target.
   *
   * @canon spec-a-constant-index-outside-a-vector-or-matrix-is-refused
   */
  it("refuses a constant index outside a vector or matrix on every target", () => {
    const read = (index: number) => () => Fn(() => vec4(vec3(1, 2, 3).toVar().element(int(index)), 0, 0, 1).toVar())();
    const write = (index: number) => () =>
      Fn(() => {
        const v = vec3(1, 2, 3).toVar();
        v.element(int(index)).assign(float(5));
        return v;
      })();
    const column = (index: number, matrix: () => any) => () => Fn(() => matrix().toVar().element(int(index)).toVar())();
    const threeColumns = () => mat3(1, 0, 0, 0, 1, 0, 0, 0, 1);
    const twoColumns = () => mat2x3(1, 0, 0, 0, 1, 0);
    const compilers: Array<[string, (build: () => any) => unknown]> = [
      ["GLSL", (build) => compileGlsl.fragment(build())],
      ["WGSL", (build) => compileWgsl.fragment(build())],
      ["JS", (build) => cpuCompilers[0]!(build)],
      ["WASM", (build) => cpuCompilers[1]!(build)],
    ];
    const writeColumn = (index: number) => () =>
      Fn(() => {
        const m = threeColumns().toVar();
        m.element(int(index)).assign(vec3(1, 2, 3));
        return vec4(0);
      })();
    const writeComponent = (column: number, component: number) => () =>
      Fn(() => {
        const m = threeColumns().toVar();
        m.element(int(column)).element(int(component)).assign(float(1));
        return vec4(0);
      })();
    for (const [name, compile] of compilers) {
      expect(() => compile(writeColumn(2)), `${name} write column 2`).not.toThrow();
      expect(() => compile(writeColumn(3)), `${name} write column 3`).toThrowRmsl(NodeError, "index-out-of-range");
      expect(() => compile(writeComponent(1, 2)), `${name} write component 2`).not.toThrow();
      expect(() => compile(writeComponent(3, 0)), `${name} write column 3 component 0`).toThrowRmsl(
        NodeError,
        "index-out-of-range",
      );
      expect(() => compile(writeComponent(0, 3)), `${name} write component 3`).toThrowRmsl(
        NodeError,
        "index-out-of-range",
      );
    }
    const folded = (index: () => any) => () =>
      Fn(() => vec4(vec3(1, 2, 3).toVar().element(index()), 0, 0, 1).toVar())();
    for (const [name, compile] of compilers) {
      expect(() => compile(folded(() => int(1).add(int(1)))), `${name} read 1 + 1`).not.toThrow();
      expect(() => compile(folded(() => int(1).add(int(3)))), `${name} read 1 + 3`).toThrowRmsl(
        NodeError,
        "index-out-of-range",
      );
      const foldedColumn = () =>
        Fn(() =>
          threeColumns()
            .toVar()
            .element(int(2).mul(int(2)))
            .toVar(),
        )();
      expect(() => compile(foldedColumn), `${name} mat3 column 2 * 2`).toThrowRmsl(NodeError, "index-out-of-range");
    }
    const throughSwizzle = (index: () => any) => () =>
      Fn(() => {
        const v = vec3(1, 2, 3).toVar();
        v.zy.element(index()).assign(float(5));
        return v;
      })();
    for (const [name, compile] of compilers) {
      expect(() => compile(throughSwizzle(() => int(1))), `${name} write .zy 1`).not.toThrow();
      for (const [index, k] of [
        [() => int(5), 5],
        [() => int(-1), -1],
        [() => int(1).add(int(1)), 2],
      ] as const) {
        expect(() => compile(throughSwizzle(index)), `${name} write .zy ${k}`).toThrowRmsl(
          NodeError,
          "index-out-of-range",
        );
      }
      const restored = () => deserialize(serialize(throughSwizzle(() => int(5))())) as Node<"vec3">;
      expect(() => compile(restored), `${name} write .zy 5 after JSON`).toThrowRmsl(NodeError, "index-out-of-range");
    }
    for (const [name, compile] of compilers) {
      expect(() => compile(read(2)), `${name} read 2`).not.toThrow();
      expect(() => compile(read(3)), `${name} read 3`).toThrowRmsl(NodeError, "index-out-of-range");
      expect(() => compile(read(-1)), `${name} read -1`).toThrowRmsl(NodeError, "index-out-of-range");
      expect(() => compile(write(3)), `${name} write 3`).toThrowRmsl(NodeError, "index-out-of-range");
      expect(() => compile(column(3, threeColumns)), `${name} mat3`).toThrowRmsl(NodeError, "index-out-of-range");
      // A mat2x3 has two columns of three rows: the count is the columns.
      expect(() => compile(column(1, twoColumns)), `${name} mat2x3 column 1`).not.toThrow();
      expect(() => compile(column(2, twoColumns)), `${name} mat2x3 column 2`).toThrowRmsl(
        NodeError,
        "index-out-of-range",
      );
    }
  });

  /**
   * A compute program that reads a buffer through `storage()` compiles on
   * every target. Spelling the same buffer as an `attribute()` is refused on
   * each, because a compute dispatch has no vertices to read one for. GLSL has
   * no compute stage at all, so the three targets here are all of them.
   *
   * @canon spec-a-compute-program-cannot-read-an-attribute
   */
  it("refuses an attribute read by a compute program, where the same buffer through storage() compiles", () => {
    const buf = instancedArray(4, "float");
    const viaStorage = () => Fn(() => buf.element(int(0)).add(1).toVar())();
    const viaAttribute = () => Fn(() => attribute("float").add(1).toVar())();

    expect(computeWgsl(viaStorage())).toContain("@compute");
    expect(() => computeWgsl(viaAttribute())).toThrowRmsl(ProgramError, "compute-cannot-use-stage-io");
    for (const compile of cpuComputeCompilers) {
      expect(() => compile(viaStorage)).not.toThrow();
      expect(() => compile(viaAttribute)).toThrowRmsl(ProgramError, "compute-cannot-use-stage-io");
    }
  });

  /**
   * A compute program that reads a uniform compiles on every target. Reading a
   * `varying()` is refused on each, because a compute dispatch has no vertex
   * stage to pass one from. GLSL has no compute stage, so the three targets
   * here are all of them.
   *
   * @canon spec-a-compute-program-cannot-read-a-varying
   */
  it("refuses a varying read by a compute program, where a uniform compiles", () => {
    const buf = instancedArray(4, "float");
    const viaUniform = () => Fn(() => buf.element(int(0)).assign(uniform("float")))();
    const viaVarying = () => Fn(() => buf.element(int(0)).assign(varying("float")))();

    expect(computeWgsl(viaUniform())).toContain("@compute");
    expect(() => computeWgsl(viaVarying())).toThrowRmsl(ProgramError, "compute-cannot-use-stage-io");
    for (const compile of cpuComputeCompilers) {
      expect(() => compile(viaUniform)).not.toThrow();
      expect(() => compile(viaVarying)).toThrowRmsl(ProgramError, "compute-cannot-use-stage-io");
    }
  });

  /**
   * An `outputStruct` is the value a fragment stage returns. A vertex stage, a
   * compute stage and a program with no stage refuse it, on both CPU targets.
   *
   * @canon spec-an-output-struct-is-refused-outside-a-fragment-stage
   */
  it.each([
    ["a vertex stage", compileJSVertex, compileWasmVertex],
    ["a compute stage", compileJSCompute, compileWasmCompute],
  ])("refuses an outputStruct in %s", (_, js, wasm) => {
    const build = () => Fn(() => outputStruct(vec4(1, 0, 0, 1)))();
    for (const compile of [js, wasm]) {
      expect(() => compile(build, { name: "main", params: [] })).toThrowRmsl(
        ProgramError,
        "output-struct-outside-fragment",
      );
    }
  });

  /**
   * @canon spec-an-output-struct-is-refused-outside-a-fragment-stage
   */
  it("refuses an outputStruct in a program compiled as a routine", () => {
    const build = () => Fn(() => outputStruct(vec4(1, 0, 0, 1)))();
    for (const compile of [compileJSRoutine, compileWasmRoutine] as CompileCpuRoutine[]) {
      expect(() => compile(build, { name: "main", params: [] })).toThrowRmsl(
        ProgramError,
        "output-struct-outside-fragment",
      );
    }
  });

  /**
   * @canon spec-a-routine-refuses-an-input-only-a-stage-has
   */
  it.each([
    ["fragCoord()", () => fragCoord().x, /fragCoord\(\) is an input of a fragment stage/],
    ["invocationIndex()", () => invocationIndex().toFloat(), /invocationIndex\(\) is an input of a compute stage/],
    ["builtinPosition()", () => builtinPosition().x, /builtinPosition\(\) is an input of a vertex stage/],
    ["builtinFragDepth()", () => builtinFragDepth(), /builtinFragDepth\(\) is an input of a fragment stage/],
    ["a varying", () => varying("float"), /a varying is an input of a vertex or fragment stage/],
    ["an attribute", () => attribute("float"), /an attribute is an input of a vertex stage/],
    ["Discard()", () => Discard(), /Discard\(\) is an input of a fragment stage, or a grid/],
  ])("refuses %s in a routine, on both CPU targets", (_, read, message) => {
    for (const compile of [compileJSRoutine, compileWasmRoutine] as CompileCpuRoutine[]) {
      expect(() => compile(() => Fn(() => read())(), { name: "main", params: [] })).toThrowRmsl(
        ProgramError,
        "stage-input-without-stage",
        message,
      );
    }
  });

  /**
   * @canon spec-a-routine-refuses-an-input-only-a-stage-has
   */
  it.each([
    ["invocationIndex()", () => invocationIndex().toFloat(), /invocationIndex\(\) is an input of a compute stage/],
    ["a varying", () => varying("float"), /a varying is an input of a vertex or fragment stage/],
    ["an attribute", () => attribute("float"), /an attribute is an input of a vertex stage/],
  ])("refuses %s in a grid, which has fragCoord() and no more, on both CPU targets", (_, read, message) => {
    for (const compile of [compileJSGrid, compileWasmGrid]) {
      expect(() => compile(() => Fn(() => read())(), { name: "main", params: [] })).toThrowRmsl(
        ProgramError,
        "stage-input-without-stage",
        message,
      );
      expect(() => compile(() => Fn(() => fragCoord().x)(), { name: "main", params: [] })).not.toThrow();
    }
  });

  /**
   * @canon spec-a-cpu-grid-evaluates-a-fragment-for-each-pixel
   */
  it("refuses to fill a grid with a program that returns nothing, on both CPU targets", () => {
    for (const compile of [compileJSGrid, compileWasmGrid]) {
      const grid = compile(
        () =>
          Fn(() => {
            outputStruct(float(1));
          })() as any,
        { name: "main", params: [] },
      );
      expect(() => grid({}, 1, 1)).toThrowRmsl(ProgramError, "grid-without-result");
    }
  });
});

describe("a cube map has no texel fetch", () => {
  const none = { name: "main", params: [] };
  /** Expects every target to refuse the graph `build` gives with a `ProgramError` of `code`. */
  const refusedOnEveryTarget = (build: () => Node<"vec4">, code: string) => {
    expect(() => compileGlsl.fragment(build())).toThrowRmsl(NodeError, code);
    expect(() => compileWgsl.fragment(build())).toThrowRmsl(NodeError, code);
    expect(() => compileJSFragment(build, none)).toThrowRmsl(NodeError, code);
    expect(() => compileWasmFragment(build, none)).toThrowRmsl(NodeError, code);
  };

  /**
   * `texture` refuses an integer cube map as the graph is built, so the graph
   * is built with a float one and its sampler made an integer cube map in the
   * saved data.
   *
   * @canon exception-an-integer-cube-map-cannot-be-fetched
   */
  it("refuses an integer cube map in a graph deserialize rebuilt, on every target", () => {
    const cube = uniform("samplerCube");
    const graph = serialize(Fn(() => vec4(texture(cube, vec3(1, 0, 0)).x, 0, 0, 1))()) as any;
    for (const n of graph.nodes) {
      if (n._t === "samplerCube") n._t = "isamplerCube";
      if (n.value?.shaderType === "samplerCube") n.value.shaderType = "isamplerCube";
    }
    refusedOnEveryTarget(() => deserialize(JSON.parse(JSON.stringify(graph))) as Node<"vec4">, "integer-cube-sampler");
  });

  /**
   * @canon exception-an-integer-cube-map-cannot-be-fetched
   */
  it("refuses textureLoad on a float cube map, on every target", () => {
    const cube = uniform("samplerCube");
    refusedOnEveryTarget(
      () => Fn(() => vec4(textureLoad(cube as any, ivec2(0, 0) as any).x, 0, 0, 1))(),
      "cube-map-texel-fetch",
    );
  });
});
