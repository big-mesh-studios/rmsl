import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  attribute,
  Break,
  builtinPosition,
  DataError,
  deserialize,
  dFdx,
  DriverError,
  float,
  Fn,
  fragCoord,
  instancedArray,
  int,
  InternalError,
  LimitError,
  mat2x3,
  type Node,
  NodeError,
  outputStruct,
  PlatformError,
  ProgramError,
  Return,
  RmslError,
  serialize,
  TargetError,
  uint,
  uniform,
  UsageError,
  varying,
  vec2,
  vec3,
  vec4,
} from "../rmsl";
import { compileGlsl, createGlsl } from "../glsl";
import { compileWgsl, createWgslContext } from "../wgsl";
import { compileJSFn, compileJSFragment, compileJSRoutine, compileJSVertex, createJs, createJsGrid } from "../js";
import { compileWasm, compileWasmFragment, compileWasmRoutine, compileWasmVertex } from "../wasm";
import { precompileShaders } from "../vite/vite";
import { node } from "../core";
import { Color } from "../scene/math/Color";
import { Euler } from "../scene/math/Euler";
import { Vector3 } from "../scene/math/Vector3";
import { Object3D } from "../scene/core/Object3D";
import { caught, caughtSync, expectRefusedOnEveryTarget, expectRmslError, expectThrows } from "../testing/errors";
import { GPU_ENABLED, installWebGpuGlobals, releaseGpu } from "../testing/gpu";
import { runInGpuPage } from "../testing/browser";

const none = { name: "main", params: [] };
const SRC = join(import.meta.dirname, "..");

/** The source files that ship: every module under `src` but tests, test tooling and the canon's stubs. */
function shippedSources(): string[] {
  const files: string[] = [];
  const walk = (dir: string) => {
    for (const name of readdirSync(dir)) {
      const path = join(dir, name);
      if (statSync(path).isDirectory()) {
        if (!["testing", "benches", "canon", "fixtures"].includes(name)) walk(path);
      } else if (/\.ts$/.test(name) && !/\.test(-d)?\.ts$|\.bench\.ts$/.test(name)) files.push(path);
    }
  };
  walk(SRC);
  return files;
}

describe("no module of rmsl throws a plain error", () => {
  /**
   * Every module that ships is read for an error built from a class other than
   * the ones `src/errors.ts` declares.
   *
   * @canon spec-no-module-of-rmsl-throws-a-plain-error
   */
  it("finds no plain Error, TypeError or RangeError built in a module that ships", () => {
    const plain = shippedSources().flatMap((file) =>
      readFileSync(file, "utf8")
        .split("\n")
        .flatMap((line, i) =>
          /new (?:Type|Range|Syntax)?Error\(|throw\s+["'`]/.test(line)
            ? [`${file.slice(SRC.length + 1)}:${i + 1}`]
            : [],
        ),
    );
    expect(plain).toEqual([]);
  });
});

describe("an error is an instance of the class of its kind", () => {
  /**
   * Each node is wrong whatever program holds it, so the builder refuses it as
   * it is made, before any target compiles anything.
   *
   * @canon spec-a-node-wrong-in-itself-is-a-node-error
   */
  it.each<[string, string, () => unknown]>([
    [
      "an assignment to a uniform",
      "assign-to-read-only",
      () => Fn(() => (uniform("vec3") as any).assign(vec3(1, 2, 3)))(),
    ],
    [
      "an assignment to a read-only storage element",
      "assign-to-read-only-storage",
      () => Fn(() => (instancedArray(4, "float").toReadOnly().element(int(0)) as any).assign(float(1)))(),
    ],
    [
      "an assignment to a whole storage buffer",
      "assign-to-whole-storage",
      () => Fn(() => (instancedArray(4, "vec3") as any).assign(vec3(1, 2, 3)))(),
    ],
    [
      "an assignment through a swizzle that repeats a component",
      "swizzle-repeats-component",
      () => Fn(() => (vec3(1, 2, 3).toVar() as any).xx.assign(vec2(1, 2)))(),
    ],
    [
      "a whole storage buffer read as a value",
      "storage-read-as-a-whole",
      () => (instancedArray(4, "float") as any).add(1),
    ],
    [
      "an outputStruct used as an operand",
      "output-struct-misplaced",
      () => float(1).add(outputStruct(float(1)) as any),
    ],
    ["a fold of literals that is not finite", "constant-not-finite", () => float(1).div(0)],
    ["a constant index past a vector", "index-out-of-range", () => vec3(1, 2, 3).element(int(5))],
    [
      "the inverse of a matrix that is not square",
      "inverse-of-non-square",
      () => (mat2x3(1, 0, 0, 0, 1, 0) as any).inverse(),
    ],
    ["a cross of two vectors that are not vec3", "cross-needs-vec3", () => (vec2(1, 0) as any).cross(vec2(0, 1))],
    [
      "a Return in a layout Fn that gives a value",
      "return-without-value",
      () =>
        Fn(() => {
          Return();
          return float(1);
        }).setLayout({ name: "f", type: "float", inputs: [] })(),
    ],
    [
      "a layout Fn that writes the position",
      "layout-fn-uses-stage-value",
      () =>
        Fn(() => {
          builtinPosition().assign(vec4(0, 0, 0, 1));
        }).setLayout({ name: "f", type: "void", inputs: [] })(),
    ],
  ])("refuses %s with a NodeError as the graph is built", (_, code, build) => {
    expectThrows(build, NodeError, code, { target: "none" });
  });

  /**
   * A parameter exists only inside a compile, so the refusal comes as that
   * compile builds the graph, before it compiles anything.
   *
   * @canon spec-a-node-wrong-in-itself-is-a-node-error
   */
  it("refuses an assignment to a parameter of the compiled function with a NodeError as the graph is built", () => {
    expectThrows(
      () =>
        compileJSFn(
          (x: Node<"float">) =>
            Fn(() => {
              (x as any).assign(float(2));
              return x;
            })(),
          { name: "main", params: [{ name: "x", type: "float" }] },
        ),
      NodeError,
      "assign-to-parameter",
    );
  });

  /**
   * The sum is built of two ints and rewired in the saved data to an int and a
   * uint, which the builder would have refused.
   *
   * @canon spec-a-node-wrong-in-itself-is-a-node-error
   */
  it("refuses a node wrong in itself with a NodeError as it is built, and in a rebuilt graph on every target", () => {
    const count = uniform("int");
    const mask = uniform("uint");
    expectThrows(() => count.add(mask as any), NodeError, "operands-of-different-kinds", { target: "none" });
    const graph = serialize(Fn(() => vec4(count.add(count).toFloat(), mask.toFloat(), 0, 1))()) as any;
    const sum = graph.nodes.find((n: any) => n.type === "add");
    sum.params[1] = graph.nodes.findIndex((n: any) => n.type === "uniform" && n._t === "uint");
    const rebuilt = () => deserialize(JSON.parse(JSON.stringify(graph))) as any;
    expectRefusedOnEveryTarget(
      {
        glsl: () => compileGlsl.fragment(rebuilt()),
        wgsl: () => compileWgsl.fragment(rebuilt()),
        js: () => compileJSFragment(rebuilt, none),
        wasm: () => compileWasmFragment(rebuilt, none),
      },
      NodeError,
      "operands-of-different-kinds",
    );
  });

  /**
   * @canon spec-a-program-that-breaks-a-rule-is-refused-with-a-program-error
   */
  it("refuses a Break outside a loop with a ProgramError on every target", () => {
    const build = () =>
      Fn(() => {
        Break();
        return vec4(0, 0, 0, 1);
      })();
    expectRefusedOnEveryTarget(
      {
        glsl: () => compileGlsl.fragment(build()),
        wgsl: () => compileWgsl.fragment(build()),
        js: () => compileJSFragment(build, none),
        wasm: () => compileWasmFragment(build, none),
      },
      ProgramError,
      "jump-outside-a-loop",
    );
  });

  /**
   * @canon spec-a-compile-call-with-wrong-options-is-refused-with-a-program-error
   */
  it("refuses gpuUniformLayout beside 64-bit floats with a ProgramError", () => {
    const u = uniform("float");
    expectThrows(
      () =>
        compileWasmRoutine(() => u.add(1), {
          ...none,
          gpuUniformLayout: { offsets: {}, size: 16 },
        } as any),
      ProgramError,
      "gpu-uniform-layout-needs-f32",
      { target: "wasm" },
    );
  });

  /**
   * @canon spec-stages-that-do-not-fit-are-refused-with-a-program-error
   */
  it("refuses a varying the vertex stage writes and the fragment stage never reads with a ProgramError", () => {
    const pos = attribute("vec3");
    const unused = varying("float");
    const vertex = () =>
      Fn(() => {
        unused.assign(float(1));
        return vec4(pos, 1);
      })();
    const fragment = () => Fn(() => vec4(1, 0, 0, 1))();
    expectThrows(() => compileWasm(vertex, fragment), ProgramError, "varying-never-read", { target: "wasm" });
  });

  /**
   * @canon spec-a-build-module-with-malformed-exports-is-refused-with-a-program-error
   */
  it("refuses a precompiled module with no default export with a ProgramError", async () => {
    const plugin = precompileShaders({ include: "no-default.ts" }) as any;
    const id = join(SRC, "vite/fixtures/no-default.ts");
    const error = await caught(() => plugin.transform("export const x = 1;\n", id));
    expectRmslError(error, ProgramError, "module-has-no-default-export");
  });

  /**
   * @canon spec-host-data-of-the-wrong-kind-or-size-is-refused-with-a-usage-error
   */
  it("refuses an Int32Array for a float attribute with a UsageError", () => {
    const pos = attribute("vec3");
    const adapter = createJs(
      () => Fn(() => vec4(pos, 1))(),
      () => Fn(() => vec4(1, 0, 0, 1))(),
    );
    expectThrows(() => adapter.setAttribute(pos, new Int32Array(9) as any), UsageError, "host-array-of-another-kind", {
      target: "js",
    });
  });

  /**
   * @canon spec-a-wrong-call-of-an-adapter-is-refused-with-a-usage-error
   */
  it("refuses a draw before attach with a UsageError", () => {
    const adapter = createJsGrid({ draw: Fn(() => vec4(1, 0, 0, 1))() });
    expectThrows(() => adapter.draw(), UsageError, "not-attached", { target: "js" });
  });

  /**
   * @canon spec-a-refusal-by-one-target-is-a-target-error
   */
  it("refuses a derivative on the CPU targets with a TargetError, where a GPU target compiles it", () => {
    const build = () => Fn(() => vec4(dFdx(fragCoord().x), 0, 0, 1))();
    expect(() => compileGlsl.fragment(build())).not.toThrow();
    expectRefusedOnEveryTarget(
      { js: () => compileJSFragment(build, none), wasm: () => compileWasmFragment(build, none) },
      TargetError,
      "no-derivatives",
    );
  });

  /**
   * @canon spec-a-malformed-saved-graph-is-a-data-error
   */
  it("refuses a saved uniform with no value with a DataError", () => {
    const graph = serialize([Fn(() => vec4(uniform("float").add(float(1)), 0, 0, 1))()]) as any;
    const at = graph.nodes.findIndex((n: any) => n.type === "uniform");
    delete graph.nodes[at].value;
    expectThrows(() => deserialize(graph), DataError, "node-without-value", { target: "none" });
  });

  /**
   * A canvas whose `getContext` gives nothing stands for a browser without WebGL 2.
   *
   * @canon spec-a-missing-platform-is-a-platform-error
   */
  it("throws a PlatformError when the canvas gives no WebGL 2 context", () => {
    const pos = attribute("vec3");
    const adapter = createGlsl(Fn(() => vec4(pos, 1))(), Fn(() => vec4(1, 0, 0, 1))());
    expectThrows(() => adapter.attach({ getContext: () => null } as any), PlatformError, "webgl2-unavailable", {
      target: "glsl",
    });
  });

  /**
   * @canon spec-an-inconsistent-compiler-is-an-internal-error
   */
  it("throws an InternalError for a node of a kind rmsl does not have", () => {
    expectThrows(() => node({ _t: "float", type: "no-such-kind" }), InternalError, "node-of-no-kind");
  });

  /**
   * @canon spec-an-error-is-named-for-its-class
   */
  it("names each error for its class", () => {
    const classes = [
      RmslError,
      NodeError,
      ProgramError,
      UsageError,
      TargetError,
      DataError,
      PlatformError,
      LimitError,
      DriverError,
      InternalError,
    ] as const;
    const names = classes.map(
      (Class) =>
        new (Class as any)("code", "message", { target: "js", limit: 1, requested: 2, log: "", index: 0 }).name,
    );
    expect(names).toEqual([
      "RmslError",
      "NodeError",
      "ProgramError",
      "UsageError",
      "TargetError",
      "DataError",
      "PlatformError",
      "LimitError",
      "DriverError",
      "InternalError",
    ]);
  });
});

describe("a code and a target", () => {
  /**
   * @canon spec-a-code-names-what-went-wrong-and-not-the-backend
   */
  it("gives fragCoord() in a vertex stage one code on every target", () => {
    const build = () => Fn(() => vec4(fragCoord().x, 0, 0, 1))();
    expectRefusedOnEveryTarget(
      {
        glsl: () => compileGlsl.vertex(build()),
        wgsl: () => compileWgsl.vertex(build()),
        js: () => compileJSVertex(build, none),
        wasm: () => compileWasmVertex(build, none),
      },
      ProgramError,
      "builtin-outside-its-stage",
    );
  });

  /**
   * @canon spec-an-error-thrown-in-a-backend-call-carries-its-target
   */
  it("names the backend whose compile refused the program", () => {
    const build = () => Fn(() => vec4(fragCoord().x, 0, 0, 1))();
    expectThrows(() => compileWgsl.vertex(build()), ProgramError, "builtin-outside-its-stage", { target: "wgsl" });
    expectThrows(() => compileJSVertex(build, none), ProgramError, "builtin-outside-its-stage", { target: "js" });
  });

  /**
   * @canon spec-an-error-from-no-backend-carries-no-target
   */
  it("names no backend for a mistake made while the graph is built, inside a compile or outside one", () => {
    expectThrows(() => float(1).add(uint(2) as any), NodeError, "operands-of-different-kinds", { target: "none" });
    expectThrows(
      () => compileJSRoutine(() => float(1).add(uint(2) as any), none),
      NodeError,
      "operands-of-different-kinds",
      { target: "none" },
    );
    expectThrows(() => deserialize({} as any), DataError, "graph-shape", { target: "none" });
  });

  /**
   * @canon spec-a-data-error-carries-the-index-of-the-malformed-entry
   */
  it("gives the index of the node of a saved graph that names a child it lacks", () => {
    const graph = serialize([Fn(() => vec4(uniform("float").add(float(1)), 0, 0, 1))()]) as any;
    const at = graph.nodes.findIndex((n: any) => n.params?.length);
    graph.nodes[at].params[0] = graph.nodes.length + 4;
    const error = caughtSync(() => deserialize(graph));
    expectRmslError(error, DataError, "index-out-of-range");
    expect((error as DataError).index).toBe(at);
  });

  /**
   * Every module that ships is read for a message that opens with a prefix
   * naming rmsl. A console warning, which prints no class name, keeps its prefix.
   *
   * @canon spec-a-message-carries-no-prefix
   */
  it("finds no message that opens with [RMSL] or rmsl:", () => {
    const prefixed = shippedSources().flatMap((file) =>
      readFileSync(file, "utf8")
        .split("\n")
        .flatMap((line, i) =>
          !line.includes("console.") && /["'`](\[RMSL[^\]]*\] ?|rmsl: )/.test(line)
            ? [`${file.slice(SRC.length + 1)}:${i + 1}`]
            : [],
        ),
    );
    expect(prefixed).toEqual([]);
  });
});

describe.skipIf(!GPU_ENABLED)("errors of a WebGPU device", () => {
  let uninstall: (() => void) | undefined;
  beforeAll(async () => {
    uninstall = await installWebGpuGlobals();
  });
  afterAll(() => uninstall?.());

  /**
   * A device with WebGPU's default limit of 8 storage buffers in one shader stage.
   *
   * @canon spec-work-past-a-device-limit-is-a-limit-error
   */
  it("refuses more storage buffers than a shader stage binds with a LimitError", async () => {
    const device = await (await navigator.gpu.requestAdapter())!.requestDevice();
    const context = await createWgslContext({ device });
    const inputs = Array.from({ length: 8 }, () => instancedArray(1, "uint"));
    const out = instancedArray(1, "uint");
    const sum = Fn(() => {
      out.element(0).assign(inputs.reduce((total, input) => total.add(input.element(0)), uint(0)));
    })().compute(1);
    expectThrows(() => context.compute(sum), LimitError, "storage-buffer-count", { target: "wgsl" });
    context.destroy();
    device.destroy();
  });

  /**
   * @canon spec-a-limit-error-carries-the-limit-and-the-amount-requested
   */
  it("gives the limit and the amount requested", async () => {
    const device = await (await navigator.gpu.requestAdapter())!.requestDevice();
    const context = await createWgslContext({ device });
    const inputs = Array.from({ length: 8 }, () => instancedArray(1, "uint"));
    const out = instancedArray(1, "uint");
    const sum = Fn(() => {
      out.element(0).assign(inputs.reduce((total, input) => total.add(input.element(0)), uint(0)));
    })().compute(1);
    const error = caughtSync(() => context.compute(sum)) as LimitError;
    expect([error.limit, error.requested]).toEqual([8, 9]);
    context.destroy();
    device.destroy();
  });
});

/**
 * A `bool` attribute, which GLSL ES 3.00 does not allow as a vertex input, so
 * the driver refuses the vertex shader rmsl writes for it.
 */
const DRIVER_SCENE = `
import { Fn, attributeRaw, builtinPosition, float, vec4 } from "../rmsl";
import { createGlsl } from "../glsl";
globalThis.__rmslDriverError = () => {
  const pos = attributeRaw("pos", "vec3");
  const flag = attributeRaw("flag", "bool");
  const vertex = Fn(() => builtinPosition().assign(vec4(pos, flag.select(float(1), float(0)))))();
  const fragment = Fn(() => vec4(0, 1, 0, 1))();
  const canvas = document.createElement("canvas");
  try {
    createGlsl(vertex, fragment).attach(canvas);
    return { thrown: false };
  } catch (error) {
    return { name: error.name, code: error.code, target: error.target, log: error.log };
  }
};
`;

describe.skipIf(!GPU_ENABLED)("errors of a WebGL 2 driver", () => {
  afterAll(async () => {
    await releaseGpu();
  }, 120_000);

  /**
   * @canon spec-a-driver-refusal-is-a-driver-error
   */
  it("throws a DriverError for a vertex shader the driver does not compile", async () => {
    const result = await runInGpuPage(DRIVER_SCENE, "__rmslDriverError", import.meta.dirname);
    expect(result).toMatchObject({ name: "DriverError", code: "shader-compile", target: "glsl" });
  }, 120_000);

  /**
   * @canon spec-a-driver-error-carries-the-log-of-the-driver
   */
  it("gives the driver's log", async () => {
    const result = await runInGpuPage(DRIVER_SCENE, "__rmslDriverError", import.meta.dirname);
    expect(result.log).toMatch(/bool/);
  }, 120_000);
});

describe("the scene departs from three.js where three.js lets a mistake through", () => {
  /**
   * @canon exception-a-vector-component-out-of-range-is-a-usage-error
   */
  it("refuses a vector component past z with a UsageError", () => {
    expectThrows(() => new Vector3().setComponent(5, 1), UsageError, "component-out-of-range");
  });

  /**
   * @canon exception-an-unknown-euler-order-or-colour-is-refused
   */
  it("refuses an unknown Euler order and a colour it cannot read with a UsageError", () => {
    expectThrows(
      () => new Euler(0, 0, 0, "XZZ" as any).setFromRotationMatrix(new (Object3D as any)().matrix),
      UsageError,
      "unknown-euler-order",
    );
    expectThrows(() => new Color().setStyle("bogus(1,2,3)"), UsageError, "unknown-colour-style");
  });

  /**
   * @canon exception-an-object-added-to-itself-is-refused
   */
  it("refuses an object added to itself with a UsageError", () => {
    const object = new Object3D();
    expectThrows(() => object.add(object), UsageError, "object-added-to-itself");
  });
});
