import { describe, expect, it } from "vitest";
import wabtInit from "wabt";
import {
  attribute,
  builtinFragDepth,
  builtinPosition,
  Discard,
  float,
  fragCoord,
  Fn,
  If,
  instancedArray,
  For,
  Loop,
  While,
  int,
  invocationIndex,
  ivec2,
  mat3,
  mix,
  select,
  step,
  storage,
  StorageBufferAttribute,
  textureLoad,
  uniform,
  varying,
  vec2,
  vec3,
  vec4,
  type Node,
  instanceIndex,
  texture,
  uint,
  vertexIndex,
} from "../rmsl";
import { compileJS, compileJSFragment, compileJSCompute, compileJSRoutine } from "../js";
import type { CompileCpuRoutine } from "../backends/cpu";
import { instantiateRasterizer } from "../backends/wasm/rasterizer";
import { CONDITION_PAIRS, inlineFnPaths } from "../testing/inline-fn-paths";
import {
  createWasmCompute,
  compileWasm,
  compileWasmCompute,
  compileWasmFn,
  compileWasmRoutine,
  compileWasmFragment,
  compileWasmGrid,
  instantiateWasmCompute,
  instantiateWasmGrid,
  instantiateWasmRoutine,
} from "../wasm";

const none = { name: "main", params: [] };

/** Both CPU targets, by name, as a compiler of a routine. */
const computes = [
  ["JS", compileJSCompute],
  ["WASM", compileWasmCompute],
] as const;

const cpuTargets: [string, CompileCpuRoutine][] = [
  ["JS", compileJSRoutine],
  ["WASM", compileWasmRoutine],
];

/** Compiles `build` to WASM, instantiates the module by hand, and calls its `main` with `args`. */
function callMain(build: (...args: any[]) => Node<any>, params: { name: string; type: "float" }[], args: number[]) {
  const compiled = compileWasmFn(build, { name: "main", params });
  const memory = new WebAssembly.Memory({ initial: compiled.memoryPages });
  const instance = new WebAssembly.Instance(new WebAssembly.Module(compiled.bytes.buffer as ArrayBuffer), {
    math: Math as unknown as WebAssembly.ModuleImports,
    env: { memory },
  });
  const returned = (instance.exports.main as (...args: number[]) => unknown)(...args);
  return { returned, kinds: compiled.params.map((p) => p.kind) };
}

describe("a WASM function's result", () => {
  /**
   * @canon spec-a-wasm-function-returns-a-scalar-result-as-its-value
   */
  it("returns a float result from main itself", () => {
    const { returned, kinds } = callMain((a: Node<"float">) => a.mul(3), [{ name: "a", type: "float" }], [2]);
    expect(returned).toBe(6);
    expect(kinds).toEqual(["param"]);
  });

  /**
   * @canon spec-a-wasm-function-returns-a-vector-result-through-memory
   */
  it("returns nothing from main for a vec2 result, and names a memory slot for it", () => {
    const { returned, kinds } = callMain((a: Node<"float">) => vec2(a, 1), [{ name: "a", type: "float" }], [2]);
    expect(returned).toBeUndefined();
    expect(kinds).toEqual(["param", "valueMemory"]);
  });
});

describe("a CPU target's filter", () => {
  /** Two texels, 0 and 100. Halfway between them reads 100 unfiltered and 50 blended. */
  const texels = { data: [0, 0, 0, 0, 100, 100, 100, 100], width: 2, height: 1 };

  /**
   * @canon spec-a-cpu-target-filters-by-the-magnification-filter-alone
   */
  it.each(cpuTargets)("filters by magFilter alone on %s", (_, compile) => {
    let tex: any;
    const build = () =>
      Fn(() => {
        tex = uniform("sampler2D");
        return texture(tex, vec2(0.5, 0.5)).x;
      })();
    const routine = compile(build as any, none);
    const red = (image: object) => routine({ textures: { [tex.name]: { ...texels, ...image } } });
    expect(red({ minFilter: "linear" })).toBe(100);
    expect(red({ magFilter: "linear", minFilter: "nearest" })).toBe(50);
  });
});

describe("a CPU target's storage buffers", () => {
  /**
   * @canon spec-an-element-no-invocation-writes-keeps-the-host-value
   */
  it.each(computes)("keeps the elements no invocation writes on %s", (_, compile) => {
    const buffer = instancedArray(4, "float");
    const build = () =>
      Fn(() => {
        buffer.element(invocationIndex().mul(2)).assign(float(9));
      })();
    const data = new Float64Array([1, 2, 3, 4]);
    compile(build as any, none)({ storages: { [buffer.name]: data } }, 2);
    expect(Array.from(data)).toEqual([9, 2, 9, 4]);
  });

  /**
   * Two nodes over one buffer, one of them read-only, in either order.
   *
   * @canon spec-a-wasm-routine-copies-back-a-buffer-another-node-reads-only
   */
  it.each([
    ["read-only node first", true],
    ["writable node first", false],
  ])("copies back a buffer written through one node, %s", (_, readOnlyFirst) => {
    const shared = new StorageBufferAttribute(2, 1);
    const build = () =>
      Fn(() => {
        const first = readOnlyFirst ? storage(shared, "float").toReadOnly() : storage(shared, "float");
        const second = readOnlyFirst ? storage(shared, "float") : storage(shared, "float").toReadOnly();
        const [source, target] = readOnlyFirst ? [first, second] : [second, first];
        const i = invocationIndex();
        (target as ReturnType<typeof storage<"float">>).element(i).assign(source.element(i).mul(2));
      })();
    const data = new Float64Array([1, 2]);
    compileWasmCompute(build as any, none)({ storages: { [storage(shared, "float").name]: data } }, 2);
    expect(Array.from(data)).toEqual([2, 4]);
  });
});

describe("a CPU rasterizer's triangles", () => {
  const position = attribute("vec3");
  const vertex = () => Fn(() => builtinPosition().assign(vec4(position.x, position.y, position.z, 1)))();
  const fragment = () => Fn(() => vec4(1, 1, 1, 1))();
  const counterClockwise = new Float64Array([-1, -1, 0, 1, -1, 0, -1, 1, 0]);
  const clockwise = new Float64Array([-1, -1, 0, -1, 1, 0, 1, -1, 0]);
  const rasterizers = [
    ["JS", () => compileJS(vertex as any, fragment as any)],
    ["WASM", () => compileWasm(vertex as any, fragment as any)],
  ] as const;

  /**
   * Three pixel centres lie on the triangle's long edge, a right edge, which
   * shades none of them, so it covers the three below it.
   *
   * @canon spec-a-cpu-rasterizer-draws-a-triangle-whichever-way-it-winds
   */
  it.each(rasterizers)("draws the same pixels for either winding on %s", (_, make) => {
    const routine = make();
    const covered = (triangle: Float64Array) =>
      Array.from(
        routine.draw(
          { attributes: { [position.name]: triangle } },
          { width: 3, height: 3, clear: true, clearDepth: true },
        ),
      ).filter((_, i) => i % 4 === 3);
    expect(covered(counterClockwise)).toEqual(covered(clockwise));
    expect(covered(clockwise).filter((alpha) => alpha === 1)).toHaveLength(3);
  });
});

/** A triangle at depth `z` that covers the whole viewport. */
const screen = (z = 0) => new Float64Array([-1, -1, z, 3, -1, z, -1, 3, z]);

/** A vertex stage that places `pos` as given, and a fragment stage that draws `color`. */
function flat() {
  const pos = attribute("vec3");
  const color = uniform("vec4");
  const vertex = () => Fn(() => builtinPosition().assign(vec4(pos, 1)))();
  const fragment = () => Fn(() => color)();
  return { pos, color, routine: compileWasm(vertex as any, fragment as any) };
}

describe("the WASM rasterizer's memory", () => {
  /**
   * A vertex stage's texture survives the next draw: its heap lies after the
   * fragment stage's layout, which the fragment inputs of each draw write.
   *
   * @canon spec-the-wasm-rasterizer-links-its-stages-in-one-memory
   */
  it("keeps a vertex-stage texture intact across draws on WASM", () => {
    const pos = attribute("vec3");
    const tex = uniform("sampler2D");
    const shade = varying("float");
    const tint = uniform("vec4");
    const vertex = () =>
      Fn(() => {
        shade.assign(textureLoad(tex, ivec2(0, 0)).x);
        builtinPosition().assign(vec4(pos, 1));
      })();
    const fragment = () => Fn(() => vec4(shade, tint.x, tint.y, 1))();
    const routine = compileWasm(vertex as any, fragment as any);
    const ctx = {
      attributes: { [pos.name]: screen() },
      uniforms: { [tint.name]: [0.25, 0.5, 0, 0] },
      textures: { [tex.name]: { data: new Float32Array(64).fill(9), width: 4, height: 4 } },
    };
    expect(Array.from(routine.draw(ctx, { width: 1, height: 1 }))).toEqual([9, 0.25, 0.5, 1]);
    expect(Array.from(routine.draw(ctx, { width: 1, height: 1 }))).toEqual([9, 0.25, 0.5, 1]);
  });

  /**
   * A draw after a smaller draw gives what a fresh routine gives: the depth
   * buffer lies after the regions of the draw, so a larger draw does not lay
   * its vertices or colours over it.
   *
   * @canon spec-a-rasterizer-keeps-the-closer-fragment
   */
  it("draws the same after a smaller draw as on a fresh routine on WASM", () => {
    const fresh = flat();
    const used = flat();
    const four = new Float64Array(Array.from({ length: 4 }, () => Array.from(screen())).flat());
    const options = { width: 2, height: 2, clear: true, clearDepth: true };
    const expected = fresh.routine.draw(
      { attributes: { [fresh.pos.name]: four }, uniforms: { [fresh.color.name]: [0, 0, 1, 1] } },
      options,
    );
    used.routine.draw(
      { attributes: { [used.pos.name]: screen() }, uniforms: { [used.color.name]: [1, 0, 0, 1] } },
      { width: 1, height: 1 },
    );
    const got = used.routine.draw(
      { attributes: { [used.pos.name]: four }, uniforms: { [used.color.name]: [0, 0, 1, 1] } },
      options,
    );
    expect(Array.from(got)).toEqual(Array.from(expected));
  });

  /**
   * The depth buffer keeps its values when a later draw's regions reach it and
   * it moves. A far triangle is drawn, then a nearer one with more vertices,
   * which is accepted, and then a farther one with more again, which is
   * rejected by the depth the nearer one left. A depth buffer that lost its
   * values on the move would reject the nearer one, as zero depth rejects all.
   *
   * @canon spec-a-rasterizer-keeps-the-closer-fragment
   */
  it("keeps the closer fragment of earlier draws when later draws move the depth buffer", () => {
    const { pos, color, routine } = flat();
    const triangles = (z: number, count: number) =>
      new Float64Array(Array.from({ length: count }, () => Array.from(screen(z))).flat());
    const draw = (z: number, count: number, rgba: number[], clearDepth = false) =>
      Array.from(
        routine.draw(
          { attributes: { [pos.name]: triangles(z, count) }, uniforms: { [color.name]: rgba } },
          { width: 1, height: 1, clear: true, clearDepth },
        ),
      );
    expect(draw(0.6, 1, [1, 0, 0, 1], true)).toEqual([1, 0, 0, 1]);
    expect(draw(0.2, 2, [0, 0, 1, 1])).toEqual([0, 0, 1, 1]);
    expect(draw(0.4, 4, [0, 1, 0, 1])).toEqual([0, 0, 0, 0]);
  });

  /**
   * The rasterizer module is built from its `.wat` twice: importing its memory
   * as it is, and importing it shared.
   *
   * @canon spec-compile-wasm-makes-its-memory-as-its-modules-declare
   */
  it("links its rasterizer against a memory that is not shared, and a shared one of any maximum, on WASM", () => {
    const stages = { vertex: { main: () => {} }, fragment: { main: () => {} } };
    for (const [memory, shared] of [
      [new WebAssembly.Memory({ initial: 1 }), false],
      [new WebAssembly.Memory({ initial: 1, maximum: 2, shared: true }), true],
      [new WebAssembly.Memory({ initial: 1, maximum: 65536, shared: true }), true],
    ] as const) {
      expect(() => instantiateRasterizer(stages.vertex.main, stages.fragment.main, memory, shared)).not.toThrow();
    }
  });

  /**
   * `compileWasm` makes its own memory as the modules it compiled declare it,
   * shared and with their maximum, so they link.
   *
   * @canon spec-compile-wasm-makes-its-memory-as-its-modules-declare
   */
  it("compiles a vertex and fragment pair with sharedMemory and no memory on WASM", () => {
    const pos = attribute("vec3");
    const vertex = () => Fn(() => builtinPosition().assign(vec4(pos, 1)))();
    const fragment = () => Fn(() => vec4(1, 0, 0, 1))();
    const routine = compileWasm(vertex as any, fragment as any, { sharedMemory: true });
    expect(Array.from(routine.draw({ attributes: { [pos.name]: screen() } }, { width: 1, height: 1 }))).toEqual([
      1, 0, 0, 1,
    ]);
  });
});

describe("the exports of a WASM module", () => {
  /** The names a module compiled from `build` exports. */
  const exportsOf = (build: () => any, name: string) =>
    WebAssembly.Module.exports(new WebAssembly.Module(compileWasmFn(build, { name, params: [] }).bytes as any)).map(
      (entry) => entry.name,
    );

  /**
   * @canon spec-a-wasm-module-exports-its-loops-under-the-reserved-prefix
   */
  it.each(["draw", "compute"])("exports a program named %s beside its own loops", (name) => {
    expect(exportsOf(() => Fn(() => fragCoord().x)(), name).sort()).toEqual(
      ["_rmsl_compute", "_rmsl_draw", name].sort(),
    );
  });

  /**
   * @canon spec-a-wasm-module-exports-its-loops-under-the-reserved-prefix
   */
  it("refuses a program name in the reserved prefix", () => {
    expect(() => exportsOf(() => Fn(() => float(1))(), "_rmsl_draw")).toThrow(/"_rmsl_draw".*reserve/);
  });

  /**
   * @canon spec-a-wasm-module-exports-its-loops-under-the-reserved-prefix
   */
  it("refuses to instantiate a program under a name in the reserved prefix", () => {
    const compiled = compileWasmFn(() => Fn(() => float(1))(), { name: "main", params: [] });
    expect(() => instantiateWasmRoutine(compiled, "_rmsl_compute")).toThrow(/"_rmsl_compute".*reserve/);
  });

  /**
   * @canon spec-a-wasm-stage-module-exports-no-loop-of-its-own
   */
  it.each(["vertex", "fragment"] as const)("exports no loop of its own from a %s stage module", (stage) => {
    const compiled = compileWasmFn(() => Fn(() => vec4(1, 0, 0, 1))(), { name: "main", params: [], stage });
    const names = WebAssembly.Module.exports(new WebAssembly.Module(compiled.bytes as any)).map((e) => e.name);
    expect(names).toEqual(["main"]);
  });

  /**
   * @canon spec-a-wasm-stage-module-exports-no-loop-of-its-own
   */
  it("refuses to instantiate a stage module as a grid or a compute stage", () => {
    const compiled = compileWasmFn(() => Fn(() => vec4(1, 0, 0, 1))(), { name: "main", params: [], stage: "fragment" });
    expect(() => instantiateWasmGrid(compiled, "main")).toThrow(/exports no "_rmsl_draw"/);
    expect(() => instantiateWasmCompute(compiled, "main")).toThrow(/exports no "_rmsl_compute"/);
  });

  /**
   * @canon spec-a-wasm-routine-loops-inside-its-module
   */
  it("exports a dispatch loop for a program that reads neither storage nor invocationIndex", () => {
    expect(exportsOf(() => Fn(() => float(1).toVar())(), "main")).toContain("_rmsl_compute");
  });

  /**
   * @canon spec-a-cpu-adapter-computes-synchronously
   */
  it("creates a compute adapter under its default name, and computes", () => {
    const buf = instancedArray(2, "float");
    const adapter = createWasmCompute(
      Fn(() => {
        buf.element(invocationIndex()).assign(float(3));
      })(),
    );
    const data = new Float32Array(2);
    adapter.setAttribute(buf.name, data);
    adapter.compute();
    expect(Array.from(data)).toEqual([3, 3]);
  });
});

describe("the imports of a WASM module", () => {
  /** What a module compiled from `build` imports, as `module.name`, sorted. */
  const importsOf = (build: () => any, stage?: "compute") =>
    WebAssembly.Module.imports(
      new WebAssembly.Module(compileWasmFn(build, { name: "main", params: [], stage }).bytes as any),
    )
      .map((entry) => `${entry.module}.${entry.name}`)
      .sort();

  /**
   * @canon spec-a-wasm-program-without-a-transcendental-imports-its-memory-alone
   */
  it("imports its memory alone, whatever the program reads or writes", () => {
    const gain = uniform("float");
    const image = uniform("sampler2D");
    const values = instancedArray(4, "float");
    expect(importsOf(() => Fn(() => texture(image, vec2(0.5, 0.5)).x.mul(gain))())).toEqual(["env.memory"]);
    expect(importsOf(() => Fn(() => vec4(fragCoord().x, 0, 0, 1))())).toEqual(["env.memory"]);
    const write = () =>
      Fn(() => {
        values.element(invocationIndex()).assign(gain);
      })();
    expect(importsOf(write, "compute")).toEqual(["env.memory"]);
  });

  /**
   * @canon exception-a-wasm-transcendental-is-imported-from-the-host
   */
  it("imports from math each transcendental it calls, and nothing else", () => {
    const x = uniform("float");
    expect(importsOf(() => Fn(() => x.sin().add(x.pow(2)))())).toEqual(["env.memory", "math.pow", "math.sin"]);
  });
});

describe("bytes compiled with compileWasmFn", () => {
  /**
   * @canon spec-a-cpu-grid-evaluates-a-fragment-for-each-pixel
   */
  it("instantiate as a grid", () => {
    const compiled = compileWasmFn(() => Fn(() => fragCoord().x)() as any, { name: "main", params: [] });
    const grid = instantiateWasmGrid<"float">(compiled, "main");
    expect(Array.from(grid({}, 2, 1))).toEqual([0.5, 1.5]);
  });

  /**
   * @canon spec-a-cpu-compute-stage-runs-one-invocation-per-index
   */
  it("instantiate as a compute stage", () => {
    const buffer = instancedArray(2, "float");
    const compiled = compileWasmFn(
      () =>
        Fn(() => {
          buffer.element(invocationIndex()).assign(float(7));
        })() as any,
      { name: "main", params: [] },
    );
    const compute = instantiateWasmCompute(compiled, "main");
    const data = new Float64Array(2);
    compute({ storages: { [buffer.name]: data } }, 2);
    expect(Array.from(data)).toEqual([7, 7]);
  });
});

describe("a WASM routine's results", () => {
  /**
   * @canon spec-a-cpu-routine-answers-one-fragment-per-call
   */
  it("keeps the pixels a WASM routine drew when it draws again", () => {
    const routine = compileWasmGrid((a: any) => Fn(() => vec2(a, a.add(1)).toVar())(), {
      name: "main",
      params: [{ name: "a", type: "float" }],
    });
    const first = routine({ params: { a: 0.5 } }, 1, 1);
    routine({ params: { a: 100.5 } }, 1, 1);
    expect(Array.from(first)).toEqual([0.5, 1.5]);
  });

  /**
   * @canon spec-a-fragment-stage-returns-its-colour-and-outputs
   */
  it("returns null for a discarded fragment on WASM", () => {
    const build = () =>
      Fn(() => {
        Discard();
        return float(1);
      })();
    expect(compileWasmFragment(build, { ...none })({})).toBeNull();
  });

  /**
   * A stage that returns nothing ends in the value of an inline `Fn` that
   * discards; the inline `Fn`'s statements run where it is called, so the
   * discard runs though the final value holds nothing.
   *
   * @canon spec-a-fragment-stage-returns-its-colour-and-outputs
   */
  it("discards through an inline Fn that a stage writing no colour ends in, on JS and WASM", () => {
    const drop = uniform("float");
    const discarder = Fn(() => {
      If(drop.greaterThan(0.5), () => Discard());
    });
    const build = () => Fn(() => discarder())();
    for (const compile of [compileWasmFragment, compileJSFragment]) {
      const stage = (compile as any)(build, { ...none });
      expect(stage({ uniforms: { [drop.name]: 1 } })).toBeNull();
      expect(stage({ uniforms: { [drop.name]: 0 } })).not.toBeNull();
    }
  });
});

describe("an inline Fn whose value an operation reads more than once", () => {
  const counter = instancedArray(1, "float");
  /** An inline `Fn` that counts its runs in `counter` and returns `value`. */
  const counted = (value: number) =>
    Fn(() => {
      counter.element(int(0)).addAssign(1);
      return float(value);
    })() as any;
  const run = (compile: CompileCpuRoutine, build: () => any) => {
    const data = new Float64Array(1);
    const result = compile(build, none)({ storages: { [counter.name]: data } });
    return { runs: data[0], result };
  };

  /**
   * @canon spec-an-inline-fn-read-in-one-statement-runs-before-it
   */
  it.each(cpuTargets)("%s: runs it once when fract reads its value twice", (_, compile) => {
    expect(run(compile, () => counted(2.75).fract())).toEqual({ runs: 1, result: 0.75 });
  });

  /**
   * @canon spec-an-inline-fn-read-in-one-statement-runs-before-it
   */
  it.each(cpuTargets)("%s: runs it once when sign reads its value twice", (_, compile) => {
    expect(run(compile, () => counted(-3).sign())).toEqual({ runs: 1, result: -1 });
  });

  /**
   * @canon spec-an-inline-fn-read-in-one-statement-runs-before-it
   */
  it.each(cpuTargets)("%s: runs it once when the integer abs reads its value three times", (_, compile) => {
    expect(run(compile, () => counted(-3).toInt().abs().toFloat())).toEqual({ runs: 1, result: 3 });
  });

  /**
   * @canon spec-an-inline-fn-read-in-one-statement-runs-before-it
   */
  it.each(cpuTargets)(
    "%s: runs it once when a vector operation takes it as a scalar for every component",
    (_, compile) => {
      expect(run(compile, () => vec3(1, 2, 3).mul(counted(2)).z)).toEqual({ runs: 1, result: 6 });
    },
  );

  /**
   * @canon spec-an-inline-fn-read-in-one-statement-runs-before-it
   */
  it.each(cpuTargets)("%s: runs it once when clamp takes it as a scalar bound", (_, compile) => {
    expect(run(compile, () => vec3(1, 2, 3).clamp(counted(1.5), 2.5).x)).toEqual({ runs: 1, result: 1.5 });
  });

  /**
   * @canon exception-an-inline-fn-read-more-than-once-runs-before-every-read
   */
  it("runs it once before an If whose branches both read its value, on WASM", () => {
    const taken = uniform("float");
    const result = instancedArray(1, "float");
    const shared = counted(2);
    const build = () =>
      Fn(() => {
        If(taken.greaterThan(0), () => {
          result.element(int(0)).assign(shared.fract().add(shared));
        }).Else(() => {
          result.element(int(0)).assign(shared.mul(10));
        });
      })();
    const read = (branch: number) => {
      const data = new Float64Array(1);
      const out = new Float64Array(1);
      compileWasmRoutine(
        build,
        none,
      )({
        storages: { [counter.name]: data, [result.name]: out },
        uniforms: { [taken.name]: branch },
      });
      return { runs: data[0], value: out[0] };
    };
    expect(read(1)).toEqual({ runs: 1, value: 2 });
    expect(read(0)).toEqual({ runs: 1, value: 20 });
  });

  /**
   * @canon spec-an-inline-fn-read-in-one-statement-runs-before-it
   */
  it.each(cpuTargets)("%s: runs it once when mix takes it as the weight of every component", (_, compile) => {
    expect(run(compile, () => vec3(0, 0, 0).mix(vec3(4, 4, 4), counted(0.5)).z)).toEqual({ runs: 1, result: 2 });
  });
});

describe("an inline Fn read on several paths of a program", () => {
  const { counter, result, first, second, paths } = inlineFnPaths();
  const cases = cpuTargets.flatMap(([target, compile]) =>
    paths.map((path) => [target, path.name, compile, path] as const),
  );

  /**
   * @canon exception-an-inline-fn-read-more-than-once-runs-before-every-read
   */
  it.each(cases)("%s: runs an inline Fn read %s once on every path that reads it", (_, __, compile, path) => {
    const routine = compile(path.build, none);
    for (const [f, s] of CONDITION_PAIRS) {
      const runs = new Float64Array(1);
      const value = new Float64Array(1);
      routine({
        storages: { [counter.name]: runs, [result.name]: value },
        uniforms: { [first.name]: f ? 1 : 0, [second.name]: s ? 1 : 0 },
      });
      expect({ runs: runs[0], value: value[0] }, `first ${f}, second ${s}`).toEqual({
        runs: path.runs(f, s),
        value: path.value(f, s),
      });
    }
  });
});

describe("an inline Fn read in a loop", () => {
  const counter = instancedArray(1, "float");
  const result = instancedArray(1, "float");
  /** An `Fn` that adds one to `counter` and returns 2. */
  const counted = Fn(() => {
    counter.element(int(0)).addAssign(1);
    return float(2);
  });
  const run = (compile: CompileCpuRoutine, build: () => any) => {
    const runs = new Float64Array(1);
    const value = new Float64Array(1);
    compile(build, none)({ storages: { [counter.name]: runs, [result.name]: value } });
    return { runs: runs[0], value: value[0] };
  };

  /**
   * The `Fn` is called outside the program's body, and the loop body reads
   * its value once.
   *
   * @canon spec-an-inline-fn-read-once-in-a-loop-runs-on-every-iteration
   */
  it.each(cpuTargets)("%s: runs an inline Fn read once in a loop body on every iteration", (_, compile) => {
    const build = () => {
      const shared = counted() as any;
      return Fn(() => {
        Loop(4, () => {
          result.element(int(0)).addAssign(shared);
        });
      })();
    };
    expect(run(compile, build)).toEqual({ runs: 4, value: 8 });
  });

  /**
   * @canon spec-an-inline-fn-runs-where-it-is-called
   */
  it.each(cpuTargets)("%s: runs an Fn called before a loop once", (_, compile) => {
    const build = () =>
      Fn(() => {
        const shared = counted() as any;
        Loop(4, () => {
          result.element(int(0)).addAssign(shared);
        });
      })();
    expect(run(compile, build)).toEqual({ runs: 1, value: 8 });
  });
});

describe("an inline Fn read more than once inside another inline Fn read once", () => {
  const counter = instancedArray(1, "float");
  const result = instancedArray(1, "float");
  const taken = uniform("float");
  /** An `Fn` called outside the program's body, that adds one to `counter` and returns 2. */
  const counted = () =>
    Fn(() => {
      counter.element(int(0)).addAssign(1);
      return float(2);
    })() as any;
  /** A program that writes `value` to `result`. */
  const writing = (value: any) =>
    Fn(() => {
      result.element(int(0)).assign(value);
    })();
  const run = (compile: CompileCpuRoutine, build: () => any, branch = 0) => {
    const runs = new Float64Array(1);
    const value = new Float64Array(1);
    compile(
      build,
      none,
    )({
      storages: { [counter.name]: runs, [result.name]: value },
      uniforms: { [taken.name]: branch },
    });
    return { runs: runs[0], value: value[0] };
  };
  /** An `Fn` called outside the program's body, whose loop body reads `f` twice, three times over. */
  const looping = (f: any) =>
    Fn(() => {
      const t = float(0).toVar();
      Loop(3, () => {
        t.addAssign(f);
        t.addAssign(f.mul(10));
      });
      return t;
    })() as any;

  /**
   * @canon spec-an-inline-fn-read-only-inside-another-runs-in-the-block-there
   */
  it.each(cpuTargets)("%s: runs it on every iteration of the other's loop that reads it", (_, compile) => {
    const build = () => writing(looping(counted()));
    expect(run(compile, build)).toEqual({ runs: 3, value: 66 });
  });

  /**
   * @canon spec-an-inline-fn-read-only-inside-another-runs-in-the-block-there
   */
  it.each(cpuTargets)("%s: runs it only when the other's branch that reads it runs", (_, compile) => {
    const build = () => {
      const f = counted();
      const g = Fn(() => {
        const t = float(0).toVar();
        If(taken.greaterThan(0), () => {
          t.assign(f);
          t.addAssign(f.mul(10));
        });
        return t;
      })() as any;
      return writing(g);
    };
    expect(run(compile, build, 1)).toEqual({ runs: 1, value: 22 });
    expect(run(compile, build, 0)).toEqual({ runs: 0, value: 0 });
  });

  /**
   * The outer `Fn` reads the middle one twice per iteration of a loop of its
   * own, and the middle one reads the innermost twice per iteration of its loop.
   *
   * @canon spec-an-inline-fn-read-only-inside-another-runs-in-the-block-there
   */
  it.each(cpuTargets)("%s: runs it in the innermost block at every level of nesting", (_, compile) => {
    const build = () => {
      const g = looping(counted());
      const h = Fn(() => {
        const t = float(0).toVar();
        Loop(2, () => {
          t.addAssign(g);
          t.addAssign(g.mul(0));
        });
        return t;
      })() as any;
      return writing(h);
    };
    expect(run(compile, build)).toEqual({ runs: 6, value: 132 });
  });

  /**
   * The statements of a loop condition run before every test, and they read
   * the value twice.
   *
   * @canon spec-an-inline-fn-read-only-in-the-statements-of-a-while-condition-runs-before-every-test
   */
  it.each(cpuTargets)("%s: runs it before every test of a loop condition that reads it", (_, compile) => {
    const build = () => {
      const f = counted();
      return Fn(() => {
        const n = float(0).toVar();
        While(
          () => {
            const limit = f.add(f).toVar();
            return n.lessThan(limit);
          },
          () => {
            n.addAssign(1);
          },
        );
        result.element(int(0)).assign(n);
      })();
    };
    expect(run(compile, build)).toEqual({ runs: 5, value: 4 });
  });
});

describe("an inline Fn read more than once in a For condition", () => {
  /**
   * The condition reads the value twice, and runs before each of the four
   * tests of a loop that runs three times.
   *
   * @canon spec-an-inline-fn-read-only-in-a-for-condition-runs-before-every-test
   */
  it.each(cpuTargets)("%s: runs it before every test", (_, compile) => {
    const counter = instancedArray(1, "float");
    const build = () => {
      const f = Fn(() => {
        counter.element(int(0)).addAssign(1);
        return float(3);
      })() as any;
      return Fn(() => {
        For(
          () => float(0).toVar(),
          (n) => n.lessThan(f.add(f.mul(0))),
          (n) => n.addAssign(1),
          () => {},
        );
      })();
    };
    const runs = new Float64Array(1);
    compile(build, none)({ storages: { [counter.name]: runs } });
    expect(runs[0]).toBe(4);
  });

  /**
   * The condition is the value of the `Fn` itself, which counts its runs and
   * stops the loop at the third.
   *
   * @canon spec-an-inline-fn-read-only-in-a-for-condition-runs-before-every-test
   */
  it.each(cpuTargets)("%s: runs it before every test of a condition that is its value", (_, compile) => {
    const counter = instancedArray(1, "float");
    const build = () => {
      const f = Fn(() => {
        counter.element(int(0)).addAssign(1);
        return counter.element(int(0)).lessThan(3);
      })() as any;
      return Fn(() => {
        For(
          () => int(0).toVar(),
          () => f,
          (i) => i.addAssign(1),
          () => {},
        );
      })();
    };
    const runs = new Float64Array(1);
    compile(build, none)({ storages: { [counter.name]: runs } });
    expect(runs[0]).toBe(3);
  });

  /**
   * The condition is the value of the `Fn`, and the body reads it too, once or
   * twice. The `For` is the innermost block's statement that holds every read,
   * so the `Fn` runs once, before the loop, and the condition tests what it returns.
   *
   * @canon spec-an-inline-fn-a-for-condition-and-body-read-runs-once-before-the-loop
   */
  it.each(cpuTargets.flatMap(([name, compile]) => [1, 2].map((reads) => [name, reads, compile] as const)))(
    "%s: runs it once before a For whose condition is its value and whose body reads it %i times",
    (_, reads, compile) => {
      const counter = instancedArray(1, "float");
      const steps = instancedArray(1, "float");
      const build = () => {
        const f = Fn(() => {
          counter.element(int(0)).addAssign(1);
          return steps.element(int(0)).lessThan(3);
        })() as any;
        return Fn(() => {
          For(
            () => int(0).toVar(),
            () => f,
            (i) => i.addAssign(1),
            () => {
              steps.element(int(0)).addAssign(1);
              for (let k = 0; k < reads; k++) steps.element(int(0)).addAssign(select(f, float(0), float(0)));
            },
          );
        })();
      };
      const runs = new Float64Array(1);
      const taken = new Float64Array(1);
      compile(build, none)({ storages: { [counter.name]: runs, [steps.name]: taken } });
      expect({ runs: runs[0], steps: taken[0] }).toEqual({ runs: 1, steps: 3 });
    },
  );
});

describe("an inline Fn read more than once only inside another read from two roots", () => {
  /**
   * The outer `Fn` reads the inner one in both branches of an `If`, and two
   * roots read the outer one. The inner one runs before that `If`, inside the
   * outer one, so it runs on either branch.
   *
   * @canon spec-an-inline-fn-read-only-inside-another-runs-in-the-block-there
   */
  it.each(cpuTargets)("%s: runs it once in the block there that holds every read", (_, compile) => {
    const counter = instancedArray(1, "float");
    const result = instancedArray(2, "float");
    const taken = uniform("float");
    const build = () => {
      const g = Fn(() => {
        counter.element(int(0)).addAssign(1);
        return float(2);
      })() as any;
      const f = Fn(() => {
        const t = float(0).toVar();
        If(taken.greaterThan(0), () => {
          t.assign(g);
        }).Else(() => {
          t.assign(g.mul(10));
        });
        return t;
      })() as any;
      return [
        Fn(() => {
          result.element(int(0)).assign(f);
        })(),
        Fn(() => {
          result.element(int(1)).assign(f.add(1));
        })(),
      ];
    };
    for (const [branch, value] of [
      [1, 2],
      [0, 20],
    ]) {
      const runs = new Float64Array(1);
      const data = new Float64Array(2);
      compile(build as any, none)({
        storages: { [counter.name]: runs, [result.name]: data },
        uniforms: { [taken.name]: branch },
      });
      expect({ runs: runs[0], values: Array.from(data) }).toEqual({ runs: 1, values: [value, value + 1] });
    }
  });
});

describe("an inline Fn read in the program and inside another read twice", () => {
  /**
   * The program reads the inner `Fn` once, and twice the outer one, whose
   * statements read the inner one. Every read is in the program's block.
   *
   * @canon exception-an-inline-fn-read-more-than-once-runs-before-every-read
   */
  it.each(cpuTargets)("%s: runs it once, before the first read", (_, compile) => {
    const counter = instancedArray(1, "float");
    const result = instancedArray(3, "float");
    const build = () => {
      const g = Fn(() => {
        counter.element(int(0)).addAssign(1);
        return float(2);
      })() as any;
      const f = Fn(() => {
        const t = g.mul(10).toVar();
        return t;
      })() as any;
      return Fn(() => {
        result.element(int(0)).assign(g);
        result.element(int(1)).assign(f);
        result.element(int(2)).assign(f.add(1));
      })();
    };
    const runs = new Float64Array(1);
    const data = new Float64Array(3);
    compile(build, none)({ storages: { [counter.name]: runs, [result.name]: data } });
    expect({ runs: runs[0], values: Array.from(data) }).toEqual({ runs: 1, values: [2, 20, 21] });
  });
});

describe("an inline Fn that discards, read in both branches of an If", () => {
  /**
   * The `Fn` discards the fragment, and each branch reads its value, so
   * the fragment is dropped on either path.
   *
   * @canon spec-a-discard-in-an-inline-fn-drops-the-fragment-where-it-runs
   */
  it("drops the fragment on either path, on JS and WASM", () => {
    const side = uniform("float");
    const build = () => {
      const shared = Fn(() => {
        Discard();
        return float(1);
      })() as any;
      return Fn(() => {
        const colour = vec4(0, 0, 0, 1).toVar();
        If(side.greaterThan(0), () => {
          colour.assign(vec4(shared, 0, 0, 1));
        }).Else(() => {
          colour.assign(vec4(0, shared, 0, 1));
        });
        return colour;
      })();
    };
    for (const compile of [compileJSFragment, compileWasmFragment]) {
      const stage = (compile as any)(build, { ...none });
      expect(stage({ uniforms: { [side.name]: 1 } })).toBeNull();
      expect(stage({ uniforms: { [side.name]: 0 } })).toBeNull();
    }
  });
});

describe("the operands of an operation that run statements", () => {
  const counter = instancedArray(1, "float");
  const count = () => counter.element(int(0));
  /** An inline `Fn` that adds one to `counter` and returns what it holds then. */
  const increment = () =>
    Fn(() => {
      count().addAssign(1);
      return count().toVar();
    })() as any;
  /** An inline `Fn` that multiplies `counter` by ten and returns a vector of what it holds then. */
  const scale = () =>
    Fn(() => {
      count().mulAssign(10);
      return vec3(count()).toVar();
    })() as any;
  const run = (compile: CompileCpuRoutine, build: () => any) => {
    const data = new Float64Array([1]);
    const result = compile(build, none)({ storages: { [counter.name]: data } });
    return { counter: data[0], result };
  };

  /**
   * @canon spec-the-operands-of-an-operation-run-in-the-order-it-takes-them
   */
  it.each(cpuTargets)("%s: runs a vector operand before the scalar operand after it", (_, compile) => {
    expect(run(compile, () => scale().add(increment()).x)).toEqual({ counter: 11, result: 21 });
  });

  /**
   * @canon spec-the-operands-of-an-operation-run-in-the-order-it-takes-them
   */
  it.each(cpuTargets)("%s: runs the vectors of mix before its scalar weight", (_, compile) => {
    expect(run(compile, () => mix(scale(), vec3(0, 0, 0), increment()).x)).toEqual({ counter: 11, result: -100 });
  });

  /**
   * @canon spec-the-operands-of-an-operation-run-in-the-order-it-takes-them
   */
  it.each(cpuTargets)("%s: runs the scalar condition of select before its vectors", (_, compile) => {
    expect(run(compile, () => select(increment().greaterThan(1), scale(), vec3(0, 0, 0)).x)).toEqual({
      counter: 20,
      result: 20,
    });
  });

  /**
   * @canon spec-the-operands-of-an-operation-run-in-the-order-it-takes-them
   */
  it.each(cpuTargets)("%s: runs the scalar condition of select before its scalars", (_, compile) => {
    const pick = () => select(increment().greaterThan(1), scale().x, float(0));
    expect(run(compile, pick)).toEqual({ counter: 20, result: 20 });
  });

  /**
   * @canon spec-the-operands-of-an-operation-run-in-the-order-it-takes-them
   */
  it.each(cpuTargets)("%s: runs the edge of step before its value", (_, compile) => {
    expect(run(compile, () => step(scale().x, increment()))).toEqual({ counter: 11, result: 1 });
  });
});

describe("a coordinate far past the edge on WASM", () => {
  /**
   * @canon spec-a-cpu-target-wraps-as-the-texture-asks
   */
  it("clamps a sampling coordinate far past the edge", () => {
    const tex = uniform("sampler2D");
    const routine = compileWasmRoutine((a: any) => Fn(() => texture(tex, vec2(a, 0.5)).x.toVar())(), {
      name: "main",
      params: [{ name: "a", type: "float" as const }],
    });
    const image = { data: [1, 2], width: 2, height: 1, channels: 1 as const };
    expect(routine({ params: { a: 1e12 }, textures: { [tex.name]: image } })).toBe(2);
  });

  /**
   * @canon spec-a-triangle-off-screen-draws-nothing
   */
  it("draws nothing for a triangle far off screen", () => {
    const { pos, routine } = flat();
    const far = new Float64Array([1e12, 0, 0, 2e12, 0, 0, 1e12, 1, 0]);
    expect(Array.from(routine.draw({ attributes: { [pos.name]: far } }, { width: 1, height: 1 }))).toEqual([
      0, 0, 0, 0,
    ]);
  });
});

describe("an integer texel on WASM", () => {
  /**
   * @canon spec-an-integer-texture-is-fetched-unfiltered
   */
  it("is read exactly, a uint above the largest float-exact one and an int at its maximum included", () => {
    const unsigned = uniform("usampler2D");
    const signed = uniform("isampler2D");
    const readUnsigned = compileWasmRoutine(
      () =>
        Fn(() =>
          textureLoad(unsigned as any, ivec2(0, 0))
            .x.toFloat()
            .toVar(),
        )(),
      none,
    );
    const readSigned = compileWasmRoutine(
      () =>
        Fn(() =>
          textureLoad(signed as any, ivec2(0, 0))
            .x.toFloat()
            .toVar(),
        )(),
      none,
    );
    const image = (value: number) => ({ data: [value], width: 1, height: 1, channels: 1 as const });
    expect(readUnsigned({ textures: { [unsigned.name]: image(4294967295) } })).toBe(4294967295);
    expect(readSigned({ textures: { [signed.name]: image(2147483647) } })).toBe(2147483647);
    expect(readSigned({ textures: { [signed.name]: image(-2147483648) } })).toBe(-2147483648);
  });
});

describe("a matrix column read on WASM", () => {
  /**
   * One read of a component of a column, at constant indices, adds one load
   * to the module: a second read beside the first costs that much.
   *
   * @canon spec-a-wasm-routine-reads-a-matrix-column-where-it-lies
   */
  it("reads a component of a column at constant indices with one load", async () => {
    const wabt = await wabtInit();
    const loads = (read: (m: any) => Node<"float">) => {
      const compiled = compileWasmFn(
        () =>
          Fn(() => {
            const m = mat3(1, 2, 3, 4, 5, 6, 7, 8, 9).toVar();
            return read(m);
          })(),
        { name: "main", params: [] },
      );
      return (
        wabt
          .readWasm(compiled.bytes, {})
          .toText({})
          .match(/f64\.load/g) ?? []
      ).length;
    };
    const once = loads((m) => m.element(int(2)).y);
    expect(loads((m) => m.element(int(2)).y.add(m.element(int(1)).z)) - once).toBe(1);
  });

  /**
   * @canon spec-a-wasm-routine-reads-a-matrix-column-where-it-lies
   */
  it("reads a column of a storage element from the buffer, and as zero past its end", () => {
    const matrices = instancedArray(1, "mat3");
    const out = instancedArray(2, "vec3");
    const root = Fn(() => {
      out.element(int(0)).assign(matrices.element(int(0)).element(int(1)));
      out.element(int(1)).assign(vec3(matrices.element(int(4)).element(int(1)).y, 0, 0));
    })();
    const adapter = createWasmCompute(root, { name: "step" });
    adapter.setAttribute(matrices.name, Float32Array.of(1, 2, 3, 4, 5, 6, 7, 8, 9));
    const data = new Float32Array(6);
    adapter.setAttribute(out.name, data);
    adapter.compute();
    expect(Array.from(data)).toEqual([4, 5, 6, 0, 0, 0]);
  });
});

describe("the fragments the WASM rasterizer draws", () => {
  /**
   * @canon spec-an-attribute-is-an-input-of-the-vertex-stage
   */
  it("passes an int attribute to the vertex stage on WASM", () => {
    const pos = attribute("vec3");
    const k = attribute("int");
    const shade = varying("float");
    const vertex = () =>
      Fn(() => {
        shade.assign(k.toFloat());
        builtinPosition().assign(vec4(pos, 1));
      })();
    const fragment = () => Fn(() => vec4(shade, 0, 0, 1))();
    const routine = compileWasm(vertex as any, fragment as any);
    const got = routine.draw(
      { attributes: { [pos.name]: screen(), [k.name]: Int32Array.of(5, 5, 5) } },
      { width: 1, height: 1 },
    );
    expect(Array.from(got)).toEqual([5, 0, 0, 1]);
  });

  /**
   * @canon spec-a-flat-varying-takes-the-first-vertex
   */
  it("passes an int varying from the vertex to the fragment stage on WASM", () => {
    const pos = attribute("vec3");
    const k = varying("int");
    const vertex = () =>
      Fn(() => {
        k.assign(int(5));
        builtinPosition().assign(vec4(pos, 1));
      })();
    const fragment = () => Fn(() => vec4(k.toFloat(), 0, 0, 1))();
    const routine = compileWasm(vertex as any, fragment as any);
    expect(Array.from(routine.draw({ attributes: { [pos.name]: screen() } }, { width: 1, height: 1 }))).toEqual([
      5, 0, 0, 1,
    ]);
  });

  /**
   * @canon spec-discard-drops-the-fragment
   */
  it("lets a discarded fragment leave the depth buffer as it was on WASM", () => {
    const pos = attribute("vec3");
    const drop = uniform("float");
    const color = uniform("vec4");
    const vertex = () => Fn(() => builtinPosition().assign(vec4(pos, 1)))();
    const fragment = () =>
      Fn(() => {
        If(drop.greaterThan(0.5), () => Discard());
        return color;
      })();
    const routine = compileWasm(vertex as any, fragment as any);
    const draw = (z: number, dropped: number, rgba: number[], clearDepth = false) =>
      routine.draw(
        { attributes: { [pos.name]: screen(z) }, uniforms: { [drop.name]: dropped, [color.name]: rgba } },
        { width: 1, height: 1, clear: true, clearDepth },
      );
    draw(0.25, 1, [1, 0, 0, 1], true);
    expect(Array.from(draw(0.5, 0, [0, 0, 1, 1]))).toEqual([0, 0, 1, 1]);
  });

  /**
   * @canon spec-discard-drops-the-fragment
   */
  it("leaves the pixel of a discarded fragment cleared on WASM", () => {
    const pos = attribute("vec3");
    const side = varying("float");
    const vertex = () =>
      Fn(() => {
        side.assign(pos.x);
        builtinPosition().assign(vec4(pos, 1));
      })();
    const fragment = () =>
      Fn(() => {
        If(side.greaterThan(0), () => Discard());
        return vec4(1, 0, 0, 1);
      })();
    const routine = compileWasm(vertex as any, fragment as any);
    const got = routine.draw({ attributes: { [pos.name]: screen() } }, { width: 2, height: 1, clear: true });
    expect(Array.from(got)).toEqual([1, 0, 0, 1, 0, 0, 0, 0]);
  });

  /**
   * @canon spec-a-rasterizer-keeps-the-closer-fragment
   */
  it("tests the depth the fragment stage writes on WASM", () => {
    const pos = attribute("vec3");
    const depth = uniform("float");
    const color = uniform("vec4");
    const vertex = () => Fn(() => builtinPosition().assign(vec4(pos, 1)))();
    const fragment = () =>
      Fn(() => {
        builtinFragDepth().assign(depth);
        return color;
      })();
    const routine = compileWasm(vertex as any, fragment as any);
    const draw = (z: number, d: number, rgba: number[], clearDepth = false) =>
      routine.draw(
        { attributes: { [pos.name]: screen(z) }, uniforms: { [depth.name]: d, [color.name]: rgba } },
        { width: 1, height: 1, clear: true, clearDepth },
      );
    // Each depth the fragment writes reverses the order its triangle's own depth gives.
    draw(0.25, 0.9, [1, 0, 0, 1], true);
    expect(Array.from(draw(0.5, 0, [0, 0, 1, 1]))).toEqual([0, 0, 1, 1]);
  });

  /**
   * @canon spec-the-wasm-rasterizer-draws-what-the-js-rasterizer-draws
   */
  it("gives each fragment the centre of its pixel as fragCoord on WASM", () => {
    const pos = attribute("vec3");
    const vertex = () => Fn(() => builtinPosition().assign(vec4(pos, 1)))();
    const fragment = () => Fn(() => vec4(fragCoord(), 0, 1))();
    const routine = compileWasm(vertex as any, fragment as any);
    const got = routine.draw({ attributes: { [pos.name]: screen() } }, { width: 2, height: 1 });
    expect(Array.from(got)).toEqual([0.5, 0.5, 0, 1, 1.5, 0.5, 0, 1]);
  });

  /**
   * @canon spec-a-pixel-on-a-shared-edge-is-shaded-once
   */
  it("gives a pixel on a shared edge to one triangle whatever their order on WASM", () => {
    const { pos, color, routine } = flat();
    const upper = new Float64Array([-1, 1, 0, 1, -1, 0, 1, 1, 0]);
    const lower = new Float64Array([-1, 1, 0, -1, -1, 0, 1, -1, 0]);
    const draw = (triangle: Float64Array, rgba: number[], clear: boolean) =>
      Array.from(
        routine.draw(
          { attributes: { [pos.name]: triangle }, uniforms: { [color.name]: rgba } },
          { width: 3, height: 3, clear, clearDepth: clear },
        ),
      );
    draw(upper, [1, 0, 0, 1], true);
    const upperFirst = draw(lower, [0, 0, 1, 1], false);
    draw(lower, [0, 0, 1, 1], true);
    const lowerFirst = draw(upper, [1, 0, 0, 1], false);
    expect(upperFirst).toEqual(lowerFirst);
  });
});

describe("the index accessors on WASM", () => {
  /** A triangle over the pixel only when its vertices read the indices 3, 4 and 5, coloured by its instance. */
  function indexed() {
    const pos = attribute("vec3");
    const instance = varying("float");
    const vertex = () =>
      Fn(() => {
        const v = vertexIndex();
        instance.assign(instanceIndex().toFloat().add(0.5));
        builtinPosition().assign(
          vec4(v.equal(uint(4)).select(float(3), float(-1)), v.equal(uint(5)).select(float(3), float(-1)), 0, 1),
        );
      })();
    const fragment = () => Fn(() => vec4(instance, 0, 0, 1))();
    const routine = compileWasm(vertex as any, fragment as any);
    return routine.draw(
      { attributes: { [pos.name]: new Float64Array(18) } },
      { width: 1, height: 1, first: 3, count: 3 },
    );
  }

  /**
   * @canon spec-a-cpu-vertex-stage-reads-the-vertex-it-runs-for
   */
  it("reads the vertex index counted from the start of the attributes on WASM", () => {
    expect(Array.from(indexed())[3]).toBe(1);
  });

  /**
   * @canon spec-a-cpu-vertex-stage-reads-instance-zero
   */
  it("reads instance 0 on WASM", () => {
    expect(Array.from(indexed())[0]).toBe(0.5);
  });

  /**
   * @canon spec-the-index-accessors-are-read-only
   */
  it("refuses a write to the vertex index on WASM", () => {
    const vertex = () =>
      Fn(() => {
        (vertexIndex() as any).assign(uint(1));
        builtinPosition().assign(vec4(0, 0, 0, 1));
      })();
    expect(() => compileWasm(vertex as any, (() => Fn(() => vec4(1))()) as any)).toThrow(/built-in input/);
  });
});

describe("the count of a draw that names none on WASM", () => {
  /**
   * The program reads only \`pos\`, so the 12 values of \`unused\`, passed first, count for nothing.
   *
   * @canon spec-a-rasterizer-takes-its-count-from-the-first-attribute-it-reads
   */
  it("counts from the first attribute the program reads on WASM", () => {
    const unused = attribute("vec2");
    const pos = attribute("vec3");
    const vertex = () => Fn(() => builtinPosition().assign(vec4(pos, 1)))();
    const fragment = () => Fn(() => vec4(0, 1, 0, 1))();
    const routine = compileWasm(vertex as any, fragment as any);
    const pixel = routine.draw(
      {
        attributes: { [unused.name]: new Float64Array(12), [pos.name]: Float64Array.of(-1, -1, 0, 3, -1, 0, -1, 3, 0) },
      },
      { width: 1, height: 1 },
    );
    expect(Array.from(pixel)).toEqual([0, 1, 0, 1]);
  });
});
