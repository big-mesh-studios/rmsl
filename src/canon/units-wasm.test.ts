import { describe, expect, it } from "vitest";
import {
  attribute,
  builtinPosition,
  float,
  Fn,
  If,
  instancedArray,
  int,
  invocationIndex,
  ivec2,
  storage,
  StorageBufferAttribute,
  textureLoad,
  uniform,
  varying,
  vec2,
  vec3,
  vec4,
  type Node,
} from "../rmsl";
import { compileJS, compileJSRoutine } from "../js";
import { compileWasm, compileWasmFn, compileWasmRoutine } from "../wasm";

const none = { name: "main", params: [] };

/** Both CPU targets, by name, as a compiler of a routine. */
const cpuTargets = [
  ["JS", compileJSRoutine],
  ["WASM", compileWasmRoutine],
] as const;

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
        return tex.texture(vec2(0.5, 0.5)).x;
      })();
    const routine = compile(build as any, none);
    const red = (texture: object) => routine.run({ textures: { [tex.name]: { ...texels, ...texture } } });
    expect(red({ minFilter: "linear" })).toBe(100);
    expect(red({ magFilter: "linear", minFilter: "nearest" })).toBe(50);
  });
});

describe("a CPU target's storage buffers", () => {
  /**
   * @canon spec-an-element-no-invocation-writes-keeps-the-host-value
   */
  it.each(cpuTargets)("keeps the elements no invocation writes on %s", (_, compile) => {
    const buffer = instancedArray(4, "float");
    const build = () =>
      Fn(() => {
        buffer.element(invocationIndex().mul(2)).assign(float(9));
      })();
    const data = new Float64Array([1, 2, 3, 4]);
    compile(build as any, none).compute({ storages: { [buffer.name]: data } }, 2);
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
    compileWasmRoutine(build as any, none).compute({ storages: { [storage(shared, "float").name]: data } }, 2);
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
    ["JS", () => compileJS(vertex as any, fragment as any, { attributeTypes: { [position.name]: "vec3" } })],
    ["WASM", () => compileWasm(vertex as any, fragment as any)],
  ] as const;

  /**
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
    expect(covered(clockwise).filter((alpha) => alpha === 1)).toHaveLength(6);
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

describe("an inline Fn whose value an operation reads more than once", () => {
  const counter = instancedArray(1, "float");
  /** An inline `Fn` that counts its runs in `counter` and returns `value`. */
  const counted = (value: number) =>
    Fn(() => {
      counter.element(int(0)).addAssign(1);
      return float(value);
    })() as any;
  const run = (compile: typeof compileJSRoutine, build: () => any) => {
    const data = new Float64Array(1);
    const result = compile(build, none).run({ storages: { [counter.name]: data } });
    // A WASM routine wraps a result in an object (#62), so read the value out of it.
    return { runs: data[0], result: typeof result === "object" && result !== null ? result.value : result };
  };

  /**
   * @canon spec-an-inline-fn-runs-once
   */
  it.each(cpuTargets)("%s: runs it once when fract reads its value twice", (_, compile) => {
    expect(run(compile, () => counted(2.75).fract())).toEqual({ runs: 1, result: 0.75 });
  });

  /**
   * @canon spec-an-inline-fn-runs-once
   */
  it.each(cpuTargets)("%s: runs it once when sign reads its value twice", (_, compile) => {
    expect(run(compile, () => counted(-3).sign())).toEqual({ runs: 1, result: -1 });
  });

  /**
   * @canon spec-an-inline-fn-runs-once
   */
  it.each(cpuTargets)("%s: runs it once when the integer abs reads its value three times", (_, compile) => {
    expect(run(compile, () => counted(-3).toInt().abs().toFloat())).toEqual({ runs: 1, result: 3 });
  });

  /**
   * @canon spec-an-inline-fn-runs-once
   */
  it.each(cpuTargets)(
    "%s: runs it once when a vector operation takes it as a scalar for every component",
    (_, compile) => {
      expect(run(compile, () => vec3(1, 2, 3).mul(counted(2)).z)).toEqual({ runs: 1, result: 6 });
    },
  );

  /**
   * @canon spec-an-inline-fn-runs-once
   */
  it.each(cpuTargets)("%s: runs it once when clamp takes it as a scalar bound", (_, compile) => {
    expect(run(compile, () => vec3(1, 2, 3).clamp(counted(1.5), 2.5).x)).toEqual({ runs: 1, result: 1.5 });
  });

  /**
   * @canon spec-an-inline-fn-runs-once
   */
  it("runs it in the branch that reads its value, and once there on WASM", () => {
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
      compileWasmRoutine(build, none).run({
        storages: { [counter.name]: data, [result.name]: out },
        uniforms: { [taken.name]: branch },
      });
      return { runs: data[0], value: out[0] };
    };
    expect(read(1)).toEqual({ runs: 1, value: 2 });
    expect(read(0)).toEqual({ runs: 1, value: 20 });
  });

  /**
   * @canon spec-an-inline-fn-runs-once
   */
  it.each(cpuTargets)("%s: runs it once when mix takes it as the weight of every component", (_, compile) => {
    expect(run(compile, () => vec3(0, 0, 0).mix(vec3(4, 4, 4), counted(0.5)).z)).toEqual({ runs: 1, result: 2 });
  });
  /**
   * @canon spec-a-cpu-routine-answers-one-fragment-per-call
   */
  it("keeps the pixels a WASM routine drew when it draws again", () => {
    const routine = compileWasmRoutine((a: any) => Fn(() => vec2(a, a.add(1)).toVar())(), {
      name: "main",
      params: [{ name: "a", type: "float" }],
    });
    const first = routine.draw({ params: { a: 0.5 } }, 1, 1);
    routine.draw({ params: { a: 100.5 } }, 1, 1);
    expect(Array.from(first)).toEqual([0.5, 1.5]);
  });
});
