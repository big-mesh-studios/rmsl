import { describe, expect, it } from "vitest";
import {
  attribute,
  builtinPosition,
  float,
  Fn,
  instancedArray,
  invocationIndex,
  storage,
  StorageBufferAttribute,
  uniform,
  vec2,
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
