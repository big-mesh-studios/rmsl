import { describe, expect, it } from "vitest";
import {
  attribute,
  builtinPosition,
  Fn,
  fragCoord,
  float,
  instancedArray,
  invocationIndex,
  normalize,
  smoothstep,
  uniform,
  varying,
  vec3,
  vec4,
  type Node,
} from "./rmsl";
import { compileJS, compileJSCompute, compileJSGrid, compileJSRoutine, createJsCompute } from "./js";
import { compileWasm, compileWasmCompute, compileWasmGrid, compileWasmRoutine, createWasmCompute } from "./wasm";
import type { CompileCpuRoutine } from "./backends/cpu";

const f = Math.fround;
const routines: [string, CompileCpuRoutine][] = [
  ["JS", compileJSRoutine],
  ["WASM", compileWasmRoutine],
];
const two = {
  name: "main",
  params: [
    { name: "a", type: "float" as const },
    { name: "b", type: "float" as const },
  ],
};

describe("a CPU compile at float: f32", () => {
  /**
   * @canon spec-a-cpu-target-at-f32-rounds-every-float-value-it-computes
   */
  it.each(routines)("%s: gives a basic operation the 32-bit result", (_, compile) => {
    const ops = {
      add: [(a: Node<"float">, b: Node<"float">) => a.add(b), (a: number, b: number) => f(f(a) + f(b))],
      sub: [(a: Node<"float">, b: Node<"float">) => a.sub(b), (a: number, b: number) => f(f(a) - f(b))],
      mul: [(a: Node<"float">, b: Node<"float">) => a.mul(b), (a: number, b: number) => f(f(a) * f(b))],
      div: [(a: Node<"float">, b: Node<"float">) => a.div(b), (a: number, b: number) => f(f(a) / f(b))],
      sqrt: [(a: Node<"float">) => a.sqrt(), (a: number) => f(Math.sqrt(f(a)))],
    } as const;
    const pairs = [
      [0.1, 0.2],
      [1 / 3, 3],
      [123456.789, 0.000123],
      [16777217, 1],
      [2.5, -7.25],
    ];
    for (const [name, [build, expected]] of Object.entries(ops)) {
      const run = compile((a: any, b: any) => Fn(() => (build as any)(a, b).toVar())(), { ...two, float: "f32" });
      for (const [a, b] of pairs) {
        expect(run({ params: { a, b } }), `${name}(${a}, ${b})`).toBe((expected as any)(a, b));
      }
    }
  });

  /**
   * @canon spec-a-cpu-target-at-f32-rounds-every-float-value-it-computes
   */
  it.each(routines)("%s: folds a literal operation as it computes it at run time", (_, compile) => {
    const run = compile(() => Fn(() => float(0.1).add(0.2).toVar())(), { name: "main", params: [], float: "f32" });
    expect(run({})).toBe(f(f(0.1) + f(0.2)));
  });

  /**
   * @canon spec-a-cpu-target-at-f32-rounds-every-float-value-it-computes
   * @canon spec-wasm-and-js-give-the-same-float-bits
   */
  it("gives the same 32-bit bits on JS and WASM, built-in functions and inputs included", () => {
    const dir = uniform("vec3");
    const t = uniform("float");
    const build = () =>
      Fn(() => {
        const n = normalize(dir.add(vec3(0.1, 0.2, 0.3)));
        const s = smoothstep(0.1, 0.9, t).add(n.dot(vec3(1, 2, 3)).sin());
        return n
          .mul(s)
          .add(vec3(t, t.mul(t), 1 / 3))
          .toVar();
      })();
    const options = { name: "main", params: [], float: "f32" as const };
    const ctx = { uniforms: { [dir.name]: [0.3, -1.7, 2.9], [t.name]: 0.4 } };
    const js = compileJSRoutine(build as any, options)(ctx) as Float32Array;
    const wasm = compileWasmRoutine(build as any, options)(ctx) as Float32Array;
    expect(wasm).toEqual(js);
    for (const component of js) expect(f(component)).toBe(component);
  });

  /**
   * @canon spec-a-cpu-target-at-f32-rounds-every-float-value-it-computes
   */
  it.each([
    ["JS", compileJSCompute],
    ["WASM", compileWasmCompute],
  ] as const)("%s: reads a storage buffer the host passes as plain numbers at 32 bits", (_, compile) => {
    const buf = instancedArray(2, "vec2");
    const stage = compile(
      () =>
        Fn(() => {
          const i = invocationIndex();
          buf.element(i).assign(buf.element(i).mul(1));
        })() as any,
      { name: "step", params: [], float: "f32" },
    );
    const data = [0.1, 0.2, 1 / 3, 16777217];
    stage({ storages: { [buf.name]: data } }, 2);
    expect(data).toEqual([f(0.1), f(0.2), f(1 / 3), f(16777217)]);
  });

  /**
   * @canon spec-a-cpu-target-at-f32-rounds-every-float-value-it-computes
   */
  it.each([
    ["JS", compileJSCompute],
    ["WASM", compileWasmCompute],
  ] as const)("%s: leaves a storage element it does not write as the host gave it", (_, compile) => {
    const buf = instancedArray(2, "vec2");
    const stage = compile(
      () =>
        Fn(() => {
          buf.element(invocationIndex()).x.assign(float(0.2));
        })() as any,
      { name: "step", params: [], float: "f32" },
    );
    const data = Float64Array.of(0.1, 0.2, 0.3, 0.4);
    stage({ storages: { [buf.name]: data } }, 1);
    expect(Array.from(data)).toEqual([f(0.2), 0.2, 0.3, 0.4]);
  });

  /**
   * @canon spec-a-cpu-target-at-f32-rounds-every-float-value-it-computes
   * @canon spec-the-wasm-rasterizer-draws-what-the-js-rasterizer-draws
   */
  it("draws the same 32-bit pixels through the JS and the WASM rasterizer", async () => {
    const position = attribute("vec3");
    const colour = attribute("vec3");
    const shade = varying("vec3");
    const vertex = () =>
      Fn(() => {
        shade.assign(colour);
        builtinPosition().assign(vec4(position.x, position.y, position.z, 1));
      })();
    // Each varying arrives interpolated in 64 bits, and is read at 32 before it is multiplied.
    const fragment = () => Fn(() => vec4(shade.mul(3).x, shade.y.mul(7), shade.z, 1))();
    const attributes = {
      [position.name]: new Float64Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]),
      [colour.name]: new Float64Array([0.1, 0.2, 0.3, 0.7, 0.11, 0.13, 0.17, 0.19, 0.23]),
    };
    const draw = { width: 4, height: 4, clear: true, clearDepth: true };
    const js = compileJS(vertex, fragment, {
      attributeTypes: { [position.name]: "vec3", [colour.name]: "vec3" },
      float: "f32",
    }).draw({ attributes }, draw);
    const wasm = await compileWasm(vertex, fragment, { float: "f32" }).draw({ attributes }, draw);
    expect(Array.from(wasm)).toEqual(Array.from(js));
    for (const component of js) expect(f(component)).toBe(component);
  });

  /**
   * @canon spec-a-cpu-target-at-f32-rounds-every-float-value-it-computes
   * @canon spec-the-wasm-rasterizer-draws-what-the-js-rasterizer-draws
   */
  it("reads an attribute at 32 bits in the WASM rasterizer's vertex stage", async () => {
    const position = attribute("vec3");
    const colour = attribute("vec3");
    const shade = varying("vec3");
    // The vertex stage computes with the attribute, so an unrounded one changes the varying it writes.
    const vertex = () =>
      Fn(() => {
        shade.assign(colour.mul(3));
        builtinPosition().assign(vec4(position.x, position.y, position.z, 1));
      })();
    const fragment = () => Fn(() => vec4(shade, 1))();
    const attributes = {
      [position.name]: new Float64Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]),
      [colour.name]: new Float64Array([0.1, 0.2, 0.3, 0.7, 0.11, 0.13, 0.17, 0.19, 0.23]),
    };
    const draw = { width: 4, height: 4, clear: true, clearDepth: true };
    const js = compileJS(vertex, fragment, {
      attributeTypes: { [position.name]: "vec3", [colour.name]: "vec3" },
      float: "f32",
    }).draw({ attributes }, draw);
    const wasm = await compileWasm(vertex, fragment, { float: "f32" }).draw({ attributes }, draw);
    expect(Array.from(wasm)).toEqual(Array.from(js));
  });

  /**
   * @canon spec-a-cpu-compile-can-run-a-program-at-64-bit-precision
   */
  it.each([
    ["JS", createJsCompute],
    ["WASM", createWasmCompute],
  ] as const)("%s: an adapter takes float too", (_, create) => {
    const buf = instancedArray(1, "float");
    const t = uniform("float");
    const adapter = create(
      Fn(() => {
        buf.element(invocationIndex()).assign(t.add(0.2));
      })(),
      { name: "step", float: "f32" },
    );
    const data = new Float64Array(1);
    adapter.setAttribute(buf.name, data as any);
    adapter.setUniform(t, 0.1);
    adapter.compute();
    expect(data[0]).toBe(f(f(0.1) + f(0.2)));
  });
});

describe("a CPU grid at float: f32", () => {
  /**
   * @canon spec-a-grid-fills-a-float64-array-for-a-float-result
   */
  it("fills an out in the WASM module's own memory with the values the JS grid gives", () => {
    const memory = new WebAssembly.Memory({ initial: 2, maximum: 2, shared: true });
    const build = () => Fn(() => vec4(fragCoord().x.mul(0.1), 0.5, 0.25, 1))() as any;
    const grid = compileWasmGrid(build, { name: "main", params: [], memory, sharedMemory: true, float: "f32" });
    const offset = 65536;
    const out = new Float32Array(memory.buffer, offset, 8);
    const guard = new Float32Array(memory.buffer, offset + out.byteLength, 8).fill(7);
    expect(grid({}, 2, 1, out)).toBe(out);
    const js = compileJSGrid(build, { name: "main", params: [], float: "f32" })({}, 2, 1);
    expect(Array.from(out)).toEqual(Array.from(js));
    expect(Array.from(guard)).toEqual(new Array(8).fill(7));
  });

  /**
   * @canon spec-a-grid-fills-a-float64-array-for-a-float-result
   */
  it("refuses an out of another kind in the WASM module's own memory", () => {
    const memory = new WebAssembly.Memory({ initial: 2, maximum: 2, shared: true });
    const build = () => Fn(() => vec4(fragCoord().x, 0.5, 0.25, 1))() as any;
    const grid = compileWasmGrid(build, { name: "main", params: [], memory, sharedMemory: true, float: "f32" });
    const out = new Float64Array(memory.buffer, 65536, 8);
    expect(() => grid({}, 2, 1, out as any)).toThrow(/Float32Array/);
  });
});

describe("a CPU compile at the default width", () => {
  /**
   * @canon spec-a-cpu-compile-can-run-a-program-at-64-bit-precision
   */
  it.each(routines)("%s: computes a float in 64 bits", (_, compile) => {
    const run = compile((a: any, b: any) => Fn(() => a.add(b).toVar())(), two);
    expect(run({ params: { a: 0.1, b: 0.2 } })).toBe(0.1 + 0.2);
  });
});
