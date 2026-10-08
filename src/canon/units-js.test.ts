import { Session } from "node:inspector/promises";
import { setFlagsFromString } from "node:v8";
import { runInNewContext } from "node:vm";
import { afterAll, afterEach, describe, expect, it } from "vitest";
import {
  attribute,
  bool,
  builtinPosition,
  bvec2,
  Discard,
  Fn,
  float,
  If,
  int,
  Loop,
  fragCoord,
  instancedArray,
  invocationIndex,
  ivec2,
  mat2,
  mix,
  normalize,
  select,
  smoothstep,
  outputStruct,
  Return,
  uint,
  textureLoad,
  uniform,
  varying,
  vec2,
  vec3,
  vec4,
  type Node,
  texture,
} from "../rmsl";
import type { CompileCpuRoutine } from "../backends/cpu";
import { jsHelperSource } from "../backends/js/js";
import { roundHalfToEven } from "../backends/shared";
import {
  compileJS,
  compileJSCompute,
  compileJSFn,
  compileJSFragment,
  compileJSGrid,
  compileJSRoutine,
  compileJSVertex,
  createJsCompute,
  createJsGrid,
  createJsRoutine,
} from "../js";
import {
  compileWasm,
  compileWasmCompute,
  compileWasmFragment,
  compileWasmGrid,
  compileWasmRoutine,
  compileWasmVertex,
  createWasmCompute,
  createWasmGrid,
  createWasmRoutine,
} from "../wasm";
import {
  assertEvaluationsOfTheTestAgree,
  assertRecordedEvaluationsAgree,
  closeEvaluators,
  evaluateRecording,
} from "../testing/shader-eval";
import type { ComputeStage, FragmentStage, VertexStage } from "../backends/cpu";

// Each test's programs are compared after it, so a disagreement fails the test that made the program.
afterEach(assertEvaluationsOfTheTestAgree, 120_000);

afterAll(async () => {
  profiler?.disconnect();
  await assertRecordedEvaluationsAgree();
  await closeEvaluators();
}, 120_000);

const none = { name: "main", params: [] };

/** One inspector session for the file: connecting one deoptimizes the code it finds running. */
let profiler: Session | undefined;

/**
 * The bytes `run` allocates a call, over `runs` calls after `warm` calls, by
 * V8's sampling heap profiler, collected objects included. Only the library's
 * code counts: a compiled program has no source URL, and the rest of the
 * library lies under `src/backends/`. It measures twice and keeps the smaller:
 * another worker's load only adds to a measurement, and an allocation the
 * calls make shows in both.
 */
async function allocatedBy(run: () => unknown, warm: number, runs: number): Promise<number> {
  if (!profiler) {
    profiler = new Session();
    profiler.connect();
    await profiler.post("HeapProfiler.enable");
  }
  for (let k = 0; k < warm; k++) run();
  let least = Infinity;
  for (let round = 0; round < 2; round++) {
    await profiler.post("HeapProfiler.startSampling", {
      samplingInterval: 128,
      includeObjectsCollectedByMajorGC: true,
      includeObjectsCollectedByMinorGC: true,
    });
    for (let k = 0; k < runs; k++) run();
    const { profile } = await profiler.post("HeapProfiler.stopSampling");
    let allocated = 0;
    const walk = (node: any): void => {
      const url: string = node.callFrame.url;
      if (url === "" || url.includes("/src/backends/")) allocated += node.selfSize;
      for (const child of node.children) walk(child);
    };
    walk(profile.head);
    least = Math.min(least, allocated / runs);
  }
  return least;
}

const cpuTargets: [string, CompileCpuRoutine][] = [
  ["JS", compileJSRoutine],
  ["WASM", compileWasmRoutine as CompileCpuRoutine],
];

const grids = [
  ["JS", compileJSGrid],
  ["WASM", compileWasmGrid],
] as const;

const rasterizers: [string, typeof compileJS][] = [
  ["JS", compileJS],
  ["WASM", compileWasm as unknown as typeof compileJS],
];

const routineAdapters: [string, typeof createJsGrid][] = [
  ["JS", createJsGrid],
  ["WASM", createWasmGrid as unknown as typeof createJsGrid],
];

/** The bytes a routine adapter puts on a one-pixel canvas for `draw`. */
function shownOnCanvas(create: typeof createJsGrid, draw: Node<any>): number[] {
  const hadImageData = "ImageData" in globalThis;
  if (!hadImageData) {
    (globalThis as any).ImageData = class {
      data: Uint8ClampedArray;
      constructor(
        public width: number,
        public height: number,
      ) {
        this.data = new Uint8ClampedArray(width * height * 4);
      }
    };
  }
  try {
    let shown!: ImageData;
    const canvas = { width: 1, height: 1, getContext: () => ({ putImageData: (image: ImageData) => (shown = image) }) };
    const adapter = create({ draw, name: "shade" });
    adapter.attach(canvas as unknown as HTMLCanvasElement);
    adapter.draw();
    return Array.from(shown.data);
  } finally {
    if (!hadImageData) delete (globalThis as any).ImageData;
  }
}

describe("the JS target's internal decisions, on every target they claim", () => {
  /**
   * @canon spec-a-cpu-routine-returns-its-value
   */
  it.each(cpuTargets)("%s: returns a vector result as a bare array", (_, compile) => {
    const routine = compile(() => Fn(() => vec3(1, 2, 3))() as any, none);
    expect(routine({})).toEqual(new Float64Array([1, 2, 3]));
  });

  /**
   * `mat2(1, 2, 3, 4)` has the columns (1, 2) and (3, 4).
   *
   * @canon spec-a-cpu-routine-returns-a-matrix-as-its-columns-in-one-array
   */
  it.each(cpuTargets)("%s: returns a matrix as its columns in one flat array", (_, compile) => {
    const routine = compile(() => Fn(() => mat2(vec2(1, 2), vec2(3, 4)))() as any, none);
    expect(routine({})).toEqual(new Float64Array([1, 2, 3, 4]));
  });

  /**
   * @canon spec-a-vector-converted-to-a-scalar-takes-its-first-component
   */
  it("takes the first component of a vector converted to a scalar on every target", () => {
    expect(evaluateRecording((a, b) => vec3(a, b, b).toFloat(), [4.5, 9])).toBe(4.5);
    expect(evaluateRecording((a, b) => vec3(a, b, b).toInt().toFloat(), [2.7, 9])).toBe(2);
    expect(evaluateRecording((a, b) => vec2(a, b).toUint().toFloat(), [3.9, 9])).toBe(3);
  });

  /**
   * A builder that declares its own uniform names a slot each time it runs,
   * so the count of runs shows on the slot the routine reads.
   *
   * @canon spec-a-cpu-compiler-calls-its-builder-once
   */
  it.each(grids)("%s: calls the builder it is given once", (_, compile) => {
    let calls = 0;
    let scale!: Node<"float">;
    const grid = compile(() => {
      calls++;
      scale = uniform("float");
      return Fn(() => fragCoord().x.mul(scale))();
    }, none);
    expect(calls).toBe(1);
    expect(Array.from(grid({ uniforms: { [(scale as any).name]: 2 } }, 1, 1))).toEqual([1]);
  });

  /**
   * The sample at u = 0.5 falls on the border of two texels. A nearest filter
   * takes the second, 20, and a linear one blends them to 15.
   *
   * @canon spec-a-cpu-target-filters-by-the-magnification-filter-alone
   */
  it.each(cpuTargets)(
    "%s: filters by the magnification filter, whatever the minification filter asks",
    (_, compile) => {
      let tex!: any;
      const program = Fn(() => {
        tex = uniform("sampler2D");
        return texture(tex, vec2(0.5, 0.5));
      })();
      const routine = compile(() => program, none);
      const texels = { data: new Float32Array([10, 10, 10, 10, 20, 20, 20, 20]), width: 2, height: 1 };
      const red = (filters: { magFilter: "nearest" | "linear"; minFilter: "nearest" | "linear" }) =>
        (routine({ textures: { [tex.name]: { ...texels, ...filters } } }) as Float64Array)[0];
      expect(red({ magFilter: "nearest", minFilter: "linear" })).toBe(20);
      expect(red({ magFilter: "linear", minFilter: "nearest" })).toBe(15);
    },
  );

  /**
   * @canon spec-a-grid-fills-a-float64-array-for-a-float-result
   */
  it.each(grids)("%s: fills a float result into a Float64Array", (_, compile) => {
    const out = compile(() => Fn(() => fragCoord().x.add(0.25))(), none)({}, 2, 1);
    expect(out).toBeInstanceOf(Float64Array);
    expect(Array.from(out)).toEqual([0.75, 1.75]);
  });

  /**
   * @canon spec-a-grid-fills-an-int32-array-for-an-int-result
   */
  it.each(grids)("%s: fills an int result into an Int32Array", (_, compile) => {
    const out = compile(() => Fn(() => fragCoord().x.toInt().sub(2))(), none)({}, 2, 1);
    expect(out).toBeInstanceOf(Int32Array);
    expect(Array.from(out)).toEqual([-2, -1]);
  });

  /**
   * 3000000000 lies past the largest `int`, so only an unsigned array holds it.
   *
   * @canon spec-a-grid-fills-a-uint32-array-for-a-uint-result
   */
  it.each(grids)("%s: fills a uint result into a Uint32Array", (_, compile) => {
    const out = compile(() => Fn(() => fragCoord().x.toUint().add(uint(3000000000)))(), none)({}, 2, 1);
    expect(out).toBeInstanceOf(Uint32Array);
    expect(Array.from(out)).toEqual([3000000000, 3000000001]);
  });

  /**
   * @canon spec-a-grid-writes-a-bool-result-as-one-or-zero-in-an-int32-array
   */
  it.each(grids)("%s: fills a bool result as 1 or 0 into an Int32Array", (_, compile) => {
    const out = compile(() => Fn(() => fragCoord().x.greaterThan(1))(), none)({}, 2, 1);
    expect(out).toBeInstanceOf(Int32Array);
    expect(Array.from(out)).toEqual([0, 1]);
  });

  /**
   * The same triangle covers the one pixel, its vertices given once
   * counter-clockwise and once clockwise.
   *
   * @canon spec-a-cpu-rasterizer-draws-a-triangle-whichever-way-it-winds
   */
  it.each(rasterizers)("%s: draws a triangle whichever way its vertices wind", (_, compileRaster) => {
    const pos = attribute("vec3");
    const routine = compileRaster(
      () => Fn(() => builtinPosition().assign(vec4(pos, 1)))() as any,
      () => Fn(() => vec4(1, 0, 0, 1))() as any,
      { attributeTypes: { [pos.name]: "vec3" } },
    );
    const draw = (triangle: number[]) =>
      Array.from(
        routine.draw(
          { attributes: { [pos.name]: new Float64Array(triangle) } },
          { width: 1, height: 1, clear: true, clearDepth: true },
        ),
      );
    expect(draw([-1, -1, 0, 3, -1, 0, -1, 3, 0])).toEqual([1, 0, 0, 1]);
    expect(draw([-1, -1, 0, -1, 3, 0, 3, -1, 0])).toEqual([1, 0, 0, 1]);
  });

  /**
   * 0.5 rounds to 128, 1.2 clamps to 255, -0.1 clamps to 0 and 0.25 rounds
   * to 64.
   *
   * @canon spec-a-cpu-adapter-writes-a-channel-as-a-rounded-clamped-byte
   */
  it.each(routineAdapters)("%s: writes each channel on the canvas as a rounded byte, clamped", (_, create) => {
    expect(shownOnCanvas(create, Fn(() => vec4(1.2, -0.1, 0.25, 0.5))())).toEqual([255, 0, 64, 128]);
  });
});

describe("what the JS target emits for an operand a constructor reads several times", () => {
  const param = { name: "main", params: [{ name: "a", type: "float" as const }] };

  /**
   * @canon spec-every-node-is-emitted-once
   */
  it("emits a vector operand of a constructor once", () => {
    const source = compileJSFn(
      (a: any) =>
        Fn(() => {
          const v = vec3(a, a, a).toVar();
          return vec4(v.add(v), 1).x;
        })(),
      param,
    );
    // The sum is written out once: one line writes its first component.
    expect(source.match(/\[0\] = (_rmsl_\w+)\[0\] \+ \1\[0\];/g)).toHaveLength(1);
  });

  /**
   * @canon spec-every-node-is-emitted-once
   */
  it("emits a scalar operand of a broadcast constructor once", () => {
    const source = compileJSFn((a: any) => Fn(() => vec3(a.sin().mul(2)).x)(), param);
    expect(source.match(/Math\.sin\(/g)?.length).toBe(1);
  });

  /**
   * @canon spec-every-node-is-emitted-once
   */
  it("emits a column operand of a matrix constructor once", () => {
    const source = compileJSFn(
      (a: any) =>
        Fn(() => {
          const column = vec2(a, a).toVar();
          return mat2(column.add(column), column).element(0).x;
        })(),
      param,
    );
    // The sum is written out once: one line writes its first component.
    expect(source.match(/\[0\] = (_rmsl_\w+)\[0\] \+ \1\[0\];/g)).toHaveLength(1);
  });
});

describe("a JS routine's results", () => {
  const param = { name: "main", params: [{ name: "a", type: "float" as const }] };

  /**
   * @canon spec-a-cpu-routine-answers-one-fragment-per-call
   */
  it("keeps the value a JS routine returned when it is called again", () => {
    const run = compileJSRoutine((a: any) => Fn(() => vec3(a, a, a).toVar())(), param);
    const first = run({ params: { a: 1 } });
    run({ params: { a: 2 } });
    expect(first).toEqual(new Float64Array([1, 1, 1]));
  });

  /**
   * @canon spec-a-rasterizer-gives-each-vertex-its-own-position
   */
  it.each(rasterizers)(
    "rasterizes each vertex at its own position when the position is a variable on %s",
    (_, compile) => {
      const position = attribute("vec3");
      const routine = compile(
        () =>
          Fn(() => {
            const p = vec4(position.x, position.y, position.z, 1).toVar();
            builtinPosition().assign(p);
          })() as any,
        () => Fn(() => vec4(1, 1, 1, 1).toVar())() as any,
        { attributeTypes: { [position.name]: "vec3" } },
      );
      const image = routine.draw(
        { attributes: { [position.name]: new Float64Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]) } },
        { width: 2, height: 2, clear: true, clearDepth: true },
      );
      expect(Array.from(image)).toEqual(new Array(16).fill(1));
    },
  );

  /**
   * @canon spec-a-cpu-routine-answers-one-fragment-per-call
   */
  it.each([false, true])("returns a copy of an array the caller passed in, with reentrant %s", (reentrant) => {
    const input = uniform("vec3");
    const routine = compileJSRoutine(() => Fn(() => input.add(0).toVar())(), { ...none, reentrant });
    const passed = [1, 2, 3];
    const first = routine({ uniforms: { [input.name]: passed } });
    (first as Float64Array)[0] = 9;
    expect(passed).toEqual([1, 2, 3]);
    expect(routine({ uniforms: { [input.name]: passed } })).toEqual(new Float64Array([1, 2, 3]));
  });
});

/** A JS rasterizer drawing one flat-coloured triangle list, its colour a uniform. */
function flatRasterizer(fragment?: (color: Node<"vec4">, drop: Node<"float">) => Node<"vec4">) {
  const position = attribute("vec3");
  const color = uniform("vec4");
  const drop = uniform("float");
  const routine = compileJS(
    () => Fn(() => builtinPosition().assign(vec4(position.x, position.y, position.z, 1)))() as any,
    () => Fn(() => (fragment ? fragment(color, drop) : color).toVar())() as any,
    { attributeTypes: { [position.name]: "vec3" } },
  );
  const draw = (triangles: number[], rgba: number[], options: Record<string, unknown> = {}, dropped = 0) =>
    routine.draw(
      {
        attributes: { [position.name]: new Float64Array(triangles) },
        uniforms: { [color.name]: rgba, [drop.name]: dropped },
      },
      { width: 2, height: 2, ...options },
    );
  return draw;
}

const screenAt = (z: number) => [-1, -1, z, 3, -1, z, -1, 3, z];

describe("a varying a CPU vertex stage does not write", () => {
  /**
   * @canon spec-a-cpu-vertex-stage-gives-zero-for-a-varying-a-call-does-not-write
   */
  it("is 0 in what the call returns, whatever the call before wrote", () => {
    const shade = varying("float");
    const pair = varying("vec2");
    const write = uniform("int");
    const build = () =>
      Fn(() => {
        If(write.equal(int(1)), () => {
          shade.assign(float(0.25));
          pair.assign(vec2(1, 2));
        });
        builtinPosition().assign(vec4(0, 0, 0, 1));
      })() as any;
    for (const compile of [compileJSVertex, compileWasmVertex]) {
      const stage = (compile as any)(build, { name: "main", params: [] });
      const written = stage({ uniforms: { [write.name]: 1 } });
      expect(written.varyings[shade.name]).toBe(0.25);
      expect(Array.from(written.varyings[pair.name])).toEqual([1, 2]);
      const skipped = stage({ uniforms: { [write.name]: 0 } });
      expect(skipped.varyings[shade.name]).toBe(0);
      expect(Array.from(skipped.varyings[pair.name])).toEqual([0, 0]);
    }
  });

  /**
   * @canon spec-a-cpu-vertex-stage-gives-zero-for-a-varying-a-call-does-not-write
   */
  it("clears on JS only a varying a call may leave unwritten", () => {
    const always = varying("vec3");
    const sometimes = varying("vec2");
    const write = uniform("int");
    const source = compileJSFn(
      () =>
        Fn(() => {
          always.assign(vec3(1, 2, 3));
          If(write.equal(int(1)), () => {
            sometimes.assign(vec2(1, 2));
          });
          builtinPosition().assign(vec4(0, 0, 0, 1));
        })(),
      { name: "main", params: [], stage: "vertex" },
    );
    expect(source.match(/\.fill\(0\)/g)).toHaveLength(1);
    expect(source).toContain(`res.varyings["${sometimes.name}"] = `);
  });

  /**
   * @canon spec-a-cpu-vertex-stage-gives-zero-for-a-varying-a-call-does-not-write
   */
  it("is 0 when the call returns before the statement that writes it", () => {
    const shade = varying("float");
    const pair = varying("vec2");
    const stop = uniform("int");
    const build = () =>
      Fn(() => {
        builtinPosition().assign(vec4(0, 0, 0, 1));
        If(stop.equal(int(1)), () => {
          Return();
        });
        shade.assign(float(0.25));
        pair.assign(vec2(1, 2));
      })() as any;
    const source = compileJSFn(build, { name: "main", params: [], stage: "vertex" });
    expect(source).toContain(`res.varyings["${shade.name}"] = 0;`);
    expect(source.match(/\.fill\(0\)/g)).toHaveLength(1);
    const stage = (compileWasmVertex as any)(build, { name: "main", params: [] });
    expect(stage({ uniforms: { [stop.name]: 0 } }).varyings[shade.name]).toBe(0.25);
    const returned = stage({ uniforms: { [stop.name]: 1 } });
    expect(returned.varyings[shade.name]).toBe(0);
    expect(Array.from(returned.varyings[pair.name])).toEqual([0, 0]);
  });

  /**
   * @canon spec-a-cpu-vertex-stage-gives-zero-for-a-varying-a-call-does-not-write
   */
  it("interpolates a bool varying as 1 or 0 on the JS rasterizer", () => {
    const position = attribute("vec3");
    const flag = attribute("float");
    const hit = varying("bool");
    const raster = compileJS(
      () =>
        Fn(() => {
          If(flag.greaterThan(0.5), () => {
            hit.assign(bool(true));
          });
          builtinPosition().assign(vec4(position, 1));
        })() as any,
      () => Fn(() => vec4(select(hit, float(1), float(0)), 0, 0, 1))() as any,
      { attributeTypes: { [position.name]: "vec3", [flag.name]: "float" } },
    );
    const ctx = {
      attributes: {
        [position.name]: Float64Array.of(-1, -1, 0, 3, -1, 0, -1, 3, 0),
        [flag.name]: Float64Array.of(1, 1, 1),
      },
    };
    // Every vertex writes true, so every fragment of the triangle, which covers the target, reads it.
    const red = Array.from(raster.draw(ctx, { width: 4, height: 4 })).filter((_, i) => i % 4 === 0);
    expect(red).toEqual(Array(16).fill(1));
  });

  /**
   * @canon spec-a-cpu-vertex-stage-gives-zero-for-a-varying-a-call-does-not-write
   */
  it("is 0 for the vertices of a draw that do not write it, on both rasterizers", () => {
    const position = attribute("vec3");
    const flag = attribute("float");
    const shade = varying("float");
    const vertex = () =>
      Fn(() => {
        If(flag.greaterThan(0.5), () => {
          shade.assign(float(1));
        });
        builtinPosition().assign(vec4(position, 1));
      })() as any;
    const fragment = () => Fn(() => vec4(shade, 0, 0, 1))() as any;
    const ctx = {
      attributes: {
        [position.name]: Float64Array.of(-1, -1, 0, 3, -1, 0, -1, 3, 0),
        [flag.name]: Float64Array.of(1, 0, 0),
      },
    };
    const options = { width: 4, height: 4 };
    const js = compileJS(vertex, fragment, { attributeTypes: { [position.name]: "vec3", [flag.name]: "float" } });
    const wasm = compileWasm(vertex, fragment);
    const drawn = Array.from(js.draw(ctx, options));
    // Only the first vertex writes the varying, so it fades from 1 there to 0 at the others.
    expect(drawn.filter((_, i) => i % 4 === 0).some((red) => red < 1)).toBe(true);
    expect(Array.from(wasm.draw(ctx, options))).toEqual(drawn);
  });
});

describe("the varyings a JS rasterizer interpolates", () => {
  /**
   * @canon spec-a-js-rasterizer-interpolates-every-varying-the-vertex-stage-writes
   */
  it("interpolates a varying the first vertex does not write, as 0 at that vertex", () => {
    const position = attribute("vec3");
    const shade = varying("float");
    const raster = compileJS(
      () =>
        Fn(() => {
          If(position.x.greaterThan(0), () => {
            shade.assign(float(1));
          });
          builtinPosition().assign(vec4(position, 1));
        })() as any,
      () => Fn(() => vec4(shade, 0, 0, 1))() as any,
      { attributeTypes: { [position.name]: "vec3" } },
    );
    const ctx = { attributes: { [position.name]: Float64Array.of(-1, -1, 0, 3, -1, 0, -1, 3, 0) } };
    for (let draw = 0; draw < 2; draw++) {
      const red = Array.from(raster.draw(ctx, { width: 4, height: 4 })).filter((_, i) => i % 4 === 0);
      // The varying grows from 0 at the left vertices to 1 at the right one.
      expect(red.every((value) => value >= 0 && value <= 1)).toBe(true);
      expect(red[3]).toBeGreaterThan(red[0]!);
    }
  });
});

describe("a JS rasterizer's discarded fragment", () => {
  /**
   * @canon spec-break-continue-return-and-discard-leave-where-tsl-leaves
   */
  it("leaves the colour under a discarded fragment as it was", () => {
    const draw = flatRasterizer((color, drop) => {
      If(drop.greaterThan(0), () => Discard());
      return color;
    });
    const composes = { clear: false, clearDepth: false };
    draw(screenAt(0.5), [1, 0, 0, 1]);
    expect(Array.from(draw(screenAt(0.25), [0, 1, 0, 1], composes, 1).slice(0, 4))).toEqual([1, 0, 0, 1]);
  });
});

/** A compile function of either CPU target for a stage, as these tests call it. */
type CompileStage<T> = (fn: (...args: any[]) => any, options: { name: string; params: never[] }) => T;
const stages: [string, CompileStage<VertexStage>, CompileStage<FragmentStage<any>>, CompileStage<ComputeStage>][] = [
  ["JS", compileJSVertex, compileJSFragment, compileJSCompute],
  ["WASM", compileWasmVertex, compileWasmFragment, compileWasmCompute],
];

describe("a CPU stage's result", () => {
  /**
   * @canon spec-a-vertex-stage-returns-its-position-and-varyings
   */
  it.each(stages)("returns the position and the varyings a vertex stage writes on %s", (_, compileVertex) => {
    const place = attribute("vec3");
    const tint = varying("vec2");
    const stage = compileVertex(
      () =>
        Fn(() => {
          tint.assign(vec2(place.x, place.y));
          builtinPosition().assign(vec4(place, 1));
        })(),
      none,
    );
    const result = stage({ attributes: { [place.name]: [1, 2, 3] } });
    expect(result.position).toEqual(new Float64Array([1, 2, 3, 1]));
    expect(Object.values(result.varyings)).toEqual([new Float64Array([1, 2])]);
  });

  /**
   * @canon spec-a-vertex-stage-returns-its-position-and-varyings
   */
  it.each(stages)(
    "returns the vec4 a vertex stage returns as its position, with no varyings on %s",
    (_, compileVertex) => {
      const place = attribute("vec3");
      const stage = compileVertex(() => Fn(() => vec4(place, 1))(), none);
      expect(stage({ attributes: { [place.name]: [1, 2, 3] } })).toEqual({
        position: new Float64Array([1, 2, 3, 1]),
        varyings: {},
      });
    },
  );

  /**
   * @canon spec-a-fragment-stage-returns-its-colour-and-outputs
   */
  it.each(stages)(
    "returns the colour of a fragment stage as a vec4, a vec3 with an opaque alpha on %s",
    (_, __, compileFragment) => {
      expect(compileFragment(() => Fn(() => vec4(1, 2, 3, 4))(), none)({})).toEqual({
        value: new Float64Array([1, 2, 3, 4]),
        outputs: [],
      });
      expect(compileFragment(() => Fn(() => vec3(1, 2, 3))(), none)({})).toEqual({
        value: new Float64Array([1, 2, 3, 1]),
        outputs: [],
      });
    },
  );

  /**
   * @canon spec-an-output-struct-writes-each-member-at-its-position
   */
  it.each(stages)(
    "returns the members of an outputStruct by position, with no colour on %s",
    (_, __, compileFragment) => {
      const stage = compileFragment(() => Fn(() => outputStruct(float(7), vec3(1, 2, 3)))(), none);
      expect(stage({})).toEqual({ value: undefined, outputs: [7, new Float64Array([1, 2, 3])] });
    },
  );

  /**
   * @canon spec-a-fragment-stage-returns-its-colour-and-outputs
   */
  it.each(stages)("returns null for a fragment that discards on %s", (_, __, compileFragment) => {
    const stage = compileFragment(
      () =>
        Fn(() => {
          Discard();
          return vec4(1, 2, 3, 4);
        })(),
      none,
    );
    expect(stage({})).toBeNull();
  });

  /**
   * @canon spec-a-compute-stage-dispatches-and-returns-nothing
   */
  it.each(stages)(
    "dispatches a compute stage over its indices and names the storage it reads on %s",
    (_, __, ___, compileCompute) => {
      const buffer = instancedArray(4, "float");
      const stage = compileCompute(
        () =>
          Fn(() => {
            buffer.element(invocationIndex().mul(2)).assign(float(9));
          })(),
        none,
      );
      const data = new Float64Array([1, 2, 3, 4]);
      expect(stage({ storages: { [buffer.name]: data } }, 2)).toBeUndefined();
      expect(Array.from(data)).toEqual([9, 2, 9, 4]);
      expect(stage.storageTypes[buffer.name]).toBe("float");
    },
  );

  /**
   * @canon spec-a-cpu-grid-evaluates-a-fragment-for-each-pixel
   */
  it.each(grids)("%s: evaluates every pixel of a grid at the centre of the pixel", (_, compile) => {
    const grid = compile(() => Fn(() => vec4(fragCoord().x, fragCoord().y, 0, 1))(), none);
    expect(Array.from(grid({}, 2, 2))).toEqual([0.5, 0.5, 0, 1, 1.5, 0.5, 0, 1, 0.5, 1.5, 0, 1, 1.5, 1.5, 0, 1]);
  });

  /**
   * @canon spec-a-grid-writes-a-discarded-pixel-as-zero
   */
  it.each(grids)("%s: writes a pixel that discards as zero in every channel", (_, compile) => {
    const grid = compile(
      () =>
        Fn(() => {
          If(fragCoord().x.greaterThan(1), () => {
            Discard();
          });
          return vec4(1, 2, 3, 4);
        })(),
      none,
    );
    const out = new Float64Array(8).fill(9);
    grid({}, 2, 1, out);
    expect(Array.from(out)).toEqual([1, 2, 3, 4, 0, 0, 0, 0]);
  });
});

describe("an adapter of a routine", () => {
  const routineAdapters = [
    ["JS", createJsRoutine],
    ["WASM", createWasmRoutine],
  ] as const;

  /**
   * @canon spec-a-cpu-routine-adapter-calls-its-routine-with-what-it-was-given
   */
  it.each(routineAdapters)(
    "%s: calls the routine with the uniform the host set, and again with the next",
    (_, create) => {
      const gain = uniform("float");
      const adapter = create(Fn(() => gain.mul(2).add(1))(), none);
      adapter.setUniform(gain, 3);
      expect(adapter.run()).toBe(7);
      adapter.setUniform(gain, 10);
      expect(adapter.run()).toBe(21);
      adapter.destroy();
    },
  );

  /**
   * @canon spec-a-cpu-routine-adapter-calls-its-routine-with-what-it-was-given
   */
  it.each(routineAdapters)("%s: calls the routine with the texture the host set", (_, create) => {
    const tex = uniform("sampler2D");
    const adapter = create(Fn(() => textureLoad(tex, ivec2(1, 0)).x)(), none);
    adapter.setTexture(tex, { data: [10, 99], width: 2, height: 1, channels: 1 });
    expect(adapter.run()).toBe(99);
  });
});

describe("what a JS routine allocates per call", () => {
  const param = { name: "main", params: [{ name: "a", type: "float" as const }] };

  /**
   * @canon spec-a-js-routine-allocates-nothing-per-call
   */
  it("computes a scalar smoothstep without a closure on JS", () => {
    const source = compileJSFn((a: any) => Fn(() => smoothstep(0, 1, a).toVar())(), param);
    expect(source).not.toMatch(/function\s*\(t\)/);
  });
  /**
   * @canon spec-a-js-routine-allocates-nothing-per-call
   */
  it("writes a component-wise select into an output argument on JS", () => {
    const source = compileJSFn(
      (a: any) => Fn(() => select(vec3(a, 1, -1).greaterThan(vec3(0, 0, 0)), vec3(1, 2, 3), vec3(4, 5, 6)).toVar())(),
      param,
    );
    expect(source).not.toContain("_copy(_bselect(");
  });
  /**
   * @canon spec-a-js-routine-allocates-nothing-per-call
   */
  it("samples a cube map without allocating on JS", () => {
    const cube = uniform("samplerCube");
    const source = compileJSFn(() => Fn(() => texture(cube, vec3(1, 0, 0)).toVar())(), none);
    expect(source).not.toMatch(/_cubeFace\([^)]*\[0, 0, 0\]\)/);
  });
  /**
   * @canon spec-a-js-routine-allocates-nothing-per-call
   */
  it("multiplies a matrix into itself without a copy on JS", () => {
    const source = compileJSFn(
      (a: any) =>
        Fn(() => {
          const m = mat2(a, 0, 0, 1).toVar();
          m.assign(m.mul(m));
          return m;
        })(),
      param,
    );
    expect(source).not.toContain(".slice()");
  });

  /**
   * @canon spec-a-js-routine-allocates-nothing-per-call
   */
  it("still squares a matrix into itself, from the operand as it was", () => {
    const squared = compileJSRoutine(
      (a: any) =>
        Fn(() => {
          const m = mat2(vec2(a, 0), vec2(1, 1)).toVar();
          m.assign(m.mul(m));
          return m;
        })(),
      param,
    );
    // The columns (a, 0) and (1, 1), squared, are (a * a, 0) and (a + 1, 1).
    expect(squared({ params: { a: 3 } })).toEqual(new Float64Array([9, 0, 4, 1]));
    expect(squared({ params: { a: 2 } })).toEqual(new Float64Array([4, 0, 3, 1]));
  });
  /**
   * @canon spec-a-js-routine-allocates-nothing-per-call
   */
  it("reads a matrix column inside an expression without a copy on JS", () => {
    const source = compileJSFn((a: any) => Fn(() => mat2(1, 2, 3, 4).toVar().element(a.toInt()).x)(), param);
    expect(source).not.toContain(".slice(");
  });

  /**
   * @canon spec-a-js-routine-allocates-nothing-per-call
   */
  it("reads the values of what it no longer allocates", () => {
    const run = compileJSRoutine(
      (a: any) =>
        Fn(() => {
          const m = mat2(vec2(a, 2), vec2(3, 4)).toVar();
          const diag = mat2(a).toVar();
          const broadcast = vec3(a).toVar();
          const built = vec3(a, 1, 2).toVar();
          return m
            .element(1)
            .y.add(diag.element(0).x)
            .add(broadcast.y)
            .add(built.z)
            .add(vec3(1, 2, 3).y);
        })(),
      param,
    );
    // 4 (column 1, y) + 5 (the diagonal) + 5 (broadcast) + 2 (built) + 2 (constant).
    expect(run({ params: { a: 5 } })).toBe(18);
    expect(run({ params: { a: 1 } })).toBe(10);
  });

  /**
   * @canon spec-a-compiled-js-function-returns-its-result-in-a-slot
   */
  it("returns a vector in a slot it reuses, unless it is reentrant", () => {
    const source = (reentrant: boolean) =>
      new Function(compileJSFn((a: any) => Fn(() => vec3(a, 1, 2).toVar())(), { ...param, reentrant }))() as (
        ctx: unknown,
      ) => Float64Array;
    const shared = source(false);
    const first = shared({ params: { a: 1 } });
    const second = shared({ params: { a: 7 } });
    expect(second).toBe(first);
    expect(first).toEqual(new Float64Array([7, 1, 2]));
    const own = source(true);
    const third = own({ params: { a: 1 } });
    expect(own({ params: { a: 7 } })).not.toBe(third);
    expect(third).toEqual(new Float64Array([1, 1, 2]));
  });

  /**
   * @canon spec-a-compiled-js-function-returns-its-result-in-a-slot
   */
  it("returns a stage's result in an object it reuses, unless it is reentrant", () => {
    const shade = varying("float");
    const u = uniform("float");
    const source = (reentrant: boolean) =>
      new Function(
        compileJSFn(
          () =>
            Fn(() => {
              shade.assign(u);
              builtinPosition().assign(vec4(0, 0, 0, 1));
            })(),
          { name: "main", params: [], stage: "vertex", reentrant },
        ),
      )() as (ctx: unknown) => { varyings: Record<string, number> };
    const shared = source(false);
    const first = shared({ uniforms: { [u.name]: 1 } });
    const second = shared({ uniforms: { [u.name]: 2 } });
    expect(second).toBe(first);
    expect(first.varyings[shade.name]).toBe(2);
    const own = source(true);
    const third = own({ uniforms: { [u.name]: 1 } });
    expect(own({ uniforms: { [u.name]: 2 } })).not.toBe(third);
    expect(third.varyings[shade.name]).toBe(1);
  });

  /**
   * @canon spec-a-compiled-js-function-returns-its-result-in-a-slot
   */
  it("copies the vector a routine, a stage and a grid return", () => {
    const routine = compileJSRoutine((a: any) => Fn(() => vec3(a, 1, 2).toVar())(), param);
    const first = routine({ params: { a: 1 } });
    routine({ params: { a: 7 } });
    expect(first).toEqual(new Float64Array([1, 1, 2]));
  });
});

describe("a CPU compute adapter's storage", () => {
  /**
   * @canon spec-a-cpu-compute-dispatch-allocates-nothing
   */
  it("writes nothing to the context a JS compute stage is given", () => {
    const buf = instancedArray(3, "float");
    const stage = compileJSCompute(
      () => Fn(() => buf.element(invocationIndex()).assign(invocationIndex().toFloat()))() as any,
      { name: "main", params: [] },
    );
    const data = new Float64Array(3);
    const ctx = Object.freeze({ storages: Object.freeze({ [buf.name]: data }) });
    stage(ctx, 3);
    expect(Array.from(data)).toEqual([0, 1, 2]);
    expect(Object.keys(ctx)).toEqual(["storages"]);
  });

  /**
   * @canon spec-a-compute-adapter-takes-a-storage-buffer-as-one-flat-typed-array
   */
  it("writes a vector storage buffer given as a flat typed array on JS", () => {
    const buf = instancedArray(2, "vec2");
    const adapter = createJsCompute(Fn(() => buf.element(invocationIndex()).assign(vec2(3, 4)))());
    const data = new Float32Array(4);
    adapter.setAttribute(buf.name, data);
    adapter.compute();
    expect(Array.from(data)).toEqual([3, 4, 3, 4]);
  });
  /**
   * @canon spec-a-compute-adapter-takes-a-storage-buffer-as-one-flat-typed-array
   */
  it("writes a vector storage buffer given as a flat typed array on WASM", () => {
    const buf = instancedArray(2, "vec2");
    const adapter = createWasmCompute(Fn(() => buf.element(invocationIndex()).assign(vec2(3, 4)))(), { name: "step" });
    const data = new Float32Array(4);
    adapter.setAttribute(buf.name, data);
    adapter.compute();
    expect(Array.from(data)).toEqual([3, 4, 3, 4]);
  });
});

describe("a CPU compute stage's vector buffer", () => {
  const computes = [
    ["JS", compileJSCompute],
    ["WASM", compileWasmCompute],
  ] as const;

  /**
   * @canon spec-a-cpu-compute-stage-takes-a-storage-buffer-as-one-flat-array
   */
  it.each(computes)("%s: reads and writes the element of a vector buffer in the flat array", (_, compile) => {
    const buf = instancedArray(2, "vec2");
    const stage = compile(
      () =>
        Fn(() => {
          const i = invocationIndex();
          buf.element(i).assign(buf.element(i).mul(2));
        })() as any,
      none,
    );
    const data = Float32Array.of(1, 2, 3, 4);
    stage({ storages: { [buf.name]: data } }, 2);
    expect(Array.from(data)).toEqual([2, 4, 6, 8]);
  });

  /**
   * @canon spec-a-cpu-compute-stage-takes-a-storage-buffer-as-one-flat-array
   */
  it.each(computes)("%s: reaches a column and a component of a matrix element in the flat array", (_, compile) => {
    const buf = instancedArray(2, "mat2");
    const stage = compile(
      () =>
        Fn(() => {
          const i = invocationIndex();
          buf.element(i).element(1).assign(buf.element(i).element(0).add(10));
          buf.element(i).element(0).y.assign(i.toFloat().add(100));
        })() as any,
      none,
    );
    const data = Float32Array.of(1, 2, 0, 0, 3, 4, 0, 0);
    stage({ storages: { [buf.name]: data } }, 2);
    expect(Array.from(data)).toEqual([1, 100, 11, 12, 3, 101, 13, 14]);
  });
});

describe("the slots of a JS function", () => {
  /**
   * @canon spec-a-js-program-keeps-its-vectors-in-views-of-one-buffer
   */
  it("keeps each vector in a typed view of its kind into one buffer", () => {
    const f = uniform("vec3");
    const i = uniform("ivec2");
    const u = uniform("uvec4");
    const b = uniform("bvec2");
    const build = () =>
      Fn(() => {
        f.add(1).toVar();
        i.add(1).toVar();
        u.add(1).toVar();
        b.not().toVar();
      })();
    const at = (float?: "f32") => compileJSFn(build as any, { ...none, float }).split("return function")[0]!;
    const source = at();
    expect(source.match(/new ArrayBuffer\(/g)).toHaveLength(1);
    expect(source).toMatch(/= new Float64Array\(_rmsl_slots, \d+, 3\);/);
    expect(source).toMatch(/= new Int32Array\(_rmsl_slots, \d+, 2\);/);
    expect(source).toMatch(/= new Uint32Array\(_rmsl_slots, \d+, 4\);/);
    // An ivec2 and a bvec2 uniform, each copied into a slot, and each one's result.
    expect(source.match(/= new Int32Array\(_rmsl_slots, \d+, 2\);/g)).toHaveLength(4);
    expect(at("f32")).toMatch(/= new Float32Array\(_rmsl_slots, \d+, 3\);/);
  });

  /**
   * @canon spec-a-js-program-keeps-its-vectors-in-views-of-one-buffer
   */
  it("lays an 8-byte view on a multiple of 8, whatever slots come before it", () => {
    const i = uniform("ivec3");
    const f = uniform("vec2");
    const run = compileJSRoutine(
      () =>
        Fn(() => {
          i.add(1).toVar();
          return f.add(1).toVar();
        })() as any,
      none,
    );
    expect(run({ uniforms: { [i.name]: [1, 2, 3], [f.name]: [0.5, 1.5] } })).toEqual(new Float64Array([1.5, 2.5]));
  });

  /**
   * @canon spec-a-js-program-keeps-its-vectors-in-views-of-one-buffer
   */
  it("reads a component of a boolean vector as true or false", () => {
    const flags = () => bvec2(true, false).toVar();
    const programs: [string, () => Node<"bool">][] = [
      ["x", () => flags().x],
      ["y", () => flags().y],
      ["y and true", () => flags().y.and(bool(true))],
      ["x or false", () => flags().x.or(bool(false))],
    ];
    for (const [name, build] of programs) {
      const program = () => Fn(build)() as any;
      const wasm = compileWasmRoutine(program, none)({});
      expect([name, compileJSRoutine(program, none)({})]).toEqual([name, wasm]);
      expect(typeof wasm).toBe("boolean");
    }
  });
});

describe("a JS storage access outside its buffer", () => {
  /**
   * @canon spec-a-storage-access-outside-its-buffer-reads-zero-and-writes-nothing
   */
  it("reads a storage element past either end of its buffer as zero, as WASM does", () => {
    const scalars = instancedArray(2, "float");
    const pairs = instancedArray(2, "vec2");
    const params = [{ name: "i", type: "int" as const }];
    const build = (i: Node<"int">) =>
      Fn(() => scalars.element(i).add(pairs.element(i).y).add(pairs.element(i).x.mul(10)).toVar())() as any;
    const storages = () => ({ [scalars.name]: Float32Array.of(1, 2), [pairs.name]: [3, 4, 5, 6] });
    for (const i of [10, -1, 2]) {
      const ctx = { params: { i }, storages: storages() };
      expect(compileJSRoutine(build, { name: "main", params })(ctx)).toBe(0);
      expect(compileWasmRoutine(build, { name: "main", params })({ params: { i }, storages: storages() })).toBe(0);
    }
    const whole = (i: Node<"int">) => Fn(() => pairs.element(i).toVar())() as any;
    expect(compileJSRoutine(whole, { name: "main", params })({ params: { i: 5 }, storages: storages() })).toEqual(
      new Float64Array([0, 0]),
    );
  });

  /**
   * @canon spec-a-storage-access-outside-its-buffer-reads-zero-and-writes-nothing
   */
  it("reads and writes nothing of an element the buffer holds only some components of", () => {
    const pairs = instancedArray(3, "vec2");
    const read = () =>
      Fn(() =>
        pairs
          .element(int(2))
          .x.add(pairs.element(int(2)).toVar().y.mul(10))
          .toVar(),
      )() as any;
    const write = () => Fn(() => pairs.element(int(2)).x.assign(float(9)))() as any;
    for (const [routine, compute] of [
      [compileJSRoutine, compileJSCompute],
      [compileWasmRoutine, compileWasmCompute],
    ] as const) {
      const storages = { [pairs.name]: Float64Array.of(1, 2, 3, 4, 5) };
      expect((routine as CompileCpuRoutine)(read, { name: "main", params: [] })({ storages })).toBe(0);
      compute(write, { name: "main", params: [] })({ storages }, 1);
      expect(Array.from(storages[pairs.name]!)).toEqual([1, 2, 3, 4, 5]);
    }
  });

  /**
   * @canon spec-a-storage-access-outside-its-buffer-reads-zero-and-writes-nothing
   */
  it("writes nothing past either end of a buffer, a plain array's included", () => {
    const scalars = instancedArray(2, "float");
    const pairs = instancedArray(2, "vec2");
    const build = () =>
      Fn(() => {
        scalars.element(int(7)).assign(float(9));
        scalars.element(int(-1)).assign(float(9));
        pairs.element(int(5)).assign(vec2(9, 9));
        pairs.element(int(-1)).x.assign(float(9));
        pairs.element(int(2)).xy.assign(vec2(9, 9));
      })() as any;
    for (const compile of [compileJSCompute, compileWasmCompute]) {
      const storages = { [scalars.name]: [1, 2], [pairs.name]: [3, 4, 5, 6] };
      compile(build, { name: "main", params: [] })({ storages }, 1);
      expect(storages).toEqual({ [scalars.name]: [1, 2], [pairs.name]: [3, 4, 5, 6] });
    }
  });
});

describe("a JS assignment to what the program also reads", () => {
  /**
   * @canon spec-a-var-can-be-assigned
   */
  it.each(["f64", "f32"] as const)("writes a target it read before, at %s, as WASM does", (width) => {
    const buf = instancedArray(3, "float");
    const pairs = instancedArray(2, "vec2");
    const build = () =>
      Fn(() => {
        const e = buf.element(invocationIndex());
        If(e.greaterThan(1), () => {
          e.assign(float(0));
        });
        const v = vec3(1, 2, 3).toVar();
        const k = int(invocationIndex());
        If(v.element(k).greaterThan(1), () => {
          v.element(k).assign(float(9));
        });
        const flags = bvec2(true, false).toVar();
        If(flags.x, () => {
          flags.x.assign(bool(false));
        });
        const p = pairs.element(uint(0));
        If(p.x.greaterThan(0), () => {
          p.assign(vec2(v.element(k), select(flags.x, 1, 2)));
        });
      })() as any;
    const run = (compile: typeof compileJSCompute | typeof compileWasmCompute) => {
      const storages = { [buf.name]: Float64Array.of(0.5, 2, 3), [pairs.name]: Float64Array.of(1, 1, 1, 1) };
      compile(build, { name: "main", params: [], float: width })({ storages }, 3);
      return storages;
    };
    const js = run(compileJSCompute);
    expect(Array.from(js[buf.name]!)).toEqual([0.5, 0, 0]);
    expect(Array.from(js[pairs.name]!)).toEqual([9, 2, 1, 1]);
    expect(js).toEqual(run(compileWasmCompute));
  });
});

describe("the element-wise operations of a JS function", () => {
  /**
   * @canon spec-a-js-function-writes-out-what-would-cross-a-call
   */
  it("are written out component by component, with no helper and no question of shape", () => {
    const v = uniform("vec3");
    const t = uniform("float");
    const build = () => Fn(() => mix(v.mul(t), t.mul(v), t).toVar())() as any;
    const source = compileJSFn(build, none);
    expect(source).not.toMatch(/function _v3(mul|mix)/);
    expect(source).not.toContain("typeof");
    // A scalar that is not a local is read into one once, before the components are written.
    expect(source).toMatch(
      /(_rmsl_t\d+) = \(_rmsl_in_uniforms\["_rmsl_u\d+"\] \?\? 0\);[\s\S]*\[0\] = _rmsl_\w+\[0\] \* \1;/,
    );
    expect(compileJSRoutine(build, none)({ uniforms: { [v.name]: [1, 2, 3], [t.name]: 0.5 } })).toEqual(
      new Float64Array([0.5, 1, 1.5]),
    );
  });

  /**
   * @canon spec-a-js-function-writes-out-what-would-cross-a-call
   */
  it("writes a dot product, a length and a distance out, summing from zero in order", () => {
    const a = uniform("vec3");
    const b = uniform("vec3");
    const build = () => Fn(() => a.dot(b).add(a.length()).add(a.distance(b)).toVar())() as any;
    const source = compileJSFn(build, none);
    expect(source).not.toMatch(/_vdot|_vlen|_vdist/);
    expect(source).toMatch(/\(0 \+ \w+\[0\] \* \w+\[0\] \+ \w+\[1\] \* \w+\[1\] \+ \w+\[2\] \* \w+\[2\]\)/);
    // A product of -0 summed from zero is +0, as the helper and WASM give it.
    const ctx = { uniforms: { [a.name]: [-0, 0, 0], [b.name]: [1, 0, 0] } };
    const js = compileJSRoutine(() => Fn(() => a.dot(b).toVar())() as any, none)(ctx);
    const wasm = compileWasmRoutine(() => Fn(() => a.dot(b).toVar())() as any, none)(ctx);
    expect(Object.is(js, wasm)).toBe(true);
  });

  /**
   * @canon spec-a-js-function-writes-out-what-would-cross-a-call
   */
  it("keeps a written-out smoothstep whole inside the expression around it", () => {
    const u = uniform("float");
    const build = () =>
      Fn(() =>
        float(1)
          .div(smoothstep(0, 1, u))
          .toVar(),
      )() as any;
    const ctx = { uniforms: { [u.name]: 0.25 } };
    expect(compileJSRoutine(build, none)(ctx)).toBe(1 / (0.25 * 0.25 * (3 - 2 * 0.25)));
    expect(compileJSRoutine(build, none)(ctx)).toBe(compileWasmRoutine(build, none)(ctx));
  });

  /**
   * @canon spec-a-js-function-writes-out-what-would-cross-a-call
   */
  it("reads a scalar operand before it writes a component, even one of the slot itself", () => {
    const programs = [
      () =>
        Fn(() => {
          const v = vec3(2, 3, 4).toVar();
          v.assign(v.mul(v.x));
          return v;
        })() as any,
      () =>
        Fn(() => {
          const v = vec3(2, 3, 4).toVar();
          v.assign(v.x.add(v));
          return v;
        })() as any,
    ];
    expect(compileJSRoutine(programs[0]!, none)({})).toEqual(new Float64Array([4, 6, 8]));
    expect(compileJSRoutine(programs[1]!, none)({})).toEqual(new Float64Array([4, 5, 6]));
    for (const program of programs) {
      expect(compileJSRoutine(program, none)({})).toEqual(compileWasmRoutine(program, none)({}));
    }
  });
});

describe("the scalars and inputs of a JS function", () => {
  /**
   * @canon spec-a-js-function-keeps-a-scalar-in-a-local
   */
  it("declares its scalars in the function and its vectors outside it", () => {
    const t = uniform("float");
    const source = compileJSFn(() => Fn(() => t.mul(2).sin().add(vec3(1).x).toVar())() as any, none);
    const [outside, inside] = source.split("return function");
    expect(outside).not.toMatch(/^let _rmsl_\w+;$/m);
    expect(inside).toMatch(/^\s*let _rmsl_\w+(, _rmsl_\w+)*;$/m);
  });

  /**
   * @canon spec-a-js-function-copies-a-host-vector-into-a-slot-of-its-kind
   */
  it("copies a vector the host passes into a slot of its kind before it reads it", () => {
    const v = uniform("vec3");
    const i = uniform("ivec2");
    const source = compileJSFn(() => Fn(() => vec3(v.add(1).x, i.add(1).toFloat()).toVar())() as any, none);
    // Copied one component at a time where it is read, into a slot of the input's kind.
    expect(source).toMatch(
      /(_rmsl_\w+) = _rmsl_in_uniforms\["_rmsl_u\d+"\] \?\? _zeros;\s+(_rmsl_t\d+)\[0\] = \1\[0\] \?\? 0;/,
    );
    expect(source).toMatch(/= new Int32Array\(_rmsl_slots, \d+, 2\);/);
    const run = compileJSRoutine(() => Fn(() => v.add(1).toVar())() as any, none);
    expect(run({ uniforms: { [v.name]: [1, 2, 3] } })).toEqual(new Float64Array([2, 3, 4]));
    expect(run({ uniforms: { [v.name]: Float32Array.of(1, 2, 3) } })).toEqual(new Float64Array([2, 3, 4]));
  });

  /**
   * @canon spec-a-js-function-copies-a-host-vector-into-a-slot-of-its-kind
   */
  it("reads its copy of a host vector again through a write, until the block ends", () => {
    const v = uniform("vec3");
    const build = () =>
      Fn(() => {
        const sum = v.add(1).toVar();
        sum.assign(sum.add(v));
        If(sum.x.greaterThan(0), () => {
          sum.assign(sum.mul(v));
        });
        return sum;
      })() as any;
    const source = compileJSFn(build, none);
    // Once: the copy made before the block has run when the block runs.
    expect(source.match(/= _rmsl_in_uniforms\["_rmsl_u\d+"\] \?\? _zeros;/g)).toHaveLength(1);
    expect(compileJSRoutine(build, none)({ uniforms: { [v.name]: [1, 2, 3] } })).toEqual(new Float64Array([3, 10, 21]));
  });

  /**
   * @canon spec-a-js-function-copies-a-host-vector-into-a-slot-of-its-kind
   */
  it("copies a host vector first read in a loop once, before the outermost loop", () => {
    const v = uniform("vec3");
    const build = () =>
      Fn(() => {
        const sum = vec3(0).toVar();
        Loop(int(3), () => {
          Loop(int(2), () => {
            sum.addAssign(v);
          });
        });
        return sum.add(v);
      })() as any;
    const source = compileJSFn(build, none);
    const copies = source.match(/= _rmsl_in_uniforms\["_rmsl_u\d+"\] \?\? _zeros;/g);
    expect(copies).toHaveLength(1);
    expect(source.indexOf(copies![0]!)).toBeLessThan(source.indexOf("for ("));
    expect(compileJSRoutine(build, none)({ uniforms: { [v.name]: [1, 2, 3] } })).toEqual(new Float64Array([7, 14, 21]));
  });

  /**
   * @canon spec-a-js-function-copies-a-host-vector-into-a-slot-of-its-kind
   */
  it("reads a host vector it copies before a loop as zero when the host leaves it out", () => {
    const v = uniform("vec3");
    const n = uniform("int");
    const build = () =>
      Fn(() => {
        const sum = vec3(0).toVar();
        Loop(n, () => {
          sum.addAssign(v);
        });
        return sum;
      })() as any;
    const run = compileJSRoutine(build, none);
    expect(run({ uniforms: { [n.name]: 0 } })).toEqual(new Float64Array([0, 0, 0]));
    expect(run({ uniforms: { [n.name]: 2 } })).toEqual(new Float64Array([0, 0, 0]));
    expect(run({ uniforms: { [n.name]: 2, [v.name]: [1, 2, 3] } })).toEqual(new Float64Array([2, 4, 6]));
  });

  /**
   * @canon spec-a-js-routine-allocates-nothing-per-call
   */
  it("allocates nothing per call of a shading program fed plain arrays", async () => {
    const n = uniform("vec3");
    const l = uniform("vec3");
    const base = uniform("vec3");
    const rough = uniform("float");
    const build = () =>
      Fn(() => {
        const nl = n.dot(l).max(0).toVar();
        const spec = smoothstep(0.5, 1, nl).mul(rough).toVar();
        return mix(base.mul(nl), vec3(1), spec)
          .add(normalize(n.add(l)).mul(0.05))
          .toVar();
      })() as any;
    const ctx = {
      uniforms: { [n.name]: [0, 1, 0], [l.name]: [0.3, 0.8, 0.5], [base.name]: [0.8, 0.2, 0.1], [rough.name]: 0.37 },
    };
    // A routine copies its result out, so the program is called through the function the routine wraps.
    const raw = new Function(compileJSFn(build, none))() as (c: unknown) => unknown;
    // An object is 16 bytes at the least, so a call that allocated one would count that many.
    expect(await allocatedBy(() => raw(ctx), 50000, 20000)).toBeLessThan(1);
  });

  /**
   * @canon spec-a-js-grid-allocates-nothing-per-pixel
   */
  it.each([
    ["a vector", () => vec4(fragCoord().x.mul(0.1), fragCoord().y, 0.5, 1)],
    ["a scalar", () => fragCoord().x.mul(fragCoord().y)],
  ] as const)("allocates nothing per pixel of a grid of %s", async (_, root) => {
    const grid = compileJSGrid(() => Fn(() => root().toVar())() as any, none);
    const ctx = {};
    const out = grid(ctx, 16, 16);
    // A fill calls the program 256 times, and an object is 16 bytes at the least.
    expect(await allocatedBy(() => grid(ctx, 16, 16, out as any), 2000, 200)).toBeLessThan(64);
  });

  /**
   * @canon spec-a-js-draw-allocates-nothing-per-vertex-or-fragment
   */
  it("allocates nothing per vertex or fragment of a rasterized draw", async () => {
    const position = attribute("vec3");
    const colour = attribute("vec3");
    const shade = varying("vec3");
    const raster = compileJS(
      () =>
        Fn(() => {
          shade.assign(colour.mul(0.5));
          builtinPosition().assign(vec4(position, 1));
        })(),
      () => Fn(() => vec4(shade, 1))(),
      { attributeTypes: { [position.name]: "vec3", [colour.name]: "vec3" } },
    );
    // Two triangles covering a 16 by 16 target: 6 vertices and 256 fragments a draw.
    const ctx = {
      attributes: {
        [position.name]: Float64Array.of(-1, -1, 0, 1, -1, 0, -1, 1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0),
        [colour.name]: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1),
      },
    };
    const options = { width: 16, height: 16, clear: true, clearDepth: true };
    // A draw calls a stage 262 times, and an object is 16 bytes at the least.
    expect(await allocatedBy(() => raster.draw(ctx, options), 2000, 200)).toBeLessThan(64);
  });
});

describe("what a JS program keeps after a call", () => {
  /** Whether the object `call` is given is collected once the call returns and nothing else holds it. */
  async function collectedAfter(call: (input: object) => void): Promise<boolean> {
    setFlagsFromString("--expose-gc");
    const gc = runInNewContext("gc") as () => void;
    let ref!: WeakRef<object>;
    (() => {
      const input = {};
      ref = new WeakRef(input);
      call(input);
    })();
    // A WeakRef keeps its target until the job that made it ends.
    await new Promise((resolve) => setTimeout(resolve, 0));
    gc();
    return ref.deref() === undefined;
  }

  /**
   * @canon spec-a-js-program-keeps-no-host-input-after-a-call
   */
  it("keeps none of the inputs of a grid fill, a compute dispatch or a draw", async () => {
    const grid = compileJSGrid(() => Fn(() => fragCoord().x.toVar())() as any, none);
    expect(await collectedAfter((uniforms) => grid({ uniforms } as any, 2, 2))).toBe(true);

    const buf = instancedArray(2, "float");
    const stage = compileJSCompute(() => Fn(() => buf.element(invocationIndex()).assign(float(1)))() as any, {
      name: "main",
      params: [],
    });
    expect(
      await collectedAfter((uniforms) => stage({ uniforms, storages: { [buf.name]: new Float64Array(2) } } as any, 2)),
    ).toBe(true);

    const position = attribute("vec3");
    const raster = compileJS(
      () => Fn(() => builtinPosition().assign(vec4(position, 1)))(),
      () => Fn(() => vec4(1))(),
      { attributeTypes: { [position.name]: "vec3" } },
    );
    const attributes = { [position.name]: Float64Array.of(-1, -1, 0, 1, -1, 0, -1, 1, 0) };
    expect(
      await collectedAfter((uniforms) => raster.draw({ attributes, uniforms } as any, { width: 2, height: 2 })),
    ).toBe(true);
  });
});

describe("the rounding helper of JS", () => {
  /**
   * @canon spec-round-takes-a-half-to-the-even-integer
   */
  it("rounds as constant folding does, for every value of a sweep", () => {
    const helper = new Function(`${jsHelperSource("roundEven")}; return _rmsl_roundEven;`)() as (x: number) => number;
    const values = [
      0, -0, 0.5, -0.5, 1.5, 2.5, -1.5, -2.5, 0.49999999999999994, 4503599627370495.5, 1e300, -1e300, 7.25, -7.75,
    ];
    for (const x of values) expect(Object.is(helper(x), roundHalfToEven(x)), `round(${x})`).toBe(true);
  });

  /**
   * @canon spec-round-takes-a-half-to-the-even-integer
   */
  it("comes with every compiled function that rounds a scalar or a vector", () => {
    const scalar = compileJSFn(() => Fn(() => uniform("float").round())() as any, none);
    const vector = compileJSFn(() => Fn(() => vec3(uniform("float")).round())() as any, none);
    for (const compiled of [scalar, vector]) expect(String(compiled)).toContain("function _rmsl_roundEven");
  });
});
