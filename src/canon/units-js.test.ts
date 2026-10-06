import { afterAll, describe, expect, it } from "vitest";
import {
  attribute,
  builtinPosition,
  Discard,
  Fn,
  float,
  If,
  fragCoord,
  instancedArray,
  invocationIndex,
  mat2,
  outputStruct,
  uint,
  uniform,
  varying,
  vec2,
  vec3,
  vec4,
  type Node,
} from "../rmsl";
import type { CompileCpuRoutine } from "../backends/cpu";
import {
  compileJS,
  compileJSCompute,
  compileJSFn,
  compileJSFragment,
  compileJSGrid,
  compileJSRoutine,
  compileJSVertex,
  createJsRoutine,
} from "../js";
import {
  compileWasm,
  compileWasmCompute,
  compileWasmFragment,
  compileWasmGrid,
  compileWasmRoutine,
  compileWasmVertex,
  createWasmRoutine,
} from "../wasm";
import { assertRecordedEvaluationsAgree, closeEvaluators, evaluateRecording } from "../testing/shader-eval";

afterAll(async () => {
  await assertRecordedEvaluationsAgree();
  await closeEvaluators();
}, 120_000);

const none = { name: "main", params: [] };

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

const routineAdapters: [string, typeof createJsRoutine][] = [
  ["JS", createJsRoutine],
  ["WASM", createWasmRoutine as unknown as typeof createJsRoutine],
];

/** The bytes a routine adapter puts on a one-pixel canvas for `draw`. */
function shownOnCanvas(create: typeof createJsRoutine, draw: Node<any>): number[] {
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
   * @canon spec-a-cpu-routine-returns-its-value-or-a-result
   */
  it.each(cpuTargets)("%s: returns a vector result as a bare array", (_, compile) => {
    const routine = compile(() => Fn(() => vec3(1, 2, 3))() as any, none);
    expect(routine.run({})).toEqual([1, 2, 3]);
  });

  /**
   * `mat2(1, 2, 3, 4)` has the columns (1, 2) and (3, 4).
   *
   * @canon spec-a-cpu-routine-returns-a-matrix-as-its-columns-in-one-array
   */
  it.each(cpuTargets)("%s: returns a matrix as its columns in one flat array", (_, compile) => {
    const routine = compile(() => Fn(() => mat2(vec2(1, 2), vec2(3, 4)))() as any, none);
    expect(routine.run({})).toEqual([1, 2, 3, 4]);
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
  it.each(cpuTargets)("%s: calls the builder it is given once", (_, compile) => {
    let calls = 0;
    let scale!: Node<"float">;
    const routine = compile(() => {
      calls++;
      scale = uniform("float");
      return Fn(() => fragCoord().x.mul(scale))();
    }, none);
    expect(calls).toBe(1);
    expect(routine.run({ uniforms: { [(scale as any).name]: 2 }, fragCoord: [3, 0] })).toBe(6);
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
        return tex.texture(vec2(0.5, 0.5));
      })();
      const routine = compile(() => program, none);
      const texture = { data: new Float32Array([10, 10, 10, 10, 20, 20, 20, 20]), width: 2, height: 1 };
      const red = (filters: { magFilter: "nearest" | "linear"; minFilter: "nearest" | "linear" }) =>
        (routine.run({ textures: { [tex.name]: { ...texture, ...filters } } }) as number[])[0];
      expect(red({ magFilter: "nearest", minFilter: "linear" })).toBe(20);
      expect(red({ magFilter: "linear", minFilter: "nearest" })).toBe(15);
    },
  );

  /**
   * @canon spec-a-grid-fills-a-float64-array-for-a-float-result
   */
  it.each(grids)("%s: fills a float result into a Float64Array", (_, compile) => {
    const out = compile(() => Fn(() => fragCoord().x.add(0.25))(), none).fill({}, 2, 1);
    expect(out).toBeInstanceOf(Float64Array);
    expect(Array.from(out)).toEqual([0.75, 1.75]);
  });

  /**
   * @canon spec-a-grid-fills-an-int32-array-for-an-int-result
   */
  it.each(grids)("%s: fills an int result into an Int32Array", (_, compile) => {
    const out = compile(() => Fn(() => fragCoord().x.toInt().sub(2))(), none).fill({}, 2, 1);
    expect(out).toBeInstanceOf(Int32Array);
    expect(Array.from(out)).toEqual([-2, -1]);
  });

  /**
   * 3000000000 lies past the largest `int`, so only an unsigned array holds it.
   *
   * @canon spec-a-grid-fills-a-uint32-array-for-a-uint-result
   */
  it.each(grids)("%s: fills a uint result into a Uint32Array", (_, compile) => {
    const out = compile(() => Fn(() => fragCoord().x.toUint().add(uint(3000000000)))(), none).fill({}, 2, 1);
    expect(out).toBeInstanceOf(Uint32Array);
    expect(Array.from(out)).toEqual([3000000000, 3000000001]);
  });

  /**
   * @canon spec-a-grid-writes-a-bool-result-as-one-or-zero-in-an-int32-array
   */
  it.each(grids)("%s: fills a bool result as 1 or 0 into an Int32Array", (_, compile) => {
    const out = compile(() => Fn(() => fragCoord().x.greaterThan(1))(), none).fill({}, 2, 1);
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
    expect(source.match(/(?<!function )_v3add\(/g)?.length).toBe(1);
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
    expect(source.match(/(?<!function )_v2add\(/g)?.length).toBe(1);
  });
});

describe("a JS routine's results", () => {
  const param = { name: "main", params: [{ name: "a", type: "float" as const }] };

  /**
   * @canon spec-a-cpu-routine-answers-one-fragment-per-call
   */
  it("keeps the value a JS routine returned when it is called again", () => {
    const run = compileJSRoutine((a: any) => Fn(() => vec3(a, a, a).toVar())(), param);
    const first = run.run({ params: { a: 1 } });
    run.run({ params: { a: 2 } });
    expect(first).toEqual([1, 1, 1]);
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
    const first = routine.run({ uniforms: { [input.name]: passed } });
    (first as number[])[0] = 9;
    expect(passed).toEqual([1, 2, 3]);
    expect(routine.run({ uniforms: { [input.name]: passed } })).toEqual([1, 2, 3]);
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

const stages = [
  ["JS", compileJSVertex, compileJSFragment, compileJSCompute],
  ["WASM", compileWasmVertex, compileWasmFragment, compileWasmCompute],
] as const;

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
    const result = stage.run({ attributes: { [place.name]: [1, 2, 3] } });
    expect(result.position).toEqual([1, 2, 3, 1]);
    expect(Object.values(result.varyings)).toEqual([[1, 2]]);
  });

  /**
   * @canon spec-a-vertex-stage-returns-its-position-and-varyings
   */
  it.each(stages)(
    "returns the vec4 a vertex stage returns as its position, with no varyings on %s",
    (_, compileVertex) => {
      const place = attribute("vec3");
      const stage = compileVertex(() => Fn(() => vec4(place, 1))(), none);
      expect(stage.run({ attributes: { [place.name]: [1, 2, 3] } })).toEqual({ position: [1, 2, 3, 1], varyings: {} });
    },
  );

  /**
   * @canon spec-a-fragment-stage-returns-its-colour-and-outputs
   */
  it.each(stages)(
    "returns the colour of a fragment stage as a vec4, a vec3 with an opaque alpha on %s",
    (_, __, compileFragment) => {
      expect(compileFragment(() => Fn(() => vec4(1, 2, 3, 4))(), none).run({})).toEqual({
        value: [1, 2, 3, 4],
        outputs: [],
      });
      expect(compileFragment(() => Fn(() => vec3(1, 2, 3))(), none).run({})).toEqual({
        value: [1, 2, 3, 1],
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
      expect(stage.run({})).toEqual({ value: undefined, outputs: [7, [1, 2, 3]] });
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
    expect(stage.run({})).toBeNull();
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
      expect(stage.dispatch({ storages: { [buffer.name]: data } }, 2)).toBeUndefined();
      expect(Array.from(data)).toEqual([9, 2, 9, 4]);
      expect(stage.storageTypes[buffer.name]).toBe("float");
    },
  );

  /**
   * @canon spec-a-fragment-stage-draws-a-grid-with-quad
   */
  it.each(stages)(
    "shades every pixel of a grid with quad, at the centre of the pixel on %s",
    (_, __, compileFragment) => {
      const stage = compileFragment(() => Fn(() => vec4(fragCoord().x, fragCoord().y, 0, 1))(), none);
      expect(Array.from(stage.quad({}, 2, 2))).toEqual([
        0.5, 0.5, 0, 1, 1.5, 0.5, 0, 1, 0.5, 1.5, 0, 1, 1.5, 1.5, 0, 1,
      ]);
    },
  );

  /**
   * @canon spec-a-grid-writes-a-discarded-pixel-as-zero
   */
  it.each(stages)("writes a pixel that discards as zero in every channel with quad on %s", (_, __, compileFragment) => {
    const stage = compileFragment(
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
    stage.quad({}, 2, 1, out);
    expect(Array.from(out)).toEqual([1, 2, 3, 4, 0, 0, 0, 0]);
  });
});
