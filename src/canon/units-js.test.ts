import { afterAll, describe, expect, it } from "vitest";
import {
  attribute,
  builtinPosition,
  Fn,
  float,
  fragCoord,
  mat2,
  uint,
  uniform,
  vec2,
  vec3,
  vec4,
  type Node,
} from "../rmsl";
import { compileJS, compileJSFn, compileJSRoutine, createJsRoutine } from "../js";
import { compileWasm, compileWasmRoutine, createWasmRoutine } from "../wasm";
import { assertRecordedEvaluationsAgree, closeEvaluators, evaluateRecording } from "../testing/shader-eval";

afterAll(async () => {
  await assertRecordedEvaluationsAgree();
  await closeEvaluators();
}, 120_000);

const none = { name: "main", params: [] };

const cpuTargets: [string, typeof compileJSRoutine][] = [
  ["JS", compileJSRoutine],
  ["WASM", compileWasmRoutine as typeof compileJSRoutine],
];

const rasterizers: [string, typeof compileJS][] = [
  ["JS", compileJS],
  ["WASM", compileWasm as unknown as typeof compileJS],
];

const routineAdapters: [string, typeof createJsRoutine][] = [
  ["JS", createJsRoutine],
  ["WASM", createWasmRoutine as unknown as typeof createJsRoutine],
];

/** The value a CPU routine returned, taken out of a result object if it wrapped one. */
const valueOf = (result: unknown) =>
  typeof result === "object" && result !== null && !Array.isArray(result) && "value" in result
    ? (result as { value: unknown }).value
    : result;

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
   * `mat2(1, 2, 3, 4)` has the columns (1, 2) and (3, 4). A WASM routine
   * wraps the value in a result object, which bug
   * wasm-wraps-a-vector-result-in-a-result-object records, so the test reads
   * the value out of it.
   *
   * @canon spec-a-cpu-routine-returns-a-matrix-as-its-columns-in-one-array
   */
  it.each(cpuTargets)("%s: returns a matrix as its columns in one flat array", (_, compile) => {
    const routine = compile(() => Fn(() => mat2(vec2(1, 2), vec2(3, 4)))() as any, none);
    expect(valueOf(routine.run({}))).toEqual([1, 2, 3, 4]);
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
    expect(valueOf(routine.run({ uniforms: { [(scale as any).name]: 2 }, fragCoord: [3, 0] }))).toBe(6);
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
        (valueOf(routine.run({ textures: { [tex.name]: { ...texture, ...filters } } })) as number[])[0];
      expect(red({ magFilter: "nearest", minFilter: "linear" })).toBe(20);
      expect(red({ magFilter: "linear", minFilter: "nearest" })).toBe(15);
    },
  );

  /**
   * @canon spec-draw-fills-a-float64-array-for-a-float-result
   */
  it.each(cpuTargets)("%s: draws a float result into a Float64Array", (_, compile) => {
    const out = compile(() => Fn(() => fragCoord().x.add(0.25))() as any, none).draw({}, 2, 1);
    expect(out).toBeInstanceOf(Float64Array);
    expect(Array.from(out)).toEqual([0.75, 1.75]);
  });

  /**
   * @canon spec-draw-fills-an-int32-array-for-an-int-result
   */
  it.each(cpuTargets)("%s: draws an int result into an Int32Array", (_, compile) => {
    const out = compile(() => Fn(() => fragCoord().x.toInt().sub(2))() as any, none).draw({}, 2, 1);
    expect(out).toBeInstanceOf(Int32Array);
    expect(Array.from(out)).toEqual([-2, -1]);
  });

  /**
   * 3000000000 lies past the largest `int`, so only an unsigned array holds it.
   *
   * @canon spec-draw-fills-a-uint32-array-for-a-uint-result
   */
  it.each(cpuTargets)("%s: draws a uint result into a Uint32Array", (_, compile) => {
    const out = compile(() => Fn(() => fragCoord().x.toUint().add(uint(3000000000)))() as any, none).draw({}, 2, 1);
    expect(out).toBeInstanceOf(Uint32Array);
    expect(Array.from(out)).toEqual([3000000000, 3000000001]);
  });

  /**
   * @canon spec-draw-writes-a-bool-result-as-one-or-zero-in-an-int32-array
   */
  it.each(cpuTargets)("%s: draws a bool result as 1 or 0 into an Int32Array", (_, compile) => {
    const out = compile(() => Fn(() => fragCoord().x.greaterThan(1))() as any, none).draw({}, 2, 1);
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
