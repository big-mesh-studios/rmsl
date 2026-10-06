import { describe, expect, it } from "vitest";
import {
  attribute,
  builtinPosition,
  Discard,
  Fn,
  float,
  If,
  instancedArray,
  int,
  invocationIndex,
  ivec2,
  mat2,
  mat3,
  select,
  smoothstep,
  textureLoad,
  uniform,
  uniformArray,
  vec2,
  vec3,
  vec4,
  type Node,
} from "../rmsl";
import { compileJS, compileJSFn, compileJSRoutine, createJsCompute, createJsGrid } from "../js";
import { evaluateJS } from "../testing/shader-eval";

const param = { name: "main", params: [{ name: "a", type: "float" as const }] };
const none = { name: "main", params: [] };

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

const checker = { data: Float32Array.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), width: 2, height: 2 };

describe("known bugs of the JS target, each failing until its fix", () => {
  /**
   * The statements of an inline `Fn` result are emitted with the first read of
   * its value, so the other branch of an `If` gets the value and not the
   * statements.
   *
   * @canon bug-js-runs-an-inline-fn-only-on-the-path-that-first-reads-it
   */
  it.fails("runs an inline Fn read in both branches of an If on the branch taken, on JS", () => {
    const counter = instancedArray(1, "float");
    const shared = Fn(() => {
      counter.element(int(0)).addAssign(1);
      return float(2);
    })() as any;
    const taken = uniform("float");
    const result = instancedArray(1, "float");
    const build = () =>
      Fn(() => {
        If(taken.greaterThan(0), () => {
          result.element(int(0)).assign(shared.fract().add(shared));
        }).Else(() => {
          result.element(int(0)).assign(shared.mul(10));
        });
      })();
    const runs = (branch: number) => {
      const data = new Float64Array(1);
      compileJSRoutine(build as any, none)({
        storages: { [counter.name]: data, [result.name]: new Float64Array(1) },
        uniforms: { [taken.name]: branch },
      });
      return data[0];
    };
    expect(runs(1)).toBe(1);
    expect(runs(0)).toBe(1);
  });

  /**
   * The JS rasterizer writes the depth of a fragment before running it, so a
   * fragment that discards still hides what is drawn behind it later.
   *
   * @canon bug-the-cpu-rasterizers-write-the-depth-of-a-discarded-fragment
   */
  it.fails("writes no depth for a discarded fragment in the JS rasterizer", () => {
    const draw = flatRasterizer((color, drop) => {
      If(drop.greaterThan(0), () => Discard());
      return color;
    });
    draw(screenAt(0.25), [0, 1, 0, 1], {}, 1);
    expect(Array.from(draw(screenAt(0.5), [0, 0, 1, 1], { clear: false, clearDepth: false }).slice(0, 4))).toEqual([
      0, 0, 1, 1,
    ]);
  });

  /**
   * On JS, `textureLoad` outside the texture into a variable leaves the
   * variable as it was, rather than writing zero into it.
   *
   * @canon bug-js-keeps-a-stale-texel-out-of-range
   */
  it.fails("reads zero for a texel out of range into a variable that held a texel on JS", () => {
    const tex = uniform("sampler2D");
    const build = (a: any) =>
      Fn(() => {
        const v = textureLoad(tex, ivec2(0, 0)).toVar();
        v.assign(textureLoad(tex, ivec2(a.toInt(), 0)));
        return v;
      })();
    const run = compileJSRoutine(build, param);
    expect(run({ params: { a: 5 }, textures: { [tex.name]: checker } })).toEqual([0, 0, 0, 0]);
  });

  /**
   * On JS, a vector component read by a run-time index past the end gives
   * `undefined`.
   *
   * @canon bug-js-reads-a-vector-component-out-of-range-as-undefined
   */
  it.fails("reads the last component for a run-time index past a vector on JS", () => {
    expect(evaluateJS((a) => vec4(1, 2, 3, 4).element(a.toInt()), [9])).toBe(4);
  });

  /**
   * On JS, a matrix column read by a run-time index past the end gives an
   * empty array.
   *
   * @canon bug-js-reads-a-matrix-column-out-of-range-as-empty
   */
  it.fails("reads the last column for a run-time index past a matrix on JS", () => {
    const m = () => mat3(1, 2, 3, 4, 5, 6, 7, 8, 9);
    expect(evaluateJS((a) => m().element(a.toInt()), [9])).toEqual([7, 8, 9]);
  });

  /**
   * On JS, a write to a vector component by a run-time index past the end
   * adds a component to the vector instead of writing the last one.
   *
   * @canon bug-js-writes-a-vector-component-out-of-range-past-its-end
   */
  it.fails("writes the last component for a run-time index past a vector on JS", () => {
    const write = Fn((a: Node<"float">) => {
      const v = vec4(1, 2, 3, 4).toVar();
      v.element(a.toInt()).assign(float(20));
      return v;
    });
    expect(evaluateJS((a) => write(a), [9])).toEqual([1, 2, 3, 20]);
  });

  /**
   * On JS, a uniform array element read by a run-time index past the end
   * gives `undefined`.
   *
   * @canon bug-js-reads-a-uniform-array-element-out-of-range-as-undefined
   */
  it.fails("reads the last element for a run-time index past a uniform array on JS", () => {
    const items = uniformArray("float", 4);
    const run = compileJSRoutine((a: any) => Fn(() => items.element(a.toInt()).add(0).toVar())(), param);
    expect(run({ params: { a: 9 }, uniforms: { [items.name]: [1, 2, 3, 4] } })).toBe(4);
  });

  /**
   * A scalar `smoothstep` on JS builds a closure on every call.
   *
   * @canon bug-js-smoothstep-allocates-a-closure-per-call
   */
  it.fails("computes a scalar smoothstep without a closure on JS", () => {
    const source = compileJSFn((a: any) => Fn(() => smoothstep(0, 1, a).toVar())(), param);
    expect(source).not.toMatch(/function\s*\(t\)/);
  });

  /**
   * On JS, a vector or matrix computed outside an assignment is built as a new
   * array on every call: a matrix column is copied with `slice`, a scalar
   * matrix through `_matDiag`, and a constant vector as an array literal.
   *
   * @canon bug-js-allocates-a-vector-computed-outside-an-assignment
   */
  it.fails("reads a matrix column inside an expression without a copy on JS", () => {
    const source = compileJSFn((a: any) => Fn(() => mat2(1, 2, 3, 4).toVar().element(a.toInt()).x)(), param);
    expect(source).not.toContain(".slice(");
  });

  /**
   * On JS, a component-wise `select` calls its helper with no output
   * argument, so the helper allocates its result on every call.
   *
   * @canon bug-js-select-allocates-its-result-per-call
   */
  it.fails("writes a component-wise select into an output argument on JS", () => {
    const source = compileJSFn(
      (a: any) => Fn(() => select(vec3(a, 1, -1).greaterThan(vec3(0, 0, 0)), vec3(1, 2, 3), vec3(4, 5, 6)).toVar())(),
      param,
    );
    expect(source).not.toContain("_copy(_bselect(");
  });

  /**
   * On JS, sampling a cube map allocates an array for the face it picks on
   * every call.
   *
   * @canon bug-js-cube-map-allocates-its-face-per-call
   */
  it.fails("samples a cube map without allocating on JS", () => {
    const cube = uniform("samplerCube");
    const source = compileJSFn(() => Fn(() => cube.texture(vec3(1, 0, 0)).toVar())(), none);
    expect(source).not.toMatch(/_cubeFace\([^)]*\[0, 0, 0\]\)/);
  });

  /**
   * On JS, a matrix product written into one of its own operands copies that
   * operand with `slice` on every call.
   *
   * @canon bug-js-matrix-product-into-its-operand-allocates
   */
  it.fails("multiplies a matrix into itself without a copy on JS", () => {
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
   * The JS rasterizer shades a pixel centre on an edge two triangles share
   * with both, so the triangle drawn last wins it.
   *
   * @canon bug-js-rasterizer-shades-a-shared-edge-twice
   */
  it.fails("gives a pixel on a shared edge to one triangle whatever their order in the JS rasterizer", () => {
    const draw = flatRasterizer();
    const upper = [-1, 1, 0, 1, -1, 0, 1, 1, 0];
    const lower = [-1, 1, 0, -1, -1, 0, 1, -1, 0];
    const composes = { clear: false, clearDepth: false };
    draw(upper, [1, 0, 0, 1]);
    const upperFirst = Array.from(draw(lower, [0, 0, 1, 1], composes));
    draw(lower, [0, 0, 1, 1]);
    const lowerFirst = Array.from(draw(upper, [1, 0, 0, 1], composes));
    expect(upperFirst).toEqual(lowerFirst);
  });

  /**
   * The JS rasterizer draws a triangle whose depth lies below zero, which
   * WebGPU clips away.
   *
   * @canon bug-js-rasterizer-draws-a-triangle-below-zero-depth
   */
  it.fails("clips a triangle below zero depth in the JS rasterizer", () => {
    const draw = flatRasterizer();
    const image = draw(screenAt(-0.5), [1, 0, 0, 1]);
    expect(Array.from(image)).toEqual(new Array(16).fill(0));
  });

  /**
   * On JS, converting a negative float to `uint` wraps it to a large unsigned
   * integer instead of clamping it to zero.
   *
   * @canon bug-js-leaves-a-float-outside-an-integer-range-unclamped
   */
  it.fails("clamps a negative float converted to uint to zero on JS", () => {
    const run = compileJSRoutine((a: any) => Fn(() => a.toUint().toVar())(), param);
    expect(run({ params: { a: -1.5 } })).toBe(0);
  });

  /**
   * On JS, an element past the end of a shorter array the call passes reads
   * as `undefined`, which makes `NaN`.
   *
   * @canon bug-js-reads-a-uniform-array-element-the-host-leaves-out-as-nan
   */
  it.fails("reads a uniform array element the call leaves out as zero on JS", () => {
    const items = uniformArray("float", 3);
    const run = compileJSRoutine(() => Fn(() => items.element(int(2)).add(0).toVar())(), none);
    expect(run({ uniforms: { [items.name]: [1, 2] } })).toBe(0);
  });

  /**
   * `createJsCompute` reads a vector storage buffer as one array per element,
   * so the flat typed array `setAttribute` takes ends up as `NaN`.
   *
   * @canon bug-the-cpu-compute-adapters-take-a-vector-storage-element-as-an-array
   */
  it.fails("writes a vector storage buffer given as a flat typed array on JS", () => {
    const buf = instancedArray(2, "vec2");
    const adapter = createJsCompute(Fn(() => buf.element(invocationIndex()).assign(vec2(3, 4)))());
    const data = new Float32Array(4);
    adapter.setAttribute(buf.name, data);
    adapter.compute();
    expect(Array.from(data)).toEqual([3, 4, 3, 4]);
  });
});
