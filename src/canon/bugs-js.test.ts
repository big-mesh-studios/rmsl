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
  ivec2,
  mat2,
  textureLoad,
  uniform,
  vec3,
  vec4,
  type Node,
} from "../rmsl";
import { compileJS, compileJSFn, compileJSRoutine, createJsGrid } from "../js";

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
      compileJSRoutine(
        build as any,
        none,
      )({
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
});
