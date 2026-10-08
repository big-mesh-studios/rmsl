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
  mat2,
  uniform,
  vec3,
  vec4,
  type Node,
} from "../rmsl";
import { compileJS, compileJSFn, compileJSRoutine, createJsGrid } from "../js";

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
});
