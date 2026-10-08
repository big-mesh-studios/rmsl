import { describe, expect, it } from "vitest";
import { Fn, float, instanceIndex, int, mat3, vec4, vertexIndex } from "../rmsl";
import { compileJSRoutine, compileJSVertex } from "../js";
import { compileWasmVertex } from "../wasm";

const param = { name: "main", params: [{ name: "a", type: "float" as const }] };
const none = { name: "main", params: [] };

describe("known bugs, each failing until its fix", () => {
  /**
   * @canon bug-the-cpu-targets-compile-no-index-accessors
   */
  it.fails("compiles vertexIndex and instanceIndex on the CPU targets", () => {
    const build = () => Fn(() => vec4(vertexIndex().toFloat(), instanceIndex().toFloat(), 0, 1))();
    expect(() => compileJSVertex(build, { ...none })).not.toThrow();
    expect(() => compileWasmVertex(build, { ...none })).not.toThrow();
  });
});
