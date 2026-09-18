import { bench, describe } from "vitest";
import { compileJSRoutine, rasterizeTriangles } from "../js";
import { attribute, builtinPosition, cos, Fn, sin, uniform, varying, vec3, vec4 } from "../rmsl";
import { compileWasm } from "../wasm";

// A rotating, per-vertex-colored quad — the same scene apps/adapters'
// js-vtx/wasm-vtx demo draws, at a size and vertex count large enough to
// get a real signal (a single 2x2 quad at 512x512 is over 99% empty pixels).
for (const SIZE of [128, 512]) {
  describe(`rasterizeTriangles(compileJSRoutine) vs createWasm: a rotating quad, ${SIZE}x${SIZE}`, () => {
    const pos = attribute("vec2");
    const time = uniform("float");
    const vColor = varying("vec3");

    const vertexFn = () =>
      Fn(() => {
        const c = cos(time);
        const s = sin(time);
        const x = pos.x.mul(c).sub(pos.y.mul(s));
        const y = pos.x.mul(s).add(pos.y.mul(c));
        vColor.assign(vec3(pos.x.mul(0.5).add(0.5), pos.y.mul(0.5).add(0.5), sin(time).mul(0.5).add(0.5)));
        builtinPosition().assign(vec4(x, y, 0, 1));
      })();
    const fragmentFn = () => Fn(() => vec4(vColor, 1))();

    // A quad covering most of the viewport, drawn as a 6-vertex (2-triangle)
    // non-indexed list — the same TRIANGLE_STRIP_QUAD shape the demo uses.
    const QUAD = new Float64Array([-0.9, -0.9, 0.9, -0.9, -0.9, 0.9, 0.9, -0.9, -0.9, 0.9, 0.9, 0.9]);

    const jsVertex = compileJSRoutine(vertexFn as any, { name: "vtx", params: [], stage: "vertex" });
    const jsFragment = compileJSRoutine(fragmentFn as any, { name: "frag", params: [] });

    const wasmRoutine = compileWasm(vertexFn as any, fragmentFn as any);
    const wasmCtx = { attributes: { [pos.name]: QUAD }, uniforms: { [time.name]: 0.5 } };

    bench("rasterizeTriangles(compileJSRoutine) — host-mediated vertex/fragment loop", () => {
      rasterizeTriangles(jsVertex, jsFragment, {
        attributes: { [pos.name]: QUAD },
        attributeTypes: { [pos.name]: "vec2" },
        uniforms: { [time.name]: 0.5 },
        width: SIZE,
        height: SIZE,
        componentCount: 4,
      });
    });

    bench("createWasm — vertex/clip/triangle loop inside one WASM call", () => {
      wasmRoutine.draw(wasmCtx, 6, SIZE, SIZE, undefined, true);
    });
  });
}
