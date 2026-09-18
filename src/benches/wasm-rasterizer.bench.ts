import { bench, describe } from "vitest";
import { compileJS } from "../js";
import { attribute, builtinPosition, cos, Fn, sin, uniform, varying, vec3, vec4 } from "../rmsl";
import { compileWasm } from "../wasm";

// A rotating, per-vertex-colored quad — the same scene apps/adapters'
// js-vtx/wasm-vtx demo draws, at a size and vertex count large enough to
// get a real signal (a single 2x2 quad at 512x512 is over 99% empty pixels).
// Both sides now have the same near-plane-clip + LEQUAL-depth-test scope —
// compileJS ported rasterizer.wat's algorithm to plain JS (see
// src/backends/js/rasterizer.ts), so this compares the two real
// rasterizer implementations, not compileJS against the older, simpler
// rasterizeTriangles (see this file's own history for that comparison).
for (const SIZE of [128, 512]) {
  describe(`compileJS vs createWasm: a rotating quad, ${SIZE}x${SIZE}`, () => {
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

    const jsRoutine = compileJS(vertexFn as any, fragmentFn as any, { attributeTypes: { [pos.name]: "vec2" } });
    const wasmRoutine = compileWasm(vertexFn as any, fragmentFn as any);
    const ctx = { attributes: { [pos.name]: QUAD }, uniforms: { [time.name]: 0.5 } };

    bench("compileJS — vertex/clip/triangle loop in plain JS", () => {
      jsRoutine.draw(ctx, 6, SIZE, SIZE, undefined, true);
    });

    bench("createWasm — vertex/clip/triangle loop inside one WASM call", () => {
      wasmRoutine.draw(ctx, 6, SIZE, SIZE, undefined, true);
    });
  });
}
