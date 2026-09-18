# adapters

Six adapters, two shapes. Demonstrates that all four RMSL backends can draw
the same rotating, per-vertex-colored quad, despite having very different
native shapes:

- `createGlsl` (WebGL) and `createWgsl` (WebGPU) draw the quad directly,
  through their own vertex/fragment stages.
- `createJs`/`createWasm` link the same vertex/fragment pair against a
  generic triangle rasterizer instead of a GPU (near-plane clipping, a
  LEQUAL depth test) — `src/backends/js/rasterizer.ts` and
  `src/backends/wasm/rasterizer.ts` respectively, the plain-JS and
  real-WASM implementations of the same algorithm. The vertex loop,
  clipping, and rasterization all run inside the compiled callable, not
  host-mediated per vertex/pixel.
- `createJsRoutine`/`createWasmRoutine`'s own `batch` wraps a per-pixel
  `fragCoord()` program instead — a full-screen gradient, not a quad, with
  no vertex stage or attributes at all.
