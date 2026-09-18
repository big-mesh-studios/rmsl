# adapters

Three shapes, six adapters. Demonstrates that all four RMSL backends draw the
same rotating quad, despite having different native shapes:

- `createGlsl` (WebGL) and `createWgsl` (WebGPU) draw the quad directly,
  through their own vertex/fragment stages.
- `createJs`/`createWasm`'s own `draw` wraps a per-pixel `fragCoord()`
  program instead — a full-screen shape, not a quad — so `rasterizeTriangles`
  (`src/backends/cpu-rasterizer.ts`, prototype) is used to run the *same*
  vertex/fragment pair through `compileJS`/`compileWasm`'s "vertex" stage and
  a software rasterizer, drawing the identical quad entirely on the CPU.
