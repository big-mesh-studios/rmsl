# ecs

A tiny entity-component-system demo showing one RMSL `Fn` graph compiled
and run identically across all three CPU/GPU compute backends.

- `system.ts` — the one system this demo has: integrate position by
  velocity, bounce off the canvas edges. Built once as an RMSL `Fn` graph,
  and that single graph is what `compileJS`, `compileWasm` and
  `compileWgsl.compute` each compile, so all three backends run the exact
  same logic on different hardware. Written against
  `storage()`/`invocationIndex()` rather than
  `attribute()`/`uniform()`/`output()`: position and velocity are
  `read_write` storage arrays indexed by the current invocation, so a WGSL
  compute dispatch processes the whole entity buffer in one call, while
  `compileJS`/`compileWasm` call the same compiled function once per entity
  with `index` set to that entity's position (see `main.ts`'s `stepCPU`).
- `gpu-renderer.ts` — reads the WGSL adapter's own storage buffers directly
  as vertex data; the compute pass and this render pass share the same
  `GPUBuffer` objects, so drawing never round-trips position data through
  the CPU.
