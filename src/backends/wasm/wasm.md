# WASM backend architecture

This covers `wasm.ts`'s compile pipeline end to end — how an `Fn` becomes a
`WebAssembly.Module` and gets called back into from JS. `rasterizer.md`
covers the fixed rasterizer module this pipeline's output gets linked
against; read this one first, since the rasterizer's constraints (zero-arg
imports, one shared memory) only make sense in terms of the choices made
here.

## The two-pass compiler (`compileWasmFn`)

`compileWasmFn` never touches a WASM toolchain — it walks the `Fn`'s AST and
hand-encodes the module's bytes directly (see `WASM_OP`'s doc comment: every
opcode used is checked empirically, not quoted from a spec). It runs in two
passes over the same tree:

1. **`collect`** — walks the whole AST once, before any bytecode exists, and
   decides where every value the body references will live: a WASM param, a
   WASM local, or a fixed byte offset in linear memory. Nothing is emitted
   yet; this pass only populates the various `*Address`/`param`/`local` maps
   in `compileWasmFn`'s closure.
2. **The emit walk** (`walkExpr`/`walkStmt`/`finalValueBytes`) — walks the
   tree again, this time turning each node into WASM bytecode, looking up
   the slot `collect` already decided for every var/uniform/attribute/
   varying/storage reference. No planning happens here; by this point every
   address is frozen.

Splitting compilation this way means the emitter never has to backtrack —
`collect` has already answered "where does this value live" for every node
before the first opcode is written.

## Where a value lives: param, local, or memory

Three kinds of storage, each used for a different reason:

### WASM locals

A body-local variable (`const x = a.mul(2)`), always a local (`addLocal`).
Never host-visible; pure scratch inside the call.

### WASM params

A scalar `Fn` argument (`options.params`, see the `"var"` case in
`collect`) is always a real function parameter, `local.get`-readable,
regardless of any option. This path is unconditional: it has nothing to do
with `scalarsInMemory` below.

### Linear memory

A fixed byte address, written by the host before the call and/or read
after. Two things are *always* memory-resident, with no param path
available at all: aggregates (`vec3`, `mat4`, ... — a WASM function param
can only be one scalar, so there's no other option), and a fragment
stage's varying inputs (interpolated and written by the rasterizer
between the vertex and fragment calls — see `rasterizer.md`'s
vertex/triangle passes).

`uniform()`/`attribute()`/`varying()` calls are the interesting case: by
default, each distinct scalar one becomes an *implicit* extra WASM param —
appended after the explicit `Fn` params, invisible in the shader source
itself, but a real part of the compiled function's WASM-level signature
(`collect`'s `"uniform"`/`"attribute"`/`"varying"` cases, each gated on
`isAggregate(v.shaderType) || options.scalarsInMemory`). Turning on
`options.scalarsInMemory` forces every one of those into memory instead,
joining the two always-memory cases above. A plain `local.get` measurably
beats a memory read for a called-once function (`docs/wasm-benchmarks.md`),
which is why this isn't the default — it exists specifically so the
rasterizer (below) can force every shader onto one shared, zero-arg call
shape.

`storage()` is always memory-resident regardless of `scalarsInMemory` too,
for a different reason: a `read_write`/`write` storage has to be readable
back into `ctx.storages[slot][ctx.index]` after the call, which only a fixed
address makes possible.

## What the compiled module looks like

`compileWasmFn`'s output (`CompiledWasm`) is raw module bytes plus the
metadata a host needs to call it: `params` (the ordered list of WASM params
*and* memory-resident inputs — `WasmParam[]`, tagged by `kind`), the result
type, and how much linear memory the compile-time layout used
(`textureHeapBase`, appended-after for texture pixel data at instantiation
time).

The module always exports `main` (`options.name`) and, when the `Fn`'s
result isn't `void`, a second export, `batch(width, height, bufferBase)` —
a WASM-side pixel loop that calls `main` once per pixel (feeding it
`fragCoord()`) and writes results into the caller's buffer, so a whole-image
evaluation only pays the JS↔WASM marshalling cost once instead of once per
pixel (see "A whole grid in one call" in `docs/wasm-benchmarks.md`).

Memory is *imported*, not owned by the module (`env.memory`) — this is what
lets `instantiateWasmRoutine` hand multiple instances the same memory, and
is what lets the rasterizer's vertex and fragment modules share one memory
with each other and with the rasterizer module itself.

## From bytes to a callable (`instantiateWasmRoutine`)

`instantiateWasmRoutine` is deliberately a separate export from
`compileWasmFn`, not folded into `compileWasmRoutine`: a build step
(`precompileWasm`, `src/vite/vite.ts`) can compile ahead of time and ship
only the bytes plus this instantiation glue — never the AST walker that
produced them.

It does three things:

1. Instantiates the `WebAssembly.Module`, importing `math` (the host's own
   `Math` object, called by index for `sin`/`pow`/...) and `env.memory`
   (freshly allocated, sized for the compile-time layout, unless a memory
   was passed in).
2. Builds a `createWasmInputMarshaller` for the compiled `params` list: on
   each `invoke()`/`batch()` call, it walks that list and, per entry's
   `kind`, either pushes a JS number onto the WASM call's `args` (the
   `param`/`uniform`/`attribute`/`varying`/`invocationIndex` kinds) or
   writes the value into memory at its fixed address (every `*Memory` kind).
   Textures get the same treatment, appended into memory past
   `textureHeapBase`, with a cache that skips re-uploading an unchanged
   texture object between calls.
3. Wraps the result in `invoke`/`batch` (satisfying `CpuRoutine`): call the
   WASM export with the marshalled args, then read every memory-resident
   *output* (`outputMemory`/`varyingOutputMemory`/`positionMemory`/
   `fragDepthMemory`/`valueMemory`) back out of memory into a
   `CpuShaderResult` — or, when there are no memory outputs at all, just
   reinterpret the WASM call's own return value (with a `>>> 0` for `uint`,
   since the boundary always returns a signed i32).

`createWasmInputMarshaller` is exported on its own because it's shared: both
`instantiateWasmRoutine` (one call per `invoke()`/`batch()`) and the
rasterizer's `compileWasm` (below — one call per `draw()`) need the exact
same "`CpuShaderContext` → WASM args + memory writes" translation.

## The rasterizer: reusing the same compiled shaders, differently

`compileWasm(vertexFn, fragmentFn, options)` (`rasterizer.ts`) is not a
third compilation strategy — it calls `compileWasmFn` on both `Fn`s exactly
as `compileWasmRoutine` does, with `scalarsInMemory: true` forced on, and
then never calls `instantiateWasmRoutine` at all. Instead it instantiates
both compiled modules against one shared memory, and links their exported
`main` functions as the `vertex`/`fragment` imports of the separate, fixed
`rasterizer.wat` module (`RASTERIZER_VERTEX_IMPORT`/
`RASTERIZER_FRAGMENT_IMPORT` — see `rasterizer.md`). Forcing
`scalarsInMemory` is what makes that linking possible: the rasterizer's
`rasterize` export calls its imported vertex/fragment functions with a
fixed, zero-arg call shape it never varies per shader, so both compiled
`main`s have to present that same shape regardless of how many uniforms or
attributes the original `Fn` referenced.

Two marshallers still run per `draw()` call — one for the vertex module's
own uniforms/textures, one for the fragment module's — but each excludes
what the rasterizer itself now owns: the vertex marshaller skips
`attributeMemory` (the rasterizer's vertex pass pokes per-vertex attribute
bytes into that address itself, inside WASM, once per vertex — seeing the
whole call as one per-draw batch rather than one per-vertex call), and the
fragment marshaller skips `varyingMemory` (the rasterizer's triangle pass
writes interpolated varyings there per covered pixel).

`createWasmRoutine`/`createWasmCompute`/`createWasm` (`adapter-wasm.ts`)
are the outermost layer, each wrapping `compileWasmRoutine`/`compileWasm`
for one of three distinct shapes rather than living as options on a
single entry point: `createWasm` is the render-pipeline shape (a
vertex/fragment pair through the rasterizer), `createWasmCompute` is the
compute-pipeline shape (a `storage()`/`invocationIndex()` program,
exposed through its own `WasmComputeAdapter` — `setAttribute`/
`setUniform`/`compute()`, no `draw()`/`attach()` in the type at all), and
`createWasmRoutine` is what's left over — a plain CPU-callable
(`fragCoord()`-driven, `.batch()`'s in-WASM loop) with no wgpu pipeline
equivalent, the same niche `compileJS`/`compileWasmFn` exist for in the
first place. `createGlsl`/`createWgsl` implement the same `Adapter`
interface `createWasm`/`createWasmRoutine` do, so calling code doesn't
need to know which backend it got — `createWasmCompute` deliberately
doesn't, since a compute-pipeline-shaped adapter has no `draw()` to be
interchangeable about.

## Summary

```
Fn(s)
  │
  ├─ compileWasmFn ── collect (plan slots/addresses) ── emit (walk → bytecode)
  │                                    │
  │                     param / local / memory, decided per node
  │                                    │
  ▼                                    ▼
CompiledWasm (bytes + WasmParam[] + resultType + textureHeapBase)
  │
  ├─ instantiateWasmRoutine ─ one module, own memory ─ CpuRoutine (invoke/batch)
  │        used by:  compileWasmRoutine  (direct calls, no rasterizer)
  │             ├─ createWasmRoutine   (batch/fragCoord shape, Adapter wrapper)
  │             └─ createWasmCompute   (storage()/invocationIndex() shape, WasmComputeAdapter)
  │
  └─ compileWasm ─ two modules (scalarsInMemory: true), shared memory,
       linked as vertex/fragment imports into rasterizer.wat's `rasterize` ─
       WasmRasterRoutine (draw)
            used by: createWasm (Adapter wrapper)
```
