# Generic rasterizer module

See `ROADMAP.md`'s "generic, precompiled rasterizer module" open question
for the design this implements. `buildRasterizerModule()` builds one
fixed WASM module, compiled once regardless of shader, with a
`"rasterize"` export ({@link RASTERIZE_PARAMS} in `rasterizer.ts` for the
exact argument list and order).

`rasterize()` does two passes over one shared `env.memory`:

1. **Vertex pass** — loops `vertexCount` times: for each attribute
   descriptor in `attrDescBase`/`attrDescCount`, byte-copies that slot's
   bytes from `attrSrcBase` into the imported vertex function's own
   attribute address for that slot, calls it, then byte-copies its
   written `vec4` position (and one shared varying blob, `varyingBytes`
   long) out to `positionsOutBase`/`varyingsOutBase`.
2. **Triangle pass** — for each non-indexed triangle: perspective divide
   and screen-space mapping, a degenerate-area skip, a clamped bounding
   box, and per pixel in that box an edge-function coverage test.
   Covered pixels perspective-correct interpolate the varying blob into
   the imported fragment function's varying-input address, call it, and
   byte-copy its `vec4` result into `outputBase`.

`emitByteCopyLoop` is the one raw-byte copy primitive both passes reuse.
Any number of attribute slots are supported via a runtime descriptor
table (`AttributeDescriptor`/`writeAttributeDescriptors`) — each entry is
`[srcOffset, destAddress, sizeBytes]`, letting the source buffer pack
slots in any per-vertex layout the host chooses. Varyings work the same
way, via `VaryingDescriptor`/`writeVaryingDescriptors` — each entry is
`[recordOffset, vertexSrcAddress, fragmentDestAddress, sizeBytes]`, since
a varying (unlike an attribute) has two addresses to reconcile, one per
stage, matched by the shared node's slot name.

## Scope (v1)

- Every attribute/varying must be an aggregate (vector) type. A scalar
  (non-aggregate) uniform/attribute/varying becomes a real WASM function
  parameter even in an otherwise zero-arg program (see the next bullet),
  which this rasterizer can't drive.
- No index buffer, no near/far clipping, no depth test, no antialiasing —
  same gaps `ROADMAP.md`'s rasterizer checklist already tracks for
  `cpu-rasterizer.ts`'s own `rasterizeTriangles`.
- Both the vertex and fragment module must be zero-arg/zero-return
  (`compileWasmFn`'s shape whenever every uniform/attribute/varying is
  memory-resident) — a scalar-only fragment program isn't drivable here.
