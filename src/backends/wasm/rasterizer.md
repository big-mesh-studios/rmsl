# Generic rasterizer module

See `ROADMAP.md`'s "generic, precompiled rasterizer module" open question
for the design this implements. Unlike the rest of the WASM backend
(`compileWasmFn`/`compileWasm` in `wasm.ts`, which compile an arbitrary,
dynamically-constructed shader graph at runtime and so can't depend on a
WASM toolchain), this module's structure never varies per shader — it's
authored directly as `rasterizer.wat`, and `rasterizer.ts` imports it
like any other module (`import RASTERIZER_WASM_BYTES from "./rasterizer.wat"`).
`compileWat` (`src/vite/vite.ts`) compiles `.wat` to bytes at build time
via `wabt`, wired into both `vite.config.ts` and `vitest.config.ts`; no
`.wat` source or `wabt` reference reaches the built `dist/wasm.js`.
`buildRasterizerModule()` just returns those bytes — one fixed module,
compiled once regardless of shader, with a `"rasterize"` export
({@link RASTERIZE_PARAMS} in `rasterizer.ts` for the exact argument list
and order).

`rasterize()` does three passes over one shared `env.memory`:

1. **Vertex pass** — loops `vertexCount` times: for each attribute
   descriptor in `attrDescBase`/`attrDescCount`, byte-copies that slot's
   bytes from `attrSrcBase` into the imported vertex function's own
   attribute address for that slot, calls it, then byte-copies its
   written `vec4` position (and one shared varying blob, `varyingBytes`
   long) out to `positionsOutBase`/`varyingsOutBase`.
2. **Clip pass** — for each original (non-indexed) triangle, clips it
   against the homogeneous near plane `w > W_CLIP_EPS` (Sutherland-
   Hodgman, single plane): a triangle fully in front passes through
   unchanged, one fully behind is dropped, and one straddling the plane
   is cut into a quad and fan-triangulated into two. Cut vertices lerp
   both the clip-space position and the whole varying blob linearly (not
   perspective-corrected — correct for clip space, pre-divide). Results
   land in `clippedPositionsOutBase`/`clippedVaryingsOutBase`, using
   `clipScratchBase` as scratch for the up-to-4-vertex intermediate
   polygon.
3. **Triangle pass** — for each clipped triangle: perspective divide and
   screen-space mapping, a degenerate-area skip, a clamped bounding box,
   and per pixel in that box an edge-function coverage test. A covered
   pixel's depth (NDC `z/w`, interpolated with the same plain barycentric
   weights screen coordinates use — already affine in screen space, no
   perspective correction needed) is compared against `depthBufferBase`;
   only a closer-or-equal pixel writes both the new depth and, after
   perspective-correct interpolating the varying blob into the imported
   fragment function's varying-input address and calling it, its `vec4`
   result into `outputBase`. The host must pre-clear `depthBufferBase` to
   a large value before the first draw over it.

`$byteCopy` (in `rasterizer.wat`) is the one raw-byte copy primitive both
passes reuse. Any number of attribute slots are supported via a runtime descriptor
table (`AttributeDescriptor`/`writeAttributeDescriptors`) — each entry is
`[srcOffset, destAddress, sizeBytes]`, letting the source buffer pack
slots in any per-vertex layout the host chooses. Varyings work the same
way, via `VaryingDescriptor`/`writeVaryingDescriptors` — each entry is
`[recordOffset, vertexSrcAddress, fragmentDestAddress, sizeBytes]`, since
a varying (unlike an attribute) has two addresses to reconcile, one per
stage, matched by the shared node's slot name.

## Scope (v1)

- Both the vertex and fragment module must be zero-arg/zero-return.
  Aggregates already compile that way; a scalar uniform/attribute/varying
  needs `compileWasmFn`'s `scalarsInMemory: true` option to force it into
  memory too instead of a real WASM function parameter, which this
  rasterizer's fixed-arity imports can't accept.
- No index buffer, no antialiasing — same gaps `ROADMAP.md`'s rasterizer
  checklist already tracks for `cpu-rasterizer.ts`'s own
  `rasterizeTriangles`.
- Depth test is a plain LEQUAL z-buffer — no depth write mask, no
  stencil, no blending (a passing pixel always overwrites).
- Clipping is near-plane (`w`) only — no far-plane or screen-bounds
  frustum clipping (the per-pixel bbox clamp still handles screen
  bounds, as before).
