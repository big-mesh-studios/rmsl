# Generic rasterizer module

See `wasm.md` for the WASM backend's compile pipeline as a whole — this
module is the consumer of that pipeline's output, not a description of it.

Unlike the rest of the WASM backend
(`compileWasmFn`/`compileWasm` in `wasm.ts`, which compile an arbitrary,
dynamically-constructed shader graph at runtime and so can't depend on a
WASM toolchain), this module's structure never varies per shader — it's
authored directly as `rasterizer.wat`, and `rasterizer.ts` imports it
like any other module (`import RASTERIZER_WASM_BYTES from "./rasterizer.wat"`).
`compileWat` (`src/vite/vite.ts`) compiles `.wat` to bytes at build time
via `wabt`, wired into both `vite.config.ts` and `vitest.config.ts`; no
`.wat` source or `wabt` reference reaches the built `dist/wasm.js`.
`buildRasterizerModule()` returns those bytes — one fixed module,
compiled once regardless of shader, with a `"rasterize"` export
({@link RASTERIZE_PARAMS} in `rasterizer.ts` for the exact argument list
and order).
Given `true`, it returns the same module importing its memory shared, with
the largest maximum a 32-bit memory has, because a shared memory links only
against an import declared shared, and against one whose maximum is at least
its own. `compileWat` builds both modules from the one `.wat`.

`rasterize()` does three passes over one shared `env.memory`:

## Passes

### 1. Vertex pass

Loops `vertexCount` times: for each attribute descriptor in
`attrDescBase`/`attrDescCount`, byte-copies that slot's bytes from
`attrSrcBase` into the imported vertex function's own attribute address for
that slot, calls it, then byte-copies its written `vec4` position (and one
shared varying blob, `varyingBytes` long) out to
`positionsOutBase`/`varyingsOutBase`.

### 2. Clip pass

For each original (non-indexed) triangle, clips it against the homogeneous
near plane `w > W_CLIP_EPS` (Sutherland-Hodgman, single plane): a triangle
fully in front passes through unchanged, one fully behind is dropped, and
one straddling the plane is cut into a quad and fan-triangulated into two.
Cut vertices lerp both the clip-space position and the whole varying blob
linearly (not perspective-corrected — correct for clip space, pre-divide).
Results land in `clippedPositionsOutBase`/`clippedVaryingsOutBase`, using
`clipScratchBase` as scratch for the up-to-4-vertex intermediate polygon.

### 3. Triangle pass

For each clipped triangle: perspective divide and screen-space mapping, a
degenerate-area skip, a clamped bounding box, and per pixel in that box an
edge-function coverage test. A pixel centre that lies on an edge belongs to
the triangle whose edge, wound so its inside is positive, runs down the
screen or left along it, so a pixel on an edge two triangles share is shaded
once.

A covered pixel's depth is NDC `z/w`, interpolated with the plain
barycentric weights screen coordinates use, since it is affine in screen
space. A triangle wholly outside depth 0 to 1 draws nothing, and one that
crosses that range drops each pixel outside it, which is what clipping at
depth 0 and 1 drops. A pixel then:

1. fails early when its depth is farther than the stored one, unless the
   fragment function writes its own depth;
2. has the varying blob interpolated, perspective-correct, into the
   fragment function's varying-input address, its centre written to the
   `fragCoord` address, the discard flag cleared, and the interpolated depth
   written to the fragment depth address;
3. runs the fragment function;
4. is dropped when the function discarded, or when the depth it wrote is
   farther than the stored one;
5. stores its depth, and its `vec4` result into `outputBase`.

Each address the fragment function writes or reads besides its varyings
comes with a flag, 1 when the function has it and 0 when it does not:
`writesColour` for `fragmentValueAddress`, `writesFragCoord`, `writesDepth`
and `mayDiscard`. A function that writes no colour leaves the pixel as it
was. The host must pre-clear `depthBufferBase` to a large value before the
first draw over it, and again before a draw of another size.

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
- No index buffer, no antialiasing.
- Depth test is a plain LEQUAL z-buffer — no depth write mask, no
  stencil, no blending (a passing pixel always overwrites).
- Geometric clipping is near-plane (`w`) only; depth 0 to 1 is held per
  pixel, and the per-pixel bbox clamp handles screen bounds.
