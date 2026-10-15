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

`rasterize()` does two passes over one shared `env.memory`:

## Passes

### 1. Vertex pass

Loops `vertexCount` times: for each attribute descriptor in
`attrDescBase`/`attrDescCount`, byte-copies that slot's bytes from
`attrSrcBase` into the imported vertex function's own attribute address for
that slot, calls it, then byte-copies its written `vec4` position (and one
shared varying blob, `varyingBytes` long) out to
`positionsOutBase`/`varyingsOutBase`.

### 2. Triangle pass

Each original (non-indexed) triangle is drawn before the next, one of two
ways, as `rasterizer.ts` in the JS backend draws it.

**By its lines.** A triangle whose vertices do not lie in one plane with the
eye is tested pixel by pixel against its own lines. Each vertex is scaled by
a power of two that brings its largest component from 1 to 2, and a
component smaller than `2^-300` of that is taken as 0. The line opposite
vertex `e` is the cross product of the other two in homogeneous `(x, y, w)`,
kept as four exact terms for each coefficient (Dekker's product). A pixel
centre `(x + 0.5, y + 0.5)` has the ray `X = (2x + 1 - width) * height`,
`Y = (height - 2y - 1) * width`, `W = width * height`, whole numbers an f64
holds exactly. A line's value there is decided by a float filter, and within
its bound of rounding by summing its terms exactly as an expansion of floats
(Shewchuk's adaptive predicates), so a centre exactly on an edge goes by the
top-left rule at any size of target and any scale of the positions. The
point's `z` and `w - z`, each the sum over the vertices of that component
times the vertex's line, are decided the same way, so a centre exactly at
depth 0 or 1 is drawn. The depth is their ratio, and each vertex weighs its
line there times its scale over the greatest scale among the vertices whose
lines are not 0. A triangle with a vertex at `vec4(0)` and the other two in
the clip volume is drawn the same way, with that vertex at the centre at
depth 0, and all the weight. The pixels tested are those of the vertices'
projections when every `w` is above 0; otherwise those of the polygon the
clip leaves of the scaled vertices (below), or the whole target when a
vertex of that polygon has a `w` below `2^-24`, whose projection the rounding
of its cut could move by more than half a pixel.

**By the clip.** Any other triangle lies in one plane with the eye. It is
clipped to WebGPU's clip volume (Sutherland-Hodgman, one plane after
another: `w + x`, `w - x`, `w + y`, `w - y`, `z`, `w - z`, each kept where it
is at least 0), and draws only when the polygon left, at most 9 vertices,
has a vertex at `vec4(0)`, which lies at the centre and gives it an area.
Cut vertices lerp both the clip-space position and the whole varying blob
linearly, from the vertex the polygon reaches first. A cut vertex whose four
components are all no more than `2^-48` of the largest sum it took, a
component of the vertex it starts from and its step towards the other added
by size, is rounding alone, and becomes `vec4(0)`. One whose `w` alone is 0 keeps its other
components for the next plane to cut. The polygon lives in
`clipScratchBase`, two halves of 9 vertex slots each that the planes take
turns writing, and is fan-triangulated from its vertex at `vec4(0)` into
`clippedPositionsOutBase`/`clippedVaryingsOutBase`, room for 7 triangles.
Each triangle of the fan is rasterized by its projections, with `vec4(0)` at
the centre at depth 0. Perspective weighs a vertex by the inverse of its
`w`, so every fragment takes the varyings of the vertex at `vec4(0)`, on the
edge opposite it too, and the flat ones of the triangle's first vertex.

A covered pixel then:

1. fails early when its depth is farther than the stored one, unless the
   fragment function writes its own depth;
2. has the varying blob interpolated, each vertex by its weight over the sum
   of the weights, and each flat varying copied from the triangle's first
   vertex, into the fragment function's varying-input address, its
   centre written to the `fragCoord` address, the discard flag cleared, and
   the interpolated depth written to the fragment depth address;
3. runs the fragment function;
4. is dropped when the function discarded, or when the depth it wrote,
   clamped to 0 to 1, is farther than the stored one;
5. stores its depth, and its `vec4` result into `outputBase`.

Each address the fragment function writes or reads besides its varyings
comes with a flag, 1 when the function has it and 0 when it does not:
`writesColour` for `fragmentValueAddress`, `readsFragCoord`, `usesDepth`
and `mayDiscard`. `writesDepth` is 1 when the function assigns its depth,
and 0 when it only reads it. A function that writes no colour leaves the pixel as it
was. The host must pre-clear `depthBufferBase` to a large value before the
first draw over it, and again before a draw of another size.

`$byteCopy` (in `rasterizer.wat`) is the one raw-byte copy primitive both
passes reuse. Any number of attribute slots are supported via a runtime descriptor
table (`AttributeDescriptor`/`writeAttributeDescriptors`) — each entry is
`[srcOffset, destAddress, sizeBytes]`, letting the source buffer pack
slots in any per-vertex layout the host chooses. Varyings work the same
way, via `VaryingDescriptor`/`writeVaryingDescriptors` — each entry is
`[recordOffset, vertexSrcAddress, fragmentDestAddress, sizeBytes, flat]`,
since a varying (unlike an attribute) has two addresses to reconcile, one
per stage, matched by the shared node's slot name. A `flat` varying is
copied from the triangle's first vertex wherever the others interpolate: a
pixel takes it from the first vertex of the original triangle it lies in,
clipped or not.

## Scope (v1)

- Both the vertex and fragment module must be zero-arg/zero-return.
  Aggregates already compile that way; a scalar uniform/attribute/varying
  needs `compileWasmFn`'s `scalarsInMemory: true` option to force it into
  memory too instead of a real WASM function parameter, which this
  rasterizer's fixed-arity imports can't accept.
- No index buffer, no antialiasing.
- Depth test is a plain LEQUAL z-buffer — no depth write mask, no
  stencil, no blending (a passing pixel always overwrites).
- A triangle drawn by its lines is not clipped: its lines, depth 0 to 1
  and the target's bounds are held per pixel, and the clip only bounds the
  pixels of one with a vertex at or behind the eye. Only a triangle in one
  plane with the eye is drawn by the polygon the clip leaves.
