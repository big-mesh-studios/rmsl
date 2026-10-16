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

The host lays out one record for each vertex the draw names, once however
many indices name it (`TriangleAssembly` in `src/backends/assemble.ts`
decides which), with an attribute past the end of its array written as
zero. The pass loops `vertexCount` times, once per record: for each attribute descriptor in
`attrDescBase`/`attrDescCount`, byte-copies that slot's bytes from
`attrSrcBase` into the imported vertex function's own attribute address for
that slot, calls it, then byte-copies its written `vec4` position (and one
shared varying blob, `varyingBytes` long) out to
`positionsOutBase`/`varyingsOutBase`.

### 2. Triangle pass

The triangles come from `triangleBase`: `triangleCount` entries of three
`i32` vertex records each, which the host assembles from the draw's
topology and indices. Each is wound as WebGPU winds it, with an odd strip
triangle reversed, and its first vertex gives its flat varyings. A
`triangleBase` of 0 takes each three records in turn instead.

Each triangle is drawn before the next, one of two ways, as `rasterizer.ts`
in the JS backend draws it. A draw that culls a face (`cullSign`, the sign
of the screen area such a face has, or 0) skips a triangle drawn by its
lines whose sign, which is opposite to its screen area's as y runs down the
screen, is opposite to `cullSign`, and likewise a triangle in one plane with
the eye whose drawn triangle winds so.

**By its lines.** A triangle whose vertices do not lie in one plane with the
eye is tested pixel by pixel against its own lines. Each vertex is scaled by
a power of two that brings its largest component from 1 to 2, and a
component smaller than `2^-300` of that is taken as 0. The line opposite
vertex `e` is the cross product of the other two in homogeneous `(x, y, w)`,
kept as four exact terms for each coefficient (Dekker's product). A pixel
centre `(x + 0.5, y + 0.5)` has the ray `X = (2x + 1 - width) * height`,
`Y = (height - 2y - 1) * width`, `W = width * height`, whole numbers an f64
holds exactly. A line's value there is the float sum of its coefficients
times the ray when that is at least `2^-16` of its terms' sizes, `2^32` times
its bound of rounding; otherwise its terms are summed exactly as an expansion
of floats (Shewchuk's adaptive predicates), whose sign is the line's and whose
float sum is its value. So a centre exactly on an edge goes by the top-left
rule at any size of target and any scale of the positions. The point's `z` and
`w - z`, each the sum over the vertices of that component times the vertex's
line, are decided the same way, so a centre exactly at depth 0 or 1 is drawn;
a triangle whose vertices all lie at a depth from 0 to 1 in front of the eye
needs neither test, as every point inside its lines lies there too. The part
of each line's value that a row of pixels shares is computed once per row.
The depth is a plane on the screen, the point's `z` along the ray over the
triangle's determinant times `W`, its three coefficients each summed exactly
once per triangle, and held to 0 to 1, where those signs put the point. Each
vertex weighs its line there times its scale over the greatest scale among the
vertices whose lines are not 0. A triangle with a vertex at `vec4(0)` and the other two in
the clip volume is drawn the same way, with that vertex at the centre at
depth 0, and all the weight. The pixels tested are those of the vertices'
projections when every `w` is above 0; otherwise those of the polygon the
clip leaves of the scaled vertices (below), or the whole target when a
vertex of that polygon has a `w` below `2^-24`, whose projection the rounding
of its cut could move by more than half a pixel.

**The clip.** A triangle drawn by its lines with a vertex at or behind the
eye is clipped to WebGPU's clip volume (Sutherland-Hodgman, one plane after
another: `w + x`, `w - x`, `w + y`, `w - y`, `z`, `w - z`, each kept where it
is at least 0) only to bound the pixels it tests. Cut vertices lerp both the
clip-space position and the whole varying blob linearly, from the vertex the
polygon reaches first. A cut vertex whose four components are all no more
than `2^-48` of the largest sum it took, a component of the vertex it starts
from and its step towards the other added by size, is rounding alone, and
becomes `vec4(0)`. One whose `w` alone is 0 keeps its other components for
the next plane to cut. The polygon lives in `clipScratchBase`, two halves of
9 vertex slots each that the planes take turns writing.

**In one plane with the eye.** Any other triangle lies in one plane with the
eye: its `x`, `y` and `w` lie in one plane through the eye, so every line is
a multiple of that plane, and the `w` coefficient of each, the cross product
of the other two vertices' `x` and `y`, is the weight of its vertex in the
triangle's point of `x`, `y` and `w` 0, times one factor. The triangle draws
only when those weights share a sign and its vertices' `z` by them sum to 0
exactly: then it contains `vec4(0)`, decided from the given vertices and not
from any cut. It draws the triangle on the screen from the centre to the two
ends of the segment that the rest of its part inside the clip volume
projects to. A point of that plane is placed by its `x` and `y`, so the one
on a pixel's ray by the ray's `X` and `Y`: by a line `(a, b, c)` whose weight
is not 0, its `w` is `-(aX + bY) / c` and its `z` is `(dX X + dY Y) / c`. A
centre is covered where that point's `w` is at most `W`, the point lies in
the clip volume (`w ± x`, `w ± y`, `z` and `w - z` at least 0), and it lies
within each edge of the triangle that runs through `vec4(0)`: at most 10
sides, each a value of the ray whose coefficients are eight exact terms,
decided as a line is, by its float value outside `2^-48` of its terms' sizes
and by its exact sum within, a centre on a side going by the top-left rule.
The depth is `(dX X + dY Y) / (c W)`, summed exactly where its float value's
bound of rounding passes `2^-32` of `c W`. Every fragment takes the varyings
of the triangle's point at `vec4(0)`, each vertex weighed by its line's `w`
coefficient over its own scale, on every side too, and the flat ones of the triangle's first
vertex. The sides follow the lines in the lines block, from `+6824`.

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
- No antialiasing. Indices, strips and restarts are assembled by the host;
  the module sees only the triangle list.
- Depth test is a plain LEQUAL z-buffer — no depth write mask, no
  stencil, no blending (a passing pixel always overwrites).
- No triangle is drawn by the polygon the clip leaves. A triangle drawn by
  its lines holds its lines, depth 0 to 1 and the target's bounds per pixel,
  and the clip only bounds the pixels of one with a vertex at or behind the
  eye. A triangle in one plane with the eye holds the sides of what it draws
  per pixel, over the whole target.
