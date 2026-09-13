# A shared layout IR for uniforms, instance data, and textures

**Status: exploratory.** This is a design sketch, not a committed roadmap
item — nothing here is scheduled, and no phase number depends on it. It
came out of a conversation about where RMSL's CPU/GPU backend split could
go next, starting from the observation that a byte layout derived from a
`ShaderType` is a real code entity (`src/rmsl-wasm.ts`'s Phase 3 memory
design, `src/rmsl-wgsl.ts`'s `wgslUniformLayout`) that today gets computed
independently, and slightly differently, in more than one place.

## The problem

RMSL already compiles one node graph to four backends (GLSL, WGSL, JS,
WASM), but "the byte layout of a `vec3`/`mat4`/instance struct" is not one
of the things that gets shared across them. Three real, separate
implementations already exist:

- **`wgslUniformLayout`** (`src/rmsl-wgsl.ts:216`) computes WGSL's
  uniform-address-space layout rules (16-byte array-stride rounding, widened
  storage for anything too narrow to align) so the generated `struct` and
  whatever writes the actual uniform buffer agree on offsets. Its own
  comment notes three separate places already have to agree with each
  other by construction, not by sharing code.
- **Phase 3's WASM linear-memory allocator** (`src/rmsl-wasm.ts`,
  `allocateFor`/`componentSizeOf`/`elementKindOf`) computes a *different*
  layout for the same shader types: byte-packed, no padding, `align=0`
  everywhere, because nothing on the WASM side ever needed to match a GPU
  buffer's alignment rules — it only ever talks to itself and to
  `compileWasm`'s JS-side `DataView` writer.
- **Instance/attribute buffers** (`src/scene/geometries/BufferAttribute.ts`,
  and the per-instance packing in `WebGLRenderer.ts`/`WebGPURenderer.ts`)
  hand-roll a third notion of "stride and offset for this field," specific
  to vertex/instance buffers rather than uniform blocks.

None of these are wrong for what they do today. They're just three
independent answers to a question — "where does this value's Nth component
live in memory?" — that only depends on one thing: the `ShaderType`. The
idea explored here is making that question have one answer, computed once,
that every backend and every buffer-packing call site reads instead of
re-deriving.

## Sketch of the IR

Not a real API — a shape to argue about. For a given `ShaderType`:

```ts
type FieldLayout = {
  byteOffset: number;
  byteSize: number;
  componentKind: "f32" | "f64" | "i32" | "u32" | "bool32";
  componentCount: number;
};

type TypeLayout = {
  byteSize: number;
  align: number;               // 0 = unconstrained (today's WASM choice)
  fields: FieldLayout[];        // one per scalar component, in declared order
};

function layoutOf(type: ShaderType, rules: LayoutRules): TypeLayout;
```

`LayoutRules` is the part that has to stay pluggable, not unified away:
GPU uniform-address-space rules (std140-ish, WGSL's own variant), GPU
storage-buffer rules (std430-ish, tighter), and "no constraint, just pack
tightly" (what WASM and a plain JS array both want) are genuinely different
answers for the same type, not implementation accidents to paper over. The
win isn't "one layout for everything" — it's "one *function* that knows how
to compute any of them, instead of three hand-written ones that happen to
overlap."

Every current consumer becomes a thin renderer over `TypeLayout` instead of
its own layout logic:

- `wgslUniformLayout` becomes `layoutOf(type, WGSL_UNIFORM_RULES)` plus the
  existing struct-text emission.
- Phase 3's WASM allocator becomes `layoutOf(type, PACKED_RULES)` plus the
  existing bump allocator (which only needs `byteSize`/`align` from it, not
  a redesign of `allocateFor`).
- `BufferAttribute`/instance packing becomes `layoutOf(type,
  VERTEX_RULES)`.

## What this could open up

- **Zero-copy CPU→GPU data flow.** If a WASM module's linear memory used
  `layoutOf(type, GPU_STORAGE_RULES)` for a given buffer instead of the
  packed rules, the exact bytes `compileWasm` (or a future "compile a whole
  CPU stage" mode) writes could go straight into
  `device.queue.writeBuffer(gpuBuffer, 0, wasmMemory.buffer, offset,
  length)` — no JS-side repacking step between "WASM computed this" and
  "GPU can read this."
- **One stride/offset derivation for instancing**, instead of
  `BufferAttribute`/`WebGLRenderer`/`WebGPURenderer` each carrying their
  own version that can drift out of sync by hand-edit.
- **CPU reads of GPU-authored textures with guaranteed-matching layout.**
  Phase 6 (texture sampling, `ROADMAP.md`) already needs a real memory
  layout for WASM; if it's `layoutOf` under the GPU's own texture rules, a
  WASM-side heightmap raycast reads the identical bytes the GPU is
  sampling, with no separately-maintained CPU copy to keep in sync.
- **A cross-language ABI, not just a TS convenience.** `LayoutRules` are
  just data — a generated `#[repr(C)]` Rust struct or a C header could
  describe the same layout, which matters directly for the
  wgpu-native/"no JS engine" native-build idea from the same conversation
  this came out of.
- **Alignment-bug tooling.** std140's "why is my vec3 secretly 16 bytes"
  surprises stop being tribal knowledge per backend and become one thing a
  layout inspector can print, for any `LayoutRules`.

## Real tension to design around, not gloss over

Phase 3's WASM memory design explicitly chose **no padding, `align=0`
everywhere** (`ROADMAP.md`, "Vectors and matrices live in linear memory
now") — a deliberate simplification, correct because nothing on that side
ever needed to match a GPU buffer. Making WASM's layout GPU-compatible
*by default* would be a real, backwards-incompatible change to an
already-shipped, tested design, not a free generalization. Any version of
this that ships has to keep "tightly packed, CPU-only" as one of the
pluggable `LayoutRules` — the WASM backend should only switch to a
GPU-shaped layout for a specific buffer that's actually headed to the GPU,
not universally.

## Non-goals (for now)

- Not a proposal to change how any backend already declares or packs
  layout today — this is additive (a shared function multiple call sites
  *could* adopt) not a rewrite.
- Not the async GPU↔CPU scheduling problem from the same conversation
  (readback latency, auto-partitioning a graph across backends) — that's a
  separate, harder problem this doesn't attempt to solve, even though a
  shared layout is a prerequisite for it.
- Not scoped to a specific phase number in `ROADMAP.md` — it touches all
  four backends, not just WASM, so it doesn't belong under that
  WASM-specific roadmap.

## If this ever gets picked up

Smallest possible first step: extract `wgslUniformLayout`'s core
offset/stride computation and Phase 3's WASM `allocateFor` into calls to
one shared `layoutOf(type, rules)`, with each backend's existing
`LayoutRules` (WGSL uniform rules; WASM's packed rules) as the only two
`rules` values that exist at first — no GPU storage-buffer rules, no
instance-buffer unification, nothing cross-language yet. That proves the
one-function-many-rules shape holds before spending any effort on the
capabilities above.
