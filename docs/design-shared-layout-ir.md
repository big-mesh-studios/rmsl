# A shared layout IR for uniforms, instance data, and textures

**Status: exploratory, stage 1 landed, stage 2 landed and found a real
gap.** This started as a design sketch, not a committed roadmap item — no
phase number depends on it. It came out of a conversation about where
RMSL's CPU/GPU backend split could go next, starting from the observation
that a byte layout derived from a `ShaderType` is a real code entity
(`src/rmsl-wasm.ts`'s Phase 3 memory design, `src/rmsl-wgsl.ts`'s
`wgslUniformLayout`) that today gets computed independently, and slightly
differently, in more than one place. Stage 1 (the behavior-preserving
refactor, `src/rmsl-layout.ts`) proved the placement *algorithm* can be
shared. Stage 2 (below) tried to prove the actual interop claim — a WASM
computation's uniforms landing at the exact byte offsets a real WGSL
uniform buffer would use — and found it's only half true: offsets can
match exactly, but this backend's `float` storage (f64) is twice the width
`wgslUniformLayout`'s offsets assume (`f32`), so two GPU-adjacent uniforms
placed at those offsets can end up overlapping in this backend's actual
writes and corrupting each other. The zero-copy capability this doc
originally pitched does **not** hold today without also closing that
width gap — see "Stage 2" below for what was actually built and found.

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
- **`WebGLRenderer.ts`'s vertex/instance attribute stride** computes a
  third, simpler version of the same "how big is this value" question
  (`stride = attr.itemSize * format.bytes`) — simpler because this codebase
  doesn't interleave multiple attributes into one buffer today (confirmed
  in the code itself: "One buffer carries one attribute... An interleaved
  attribute would take both numbers from the attribute rather than deriving
  them here"), so there's no multi-member placement problem here yet, only
  a third place computing one value's own byte size. GLSL/WebGL uniforms
  don't go through a packed buffer at all — `rmsl-glsl.ts` emits one
  `uniform` declaration per value, set individually via `gl.uniformXfv`,
  so there's nothing to pack there either, unless this ever adopts WebGL2
  uniform buffer objects.

None of these are wrong for what they do today. They're just three
independent answers to a question — "where does this value's Nth component
live in memory?" — that only depends on one thing: the `ShaderType`. The
idea explored here is making that question have one answer, computed once,
that every backend and every buffer-packing call site reads instead of
re-deriving.

## Sketch of the allocator

Not a real API — a shape to argue about. The first sketch of this
(`layoutOf(type, rules)`) computed one type's own size and alignment in
isolation. That's too narrow: `wgslUniformLayout` isn't just a size table —
it's an *allocator* over a whole list of heterogeneous members, and the
part of it worth sharing is the placement algorithm, not just a lookup.

```ts
type Member = { slot: string; type: ShaderType; length?: number };
type PlacedMember = Member & { offset: number; size: number; stride?: number };

type AllocRules = {
  sizeAndAlignOf(type: ShaderType): { size: number; align: number };
  reorderByAlignment: boolean;       // WGSL: true, to minimize padding.
  widenNarrowArrayElements: boolean; // WGSL's f32[] -> vec4<f32>[] quirk.
  arrayStrideRoundedTo?: number;     // WGSL: 16. Packed/CPU rules: none.
  structAlignMinimum: number;        // WGSL: 4. Packed/CPU rules: 1.
};

function planLayout(
  members: Member[],
  rules: AllocRules,
): { members: PlacedMember[]; size: number; align: number };
```

`AllocRules` is the part that has to stay pluggable, not unified away: GPU
uniform-address-space rules (std140-ish, WGSL's own variant), GPU
storage-buffer rules (std430-ish, tighter), and "no constraint, just pack
tightly in declaration order" (what WASM and a plain JS array both want)
are genuinely different answers, not implementation accidents to paper
over. The win isn't "one layout for everything" — it's "one *allocator*
that knows how to run any of them, instead of one hand-written struct
packer (WGSL) and one hand-written bump allocator (WASM) that happen to
overlap in what they're actually deciding."

Every current consumer becomes a thin wrapper over `planLayout` instead of
its own placement logic:

- `wgslUniformLayout` becomes `planLayout(members, WGSL_UNIFORM_RULES)`
  plus the existing struct-text emission — the same reordering and
  array-widening it already does, just factored out as data (`AllocRules`)
  instead of hard-coded into the function.
- Phase 3's WASM allocator becomes `planLayout(members, PACKED_RULES)`
  (`reorderByAlignment: false`, matching its current declaration-order
  bump behavior) feeding the existing address maps — no change in what
  addresses it hands out today.
- `BufferAttribute`/instance packing becomes `planLayout(members,
  VERTEX_RULES)`.

The capability this unlocks that a per-type-only sketch couldn't: a WASM
computation that needs to feed a *specific* WGSL uniform struct can call
`planLayout(sameMembers, WGSL_UNIFORM_RULES)` itself — same reordering,
same widening, same offsets `wgslUniformLayout` would produce for that
struct — and write its output there, instead of computing its own
(differently-ordered) packed layout and hoping it happens to match.

## What this could open up

- **Zero-copy CPU→GPU data flow.** If a WASM module allocated a given
  buffer via `planLayout(members, GPU_STORAGE_RULES)` instead of its own
  packed rules, the exact bytes `compileWasm` (or a future "compile a whole
  CPU stage" mode) writes could go straight into
  `device.queue.writeBuffer(gpuBuffer, 0, wasmMemory.buffer, offset,
  length)` — no JS-side repacking step between "WASM computed this" and
  "GPU can read this."
- **One placement algorithm for instancing**, instead of
  `BufferAttribute`/`WebGLRenderer`/`WebGPURenderer` each carrying their
  own version that can drift out of sync by hand-edit.
- **CPU reads of GPU-authored textures with guaranteed-matching layout.**
  Phase 6 (texture sampling, `ROADMAP.md`) already needs a real memory
  layout for WASM; if it's planned under the GPU's own texture rules, a
  WASM-side heightmap raycast reads the identical bytes the GPU is
  sampling, with no separately-maintained CPU copy to keep in sync.
- **A cross-language ABI, not just a TS convenience.** `AllocRules` are
  just data — a generated `#[repr(C)]` Rust struct or a C header could
  describe the same layout, which matters directly for the
  wgpu-native/"no JS engine" native-build idea from the same conversation
  this came out of.
- **Alignment-bug tooling.** std140's "why is my vec3 secretly 16 bytes"
  surprises stop being tribal knowledge per backend and become one thing a
  layout inspector can print, for any `AllocRules`.

**What this does *not* give you: one live GPU buffer shared between WebGL
and WebGPU.** Those are separate browser APIs with separate buffer objects
(`WebGLBuffer` vs `GPUBuffer`) — there is no browser API to hand one GPU
allocation to both, no matter how identical the layout is. What a shared
allocator gets you there is narrower but still real: compute the packed
bytes once and have `gl.bufferData()` and `queue.writeBuffer()` both read
from that same source, instead of two hand-written, independently-drifting
packing routines for a material that has to render through both backends.
True zero-copy sharing of one GPU allocation between two different GPU
APIs is a real capability (DMA-BUF, Vulkan/D3D external-memory extensions,
`IOSurface`) — just not one reachable from a browser tab; it only exists
off the web platform entirely, which is the native/wgpu-native thread from
the same conversation, not something this design touches.

## Real tension to design around, not gloss over

Phase 3's WASM memory design explicitly chose **no padding, `align=0`
everywhere, declaration order, no reordering** (`ROADMAP.md`, "Vectors and
matrices live in linear memory now") — a deliberate simplification, correct
because nothing on that side ever needed to match a GPU buffer. Making
WASM's layout GPU-compatible *by default* — including running the same
alignment-driven reordering WGSL does — would be a real,
backwards-incompatible change to an already-shipped, tested design's
addresses, not a free generalization. Any version of this that ships has
to keep "tightly packed, declaration order, CPU-only" as one of the
pluggable `AllocRules` — the WASM backend should only opt a *specific*
buffer into GPU-shaped placement (and thus GPU-shaped reordering) when
that buffer is actually headed to the GPU, never universally, and never by
changing what address an existing packed-only value gets today.

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

## Stage 1 — landed

`src/rmsl-layout.ts` now has `planLayout(members, rules)`.
`wgslUniformLayout` (`src/rmsl-wgsl.ts`) is a thin wrapper over
`planLayout(members, WGSL_UNIFORM_RULES)` — same reordering, same
array-widening, same offsets it always produced, now expressed as an
`AllocRules` value instead of hard-coded into the function. Phase 3's WASM
`allocateFor` (`src/rmsl-wasm.ts`) calls
`planLayout([{slot: t, type: t}], PACKED_RULES)` — a single-member list,
not a batch of everything `collect()` discovers, which is a deliberately
smaller change than first planned: `collect()` walks the AST and
discovers uniforms/vars/scratch nodes one at a time as it encounters them,
so batching them into one `planLayout` call would mean restructuring that
walk into two passes (discover the full list, *then* place it) — real
extra risk for a change meant to be behavior-preserving. A single-member
call to the same shared algorithm gets identical addresses (with
`reorderByAlignment: false`, a length-one list can't be reordered) at much
lower risk, and still means both backends' sizing rules live in one place
instead of two. Confirmed behavior-preserving by the full existing test
suite passing unchanged — no new test needed, since nothing observable was
supposed to change.

The type-vocabulary mismatch flagged above is still unresolved by this
step, on purpose: `wgslUniformLayout`'s callers already hand it WGSL-spelled
type strings (`"vec3<f32>"`), Phase 3's callers already use RMSL's own
`ShaderType` (`"vec3"`), and `planLayout` stays agnostic — a member's
`type` is an opaque string its own `rules.sizeAndAlignOf` interprets
however that target already does. Unifying that vocabulary is exactly what
stage 2 would force, since sharing one `members` list between a WGSL call
and a WASM call means both need to agree on how a type is spelled.

## Stage 2 — landed, claim half-confirmed

`compileWasmFn`/`compileWasm` gained an experimental option,
`gpuUniformLayout: { offsets, totalSize }` (`src/rmsl-wasm.ts`), that
places specific aggregate uniforms at caller-given byte offsets instead of
this backend's own packed allocation — everything else it needs (locals,
scratch, non-overridden uniforms) starts its own bump-allocated region
right after `totalSize`, so it can never collide with an overridden
address. `src/rmsl-layout-interop.test.ts` is the actual proof:

- **What holds**: given a real `wgslUniformLayout` computation for two
  differently-aligned uniforms (a `vec3` and a `vec2`, deliberately chosen
  so WGSL's alignment rules reorder them), the WASM backend's own
  `WasmParam` addresses come back *exactly* equal to `wgslUniformLayout`'s
  offsets — not re-derived, not coincidentally equal for a trivial
  single-member case, and correctly reordered (the `vec3` lands before the
  `vec2` despite being declared second).
- **What doesn't**: `wgslUniformLayout`'s offsets assume each `float`
  component is WGSL's 4-byte `f32`; this backend always stores `float` as
  an 8-byte f64 (`ROADMAP.md`, "`float` is f64"). A `vec3` at GPU offset 0
  occupies 12 bytes in a real WGSL buffer but 24 in this backend's actual
  memory — enough to spill straight over a `vec2` GPU-placed at offset 16,
  the very next member. The second test in that file demonstrates this
  concretely: real uniform values go in, and the computed result comes
  back neither correct nor derived from those values at all, because one
  uniform's write clobbered the other's before it was ever read.

So the zero-copy capability pitched earlier in this doc is **not achieved
by stage 2 alone** — matching offsets was necessary but not sufficient.
Closing the gap needs this backend to store a GPU-bound `float` as an f32
(new `f32.load`/`f32.store` WASM opcodes, and a conversion wherever such a
value re-enters this backend's otherwise-all-f64 internal computation) —
a separate, materially bigger change than an address override, not
attempted here. `GpuUniformLayout`'s doc comment in `rmsl-wasm.ts` carries
this warning directly, since the option as it exists today is unsafe to
use with real `wgslUniformLayout` offsets whenever two members end up
GPU-adjacent closer together than this backend's f64 spacing needs.

No GPU storage-buffer rules, no instance-buffer unification, nothing
cross-language yet, and no f32 storage — all of those wait on the width
gap above actually getting closed, which is the real next step if this
gets picked up again, not any of the items originally listed here.
