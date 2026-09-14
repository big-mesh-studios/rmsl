# uniformArray for the WASM backend — design

## Problem

`uniformArray(type, length)` already works in the GLSL, WGSL, and JS
backends. The WASM backend (`src/backends/rmsl-wasm.ts`) is the only one
that throws: both a `uniformArray` node and a `uniformArrayElement` node
fall through to the `default` cases of `walkExpr` / `materializeIfNeeded`
and throw `[RMSL] compileWasmFn: unsupported node type ...`.

ROADMAP.md anticipated this. Phase 3 deliberately deferred it: "the memory
design leaves room for it — an array element's address just needs a
dynamically-computed offset, which nothing in this phase's scope required."

## Node graph facts (already true, not new)

- `uniformArray(shaderType, length)` produces a node
  `{ type: "uniformArray", _t: shaderType, value: { id, slot, shaderType, length }, name: slot }`,
  plus an `.element(index)` closure. The uniform-array node itself is not an
  expression; only `.element()` opens it.
- `uniformArrayElement` is `{ type: "uniformArrayElement", _t: shaderType, params: [arrayNode, indexNode] }`.
  `params[0]` is the uniform-array node; `params[1]` is the index.
- A bare JS number index becomes a **float** node (`wrapValue`), so a
  backend must convert it (`int(2.0)`, `i32(2f)`). `int(...)`/`uint(...)`
  indices are already integer nodes.
- Array elements are always **reads**. There is no assign path for a
  uniform-array element (uniform buffers are read-only in every backend).
- A `uniformArrayElement` may be the function's root result
  (aggregate or scalar), which routes through `finalValueBytes`.

## Approach chosen: element-read scratch + dynamic loads (A)

Extend the existing scratch/address machinery rather than introduce a new
addressing concept.

- Give each aggregate-typed `uniformArrayElement` node a per-element scratch
  address (the existing `isScratchNode`/`scratchAddress` WeakMap pattern);
  keep `nodeAddress`'s hard `number` contract.
- Emit dynamic loads (`loadDynamic`, already used for the texture heap) from
  `base + trunc(index) * stride + k * rawCompSize` into the scratch
  (aggregate) or onto the WASM stack (scalar).
- Host-side, marshal the whole array into linear memory before each call,
  like aggregate uniforms already are.

Rejected alternatives:

- **B — a general "dynamic-address" node.** Would change `nodeAddress`'s
  `number` contract and touch far more of the file, to serve hypothetical
  features (`buffer()`, mutable array locals) nothing needs today. YAGNI.
- **C — host-side GPU unpack** (copy the f32/GPU-stride buffer into the
  packed f64 region every call). Re-copies the whole array when one element
  is read, diverges from the backend's lazy promote-on-read design for
  plain GPU uniforms, and silently nearest-f32s values.

## Compiler internals

### A helper for a runtime element address

New module-level helper emitting bytes that compute
`base + index_i32 * elementStride` (the element's first component):

```
addr(node, indexBytes) = [i32Const(base), indexBytes, i32Const(elementStride), i32Mul, i32Add]
```

using `loadDynamic`'s recipe (rmsl-wasm.ts:432). Component `k` adds
`k * rawCompSize` (`rawCompSize` is 4 for a narrow float element, else
`componentSizeOf(kind)`).

Index conversion (matches GLSL `int(2.0)` / WGSL `i32(2f)`):

- `float` index → `[...walkExpr(index), WASM_OP.i32TruncF64S]` — the same
  opcode the existing `construct` float→int cast already uses.
- `int`/`uint`/`bool` index → `walkExpr(index)` unchanged (already i32).
- No bounds check — an OOB index reads whatever bytes are at the computed
  address, identical to the other backends' silence.

### `collect`

- New `"uniformArray"` case: record
  `uniformArrayInfo[slot] = { base, elementStride, narrow }` and push a
  `uniformArrayMemory` WasmParam (see "Data model"). For the packed (non-GPU)
  case, `base = allocateBytes(length * elementSize)` where
  `elementSize = componentCountOf(t) * componentSizeOf(elementKindOf(t))`,
  `elementStride = elementSize`, `narrow = false`.
- `isScratchNode`: return `true` for `node.type === "uniformArrayElement"`
  when `isAggregate(node._t)`. The existing scratch-allocating tail of
  `collect` then gives each such node a one-element scratch automatically.

### `materializeIfNeeded`

New `"uniformArrayElement"` case (aggregate elements only; scalar elements
never reach it). For each of the element's `width` components:

```
[base + trunc(index)*elementStride + k*rawCompSize, load, (promote if narrow)]
  -> storeComponent(scratchAddr, kind, k*compSize)
```

- Packed float: `f64.load`.
- Narrow float: `f32.load` + `f64.promote_f32` — the exact byte recipe
  `emitGpuUniformPromote` already uses (rmsl-wasm.ts:503).
- int/uint/bool: `i32.load` in both layouts (already 4 bytes; `narrow` only
  affects float).

After this, every downstream consumer reads the scratch through the
unchanged `nodeAddress` + `readComponent`, and an aggregate element can also
be the function root via `finalValueBytes`'s existing aggregate path.

### `walkExpr`

New scalar `"uniformArrayElement"` case: compute the element address bytes,
load once (`f64.load` for float, `i32.load` otherwise), promote when narrow.

## Data model and host marshalling

### New `WasmParam` variant

```ts
{ kind: "uniformArrayMemory"; slot: string; shaderType: ShaderType; length: number;
  address: number; elementStride: number; narrow?: boolean }
```

- `address` = packed base (f64 layout) normally, the GPU offset when narrow.
- `elementStride` = byte distance between consecutive elements in the
  marshalling layout: `elementSize` packed, the WGSL-computed stride for GPU.
- `narrow` = float elements travel as f32 (GPU shareable).

### `writeArrayToMemory` (host helper, next to `writeAggregateToMemory`)

Loops `length` elements at `address + i * elementStride`:

- Aggregate element type (`width > 1`): each element is an array of
  components (`[[0,0,0,0], …]`); write with `setFloat64` / `setFloat32`
  (narrow) / `setInt32` per component, same conversions as
  `writeAggregateToMemory`.
- Scalar element type: each element is a bare value (`[1,2,3,4]`); one write
  per element.
- Tail elements beyond what the host supplies are left untouched (mirrors
  how `writeAggregateToMemory` writes only what is present).

Host contract is identical to `compileJS`: `ctx.uniforms[slot]`.

### `marshalInputs`

New switch case routing `uniformArrayMemory` →
`writeArrayToMemory(view, p.address, p.shaderType, p.length, value, p.elementStride, p.narrow)`.
Array elements never occupy WASM function args (even a scalar-typed array
lives in memory). The scalar-args path is unchanged.

## GPU layout path

For a slot with `gpuUniformLayout.offsets[slot]` defined:

- `collect`: no packed array region is allocated. Record
  `{ base: gpuOffset, elementStride: strides[slot], narrow: true }` and push
  the WasmParam with those values.
- Read path: `base + trunc(index) * gpuStride`; float components promote
  f32→f64; int/uint/bool read with plain `i32.load`.
- Marshalling: each float element's components written as f32 at
  `gpuOffset + i * gpuStride` — byte-for-byte what the WGSL uniform buffer
  for the same member contains, so a renderer pre-filling the region and the
  WASM side reading it agree exactly.
- A `gpuUniformLayout` array member **without** `strides[slot]` is a throw at
  `collect` time (`[RMSL] compileWasmFn: ...`), never a miscompile.
- Float elements of a GPU-placed array are only as precise as f32, the same
  real-but-inherent limit the plain-aggregate GPU path already pins
  (`rmsl-layout-interop.test.ts:92-108`).

### `GpuUniformLayout` gains `strides?: Record<string, number>`

Per-array element stride in the host's GPU buffer. The caller already has it
from `wgslUniformLayout(...).members[i].stride`, which is reported for array
members exactly because "the stride is what a caller cannot guess"
(rmsl-usage.test.ts:2013). Without it the backend cannot compute a GPU
element's byte offset (WGSL rounds array stride to 16, and it differs by
element type), and the codegen multiply would be wrong.

## Behavior edges (all deliberate)

- **Out-of-range index**: unclamped; reads whatever bytes sit at the
  computed address. Matches GLSL/WGSL/JS, which signal nothing on OOB.
- **Float index outside i32 range**: `i32.trunc_f64_s` traps — identical to
  the existing `int(float)` cast behavior; consistent, not new.
- **GPU float elements** round to f32 (pinned by interop tests).
- **Shorter host array**: tail reads whatever's in memory, like existing
  uniform writes.

## Testing

- `src/backends/rmsl-wasm.test.ts:118-121` — the existing rejection test
  (`uniformArray("float", 4).element(int(0))` throwing) flips to a positive
  case and is rewritten.
- New `rmsl-wasm.test.ts` block:
  - scalar-element read (`float`, `int`, `bool` arrays), constant index.
  - vec4-element read and `.toVar()`, element as function root.
  - the realistic For-loop sum pattern from `rmsl-usage.test.ts:1846`
    (runtime index, no `int(...)` on the index — it's a float loop counter).
  - host marshalling shape: `ctx.uniforms[slot]` as `number[][]` (aggregate
    elements) and `number[]` (scalar elements).
- `src/rmsl-layout-interop.test.ts` — GPU-placed array at a
  `wgslUniformLayout` offset + stride (pins the typed value, f32 rounding,
  and no corruption of an adjacent member).
- Phase 7's cross-backend recording: run `rmsl-js.test.ts` /
  `rmsl-eval.test.ts` — the `[shader-eval] WASM: N of 79 ... not yet
  supported` count drops, leaving only the non-square matrix-multiply
  case(s).

## Future note (WebGL2 UBO, recorded in ROADMAP.md)

If `WebGLRenderer` ever gains uniform buffer objects, a third `AllocRules`
for `std140` (GLSL spellings, `reorderByAlignment: false`) slots into the
same shared `planLayout` — std140 agrees with the WGSL rules on every axis
except declaration-order member placement, which is already the flag's
"false" behavior. That would let the WASM `gpuUniformLayout` seam back a
WebGL buffered draw too. Not built here; the renderer has no UBO path yet.

## Files touched

- `src/backends/rmsl-wasm.ts` — `WasmParam` variant, `GpuUniformLayout` +
  `strides`, `uniformArrayInfo`/`uniformArrayAddress` maps, `collect`,
  `isScratchNode`, helper + `materializeIfNeeded` + `walkExpr` cases,
  `writeArrayToMemory`, `marshalInputs`.
- `src/backends/rmsl-wasm.test.ts` — reject-test rewrite + positive block.
- `src/rmsl-layout-interop.test.ts` — GPU-placed array cases.
- `ROADMAP.md` — already updated (Open questions, WebGL2 UBO).