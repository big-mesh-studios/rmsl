# WASM backend roadmap

A fourth compiler backend, alongside GLSL/WGSL/JS: compile an `Fn` straight to
a raw WASM binary module (hand-encoded bytes, no wabt/binaryen) instead of JS
source. It targets the same CPU-eval niche `compileJS` serves — screen
picking, ray-march hit tests, anything calling a compiled shader graph once
per pixel/click from plain JS — where per-call overhead matters more than
raw throughput on a hot, already-warm loop.

The full benchmark history and measurement-by-measurement rationale behind
the claims and design decisions below lives in
[`docs/wasm-benchmarks.md`](docs/wasm-benchmarks.md) — kept separate so this
file stays a scannable status/plan document rather than a chronological log.

## Table of contents

- [Status: Phase 1 through Phase 8 landed](#status-phase-1-through-phase-8-landed)
- [Software rasterizer(s): feature-complete checklist](#software-rasterizers-feature-complete-checklist)
- [Open questions](#open-questions)
- [Design decisions already made](#design-decisions-already-made)
- [Known issues found along the way](#known-issues-found-along-the-way-not-wasm-specific)
- [Non-goals](#non-goals)

## Status: Phase 1 through Phase 8 landed

`compileWasmFn` and `compileWasm` exist in `src/wasm.ts`, next to
`glsl.ts`/`wgsl.ts`/`js.ts` (see CONTRIBUTING.md for
the file layout). Tests are in `src/wasm.test.ts` and
`src/layout.test.ts`.

**What it covers**, Phase 1's validated slice, Phase 2's full scalar op
parity, Phase 3's vectors/matrices as first-class values, Phase 4's control
flow, Phase 5's shader-stage surface, and Phase 6's texture sampling:

- `compileWasmFn(fn, options: CompileWasmFnOptions): { bytes: Uint8Array,
params: WasmParam[], resultType: ShaderType }` — the module plus a
  description of what each exported-function argument (and the result)
  means.
- `compileWasm(fn, options): (ctx: CpuShaderContext) => number | boolean |
CpuShaderResult` — same call signature as `compileJS`. A plain
  scalar-returning program returns the bare value (a `"bool"` result comes
  back as a real boolean, a `"uint"` one reinterpreted from WASM's
  always-signed i32 return, matching `compileJS`); a program using the
  shader-stage surface returns a `CpuShaderResult`, identical in shape to
  what `compileJS` returns for the same program.
- Explicit function params (`options.params`), scalar float/int/uint/
  bool uniforms and params, and `attribute()`/a fragment-stage
  `varying()`/`fragCoord()` as further inputs — see "The shader-stage
  surface: one direction reuses uniform-marshalling, the other reads
  memory back after the call" below.
- Full scalar arithmetic: `+ - * / %`, `min`/`max`, `sqrt`/`inverseSqrt`,
  `abs`/`sign`/`negate`/`floor`/`ceil`/`trunc`/`fract`/`round`, every
  comparison, `and`/`or`/`not`, every bitwise op, and the transcendental
  family (`sin`/`cos`/`tan`/.../`exp`/`log`/`log2`/`exp2`/`pow`/`atan2`) via
  a WASM import calling the real `Math` object — see "Transcendentals import
  `Math`, they don't get a polynomial" below. `int`/`uint` use real `i32`,
  with signed/unsigned opcode variants chosen per operand type and explicit
  conversions for `float(int)`/`int(float)`/etc. casts.
- vec2/vec3/vec4/mat2/mat3/mat4 (and `ivec*`/`uvec*`/`bvec*`) as first-class
  intermediate values, backed by a real WASM linear memory: construct
  (mixed-arity, matrix-from-columns, matrix-from-scalar diagonal), literal
  vectors/matrices, `toVar()`/`.assign()` (including reuse across an
  `If`/`Else` branch), swizzle read and write (single- and multi-component,
  including a swizzled assignment target), `dot` (any width, not just
  vec3), and componentwise `add`/`sub`/`mul`/`div` (vector±vector and
  vector±scalar broadcast). Aggregate function params work the same way as
  aggregate uniforms — see "Vectors and matrices live in linear memory now"
  below.
- `If`/`Else` via `toVar()`/`.assign()`, compiled to WASM's structured
  `if`/`else`/`end`.
- `for`/`while`/`Break`/`Continue`/`Return`/`Discard`, compiled to WASM's
  structured `block`/`loop`/`br`/`br_if` — see "Control flow compiles
  through one loop shape and one exit block" below. `Loop(count, body)` and
  `Switch(selector, chain)` need no separate handling: both desugar to
  `For`/`if`-chains before this backend ever sees them (`core.ts`), so
  they already worked once `for`/`if` did.
- `cross` (vec3 only), `length`, `distance`, `normalize` (leaves a
  zero-length vector unchanged rather than dividing by zero, matching the
  JS backend), and `reflect`.
- `mat.mul(vec)` (`matVecMul` — a dedicated node type, not the generic
  `"mul"`; correctly implies a homogeneous `w=1` and drops a row for a
  vector one component short of the matrix's column count, e.g.
  `mat4 * vec3`) and `mat.mul(mat)` (a real matrix product for two
  same-shape square matrices, matching the JS backend's own square-only
  limit — **was previously a silent miscompile**: the generic `"mul"`
  dispatch treated it as componentwise multiplication instead, since
  nothing distinguished "two matrix operands" from "vector operands" before
  this landed; `mat.mul(scalar)` was already correct and is untouched).
- `stage`/`derivatives`/`reentrant` options, matching `CompileJSOptions`;
  `attribute()`/a fragment-stage `varying()`/`fragCoord()` as inputs;
  `output()`/a vertex-stage `varying()`/`builtinPosition()`/
  `builtinFragDepth()` as outputs, including a program's own result
  becoming the implicit vertex position when `builtinPosition()` was never
  written explicitly — see "The shader-stage surface" below. A stage root
  can also be an array of nodes (the `Fn(() => [a, b])` pattern
  `compileGLSL.vertex`/`compileWGSL.vertex` already accept): every entry
  compiles for its side effects, and only the last one's value and type
  feed the stage's single result slot — matching `compileJS`, whose own
  multi-return support landed alongside this one.
- `textureSize`, `textureLoad`, and `texture`/`textureLod` (nearest,
  bilinear, and trilinear filtering; `repeat`/`mirror`/`clamp` wrapping) for
  `sampler2D`/`sampler3D`/`samplerCube` and the 2D/3D integer
  (`isampler*`/`usampler*`) variants — matching `compileJS`'s own scope
  exactly: no mipmap/LOD (`textureLod`'s third argument is compiled nowhere,
  same as `compileJS`), no integer cube sampling (`isamplerCube`/
  `usamplerCube`). Texture pixel data is copied into the compiled module's
  own linear memory rather than sampled through a call back into
  JavaScript — see "Texture data lives in linear memory, not behind a host
  call" below.

**What throws today** (deliberately): `isamplerCube`/`usamplerCube`.
(Non-square matrix×matrix multiply, float `samplerCube` sampling, and
multi-return used to be on this list; all three are supported now — the
shape check that remains in
`emitMatMatMulStores` is defense-in-depth against a hand-built node, since
core already rejects a mismatched product at construction.)
`compileWasmFn` throws `[RMSL] compileWasmFn: unsupported node type in
<expr|vector|statement> position: "<type>"` naming exactly what's missing,
which is also the fastest way to find the next thing worth doing here.

## Software rasterizer(s): feature-complete checklist

What a feature-complete software rasterizer needs, against what
`src/backends/cpu-rasterizer.ts` has today. Checked items are implemented;
unchecked ones are the gap between the current prototype and "real
rasterizer" — most were already called out piecemeal in the "`
rasterizeTriangles` is a prototype" bullet in "Open questions" below,
gathered here as one list to track over time rather than re-derive each
time.

This checklist tracks `cpu-rasterizer.ts` specifically, not the generic
WASM rasterizer module below (`src/backends/wasm/rasterizer.ts`/`.wat`) —
that module is already ahead of this list on near-plane clipping and the
depth test (see "generic, precompiled rasterizer module" under "Open
questions"), which remain unchecked here since `cpu-rasterizer.ts` itself
still lacks both.

**Vertex stage**

- [x] Run compiled vertex program once per vertex, attributes in
- [x] Implicit position from a plain `vec4` return (no explicit
      `builtinPosition()` needed)
- [x] Varyings out, carried per-vertex
- [ ] Vertex reuse / index buffer (currently sequential triples only — no
      shared vertices across triangles)
- [ ] Multiple vertex streams / interleaved vs. planar attribute layout
      choice

**Primitive assembly & clipping**

- [x] Triangle list (non-indexed)
- [ ] Indexed triangles (`drawElements`-style)
- [ ] Triangle strips/fans
- [ ] Near/far clipping (`w <= 0` currently produces garbage/`Infinity`/
      `NaN` instead of being clipped)
- [ ] Frustum/guard-band clipping against screen bounds (currently relies
      on the per-pixel bbox clamp only)
- [ ] Backface culling (winding-based)
- [x] Degenerate/zero-area triangle skip (skipped outright, not
      subdivided)

**Rasterization**

- [x] Edge-function coverage test, per-pixel-center sampling
- [x] Screen-space bounding box per triangle
- [x] Perspective-correct barycentric interpolation of varyings
- [ ] Antialiasing (MSAA-style multi-sample, or coverage-based edge AA) —
      currently one sample per pixel center
- [ ] Tiling/binning for large triangles or many triangles (currently
      naive full-bbox scan per triangle)
- [ ] Provoking-vertex / flat-shading mode (no interpolation for `flat`
      varyings)

**Depth & output**

- [ ] Depth test / z-buffer (triangles currently paint strictly in draw
      order, no occlusion)
- [ ] Depth write control / depth range
- [ ] Stencil test
- [ ] Blending (alpha blend, additive, etc. — currently a flat overwrite
      per pixel)
- [ ] Multiple render targets
- [x] Single flat output buffer, `componentCount` wide
- [ ] `componentCount` inferred from the fragment program's actual return
      type (currently a caller-supplied constant; a mismatch silently
      reads zeros/garbage)

**Fragment stage**

- [x] Run compiled fragment program once per covered pixel, interpolated
      varyings + `fragCoord` in
- [x] Uniforms threaded through to both stages
- [x] Textures threaded through to both stages (**unverified** — never
      exercised by the demo)
- [ ] Derivatives (`dFdx`/`dFdy`) — needs quad-based (2x2) fragment
      grouping, not per-pixel evaluation
- [ ] Discard (`Discard()`'s real "no output" meaning has no
      rasterizer-level effect yet — see the WASM backend's own open
      question on this)
- [ ] Fragment depth override (`builtinFragDepth()`)

**Performance / integration**

- [ ] Batched call boundary (`.batch()`-style) instead of one `invoke()`
      per vertex/pixel
- [ ] Full in-WASM triangle loop (no host-mediated vertex→fragment
      handoff) — see the generic-rasterizer-module design above
- [ ] `createJs`/`createWasm` adapter integration — no
      `setAttribute`/`setUniform` ergonomics, no pending-value replay, no
      `.draw()` on an adapter object (currently a standalone function over
      already-compiled callables)
- [ ] Shared framebuffer/depth buffer across multiple draw calls/programs
      in one frame (Approach A/B above)

**Testing**

- [ ] Correctness verified against GLSL/WGSL output for the same program
      (currently unverified beyond one demo triangle)
- [ ] Cross-backend recording hookup (the same `shader-eval.ts` machinery
      Phase 7 gave `compileWasm`/`compileJS`)

## Design decisions already made

- **Lives in `src/wasm.ts`, alongside `src/glsl.ts`/`wgsl.ts`/
  `js.ts`** — the compiler was later split out of the original
  single `src/rmsl.ts` file by concern (see `CONTRIBUTING.md`), and this
  backend followed the same one-file-per-backend pattern. It reuses the same
  untyped internal node shape (`node.type`/`node.params`/`node.value`)
  `compileJSNode` already switches on, imported from `core.ts` and
  `src/backends/shared.ts` — no new node representation to keep in sync.
- **`float` is f64, `int`/`uint`/`bool` are real `i32`** — not the JS
  backend's approach of collapsing every declared type into one JS number.
  GLSL/WGSL already carry each node's declared type through to their output;
  WASM has that same type information (`node._t`) and, unlike JavaScript, an
  actual integer type to put it in. This also matches GLSL/WGSL's 32-bit
  integer wraparound on overflow, which a WASM int stored as f64 would not.
  Every op picks its opcode from the operand's `scalarKindOf(node._t)` — see
  `compileWasmFn`'s `binaryArith`/`comparison`/`minOrMax` helpers — and a
  scalar `construct` node (a `float(x)`/`int(x)`/etc. cast) becomes an
  explicit conversion or comparison, not a no-op, whenever the source and
  target aren't both f64 or both i32. `bool` is i32 0/1, the same
  representation WASM's own comparison opcodes and `if` condition already
  use, so it needed no separate representation of its own.
- **Transcendentals import `Math`, they don't get a polynomial.** `sin`,
  `cos`, `exp`, `log`, `pow`, and the rest of that family have no WASM
  opcode. Rather than hand-roll an approximation — real risk in a codebase
  whose own testing philosophy is built around "most mistakes here are
  silent" (CONTRIBUTING.md) — the compiled module imports them from a
  namespace named `"math"`, and `compileWasm` hands the real `Math` object
  as that namespace's implementation, unchanged: `Math.sin`, `Math.cos`, etc.
  already have the right names, so no translation layer is needed. A module
  that ends up needing none of them declares no imports at all and simply
  ignores the namespace it was handed. `exp2(x)` reuses the `pow` import as
  `pow(2, x)`, matching how the JS backend implements it (`Math.pow(2, x)`);
  `inverseSqrt(x)` and `round(x)` stay opcode-only (`1 / sqrt(x)`,
  `floor(x + 0.5)` — the latter because `f64.nearest` rounds half-to-even,
  not `Math.round`'s round-half-up).
- **No sub-expression caching.** Where a value is needed twice with no WASM
  opcode that takes it once — `select`'s two branches, `sign`'s two
  comparisons, float `mod`'s `a - b * floor(a / b)` — the operand's bytes
  are just emitted again rather than computed once into a scratch local
  (compare `compileJS`'s `jsNewTemp`, which this backend has no equivalent
  of yet). Always correct, since nothing in this DSL's expression position
  has a side effect to duplicate; only ever a size/speed cost, and one this
  early backend hasn't needed to solve yet.
- **Synchronous instantiation** (`new WebAssembly.Instance(new
WebAssembly.Module(bytes))`), matching `compileJS`'s synchronous `new
Function(source)()`. Fine for the module sizes here; revisit if a module
  grows large enough that sync compilation stalls a browser main thread (see
  Open questions).
- **Vectors and matrices live in linear memory now.** Phase 3 settled the
  ROADMAP's own open question in favor of a real WASM linear memory (one
  memory, declared and exported as `"memory"`) over the cheaper
  "split into N scalar slots" alternative. Every address is a **compile-time
  constant** — there's no dynamic allocation, stack pointer, or (until
  `uniformArray`/loops need one) dynamically-computed offset. A bump
  allocator during the existing collect pass hands out byte offsets to:
  aggregate uniforms and function params (keyed by slot/name — `WasmParam`
  gained `"paramMemory"`/`"uniformMemory"` variants that carry an `address`
  and never occupy a WASM function argument at all; `compileWasm` writes
  their components straight into the instance's exported memory via a
  `DataView` before every call), `let`-bound aggregate vars (keyed by
  varName), and one dedicated scratch slot per vector/matrix-_producing_
  expression node (construct, literal, multi-component swizzle,
  componentwise arithmetic — keyed by node object identity via a `WeakMap`,
  so a repeated reference like `dot(v, v)` shares one slot). Components are
  stored at their natural width (f64 for float-family types, i32 for
  int/uint/bool-family types), with no padding — every load/store uses
  `align=0` since the allocator doesn't guarantee natural alignment, and
  WASM's alignment immediate is only a perf hint, not a correctness
  requirement. Scratch slots are never freed or reused (no arena/stack —
  there's no recursion, and a future loop iteration just re-runs the same
  static address), and every aggregate sub-node is materialized (its store
  bytecode emitted) _exactly once_ per syntactic use regardless of how many
  of its components get read afterward, so nesting doesn't blow up
  proportionally to width — see `materializeIfNeeded`/`nodeAddress` in
  `src/wasm.ts`. Scalar locals/params/uniforms are untouched by any of
  this; only aggregate values moved into memory.
- **Control flow compiles through one loop shape and one exit block.**
  `Loop`/`Switch` needed no work at all — both desugar to `For`/nested
  `"if"` nodes in `core.ts` before this backend ever sees them, so the
  real new surface was `for`/`while`/`break`/`continue`/`return`/`discard`.
  `for` and `while` both compile through the same nested shape, `block
{ loop { <cond>; br_if (out to block) ; block { <body> } ; <update>; br
(back to loop) } }` — the inner `block` around `body` exists specifically
  so `Continue` (`br` to that block) still runs a `for`'s `update` clause
  before re-testing the condition, rather than skipping it by jumping
  straight back to the condition check. `Break`/`Continue` need WASM's
  `br`/`br_if` **relative label index** (how many enclosing structured
  blocks to jump out through, fixed at compile time, not a runtime target),
  computed via a `depth` parameter threaded through `walkStmt` (incremented
  for every enclosing `if`/`for`/`while`, not just loops — an `if`'s
  branches are structured WASM blocks too, so a `Break()` inside
  `If(x, () => Break())` inside a loop must count the `if`'s own label) and
  a `loopStack` recording each open loop's break/continue target depths.
  `Return()`/`Discard()` carry no value in this DSL (there's no
  `Return(value)` — only a bare early-exit), so both `br` to one
  function-exit `block` wrapped unconditionally around the whole body (a
  no-op for any function that never uses either) — pushing a zero/false
  sentinel first when the function still declares a real scalar WASM
  result (a plain function, `!needsResult`), or nothing at all when it
  doesn't (a stage-mode function, `needsResult` — see the next bullet;
  its own result, if any, already lives in memory by then). `Discard`'s
  real "no fragment output" meaning still has no representation; it
  remains identical to `Return()`.
- **The shader-stage surface: one direction reuses uniform-marshalling,
  the other reads memory back after the call.** `attribute()` and a
  fragment-stage `varying()` are input-direction — same shape as a
  uniform (`WasmParam`'s `"attribute"`/`"attributeMemory"`/
  `"varying"`/`"varyingMemory"` kinds), just sourced from
  `ctx.attributes`/`ctx.varyings`. `fragCoord()` is the same, with one
  canonical address instead of one per slot, since every reference in a
  program is the same input. `output()`, a vertex-stage `varying()`,
  `builtinPosition()`, and `builtinFragDepth()` are output-direction —
  the first time this backend ever needed to read its own memory back
  _after_ a call, not just write it before one. A `needsResult` flag
  (mirroring `compileJS`'s own `ctx.jsNeedsRes`) tracks whether any of
  these were used, or whether an explicit `"vertex"` stage was requested
  at all (a vertex stage's own result always maps to its position one way
  or another, so it's never just a plain WASM return, even if
  `builtinPosition()` itself is never mentioned — found by a test that
  didn't initially catch a vertex-stage program failing to produce a
  position, until `needsResult` was seeded from `options.stage ===
"vertex"` directly rather than solely from which nodes a program
  happened to use). `needsResult` false (every pre-Phase-5 program, and
  any fragment-stage one that only reads inputs) keeps the function's
  original single-scalar-result shape, byte-for-byte; true switches it to
  a zero-result shape, with `output()`/`varying()`/`builtinPosition()`/
  `builtinFragDepth()`/the function's own value all read back by
  `compileWasm` afterward into a `CpuShaderResult` — identical to what
  `compileJS` already returns for the same program. `assertStageResult`
  (`src/backends/shared.ts`) is reused directly, unmodified — it only
  ever needed plain primitives (`shaderStage`/`lastType`/`positionWritten`),
  not a full `CompileCtx`.
- **`.batch()` (originally named `.draw()` — renamed once the same
  `CpuRoutine` shape started covering vertex/fragment/compute invocations
  too, not just image rendering; see "A single shared-memory WASM module
  ..." below) is a second exported function sharing `main`'s bytecode via
  `call`, not a second compile mode or a copy of `main`'s body.**
  `collect()`/`walkStmt`/`walkExpr` compile `main` exactly as they always
  have; the module-assembly step at the end of `compileWasmFn` separately
  builds a `"batch"` function (only when the root produces a value at all)
  whose own body is just a `y`/`x` loop writing `fragCoord()`, calling
  `main` by function index, and copying the result into a growable output
  buffer. `compileWasm` exposes it as `.batch(ctx, width, height)` on the
  `CpuRoutine` it returns. See `docs/wasm-benchmarks.md`'s "A whole grid in
  one call: `.draw()`" (kept under its original name there — a historical
  measurement, not current API surface) for the design and the measured
  speedup.
- **`clamp`/`mix`/`step`/`smoothstep` compile through the same
  aggregate-value machinery vectors already use.** Each is a composite
  formula rather than a single opcode (`clamp`: nested `min`/`max`; `mix`:
  `a*(1-t)+b*t`; `step`: a `select` on `x < edge`; `smoothstep`: `clamp`
  then `t*t*(3-2*t)`), so each gets its own `isScratchNode` entry, a
  `materializeIfNeeded` case, and an `emit*Stores` function that walks its
  operands componentwise into a scratch address — exactly the pattern
  `emitComponentwiseStores` already established for plain vector
  arithmetic, just with a formula in place of a single opcode per
  component. A scalar-only fast path (`walkExpr`'s own `"clamp"`/`"mix"`/
  `"step"`/`"smoothstep"` cases) skips the scratch address entirely and
  leaves the value on the WASM stack, matching how every other scalar op
  here already works. `UNIFORM_OPERAND_OPS` (`core.ts`) already
  broadcasts a scalar operand to match the defining operand's width at
  AST-construction time for all of these _except_ `mix`'s `t`, which is
  deliberately left unbroadcast so a single scale factor can drive a
  vector `mix` — so `mix`'s codegen is the one of the four that checks
  `componentCountOf(t._t) === 1` and reads `t` once instead of
  once per component. `smoothstep`'s `t` is computed once per call site
  and reused for its other two uses via a shared WASM local (`local.tee`
  followed by two `local.get`s) rather than recomputed three times —
  the one place this backend caches a sub-expression instead of following
  its usual "recompute, don't cache" style (see `mod`'s comment), because
  `t` here is itself a composite expression and WASM's own execution model
  (locals are per call frame, execution is strictly sequential — no
  reentrancy hazard even for a nested `smoothstep` call) makes the local
  safe to share across every call site rather than needing one local per
  site.

## Open questions

- **Should scalar-as-function-param be chopped in favor of always
  memory-resident?** `scalarsInMemory` (see "Design decisions already
  made" — added for the generic rasterizer module, which needs every
  compiled shader to share one fixed, zero-argument call signature) is
  opt-in specifically to avoid regressing the default path: a plain
  `local.get` measurably beats a memory read for a cheap, called-once
  function (`docs/wasm-benchmarks.md`'s whole "Why" section is built on
  that ~2.5-3.4x scalar-case measurement), which is the documented reason
  `compileWasmFn` exists as an alternative to `compileJS` at all. Chopping
  the param path and always going through memory would simplify
  `walkExpr`'s uniform/attribute/varying cases to one branch instead of
  two, at the cost of that regression for every caller, not just the
  rasterizer. Raised as a pure code-complexity question, not a
  performance one — revisit only if the two-path complexity actually
  becomes a maintenance burden, or if a re-benchmark someday shows the
  wrapper/marshalling overhead already swamps the param-vs-memory
  difference anyway (plausible, unverified). Not planned.
- **Async instantiation.** `WebAssembly.instantiate` (async) is the
  browser-recommended path for anything but a tiny module; sync
  `new WebAssembly.Module()` blocks the main thread past some size. Revisit
  once real programs are large enough to measure this, rather than guessing
  now.
- **Reentrancy.** `compileJS`'s `reentrant` option exists because its
  scratch slots are shared across calls by default. WASM locals are already
  per-call-frame, so this concern may simply not exist here — confirm once
  Phase 5 needs the option to make sense of at all.
- **Whole-function f32 arithmetic mode.** `GpuUniformLayout` (stage 2 of
  `docs/design-shared-layout-ir.md`) only narrows a uniform's _storage_ to
  f32 at the memory boundary — internal arithmetic always stays f64
  (matching `compileJS`), same as an ordinary uniform. A different,
  materially bigger idea came up alongside it: a per-`Fn` mode where _every_
  scalar float op (`add`/`mul`/`sqrt`/...) actually computes in f32
  throughout, matching what a real GPU shader would compute bit-for-bit at
  every intermediate step, not just at the uniform boundary — useful for a
  CPU-side computation meant to verify or shadow a GPU one exactly. Not
  designed or started: it would need an f32 variant of nearly every
  scalar-op-emitting function in `wasm.ts`, not one contained seam like
  the uniform-boundary case. No known driving use case yet — revisit only if
  one shows up.
- **Transcendentals still call back into JavaScript.** Phase 6 rejected a
  host-import design for texture sampling specifically because it would
  keep a texture-using shader from ever running standalone, without a JS
  engine behind it — but `sin`/`cos`/`exp`/`log`/`pow`/etc. (Phase 1, see
  "Transcendentals import `Math`, they don't get a polynomial" above) still
  work exactly that way, and Phase 6 didn't revisit it. So the "runs without
  a JS engine" property Phase 6 bought for textures doesn't actually hold
  yet for any program that also uses a transcendental function — the
  backend is inconsistent on this point right now, not fully standalone.
  Closing it would mean hand-rolling polynomial approximations, which is
  exactly the risk that made the `Math`-import choice deliberate in the
  first place (see that section) — so this isn't a small follow-up, it's a
  real tradeoff to make consciously if the standalone/native use case ever
  becomes a real target rather than exploratory. Revisit then, not before.
- **A `construct` converting between a `bool` component and a `float` one
  has no defined WASM behavior.** Found while fixing `emitConstructStores`
  (`wasm.ts`) for the real bug Phase 7's cross-backend recording
  hookup surfaced, `vec3(...).toIVec3()` truncating incorrectly because
  the float-to-int/uint conversion it needs was simply missing. The fix
  (`convertComponent`) handles float↔int and float↔uint (truncate toward
  zero one way, exact widening the other, matching GLSL/WGSL/JS), but
  deliberately leaves a `bool` on either side of that boundary untouched —
  not because it's known to work, but because no construct in the DSL
  exercises it today (nothing in the recorded test suite hit it, so there
  was nothing to fix against). Revisit if a real case shows up: the open
  question is what it should even mean (`float(someBool)` as 0.0/1.0 is an
  obvious guess, `bool(someFloat)` is less obvious — truncate-then-nonzero,
  like C, or exactly nonzero, which differ for values in (-1, 0) ∪ (0, 1)).
- **WebGL2 uniform buffer objects, if the renderer ever gains them, should
  go through the same shared allocator (`layout.ts`), not a fork.**
  `WebGLRenderer` today uploads every uniform with a per-uniform
  `gl.uniform*v` call against its `uniformLocations` map and never writes a
  packed byte region, so it has nothing to do with `gpuUniformLayout` — but
  WebGL2's `std140` block rules are the rules WGSL's uniform address space
  mirrors, so they agree with the shared allocator's WGSL rules on every
  axis except one: std140 offsets follow **declaration order**, while the
  WGSL rules reorder members by alignment to minimize padding. That axis is
  already a knob (`AllocRules.reorderByAlignment: false`), which is exactly
  what an `std140` rule set would set — alongside GLSL's own type spellings
  in `sizeAndAlignOf` and the GLSL backend emitting its uniform block in
  declaration order. The payoff would be a third consumer of `planLayout`
  placing WebGL buffered draws at the same byte offsets the WASM
  `gpuUniformLayout` seam already reads, so one CPU shadow could back both
  WebGPU and WebGL UBO draws. Not scoped or started — the renderer has no
  UBO path to hook into yet, so record the design implication, don't build
  it.
- **Typed errors, if programmatic consumers ever need to tell failure
  kinds apart.** Today every `[RMSL]`-prefixed `new Error` is untyped: a
  caller can only distinguish a validation failure (bad operand types,
  invalid shapes) from a backend coverage gap (`compileWasmFn`-prefixed
  throws) or an internal misuse ("internal error") by string-matching
  `message`, which is exactly what `isWasmUnsupported` in
  `src/testing/shader-eval.ts` already does for the coverage-gap case. That
  works, but it keys behaviour off message text. A real `RmslError`
  hierarchy (or adding a `kind`/`code` field) was considered for the
  matrix-shape validation added alongside non-square matrix multiply and
  deliberately _not_ chosen — the shapes are rejected at node construction
  for every backend at once, so no backend needed to read the error's kind.
  Revisit only if the public API needs to expose and dispatch on failure
  kind in its own code.
- **`rasterizeTriangles` is a prototype, missing a lot a real rasterizer
  has.** Scoped deliberately narrow (matching `createGlsl`'s own default
  draw's scope in some cases, just genuinely absent in others) — recorded
  so the gaps are explicit rather than discovered by surprise:
  - No depth test / z-buffer — triangles paint in draw order only, no
    occlusion.
  - No near/far clipping — a vertex behind the eye (`w <= 0`) produces
    garbage or `Infinity`/`NaN` screen coordinates instead of being
    clipped.
  - No index buffer — plain sequential triples only (matches
    `createGlsl`'s current default, not general mesh data).
  - Degenerate/zero-area triangles are skipped outright, not subdivided
    or handled specially.
  - No antialiasing — a hard edge-function coverage test, one sample per
    pixel center.
  - `textures` is threaded through to both stages but never exercised by
    the demo — unverified.
  - `componentCount` is a caller-supplied constant, not inferred from the
    fragment program's actual return type, so a mismatch silently reads
    zeros/garbage instead of erroring.
  - No `createJs`/`createWasm` integration — it's a standalone function
    over already-compiled vertex/fragment callables, not hung off an
    adapter the way `createWgsl`'s vertex+fragment option is: no
    `setAttribute`/`setUniform` ergonomics, no pending-value replay, no
    `.draw()` on an adapter object.
  - No tests — correctness against GLSL/WGSL output for the same program
    is unverified beyond the one demo triangle.
  - Naive per-pixel bounding-box loop, no tiling/binning — fine for a
    512x512 demo, not a real workload's performance profile (see the
    batched-entry-point bullet below for the specific WASM-side cost this
    also masks).
- **A single shared-memory WASM module for the whole vertex+rasterize+
  fragment loop, not per-call marshalling.** `src/backends/cpu-
  rasterizer.ts` (prototype, `apps/adapters`'s js-vtx/wasm-vtx demo) runs
  a compiled vertex/fragment pair through a software triangle rasterizer
  from the host side, calling the compiled `vertex` callable once per
  vertex and the compiled `fragment` callable once per *covered pixel* —
  for `compileWasm` specifically, every one of those is a JS→WASM boundary
  crossing, exactly the cost `.batch()` was already built to amortize for
  the fragment-only, no-attributes case (see `docs/wasm-benchmarks.md`'s
  "A whole grid in one call: `.draw()`" — kept under its original name
  there, since that's a historical measurement; the method itself is
  `.batch()` today, see
  "`CpuRoutine`" in `docs/wasm.md`). A batched entry point (an array of
  pre-interpolated varying sets handed to WASM in one call) would help, but
  doesn't remove the deeper cost: the host still owns the vertex loop and
  the raster math, so there's still at least one crossing per triangle. The
  real fix is `.batch()`'s own model taken all the way: compile the vertex
  stage and the fragment stage as two functions in **one** WASM module
  sharing one linear memory (attribute/uniform/varying storage, the same
  allocator Phase 3 already built), plus a third exported function — call
  it `"drawTriangles"`, the vertex-stage analogue of `.batch()`'s existing
  `"batch"` export — that runs the vertex loop, the edge-function/
  barycentric math, and the fragment `call`s entirely inside WASM, writing
  straight into the output buffer. The host side then shrinks to: upload
  attribute buffers into linear memory once, call
  `drawTriangles(vertexCount, width, height)` once per frame, read back
  one buffer — zero JS↔WASM crossings in the hot loop, not just fewer.
  Not designed or started — this is new `compileWasm`/`compileWasmFn`
  codegen (linking two stages' bytecode into one module, plus the raster
  loop itself as WASM bytecode, not JS), a materially bigger lift than a
  batched marshalling entry point. Revisit before trying to make the
  rasterizer prototype fast, not before.

  A typical GLSL app draws several *different* programs (materials) into
  one shared framebuffer/depth buffer, not one program per frame — worth
  recording how that composes with a one-module-per-program design before
  it's built, since it isn't free:

  - **Approach A — each program keeps its own private linear memory
    (today's model); the shared color/depth buffers are copied in and out
    per draw call.** `drawTriangles(vertexCount, width, height,
    colorBufferIn?, depthBufferIn?)` copies the shared buffers into its own
    memory at the start of the call and writes them back at the end — the
    same `out?` shape `.batch()` already has, just carrying a depth buffer
    along too so depth testing works *across* programs, not only within
    one. Cost: one buffer copy per *draw call*, not per pixel/vertex — for
    a 512x512 RGBA+depth buffer that's a few MB copied a handful of times a
    frame, nowhere near the per-pixel crossing cost this whole redesign
    exists to avoid. Small, additive change to what's already sketched
    above — the natural first thing to build.
  - **Approach B — one `WebAssembly.Memory` imported by every compiled
    program**, with the host handing out fixed offsets: one shared region
    for the color+depth buffers, one private scratch region per program.
    Every `drawTriangles` call then writes directly into the shared
    framebuffer with zero copying at all — N draw calls from N different
    programs cost exactly N calls. Strictly faster than A, but a real
    architectural change: every compiled module goes from declaring its
    own memory (`(memory (export "memory") 1)`) to *importing* one the
    host creates and grows, and each program's compile-time address
    allocator needs to know it's carving out of a shared space rather than
    owning memory 0..N itself. Reach for this only if A's per-draw-call
    copy cost actually shows up in a benchmark — same measure-before-
    optimizing discipline the rest of this file follows.

  **A third shape, considered alongside A/B and not yet reconciled with
  either: a generic, precompiled rasterizer module instead of per-program
  codegen.** Rather than generating `drawTriangles` bytecode fresh for
  every vertex/fragment pair (baking in that program's own attribute
  count/widths and varying count as compile-time constants), compile the
  triangle loop — edge functions, bounding box, perspective-correct
  interpolation — exactly **once**, as its own fixed WASM module driven
  entirely by runtime parameters: attribute stride, attribute count,
  varying stride, varying count, vertex count, width/height. It never
  knows what an attribute or varying *means*, only how many bytes to copy
  and interpolate. The vertex and fragment stages stay exactly what
  `compileWasmFn` already produces today (no new codegen there at all);
  the rasterizer module takes their compiled exports as WASM
  imports — `call_indirect` (or two ordinary imports, one per stage) into
  the vertex module's `main` and the fragment module's `main` — and all
  three share one `WebAssembly.Memory` the same way Approach B's shared
  framebuffer does, so no host round-trip carries attributes/varyings
  across the boundary either.

  This ends up as **three WASM programs per draw** (compiled vertex,
  compiled fragment, one rasterizer runtime shared by every program in the
  whole app) instead of Approach A/B's **one fused module per program**.
  Tradeoffs against the fused design above:

  - **Reuse.** The rasterizer module is compiled once, ever — adding a new
    material/shader pair costs zero new rasterizer codegen, just two
    ordinary `compileWasmFn` calls for its vertex/fragment. The fused
    design regenerates the entire triangle loop's bytecode per
    vertex/fragment pair.
  - **Cost per call.** The generic module pays a `call_indirect` (or an
    imported-function call) into vertex/fragment per vertex/pixel instead
    of the fused design's plain internal `call` — still a WASM→WASM call,
    not a JS↔WASM crossing, so it keeps the actual boundary-elimination
    win the whole redesign is for; whether the indirection itself is
    measurable is an open question, not assumed either way (see
    "measure-before-optimizing" elsewhere in this file).
  - **Codegen surface.** Nearly all of the "several hundred lines of new
    `wasm.ts` codegen" estimate for a fused `drawTriangles` applies here
    too (the edge-function/interpolation arithmetic still has to become
    bytecode), but it's written and debugged exactly once rather than
    being one more thing every future stage-compilation path has to get
    right.

  Recommended as the v1 target over the fully fused design specifically
  *because* it's compiled once: lower ongoing cost as more shader pairs
  get added, and it isolates the genuinely new, risky part (rasterizer
  codegen) from the part that's already proven (`compileWasmFn` itself).
  Scope for that v1, deliberately narrow, matching `rasterizeTriangles`'s
  own current scope (see the bullet above): non-indexed triangle list, no
  clipping, no depth test — general GPU-emulation features (indexed
  draws, clipping, depth, multiple render targets) stay explicitly out of
  scope until a real need shows up, not built ahead of one. The
  parameter-passing convention landed as plain `i32` args (see
  `RASTERIZE_PARAMS` below), not a metadata block in shared memory.

  **v1 landed, now generalized further**: `src/backends/wasm/rasterizer.ts`
  (design in `src/backends/wasm/rasterizer.md`) builds this generic module
  and covers the vertex loop, triangle setup, edge-function coverage test,
  perspective-correct varying interpolation, any number of independently-
  addressed attribute/varying slots (via runtime descriptor tables),
  near-plane (`w`) clipping (Sutherland-Hodgman, single plane — a
  straddling triangle is cut into a quad and fan-triangulated, not just
  culled whole), and a LEQUAL depth test/z-buffer (NDC `z/w`, interpolated
  with the same plain barycentric weights screen coordinates already use)
  — all verified against either `rasterizeTriangles`'s own `compileJS`-
  driven output, an independent JS clip+raster reference for the clipping
  case, or (for the depth test) drawing two overlapping triangles in both
  orders and checking the closer one wins either way. Still open: an
  index buffer and far-plane or screen-bounds frustum clipping.

  **`compileWasm(vertexFn, fragmentFn)` and `createWasm` landed**,
  finishing the `createJs`/`createWasm` adapter integration bullet this
  paragraph used to list as open. `compileWasm` links a compiled
  vertex/fragment pair against this module into one `WasmRasterRoutine`
  (`draw()`/`clearDepth()`) — see `src/backends/wasm/rasterizer.ts`. Both
  stages compile with `scalarsInMemory: true` over one shared memory (the
  fragment stage's layout placed after the vertex stage's via
  `memoryBase`); attributes are interleaved into the rasterizer's own
  descriptor-driven copy region per `draw()` call, uniforms/textures
  marshal once per call via a `WasmParam` marshaller factored out of
  `instantiateWasm`, and the depth buffer gets a stable address so it
  persists across `draw()` calls until `clearDepth()`. `createWasm`
  (`src/backends/wasm/adapter-wasm.ts`) wraps that routine in the uniform
  `Adapter` interface — `setAttribute`/`setUniform` collect draw state,
  `attach()` opens a 2D canvas, and `draw({ vertexCount })` auto-clears
  depth (one call, one frame) before rendering into it — the real
  vertex+attribute+triangle counterpart to `createWasmRoutine`'s existing
  per-pixel `compute`/`batch` path. Not yet covered by an automated test:
  `createWasm`'s own `draw()`, since it needs a real `CanvasRenderingContext2D`
  (`ImageData` isn't available in this repo's plain-Node vitest
  environment) — `compileWasm`'s `WasmRasterRoutine` underneath it is
  fully tested in `compile-wasm.test.ts` instead.

  **Authored as `.wat`, not TS codegen.** Because this module's structure
  never varies per shader (unlike `compileWasmFn`/`compileWasmRoutine`,
  which must compile an arbitrary, dynamically-constructed graph at
  runtime and so can't depend on a WASM toolchain), the "no wabt/binaryen"
  constraint above doesn't apply to it. `rasterizer.ts` used to hand-encode the same
  bytes as TS combinators simulating WASM's own function/param/local
  scoping one level removed; it's now authored directly as
  `src/backends/wasm/rasterizer.wat`, with a new `compileWat` Vite plugin
  (`src/vite/vite.ts`) compiling `.wat` to bytes at build time via `wabt`
  (wired into both `vite.config.ts` and `vitest.config.ts`), so neither
  `wabt` nor any `.wat` source reaches the built `dist/wasm.js` — only the
  inlined bytes do, same as before.

  If B is ever built, one non-obvious constraint to design around up
  front: growing a `WebAssembly.Memory` (`memory.grow`, from the host or
  from an imported call) never races an in-progress call — this backend's
  execution model is synchronous and single-threaded, so nothing runs
  concurrently while `grow` executes; the host only ever grows *between*
  calls, exactly when it already wants to (right before a draw call that
  needs more space). The real hazard is buffer **identity**, not
  concurrency: for an ordinary (non-`shared`) `WebAssembly.Memory`,
  `grow()` replaces `memory.buffer` with a brand-new `ArrayBuffer` and
  detaches the old one — already true of the existing WASM texture heap
  (see "Texture data lives in linear memory, not behind a host call"
  above, "growing detaches the old `ArrayBuffer`"). With *one* memory
  shared by several modules, growing it from any one of them invalidates
  every previously-taken `DataView`/`TypedArray` over it, for every
  module, not just the one that triggered the grow — so the host would
  need to re-derive every cached view from `memory.buffer` fresh after any
  grow, everywhere one is held. Creating the memory with `shared: true` (a
  growable `SharedArrayBuffer` instead of a growable `ArrayBuffer`) avoids
  this — a `SharedArrayBuffer` can't be detached, so growth extends it in
  place and every existing view stays valid — at the cost of needing a
  cross-origin-isolated page (COOP/COEP headers) to exist in a browser at
  all, a real deployment constraint plain `WebAssembly.Memory` doesn't
  have. Not decided — recorded so the tradeoff is visible before B gets
  built, not discovered partway through.

  **Addendum: `memoryBase` (below, under "Design decisions already made")
  generalizes to the mandelbrot worker pool's own hazard, not just to this
  rasterizer design.** `apps/mandelbrot/src/wasmWorkerPool.ts` gives each
  worker a private `WebAssembly.Memory` specifically because concurrent
  instances of the *same* compiled module all use the *same* fixed
  addresses (fragCoord, uniforms, ...) — sharing memory would mean two
  workers scrambling each other's mid-computation values. Compiling each
  worker's instance with its own `memoryBase` over one shared, `shared:
  true` memory would put every worker's fixed addresses in disjoint byte
  ranges, making concurrent writes safe (disjoint regions, not a data
  race) without per-slice `postMessage` copying at all. Two things this
  doesn't solve on its own, left open rather than assumed away: the
  COOP/COEP deployment constraint above still applies unchanged, and the
  growth-coordination hazard immediately above compounds once *several
  workers* can each independently decide to grow the same shared memory
  for their own texture/output buffer — `grow` itself is atomic per the
  WASM spec, but two workers each expanding "their own" region
  independently is a logical layout conflict, not just a thread-safety
  one, that the existing per-instance `lastSizes`/`needsRepack` caching in
  `instantiateWasm` was never written to coordinate across instances. The
  narrow fix would be giving each worker a fixed, pre-sized region up
  front (no per-worker dynamic heap growth) rather than solving general
  cross-worker growth coordination. Not designed or started — recorded
  here since it surfaced while discussing this rasterizer design, not
  because it's scoped as part of it.
- **Audio/DSP and multi-backend "audiovisual" use cases.** Purely
  exploratory — not scoped into any phase above, a set of ideas that came
  up while dreaming about what compiling one shared source to both WASM and
  WGSL/GLSL could open up:
  - An `AudioWorklet`'s real-time, per-block call pattern (`process()`
    called at a fixed rate, one call per 128-sample block) is exactly the
    "called in a loop, GC pauses are unacceptable" profile the
    wasm-vs-js benchmarks (see "Why" above) found this backend actually
    winning at, unlike the cheap-single-call case it currently loses.
  - Persistent DSP state (a delay line, a filter's history) needs no new
    backend capability — it's the same shape as GPU feedback (a buffer
    that lives outside the shader/kernel, read in and written back out
    each invocation, not state the shader itself holds): Phase 5's
    `output()`/`uniform()` round-trip already covers a scalar bit of
    state; Phase 6's texture/heap machinery already generalizes that to a
    whole buffer.
  - The "audiovisual" idea itself: the same source compiled once to both
    WASM (driving an audio thread) and WGSL (driving a visual shader),
    each independently fed the _same_ time/parameter values by the
    harness, rather than any data flowing between the two at runtime. When
    a program's feedback recurrence only ever depends on values both sides
    already have, this is just deterministic parallel simulation — same
    rule, same inputs, so the two trajectories agree without ever
    synchronizing, the same trick lockstep multiplayer netcode uses (sync
    inputs, not state). Two things would break that agreement in practice:
    numerical precision (this backend computes in f64, WGSL in f32 — a
    one-shot rounding difference is cheap, per
    `docs/design-shared-layout-ir.md`'s own finding, but compounded over
    thousands of feedback iterations in a resonant filter or delay line
    could genuinely drift the two apart), and step rate (audio's
    recurrence naturally advances per-sample or per-block, far finer-
    grained than a visual frame, so the visual side replaying "the same"
    recurrence has to decide whether it steps at audio rate internally too
    or uses a different effective rule at frame rate).

  None of this is scoped, designed, or planned — recorded so the
  exploration isn't lost, not as a commitment to build any of it.

## Known issues found along the way (not WASM-specific)

- ~~**`mat2`/`mat2x3`/`mat2x4`/`mat3x2`/`mat3x4`/`mat4x2`/`mat4x3` have no
  "columns of vector nodes" constructor overload.**~~ — fixed.
  `mat3(colA, colB, colC)` and `mat4(colA, colB, colC, colD)` were
  hand-written functions in `core.ts` with a dedicated `args.every(isNode)`
  branch for exactly this; every other matrix type was built by the generic
  `makeMatConstructor`, which only special-cased a single node/number
  argument (diagonal) or zero arguments (identity) — anything else,
  including column-vector nodes, fell through to
  `node({_t: t, type: t, value: args})`, a _literal_ node whose `value`
  ended up holding `Node` objects instead of numbers. Nothing validated this
  at construction time, so `mat2(vec2(1,2), vec2(3,4))` built silently and
  only broke downstream — found while writing a WASM backend test for
  matrix multiply, where it surfaced as every component reading back `NaN`.
  Affected every backend (GLSL/WGSL/JS), not just WASM: it was a gap in
  `core.ts`'s public API, not something a compiler backend could work
  around. `makeMatConstructor` now takes the type's column count and gets
  the same `args.length === columns && args.every(isNode)` branch
  `mat3`/`mat4` already had, routing a columns-of-vectors call through
  `"construct"` instead of the broken literal fallback — see
  `usage.test.ts`'s "builds every matrix shape from columns of vector
  nodes, not just mat3/mat4".

## Non-goals

Not a GPU-side WASM or SPIR-V backend — this is CPU-only, the same niche
`compileJS` already fills. GPU compute is GLSL/WGSL's job.
