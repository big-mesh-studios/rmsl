# WASM backend roadmap

A fourth compiler backend, alongside GLSL/WGSL/JS: compile an `Fn` straight to
a raw WASM binary module (hand-encoded bytes, no wabt/binaryen) instead of JS
source. It targets the same CPU-eval niche `compileJS` serves — screen
picking, ray-march hit tests, anything calling a compiled shader graph once
per pixel/click from plain JS — where per-call overhead matters more than
raw throughput on a hot, already-warm loop.

## Why

A throwaway prototype (now deleted; see git history around the commit that
introduced `compileWasmFn`/`compileWasm` in `src/rmsl.ts` for its own
history) answered the feasibility question first: yes, the same node graph
that already feeds three interchangeable backends can feed a fourth, and yes,
it's faster. Two measured scenarios, repeated runs:

- Pure scalar arithmetic (`sqrt(a*a + b*b + c*c)`, three float uniforms):
  WASM ran **~5-8x** faster than the equivalent `compileJS` output.
- A branch plus a vector op (`if (dot(dir, target) > threshold) ...`, two
  vec3 uniforms): the win shrinks to **~3-5x** once a vec3 has to cross the
  WASM boundary as three scalar params (no linear memory yet) and a taken
  branch enters the picture — but it holds.

Root cause, as far as the prototype dug: `compileJS`'s uniform reads are
`ctx.uniforms["_rmsl_uN"]`, a string-keyed property lookup, repeated per
operand; WASM gets a plain `local.get`. That gap is wide enough that
splitting a vector into scalars and taking a branch don't close it.

**These two numbers are from the original prototype, before Phase 3's real
linear memory existed and before `compileWasm` had its current ctx-
marshalling wrapper — they no longer reproduce and should not be quoted as
the current state.** A re-measurement below, using committed, reproducible
benchmark files pinned to a specific commit, supersedes them for anything
but historical interest.

### Re-measured with `src/rmsl-wasm-vs-js.bench.ts`/`rmsl-wasm-loop.bench.ts` at commit `7b90863`

An earlier attempt at this re-measurement used a one-off script and
`performance.now()`, on a machine that turned out to be under heavy,
unrelated load (a misbehaving editor extension eating a full CPU core) —
that attempt produced numbers swinging by more than an order of magnitude
between runs and should be disregarded; it's why a **committed** benchmark
file exists at all now, rather than another throwaway script. The numbers
below are from `npx vitest bench` (tinybench under the hood, which runs
each case for a time budget rather than a fixed count and reports relative
margin of error), two runs each, on an otherwise idle machine, with the two
runs agreeing within a few percent (rme ≤ ~1.7% throughout — reproduce with
`npx vitest bench src/rmsl-wasm-vs-js.bench.ts src/rmsl-wasm-loop.bench.ts`
at this commit):

| Scenario | Result |
|---|---|
| Scalar `sqrt(a*a+b*b+c*c)`, three float params, through `compileWasm` | `compileJS` **~2.5-3.4x faster** |
| Same, calling the raw exported WASM function directly (bypassing `compileWasm`'s ctx wrapper) | `compileJS` ~1.0-1.15x faster — essentially a tie |
| vec3 `dot` + `If`/`Else` (uniforms, through Phase 3's linear memory), through `compileWasm` | `compileJS` **~3.1-3.2x faster** |
| `For` loop, 64 iterations of `sum += sqrt(i)` per call | `compileWasm` **~4.05x faster**, both runs agreeing to 2 decimal places |

So the original claim doesn't hold for a cheap, called-once function — it's
backwards, though nowhere near as dramatically as the disregarded noisy
attempt suggested. Same two-part cause as before, now with real numbers
behind it:

1. **`compileWasm`'s JS-side wrapper costs real time for trivial calls.**
   The raw exported WASM call ties `compileJS` almost exactly (~1.0-1.15x);
   going through `compileWasm`'s wrapper (`params` array iteration, a
   `kind` branch per entry, property lookups by name into
   `ctx.params`/`ctx.uniforms`) is where essentially the entire 2.5-3.4x
   gap comes from.
2. Once memory is involved (the vector case), the wrapper cost and the
   `DataView` writes together land at roughly the same ~3x gap as the
   scalar wrapper alone — see the linear-memory-specific A/B right below
   for what changed and didn't.

The loop case is the one place `compileWasm` wins outright: 64 iterations
of real work amortize the fixed call/wrapper cost, and instruction
execution — what WASM was always expected to win at — dominates instead.

**Practical read:** at Phase 1-4, `compileWasm` is a worse choice than
`compileJS` for the exact "cheap function, called once per pixel/click"
pattern its own opening paragraph names as the target niche, and a better
one once there's a loop or enough per-call work to amortize the wrapper.
Revisit before recommending `compileWasm` over `compileJS` for that niche
specifically — either the wrapper needs to get cheaper (Phase 7 territory:
this is exactly the "real workload, not a microbenchmark" gap that phase
already flags), or the niche description needs updating.

### A/B: what Phase 3's linear memory itself cost or saved

The re-measurement above doesn't separate linear memory's own effect from
the wrapper/call-boundary cost in general — both scenarios ran on code that
already had one or the other. To isolate it, `src/rmsl-wasm-vs-js.bench.ts`
as of commit `7b90863` was copied unmodified — `git show
7b90863:src/rmsl-wasm-vs-js.bench.ts` — into a worktree checked out at
`9b845b7` (the commit immediately before linear memory landed, `f58c93b`)
and run there, two runs, same idle-machine conditions:

| Scenario | Before linear memory (`9b845b7`) | After (`7b90863`) |
|---|---|---|
| Scalar `sqrt(...)`, through `compileWasm`'s wrapper | `compileJS` ~2.5x faster | `compileJS` ~2.5-3.4x faster — **slightly worse** |
| vec3 `dot` + `If`/`Else`, through `compileWasm`'s wrapper | `compileJS` ~6.3x faster | `compileJS` ~3.1-3.2x faster — **roughly 2x better** |

Linear memory is not the source of the scalar-function slowdown — a
function with no aggregate values touches none of it, and the modest
regression there (a few percent to ~30%, noisier than the other numbers
here) tracks `WasmParam` growing from two kinds to four
(`"param"`/`"uniform"`/`"paramMemory"`/`"uniformMemory"`), meaning every
call now branches through more `kind` checks even when the extra kinds are
never hit — a cost of the wrapper's added generality, not of memory access.

For the vector case, linear memory is a clear **win**: before, a vec3
uniform meant three separate scalar `WasmParam` entries (six just for
`dir`/`target`), each needing its own name lookup and axis-indexed read out
of `ctx.uniforms` on every call; now it's one `"uniformMemory"` entry per
vector, with all three components copied by `writeAggregateToMemory` in one
pass. Roughly halving the JS-vs-WASM gap (6.3x down to ~3.1x) came from
that — fewer round trips through the `params` array outweighing the added
`DataView` write.

## Status: Phase 1 through Phase 4 landed

`compileWasmFn` and `compileWasm` exist in `src/rmsl-wasm.ts`, next to
`rmsl-glsl.ts`/`rmsl-wgsl.ts`/`rmsl-compile-js.ts` (see CONTRIBUTING.md for
the file layout). Tests are in `src/rmsl-wasm.test.ts`.

**What it covers**, Phase 1's validated slice, Phase 2's full scalar op
parity, Phase 3's vectors/matrices as first-class values, and Phase 4's
control flow:

- `compileWasmFn(fn, options): { bytes: Uint8Array, params: WasmParam[],
  resultType: ShaderType }` — the module plus a description of what each
  exported-function argument (and the result) means.
- `compileWasm(fn, options): (ctx: JsShaderContext) => number | boolean` —
  same call signature as `compileJS`, for drop-in comparison on the ops it
  supports. A `"bool"` result comes back as a real boolean and a `"uint"`
  one is reinterpreted from WASM's always-signed i32 return, matching what
  `compileJS` hands back for the same declared types.
- Explicit function params (`options.params`), and scalar float/int/uint/
  bool uniforms and params.
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
  `For`/`if`-chains before this backend ever sees them (`rmsl-core.ts`), so
  they already worked once `for`/`if` did.

**What throws today** (deliberately — see the Phase list below for when each
lands): `cross`/`length`/`normalize`/`distance`/`reflect`, matrix×vector and
matrix×matrix multiply, `clamp`/`mix`/`step`/`smoothstep` (composite ops
with no single WASM opcode — deliberately out of Phase 2's "has a direct
opcode" scope), `uniformArray`, `output()`/`varying()`/`attribute()`/
`builtinPosition()`/`builtinFragDepth()`/`fragCoord()`,
`textureLoad`/`texture`/`textureSize`, multi-return, and any non-scalar
function *result* (only intermediate values are first-class aggregates now
— the root a compiled `Fn` returns must still be a scalar). `compileWasmFn`
throws `[RMSL] compileWasmFn: unsupported node type in
<expr|vector|statement> position: "<type>"` naming exactly what's missing,
which is also the fastest way to find the next thing worth doing here.

## Design decisions already made

- **Lives in `src/rmsl-wasm.ts`, alongside `src/rmsl-glsl.ts`/`rmsl-wgsl.ts`/
  `rmsl-compile-js.ts`** — the compiler was later split out of the original
  single `src/rmsl.ts` file by concern (see `CONTRIBUTING.md`), and this
  backend followed the same one-file-per-backend pattern. It reuses the same
  untyped internal node shape (`node.type`/`node.params`/`node.value`)
  `compileJSNode` already switches on, imported from `rmsl-core.ts` and
  `rmsl-compiler-shared.ts` — no new node representation to keep in sync.
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
  varName), and one dedicated scratch slot per vector/matrix-*producing*
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
  bytecode emitted) *exactly once* per syntactic use regardless of how many
  of its components get read afterward, so nesting doesn't blow up
  proportionally to width — see `materializeIfNeeded`/`nodeAddress` in
  `src/rmsl-wasm.ts`. Scalar locals/params/uniforms are untouched by any of
  this; only aggregate values moved into memory.
- **Control flow compiles through one loop shape and one exit block.**
  `Loop`/`Switch` needed no work at all — both desugar to `For`/nested
  `"if"` nodes in `rmsl-core.ts` before this backend ever sees them, so the
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
  `Return(value)` — only a bare early-exit) but a compiled `Fn` always
  declares a real scalar WASM result type, so both push a zero/false
  sentinel of that type and `br` to one function-exit `block` wrapped
  unconditionally around the whole body (3 bytes, a no-op for any function
  that never uses either) — `Discard`'s real "no fragment output" meaning
  still has no representation and won't until Phase 5's shader-stage
  surface exists; for now it's identical to `Return()`.
- **No shader-stage concept yet.** `compileWasmFn`'s options are
  `CompileFnOptions` (`name` + `params`), the same shape
  `compileGLSLFn`/`compileWGSLFn` use for standalone functions — not
  `CompileJSOptions`'s `stage`/`derivatives`/`reentrant`. A plain
  float-returning function is a much smaller surface than a full fragment/
  vertex stage; Phase 5 is where that gap gets closed.

## Phased plan

Roughly ordered by what unblocks the most; not a commitment to build all of
it.

### ~~Phase 2 — full scalar op parity~~ — done
See "Status" and "Design decisions already made" above for what landed:
every remaining `jsBinaryOp`/`jsUnaryMath` entry with a direct WASM opcode
(or a trivial derivation of one, like float `mod`/`round`/`fract`), every
comparison, logical, and bitwise op, and `int`/`uint`/`bool` as real `i32`.
Explicitly *not* included, and not planned for a later phase to sneak back
in under this name: `clamp`/`mix`/`step`/`smoothstep` — composite ops with
no single opcode, closer in spirit to Phase 3's vector work than to this
phase's "one opcode per op" scope.

### ~~Phase 3 — vectors and matrices as first-class values~~ — done
See "Status" and "Vectors and matrices live in linear memory now" above for
the layout that shipped and what it covers: construct/literal/`toVar()`/
`.assign()`/swizzle read+write/`dot`/componentwise `add`/`sub`/`mul`/`div`
across vec2/vec3/vec4/mat2/mat3/mat4 (+ `ivec*`/`uvec*`/`bvec*`), backed by
a real WASM linear memory rather than the scalar-slot-splitting Phase 1/2
used for vec3. Explicitly *not* included, and left for a later phase:
`cross`/`length`/`normalize`/`distance`/`reflect`, matrix×vector/
matrix×matrix multiply, and `uniformArray` (the memory design leaves room
for it — an array element's address just needs a dynamically-computed
offset, which nothing in this phase's scope required — but it wasn't
implemented here).

### ~~Phase 4 — control flow parity~~ — done
See "Status" and "Control flow compiles through one loop shape and one exit
block" above for what shipped: `for`/`while`/`Break`/`Continue`/`Return`/
`Discard`, plus `Loop`/`Switch` for free since they desugar before reaching
this backend. `Discard` still has no real scalar-function equivalent — it
compiles to the same zero/false-sentinel early exit `Return()` does, a
placeholder until Phase 5's shader-stage surface gives it actual meaning.

### Phase 5 — shader-stage surface
`output()`, `varying()`, `attribute()`, `builtinPosition()`,
`builtinFragDepth()`, `fragCoord()`, multi-return, the `stage`/`derivatives`/
`reentrant` options `compileJS` has. This is what turns "compiles a plain
function" into "is an alternative to `compileJS` for a real shader graph,"
matching `CompileJSOptions` instead of `CompileFnOptions`.

### Phase 6 — texture sampling
Phase 3's linear memory is sized for a handful of fixed-size aggregates,
laid out once at compile time; `JsTextureData` is arbitrary-sized pixel
data, unknown until a texture uniform is actually bound. Needs its own
memory region (likely grown with `memory.grow` rather than baked into the
Phase 3 bump allocator's compile-time total), a way to copy a texture's
data into WASM memory before each call (or keep it resident and re-copy
only on change), and filtering/wrap logic implemented either in emitted
WASM or as an imported host function.

### Phase 7 — parity testing infrastructure
Once coverage is broad enough, hook `compileWasm` into
`src/testing/shader-eval.ts`'s recording the way `compileGLSL`/`compileWGSL`
already are (see CONTRIBUTING.md's "Validity"/"Values" test layers), so a
case written once in `rmsl-js.test.ts`-style files is checked against WASM
automatically instead of needing its own file. Also: benchmark against
realistic workloads, not the microbenchmarks that shaped Phase 1 — a real
picking scene's actual call pattern, not a tight synthetic loop. See
`src/rmsl-wasm-vs-js.bench.ts`/`rmsl-wasm-loop.bench.ts` and the
re-measurement under "Why" above: the current `compileWasm` wrapper loses
to `compileJS` on a cheap per-call microbenchmark and only wins once
there's a loop, so this phase's benchmarking work should specifically pin
down where the crossover point is and whether the wrapper itself can get
cheaper, not just confirm a win on a friendlier workload — and should keep
using (or extending) these committed bench files rather than another
throwaway script, so results stay reproducible run to run.

### Phase 8 — tooling and docs
A `docs/wasm.md` page (or a section in `docs/compilation.md`), and a vite
precompile story analogous to `precompileShaders`/`precompileJS` if this is
going to ship a `.wasm` asset rather than generate one at runtime.

## Open questions

- **Async instantiation.** `WebAssembly.instantiate` (async) is the
  browser-recommended path for anything but a tiny module; sync
  `new WebAssembly.Module()` blocks the main thread past some size. Revisit
  once real programs are large enough to measure this, rather than guessing
  now.
- **Does this ever reach the public API surface** (README's "Three backends"
  becoming four, a `compileWasm` export from the package root) or does it
  stay an internal/experimental path indefinitely? Depends on how far the
  phases above get and whether the win keeps holding as coverage grows.
- **Reentrancy.** `compileJS`'s `reentrant` option exists because its
  scratch slots are shared across calls by default. WASM locals are already
  per-call-frame, so this concern may simply not exist here — confirm once
  Phase 5 needs the option to make sense of at all.

## Known issues found along the way (not WASM-specific)

- **`mat2`/`mat2x3`/`mat2x4`/`mat3x2`/`mat3x4`/`mat4x2`/`mat4x3` have no
  "columns of vector nodes" constructor overload.** `mat3(colA, colB, colC)`
  and `mat4(colA, colB, colC, colD)` are hand-written functions in
  `rmsl-core.ts` with a dedicated `args.every(isNode)` branch for exactly
  this; every other matrix type is built by the generic
  `makeMatConstructor`, which only special-cases a single node/number
  argument (diagonal) or zero arguments (identity) — anything else,
  including column-vector nodes, falls through to
  `node({_t: t, type: t, value: args})`, a *literal* node whose `value`
  ends up holding `Node` objects instead of numbers. Nothing validates this
  at construction time, so `mat2(vec2(1,2), vec2(3,4))` builds silently and
  only breaks downstream — found while writing a WASM backend test for
  matrix multiply, where it surfaced as every component reading back `NaN`.
  Affects every backend (GLSL/WGSL/JS), not just WASM: this is a gap in
  `rmsl-core.ts`'s public API, not something a compiler backend can work
  around. Not fixed here — filed as a note rather than a fix since it's
  outside this roadmap's scope (the WASM backend) and deserves its own
  look at whether to extend `makeMatConstructor` with the same
  `args.every(isNode)` branch `mat3`/`mat4` already have, or to make the
  literal-fallback path throw when given non-number args instead of
  silently accepting them.

## Non-goals

Not a GPU-side WASM or SPIR-V backend — this is CPU-only, the same niche
`compileJS` already fills. GPU compute is GLSL/WGSL's job.
