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
already flags), or the niche description needs updating. **Refined by the
crossover measurement right below: the "enough per-call work" threshold
turns out to be very low — a handful of loop iterations, not dozens.**

### Crossover point: how many loop iterations before `compileWasm` wins

The numbers above pin down exactly two points — a loop-free scalar call
(`compileJS` wins) and one arbitrarily chosen 64-iteration loop
(`compileWasm` wins) — without saying where between them the win actually
starts. `src/rmsl-wasm-crossover.bench.ts` at commit `b629a19` sweeps the
same `sum of sqrt(i)` loop workload across iteration counts, each compiled
once up front (`npx vitest bench src/rmsl-wasm-crossover.bench.ts`, two
runs, otherwise idle machine):

| Loop length | Run 1 | Run 2 |
|---|---|---|
| 1 | `compileJS` 1.24x faster | `compileJS` 1.25x faster |
| 2 | `compileJS` 1.28x faster | `compileJS` 1.25x faster |
| 4 | `compileWasm` 1.08x faster | `compileWasm` 1.21x faster |
| 8 | `compileWasm` 1.41x faster | `compileWasm` 1.47x faster |
| 16 | `compileWasm` 2.30x faster | `compileWasm` 2.46x faster |
| 32 | `compileWasm` 2.87x faster | `compileWasm` 3.14x faster |
| 64 | `compileWasm` 3.57x faster | `compileWasm` 4.08x faster |
| 128 | `compileWasm` 4.24x faster | `compileWasm` 4.37x faster |

Both runs agree on which side of the crossover every length falls on
(only 4 iterations wobbles between a 1.08x and a 1.21x win, never a loss),
so the crossover for this workload sits **between 2 and 4 loop iterations**
— strikingly low. Most of the fixed wrapper cost the earlier scalar-call
measurement blamed for `compileWasm`'s loss turns out to have nothing to
do with looping specifically: the moment a program is loop-shaped at all
(even a loop that only runs once or twice), it's already close to
break-even, and three or four iterations of real work tip it into a win.

One structural point worth being precise about, not blurring together:
this crossover is measured entirely *within* loop-shaped programs (1
through 128 iterations of the same `For`), not as a continuous sweep
starting from the loop-free scalar case above — a 1-iteration `for` loop
and a loop-free function are different compiled shapes (the former still
emits `block`/`loop`/`br` structure WASM has to set up and JS has to
enter), not two points on one line. So the accurate statement is: **a
loop-free scalar call stays a `compileJS` win by ~2.5-3.4x; a loop of any
length 4 or more, for this workload, is already a `compileWasm` win** —
not "compileWasm needs N iterations of amortization starting from zero."

This directly narrows the other half of Phase 7's original question —
whether the wrapper itself needs to get cheaper before `compileWasm` is a
reasonable default. For any workload that loops at all, it already isn't
the bottleneck this benchmark can find; whether it's worth cheapening
further is now specifically a question about the loop-free, called-once
case (`rmsl-wasm-vs-js.bench.ts`'s scalar scenario), not a general one —
left for a separate pass, since narrowing where to look was this
benchmark's job, not fixing it.

### Texture sampling was dramatically slower than `compileJS` — fixed

`src/rmsl-wasm-texture.bench.ts` at commit `3f0ef46` first measured the
three Phase 6 texture operations against an 8x8 texture (`npx vitest bench
src/rmsl-wasm-texture.bench.ts`, two runs, otherwise idle machine):

| Scenario | Run 1 | Run 2 |
|---|---|---|
| `textureSize()` (metadata only, no sampling math) | `compileJS` 11.12x faster | `compileJS` 11.08x faster |
| `textureLoad()` (one unfiltered texel) | `compileJS` 14.57x faster | `compileJS` 14.57x faster |
| `texture()` (bilinear filtering) | `compileJS` 18.66x faster | `compileJS` 19.12x faster |

This was a real regression from the "runs standalone, no host call needed"
story Phase 6 was built around, not noise — but the cause was precise, not
mysterious: `compileWasm`'s wrapper unconditionally copied the *entire*
bound texture into its linear-memory heap on **every call**, regardless of
whether the same texture had just been copied in the call before. Even
`textureSize()`, whose own sampling math is nothing more than reading two
`i32` fields, already lost by 11x — proof the copy itself, not the
per-texel math, was most of the cost.

Fixed at commit `6b70377`: each texture slot now caches which texture
object (by reference) currently occupies its heap region, and a call
whose bound texture is the exact same object as last call's skips writing
it entirely — the realistic pattern this backend's niche implies (bind a
texture once, call the compiled function repeatedly with different
coordinates) now pays the copy exactly once, not every call. Re-measured
at the same commit, two runs:

| Scenario | Run 1 | Run 2 |
|---|---|---|
| `textureSize()` | `compileJS` 2.07x faster | `compileJS` 2.12x faster |
| `textureLoad()` | `compileJS` 3.33x faster | `compileJS` 3.25x faster |
| `texture()` (bilinear filtering) | `compileJS` 13.16x faster | `compileJS` 13.09x faster |

`textureSize()` and `textureLoad()` improved dramatically (11x → ~2x,
14.6x → ~3.3x) — consistent with the copy being their dominant cost, now
mostly gone (what's left is likely the same ordinary per-call wrapper
overhead the plain-scalar benchmark already found, ~2.5-3.4x). `texture()`
barely moved (19x → 13x), which is itself informative: its cost was never
mainly the copy, so removing the copy couldn't fix it.

Confirmed directly, not just suspected: a fourth scenario added to the same
benchmark compares `texture()` sampling the same texture with
`magFilter: "nearest"` against the existing `magFilter: "linear"` case.
`compileWasm`'s own raw throughput is the same for both, within noise,
across two runs (`compileWasm`: 1,244,856/1,263,353 hz nearest vs
1,258,009/1,266,390 hz bilinear), while `compileJS` gets measurably
*cheaper* for nearest (its own codegen actually takes a shorter branch) —
which is exactly why nearest's ratio (17.47x/19.25x) looks *worse* than
bilinear's (13.09-13.36x): `compileJS` improved and `compileWasm` didn't
move at all. This directly confirms `emitTextureSampleStores`'s own doc
comment: it computes *both* the nearest and the bilinear value
unconditionally on every sample and only `select`s between them at run
time on the texture's own `magFilter`, so the expensive path's bytecode
runs whether or not a program ever asks for it — the cost isn't "bilinear
filtering is expensive", it's "this codegen always pays bilinear's cost".

Fixed at commit `83200c6`: `emitTextureSampleStores` now branches on
`magFilter` with a real WASM `if`/`else` instead of computing both paths
and `select`ing — the one deliberate departure from this function's
otherwise-branchless style, exactly because `select` was the mechanism
paying for the unused path. Re-measured at the same commit, two runs:

| Scenario | Run 1 | Run 2 |
|---|---|---|
| `texture()`, nearest filtering | `compileJS` 3.60x faster | `compileJS` 3.68x faster |
| `texture()`, bilinear filtering | `compileJS` 11.73x faster | `compileJS` 12.04x faster |

Nearest sampling through `texture()` dropped from ~17-19x slower to
**~3.6x** — right in line with `textureLoad()`'s own ~3.3-3.5x, exactly as
expected once it no longer pays for bilinear math it never uses. Bilinear
itself improved too, a smaller but real amount (13x → ~12x), since it no
longer wastes time computing the now-unused nearest path either.

A third fix, at commit `8062a35`: `bilinear()`'s lerp used the same
`a + (b-a)*t` form `compileJS`'s `_lerp2`/`_tex2d`/`_tex3d` use — harmless
there (JS just reads the same array slot a second time), but `a`/`b` here
are usually a full `texelChannel` fetch (a dynamically-addressed memory
load, a channel-present `select`, a divide), so duplicating it is
expensive — and the two horizontal lerps were then duplicated *again* by
the outer vertical lerp, so one corner's fetch was emitted **four times**
per channel (the 3D case duplicated its two bilinear results the same way
on top of that). `a*(1-t) + b*t` is the same value needing `a`/`b` each
exactly once, duplicating only the cheap blend weight `t` instead —
applied at all three lerp levels via one shared `lerp()` helper.
Re-measured, two runs:

| Scenario | Run 1 | Run 2 |
|---|---|---|
| `texture()`, bilinear filtering | `compileJS` 6.79x faster | `compileJS` 6.80x faster |

Bilinear's gap nearly halved again (~12x → ~6.8x) — `compileWasm`'s own
raw throughput almost doubled (1.35M → 2.46M hz), from a pure arithmetic
reformulation needing no new capability at all (no locals, no branches).

A fourth fix, at commit `1c19984`: `emitTextureSampleStores` recomputed
every wrap-addressed tap coordinate — nearest mode's own wrapped index,
bilinear/trilinear's wrapped tap indices, the blend weights — fresh inside
each of the 4 channels' own call, even though none of it depends on which
channel is being read. These are now computed exactly once per sample and
stored into fixed scratch addresses right after the node's own vec4 result
(60 bytes, reserved the same way `normalize`/`reflect` already reserve one
scratch scalar for themselves), with every channel just loading them back
— a fixed-address load being cheap to repeat, unlike the
floor/multiply/wrap chain it replaces. This is the first place in this
file using a scratch address as a genuine compiler-managed temporary
rather than a node's own output value — closing the gap the file's former
"no sub-expression caching" design note called out as accepted but
unaddressed. Re-measured, two runs:

| Scenario | Run 1 | Run 2 |
|---|---|---|
| `texture()`, nearest filtering | `compileJS` 3.02x faster | `compileJS` 3.01x faster |
| `texture()`, bilinear filtering | `compileJS` 2.62x faster | `compileJS` 2.59x faster |

Bilinear's gap dropped sharply again (~6.8x → ~2.6x, `compileWasm`'s own
throughput nearly tripling, 2.46M → 6.36M hz) — filtered `texture()`
sampling is now in the same range as this backend's ordinary per-call
wrapper overhead (`textureSize()`/`textureLoad()` are ~2-3.3x, the plain
scalar case is ~2.5-3.4x), rather than a distinct, much larger cost.
Nearest filtering improved too (~3.7x → ~3.0x), for the same reason: its
own wrapped index was also being recomputed per channel before this.

Four fixes, one investigation: from an initial 11-19x, texture sampling
now costs roughly what any other `compileWasm` call costs — the copy
cache, the `magFilter` branch, the lerp reformulation, and this per-sample
setup each closed a real, distinct, measured piece of the gap, none of
them needing to guess at what the next bottleneck would be before
measuring it.

A fifth, found later while investigating why `.draw()` combined with a
texture was slower than expected (see "A whole grid in one call:
`.draw()`" below): `emitTexelFetchStores` — `textureLoad()`'s own
codegen, and also what `texture()`/`textureLod()` fall back to for an
integer sampler — had the exact same per-channel redundancy the filtered
path above was already fixed for (its bounds check and safe, clamped
texel index recomputed once per channel instead of once per texel), just
never applied there. A single isolated call barely shows it (wrapper
overhead dominates one call, masking savings on a small amount of
per-texel work), but it mattered once many texel fetches happen inside
one `.draw()` call, where wrapper overhead is already amortized away.

One real limit on how bad any of this is in practice, worth stating
precisely regardless of which part is fixed: even before the copy-caching
fix, the copy happened once per **call** to the compiled function, not
once per **texture() invocation** inside it — a function that samples the
same bound texture many times in a loop (a blur kernel, several ray-march
steps) always paid that cost once for the whole call, not once per sample,
so the very first measurement's single-sample-per-call numbers were
already a worst case for how the copy amortizes, not representative of
every texture-using program.

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

### A whole grid in one call: `.draw()`

Every measurement above points the same direction: `compileWasm`'s own
per-call marshalling cost is what makes it lose to `compileJS` for a
cheap, called-once function, and it wins as soon as there's enough work
in one call to amortize that cost — the crossover benchmark found a
handful of loop iterations was already enough, and the exact same cost
was what made filtered texture sampling look far worse than it actually
was until that got fixed too (see both sections above). Rendering a
`width x height` image one pixel at a time is the same problem at a much
larger scale — tens of thousands to millions of calls instead of one — so
`compileWasm`'s returned callable has a `.draw(ctx, width, height)` method
that moves the pixel loop inside the compiled module instead of leaving
it in JS.

The design settled on two things worth being explicit about, since an
earlier version of this did both differently:

- **Dimensions are a per-call argument, not a compile-time constant.**
  `width`/`height` are ordinary runtime `i32` values passed to a second
  exported WASM function, `"draw"`, not baked into the loop bound at
  compile time — the same compiled function can render a `2x2` grid on
  one call and a `1920x1080` one on the next.
- **Every compiled function can do both a single-pixel call and a
  whole-grid `.draw()` — there is no separate compile mode to choose
  between.** `"draw"` shares `main`'s own compiled body through a real
  WASM `call` instruction rather than a copy of its bytecode inlined into
  a loop: `main` compiles exactly as it always has, and `"draw"`'s loop
  writes each iteration's pixel coordinate into the same fixed
  `fragCoord()` address `main` already reads, then `call`s `main` once
  per pixel and copies its result into a growable output buffer (grown
  the same way the texture heap already grows, since a draw buffer's size
  isn't known until the `width`/`height` for that particular call are).
  Sharing `main`'s bytecode via a real call, rather than re-inlining a
  copy of it wrapped in extra loop structure, also sidesteps an entire
  class of bug an earlier version had to solve by hand: nothing about
  `Return()`/`Discard()`/`Break()`/`Continue()`'s branch-depth bookkeeping
  inside `main`'s body needs to change at all, regardless of whether
  `draw` ever calls it.

One real, useful side effect: a plain (non-stage) function returning an
aggregate type used to throw `"only supports a scalar result"`, since a
WASM function can only return one scalar natively. `needsResult` now
recognizes an aggregate root the same way it already recognizes an
explicit vertex stage, routing it through the same memory-based path a
stage program uses instead of throwing — which is what lets a per-pixel
`vec4` color work with `.draw()` with no stage or `output()` involved at
all.

`src/rmsl-wasm-draw.bench.ts` measures a `sqrt(distance to a uniform
center)` program, swept across a 128x128 and a 512x512 grid (a 16x
difference in pixel count, to check the win holds at scale rather than
resting on one arbitrarily chosen size), two runs, otherwise idle machine.
The per-pixel comparison loops reuse one `ctx` object and mutate its
`fragCoord` array in place rather than allocating a fresh one per pixel —
an earlier version of this measurement didn't, which piled up real
garbage-collection pressure unrelated to either backend's own per-pixel
cost, one that grew with the grid size and made an early size sweep
actively misleading (see the file's own history for the numbers that
mistake produced):

| Scenario | 128x128, Run 1 | 128x128, Run 2 | 512x512, Run 1 | 512x512, Run 2 |
|---|---|---|---|---|
| `.draw()` vs. `compileWasm` called once per pixel | 52.31x faster | 52.71x faster | 59.44x faster | 58.86x faster |
| `.draw()` vs. `compileJS` called once per pixel | 5.49x faster | 5.49x faster | 10.54x faster | 10.42x faster |

The second row is the comparison that actually matters — `compileJS`
called once per pixel is the realistic alternative anyone would reach for
today, not a per-pixel `compileWasm` loop — and here `.draw()`'s win
against it actually *grows* with grid size (~5.5x at 128x128, ~10.5x at
512x512): `compileJS`'s own per-pixel call overhead scales with the pixel
count same as anything else, so amortizing it across a bigger single
`.draw()` call pays off more, not less, at scale.

**Fixed**: a texture uniform and `.draw()` now work together. The draw
buffer's base address became a third runtime argument to the exported
`"draw"` function (after `width`/`height`) instead of a compile-time
constant, computed fresh on every `.draw()` call as wherever that call's
own texture heap actually ends — `textureHeapBase` itself, unchanged, when
there are no textures at all. Finding this also surfaced a real,
previously-latent bug: the offset handed to a `Float64Array` constructor
must be a multiple of 8, and nothing about this backend's own compile-time
bump allocator keeps addresses aligned (a texture's 44-byte metadata block
is the concrete case that doesn't) — any `.draw()`-returning function
whose compile-time allocation happened to land on a non-8-aligned address
would have thrown the first time a result was actually read back. Fixed by
rounding the computed buffer base up to the next multiple of 8.

Combining the two is correct. Its first measurement found a real
performance surprise, since fixed — worth recording both states plainly
rather than only the final number. Same benchmark file, a scenario
sampling a texture sized to match the grid once per pixel via
`textureLoad()`, same two grid sizes, two runs each:

| Scenario | 128x128, Run 1 | 128x128, Run 2 | 512x512, Run 1 | 512x512, Run 2 |
|---|---|---|---|---|
| First measurement | `compileJS` 1.26x faster | `compileJS` 1.33x faster | `.draw()` 1.08x faster | `.draw()` 1.07x faster |
| After the fix below | `.draw()` 1.12x faster | `.draw()` 1.13x faster | `.draw()` 1.62x faster | `.draw()` 1.64x faster |

The first measurement found `compileJS` actually winning at 128x128 — the
first case found where `.draw()` was the wrong choice. Not a caching bug
(confirmed directly: timing repeated calls with the same texture shows
the first call paying a real copy-in cost and every call after it roughly
4x cheaper, exactly the reference-equality cache working as designed) but
a real, reproducible finding: `.draw()` eliminates *per-call* marshalling
overhead, and that's still true here, but `textureLoad()`'s own
*per-pixel* cost (a bounds-checked, dynamically-addressed fetch — several
`select`s and a memory load) was real work that didn't go away, and at
this grid size it outweighed the marshalling savings entirely.

The cause turned out to be fixable, not inherent: `emitTexelFetchStores`
(`rmsl-wasm.ts`) recomputed its bounds check and its safe, clamped texel
index once per channel — 4 times over, for values that don't depend on
which channel is being read — the exact same redundancy `texture()`'s own
filtered sampling path had already been fixed for (see "Texture sampling
was dramatically slower..." above), just not yet applied to the unfiltered
path. Computing both exactly once per texel instead (the same "extra
scratch right after the node's own result" treatment used throughout this
file) flipped the result at both grid sizes: `.draw()` now wins by
roughly 1.1x at 128x128 and roughly 1.6x at 512x512, up from ~1.07x. Still
nowhere near the texture-free scenario's ~10x at the same size — a real,
inherent per-pixel cost for texture sampling remains — but `.draw()` is no
longer the wrong choice for this workload at either size measured.

## Status: Phase 1 through Phase 7 landed (except multi-return)

`compileWasmFn` and `compileWasm` exist in `src/rmsl-wasm.ts`, next to
`rmsl-glsl.ts`/`rmsl-wgsl.ts`/`rmsl-compile-js.ts` (see CONTRIBUTING.md for
the file layout). Tests are in `src/rmsl-wasm.test.ts` and
`src/rmsl-layout-interop.test.ts`.

**What it covers**, Phase 1's validated slice, Phase 2's full scalar op
parity, Phase 3's vectors/matrices as first-class values, Phase 4's control
flow, Phase 5's shader-stage surface, and Phase 6's texture sampling:

- `compileWasmFn(fn, options: CompileWasmFnOptions): { bytes: Uint8Array,
  params: WasmParam[], resultType: ShaderType }` — the module plus a
  description of what each exported-function argument (and the result)
  means.
- `compileWasm(fn, options): (ctx: JsShaderContext) => number | boolean |
  JsShaderResult` — same call signature as `compileJS`. A plain
  scalar-returning program returns the bare value (a `"bool"` result comes
  back as a real boolean, a `"uint"` one reinterpreted from WASM's
  always-signed i32 return, matching `compileJS`); a program using the
  shader-stage surface returns a `JsShaderResult`, identical in shape to
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
  `For`/`if`-chains before this backend ever sees them (`rmsl-core.ts`), so
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
  written explicitly — see "The shader-stage surface" below. Multi-return
  is the one thing from the original Phase 5 wishlist *not* included —
  `compileJS` doesn't have it either, so there was nothing to port; see
  the Phase 5 writeup below for why.
- `textureSize`, `textureLoad`, and `texture`/`textureLod` (nearest,
  bilinear, and trilinear filtering; `repeat`/`mirror`/`clamp` wrapping) for
  `sampler2D`/`sampler3D` and their integer (`isampler*`/`usampler*`)
  variants — matching `compileJS`'s own scope exactly: no cube maps, no
  mipmap/LOD (`textureLod`'s third argument is compiled nowhere, same as
  `compileJS`). Texture pixel data is copied into the compiled module's own
  linear memory rather than sampled through a call back into JavaScript —
  see "Texture data lives in linear memory, not behind a host call" below.

**What throws today** (deliberately — see the Phase list below for when each
lands): non-square or mismatched-shape matrix×matrix multiply,
`uniformArray`, cube-map sampling, and multi-return.
`compileWasmFn` throws `[RMSL] compileWasmFn: unsupported node type in
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
  *after* a call, not just write it before one. A `needsResult` flag
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
  `compileWasm` afterward into a `JsShaderResult` — identical to what
  `compileJS` already returns for the same program. `assertStageResult`
  (`rmsl-compiler-shared.ts`) is reused directly, unmodified — it only
  ever needed plain primitives (`shaderStage`/`lastType`/`positionWritten`),
  not a full `CompileCtx`.
- **`.draw()` is a second exported function sharing `main`'s bytecode via
  `call`, not a second compile mode or a copy of `main`'s body.**
  `collect()`/`walkStmt`/`walkExpr` compile `main` exactly as they always
  have; the module-assembly step at the end of `compileWasmFn` separately
  builds a `"draw"` function (only when the root produces a value at all)
  whose own body is just a `y`/`x` loop writing `fragCoord()`, calling
  `main` by function index, and copying the result into a growable output
  buffer. `compileWasm` exposes it as `.draw(ctx, width, height)` on the
  callable it returns. See "A whole grid in one call: `.draw()`" above for
  the design and the measured speedup.
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
  here already works. `UNIFORM_OPERAND_OPS` (`rmsl-core.ts`) already
  broadcasts a scalar operand to match the defining operand's width at
  AST-construction time for all of these *except* `mix`'s `t`, which is
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

## Phased plan

Roughly ordered by what unblocks the most; not a commitment to build all of
it.

### ~~Phase 2 — full scalar op parity~~ — done
See "Status" and "Design decisions already made" above for what landed:
every remaining `jsBinaryOp`/`jsUnaryMath` entry with a direct WASM opcode
(or a trivial derivation of one, like float `mod`/`round`/`fract`), every
comparison, logical, and bitwise op, and `int`/`uint`/`bool` as real `i32`.
Explicitly *not* included at the time: `clamp`/`mix`/`step`/`smoothstep` —
composite ops with no single opcode, closer in spirit to Phase 3's vector
work than to this phase's "one opcode per op" scope. These four landed
later, after Phase 6 (see "`clamp`/`mix`/`step`/`smoothstep` compile through
the same aggregate-value machinery vectors already use" below), reusing the
`materializeIfNeeded`/scratch-address machinery Phase 3 built rather than
needing anything new of their own.

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

### ~~Phase 5 — shader-stage surface~~ — done, except multi-return
`output()`, `varying()`, `attribute()`, `builtinPosition()`,
`builtinFragDepth()`, `fragCoord()`, and the `stage`/`derivatives`/
`reentrant` options all landed, matching `CompileJSOptions` instead of
`CompileFnOptions` — this is now a real alternative to `compileJS` for a
full vertex or fragment stage, not just a plain function.

Multi-return was explicitly **not** ported here, on purpose, not as an
oversight: research while planning this phase found `compileJS` itself
doesn't support it either — `compileJSFn` throws the identical "does not
support multi-return functions" error `compileWasmFn` already gave. Real
multi-return only exists one layer up, in `compileGLSL.vertex`/
`compileWGSL.vertex`'s stage-root (`Node | readonly Node[]`) handling, which
is genuinely new ground beyond `compileJS` parity, not a port of something
`compileJS` already has — left for a separate, later decision rather than
folded into this phase under the same name.

Two design points worth remembering if this gets touched again:
- **`compileWasmFn`'s "root must be a scalar" restriction only applies when
  `needsResult` is false** (mirroring `compileJS`'s own `ctx.jsNeedsRes`) —
  the moment a program touches `output()`/a vertex `varying()`/
  `builtinPosition()`/`builtinFragDepth()`, or an explicit `"vertex"` stage
  is requested at all, the compiled function's result arity switches from
  one (a plain WASM return) to zero, with everything (the function's own
  value included) read back from memory afterward instead.
- **A vertex stage's own result is the implicit position** whenever
  `builtinPosition()` was never explicitly written — `assertStageResult`
  requires it to be a `vec4` in exactly that case, and it's written directly
  into the same memory `builtinPosition()` would use, not a separate
  "value" slot. This was found by a test that initially didn't catch the
  case — `Fn(() => float(0))()` compiled with `stage: "vertex"` did not
  throw until `needsResult` was seeded from `options.stage === "vertex"`
  directly, not solely from what nodes the program happened to use.
- **Texture data lives in linear memory, not behind a host call.** An
  earlier option considered for Phase 6 was sampling through a JS host
  import, the same way the transcendental math functions already work —
  smaller to build, but it would mean a texture-sampling shader could never
  run standalone (no texture data without a JS engine to call back into),
  which cuts against part of why this backend exists in the first place.
  Instead, a texture uniform gets a small, fixed-size **metadata block** at
  a compile-time address (recording where its pixel data currently sits
  plus its width/height/depth/channel count/filter/wrap settings) and a
  share of a separate, **growable heap** for the pixel data itself, both
  populated by `compileWasm`'s wrapper before every call — unlike every
  other uniform kind here, a texture's size isn't fixed at compile time, so
  it can't get an ordinary Phase-3-style scratch address. The heap starts
  right after every compile-time-fixed allocation and is packed fresh each
  call (`compileWasm` sums the bound textures' byte sizes, calls the
  module's own `memory.grow` if the current buffer is too small, then
  refreshes its `DataView` — growing detaches the old `ArrayBuffer`).
  Pixel data is always stored as `f64` regardless of the source
  `TypedArray`'s own width, matching this backend's existing "`float` is
  f64" convention and keeping the sampling bytecode itself free of any
  per-texture width bookkeeping. A program that never uses a texture never
  touches any of this — no heap, no `memory.grow` call, byte-for-byte
  identical to before Phase 6.
- **Every sampling conditional is a `select`, matching the rest of this
  file's style.** Wrap mode (`clamp`/`repeat`/`mirror`) and the
  nearest-versus-linear filter choice are both runtime values read from a
  texture's own metadata, and — like this file's existing ternary/`min`/
  `max` handling — both outcomes are always computed and `select` just
  picks one, rather than branching. `textureLoad()`'s out-of-range case
  needed one extra safety step this pattern doesn't need elsewhere: since
  `select`'s "not taken" branch still executes, the address it reads must
  already be in range even when the real coordinate is wild, so the actual
  memory address always comes from a *clamped* copy of the coordinate,
  while the real, unclamped one only ever feeds the bounds comparison
  itself. `texture()`/`textureLod()` need no such clamp — wrapping already
  guarantees an in-range tap index by construction, so the wrapped index
  doubles as the safe one.

### ~~Phase 6 — texture sampling~~ — done, performance now comparable to any other call
See "Status" and "Texture data lives in linear memory, not behind a host
call" above for what shipped: `textureSize`/`textureLoad`/`texture`/
`textureLod` at the same scope `compileJS` itself has (`sampler2D`/
`sampler3D`, float and integer variants, no cube maps, no mipmap/LOD).
Correct throughout; performance was initially 11-19x slower than
`compileJS` per call. Four fixes landed — caching the last-copied texture
per slot by reference, branching on `magFilter` with a real `if`/`else`
instead of computing both the nearest and the filtered value on every
sample, reformulating the bilinear/trilinear lerp to stop duplicating an
expensive fetched value (`a + (b-a)*t` needs `a` twice; `a*(1-t) + b*t`
needs it once), and computing every per-sample wrap-addressed coordinate
once instead of once per channel. See "Why" above, "Texture sampling was
dramatically slower than `compileJS` — fixed". All four texture operations
now cost roughly what any other `compileWasm` call costs (~2-3.5x
`compileJS`'s own per-call cost, the same range the plain scalar case
already has) instead of a distinct, much larger penalty.

### ~~Phase 7 — parity testing infrastructure~~ — done
`compileWasm` is now a third backend checked by
`src/testing/shader-eval.ts`'s recording, alongside `compileGLSL`/
`compileWGSL` (see CONTRIBUTING.md's "Validity"/"Values" test layers) — a
case written once in `rmsl-js.test.ts`/`rmsl-eval.test.ts`-style files is
now checked against WASM automatically too, with no change needed to any
existing call site. `evaluateWASM` (`shader-eval.ts`) runs synchronously
alongside the CPU target, needing no browser or graphics device, so it
isn't gated by `RMSL_SKIP_GPU`/`RMSL_SKIP_SHADER_EVALUATION` the way
GLSL/WGSL are. Its result is compared against the CPU target with exact
equality, not `floatTolerance` — this backend's `float` is f64, matching
`compileJS`'s own JS-number arithmetic bit for bit (including the
transcendental functions, which both backends call through the literal
same `Math` object), so a real difference is a bug, never rounding.

Coverage wasn't "broad enough" yet in the sense the phase originally
imagined — several recorded cases exercise `clamp`/`mix`/`step`/
`smoothstep`/`uniformArray`/non-square matrix multiply, none of which
`compileWasmFn` supports (see "What throws today" above). Rather than
wait, a `compileWasmFn`-thrown error (always prefixed `"[RMSL]
compileWasmFn"`, confirmed across every throw site in `rmsl-wasm.ts`) is
treated as a countable, visible skip instead of a failure — reported as a
`[shader-eval] WASM: N of M ... not yet supported` line — while anything
else (a genuine `WebAssembly.RuntimeError` trap, a `CompileError`/
`LinkError`, or a numeric mismatch) still fails the run exactly like a
GLSL/WGSL disagreement does. This is what let the hookup land now instead
of waiting for full coverage, without weakening what the suite actually
guarantees.

This immediately found two real, previously undetected bugs, exactly the
value this phase was for:
- `compileWasm`'s wrapper looked up the compiled export as the literal
  string `instance.exports.main`, ignoring `options.name` entirely — silently
  broken for any function name other than `"main"`, undetected until now
  because every existing WASM test happened to use that exact name.
- `Switch`'s `Case()` (`rmsl-core.ts`) typed every case value as `float`
  regardless of the selector's actual `int`/`uint` type (`wrapValue`
  defaults a bare number to `float`, and nothing corrected it afterward).
  GLSL and WGSL both silently papered over the resulting type mismatch
  with an implicit cast at comparison codegen, so it never produced a
  wrong answer there — only the WASM backend's comparison codegen, which
  assumes both operands already match, turned it into a `WebAssembly.
  CompileError`. Fixed at the source (`typedOperand(v, selector._t)`
  instead of `wrapValue(v)`), which also removed the now-unnecessary casts
  from GLSL/WGSL's own output.
- A third, unrelated gap found by the same process: `emitConstructStores`
  (`rmsl-wasm.ts`) copied a construct's components straight from source to
  target with no conversion at all, so `vec3(...).toIVec3()` tried to
  `i32.store` a raw `f64.load`'s bits — a real `CompileError`, not a wrong
  number, but still a case that had simply never been exercised by
  `rmsl-wasm.test.ts`'s own hand-written cases before this. Fixed
  (`convertComponent`) for float↔int/uint; see "Open questions" for the
  one case deliberately left unhandled (a `bool` on either side of that
  conversion).

The realistic-workload benchmarking half landed too — see "Why" above,
"Crossover point: how many loop iterations before `compileWasm` wins".
The crossover for a `sum of sqrt(i)`-shaped loop turned out to be between
2 and 4 iterations, far lower than the 64-iteration case that originally
established a win existed at all — so "does the wrapper need to get
cheaper" narrows to specifically the loop-free, called-once case
(`rmsl-wasm-vs-js.bench.ts`'s scalar scenario), which is real, separate
work left for later rather than folded into this phase.

### ~~Phase 7.5 — `clamp`/`mix`/`step`/`smoothstep`~~ — done
See "`clamp`/`mix`/`step`/`smoothstep` compile through the same
aggregate-value machinery vectors already use" above for the design. All
four now work in both scalar and componentwise vector form, verified by
`rmsl-wasm.test.ts`'s dedicated test block and by Phase 7's cross-backend
recording (`npx vitest run src/rmsl-js.test.ts src/rmsl-eval.test.ts`),
whose `[shader-eval] WASM: N of 79 ... not yet supported` count dropped
from 17 to 3 as a direct result — the remaining 3 are `uniformArray`/
non-square matrix multiply cases, still open per "What throws today"
above.

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
- **Whole-function f32 arithmetic mode.** `GpuUniformLayout` (stage 2 of
  `docs/design-shared-layout-ir.md`) only narrows a uniform's *storage* to
  f32 at the memory boundary — internal arithmetic always stays f64
  (matching `compileJS`), same as an ordinary uniform. A different,
  materially bigger idea came up alongside it: a per-`Fn` mode where *every*
  scalar float op (`add`/`mul`/`sqrt`/...) actually computes in f32
  throughout, matching what a real GPU shader would compute bit-for-bit at
  every intermediate step, not just at the uniform boundary — useful for a
  CPU-side computation meant to verify or shadow a GPU one exactly. Not
  designed or started: it would need an f32 variant of nearly every
  scalar-op-emitting function in `rmsl-wasm.ts`, not one contained seam like
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
  (`rmsl-wasm.ts`) for the real bug Phase 7's cross-backend recording
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
    each independently fed the *same* time/parameter values by the
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
