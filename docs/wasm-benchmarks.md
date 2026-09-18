# WASM backend: benchmark history and rationale

The measurement narrative behind `compileWasm`/`compileWasmFn`
(`src/backends/wasm/wasm.ts`) — why the backend exists, every
re-measurement taken as it grew, and the fixes each one led to. Moved out
of `ROADMAP.md` to keep that file scannable as a status/plan document;
this file is the historical record backing its claims.

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

### Re-measured with `src/wasm-vs-js.bench.ts`/`wasm-loop.bench.ts` at commit `7b90863`

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
`npx vitest bench src/wasm-vs-js.bench.ts src/wasm-loop.bench.ts`
at this commit):

| Scenario                                                                                      | Result                                                                  |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Scalar `sqrt(a*a+b*b+c*c)`, three float params, through `compileWasm`                         | `compileJS` **~2.5-3.4x faster**                                        |
| Same, calling the raw exported WASM function directly (bypassing `compileWasm`'s ctx wrapper) | `compileJS` ~1.0-1.15x faster — essentially a tie                       |
| vec3 `dot` + `If`/`Else` (uniforms, through Phase 3's linear memory), through `compileWasm`   | `compileJS` **~3.1-3.2x faster**                                        |
| `For` loop, 64 iterations of `sum += sqrt(i)` per call                                        | `compileWasm` **~4.05x faster**, both runs agreeing to 2 decimal places |

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
starts. `src/wasm-crossover.bench.ts` at commit `b629a19` sweeps the
same `sum of sqrt(i)` loop workload across iteration counts, each compiled
once up front (`npx vitest bench src/wasm-crossover.bench.ts`, two
runs, otherwise idle machine):

| Loop length | Run 1                      | Run 2                      |
| ----------- | --------------------------- | --------------------------- |
| 1           | `compileJS` 1.24x faster   | `compileJS` 1.25x faster   |
| 2           | `compileJS` 1.28x faster   | `compileJS` 1.25x faster   |
| 4           | `compileWasm` 1.08x faster | `compileWasm` 1.21x faster |
| 8           | `compileWasm` 1.41x faster | `compileWasm` 1.47x faster |
| 16          | `compileWasm` 2.30x faster | `compileWasm` 2.46x faster |
| 32          | `compileWasm` 2.87x faster | `compileWasm` 3.14x faster |
| 64          | `compileWasm` 3.57x faster | `compileWasm` 4.08x faster |
| 128         | `compileWasm` 4.24x faster | `compileWasm` 4.37x faster |

Both runs agree on which side of the crossover every length falls on
(only 4 iterations wobbles between a 1.08x and a 1.21x win, never a loss),
so the crossover for this workload sits **between 2 and 4 loop iterations**
— strikingly low. Most of the fixed wrapper cost the earlier scalar-call
measurement blamed for `compileWasm`'s loss turns out to have nothing to
do with looping specifically: the moment a program is loop-shaped at all
(even a loop that only runs once or twice), it's already close to
break-even, and three or four iterations of real work tip it into a win.

One structural point worth being precise about, not blurring together:
this crossover is measured entirely _within_ loop-shaped programs (1
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
case (`wasm-vs-js.bench.ts`'s scalar scenario), not a general one —
left for a separate pass, since narrowing where to look was this
benchmark's job, not fixing it.

### Texture sampling was dramatically slower than `compileJS` — fixed

`src/wasm-texture.bench.ts` at commit `3f0ef46` first measured the
three Phase 6 texture operations against an 8x8 texture (`npx vitest bench
src/wasm-texture.bench.ts`, two runs, otherwise idle machine):

| Scenario                                          | Run 1                     | Run 2                     |
| --------------------------------------------------- | --------------------------- | --------------------------- |
| `textureSize()` (metadata only, no sampling math) | `compileJS` 11.12x faster | `compileJS` 11.08x faster |
| `textureLoad()` (one unfiltered texel)            | `compileJS` 14.57x faster | `compileJS` 14.57x faster |
| `texture()` (bilinear filtering)                  | `compileJS` 18.66x faster | `compileJS` 19.12x faster |

This was a real regression from the "runs standalone, no host call needed"
story Phase 6 was built around, not noise — but the cause was precise, not
mysterious: `compileWasm`'s wrapper unconditionally copied the _entire_
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

| Scenario                         | Run 1                     | Run 2                     |
| ----------------------------------- | --------------------------- | --------------------------- |
| `textureSize()`                  | `compileJS` 2.07x faster  | `compileJS` 2.12x faster  |
| `textureLoad()`                  | `compileJS` 3.33x faster  | `compileJS` 3.25x faster  |
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
_cheaper_ for nearest (its own codegen actually takes a shorter branch) —
which is exactly why nearest's ratio (17.47x/19.25x) looks _worse_ than
bilinear's (13.09-13.36x): `compileJS` improved and `compileWasm` didn't
move at all. This directly confirms `emitTextureSampleStores`'s own doc
comment: it computes _both_ the nearest and the bilinear value
unconditionally on every sample and only `select`s between them at run
time on the texture's own `magFilter`, so the expensive path's bytecode
runs whether or not a program ever asks for it — the cost isn't "bilinear
filtering is expensive", it's "this codegen always pays bilinear's cost".

Fixed at commit `83200c6`: `emitTextureSampleStores` now branches on
`magFilter` with a real WASM `if`/`else` instead of computing both paths
and `select`ing — the one deliberate departure from this function's
otherwise-branchless style, exactly because `select` was the mechanism
paying for the unused path. Re-measured at the same commit, two runs:

| Scenario                        | Run 1                     | Run 2                     |
| ---------------------------------- | --------------------------- | --------------------------- |
| `texture()`, nearest filtering  | `compileJS` 3.60x faster  | `compileJS` 3.68x faster  |
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
expensive — and the two horizontal lerps were then duplicated _again_ by
the outer vertical lerp, so one corner's fetch was emitted **four times**
per channel (the 3D case duplicated its two bilinear results the same way
on top of that). `a*(1-t) + b*t` is the same value needing `a`/`b` each
exactly once, duplicating only the cheap blend weight `t` instead —
applied at all three lerp levels via one shared `lerp()` helper.
Re-measured, two runs:

| Scenario                        | Run 1                    | Run 2                    |
| ---------------------------------- | -------------------------- | -------------------------- |
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

| Scenario                        | Run 1                    | Run 2                    |
| ---------------------------------- | -------------------------- | -------------------------- |
| `texture()`, nearest filtering  | `compileJS` 3.02x faster | `compileJS` 3.01x faster |
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
already had one or the other. To isolate it, `src/wasm-vs-js.bench.ts`
as of commit `7b90863` was copied unmodified — `git show
7b90863:src/wasm-vs-js.bench.ts` — into a worktree checked out at
`9b845b7` (the commit immediately before linear memory landed, `f58c93b`)
and run there, two runs, same idle-machine conditions:

| Scenario                                                  | Before linear memory (`9b845b7`) | After (`7b90863`)                                    |
| ------------------------------------------------------------ | ----------------------------------- | ------------------------------------------------------- |
| Scalar `sqrt(...)`, through `compileWasm`'s wrapper       | `compileJS` ~2.5x faster         | `compileJS` ~2.5-3.4x faster — **slightly worse**    |
| vec3 `dot` + `If`/`Else`, through `compileWasm`'s wrapper | `compileJS` ~6.3x faster         | `compileJS` ~3.1-3.2x faster — **roughly 2x better** |

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

`src/benches/wasm-draw.bench.ts` measures a `sqrt(distance to a uniform
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

| Scenario                                          | 128x128, Run 1 | 128x128, Run 2 | 512x512, Run 1 | 512x512, Run 2 |
| ---------------------------------------------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| `.draw()` vs. `compileWasm` called once per pixel | 52.31x faster  | 52.71x faster  | 59.44x faster  | 58.86x faster  |
| `.draw()` vs. `compileJS` called once per pixel   | 5.49x faster   | 5.49x faster   | 10.54x faster  | 10.42x faster  |

The second row is the comparison that actually matters — `compileJS`
called once per pixel is the realistic alternative anyone would reach for
today, not a per-pixel `compileWasm` loop — and here `.draw()`'s win
against it actually _grows_ with grid size (~5.5x at 128x128, ~10.5x at
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

| Scenario            | 128x128, Run 1           | 128x128, Run 2           | 512x512, Run 1         | 512x512, Run 2         |
| ---------------------- | --------------------------- | --------------------------- | ------------------------ | ------------------------ |
| First measurement   | `compileJS` 1.26x faster | `compileJS` 1.33x faster | `.draw()` 1.08x faster | `.draw()` 1.07x faster |
| After the fix below | `.draw()` 1.12x faster   | `.draw()` 1.13x faster   | `.draw()` 1.62x faster | `.draw()` 1.64x faster |

The first measurement found `compileJS` actually winning at 128x128 — the
first case found where `.draw()` was the wrong choice. Not a caching bug
(confirmed directly: timing repeated calls with the same texture shows
the first call paying a real copy-in cost and every call after it roughly
4x cheaper, exactly the reference-equality cache working as designed) but
a real, reproducible finding: `.draw()` eliminates _per-call_ marshalling
overhead, and that's still true here, but `textureLoad()`'s own
_per-pixel_ cost (a bounds-checked, dynamically-addressed fetch — several
`select`s and a memory load) was real work that didn't go away, and at
this grid size it outweighed the marshalling savings entirely.

The cause turned out to be fixable, not inherent: `emitTexelFetchStores`
(`wasm.ts`) recomputed its bounds check and its safe, clamped texel
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

(`.draw()` was later renamed `.batch()` once the same `CpuRoutine` shape
started covering vertex/fragment/compute invocations too, not just image
rendering — see `ROADMAP.md`'s "Design decisions already made" for the
rename and `docs/wasm.md` for the current API surface. Kept under its
original name in this file since it's a historical measurement, not
current API documentation.)

### `createWasm` vs `rasterizeTriangles(compileJS)`: the same "whole grid in one call" win, for real geometry

The generic rasterizer module (`src/backends/wasm/rasterizer.ts`/`.wat`,
see `ROADMAP.md`'s "generic, precompiled rasterizer module") is the same
`.batch()` idea one level up: instead of moving a per-*pixel* loop inside
WASM, it moves the per-*vertex* and per-*pixel* loop — vertex transform,
near-plane clipping, edge-function coverage, perspective-correct
interpolation, depth test — inside one WASM call, driven by `compileWasm`/
`createWasm`. The realistic alternative it replaces is
`rasterizeTriangles` (`src/backends/cpu-rasterizer.ts`) driving a
`compileJS`-compiled vertex/fragment pair from the host side — a real JS
function call per vertex and per covered pixel, the same per-call
overhead `.batch()` already amortizes for a fragment-only program.

`src/benches/wasm-rasterizer.bench.ts` measures a rotating, per-vertex-
colored quad (2 triangles, 6 non-indexed vertices — the same scene
`apps/adapters`' `js-vtx`/`wasm-vtx` demo draws) at 128x128 and 512x512,
two runs, otherwise idle machine:

| Scenario                                           | 128x128, Run 1 | 128x128, Run 2 | 512x512, Run 1 | 512x512, Run 2 |
| --------------------------------------------------- | -------------- | -------------- | -------------- | -------------- |
| `createWasm` vs. `rasterizeTriangles(compileJS)` | 8.04x faster   | 8.08x faster   | 8.44x faster   | 8.85x faster   |

Consistent with `.draw()`'s own result above: the win holds — and grows
slightly — at the larger grid size, since `rasterizeTriangles`'s per-pixel
JS call overhead scales with pixel count same as anything else, while
`createWasm`'s one WASM call amortizes it regardless of grid size. This
motivated `apps/adapters`' own `wasm-vtx` demo option switching from
`compileWasmRoutine` + `rasterizeTriangles` (which never exercised this
module at all) to `createWasm` directly.

### `compileJS` vs `createWasm`: now a fair fight — both sides have the same clip/depth scope

`compileJS` (`src/backends/js/rasterizer.ts`) ported `rasterizer.wat`'s
algorithm to plain JS — near-plane clipping and a persistent LEQUAL depth
buffer, the same scope `WasmRasterRoutine` has — so `apps/adapters`'
`js-vtx` demo option could switch from `rasterizeTriangles` (no clipping,
no depth test) to `createJs` too, the same way `wasm-vtx` switched to
`createWasm` above. That makes the section above's comparison stale as a
"which backend is faster" measurement — it compared `createWasm` against
a *simpler* JS implementation, not against JS's own feature-equivalent
one — so `wasm-rasterizer.bench.ts` was updated to compare `compileJS`
against `createWasm` directly instead of `rasterizeTriangles`:

| Scenario                            | 128x128, Run 1 | 128x128, Run 2 | 512x512, Run 1 | 512x512, Run 2 |
| ------------------------------------ | -------------- | -------------- | -------------- | -------------- |
| `createWasm` vs. `compileJS`        | 9.16x faster   | 9.27x faster   | 9.19x faster   | 7.83x faster   |

Similar magnitude to the `rasterizeTriangles` comparison above, which
makes sense: both JS implementations pay the same fundamental cost
`compileWasm`'s design exists to amortize (a real function call per
vertex and per covered pixel from JS, vs. one WASM call doing the same
work internally) — adding clipping/depth to the JS side made it more
_capable_, not faster, since the added work (the clip test, the depth
compare) is itself plain JS running per vertex/pixel same as before.

**Why the demo app shows "same fps" for both, then:** `requestAnimationFrame`
caps at the display's refresh rate (typically 60Hz), and this benchmark's
scene (one quad, 2 triangles) is cheap enough that *both* backends clear
that ceiling with room to spare — so the demo's FPS reading measures "did
we hit vsync," not "how fast is this backend," for either option. The
~8-9x gap above is the real number; it's just invisible behind the cap
for a scene this small. It would show up directly in the demo with either
a much bigger scene (more triangles/pixels) or an FPS counter that
doesn't cap at vsync (e.g. summing per-frame CPU time instead of counting
frames).
