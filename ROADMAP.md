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

## Status: Phase 1 landed

`compileWasmFn` and `compileWasm` exist in `src/rmsl.ts`, next to
`compileGLSLFn`/`compileWGSLFn` (CONTRIBUTING.md: "almost everything lives in
`src/rmsl.ts`" — this follows that rather than introducing a new module).
Tests are in `src/rmsl-wasm.test.ts`.

**What it covers**, exactly the prototype's validated slice, promoted to the
real API shape:

- `compileWasmFn(fn, options): { bytes: Uint8Array, params: WasmParam[] }` —
  the module plus a description of what each exported-function argument
  means.
- `compileWasm(fn, options): (ctx: JsShaderContext) => number` — same call
  signature as `compileJS`, for drop-in comparison on the ops it supports.
- Explicit function params (`options.params`), float uniforms, and vec3
  uniforms (split into three f64 params — see "Vectors have no home yet"
  below).
- `+`, `-`, `*`, `sqrt`, `greaterThan`, vec3 `dot`.
- `If`/`Else` via `toVar()`/`.assign()`, compiled to WASM's structured
  `if`/`else`/`end` — the same shape the prototype validated.

**What throws today** (deliberately — see the Phase list below for when each
lands): every other binary/unary op, `for`/`while`/`Loop`/`Switch`, vec2/vec4/
mat*, int/uint/bool, `uniformArray`, swizzles, `output()`/`varying()`/
`attribute()`/`builtinPosition()`/`builtinFragDepth()`/`fragCoord()`,
`textureLoad`/`texture`/`textureSize`, multi-return, and any non-`"float"`
result. `compileWasmFn` throws `[RMSL] compileWasmFn: unsupported node type
in <expr|vec3|statement> position: "<type>"` naming exactly what's missing,
which is also the fastest way to find the next thing worth doing here.

## Design decisions already made

- **Lives in `src/rmsl-wasm.ts`, alongside `src/rmsl-glsl.ts`/`rmsl-wgsl.ts`/
  `rmsl-compile-js.ts`** — the compiler was later split out of the original
  single `src/rmsl.ts` file by concern (see `CONTRIBUTING.md`), and this
  backend followed the same one-file-per-backend pattern. It reuses the same
  untyped internal node shape (`node.type`/`node.params`/`node.value`)
  `compileJSNode` already switches on, imported from `rmsl-core.ts` and
  `rmsl-compiler-shared.ts` — no new node representation to keep in sync.
- **f64 everywhere, for now** — but only because Phase 1 never touches
  anything but `"float"`. This matches the JS backend's exact-arithmetic
  semantics (no tolerance needed comparing the two, only for transcendentals
  later — see CONTRIBUTING.md's testing section on this same point for
  `compileJS`), but the JS backend collapses `int`/`uint` to the same f64
  representation as `float` only because JS numbers have no other option.
  WASM does have another option — real `i32` — and Phase 2 uses it: see the
  "int/uint use real i32" decision there. So "f64 everywhere" describes
  Phase 1's actual scope, not a permanent design stance.
- **Synchronous instantiation** (`new WebAssembly.Instance(new
  WebAssembly.Module(bytes))`), matching `compileJS`'s synchronous `new
  Function(source)()`. Fine for the module sizes here; revisit if a module
  grows large enough that sync compilation stalls a browser main thread (see
  Open questions).
- **Vectors have no home yet.** There is no linear memory in this backend
  yet, so a vec3 uniform becomes three separate f64 WASM params rather than
  one aggregate — `WasmParam`'s `{ kind: "uniform", slot, axis }` shape
  records which. This is a deliberate, load-bearing simplification, not an
  oversight: it's what let Phase 1 ship without first deciding a memory
  layout. It will need to change before uniform arrays, textures, or a
  `toVar()`'d vector are possible (see Phase 3 and Phase 6).
- **No shader-stage concept yet.** `compileWasmFn`'s options are
  `CompileFnOptions` (`name` + `params`), the same shape
  `compileGLSLFn`/`compileWGSLFn` use for standalone functions — not
  `CompileJSOptions`'s `stage`/`derivatives`/`reentrant`. A plain
  float-returning function is a much smaller surface than a full fragment/
  vertex stage; Phase 5 is where that gap gets closed.

## Phased plan

Roughly ordered by what unblocks the most; not a commitment to build all of
it.

### Phase 2 — full scalar op parity
Every remaining `jsBinaryOp`/`jsUnaryMath` entry in the JS backend that has a
direct WASM opcode: `div`, `mod`, `min`, `max`, `pow`, the trig/exponential
family (`sin`/`cos`/.../`exp`/`log` have no native WASM opcode — decide
per-op whether to hand-roll a polynomial approximation or import a host
function, mirroring how a real wasm toolchain's libm would), comparisons
(`lessThan`, `equal`, ...), logical (`and`/`or`/`not`), bitwise.

**`int`/`uint` use real `i32`, not f64** — mirroring GLSL/WGSL, not the JS
backend. GLSL/WGSL already know each node's declared type (`node._t`) and
use it to pick `int`/`i32` text over `float`/`f32`; WASM has that same type
information available and, unlike JS, a real integer type to put it in. The
JS backend gets away with f64-only because JS numbers don't distinguish
int from float at all and its bitwise operators silently coerce through
`ToInt32` — WASM has no such coercion, and every instruction (`f64.add` vs
`i32.add`) is a distinct, explicitly-chosen opcode. Copying JS's approach
here would also cost real correctness: GLSL/WGSL integers wrap at 32 bits
on overflow and f64 arithmetic doesn't, so an f64-collapsed WASM int would
silently disagree with the other three backends on any shader relying on
wraparound. This is the first place the backend needs to track a concrete
WASM storage type (`f64` vs `i32`) per node rather than assuming one type
for everything, including explicit conversions where an expression mixes
the two (`float(intValue)`, `intValue.mul(floatValue)`, ...). `bool` likely
rides along as `i32` too, matching WASM's own boolean-as-i32 convention
(which Phase 1's comparison ops — `f64.gt` etc. — already produce).

### Phase 3 — vectors and matrices as first-class values
The load-bearing decision: keep the "split into N scalar slots" approach
(cheap, no memory, but doesn't scale to `uniformArray`, swizzled writes, or a
`toVar()`'d vector reused across a branch) or introduce WASM linear memory
with a fixed struct layout (an aggregate finally has one address, but now
every read/write is a `load`/`store` at an offset instead of a `local.get`,
and offsets have to be planned). Prototype's dot-product test dodged this by
never storing a vector, only reducing it immediately — a `toVar()`'d vec3
forces the question. vec2/vec4/mat2-4 follow whichever layout gets picked.

### Phase 4 — control flow parity
`for`/`while`/`Loop`/`Break`/`Continue`/`Return`/`Switch`/`Discard`. WASM's
structured `loop`/`br`/`br_if` map reasonably directly to `for`/`while`; the
prototype's `if`/`else` precedent shows the shape. `Discard` has no scalar-
function equivalent — needs a sentinel return or Phase 5's stage concept to
mean anything.

### Phase 5 — shader-stage surface
`output()`, `varying()`, `attribute()`, `builtinPosition()`,
`builtinFragDepth()`, `fragCoord()`, multi-return, the `stage`/`derivatives`/
`reentrant` options `compileJS` has. This is what turns "compiles a plain
function" into "is an alternative to `compileJS` for a real shader graph,"
matching `CompileJSOptions` instead of `CompileFnOptions`.

### Phase 6 — texture sampling
The one place linear memory stops being optional: `JsTextureData` is
arbitrary-sized pixel data, not a handful of scalars. Needs a real memory
layout, a way to copy a texture's data into WASM memory before each call (or
keep it resident and re-copy only on change), and filtering/wrap logic
implemented either in emitted WASM or as an imported host function.

### Phase 7 — parity testing infrastructure
Once coverage is broad enough, hook `compileWasm` into
`src/testing/shader-eval.ts`'s recording the way `compileGLSL`/`compileWGSL`
already are (see CONTRIBUTING.md's "Validity"/"Values" test layers), so a
case written once in `rmsl-js.test.ts`-style files is checked against WASM
automatically instead of needing its own file. Also: benchmark against
realistic workloads, not the microbenchmarks that shaped Phase 1 — a real
picking scene's actual call pattern, not a tight synthetic loop.

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

## Non-goals

Not a GPU-side WASM or SPIR-V backend — this is CPU-only, the same niche
`compileJS` already fills. GPU compute is GLSL/WGSL's job.
