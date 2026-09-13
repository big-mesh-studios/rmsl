# Code review — `fix/compiler-codegen-bugs`

Reviewed 2026-07-18 at `18b211a`, 19 commits ahead of `origin/main`. Ten finder
passes over `git diff main...HEAD`, then every surviving candidate re-verified by
running the compiler and reading the emitted shader text.

This is the complete set of findings, not the subset that fitted a report. Items
1–27 came from the first pass, 28–38 were produced by the same passes but cut by
an output cap, and 39 onwards were found while fixing the others.

Baseline at review time: typecheck clean, 142 tests green, suite 25.1s.
Now: typecheck clean, 191 tests plus 17 type tests green, suite ~8s.

## Status

| #   | Finding                                                            | Status               |
| --- | ------------------------------------------------------------------ | -------------------- |
| 1   | Nodes reached twice were compiled twice                            | `2c1d820`            |
| 2   | Matrix-from-matrix constructor expanded as a diagonal              | `6c79295`            |
| 3   | `compileWGSLFn` called a helper it never emitted                   | `a24ae1b`            |
| 4   | GLSL fragment emitted no colour write                              | `19e66e8`            |
| 5   | GLSL wrote the stage result into every declared output             | `19e66e8`            |
| 6   | `mod` meant three different things                                 | `5b64632`, `c7b2f4c` |
| 7   | Comparison result type contradicted its signature                  | `27d503f`            |
| 8   | `"0.0"` sentinel collides with a real zero                         | `8f33709`            |
| 9   | Vertex guard rejects an explicit `builtinPosition()` write         | `8f33709`            |
| 10  | WGSL drops all but the last loop update statement                  | `027d1d1`            |
| 11  | `REDUCING_OPS` omits `matrixElement`                               | `85ff9fd`            |
| 12  | Integer-literal typing absent in `comp()`, too narrow in `op()`    | `271c43c`            |
| 13  | `bvec` uniforms are non-host-shareable in WGSL                     | `3685249`            |
| 14  | Nested swizzle assignment emits an unassignable target             | `1a2e2b3`            |
| 15  | `builtinPosition()` in a fragment stage is undeclared              | `1a2e2b3`            |
| 16  | `evaluateWGSL` returned 0 for a shader that never compiled         | `027d63c`            |
| 17  | An empty GLSL info log read as a valid shader                      | `03143d9`            |
| 18  | Only the last failing shader per test was reported                 | `03143d9`            |
| 19  | A run that recorded nothing passed                                 | `03143d9`            |
| 20  | One flag silently disabled both GPU layers                         | `b1b52f4`            |
| 21  | The graphics device was never released                             | `b1b52f4`            |
| 22  | Tolerance was tighter than one float's last place                  | `b1b52f4`            |
| 23  | The Stryker skip flag was never set                                | `b1b52f4`            |
| 24  | WGSL validation ran one round trip per shader                      | `fce6519`            |
| 25  | `evaluateGLSL` opens a browser page per call                       | `4a1e1ca`            |
| 26  | Version not bumped despite breaking changes                        | not ours             |
| 27  | `docs/compilation.md` vertex example now throws                    | PR note              |
| 28  | `inverse()` sends every non-mat3 to the mat4 helper                | `9fe75ae`            |
| 29  | The two `case "let"` arms return different expressions             | **open**             |
| 30  | Shared subexpressions are re-emitted rather than named             | follow-up            |
| 31  | `NodeOps` denies operations the runtime implements                 | `42da16c`            |
| 32  | The raw-array test's assertions cannot fail                        | **open**             |
| 33  | The two testing modules duplicate their GPU setup                  | `4a1e1ca`            |
| 34  | Stage-output typing is a runtime throw, not a signature            | `1288d0e`            |
| 35  | Nothing ensures a new test file gets validated                     | **open**             |
| 36  | Check-then-act race on the browser singleton                       | `4a1e1ca`            |
| 37  | `op()` and `comp()` allocate on every operator call                | **open**             |
| 38  | Uniform bindings ordered by string comparison                      | **open**             |
| 39  | Skip warnings were invisible                                       | `b1b52f4`            |
| 40  | Thirty-four casts to any in the tests, hiding checks               | `7670845`            |
| 41  | `Node<any>` let the standalone compilers accept anything           | `d6477a2`            |
| 42  | The recording stand-ins were typed as taking anything              | `1288d0e`            |
| 43  | A vertex stage returning nothing was never checked                 | `1288d0e`            |
| 44  | Caching the rendering context made the suite flaky                 | `fd36c06`            |
| 45  | Matrix-vector transforms are never value-checked                   | **open**             |
| 46  | The value harness reads back one float, so only floats are checked | **open**             |

Thirty-six fixed, seven open, one deferred. Items 40 to 44 were found while fixing the
others; 44 was introduced by the fix for 25 and 33 rather than found in the
code as it stood. #26 belongs to whoever publishes the
package, not to this branch; #27 is called out in the pull request instead of
being changed here.

## Notes on what was fixed

**#1** was fixed at the traversal layer rather than patched: the compiler
memoises what each node produced, keyed by the node itself, so the name-based
`ctx.declared` bookkeeping is gone. That also closed the silently-doubled side
effects a narrower fix would have left.

**#6** needed a semantic decision, not just a fix. Floored, following GLSL's
`mod()`, with WGSL and the constant folder brought into line and the choice
recorded in `docs/api.md`. Integers keep each backend's `%`, since GLSL leaves
that undefined for negative operands. Three.js TSL independently makes all three
of the same choices, down to the same subtraction; `c7b2f4c` follows it the rest
of the way by emitting a helper per width instead of inlining.

**#7** is enforced by the type checker, so it is pinned by `src/rmsl.test-d.ts`
rather than a runtime assertion.

**#39** was not in the original review. The validation layer already tried to
announce a skipped run, but used `console.warn`, which vitest intercepts without
surfacing — so the message saying nothing had been checked was itself unseen.
Both layers write to stderr now.

**#23** could not be fixed by setting the variable its comment named, because
Stryker cannot set one from its config file at all. It needed an npm script.

---

# Open findings

## 26. Version not bumped despite breaking changes — not ours

Left to whoever publishes the package. Recorded because the branch is a major
version's worth of change: `.node()` removed, vector comparisons returning a
boolean vector, a scalar compared against a vector now a type error, fractional
and negative literals beside integers now refused, `builtinPosition()` in a
fragment stage now refused, `inverse()` on a non-square matrix now refused,
`mod` answering differently for negative operands, GLSL fragment shaders gaining
an implicit colour output, and GLSL no longer writing the stage result into
declared outputs.

## 27. `docs/compilation.md` vertex example now throws — PR note

The documented multi-return example ends in two floats and is passed to
`compileGLSL.vertex`, which now refuses a result that cannot become a position.
The error is right and the example predates it. Called out in the pull request
rather than changed here.

## 29. The two `case "let"` arms return different expressions

`src/rmsl.ts`

GLSL returns `expr: lhs.expr`; WGSL returns `expr: varName`, where
`varName = node.params[0].varName || lhs.expr`. They coincide whenever `varName`
is unset, so this is latent rather than active, and memoisation now guarantees a
node compiles to one expression per backend. Still worth collapsing: the two arms
are otherwise the same code, and a divergence here is the kind that surfaces as
an undefined identifier much later.

## 30. Shared subexpressions are re-emitted rather than named — follow-up

Deliberately out of scope for this branch, which is about the compiler emitting
correct shaders. Reducing common subexpressions is an optimisation, and mixing
one in would blur what the branch is for.

Memoisation stops a node's _statements_ being emitted twice, but a pure
expression reached from several places still has its text substituted at each
one. Measured on `apps/shared/shader.ts`: the generated GLSL contains
`_rmsl_u2 * vec4(_rmsl_v1.x, _rmsl_v1.y, -1.0, 1.0)` three times on one line, so
every fragment runs three matrix-vector products where the source wrote one — 48
multiply-adds instead of 16, at roughly 2M fragments a frame.

The information needed is already there: the memo map added for #1 knows which
nodes are reached more than once. Spilling those into a generated temporary is
the whole change.

Worth raising as an issue rather than carrying here. Three.js TSL does not do
this either, which is either a reason to be careful or an opportunity, depending
on why — worth establishing before building anything.

## 45. Matrix-vector transforms are never value-checked

Found by applying surviving mutants by hand and running the full suite, real
compilers included. Two mutations of the matrix-times-vector emitter are not
noticed by anything:

- `mat * vec4(v, 1.0)` becomes `vec4(v, 1.0) * mat`. Both are valid GLSL — the
  second multiplies by the transpose — and the results differ.
- `vec4(v, 1.0)` becomes `vec4(v, 0.0)`, which is the difference between
  transforming a point and a direction. In a real shader that silently drops
  the translation from a model or view matrix, putting the geometry at the
  origin.

This is the operation a vertex shader is mostly made of, and `apps/shared`
depends on it. Both emit shaders that compile, so validation cannot see it, and
the value checks read back a single float, so they never look. See #46.

## 46. The value harness reads back one float, so only floats are checked

`evaluateGLSL` renders to a one-pixel target and returns the red channel;
`evaluateWGSL` reads one f32 from a storage buffer. So the only expressions
whose _result_ is ever checked are those reducing to a float. Everything vector
or matrix valued is checked for compiling and nothing else.

Reading back all four channels, and a vec4 from the storage buffer, would cover
the vector operations — including #45, and swizzles, constructors and the
component-wise builtins, none of which currently have a value check.

## 32. The raw-array test's assertions cannot fail

`src/rmsl-usage.test.ts` — "wraps raw arrays by length"

For `vec2(1, 1).mult([1, 2])`, the emitted GLSL contains `vec2(1, 1)` from the
receiver regardless of how `wrapValue` handled the array, so
`expect(glsl).toContain("vec2(")` passes even if the array were wrapped as a
float or at the wrong width. The same holds for the vec3, vec4 and WGSL rows.
Only the shader validator would notice, which the test's own comment concedes.

`wrapValue`'s length-9 (mat3), length-16 (mat4) and fallback branches remain
uncovered.

Confirmed by applying the mutation: forcing every array to wrap as a vec3 leaves
this test passing. The shader validation does catch that particular defect, so
the consequence is smaller than it first appears — but the test is not what is
catching it, and it reads as though it were.

## 35. Nothing ensures a new test file gets validated

`src/testing/shader-validity.ts`

Validation is reachable only through the aliased import plus a hand-written
`afterAll` in one file. `src/rmsl-eval.test.ts` is already the counterexample —
it imports `compileWGSLFn` directly and records nothing. Module state is isolated
per file, so a second file cannot share the existing hook; every future test file
must repeat both the aliasing and the `afterAll`, and forgetting either fails
open.

#19 added a floor for the case where _nothing_ is recorded, which catches the
alias being lost from the existing file but not a new file that never had it.
A vitest setup file, or a lint rule, would close the rest.

## 37. `op()` and `comp()` allocate on every operator call

`src/rmsl.ts`

`op()` builds three intermediate arrays where the previous version built one, and
its broadcast pass reallocates even when no operand needs widening. `comp()`
allocates a `widths` array on every comparison including scalar ones. Benchmarked
at roughly 3.5× over the pre-branch shape, which is about 0.08ms per 200-line
shader — small for a compile-once library, and worth doing only while the file is
open for another reason.

## 38. Uniform bindings ordered by string comparison

`src/rmsl.ts` — `compileFnBody`

Uniforms are sorted with `slot.localeCompare`, which orders `_rmsl_u10` before
`_rmsl_u2`. Only matters if a host assigns bindings positionally rather than by
reflection. Not verified against a real host.

---

# Structural note

Findings 11, 12, 13 and 28 are one problem wearing four hats. An operation's
result type is decided in one of several hand-maintained places:

- `REDUCING_OPS` — operations whose result is not the first operand's type
- `VALUE_OPERAND` — where the defining operand sits
- `UNIFORM_OPERAND_OPS` — operations needing broadcast
- `TYPE_WIDTH` — component counts, no matrix entries
- `MATRIX_DIMENSIONS` — matrix shapes
- hardcoded `_t` at construction sites (`texture`, `multVec`, `all`, `any`)
- `tryFold` — a third independent definition of each operation's meaning
- the two backend switches

Every table is individually correct. Every one is also already missing an entry.
Adding one operation means editing, in lockstep: `ShaderType`, the `NodeOps`
registry, the operation interface, the `NodeImpl` method, three or four tables,
`typeToGLSL`, `typeToWGSL`, `tryFold`, both switches, and the docs. Nothing fails
if a step is skipped — the symptom is a wrong `_t`, an unbroadcast operand, or a
silently unfolded constant, surfacing later as an invalid shader.

A single operation-definition table — result type, arity, per-backend rendering,
optional fold — would make an operation's metadata either complete or visibly
absent. `CONTRIBUTING.md` documents the current process; it is not a substitute
for fixing it.

Two smaller versions of the same theme:

- `op()` and `comp()` each grew their own copy of the scalar-broadcast loop.
- The `WGSL_HELPERS` matrix inverses re-implement the JavaScript `mat4Inverse`
  cofactor expansion in a second language with no shared source and no
  cross-check. They have already drifted: the JavaScript version guards
  `det === 0`, the WGSL version divides by zero. `_rmsl_inverse3` has no
  JavaScript counterpart at all. Both helper bodies were checked numerically and
  are correct column-major inverses.

---

# Settled against Three.js TSL

Where a question was genuinely open, TSL was read rather than reasoned about,
since it targets both backends from one graph and has had far more exposure.

- **Floored `mod`.** TSL polyfills `tsl_mod_float` as `x - y * floor(x / y)`,
  one variant per width, leaves GLSL its native `mod()`, and keeps `%` for
  integers. rmsl now matches on all four counts.
- **Widening narrow uniform array elements.** TSL stores an array of `f32` as
  `array<vec4<f32>, N>` and reads back the leading component, for the same
  stride rule. rmsl does the same.
- **No array of separate texture bindings.** TSL does not offer one either; it
  supports layered array textures — `texture_2d_array<f32>` and
  `sampler2DArray`, one binding with many layers — which both languages have.
  So refusing `uniformArray` of a texture follows TSL, and a layered array
  texture is the feature to add if one is wanted.
- **Component-wise comparison result types.** Already noted in the source as
  matching how TSL types them.

Not everything should follow TSL — it does not reduce common subexpressions
(#30), which is an opportunity rather than a precedent. The point is to know
which way one is going and why.

# Checked and found clean

- `.node()` removal — no surviving call sites anywhere in the repo.
- Scalar broadcast for vectors — `vec3.clamp(0,1)` emits `clamp(v, vec3(0.0), vec3(1.0))`.
- WGSL diagonal expansion for square matrices.
- Both WGSL inverse helper bodies, numerically.

# How to reproduce any of this

Drop a temporary test file in `src/` that calls the compiler and writes its
output to a file — vitest suppresses `console.log` — run
`npx vitest run src/<name>.test.ts`, then delete it. Line numbers are
deliberately omitted here because they rot; the symbols are greppable.
