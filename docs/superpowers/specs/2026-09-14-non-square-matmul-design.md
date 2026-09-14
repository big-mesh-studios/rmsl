# Non-square matrix multiplication across all backends — design

## Problem

`mat * mat` in RMSL only works for square same-type pairs (`mat4 * mat4`,
`mat2 * mat2`, ...). GLSL and WGSL already emit the product as a native `*`
and support any shape pair whose inner dimensions match; the JS and WASM
backends are the ones that lack the loop shapes:

- **JS** (`rmsl-compile-js.ts:671`): `jsMatMul` throws
  `[RMSL] The JS target does not yet support non-square matrix multiplication.`
  and its helper name / loop only encode one square shape.
- **WASM** (`rmsl-wasm.ts:2081`): `emitMatMatMulStores` throws
  `[RMSL] compileWasmFn: does not yet support non-square or mismatched-shape
  matrix multiplication` when `aType !== bType || cols !== rows`.

On top of the missing loops there is a typing gap: a `mat.mul(mat)` node is
built by `op("mul")`, which types the result as the *widest* operand and
which for two matrices means the left operand's type. That is harmless for
`mat4 * mat4` but wrong for a valid non-square product (`mat2x3 * mat3x2`
must produce `mat3x3`, not `mat2x3`) and silently wrong for a mismatched pair
(`mat2x3 * mat2x4` — GLSL/WGSL emit a shader that fails to compile, JS/WASM
throw or compute nonsense).

ROADMAP.md lists non-square matrix multiply in "What throws today"; a note in
the Phase 7.5 writeup claimed the remaining WASM evaluation skips were
non-square matrix-multiply cases — investigation showed that is wrong (all
three recorded skips are the bvec→numeric-vec constructor case), so this
feature is not measurable through the existing skip count and needs its own
tests.

## Matrix product rule

`A(C1×R1) × B(C2×R2)` is valid iff `C1 === R2`. The result is `matC2xR1`.
Column-major storage everywhere: component `(col, row)` of the result is

```
out[col*R1 + row] = Σ_k a[k*R1 + row] * b[col*R2 + k],   k ∈ [0, C1)
```

Both `matVecMul` (JS `mvm` helper, WASM `emitMatVecMulStores`) and the
existing square `jsMatMul`/`emitMatMatMulStores` already index with exactly
this convention; the generalisation is purely extracting both operands'
shapes and emitting counts `cR × rL`.

## Approach chosen

### (A) Construction-time typing and validation, one branch at the core

In `NodeImpl.mul()` (`rmsl-core.ts:743`), add a mat×mat branch before the
fall-through to `op("mul")`, mirroring the existing `matVecMul` branch:

- both operands are matrices (`MATRIX_DIMENSIONS` has keys for both `_t`s);
- compute `[cL, rL]` and `[cR, rR]` from the two types;
- if `cL !== rR`, throw the house-style untyped error, e.g.
  `[RMSL] A mat2x3 cannot multiply a mat2x4: the left has 2 columns but the
  right has 4 rows, and matrix multiplication needs the left's columns to
  equal the right's rows.`;
- otherwise return a `"mul"` node built directly with
  `_t = "mat" + cR + "x" + rL`, bypassing `op()`'s widest operand typing.

Consequences:

- GLSL/WGSL need no change — they keep emitting native `*`, and the 
  construction-time reject makes the invalid-combo emission path unreachable.
- The erroneous `op()` typing is fixed for every backend at once.
- The DSL's static types stay loose (`ArithOps.mul` returns `Node<A>`),
  consistent with the pre-existing `mat4 * vec3` loose typing — runtime
  `_t` is authoritative and is what all compilers read. No `.d.ts` changes.

Rejected alternatives:

- **B — keep backend-only throws.** Leaves GLSL/WGSL emitting shaders that
  fail to compile for mismatched pairs, and leaves each new backend to
  rediscover the rule. The core already owns operand typing (matVecMul,
  inverse, int/uint negativity), so a shape rule belongs there.
- **C — typed errors / error classes for the new throw.** Deliberately not
  chosen; see ROADMAP.md "Open questions" (typed errors recorded as a future
  idea). The new throw uses the plain `[RMSL]`-prefixed `Error` convention.

### (B) JS backend: generalise the matmul helper

`jsMatMul` (`rmsl-compile-js.ts:671`):

- read both dims `[cL, rL] = MATRIX_DIMENSIONS[aType]`,
  `[cR, rR] = MATRIX_DIMENSIONS[bType]`;
- helper name encoding both factor shapes, e.g. `matmul{cL}x{rL}x{cR}x{rR}`
  (the existing name `mat{c}x{r}mul` only encodes the square shape);
- `jsCompileHelper` (`rmsl-compile-js.ts:437`) gets the matching case and
  emits the general column-major loop from the product rule (`out[col*rL+row]
  = Σ_k a[k*rL+row] * b[col*rR+k]`);
- the non-square throw is removed; a mismatched inner-dims check stays as an
  internal-invariant throw (unreachable now that construction rejects it).

The `"mul"` dispatch (`rmsl-compile-js.ts:1116`) already routes mat×mat to
`jsMatMul`; mat×scalar stays elementwise via `jsVectorBinary`.

### (C) WASM backend: generalise `emitMatMatMulStores`

`emitMatMatMulStores` (`rmsl-wasm.ts:2081`):

- read both dims from `aType`/`bType`;
- loop `col ∈ [0, cR)`, `row ∈ [0, rL)`, `k ∈ [0, cL)` with
  `loadComponent(aAddr, "float", (k*rL + row) * 8)` and
  `loadComponent(bAddr, "float", (col*rR + k) * 8)`, storing at
  `(col*rL + row) * 8`;
- the square-only `if (aType !== bType || cols !== rows) throw` becomes a
  defense-in-depth inner-dims check (left `cL` vs right `rR`).

The `"mul"` dispatch (`rmsl-wasm.ts:1431`) already routes both-matrices to
this emitter.

## Harness extension: compare whole value arrays

The recorded-evaluation harness (`src/testing/shader-eval.ts`) currently
compares a single float per program across JS/WASM (exact f64) and
GLSL/WGSL (f32 tolerance). To satisfy "tests that compare results of these
non-square matrix operations" across all four backends, the recorded result
must be the full product matrix, not a scalar reduction.

Design:

- **Representation.** `RecordedEvaluation.js` becomes `number | number[]`
  (column-major for matrices — the natural flat order of every backend: JS
  arrays, WASM memory at `(col*rows+row)`, GLSL/WGSL column-major storage).
  `Build` broadens to a float/vec/mat-valued production; the harness derives
  the root type via the existing `build(var_(a0,"float"), ...)` ergonomic and
  its component count (`componentCountOf`) to size the readbacks.
- **JS (`evaluateJS`).** `compileJSFn` already returns a flat array for an
  aggregate root; stop truncating through `value[0]` and return the whole
  array.
- **WASM (`evaluateWASM`).** An aggregate root routes through `valueMemory`
  and `compileWasm`'s callable returns `{ value: number[] }` — today the
  harness would hand back that object (a latent path nothing exercised).
  Return `value.value`.
- **GLSL (`evaluateGLSL`).** For an aggregate root, emit the compiled fn
  plus a main that computes the root into a local and a constant-index
  component reader `el(r, i)` (unrolled `if`s with literal indices — GLSL ES
  3.00 requires constant-index-expression matrix indexing), selected by pixel:
  render to a `W × 1` RGBA32F texture, `W = ceil(N/4)`,
  `base = 4 * int(gl_FragCoord.x)`, write
  `vec4(el(r,base), el(r,base+1), el(r,base+2), el(r,base+3))` (padding reads
  return 0), `readPixels` and slice to `N`. For a scalar root the existing
  single-pixel path is unchanged.
- **WGSL (`evaluateWGSL`).** For an aggregate root, emit the compiled fn plus
  `var<storage, read_write> result: array<f32>` and a main that stores each
  component via unrolled constant-index writes per column (`result[i] =
  r[col].x`, … — storage arrays accept constant indices; matrices in WGSL do
  not support runtime column indexing). Scalar root keeps `result[0] = …`.
- **Comparison.** In `assertRecordedEvaluationsAgree`: element-wise — exact
  for WASM vs CPU, `floatTolerance(js[i])` per element for GLSL/WGSL; lengths
  must match; failure messages render the whole (short) arrays.

No call-site changes: existing recordings all return floats and keep the
scalar path byte-for-byte.

## Tests

TDD order, honouring "start with JS, then port to WASM":

1. **Core rule** (`rmsl.test.ts` / a matmul block): `mat2x3.mul(mat3x2)` has
   `_t === "mat3x3"`, mismatched pairs throw at construction, valid square
   behaviour (`mat4*mat4`) unchanged.
2. **JS backend** (`rmsl.test.ts` or `rmsl-js.test.ts`): `evaluateJS`-driven
   pins of the exact product matrix for each valid shape; construction
   rejection is already covered by (1).
3. **Harness + cross-backend recording**: recorded programs of the shape
   `(matCxR(a0..).mul(matRxD(b0..)))` built from float params, asserted in
   `evalScalar`-style files; GLSL/WGSL/JS compare full matrices through the
   new array path. WASM still counts these as visible "not supported yet"
   skips (the old matmul throw carries the `[RMSL] compileWasmFn` prefix),
   so the suite stays green mid-migration.
4. **WASM backend**: same exact-matrix pins via `evaluateWASM`; the recorded
   cases flip from skip to real comparison, dropping the `[shader-eval] WASM`
   skip count back to the genuine residual (bvec→numeric-vec constructor).

Representative shape set (dims 2/3/4), each with a mismatched rejection
alongside: `mat2x3×mat3x2 → mat3x3`, `mat3x2×mat2x3 → mat2x2`,
`mat2×mat3x2 → mat3x2`, `mat2x4×mat4x2 → mat4x4`, `mat4x2×mat2x4 → mat2x2`,
and `mat4×mat2x4 → mat2x4` (the "cut a mat4 to a mat2x4" idiom, GLSL's own
conversion-matrix case).

## Behavior edges (all deliberate)

- **Mismatched pair** → construction-time throw for every backend; no backend
  can be reached with a bad product anymore.
- **The `vec.mul(mat)` quirk** (vector on the left is typed as a matrix and
  compiled elementwise) is pre-existing and untouched; a vector never enters
  the new branch.
- **f32 backends** round each component; comparison tolerance is per-element
  `floatTolerance`, identical to the scalar path.
- **Array-vs-scalar result comparison** never mixes kinds: a build returning
  a scalar records a number; a build returning an aggregate records an array.

## Open verification points

Claims the design relies on that must be confirmed in code (all in the TDD
step named):

1. `compileJSFn` returns a flat column-major array for a matrix root
   (the JS-backend step makes a pinned product of a non-square pair the
   first test that forces it).
2. `compileWasm`'s aggregate-root callable returns `{ value: number[] }` for
   matrices specifically (the harness's `evaluateWASM` array path, step 3).
3. `compileGLSLFn`/`compileWGSLFn` return the product matrix type from `mul`
   — i.e. an aggregate root flows through their constructor/loop emission
   unchanged (steps 1-3 cover it via recordings).

## Files touched

- `src/rmsl-core.ts` — mat×mat branch in `NodeImpl.mul` (product type +
  construction-time rejection).
- `src/backends/rmsl-compile-js.ts` — `jsMatMul` (both dims, relaxed guard),
  helper-name encoding, `jsCompileHelper` matmul case.
- `src/backends/rmsl-wasm.ts` — `emitMatMatMulStores` generalisation.
- `src/testing/shader-eval.ts` — `number | number[]` results, `evaluateJS`/
  `evaluateWASM`/`evaluateGLSL`/`evaluateWGSL` aggregate readbacks, array
  comparison in `assertRecordedEvaluationsAgree`.
- Tests — core matmul block, JS pins, cross-backend recordings, WASM pins.
- `ROADMAP.md` — "What throws today": remove non-square matrix multiply;
  Phase 7.5 note gains the correct residual (bvec→numeric-vec constructor).