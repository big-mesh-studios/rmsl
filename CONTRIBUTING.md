# Contributing to RMSL

RMSL builds a shader node graph in TypeScript and emits GLSL ES 3.00 and WGSL.
Almost everything lives in `src/rmsl.ts`.

Most mistakes here are silent: a wrong result type or a missing table entry
produces a shader that reads fine and that no driver accepts. The test setup
exists to catch those, so it is worth reading first.

## Setup

```bash
pnpm install
pnpm test:setup   # Chromium build used to compile GLSL, once
pnpm test
```

`pnpm type-check` and `pnpm build` do what they say.

## Tests

Three layers.

**Text.** `src/rmsl-usage.test.ts` asserts on the generated source. Cheap and
weak — `refract(I, N)` still contains `refract(`.

**Validity.** Automatic, and the important one. That file imports the compilers
under an alias:

```typescript
import {
  recordingGLSL as compileGLSL,
  recordingWGSL as compileWGSL,
} from "./testing/shader-validity";
```

The stand-ins compile every program to *both* backends and record it. An
`afterAll` hands the whole set to Chromium's WebGL2 compiler and to Dawn, and
fails the run on any rejection. A test asserting only on GLSL still has its WGSL
output checked by a real driver.

So: **write codegen tests in `src/rmsl-usage.test.ts`.** A new file importing
`compileGLSL` from `../rmsl` directly gets the text layer only, silently.

**Values.** `src/rmsl-eval.test.ts` runs the expression on real hardware and
reads the number back from both backends. Text and validity both pass for a
shader that computes the wrong thing — swapping `min` for `max` is invisible to
them. Operands arrive as function parameters, not literals, or constant folding
computes the answer before codegen runs.

Add a case here when operands could plausibly be emitted in the wrong order:
`min`/`max`, the edge order in `step(edge, x)`.

**Types.** `src/rmsl.test-d.ts` asserts what the signatures promise, with
`expectTypeOf`. Several defects here were a signature saying one thing while
the node was another — a comparison declared as returning a single boolean
while building a boolean vector — which no runtime assertion can see.

```bash
pnpm test:types    # just these
pnpm type-check    # catches them too, as ordinary type errors
```

Add a case whenever an operation's result type is not simply its receiver's:
anything that reduces, anything whose defining operand is not the first.

### Skipping the GPU layers

Both GPU-backed layers can be turned off, which is what makes a fast inner loop
and a workable mutation run:

| Variable | Turns off |
|---|---|
| `RMSL_SKIP_GPU` | both |
| `RMSL_SKIP_SHADER_VALIDATION` | validity only |
| `RMSL_SKIP_SHADER_EVALUATION` | evaluation only |

```bash
pnpm test:fast     # RMSL_SKIP_GPU=1, well under a second
pnpm test:mutate   # mutation testing with both layers off
```

Skipping is announced on stderr, because a run without these layers proves much
less than it appears to — the score from a mutation run measures the text
assertions alone. Run the full `pnpm test` before opening a pull request.

## Adding an operation

A method on `Node` — `.mix()`, `.length()`, `.lessThan()`.

1. **Declare it** on an operation interface (`ArithOps`, `FloatMathOps`,
   `ComparisonOps`, `VecCommonOps`, `MatOps`, …). Declare the *result* type:
   `length(): Node<"float">`. Do not declare one operation on two interfaces
   that both apply to the same type — the checker picks one arbitrarily, which
   is how `step` once shipped broken.
2. **Add to `NodeOps`** only if you introduced a new interface.
3. **Implement on `NodeImpl`**, routing through `op1` (unary), `op` (n-ary),
   `comp` (comparisons), or a direct `node({...})` when the result type is
   unrelated to the operands. Argument order follows the target languages, not
   the fluent chain — `step` puts the receiver last. `NodeImpl` does not
   `implements` the operation interfaces, so nothing warns you if you skip this.
4. **`REDUCING_OPS`** — if the result type differs from the operand type.
   Skipped: `vec3 x = length(v);`.
5. **`VALUE_OPERAND`** — if the operand defining the result type is not the
   first. Skipped: `vec3.step(0.5)` types itself from the edge.
6. **`UNIFORM_OPERAND_OPS`** — if every operand must share the defining
   operand's type, so a scalar gets broadcast. Skipped: GLSL usually still
   accepts it and WGSL does not. Leave out operations whose argument is
   genuinely scalar — `mix(a, b, t)`, `refract(i, n, eta)`.
7. **`tryFold`** — optional, for operations with an exact scalar JavaScript
   equivalent.
8. **GLSL case** in `compileGLSLNode`, via `unaryGLSL` / `binaryGLSL` /
   `ternaryGLSL` / `comparisonGLSL`. `binaryGLSL(node, ctx, "pow", true)`
   renders a call; without the flag, an infix operator.
9. **WGSL case** in `compileWGSLNode`, via `unaryWGSL` / `binaryWGSL` /
   `ternaryWGSL` / `shiftWGSL`. There is no `comparisonWGSL` — WGSL comparisons
   are plain operators.
10. **Check the backends agree** on name, argument order and semantics.
    `inversesqrt` is `inverseSqrt` in WGSL; GLSL needs `lessThan(a, b)` for
    vectors; WGSL shifts take `u32`; WGSL has no `inverse()` and pulls a helper
    from `WGSL_HELPERS`.
11. **Document** in `docs/api.md`.
12. **Test** in `src/rmsl-usage.test.ts`, driven from a `uniform(...)` — with
    literals the expression folds and codegen never runs.

Both switches throw on an unhandled node type, so steps 8 and 9 fail loudly —
provided a test exercises them.

## Adding a shader type

1. **`ShaderType` union.**
2. **`NodeOps` row** — `{}` if it carries no operations. The only step the type
   checker enforces, as
   `error TS2536: Type 'A' cannot be used to index type 'NodeOps'`. It will not
   name your type.
3. **`typeToGLSL`** — a missing row falls back to `float` rather than erroring.
4. **`typeToWGSL`** — falls back to `f32`.
5. **`TYPE_WIDTH`**, if the type is scalar- or vector-shaped. Drives comparison
   result width, scalar broadcasting and WGSL narrowing; a missing row means
   width 1, which quietly disables all three.
6. **Literal case** in both switches, if the type has literal values.
7. **Constructor**, if the type is constructible. The `bvec` types have none —
   they only arise as comparison results.
8. **`wrapValue` and the `…Like` aliases**, if raw JavaScript values should
   coerce to it. `wrapValue` reads arrays by length, so an array-shaped type
   collides with the existing 2/3/4/9/16 mapping.
9. **Swizzle map**, if the type is swizzleable. The runtime `swizzle()` derives
   its result from pattern length and from the source type's `ivec`/`uvec`
   prefix (a single component of an integer vector is that integer scalar), so
   the type-level map is what keeps a wrong swizzle unreachable.
10. **Backend special cases.** Grep for `_t ===`: `comparisonGLSL`, the GLSL
    `not` case, the WGSL `construct` narrowing, the WGSL uniform workaround that
    stores `bool` as `u32`, `isWgslTexture` / `isSamplerType` (integer textures
    take no WGSL sampler and fetch with `textureLoad`), `MATRIX_DIMENSIONS`,
    and the coercion ladders in the binary and ternary emitters.
11. **Document** in `docs/api.md` and the type tables in `docs/compilation.md`.
12. **Test** through the aliased compilers — the only automated check that steps
    3, 4, 5 and 10 were done.

## What catches what

| Mistake | Caught by |
|---|---|
| `ShaderType` with no `NodeOps` row | `tsc` |
| Missing `NodeImpl` method | test run — `TypeError` |
| Missing backend switch case | test run — throws, if exercised |
| Missing `REDUCING_OPS` / `VALUE_OPERAND` | real-compiler validation |
| Missing `UNIFORM_OPERAND_OPS` | Dawn only |
| Missing `typeToGLSL` / `typeToWGSL` / `TYPE_WIDTH` | real-compiler validation |
| Declared but never implemented | nothing |
| Wrong folding arithmetic, or swapped operands | evaluation layer |

The further down that table, the more your test needs to go through
`src/rmsl-usage.test.ts`.

## Pull requests

`pnpm type-check` and a full `pnpm test` with validation on. One logical change
per commit. New behaviour needs a test that would fail without it.
