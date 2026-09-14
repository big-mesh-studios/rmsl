# Non-square matrix multiplication — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow `mul` between any two matrices whose shapes meet (left columns === right rows), typed at the column/row product and supported by every backend (JS, WASM; GLSL/WGSL already support it natively).

**Architecture:** Core (`rmsl-core.ts`) gains the type-level product (`MatrixProduct`/`MatrixColumns`/`MatrixRows`/`MatName`) and a runtime branch in `NodeImpl.mul` that checks `left.columns === right.rows` and builds a `"mul"` node typed at the product shape (or throws with a message naming the mismatch). GLSL/WGSL need no backend change — they already emit `a * b` natively for any compatible shapes, so the node's own `_t` (now correct) is all they need. JS's `jsMatMul`/`jsCompileHelper` generalize from square-only `mat{n}mul` to `matmul{cL}x{rL}x{cR}x{rR}`. WASM's `emitMatMatMulStores` generalizes the same way, and the pre-existing non-square guard becomes a defense-in-depth internal-error check (core already rejects invalid shapes before any backend sees them). The shader-eval harness must widen from scalar-only round-tripping to arbitrary-size aggregate (vector/matrix) round-tripping so matrix-shaped recordings can be cross-checked JS vs GLSL vs WGSL vs WASM.

**Tech Stack:** TypeScript, GLSL ES 3.00, WGSL, hand-encoded WASM bytecode, Vitest (incl. `expectTypeOf` test-d), Prettier, pnpm. Spec: `docs/superpowers/specs/2026-09-14-non-square-matmul-design.md`.

**Verification commands used throughout:**

- Fast per-file test: `pnpm vitest run <file>`
- Type check: `pnpm type-check`
- Type-level tests: `pnpm test:types`
- Format check: `pnpm format:check`
- Full suite (CPU only): `pnpm test:fast`

---

## File structure

- `src/rmsl-core.ts` — type helpers, `MatOps`/`RectMatOps` interfaces, `NodeOps` registry, `NodeImpl.mul` runtime branch.
- `src/rmsl.test-d.ts` — type-level pins for the matrix product.
- `src/rmsl-usage.test.ts` — construction/throw/emitted-type pins.
- `src/backends/rmsl-js.ts` — `jsMatMul`, `jsCompileHelper`.
- `src/backends/rmsl-js.test.ts` — direct JS compile pins.
- `src/backends/rmsl-wasm.ts` — `emitMatMatMulStores`.
- `src/backends/rmsl-wasm.test.ts` — direct WASM value pins.
- `src/test-helpers/shader-eval.ts` — harness generalization (aggregate round-trip).
- `src/test-helpers/shader-eval.test.ts` — harness coverage.
- `src/rmsl-js.test.ts` (or equivalent cross-backend recording file) — non-square matmul recordings.
- `ROADMAP.md` — "what throws today" correction.

---

## Task 1: Core rule + typing of `mul`

**Files:** `src/rmsl-core.ts`, `src/rmsl.test-d.ts`, `src/rmsl-usage.test.ts`

- [ ] **Step 1: Write failing type-level pins**

In `src/rmsl.test-d.ts`, add:

```ts
it("types a matrix times a matrix by the column/row product", () => {
  expectTypeOf(uniform("mat2x3").mul(uniform("mat3x2"))).toEqualTypeOf<Node<"mat3x3">>();
  expectTypeOf(uniform("mat3x2").mul(uniform("mat2x3"))).toEqualTypeOf<Node<"mat2x2">>();
  expectTypeOf(uniform("mat2").mul(uniform("mat3x2"))).toEqualTypeOf<Node<"mat3x2">>();
  expectTypeOf(uniform("mat4").mul(uniform("mat2x4"))).toEqualTypeOf<Node<"mat2x4">>();
  expectTypeOf(uniform("mat2x4").mul(uniform("mat4x2"))).toEqualTypeOf<Node<"mat4x4">>();
  expectTypeOf(uniform("mat4").mul(uniform("mat4"))).toEqualTypeOf<Node<"mat4">>();
});

it("rejects a matrix product whose shapes do not meet", () => {
  expectTypeOf(uniform("mat2x3").mul(uniform("mat2x4"))).toEqualTypeOf<never>();
});
```

Run: `pnpm test:types` — expect a compile error (no generic `mul` overload exists yet for mismatched/rectangular pairs).

- [ ] **Step 2: Add type helpers in `rmsl-core.ts`**

Near the existing matrix type definitions, add:

```ts
export type MatrixType = "mat2" | "mat3" | "mat4" | "mat2x3" | "mat2x4" | "mat3x2" | "mat3x4" | "mat4x2" | "mat4x3";

type MatrixColumns<M extends MatrixType> = M extends `mat${infer C}x${string}`
  ? C extends `${infer N extends number}`
    ? N
    : never
  : M extends `mat${infer N extends number}`
    ? N
    : never;

type MatrixRows<M extends MatrixType> = M extends `mat${string}x${infer R}`
  ? R extends `${infer N extends number}`
    ? N
    : never
  : M extends `mat${infer N extends number}`
    ? N
    : never;

type MatName<C extends number, R extends number> = C extends R ? `mat${C}` : `mat${C}x${R}`;

export type MatrixProduct<A extends MatrixType, B extends MatrixType> =
  MatrixColumns<A> extends MatrixRows<B> ? MatName<MatrixColumns<B>, MatrixRows<A>> : never;
```

- [ ] **Step 3: Add the generic `mul` overload**

Add `Self extends MatrixType` as the first type parameter to `RectMatOps` (and thread it through wherever `RectMatOps` is instantiated in the `NodeOps` registry — update each entry to pass its own matrix name as `Self`). In both `MatOps` and `RectMatOps`, replace the square-only `mul(other: Node<Self>): Node<Self>` overload with:

```ts
mul<O extends MatrixType>(
  other: Node<O>,
): MatrixProduct<Self, O> extends infer P
  ? P extends ShaderType ? Node<P> : never
  : never;
```

Keep the existing `mul(other: Node<Vec>): Node<Vec>` (mat×vec) and `mul(other: Node<Shorter>): Node<Self>` (mat×scalar-ish) overloads unchanged, ordered after the matrix overload so TS tries the matrix candidate first.

Run `pnpm test:types` — Step 1's pins should now pass.

- [ ] **Step 4: Write failing runtime/usage pins**

In `src/rmsl-usage.test.ts`, add pins that: (a) construct `uniform("mat2x3").mul(uniform("mat3x2"))` and assert `._t === "mat3x3"`; (b) assert `uniform("mat2x3").mul(uniform("mat2x4"))` throws, with a message naming both types and the shape mismatch (columns vs rows); (c) compile a non-square product through GLSL and WGSL and assert the emitted source `toContain("mat3x3")`.

Run the test file — expect failure (today `NodeImpl.mul` for mat×mat falls through to `op("mul", this, other)`, which types/builds at the _left operand's_ type, not the product).

- [ ] **Step 5: Implement the runtime branch**

In `NodeImpl.mul` in `rmsl-core.ts`, after the existing mat×vec (`matVecMul`) branch and before the final generic `op("mul", ...)` fallback, add:

```ts
let otherType = other instanceof BaseNode ? other._t : undefined;
let shape = MATRIX_DIMENSIONS[this._t];
let otherShape = otherType !== undefined ? MATRIX_DIMENSIONS[otherType] : undefined;
if (shape !== undefined && otherShape !== undefined) {
  let [c1, r1] = shape;
  let [c2, r2] = otherShape;
  if (c1 !== r2) {
    throw new Error(
      `[RMSL] A ${this._t} cannot multiply a ${otherType}: the left has ${c1} ` +
        `column(s) but the right has ${r2} row(s), and a matrix product needs ` +
        `the left's columns to equal the right's rows.`,
    );
  }
  let resultType = c2 === r1 ? `mat${c2}` : `mat${c2}x${r1}`;
  return node({
    _t: resultType,
    type: "mul",
    params: [this as BaseNode<ShaderType>, wrapValue(other) as BaseNode<ShaderType>],
  });
}
```

(Adjust identifier names/import of `MATRIX_DIMENSIONS`/`node`/`wrapValue`/`BaseNode` to match what's already in scope in this function — follow the pattern used by the adjacent `matVecMul` branch.)

Run `src/rmsl-usage.test.ts` — Step 4's pins should pass. Run `pnpm test:types` and `pnpm test:fast` to confirm no regressions (square mat×mat, mat×vec, mat×scalar all unchanged).

- [ ] **Step 6: Commit**

---

## Task 2: JS backend — generalize matrix multiply

**Files:** `src/backends/rmsl-js.ts`, `src/backends/rmsl-js.test.ts`

- [ ] **Step 1: Write failing direct-compile pins**

In `rmsl-js.test.ts`, add a test that builds `uniform("mat2x3").mul(uniform("mat3x2"))`, compiles with `compileJSFn`, feeds concrete column-major arrays for both operands, and asserts the exact 9-element result array (hand-computed).

Run — expect failure: `jsMatMul` currently asserts `_t` cols === rows (square only) and/or the helper name collides.

- [ ] **Step 2: Generalize `jsMatMul`**

Replace the square-only helper-name construction with one keyed on all four dimensions:

```ts
export function jsMatMul(node: BaseNode<ShaderType>, ctx: CompileCtx): CompiledNode {
  let aType = node.params![0]!._t;
  let bType = node.params![1]!._t;
  let [cL, rL] = MATRIX_DIMENSIONS[aType];
  let [cR, rR] = MATRIX_DIMENSIONS[bType];
  let name = `matmul${cL}x${rL}x${cR}x${rR}`;
  jsRequireHelper(ctx, name);
  let a = jsCompileOperand(node.params![0]!, ctx);
  let b = jsCompileOperand(node.params![1]!, ctx);
  if (ctx.outTarget) {
    return {
      decls: [...a.decls, ...b.decls],
      body: [...a.body, ...b.body, `_${name}(${a.expr}, ${b.expr}, ${ctx.outTarget});`],
      expr: ctx.outTarget,
    };
  }
  return {
    decls: [...a.decls, ...b.decls],
    body: [...a.body, ...b.body],
    expr: `_${name}(${a.expr}, ${b.expr})`,
  };
}
```

- [ ] **Step 3: Generalize `jsCompileHelper`**

Split the combined `mat{n}x{n}(mul|T)` regex into a matmul regex and keep the transpose regex separate:

```ts
let mmMul = /^matmul(\d+)x(\d+)x(\d+)x(\d+)$/.exec(name);
if (mmMul) {
  let cL = Number(mmMul[1]);
  let rL = Number(mmMul[2]);
  let cR = Number(mmMul[3]);
  let rR = Number(mmMul[4]);
  let lines: string[] = [];
  for (let col = 0; col < cR; col++) {
    for (let row = 0; row < rL; row++) {
      let terms: string[] = [];
      for (let k = 0; k < cL; k++) terms.push(`a[${k * rL + row}] * b[${col * rR + k}]`);
      lines.push(`  out[${col * rL + row}] = ${terms.join(" + ")};`);
    }
  }
  return (
    `function _${name}(a, b, out) {\n` +
    `  out = out || new Array(${cR * rL});\n` +
    `  if (out === a) a = a.slice();\n` +
    `  if (out === b) b = b.slice();\n${lines.join("\n")}\n  return out;\n}`
  );
}

let mmT = /^mat(\d+)x(\d+)T$/.exec(name);
if (mmT) {
  let cols = Number(mmT[1]);
  let rows = Number(mmT[2]);
  let lines: string[] = [];
  for (let c = 0; c < cols; c++)
    for (let r = 0; r < rows; r++) lines.push(`  out[${r * cols + c}] = m[${c * rows + r}];`);
  return (
    `function _${name}(m, out) {\n` + `  out = out || new Array(${cols * rows});\n${lines.join("\n")}\n  return out;\n}`
  );
}
```

Remove the old combined `mat{n}x{n}(mul|T)` branch. Keep every other helper branch untouched.

Run `rmsl-js.test.ts` — Step 1's pin should pass, plus all existing square-matrix multiply/transpose tests should still pass unchanged (same generated code for the square case, since `cL===rL===cR===rR` collapses to the old formula).

- [ ] **Step 4: Commit**

---

## Task 3: Shader-eval harness — support aggregate (vector/matrix) round-trips

**Files:** `src/test-helpers/shader-eval.ts`, `src/test-helpers/shader-eval.test.ts`

- [ ] **Step 1: Write failing harness coverage tests**

Add tests in `shader-eval.test.ts` that round-trip a `vec3`-returning `Build` and a `mat3x3`-returning `Build` through `evaluateJS`, `evaluateGLSL`, `evaluateWGSL`, and `evaluateWASM`, asserting each returns the full element array (not just the first component). These can use existing supported ops (e.g. `a.add(b)` on vectors) since matrix product support in JS/WASM isn't required for this task.

Run — expect failure: today's harness only round-trips scalars (`Node<"float">`).

- [ ] **Step 2: Widen `Build` and add `componentCountOf`**

```ts
export type EvaluableRoot =
  | Node<"float">
  | Node<"vec2">
  | Node<"vec3">
  | Node<"vec4">
  | Node<"mat2">
  | Node<"mat2x3">
  | Node<"mat2x4">
  | Node<"mat3">
  | Node<"mat3x2">
  | Node<"mat3x4">
  | Node<"mat4">
  | Node<"mat4x2">
  | Node<"mat4x3">;

export type Build = (...args: Node<"float">[]) => EvaluableRoot;

function componentCountOf(t: ShaderType): number {
  let width = TYPE_WIDTH[t];
  if (width !== undefined) return width;
  let shape = MATRIX_DIMENSIONS[t];
  if (shape !== undefined) return shape[0] * shape[1];
  return 1;
}

function rootComponentCount(build: Build, argCount: number): number {
  let probes = Array.from({ length: argCount }, (_, i) => var_(`a${i}`, "float"));
  let root = build(...probes);
  return componentCountOf(root._t);
}
```

- [ ] **Step 3: Widen `evaluateJS`/`evaluateWASM` to return `number | number[]`**

Both backends already return arrays for aggregate roots (JS: `compileJSFn` result array; WASM: `readValueFromMemory` via `shaderResult.value`). Change each evaluator's return type to `number | number[]` and return the raw value (array or scalar) instead of unconditionally indexing `[0]`.

- [ ] **Step 4: Widen `evaluateGLSL`**

Keep the existing 1-component fast path (current behavior) when `rootComponentCount(...) === 1`. Otherwise: compute `n = rootComponentCount(build, args.length)`, `width = Math.ceil(n / 4)`; generate a fragment shader that assigns `TYPE r = rmsl_eval(...)` (`TYPE` from the fn's declared return type) then, per output pixel, selects among unrolled constant-index reads (`r[col].x`/`.y`/`.z`/`.w`, `col = floor(elementIndex / rows)`, component = `elementIndex % rows`) gated on `int(gl_FragCoord.y) == row` for `row` in `0..width`. Render into a `width × 1` `RGBA32F` framebuffer, `readPixels` into a `Float32Array(4 * width)`, return `Array.from(pixels.subarray(0, n))`.

- [ ] **Step 5: Add `runWGSLElements` and widen `evaluateWGSL`**

```ts
export async function runWGSLElements(code: string, floatCount: number): Promise<Float32Array> {
  // same device/pipeline setup as runWGSL, but the storage buffer and
  // readback buffer are sized 4 * floatCount bytes instead of 4 bytes.
  ...
}

export async function runWGSL(code: string): Promise<number> {
  return (await runWGSLElements(code, 1))[0]!;
}
```

Generate the WGSL entry point as `let r = rmsl_eval(...); result[i] = r[col][row];` for each element with a literal `i`/`col`/`row` (WGSL matrix indexing requires constant indices, which unrolling satisfies). For `evaluateWGSL`, compute `n` the same way as GLSL, call `runWGSLElements(code, n)`, return `Array.from(result.subarray(0, n))` (or the scalar path unchanged when `n === 1`).

- [ ] **Step 6: Run and verify**

Run `shader-eval.test.ts` — Step 1's pins pass. Run `pnpm test:fast` — existing scalar-only recordings still pass (scalar path unchanged).

- [ ] **Step 7: Commit**

---

## Task 4: Cross-backend recordings for non-square matmul (JS/GLSL/WGSL; WASM skipped)

**Files:** wherever existing matrix-multiply recordings live (likely `src/rmsl-js.test.ts` or a dedicated recordings file — locate via existing square-matmul recording tests) and `assertRecordedEvaluationsAgree`.

- [ ] **Step 1: Add an `evalMatrix` helper and non-square recordings**

Add a small helper mirroring the existing vector/scalar recording helpers, e.g.:

```ts
async function evalMatrix(build: Build, args: number[] = []) {
  return assertRecordedEvaluationsAgree(build, args);
}
```

Add recordings for representative non-square products: `mat2x3 * mat3x2`, `mat3x2 * mat2x3`, `mat2 * mat3x2`, `mat4 * mat2x4`, `mat2x4 * mat4x2` — same shapes pinned in Task 1's type-level test. Use concrete literal operands (via `uniform`/constant nodes) and hand-computed expected arrays, or rely on `assertRecordedEvaluationsAgree`'s cross-backend comparison (JS vs GLSL vs WGSL) without a separate hand-computed oracle if that's the existing pattern in this file.

- [ ] **Step 2: Run**

Run the recordings file. Expect: JS/GLSL/WGSL agree (Tasks 1–3 landed); WASM is still guarded (pre-Task-5) and throws `[RMSL] compileWasmFn: ...`, which `assertRecordedEvaluationsAgree`'s `isWasmUnsupported` detection already treats as a visible skip (matching the existing `vec`→`vec3` skip pattern) — confirm the skip count increases by the number of new recordings and no test fails.

- [ ] **Step 3: Commit**

---

## Task 5: WASM backend — generalize matrix multiply

**Files:** `src/backends/rmsl-wasm.ts`, `src/backends/rmsl-wasm.test.ts`

- [ ] **Step 1: Write failing direct-value pins**

In `rmsl-wasm.test.ts`, add a test building `uniform("mat2x3").mul(uniform("mat3x2"))`, compiling with `compileWasm`, feeding concrete uniforms, and asserting the exact 9-element `.value` array.

Run — expect failure (current guard throws for non-square).

- [ ] **Step 2: Generalize `emitMatMatMulStores`**

```ts
function emitMatMatMulStores(node: any, addr: number): number[] {
  let [a, b] = node.params;
  let aType = a._t as string;
  let bType = b._t as string;
  let [cL, rL] = MATRIX_DIMENSIONS[aType];
  let [cR, rR] = MATRIX_DIMENSIONS[bType];
  if (cL !== rR) {
    throw new Error(`[RMSL] compileWasmFn: internal error, mismatched matrix product ("${aType}" x "${bType}")`);
  }
  let out = [...materializeIfNeeded(a), ...materializeIfNeeded(b)];
  let aAddr = nodeAddress(a);
  let bAddr = nodeAddress(b);
  for (let col = 0; col < cR; col++) {
    for (let row = 0; row < rL; row++) {
      let terms: number[] = [];
      for (let k = 0; k < cL; k++) {
        let term = [
          ...loadComponent(aAddr, "float", (k * rL + row) * 8),
          ...loadComponent(bAddr, "float", (col * rR + k) * 8),
          WASM_OP.f64Mul,
        ];
        terms = k === 0 ? term : [...terms, ...term, WASM_OP.f64Add];
      }
      out.push(...storeComponent(addr, "float", (col * rL + row) * 8, terms));
    }
  }
  return out;
}
```

(Keep whatever the existing function's exact helper names/signatures are — `materializeIfNeeded`/`nodeAddress`/`loadComponent`/`storeComponent`/`WASM_OP` should already exist; adjust to match.) This keeps the mismatch check as an internal-error guard (unreachable via the public API since core now rejects invalid shapes before any node reaches this function), not a user-facing "unsupported" throw.

- [ ] **Step 3: Run**

Run `rmsl-wasm.test.ts` — Step 1's pin passes; existing square-matrix multiply pins still pass unchanged (same arithmetic when `cL===rL===cR===rR`).

Re-run Task 4's recordings file — the previously-skipped non-square WASM comparisons now execute and agree with JS/GLSL/WGSL instead of being skipped.

- [ ] **Step 4: Commit**

---

## Task 6: Documentation and full-suite gate

**Files:** `ROADMAP.md`

- [ ] **Step 1: Update "what throws today"**

Remove non-square matrix multiplication from the list of things that throw. If a Phase 7.5 (or similar) "known gaps" section lists it, remove the entry; leave other unrelated open items (e.g. bvec→numeric-vec constructor) untouched.

- [ ] **Step 2: Full verification**

Run, in order:

- `pnpm format` (or `pnpm format:check` then fix)
- `pnpm type-check`
- `pnpm test:types`
- `pnpm test:fast`

All must pass with zero regressions.

- [ ] **Step 3: Commit**
