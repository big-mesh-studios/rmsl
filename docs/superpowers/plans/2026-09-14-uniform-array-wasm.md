# uniformArray for the WASM backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `uniformArray` for the WASM backend by reading array elements through dynamically-computed linear-memory addresses, with matching host marshalling and GPU-layout interop.

**Architecture:** Each uniform array gets a compile-time base address and per-element stride; an element read computes `base + trunc(index) * stride` at runtime and loads from it (`loadDynamic`). Aggregate-typed element reads materialize into a per-node scratch (existing `isScratchNode`/`scratchAddress` machinery), so `nodeAddress`'s `number` contract is untouched. A GPU-placed array reads f32 from a `wgslUniformLayout`-shaped region and promotes to f64, mirroring `emitGpuUniformPromote`.

**Tech Stack:** TypeScript, hand-encoded WASM bytecode (no wabt/binaryen), Vitest, Prettier, pnpm. Spec: `docs/superpowers/specs/2026-09-14-uniform-array-wasm-design.md`.

**Verification commands used throughout:**

- Fast per-file test: `pnpm vitest run src/backends/rmsl-wasm.test.ts`
- Type check: `pnpm type-check`
- Format check: `pnpm format:check`
- Full suite (CPU only): `pnpm test:fast`

---

## File structure

- `src/backends/rmsl-wasm.ts` — all compiler + host-marshalling changes (this file holds the whole backend; no split — follow existing pattern).
- `src/backends/rmsl-wasm.test.ts` — WASM-specific behavior tests.
- `src/rmsl-layout-interop.test.ts` — GPU-layout interop tests.
- `ROADMAP.md` — already updated (Open questions).

---

## Task 1: Scalar array element read, constant index (MVP)

**Files:**

- Modify: `src/backends/rmsl-wasm.ts` (add `WasmParam` variant, `uniformArrayInfo` map, `uniformArrayElementAddress` helper, `writeArrayToMemory`, `collect` `"uniformArray"` case, `walkExpr` `"uniformArrayElement"` case, `marshalInputs` case)
- Test: `src/backends/rmsl-wasm.test.ts:118-121`

- [ ] **Step 1: Replace the rejection test with a positive read**

Replace the test at `src/backends/rmsl-wasm.test.ts:118-121`:

```ts
it("reads a float uniform array element by a constant index", () => {
  let arr!: any;
  const build = () => {
    arr = uniformArray("float", 4);
    return arr.element(int(1));
  };
  const fn = compileWasm(build, { name: "main", params: [] });
  expect(fn({ uniforms: { [arr.name]: [10, 20, 30, 40] } })).toBe(20);
});
```

(`uniformArray` and `int` are already imported in this file.)

- [ ] **Step 2: Run to verify it fails**

Run: `pnpm vitest run src/backends/rmsl-wasm.test.ts`
Expected: FAIL — `[RMSL] compileWasmFn: unsupported node type in expression position: "uniformArrayElement"`.

- [ ] **Step 3: Add the `WasmParam` variant**

In the `WasmParam` union (`src/backends/rmsl-wasm.ts`, after the `"uniformMemory"` entry ~line 33), add:

```ts
  | { kind: "uniformArrayMemory"; slot: string; shaderType: ShaderType; length: number; address: number; elementStride: number; narrow?: boolean };
```

- [ ] **Step 4: Add the address helper**

After `storeDynamic` (~line 442), add:

```ts
/** Bytes computing `base + index * elementStride` — a uniform array element's first component address. */
function uniformArrayElementAddress(base: number, elementStride: number, indexBytes: number[]): number[] {
  return [...i32ConstBytes(base), ...indexBytes, ...i32ConstBytes(elementStride), WASM_OP.i32Mul, WASM_OP.i32Add];
}
```

- [ ] **Step 5: Add the host `writeArrayToMemory` helper**

After `writeAggregateToMemory` (~line 550), add:

```ts
/** Host-side: writes a uniform array (array of elements, or bare scalars for a scalar element type) into linear memory. */
function writeArrayToMemory(
  view: DataView,
  address: number,
  shaderType: ShaderType,
  length: number,
  value: any,
  elementStride: number,
  narrow?: boolean,
): void {
  const kind = elementKindOf(shaderType);
  const compSize = narrow && kind === "float" ? 4 : componentSizeOf(kind);
  const width = componentCountOf(shaderType);
  const arr = value as ArrayLike<any>;
  const n = Math.min(arr.length, length); // a shorter host array leaves the tail untouched
  for (let i = 0; i < n; i++) {
    const el = arr[i];
    const base = address + i * elementStride;
    if (width === 1) {
      const num = typeof el === "boolean" ? (el ? 1 : 0) : (el as number);
      if (kind === "float") {
        if (narrow) view.setFloat32(base, num, true);
        else view.setFloat64(base, num, true);
      } else {
        view.setInt32(base, num, true);
      }
    } else {
      for (let k = 0; k < width; k++) {
        const raw = el[k];
        const num = typeof raw === "boolean" ? (raw ? 1 : 0) : (raw as number);
        const at = base + k * compSize;
        if (kind === "float") {
          if (narrow) view.setFloat32(at, num, true);
          else view.setFloat64(at, num, true);
        } else {
          view.setInt32(at, num, true);
        }
      }
    }
  }
}
```

- [ ] **Step 6: Register the per-array layout info**

Next to `uniformAddress` (~line 667), add:

```ts
const uniformArrayInfo = new Map<string, { base: number; elementStride: number; narrow: boolean }>();
```

- [ ] **Step 7: Add the `collect` case (packed path)**

After the `"uniform"` case in `collect` (~line 1027), add:

```ts
case "uniformArray": {
  if (uniformArrayInfo.has(node.value.slot)) break;
  const shaderType = node.value.shaderType as string;
  const length = node.value.length as number;
  const elementSize = componentCountOf(shaderType) * componentSizeOf(elementKindOf(shaderType));
  if (options.gpuUniformLayout?.offsets[node.value.slot] !== undefined) {
    throw new Error("[RMSL] compileWasmFn: GPU-placed uniform arrays are not implemented yet");
  }
  const address = allocateBytes(elementSize * length);
  uniformArrayInfo.set(node.value.slot, { base: address, elementStride: elementSize, narrow: false });
  memoryParams.push({
    kind: "uniformArrayMemory",
    slot: node.value.slot,
    shaderType,
    length,
    address,
    elementStride: elementSize,
  });
  break;
}
```

- [ ] **Step 8: Add the `walkExpr` scalar case**

Before the `default` in `walkExpr` (~line 2458), add:

```ts
case "uniformArrayElement": {
  const info = uniformArrayInfo.get(node.params[0].value.slot);
  if (info === undefined) {
    throw new Error(`[RMSL] compileWasmFn: internal error, unaddressed uniform array "${node.params[0].value.slot}"`);
  }
  const kind = elementKindOf(node._t as string);
  const index = node.params[1];
  const indexBytes =
    scalarKindOf(index._t as string) === "float"
      ? [...walkExpr(index), WASM_OP.i32TruncF64S]
      : walkExpr(index);
  return loadDynamic(uniformArrayElementAddress(info.base, info.elementStride, indexBytes), kind);
}
```

- [ ] **Step 9: Add the `marshalInputs` case**

After the `"uniformMemory"` case in `marshalInputs` (~line 2762), add:

```ts
case "uniformArrayMemory":
  writeArrayToMemory(view, p.address, p.shaderType, p.length, (ctx.uniforms as any)?.[p.slot], p.elementStride, p.narrow);
  break;
```

- [ ] **Step 10: Run to verify it passes**

Run: `pnpm vitest run src/backends/rmsl-wasm.test.ts`
Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add src/backends/rmsl-wasm.ts src/backends/rmsl-wasm.test.ts
git commit -m "Read scalar uniform array elements over linear memory in the WASM backend"
```

---

## Task 2: Aggregate (vec4) element read via scratch

**Files:**

- Modify: `src/backends/rmsl-wasm.ts` (`isScratchNode`, `materializeIfNeeded`)
- Test: `src/backends/rmsl-wasm.test.ts`

- [ ] **Step 1: Write the failing test**

Add to `rmsl-wasm.test.ts` (imports `vec4` already present):

```ts
describe("WASM backend: uniform arrays", () => {
  it("reads a vec4 uniform array element by a constant index", () => {
    let arr!: any;
    const build = () => {
      arr = uniformArray("vec4", 4);
      return arr.element(int(2)).x;
    };
    const fn = compileWasm(build, { name: "main", params: [] });
    const values = [
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [7, 8, 9, 10],
      [0, 0, 0, 0],
    ];
    expect(fn({ uniforms: { [arr.name]: values } })).toBe(7);
  });

  it("lets a uniform array element back a toVar()", () => {
    let arr!: any;
    const build = () => {
      arr = uniformArray("vec3", 3);
      return arr.element(int(1)).toVar().x;
    };
    const fn = compileWasm(build, { name: "main", params: [] });
    expect(
      fn({
        uniforms: {
          [arr.name]: [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ],
        },
      }),
    ).toBe(4);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `pnpm vitest run src/backends/rmsl-wasm.test.ts`
Expected: FAIL — the vec4 read hits `"unsupported node type in vector position: uniformArrayElement"` (from `materializeIfNeeded`'s default).

- [ ] **Step 3: Make `isScratchNode` claim aggregate element nodes**

In `isScratchNode` (`src/backends/rmsl-wasm.ts`, ~line 475), add after the `construct` line:

```ts
if (node.type === "uniformArrayElement") return true;
```

(`isScratchNode` already returns `false` for non-aggregate `_t` at its top, so scalar elements never get a scratch.)

- [ ] **Step 4: Add the `materializeIfNeeded` case**

After the `"uniform"` case in `materializeIfNeeded` (~line 1276), add:

```ts
case "uniformArrayElement": {
  const info = uniformArrayInfo.get(node.params[0].value.slot);
  if (info === undefined) {
    throw new Error(`[RMSL] compileWasmFn: internal error, unaddressed uniform array "${node.params[0].value.slot}"`);
  }
  const kind = elementKindOf(node._t as string);
  const compSize = componentSizeOf(kind);
  const width = componentCountOf(node._t as string);
  const index = node.params[1];
  const indexBytes =
    scalarKindOf(index._t as string) === "float"
      ? [...walkExpr(index), WASM_OP.i32TruncF64S]
      : walkExpr(index);
  const baseAddr = nodeAddress(node);
  const out: number[] = [];
  for (let k = 0; k < width; k++) {
    const addrBytes = [
      ...uniformArrayElementAddress(info.base, info.elementStride, indexBytes),
      ...i32ConstBytes(k * compSize),
      WASM_OP.i32Add,
    ];
    out.push(...storeComponent(baseAddr, kind, k * compSize, loadDynamic(addrBytes, kind)));
  }
  return out;
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `pnpm vitest run src/backends/rmsl-wasm.test.ts`
Expected: PASS for both new tests and everything before.

- [ ] **Step 6: Commit**

```bash
git add src/backends/rmsl-wasm.ts src/backends/rmsl-wasm.test.ts
git commit -m "Materialize aggregate uniform array elements into a scratch address"
```

---

## Task 3: Float index conversion and bool arrays

**Files:**

- Test: `src/backends/rmsl-wasm.test.ts`

The conversion itself already landed in Task 1/2 (`i32.trunc_f64_s`); this task pins it and the bool element kind.

- [ ] **Step 1: Write the failing tests**

Add to the `"WASM backend: uniform arrays"` describe:

```ts
it("converts a plain-number (float) index", () => {
  let arr!: any;
  const build = () => {
    arr = uniformArray("float", 4);
    return arr.element(2.0);
  };
  const fn = compileWasm(build, { name: "main", params: [] });
  expect(fn({ uniforms: { [arr.name]: [10, 20, 30, 40] } })).toBe(30);
});

it("reads a bool uniform array element", () => {
  let arr!: any;
  const build = () => {
    arr = uniformArray("bool", 4);
    return arr.element(int(1));
  };
  const fn = compileWasm(build, { name: "main", params: [] });
  expect(fn({ uniforms: { [arr.name]: [true, false, true, false] } })).toBe(false);
});
```

- [ ] **Step 2: Run to verify it passes**

Run: `pnpm vitest run src/backends/rmsl-wasm.test.ts`
Expected: PASS — `2.0` truncates to `2` (reads `30`); bool is stored/read as i32 0/1 and the `"bool"` result comes back via `compileWasm`'s `result !== 0`.

- [ ] **Step 3: Commit**

```bash
git add src/backends/rmsl-wasm.test.ts
git commit -m "Pin float-index truncation and bool elements for uniform arrays"
```

---

## Task 4: Runtime index in a `For` loop

**Files:**

- Test: `src/backends/rmsl-wasm.test.ts`

- [ ] **Step 1: Write the failing test**

Add to the same describe:

```ts
it("indexes a vec4 uniform array with a runtime index inside a loop", () => {
  let arr!: any;
  const build = () => {
    arr = uniformArray("vec4", 24);
    let total = vec4(0, 0, 0, 0).toVar();
    For(
      () => float(0).toVar(),
      (i) => i.lessThan(24),
      (i) => i.assign(i.add(1)),
      (i) => {
        total.assign(total.add(arr.element(i)));
      },
    );
    return total.x;
  };
  const fn = compileWasm(build, { name: "main", params: [] });
  const values = Array.from({ length: 24 }, (_, i) => [i, 0, 0, 0]);
  expect(fn({ uniforms: { [arr.name]: values } })).toBe(276); // sum of 0..23
});
```

This is the pattern from `rmsl-usage.test.ts:1846`: a **float** loop counter becomes the index (so `i32.trunc_f64_s` fires every iteration), and the element read is a dynamic address. (`For`/`float` are already imported in this file.)

- [ ] **Step 2: Run to verify it passes**

Run: `pnpm vitest run src/backends/rmsl-wasm.test.ts`
Expected: PASS — `0+1+…+23 = 276`.

- [ ] **Step 3: Commit**

```bash
git add src/backends/rmsl-wasm.test.ts
git commit -m "Test a runtime uniform array index inside a WASM loop"
```

---

## Task 5: Uniform array element as function root

**Files:**

- Test: `src/backends/rmsl-wasm.test.ts`

This likely passes already (aggregate root flows through `finalValueBytes` → `materializeIfNeeded`, both covered since Task 2); it pins the behavior.

- [ ] **Step 1: Write the test**

```ts
it("supports a uniform array element as the function root", () => {
  let arr!: any;
  const build = () => {
    arr = uniformArray("vec3", 3);
    return arr.element(int(1));
  };
  const fn = compileWasm(build, { name: "main", params: [] });
  const result = fn({
    uniforms: {
      [arr.name]: [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ],
    },
  }) as any;
  expect(result.value).toEqual([4, 5, 6]);
});
```

- [ ] **Step 2: Run**

Run: `pnpm vitest run src/backends/rmsl-wasm.test.ts`
Expected: PASS. If it does not, the gap is the aggregate root path at `finalValueBytes` — investigate there before fixing.

- [ ] **Step 3: Commit**

```bash
git add src/backends/rmsl-wasm.test.ts
git commit -m "Pin a uniform array element as the WASM function root"
```

---

## Task 6: GPU-layout path for arrays

**Files:**

- Modify: `src/backends/rmsl-wasm.ts` (`GpuUniformLayout.strides`, `collect` GPU branch, narrow promote in `materializeIfNeeded`/`walkExpr`)
- Test: `src/rmsl-layout-interop.test.ts`

- [ ] **Step 1: Write the failing interop test**

Add `uniformArray` to the `./rmsl` import in `src/rmsl-layout-interop.test.ts` (currently `Fn, uniform, wgslUniformLayout`), then append to the describe:

```ts
it("places a uniform array at wgslUniformLayout's offset and stride, f32-accurate", () => {
  const arr = uniformArray("vec4", 2);
  const layout = wgslUniformLayout([{ slot: arr.name, type: wgslType("vec4"), length: 2 }]);
  const member = layout.members.find((m) => m.name === arr.name)!;
  expect(member.stride).toBe(16);

  const options: CompileWasmFnOptions = {
    name: "main",
    params: [],
    gpuUniformLayout: {
      offsets: { [arr.name]: member.offset },
      strides: { [arr.name]: member.stride! },
      totalSize: layout.size,
    },
  };
  const fn = compileWasm(() => Fn(() => arr.element(int(1)).x)() as any, options);
  // 0.1 is not exact in f32; the GPU path stores f32, so it reads back fround(0.1).
  expect(
    fn({
      uniforms: {
        [arr.name]: [
          [0, 0, 0, 0],
          [0.1, 0, 0, 0],
        ],
      },
    }),
  ).toBe(Math.fround(0.1));
  expect(Math.fround(0.1)).not.toBe(0.1);
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `pnpm vitest run src/rmsl-layout-interop.test.ts`
Expected: FAIL — `"GPU-placed uniform arrays are not implemented yet"` from the Task 1 `collect` guard.

- [ ] **Step 3: Extend `GpuUniformLayout`**

In the `GpuUniformLayout` type (`src/backends/rmsl-wasm.ts`, ~line 66), add:

```ts
  /** Per-uniform-array element stride in the host buffer (its wgslUniformLayout stride). */
  strides?: Record<string, number>;
```

- [ ] **Step 4: Replace the `collect` GPU guard with the real path**

Replace the acts in the `"uniformArray"` collect case added in Task 1:

```ts
if (options.gpuUniformLayout?.offsets[node.value.slot] !== undefined) {
  throw new Error("[RMSL] compileWasmFn: GPU-placed uniform arrays are not implemented yet");
}
const address = allocateBytes(elementSize * length);
uniformArrayInfo.set(node.value.slot, { base: address, elementStride: elementSize, narrow: false });
memoryParams.push({
  kind: "uniformArrayMemory",
  slot: node.value.slot,
  shaderType,
  length,
  address,
  elementStride: elementSize,
});
```

with:

```ts
const gpuOffset = options.gpuUniformLayout?.offsets[node.value.slot];
if (gpuOffset !== undefined) {
  const gpuStride = options.gpuUniformLayout?.strides?.[node.value.slot];
  if (gpuStride === undefined) {
    throw new Error(
      `[RMSL] compileWasmFn: gpuUniformLayout for uniform array "${node.value.slot}" needs a matching strides value`,
    );
  }
  uniformArrayInfo.set(node.value.slot, { base: gpuOffset, elementStride: gpuStride, narrow: true });
  memoryParams.push({
    kind: "uniformArrayMemory",
    slot: node.value.slot,
    shaderType,
    length,
    address: gpuOffset,
    elementStride: gpuStride,
    narrow: true,
  });
  break;
}
const address = allocateBytes(elementSize * length);
uniformArrayInfo.set(node.value.slot, { base: address, elementStride: elementSize, narrow: false });
memoryParams.push({
  kind: "uniformArrayMemory",
  slot: node.value.slot,
  shaderType,
  length,
  address,
  elementStride: elementSize,
});
```

- [ ] **Step 5: Add the f32 promote to both read paths**

In `materializeIfNeeded`'s `"uniformArrayElement"` case, change the loop body so the per-component raw size and load vary with `narrow`:

```ts
  const rawCompSize = info.narrow && kind === "float" ? 4 : compSize;
  ...
  for (let k = 0; k < width; k++) {
    const addrBytes = [
      ...uniformArrayElementAddress(info.base, info.elementStride, indexBytes),
      ...i32ConstBytes(k * rawCompSize),
      WASM_OP.i32Add,
    ];
    const loaded =
      info.narrow && kind === "float"
        ? [...addrBytes, WASM_OP.f32Load, 0x00, 0x00, WASM_OP.f64PromoteF32]
        : loadDynamic(addrBytes, kind);
    out.push(...storeComponent(baseAddr, kind, k * compSize, loaded));
  }
```

In `walkExpr`'s `"uniformArrayElement"` case, change the return so a narrow float promotes:

```ts
const addrBytes = uniformArrayElementAddress(info.base, info.elementStride, indexBytes);
if (info.narrow && kind === "float") {
  return [...addrBytes, WASM_OP.f32Load, 0x00, 0x00, WASM_OP.f64PromoteF32];
}
return loadDynamic(addrBytes, kind);
```

- [ ] **Step 6: Run to verify it passes**

Run: `pnpm vitest run src/rmsl-layout-interop.test.ts`
Expected: PASS (element `1` read from `offset + 1×16`, f32 → `Math.fround(0.1)`). Then re-run `pnpm vitest run src/backends/rmsl-wasm.test.ts` to confirm the packed path is unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/backends/rmsl-wasm.ts src/rmsl-layout-interop.test.ts
git commit -m "Place WASM uniform arrays in a WGSL-shaped GPU buffer at f32 precision"
```

---

## Task 7: Cross-backend recording and full-suite gate

**Files:** none (verification only)

- [ ] **Step 1: Run the Phase 7 recording**

Run: `pnpm vitest run src/rmsl-js.test.ts src/rmsl-eval.test.ts`
Expected: PASS, and the `[shader-eval] WASM: N of 79 … not yet supported` line shows the count dropping with `uniformArray` gone (only non-square matrix-multiply cases remain). If a numeric mismatch appears, fix the divergence in `rmsl-wasm.ts` before proceeding.

- [ ] **Step 2: Type check**

Run: `pnpm type-check`
Expected: no errors.

- [ ] **Step 3: Format check**

Run: `pnpm format:check`
Expected: no files need formatting.

- [ ] **Step 4: Full CPU suite**

Run: `pnpm test:fast`
Expected: entire suite passes.

- [ ] **Step 5: Commit any stragglers**

```bash
git status
```

If anything is uncommitted, commit it with a message describing the change.

---

## Self-review notes

- **Spec coverage:** WasmParam variant (T1), `GpuUniformLayout.strides` (T6), `uniformArrayInfo`/collect (T1/T6), scratch + materialize for aggregate elements (T2), scalar walkExpr + index trunc (T1/T3), loop/dynamic index (T4), aggregate root (T5), marshalling shape + `writeArrayToMemory` (T1), GPU promote path (T6), missing-stride throw (T6). All spec sections map to a task.
- **Type consistency:** `uniformArrayElementAddress(base, elementStride, indexBytes)` signature is identical across T1, T2, T6. `uniformArrayInfo` shape `{ base, elementStride, narrow }` consistent everywhere. `writeArrayToMemory`'s parameter order matches the `marshalInputs` call.
- **Deliberate delegate:** Task 5 may pass without new code (covered by Task 2 machinery) — intentionally framed as a pin, not a red test.
