import { Node, ShaderType, TYPE_WIDTH, MATRIX_DIMENSIONS, var_ } from "./rmsl-core";
import { CompileFnOptions, COMPONENT_INDEX, resolveSwizzleTarget, assertStageResult } from "./rmsl-compiler-shared";
import { JsShaderContext, JsShaderResult, JsTextureData, JsTextureWrap } from "./rmsl-compile-js";
import { AllocRules, planLayout } from "./rmsl-layout";
// === WASM backend (see ROADMAP.md for what this does and doesn't cover yet) ===
//
// Compiles a plain, non-stage Fn straight to a WASM binary module instead of
// JS source, for the same CPU-eval niche `compileJS` serves (screen picking,
// ray-march hit tests) where per-call overhead matters. Phase 3 added a real
// WASM linear memory: every vector/matrix value lives at a compile-time-fixed
// byte address, and vector/matrix ops compile to loads/stores at that address
// rather than juggling N separate scalar values. ROADMAP.md maps what's next.
//
// Every opcode below was checked empirically (a minimal module built and run
// against a known answer) before use, not taken from memory — a
// wrong-but-still-valid opcode produces a module that *runs* and gives the
// wrong number, which is exactly the silent-miscompile failure mode
// CONTRIBUTING.md's testing section is about.

/**
 * One entry per parameter/uniform the compiled module needs data for.
 * `"param"`/`"uniform"` are scalar and occupy an actual WASM function
 * argument, in call order among themselves. `"paramMemory"`/`"uniformMemory"`
 * are aggregate (vector/matrix) values that never touch the WASM argument
 * list at all — they live at a fixed address in the module's exported
 * linear memory, and `compileWasm` writes their components there directly
 * via a `DataView` before every call.
 */
export type WasmParam =
  | { kind: "param"; name: string; shaderType: ShaderType }
  | { kind: "uniform"; slot: string; shaderType: ShaderType }
  | { kind: "paramMemory"; name: string; shaderType: ShaderType; address: number }
  | {
    kind: "uniformMemory"; slot: string; shaderType: ShaderType; address: number;
    /** Set only for a `gpuUniformLayout`-placed uniform: `address` is a raw
     * GPU-shaped slot (e.g. `f32` per float component, half this backend's
     * usual `f64`) rather than this backend's own packed representation —
     * `compileWasm` writes the narrower width there instead of its usual
     * one. See `GpuUniformLayout`. */
    narrow?: boolean;
  }
  // Phase 5 input direction — `compileWasm` writes these before the call,
  // exactly like a "param"/"uniform" pair, just sourced from
  // `ctx.attributes`/`ctx.varyings`/`ctx.fragCoord` instead of
  // `ctx.params`/`ctx.uniforms`. `varying` here is only ever the
  // fragment-stage (read) direction — a vertex stage's `varying()` is
  // output-direction, a different `WasmParam` kind entirely.
  | { kind: "attribute"; slot: string; shaderType: ShaderType }
  | { kind: "attributeMemory"; slot: string; shaderType: ShaderType; address: number }
  | { kind: "varying"; slot: string; shaderType: ShaderType }
  | { kind: "varyingMemory"; slot: string; shaderType: ShaderType; address: number }
  | { kind: "fragCoordMemory"; address: number }
  // Phase 5 output direction — the mirror image: `compileWasm` reads these
  // *after* the call, assembling a `JsShaderResult`-shaped object instead
  // of returning a bare value. Only present when the compiled function
  // actually used one of `output()`/a vertex `varying()`/`builtinPosition()`/
  // `builtinFragDepth()`, or has a value-producing result alongside them —
  // see `needsResult` in `compileWasmFn`.
  | { kind: "outputMemory"; slot: string; shaderType: ShaderType; address: number }
  | { kind: "varyingOutputMemory"; slot: string; shaderType: ShaderType; address: number }
  | { kind: "positionMemory"; address: number }
  | { kind: "fragDepthMemory"; address: number }
  | { kind: "valueMemory"; shaderType: ShaderType; address: number }
  // Phase 6: a texture uniform. Unlike every other uniform kind, its data
  // (dimensions, pixels) isn't fixed-size at compile time, so it gets no
  // ordinary scratch address for its *value* — only a small, fixed-size
  // metadata block (see `TEXTURE_META_*` byte offsets) recording where its
  // pixel data currently lives in the growable texture heap
  // (`compileWasm`'s wrapper places it fresh before every call) plus its
  // shape (width/height/depth/channels/filter/wrap). `compileWasm` writes
  // this block *and* the pixel data every call, exactly like a `uniform`
  // value, just shaped differently.
  | { kind: "textureMemory"; slot: string; samplerType: ShaderType; metadataAddress: number };

export type CompiledWasm = {
  /** The raw WASM binary module, exporting `options.name` and `"memory"`. */
  bytes: Uint8Array;
  params: WasmParam[];
  /** The Fn's declared return type — `compileWasm` reads this to convert a
   * `"bool"` result's 0/1 back to a real boolean, matching `compileJS`. */
  resultType: ShaderType;
  /** Phase 6: the byte address right after every compile-time-fixed
   * allocation (scratch, uniforms, texture metadata, ...) — where
   * `compileWasm`'s wrapper starts packing texture pixel data fresh before
   * every call, growing the module's memory first if needed. Meaningless
   * when `params` has no `"textureMemory"` entry. */
  textureHeapBase: number;
  /** Present whenever the Fn produces a non-`"void"` value (regardless of
   * whether `compileWasm`'s `.draw()` ever actually gets called) — the
   * exported `"draw"` function shares `main`'s own compiled body via a
   * real WASM `call`, rendering a `width x height` grid (both runtime
   * arguments, decided per call) into a buffer instead of one call per
   * pixel. That buffer's base address is *also* a runtime argument to
   * `"draw"` (not included here) — `compileWasm` computes it fresh every
   * call as `textureHeapBase` plus however many bytes that call's own
   * textures occupy, so a texture-using `.draw()` call places its output
   * right after wherever that call's texture heap actually ends. */
  draw?: { componentCount: number; kind: "float" | "int" | "uint" | "bool" };
};

export const WASM_OP = {
  end: 0x0b,
  block: 0x02,
  loop: 0x03,
  br: 0x0c,
  brIf: 0x0d,
  localGet: 0x20,
  localSet: 0x21,
  call: 0x10,
  select: 0x1b,

  i32Const: 0x41,
  f64Const: 0x44,

  i32Load: 0x28,
  f32Load: 0x2a,
  f64Load: 0x2b,
  i32Store: 0x36,
  f64Store: 0x39,
  f64PromoteF32: 0xbb,

  i32Eqz: 0x45,
  i32Eq: 0x46,
  i32Ne: 0x47,
  i32LtS: 0x48,
  i32LtU: 0x49,
  i32GtS: 0x4a,
  i32GtU: 0x4b,
  i32LeS: 0x4c,
  i32LeU: 0x4d,
  i32GeS: 0x4e,
  i32GeU: 0x4f,

  f64Eq: 0x61,
  f64Ne: 0x62,
  f64Lt: 0x63,
  f64Gt: 0x64,
  f64Le: 0x65,
  f64Ge: 0x66,

  i32Add: 0x6a,
  i32Sub: 0x6b,
  i32Mul: 0x6c,
  i32DivS: 0x6d,
  i32DivU: 0x6e,
  i32RemS: 0x6f,
  i32RemU: 0x70,
  i32And: 0x71,
  i32Or: 0x72,
  i32Xor: 0x73,
  i32Shl: 0x74,
  i32ShrS: 0x75,
  i32ShrU: 0x76,

  f64Abs: 0x99,
  f64Neg: 0x9a,
  f64Ceil: 0x9b,
  f64Floor: 0x9c,
  f64Trunc: 0x9d,
  f64Sqrt: 0x9f,
  f64Add: 0xa0,
  f64Sub: 0xa1,
  f64Mul: 0xa2,
  f64Div: 0xa3,
  f64Min: 0xa4,
  f64Max: 0xa5,

  i32TruncF64S: 0xaa,
  i32TruncF64U: 0xab,
  f64ConvertI32S: 0xb7,
  f64ConvertI32U: 0xb8,

  if_: 0x04,
  else_: 0x05,
} as const;

export const WASM_F64 = 0x7c;
export const WASM_I32 = 0x7f;
export const WASM_FUNC = 0x60;
export const WASM_BLOCKTYPE_VOID = 0x40;

/** A node's scalar value kind — everything at this phase is one of these
 * four, stored as either f64 (`float`) or i32 (the other three). */
type ScalarKind = "float" | "int" | "uint" | "bool";

function scalarKindOf(t: string | undefined): ScalarKind {
  return t === "int" || t === "uint" || t === "bool" ? t : "float";
}

function wasmTypeOf(kind: ScalarKind): number {
  return kind === "float" ? WASM_F64 : WASM_I32;
}

/** The scalar kind an aggregate type's components are stored as — the
 * `vec/mat` prefix says nothing else about layout, only this. */
function elementKindOf(t: string): ScalarKind {
  if (t.startsWith("ivec")) return "int";
  if (t.startsWith("uvec")) return "uint";
  if (t.startsWith("bvec")) return "bool";
  return "float"; // vecN and every matM are float-component.
}

/** Bytes one component of `t` occupies in linear memory: 8 for a float
 * component (f64), 4 for int/uint/bool (i32). */
function componentSizeOf(kind: ScalarKind): number {
  return kind === "float" ? 8 : 4;
}

/** Total scalar component count of `t` — 1 for a plain scalar, vector width
 * for a vecN, `cols*rows` for a matrix. */
function componentCountOf(t: string): number {
  const width = TYPE_WIDTH[t];
  if (width !== undefined) return width;
  const shape = MATRIX_DIMENSIONS[t];
  if (shape !== undefined) return shape[0] * shape[1];
  return 1;
}

function isAggregate(t: string): boolean {
  return componentCountOf(t) > 1;
}

/** A texture uniform's `shaderType` is never an aggregate `componentCountOf`
 * knows how to size (samplers carry no component count at all) — this is
 * the separate check `collect()` runs first to route it to texture-metadata
 * handling instead of the ordinary aggregate/scalar uniform paths. */
function isSamplerType(t: string): boolean {
  return t.startsWith("sampler") || t.startsWith("isampler") || t.startsWith("usampler");
}

/** An integer texture (`isampler*`/`usampler*`) is never filterable in
 * either language — `texture()`/`textureLod()` on one takes the same
 * unfiltered-fetch path `textureLoad()` always takes, matching `compileJS`
 * exactly (see the doc comment on `ISampler2DOps`, `rmsl-core.ts`). */
function isIntegerSamplerType(t: string): boolean {
  return t.startsWith("isampler") || t.startsWith("usampler");
}

/** `texture()`/`textureLoad()` support every 2D/3D sampler kind — float,
 * signed, and unsigned — exactly like `compileJS` does (an integer sampler
 * just takes the unfiltered-fetch path instead of the wrap/filter one,
 * since integer textures aren't filterable in either language). Only cube
 * samplers are unsupported, matching `compileJS`'s own restriction. */
function assertSampled2Dor3D(t: string): void {
  if (!t.endsWith("2D") && !t.endsWith("3D")) {
    throw new Error("[RMSL] compileWasmFn: texture uniforms support sampler2D/sampler3D (and their integer variants) only.");
  }
}

/** Fixed byte layout of a texture uniform's metadata block (44 bytes,
 * allocated once per slot via `allocateBytes`). `dataAddr` and every other
 * field are rewritten by `compileWasm`'s wrapper before every call — none
 * of it is known at compile time beyond "this slot exists". */
const TEX_META_UNORM_DIVISOR = 0; // f64: 255 for Uint8Array/Uint8ClampedArray source data, 1 otherwise
const TEX_META_DATA_ADDR = 8; // i32: this call's heap offset for the pixel data
const TEX_META_WIDTH = 12; // i32
const TEX_META_HEIGHT = 16; // i32
const TEX_META_DEPTH = 20; // i32: 0 for a 2D texture
const TEX_META_CHANNELS = 24; // i32: 1-4
const TEX_META_FILTER = 28; // i32: 0 = nearest, 1 = linear (magFilter only — minFilter is unused, matching compileJS)
const TEX_META_WRAP_S = 32; // i32: 0 = clamp, 1 = repeat, 2 = mirror
const TEX_META_WRAP_T = 36; // i32
const TEX_META_WRAP_R = 40; // i32: 3D only
const TEXTURE_META_STRIDE = 44;

/**
 * This backend's placement rules for `planLayout` (src/rmsl-layout.ts):
 * declaration order (never reordered — nothing here shares a struct with a
 * GPU buffer, so there's no padding to minimize), no array-element widening,
 * no stride rounding, and no whole-allocation alignment requirement — matching
 * the file's existing "no padding, byte-packed" design (see
 * `ROADMAP.md`, "Vectors and matrices live in linear memory now"). `type`
 * here is always an RMSL `ShaderType`.
 */
const PACKED_RULES: AllocRules = {
  sizeAndAlignOf(type) {
    return { size: componentCountOf(type) * componentSizeOf(elementKindOf(type)), align: 1 };
  },
  reorderByAlignment: false,
  structAlignMinimum: 1,
};

/**
 * Math functions with no WASM opcode, called through an import named
 * `"math"` — the same names as `Math`'s own, so `compileWasm` can hand the
 * real `Math` object as the import's namespace with no translation.
 */
const MATH_UNARY_IMPORTS = new Set([
  "sin", "cos", "tan", "asin", "acos", "atan",
  "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
  "exp", "log", "log2",
]);
const MATH_BINARY_IMPORTS = new Set(["pow", "atan2"]);

function wasmUleb128(n: number): number[] {
  const out: number[] = [];
  do {
    let byte = n & 0x7f;
    n >>>= 7;
    if (n !== 0) byte |= 0x80;
    out.push(byte);
  } while (n !== 0);
  return out;
}

/** Signed LEB128, for `i32.const` — a plain `wasmUleb128` would encode a
 * negative operand as an enormous positive one instead of sign-extending. */
function wasmSleb128(n: number): number[] {
  const out: number[] = [];
  let more = true;
  while (more) {
    let byte = n & 0x7f;
    n >>= 7;
    if ((n === 0 && (byte & 0x40) === 0) || (n === -1 && (byte & 0x40) !== 0)) more = false;
    else byte |= 0x80;
    out.push(byte);
  }
  return out;
}

export function wasmF64Bytes(value: number): number[] {
  const buf = new ArrayBuffer(8);
  new DataView(buf).setFloat64(0, value, true);
  return [...new Uint8Array(buf)];
}

export function wasmSection(id: number, body: number[]): number[] {
  return [id, ...wasmUleb128(body.length), ...body];
}

export function wasmVec(items: number[][]): number[] {
  return [...wasmUleb128(items.length), ...items.flat()];
}

function wasmStrBytes(s: string): number[] {
  const b = [...new TextEncoder().encode(s)];
  return [...wasmUleb128(b.length), ...b];
}

/**
 * Compile an Fn to a raw WASM binary module. `fn` must return a plain
 * scalar (`"float"`, `"int"`, `"uint"`, or `"bool"`) — no multi-return, no
 * `output()`/`varying()`/`attribute()`, and no control flow beyond
 * `If`/`Else` (see ROADMAP.md for the rest). Vectors and matrices are fully
 * first-class as *intermediate* values (construct, `toVar()`, `assign()`,
 * swizzle read/write, `dot`, componentwise `add`/`sub`/`mul`/`div`) — only
 * the function's own result must still be a scalar.
 */
/**
 * Stage 2 of `docs/design-shared-layout-ir.md`: place specific *aggregate*
 * uniforms at caller-given byte offsets (typically from `wgslUniformLayout`)
 * instead of this backend's own packed allocation, so the JS host can write
 * a GPU-shaped uniform buffer's bytes directly, with no repacking step for
 * this backend to read them. Experimental — not part of the stable API.
 *
 * `wgslUniformLayout`'s offsets assume each `float` component is WGSL's
 * 4-byte `f32`; this backend's own arithmetic is always f64 (`ROADMAP.md`,
 * "`float` is f64"), and an earlier version of this option used the
 * caller's raw offset as this backend's *only* address for that uniform —
 * safe for offset placement, but a real correctness bug once two
 * GPU-adjacent, 4-byte-spaced members met this backend's 8-byte-per-
 * component writes: one could spill straight over the next (see git
 * history around `rmsl-layout-interop.test.ts`'s original corruption
 * case). Fixed by giving a GPU-placed uniform *two* addresses instead of
 * one: the caller's raw offset, exactly `narrow` (`f32`) as the caller
 * expects and never touched by this backend's arithmetic directly, and an
 * ordinary packed scratch address like any other uniform gets. Reading the
 * uniform promotes narrow → f64 once, into that scratch address (see
 * `materializeIfNeeded`'s `"uniform"` case) — every consumer downstream
 * (`dot`, swizzles, `emitConstructStores`, ...) reads the ordinary scratch
 * address exactly as it always has, with no awareness a GPU-facing
 * representation exists at all.
 */
export type GpuUniformLayout = {
  /** Byte offset for each overridden uniform, keyed by slot (the `.name`
   * on the `uniform()` node) — everything `wgslUniformLayout` already
   * reports as `WgslUniformMember.name`/`.offset`. */
  offsets: Record<string, number>;
  /** The whole layout's total size (`wgslUniformLayout`'s `.size`) — the
   * bump allocator for everything else this function needs (locals,
   * scratch, non-overridden uniforms) starts right after it, so nothing it
   * places can ever land inside the reserved region. */
  totalSize: number;
};

export type CompileWasmFnOptions = CompileFnOptions & {
  gpuUniformLayout?: GpuUniformLayout;
  /** Phase 5 of `ROADMAP.md`: `undefined` (the default) is today's plain
   * scalar-returning function — every existing behavior stays exactly as
   * it was. Set to compile a full vertex or fragment stage instead:
   * `varying()` becomes a write (collected into the result) in `"vertex"`
   * and a read (from `ctx.varyings`) in `"fragment"`; `builtinPosition()`
   * is only writable in `"vertex"`; `builtinFragDepth()`/`fragCoord()`
   * only in `"fragment"`; and the function's own result no longer has to
   * be a scalar — it becomes the stage's `res.value`, exactly matching
   * `compileJS`'s `CompileJSOptions.stage`. */
  stage?: "vertex" | "fragment";
  /** `dFdx`/`dFdy`/`fwidth` have no meaning on a CPU target. `"throw"`
   * (the default, matching `compileJS`) rejects them; `"zero"` evaluates
   * them as `0` instead — matching `compileJS`'s own two options exactly. */
  derivatives?: "throw" | "zero";
  /**
   * Accepted for API parity with `compileJS`'s `CompileJSOptions` —
   * **has no effect**. `compileJS`'s `reentrant` exists because its
   * default hoists scratch to module-scope `let`s shared across every
   * call; this backend's WASM locals are already allocated fresh per call
   * frame by the VM itself, so there is no shared-scratch hazard to opt
   * out of here regardless of this option's value (see `ROADMAP.md`'s
   * former "Reentrancy" open question, now resolved this way).
   */
  reentrant?: boolean;
};

export function compileWasmFn(
  fn: (...args: any[]) => Node<ShaderType>,
  options: CompileWasmFnOptions,
): CompiledWasm {
  const paramNodes = options.params.map(p => var_(p.name, p.type));
  const root = fn(...paramNodes) as any;
  if (Array.isArray(root)) {
    throw new Error("[RMSL] compileWasmFn does not support multi-return functions.");
  }
  // The "root must be a plain scalar" check used to happen right here — it
  // now has to wait until after `collect()` (below) has run, since whether
  // that restriction still applies at all depends on `needsResult`, which
  // isn't known until the whole tree has been walked once (see the check
  // right after `collect(root)`).

  const paramTypeByName = new Map(options.params.map(p => [p.name, p.type]));
  const fnParamNames = new Set(options.params.map(p => p.name));
  // Matches `compileJS`'s own default exactly (`compileJSFn`: `options.stage
  // ?? "fragment"`) — a program can read fragment-only builtins without
  // having to pass `stage` explicitly, the same as compileJS allows today.
  const effectiveStage: "vertex" | "fragment" = options.stage ?? "fragment";

  // --- pass 1: collect the WASM param/local index space, every math import
  // the program needs, and every linear-memory address a vector/matrix value
  // needs (uniforms, params, `let`-bound vars, and one scratch slot per
  // vector/matrix-producing expression node) — all in first-seen order,
  // before any instruction bytes reference an index or address. ---
  // Scalar-only — these are the ones that actually occupy a WASM function
  // argument; every kind here always carries `shaderType`, unlike the wider
  // `WasmParam` union (a memory-only kind like `"fragCoordMemory"` doesn't).
  type ScalarWasmParam = Extract<WasmParam, { kind: "param" | "uniform" | "attribute" | "varying" }>;
  const params: ScalarWasmParam[] = [];
  const paramIndex = new Map<string, number>();
  const localSlots: string[] = [];
  const localIndex = new Map<string, number>();
  const localType = new Map<string, ScalarKind>();
  const importsUsed = new Set<string>();

  const memoryParams: WasmParam[] = [];
  const paramAddress = new Map<string, number>();
  const varAddress = new Map<string, number>();
  const uniformAddress = new Map<string, number>();
  // Present only for a `gpuUniformLayout`-placed uniform: its caller-given,
  // narrow (f32-per-float-component) raw offset — `uniformAddress` above
  // still holds its *own*, ordinary packed scratch address; this map is
  // only consulted by `materializeIfNeeded`'s promotion step.
  const gpuRawUniformAddress = new Map<string, number>();
  // Phase 5 input direction: `attribute()` always reads from `ctx.attributes`
  // regardless of stage; `varying()` reads from `ctx.varyings` only in a
  // fragment stage (a vertex stage's `varying()` is output-direction — see
  // the "varying"/"builtinPosition" handling in `walkStmt`'s assign case).
  const attributeAddress = new Map<string, number>();
  const varyingAddress = new Map<string, number>();
  // `fragCoord()` has no per-call slot — every reference in one program is
  // the same input, so one address serves all of them, unlike every other
  // node type here which is keyed by slot or by node identity.
  let fragCoordAddress: number | undefined;
  // Phase 5 output direction: written during the function body, read back
  // by `compileWasm` *after* the call — the first backend direction that
  // ever needs that. `needsResult` mirrors `compileJS`'s own
  // `ctx.jsNeedsRes`: false for every program that never touches any of
  // these (every existing test), in which case the function keeps its
  // original single-scalar-result shape untouched; true the moment any of
  // them is used, switching the compiled function to a zero-result shape
  // with everything read back from memory instead (see the `code`/type
  // section assembly near the end of this function). An explicit `"vertex"`
  // stage seeds this `true` unconditionally, even if nothing else in the
  // program does — a vertex stage's own result always maps to the implicit
  // position unless `builtinPosition()` was written some other way
  // (`assertStageResult`), so it's never just a plain WASM return value.
  // An aggregate root (`vec4`, ...) seeds it too: a plain WASM function can
  // only ever return one scalar, so anything wider always has to go
  // through memory regardless of stage — this is what lets `draw()` (see
  // `compileWasm`) support a per-pixel `vec4` color with no stage/output()
  // involved at all.
  let needsResult = options.stage === "vertex" || isAggregate(root._t as string);
  let positionWritten = false;
  const outputAddress = new Map<string, number>();
  const varyingOutputAddress = new Map<string, number>();
  let positionAddress: number | undefined;
  let fragDepthAddress: number | undefined;
  // Phase 6: each texture uniform slot's fixed metadata-block address (see
  // `TEXTURE_META_*`). The pixel data itself gets no compile-time address —
  // it lives in a growable heap `compileWasm`'s wrapper packs fresh before
  // every call (see `textureHeapBase` below and `CompiledWasm.textureHeapBase`).
  const textureMetadataAddress = new Map<string, number>();
  const scratchAddress = new WeakMap<object, number>();
  // Reserve [0, totalSize) for a caller-supplied GPU uniform layout, if any
  // — everything this backend places itself starts after it, so it can
  // never collide with an overridden uniform's address.
  let memCursor = options.gpuUniformLayout?.totalSize ?? 0;

  function allocateBytes(size: number): number {
    const addr = memCursor;
    memCursor += size;
    return addr;
  }
  function allocateFor(t: string): number {
    // A single-member call: with `reorderByAlignment: false` there's nothing
    // to reorder against, so this is exactly `allocateBytes(size)` for the
    // one type's own byte size — routed through the shared allocator so this
    // backend's sizing rules live in one place (`PACKED_RULES`) instead of
    // being computed inline here too.
    const { size } = planLayout([{ slot: t, type: t }], PACKED_RULES);
    return allocateBytes(size);
  }

  function addParam(spec: ScalarWasmParam, key: string): void {
    if (!paramIndex.has(key)) {
      paramIndex.set(key, params.length);
      params.push(spec);
    }
  }
  function addLocal(varName: string, kind: ScalarKind): void {
    if (!localIndex.has(varName)) {
      localIndex.set(varName, localSlots.length);
      localSlots.push(varName);
      localType.set(varName, kind);
    }
  }

  /** Every node whose value is an aggregate that isn't already addressed by
   * `paramAddress`/`varAddress`/`uniformAddress` — a construct, a literal
   * vector/matrix, a multi-component swizzle read, or a componentwise
   * arithmetic op — gets its own dedicated scratch address, keyed by node
   * identity so a repeated reference (`dot(v, v)`) shares one slot. */
  function isScratchNode(node: any): boolean {
    const t = node._t as string;
    if (!isAggregate(t)) return false;
    if (node.type === "construct") return true;
    if (node.type === t) return true; // literal vector/matrix
    if (node.type === "swizzle" && (node.value as string).length > 1) return true;
    if (node.type === "cross" || node.type === "reflect" || node.type === "normalize" || node.type === "matVecMul") return true;
    if (node.type === "dFdx" || node.type === "dFdy" || node.type === "fwidth") return true;
    if (node.type === "textureSize" || node.type === "textureLoad") return true;
    if (node.type === "texture" || node.type === "textureLod") return true;
    return node.type === "add" || node.type === "sub" || node.type === "mul" || node.type === "div";
  }

  function collect(node: any): void {
    if (node === null || typeof node !== "object") return;
    if (node.type === "var" && fnParamNames.has(node.value?.varName)) {
      const name = node.value.varName;
      const t = paramTypeByName.get(name)!;
      if (isAggregate(t)) {
        if (!paramAddress.has(name)) {
          const addr = allocateFor(t);
          paramAddress.set(name, addr);
          memoryParams.push({ kind: "paramMemory", name, shaderType: t, address: addr });
        }
      } else {
        addParam({ kind: "param", name, shaderType: t }, `param:${name}`);
      }
    } else if (node.type === "uniform" && isSamplerType(node.value.shaderType)) {
      // No 2D/3D-only restriction here — `textureSize()` works for any
      // sampler kind including cube, matching `compileJS`'s own
      // `_texSize` (no restriction at all). Only `texture()`/`textureLoad()`
      // restrict to 2D/3D, checked where those node types are compiled.
      const v = node.value;
      if (!textureMetadataAddress.has(v.slot)) {
        const addr = allocateBytes(TEXTURE_META_STRIDE);
        textureMetadataAddress.set(v.slot, addr);
        memoryParams.push({ kind: "textureMemory", slot: v.slot, samplerType: v.shaderType, metadataAddress: addr });
      }
    } else if (node.type === "uniform") {
      const v = node.value;
      if (isAggregate(v.shaderType)) {
        if (!uniformAddress.has(v.slot)) {
          // A GPU-placed uniform gets its own ordinary packed scratch
          // address like any other uniform — `nodeAddress`/every consumer
          // reads that, never the raw GPU offset directly (see
          // `materializeIfNeeded`'s "uniform" case for the narrow -> f64
          // promotion that keeps the two in sync).
          const addr = allocateFor(v.shaderType);
          uniformAddress.set(v.slot, addr);
          const gpuOffset = options.gpuUniformLayout?.offsets[v.slot];
          if (gpuOffset !== undefined) {
            gpuRawUniformAddress.set(v.slot, gpuOffset);
            memoryParams.push({ kind: "uniformMemory", slot: v.slot, shaderType: v.shaderType, address: gpuOffset, narrow: true });
          } else {
            memoryParams.push({ kind: "uniformMemory", slot: v.slot, shaderType: v.shaderType, address: addr });
          }
        }
      } else {
        addParam({ kind: "uniform", slot: v.slot, shaderType: v.shaderType }, `uniform:${v.slot}`);
      }
    } else if (node.type === "attribute") {
      const v = node.value;
      if (isAggregate(v.shaderType)) {
        if (!attributeAddress.has(v.slot)) {
          const addr = allocateFor(v.shaderType);
          attributeAddress.set(v.slot, addr);
          memoryParams.push({ kind: "attributeMemory", slot: v.slot, shaderType: v.shaderType, address: addr });
        }
      } else {
        addParam({ kind: "attribute", slot: v.slot, shaderType: v.shaderType }, `attribute:${v.slot}`);
      }
    } else if (node.type === "varying" && effectiveStage === "fragment") {
      // A vertex stage's `varying()` is output-direction instead — handled
      // in `walkStmt`'s assign case, not here (nothing to collect on read,
      // since it's never read in a vertex stage program the way it's
      // written here for a fragment one).
      const v = node.value;
      if (isAggregate(v.shaderType)) {
        if (!varyingAddress.has(v.slot)) {
          const addr = allocateFor(v.shaderType);
          varyingAddress.set(v.slot, addr);
          memoryParams.push({ kind: "varyingMemory", slot: v.slot, shaderType: v.shaderType, address: addr });
        }
      } else {
        addParam({ kind: "varying", slot: v.slot, shaderType: v.shaderType }, `varying:${v.slot}`);
      }
    } else if (node.type === "fragCoord") {
      if (effectiveStage !== "fragment") {
        throw new Error("[RMSL] compileWasmFn: fragCoord() can only be used in fragment shaders");
      }
      if (fragCoordAddress === undefined) {
        fragCoordAddress = allocateFor("vec2");
        memoryParams.push({ kind: "fragCoordMemory", address: fragCoordAddress });
      }
    } else if (node.type === "output") {
      needsResult = true;
      const v = node.value;
      if (!outputAddress.has(v.slot)) {
        const addr = allocateFor(v.shaderType);
        outputAddress.set(v.slot, addr);
        memoryParams.push({ kind: "outputMemory", slot: v.slot, shaderType: v.shaderType, address: addr });
      }
    } else if (node.type === "varying" && effectiveStage === "vertex") {
      // Output-direction here (see the fragment-stage "varying" branch
      // above for the read direction) — allocated on first encounter
      // whether that's a read-back or the assign that writes it, exactly
      // like "output" above.
      needsResult = true;
      const v = node.value;
      if (!varyingOutputAddress.has(v.slot)) {
        const addr = allocateFor(v.shaderType);
        varyingOutputAddress.set(v.slot, addr);
        memoryParams.push({ kind: "varyingOutputMemory", slot: v.slot, shaderType: v.shaderType, address: addr });
      }
    } else if (node.type === "builtinPosition") {
      needsResult = true;
      if (positionAddress === undefined) {
        positionAddress = allocateFor("vec4");
        memoryParams.push({ kind: "positionMemory", address: positionAddress });
      }
    } else if (node.type === "builtinFragDepth") {
      if (effectiveStage !== "fragment") {
        throw new Error("[RMSL] compileWasmFn: builtinFragDepth() can only be used in fragment shaders");
      }
      needsResult = true;
      if (fragDepthAddress === undefined) {
        fragDepthAddress = allocateFor("float");
        memoryParams.push({ kind: "fragDepthMemory", address: fragDepthAddress });
      }
    } else if (node.type === "assign" && node.params[0].type === "builtinPosition") {
      // The one thing that specifically depends on *writing* rather than
      // merely referencing builtinPosition — `assertStageResult` below
      // needs to know a vertex stage supplied its own position, so its
      // result doesn't have to be a vec4 too.
      positionWritten = true;
    } else if (node.type === "let") {
      const targetNode = node.params[0];
      const t = targetNode._t as string;
      if (isAggregate(t)) {
        const varName = targetNode.value.varName;
        if (!varAddress.has(varName)) varAddress.set(varName, allocateFor(t));
      } else {
        addLocal(targetNode.value.varName, scalarKindOf(t));
      }
    } else if (MATH_UNARY_IMPORTS.has(node.type) || MATH_BINARY_IMPORTS.has(node.type)) {
      importsUsed.add(node.type);
    } else if (node.type === "exp2") {
      // exp2(x) compiles to a call to the imported pow(2, x) — see walkExpr —
      // so that import has to be registered here too, even though "exp2"
      // itself isn't one of the imported names.
      importsUsed.add("pow");
    }
    if (isScratchNode(node) && !scratchAddress.has(node)) {
      const addr = allocateFor(node._t as string);
      // `normalize`/`reflect` each need one scalar (the length, the
      // reflection dot product) read back multiple times while computing
      // every output component — rather than inventing a WASM-local scratch
      // mechanism, one extra f64 slot right after the node's own output
      // components holds it, computed once.
      if (node.type === "normalize" || node.type === "reflect") allocateBytes(8);
      if ((node.type === "texture" || node.type === "textureLod") && !isIntegerSamplerType(node.params[0]._t as string)) {
        // Scratch for every value `emitTextureSampleStores` computes once
        // per sample and every one of the 4 channels then reuses — nearest
        // mode's own wrapped x/y/z, bilinear/trilinear's wrapped tap
        // indices, and the blend weights — instead of each channel
        // recomputing all of it from scratch. Same "extra scratch right
        // after the node's own result" treatment as normalize/reflect
        // above, just a lot more of it (3 nearest indices + 6 tap indices,
        // all i32, plus 3 f64 blend weights = 60 bytes).
        allocateBytes(60);
      }
      scratchAddress.set(node, addr);
    }
    if (Array.isArray(node.params)) for (const p of node.params) collect(p);
  }
  collect(root);

  // Now that the whole tree has been walked once, `needsResult` is settled
  // — decide what the root is actually allowed to be, and where its own
  // value (if it has one) will live.
  let resultKind: ScalarKind;
  let valueAddress: number | undefined;
  if (needsResult) {
    assertStageResult(effectiveStage, root._t === "void" ? undefined : (root._t as string), positionWritten);
    resultKind = "float"; // unused for the function's own WASM result type in this mode (see below) — a value, never read as a bare WASM return.
    if (effectiveStage === "vertex" && !positionWritten) {
      // "Otherwise the result becomes the position" (assertStageResult's
      // own wording) — already required to be a vec4 by the check above,
      // so it's written directly into the position slot instead of a
      // separate "value" one; `JsShaderResult.value` stays unset here,
      // matching `compileJS`.
      if (positionAddress === undefined) {
        positionAddress = allocateFor("vec4");
        memoryParams.push({ kind: "positionMemory", address: positionAddress });
      }
      valueAddress = positionAddress;
    } else if (root._t !== "void") {
      const t = root._t as string;
      valueAddress = isAggregate(t) ? allocateFor(t) : allocateBytes(componentSizeOf(scalarKindOf(t)));
      memoryParams.push({ kind: "valueMemory", shaderType: t as ShaderType, address: valueAddress });
    }
  } else {
    if (root._t !== "float" && root._t !== "int" && root._t !== "uint" && root._t !== "bool") {
      throw new Error(`[RMSL] compileWasmFn only supports a scalar result so far, got "${root._t}".`);
    }
    resultKind = scalarKindOf(root._t);
  }

  const importNames = [...importsUsed].sort();
  const importIndexOf = new Map(importNames.map((name, i) => [name, i]));

  // --- pass 2: emit instruction bytes against the now-fixed index/address
  // space. ---
  function localSlotIndex(varName: string): number {
    const i = localIndex.get(varName);
    if (i === undefined) throw new Error(`[RMSL] compileWasmFn: read of undeclared var "${varName}"`);
    return params.length + i;
  }
  function paramSlotIndex(key: string): number {
    const i = paramIndex.get(key);
    if (i === undefined) throw new Error(`[RMSL] compileWasmFn: internal error, unindexed slot "${key}"`);
    return i;
  }
  function callImport(name: string): number[] {
    return [WASM_OP.call, ...wasmUleb128(importIndexOf.get(name)!)];
  }

  /** The fixed byte address of an aggregate-valued node — a pure
   * compile-time lookup, never bytecode, since nothing in this phase needs
   * a dynamically-computed address. */
  function nodeAddress(node: any): number {
    if (node.type === "var") {
      const name = node.value.varName;
      const addr = fnParamNames.has(name) ? paramAddress.get(name) : varAddress.get(name);
      if (addr === undefined) throw new Error(`[RMSL] compileWasmFn: read of undeclared aggregate var "${name}"`);
      return addr;
    }
    if (node.type === "uniform") {
      const addr = uniformAddress.get(node.value.slot);
      if (addr === undefined) throw new Error(`[RMSL] compileWasmFn: internal error, unaddressed uniform "${node.value.slot}"`);
      return addr;
    }
    if (node.type === "attribute") {
      const addr = attributeAddress.get(node.value.slot);
      if (addr === undefined) throw new Error(`[RMSL] compileWasmFn: internal error, unaddressed attribute "${node.value.slot}"`);
      return addr;
    }
    if (node.type === "varying" && effectiveStage === "fragment") {
      const addr = varyingAddress.get(node.value.slot);
      if (addr === undefined) throw new Error(`[RMSL] compileWasmFn: internal error, unaddressed varying "${node.value.slot}"`);
      return addr;
    }
    if (node.type === "fragCoord") {
      if (fragCoordAddress === undefined) throw new Error("[RMSL] compileWasmFn: internal error, unaddressed fragCoord");
      return fragCoordAddress;
    }
    if (node.type === "output") {
      const addr = outputAddress.get(node.value.slot);
      if (addr === undefined) throw new Error(`[RMSL] compileWasmFn: internal error, unaddressed output "${node.value.slot}"`);
      return addr;
    }
    if (node.type === "varying" && effectiveStage === "vertex") {
      const addr = varyingOutputAddress.get(node.value.slot);
      if (addr === undefined) throw new Error(`[RMSL] compileWasmFn: internal error, unaddressed varying "${node.value.slot}"`);
      return addr;
    }
    if (node.type === "builtinPosition") {
      // Only a vertex stage may *read* its own position back — matches
      // `compileJS`'s `assertPositionIsReadable` exactly; writing it (the
      // `walkStmt` assign-target case) never calls `nodeAddress`, so this
      // check only ever fires for a read.
      if (effectiveStage !== "vertex") {
        throw new Error(
          "[RMSL] compileWasmFn: builtinPosition() is the vertex stage's output position, and a "
          + "fragment stage cannot read it. Pass the value you need through a "
          + "varying() instead.",
        );
      }
      if (positionAddress === undefined) throw new Error("[RMSL] compileWasmFn: internal error, unaddressed builtinPosition");
      return positionAddress;
    }
    if (node.type === "builtinFragDepth") {
      if (fragDepthAddress === undefined) throw new Error("[RMSL] compileWasmFn: internal error, unaddressed builtinFragDepth");
      return fragDepthAddress;
    }
    const addr = scratchAddress.get(node);
    if (addr === undefined) throw new Error(`[RMSL] compileWasmFn: internal error, unaddressed node "${node.type}"`);
    return addr;
  }

  function loadComponent(addr: number, kind: ScalarKind, byteOffset: number): number[] {
    return [...i32ConstBytes(addr), kind === "float" ? WASM_OP.f64Load : WASM_OP.i32Load, 0x00, ...wasmUleb128(byteOffset)];
  }
  function storeComponent(addr: number, kind: ScalarKind, byteOffset: number, valueBytes: number[]): number[] {
    return [...i32ConstBytes(addr), ...valueBytes, kind === "float" ? WASM_OP.f64Store : WASM_OP.i32Store, 0x00, ...wasmUleb128(byteOffset)];
  }

  /** The dynamic-address counterpart of `loadComponent`: every other load in
   * this file targets a compile-time-constant address (`i32ConstBytes(addr)`
   * pushed as the base, with the component's own byte offset folded into
   * the load instruction's immediate) — texture heap access is the first
   * case where the address itself is only known at run time (it depends on
   * `compileWasm`'s per-call heap packing), so `addrBytes` is arbitrary
   * bytecode leaving the full address on the stack and the instruction's
   * own immediate offset is always 0. */
  function loadDynamic(addrBytes: number[], kind: ScalarKind): number[] {
    return [...addrBytes, kind === "float" ? WASM_OP.f64Load : WASM_OP.i32Load, 0x00, 0x00];
  }

  /** The dynamic-address counterpart of `storeComponent`, mirroring
   * `loadDynamic` — needed for `draw`'s per-pixel output buffer, whose
   * write address depends on the loop's own runtime `x`/`y` counters
   * rather than being a compile-time constant. */
  function storeDynamic(addrBytes: number[], kind: ScalarKind, valueBytes: number[]): number[] {
    return [...addrBytes, ...valueBytes, kind === "float" ? WASM_OP.f64Store : WASM_OP.i32Store, 0x00, 0x00];
  }

  /** Populate `nodeAddress(node)` with `node`'s value, for any node type
   * `isScratchNode` addresses — a `"var"` needs no work (its data is
   * already valid, set by a prior `let`/`assign`), and an ordinary
   * `"uniform"` needs none either (`compileWasm`'s JS wrapper already
   * wrote it at `nodeAddress(node)` before the call) — but a
   * `gpuUniformLayout`-placed one needs its narrow (f32) components
   * promoted into that ordinary address first; see `emitGpuUniformPromote`.
   * Always materializes an aggregate sub-node exactly once, however many
   * of its components end up read, so nesting doesn't blow up
   * proportionally to width. */
  function materializeIfNeeded(node: any): number[] {
    switch (node.type) {
      case "var":
        return [];
      case "uniform": {
        const rawAddr = gpuRawUniformAddress.get(node.value.slot);
        return rawAddr === undefined ? [] : emitGpuUniformPromote(node, nodeAddress(node), rawAddr);
      }
      case "attribute":
      case "fragCoord":
        return [];
      case "varying":
      case "output":
      case "builtinPosition":
      case "builtinFragDepth":
        // A fragment-stage `varying()` read: data already valid, written by
        // `compileWasm` before the call, exactly like an attribute or
        // uniform. Every other case here is output-direction, read back
        // *after* an earlier `.assign()` within the same program — no work
        // needed either way, since nothing needs converting or copying to
        // make a prior write visible to a later read of the same address.
        return [];
      case "construct":
        return emitConstructStores(node, nodeAddress(node));
      case "swizzle":
        if ((node.value as string).length <= 1) {
          throw new Error("[RMSL] compileWasmFn: internal error, materializing a scalar swizzle");
        }
        return emitSwizzleStores(node, nodeAddress(node));
      case "add":
      case "sub":
      case "div":
        return emitComponentwiseStores(node, nodeAddress(node));
      case "mul":
        // Componentwise `mul` covers vector±vector and vector±scalar
        // broadcast (see emitComponentwiseStores), but a matrix times
        // another matrix means a real matrix product, not a per-component
        // one — `mat.mul(scalar)` stays componentwise (only one operand is
        // a matrix there).
        if (MATRIX_DIMENSIONS[node.params[0]._t] !== undefined && MATRIX_DIMENSIONS[node.params[1]._t] !== undefined) {
          return emitMatMatMulStores(node, nodeAddress(node));
        }
        return emitComponentwiseStores(node, nodeAddress(node));
      case "matVecMul":
        return emitMatVecMulStores(node, nodeAddress(node));
      case "cross":
        return emitCrossStores(node, nodeAddress(node));
      case "normalize":
        return emitNormalizeStores(node, nodeAddress(node));
      case "reflect":
        return emitReflectStores(node, nodeAddress(node));
      case "dFdx":
      case "dFdy":
      case "fwidth":
        return emitDerivativeZeroStores(node, nodeAddress(node));
      case "textureSize":
        return emitTextureSizeStores(node, nodeAddress(node));
      case "textureLoad":
        return emitTexelFetchStores(node, nodeAddress(node));
      case "texture":
      case "textureLod":
        return emitTextureSampleStores(node, nodeAddress(node));
      default:
        if (node.type === node._t && Array.isArray(node.value)) {
          return emitLiteralStores(node, nodeAddress(node));
        }
        throw new Error(`[RMSL] compileWasmFn: unsupported node type in vector position: "${node.type}"`);
    }
  }

  /** Promote a `gpuUniformLayout`-placed uniform's narrow (`f32`) components
   * at `rawAddr` into its own ordinary packed (`f64`) scratch address
   * `addr`, so every consumer downstream reads `addr` exactly like any
   * other uniform, unaware a narrower representation exists at all. Only
   * float-family uniforms need this — `wgslUniformLayout`'s `i32`/`u32`
   * component width already matches this backend's own, so an int/uint
   * uniform (not reachable here today, since `GpuUniformLayout` is
   * aggregate-only, but kept correct for when it is) would need no
   * conversion, just a plain same-width copy. */
  function emitGpuUniformPromote(node: any, addr: number, rawAddr: number): number[] {
    const kind = elementKindOf(node._t as string);
    const width = componentCountOf(node._t as string);
    const rawCompSize = kind === "float" ? 4 : componentSizeOf(kind);
    const compSize = componentSizeOf(kind);
    const out: number[] = [];
    for (let k = 0; k < width; k++) {
      const rawBytes = kind === "float"
        ? [...i32ConstBytes(rawAddr + k * rawCompSize), WASM_OP.f32Load, 0x00, ...wasmUleb128(0), WASM_OP.f64PromoteF32]
        : loadComponent(rawAddr, kind, k * rawCompSize);
      out.push(...storeComponent(addr, kind, k * compSize, rawBytes));
    }
    return out;
  }

  function emitConstructStores(node: any, addr: number): number[] {
    const targetType = node._t as string;
    const matShape = MATRIX_DIMENSIONS[targetType];
    if (matShape) {
      const [cols, rows] = matShape;
      if (node.params.length === 1 && componentCountOf(node.params[0]._t) > 1) {
        throw new Error('[RMSL] compileWasmFn: unsupported node type in vector position: "matrix-from-matrix construct"');
      }
      const out: number[] = [];
      if (node.params.length === 1) {
        // A single scalar param means a diagonal matrix — every off-diagonal
        // cell is zero, and the scalar expression is recomputed per
        // diagonal cell (no sub-expression caching, same tradeoff the rest
        // of this file already accepts).
        for (let c = 0; c < cols; c++) {
          for (let r = 0; r < rows; r++) {
            const valueBytes = c === r ? walkExpr(node.params[0]) : f64ConstBytes(0);
            out.push(...storeComponent(addr, "float", (c * rows + r) * 8, valueBytes));
          }
        }
        return out;
      }
      // One param per column.
      node.params.forEach((colNode: any, c: number) => {
        out.push(...materializeIfNeeded(colNode));
        const colAddr = nodeAddress(colNode);
        for (let r = 0; r < rows; r++) {
          out.push(...storeComponent(addr, "float", (c * rows + r) * 8, loadComponent(colAddr, "float", r * 8)));
        }
      });
      return out;
    }

    const targetKind = elementKindOf(targetType);
    const compSize = componentSizeOf(targetKind);
    const out: number[] = [];
    let compIndex = 0;
    for (const p of node.params) {
      const pWidth = componentCountOf(p._t);
      if (pWidth === 1) {
        const pKind = scalarKindOf(p._t as string);
        out.push(...storeComponent(addr, targetKind, compIndex * compSize, convertComponent(walkExpr(p), pKind, targetKind)));
        compIndex++;
      } else {
        out.push(...materializeIfNeeded(p));
        const pAddr = nodeAddress(p);
        const pKind = elementKindOf(p._t);
        const pCompSize = componentSizeOf(pKind);
        for (let k = 0; k < pWidth; k++) {
          out.push(...storeComponent(addr, targetKind, compIndex * compSize, convertComponent(loadComponent(pAddr, pKind, k * pCompSize), pKind, targetKind)));
          compIndex++;
        }
      }
    }
    return out;
  }

  /**
   * Convert a value already on the stack from `fromKind` to `toKind` for a
   * `construct` whose target type differs from a param's own — e.g.
   * `vec3(...).toIVec3()`, a per-component float-to-int truncation the
   * source vec3's own components don't carry. Matches GLSL/WGSL/JS's own
   * scalar-cast semantics: truncation toward zero into an integer, exact
   * widening into a float. `int`/`uint`/`bool` share one representation
   * (i32) so converting between those three is always a no-op; a `bool`
   * on either side of a float boundary is not a case any construct in this
   * DSL exercises today, so it stays untouched rather than guessing a
   * semantics nothing currently tests.
   */
  function convertComponent(valueBytes: number[], fromKind: ScalarKind, toKind: ScalarKind): number[] {
    if (fromKind === toKind || fromKind === "bool" || toKind === "bool") return valueBytes;
    if (fromKind === "float") {
      return [...valueBytes, toKind === "uint" ? WASM_OP.i32TruncF64U : WASM_OP.i32TruncF64S];
    }
    if (toKind === "float") {
      return [...valueBytes, fromKind === "uint" ? WASM_OP.f64ConvertI32U : WASM_OP.f64ConvertI32S];
    }
    return valueBytes; // int <-> uint: identical bit pattern.
  }

  /** `dFdx`/`dFdy`/`fwidth` have no meaning on a CPU target — only reachable
   * when `options.derivatives === "zero"` was explicitly chosen (the
   * default, `"throw"`, is checked at the top of `walkExpr`'s scalar case,
   * which every aggregate reference to one of these nodes also passes
   * through first via `readComponent`/`dot`/etc. calling `materializeIfNeeded`
   * — but a purely aggregate derivative, e.g. `dFdx(vec3)`, never visits
   * `walkExpr` at all, so the check is repeated here too). */
  function assertDerivativesAllowed(node: any): void {
    if (options.derivatives === "zero") return;
    throw new Error(
      `[RMSL] compileWasmFn: ${node.type}() has no meaning on the CPU target. `
      + `Compile with { derivatives: "zero" } to evaluate it as 0.`,
    );
  }

  function emitDerivativeZeroStores(node: any, addr: number): number[] {
    assertDerivativesAllowed(node);
    const kind = elementKindOf(node._t as string);
    const compSize = componentSizeOf(kind);
    const width = componentCountOf(node._t as string);
    const out: number[] = [];
    for (let k = 0; k < width; k++) {
      out.push(...storeComponent(addr, kind, k * compSize, kind === "float" ? f64ConstBytes(0) : i32ConstBytes(0)));
    }
    return out;
  }

  /** `textureSize()` needs no sampling math at all — just a read of the
   * dimensions `compileWasm`'s wrapper already wrote into this texture's
   * metadata block before the call. */
  function emitTextureSizeStores(node: any, addr: number): number[] {
    const metaAddr = textureMetadataAddress.get(node.params[0].value.slot);
    if (metaAddr === undefined) {
      throw new Error(`[RMSL] compileWasmFn: internal error, unaddressed texture "${node.params[0].value.slot}"`);
    }
    const width = componentCountOf(node._t as string); // 2 for uvec2, 3 for uvec3
    const out: number[] = [];
    const fieldOffset = [TEX_META_WIDTH, TEX_META_HEIGHT, TEX_META_DEPTH];
    for (let k = 0; k < width; k++) {
      out.push(...storeComponent(addr, "uint", k * 4, loadComponent(metaAddr, "uint", fieldOffset[k])));
    }
    return out;
  }

  /**
   * Shared bytecode for an unfiltered texel fetch: `textureLoad()` for any
   * sampler kind, and `texture()`/`textureLod()` on an integer sampler
   * (`emitTextureSampleStores` routes the float-sampler, filtered case
   * elsewhere). Mirrors `_texFetch2d`/`_texFetch3d`/`_texFetchUnorm2d`/
   * `_texFetchUnorm3d` (`rmsl-compile-js.ts`) exactly: coordinates are
   * already texel-space integers (no UV scaling), every axis is
   * bounds-checked against the metadata's width/height/depth, and — only
   * when every axis is in range — up to 4 channels are read (missing
   * green/blue default to 0, missing alpha to 1), a float sampler's raw
   * value divided by its unorm divisor. Out of range gives a literal
   * all-zero result (including alpha), matching `compileJS` — the
   * "alpha defaults to 1" rule is a missing-*channel* fallback, not a
   * missing-*texel* one.
   *
   * Every conditional here is a `select`, not a branch, matching this
   * file's existing ternary/min/max style — which means both the "in
   * range" and "out of range" values are always computed, so the memory
   * address actually dereferenced is always clamped into range first
   * (`safeAxis`) regardless of how wild the real coordinate is; the real,
   * unclamped coordinate only ever feeds the bounds comparison, never an
   * address. Every helper below re-emits its bytecode fresh on each call
   * (recompute, not cache) — consistent with this file's existing
   * `selectExpr`/`minOrMax`, which already accept the same tradeoff.
   */
  function emitTexelFetchStores(node: any, addr: number): number[] {
    const samplerNode = node.params[0];
    const coordsNode = node.params[1];
    const samplerType = samplerNode._t as string;
    assertSampled2Dor3D(samplerType);
    const is3D = samplerType.endsWith("3D");
    const isInteger = isIntegerSamplerType(samplerType);
    // A plain `const` narrowed by an `if` loses that narrowing inside the
    // nested `function` declarations below (hoisted, so TS can't assume
    // they only run after the check) — resolving it to a definite `number`
    // up front sidesteps that instead.
    const metaAddr: number = ((): number => {
      const a = textureMetadataAddress.get(samplerNode.value.slot);
      if (a === undefined) throw new Error(`[RMSL] compileWasmFn: internal error, unaddressed texture "${samplerNode.value.slot}"`);
      return a;
    })();
    const materialize = materializeIfNeeded(coordsNode);
    const coordsAddr = nodeAddress(coordsNode);
    const coordKind = elementKindOf(coordsNode._t as string); // "int" or "uint"
    const dims = is3D ? [TEX_META_WIDTH, TEX_META_HEIGHT, TEX_META_DEPTH] : [TEX_META_WIDTH, TEX_META_HEIGHT];

    const rawAxis = (k: number) => loadComponent(coordsAddr, "int", k * 4);
    const dimAxis = (offset: number) => loadComponent(metaAddr, "int", offset);

    function axisOOB(k: number): number[] {
      const tooHigh = [...rawAxis(k), ...dimAxis(dims[k]), coordKind === "uint" ? WASM_OP.i32GeU : WASM_OP.i32GeS];
      if (coordKind === "uint") return tooHigh; // never negative — the "< 0" half is always false
      const tooLow = [...rawAxis(k), ...i32ConstBytes(0), WASM_OP.i32LtS];
      return [...tooLow, ...tooHigh, WASM_OP.i32Or];
    }
    function oobFlag(): number[] {
      let flag = axisOOB(0);
      for (let k = 1; k < dims.length; k++) flag = [...flag, ...axisOOB(k), WASM_OP.i32Or];
      return flag;
    }

    // clamp(raw, 0, dim-1) — a safe index to dereference even when `raw`
    // itself is wildly out of range (see the doc comment above).
    function safeAxis(k: number): number[] {
      const raw = rawAxis(k);
      const dimMinus1 = [...dimAxis(dims[k]), ...i32ConstBytes(1), WASM_OP.i32Sub];
      const nonNegative = coordKind === "uint" ? raw : selectExpr(i32ConstBytes(0), raw, [...raw, ...i32ConstBytes(0), WASM_OP.i32LtS]);
      const tooHigh = coordKind === "uint" ? [...raw, ...dimMinus1, WASM_OP.i32GtU] : [...nonNegative, ...dimMinus1, WASM_OP.i32GtS];
      return selectExpr(dimMinus1, nonNegative, tooHigh);
    }

    // (y*width + x), or ((z*height + y)*width + x) for 3D — `_texFetch2d`/
    // `_texFetch3d`'s index formula exactly.
    function texelIndexBytes(): number[] {
      const x = safeAxis(0);
      const y = safeAxis(1);
      const yx = [...y, ...dimAxis(TEX_META_WIDTH), WASM_OP.i32Mul, ...x, WASM_OP.i32Add];
      if (!is3D) return yx;
      const z = safeAxis(2);
      const zy = [...z, ...dimAxis(TEX_META_HEIGHT), WASM_OP.i32Mul, ...y, WASM_OP.i32Add];
      return [...zy, ...dimAxis(TEX_META_WIDTH), WASM_OP.i32Mul, ...x, WASM_OP.i32Add];
    }

    // dataAddr + (texelIndex*channels + i) * 8 — the heap always stores
    // one f64 per component, regardless of sampler kind.
    function elemAddrBytes(i: number): number[] {
      const dataAddr = loadComponent(metaAddr, "int", TEX_META_DATA_ADDR);
      const channels = loadComponent(metaAddr, "int", TEX_META_CHANNELS);
      const elemOffset = [...texelIndexBytes(), ...channels, WASM_OP.i32Mul, ...i32ConstBytes(i), WASM_OP.i32Add];
      const byteOffset = [...elemOffset, ...i32ConstBytes(3), WASM_OP.i32Shl]; // * 8
      return [...dataAddr, ...byteOffset, WASM_OP.i32Add];
    }

    function channelValue(i: number): number[] {
      const present = [...i32ConstBytes(i), ...loadComponent(metaAddr, "int", TEX_META_CHANNELS), WASM_OP.i32LtS];
      if (isInteger) {
        const truncOp = samplerType.startsWith("isampler") ? WASM_OP.i32TruncF64S : WASM_OP.i32TruncF64U;
        const fetched = [...loadDynamic(elemAddrBytes(i), "float"), truncOp];
        const missingChannelDefault = i === 3 ? i32ConstBytes(1) : i32ConstBytes(0);
        const inRange = selectExpr(fetched, missingChannelDefault, present);
        return selectExpr(i32ConstBytes(0), inRange, oobFlag());
      }
      const divisor = loadComponent(metaAddr, "float", TEX_META_UNORM_DIVISOR);
      const fetched = [...loadDynamic(elemAddrBytes(i), "float"), ...divisor, WASM_OP.f64Div];
      const missingChannelDefault = i === 3 ? f64ConstBytes(1) : f64ConstBytes(0);
      const inRange = selectExpr(fetched, missingChannelDefault, present);
      return selectExpr(f64ConstBytes(0), inRange, oobFlag());
    }

    const targetKind = elementKindOf(node._t as string);
    const compSize = componentSizeOf(targetKind);
    const out = [...materialize];
    for (let i = 0; i < 4; i++) {
      out.push(...storeComponent(addr, targetKind, i * compSize, channelValue(i)));
    }
    return out;
  }

  /**
   * `texture()`/`textureLod()` on a float sampler: wrap-addressed,
   * optionally bilinear/trilinear-filtered sampling, porting `_tex2d`/
   * `_tex3d`/`_wrap`/`_lerp2` (`rmsl-compile-js.ts`) to bytecode. An
   * integer sampler is never filterable in either language and instead
   * takes the exact same unfiltered path `textureLoad()` uses (see
   * `emitTexelFetchStores`) — `texture()`'s coordinates on an integer
   * sampler are already texel-space, not normalized UVs (`ISampler2DOps`'s
   * doc comment, `rmsl-core.ts`), so no wrap/UV-scaling code ever applies
   * to them regardless.
   *
   * Unlike `textureLoad()`, wrapping (not clamping-for-safety) already
   * guarantees every tap index this function computes is in `[0, dim)`, so
   * no separate "safe address" step is needed here — the wrapped index
   * *is* the safe index. And there's no "out of range" case at all: a
   * `texture()` call always returns some in-range, wrap-addressed sample.
   *
   * Every conditional except `magFilter` (wrap mode, missing-channel
   * default) is a `select`, matching this file's existing ternary/min/max
   * style and this phase's own `emitTexelFetchStores`. What is *not*
   * duplicated any more: every value that doesn't depend on which of the
   * 4 channels is being read — the wrapped tap indices, the blend
   * weights, nearest mode's own wrapped index — is computed exactly once
   * per sample into fixed scratch addresses right after this node's own
   * result (see `collect`'s `isScratchNode` extra-allocation block), and
   * every channel just loads it back. Before this, each of the 4 channels
   * recomputed all of that from scratch, which a benchmark showed was
   * most of filtered sampling's remaining cost after the `magFilter`
   * branch and lerp-reformulation fixes (see `ROADMAP.md`'s texture
   * performance section) — this is the first place in this file using a
   * scratch address as a genuine compiler-managed temporary rather than
   * a node's own output value.
   */
  function emitTextureSampleStores(node: any, addr: number): number[] {
    const samplerNode = node.params[0];
    const coordsNode = node.params[1];
    const samplerType = samplerNode._t as string;
    assertSampled2Dor3D(samplerType);
    if (isIntegerSamplerType(samplerType)) return emitTexelFetchStores(node, addr);

    const is3D = samplerType.endsWith("3D");
    const metaAddr: number = ((): number => {
      const a = textureMetadataAddress.get(samplerNode.value.slot);
      if (a === undefined) throw new Error(`[RMSL] compileWasmFn: internal error, unaddressed texture "${samplerNode.value.slot}"`);
      return a;
    })();
    const materialize = materializeIfNeeded(coordsNode);
    const coordsAddr = nodeAddress(coordsNode);

    const uv = (k: number): number[] => loadComponent(coordsAddr, "float", k * 8);
    const dimI32 = (offset: number): number[] => loadComponent(metaAddr, "int", offset);
    const dimF64 = (offset: number): number[] => [...dimI32(offset), WASM_OP.f64ConvertI32S];

    // Scratch right after this node's own vec4 result (`collect`'s
    // `isScratchNode` extra-allocation block reserves 60 bytes there for
    // exactly this) — every per-sample value below, computed once and
    // read back cheaply by every channel instead of being recomputed.
    const scratch = addr + 32;
    const NEAREST_X = scratch, NEAREST_Y = scratch + 4, NEAREST_Z = scratch + 8;
    const XA = scratch + 12, XB = scratch + 16, YA = scratch + 20, YB = scratch + 24, ZA = scratch + 28, ZB = scratch + 32;
    const TX = scratch + 36, TY = scratch + 44, TZ = scratch + 52;

    // repeat/mirror/clamp, dispatched on a runtime mode value (0/1/2) —
    // every formula computed unconditionally, picked with nested `select`s.
    function wrapAxis(idxBytes: number[], dimOffset: number, wrapOffset: number): number[] {
      const dim = (): number[] => dimI32(dimOffset);
      const mode = (): number[] => dimI32(wrapOffset);
      const dimMinus1 = (): number[] => [...dim(), ...i32ConstBytes(1), WASM_OP.i32Sub];
      const clampVal = selectExpr(
        i32ConstBytes(0),
        selectExpr(dimMinus1(), idxBytes, [...idxBytes, ...dimMinus1(), WASM_OP.i32GtS]),
        [...idxBytes, ...i32ConstBytes(0), WASM_OP.i32LtS],
      );
      const repeatVal = [...[...[...idxBytes, ...dim(), WASM_OP.i32RemS], ...dim(), WASM_OP.i32Add], ...dim(), WASM_OP.i32RemS];
      const twoDim = (): number[] => [...dim(), ...i32ConstBytes(2), WASM_OP.i32Mul];
      const period = [...[...[...idxBytes, ...twoDim(), WASM_OP.i32RemS], ...twoDim(), WASM_OP.i32Add], ...twoDim(), WASM_OP.i32RemS];
      const mirrorVal = selectExpr(
        period,
        [...twoDim(), ...i32ConstBytes(1), WASM_OP.i32Sub, ...period, WASM_OP.i32Sub],
        [...period, ...dim(), WASM_OP.i32LtS],
      );
      const modeIsMirror = [...mode(), ...i32ConstBytes(2), WASM_OP.i32Eq];
      const modeIsClamp = [...mode(), ...i32ConstBytes(0), WASM_OP.i32Eq];
      const repeatOrMirror = selectExpr(mirrorVal, repeatVal, modeIsMirror);
      return selectExpr(clampVal, repeatOrMirror, modeIsClamp);
    }

    // Half-texel-offset sample point, split into a floor'd integer part and
    // a [0,1) fractional blend weight — `_tex2d`/`_tex3d`'s own `fx`/`x0`/`tx`.
    function fracAxis(k: number, dimOffset: number): { i0: number[]; t: number[] } {
      const f = [...uv(k), ...dimF64(dimOffset), WASM_OP.f64Mul, ...f64ConstBytes(0.5), WASM_OP.f64Sub];
      const f0 = [...f, WASM_OP.f64Floor];
      return { i0: [...f0, WASM_OP.i32TruncF64S], t: [...f, ...f0, WASM_OP.f64Sub] };
    }

    // --- Everything above this point is channel-independent — computed
    // exactly once per sample and stored, never recomputed per channel. ---
    const nearestX = wrapAxis([...uv(0), ...dimF64(TEX_META_WIDTH), WASM_OP.f64Mul, WASM_OP.f64Floor, WASM_OP.i32TruncF64S], TEX_META_WIDTH, TEX_META_WRAP_S);
    const nearestY = wrapAxis([...uv(1), ...dimF64(TEX_META_HEIGHT), WASM_OP.f64Mul, WASM_OP.f64Floor, WASM_OP.i32TruncF64S], TEX_META_HEIGHT, TEX_META_WRAP_T);
    const fx = fracAxis(0, TEX_META_WIDTH);
    const fy = fracAxis(1, TEX_META_HEIGHT);
    const xa = wrapAxis(fx.i0, TEX_META_WIDTH, TEX_META_WRAP_S);
    const xb = wrapAxis([...fx.i0, ...i32ConstBytes(1), WASM_OP.i32Add], TEX_META_WIDTH, TEX_META_WRAP_S);
    const ya = wrapAxis(fy.i0, TEX_META_HEIGHT, TEX_META_WRAP_T);
    const yb = wrapAxis([...fy.i0, ...i32ConstBytes(1), WASM_OP.i32Add], TEX_META_HEIGHT, TEX_META_WRAP_T);
    const setup: number[] = [
      ...storeComponent(NEAREST_X, "int", 0, nearestX),
      ...storeComponent(NEAREST_Y, "int", 0, nearestY),
      ...storeComponent(XA, "int", 0, xa),
      ...storeComponent(XB, "int", 0, xb),
      ...storeComponent(YA, "int", 0, ya),
      ...storeComponent(YB, "int", 0, yb),
      ...storeComponent(TX, "float", 0, fx.t),
      ...storeComponent(TY, "float", 0, fy.t),
    ];
    if (is3D) {
      const nearestZ = wrapAxis([...uv(2), ...dimF64(TEX_META_DEPTH), WASM_OP.f64Mul, WASM_OP.f64Floor, WASM_OP.i32TruncF64S], TEX_META_DEPTH, TEX_META_WRAP_R);
      const fz = fracAxis(2, TEX_META_DEPTH);
      const za = wrapAxis(fz.i0, TEX_META_DEPTH, TEX_META_WRAP_R);
      const zb = wrapAxis([...fz.i0, ...i32ConstBytes(1), WASM_OP.i32Add], TEX_META_DEPTH, TEX_META_WRAP_R);
      setup.push(
        ...storeComponent(NEAREST_Z, "int", 0, nearestZ),
        ...storeComponent(ZA, "int", 0, za),
        ...storeComponent(ZB, "int", 0, zb),
        ...storeComponent(TZ, "float", 0, fz.t),
      );
    }

    // (y*width + x), or ((z*height + y)*width + x) for 3D.
    function texelIndexBytes(x: number[], y: number[], z: number[] | null): number[] {
      const yx = [...y, ...dimI32(TEX_META_WIDTH), WASM_OP.i32Mul, ...x, WASM_OP.i32Add];
      if (!z) return yx;
      const zy = [...z, ...dimI32(TEX_META_HEIGHT), WASM_OP.i32Mul, ...y, WASM_OP.i32Add];
      return [...zy, ...dimI32(TEX_META_WIDTH), WASM_OP.i32Mul, ...x, WASM_OP.i32Add];
    }

    // One channel's raw, unorm-divided value at an already in-range texel —
    // no bounds handling needed here, unlike `emitTexelFetchStores`,
    // because every index this function ever passes in came from `wrapAxis`.
    function texelChannelRaw(x: number[], y: number[], z: number[] | null, i: number): number[] {
      const dataAddr = dimI32(TEX_META_DATA_ADDR);
      const channels = dimI32(TEX_META_CHANNELS);
      const elemOffset = [...texelIndexBytes(x, y, z), ...channels, WASM_OP.i32Mul, ...i32ConstBytes(i), WASM_OP.i32Add];
      const byteOffset = [...elemOffset, ...i32ConstBytes(3), WASM_OP.i32Shl];
      const addrBytes = [...dataAddr, ...byteOffset, WASM_OP.i32Add];
      const divisor = loadComponent(metaAddr, "float", TEX_META_UNORM_DIVISOR);
      return [...loadDynamic(addrBytes, "float"), ...divisor, WASM_OP.f64Div];
    }
    function texelChannel(x: number[], y: number[], z: number[] | null, i: number): number[] {
      const present = [...i32ConstBytes(i), ...dimI32(TEX_META_CHANNELS), WASM_OP.i32LtS];
      const missingChannelDefault = i === 3 ? f64ConstBytes(1) : f64ConstBytes(0);
      return selectExpr(texelChannelRaw(x, y, z, i), missingChannelDefault, present);
    }

    /**
     * `a + (b-a)*t` — the form `_lerp2`/`_tex2d`/`_tex3d` use
     * (`rmsl-compile-js.ts`) — needs `a`'s own bytecode twice: once as the
     * base, once inside the subtraction. Harmless there (JS just reads the
     * same array slot twice, cheaply); costly here, where `a`/`b` are
     * often a full `texelChannel` fetch (a dynamically-addressed memory
     * load, a channel-present `select`, a divide) rather than a plain
     * value. `a*(1-t) + b*t` is the same value and needs `a` and `b` each
     * exactly once, duplicating only the cheap blend weight `t` instead.
     */
    function lerp(a: number[], b: number[], t: number[]): number[] {
      return [
        ...a, ...f64ConstBytes(1), ...t, WASM_OP.f64Sub, WASM_OP.f64Mul,
        ...b, ...t, WASM_OP.f64Mul,
        WASM_OP.f64Add,
      ];
    }
    function bilinear(xa: number[], xb: number[], ya: number[], yb: number[], z: number[] | null, tx: number[], ty: number[], i: number): number[] {
      const taa = texelChannel(xa, ya, z, i);
      const tba = texelChannel(xb, ya, z, i);
      const tab = texelChannel(xa, yb, z, i);
      const tbb = texelChannel(xb, yb, z, i);
      const lower = lerp(taa, tba, tx);
      const upper = lerp(tab, tbb, tx);
      return lerp(lower, upper, ty);
    }

    // --- Per-channel: reads the setup above back (a fixed-address load,
    // cheap to repeat) instead of recomputing it. ---
    const nx = loadComponent(NEAREST_X, "int", 0), ny = loadComponent(NEAREST_Y, "int", 0);
    const nz = is3D ? loadComponent(NEAREST_Z, "int", 0) : null;
    const xaBytes = loadComponent(XA, "int", 0), xbBytes = loadComponent(XB, "int", 0);
    const yaBytes = loadComponent(YA, "int", 0), ybBytes = loadComponent(YB, "int", 0);
    const txBytes = loadComponent(TX, "float", 0), tyBytes = loadComponent(TY, "float", 0);

    // `magFilter` is a real runtime `if`/`else`, not a `select` like every
    // other choice in this function — deliberately, unlike the rest of
    // this file's usual branchless style. `select` requires both operands
    // already computed, which for the nearest/linear choice means paying
    // for the far more expensive bilinear/trilinear path even when a
    // texture asks for nearest filtering. An `if`/`else` only ever
    // executes the branch actually taken.
    const linearStores: number[] = [];
    const nearestStores: number[] = [];
    for (let i = 0; i < 4; i++) {
      let linearValue: number[];
      if (!is3D) {
        linearValue = bilinear(xaBytes, xbBytes, yaBytes, ybBytes, null, txBytes, tyBytes, i);
      } else {
        const zaBytes = loadComponent(ZA, "int", 0), zbBytes = loadComponent(ZB, "int", 0);
        const tzBytes = loadComponent(TZ, "float", 0);
        const near = bilinear(xaBytes, xbBytes, yaBytes, ybBytes, zaBytes, txBytes, tyBytes, i);
        const far = bilinear(xaBytes, xbBytes, yaBytes, ybBytes, zbBytes, txBytes, tyBytes, i);
        linearValue = lerp(near, far, tzBytes);
      }
      linearStores.push(...storeComponent(addr, "float", i * 8, linearValue));
      nearestStores.push(...storeComponent(addr, "float", i * 8, texelChannel(nx, ny, nz, i)));
    }
    return [
      ...materialize,
      ...setup,
      ...dimI32(TEX_META_FILTER), WASM_OP.if_, WASM_BLOCKTYPE_VOID,
      ...linearStores,
      WASM_OP.else_, ...nearestStores,
      WASM_OP.end,
    ];
  }

  function emitLiteralStores(node: any, addr: number): number[] {
    const kind = elementKindOf(node._t as string);
    const compSize = componentSizeOf(kind);
    const values = node.value as (number | boolean)[];
    const out: number[] = [];
    values.forEach((v, i) => {
      const num = typeof v === "boolean" ? (v ? 1 : 0) : v;
      const bytes = kind === "float" ? f64ConstBytes(num) : i32ConstBytes(num);
      out.push(...storeComponent(addr, kind, i * compSize, bytes));
    });
    return out;
  }

  function emitSwizzleStores(node: any, addr: number): number[] {
    const src = node.params[0];
    const pattern = node.value as string;
    const kind = elementKindOf(node._t as string);
    const compSize = componentSizeOf(kind);
    const out = [...materializeIfNeeded(src)];
    const srcAddr = nodeAddress(src);
    const srcKind = elementKindOf(src._t as string);
    const srcCompSize = componentSizeOf(srcKind);
    [...pattern].forEach((ch, i) => {
      out.push(...storeComponent(addr, kind, i * compSize, loadComponent(srcAddr, srcKind, COMPONENT_INDEX[ch] * srcCompSize)));
    });
    return out;
  }

  /** Componentwise `add`/`sub`/`mul`/`div` on a vector — a scalar operand
   * (`vec3.mul(2.0)`) is broadcast by recomputing its expression per
   * component; an aggregate operand is materialized once, then read by
   * index. */
  function emitComponentwiseStores(node: any, addr: number): number[] {
    const [a, b] = node.params;
    const targetKind = elementKindOf(node._t as string);
    const compSize = componentSizeOf(targetKind);
    const width = componentCountOf(node._t as string);
    const aWidth = componentCountOf(a._t);
    const bWidth = componentCountOf(b._t);
    const out: number[] = [];
    if (aWidth > 1) out.push(...materializeIfNeeded(a));
    if (bWidth > 1) out.push(...materializeIfNeeded(b));
    const aAddr = aWidth > 1 ? nodeAddress(a) : undefined;
    const bAddr = bWidth > 1 ? nodeAddress(b) : undefined;
    let opcode: number;
    if (targetKind === "float") {
      opcode = node.type === "add" ? WASM_OP.f64Add : node.type === "sub" ? WASM_OP.f64Sub
        : node.type === "mul" ? WASM_OP.f64Mul : WASM_OP.f64Div;
    } else if (node.type === "add") opcode = WASM_OP.i32Add;
    else if (node.type === "sub") opcode = WASM_OP.i32Sub;
    else if (node.type === "mul") opcode = WASM_OP.i32Mul;
    else opcode = targetKind === "uint" ? WASM_OP.i32DivU : WASM_OP.i32DivS;
    for (let k = 0; k < width; k++) {
      const aBytes = aWidth > 1 ? loadComponent(aAddr!, targetKind, k * compSize) : walkExpr(a);
      const bBytes = bWidth > 1 ? loadComponent(bAddr!, targetKind, k * compSize) : walkExpr(b);
      out.push(...storeComponent(addr, targetKind, k * compSize, [...aBytes, ...bBytes, opcode]));
    }
    return out;
  }

  /** `mat.mul(mat)` — a real matrix product, square only (matching the
   * JS backend's own limit: `jsMatMul` throws for a non-square operand). A
   * matrix times a *scalar* never reaches here — that's still componentwise
   * (see the "mul" case in `materializeIfNeeded`). */
  function emitMatMatMulStores(node: any, addr: number): number[] {
    const [a, b] = node.params;
    const aType = a._t as string, bType = b._t as string;
    const [cols, rows] = MATRIX_DIMENSIONS[aType];
    if (aType !== bType || cols !== rows) {
      throw new Error(`[RMSL] compileWasmFn: does not yet support non-square or mismatched-shape matrix multiplication ("${aType}" x "${bType}")`);
    }
    const n = cols;
    const out = [...materializeIfNeeded(a), ...materializeIfNeeded(b)];
    const aAddr = nodeAddress(a), bAddr = nodeAddress(b);
    for (let col = 0; col < n; col++) {
      for (let row = 0; row < n; row++) {
        let terms: number[] = [];
        for (let k = 0; k < n; k++) {
          const term = [...loadComponent(aAddr, "float", (k * n + row) * 8), ...loadComponent(bAddr, "float", (col * n + k) * 8), WASM_OP.f64Mul];
          terms = k === 0 ? term : [...terms, ...term, WASM_OP.f64Add];
        }
        out.push(...storeComponent(addr, "float", (col * n + row) * 8, terms));
      }
    }
    return out;
  }

  /** `mat.mul(vec)` — always float, column-major. A vector one component
   * shorter than the matrix's column count is a position with its
   * homogeneous coordinate implied (`mat4 * vec3`) — `node._t` (computed in
   * rmsl-core.ts) already reflects the resulting, possibly-truncated width,
   * so the output row count is read from there rather than re-derived. */
  function emitMatVecMulStores(node: any, addr: number): number[] {
    const [matNode, vecNode] = node.params;
    const [cols, rows] = MATRIX_DIMENSIONS[matNode._t as string];
    const vecWidth = componentCountOf(vecNode._t);
    const outRows = componentCountOf(node._t as string);
    const out = [...materializeIfNeeded(matNode), ...materializeIfNeeded(vecNode)];
    const matAddr = nodeAddress(matNode), vecAddr = nodeAddress(vecNode);
    for (let row = 0; row < outRows; row++) {
      let terms: number[] = [];
      for (let c = 0; c < vecWidth; c++) {
        const term = [...loadComponent(matAddr, "float", (c * rows + row) * 8), ...loadComponent(vecAddr, "float", c * 8), WASM_OP.f64Mul];
        terms = c === 0 ? term : [...terms, ...term, WASM_OP.f64Add];
      }
      if (vecWidth < cols) {
        // The implied homogeneous coordinate is 1, so its term is just the
        // matrix's own entry for that row, added unmultiplied.
        terms = [...terms, ...loadComponent(matAddr, "float", (vecWidth * rows + row) * 8), WASM_OP.f64Add];
      }
      out.push(...storeComponent(addr, "float", row * 8, terms));
    }
    return out;
  }

  /** `cross(a, b)` — vec3 only, matching GLSL/the JS backend. Each of a's
   * and b's three components is used in exactly two of the three outputs;
   * materializing once and reloading by index for each use is cheap enough
   * not to need anything smarter. */
  function emitCrossStores(node: any, addr: number): number[] {
    const [a, b] = node.params;
    const width = componentCountOf(node._t as string);
    if (width !== 3) {
      throw new Error(`[RMSL] compileWasmFn: cross() needs a vec3, got width ${width}`);
    }
    const out = [...materializeIfNeeded(a), ...materializeIfNeeded(b)];
    const aAddr = nodeAddress(a), bAddr = nodeAddress(b);
    const load = (n: number, k: number) => loadComponent(n, "float", k * 8);
    const term = (a0: number[], b0: number[], a1: number[], b1: number[]) => [...a0, ...b0, WASM_OP.f64Mul, ...a1, ...b1, WASM_OP.f64Mul, WASM_OP.f64Sub];
    out.push(...storeComponent(addr, "float", 0, term(load(aAddr, 1), load(bAddr, 2), load(aAddr, 2), load(bAddr, 1))));
    out.push(...storeComponent(addr, "float", 8, term(load(aAddr, 2), load(bAddr, 0), load(aAddr, 0), load(bAddr, 2))));
    out.push(...storeComponent(addr, "float", 16, term(load(aAddr, 0), load(bAddr, 1), load(aAddr, 1), load(bAddr, 0))));
    return out;
  }

  /** `normalize(v)` — `v / length(v)`, or `v` unchanged where the length is
   * zero (matching the JS backend's div-by-zero guard). The length is
   * computed once into the one extra f64 slot `isScratchNode`'s caller
   * reserved right after this node's own output components (see
   * `collect()`), then read back by each component's `select` instead of
   * being recomputed — recomputing a sum-of-`width`-squares per component
   * would cost `O(width^2)`, unlike this file's usual "recompute a value
   * used twice" tradeoff. */
  function emitNormalizeStores(node: any, addr: number): number[] {
    const src = node.params[0];
    const kind = elementKindOf(node._t as string);
    const compSize = componentSizeOf(kind);
    const width = componentCountOf(node._t as string);
    const lengthAddr = addr + width * compSize;
    const out = [...materializeIfNeeded(src)];
    const srcAddr = nodeAddress(src);
    let sumSq: number[] = [];
    for (let k = 0; k < width; k++) {
      const term = [...loadComponent(srcAddr, kind, k * compSize), ...loadComponent(srcAddr, kind, k * compSize), WASM_OP.f64Mul];
      sumSq = k === 0 ? term : [...sumSq, ...term, WASM_OP.f64Add];
    }
    out.push(...storeComponent(lengthAddr, "float", 0, [...sumSq, WASM_OP.f64Sqrt]));
    for (let k = 0; k < width; k++) {
      const srcK = loadComponent(srcAddr, kind, k * compSize);
      const cond = [...loadComponent(lengthAddr, "float", 0), ...f64ConstBytes(0), WASM_OP.f64Gt];
      const whenPositive = [...srcK, ...loadComponent(lengthAddr, "float", 0), WASM_OP.f64Div];
      out.push(...storeComponent(addr, kind, k * compSize, selectExpr(whenPositive, srcK, cond)));
    }
    return out;
  }

  /** `reflect(i, n) = i - 2 * dot(n, i) * n`. Same "one extra scratch slot"
   * treatment as `normalize` for the dot product, reused across all `width`
   * output components. */
  function emitReflectStores(node: any, addr: number): number[] {
    const [i, n] = node.params;
    const kind = elementKindOf(node._t as string);
    const compSize = componentSizeOf(kind);
    const width = componentCountOf(node._t as string);
    const dotAddr = addr + width * compSize;
    const out = [...materializeIfNeeded(i), ...materializeIfNeeded(n)];
    const iAddr = nodeAddress(i), nAddr = nodeAddress(n);
    let dot: number[] = [];
    for (let k = 0; k < width; k++) {
      const term = [...loadComponent(nAddr, kind, k * compSize), ...loadComponent(iAddr, kind, k * compSize), WASM_OP.f64Mul];
      dot = k === 0 ? term : [...dot, ...term, WASM_OP.f64Add];
    }
    out.push(...storeComponent(dotAddr, "float", 0, dot));
    for (let k = 0; k < width; k++) {
      const bytes = [
        ...loadComponent(iAddr, kind, k * compSize),
        ...f64ConstBytes(2),
        ...loadComponent(dotAddr, "float", 0),
        WASM_OP.f64Mul,
        ...loadComponent(nAddr, kind, k * compSize),
        WASM_OP.f64Mul,
        WASM_OP.f64Sub,
      ];
      out.push(...storeComponent(addr, kind, k * compSize, bytes));
    }
    return out;
  }

  /** One scalar component of an aggregate node's value, materializing it
   * first if needed. Only used where a node's components are each read
   * exactly once — anywhere a component is read more than once
   * (`emitConstructStores`, `dot`, ...) materializes once up front instead
   * and calls `loadComponent` directly, to avoid re-materializing per read. */
  function readComponent(node: any, k: number): number[] {
    const out = [...materializeIfNeeded(node)];
    const kind = elementKindOf(node._t as string);
    out.push(...loadComponent(nodeAddress(node), kind, k * componentSizeOf(kind)));
    return out;
  }

  /** `add`/`sub`/`mul` share one opcode across int and uint (two's-complement
   * arithmetic doesn't care about signedness), and one across bool too, since
   * a bool operand only ever reaches these through `select`'s condition —
   * never directly — so only float-vs-not distinguishes the opcode. */
  function binaryArith(node: any, f64op: number, i32op: number): number[] {
    const op = scalarKindOf(node.params[0]._t) === "float" ? f64op : i32op;
    return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), op];
  }

  /** A comparison needs a third, unsigned variant when the operand type is
   * `uint` — `<`/`>`/`<=`/`>=` read a negative int's top bit as a sign, an
   * unsigned one as magnitude, and those disagree. */
  function comparison(node: any, f64op: number, i32sOp: number, i32uOp: number): number[] {
    const kind = scalarKindOf(node.params[0]._t);
    const op = kind === "float" ? f64op : kind === "uint" ? i32uOp : i32sOp;
    return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), op];
  }

  /** `select`'s stack order is `[whenTrue, whenFalse, cond]`; the two
   * branches are plain expression bytes recomputed inline (this backend
   * doesn't yet cache a shared sub-expression in a temp local — see
   * ROADMAP.md), which is only a size/speed cost, never a correctness one,
   * since nothing in this DSL's expression position has a side effect. */
  function selectExpr(whenTrue: number[], whenFalse: number[], cond: number[]): number[] {
    return [...whenTrue, ...whenFalse, ...cond, WASM_OP.select];
  }

  function i32ConstBytes(n: number): number[] {
    return [WASM_OP.i32Const, ...wasmSleb128(n | 0)];
  }
  function f64ConstBytes(n: number): number[] {
    return [WASM_OP.f64Const, ...wasmF64Bytes(n)];
  }

  /** `min`/`max` have a native opcode for float; int and uint have none, so
   * they're built from a comparison and `select`. */
  function minOrMax(a: any, b: any, kind: ScalarKind, pick: "min" | "max"): number[] {
    if (kind === "float") {
      return [...walkExpr(a), ...walkExpr(b), pick === "min" ? WASM_OP.f64Min : WASM_OP.f64Max];
    }
    const cmp = kind === "uint" ? (pick === "min" ? WASM_OP.i32LtU : WASM_OP.i32GtU)
      : (pick === "min" ? WASM_OP.i32LtS : WASM_OP.i32GtS);
    return selectExpr(walkExpr(a), walkExpr(b), [...walkExpr(a), ...walkExpr(b), cmp]);
  }

  /** Expression context: leaves exactly one value (f64, or i32 for a bool or
   * an int/uint) on the stack. */
  function walkExpr(node: any): number[] {
    switch (node.type) {
      case "float": return f64ConstBytes(node.value);
      case "int": case "uint": return i32ConstBytes(node.value);
      case "bool": return i32ConstBytes(node.value ? 1 : 0);
      case "var":
        if (fnParamNames.has(node.value.varName)) {
          return [WASM_OP.localGet, ...wasmUleb128(paramSlotIndex(`param:${node.value.varName}`))];
        }
        return [WASM_OP.localGet, ...wasmUleb128(localSlotIndex(node.value.varName))];
      case "uniform":
        return [WASM_OP.localGet, ...wasmUleb128(paramSlotIndex(`uniform:${node.value.slot}`))];
      case "attribute":
        return [WASM_OP.localGet, ...wasmUleb128(paramSlotIndex(`attribute:${node.value.slot}`))];
      case "varying":
        // Fragment stage: a scalar varying is a WASM function argument, like
        // a scalar uniform/attribute. Vertex stage: it's output-direction,
        // read back (after an earlier `.assign()`) from its own address —
        // there is no WASM argument for it at all.
        if (effectiveStage === "fragment") {
          return [WASM_OP.localGet, ...wasmUleb128(paramSlotIndex(`varying:${node.value.slot}`))];
        }
        return loadComponent(varyingOutputAddress.get(node.value.slot)!, scalarKindOf(node._t), 0);
      case "output":
        return loadComponent(outputAddress.get(node.value.slot)!, scalarKindOf(node._t), 0);
      case "builtinFragDepth":
        return loadComponent(fragDepthAddress!, "float", 0);

      case "add": return binaryArith(node, WASM_OP.f64Add, WASM_OP.i32Add);
      case "sub": return binaryArith(node, WASM_OP.f64Sub, WASM_OP.i32Sub);
      case "mul": return binaryArith(node, WASM_OP.f64Mul, WASM_OP.i32Mul);
      case "div": {
        const kind = scalarKindOf(node.params[0]._t);
        if (kind === "float") return binaryArith(node, WASM_OP.f64Div, WASM_OP.f64Div);
        return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), kind === "uint" ? WASM_OP.i32DivU : WASM_OP.i32DivS];
      }
      case "mod": {
        const kind = scalarKindOf(node.params[0]._t);
        if (kind !== "float") {
          return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), kind === "uint" ? WASM_OP.i32RemU : WASM_OP.i32RemS];
        }
        // Floored, matching GLSL's mod() (JS backend's "mod", not "imod"):
        // a - b * floor(a / b). a and b are each evaluated twice, same
        // tradeoff as selectExpr above.
        const a = node.params[0], b = node.params[1];
        return [
          ...walkExpr(a),
          ...walkExpr(b),
          ...walkExpr(a),
          ...walkExpr(b),
          WASM_OP.f64Div,
          WASM_OP.f64Floor,
          WASM_OP.f64Mul,
          WASM_OP.f64Sub,
        ];
      }
      case "min": return minOrMax(node.params[0], node.params[1], scalarKindOf(node.params[0]._t), "min");
      case "max": return minOrMax(node.params[0], node.params[1], scalarKindOf(node.params[0]._t), "max");

      case "negate": {
        const kind = scalarKindOf(node.params[0]._t);
        if (kind === "float") return [...walkExpr(node.params[0]), WASM_OP.f64Neg];
        return [...i32ConstBytes(0), ...walkExpr(node.params[0]), WASM_OP.i32Sub];
      }
      case "abs": {
        const kind = scalarKindOf(node.params[0]._t);
        if (kind === "float") return [...walkExpr(node.params[0]), WASM_OP.f64Abs];
        const x = node.params[0];
        const negated = [...i32ConstBytes(0), ...walkExpr(x), WASM_OP.i32Sub];
        const isNeg = [...walkExpr(x), ...i32ConstBytes(0), WASM_OP.i32LtS];
        return selectExpr(negated, walkExpr(x), isNeg);
      }
      case "sign": {
        const kind = scalarKindOf(node.params[0]._t);
        const x = node.params[0];
        const zero = kind === "float" ? f64ConstBytes(0) : i32ConstBytes(0);
        const one = kind === "float" ? f64ConstBytes(1) : i32ConstBytes(1);
        const minusOne = kind === "float" ? f64ConstBytes(-1) : i32ConstBytes(-1);
        const gtZero = kind === "float"
          ? [...walkExpr(x), ...f64ConstBytes(0), WASM_OP.f64Gt]
          : [...walkExpr(x), ...i32ConstBytes(0), (kind === "uint" ? WASM_OP.i32GtU : WASM_OP.i32GtS)];
        const ltZero = kind === "float"
          ? [...walkExpr(x), ...f64ConstBytes(0), WASM_OP.f64Lt]
          : [...walkExpr(x), ...i32ConstBytes(0), (kind === "uint" ? WASM_OP.i32LtU : WASM_OP.i32LtS)];
        const positiveOrZero = selectExpr(one, zero, gtZero);
        return selectExpr(minusOne, positiveOrZero, ltZero);
      }
      case "floor": return [...walkExpr(node.params[0]), WASM_OP.f64Floor];
      case "ceil": return [...walkExpr(node.params[0]), WASM_OP.f64Ceil];
      case "trunc": return [...walkExpr(node.params[0]), WASM_OP.f64Trunc];
      case "fract": {
        const x = node.params[0];
        return [...walkExpr(x), ...walkExpr(x), WASM_OP.f64Floor, WASM_OP.f64Sub];
      }
      case "round": {
        // Math.round semantics (round-half-up), not f64.nearest's
        // round-half-to-even, so this is floor(x + 0.5), not that opcode.
        return [...walkExpr(node.params[0]), ...f64ConstBytes(0.5), WASM_OP.f64Add, WASM_OP.f64Floor];
      }
      case "sqrt": return [...walkExpr(node.params[0]), WASM_OP.f64Sqrt];
      case "inverseSqrt": return [...f64ConstBytes(1), ...walkExpr(node.params[0]), WASM_OP.f64Sqrt, WASM_OP.f64Div];
      case "exp2": return [...f64ConstBytes(2), ...walkExpr(node.params[0]), ...callImport("pow")];

      case "sin": case "cos": case "tan":
      case "asin": case "acos": case "atan":
      case "sinh": case "cosh": case "tanh":
      case "asinh": case "acosh": case "atanh":
      case "exp": case "log": case "log2":
        return [...walkExpr(node.params[0]), ...callImport(node.type)];
      case "pow": case "atan2":
        return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), ...callImport(node.type)];

      case "lessThan": return comparison(node, WASM_OP.f64Lt, WASM_OP.i32LtS, WASM_OP.i32LtU);
      case "greaterThan": return comparison(node, WASM_OP.f64Gt, WASM_OP.i32GtS, WASM_OP.i32GtU);
      case "lessThanEqual": return comparison(node, WASM_OP.f64Le, WASM_OP.i32LeS, WASM_OP.i32LeU);
      case "greaterThanEqual": return comparison(node, WASM_OP.f64Ge, WASM_OP.i32GeS, WASM_OP.i32GeU);
      case "equal": return comparison(node, WASM_OP.f64Eq, WASM_OP.i32Eq, WASM_OP.i32Eq);
      case "notEqual": return comparison(node, WASM_OP.f64Ne, WASM_OP.i32Ne, WASM_OP.i32Ne);

      case "and": return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), WASM_OP.i32And];
      case "or": return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), WASM_OP.i32Or];
      case "not": return [...walkExpr(node.params[0]), WASM_OP.i32Eqz];

      case "bitAnd": return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), WASM_OP.i32And];
      case "bitOr": return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), WASM_OP.i32Or];
      case "bitXor": return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), WASM_OP.i32Xor];
      case "bitNot": return [...walkExpr(node.params[0]), ...i32ConstBytes(-1), WASM_OP.i32Xor];
      case "shiftLeft": return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), WASM_OP.i32Shl];
      case "shiftRight": {
        const kind = scalarKindOf(node.params[0]._t);
        return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), kind === "uint" ? WASM_OP.i32ShrU : WASM_OP.i32ShrS];
      }

      case "construct": {
        const targetKind = scalarKindOf(node._t);
        const source = node.params[0];
        const sourceKind = scalarKindOf(source._t);
        const bytes = walkExpr(source);
        if (sourceKind === targetKind) return bytes;
        const sourceIsFloat = sourceKind === "float";
        const targetIsFloat = targetKind === "float";
        if (sourceIsFloat && !targetIsFloat) {
          if (targetKind === "bool") return [...bytes, ...f64ConstBytes(0), WASM_OP.f64Ne];
          return [...bytes, targetKind === "uint" ? WASM_OP.i32TruncF64U : WASM_OP.i32TruncF64S];
        }
        if (!sourceIsFloat && targetIsFloat) {
          return [...bytes, sourceKind === "uint" ? WASM_OP.f64ConvertI32U : WASM_OP.f64ConvertI32S];
        }
        // int <-> uint <-> bool: same i32 bits, except a bool target needs an
        // actual "is this nonzero" test rather than a raw reinterpretation.
        if (targetKind === "bool") return [...bytes, ...i32ConstBytes(0), WASM_OP.i32Ne];
        return bytes;
      }

      case "swizzle": {
        const pattern = node.value as string;
        if (pattern.length !== 1) {
          throw new Error('[RMSL] compileWasmFn: unsupported node type in expression position: "swizzle" (multi-component)');
        }
        return readComponent(node.params[0], COMPONENT_INDEX[pattern]);
      }

      case "dot": {
        const a = node.params[0], b = node.params[1];
        const width = componentCountOf(a._t);
        const pre = [...materializeIfNeeded(a), ...materializeIfNeeded(b)];
        const aAddr = nodeAddress(a);
        const bAddr = nodeAddress(b);
        let acc: number[] = [];
        for (let k = 0; k < width; k++) {
          const term = [...loadComponent(aAddr, "float", k * 8), ...loadComponent(bAddr, "float", k * 8), WASM_OP.f64Mul];
          acc = k === 0 ? term : [...acc, ...term, WASM_OP.f64Add];
        }
        return [...pre, ...acc];
      }
      case "dFdx":
      case "dFdy":
      case "fwidth": {
        assertDerivativesAllowed(node);
        const kind = scalarKindOf(node._t);
        return kind === "float" ? f64ConstBytes(0) : i32ConstBytes(0);
      }
      case "length": {
        const src = node.params[0];
        const width = componentCountOf(src._t);
        const pre = materializeIfNeeded(src);
        const addr = nodeAddress(src);
        let sumSq: number[] = [];
        for (let k = 0; k < width; k++) {
          const term = [...loadComponent(addr, "float", k * 8), ...loadComponent(addr, "float", k * 8), WASM_OP.f64Mul];
          sumSq = k === 0 ? term : [...sumSq, ...term, WASM_OP.f64Add];
        }
        return [...pre, ...sumSq, WASM_OP.f64Sqrt];
      }
      case "distance": {
        const a = node.params[0], b = node.params[1];
        const width = componentCountOf(a._t);
        const pre = [...materializeIfNeeded(a), ...materializeIfNeeded(b)];
        const aAddr = nodeAddress(a);
        const bAddr = nodeAddress(b);
        let sumSq: number[] = [];
        for (let k = 0; k < width; k++) {
          const diff = [...loadComponent(aAddr, "float", k * 8), ...loadComponent(bAddr, "float", k * 8), WASM_OP.f64Sub];
          const term = [...diff, ...diff, WASM_OP.f64Mul];
          sumSq = k === 0 ? term : [...sumSq, ...term, WASM_OP.f64Add];
        }
        return [...pre, ...sumSq, WASM_OP.f64Sqrt];
      }
      default:
        throw new Error(`[RMSL] compileWasmFn: unsupported node type in expression position: "${node.type}"`);
    }
  }

  // The whole function body is wrapped in one outer `block` so `Return`/
  // `Discard` always have somewhere to `br` to (see the `code` assembly
  // below) — its content begins at depth 1, the depth every top-level
  // statement is now compiled at.
  const EXIT_BLOCK_DEPTH = 1;

  // Per-open-loop `br`/`br_if` targets, as the WASM label-nesting depth at
  // which that loop's break/continue block's *content* begins (see
  // `emitLoop`) — pushed on entering a "for"/"while", popped on leaving.
  // `Break`/`Continue` resolve their relative branch index against the top
  // entry; empty means neither is inside a loop.
  const loopStack: { breakDepth: number; continueDepth: number }[] = [];

  /**
   * `for`/`while` share this lowering: a `block` (the `Break` target)
   * wrapping a `loop` (re-tests `cond` each iteration) wrapping a `block`
   * (the `Continue` target) around `body` — the inner block exists so
   * `Continue` skips straight to `update` (still running it) rather than
   * jumping back to `cond` directly, which would skip a `for`'s update
   * clause entirely.
   */
  function emitLoop(initBytes: number[], condNode: any, bodyNode: any, updateNode: any | null, depth: number): number[] {
    const breakDepth = depth + 1;
    const continueDepth = depth + 3;
    loopStack.push({ breakDepth, continueDepth });
    const condBytes = walkExpr(condNode);
    const bodyBytes = walkStmt(bodyNode, continueDepth);
    const updateBytes = updateNode ? walkStmt(updateNode, depth + 2) : [];
    loopStack.pop();
    return [
      ...initBytes,
      WASM_OP.block, WASM_BLOCKTYPE_VOID,
      WASM_OP.loop, WASM_BLOCKTYPE_VOID,
      ...condBytes, WASM_OP.i32Eqz, WASM_OP.brIf, ...wasmUleb128(1),
      WASM_OP.block, WASM_BLOCKTYPE_VOID,
      ...bodyBytes,
      WASM_OP.end,
      ...updateBytes,
      WASM_OP.br, ...wasmUleb128(0),
      WASM_OP.end,
      WASM_OP.end,
    ];
  }

  /** Statement context: leaves no net stack effect. `depth` is the WASM
   * label-nesting depth of the structured blocks `node` itself sits
   * directly inside — needed so `break`/`continue`/`return`/`discard` can
   * compute the relative branch index `br`/`br_if` require. The whole
   * function body is wrapped in one exit block (see the `code` assembly
   * below), so depth starts at 1, not 0. */
  function walkStmt(node: any, depth: number): number[] {
    switch (node.type) {
      case "seq": {
        const list = node.params ?? [];
        if (node._t !== "void") {
          throw new Error("[RMSL] compileWasmFn: a value-producing seq belongs at the root, not in statement position");
        }
        return list.flatMap((s: any) => walkStmt(s, depth));
      }
      case "let":
      case "assign": {
        const target = node.params[0];
        const rhs = node.params[1];

        if (target.type === "swizzle") {
          const { base, pattern } = resolveSwizzleTarget(target);
          const baseName = (base as any).value.varName;
          const baseAddr = fnParamNames.has(baseName) ? paramAddress.get(baseName) : varAddress.get(baseName);
          if (baseAddr === undefined) {
            throw new Error(`[RMSL] compileWasmFn: assign to swizzle of undeclared var "${baseName}"`);
          }
          const kind = elementKindOf((base as any)._t);
          const compSize = componentSizeOf(kind);
          if (pattern.length === 1) {
            return storeComponent(baseAddr, kind, COMPONENT_INDEX[pattern] * compSize, walkExpr(rhs));
          }
          const out = [...materializeIfNeeded(rhs)];
          const rhsAddr = nodeAddress(rhs);
          [...pattern].forEach((ch, i) => {
            out.push(...storeComponent(baseAddr, kind, COMPONENT_INDEX[ch] * compSize, loadComponent(rhsAddr, kind, i * compSize)));
          });
          return out;
        }

        // Phase 5 output direction: `output()`/a vertex-stage `varying()`/
        // `builtinPosition()`/`builtinFragDepth()` are all assign targets
        // with their own reserved address (allocated in `collect()`) rather
        // than a `"var"` node's `value.varName` — resolve the destination
        // address and type once, then the aggregate-vs-scalar copy below is
        // identical either way.
        let targetType: string;
        let destAddr: number;
        if (target.type === "output") {
          targetType = target._t as string;
          destAddr = outputAddress.get(target.value.slot)!;
        } else if (target.type === "varying") {
          if (effectiveStage !== "vertex") {
            throw new Error("[RMSL] compileWasmFn: varying() cannot be assigned to outside a vertex stage");
          }
          targetType = target._t as string;
          destAddr = varyingOutputAddress.get(target.value.slot)!;
        } else if (target.type === "builtinPosition") {
          targetType = "vec4";
          destAddr = positionAddress!;
        } else if (target.type === "builtinFragDepth") {
          targetType = "float";
          destAddr = fragDepthAddress!;
        } else {
          targetType = target._t as string;
          const varName = target.value.varName;
          if (!isAggregate(targetType)) {
            return [...walkExpr(rhs), WASM_OP.localSet, ...wasmUleb128(localSlotIndex(varName))];
          }
          destAddr = fnParamNames.has(varName) ? paramAddress.get(varName)! : varAddress.get(varName)!;
        }

        if (!isAggregate(targetType)) {
          // A scalar output-like target (e.g. builtinFragDepth, or a plain
          // float output()) — write its value directly, no address/
          // materialize machinery needed for a single scalar.
          return storeComponent(destAddr, elementKindOf(targetType), 0, walkExpr(rhs));
        }
        const kind = elementKindOf(targetType);
        const compSize = componentSizeOf(kind);
        const width = componentCountOf(targetType);
        const out = [...materializeIfNeeded(rhs)];
        const rhsAddr = nodeAddress(rhs);
        for (let k = 0; k < width; k++) {
          out.push(...storeComponent(destAddr, kind, k * compSize, loadComponent(rhsAddr, kind, k * compSize)));
        }
        return out;
      }
      case "if": {
        const cond = walkExpr(node.params[0]);
        // "if"/"else" are themselves structured WASM blocks and occupy a
        // label index, so anything nested inside — a "break"/"continue"/
        // "return" reached via `if (x) { Break(); }` inside a loop — must
        // count this level too.
        const thenBytes = walkStmt(node.params[1], depth + 1);
        const elseNode = node.params[2];
        return [
          ...cond, WASM_OP.if_, WASM_BLOCKTYPE_VOID,
          ...thenBytes,
          ...(elseNode ? [WASM_OP.else_, ...walkStmt(elseNode, depth + 1)] : []),
          WASM_OP.end,
        ];
      }
      case "for": {
        const [initNode, condNode, updateNode, bodyNode] = node.params;
        return emitLoop(walkStmt(initNode, depth), condNode, bodyNode, updateNode, depth);
      }
      case "while": {
        const [condNode, bodyNode] = node.params;
        return emitLoop([], condNode, bodyNode, null, depth);
      }
      case "break": {
        const top = loopStack[loopStack.length - 1];
        if (!top) throw new Error('[RMSL] compileWasmFn: "Break" outside a loop');
        return [WASM_OP.br, ...wasmUleb128(depth - top.breakDepth)];
      }
      case "continue": {
        const top = loopStack[loopStack.length - 1];
        if (!top) throw new Error('[RMSL] compileWasmFn: "Continue" outside a loop');
        return [WASM_OP.br, ...wasmUleb128(depth - top.continueDepth)];
      }
      case "return":
      case "discard": {
        // Neither carries a value in this DSL (there is no `Return(value)`
        // overload). A plain scalar-returning function's exit block still
        // expects one value on exit — a zero/false sentinel of the result
        // kind, matching what ROADMAP.md already anticipated for
        // `Discard`. A stage-mode function's exit block is void instead:
        // whatever result it has already lives in memory (via `assign()`s
        // that ran before this point, if any), so there's nothing to leave
        // on the stack at all. `Discard`'s real "no fragment output"
        // meaning still has no representation yet; it compiles identically
        // to `Return()` either way.
        const sentinel = needsResult ? [] : (resultKind === "float" ? f64ConstBytes(0) : i32ConstBytes(0));
        return [...sentinel, WASM_OP.br, ...wasmUleb128(depth - EXIT_BLOCK_DEPTH)];
      }
      default:
        throw new Error(`[RMSL] compileWasmFn: unsupported node type in statement position: "${node.type}"`);
    }
  }

  /** The root's own final value: `!needsResult` leaves it on the WASM
   * stack exactly as before (becomes the function's one return value).
   * `needsResult` instead writes it into `valueAddress` (skipped
   * entirely for a `"void"` root — a stage program with no value beyond
   * whatever it wrote to `output()`/etc.) and leaves nothing on the
   * stack, matching the function's now-zero-result signature below. */
  function finalValueBytes(valueNode: any): number[] {
    if (!needsResult) return walkExpr(valueNode);
    if (valueNode._t === "void" || valueAddress === undefined) return [];
    if (isAggregate(valueNode._t as string)) {
      const kind = elementKindOf(valueNode._t as string);
      const compSize = componentSizeOf(kind);
      const width = componentCountOf(valueNode._t as string);
      const out = [...materializeIfNeeded(valueNode)];
      const srcAddr = nodeAddress(valueNode);
      for (let k = 0; k < width; k++) {
        out.push(...storeComponent(valueAddress, kind, k * compSize, loadComponent(srcAddr, kind, k * compSize)));
      }
      return out;
    }
    return storeComponent(valueAddress, scalarKindOf(valueNode._t as string), 0, walkExpr(valueNode));
  }

  // A program built with `Fn(() => { ...; return x; })()` is a "seq" node of
  // [...priorStatements, returnValue] (see the Fn implementation in
  // rmsl-core.ts); one built as a bare expression (no Fn wrapper, no
  // statements) is just that expression. Both are valid roots here.
  // Top-level statements compile at EXIT_BLOCK_DEPTH (1), since the whole
  // body is wrapped in the one exit block "return"/"discard" branch to.
  const bodyBytes = root.type === "seq"
    ? [...(root.params.slice(0, -1) as any[]).flatMap((s: any) => walkStmt(s, EXIT_BLOCK_DEPTH)), ...finalValueBytes(root.params[root.params.length - 1])]
    : finalValueBytes(root);
  // Wrapping unconditionally (rather than only when "return"/"discard"
  // appear) costs 3 bytes and is a no-op when neither is used — the exact
  // same bytes run inside a block nothing branches out of — so there's one
  // code path here, not two. `needsResult` makes the block (and the
  // function around it) void instead of single-result.
  const exitBlockType = needsResult ? WASM_BLOCKTYPE_VOID : wasmTypeOf(resultKind);
  const code = [WASM_OP.block, exitBlockType, ...bodyBytes, WASM_OP.end];

  // --- assemble the module: type section (main's signature, plus one shared
  // signature per import arity actually used), import section, function
  // section, memory section, export section (function + memory), code
  // section. ---
  const typeEntries: number[][] = [];
  let unaryImportType: number | null = null;
  let binaryImportType: number | null = null;
  function unaryImportTypeIdx(): number {
    if (unaryImportType === null) {
      unaryImportType = typeEntries.length;
      typeEntries.push([WASM_FUNC, ...wasmVec([[WASM_F64]]), ...wasmVec([[WASM_F64]])]);
    }
    return unaryImportType;
  }
  function binaryImportTypeIdx(): number {
    if (binaryImportType === null) {
      binaryImportType = typeEntries.length;
      typeEntries.push([WASM_FUNC, ...wasmVec([[WASM_F64], [WASM_F64]]), ...wasmVec([[WASM_F64]])]);
    }
    return binaryImportType;
  }
  const importEntries: number[][] = importNames.map((name) => {
    const typeIdx = MATH_BINARY_IMPORTS.has(name) ? binaryImportTypeIdx() : unaryImportTypeIdx();
    return [...wasmStrBytes("math"), ...wasmStrBytes(name), 0x00, ...wasmUleb128(typeIdx)];
  });

  const paramTypes = params.map(p => [wasmTypeOf(scalarKindOf(p.shaderType))]);
  // `needsResult`: the function's own value (if any) and everything else it
  // produces all live in memory, read back after the call — the exported
  // function itself declares zero results, not one.
  const resultTypes = needsResult ? [] : [[wasmTypeOf(resultKind)]];
  const mainTypeIdx = typeEntries.length;
  typeEntries.push([WASM_FUNC, ...wasmVec(paramTypes), ...wasmVec(resultTypes)]);
  // Function indices: imports occupy 0..importNames.length-1, so `main` —
  // the first locally-defined function — is at `importNames.length`,
  // exactly the index the export section below already uses for it.
  const mainFuncIndex = importNames.length;

  /**
   * A second exported function, `"draw"`, sharing `main`'s own compiled
   * body via a real WASM `call` rather than a separate compile mode —
   * `compileWasm` exposes it as `.draw(ctx, width, height)` on the
   * callable it returns, decided per call, not baked in at compile time.
   * Only built when there is a per-pixel value to render at all
   * (`root._t !== "void"` — a pure `output()`-writing stage program has
   * nothing for `draw` to put anywhere).
   *
   * `draw`'s own signature is `main`'s scalar params, followed by
   * `width`/`height` (both runtime `i32` values, not compile-time
   * constants — an image's dimensions are picked per call, the same way
   * any other input is). Its body is a `y`/`x` loop: write this
   * iteration's pixel coordinate into `fragCoordAddress` (skipped
   * entirely when the compiled program never calls `fragCoord()` at
   * all — nothing needs it), `call` `main` with the same scalar
   * arguments `draw` itself received, and copy the result — read
   * straight off `main`'s own WASM return value when `!needsResult`
   * (a plain scalar), or from `valueAddress` in memory when `needsResult`
   * (anything wider, or a stage program) — into a growable output
   * buffer, one dynamic-address store per component (`storeDynamic`, the
   * write-side counterpart of the texture heap's `loadDynamic`).
   *
   * The buffer's base address is a *third* runtime argument (after
   * `width`/`height`), not a compile-time constant — `compileWasm`
   * computes it fresh every `.draw()` call as `textureHeapBase` plus
   * however many bytes the current call's textures actually occupy (0 for
   * a function using no textures, recovering exactly the "starts right
   * after every other compile-time allocation" placement this had before
   * texture support needed to share the same space), so a function using
   * *both* a texture uniform and `.draw()` together places the draw
   * buffer right after wherever that call's texture heap actually ends,
   * rather than the two colliding at the same fixed address.
   */
  let drawTypeIdx: number | undefined;
  let drawFuncBody: number[] | undefined;
  const drawComponentCount = root._t === "void" ? 0 : componentCountOf(root._t as string);
  const drawComponentKind: ScalarKind = root._t === "void" ? "float" : (isAggregate(root._t as string) ? elementKindOf(root._t as string) : scalarKindOf(root._t as string));
  if (root._t !== "void") {
    const drawFuncIndex = mainFuncIndex + 1;
    const widthIdx = params.length;
    const heightIdx = params.length + 1;
    const bufferBaseIdx = params.length + 2;
    const xIdx = params.length + 3;
    const yIdx = params.length + 4;
    const getX = [WASM_OP.localGet, ...wasmUleb128(xIdx)];
    const getY = [WASM_OP.localGet, ...wasmUleb128(yIdx)];
    const compSize = componentSizeOf(drawComponentKind);
    const passThroughArgs = params.map((_, i) => [WASM_OP.localGet, ...wasmUleb128(i)]).flat();
    const callMain = [...passThroughArgs, WASM_OP.call, ...wasmUleb128(mainFuncIndex)];
    const writeFragCoord = fragCoordAddress === undefined ? [] : [
      ...storeComponent(fragCoordAddress, "float", 0, [...getX, WASM_OP.f64ConvertI32S, ...f64ConstBytes(0.5), WASM_OP.f64Add]),
      ...storeComponent(fragCoordAddress, "float", 8, [...getY, WASM_OP.f64ConvertI32S, ...f64ConstBytes(0.5), WASM_OP.f64Add]),
    ];
    // Byte offset of this pixel's first component within the draw buffer:
    // (y*width + x) * componentCount * componentSize — `width` is
    // `draw`'s own runtime argument now, not a compile-time constant.
    const pixelByteOffset = [
      ...getY, WASM_OP.localGet, ...wasmUleb128(widthIdx), WASM_OP.i32Mul, ...getX, WASM_OP.i32Add,
      ...i32ConstBytes(drawComponentCount * compSize), WASM_OP.i32Mul,
    ];
    const destAddr = (k: number) => [WASM_OP.localGet, ...wasmUleb128(bufferBaseIdx), ...pixelByteOffset, WASM_OP.i32Add, ...i32ConstBytes(k * compSize), WASM_OP.i32Add];
    const copyResult: number[] = needsResult
      ? [...callMain, ...Array.from({ length: drawComponentCount }, (_, k) =>
        storeDynamic(destAddr(k), drawComponentKind, loadComponent(valueAddress!, drawComponentKind, k * compSize))).flat()]
      : storeDynamic(destAddr(0), drawComponentKind, callMain); // always exactly 1 component here
    const perPixel = [...writeFragCoord, ...copyResult];
    const innerLoop = [ // x: 0..width
      WASM_OP.block, WASM_BLOCKTYPE_VOID,
      WASM_OP.loop, WASM_BLOCKTYPE_VOID,
      ...getX, WASM_OP.localGet, ...wasmUleb128(widthIdx), WASM_OP.i32GeS, WASM_OP.brIf, ...wasmUleb128(1),
      ...perPixel,
      ...getX, ...i32ConstBytes(1), WASM_OP.i32Add, WASM_OP.localSet, ...wasmUleb128(xIdx),
      WASM_OP.br, ...wasmUleb128(0),
      WASM_OP.end,
      WASM_OP.end,
    ];
    const outerLoop = [ // y: 0..height
      WASM_OP.block, WASM_BLOCKTYPE_VOID,
      WASM_OP.loop, WASM_BLOCKTYPE_VOID,
      ...getY, WASM_OP.localGet, ...wasmUleb128(heightIdx), WASM_OP.i32GeS, WASM_OP.brIf, ...wasmUleb128(1),
      ...i32ConstBytes(0), WASM_OP.localSet, ...wasmUleb128(xIdx),
      ...innerLoop,
      ...getY, ...i32ConstBytes(1), WASM_OP.i32Add, WASM_OP.localSet, ...wasmUleb128(yIdx),
      WASM_OP.br, ...wasmUleb128(0),
      WASM_OP.end,
      WASM_OP.end,
    ];
    const drawCode = [...i32ConstBytes(0), WASM_OP.localSet, ...wasmUleb128(yIdx), ...outerLoop];
    const drawLocalsDecl = wasmVec([[...wasmUleb128(1), WASM_I32], [...wasmUleb128(1), WASM_I32]]);
    drawFuncBody = [...drawLocalsDecl, ...drawCode, WASM_OP.end];
    drawTypeIdx = typeEntries.length;
    typeEntries.push([WASM_FUNC, ...wasmVec([...paramTypes, [WASM_I32], [WASM_I32], [WASM_I32]]), ...wasmVec([])]);
  }

  const typeSection = wasmSection(1, wasmVec(typeEntries));
  const importSection = importEntries.length > 0 ? wasmSection(2, wasmVec(importEntries)) : [];
  const funcSection = wasmSection(3, wasmVec(drawTypeIdx === undefined ? [[mainTypeIdx]] : [[mainTypeIdx], [drawTypeIdx]]));
  const memoryPages = Math.max(1, Math.ceil(memCursor / 65536));
  const memorySection = wasmSection(5, wasmVec([[0x00, ...wasmUleb128(memoryPages)]]));
  const nameBytes = wasmStrBytes(options.name);
  const exportEntries = [
    [...nameBytes, 0x00, ...wasmUleb128(mainFuncIndex)],
    [...wasmStrBytes("memory"), 0x02, ...wasmUleb128(0)],
  ];
  if (drawTypeIdx !== undefined) exportEntries.push([...wasmStrBytes("draw"), 0x00, ...wasmUleb128(mainFuncIndex + 1)]);
  const exportSection = wasmSection(7, wasmVec(exportEntries));
  // One group per local rather than run-length-compressing consecutive
  // same-type locals — larger than it needs to be, but every group is
  // independently correct, and there's no shared-type run to get wrong.
  const localsDecl = wasmVec(localSlots.map(name => [...wasmUleb128(1), wasmTypeOf(localType.get(name)!)]));
  const funcBody = [...localsDecl, ...code, WASM_OP.end];
  const codeEntries = [[...wasmUleb128(funcBody.length), ...funcBody]];
  if (drawFuncBody !== undefined) codeEntries.push([...wasmUleb128(drawFuncBody.length), ...drawFuncBody]);
  const codeSection = wasmSection(10, wasmVec(codeEntries));

  const bytes = new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, // "\0asm"
    0x01, 0x00, 0x00, 0x00, // version 1
    ...typeSection,
    ...importSection,
    ...funcSection,
    ...memorySection,
    ...exportSection,
    ...codeSection,
  ]);

  return {
    bytes, params: [...params, ...memoryParams], resultType: root._t, textureHeapBase: memCursor,
    draw: drawTypeIdx === undefined ? undefined : { componentCount: drawComponentCount, kind: drawComponentKind },
  };
}

/** Write an aggregate value's components into `view` at `address`, using
 * `shaderType` to pick the storage kind/width — the JS-side half of the
 * linear-memory design, run before every call since uniform/param values
 * can change between calls. */
function writeAggregateToMemory(view: DataView, address: number, shaderType: ShaderType, value: any, narrow?: boolean): void {
  const kind = elementKindOf(shaderType);
  // A `narrow` (GPU-shaped) uniform's float component is WGSL's 4-byte
  // f32, not this backend's usual 8-byte f64 — real, lossy rounding of the
  // f64 JS number, same as any real GPU uniform buffer would apply to this
  // exact value. int/uint/bool are already 4 bytes either way.
  const compSize = narrow && kind === "float" ? 4 : componentSizeOf(kind);
  const arr = value as ArrayLike<number | boolean>;
  for (let i = 0; i < arr.length; i++) {
    const raw = arr[i];
    const num = typeof raw === "boolean" ? (raw ? 1 : 0) : (raw as number);
    if (kind === "float") {
      if (narrow) view.setFloat32(address + i * compSize, num, true);
      else view.setFloat64(address + i * compSize, num, true);
    } else {
      view.setInt32(address + i * compSize, num, true);
    }
  }
}

/** The mirror image of `writeAggregateToMemory`: read an aggregate value's
 * components back out, converting a `bool`-kind component back to a real
 * boolean and a `uint`-kind one back to its unsigned reading — the
 * output-direction half of Phase 5, read after every call instead of
 * written before it. */
function readAggregateFromMemory(view: DataView, address: number, shaderType: ShaderType): (number | boolean)[] {
  const kind = elementKindOf(shaderType);
  const compSize = componentSizeOf(kind);
  const width = componentCountOf(shaderType);
  const out: (number | boolean)[] = [];
  for (let i = 0; i < width; i++) {
    if (kind === "float") {
      out.push(view.getFloat64(address + i * compSize, true));
    } else {
      const raw = view.getInt32(address + i * compSize, true);
      out.push(kind === "bool" ? raw !== 0 : kind === "uint" ? raw >>> 0 : raw);
    }
  }
  return out;
}

/** A plain scalar's own kind (`scalarKindOf`, not `elementKindOf` —
 * `elementKindOf` is only correct for a genuine aggregate type's prefix,
 * and would misread e.g. a bare `"int"` as float-width). */
function readScalarFromMemory(view: DataView, address: number, shaderType: ShaderType): number | boolean {
  const kind = scalarKindOf(shaderType);
  if (kind === "float") return view.getFloat64(address, true);
  const raw = view.getInt32(address, true);
  if (kind === "bool") return raw !== 0;
  return kind === "uint" ? raw >>> 0 : raw;
}

function readValueFromMemory(view: DataView, address: number, shaderType: ShaderType): unknown {
  return isAggregate(shaderType) ? readAggregateFromMemory(view, address, shaderType) : readScalarFromMemory(view, address, shaderType);
}

const WRAP_MODE_CODE: Record<JsTextureWrap, number> = { clamp: 0, repeat: 1, mirror: 2 };

/** Write one texture uniform's metadata block and pixel data into `view` at
 * `heapAddr` — the linear-memory counterpart of `writeAggregateToMemory`,
 * run fresh before every call since a texture's shape and data can change
 * call to call, just like a uniform's value can. Defaults mirror
 * `compileJS`'s own exactly (`_chan`/`_unorm`, `rmsl-compile-js.ts`):
 * channels default to 4, filter/wrap default to nearest/clamp, and the
 * unorm divisor is 255 only for `Uint8Array`/`Uint8ClampedArray` data. */
function writeTextureToMemory(view: DataView, metaAddr: number, heapAddr: number, tex: JsTextureData): void {
  const channels = tex.channels ?? 4;
  const depth = tex.depth ?? 0;
  const isByteData = tex.data instanceof Uint8Array || tex.data instanceof Uint8ClampedArray;
  view.setFloat64(metaAddr + TEX_META_UNORM_DIVISOR, isByteData ? 255 : 1, true);
  view.setInt32(metaAddr + TEX_META_DATA_ADDR, heapAddr, true);
  view.setInt32(metaAddr + TEX_META_WIDTH, tex.width, true);
  view.setInt32(metaAddr + TEX_META_HEIGHT, tex.height, true);
  view.setInt32(metaAddr + TEX_META_DEPTH, depth, true);
  view.setInt32(metaAddr + TEX_META_CHANNELS, channels, true);
  view.setInt32(metaAddr + TEX_META_FILTER, tex.magFilter === "linear" ? 1 : 0, true);
  view.setInt32(metaAddr + TEX_META_WRAP_S, WRAP_MODE_CODE[tex.wrapS ?? "clamp"], true);
  view.setInt32(metaAddr + TEX_META_WRAP_T, WRAP_MODE_CODE[tex.wrapT ?? "clamp"], true);
  view.setInt32(metaAddr + TEX_META_WRAP_R, WRAP_MODE_CODE[tex.wrapR ?? "clamp"], true);
  const count = tex.width * tex.height * (depth || 1) * channels;
  for (let i = 0; i < count; i++) {
    view.setFloat64(heapAddr + i * 8, tex.data[i] as number, true);
  }
}

/** Total bytes one texture's pixel data occupies in the heap, stored as f64
 * per component regardless of its own source `TypedArray`'s width —
 * matches this backend's existing "float is f64" convention and keeps
 * sampling arithmetic free of any per-texture width bookkeeping. */
function textureByteSize(tex: JsTextureData): number {
  return tex.width * tex.height * (tex.depth || 1) * (tex.channels ?? 4) * 8;
}

/**
 * The callable `compileWasm` returns: `(ctx) => number | boolean` for the
 * subset of the DSL this backend covers so far, matching `compileJS`'s
 * call signature (a `JsShaderContext`'s `params`/`uniforms` in, a scalar
 * out) — or a `JsShaderResult` for a program that uses `output()`/a
 * vertex `varying()`/`builtinPosition()`/`builtinFragDepth()`, exactly
 * matching what `compileJS` returns for the same program. `.draw()` is
 * always present alongside it: render a `width x height` grid in one call
 * instead of one call per pixel, sharing the exact same compiled body —
 * see `CompiledWasm.draw`'s doc comment (`compileWasmFn`) for the design.
 */
export type WasmCallable = ((ctx: JsShaderContext) => number | boolean | JsShaderResult) & {
  draw(ctx: JsShaderContext, width: number, height: number): Float64Array | Int32Array | Uint32Array;
};

/**
 * Compile an Fn to a `WasmCallable` (see its own doc comment).
 */
export function compileWasm(
  fn: (...args: any[]) => Node<ShaderType>,
  options: CompileWasmFnOptions,
): WasmCallable {
  const { bytes, params, resultType, textureHeapBase, draw } = compileWasmFn(fn, options);
  // A module that imports nothing ignores an unused "math" namespace, so
  // this is passed unconditionally rather than only when needed. `Math`'s
  // own methods have the same names, so it's handed over directly.
  const instance = new WebAssembly.Instance(new WebAssembly.Module(bytes.buffer as ArrayBuffer), { math: Math as unknown as WebAssembly.ModuleImports });
  const wasmMain = instance.exports[options.name] as (...args: number[]) => number;
  const wasmDraw = draw ? (instance.exports.draw as (...args: number[]) => void) : undefined;
  const memory = instance.exports.memory as WebAssembly.Memory;
  // Reassigned (not `const`) because growing `memory` for texture data or
  // a `.draw()` output buffer (below) detaches the buffer this `DataView`
  // was built on.
  let view = new DataView(memory.buffer);
  // Fixed for this compiled function — never varies call to call — so
  // computed once rather than re-scanning `params` on every call.
  const outputParams = params.filter((p): p is Extract<WasmParam, { kind: "outputMemory" | "varyingOutputMemory" | "positionMemory" | "fragDepthMemory" | "valueMemory" }> =>
    p.kind === "outputMemory" || p.kind === "varyingOutputMemory" || p.kind === "positionMemory" || p.kind === "fragDepthMemory" || p.kind === "valueMemory");
  const textureParams = params.filter((p): p is Extract<WasmParam, { kind: "textureMemory" }> => p.kind === "textureMemory");
  // Per-slot cache for the texture heap below: which JsTextureData object
  // (by reference) currently occupies each slot's region, and the byte
  // size that layout was packed for. `null` forces the first call to pack
  // and write everything, the same as if this cache didn't exist.
  const lastTexture: (JsTextureData | undefined)[] = new Array(textureParams.length);
  let lastSizes: number[] | null = null;

  /** Shared between a plain call and `.draw()`: write every uniform/param/
   * texture this compiled function needs into linear memory (or into the
   * scalar `args` list a WASM call itself takes), growing `memory` first
   * if a texture needs more room than it currently has. Also reports
   * where the texture heap actually ends for this call — `textureHeapBase`
   * when there are no textures at all — which `.draw()` uses as the base
   * for its own output buffer, so the two never collide even though
   * neither's size is known until call time. */
  function marshalInputs(ctx: JsShaderContext): { args: number[]; textureHeapEnd: number } {
    // Texture data is call-time-sized, unlike every other param kind here,
    // so it's packed into a growable heap (starting at `textureHeapBase`)
    // rather than at a compile-time-fixed address. A program with no
    // texture uniforms never touches any of this — `textureParams` is
    // empty, `memory` never grows, byte-for-byte identical to before this
    // existed.
    let textureHeapEnd = textureHeapBase;
    if (textureParams.length > 0) {
      const textures = textureParams.map(p => (ctx.textures as any)?.[p.slot] as JsTextureData);
      const sizes = textures.map(textureByteSize);
      const heapOffsets: number[] = [];
      let heapCursor = textureHeapBase;
      for (const size of sizes) {
        heapOffsets.push(heapCursor);
        heapCursor += size;
      }
      textureHeapEnd = heapCursor;
      // A texture's *placement* depends on every earlier slot's current
      // size (they're packed back to back), so a size change anywhere
      // forces every slot to be rewritten at its (possibly new) offset —
      // not just the slot whose size actually changed. This only happens
      // the first call, and again only if some texture's dimensions
      // change call to call, which real usage (bind once, sample every
      // frame with different coordinates) does not do.
      const needsRepack = lastSizes === null || sizes.some((s, i) => s !== lastSizes![i]);
      if (needsRepack && heapCursor > memory.buffer.byteLength) {
        memory.grow(Math.ceil((heapCursor - memory.buffer.byteLength) / 65536));
        view = new DataView(memory.buffer);
      }
      textureParams.forEach((p, i) => {
        // A slot whose texture object is the exact same reference as last
        // call needs no work at all — its metadata and pixel bytes in
        // linear memory are still exactly what they were. This is the
        // whole point of the cache: the realistic pattern (bind a texture
        // once, call the compiled function repeatedly with different
        // coordinates) skips the entire copy on every call but the first.
        if (!needsRepack && textures[i] === lastTexture[i]) return;
        writeTextureToMemory(view, p.metadataAddress, heapOffsets[i], textures[i]);
        lastTexture[i] = textures[i];
      });
      lastSizes = sizes;
    }
    const args: number[] = [];
    for (const p of params) {
      switch (p.kind) {
        case "textureMemory":
          break; // written above, before this loop, not per-argument here
        case "param":
          args.push((ctx.params as any)?.[p.name] as number);
          break;
        case "uniform":
          args.push((ctx.uniforms as any)?.[p.slot] as number);
          break;
        case "attribute":
          args.push((ctx.attributes as any)?.[p.slot] as number);
          break;
        case "varying":
          args.push((ctx.varyings as any)?.[p.slot] as number);
          break;
        case "paramMemory":
          writeAggregateToMemory(view, p.address, p.shaderType, (ctx.params as any)?.[p.name]);
          break;
        case "uniformMemory":
          writeAggregateToMemory(view, p.address, p.shaderType, (ctx.uniforms as any)?.[p.slot], p.narrow);
          break;
        case "attributeMemory":
          writeAggregateToMemory(view, p.address, p.shaderType, (ctx.attributes as any)?.[p.slot]);
          break;
        case "varyingMemory":
          writeAggregateToMemory(view, p.address, p.shaderType, (ctx.varyings as any)?.[p.slot]);
          break;
        case "fragCoordMemory":
          // Not written for a `.draw()` call — `draw`'s own loop overwrites
          // this same address fresh every pixel regardless, so there is
          // nothing for this line to usefully do there. Harmless either
          // way (it would just be overwritten immediately), but `.draw()`
          // calls `wasmDraw` directly rather than through this function's
          // ordinary post-`marshalInputs` path, never reaching this case
          // with a meaningfully different `ctx.fragCoord` per pixel to
          // begin with.
          writeAggregateToMemory(view, p.address, "vec2", ctx.fragCoord ?? [0, 0]);
          break;
      }
    }
    return { args, textureHeapEnd };
  }

  const callable = ((ctx: JsShaderContext): number | boolean | JsShaderResult => {
    const { args } = marshalInputs(ctx);
    const result = wasmMain(...args);
    if (outputParams.length === 0) {
      // The JS/WASM call boundary always surfaces an i32 return as a signed
      // number; a "uint" result above 2^31-1 needs reinterpreting as
      // unsigned, the same way ctx.uniforms/ctx.params values are read as
      // unsigned going in (JS numbers don't distinguish, so no equivalent
      // step is needed there — only coming back out through a fixed-width
      // return does).
      if (resultType === "bool") return result !== 0;
      if (resultType === "uint") return result >>> 0;
      return result;
    }
    // `needsResult` mode: the function declared zero WASM results — its
    // value (if any) and everything else it produced all live in memory,
    // read back here into exactly the shape `compileJS` returns for the
    // same program.
    const shaderResult: JsShaderResult = {};
    for (const p of outputParams) {
      switch (p.kind) {
        case "outputMemory":
          (shaderResult.outputs ??= {})[p.slot] = readValueFromMemory(view, p.address, p.shaderType);
          break;
        case "varyingOutputMemory":
          (shaderResult.varyings ??= {})[p.slot] = readValueFromMemory(view, p.address, p.shaderType);
          break;
        case "positionMemory":
          shaderResult.position = readAggregateFromMemory(view, p.address, "vec4") as number[];
          break;
        case "fragDepthMemory":
          shaderResult.fragDepth = view.getFloat64(p.address, true);
          break;
        case "valueMemory":
          shaderResult.value = readValueFromMemory(view, p.address, p.shaderType);
          break;
      }
    }
    return shaderResult;
  }) as WasmCallable;

  callable.draw = (ctx: JsShaderContext, width: number, height: number): Float64Array | Int32Array | Uint32Array => {
    if (!draw || !wasmDraw) {
      throw new Error("[RMSL] compileWasm: this function produces no value to render — draw() needs a non-\"void\" result.");
    }
    const { args, textureHeapEnd } = marshalInputs(ctx);
    // The draw buffer starts right after wherever this call's texture heap
    // actually ends — `textureHeapBase` itself when there are no textures
    // at all, recovering exactly the placement this had before texture
    // support needed to share the same space. Computed fresh every call,
    // same as the texture heap's own placement already is, so the two
    // never collide regardless of how either one's size changes call to
    // call. Rounded up to a multiple of 8: nothing about this backend's
    // own compile-time bump allocator keeps addresses aligned (a texture's
    // 44-byte metadata block is the concrete case that doesn't), but
    // `Float64Array`'s constructor requires an 8-byte-aligned offset, and
    // rounding up here is the one place that needs to know or care —
    // `draw`'s own bytecode has no alignment requirement of its own, so it
    // takes whatever value this computes exactly as given.
    const bufferBase = Math.ceil(textureHeapEnd / 8) * 8;
    const pixelCount = width * height * draw.componentCount;
    const neededBytes = bufferBase + pixelCount * componentSizeOf(draw.kind);
    if (neededBytes > memory.buffer.byteLength) {
      memory.grow(Math.ceil((neededBytes - memory.buffer.byteLength) / 65536));
      view = new DataView(memory.buffer);
    }
    wasmDraw(...args, width, height, bufferBase);
    // A fresh, lightweight view every call (not a copy, and not cached
    // outside this method) rather than one built once — `memory.grow`
    // above, or one triggered by a texture in the same `ctx`, detaches
    // whatever buffer an earlier view pointed at, so only reading
    // `memory.buffer` fresh here is guaranteed never to be stale.
    if (draw.kind === "float") return new Float64Array(memory.buffer, bufferBase, pixelCount);
    if (draw.kind === "uint") return new Uint32Array(memory.buffer, bufferBase, pixelCount);
    return new Int32Array(memory.buffer, bufferBase, pixelCount);
  };

  return callable;
}
