/**
 * RMSL CPU backend: compiles an RMSL function into a small WebAssembly
 * module. Scalars flow through WASM params/locals; vectors, matrices and
 * pipeline I/O live at fixed offsets in an exported linear memory that the
 * host reads and writes around each call. Opcodes below are verified
 * empirically against a known answer, not quoted from memory: a
 * wrong-but-valid opcode runs and silently miscompiles.
 */
import { MATRIX_DIMENSIONS, Node, ShaderType, StorageAccess, var_ } from "../../core";
import { AllocRules, planLayout } from "../../layout";
import {
  componentCountOf,
  CpuDrawBuffer,
  CpuRenderer,
  CpuShaderContext,
  CpuShaderResult,
  CpuTextureData,
  CpuTextureWrap,
  elementKindOf,
  isAggregate,
  ScalarKind,
  scalarKindOf,
} from "../cpu";
import { assertStageResult, CompileFnOptions, COMPONENT_INDEX, resolveSwizzleTarget } from "../shared";

/**
 * How a value crosses the module boundary. Scalar kinds are WASM params;
 * "Memory" kinds live at fixed linear-memory offsets. The output/varying/
 * position/fragDepth/value kinds are written by the call and read back
 * after it; "narrow" floats are stored as f32 instead of f64.
 */
export type WasmParam =
  | { kind: "param"; name: string; shaderType: ShaderType }
  | { kind: "uniform"; slot: string; shaderType: ShaderType }
  | { kind: "paramMemory"; name: string; shaderType: ShaderType; address: number }
  | {
      kind: "uniformMemory";
      slot: string;
      shaderType: ShaderType;
      address: number;

      narrow?: boolean;
    }
  | { kind: "attribute"; slot: string; shaderType: ShaderType }
  | { kind: "invocationIndex"; shaderType: ShaderType }
  | { kind: "attributeMemory"; slot: string; shaderType: ShaderType; address: number }
  | { kind: "varying"; slot: string; shaderType: ShaderType }
  | { kind: "varyingMemory"; slot: string; shaderType: ShaderType; address: number }
  | { kind: "fragCoordMemory"; address: number }
  | { kind: "outputMemory"; slot: string; shaderType: ShaderType; address: number }
  | { kind: "varyingOutputMemory"; slot: string; shaderType: ShaderType; address: number }
  | { kind: "positionMemory"; address: number }
  | { kind: "fragDepthMemory"; address: number }
  | { kind: "valueMemory"; shaderType: ShaderType; address: number }
  | { kind: "textureMemory"; slot: string; samplerType: ShaderType; metadataAddress: number }
  | {
      kind: "uniformArrayMemory";
      slot: string;
      shaderType: ShaderType;
      length: number;
      address: number;
      elementStride: number;
      narrow?: boolean;
    }
  | {
      /**
       * A `storage()` slot, called once per element (like `invocationIndex`)
       * rather than once over a whole buffer — so unlike `uniformArrayMemory`
       * this is one value, marshalled from/to `ctx.storages[slot][ctx.index]`
       * around the call instead of a compile-time-sized array.
       */
      kind: "storageMemory";
      slot: string;
      shaderType: ShaderType;
      address: number;
      access: StorageAccess;
    };

/**
 * A compiled module: raw bytes plus the contract the host uses to marshal
 * values, and "draw" metadata when whole-image rendering is available.
 */
export type CompiledWasm = {
  bytes: Uint8Array;
  params: WasmParam[];

  resultType: ShaderType;

  textureHeapBase: number; // heap starts at this offset; below it is the compile-time layout

  memoryPages: number; // pages needed for the compile-time layout; grows from here as textures/draw buffers are marshalled

  sharedMemory: boolean; // whether the module's memory import was declared shared (must match the instantiated memory exactly)

  maxMemoryPages: number; // the maximum this module's memory import declared — only enforced when sharedMemory is true

  draw?: { componentCount: number; kind: "float" | "int" | "uint" | "bool" }; // set when "draw" is exported
};

/**
 * Offsets where the host will store GPU-placed uniforms (float as f32),
 * reserving a memory region sized by totalSize. Reading such a uniform
 * promotes it into the packed scratch layout instead of reading in place.
 */
export type GpuUniformLayout = {
  offsets: Record<string, number>;

  totalSize: number;

  /**
   * Per-uniform-array element stride in the host's narrow buffer (their
   * wgslUniformLayout stride), required for every GPU-placed array.
   */
  strides?: Record<string, number>;
};

/**
 * stage: vertex makes varying/builtinPosition etc. OUTPUTS, fragment makes
 * them inputs (default). derivatives: CPU has no derivatives — throw
 * (default) or evaluate to 0. reentrant: accepted for parity, no effect —
 * WASM locals are fresh per call frame, so there is no shared scratch to
 * privatize.
 */
export type CompileWasmFnOptions = CompileFnOptions & {
  gpuUniformLayout?: GpuUniformLayout;

  stage?: "vertex" | "fragment";

  derivatives?: "throw" | "zero";

  reentrant?: boolean;

  /**
   * Draw into an externally owned WebAssembly.Memory instead of one this
   * module allocates for itself — e.g. a `shared: true`, SharedArrayBuffer-
   * backed memory so multiple worker-hosted instances can draw into disjoint
   * regions of one buffer. The caller is responsible for sizing/growing it
   * (an externally owned memory can't be grown from inside a module that
   * doesn't own it, past whatever `maximum` it was created with).
   */
  memory?: WebAssembly.Memory;

  /**
   * Declares the module's memory import as `shared: true` — required
   * whenever `memory` (or the memory a caller will later instantiate this
   * same compiled module with) is itself a shared, SharedArrayBuffer-backed
   * WebAssembly.Memory: the engine rejects instantiation unless the
   * import's declared shared-ness matches the actual memory object exactly.
   * A shared import also needs a declared maximum — see `maxMemoryPages`.
   */
  sharedMemory?: boolean;

  /** The memory import's declared maximum page count, only meaningful when `sharedMemory` is true. Defaults to 65536 (the full 4GiB wasm32 address space). */
  maxMemoryPages?: number;
};

// Per-texture metadata block written by writeTextureToMemory(), reserved
// per sampler slot. TEX_META_DATA_ADDR points at the heap-relative pixel
// data (stored as f64 per channel); TEX_META_UNORM_DIVISOR is 255 for
// byte data and 1 for float data.
const TEX_META_UNORM_DIVISOR = 0;
const TEX_META_DATA_ADDR = 8;
const TEX_META_WIDTH = 12;
const TEX_META_HEIGHT = 16;
const TEX_META_DEPTH = 20;
const TEX_META_CHANNELS = 24;
const TEX_META_FILTER = 28;
const TEX_META_WRAP_S = 32;
const TEX_META_WRAP_T = 36;
const TEX_META_WRAP_R = 40;
const TEXTURE_META_STRIDE = 44;

/**
 * Packed layout (no alignment padding, no reorder): WASM linear memory has
 * no struct type, so aggregates are stored as flat byte runs.
 */
const PACKED_RULES: AllocRules = {
  sizeAndAlignOf(type) {
    const kind = isAggregate(type) ? elementKindOf(type) : scalarKindOf(type);
    return { size: componentCountOf(type) * componentSizeOf(kind), align: 1 };
  },
  reorderByAlignment: false,
  structAlignMinimum: 1,
};

/** Math.* functions imported from the host and called by index (exp2 lowers to pow). */
const MATH_UNARY_IMPORTS = new Set([
  "sin",
  "cos",
  "tan",
  "asin",
  "acos",
  "atan",
  "sinh",
  "cosh",
  "tanh",
  "asinh",
  "acosh",
  "atanh",
  "exp",
  "log",
  "log2",
]);

const MATH_BINARY_IMPORTS = new Set(["pow", "atan2"]);

const WRAP_MODE_CODE: Record<CpuTextureWrap, number> = { clamp: 0, repeat: 1, mirror: 2 }; // codes the wrapAxis() loop switches on

export const WASM_F64 = 0x7c;
export const WASM_I32 = 0x7f;
export const WASM_FUNC = 0x60;
export const WASM_BLOCKTYPE_VOID = 0x40;

export const WASM_OP = {
  /** `end`: terminates a block, loop, or if. */
  end: 0x0b,
  /** `block`: begins a block with a block type. */
  block: 0x02,
  /** `loop`: begins a loop with a block type; branches to its start. */
  loop: 0x03,
  /** `br`: unconditional branch to a label. */
  br: 0x0c,
  /** `brIf`: conditionally (on an i32) branch to a label. */
  brIf: 0x0d,
  /** `local.get`: push the value of a local onto the stack. */
  localGet: 0x20,
  /** `local.set`: pop the stack into a local. */
  localSet: 0x21,
  /** `local.tee`: pop the stack into a local, leaving the value on the stack. */
  localTee: 0x22,
  /** `call`: call a function by index. */
  call: 0x10,
  /** `select`: pick one of two values based on an i32 condition. */
  select: 0x1b,

  /** `i32.const`: push a signed LEB128 i32 constant. */
  i32Const: 0x41,
  /** `f64.const`: push a f64 constant (8 raw bytes). */
  f64Const: 0x44,

  /** `i32.load`: load an i32 from memory. */
  i32Load: 0x28,
  /** `f32.load`: load an f32 from memory. */
  f32Load: 0x2a,
  /** `f64.load`: load an f64 from memory. */
  f64Load: 0x2b,
  /** `i32.store`: store an i32 to memory. */
  i32Store: 0x36,
  /** `f64.store`: store an f64 to memory. */
  f64Store: 0x39,
  /** `f64.promote_f32`: widen an f32 to an f64. */
  f64PromoteF32: 0xbb,

  /** `i32.eqz`: 1 if the i32 is zero, else 0. */
  i32Eqz: 0x45,
  /** `i32.eq`: equal. */
  i32Eq: 0x46,
  /** `i32.ne`: not equal. */
  i32Ne: 0x47,
  /** `i32.lt_s`: less than, signed. */
  i32LtS: 0x48,
  /** `i32.lt_u`: less than, unsigned. */
  i32LtU: 0x49,
  /** `i32.gt_s`: greater than, signed. */
  i32GtS: 0x4a,
  /** `i32.gt_u`: greater than, unsigned. */
  i32GtU: 0x4b,
  /** `i32.le_s`: less than or equal, signed. */
  i32LeS: 0x4c,
  /** `i32.le_u`: less than or equal, unsigned. */
  i32LeU: 0x4d,
  /** `i32.ge_s`: greater than or equal, signed. */
  i32GeS: 0x4e,
  /** `i32.ge_u`: greater than or equal, unsigned. */
  i32GeU: 0x4f,

  /** `f64.eq`: equal. */
  f64Eq: 0x61,
  /** `f64.ne`: not equal. */
  f64Ne: 0x62,
  /** `f64.lt`: less than. */
  f64Lt: 0x63,
  /** `f64.gt`: greater than. */
  f64Gt: 0x64,
  /** `f64.le`: less than or equal. */
  f64Le: 0x65,
  /** `f64.ge`: greater than or equal. */
  f64Ge: 0x66,

  /** `i32.add`: addition, wrapping around on overflow. */
  i32Add: 0x6a,
  /** `i32.sub`: subtraction, wrapping around on underflow. */
  i32Sub: 0x6b,
  /** `i32.mul`: multiplication, wrapping around on overflow. */
  i32Mul: 0x6c,
  /** `i32.div_s`: signed division (traps on zero, MIN/-1). */
  i32DivS: 0x6d,
  /** `i32.div_u`: unsigned division (traps on zero). */
  i32DivU: 0x6e,
  /** `i32.rem_s`: signed remainder (traps on zero, MIN/-1). */
  i32RemS: 0x6f,
  /** `i32.rem_u`: unsigned remainder (traps on zero). */
  i32RemU: 0x70,
  /** `i32.and`: bitwise AND. */
  i32And: 0x71,
  /** `i32.or`: bitwise OR. */
  i32Or: 0x72,
  /** `i32.xor`: bitwise XOR. */
  i32Xor: 0x73,
  /** `i32.shl`: shift left. */
  i32Shl: 0x74,
  /** `i32.shr_s`: arithmetic (sign-extending) shift right. */
  i32ShrS: 0x75,
  /** `i32.shr_u`: logical (zero-filling) shift right. */
  i32ShrU: 0x76,

  /** `f64.abs`: absolute value. */
  f64Abs: 0x99,
  /** `f64.neg`: negation. */
  f64Neg: 0x9a,
  /** `f64.ceil`: round toward +infinity. */
  f64Ceil: 0x9b,
  /** `f64.floor`: round toward -infinity. */
  f64Floor: 0x9c,
  /** `f64.trunc`: round toward zero. */
  f64Trunc: 0x9d,
  /** `f64.sqrt`: square root. */
  f64Sqrt: 0x9f,
  /** `f64.add`: addition. */
  f64Add: 0xa0,
  /** `f64.sub`: subtraction. */
  f64Sub: 0xa1,
  /** `f64.mul`: multiplication. */
  f64Mul: 0xa2,
  /** `f64.div`: division. */
  f64Div: 0xa3,
  /** `f64.min`: minimum of the two values. */
  f64Min: 0xa4,
  /** `f64.max`: maximum of the two values. */
  f64Max: 0xa5,

  /** `i32.trunc_f64_s`: truncate an f64 toward zero to a signed i32 (traps on NaN or out of range). */
  i32TruncF64S: 0xaa,
  /** `i32.trunc_f64_u`: truncate an f64 toward zero to an unsigned i32 (traps on NaN or out of range). */
  i32TruncF64U: 0xab,
  /** `f64.convert_i32_s`: widen a signed i32 to an f64. */
  f64ConvertI32S: 0xb7,
  /** `f64.convert_i32_u`: widen an unsigned i32 to an f64. */
  f64ConvertI32U: 0xb8,

  /** `if` control opcode: starts an if block, followed by a block type, then the then-branch. */
  if_: 0x04,
  /** `else` marker: separates the then-branch from the else-branch inside an if block. */
  else_: 0x05,
} as const;

function wasmTypeOf(kind: ScalarKind): number {
  return kind === "float" ? WASM_F64 : WASM_I32;
}

/** Bytes per component: 8 for float, 4 for int/uint/bool. */
function componentSizeOf(kind: ScalarKind): number {
  return kind === "float" ? 8 : 4;
}

function isSamplerType(t: string): boolean {
  return t.startsWith("sampler") || t.startsWith("isampler") || t.startsWith("usampler");
}

function isIntegerSamplerType(t: string): boolean {
  return t.startsWith("isampler") || t.startsWith("usampler");
}

/** `textureLoad()`/integer-sampler fetch only supports 2D/3D — reject 1D/cube/texture-arrays early. */
function assertSampled2Dor3D(t: string): void {
  if (!t.endsWith("2D") && !t.endsWith("3D")) {
    throw new Error(
      "[RMSL] compileWasmFn: texture uniforms support sampler2D/sampler3D (and their integer variants) only.",
    );
  }
}

/** `texture()`/`textureLod()` additionally supports samplerCube (float only) — reject 1D/texture-arrays early. */
function assertSampledTextureType(t: string): void {
  if (!t.endsWith("2D") && !t.endsWith("3D") && !t.endsWith("Cube")) {
    throw new Error(
      "[RMSL] compileWasmFn: texture uniforms support sampler2D/sampler3D/samplerCube (2D/3D integer variants too) only.",
    );
  }
}

/**
 * Unsigned LEB128 encoding: the unsigned variable-length integer format used
 * for indices and lengths throughout the wasm binary format. Emits the number
 * in 7-bit groups, least significant first, with the high bit of every byte
 * except the last set to signal continuation.
 */
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

/**
 * Signed LEB128 encoding: the signed variant needed by i32.const, whose
 * operands are sign-extended across the 7-bit groups so negative literals
 * round-trip. Emits continuation bytes until the current group already carries
 * the sign extension for what remains.
 */
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

function wasmStrBytes(s: string): number[] {
  const b = [...new TextEncoder().encode(s)];
  return [...wasmUleb128(b.length), ...b];
}

/** Emits i32.const plus the signed LEB128 operand. */
function i32ConstBytes(n: number): number[] {
  return [WASM_OP.i32Const, ...wasmSleb128(n | 0)];
}

/** Emits f64.const plus the raw 8-byte little-endian operand. */
function f64ConstBytes(n: number): number[] {
  return [WASM_OP.f64Const, ...wasmF64Bytes(n)];
}

/** WASM select pops [whenTrue, whenFalse, cond]; note BOTH value arms are always evaluated. */
function selectExpr(whenTrue: number[], whenFalse: number[], cond: number[]): number[] {
  return [...whenTrue, ...whenFalse, ...cond, WASM_OP.select];
}

/** [base, load, memarg: align=0, offset] — loads one component at a constant address. */
function loadComponent(addr: number, kind: ScalarKind, byteOffset: number): number[] {
  return [
    ...i32ConstBytes(addr),
    kind === "float" ? WASM_OP.f64Load : WASM_OP.i32Load,
    0x00,
    ...wasmUleb128(byteOffset),
  ];
}

/** [base, value, store, align=0, offset] — stores one component at a constant address. */
function storeComponent(addr: number, kind: ScalarKind, byteOffset: number, valueBytes: number[]): number[] {
  return [
    ...i32ConstBytes(addr),
    ...valueBytes,
    kind === "float" ? WASM_OP.f64Store : WASM_OP.i32Store,
    0x00,
    ...wasmUleb128(byteOffset),
  ];
}

/**
 * As the static helpers, but the base address is itself computed at runtime
 * (texture heap, draw output).
 */
function loadDynamic(addrBytes: number[], kind: ScalarKind): number[] {
  return [...addrBytes, kind === "float" ? WASM_OP.f64Load : WASM_OP.i32Load, 0x00, 0x00];
}

/**
 * As the static helpers, but the base address is itself computed at runtime
 * (texture heap, draw output).
 */
function storeDynamic(addrBytes: number[], kind: ScalarKind, valueBytes: number[]): number[] {
  return [...addrBytes, ...valueBytes, kind === "float" ? WASM_OP.f64Store : WASM_OP.i32Store, 0x00, 0x00];
}

/** Bytes computing `base + index * elementStride` — a uniform array element's first component address. */
function uniformArrayElementAddress(base: number, elementStride: number, indexBytes: number[]): number[] {
  return [...i32ConstBytes(base), ...indexBytes, ...i32ConstBytes(elementStride), WASM_OP.i32Mul, WASM_OP.i32Add];
}

/** Casts one scalar to another kind: f64<->i32 via trunc/convert; bools stay as raw bit values. */
function convertComponent(valueBytes: number[], fromKind: ScalarKind, toKind: ScalarKind): number[] {
  if (fromKind === toKind || fromKind === "bool" || toKind === "bool") return valueBytes;
  if (fromKind === "float") {
    return [...valueBytes, toKind === "uint" ? WASM_OP.i32TruncF64U : WASM_OP.i32TruncF64S];
  }
  if (toKind === "float") {
    return [...valueBytes, fromKind === "uint" ? WASM_OP.f64ConvertI32U : WASM_OP.f64ConvertI32S];
  }
  return valueBytes;
}

/** float min/max use native f64.min/max; int/uint fall back to a compare + select (bytes in). */
function minMaxBytes(a: number[], b: number[], kind: ScalarKind, pick: "min" | "max"): number[] {
  if (kind === "float") return [...a, ...b, pick === "min" ? WASM_OP.f64Min : WASM_OP.f64Max];
  const cmp =
    kind === "uint"
      ? pick === "min"
        ? WASM_OP.i32LtU
        : WASM_OP.i32GtU
      : pick === "min"
        ? WASM_OP.i32LtS
        : WASM_OP.i32GtS;
  return selectExpr(a, b, [...a, ...b, cmp]);
}

/**
 * True for aggregate-typed expressions that are anonymous results (not a
 * var/uniform/param...) and therefore need their own fixed address to be
 * materialized into before any component can be read.
 */
const SCRATCH_NODE_TYPES = new Set([
  "construct",
  "uniformArrayElement",
  "cross",
  "reflect",
  "normalize",
  "matVecMul",
  "dFdx",
  "dFdy",
  "fwidth",
  "textureSize",
  "textureLoad",
  "texture",
  "textureLod",
  "clamp",
  "mix",
  "step",
  "smoothstep",
  "select",
  "add",
  "sub",
  "mul",
  "div",
]);

function isScratchNode(node: any): boolean {
  const t = node._t as string;
  if (!isAggregate(t)) return false;
  if (node.type === t) return true;
  if (node.type === "swizzle") return (node.value as string).length > 1;
  return SCRATCH_NODE_TYPES.has(node.type);
}

/**
 * Promotes a gpu-placed uniform from the host's narrow f32 region into the
 * packed f64 layout.
 */
function emitGpuUniformPromote(node: any, addr: number, rawAddr: number): number[] {
  const kind = elementKindOf(node._t as string);
  const width = componentCountOf(node._t as string);
  const rawCompSize = kind === "float" ? 4 : componentSizeOf(kind);
  const compSize = componentSizeOf(kind);
  const out: number[] = [];
  for (let k = 0; k < width; k++) {
    const rawBytes =
      kind === "float"
        ? [...i32ConstBytes(rawAddr + k * rawCompSize), WASM_OP.f32Load, 0x00, ...wasmUleb128(0), WASM_OP.f64PromoteF32]
        : loadComponent(rawAddr, kind, k * rawCompSize);
    out.push(...storeComponent(addr, kind, k * compSize, rawBytes));
  }
  return out;
}

/** Stores a literal aggregate component-wise at addr. */
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

/**
 * Host-side: packs an array value into linear memory. With `narrow` plus a
 * float type, components are stored as f32 (truncating to match the GPU
 * layout); otherwise they mirror the WASM f64/i32 layout exactly.
 */
function writeAggregateToMemory(
  view: DataView,
  address: number,
  shaderType: ShaderType,
  value: any,
  narrow?: boolean,
): void {
  const kind = elementKindOf(shaderType);

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
  const kind = isAggregate(shaderType) ? elementKindOf(shaderType) : scalarKindOf(shaderType);
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

/** Host-side: reads an array value back, converting i32 bits to bool/uint as needed. */
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

/** Host-side: reads one scalar component back from memory. */
function readScalarFromMemory(view: DataView, address: number, shaderType: ShaderType): number | boolean {
  const kind = scalarKindOf(shaderType);
  if (kind === "float") return view.getFloat64(address, true);
  const raw = view.getInt32(address, true);
  if (kind === "bool") return raw !== 0;
  return kind === "uint" ? raw >>> 0 : raw;
}

function readValueFromMemory(view: DataView, address: number, shaderType: ShaderType): unknown {
  return isAggregate(shaderType)
    ? readAggregateFromMemory(view, address, shaderType)
    : readScalarFromMemory(view, address, shaderType);
}

/** Host-side: writes one scalar value into memory — writeAggregateToMemory's counterpart for a non-array type. */
function writeScalarToMemory(view: DataView, address: number, shaderType: ShaderType, value: unknown): void {
  const kind = scalarKindOf(shaderType);
  const num = typeof value === "boolean" ? (value ? 1 : 0) : ((value as number | undefined) ?? 0);
  if (kind === "float") view.setFloat64(address, num, true);
  else view.setInt32(address, num, true);
}

/** Host-side: writes a value (scalar or aggregate) into memory — the write-side counterpart of readValueFromMemory. */
function writeValueToMemory(view: DataView, address: number, shaderType: ShaderType, value: unknown): void {
  if (isAggregate(shaderType)) writeAggregateToMemory(view, address, shaderType, value);
  else writeScalarToMemory(view, address, shaderType, value);
}

/**
 * Host-side: writes a texture's metadata block and pixel data into the heap
 * region reserved by marshalInputs. Channels default to 4; the unorm
 * divisor is 255 for byte arrays and 1 for float data.
 *
 * A cube map has no `depth` field on its `CpuTextureData` — 6 is a property
 * of being a cube, not of the texture, the same convention the JS backend
 * uses — so its 6 layers are counted from `isCube` here, not from `tex`
 * itself. Its wrap modes are always clamp too, whatever `tex.wrapS`/`wrapT`
 * say: neither GPU backend honors a wrap mode on a cube sampler either.
 */
function writeTextureToMemory(
  view: DataView,
  metaAddr: number,
  heapAddr: number,
  tex: CpuTextureData,
  isCube: boolean,
): void {
  const channels = tex.channels ?? 4;
  const depth = isCube ? 6 : (tex.depth ?? 0);
  const isByteData = tex.data instanceof Uint8Array || tex.data instanceof Uint8ClampedArray;
  view.setFloat64(metaAddr + TEX_META_UNORM_DIVISOR, isByteData ? 255 : 1, true);
  view.setInt32(metaAddr + TEX_META_DATA_ADDR, heapAddr, true);
  view.setInt32(metaAddr + TEX_META_WIDTH, tex.width, true);
  view.setInt32(metaAddr + TEX_META_HEIGHT, tex.height, true);
  view.setInt32(metaAddr + TEX_META_DEPTH, depth, true);
  view.setInt32(metaAddr + TEX_META_CHANNELS, channels, true);
  view.setInt32(metaAddr + TEX_META_FILTER, tex.magFilter === "linear" ? 1 : 0, true);
  view.setInt32(metaAddr + TEX_META_WRAP_S, isCube ? WRAP_MODE_CODE.clamp : WRAP_MODE_CODE[tex.wrapS ?? "clamp"], true);
  view.setInt32(metaAddr + TEX_META_WRAP_T, isCube ? WRAP_MODE_CODE.clamp : WRAP_MODE_CODE[tex.wrapT ?? "clamp"], true);
  view.setInt32(metaAddr + TEX_META_WRAP_R, WRAP_MODE_CODE[tex.wrapR ?? "clamp"], true);
  const count = tex.width * tex.height * (depth || 1) * channels;
  for (let i = 0; i < count; i++) {
    view.setFloat64(heapAddr + i * 8, tex.data[i] as number, true);
  }
}

/** Heap bytes for a texture: f64 per component. A cube map is always 6 layers. */
function textureByteSize(tex: CpuTextureData, isCube: boolean): number {
  return tex.width * tex.height * (isCube ? 6 : tex.depth || 1) * (tex.channels ?? 4) * 8;
}

/** Little-endian f64 bytes, as used by f64.const. */
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

/**
 * Compiles an RMSL function into a WASM module in two passes: "collect"
 * walks the AST once, giving each scalar a param/local slot and each
 * aggregate a fixed memory address; the emit helpers then generate bytes
 * against that frozen layout. When the root is an aggregate or the stage
 * writes pipeline outputs, the module declares zero results and values
 * round-trip through the linear memory instead.
 */
export function compileWasmFn(
  fn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  options: CompileWasmFnOptions,
): CompiledWasm {
  const paramNodes = options.params.map((p) => var_(p.name, p.type));
  const rawResult = fn(...paramNodes) as any;
  // Fn's array-return sugar wraps every returned item in its own "seq" node
  // carrying the *same* captured statements plus that item as its tail value
  // (src/core.ts's Fn), so the last item alone already reproduces the whole
  // body: only its value feeds the stage's single result slot, matching
  // compileGlsl/compileWgsl's "last array entry wins" convention.
  const root = Array.isArray(rawResult) ? rawResult[rawResult.length - 1] : rawResult;

  const paramTypeByName = new Map(options.params.map((p) => [p.name, p.type]));
  const fnParamNames = new Set(options.params.map((p) => p.name));

  const effectiveStage: "vertex" | "fragment" = options.stage ?? "fragment";

  // Scratch state for the planning pass. Scalars land in the WASM param space (params) or
  // the WASM local space (localSlots); aggregates and stage I/O get fixed
  // memory addresses recorded in the *Address maps. Everything is resolved
  // up front so the byte emitters never need to re-plan.
  type ScalarWasmParam = Extract<
    WasmParam,
    { kind: "param" | "uniform" | "attribute" | "varying" | "invocationIndex" }
  >;
  const params: ScalarWasmParam[] = [];
  const paramIndex = new Map<string, number>();
  const localSlots: string[] = [];
  const localIndex = new Map<string, number>();
  const localType = new Map<string, ScalarKind>();
  const importsUsed = new Set<string>();

  // memory-kind params, host-written before the call (or read after)
  const memoryParams: WasmParam[] = [];

  const paramAddress = new Map<string, number>();
  const varAddress = new Map<string, number>();
  const uniformAddress = new Map<string, number>();
  const uniformArrayInfo = new Map<string, { base: number; elementStride: number; narrow: boolean }>();

  // host offsets of GPU-placed uniforms (f32)
  const gpuRawUniformAddress = new Map<string, number>();

  const attributeAddress = new Map<string, number>();
  const varyingAddress = new Map<string, number>();
  const storageAddress = new Map<string, number>();

  // one shared per-pixel input slot, allocated on first use
  let fragCoordAddress: number | undefined;

  // needsResult: the WASM function returns nothing; the result is written
  // to memory (valueAddress below) and read back by the host.
  let needsResult = options.stage === "vertex" || isAggregate(root._t as string);
  let positionWritten = false;
  const outputAddress = new Map<string, number>();
  const varyingOutputAddress = new Map<string, number>();
  let positionAddress: number | undefined;
  let fragDepthAddress: number | undefined;

  const textureMetadataAddress = new Map<string, number>();
  const scratchAddress = new WeakMap<object, number>();

  let memCursor = options.gpuUniformLayout?.totalSize ?? 0; // cursor past the host's reserved region

  // Pass 1: plan the whole tree — record slots/addresses and imported
  // math names — before any bytecode is emitted.
  collect(root);

  let resultKind: ScalarKind;
  let valueAddress: number | undefined;

  if (needsResult) {
    assertStageResult(effectiveStage, root._t === "void" ? undefined : (root._t as string), positionWritten);
    // placeholder: with needsResult the module returns void, so it is never used
    resultKind = "float";
    if (effectiveStage === "vertex" && !positionWritten) {
      // vertex fn that never wrote gl_Position: route its value there anyway
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

  // Pass 2: emit bytecode for the body, looking up the slots/addresses
  // planned in Pass 1 (collect above). Direct function bodies are a single
  // value expression; a seq body is statements followed by one final value.
  // The root body sits in one outer block (EXIT_BLOCK_DEPTH); return/discard
  // branch out of it, and loopStack tracks break/continue targets.
  const EXIT_BLOCK_DEPTH = 1;
  const loopStack: { breakDepth: number; continueDepth: number }[] = [];
  // Tracks the block depth in scope wherever expression evaluation currently
  // sits, so a "seq" node encountered mid-expression (a nested Fn call's
  // result) can lower its leading statements with walkStmt at the right
  // depth. walkStmt refreshes it on every call, so it's always accurate by
  // the time a nested seq is walked from within that statement.
  let currentStmtDepth = EXIT_BLOCK_DEPTH;
  const bodyBytes =
    root.type === "seq"
      ? [
          ...(root.params.slice(0, -1) as any[]).flatMap((s: any) => walkStmt(s, EXIT_BLOCK_DEPTH)),
          ...finalValueBytes(root.params[root.params.length - 1]),
        ]
      : finalValueBytes(root);

  const exitBlockType = needsResult ? WASM_BLOCKTYPE_VOID : wasmTypeOf(resultKind); // the outer block carries the function's result type (or void)
  const code = [WASM_OP.block, exitBlockType, ...bodyBytes, WASM_OP.end];

  // Module assembly: type section, math imports, the main function (whose type
  // carries every scalar param and, when !needsResult, one result), a linear
  // memory sized for memCursor, exports, and the code section.
  const typeEntries: number[][] = [];
  let unaryImportType: number | null = null;
  let binaryImportType: number | null = null;

  const importEntries: number[][] = importNames.map((name) => {
    const typeIdx = MATH_BINARY_IMPORTS.has(name) ? binaryImportTypeIdx() : unaryImportTypeIdx();
    return [...wasmStrBytes("math"), ...wasmStrBytes(name), 0x00, ...wasmUleb128(typeIdx)];
  });

  const paramTypes = params.map((p) => [wasmTypeOf(scalarKindOf(p.shaderType))]);
  const resultTypes = needsResult ? [] : [[wasmTypeOf(resultKind)]];
  const mainTypeIdx = typeEntries.length;
  typeEntries.push([WASM_FUNC, ...wasmVec(paramTypes), ...wasmVec(resultTypes)]);

  const mainFuncIndex = importNames.length; // imports come first in the module's function index space

  let drawTypeIdx: number | undefined;
  let drawFuncBody: number[] | undefined;
  const drawComponentCount = root._t === "void" ? 0 : componentCountOf(root._t as string);
  const drawComponentKind: ScalarKind =
    root._t === "void"
      ? "float"
      : isAggregate(root._t as string)
        ? elementKindOf(root._t as string)
        : scalarKindOf(root._t as string);

  // A "draw(width, height, bufferBase)" export: loops every pixel, runs the
  // main function (feeding the pixel in as fragCoord), and stores the output
  // into the caller's buffer — the CPU path for whole-image rendering.
  if (root._t !== "void") {
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
    // pixel centers land at (x + 0.5, y + 0.5) — the same convention js uses
    const writeFragCoord =
      fragCoordAddress === undefined
        ? []
        : [
            ...storeComponent(fragCoordAddress, "float", 0, [
              ...getX,
              WASM_OP.f64ConvertI32S,
              ...f64ConstBytes(0.5),
              WASM_OP.f64Add,
            ]),
            ...storeComponent(fragCoordAddress, "float", 8, [
              ...getY,
              WASM_OP.f64ConvertI32S,
              ...f64ConstBytes(0.5),
              WASM_OP.f64Add,
            ]),
          ];

    const pixelByteOffset = [
      ...getY,
      WASM_OP.localGet,
      ...wasmUleb128(widthIdx),
      WASM_OP.i32Mul,
      ...getX,
      WASM_OP.i32Add,
      ...i32ConstBytes(drawComponentCount * compSize),
      WASM_OP.i32Mul,
    ];
    const destAddr = (k: number) => [
      WASM_OP.localGet,
      ...wasmUleb128(bufferBaseIdx),
      ...pixelByteOffset,
      WASM_OP.i32Add,
      ...i32ConstBytes(k * compSize),
      WASM_OP.i32Add,
    ];
    const copyResult: number[] = needsResult
      ? [
          ...callMain,
          ...Array.from({ length: drawComponentCount }, (_, k) =>
            storeDynamic(destAddr(k), drawComponentKind, loadComponent(valueAddress!, drawComponentKind, k * compSize)),
          ).flat(),
        ]
      : storeDynamic(destAddr(0), drawComponentKind, callMain);
    const perPixel = [...writeFragCoord, ...copyResult];
    const innerLoop = [
      WASM_OP.block,
      WASM_BLOCKTYPE_VOID,
      WASM_OP.loop,
      WASM_BLOCKTYPE_VOID,
      ...getX,
      WASM_OP.localGet,
      ...wasmUleb128(widthIdx),
      WASM_OP.i32GeS,
      WASM_OP.brIf,
      ...wasmUleb128(1),
      ...perPixel,
      ...getX,
      ...i32ConstBytes(1),
      WASM_OP.i32Add,
      WASM_OP.localSet,
      ...wasmUleb128(xIdx),
      WASM_OP.br,
      ...wasmUleb128(0),
      WASM_OP.end,
      WASM_OP.end,
    ];
    const outerLoop = [
      WASM_OP.block,
      WASM_BLOCKTYPE_VOID,
      WASM_OP.loop,
      WASM_BLOCKTYPE_VOID,
      ...getY,
      WASM_OP.localGet,
      ...wasmUleb128(heightIdx),
      WASM_OP.i32GeS,
      WASM_OP.brIf,
      ...wasmUleb128(1),
      ...i32ConstBytes(0),
      WASM_OP.localSet,
      ...wasmUleb128(xIdx),
      ...innerLoop,
      ...getY,
      ...i32ConstBytes(1),
      WASM_OP.i32Add,
      WASM_OP.localSet,
      ...wasmUleb128(yIdx),
      WASM_OP.br,
      ...wasmUleb128(0),
      WASM_OP.end,
      WASM_OP.end,
    ];
    const drawCode = [...i32ConstBytes(0), WASM_OP.localSet, ...wasmUleb128(yIdx), ...outerLoop];
    const drawLocalsDecl = wasmVec([
      [...wasmUleb128(1), WASM_I32], // one local group per loop counter (xIdx, yIdx)
      [...wasmUleb128(1), WASM_I32],
    ]);

    drawFuncBody = [...drawLocalsDecl, ...drawCode, WASM_OP.end];
    drawTypeIdx = typeEntries.length;
    typeEntries.push([WASM_FUNC, ...wasmVec([...paramTypes, [WASM_I32], [WASM_I32], [WASM_I32]]), ...wasmVec([])]);
  }

  const memoryPages = Math.max(1, Math.ceil(memCursor / 65536)); // 65536 bytes per WASM memory page
  // memory is imported rather than owned by the module, so a caller can hand
  // multiple instances the same (optionally SharedArrayBuffer-backed) memory.
  // The shared-ness of an import is a static part of the module (the engine
  // rejects instantiation if it doesn't exactly match the memory object
  // handed in), so it has to be a compile-time option, not a runtime one —
  // a shared import always needs a declared maximum too.
  const sharedMemory = options.sharedMemory ?? false;
  const maxMemoryPages = options.maxMemoryPages ?? 65536; // 65536 pages = the full 4GiB wasm32 address space
  const memoryLimitsBytes = sharedMemory
    ? [0x03, ...wasmUleb128(memoryPages), ...wasmUleb128(maxMemoryPages)]
    : [0x00, ...wasmUleb128(memoryPages)];
  const memoryImportEntry = [...wasmStrBytes("env"), ...wasmStrBytes("memory"), 0x02, ...memoryLimitsBytes];

  const typeSection = wasmSection(1, wasmVec(typeEntries));
  const importSection = wasmSection(2, wasmVec([...importEntries, memoryImportEntry]));
  const funcSection = wasmSection(
    3,
    wasmVec(drawTypeIdx === undefined ? [[mainTypeIdx]] : [[mainTypeIdx], [drawTypeIdx]]),
  );
  const nameBytes = wasmStrBytes(options.name);
  const exportEntries = [[...nameBytes, 0x00, ...wasmUleb128(mainFuncIndex)]];

  if (drawTypeIdx !== undefined) {
    exportEntries.push([...wasmStrBytes("draw"), 0x00, ...wasmUleb128(mainFuncIndex + 1)]);
  }

  const exportSection = wasmSection(7, wasmVec(exportEntries));

  const localsDecl = wasmVec(localSlots.map((name) => [...wasmUleb128(1), wasmTypeOf(localType.get(name)!)]));
  const funcBody = [...localsDecl, ...code, WASM_OP.end];
  const codeEntries = [[...wasmUleb128(funcBody.length), ...funcBody]];

  if (drawFuncBody !== undefined) {
    codeEntries.push([...wasmUleb128(drawFuncBody.length), ...drawFuncBody]);
  }

  const codeSection = wasmSection(10, wasmVec(codeEntries));

  // prettier-ignore
  const bytes = new Uint8Array([
    // "\0asm" magic
    0x00, 0x61, 0x73, 0x6d,
    // version 1 (u32, little-endian)
    0x01, 0x00, 0x00, 0x00,
    ...typeSection,
    ...importSection,
    ...funcSection,
    ...exportSection,
    ...codeSection,
  ]);

  return {
    bytes,
    params: [...params, ...memoryParams],
    resultType: root._t,
    textureHeapBase: memCursor, // host texture heaps are appended at the end of the compile-time layout
    memoryPages, // initial page count a default (non-shared) memory should be created with
    sharedMemory,
    maxMemoryPages,
    draw: drawTypeIdx === undefined ? undefined : { componentCount: drawComponentCount, kind: drawComponentKind },
  };

  /** Advances a fixed-size region from the cursor. */
  function allocateBytes(size: number): number {
    const addr = memCursor;
    memCursor += size;
    return addr;
  }

  /** Bytes needed to store one value of type t under PACKED_RULES. */
  function allocateFor(t: string): number {
    const { size } = planLayout([{ slot: t, type: t }], PACKED_RULES);
    return allocateBytes(size);
  }

  /** Registers a scalar into the WASM param space, deduped by key. */
  function addParam(spec: ScalarWasmParam, key: string): void {
    if (!paramIndex.has(key)) {
      paramIndex.set(key, params.length);
      params.push(spec);
    }
  }

  /** Registers a scalar WASM local, deduped by var name. */
  function addLocal(varName: string, kind: ScalarKind): void {
    if (!localIndex.has(varName)) {
      localIndex.set(varName, localSlots.length);
      localSlots.push(varName);
      localType.set(varName, kind);
    }
  }

  /** Walk the AST, allocating every slot/address the emitters will use. */
  function collect(node: any): void {
    if (node === null || typeof node !== "object") return;
    switch (node.type) {
      case "var": {
        if (!fnParamNames.has(node.value?.varName)) break;
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
        break;
      }

      case "uniform": {
        const v = node.value;
        if (isSamplerType(v.shaderType)) {
          if (!textureMetadataAddress.has(v.slot)) {
            const addr = allocateBytes(TEXTURE_META_STRIDE);
            textureMetadataAddress.set(v.slot, addr);
            memoryParams.push({
              kind: "textureMemory",
              slot: v.slot,
              samplerType: v.shaderType,
              metadataAddress: addr,
            });
          }
        } else if (isAggregate(v.shaderType)) {
          if (!uniformAddress.has(v.slot)) {
            const addr = allocateFor(v.shaderType);
            uniformAddress.set(v.slot, addr);
            const gpuOffset = options.gpuUniformLayout?.offsets[v.slot];
            if (gpuOffset !== undefined) {
              gpuRawUniformAddress.set(v.slot, gpuOffset);
              memoryParams.push({
                kind: "uniformMemory",
                slot: v.slot,
                shaderType: v.shaderType,
                address: gpuOffset,
                narrow: true,
              });
            } else {
              memoryParams.push({ kind: "uniformMemory", slot: v.slot, shaderType: v.shaderType, address: addr });
            }
          }
        } else {
          addParam({ kind: "uniform", slot: v.slot, shaderType: v.shaderType }, `uniform:${v.slot}`);
        }
        break;
      }

      case "uniformArray": {
        if (uniformArrayInfo.has(node.value.slot)) break;
        const shaderType = node.value.shaderType as ShaderType;
        const length = node.value.length as number;
        const elementSize =
          componentCountOf(shaderType) *
          componentSizeOf(isAggregate(shaderType) ? elementKindOf(shaderType) : scalarKindOf(shaderType));
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
        break;
      }

      case "attribute": {
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
        break;
      }

      case "storage": {
        // Like `attribute`, but always memory-backed (never a plain param)
        // since a read_write/write storage has to be readable back after the
        // call — one call per element, marshalled from/to
        // `ctx.storages[slot][ctx.index]` (see instantiateWasm).
        const v = node.value;
        if (!storageAddress.has(v.slot)) {
          const addr = allocateFor(v.shaderType);
          storageAddress.set(v.slot, addr);
          if (v.access !== "read") needsResult = true;
          memoryParams.push({
            kind: "storageMemory",
            slot: v.slot,
            shaderType: v.shaderType,
            address: addr,
            access: v.access,
          });
        }
        break;
      }

      case "invocationIndex": {
        addParam({ kind: "invocationIndex", shaderType: "uint" }, "invocationIndex");
        break;
      }

      case "varying": {
        const v = node.value;
        if (effectiveStage === "fragment") {
          // fragment: varyings are per-call inputs, written by the host
          if (isAggregate(v.shaderType)) {
            if (!varyingAddress.has(v.slot)) {
              const addr = allocateFor(v.shaderType);
              varyingAddress.set(v.slot, addr);
              memoryParams.push({ kind: "varyingMemory", slot: v.slot, shaderType: v.shaderType, address: addr });
            }
          } else {
            addParam({ kind: "varying", slot: v.slot, shaderType: v.shaderType }, `varying:${v.slot}`);
          }
        } else {
          // vertex: varyings are outputs the host reads back
          needsResult = true;
          if (!varyingOutputAddress.has(v.slot)) {
            const addr = allocateFor(v.shaderType);
            varyingOutputAddress.set(v.slot, addr);
            memoryParams.push({ kind: "varyingOutputMemory", slot: v.slot, shaderType: v.shaderType, address: addr });
          }
        }
        break;
      }

      case "fragCoord": {
        // fragment-only shared per-pixel input slot
        if (effectiveStage !== "fragment") {
          throw new Error("[RMSL] compileWasmFn: fragCoord() can only be used in fragment shaders");
        }
        if (fragCoordAddress === undefined) {
          fragCoordAddress = allocateFor("vec2");
          memoryParams.push({ kind: "fragCoordMemory", address: fragCoordAddress });
        }
        break;
      }

      case "output": {
        // pipeline output: forces needsResult so the host can read it
        needsResult = true;
        const v = node.value;
        if (!outputAddress.has(v.slot)) {
          const addr = allocateFor(v.shaderType);
          outputAddress.set(v.slot, addr);
          memoryParams.push({ kind: "outputMemory", slot: v.slot, shaderType: v.shaderType, address: addr });
        }
        break;
      }

      case "builtinPosition": {
        // vertex output (gl_Position), write-only in the vertex stage
        needsResult = true;
        if (positionAddress === undefined) {
          positionAddress = allocateFor("vec4");
          memoryParams.push({ kind: "positionMemory", address: positionAddress });
        }
        break;
      }

      case "builtinFragDepth": {
        // fragment output (gl_FragDepth)
        if (effectiveStage !== "fragment") {
          throw new Error("[RMSL] compileWasmFn: builtinFragDepth() can only be used in fragment shaders");
        }
        needsResult = true;
        if (fragDepthAddress === undefined) {
          fragDepthAddress = allocateFor("float");
          memoryParams.push({ kind: "fragDepthMemory", address: fragDepthAddress });
        }
        break;
      }

      case "assign": {
        // remember a direct gl_Position write
        if (node.params[0].type === "builtinPosition") positionWritten = true;
        break;
      }

      case "let": {
        // aggregate let reserves a varAddress slot; scalar lets become WASM locals
        const targetNode = node.params[0];
        const t = targetNode._t as string;
        if (isAggregate(t)) {
          const varName = targetNode.value.varName;
          if (!varAddress.has(varName)) varAddress.set(varName, allocateFor(t));
        } else {
          addLocal(targetNode.value.varName, scalarKindOf(t));
        }
        break;
      }

      case "exp2":
        importsUsed.add("pow"); // exp2(x) = pow(2, x) via the host import
        break;

      case "smoothstep":
        addLocal("$smoothstep_t", "float"); // one shared temp local for t (dedup keeps it single)
        break;

      default:
        if (MATH_UNARY_IMPORTS.has(node.type) || MATH_BINARY_IMPORTS.has(node.type)) importsUsed.add(node.type);
        break;
    }
    if (isScratchNode(node) && !scratchAddress.has(node)) {
      const addr = allocateFor(node._t as string);

      // Extra scratch right after the value slot: 8 for normalize/reflect
      // (length/dot), 60 for filtered sampling (136 for a cube sampler,
      // which needs the face-selection scratch on top), 8 for texel fetch.
      if (node.type === "normalize" || node.type === "reflect") allocateBytes(8);
      if (
        (node.type === "texture" || node.type === "textureLod") &&
        !isIntegerSamplerType(node.params[0]._t as string)
      ) {
        allocateBytes((node.params[0]._t as string).endsWith("Cube") ? 136 : 60);
      }
      if (
        node.type === "textureLoad" ||
        ((node.type === "texture" || node.type === "textureLod") && isIntegerSamplerType(node.params[0]._t as string))
      ) {
        allocateBytes(8);
      }
      scratchAddress.set(node, addr);
    }
    if (Array.isArray(node.params)) for (const p of node.params) collect(p);
  }

  function localSlotIndex(varName: string): number {
    const i = localIndex.get(varName);
    if (i === undefined) throw new Error(`[RMSL] compileWasmFn: read of undeclared var "${varName}"`);
    return params.length + i; // WASM indexes locals after the params, hence + params.length
  }

  function paramSlotIndex(key: string): number {
    const i = paramIndex.get(key);
    if (i === undefined) throw new Error(`[RMSL] compileWasmFn: internal error, unindexed slot "${key}"`);
    return i;
  }

  function callImport(name: string): number[] {
    return [WASM_OP.call, ...wasmUleb128(importIndexOf.get(name)!)];
  }

  /** Fixed linear-memory address of an aggregate-typed node, from the map for its kind. */
  function nodeAddress(node: any): number {
    switch (node.type) {
      case "var": {
        const name = node.value.varName;
        const addr = fnParamNames.has(name) ? paramAddress.get(name) : varAddress.get(name);
        if (addr === undefined) throw new Error(`[RMSL] compileWasmFn: read of undeclared aggregate var "${name}"`);
        return addr;
      }

      case "uniform": {
        const addr = uniformAddress.get(node.value.slot);
        if (addr === undefined)
          throw new Error(`[RMSL] compileWasmFn: internal error, unaddressed uniform "${node.value.slot}"`);
        return addr;
      }

      case "attribute": {
        const addr = attributeAddress.get(node.value.slot);
        if (addr === undefined)
          throw new Error(`[RMSL] compileWasmFn: internal error, unaddressed attribute "${node.value.slot}"`);
        return addr;
      }

      case "storage": {
        const addr = storageAddress.get(node.value.slot);
        if (addr === undefined)
          throw new Error(`[RMSL] compileWasmFn: internal error, unaddressed storage "${node.value.slot}"`);
        return addr;
      }

      // Per-call model: this call already represents one element, so a
      // storageElement's address is simply its storage's own address — the
      // index expression selects nothing at this level (it does on the WGSL
      // target, which reads the whole buffer in one invocation instead).
      case "storageElement":
        return nodeAddress(node.params[0]);

      case "varying": {
        if (effectiveStage === "fragment") {
          const addr = varyingAddress.get(node.value.slot);
          if (addr === undefined)
            throw new Error(`[RMSL] compileWasmFn: internal error, unaddressed varying "${node.value.slot}"`);
          return addr;
        }
        const addr = varyingOutputAddress.get(node.value.slot);
        if (addr === undefined)
          throw new Error(`[RMSL] compileWasmFn: internal error, unaddressed varying "${node.value.slot}"`);
        return addr;
      }

      case "fragCoord": {
        if (fragCoordAddress === undefined)
          throw new Error("[RMSL] compileWasmFn: internal error, unaddressed fragCoord");
        return fragCoordAddress;
      }

      case "output": {
        const addr = outputAddress.get(node.value.slot);
        if (addr === undefined)
          throw new Error(`[RMSL] compileWasmFn: internal error, unaddressed output "${node.value.slot}"`);
        return addr;
      }

      case "builtinPosition": {
        if (effectiveStage !== "vertex") {
          throw new Error(
            "[RMSL] compileWasmFn: builtinPosition() is the vertex stage's output position, and a " +
              "fragment stage cannot read it. Pass the value you need through a " +
              "varying() instead.",
          );
        }
        if (positionAddress === undefined)
          throw new Error("[RMSL] compileWasmFn: internal error, unaddressed builtinPosition");
        return positionAddress;
      }

      case "builtinFragDepth": {
        if (fragDepthAddress === undefined)
          throw new Error("[RMSL] compileWasmFn: internal error, unaddressed builtinFragDepth");
        return fragDepthAddress;
      }

      case "seq":
        // a nested Fn call's result wraps its statements + final value in a
        // seq node; its address is just wherever its final value already lives.
        return nodeAddress(node.params[node.params.length - 1]);

      default: {
        const addr = scratchAddress.get(node);
        if (addr === undefined)
          throw new Error(`[RMSL] compileWasmFn: internal error, unaddressed node "${node.type}"`);
        return addr;
      }
    }
  }

  /**
   * Emits the stores that guarantee node's aggregate value sits in memory
   * at nodeAddress(node). Most inputs already live there; gpu-placed
   * uniforms get promoted, and pure expressions are computed into their
   * scratch address on first use.
   */
  function materializeIfNeeded(node: any): number[] {
    switch (node.type) {
      case "var":
        return [];
      case "uniform": {
        const rawAddr = gpuRawUniformAddress.get(node.value.slot); // host f32 region: promote it on read
        return rawAddr === undefined ? [] : emitGpuUniformPromote(node, nodeAddress(node), rawAddr);
      }
      case "uniformArrayElement": {
        const info = uniformArrayInfo.get(node.params[0].value.slot);
        if (info === undefined) {
          throw new Error(
            `[RMSL] compileWasmFn: internal error, unaddressed uniform array "${node.params[0].value.slot}"`,
          );
        }
        const kind = elementKindOf(node._t as string);
        const compSize = componentSizeOf(kind);
        const rawCompSize = info.narrow && kind === "float" ? 4 : compSize;
        const width = componentCountOf(node._t as string);
        const index = node.params[1];
        const indexBytes =
          scalarKindOf(index._t as string) === "float" ? [...walkExpr(index), WASM_OP.i32TruncF64S] : walkExpr(index);
        const baseAddr = nodeAddress(node);
        const out: number[] = [];
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
        return out;
      }
      case "attribute":
      case "fragCoord":
      case "storage":
      // Already resident at nodeAddress(node) — the host wrote it there (see
      // "storageMemory" in instantiateWasm) before this call.
      case "storageElement":
        return [];
      case "varying":
      case "output":
      case "builtinPosition":
      case "builtinFragDepth":
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
        // a true matrix product only when BOTH operands are matrices
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
      case "clamp":
        return emitClampStores(node, nodeAddress(node));
      case "mix":
        return emitMixStores(node, nodeAddress(node));
      case "step":
        return emitStepStores(node, nodeAddress(node));
      case "smoothstep":
        return emitSmoothstepStores(node, nodeAddress(node));
      case "select":
        return emitSelectStores(node, nodeAddress(node));
      case "seq": {
        // a nested Fn call's aggregate result: statements, then materialize
        // the final value (which nodeAddress() already resolves to for this node).
        const stmts = node.params.slice(0, -1) as any[];
        const final = node.params[node.params.length - 1];
        return [...stmts.flatMap((s) => walkStmt(s, currentStmtDepth)), ...materializeIfNeeded(final)];
      }
      default:
        if (node.type === node._t && Array.isArray(node.value)) {
          // node.type matching the type name with an array value identifies a literal
          return emitLiteralStores(node, nodeAddress(node));
        }
        throw new Error(`[RMSL] compileWasmFn: unsupported node type in vector position: "${node.type}"`);
    }
  }

  /**
   * Stores a constructor's result. Matrices are column-major with one
   * column per param (a single scalar param builds a diagonal matrix);
   * vector targets copy each param component-wise, spread from scalars.
   */
  function emitConstructStores(node: any, addr: number): number[] {
    const targetType = node._t as string;
    const matShape = MATRIX_DIMENSIONS[targetType];
    if (matShape) {
      const [cols, rows] = matShape;
      if (node.params.length === 1 && componentCountOf(node.params[0]._t) > 1) {
        throw new Error(
          '[RMSL] compileWasmFn: unsupported node type in vector position: "matrix-from-matrix construct"',
        );
      }
      const out: number[] = [];
      if (node.params.length === 1) {
        // single scalar param -> diagonal matrix (identity-scaled)
        for (let c = 0; c < cols; c++) {
          for (let r = 0; r < rows; r++) {
            const valueBytes = c === r ? walkExpr(node.params[0]) : f64ConstBytes(0);
            out.push(...storeComponent(addr, "float", (c * rows + r) * 8, valueBytes));
          }
        }
        return out;
      }

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
        out.push(
          ...storeComponent(addr, targetKind, compIndex * compSize, convertComponent(walkExpr(p), pKind, targetKind)),
        );
        compIndex++;
      } else {
        out.push(...materializeIfNeeded(p));
        const pAddr = nodeAddress(p);
        const pKind = elementKindOf(p._t);
        const pCompSize = componentSizeOf(pKind);
        for (let k = 0; k < pWidth; k++) {
          out.push(
            ...storeComponent(
              addr,
              targetKind,
              compIndex * compSize,
              convertComponent(loadComponent(pAddr, pKind, k * pCompSize), pKind, targetKind),
            ),
          );
          compIndex++;
        }
      }
    }
    return out;
  }

  /** CPU has no derivative opcodes: throw, or emit zeros under { derivatives: "zero" }. */
  function assertDerivativesAllowed(node: any): void {
    if (options.derivatives === "zero") return;
    throw new Error(
      `[RMSL] compileWasmFn: ${node.type}() has no meaning on the CPU target. ` +
        `Compile with { derivatives: "zero" } to evaluate it as 0.`,
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

  /**
   * textureSize(): copies width/height(/depth) out of the metadata block as
   * uints. No 2D/3D restriction — valid for cube too, matching compileJS.
   */
  function emitTextureSizeStores(node: any, addr: number): number[] {
    const metaAddr = textureMetadataAddress.get(node.params[0].value.slot);
    if (metaAddr === undefined) {
      throw new Error(`[RMSL] compileWasmFn: internal error, unaddressed texture "${node.params[0].value.slot}"`);
    }
    const width = componentCountOf(node._t as string);
    const out: number[] = [];
    const fieldOffset = [TEX_META_WIDTH, TEX_META_HEIGHT, TEX_META_DEPTH];
    for (let k = 0; k < width; k++) {
      out.push(...storeComponent(addr, "uint", k * 4, loadComponent(metaAddr, "uint", fieldOffset[k])));
    }
    return out;
  }

  /**
   * textureLoad(texture, coord): raw, unfiltered per-texel fetch. OOB
   * coords yield zero (alpha included); missing channels default to 0
   * (alpha 1). Integer samplers truncate the stored f64 channels; float
   * samplers divide each channel by its unorm divisor.
   */
  function emitTexelFetchStores(node: any, addr: number): number[] {
    const samplerNode = node.params[0];
    const coordsNode = node.params[1];
    const samplerType = samplerNode._t as string;
    assertSampled2Dor3D(samplerType);
    const is3D = samplerType.endsWith("3D");
    const isInteger = isIntegerSamplerType(samplerType);

    const metaAddr: number = ((): number => {
      // IIFE: keeps a definitely-narrowed const usable inside the nested function declarations below
      const a = textureMetadataAddress.get(samplerNode.value.slot);
      if (a === undefined)
        throw new Error(`[RMSL] compileWasmFn: internal error, unaddressed texture "${samplerNode.value.slot}"`);
      return a;
    })();
    const materialize = materializeIfNeeded(coordsNode);
    const coordsAddr = nodeAddress(coordsNode);
    const coordKind = elementKindOf(coordsNode._t as string);
    const dims = is3D ? [TEX_META_WIDTH, TEX_META_HEIGHT, TEX_META_DEPTH] : [TEX_META_WIDTH, TEX_META_HEIGHT];

    const rawAxis = (k: number) => loadComponent(coordsAddr, "int", k * 4);
    const dimAxis = (offset: number) => loadComponent(metaAddr, "int", offset);

    const targetKind = elementKindOf(node._t as string);
    const compSize = componentSizeOf(targetKind);

    const scratch = addr + componentCountOf(node._t as string) * compSize; // slots right after the value: an OOB flag + clamped linear texel index
    const OOB_FLAG = scratch;
    const TEXEL_INDEX = scratch + 4;
    const setup = [
      ...storeComponent(OOB_FLAG, "int", 0, oobFlag()),
      ...storeComponent(TEXEL_INDEX, "int", 0, texelIndexBytes()),
    ];

    const out = [...materialize, ...setup];
    for (let i = 0; i < 4; i++) {
      out.push(...storeComponent(addr, targetKind, i * compSize, channelValue(i)));
    }

    return out;

    // out of range when coord < 0 or coord >= dim; unsigned coords skip the low bound
    function axisOOB(k: number): number[] {
      const tooHigh = [...rawAxis(k), ...dimAxis(dims[k]), coordKind === "uint" ? WASM_OP.i32GeU : WASM_OP.i32GeS];
      if (coordKind === "uint") return tooHigh;
      const tooLow = [...rawAxis(k), ...i32ConstBytes(0), WASM_OP.i32LtS];
      return [...tooLow, ...tooHigh, WASM_OP.i32Or];
    }

    function oobFlag(): number[] {
      let flag = axisOOB(0);
      for (let k = 1; k < dims.length; k++) flag = [...flag, ...axisOOB(k), WASM_OP.i32Or];
      return flag;
    }

    // clamp coord into [0, dim-1] first: selects evaluate BOTH sides, so an
    // out-of-range index must never reach a memory load
    function safeAxis(k: number): number[] {
      const raw = rawAxis(k);
      const dimMinus1 = [...dimAxis(dims[k]), ...i32ConstBytes(1), WASM_OP.i32Sub];
      const nonNegative =
        coordKind === "uint" ? raw : selectExpr(i32ConstBytes(0), raw, [...raw, ...i32ConstBytes(0), WASM_OP.i32LtS]);
      const tooHigh =
        coordKind === "uint" ? [...raw, ...dimMinus1, WASM_OP.i32GtU] : [...nonNegative, ...dimMinus1, WASM_OP.i32GtS];
      return selectExpr(dimMinus1, nonNegative, tooHigh);
    }

    function texelIndexBytes(): number[] {
      const x = safeAxis(0);
      const y = safeAxis(1);
      const yx = [...y, ...dimAxis(TEX_META_WIDTH), WASM_OP.i32Mul, ...x, WASM_OP.i32Add];
      if (!is3D) return yx;
      const z = safeAxis(2);
      const zy = [...z, ...dimAxis(TEX_META_HEIGHT), WASM_OP.i32Mul, ...y, WASM_OP.i32Add];
      return [...zy, ...dimAxis(TEX_META_WIDTH), WASM_OP.i32Mul, ...x, WASM_OP.i32Add];
    }

    function elemAddrBytes(i: number): number[] {
      const dataAddr = loadComponent(metaAddr, "int", TEX_META_DATA_ADDR);
      const channels = loadComponent(metaAddr, "int", TEX_META_CHANNELS);
      const elemOffset = [
        ...loadComponent(TEXEL_INDEX, "int", 0),
        ...channels,
        WASM_OP.i32Mul,
        ...i32ConstBytes(i),
        WASM_OP.i32Add,
      ];
      const byteOffset = [...elemOffset, ...i32ConstBytes(3), WASM_OP.i32Shl];
      return [...dataAddr, ...byteOffset, WASM_OP.i32Add];
    }

    // missing channels default to 0 (alpha -> 1); OOB coords yield all-zero
    function channelValue(i: number): number[] {
      const present = [...i32ConstBytes(i), ...loadComponent(metaAddr, "int", TEX_META_CHANNELS), WASM_OP.i32LtS];
      const oob = loadComponent(OOB_FLAG, "int", 0);
      if (isInteger) {
        const truncOp = samplerType.startsWith("isampler") ? WASM_OP.i32TruncF64S : WASM_OP.i32TruncF64U;
        const fetched = [...loadDynamic(elemAddrBytes(i), "float"), truncOp];
        const missingChannelDefault = i === 3 ? i32ConstBytes(1) : i32ConstBytes(0);
        const inRange = selectExpr(fetched, missingChannelDefault, present);
        return selectExpr(i32ConstBytes(0), inRange, oob);
      }
      const divisor = loadComponent(metaAddr, "float", TEX_META_UNORM_DIVISOR);
      const fetched = [...loadDynamic(elemAddrBytes(i), "float"), ...divisor, WASM_OP.f64Div];
      const missingChannelDefault = i === 3 ? f64ConstBytes(1) : f64ConstBytes(0);
      const inRange = selectExpr(fetched, missingChannelDefault, present);
      return selectExpr(f64ConstBytes(0), inRange, oob);
    }
  }

  /**
   * texture()/textureLod(): bilinear filtering (trilinear for 3D) with
   * per-axis wrap modes. The four/eight corner indices and blend factors
   * are precomputed into scratch once, then reused per channel. Integer
   * samplers get no filtering, so they fall through to the raw fetch.
   */
  function emitTextureSampleStores(node: any, addr: number): number[] {
    const samplerNode = node.params[0];
    const coordsNode = node.params[1];
    const samplerType = samplerNode._t as string;
    assertSampledTextureType(samplerType);
    const isCube = samplerType.endsWith("Cube");
    if (isCube && isIntegerSamplerType(samplerType)) {
      throw new Error(
        "[RMSL] compileWasmFn: samplerCube supports texture()/textureLod() as a float sampler only " +
          "(isamplerCube/usamplerCube aren't supported yet).",
      );
    }
    if (isIntegerSamplerType(samplerType)) return emitTexelFetchStores(node, addr);

    const is3D = samplerType.endsWith("3D");
    const metaAddr: number = ((): number => {
      // IIFE: keeps a definitely-narrowed const usable inside the nested function declarations below
      const a = textureMetadataAddress.get(samplerNode.value.slot);
      if (a === undefined)
        throw new Error(`[RMSL] compileWasmFn: internal error, unaddressed texture "${samplerNode.value.slot}"`);
      return a;
    })();
    const materialize = materializeIfNeeded(coordsNode);
    const coordsAddr = nodeAddress(coordsNode);

    const scratch = addr + 32; // scratch right after the value: corner indices + blend factors reused per channel
    // A cube sample's own scratch, past the 2D/3D fields below — the
    // direction's components and their absolute values, which axis is
    // dominant, and the face/u/v the face-selection math lands on.
    const CUBE_X = scratch + 60,
      CUBE_Y = scratch + 68,
      CUBE_Z = scratch + 76,
      CUBE_AX = scratch + 84,
      CUBE_AY = scratch + 92,
      CUBE_AZ = scratch + 100,
      CUBE_XDOM = scratch + 108,
      CUBE_YDOM = scratch + 112,
      CUBE_U = scratch + 116,
      CUBE_V = scratch + 124,
      CUBE_FACE = scratch + 132;

    const rawCoord = (k: number): number[] => loadComponent(coordsAddr, "float", k * 8);
    // For a cube sample, "u"/"v" (axis 0/1) are the face-local coordinate the
    // face-selection math below computes, not the raw direction components.
    const uv = (k: number): number[] => (isCube ? loadComponent(k === 0 ? CUBE_U : CUBE_V, "float", 0) : rawCoord(k));
    const dimI32 = (offset: number): number[] => loadComponent(metaAddr, "int", offset);
    const dimF64 = (offset: number): number[] => [...dimI32(offset), WASM_OP.f64ConvertI32S];

    // Standard cube-map face selection: the major axis (largest absolute
    // component) picks the face — +X,-X,+Y,-Y,+Z,-Z, the face order both GPU
    // backends and three.js's own CubeTexture agree on — and the other two
    // components, divided by it, are the face-local coordinate. Mirrors
    // js.ts's _cubeFace exactly; see its comment for the derivation.
    const cubeSetup: number[] = [];
    if (isCube) {
      const x = () => loadComponent(CUBE_X, "float", 0);
      const y = () => loadComponent(CUBE_Y, "float", 0);
      const z = () => loadComponent(CUBE_Z, "float", 0);
      const ax = () => loadComponent(CUBE_AX, "float", 0);
      const ay = () => loadComponent(CUBE_AY, "float", 0);
      const az = () => loadComponent(CUBE_AZ, "float", 0);
      const xDom = () => loadComponent(CUBE_XDOM, "int", 0);
      const notXThenYDom = () => loadComponent(CUBE_YDOM, "int", 0);
      const isPos = (v: () => number[]): number[] => [...v(), ...f64ConstBytes(0), WASM_OP.f64Ge];

      const faceX = selectExpr(i32ConstBytes(0), i32ConstBytes(1), isPos(x));
      const faceY = selectExpr(i32ConstBytes(2), i32ConstBytes(3), isPos(y));
      const faceZ = selectExpr(i32ConstBytes(4), i32ConstBytes(5), isPos(z));
      const face = selectExpr(faceX, selectExpr(faceY, faceZ, notXThenYDom()), xDom());

      const ma = selectExpr(ax(), selectExpr(ay(), az(), notXThenYDom()), xDom());

      const ucX = selectExpr([...z(), WASM_OP.f64Neg], z(), isPos(x));
      const ucNotX = selectExpr(x(), selectExpr(x(), [...x(), WASM_OP.f64Neg], isPos(z)), notXThenYDom());
      const uc = selectExpr(ucX, ucNotX, xDom());

      const vcX = [...y(), WASM_OP.f64Neg];
      const vcNotX = selectExpr(selectExpr(z(), [...z(), WASM_OP.f64Neg], isPos(y)), vcX, notXThenYDom());
      const vc = selectExpr(vcX, vcNotX, xDom());

      const uvFrom = (component: number[]): number[] => [
        ...component,
        ...ma,
        WASM_OP.f64Div,
        ...f64ConstBytes(1),
        WASM_OP.f64Add,
        ...f64ConstBytes(0.5),
        WASM_OP.f64Mul,
      ];

      cubeSetup.push(
        ...storeComponent(CUBE_X, "float", 0, rawCoord(0)),
        ...storeComponent(CUBE_Y, "float", 0, rawCoord(1)),
        ...storeComponent(CUBE_Z, "float", 0, rawCoord(2)),
        ...storeComponent(CUBE_AX, "float", 0, [...x(), WASM_OP.f64Abs]),
        ...storeComponent(CUBE_AY, "float", 0, [...y(), WASM_OP.f64Abs]),
        ...storeComponent(CUBE_AZ, "float", 0, [...z(), WASM_OP.f64Abs]),
        ...storeComponent(CUBE_XDOM, "int", 0, [
          ...ax(),
          ...ay(),
          WASM_OP.f64Ge,
          ...ax(),
          ...az(),
          WASM_OP.f64Ge,
          WASM_OP.i32And,
        ]),
        ...storeComponent(CUBE_YDOM, "int", 0, [...ay(), ...az(), WASM_OP.f64Ge]),
        ...storeComponent(CUBE_FACE, "int", 0, face),
        ...storeComponent(CUBE_U, "float", 0, uvFrom(uc)),
        ...storeComponent(CUBE_V, "float", 0, uvFrom(vc)),
      );
    }

    const NEAREST_X = scratch,
      NEAREST_Y = scratch + 4,
      NEAREST_Z = scratch + 8;
    const XA = scratch + 12,
      XB = scratch + 16,
      YA = scratch + 20,
      YB = scratch + 24,
      ZA = scratch + 28,
      ZB = scratch + 32;
    const TX = scratch + 36,
      TY = scratch + 44,
      TZ = scratch + 52;

    // wraps a texel index per mode: 0 clamp, 1 repeat, 2 mirror. Mirror works
    // on a period of 2*dim: idx % 2d cycles, then reflects back into [0, d).
    function wrapAxis(idxBytes: number[], dimOffset: number, wrapOffset: number): number[] {
      const dim = (): number[] => dimI32(dimOffset);
      const mode = (): number[] => dimI32(wrapOffset);
      const dimMinus1 = (): number[] => [...dim(), ...i32ConstBytes(1), WASM_OP.i32Sub];
      const clampVal = selectExpr(
        i32ConstBytes(0),
        selectExpr(dimMinus1(), idxBytes, [...idxBytes, ...dimMinus1(), WASM_OP.i32GtS]),
        [...idxBytes, ...i32ConstBytes(0), WASM_OP.i32LtS],
      );
      const repeatVal = [
        ...[...[...idxBytes, ...dim(), WASM_OP.i32RemS], ...dim(), WASM_OP.i32Add],
        ...dim(),
        WASM_OP.i32RemS,
      ];
      const twoDim = (): number[] => [...dim(), ...i32ConstBytes(2), WASM_OP.i32Mul];
      const period = [
        ...[...[...idxBytes, ...twoDim(), WASM_OP.i32RemS], ...twoDim(), WASM_OP.i32Add],
        ...twoDim(),
        WASM_OP.i32RemS,
      ];
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

    // pixel-space index = uv*dim - 0.5 (half-texel center); i0 = floor, t = fract in [0,1)
    function fracAxis(k: number, dimOffset: number): { i0: number[]; t: number[] } {
      const f = [...uv(k), ...dimF64(dimOffset), WASM_OP.f64Mul, ...f64ConstBytes(0.5), WASM_OP.f64Sub];
      const f0 = [...f, WASM_OP.f64Floor];
      return { i0: [...f0, WASM_OP.i32TruncF64S], t: [...f, ...f0, WASM_OP.f64Sub] };
    }

    const nearestX = wrapAxis(
      [...uv(0), ...dimF64(TEX_META_WIDTH), WASM_OP.f64Mul, WASM_OP.f64Floor, WASM_OP.i32TruncF64S],
      TEX_META_WIDTH,
      TEX_META_WRAP_S,
    );
    const nearestY = wrapAxis(
      [...uv(1), ...dimF64(TEX_META_HEIGHT), WASM_OP.f64Mul, WASM_OP.f64Floor, WASM_OP.i32TruncF64S],
      TEX_META_HEIGHT,
      TEX_META_WRAP_T,
    );
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
      const nearestZ = wrapAxis(
        [...uv(2), ...dimF64(TEX_META_DEPTH), WASM_OP.f64Mul, WASM_OP.f64Floor, WASM_OP.i32TruncF64S],
        TEX_META_DEPTH,
        TEX_META_WRAP_R,
      );
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

    function texelIndexBytes(x: number[], y: number[], z: number[] | null): number[] {
      const yx = [...y, ...dimI32(TEX_META_WIDTH), WASM_OP.i32Mul, ...x, WASM_OP.i32Add];
      if (!z) return yx;
      const zy = [...z, ...dimI32(TEX_META_HEIGHT), WASM_OP.i32Mul, ...y, WASM_OP.i32Add];
      return [...zy, ...dimI32(TEX_META_WIDTH), WASM_OP.i32Mul, ...x, WASM_OP.i32Add];
    }

    function texelChannelRaw(x: number[], y: number[], z: number[] | null, i: number): number[] {
      const dataAddr = dimI32(TEX_META_DATA_ADDR);
      const channels = dimI32(TEX_META_CHANNELS);
      const elemOffset = [
        ...texelIndexBytes(x, y, z),
        ...channels,
        WASM_OP.i32Mul,
        ...i32ConstBytes(i),
        WASM_OP.i32Add,
      ];
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

    function lerp(a: number[], b: number[], t: number[]): number[] {
      // a*(1-t) + b*t — recomposed so a and b are each emitted exactly once
      return [
        ...a,
        ...f64ConstBytes(1),
        ...t,
        WASM_OP.f64Sub,
        WASM_OP.f64Mul,
        ...b,
        ...t,
        WASM_OP.f64Mul,
        WASM_OP.f64Add,
      ];
    }

    function bilinear(
      xa: number[],
      xb: number[],
      ya: number[],
      yb: number[],
      z: number[] | null,
      tx: number[],
      ty: number[],
      i: number,
    ): number[] {
      const taa = texelChannel(xa, ya, z, i);
      const tba = texelChannel(xb, ya, z, i);
      const tab = texelChannel(xa, yb, z, i);
      const tbb = texelChannel(xb, yb, z, i);
      const lower = lerp(taa, tba, tx);
      const upper = lerp(tab, tbb, tx);
      return lerp(lower, upper, ty);
    }

    const nx = loadComponent(NEAREST_X, "int", 0),
      ny = loadComponent(NEAREST_Y, "int", 0);
    const nz = is3D ? loadComponent(NEAREST_Z, "int", 0) : isCube ? loadComponent(CUBE_FACE, "int", 0) : null;
    const xaBytes = loadComponent(XA, "int", 0),
      xbBytes = loadComponent(XB, "int", 0);
    const yaBytes = loadComponent(YA, "int", 0),
      ybBytes = loadComponent(YB, "int", 0);
    const txBytes = loadComponent(TX, "float", 0),
      tyBytes = loadComponent(TY, "float", 0);

    const linearStores: number[] = [];
    const nearestStores: number[] = [];
    for (let i = 0; i < 4; i++) {
      let linearValue: number[];
      if (isCube) {
        // The face never blends into a neighbour — nz is a single, fixed
        // face index for this sample, so one bilinear tap within it (not a
        // near/far lerp across z) is the whole story.
        linearValue = bilinear(xaBytes, xbBytes, yaBytes, ybBytes, nz, txBytes, tyBytes, i);
      } else if (!is3D) {
        linearValue = bilinear(xaBytes, xbBytes, yaBytes, ybBytes, null, txBytes, tyBytes, i);
      } else {
        const zaBytes = loadComponent(ZA, "int", 0),
          zbBytes = loadComponent(ZB, "int", 0);
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
      ...cubeSetup,
      ...setup,
      // real if/else, not select: the nearest path is cheap, and select would always compute both
      ...dimI32(TEX_META_FILTER),
      WASM_OP.if_,
      WASM_BLOCKTYPE_VOID,
      ...linearStores,
      WASM_OP.else_,
      ...nearestStores,
      WASM_OP.end,
    ];
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
      out.push(
        ...storeComponent(addr, kind, i * compSize, loadComponent(srcAddr, srcKind, COMPONENT_INDEX[ch] * srcCompSize)),
      );
    });
    return out;
  }

  /**
   * Element-wise op over same-width vectors, with scalar operands broadcast
   * by re-evaluating them (walkExpr) per component; aggregate operands are
   * materialized once and loaded per component.
   */
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
      opcode =
        node.type === "add"
          ? WASM_OP.f64Add
          : node.type === "sub"
            ? WASM_OP.f64Sub
            : node.type === "mul"
              ? WASM_OP.f64Mul
              : WASM_OP.f64Div;
    } else if (node.type === "add") opcode = WASM_OP.i32Add;
    else if (node.type === "sub") opcode = WASM_OP.i32Sub;
    else if (node.type === "mul") opcode = WASM_OP.i32Mul;
    else opcode = targetKind === "uint" ? WASM_OP.i32DivU : WASM_OP.i32DivS; // div needs the unsigned variant; add/sub/mul share the same bits
    for (let k = 0; k < width; k++) {
      const aBytes = aWidth > 1 ? loadComponent(aAddr!, targetKind, k * compSize) : walkExpr(a);
      const bBytes = bWidth > 1 ? loadComponent(bAddr!, targetKind, k * compSize) : walkExpr(b);
      out.push(...storeComponent(addr, targetKind, k * compSize, [...aBytes, ...bBytes, opcode]));
    }
    return out;
  }

  /** clamp = max(x, lo) then min(hi); an core guarantee keeps all three params the same width. */
  function emitClampStores(node: any, addr: number): number[] {
    const [x, lo, hi] = node.params;
    const targetKind = elementKindOf(node._t as string);
    const compSize = componentSizeOf(targetKind);
    const width = componentCountOf(node._t as string);
    const out = [...materializeIfNeeded(x), ...materializeIfNeeded(lo), ...materializeIfNeeded(hi)];
    const xAddr = nodeAddress(x),
      loAddr = nodeAddress(lo),
      hiAddr = nodeAddress(hi);
    for (let k = 0; k < width; k++) {
      const xk = loadComponent(xAddr, targetKind, k * compSize);
      const lok = loadComponent(loAddr, targetKind, k * compSize);
      const hik = loadComponent(hiAddr, targetKind, k * compSize);
      out.push(
        ...storeComponent(
          addr,
          targetKind,
          k * compSize,
          minMaxBytes(minMaxBytes(xk, lok, targetKind, "max"), hik, targetKind, "min"),
        ),
      );
    }
    return out;
  }

  /** mix(a, b, t) = a + t*(b - a) per component; t may be scalar (re-evaluated) or same-width vector. */
  function emitMixStores(node: any, addr: number): number[] {
    const [a, b, t] = node.params;
    const width = componentCountOf(node._t as string);
    const tWidth = componentCountOf(t._t as string);
    const out = [...materializeIfNeeded(a), ...materializeIfNeeded(b)];
    if (tWidth > 1) out.push(...materializeIfNeeded(t));
    const aAddr = nodeAddress(a),
      bAddr = nodeAddress(b);
    const tAddr = tWidth > 1 ? nodeAddress(t) : undefined;
    for (let k = 0; k < width; k++) {
      const ak = loadComponent(aAddr, "float", k * 8);
      const bk = loadComponent(bAddr, "float", k * 8);
      const tk = tWidth > 1 ? loadComponent(tAddr!, "float", k * 8) : walkExpr(t);
      out.push(
        ...storeComponent(addr, "float", k * 8, [
          ...ak,
          ...tk,
          ...bk,
          ...ak,
          WASM_OP.f64Sub,
          WASM_OP.f64Mul,
          WASM_OP.f64Add,
        ]),
      );
    }
    return out;
  }

  /** step(edge, x) = 0.0 when x < edge, else 1.0. */
  function emitStepStores(node: any, addr: number): number[] {
    const [edge, x] = node.params;
    const width = componentCountOf(node._t as string);
    const out = [...materializeIfNeeded(edge), ...materializeIfNeeded(x)];
    const edgeAddr = nodeAddress(edge),
      xAddr = nodeAddress(x);
    for (let k = 0; k < width; k++) {
      const ek = loadComponent(edgeAddr, "float", k * 8);
      const xk = loadComponent(xAddr, "float", k * 8);
      out.push(
        ...storeComponent(
          addr,
          "float",
          k * 8,
          selectExpr(f64ConstBytes(0), f64ConstBytes(1), [...xk, ...ek, WASM_OP.f64Lt]),
        ),
      );
    }
    return out;
  }

  /**
   * select(cond, a, b) = a when cond, else b, per component. `cond`'s width
   * follows core.ts's `select`: a scalar bool broadcasts, a bvecN selects
   * component-by-component against a and b's (equal, matching) width.
   */
  function emitSelectStores(node: any, addr: number): number[] {
    const [cond, a, b] = node.params;
    const targetKind = elementKindOf(node._t as string);
    const compSize = componentSizeOf(targetKind);
    const width = componentCountOf(node._t as string);
    const condWidth = componentCountOf(cond._t as string);
    const out = [...materializeIfNeeded(a), ...materializeIfNeeded(b)];
    if (condWidth > 1) out.push(...materializeIfNeeded(cond));
    const aAddr = nodeAddress(a),
      bAddr = nodeAddress(b);
    const condAddr = condWidth > 1 ? nodeAddress(cond) : undefined;
    for (let k = 0; k < width; k++) {
      const ak = loadComponent(aAddr, targetKind, k * compSize);
      const bk = loadComponent(bAddr, targetKind, k * compSize);
      const condK = condWidth > 1 ? loadComponent(condAddr!, "bool", k * componentSizeOf("bool")) : walkExpr(cond);
      out.push(...storeComponent(addr, targetKind, k * compSize, selectExpr(ak, bk, condK)));
    }
    return out;
  }

  /** smoothstep(e0, e1, x) per component, via the shared t computation in emitSmoothstepValue. */
  function emitSmoothstepStores(node: any, addr: number): number[] {
    const [e0, e1, x] = node.params;
    const width = componentCountOf(node._t as string);
    const out = [...materializeIfNeeded(e0), ...materializeIfNeeded(e1), ...materializeIfNeeded(x)];
    const e0Addr = nodeAddress(e0),
      e1Addr = nodeAddress(e1),
      xAddr = nodeAddress(x);
    for (let k = 0; k < width; k++) {
      const value = emitSmoothstepValue(
        loadComponent(e0Addr, "float", k * 8),
        loadComponent(e1Addr, "float", k * 8),
        loadComponent(xAddr, "float", k * 8),
      );
      out.push(...storeComponent(addr, "float", k * 8, value));
    }
    return out;
  }

  /**
   * A matCLxRL times a matCRxRR: C[col,row] = sum_k A[k,row] * B[col,k], for
   * col in [0,CR), row in [0,RL), k in [0,CL) (CL === RR, checked below).
   * Square is the CL===RL===CR===RR special case, not a separate code path.
   */
  function emitMatMatMulStores(node: any, addr: number): number[] {
    const [a, b] = node.params;
    const aType = a._t as string,
      bType = b._t as string;
    const [cL, rL] = MATRIX_DIMENSIONS[aType];
    const [cR, rR] = MATRIX_DIMENSIONS[bType];
    // Core already rejects a mismatched product when the node is built, so
    // this is defense-in-depth against a hand-built node, not a coverage gap.
    if (cL !== rR) {
      throw new Error(`[RMSL] compileWasmFn: internal error, mismatched matrix product ("${aType}" x "${bType}")`);
    }
    const out = [...materializeIfNeeded(a), ...materializeIfNeeded(b)];
    const aAddr = nodeAddress(a),
      bAddr = nodeAddress(b);
    for (let col = 0; col < cR; col++) {
      for (let row = 0; row < rL; row++) {
        let terms: number[] = [];
        for (let k = 0; k < cL; k++) {
          const term = [
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

  /** mat * vec per row: outRow = sum_c mat[row,c]*vec[c]; a narrow vec ends with an implicit 1. */
  function emitMatVecMulStores(node: any, addr: number): number[] {
    const [matNode, vecNode] = node.params;
    const [cols, rows] = MATRIX_DIMENSIONS[matNode._t as string];
    const vecWidth = componentCountOf(vecNode._t);
    const outRows = componentCountOf(node._t as string);
    const out = [...materializeIfNeeded(matNode), ...materializeIfNeeded(vecNode)];
    const matAddr = nodeAddress(matNode),
      vecAddr = nodeAddress(vecNode);
    for (let row = 0; row < outRows; row++) {
      let terms: number[] = [];
      for (let c = 0; c < vecWidth; c++) {
        const term = [
          ...loadComponent(matAddr, "float", (c * rows + row) * 8),
          ...loadComponent(vecAddr, "float", c * 8),
          WASM_OP.f64Mul,
        ];
        terms = c === 0 ? term : [...terms, ...term, WASM_OP.f64Add];
      }
      if (vecWidth < cols) {
        // e.g. mat4*vec3: the vec ends with an implicit 1, so this picks up the translation column
        terms = [...terms, ...loadComponent(matAddr, "float", (vecWidth * rows + row) * 8), WASM_OP.f64Add];
      }
      out.push(...storeComponent(addr, "float", row * 8, terms));
    }
    return out;
  }

  /** cross(a, b): the standard component formula (vec3 only). */
  function emitCrossStores(node: any, addr: number): number[] {
    const [a, b] = node.params;
    const width = componentCountOf(node._t as string);
    if (width !== 3) {
      throw new Error(`[RMSL] compileWasmFn: cross() needs a vec3, got width ${width}`);
    }
    const out = [...materializeIfNeeded(a), ...materializeIfNeeded(b)];
    const aAddr = nodeAddress(a),
      bAddr = nodeAddress(b);
    const load = (n: number, k: number) => loadComponent(n, "float", k * 8);
    const term = (a0: number[], b0: number[], a1: number[], b1: number[]) => [
      ...a0,
      ...b0,
      WASM_OP.f64Mul,
      ...a1,
      ...b1,
      WASM_OP.f64Mul,
      WASM_OP.f64Sub,
    ];
    out.push(...storeComponent(addr, "float", 0, term(load(aAddr, 1), load(bAddr, 2), load(aAddr, 2), load(bAddr, 1))));
    out.push(...storeComponent(addr, "float", 8, term(load(aAddr, 2), load(bAddr, 0), load(aAddr, 0), load(bAddr, 2))));
    out.push(
      ...storeComponent(addr, "float", 16, term(load(aAddr, 0), load(bAddr, 1), load(aAddr, 1), load(bAddr, 0))),
    );
    return out;
  }

  /** v / |v| with a zero-length guard (f64 division of 0 traps): keeps the source vector. */
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
      const term = [
        ...loadComponent(srcAddr, kind, k * compSize),
        ...loadComponent(srcAddr, kind, k * compSize),
        WASM_OP.f64Mul,
      ];
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

  /** reflect(i, n) = i - 2*dot(n, i)*n; the dot is cached in the slot right after the value. */
  function emitReflectStores(node: any, addr: number): number[] {
    const [i, n] = node.params;
    const kind = elementKindOf(node._t as string);
    const compSize = componentSizeOf(kind);
    const width = componentCountOf(node._t as string);
    const dotAddr = addr + width * compSize;
    const out = [...materializeIfNeeded(i), ...materializeIfNeeded(n)];
    const iAddr = nodeAddress(i),
      nAddr = nodeAddress(n);
    let dot: number[] = [];
    for (let k = 0; k < width; k++) {
      const term = [
        ...loadComponent(nAddr, kind, k * compSize),
        ...loadComponent(iAddr, kind, k * compSize),
        WASM_OP.f64Mul,
      ];
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

  /** Loads component k, materializing the aggregate into memory first if needed. */
  function readComponent(node: any, k: number): number[] {
    const out = [...materializeIfNeeded(node)];
    const kind = elementKindOf(node._t as string);
    out.push(...loadComponent(nodeAddress(node), kind, k * componentSizeOf(kind)));
    return out;
  }

  /** f64 op for float operands, the i32 op otherwise — int/uint/bool share bit patterns for add/sub/mul. */
  function binaryArith(node: any, f64op: number, i32op: number): number[] {
    const op = scalarKindOf(node.params[0]._t) === "float" ? f64op : i32op;
    return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), op];
  }

  /** Comparisons pick the unsigned opcodes for uints; bools compare as ints. */
  function comparison(node: any, f64op: number, i32sOp: number, i32uOp: number): number[] {
    const kind = scalarKindOf(node.params[0]._t);
    const op = kind === "float" ? f64op : kind === "uint" ? i32uOp : i32sOp;
    return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), op];
  }

  /** float min/max use native f64.min/max; int/uint fall back to a compare + select. */
  function minOrMax(a: any, b: any, kind: ScalarKind, pick: "min" | "max"): number[] {
    if (kind === "float") {
      return [...walkExpr(a), ...walkExpr(b), pick === "min" ? WASM_OP.f64Min : WASM_OP.f64Max];
    }
    const cmp =
      kind === "uint"
        ? pick === "min"
          ? WASM_OP.i32LtU
          : WASM_OP.i32GtU
        : pick === "min"
          ? WASM_OP.i32LtS
          : WASM_OP.i32GtS;
    return selectExpr(walkExpr(a), walkExpr(b), [...walkExpr(a), ...walkExpr(b), cmp]);
  }

  function emitSmoothstepValue(e0Bytes: number[], e1Bytes: number[], xBytes: number[]): number[] {
    // result is t^2 * (3 - 2t). localTee caches the clamped t in the shared
    // $smoothstep_t local so the caller's x bytes are never re-evaluated.
    const tSlot = localSlotIndex("$smoothstep_t");
    const rawT = [...xBytes, ...e0Bytes, WASM_OP.f64Sub, ...e1Bytes, ...e0Bytes, WASM_OP.f64Sub, WASM_OP.f64Div];
    const clampedT = minMaxBytes(minMaxBytes(rawT, f64ConstBytes(0), "float", "max"), f64ConstBytes(1), "float", "min");
    const computeAndTee = [...clampedT, WASM_OP.localTee, ...wasmUleb128(tSlot)];
    const getT = [WASM_OP.localGet, ...wasmUleb128(tSlot)];
    return [
      ...computeAndTee,
      ...getT,
      WASM_OP.f64Mul,
      ...f64ConstBytes(3),
      ...f64ConstBytes(2),
      ...getT,
      WASM_OP.f64Mul,
      WASM_OP.f64Sub,
      WASM_OP.f64Mul,
    ];
  }

  /**
   * Expression pass: evaluates a node down to one value on the WASM stack —
   * an f64 for floats, an i32 for int/uint/bool. A component of an aggregate
   * is read through readComponent/materialize, never held on the stack.
   */
  function walkExpr(node: any): number[] {
    switch (node.type) {
      case "float":
        return f64ConstBytes(node.value);
      case "int":
      case "uint":
        return i32ConstBytes(node.value);
      case "bool":
        return i32ConstBytes(node.value ? 1 : 0);
      case "var":
        if (fnParamNames.has(node.value.varName)) {
          return [WASM_OP.localGet, ...wasmUleb128(paramSlotIndex(`param:${node.value.varName}`))];
        }
        return [WASM_OP.localGet, ...wasmUleb128(localSlotIndex(node.value.varName))];
      case "uniform":
        return [WASM_OP.localGet, ...wasmUleb128(paramSlotIndex(`uniform:${node.value.slot}`))];
      case "attribute":
        return [WASM_OP.localGet, ...wasmUleb128(paramSlotIndex(`attribute:${node.value.slot}`))];
      case "invocationIndex":
        return [WASM_OP.localGet, ...wasmUleb128(paramSlotIndex("invocationIndex"))];
      case "storage":
        return loadComponent(storageAddress.get(node.value.slot)!, scalarKindOf(node._t), 0);
      // The index selects nothing here: this call already is one element
      // (see nodeAddress's "storageElement" case), so it is not evaluated —
      // RMSL index expressions are pure, unlike a general subexpression.
      case "storageElement":
        return loadComponent(nodeAddress(node.params[0]), scalarKindOf(node._t), 0);
      case "varying":
        if (effectiveStage === "fragment") {
          return [WASM_OP.localGet, ...wasmUleb128(paramSlotIndex(`varying:${node.value.slot}`))];
        }
        return loadComponent(varyingOutputAddress.get(node.value.slot)!, scalarKindOf(node._t), 0); // vertex: re-read our own written output
      case "output":
        return loadComponent(outputAddress.get(node.value.slot)!, scalarKindOf(node._t), 0); // read back a previously written output
      case "builtinFragDepth":
        return loadComponent(fragDepthAddress!, "float", 0);

      case "add":
        return binaryArith(node, WASM_OP.f64Add, WASM_OP.i32Add);
      case "sub":
        return binaryArith(node, WASM_OP.f64Sub, WASM_OP.i32Sub);
      case "mul":
        return binaryArith(node, WASM_OP.f64Mul, WASM_OP.i32Mul);
      case "div": {
        const kind = scalarKindOf(node.params[0]._t);
        if (kind === "float") return binaryArith(node, WASM_OP.f64Div, WASM_OP.f64Div);
        return [
          ...walkExpr(node.params[0]),
          ...walkExpr(node.params[1]),
          kind === "uint" ? WASM_OP.i32DivU : WASM_OP.i32DivS,
        ];
      }
      case "mod": {
        const kind = scalarKindOf(node.params[0]._t);
        if (kind !== "float") {
          return [
            ...walkExpr(node.params[0]),
            ...walkExpr(node.params[1]),
            kind === "uint" ? WASM_OP.i32RemU : WASM_OP.i32RemS,
          ];
        }

        const a = node.params[0],
          b = node.params[1];
        // floored modulo (GLSL): a - b*floor(a/b); a plain f64.rem would truncate toward zero
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
      case "min":
        return minOrMax(node.params[0], node.params[1], scalarKindOf(node.params[0]._t), "min");
      case "max":
        return minOrMax(node.params[0], node.params[1], scalarKindOf(node.params[0]._t), "max");

      case "clamp": {
        const kind = scalarKindOf(node._t as string);
        const x = walkExpr(node.params[0]);
        const lo = walkExpr(node.params[1]);
        const hi = walkExpr(node.params[2]);
        return minMaxBytes(minMaxBytes(x, lo, kind, "max"), hi, kind, "min");
      }
      case "mix": {
        const [a, b, t] = node.params;
        return [
          ...walkExpr(a),
          ...walkExpr(t),
          ...walkExpr(b),
          ...walkExpr(a),
          WASM_OP.f64Sub,
          WASM_OP.f64Mul,
          WASM_OP.f64Add,
        ];
      }
      case "step": {
        const [edge, x] = node.params;
        return selectExpr(f64ConstBytes(0), f64ConstBytes(1), [...walkExpr(x), ...walkExpr(edge), WASM_OP.f64Lt]);
      }
      case "smoothstep": {
        const [e0, e1, x] = node.params;
        return emitSmoothstepValue(walkExpr(e0), walkExpr(e1), walkExpr(x));
      }
      case "select": {
        const [cond, a, b] = node.params;
        return selectExpr(walkExpr(a), walkExpr(b), walkExpr(cond));
      }

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
        const gtZero =
          kind === "float"
            ? [...walkExpr(x), ...f64ConstBytes(0), WASM_OP.f64Gt]
            : [...walkExpr(x), ...i32ConstBytes(0), kind === "uint" ? WASM_OP.i32GtU : WASM_OP.i32GtS];
        const ltZero =
          kind === "float"
            ? [...walkExpr(x), ...f64ConstBytes(0), WASM_OP.f64Lt]
            : [...walkExpr(x), ...i32ConstBytes(0), kind === "uint" ? WASM_OP.i32LtU : WASM_OP.i32LtS];
        const positiveOrZero = selectExpr(one, zero, gtZero);
        return selectExpr(minusOne, positiveOrZero, ltZero);
      }
      case "floor":
        return [...walkExpr(node.params[0]), WASM_OP.f64Floor];
      case "ceil":
        return [...walkExpr(node.params[0]), WASM_OP.f64Ceil];
      case "trunc":
        return [...walkExpr(node.params[0]), WASM_OP.f64Trunc];
      case "fract": {
        const x = node.params[0];
        // x - floor(x); x is emitted twice to keep the stack flat — cheap, side-effect free
        return [...walkExpr(x), ...walkExpr(x), WASM_OP.f64Floor, WASM_OP.f64Sub];
      }
      case "round": {
        // GLSL round = floor(x + 0.5), not f64.nearest (which rounds half-to-even)
        return [...walkExpr(node.params[0]), ...f64ConstBytes(0.5), WASM_OP.f64Add, WASM_OP.f64Floor];
      }
      case "sqrt":
        return [...walkExpr(node.params[0]), WASM_OP.f64Sqrt];
      case "inverseSqrt":
        return [...f64ConstBytes(1), ...walkExpr(node.params[0]), WASM_OP.f64Sqrt, WASM_OP.f64Div];
      case "exp2":
        return [...f64ConstBytes(2), ...walkExpr(node.params[0]), ...callImport("pow")]; // exp2(x) = pow(2, x)

      case "sin":
      case "cos":
      case "tan":
      case "asin":
      case "acos":
      case "atan":
      case "sinh":
      case "cosh":
      case "tanh":
      case "asinh":
      case "acosh":
      case "atanh":
      case "exp":
      case "log":
      case "log2":
        return [...walkExpr(node.params[0]), ...callImport(node.type)];
      case "pow":
      case "atan2":
        return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), ...callImport(node.type)];

      case "lessThan":
        return comparison(node, WASM_OP.f64Lt, WASM_OP.i32LtS, WASM_OP.i32LtU);
      case "greaterThan":
        return comparison(node, WASM_OP.f64Gt, WASM_OP.i32GtS, WASM_OP.i32GtU);
      case "lessThanEqual":
        return comparison(node, WASM_OP.f64Le, WASM_OP.i32LeS, WASM_OP.i32LeU);
      case "greaterThanEqual":
        return comparison(node, WASM_OP.f64Ge, WASM_OP.i32GeS, WASM_OP.i32GeU);
      case "equal":
        return comparison(node, WASM_OP.f64Eq, WASM_OP.i32Eq, WASM_OP.i32Eq);
      case "notEqual":
        return comparison(node, WASM_OP.f64Ne, WASM_OP.i32Ne, WASM_OP.i32Ne);

      case "and":
        return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), WASM_OP.i32And];
      case "or":
        return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), WASM_OP.i32Or];
      case "not":
        return [...walkExpr(node.params[0]), WASM_OP.i32Eqz];

      case "bitAnd":
        return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), WASM_OP.i32And];
      case "bitOr":
        return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), WASM_OP.i32Or];
      case "bitXor":
        return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), WASM_OP.i32Xor];
      case "bitNot":
        return [...walkExpr(node.params[0]), ...i32ConstBytes(-1), WASM_OP.i32Xor];
      case "shiftLeft":
        return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), WASM_OP.i32Shl];
      case "shiftRight": {
        const kind = scalarKindOf(node.params[0]._t);
        return [
          ...walkExpr(node.params[0]),
          ...walkExpr(node.params[1]),
          kind === "uint" ? WASM_OP.i32ShrU : WASM_OP.i32ShrS,
        ];
      }

      case "construct": {
        // scalar casts: float<->int via trunc/convert; bool tests "!= 0"; int/uint need no conversion
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

        if (targetKind === "bool") return [...bytes, ...i32ConstBytes(0), WASM_OP.i32Ne];
        return bytes;
      }

      case "swizzle": {
        // single-component swizzle evaluates as a scalar component load
        const pattern = node.value as string;
        if (pattern.length !== 1) {
          throw new Error(
            '[RMSL] compileWasmFn: unsupported node type in expression position: "swizzle" (multi-component)',
          );
        }
        return readComponent(node.params[0], COMPONENT_INDEX[pattern]);
      }

      case "dot": {
        const a = node.params[0],
          b = node.params[1];
        const width = componentCountOf(a._t);
        const pre = [...materializeIfNeeded(a), ...materializeIfNeeded(b)];
        const aAddr = nodeAddress(a);
        const bAddr = nodeAddress(b);
        let acc: number[] = [];
        for (let k = 0; k < width; k++) {
          const term = [
            ...loadComponent(aAddr, "float", k * 8),
            ...loadComponent(bAddr, "float", k * 8),
            WASM_OP.f64Mul,
          ];
          acc = k === 0 ? term : [...acc, ...term, WASM_OP.f64Add];
        }
        return [...pre, ...acc];
      }
      case "dFdx":
      case "dFdy":
      case "fwidth": {
        assertDerivativesAllowed(node);
        const kind = scalarKindOf(node._t); // derivatives always evaluate to zero here
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
        const a = node.params[0],
          b = node.params[1];
        const width = componentCountOf(a._t);
        const pre = [...materializeIfNeeded(a), ...materializeIfNeeded(b)];
        const aAddr = nodeAddress(a);
        const bAddr = nodeAddress(b);
        let sumSq: number[] = [];
        for (let k = 0; k < width; k++) {
          const diff = [
            ...loadComponent(aAddr, "float", k * 8),
            ...loadComponent(bAddr, "float", k * 8),
            WASM_OP.f64Sub,
          ];
          const term = [...diff, ...diff, WASM_OP.f64Mul];
          sumSq = k === 0 ? term : [...sumSq, ...term, WASM_OP.f64Add];
        }
        return [...pre, ...sumSq, WASM_OP.f64Sqrt];
      }
      case "uniformArrayElement": {
        const info = uniformArrayInfo.get(node.params[0].value.slot);
        if (info === undefined) {
          throw new Error(
            `[RMSL] compileWasmFn: internal error, unaddressed uniform array "${node.params[0].value.slot}"`,
          );
        }
        const type = node._t as string;
        const kind = isAggregate(type) ? elementKindOf(type) : scalarKindOf(type);
        const index = node.params[1];
        const indexBytes =
          scalarKindOf(index._t as string) === "float" ? [...walkExpr(index), WASM_OP.i32TruncF64S] : walkExpr(index);
        const addrBytes = uniformArrayElementAddress(info.base, info.elementStride, indexBytes);
        if (info.narrow && kind === "float") {
          return [...addrBytes, WASM_OP.f32Load, 0x00, 0x00, WASM_OP.f64PromoteF32];
        }
        return loadDynamic(addrBytes, kind);
      }
      case "seq": {
        // a nested Fn call's result: its captured statements, then its final
        // value evaluated as this expression's value.
        const stmts = node.params.slice(0, -1) as any[];
        const final = node.params[node.params.length - 1];
        return [...stmts.flatMap((s) => walkStmt(s, currentStmtDepth)), ...walkExpr(final)];
      }

      default:
        throw new Error(`[RMSL] compileWasmFn: unsupported node type in expression position: "${node.type}"`);
    }
  }

  /**
   * Lowers a for/while loop to block{loop{cond; brIf <exit>; block{body};
   * update; br <top>}}. break exits via the outer block, continue via the
   * loop label (which re-runs the update). The recorded break/continue
   * depths account for the two extra nested levels inside the body.
   */
  function emitLoop(
    initBytes: number[],
    condNode: any,
    bodyNode: any,
    updateNode: any | null,
    depth: number,
  ): number[] {
    const breakDepth = depth + 1;
    const continueDepth = depth + 3;
    loopStack.push({ breakDepth, continueDepth });
    const condBytes = walkExpr(condNode);
    const bodyBytes = walkStmt(bodyNode, continueDepth);
    const updateBytes = updateNode ? walkStmt(updateNode, depth + 2) : [];
    loopStack.pop();
    return [
      ...initBytes,
      WASM_OP.block,
      WASM_BLOCKTYPE_VOID,
      WASM_OP.loop,
      WASM_BLOCKTYPE_VOID,
      ...condBytes,
      WASM_OP.i32Eqz,
      WASM_OP.brIf,
      ...wasmUleb128(1),
      WASM_OP.block,
      WASM_BLOCKTYPE_VOID,
      ...bodyBytes,
      WASM_OP.end,
      ...updateBytes,
      WASM_OP.br,
      ...wasmUleb128(0),
      WASM_OP.end,
      WASM_OP.end,
    ];
  }

  /**
   * Statement pass: emits code that leaves no net value on the stack, and
   * records the depth so break/continue/return can compute relative branch
   * targets.
   */
  function walkStmt(node: any, depth: number): number[] {
    currentStmtDepth = depth;
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
            out.push(
              ...storeComponent(
                baseAddr,
                kind,
                COMPONENT_INDEX[ch] * compSize,
                loadComponent(rhsAddr, kind, i * compSize),
              ),
            );
          });
          return out;
        }

        let targetType: string;
        let destAddr: number;
        if (target.type === "output") {
          targetType = target._t as string;
          destAddr = outputAddress.get(target.value.slot)!;
        } else if (target.type === "storageElement") {
          targetType = target._t as string;
          destAddr = nodeAddress(target.params[0]);
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
          return storeComponent(destAddr, scalarKindOf(targetType), 0, walkExpr(rhs));
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

        const thenBytes = walkStmt(node.params[1], depth + 1);
        const elseNode = node.params[2];
        return [
          ...cond,
          WASM_OP.if_,
          WASM_BLOCKTYPE_VOID,
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
        // discard has no distinct "no fragment output" meaning yet: it leaves
        // early exactly like return, with the same exit sentinel.
        const sentinel = needsResult ? [] : resultKind === "float" ? f64ConstBytes(0) : i32ConstBytes(0); // needsResult exits a void block; otherwise carry a zero for the block's result type
        return [...sentinel, WASM_OP.br, ...wasmUleb128(depth - EXIT_BLOCK_DEPTH)];
      }
      default:
        throw new Error(`[RMSL] compileWasmFn: unsupported node type in statement position: "${node.type}"`);
    }
  }

  /** Puts the function's result where expected: the stack (scalar mode) or valueAddress (aggregate/stage mode). */
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

  // import signatures are shared and deduped: (f64)->f64 and (f64,f64)->f64
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
}

/**
 * Instantiates a compiled module and binds it to JS: marshals params and
 * textures into memory/args, calls the function, and reads results back
 * into a CpuShaderResult.
 *
 * Split out from `compileWasm` so a build-time precompile step (see
 * `precompileWasm` in `../vite/vite.ts`) can ship just the compiled bytes
 * and this instantiation glue — never the graph builder or bytecode
 * emitter that produced them.
 */
export function instantiateWasm(
  compiled: CompiledWasm,
  name: string,
  externalMemory?: WebAssembly.Memory,
): CpuRenderer {
  const { bytes, params, resultType, textureHeapBase, memoryPages, sharedMemory, maxMemoryPages, draw } = compiled;

  // no memory passed in: own one, sized for the compile-time layout, growable
  // as textures/draw buffers are marshalled in — same behavior as before this
  // module imported (rather than defined) its memory. Its shared-ness must
  // match what the module declared at compile time (see `sharedMemory`).
  const memory =
    externalMemory ??
    new WebAssembly.Memory(
      sharedMemory ? { initial: memoryPages, maximum: maxMemoryPages, shared: true } : { initial: memoryPages },
    );

  const instance = new WebAssembly.Instance(new WebAssembly.Module(bytes.buffer as ArrayBuffer), {
    math: Math as unknown as WebAssembly.ModuleImports, // host "math" namespace serving the sin/pow/... imports
    env: { memory },
  });
  const wasmMain = instance.exports[name] as (...args: number[]) => number;
  const wasmDraw = draw ? (instance.exports.draw as (...args: number[]) => void) : undefined;

  let view = new DataView(memory.buffer);

  // outputs read back after each call; textures repacked per call
  const outputParams = params.filter(
    (
      p,
    ): p is Extract<
      WasmParam,
      { kind: "outputMemory" | "varyingOutputMemory" | "positionMemory" | "fragDepthMemory" | "valueMemory" }
    > =>
      p.kind === "outputMemory" ||
      p.kind === "varyingOutputMemory" ||
      p.kind === "positionMemory" ||
      p.kind === "fragDepthMemory" ||
      p.kind === "valueMemory",
  );
  const textureParams = params.filter(
    (p): p is Extract<WasmParam, { kind: "textureMemory" }> => p.kind === "textureMemory",
  );

  // texture cache: skip re-uploading an unchanged texture object, and only
  // grow the module memory when the total footprint changes between calls
  const lastTexture: (CpuTextureData | undefined)[] = new Array(textureParams.length);
  let lastSizes: number[] | null = null;

  /**
   * Appends each texture's pixels after the compiled layout (growing memory
   * when the total footprint changes) and collects the scalar WASM args.
   * Returns the heap end — where the draw buffer starts.
   */
  function marshalInputs(ctx: CpuShaderContext): { args: number[]; textureHeapEnd: number } {
    let textureHeapEnd = textureHeapBase;
    if (textureParams.length > 0) {
      const textures = textureParams.map((p) => (ctx.textures as any)?.[p.slot] as CpuTextureData);
      const sizes = textures.map((tex, i) => textureByteSize(tex, textureParams[i]!.samplerType.endsWith("Cube")));
      const heapOffsets: number[] = [];
      let heapCursor = textureHeapBase;
      for (const size of sizes) {
        heapOffsets.push(heapCursor);
        heapCursor += size;
      }
      textureHeapEnd = heapCursor;

      const needsRepack = lastSizes === null || sizes.some((s, i) => s !== lastSizes![i]);
      if (needsRepack && heapCursor > memory.buffer.byteLength) {
        memory.grow(Math.ceil((heapCursor - memory.buffer.byteLength) / 65536));
        view = new DataView(memory.buffer); // growing detaches the old buffer, so the view is rebuilt
      }
      textureParams.forEach((p, i) => {
        if (!needsRepack && textures[i] === lastTexture[i]) return;
        writeTextureToMemory(view, p.metadataAddress, heapOffsets[i], textures[i], p.samplerType.endsWith("Cube"));
        lastTexture[i] = textures[i];
      });
      lastSizes = sizes;
    }
    const args: number[] = [];
    for (const p of params) {
      switch (p.kind) {
        case "textureMemory":
          break;
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
        case "invocationIndex":
          args.push(ctx.index ?? 0);
          break;
        case "paramMemory":
          writeAggregateToMemory(view, p.address, p.shaderType, (ctx.params as any)?.[p.name]);
          break;
        case "uniformMemory":
          writeAggregateToMemory(view, p.address, p.shaderType, (ctx.uniforms as any)?.[p.slot], p.narrow);
          break;
        case "uniformArrayMemory":
          writeArrayToMemory(
            view,
            p.address,
            p.shaderType,
            p.length,
            (ctx.uniforms as any)?.[p.slot],
            p.elementStride,
            p.narrow,
          );
          break;
        case "attributeMemory":
          writeAggregateToMemory(view, p.address, p.shaderType, (ctx.attributes as any)?.[p.slot]);
          break;
        case "varyingMemory":
          writeAggregateToMemory(view, p.address, p.shaderType, (ctx.varyings as any)?.[p.slot]);
          break;
        case "fragCoordMemory":
          // the host's fragCoord for CPU invocations; draw() overwrites it per pixel — harmless
          writeAggregateToMemory(view, p.address, "vec2", ctx.fragCoord ?? [0, 0]);
          break;
        case "storageMemory":
          if (p.access !== "write") {
            writeValueToMemory(view, p.address, p.shaderType, (ctx.storages as any)?.[p.slot]?.[ctx.index ?? 0]);
          }
          break;
      }
    }
    return { args, textureHeapEnd };
  }

  const storageOutputParams = params.filter(
    (p): p is Extract<WasmParam, { kind: "storageMemory" }> => p.kind === "storageMemory" && p.access !== "read",
  );

  function callable(ctx: CpuShaderContext): number | boolean | CpuShaderResult {
    const { args } = marshalInputs(ctx);
    const result = wasmMain(...args);

    // A read_write/write storage() is mutated in the caller's own array at
    // ctx.index, the same convention compileJS's storage support uses —
    // not folded into shaderResult, since the point is the array itself
    // stays the source of truth across calls.
    for (const p of storageOutputParams) {
      const arr = (ctx.storages as any)?.[p.slot];
      if (arr) arr[ctx.index ?? 0] = readValueFromMemory(view, p.address, p.shaderType);
    }

    // scalar mode: reinterpret the raw i32 — the WASM boundary returns it
    // signed, so a uint result needs a >>> 0 re-read
    if (outputParams.length === 0) {
      if (resultType === "bool") return result !== 0;
      if (resultType === "uint") return result >>> 0;
      return result;
    }

    const shaderResult: CpuShaderResult = {};
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
  }

  callable.draw = (ctx: CpuShaderContext, width: number, height: number, out?: CpuDrawBuffer): CpuDrawBuffer => {
    if (!draw || !wasmDraw) {
      throw new Error(
        '[RMSL] compileWasm: this function produces no value to render — draw() needs a non-"void" result.',
      );
    }
    const { args, textureHeapEnd } = marshalInputs(ctx);
    const pixelCount = width * height * draw.componentCount;

    // `out` backed by this instance's own (shared) memory: write directly at
    // its offset — this is the zero-copy multi-worker path, where each
    // worker's instance imports the same SharedArrayBuffer-backed memory and
    // `out` is a view pinning where in it this call should land.
    if (out && out.buffer === memory.buffer) {
      wasmDraw(...args, width, height, out.byteOffset);
      return out;
    }

    const bufferBase = Math.ceil(textureHeapEnd / 8) * 8; // align to 8 bytes — the typed-array constructors require it
    const neededBytes = bufferBase + pixelCount * componentSizeOf(draw.kind);
    if (neededBytes > memory.buffer.byteLength) {
      memory.grow(Math.ceil((neededBytes - memory.buffer.byteLength) / 65536));
      view = new DataView(memory.buffer); // growing detaches the old buffer, so the view is rebuilt
    }
    wasmDraw(...args, width, height, bufferBase);

    // `out` backed by a different buffer than this instance's memory: wasm
    // can only write into the memory it was instantiated with, so this has
    // to copy rather than return a view straight into wasm memory.
    if (out) {
      out.set(
        draw.kind === "float"
          ? new Float64Array(memory.buffer, bufferBase, pixelCount)
          : draw.kind === "uint"
            ? new Uint32Array(memory.buffer, bufferBase, pixelCount)
            : new Int32Array(memory.buffer, bufferBase, pixelCount),
      );
      return out;
    }

    if (draw.kind === "float") return new Float64Array(memory.buffer, bufferBase, pixelCount);
    if (draw.kind === "uint") return new Uint32Array(memory.buffer, bufferBase, pixelCount);
    return new Int32Array(memory.buffer, bufferBase, pixelCount);
  };

  return callable;
}

/** Compiles an `Fn` to WASM and instantiates it in one step — see `instantiateWasm`. */
export function compileWasm(
  fn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  options: CompileWasmFnOptions,
): CpuRenderer {
  return instantiateWasm(compileWasmFn(fn, options), options.name, options.memory);
}
