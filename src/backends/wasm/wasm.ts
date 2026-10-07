import { MATRIX_DIMENSIONS, Node, ShaderType, StorageAccess } from "../../core";
import { AllocRules, planLayout } from "../../layout";
import {
  componentCountOf,
  CpuDrawBuffer,
  CpuRoutine,
  CpuGrid,
  CpuProgram,
  GridBuffer,
  ComputeStage,
  FragmentStage,
  VertexStage,
  toFragmentResult,
  toVertexResult,
  typedArrayOf,
  typedArrayOfKind,
  FloatWidth,
  CpuShaderContext,
  CpuProgramResult,
  CpuTextureData,
  CpuTextureWrap,
  elementKindOf,
  isAggregate,
  ScalarKind,
  scalarKindOf,
} from "../cpu";
import {
  assertNotInAComputeStage,
  COMPUTE_REFUSES,
  assertStageResult,
  prepareRoots,
  assertReadsNoStageInput,
  assertOneDeclarationPerName,
  numberClashingVariables,
  assertAssignable,
  FOR_UPDATE_BLOCK_MESSAGE,
  assertLiteralIndexInRange,
  DEGREES_PER_RADIAN,
  RADIANS_PER_DEGREE,
  assignedStorageElement,
  parameterNode,
  CompileFnOptions,
  COMPONENT_INDEX,
  isLeafLiteral,
  resolveSwizzleTarget,
} from "../shared";
import { shareNodes } from "../share";
import {
  f64ConstBytes,
  forLoop,
  i32ConstBytes,
  iGeS,
  local,
  WASM_BLOCKTYPE_VOID,
  WASM_F64,
  WASM_FUNC,
  WASM_I32,
  I32_TRUNC_SAT_F64_S,
  I32_TRUNC_SAT_F64_U,
  WASM_OP,
  wasmF64Bytes,
  wasmSection,
  wasmStrBytes,
  wasmUleb128,
  wasmVec,
} from "./utils";

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
  | { kind: "discardMemory"; address: number }
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
       * A `storage()` slot. The whole buffer is copied into the heap past
       * `textureHeapBase` before a call or compute dispatch, and a writable one is
       * copied back out after; `metadataAddress` is where the buffer's heap
       * address and element count are written for the compiled code to read.
       */
      kind: "storageMemory";
      slot: string;
      shaderType: ShaderType;
      metadataAddress: number;
      access: StorageAccess;
      /** Whether the program assigns to the buffer; only then is it copied back after a call. */
      written: boolean;
    };

/**
 * A compiled module: raw bytes plus the contract the host uses to marshal
 * values, and "draw" metadata when whole-image evaluation is available.
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
  /** Whether the module exports `compute(...params, count)`, which runs `main` once per invocation index. */
  compute?: boolean;
  /** Whether the program computes every float in 32 bits, so the host rounds each float input it passes. */
  float32?: boolean;
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
export type CompileWasmFnOptions = CompileFnOptions & WasmCompileFields & WasmFloatWidth;

/**
 * The width a WASM program computes a `float` in, and the uniform layout it
 * may share with WGSL. `"f64"` is the default. `"f32"` rounds every float
 * value to 32 bits, as a GPU holds it. A `gpuUniformLayout` places uniforms
 * where a WGSL buffer of 32-bit floats has them, so it needs `"f32"`.
 */
export type WasmFloatWidth =
  { float?: FloatWidth; gpuUniformLayout?: never } | { float: "f32"; gpuUniformLayout?: GpuUniformLayout };

/**
 * The width part of `options`, for passing on as it is: a layout given at 64
 * bits reaches the compile, which refuses it.
 */
export function floatWidthOf(options: WasmFloatWidth): WasmFloatWidth {
  return { float: options.float, gpuUniformLayout: options.gpuUniformLayout } as WasmFloatWidth;
}

/** The options of a WASM compile beside its name, parameters and float width. */
export type WasmCompileFields = {
  stage?: "vertex" | "fragment" | "compute";

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

  /**
   * Byte offset the compile-time bump allocator starts from, instead of 0.
   * Exists so several independently-compiled modules can share one
   * `memory` without their compile-time-fixed addresses (uniforms,
   * scratch slots, texture metadata, ...) colliding — each module gets its
   * own non-overlapping region by being compiled with a different
   * `memoryBase`. The caller is responsible for choosing non-overlapping
   * bases (typically: compile each module once to learn how much space its
   * layout needs, then lay the next one out after it). Combining with
   * `gpuUniformLayout` adds this on top of that layout's own reserved
   * region, rather than replacing it.
   */
  memoryBase?: number;

  /**
   * Forces every scalar uniform/attribute/fragment-stage `varying()` to be
   * memory-resident, the same as an aggregate one, instead of a WASM
   * function parameter. Exists for callers (like the generic rasterizer
   * module) that call the compiled function with a fixed, zero-argument
   * signature and can't vary it per program.
   */
  scalarsInMemory?: boolean;
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

// Per-storage metadata block written by the input marshaller, reserved per
// storage slot: the heap address of the buffer's first element, then how
// many elements it holds.
const STORAGE_META_DATA_ADDR = 0;
const STORAGE_META_LENGTH = 4;
const STORAGE_META_STRIDE = 8;

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

/**
 * The WASM value type byte a scalar kind is stored as. Everything here is
 * either `f64` (`WASM_F64`, 0x7c) or `i32` (`WASM_I32`, 0x7f) — int/uint/bool
 * all share the i32 representation, distinguished only by how the surrounding
 * code interprets the bits (see {@link convertComponent}).
 */
function wasmTypeOf(kind: ScalarKind): number {
  return kind === "float" ? WASM_F64 : WASM_I32;
}

/** Bytes per component: 8 for float, 4 for int/uint/bool. */
function componentSizeOf(kind: ScalarKind): number {
  return kind === "float" ? 8 : 4;
}

/**
 * True for any sampler type, float or integer variant.
 */
function isSamplerType(t: string): boolean {
  return t.startsWith("sampler") || t.startsWith("isampler") || t.startsWith("usampler");
}

/**
 * True only for the integer sampler variants (`isampler*`/`usampler*`).
 */
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

/** WASM select pops [whenTrue, whenFalse, cond]; note BOTH value arms are always evaluated. */
function selectExpr(whenTrue: number[], whenFalse: number[], cond: number[]): number[] {
  return [...whenTrue, ...whenFalse, ...cond, WASM_OP.select];
}

/**
 * Loads one component at a constant address: push the address, then the
 * load opcode and its "memarg" — two unsigned LEB128 integers, alignment
 * hint first, then a byte offset added to the popped address at run time.
 * The alignment hint is always `0x00` (no assumed alignment) here, since
 * this compiler never proves an access is aligned; a wrong hint doesn't trap
 * or corrupt data, but is technically UB, so `0x00` is the only value it's
 * safe to always emit.
 */
function loadComponent(addr: number, kind: ScalarKind, byteOffset: number): number[] {
  return [
    ...i32ConstBytes(addr),
    kind === "float" ? WASM_OP.f64Load : WASM_OP.i32Load,
    0x00,
    ...wasmUleb128(byteOffset),
  ];
}

/**
 * Stores one component at a constant address: push the address, then the
 * value, then the store opcode and its memarg (alignment hint, always
 * `0x00` here — see {@link loadComponent} — then a byte offset).
 */
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

/** The largest `int` and `uint` a 32-bit float holds exactly, which a float converted to an integer clamps to. */
const INT_MAX_OF_F32 = 2147483520;
const UINT_MAX_OF_F32 = 4294967040;

/**
 * An f64 truncated toward zero to an `int` or `uint`, clamped to the range WebGPU clamps to, with NaN as 0. The
 * clamp runs on the f64 first, since the saturating truncation alone stops at the i32 range.
 */
function floatToInteger(valueBytes: number[], kind: "int" | "uint"): number[] {
  const low = kind === "uint" ? 0 : -2147483648;
  const high = kind === "uint" ? UINT_MAX_OF_F32 : INT_MAX_OF_F32;
  return [
    ...valueBytes,
    ...f64ConstBytes(low),
    WASM_OP.f64Max,
    ...f64ConstBytes(high),
    WASM_OP.f64Min,
    ...(kind === "uint" ? I32_TRUNC_SAT_F64_U : I32_TRUNC_SAT_F64_S),
  ];
}

/**
 * Casts one scalar to another kind: f64<->i32 via trunc/convert. A bool is an
 * i32 0 or 1, so it converts to a number as it is and from one by `!= 0`.
 */
function convertComponent(valueBytes: number[], fromKind: ScalarKind, toKind: ScalarKind): number[] {
  if (fromKind === toKind) return valueBytes;
  if (toKind === "bool") {
    return fromKind === "float"
      ? [...valueBytes, ...f64ConstBytes(0), WASM_OP.f64Ne]
      : [...valueBytes, ...i32ConstBytes(0), WASM_OP.i32Ne];
  }
  if (fromKind === "bool") return toKind === "float" ? [...valueBytes, WASM_OP.f64ConvertI32U] : valueBytes;
  if (fromKind === "float") {
    return floatToInteger(valueBytes, toKind === "uint" ? "uint" : "int");
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
  "storageElement",
  "matrixElement",
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
  "mod",
  "bitAnd",
  "bitOr",
  "bitXor",
  "bitNot",
  "shiftLeft",
  "shiftRight",
  "negate",
  "abs",
  "radians",
  "degrees",
  "min",
  "max",
  "lessThan",
  "greaterThan",
  "lessThanEqual",
  "greaterThanEqual",
  "equal",
  "notEqual",
]);

/** The comparison opcodes for each comparison node: `[float, int, uint]`. */
const COMPARISON_OPCODES: Record<string, [number, number, number]> = {
  lessThan: [WASM_OP.f64Lt, WASM_OP.i32LtS, WASM_OP.i32LtU],
  greaterThan: [WASM_OP.f64Gt, WASM_OP.i32GtS, WASM_OP.i32GtU],
  lessThanEqual: [WASM_OP.f64Le, WASM_OP.i32LeS, WASM_OP.i32LeU],
  greaterThanEqual: [WASM_OP.f64Ge, WASM_OP.i32GeS, WASM_OP.i32GeU],
  equal: [WASM_OP.f64Eq, WASM_OP.i32Eq, WASM_OP.i32Eq],
  notEqual: [WASM_OP.f64Ne, WASM_OP.i32Ne, WASM_OP.i32Ne],
};

/** One component of a comparison, on operands of `kind`. */
function comparisonBytes(type: string, a: number[], b: number[], kind: ScalarKind): number[] {
  const [f64op, i32sOp, i32uOp] = COMPARISON_OPCODES[type]!;
  return [...a, ...b, kind === "float" ? f64op : kind === "uint" ? i32uOp : i32sOp];
}

/**
 * True when `node` is one of the anonymous aggregate results
 * {@link SCRATCH_NODE_TYPES} describes: it needs its own scratch address
 * materialized before any of its components can be read.
 */
function isScratchNode(node: any): boolean {
  const t = node._t as string;
  if (!isAggregate(t)) return false;
  if (node.type === t) return true;
  if (node.type === "swizzle") return (node.value as string).length > 1;
  return SCRATCH_NODE_TYPES.has(node.type);
}

/**
 * A `storage()` node used as a value rather than through `.element(i)`. The
 * JS target hands back the whole array there, which has no WASM value to
 * match.
 */
function bareStorageError(node: any): Error {
  return new Error(
    `[RMSL] compileWasmFn: storage "${node.value.slot}" is read as a whole; read one element with .element(i).`,
  );
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
  round32?: boolean,
): void {
  const kind = elementKindOf(shaderType);

  const compSize = narrow && kind === "float" ? 4 : componentSizeOf(kind);
  const arr = value as ArrayLike<number | boolean>;
  for (let i = 0; i < arr.length; i++) {
    const raw = arr[i];
    const num = typeof raw === "boolean" ? (raw ? 1 : 0) : (raw as number);
    if (kind === "float") {
      if (narrow) view.setFloat32(address + i * compSize, num, true);
      else view.setFloat64(address + i * compSize, round32 ? Math.fround(num) : num, true);
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
  round32?: boolean,
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
        else view.setFloat64(base, round32 ? Math.fround(num) : num, true);
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
          else view.setFloat64(at, round32 ? Math.fround(num) : num, true);
        } else {
          view.setInt32(at, num, true);
        }
      }
    }
  }
}

/** Host-side: reads a vector or matrix back from memory, into the typed array it is held in on the CPU. */
function readAggregateFromMemory(
  view: DataView,
  address: number,
  shaderType: ShaderType,
  float32 = false,
): CpuDrawBuffer {
  const kind = elementKindOf(shaderType);
  const compSize = componentSizeOf(kind);
  const width = componentCountOf(shaderType);
  const out = new (typedArrayOf(shaderType, float32))(width);
  for (let i = 0; i < width; i++) {
    if (kind === "float") out[i] = view.getFloat64(address + i * compSize, true);
    else {
      const raw = view.getInt32(address + i * compSize, true);
      out[i] = kind === "bool" ? (raw !== 0 ? 1 : 0) : kind === "uint" ? raw >>> 0 : raw;
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

/**
 * Host-side: reads a value (scalar or aggregate) from memory — the
 * read-side counterpart of `writeValueToMemory`.
 */
function readValueFromMemory(view: DataView, address: number, shaderType: ShaderType, float32 = false): unknown {
  return isAggregate(shaderType)
    ? readAggregateFromMemory(view, address, shaderType, float32)
    : readScalarFromMemory(view, address, shaderType);
}

/** Host-side: writes one scalar value into memory — writeAggregateToMemory's counterpart for a non-array type. */
function writeScalarToMemory(
  view: DataView,
  address: number,
  shaderType: ShaderType,
  value: unknown,
  narrow?: boolean,
  round32?: boolean,
): void {
  const kind = scalarKindOf(shaderType);
  const num = typeof value === "boolean" ? (value ? 1 : 0) : ((value as number | undefined) ?? 0);
  if (kind === "float") {
    if (narrow) view.setFloat32(address, num, true);
    else view.setFloat64(address, round32 ? Math.fround(num) : num, true);
  } else view.setInt32(address, num, true);
}

/** Host-side: writes a value (scalar or aggregate) into memory — the write-side counterpart of readValueFromMemory. */
function writeValueToMemory(
  view: DataView,
  address: number,
  shaderType: ShaderType,
  value: unknown,
  narrow?: boolean,
  round32?: boolean,
): void {
  if (isAggregate(shaderType)) writeAggregateToMemory(view, address, shaderType, value, narrow, round32);
  else writeScalarToMemory(view, address, shaderType, value, narrow, round32);
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

/** The statements that hold a block of their own. */
const BLOCK_STATEMENTS = new Set(["if", "for", "while"]);

/** Whether a statement, or any it holds, is a block. Each node of the graph is visited once. */
function holdsBlock(node: any, seen = new Set<unknown>()): boolean {
  if (node === null || typeof node !== "object" || seen.has(node)) return false;
  seen.add(node);
  if (BLOCK_STATEMENTS.has(node.type)) return true;
  return Array.isArray(node.params) && node.params.some((param: unknown) => holdsBlock(param, seen));
}

/**
 * Compiles an RMSL function into a small WASM module in two passes:
 * "collect" walks the AST once, giving each scalar a param/local slot and
 * each aggregate a fixed memory address; the emit helpers then generate
 * bytes against that frozen layout. Scalars flow through WASM params/
 * locals; vectors, matrices and pipeline I/O live at fixed offsets in an
 * exported linear memory that the host reads and writes around each call.
 * When the root is an aggregate or the stage writes pipeline outputs, the
 * module declares zero results and values round-trip through the linear
 * memory instead.
 *
 * The opcodes this file emits (see `WASM_OP`) are verified empirically
 * against a known answer, not quoted from memory: a wrong-but-valid opcode
 * runs and silently miscompiles.
 */
export function compileWasmFn(
  fn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  options: CompileWasmFnOptions,
): CompiledWasm {
  if (options.gpuUniformLayout && options.float !== "f32") {
    throw new Error(
      `[RMSL] compileWasmFn: gpuUniformLayout shares a uniform buffer with the GPU, which holds 32-bit floats, ` +
        `but this compile computes in 64 bits. Pass float: "f32" to share the buffer, or drop gpuUniformLayout to keep 64-bit precision.`,
    );
  }
  /** Whether every float value is rounded to 32 bits, as `float: "f32"` asks. */
  const float32 = options.float === "f32";
  const paramNodes = options.params.map((p) => parameterNode(p.name, p.type));
  const rawResult = fn(...paramNodes) as any;
  // `rawResult` is either one root or an array of roots (a caller-supplied
  // array of independently-built Fns, e.g. compileWasmRoutine(..., [a, b]),
  // or Fn's own array-return sugar, where every returned item is a "seq"
  // node sharing the *same* captured statement list plus that item's own
  // tail value — src/core.ts's Fn). Every root's statements get emitted
  // (deduped below by node identity, so the shared-statement-list case
  // doesn't double-emit); only the last root's value feeds the stage's
  // single result slot, matching compileGlsl/compileWgsl's "last array
  // entry wins" convention.
  const rawNodes: any[] = Array.isArray(rawResult) ? (rawResult as any[]) : [rawResult];
  if (options.kind) assertReadsNoStageInput(rawNodes, options.kind);
  // Without a stage the function is a plain function of its context, whose
  // result can be any value.
  const resultNodes = numberClashingVariables(shareNodes(prepareRoots(options.stage, rawNodes)));
  const root = resultNodes[resultNodes.length - 1];

  const paramTypeByName = new Map(options.params.map((p) => [p.name, p.type]));
  const fnParamNames = new Set(options.params.map((p) => p.name));

  const effectiveStage: "vertex" | "fragment" | "compute" = options.stage ?? "fragment";

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
  const storageMetadataAddress = new Map<string, number>();
  /** Slots of the storage buffers the program assigns to. */
  const writtenStorage = new Set<string>();

  // one shared per-pixel input slot, allocated on first use
  let fragCoordAddress: number | undefined;

  // needsResult: the WASM function returns nothing; a result, when there is
  // one, is written to memory (valueAddress below) and read back by the host.
  // A program that returns nothing has no value slot and nothing to read back.
  let needsResult = options.stage === "vertex" || isAggregate(root._t as string) || root._t === "void";
  let positionWritten = false;
  const outputAddress = new Map<string, number>();
  const varyingOutputAddress = new Map<string, number>();
  let positionAddress: number | undefined;
  let fragDepthAddress: number | undefined;
  /** An `i32` the program sets to 1 when it discards, which the host reads back after the call. */
  let discardAddress: number | undefined;

  const textureMetadataAddress = new Map<string, number>();
  const scratchAddress = new WeakMap<object, number>();
  /** The WASM local each `storage.element(i)` access keeps its evaluated index in. */
  const storageIndexLocal = new WeakMap<object, string>();

  let memCursor = (options.gpuUniformLayout?.totalSize ?? 0) + (options.memoryBase ?? 0); // cursor past the host's reserved region(s)

  // Pass 1: plan the whole tree of every root — record slots/addresses and
  // imported math names — before any bytecode is emitted. `collect` gates
  // every allocation behind a `.has()` check, so revisiting nodes shared
  // between roots (the array-return-sugar case) is idempotent.
  assertOneDeclarationPerName(resultNodes);
  for (const n of resultNodes) collect(n);
  for (const p of memoryParams) if (p.kind === "storageMemory") p.written = writtenStorage.has(p.slot);

  let resultKind: ScalarKind;
  let valueAddress: number | undefined;

  if (options.stage !== undefined) {
    assertStageResult(
      effectiveStage,
      root._t === "void" ? undefined : (root._t as string),
      positionWritten,
      outputAddress.size > 0,
    );
  }
  if (needsResult) {
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
  // depth. walkStmt sets it for the statement it walks and restores it after,
  // so a nested statement's depth doesn't outlive that statement.
  let currentStmtDepth = EXIT_BLOCK_DEPTH;

  /**
   * The stretches of code being emitted that run together, innermost last: the
   * function itself, each branch of an `If` and each loop. What an emitted
   * expression stored in a local is there for the code that runs after it in
   * the same stretch, or in one inside it.
   */
  const regionStack: number[] = [0];
  let regionCount = 1;

  /** Emits `emit` as a region of its own. */
  function inRegion<T>(emit: () => T): T {
    regionStack.push(regionCount++);
    try {
      return emit();
    } finally {
      regionStack.pop();
    }
  }

  /**
   * The local that holds the value of an inline `Fn` result whose statements
   * were emitted once, with the region they were emitted in, so a later read in
   * that region or one inside it reads the local instead of running them again.
   */
  const onceValue = new Map<any, { local: number; region: number }>();
  // Statement nodes are emitted at most once by identity: the array-return-
  // sugar case gives every result node the *same* leading statement objects
  // (see resultNodes above), and re-walking an already-emitted statement
  // would double its bytecode (and any side effect it has).
  const emittedStmts = new Set<any>();
  const bodyBytes: number[] = [];
  resultNodes.forEach((node, i) => {
    const isLast = i === resultNodes.length - 1;
    if (node.type === "seq") {
      const stmts = node.params.slice(0, -1) as any[];
      for (const s of stmts) {
        if (emittedStmts.has(s)) continue;
        emittedStmts.add(s);
        bodyBytes.push(...walkStmt(s, EXIT_BLOCK_DEPTH));
      }
      if (isLast) bodyBytes.push(...finalValueBytes(node.params[node.params.length - 1]));
    } else if (isLast) {
      bodyBytes.push(...finalValueBytes(node));
    }
    // A non-last, non-"seq" root is a bare pure expression with no
    // statements of its own — its value is unused, so there's nothing to emit.
  });

  /**
   * At `float: "f32"`, the bytes that round each float of the attributes and
   * varyings in memory to 32 bits, once as a call starts: whether the host's
   * marshaller or the rasterizer wrote them, they arrive in 64 bits.
   */
  function roundInputsInMemory(): number[] {
    if (!float32) return [];
    const bytes: number[] = [];
    for (const p of memoryParams) {
      if (p.kind !== "attributeMemory" && p.kind !== "varyingMemory") continue;
      const aggregate = isAggregate(p.shaderType);
      if ((aggregate ? elementKindOf(p.shaderType) : scalarKindOf(p.shaderType)) !== "float") continue;
      bytes.push(...roundFloatsInMemory(p.address, aggregate ? componentCountOf(p.shaderType) : 1));
    }
    return bytes;
  }

  /** The bytes that round the `width` f64s at `addr` to 32 bits, in place. */
  function roundFloatsInMemory(addr: number, width: number): number[] {
    const bytes: number[] = [];
    for (let k = 0; k < width; k++) {
      const rounded = [...loadComponent(addr, "float", k * 8), WASM_OP.f32DemoteF64, WASM_OP.f64PromoteF32];
      bytes.push(...storeComponent(addr, "float", k * 8, rounded));
    }
    return bytes;
  }

  const exitBlockType = needsResult ? WASM_BLOCKTYPE_VOID : wasmTypeOf(resultKind); // the outer block carries the function's result type (or void)
  const code = [WASM_OP.block, exitBlockType, ...roundInputsInMemory(), ...bodyBytes, WASM_OP.end];

  // Module assembly: type section, math imports, the main function (whose type
  // carries every scalar param and, when !needsResult, one result), a linear
  // memory sized for memCursor, exports, and the code section.
  const typeEntries: number[][] = [];
  let unaryImportType: number | null = null;
  let binaryImportType: number | null = null;

  /**
   * One import entry per math function this program calls (sin, pow, ...):
   * module name "math", the function's own name, the 0x00 func import kind
   * (see the memory import below), then its type index.
   */
  const importEntries: number[][] = importNames.map((name) => {
    const typeIdx = MATH_BINARY_IMPORTS.has(name) ? binaryImportTypeIdx() : unaryImportTypeIdx();
    return [...wasmStrBytes("math"), ...wasmStrBytes(name), 0x00, ...wasmUleb128(typeIdx)];
  });

  /**
   * The main function's own type: every scalar param, in order, then either
   * no result (the program writes its result through `out`/memory instead —
   * `needsResult` false) or one result type.
   */
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
  /** Whether `draw` writes its float components as f32, as `float: "f32"` asks of the buffer it fills. */
  const drawFloat32 = float32 && drawComponentKind === "float";

  // A "draw(width, height, bufferBase)" export: loops every pixel, runs the
  // main function (feeding the pixel in as fragCoord), and stores the output
  // into the caller's buffer — the CPU path for whole-image evaluation.
  if (root._t !== "void") {
    const widthIdx = params.length;
    const heightIdx = params.length + 1;
    const bufferBaseIdx = params.length + 2;
    const xIdx = params.length + 3;
    const yIdx = params.length + 4;
    /** Pushes local `x`. */
    const getX = [WASM_OP.localGet, ...wasmUleb128(xIdx)];
    /** Pushes local `y`. */
    const getY = [WASM_OP.localGet, ...wasmUleb128(yIdx)];
    const compSize = componentSizeOf(drawComponentKind);
    /** The bytes a component takes in the caller's buffer: 4 for a float at `float: "f32"`, as a `Float32Array` holds it. */
    const outSize = drawFloat32 ? 4 : compSize;
    /**
     * Pushes every one of the main function's own params, in order —
     * `draw` and `main` share the same leading params, so this forwards
     * them unchanged rather than re-deriving them per pixel.
     */
    const passThroughArgs = params.map((_, i) => [WASM_OP.localGet, ...wasmUleb128(i)]).flat();
    /** Forwards `passThroughArgs` and calls `main`. */
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

    /**
     * `(y * width + x) * componentCount * compSize` — the pixel's byte
     * offset within one row-major, tightly packed image buffer.
     */
    const pixelByteOffset = [
      ...getY,
      WASM_OP.localGet,
      ...wasmUleb128(widthIdx),
      WASM_OP.i32Mul,
      ...getX,
      WASM_OP.i32Add,
      ...i32ConstBytes(drawComponentCount * outSize),
      WASM_OP.i32Mul,
    ];
    /**
     * `bufferBase + pixelByteOffset + k * compSize` — the address of the
     * pixel's `k`th component in the caller's output buffer.
     */
    const destAddr = (k: number) => [
      WASM_OP.localGet,
      ...wasmUleb128(bufferBaseIdx),
      ...pixelByteOffset,
      WASM_OP.i32Add,
      ...i32ConstBytes(k * outSize),
      WASM_OP.i32Add,
    ];
    /** Stores a component at `addrBytes` in the caller's buffer, a float as an f32 at `float: "f32"`. */
    const storeOut = (addrBytes: number[], valueBytes: number[]) =>
      drawFloat32
        ? [...addrBytes, ...valueBytes, WASM_OP.f32DemoteF64, WASM_OP.f32Store, 0x00, 0x00]
        : storeDynamic(addrBytes, drawComponentKind, valueBytes);
    /**
     * Calls `main` and copies its result into the output buffer. A
     * needs-result function returns nothing directly — main() already wrote
     * its result to `valueAddress` (memory) — so each component is read
     * back from there and stored per-component; a scalar-returning one is
     * stored straight from `callMain`'s own return value instead.
     */
    const copyResult: number[] = needsResult
      ? [
          ...callMain,
          ...Array.from({ length: drawComponentCount }, (_, k) =>
            storeOut(destAddr(k), loadComponent(valueAddress!, drawComponentKind, k * compSize)),
          ).flat(),
        ]
      : storeOut(destAddr(0), callMain);
    /** The whole per-pixel body: write fragCoord for this pixel, then run main() and copy its result out. */
    // A discarded pixel leaves its value memory as it was, so each pixel starts from zero
    // and a clear flag: a pixel that discards then holds zero in every channel.
    const startPixel =
      discardAddress === undefined || valueAddress === undefined
        ? []
        : [
            ...storeComponent(discardAddress, "int", 0, i32ConstBytes(0)),
            ...Array.from({ length: drawComponentCount }, (_, k) =>
              storeComponent(
                valueAddress!,
                drawComponentKind,
                k * compSize,
                drawComponentKind === "float" ? f64ConstBytes(0) : i32ConstBytes(0),
              ),
            ).flat(),
          ];
    const perPixel = [...writeFragCoord, ...startPixel, ...copyResult];
    /** `for (x = 0; x < width; x++) perPixel();`, one row. */
    const innerLoop = forLoop(xIdx, i32ConstBytes(0), iGeS(local(xIdx), local(widthIdx)), perPixel, i32ConstBytes(1));
    /** `for (y = 0; y < height; y++) innerLoop();` — the whole pixel grid. */
    const drawCode = forLoop(yIdx, i32ConstBytes(0), iGeS(local(yIdx), local(heightIdx)), innerLoop, i32ConstBytes(1));
    const drawLocalsDecl = wasmVec([
      [...wasmUleb128(1), WASM_I32], // one local group per loop counter (xIdx, yIdx)
      [...wasmUleb128(1), WASM_I32],
    ]);

    drawFuncBody = [...drawLocalsDecl, ...drawCode, WASM_OP.end];
    drawTypeIdx = typeEntries.length;
    // `WASM_FUNC` (0x60) is the functype form byte that opens every entry in
    // the type section — the byte a decoder uses to tell "this is a function
    // signature" apart from the handful of other type forms the format has.
    // It's followed by two `wasmVec`s back to back: the parameter types,
    // then the result types (empty here — this function communicates its
    // result by writing into `out` rather than returning one).
    typeEntries.push([WASM_FUNC, ...wasmVec([...paramTypes, [WASM_I32], [WASM_I32], [WASM_I32]]), ...wasmVec([])]);
  }

  let computeTypeIdx: number | undefined;
  let computeFuncBody: number[] | undefined;

  // A "compute(count)" export for a compute program: runs main once per
  // index in 0..count, passing the index as invocationIndex() and every other
  // param through unchanged, so a whole dispatch is one call from the host.
  if (paramIndex.has("invocationIndex") || storageMetadataAddress.size > 0) {
    const countIdx = params.length;
    const indexIdx = params.length + 1;
    const invocationIndexParam = paramIndex.get("invocationIndex");
    const args = params
      .map((_, i) => [WASM_OP.localGet, ...wasmUleb128(i === invocationIndexParam ? indexIdx : i)])
      .flat();
    const callMain = [...args, WASM_OP.call, ...wasmUleb128(mainFuncIndex), ...(needsResult ? [] : [WASM_OP.drop])];
    const computeCode = forLoop(
      indexIdx,
      i32ConstBytes(0),
      iGeS(local(indexIdx), local(countIdx)),
      callMain,
      i32ConstBytes(1),
    );
    const computeLocalsDecl = wasmVec([[...wasmUleb128(1), WASM_I32]]); // the loop counter
    computeFuncBody = [...computeLocalsDecl, ...computeCode, WASM_OP.end];
    computeTypeIdx = typeEntries.length;
    typeEntries.push([WASM_FUNC, ...wasmVec([...paramTypes, [WASM_I32]]), ...wasmVec([])]);
  }

  /** The module's own functions after main, in function-index order: draw, then compute, each when emitted. */
  const extraFunctions = [
    { name: "draw", typeIdx: drawTypeIdx, body: drawFuncBody },
    { name: "compute", typeIdx: computeTypeIdx, body: computeFuncBody },
  ].filter((f): f is { name: string; typeIdx: number; body: number[] } => f.typeIdx !== undefined);

  /** 65536 bytes per WASM memory page. */
  const memoryPages = Math.max(1, Math.ceil(memCursor / 65536));
  /**
   * Memory is imported rather than owned by the module, so a caller can hand
   * multiple instances the same (optionally SharedArrayBuffer-backed)
   * memory. The shared-ness of an import is a static part of the module
   * (the engine rejects instantiation if it doesn't exactly match the
   * memory object handed in), so it has to be a compile-time option, not a
   * runtime one — a shared import always needs a declared maximum too.
   */
  const sharedMemory = options.sharedMemory ?? false;
  /** 65536 pages = the full 4GiB wasm32 address space. */
  const maxMemoryPages = options.maxMemoryPages ?? 65536;
  /**
   * A "limits" encoding: a one-byte flag (0x00 = min only, 0x03 = min and
   * max, shared) followed by the min page count and, only when the flag
   * says so, the max page count — both as unsigned LEB128. 0x01/0x02
   * (max-but-not-shared forms) exist in the spec but are never emitted
   * here, since sharedness is this module's own choice, never partial.
   */
  const memoryLimitsBytes = sharedMemory
    ? [0x03, ...wasmUleb128(memoryPages), ...wasmUleb128(maxMemoryPages)]
    : [0x00, ...wasmUleb128(memoryPages)];
  /**
   * An import entry: module name, field name, then a one-byte "import kind"
   * (0x00 func, 0x01 table, 0x02 memory, 0x03 global) and the kind-specific
   * descriptor that follows it — here the memory limits just built above.
   */
  const memoryImportEntry = [...wasmStrBytes("env"), ...wasmStrBytes("memory"), 0x02, ...memoryLimitsBytes];

  /**
   * Section ids, per the spec's binary format (see {@link wasmSection}'s own
   * doc): 1 Type, 2 Import, 3 Function.
   */
  const typeSection = wasmSection(1, wasmVec(typeEntries));
  const importSection = wasmSection(2, wasmVec([...importEntries, memoryImportEntry]));
  const funcSection = wasmSection(3, wasmVec([[mainTypeIdx], ...extraFunctions.map((f) => [f.typeIdx])]));
  const nameBytes = wasmStrBytes(options.name);
  /**
   * An export entry: the export's own name, a one-byte "export kind" (0x00
   * func, 0x01 table, 0x02 memory, 0x03 global — same vocabulary as the
   * import kind above), then the kind-specific index — here a function
   * index into the (imports ++ this module's own functions) index space.
   */
  const exportEntries = [[...nameBytes, 0x00, ...wasmUleb128(mainFuncIndex)]];

  extraFunctions.forEach((f, i) => {
    exportEntries.push([...wasmStrBytes(f.name), 0x00, ...wasmUleb128(mainFuncIndex + 1 + i)]);
  });

  /** Section id 7, Export. */
  const exportSection = wasmSection(7, wasmVec(exportEntries));

  /**
   * Each local group is `[count, type]` — a run of `count` consecutive
   * locals sharing one type, which is why every group here is
   * `wasmUleb128(1)` (one local at a time): this compiler never merges
   * same-typed locals into a single run, only ever declares a fresh
   * one-local group per slot.
   */
  const localsDecl = wasmVec(localSlots.map((name) => [...wasmUleb128(1), wasmTypeOf(localType.get(name)!)]));
  const funcBody = [...localsDecl, ...code, WASM_OP.end];
  /**
   * A code entry is prefixed with its own byte length (not a `wasmVec`
   * count — a decoder skips a whole function body it doesn't want to
   * parse), then the local declarations and the instruction bytes
   * themselves.
   */
  const codeEntries = [[...wasmUleb128(funcBody.length), ...funcBody]];

  for (const f of extraFunctions) codeEntries.push([...wasmUleb128(f.body.length), ...f.body]);

  /**
   * Section id 10, Code — one entry per function declared in the Function
   * section above, in the same order, each holding that function's locals
   * and its actual instruction bytes.
   */
  const codeSection = wasmSection(10, wasmVec(codeEntries));

  /**
   * The module: an 8-byte header, then the sections built above. Sections
   * are self-delimiting (each carries its own byte length, from
   * {@link wasmSection}) and, while the spec allows most orderings, a
   * decoder expects known section ids ascending — hence Type/Import/
   * Function/Export/Code here, skipping the ids (Table, Memory, Global,
   * Start, Element) this compiler never emits.
   */
  // prettier-ignore
  const bytes = new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, // magic number: "\0asm"
    0x01, 0x00, 0x00, 0x00, // version 1, as a little-endian u32 (the only version that has ever existed)
    ...typeSection,         // section id 1: function signatures (params + result types)
    ...importSection,       // section id 2: the memory import (and any math-function imports)
    ...funcSection,         // section id 3: which type index each of this module's own functions has
    ...exportSection,       // section id 7: the entry point(s) a host can call, by name
    ...codeSection,         // section id 10: each function's locals + instruction bytes
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
    compute: computeTypeIdx !== undefined,
    float32,
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
        } else if (
          isAggregate(v.shaderType) ||
          options.scalarsInMemory ||
          options.gpuUniformLayout?.offsets[v.slot] !== undefined
        ) {
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
        assertNotInAComputeStage(effectiveStage, COMPUTE_REFUSES.attribute);
        if (isAggregate(v.shaderType) || options.scalarsInMemory) {
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
        const v = node.value;
        if (storageMetadataAddress.has(v.slot) && v.access !== "read") {
          // Two nodes over one buffer, one read-only: the buffer takes the wider access.
          const param = memoryParams.find((p: any) => p.kind === "storageMemory" && p.slot === v.slot) as any;
          if (param.access === "read") param.access = v.access;
          needsResult = true;
        }
        if (!storageMetadataAddress.has(v.slot)) {
          const addr = allocateBytes(STORAGE_META_STRIDE);
          storageMetadataAddress.set(v.slot, addr);
          if (v.access !== "read") needsResult = true;
          memoryParams.push({
            kind: "storageMemory",
            slot: v.slot,
            shaderType: v.shaderType,
            metadataAddress: addr,
            access: v.access,
            written: false,
          });
        }
        break;
      }

      case "storageElement": {
        if (!storageIndexLocal.has(node)) {
          const name = `$storage_index${localSlots.length}`;
          storageIndexLocal.set(node, name);
          addLocal(name, "int");
        }
        break;
      }

      case "invocationIndex": {
        addParam({ kind: "invocationIndex", shaderType: "uint" }, "invocationIndex");
        break;
      }

      case "varying": {
        const v = node.value;
        assertNotInAComputeStage(effectiveStage, COMPUTE_REFUSES.varying);
        if (effectiveStage === "fragment") {
          // fragment: varyings are per-call inputs, written by the host
          if (isAggregate(v.shaderType) || options.scalarsInMemory) {
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

      case "discard": {
        // the host reads this flag to tell a discarded fragment from one that returned zero
        needsResult = true;
        if (discardAddress === undefined) {
          discardAddress = allocateFor("int");
          memoryParams.push({ kind: "discardMemory", address: discardAddress });
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
        // and which storage buffers are written
        assertAssignable(node.params[0], effectiveStage);
        const element = assignedStorageElement(node.params[0]);
        if (element) writtenStorage.add(element.params[0].value.slot);
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

      case "vectorElement":
        addLocal("$element_index", "int");
        break;

      case "matrixElement":
        addLocal("$element_index", "int");
        addLocal("$column_address", "int");
        break;

      case "div":
      case "mod":
        if ((isAggregate(node._t) ? elementKindOf(node._t) : scalarKindOf(node._t)) !== "float") {
          addLocal("$int_div_a", "int");
          addLocal("$int_div_b", "int");
        }
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

  /**
   * WASM local index of a scalar `let`-bound variable, offset past the
   * function's own params (WASM indexes locals after params).
   */
  function localSlotIndex(varName: string): number {
    const i = localIndex.get(varName);
    if (i === undefined) throw new Error(`[RMSL] compileWasmFn: read of undeclared var "${varName}"`);
    return params.length + i;
  }

  /**
   * WASM param index of a scalar uniform/attribute/varying/param, keyed
   * the same way `addParam` deduped it.
   */
  function paramSlotIndex(key: string): number {
    const i = paramIndex.get(key);
    if (i === undefined) throw new Error(`[RMSL] compileWasmFn: internal error, unindexed slot "${key}"`);
    return i;
  }

  /**
   * Emits a `call` to a math/transcendental import by name (`sin`, `pow`, ...).
   */
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

      case "storage":
        throw bareStorageError(node);

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
   * Bytes for one `storage.element(index)` access: whether `index` lies inside
   * the buffer the host marshalled in, and the address of the element's `k`th
   * component. A negative index reads as a huge unsigned one, so one unsigned
   * compare against the length rejects both ends. `inBounds` evaluates the
   * index into a local that `address` reads, so it has to run first.
   */
  function storageElementAccess(node: any): { inBounds: number[]; address(k: number): number[] } {
    const storageNode = node.params[0];
    const metaAddr = storageMetadataAddress.get(storageNode.value.slot);
    if (metaAddr === undefined) {
      throw new Error(`[RMSL] compileWasmFn: internal error, unaddressed storage "${storageNode.value.slot}"`);
    }
    const type = node._t as string;
    const kind = isAggregate(type) ? elementKindOf(type) : scalarKindOf(type);
    const compSize = componentSizeOf(kind);
    const stride = componentCountOf(type) * compSize;
    const index = node.params[1];
    const indexBytes =
      scalarKindOf(index._t as string) === "float" ? [...walkExpr(index), ...I32_TRUNC_SAT_F64_U] : walkExpr(index);
    const indexLocal = wasmUleb128(localSlotIndex(storageIndexLocal.get(node)!));
    return {
      inBounds: [
        ...indexBytes,
        WASM_OP.localTee,
        ...indexLocal,
        ...loadComponent(metaAddr, "int", STORAGE_META_LENGTH),
        WASM_OP.i32LtU,
      ],
      address: (k) => [
        ...loadComponent(metaAddr, "int", STORAGE_META_DATA_ADDR),
        WASM_OP.localGet,
        ...indexLocal,
        ...i32ConstBytes(stride),
        WASM_OP.i32Mul,
        WASM_OP.i32Add,
        ...(k === 0 ? [] : [...i32ConstBytes(k * compSize), WASM_OP.i32Add]),
      ],
    };
  }

  /**
   * Copies an aggregate storage element into its scratch address at `addr`.
   * An element outside the buffer reads as zero rather than whatever memory
   * lies past it.
   */
  function emitStorageElementLoadStores(node: any, addr: number): number[] {
    const kind = elementKindOf(node._t as string);
    const compSize = componentSizeOf(kind);
    const width = componentCountOf(node._t as string);
    const element = storageElementAccess(node);
    const loads: number[] = [];
    const zeroes: number[] = [];
    for (let k = 0; k < width; k++) {
      loads.push(...storeComponent(addr, kind, k * compSize, loadDynamic(element.address(k), kind)));
      zeroes.push(...storeComponent(addr, kind, k * compSize, kind === "float" ? f64ConstBytes(0) : i32ConstBytes(0)));
    }
    return [...element.inBounds, WASM_OP.if_, WASM_BLOCKTYPE_VOID, ...loads, WASM_OP.else_, ...zeroes, WASM_OP.end];
  }

  /**
   * The component of a vector, or the column of a matrix, that a literal
   * `index` selects in `target`. One outside it is an error, as it is when
   * WGSL and GLSL compile it.
   */
  function constantIndex(target: any, index: any): number {
    return assertLiteralIndexInRange(target, index)!;
  }

  /**
   * Bytes computing the byte offset of the item a run-time `index` selects
   * among `count` items of `size` bytes: a vector's components or a matrix's
   * columns. An index past the end, or a negative one (which compares as a
   * huge unsigned one), selects the last item, so the access stays inside.
   */
  function clampedIndexOffset(index: any, count: number, size: number): number[] {
    const indexBytes =
      scalarKindOf(index._t as string) === "float" ? [...walkExpr(index), ...I32_TRUNC_SAT_F64_S] : walkExpr(index);
    const local = wasmUleb128(localSlotIndex("$element_index"));
    return [
      ...selectExpr([...indexBytes, WASM_OP.localTee, ...local], i32ConstBytes(count - 1), [
        WASM_OP.localGet,
        ...local,
        ...i32ConstBytes(count),
        WASM_OP.i32LtU,
      ]),
      ...i32ConstBytes(size),
      WASM_OP.i32Mul,
    ];
  }

  /**
   * Stores `rhs` into the component of a writable vector or matrix column
   * that `target`, a `vectorElement` node, selects. A storage
   * element's component is stored inside the element's bounds check, like a
   * whole-element store.
   */
  function emitVectorElementStore(target: any, rhs: any): number[] {
    const [vector, index] = target.params;
    const width = componentCountOf(vector._t as string);
    const kind = elementKindOf(vector._t as string);
    const compSize = componentSizeOf(kind);
    if (vector.type === "matrixElement") {
      const offset = isLeafLiteral(index)
        ? i32ConstBytes(constantIndex(vector, index) * compSize)
        : clampedIndexOffset(index, width, compSize);
      return emitColumnComponentStore(vector, offset, rhs);
    }
    if (vector.type === "storageElement") {
      const access = storageElementAccess(vector);
      const address = isLeafLiteral(index)
        ? access.address(constantIndex(vector, index))
        : [...access.address(0), ...clampedIndexOffset(index, width, compSize), WASM_OP.i32Add];
      return [
        ...access.inBounds,
        WASM_OP.if_,
        WASM_BLOCKTYPE_VOID,
        ...storeDynamic(address, kind, walkExpr(rhs)),
        WASM_OP.end,
      ];
    }
    if (vector.type === "swizzle") {
      throw new Error(
        "[RMSL] compileWasmFn: writing a component by index through a swizzle isn't supported yet; write it through the swizzle's letters, as .x",
      );
    }
    const base = nodeAddress(vector);
    if (isLeafLiteral(index)) return storeComponent(base, kind, constantIndex(vector, index) * compSize, walkExpr(rhs));
    return storeDynamic(
      [...i32ConstBytes(base), ...clampedIndexOffset(index, width, compSize), WASM_OP.i32Add],
      kind,
      walkExpr(rhs),
    );
  }

  /**
   * Bytes copying floats between a matrix column, at the address
   * `columnAddress` computes, and the fixed address `vector`: the vector's
   * `i`th component pairs with the column's row `rows[i]`. They copy into
   * the vector when `toVector`, out of it otherwise. The column's address is
   * computed once, into a local.
   */
  function copyColumn(columnAddress: number[], rows: number[], vector: number, toVector: boolean): number[] {
    const local = wasmUleb128(localSlotIndex("$column_address"));
    const out = [...columnAddress, WASM_OP.localSet, ...local];
    rows.forEach((r, i) => {
      const rowAddress = [WASM_OP.localGet, ...local, ...i32ConstBytes(r * 8), WASM_OP.i32Add];
      out.push(
        ...(toVector
          ? storeComponent(vector, "float", i * 8, loadDynamic(rowAddress, "float"))
          : storeDynamic(rowAddress, "float", loadComponent(vector, "float", i * 8))),
      );
    });
    return out;
  }

  /** The rows of a column of `matrix`, in order. */
  function columnRows(matrix: any): number[] {
    return Array.from({ length: MATRIX_DIMENSIONS[matrix._t as string][1] }, (_, r) => r);
  }

  /**
   * Bytes computing the address of the column `index` selects in a matrix
   * whose first column starts at the address `base` computes.
   */
  function columnAddress(matrix: any, index: any, base: number[]): number[] {
    const [cols, rows] = MATRIX_DIMENSIONS[matrix._t as string];
    if (isLeafLiteral(index)) {
      const c = constantIndex(matrix, index);
      return c === 0 ? base : [...base, ...i32ConstBytes(c * rows * 8), WASM_OP.i32Add];
    }
    return [...base, ...clampedIndexOffset(index, cols, rows * 8), WASM_OP.i32Add];
  }

  /** Copies the matrix column a `matrixElement` node selects into its scratch address at `addr`. */
  function emitMatrixColumnLoadStores(node: any, addr: number): number[] {
    const [matrix, index] = node.params;
    return [
      ...materializeIfNeeded(matrix),
      ...copyColumn(columnAddress(matrix, index, i32ConstBytes(nodeAddress(matrix))), columnRows(matrix), addr, true),
    ];
  }

  /**
   * The column of a writable matrix that `column`, a `matrixElement` node,
   * selects as an assignment target: the bytes
   * computing its address, and a `guard` that wraps its stores. A storage
   * element's column is stored inside the element's bounds check, like a
   * whole-element store.
   */
  function columnTarget(column: any): { address: number[]; guard(stores: number[]): number[] } {
    const [matrix, index] = column.params;
    if (matrix.type === "storageElement") {
      const access = storageElementAccess(matrix);
      return {
        address: columnAddress(matrix, index, access.address(0)),
        guard: (stores) => [...access.inBounds, WASM_OP.if_, WASM_BLOCKTYPE_VOID, ...stores, WASM_OP.end],
      };
    }
    return { address: columnAddress(matrix, index, i32ConstBytes(nodeAddress(matrix))), guard: (stores) => stores };
  }

  /**
   * Stores the scalar `rhs` into one component of the column `column`, a
   * `matrixElement` node, selects: the one at the byte offset `offset`
   * computes, after the column's own address.
   */
  function emitColumnComponentStore(column: any, offset: number[], rhs: any): number[] {
    const target = columnTarget(column);
    return target.guard(storeDynamic([...target.address, ...offset, WASM_OP.i32Add], "float", walkExpr(rhs)));
  }

  /**
   * Stores `rhs` into the rows `rows` of the column `column`, a
   * `matrixElement` node, selects: a whole column, or a swizzle of one.
   */
  function emitMatrixColumnStore(column: any, rows: number[], rhs: any): number[] {
    if (rows.length === 1) return emitColumnComponentStore(column, i32ConstBytes(rows[0]! * 8), rhs);
    const out = [...materializeIfNeeded(rhs)];
    const target = columnTarget(column);
    return [...out, ...target.guard(copyColumn(target.address, rows, nodeAddress(rhs), false))];
  }

  /**
   * Stores `rhs` into some components of a storage element, named by their
   * letters: one letter takes a scalar, several take the matching vector's
   * components in order. Inside the element's bounds check, like a whole-element store.
   */
  function emitStorageComponentStores(element: any, letters: string[], rhs: any): number[] {
    const access = storageElementAccess(element);
    const kind = elementKindOf(element._t as string);
    const compSize = componentSizeOf(kind);
    const prelude = letters.length === 1 ? [] : materializeIfNeeded(rhs);
    const rhsAddr = letters.length === 1 ? 0 : nodeAddress(rhs);
    const stores = letters.flatMap((letter, i) =>
      storeDynamic(
        access.address(COMPONENT_INDEX[letter]!),
        kind,
        letters.length === 1 ? walkExpr(rhs) : loadComponent(rhsAddr, kind, i * compSize),
      ),
    );
    return [...prelude, ...access.inBounds, WASM_OP.if_, WASM_BLOCKTYPE_VOID, ...stores, WASM_OP.end];
  }

  /**
   * Writes `rhs` into a storage element. A write outside the buffer is
   * dropped, so it cannot land in whatever memory lies past it.
   */
  function emitStorageElementStore(target: any, rhs: any): number[] {
    const type = target._t as string;
    const element = storageElementAccess(target);
    if (!isAggregate(type)) {
      const kind = scalarKindOf(type);
      return [
        ...element.inBounds,
        WASM_OP.if_,
        WASM_BLOCKTYPE_VOID,
        ...storeDynamic(element.address(0), kind, walkExpr(rhs)),
        WASM_OP.end,
      ];
    }
    const kind = elementKindOf(type);
    const compSize = componentSizeOf(kind);
    const out = [...materializeIfNeeded(rhs)];
    const rhsAddr = nodeAddress(rhs);
    const stores: number[] = [];
    for (let k = 0; k < componentCountOf(type); k++) {
      stores.push(...storeDynamic(element.address(k), kind, loadComponent(rhsAddr, kind, k * compSize)));
    }
    return [...out, ...element.inBounds, WASM_OP.if_, WASM_BLOCKTYPE_VOID, ...stores, WASM_OP.end];
  }

  /**
   * The bytes that write the value of an aggregate `node` to its address. At
   * `float: "f32"`, the float components a node computes are rounded there.
   */
  function materializeIfNeeded(node: any): number[] {
    const bytes = materializeValue(node);
    if (!float32 || bytes.length === 0 || elementKindOf(node._t as string) !== "float") return bytes;
    return [...bytes, ...roundFloatsInMemory(nodeAddress(node), componentCountOf(node._t as string))];
  }

  /**
   * Emits the stores that guarantee node's aggregate value sits in memory
   * at nodeAddress(node). Most inputs already live there; gpu-placed
   * uniforms get promoted, and pure expressions are computed into their
   * scratch address on first use.
   */
  function materializeValue(node: any): number[] {
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
          scalarKindOf(index._t as string) === "float" ? [...walkExpr(index), ...I32_TRUNC_SAT_F64_S] : walkExpr(index);
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
        return [];
      case "storage":
        throw bareStorageError(node);
      case "storageElement":
        return emitStorageElementLoadStores(node, nodeAddress(node));
      case "matrixElement":
        return emitMatrixColumnLoadStores(node, nodeAddress(node));
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
      case "mod":
      case "bitAnd":
      case "bitOr":
      case "bitXor":
      case "bitNot":
      case "shiftLeft":
      case "shiftRight":
      case "negate":
      case "abs":
      case "radians":
      case "degrees":
      case "min":
      case "max":
      case "lessThan":
      case "greaterThan":
      case "lessThanEqual":
      case "greaterThanEqual":
      case "equal":
      case "notEqual":
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
    // A single scalar fills every component. It was stored once above, so the
    // rest copy that component rather than evaluating the scalar again.
    const width = componentCountOf(targetType);
    if (node.params.length === 1 && compIndex === 1) {
      for (let k = 1; k < width; k++) {
        out.push(...storeComponent(addr, targetKind, k * compSize, loadComponent(addr, targetKind, 0)));
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

  /**
   * Stores an all-zero aggregate value at `addr`, for a derivative node
   * compiled under `{ derivatives: "zero" }`.
   */
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
   * uints. No 2D/3D restriction — valid for cube too, matching compileJSRoutine.
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
        // A texel holds an exact integer, so the truncation saturates at the type's own range, not at the floats' ends.
        const fetched = [
          ...loadDynamic(elemAddrBytes(i), "float"),
          ...(samplerType.startsWith("isampler") ? I32_TRUNC_SAT_F64_S : I32_TRUNC_SAT_F64_U),
        ];
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
      return { i0: [...f0, ...I32_TRUNC_SAT_F64_S], t: [...f, ...f0, WASM_OP.f64Sub] };
    }

    const nearestX = wrapAxis(
      [...uv(0), ...dimF64(TEX_META_WIDTH), WASM_OP.f64Mul, WASM_OP.f64Floor, ...I32_TRUNC_SAT_F64_S],
      TEX_META_WIDTH,
      TEX_META_WRAP_S,
    );
    const nearestY = wrapAxis(
      [...uv(1), ...dimF64(TEX_META_HEIGHT), WASM_OP.f64Mul, WASM_OP.f64Floor, ...I32_TRUNC_SAT_F64_S],
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
        [...uv(2), ...dimF64(TEX_META_DEPTH), WASM_OP.f64Mul, WASM_OP.f64Floor, ...I32_TRUNC_SAT_F64_S],
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

  /**
   * Stores a multi-component swizzle (`v.xyz`, `v.rgba`, ...) at `addr`,
   * one source component load per destination component, in pattern order.
   */
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
   * Element-wise op over same-width vectors — binary, or unary for `bitNot`,
   * `negate` and `abs` — with scalar operands broadcast by re-evaluating them
   * (walkExpr) per component; aggregate operands are materialized once and
   * loaded per component. A comparison reads operands of its first operand's
   * kind and stores a boolean vector.
   */
  function emitComponentwiseStores(node: any, addr: number): number[] {
    const [a, b] = node.params;
    const targetKind = elementKindOf(node._t as string);
    const compSize = componentSizeOf(targetKind);
    const width = componentCountOf(node._t as string);
    const aWidth = componentCountOf(a._t);
    // Only a comparison's operands differ in kind from its result.
    const operandKind = !COMPARISON_OPCODES[node.type]
      ? targetKind
      : aWidth > 1
        ? elementKindOf(a._t as string)
        : scalarKindOf(a._t as string);
    const operandSize = componentSizeOf(operandKind);
    const bWidth = b === undefined ? 0 : componentCountOf(b._t);
    const out: number[] = [];
    if (aWidth > 1) out.push(...materializeIfNeeded(a));
    if (bWidth > 1) out.push(...materializeIfNeeded(b));
    const aAddr = aWidth > 1 ? nodeAddress(a) : undefined;
    const bAddr = bWidth > 1 ? nodeAddress(b) : undefined;
    const combine = (aBytes: number[], bBytes: number[]): number[] => {
      const float = operandKind === "float";
      if (COMPARISON_OPCODES[node.type]) return comparisonBytes(node.type, aBytes, bBytes, operandKind);
      if (node.type === "min" || node.type === "max") return minMaxBytes(aBytes, bBytes, operandKind, node.type);
      if (node.type === "negate")
        return float ? [...aBytes, WASM_OP.f64Neg] : [...i32ConstBytes(0), ...aBytes, WASM_OP.i32Sub];
      if (node.type === "abs") {
        if (float) return [...aBytes, WASM_OP.f64Abs];
        if (operandKind === "uint") return aBytes;
        const negated = [...i32ConstBytes(0), ...aBytes, WASM_OP.i32Sub];
        return selectExpr(negated, aBytes, [...aBytes, ...i32ConstBytes(0), WASM_OP.i32LtS]);
      }
      if (node.type === "radians") return [...aBytes, ...f64ConstBytes(RADIANS_PER_DEGREE), WASM_OP.f64Mul];
      if (node.type === "degrees") return [...aBytes, ...f64ConstBytes(DEGREES_PER_RADIAN), WASM_OP.f64Mul];
      if (node.type === "mod" && float) return flooredModulo(aBytes, bBytes);
      if ((node.type === "div" || node.type === "mod") && !float) {
        return [...aBytes, ...bBytes, ...integerDivision(operandKind as "int" | "uint", node.type)];
      }
      if (node.type === "bitNot") return [...aBytes, ...i32ConstBytes(-1), WASM_OP.i32Xor];
      const opcode = {
        add: float ? WASM_OP.f64Add : WASM_OP.i32Add,
        sub: float ? WASM_OP.f64Sub : WASM_OP.i32Sub,
        mul: float ? WASM_OP.f64Mul : WASM_OP.i32Mul,
        div: WASM_OP.f64Div,
        bitAnd: WASM_OP.i32And,
        bitOr: WASM_OP.i32Or,
        bitXor: WASM_OP.i32Xor,
        shiftLeft: WASM_OP.i32Shl,
        shiftRight: operandKind === "uint" ? WASM_OP.i32ShrU : WASM_OP.i32ShrS,
      }[node.type as "add" | "sub" | "mul" | "div" | "bitAnd" | "bitOr" | "bitXor" | "shiftLeft" | "shiftRight"];
      return [...aBytes, ...bBytes, opcode];
    };
    for (let k = 0; k < width; k++) {
      const aBytes = aWidth > 1 ? loadComponent(aAddr!, operandKind, k * operandSize) : walkExpr(a);
      const bBytes =
        b === undefined ? [] : bWidth > 1 ? loadComponent(bAddr!, operandKind, k * operandSize) : walkExpr(b);
      out.push(...storeComponent(addr, targetKind, k * compSize, combine(aBytes, bBytes)));
    }
    return out;
  }

  /** clamp = max(x, lo) then min(hi); an core guarantee keeps all three params the same width. */
  function emitClampStores(node: any, addr: number): number[] {
    const [x, lo, hi] = node.params;
    const targetKind = elementKindOf(node._t as string);
    const compSize = componentSizeOf(targetKind);
    const width = componentCountOf(node._t as string);
    // A scalar bound applies to every component, and is re-evaluated for each.
    const component = (n: any) => {
      if (componentCountOf(n._t) === 1) return { setup: [], at: () => walkExpr(n) };
      const nAddr = nodeAddress(n);
      return { setup: materializeIfNeeded(n), at: (k: number) => loadComponent(nAddr, targetKind, k * compSize) };
    };
    const [xc, loc, hic] = [component(x), component(lo), component(hi)];
    const out = [...xc.setup, ...loc.setup, ...hic.setup];
    for (let k = 0; k < width; k++) {
      const xk = xc.at(k);
      const lok = loc.at(k);
      const hik = hic.at(k);
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

  /**
   * Integer `/` or `%` of the two operands on the stack, with WGSL's results
   * where WASM would trap: `x / 0` is `x`, `x % 0` is `0`, and `INT_MIN / -1`
   * is `INT_MIN`. Those cases divide by 1 instead, which gives exactly those
   * results. The operands are parked in two shared locals; both are fully
   * evaluated before either is written, so a nested division can't clobber them.
   */
  function integerDivision(kind: "int" | "uint", op: "div" | "mod"): number[] {
    const getA = [WASM_OP.localGet, ...wasmUleb128(localSlotIndex("$int_div_a"))];
    const getB = [WASM_OP.localGet, ...wasmUleb128(localSlotIndex("$int_div_b"))];
    const divisorIsZero = [...getB, WASM_OP.i32Eqz];
    const divideByOne =
      kind === "int" && op === "div"
        ? [
            ...divisorIsZero,
            ...[...getA, ...i32ConstBytes(-2147483648), WASM_OP.i32Eq],
            ...[...getB, ...i32ConstBytes(-1), WASM_OP.i32Eq],
            WASM_OP.i32And,
            WASM_OP.i32Or,
          ]
        : divisorIsZero;
    const opcode =
      op === "div"
        ? kind === "uint"
          ? WASM_OP.i32DivU
          : WASM_OP.i32DivS
        : kind === "uint"
          ? WASM_OP.i32RemU
          : WASM_OP.i32RemS;
    return [
      WASM_OP.localSet,
      ...wasmUleb128(localSlotIndex("$int_div_b")),
      WASM_OP.localSet,
      ...wasmUleb128(localSlotIndex("$int_div_a")),
      ...getA,
      ...i32ConstBytes(1),
      ...getB,
      ...divideByOne,
      WASM_OP.select,
      opcode,
    ];
  }

  /** Floored modulo, `a - b * floor(a / b)`, as GLSL's `mod()`; a plain `f64.rem` would truncate toward zero. */
  function flooredModulo(a: number[], b: number[]): number[] {
    return [...a, ...b, ...a, ...b, WASM_OP.f64Div, WASM_OP.f64Floor, WASM_OP.f64Mul, WASM_OP.f64Sub];
  }

  /**
   * Scalar `smoothstep(e0, e1, x)`: clamps `t = (x-e0)/(e1-e0)` to `[0,1]`,
   * then returns `t^2 * (3 - 2t)`.
   */
  function emitSmoothstepValue(e0Bytes: number[], e1Bytes: number[], xBytes: number[]): number[] {
    // localTee caches the clamped t in the shared $smoothstep_t local so
    // the caller's x bytes are never re-evaluated.
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
   * The bytes that push the value of a scalar `node`. At `float: "f32"`, a float
   * value is rounded to 32 bits and widened back, so it holds what an f32 holds.
   */
  function walkExpr(node: any): number[] {
    const bytes = walkExprValue(node);
    if (!float32 || node._t !== "float" || node.type === "float" || node.type === "var") return bytes;
    return [...bytes, WASM_OP.f32DemoteF64, WASM_OP.f64PromoteF32];
  }

  /**
   * Expression pass: evaluates a node down to one value on the WASM stack —
   * an f64 for floats, an i32 for int/uint/bool. A component of an aggregate
   * is read through readComponent/materialize, never held on the stack.
   */
  function walkExprValue(node: any): number[] {
    switch (node.type) {
      case "float":
        return f64ConstBytes(float32 ? Math.fround(node.value) : node.value);
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
      case "uniform": {
        const rawAddr = gpuRawUniformAddress.get(node.value.slot); // the host wrote it into the GPU layout, as an f32 if a float
        if (rawAddr !== undefined) {
          const kind = scalarKindOf(node._t);
          return kind === "float"
            ? [...i32ConstBytes(rawAddr), WASM_OP.f32Load, 0x00, ...wasmUleb128(0), WASM_OP.f64PromoteF32]
            : loadComponent(rawAddr, kind, 0);
        }
        const addr = uniformAddress.get(node.value.slot);
        if (addr !== undefined) return loadComponent(addr, scalarKindOf(node._t), 0); // scalarsInMemory: memory-resident, not a param
        return [WASM_OP.localGet, ...wasmUleb128(paramSlotIndex(`uniform:${node.value.slot}`))];
      }
      case "attribute": {
        const addr = attributeAddress.get(node.value.slot);
        if (addr !== undefined) return loadComponent(addr, scalarKindOf(node._t), 0); // scalarsInMemory: memory-resident, not a param
        return [WASM_OP.localGet, ...wasmUleb128(paramSlotIndex(`attribute:${node.value.slot}`))];
      }
      case "invocationIndex":
        return [WASM_OP.localGet, ...wasmUleb128(paramSlotIndex("invocationIndex"))];
      case "storage":
        throw bareStorageError(node);
      case "storageElement": {
        const kind = scalarKindOf(node._t);
        const element = storageElementAccess(node);
        return [
          ...element.inBounds,
          WASM_OP.if_,
          kind === "float" ? WASM_F64 : WASM_I32,
          ...loadDynamic(element.address(0), kind),
          WASM_OP.else_,
          ...(kind === "float" ? f64ConstBytes(0) : i32ConstBytes(0)),
          WASM_OP.end,
        ];
      }
      case "varying":
        if (effectiveStage === "fragment") {
          const addr = varyingAddress.get(node.value.slot);
          if (addr !== undefined) return loadComponent(addr, scalarKindOf(node._t), 0); // scalarsInMemory: memory-resident, not a param
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
      case "div":
      case "mod": {
        const kind = scalarKindOf(node.params[0]._t);
        const a = walkExpr(node.params[0]);
        const b = walkExpr(node.params[1]);
        if (kind === "int" || kind === "uint") return [...a, ...b, ...integerDivision(kind, node.type)];
        if (node.type === "div") return [...a, ...b, WASM_OP.f64Div];
        return flooredModulo(a, b);
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
      case "radians":
        return [...walkExpr(node.params[0]), ...f64ConstBytes(RADIANS_PER_DEGREE), WASM_OP.f64Mul];
      case "degrees":
        return [...walkExpr(node.params[0]), ...f64ConstBytes(DEGREES_PER_RADIAN), WASM_OP.f64Mul];
      case "fract": {
        const x = node.params[0];
        // x - floor(x); x is emitted twice to keep the stack flat — cheap, side-effect free
        return [...walkExpr(x), ...walkExpr(x), WASM_OP.f64Floor, WASM_OP.f64Sub];
      }
      case "round":
        // WGSL's round takes a half to the even neighbour, as f64.nearest does.
        return [...walkExpr(node.params[0]), WASM_OP.f64Nearest];
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
          return floatToInteger(bytes, targetKind === "uint" ? "uint" : "int");
        }
        if (!sourceIsFloat && targetIsFloat) {
          return [...bytes, sourceKind === "uint" ? WASM_OP.f64ConvertI32U : WASM_OP.f64ConvertI32S];
        }

        if (targetKind === "bool") return [...bytes, ...i32ConstBytes(0), WASM_OP.i32Ne];
        return bytes;
      }

      case "vectorElement": {
        const [vector, index] = node.params;
        const width = componentCountOf(vector._t as string);
        if (isLeafLiteral(index)) return readComponent(vector, constantIndex(vector, index));
        const kind = elementKindOf(vector._t as string);
        return [
          ...materializeIfNeeded(vector),
          ...loadDynamic(
            [
              ...i32ConstBytes(nodeAddress(vector)),
              ...clampedIndexOffset(index, width, componentSizeOf(kind)),
              WASM_OP.i32Add,
            ],
            kind,
          ),
        ];
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
        // A scalar is a vector of one: its dot is the product.
        if (!isAggregate(a._t)) return [...walkExpr(a), ...walkExpr(b), WASM_OP.f64Mul];
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
        if (!isAggregate(src._t)) return [...walkExpr(src), WASM_OP.f64Abs];
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
        if (!isAggregate(a._t)) return [...walkExpr(a), ...walkExpr(b), WASM_OP.f64Sub, WASM_OP.f64Abs];
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
          scalarKindOf(index._t as string) === "float" ? [...walkExpr(index), ...I32_TRUNC_SAT_F64_S] : walkExpr(index);
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
        // Its statements run once however often the value is read, as long as
        // the code that reads it again runs after the first read.
        const stored = onceValue.get(node);
        if (stored !== undefined && regionStack.includes(stored.region)) {
          return [WASM_OP.localGet, ...wasmUleb128(stored.local)];
        }
        const emitted = [...stmts.flatMap((s) => walkStmt(s, currentStmtDepth)), ...walkExpr(final)];
        if (stmts.length === 0 || node._t === "void" || isAggregate(node._t as string)) return emitted;
        const name = `$once${localSlots.length}`;
        addLocal(name, scalarKindOf(node._t as string));
        const local = localSlotIndex(name);
        onceValue.set(node, { local, region: regionStack[regionStack.length - 1]! });
        return [...emitted, WASM_OP.localTee, ...wasmUleb128(local)];
      }

      default:
        throw new Error(`[RMSL] compileWasmFn: unsupported node type in expression position: "${node.type}"`);
    }
  }

  /**
   * Lowers a for/while loop to block{loop{block{cond; brIf <exit>; body};
   * update; br <top>}}. break exits via the outer block, continue via the
   * inner one, which runs the update next. The condition sits in the inner
   * block with the body, so a statement it runs breaks and continues from
   * the same depth the body does.
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
    currentStmtDepth = continueDepth;
    const { condBytes, bodyBytes, updateBytes } = inRegion(() => ({
      condBytes: walkExpr(condNode),
      bodyBytes: walkStmt(bodyNode, continueDepth),
      updateBytes: updateNode ? walkStmt(updateNode, depth + 2) : [],
    }));
    loopStack.pop();
    return [
      ...initBytes,
      WASM_OP.block,
      WASM_BLOCKTYPE_VOID,
      WASM_OP.loop,
      WASM_BLOCKTYPE_VOID,
      WASM_OP.block,
      WASM_BLOCKTYPE_VOID,
      ...condBytes,
      WASM_OP.i32Eqz,
      WASM_OP.brIf,
      ...wasmUleb128(2),
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
    const outerDepth = currentStmtDepth;
    currentStmtDepth = depth;
    try {
      return walkStmtNode(node, depth);
    } finally {
      currentStmtDepth = outerDepth;
    }
  }

  /** {@link walkStmt} once `currentStmtDepth` is `depth`. */
  function walkStmtNode(node: any, depth: number): number[] {
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

        if (target.type === "vectorElement") return emitVectorElementStore(target, rhs);
        if (target.type === "matrixElement") return emitMatrixColumnStore(target, columnRows(target.params[0]), rhs);
        if (target.type === "swizzle") {
          const { base, pattern } = resolveSwizzleTarget(target);
          if ((base as any).type === "storageElement") return emitStorageComponentStores(base, [...pattern], rhs);
          if ((base as any).type === "matrixElement") {
            return emitMatrixColumnStore(
              base,
              [...pattern].map((ch) => COMPONENT_INDEX[ch]!),
              rhs,
            );
          }
          const baseAddr = nodeAddress(base);
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
          return emitStorageElementStore(target, rhs);
        } else if (target.type === "varying") {
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

        const thenBytes = inRegion(() => walkStmt(node.params[1], depth + 1));
        const elseNode = node.params[2];
        return [
          ...cond,
          WASM_OP.if_,
          WASM_BLOCKTYPE_VOID,
          ...thenBytes,
          ...(elseNode ? [WASM_OP.else_, ...inRegion(() => walkStmt(elseNode, depth + 1))] : []),
          WASM_OP.end,
        ];
      }
      case "for": {
        const [initNode, condNode, updateNode, bodyNode] = node.params;
        // The update slot of a GLSL, WGSL or JavaScript `for` takes no block, so
        // a program that put one there would run on this target alone.
        if (holdsBlock(updateNode)) {
          throw new Error(FOR_UPDATE_BLOCK_MESSAGE);
        }
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
        // discard leaves early exactly like return, with the same exit sentinel,
        // after it raises the flag the host reads.
        const sentinel = needsResult ? [] : resultKind === "float" ? f64ConstBytes(0) : i32ConstBytes(0); // needsResult exits a void block; otherwise carry a zero for the block's result type
        const raise = node.type === "discard" ? storeComponent(discardAddress!, "int", 0, i32ConstBytes(1)) : [];
        return [...raise, ...sentinel, WASM_OP.br, ...wasmUleb128(depth - EXIT_BLOCK_DEPTH)];
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

  /**
   * Type index for `(f64) -> f64`, the shared signature every unary math
   * import (`sin`, `sqrt`, ...) uses — created once and deduped.
   */
  function unaryImportTypeIdx(): number {
    if (unaryImportType === null) {
      unaryImportType = typeEntries.length;
      typeEntries.push([WASM_FUNC, ...wasmVec([[WASM_F64]]), ...wasmVec([[WASM_F64]])]);
    }
    return unaryImportType;
  }

  /**
   * Type index for `(f64, f64) -> f64`, the shared signature every binary
   * math import (`pow`, `atan2`) uses — created once and deduped.
   */
  function binaryImportTypeIdx(): number {
    if (binaryImportType === null) {
      binaryImportType = typeEntries.length;
      typeEntries.push([WASM_FUNC, ...wasmVec([[WASM_F64], [WASM_F64]]), ...wasmVec([[WASM_F64]])]);
    }
    return binaryImportType;
  }
}

/**
 * Calls a WASM export with the first `length` numbers of `args`, and returns
 * what it returns. A call through `apply` or a spread converts the list on
 * every call, so up to eleven arguments are passed one by one, which
 * allocates nothing.
 */
function callExport(f: (...args: number[]) => number | void, args: readonly number[], length: number): number | void {
  const a = args;
  switch (length) {
    case 0:
      return f();
    case 1:
      return f(a[0]!);
    case 2:
      return f(a[0]!, a[1]!);
    case 3:
      return f(a[0]!, a[1]!, a[2]!);
    case 4:
      return f(a[0]!, a[1]!, a[2]!, a[3]!);
    case 5:
      return f(a[0]!, a[1]!, a[2]!, a[3]!, a[4]!);
    case 6:
      return f(a[0]!, a[1]!, a[2]!, a[3]!, a[4]!, a[5]!);
    case 7:
      return f(a[0]!, a[1]!, a[2]!, a[3]!, a[4]!, a[5]!, a[6]!);
    case 8:
      return f(a[0]!, a[1]!, a[2]!, a[3]!, a[4]!, a[5]!, a[6]!, a[7]!);
    case 9:
      return f(a[0]!, a[1]!, a[2]!, a[3]!, a[4]!, a[5]!, a[6]!, a[7]!, a[8]!);
    case 10:
      return f(a[0]!, a[1]!, a[2]!, a[3]!, a[4]!, a[5]!, a[6]!, a[7]!, a[8]!, a[9]!);
    case 11:
      return f(a[0]!, a[1]!, a[2]!, a[3]!, a[4]!, a[5]!, a[6]!, a[7]!, a[8]!, a[9]!, a[10]!);
    default:
      return f(...a.slice(0, length));
  }
}

/** Writes `args` into `into`, which keeps them from one call to the next, and returns how many it wrote. */
function copyArgs(into: number[], args: readonly number[]): number {
  for (let i = 0; i < args.length; i++) into[i] = args[i]!;
  return args.length;
}

/**
 * Instantiates a compiled module and binds it to JS: marshals params and
 * textures into memory/args, calls the function, and reads results back
 * into a CpuProgramResult.
 *
 * Split out from `compileWasmRoutine` so a build-time precompile step (see
 * `precompileWasm` in `../vite/vite.ts`) can ship just the compiled bytes
 * and this instantiation glue — never the graph builder or bytecode
 * emitter that produced them.
 */
/**
 * Marshals a `CpuShaderContext` into one compiled module's own memory and
 * scalar args — the shared translation both `instantiateWasmRoutine` (one call per
 * `run()`/`draw()`/`compute()`) and the rasterizer's `compileWasm` (one call per
 * `draw()`, marshalling the vertex and fragment modules independently) need.
 */
export function createWasmInputMarshaller(
  params: readonly WasmParam[],
  textureHeapBase: number,
  memory: WebAssembly.Memory,
  /** Round each float input to 32 bits, for a program compiled at `float: "f32"`. */
  float32 = false,
): {
  /** With `heapStart`, the texture heap starts there for this call, as when another stage's heap lies before it. */
  marshal(ctx: CpuShaderContext, heapStart?: number): { args: number[]; heapEnd: number };
  /** Where the heap would end for `ctx`, without writing anything. */
  footprint(ctx: CpuShaderContext, heapStart?: number): number;
  writeBackStorages(ctx: CpuShaderContext): void;
} {
  const textureParams = params.filter(
    (p): p is Extract<WasmParam, { kind: "textureMemory" }> => p.kind === "textureMemory",
  );
  const storageParams = params.filter(
    (p): p is Extract<WasmParam, { kind: "storageMemory" }> => p.kind === "storageMemory",
  );
  /** Heap address of each storage buffer as the last `marshal` placed it, in `storageParams` order. */
  const storageHeapAddress: number[] = new Array(storageParams.length);
  const storageViews: (Float64Array | Int32Array | Uint32Array | undefined)[] = new Array(storageParams.length);

  // texture cache: skip re-uploading an unchanged texture object, and only
  // grow the module memory when the total footprint changes between calls
  const lastTexture: (CpuTextureData | undefined)[] = new Array(textureParams.length);
  const textureSizes: number[] = new Array(textureParams.length).fill(-1);
  const textureOffsets: number[] = new Array(textureParams.length).fill(0);
  let lastHeapBase: number | undefined;

  /** What `marshal` returns, and the argument list in it, kept from one call to the next so a call allocates nothing. */
  const args: number[] = [];
  const marshalled = { args, heapEnd: 0 };
  const storageLengths: number[] = new Array(storageParams.length).fill(0);
  let memoryView = new DataView(memory.buffer);

  /** A scalar argument as the program takes it: a float rounded to 32 bits at `float: "f32"`. */
  function scalarArg(shaderType: ShaderType, value: number): number {
    return float32 && scalarKindOf(shaderType) === "float" ? Math.fround(value) : value;
  }

  /** A view of the memory, made again only when the memory grows and its buffer changes. */
  function viewOfMemory(): DataView {
    if (memoryView.buffer !== memory.buffer) memoryView = new DataView(memory.buffer);
    return memoryView;
  }

  /**
   * Appends each texture's pixels after the compiled layout (growing memory
   * when the total footprint changes), then each storage buffer after the
   * textures, and collects the scalar WASM args. Returns the heap end —
   * where a draw buffer/further scratch can start.
   */
  function marshal(ctx: CpuShaderContext, heapStart = textureHeapBase): { args: number[]; heapEnd: number } {
    let textureHeapEnd = heapStart;
    if (textureParams.length > 0) {
      let needsRepack = heapStart !== lastHeapBase;
      let heapCursor = heapStart;
      for (let i = 0; i < textureParams.length; i++) {
        const p = textureParams[i]!;
        const size = textureByteSize((ctx.textures as any)?.[p.slot] as CpuTextureData, p.samplerType.endsWith("Cube"));
        if (size !== textureSizes[i]) needsRepack = true;
        textureSizes[i] = size;
        textureOffsets[i] = heapCursor;
        heapCursor += size;
      }
      textureHeapEnd = heapCursor;

      if (needsRepack && heapCursor > memory.buffer.byteLength) {
        memory.grow(Math.ceil((heapCursor - memory.buffer.byteLength) / 65536));
      }
      for (let i = 0; i < textureParams.length; i++) {
        const p = textureParams[i]!;
        const texture = (ctx.textures as any)?.[p.slot] as CpuTextureData;
        if (!needsRepack && texture === lastTexture[i]) continue;
        writeTextureToMemory(
          viewOfMemory(),
          p.metadataAddress,
          textureOffsets[i]!,
          texture,
          p.samplerType.endsWith("Cube"),
        );
        lastTexture[i] = texture;
      }
      lastHeapBase = heapStart;
    }
    const heapEnd = marshalStorages(ctx, textureHeapEnd);
    const view = viewOfMemory();
    let argCount = 0;
    for (const p of params) {
      switch (p.kind) {
        case "textureMemory":
          break;
        case "param":
          args[argCount++] = scalarArg(p.shaderType, (ctx.params as any)?.[p.name] as number);
          break;
        case "uniform":
          // An unset uniform reads zero, as in a zeroed GPU uniform buffer.
          args[argCount++] = scalarArg(p.shaderType, ((ctx.uniforms as any)?.[p.slot] as number | undefined) ?? 0);
          break;
        case "attribute":
          args[argCount++] = scalarArg(p.shaderType, (ctx.attributes as any)?.[p.slot] as number);
          break;
        case "varying":
          args[argCount++] = scalarArg(p.shaderType, (ctx.varyings as any)?.[p.slot] as number);
          break;
        case "invocationIndex":
          args[argCount++] = ctx.index ?? 0;
          break;
        case "paramMemory":
          writeAggregateToMemory(view, p.address, p.shaderType, (ctx.params as any)?.[p.name], false, float32);
          break;
        case "uniformMemory": {
          // Read once: each read of a float the host set boxes it anew.
          const value = (ctx.uniforms as any)?.[p.slot];
          if (value === undefined) break; // left as the zeroed memory it starts as
          if (typeof value !== "number" || isAggregate(p.shaderType)) {
            writeValueToMemory(view, p.address, p.shaderType, value, p.narrow, float32);
          } else if (scalarKindOf(p.shaderType) !== "float") view.setInt32(p.address, value, true);
          else if (p.narrow) view.setFloat32(p.address, value, true);
          else view.setFloat64(p.address, float32 ? Math.fround(value) : value, true);
          break;
        }
        case "uniformArrayMemory":
          if ((ctx.uniforms as any)?.[p.slot] === undefined) break;
          writeArrayToMemory(
            view,
            p.address,
            p.shaderType,
            p.length,
            (ctx.uniforms as any)?.[p.slot],
            p.elementStride,
            p.narrow,
            float32,
          );
          break;
        case "attributeMemory":
          writeValueToMemory(view, p.address, p.shaderType, (ctx.attributes as any)?.[p.slot]);
          break;
        case "varyingMemory":
          writeValueToMemory(view, p.address, p.shaderType, (ctx.varyings as any)?.[p.slot]);
          break;
        case "fragCoordMemory":
          // the host's fragCoord for CPU invocations; draw() overwrites it per pixel — harmless
          writeAggregateToMemory(view, p.address, "vec2", ctx.fragCoord ?? [0, 0], false, float32);
          break;
        case "storageMemory":
          break;
      }
    }
    marshalled.heapEnd = heapEnd;
    return marshalled;
  }

  /** The end of the heap for `ctx` from `heapStart`: each texture's pixels, then each storage buffer. */
  function footprint(ctx: CpuShaderContext, heapStart = textureHeapBase): number {
    let end = heapStart;
    for (const p of textureParams) {
      const tex = (ctx.textures as any)?.[p.slot] as CpuTextureData;
      end += textureByteSize(tex, p.samplerType.endsWith("Cube"));
    }
    if (storageParams.length === 0) return end;
    let cursor = Math.ceil(end / 8) * 8;
    storageParams.forEach((p) => {
      if (ctx.storageBuffers?.[p.slot]) return;
      cursor = Math.ceil((cursor + elementsOf(ctx, p) * storageElementSize(p.shaderType)) / 8) * 8;
    });
    return cursor;
  }

  /** The elements of the flat array the host passes for `p`, which holds their components one after another. */
  function elementsOf(ctx: CpuShaderContext, p: { slot: string; shaderType: ShaderType }): number {
    const array = (ctx.storages as any)?.[p.slot] as ArrayLike<number> | undefined;
    return Math.floor((array?.length ?? 0) / componentCountOf(p.shaderType));
  }

  /**
   * Copies every storage buffer into the heap from `heapBase` on, and points
   * each slot's metadata at it. A write-only buffer is copied in too: it is
   * copied back out whole, so an element no invocation wrote has to come
   * back unchanged.
   */
  function marshalStorages(ctx: CpuShaderContext, heapBase: number): number {
    if (storageParams.length === 0) return heapBase;
    let cursor = Math.ceil(heapBase / 8) * 8;
    for (let i = 0; i < storageParams.length; i++) {
      const p = storageParams[i]!;
      const resident = ctx.storageBuffers?.[p.slot];
      storageLengths[i] = resident?.length ?? elementsOf(ctx, p);
      if (resident) {
        storageHeapAddress[i] = resident.address;
        continue;
      }
      storageHeapAddress[i] = cursor;
      cursor = Math.ceil((cursor + storageLengths[i]! * storageElementSize(p.shaderType)) / 8) * 8;
    }
    if (cursor > memory.buffer.byteLength) {
      memory.grow(Math.ceil((cursor - memory.buffer.byteLength) / 65536));
    }
    const view = viewOfMemory();
    for (let i = 0; i < storageParams.length; i++) {
      const p = storageParams[i]!;
      const array = (ctx.storages as any)?.[p.slot] as ArrayLike<number> | undefined;
      view.setInt32(p.metadataAddress + STORAGE_META_DATA_ADDR, storageHeapAddress[i]!, true);
      view.setInt32(p.metadataAddress + STORAGE_META_LENGTH, storageLengths[i]!, true);
      if (!array || ctx.storageBuffers?.[p.slot]) continue;
      const heap = componentView(i, p.shaderType, storageLengths[i]!);
      if (array.length === heap.length) heap.set(array);
      else for (let k = 0; k < heap.length; k++) heap[k] = array[k]!;
    }
    return cursor;
  }

  /**
   * A typed-array view over the components of storage buffer `i` in the heap,
   * kept until the memory grows or the buffer moves or changes length, so a
   * dispatch copies a buffer in and out in one `set()` and allocates nothing.
   */
  function componentView(i: number, shaderType: ShaderType, length: number): Float64Array | Int32Array | Uint32Array {
    const base = storageHeapAddress[i]!;
    const count = length * componentCountOf(shaderType);
    const kept = storageViews[i];
    if (kept && kept.buffer === memory.buffer && kept.byteOffset === base && kept.length === count) return kept;
    const kind = isAggregate(shaderType) ? elementKindOf(shaderType) : scalarKindOf(shaderType);
    // The heap holds a float in 64 bits at either width.
    const made = new (typedArrayOfKind(kind, false))(memory.buffer, base, count) as
      Float64Array | Int32Array | Uint32Array;
    storageViews[i] = made;
    return made;
  }

  /** Copies every writable storage buffer back into the caller's array, where the last `marshal` placed it. */
  function writeBackStorages(ctx: CpuShaderContext): void {
    for (let i = 0; i < storageParams.length; i++) {
      const p = storageParams[i]!;
      if (p.access === "read" || !p.written || ctx.storageBuffers?.[p.slot]) continue;
      const array = (ctx.storages as any)?.[p.slot] as { length: number; [e: number]: number } | undefined;
      if (!array) continue;
      const heap = componentView(i, p.shaderType, elementsOf(ctx, p));
      if (ArrayBuffer.isView(array) && array.length === heap.length) (array as unknown as Float64Array).set(heap);
      else for (let k = 0; k < heap.length; k++) array[k] = heap[k]!;
    }
  }

  return { marshal, footprint, writeBackStorages };
}

/** Heap bytes one element of a storage buffer of `shaderType` takes: f64 per float component, i32 otherwise. */
function storageElementSize(shaderType: ShaderType): number {
  const kind = isAggregate(shaderType) ? elementKindOf(shaderType) : scalarKindOf(shaderType);
  return componentCountOf(shaderType) * componentSizeOf(kind);
}

export function instantiateWasmProgram(
  compiled: CompiledWasm,
  name: string,
  externalMemory?: WebAssembly.Memory,
): CpuProgram {
  const {
    bytes,
    params,
    resultType,
    textureHeapBase,
    memoryPages,
    sharedMemory,
    maxMemoryPages,
    draw: drawOutput,
    compute: hasCompute,
    float32,
  } = compiled;

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
  const wasmDraw = drawOutput ? (instance.exports.draw as (...args: number[]) => void) : undefined;
  const wasmCompute = hasCompute ? (instance.exports.compute as (...args: number[]) => void) : undefined;

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

  const { marshal: marshalInputs, writeBackStorages } = createWasmInputMarshaller(
    params,
    textureHeapBase,
    memory,
    float32,
  );
  /** The arguments of the `draw` and `compute` exports: the marshalled ones, then their own. Kept between calls. */
  const drawArgs: number[] = [];
  const computeArgs: number[] = [];
  const discardAddress = params.find((p) => p.kind === "discardMemory")?.address;

  /**
   * `CpuRoutine.run`: marshals `ctx` into the compiled function's args
   * and memory, calls it once, and reads back its result (a bare value, or
   * a `CpuProgramResult` for a stage program — see `marshalInputs`/the
   * `outputParams` loop below).
   */
  function run(ctx: CpuShaderContext): number | boolean | CpuProgramResult | null {
    const { args } = marshalInputs(ctx);
    if (discardAddress !== undefined) new DataView(memory.buffer).setInt32(discardAddress, 0, true);
    const result = callExport(wasmMain, args, args.length) as number;
    writeBackStorages(ctx);
    const view = new DataView(memory.buffer); // fresh: marshalInputs may have just grown (and detached) the buffer
    if (discardAddress !== undefined && view.getInt32(discardAddress, true) !== 0) return null;

    // scalar mode: reinterpret the raw i32 — the WASM boundary returns it
    // signed, so a uint result needs a >>> 0 re-read
    if (outputParams.length === 0) {
      if (resultType === "bool") return result !== 0;
      if (resultType === "uint") return result >>> 0;
      return result;
    }

    const shaderResult: CpuProgramResult = {};
    for (const p of outputParams) {
      switch (p.kind) {
        case "outputMemory":
          (shaderResult.outputs ??= {})[p.slot] = readValueFromMemory(view, p.address, p.shaderType, float32);
          break;
        case "varyingOutputMemory":
          (shaderResult.varyings ??= {})[p.slot] = readValueFromMemory(view, p.address, p.shaderType, float32);
          break;
        case "positionMemory":
          shaderResult.position = readAggregateFromMemory(view, p.address, "vec4", float32) as unknown as number[];
          break;
        case "fragDepthMemory":
          shaderResult.fragDepth = view.getFloat64(p.address, true);
          break;
        case "valueMemory":
          shaderResult.value = readValueFromMemory(view, p.address, p.shaderType, float32);
          break;
      }
    }
    // a program that writes no output, varying, position or depth returns its value bare, as the JS target does
    if (Object.keys(shaderResult).length === 1 && "value" in shaderResult)
      return shaderResult.value as number | boolean;
    return shaderResult;
  }

  /** The typed array the module's `draw` writes its components as: 32-bit floats at `float: "f32"`. */
  const DrawArray = typedArrayOfKind(drawOutput?.kind ?? "float", float32 === true);

  /**
   * `CpuRoutine.draw`: marshals `ctx` once, then calls the module's own
   * `draw` export to render the whole `width x height` grid in one call
   * (growing the buffer if needed) — see "A whole grid in one call" in
   * docs/wasm-benchmarks.md for why this exists.
   */
  function draw(ctx: CpuShaderContext, width: number, height: number, out?: CpuDrawBuffer): CpuDrawBuffer {
    if (!drawOutput || !wasmDraw) {
      throw new Error(
        '[RMSL] compileWasmGrid: this function produces no value to render — the grid needs a non-"void" result.',
      );
    }
    const { args, heapEnd } = marshalInputs(ctx);
    const pixelCount = width * height * drawOutput.componentCount;

    // `out` backed by this instance's own (shared) memory: write directly at
    // its offset — this is the zero-copy multi-worker path, where each
    // worker's instance imports the same SharedArrayBuffer-backed memory and
    // `out` is a view pinning where in it this call should land.
    if (out && out.buffer === memory.buffer) {
      if (!(out instanceof DrawArray)) {
        throw new Error(
          `[RMSL] compileWasmGrid: an out in the module's own memory is written in place, so it must be the ${DrawArray.name} the grid fills, not a ${out.constructor.name}.`,
        );
      }
      const n = copyArgs(drawArgs, args);
      drawArgs[n] = width;
      drawArgs[n + 1] = height;
      drawArgs[n + 2] = out.byteOffset;
      callExport(wasmDraw, drawArgs, n + 3);
      return out;
    }

    const bufferBase = Math.ceil(heapEnd / 8) * 8; // align to 8 bytes — the typed-array constructors require it
    const neededBytes = bufferBase + pixelCount * DrawArray.BYTES_PER_ELEMENT;
    if (neededBytes > memory.buffer.byteLength) {
      memory.grow(Math.ceil((neededBytes - memory.buffer.byteLength) / 65536));
    }
    const n = copyArgs(drawArgs, args);
    drawArgs[n] = width;
    drawArgs[n + 1] = height;
    drawArgs[n + 2] = bufferBase;
    callExport(wasmDraw, drawArgs, n + 3);

    // `out` backed by a different buffer than this instance's memory: wasm
    // can only write into the memory it was instantiated with, so this has
    // to copy rather than return a view straight into wasm memory.
    const pixels = new DrawArray(memory.buffer, bufferBase, pixelCount);
    if (out) {
      out.set(pixels);
      return out;
    }
    // A copy: the view would show the pixels of the next draw.
    return pixels.slice();
  }

  /**
   * `CpuRoutine.compute`: marshals `ctx` once, runs every invocation in one
   * call to the module's own `compute` export, and copies the storage
   * buffers back once at the end. A program with neither `storage()` nor
   * `invocationIndex()` has no such export, and loops `main` from here.
   */
  function compute(ctx: CpuShaderContext, count: number): void {
    const { args } = marshalInputs(ctx);
    if (wasmCompute) {
      const n = copyArgs(computeArgs, args);
      computeArgs[n] = count;
      callExport(wasmCompute, computeArgs, n + 1);
    } else for (let i = 0; i < count; i++) callExport(wasmMain, args, args.length);
    writeBackStorages(ctx);
  }

  const storageTypes = {} as Record<string, ShaderType>;
  for (const p of params) {
    if (p.kind === "storageMemory") storageTypes[p.slot] = p.shaderType;
  }

  return { run, draw, compute, storageTypes };
}

/**
 * Turns `compileWasmFn`'s output into a live function of a context: it
 * instantiates the `WebAssembly.Module` and returns its `run`, typed by the
 * type the program returns. Compile the program without a `stage`.
 */
export function instantiateWasmRoutine<A extends ShaderType = ShaderType>(
  compiled: CompiledWasm,
  name: string,
  externalMemory?: WebAssembly.Memory,
): CpuRoutine<A> {
  const program = instantiateWasmProgram(compiled, name, externalMemory);
  return (ctx) => program.run(ctx) as never;
}

/** {@link instantiateWasmRoutine} for a program compiled with `stage: "vertex"`: a vertex stage. */
export function instantiateWasmVertex(
  compiled: CompiledWasm,
  name: string,
  externalMemory?: WebAssembly.Memory,
): VertexStage {
  const program = instantiateWasmProgram(compiled, name, externalMemory);
  return (ctx) => toVertexResult(program.run(ctx));
}

/** {@link instantiateWasmRoutine} for a program compiled with `stage: "fragment"`: a fragment stage. */
export function instantiateWasmFragment<R = unknown>(
  compiled: CompiledWasm,
  name: string,
  externalMemory?: WebAssembly.Memory,
): FragmentStage<R> {
  const program = instantiateWasmProgram(compiled, name, externalMemory);
  return (ctx) => toFragmentResult<R>(program.run(ctx));
}

/** {@link instantiateWasmRoutine} for a program compiled with `stage: "compute"`: a compute stage. */
export function instantiateWasmCompute(
  compiled: CompiledWasm,
  name: string,
  externalMemory?: WebAssembly.Memory,
): ComputeStage {
  const program = instantiateWasmProgram(compiled, name, externalMemory);
  return Object.assign((ctx: CpuShaderContext, count: number) => program.compute(ctx, count), {
    storageTypes: program.storageTypes ?? {},
  });
}

/** {@link instantiateWasmRoutine} for a program of `fragCoord()`: a grid. */
export function instantiateWasmGrid<A extends ShaderType = ShaderType>(
  compiled: CompiledWasm,
  name: string,
  externalMemory?: WebAssembly.Memory,
): CpuGrid<A> {
  const program = instantiateWasmProgram(compiled, name, externalMemory);
  return (ctx, width, height, out) => program.draw(ctx, width, height, out) as GridBuffer<A>;
}

/** What a compile function takes: the options of `compileWasmFn`, without the `stage` and `kind` the function names. */
export type CompileWasmStageOptions = Omit<CompileFnOptions & WasmCompileFields, "stage" | "kind"> & WasmFloatWidth;

type WasmRoots = (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[];

/** Compiles an `Fn` to WASM and instantiates it in one step, for the stages and the grid to take what they give from. */
export function compileWasmProgram(fn: WasmRoots, options: CompileWasmFnOptions): CpuProgram {
  return instantiateWasmProgram(compileWasmFn(fn, options), options.name, options.memory);
}

/**
 * Compiles an `Fn` to WASM and instantiates it in one step, as a function of a
 * context: it reads its parameters and uniforms from `ctx`, and returns its
 * value, typed by the type the program returns. A program that reads what only
 * a stage has, such as `fragCoord()` or a varying, is refused: compile it as a
 * stage or as a grid.
 */
export function compileWasmRoutine<A extends ShaderType, W extends FloatWidth = "f64">(
  fn: (...args: any[]) => Node<A>,
  options: CompileWasmStageOptions & { float?: W },
): CpuRoutine<A, W>;
export function compileWasmRoutine<W extends FloatWidth = "f64">(
  fn: WasmRoots,
  options: CompileWasmStageOptions & { float?: W },
): CpuRoutine<ShaderType, W>;
export function compileWasmRoutine(fn: WasmRoots, options: CompileWasmStageOptions): CpuRoutine {
  return instantiateWasmRoutine(compileWasmFn(fn, { ...options, kind: "routine" }), options.name, options.memory);
}

/** Compiles an `Fn` as a vertex stage: a function that returns the position and the varyings the program writes. */
export function compileWasmVertex<W extends FloatWidth = "f64">(
  fn: WasmRoots,
  options: CompileWasmStageOptions & { float?: W },
): VertexStage<W> {
  return instantiateWasmVertex(
    compileWasmFn(fn, { ...options, stage: "vertex" }),
    options.name,
    options.memory,
  ) as unknown as VertexStage<W>;
}

/**
 * Compiles an `Fn` as a fragment stage: a function that returns the colour and
 * the members of the `outputStruct` the program returns, or `null` for a
 * discarded fragment.
 */
export function compileWasmFragment<R extends Node<ShaderType>, W extends FloatWidth = "f64">(
  fn: (...args: any[]) => R,
  options: CompileWasmStageOptions & { float?: W },
): FragmentStage<R, W>;
export function compileWasmFragment<W extends FloatWidth = "f64">(
  fn: WasmRoots,
  options: CompileWasmStageOptions & { float?: W },
): FragmentStage<unknown, W>;
export function compileWasmFragment(fn: WasmRoots, options: CompileWasmStageOptions): FragmentStage {
  return instantiateWasmFragment(compileWasmFn(fn, { ...options, stage: "fragment" }), options.name, options.memory);
}

/**
 * Compiles an `Fn` as a compute stage: a function that runs the program once
 * per index of a count, and returns nothing. It reads `invocationIndex()` and
 * writes `storage()`, and its `storageTypes` name the buffers it reads.
 */
export function compileWasmCompute(fn: WasmRoots, options: CompileWasmStageOptions): ComputeStage {
  return instantiateWasmCompute(compileWasmFn(fn, { ...options, stage: "compute" }), options.name, options.memory);
}

/** Compiles an `Fn` of `fragCoord()` as a grid: one result for each pixel, in a buffer the type of the result. */
export function compileWasmGrid<A extends ShaderType, W extends FloatWidth = "f64">(
  fn: (...args: any[]) => Node<A>,
  options: CompileWasmStageOptions & { float?: W },
): CpuGrid<A, W> {
  return instantiateWasmGrid<A>(
    compileWasmFn(fn, { ...options, kind: "grid" }),
    options.name,
    options.memory,
  ) as unknown as CpuGrid<A, W>;
}
