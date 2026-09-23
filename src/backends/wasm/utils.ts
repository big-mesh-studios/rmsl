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
  /** `drop`: pop and discard the top of the stack. */
  drop: 0x1a,
  /** `select`: pick one of two values based on an i32 condition. */
  select: 0x1b,

  /** `i32.const`: push a signed LEB128 i32 constant. */
  i32Const: 0x41,
  /** `f64.const`: push a f64 constant (8 raw bytes). */
  f64Const: 0x44,

  /** `i32.load`: load an i32 from memory. */
  i32Load: 0x28,
  /** `i32.load8_u`: load an unsigned byte, zero-extended to i32. */
  i32Load8U: 0x2d,
  /** `f32.load`: load an f32 from memory. */
  f32Load: 0x2a,
  /** `f64.load`: load an f64 from memory. */
  f64Load: 0x2b,
  /** `i32.store`: store an i32 to memory. */
  i32Store: 0x36,
  /** `i32.store8`: truncate an i32 to a byte and store it. */
  i32Store8: 0x3a,
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
  /** `i32.and`: bitwise and. */
  i32And: 0x71,
  /** `i32.or`: bitwise or. */
  i32Or: 0x72,
  /** `i32.xor`: bitwise xor. */
  i32Xor: 0x73,
  /** `i32.shl`: shift left. */
  i32Shl: 0x74,
  /** `i32.shr_s`: shift right, signed (arithmetic). */
  i32ShrS: 0x75,
  /** `i32.shr_u`: shift right, unsigned (logical). */
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
  /** `f64.nearest`: round to nearest, ties to even. */
  f64Nearest: 0x9e,
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
  /** `f64.min`: IEEE-754 minimum (NaN-propagating). */
  f64Min: 0xa4,
  /** `f64.max`: IEEE-754 maximum (NaN-propagating). */
  f64Max: 0xa5,

  /** `i32.trunc_f64_s`: truncate an f64 toward zero to a signed i32 (traps out of range). */
  i32TruncF64S: 0xaa,
  /** `i32.trunc_f64_u`: truncate an f64 toward zero to an unsigned i32 (traps out of range). */
  i32TruncF64U: 0xab,
  /** `f64.convert_i32_s`: convert a signed i32 to f64. */
  f64ConvertI32S: 0xb7,
  /** `f64.convert_i32_u`: convert an unsigned i32 to f64. */
  f64ConvertI32U: 0xb8,

  /** `if` control opcode: starts an if block, followed by a block type, then the then-branch. */
  if_: 0x04,
  /** `else` marker: separates the then-branch from the else-branch inside an if block. */
  else_: 0x05,
} as const;

/**
 * Unsigned LEB128 encoding: the unsigned variable-length integer format used
 * for indices and lengths throughout the wasm binary format. Emits the number
 * in 7-bit groups, least significant first, with the high bit of every byte
 * except the last set to signal continuation.
 */
export function wasmUleb128(n: number): number[] {
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
export function wasmSleb128(n: number): number[] {
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

/**
 * A WASM "name": UTF-8 bytes prefixed by their length as an unsigned LEB128
 * integer. Used everywhere the binary format embeds a string — import/export
 * names, the custom-section name, etc.
 */
export function wasmStrBytes(s: string): number[] {
  const b = [...new TextEncoder().encode(s)];
  return [...wasmUleb128(b.length), ...b];
}

/** Emits i32.const plus the signed LEB128 operand. */
export function i32ConstBytes(n: number): number[] {
  return [WASM_OP.i32Const, ...wasmSleb128(n | 0)];
}

/** Little-endian f64 bytes, as used by f64.const. */
export function wasmF64Bytes(value: number): number[] {
  const buf = new ArrayBuffer(8);
  new DataView(buf).setFloat64(0, value, true);
  return [...new Uint8Array(buf)];
}

/** Emits f64.const plus the raw 8-byte little-endian operand. */
export function f64ConstBytes(n: number): number[] {
  return [WASM_OP.f64Const, ...wasmF64Bytes(n)];
}

/**
 * Wraps `body` as a top-level module section: a one-byte section id, the
 * section's byte length as an unsigned LEB128 integer, then the bytes
 * themselves. A WASM module is just a magic number, a version, and a
 * sequence of these — the decoder reads the id, skips exactly `length`
 * bytes if it doesn't recognize the section, and moves on. The ids
 * `compileWasmFn`/`buildRasterizerModule` emit are the standard ones from
 * the spec's binary format: 1 = Type, 2 = Import, 3 = Function, 7 = Export,
 * 10 = Code.
 */
export function wasmSection(id: number, body: number[]): number[] {
  return [id, ...wasmUleb128(body.length), ...body];
}

/**
 * Wraps `items` as a WASM "vector": the element count as an unsigned LEB128
 * integer, followed by each element's own bytes back to back. This is the
 * binary format's generic list encoding — used for a section's own list of
 * entries (types, imports, functions, exports, ...) and for a function's
 * list of locals — so most section bodies are built by handing their entries
 * to this rather than encoding the count by hand.
 */
export function wasmVec(items: number[][]): number[] {
  return [...wasmUleb128(items.length), ...items.flat()];
}

/**
 * The `align`/`offset` pair a "natural" load/store uses: no declared
 * alignment hint and zero offset, since the caller's own address is already
 * fully computed on the stack.
 */
export const MEMARG_NATURAL: number[] = [0x00, 0x00];

export const local = (index: number): number[] => [WASM_OP.localGet, ...wasmUleb128(index)];
export const localSet = (index: number): number[] => [WASM_OP.localSet, ...wasmUleb128(index)];

/** Turns a binary WASM opcode into a bytes-in, bytes-out combinator: `bin(op)(a, b)` emits `a`, then `b`, then `op`. */
export const bin =
  (op: number) =>
  (a: number[], b: number[]): number[] => [...a, ...b, op];
/** Turns a unary WASM opcode into a bytes-in, bytes-out combinator: `un(op)(a)` emits `a`, then `op`. */
export const un =
  (op: number) =>
  (a: number[]): number[] => [...a, op];

export const fAdd = bin(WASM_OP.f64Add);
export const fSub = bin(WASM_OP.f64Sub);
export const fMul = bin(WASM_OP.f64Mul);
export const fDiv = bin(WASM_OP.f64Div);
export const fMin = bin(WASM_OP.f64Min);
export const fMax = bin(WASM_OP.f64Max);
export const fGe = bin(WASM_OP.f64Ge);
export const fGt = bin(WASM_OP.f64Gt);
export const fLe = bin(WASM_OP.f64Le);
export const fEq = bin(WASM_OP.f64Eq);
export const fFloor = un(WASM_OP.f64Floor);
export const fCeil = un(WASM_OP.f64Ceil);
export const iAdd = bin(WASM_OP.i32Add);
export const iMul = bin(WASM_OP.i32Mul);
export const iGtS = bin(WASM_OP.i32GtS);
export const iGeS = bin(WASM_OP.i32GeS);
export const iAnd = bin(WASM_OP.i32And);
export const iOr = bin(WASM_OP.i32Or);
export const iEq = bin(WASM_OP.i32Eq);
export const iNe = bin(WASM_OP.i32Ne);
export const iDivS = bin(WASM_OP.i32DivS);
export const toF64 = un(WASM_OP.f64ConvertI32S);
export const toI32 = un(WASM_OP.i32TruncF64S);

export const loadF64 = (addr: number[]): number[] => [...addr, WASM_OP.f64Load, ...MEMARG_NATURAL];
export const storeF64 = (addr: number[], value: number[]): number[] => [
  ...addr,
  ...value,
  WASM_OP.f64Store,
  ...MEMARG_NATURAL,
];
export const loadI32 = (addr: number[]): number[] => [...addr, WASM_OP.i32Load, ...MEMARG_NATURAL];

/**
 * `{ ...body }` — a plain structured block, only ever useful so `body` has
 * somewhere to jump to via {@link exitBlockIf} (a `br`/`br_if` with no
 * enclosing block is invalid WASM).
 */
export const block = (body: number[]): number[] => [WASM_OP.block, WASM_BLOCKTYPE_VOID, ...body, WASM_OP.end];

/** `if (cond) { ...then }`, no `else`. */
export const ifThen = (cond: number[], then: number[]): number[] => [
  ...cond,
  WASM_OP.if_,
  WASM_BLOCKTYPE_VOID,
  ...then,
  WASM_OP.end,
];

/**
 * `if (cond) break;` — jumps straight past the rest of the *nearest
 * enclosing* {@link block}/loop body. Only valid directly inside one.
 */
export const exitBlockIf = (cond: number[]): number[] => [...cond, WASM_OP.brIf, ...wasmUleb128(0)];

/**
 * `while (!exitCond) { ...body }`, checked at the top of every iteration —
 * WASM has no native loop-with-condition, so this compiles to the standard
 * `block { loop { br_if exit; body; br continue } }` shape by hand: the
 * outer `block` is what `br 1` (an early "stop looping" `brIf`) exits to,
 * the inner `loop` is what the trailing `br 0` repeats.
 */
export const loopUntil = (exitCond: number[], body: number[]): number[] => [
  WASM_OP.block,
  WASM_BLOCKTYPE_VOID,
  WASM_OP.loop,
  WASM_BLOCKTYPE_VOID,
  ...exitCond,
  WASM_OP.brIf,
  ...wasmUleb128(1),
  ...body,
  WASM_OP.br,
  ...wasmUleb128(0),
  WASM_OP.end,
  WASM_OP.end,
];

/**
 * `for (i = 0; !exitCond(i); i++) { ...body(i) }` — {@link loopUntil} plus
 * the counter-init/increment every counted loop needs, so a call site only
 * has to say what varies: the counter local, where it starts, its exit
 * test, and its own step.
 */
export const forLoop = (
  counter: number,
  start: number[],
  exitCond: number[],
  body: number[],
  step: number[],
): number[] => [
  ...start,
  ...localSet(counter),
  ...loopUntil(exitCond, [...body, ...iAdd(local(counter), step), ...localSet(counter)]),
];
