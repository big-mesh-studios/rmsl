import { Node, ShaderType, TYPE_WIDTH, MATRIX_DIMENSIONS, var_ } from "./rmsl-core";
import { CompileFnOptions, COMPONENT_INDEX, resolveSwizzleTarget } from "./rmsl-compiler-shared";
import { JsShaderContext } from "./rmsl-compile-js";
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
  | { kind: "uniformMemory"; slot: string; shaderType: ShaderType; address: number };

export type CompiledWasm = {
  /** The raw WASM binary module, exporting `options.name` and `"memory"`. */
  bytes: Uint8Array;
  params: WasmParam[];
  /** The Fn's declared return type — `compileWasm` reads this to convert a
   * `"bool"` result's 0/1 back to a real boolean, matching `compileJS`. */
  resultType: ShaderType;
};

export const WASM_OP = {
  end: 0x0b,
  localGet: 0x20,
  localSet: 0x21,
  call: 0x10,
  select: 0x1b,

  i32Const: 0x41,
  f64Const: 0x44,

  i32Load: 0x28,
  f64Load: 0x2b,
  i32Store: 0x36,
  f64Store: 0x39,

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
export function compileWasmFn(
  fn: (...args: any[]) => Node<ShaderType>,
  options: CompileFnOptions,
): CompiledWasm {
  const paramNodes = options.params.map(p => var_(p.name, p.type));
  const root = fn(...paramNodes) as any;
  if (Array.isArray(root)) {
    throw new Error("[RMSL] compileWasmFn does not support multi-return functions.");
  }
  const resultKind = scalarKindOf(root._t);
  if (root._t !== "float" && root._t !== "int" && root._t !== "uint" && root._t !== "bool") {
    throw new Error(`[RMSL] compileWasmFn only supports a scalar result so far, got "${root._t}".`);
  }

  const paramTypeByName = new Map(options.params.map(p => [p.name, p.type]));
  const fnParamNames = new Set(options.params.map(p => p.name));

  // --- pass 1: collect the WASM param/local index space, every math import
  // the program needs, and every linear-memory address a vector/matrix value
  // needs (uniforms, params, `let`-bound vars, and one scratch slot per
  // vector/matrix-producing expression node) — all in first-seen order,
  // before any instruction bytes reference an index or address. ---
  const params: WasmParam[] = [];
  const paramIndex = new Map<string, number>();
  const localSlots: string[] = [];
  const localIndex = new Map<string, number>();
  const localType = new Map<string, ScalarKind>();
  const importsUsed = new Set<string>();

  const memoryParams: WasmParam[] = [];
  const paramAddress = new Map<string, number>();
  const varAddress = new Map<string, number>();
  const uniformAddress = new Map<string, number>();
  const scratchAddress = new WeakMap<object, number>();
  let memCursor = 0;

  function allocateFor(t: string): number {
    const addr = memCursor;
    memCursor += componentCountOf(t) * componentSizeOf(elementKindOf(t));
    return addr;
  }

  function addParam(spec: WasmParam, key: string): void {
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
    } else if (node.type === "uniform") {
      const v = node.value;
      if (isAggregate(v.shaderType)) {
        if (!uniformAddress.has(v.slot)) {
          const addr = allocateFor(v.shaderType);
          uniformAddress.set(v.slot, addr);
          memoryParams.push({ kind: "uniformMemory", slot: v.slot, shaderType: v.shaderType, address: addr });
        }
      } else {
        addParam({ kind: "uniform", slot: v.slot, shaderType: v.shaderType }, `uniform:${v.slot}`);
      }
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
      scratchAddress.set(node, allocateFor(node._t as string));
    }
    if (Array.isArray(node.params)) for (const p of node.params) collect(p);
  }
  collect(root);

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

  /** Populate `nodeAddress(node)` with `node`'s value, for any node type
   * `isScratchNode` addresses — a `"var"`/`"uniform"` needs no work (its
   * data is already valid, set by a prior `let`/`assign` or by
   * `compileWasm`'s JS wrapper before the call). Always materializes an
   * aggregate sub-node exactly once, however many of its components end up
   * read, so nesting doesn't blow up proportionally to width. */
  function materializeIfNeeded(node: any): number[] {
    switch (node.type) {
      case "var":
      case "uniform":
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
      case "mul":
      case "div":
        return emitComponentwiseStores(node, nodeAddress(node));
      default:
        if (node.type === node._t && Array.isArray(node.value)) {
          return emitLiteralStores(node, nodeAddress(node));
        }
        throw new Error(`[RMSL] compileWasmFn: unsupported node type in vector position: "${node.type}"`);
    }
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
        out.push(...storeComponent(addr, targetKind, compIndex * compSize, walkExpr(p)));
        compIndex++;
      } else {
        out.push(...materializeIfNeeded(p));
        const pAddr = nodeAddress(p);
        const pKind = elementKindOf(p._t);
        const pCompSize = componentSizeOf(pKind);
        for (let k = 0; k < pWidth; k++) {
          out.push(...storeComponent(addr, targetKind, compIndex * compSize, loadComponent(pAddr, pKind, k * pCompSize)));
          compIndex++;
        }
      }
    }
    return out;
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
      default:
        throw new Error(`[RMSL] compileWasmFn: unsupported node type in expression position: "${node.type}"`);
    }
  }

  /** Statement context: leaves no net stack effect. */
  function walkStmt(node: any): number[] {
    switch (node.type) {
      case "seq": {
        const list = node.params ?? [];
        if (node._t !== "void") {
          throw new Error("[RMSL] compileWasmFn: a value-producing seq belongs at the root, not in statement position");
        }
        return list.flatMap(walkStmt);
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

        const targetType = target._t as string;
        const varName = target.value.varName;
        if (isAggregate(targetType)) {
          const destAddr = fnParamNames.has(varName) ? paramAddress.get(varName)! : varAddress.get(varName)!;
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

        return [...walkExpr(rhs), WASM_OP.localSet, ...wasmUleb128(localSlotIndex(varName))];
      }
      case "if": {
        const cond = walkExpr(node.params[0]);
        const thenBytes = walkStmt(node.params[1]);
        const elseNode = node.params[2];
        return [
          ...cond, WASM_OP.if_, WASM_BLOCKTYPE_VOID,
          ...thenBytes,
          ...(elseNode ? [WASM_OP.else_, ...walkStmt(elseNode)] : []),
          WASM_OP.end,
        ];
      }
      default:
        throw new Error(`[RMSL] compileWasmFn: unsupported node type in statement position: "${node.type}"`);
    }
  }

  // A program built with `Fn(() => { ...; return x; })()` is a "seq" node of
  // [...priorStatements, returnValue] (see the Fn implementation in
  // rmsl-core.ts); one built as a bare expression (no Fn wrapper, no
  // statements) is just that expression. Both are valid roots here.
  const code = root.type === "seq"
    ? [...(root.params.slice(0, -1) as any[]).flatMap(walkStmt), ...walkExpr(root.params[root.params.length - 1])]
    : walkExpr(root);

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
  const resultWasmType = wasmTypeOf(resultKind);
  const mainTypeIdx = typeEntries.length;
  typeEntries.push([WASM_FUNC, ...wasmVec(paramTypes), ...wasmVec([[resultWasmType]])]);

  const typeSection = wasmSection(1, wasmVec(typeEntries));
  const importSection = importEntries.length > 0 ? wasmSection(2, wasmVec(importEntries)) : [];
  const funcSection = wasmSection(3, wasmVec([[mainTypeIdx]]));
  const memoryPages = Math.max(1, Math.ceil(memCursor / 65536));
  const memorySection = wasmSection(5, wasmVec([[0x00, ...wasmUleb128(memoryPages)]]));
  const nameBytes = wasmStrBytes(options.name);
  const exportSection = wasmSection(7, wasmVec([
    [...nameBytes, 0x00, ...wasmUleb128(importNames.length)],
    [...wasmStrBytes("memory"), 0x02, ...wasmUleb128(0)],
  ]));
  // One group per local rather than run-length-compressing consecutive
  // same-type locals — larger than it needs to be, but every group is
  // independently correct, and there's no shared-type run to get wrong.
  const localsDecl = wasmVec(localSlots.map(name => [...wasmUleb128(1), wasmTypeOf(localType.get(name)!)]));
  const funcBody = [...localsDecl, ...code, WASM_OP.end];
  const codeSection = wasmSection(10, wasmVec([
    [...wasmUleb128(funcBody.length), ...funcBody],
  ]));

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

  return { bytes, params: [...params, ...memoryParams], resultType: root._t };
}

/** Write an aggregate value's components into `view` at `address`, using
 * `shaderType` to pick the storage kind/width — the JS-side half of the
 * linear-memory design, run before every call since uniform/param values
 * can change between calls. */
function writeAggregateToMemory(view: DataView, address: number, shaderType: ShaderType, value: any): void {
  const kind = elementKindOf(shaderType);
  const compSize = componentSizeOf(kind);
  const arr = value as ArrayLike<number | boolean>;
  for (let i = 0; i < arr.length; i++) {
    const raw = arr[i];
    const num = typeof raw === "boolean" ? (raw ? 1 : 0) : (raw as number);
    if (kind === "float") view.setFloat64(address + i * compSize, num, true);
    else view.setInt32(address + i * compSize, num, true);
  }
}

/**
 * Compile an Fn to a callable, `(ctx) => number | boolean`, matching
 * `compileJS`'s call signature for the subset of the DSL this backend
 * covers so far — a `JsShaderContext`'s `params`/`uniforms` in, a scalar out.
 */
export function compileWasm(
  fn: (...args: any[]) => Node<ShaderType>,
  options: CompileFnOptions,
): (ctx: JsShaderContext) => number | boolean {
  const { bytes, params, resultType } = compileWasmFn(fn, options);
  // A module that imports nothing ignores an unused "math" namespace, so
  // this is passed unconditionally rather than only when needed. `Math`'s
  // own methods have the same names, so it's handed over directly.
  const instance = new WebAssembly.Instance(new WebAssembly.Module(bytes.buffer as ArrayBuffer), { math: Math as unknown as WebAssembly.ModuleImports });
  const wasmMain = instance.exports.main as (...args: number[]) => number;
  const memory = instance.exports.memory as WebAssembly.Memory;
  const view = new DataView(memory.buffer);
  return (ctx: JsShaderContext): number | boolean => {
    const args: number[] = [];
    for (const p of params) {
      if (p.kind === "param") {
        args.push((ctx.params as any)?.[p.name] as number);
      } else if (p.kind === "uniform") {
        args.push((ctx.uniforms as any)?.[p.slot] as number);
      } else if (p.kind === "paramMemory") {
        writeAggregateToMemory(view, p.address, p.shaderType, (ctx.params as any)?.[p.name]);
      } else {
        writeAggregateToMemory(view, p.address, p.shaderType, (ctx.uniforms as any)?.[p.slot]);
      }
    }
    const result = wasmMain(...args);
    // The JS/WASM call boundary always surfaces an i32 return as a signed
    // number; a "uint" result above 2^31-1 needs reinterpreting as unsigned,
    // the same way ctx.uniforms/ctx.params values are read as unsigned going
    // in (JS numbers don't distinguish, so no equivalent step is needed
    // there — only coming back out through a fixed-width return does).
    if (resultType === "bool") return result !== 0;
    if (resultType === "uint") return result >>> 0;
    return result;
  };
}
