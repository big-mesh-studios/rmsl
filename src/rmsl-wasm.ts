import { Node, ShaderType, var_ } from "./rmsl-core";
import { CompileFnOptions } from "./rmsl-compiler-shared";
import { JsShaderContext } from "./rmsl-compile-js";
// === WASM backend (see ROADMAP.md for what this does and doesn't cover yet) ===
//
// Compiles a plain, non-stage Fn straight to a WASM binary module instead of
// JS source, for the same CPU-eval niche `compileJS` serves (screen picking,
// ray-march hit tests) where per-call overhead matters. There is no linear
// memory yet, so a vec3 crosses the WASM boundary as three scalar params
// rather than one aggregate value. ROADMAP.md maps what's next.
//
// Every opcode below was checked empirically (a minimal module built and run
// against a known answer) before use, not taken from memory — a
// wrong-but-still-valid opcode produces a module that *runs* and gives the
// wrong number, which is exactly the silent-miscompile failure mode
// CONTRIBUTING.md's testing section is about.

/** One entry per parameter the compiled WASM function takes, in call order. */
export type WasmParam =
  | { kind: "param"; name: string; shaderType: ShaderType }
  | { kind: "uniform"; slot: string; shaderType: ShaderType; axis?: "x" | "y" | "z" };

export type CompiledWasm = {
  /** The raw WASM binary module, exporting one function named `options.name`. */
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
export const WASM_VEC3_AXES = ["x", "y", "z"] as const;

/** A node's scalar value kind — everything at this phase is one of these
 * four, stored as either f64 (`float`) or i32 (the other three). */
type ScalarKind = "float" | "int" | "uint" | "bool";

function scalarKindOf(t: string | undefined): ScalarKind {
  return t === "int" || t === "uint" || t === "bool" ? t : "float";
}

function wasmTypeOf(kind: ScalarKind): number {
  return kind === "float" ? WASM_F64 : WASM_I32;
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
 * `If`/`Else` (see ROADMAP.md for the rest).
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

  // --- pass 1: collect the WASM param/local index space and every math
  // import the program needs, in first-seen order, before any instruction
  // bytes reference an index. ---
  const params: WasmParam[] = [];
  const paramIndex = new Map<string, number>();
  const localSlots: string[] = [];
  const localIndex = new Map<string, number>();
  const localType = new Map<string, ScalarKind>();
  const importsUsed = new Set<string>();

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

  function collect(node: any): void {
    if (node === null || typeof node !== "object") return;
    if (node.type === "var" && fnParamNames.has(node.value?.varName)) {
      const name = node.value.varName;
      addParam({ kind: "param", name, shaderType: paramTypeByName.get(name)! }, `param:${name}`);
    } else if (node.type === "uniform") {
      const v = node.value;
      if (v.shaderType === "vec3") {
        for (const axis of WASM_VEC3_AXES) {
          addParam({ kind: "uniform", slot: v.slot, shaderType: "float", axis }, `uniform:${v.slot}.${axis}`);
        }
      } else {
        addParam({ kind: "uniform", slot: v.slot, shaderType: v.shaderType }, `uniform:${v.slot}`);
      }
    } else if (node.type === "let") {
      addLocal(node.params[0].value.varName, scalarKindOf(node.params[0]._t));
    } else if (MATH_UNARY_IMPORTS.has(node.type) || MATH_BINARY_IMPORTS.has(node.type)) {
      importsUsed.add(node.type);
    } else if (node.type === "exp2") {
      // exp2(x) compiles to a call to the imported pow(2, x) — see walkExpr —
      // so that import has to be registered here too, even though "exp2"
      // itself isn't one of the imported names.
      importsUsed.add("pow");
    }
    if (Array.isArray(node.params)) for (const p of node.params) collect(p);
  }
  collect(root);

  const importNames = [...importsUsed].sort();
  const importIndexOf = new Map(importNames.map((name, i) => [name, i]));

  // --- pass 2: emit instruction bytes against the now-fixed index space. ---
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

      case "dot": {
        const [ax, ay, az] = walkVec3(node.params[0]);
        const [bx, by, bz] = walkVec3(node.params[1]);
        return [
          ...ax, ...bx, WASM_OP.f64Mul,
          ...ay, ...by, WASM_OP.f64Mul, WASM_OP.f64Add,
          ...az, ...bz, WASM_OP.f64Mul, WASM_OP.f64Add,
        ];
      }
      default:
        throw new Error(`[RMSL] compileWasmFn: unsupported node type in expression position: "${node.type}"`);
    }
  }

  /** A vec3 never sits in one WASM value slot here — no linear memory yet to
   * hold it — so it compiles to three independent scalar expressions,
   * combined lazily by whichever op consumes them (`dot`). */
  function walkVec3(node: any): [number[], number[], number[]] {
    switch (node.type) {
      case "uniform": {
        const slot = node.value.slot;
        return WASM_VEC3_AXES.map(axis =>
          [WASM_OP.localGet, ...wasmUleb128(paramSlotIndex(`uniform:${slot}.${axis}`))],
        ) as [number[], number[], number[]];
      }
      case "construct":
        return [walkExpr(node.params[0]), walkExpr(node.params[1]), walkExpr(node.params[2])];
      default:
        throw new Error(`[RMSL] compileWasmFn: unsupported node type in vec3 position: "${node.type}"`);
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
        const varName = node.params[0].value.varName;
        return [...walkExpr(node.params[1]), WASM_OP.localSet, ...wasmUleb128(localSlotIndex(varName))];
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
  // signature per import arity actually used), import section, then the
  // rest as before. ---
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
  const nameBytes = wasmStrBytes(options.name);
  const exportSection = wasmSection(7, wasmVec([
    [...nameBytes, 0x00, ...wasmUleb128(importNames.length)],
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
    ...exportSection,
    ...codeSection,
  ]);

  return { bytes, params, resultType: root._t };
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
  const AXIS_INDEX = { x: 0, y: 1, z: 2 } as const;
  return (ctx: JsShaderContext): number | boolean => {
    const args = params.map(p => {
      if (p.kind === "param") return (ctx.params as any)?.[p.name] as number;
      const value = (ctx.uniforms as any)?.[p.slot];
      return p.axis ? (value as number[])[AXIS_INDEX[p.axis]] : (value as number);
    });
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
