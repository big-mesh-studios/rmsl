import { Node, ShaderType, var_ } from "./rmsl-core";
import { CompileFnOptions } from "./rmsl-compiler-shared";
import { JsShaderContext } from "./rmsl-compile-js";
// === WASM backend (early — see ROADMAP.md for what this does and doesn't
// cover yet) ===
//
// Compiles a plain, non-stage Fn straight to a WASM binary module instead of
// JS source, for the same CPU-eval niche `compileJS` serves (screen picking,
// ray-march hit tests) where per-call overhead matters. Everything is f64;
// there is no linear memory yet, so a vec3 crosses the WASM boundary as
// three scalar params rather than one aggregate value. Op coverage is
// intentionally narrow — exactly the subset a throwaway prototype validated
// for both correctness and a real speedup over `compileJS` before this was
// promoted to a real backend. ROADMAP.md maps what's next.

/** One entry per parameter the compiled WASM function takes, in call order. */
export type WasmParam =
  | { kind: "param"; name: string }
  | { kind: "uniform"; slot: string; axis?: "x" | "y" | "z" };

export type CompiledWasm = {
  /** The raw WASM binary module, exporting one function named `options.name`. */
  bytes: Uint8Array;
  params: WasmParam[];
};

export const WASM_OP = {
  end: 0x0b,
  localGet: 0x20,
  localSet: 0x21,
  f64Const: 0x44,
  f64Gt: 0x64,
  f64Add: 0xa0,
  f64Sub: 0xa1,
  f64Mul: 0xa2,
  f64Sqrt: 0x9f,
  if_: 0x04,
  else_: 0x05,
} as const;

export const WASM_F64 = 0x7c;
export const WASM_FUNC = 0x60;
export const WASM_BLOCKTYPE_VOID = 0x40;
export const WASM_VEC3_AXES = ["x", "y", "z"] as const;

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
 * Compile an Fn to a raw WASM binary module. `fn` must return a plain
 * `"float"` — no multi-return, no `output()`/`varying()`/`attribute()`, and
 * no control flow beyond `If`/`Else` (see ROADMAP.md for the rest).
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
  if (root._t !== "float") {
    throw new Error(`[RMSL] compileWasmFn only supports a "float" result so far, got "${root._t}".`);
  }

  const fnParamNames = new Set(options.params.map(p => p.name));

  // --- pass 1: collect the WASM param and local index space, in the order
  // the tree first refers to each one, before any instruction bytes
  // reference an index. ---
  const params: WasmParam[] = [];
  const paramIndex = new Map<string, number>();
  const localSlots: string[] = [];
  const localIndex = new Map<string, number>();

  function addParam(spec: WasmParam, key: string): void {
    if (!paramIndex.has(key)) {
      paramIndex.set(key, params.length);
      params.push(spec);
    }
  }
  function addLocal(varName: string): void {
    if (!localIndex.has(varName)) {
      localIndex.set(varName, localSlots.length);
      localSlots.push(varName);
    }
  }

  function collect(node: any): void {
    if (node === null || typeof node !== "object") return;
    if (node.type === "var" && fnParamNames.has(node.value?.varName)) {
      addParam({ kind: "param", name: node.value.varName }, `param:${node.value.varName}`);
    } else if (node.type === "uniform") {
      const v = node.value;
      if (v.shaderType === "vec3") {
        for (const axis of WASM_VEC3_AXES) {
          addParam({ kind: "uniform", slot: v.slot, axis }, `uniform:${v.slot}.${axis}`);
        }
      } else {
        addParam({ kind: "uniform", slot: v.slot }, `uniform:${v.slot}`);
      }
    } else if (node.type === "let") {
      addLocal(node.params[0].value.varName);
    }
    if (Array.isArray(node.params)) for (const p of node.params) collect(p);
  }
  collect(root);

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

  /** Expression context: leaves exactly one value (f64, or i32 for a
   * comparison) on the stack. */
  function walkExpr(node: any): number[] {
    switch (node.type) {
      case "float":
        return [WASM_OP.f64Const, ...wasmF64Bytes(node.value)];
      case "var":
        if (fnParamNames.has(node.value.varName)) {
          return [WASM_OP.localGet, ...wasmUleb128(paramSlotIndex(`param:${node.value.varName}`))];
        }
        return [WASM_OP.localGet, ...wasmUleb128(localSlotIndex(node.value.varName))];
      case "uniform":
        return [WASM_OP.localGet, ...wasmUleb128(paramSlotIndex(`uniform:${node.value.slot}`))];
      case "add":
        return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), WASM_OP.f64Add];
      case "sub":
        return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), WASM_OP.f64Sub];
      case "mul":
        return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), WASM_OP.f64Mul];
      case "sqrt":
        return [...walkExpr(node.params[0]), WASM_OP.f64Sqrt];
      case "greaterThan":
        return [...walkExpr(node.params[0]), ...walkExpr(node.params[1]), WASM_OP.f64Gt];
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
  // [...priorStatements, returnValue] (see `Fn` above); one built as a bare
  // expression (no Fn wrapper, no statements) is just that expression. Both
  // are valid roots here.
  const code = root.type === "seq"
    ? [...(root.params.slice(0, -1) as any[]).flatMap(walkStmt), ...walkExpr(root.params[root.params.length - 1])]
    : walkExpr(root);

  const paramTypes = params.map(() => [WASM_F64]);
  const typeSection = wasmSection(1, wasmVec([
    [WASM_FUNC, ...wasmVec(paramTypes), ...wasmVec([[WASM_F64]])],
  ]));
  const funcSection = wasmSection(3, wasmVec([[0]]));
  const nameBytes = [...new TextEncoder().encode(options.name)];
  const exportSection = wasmSection(7, wasmVec([
    [...wasmUleb128(nameBytes.length), ...nameBytes, 0x00, 0x00],
  ]));
  const localsDecl = localSlots.length > 0
    ? wasmVec([[...wasmUleb128(localSlots.length), WASM_F64]])
    : wasmVec([]);
  const funcBody = [...localsDecl, ...code, WASM_OP.end];
  const codeSection = wasmSection(10, wasmVec([
    [...wasmUleb128(funcBody.length), ...funcBody],
  ]));

  const bytes = new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, // "\0asm"
    0x01, 0x00, 0x00, 0x00, // version 1
    ...typeSection,
    ...funcSection,
    ...exportSection,
    ...codeSection,
  ]);

  return { bytes, params };
}

/**
 * Compile an Fn to a callable, `(ctx) => number`, matching `compileJS`'s call
 * signature for the subset of the DSL this backend covers so far — a
 * `JsShaderContext`'s `params`/`uniforms` in, a `float` out.
 */
export function compileWasm(
  fn: (...args: any[]) => Node<ShaderType>,
  options: CompileFnOptions,
): (ctx: JsShaderContext) => number {
  const { bytes, params } = compileWasmFn(fn, options);
  const instance = new WebAssembly.Instance(new WebAssembly.Module(bytes.buffer as ArrayBuffer));
  const wasmMain = instance.exports.main as (...args: number[]) => number;
  const AXIS_INDEX = { x: 0, y: 1, z: 2 } as const;
  return (ctx: JsShaderContext): number => {
    const args = params.map(p => {
      if (p.kind === "param") return (ctx.params as any)?.[p.name] as number;
      const value = (ctx.uniforms as any)?.[p.slot];
      return p.axis ? (value as number[])[AXIS_INDEX[p.axis]] : (value as number);
    });
    return wasmMain(...args);
  };
}

