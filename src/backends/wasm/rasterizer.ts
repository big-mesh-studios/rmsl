import { WASM_FUNC, WASM_OP, wasmSection, wasmStrBytes, wasmUleb128, wasmVec } from "./wasm";

/** Import name the rasterizer module expects for the vertex stage's exported function. */
export const RASTERIZER_VERTEX_IMPORT = { module: "vertex", name: "main" } as const;

/** Import name the rasterizer module expects for the fragment stage's exported function. */
export const RASTERIZER_FRAGMENT_IMPORT = { module: "fragment", name: "main" } as const;

/**
 * Builds the rasterizer module's bytes: a `"rasterize"` export that calls
 * `RASTERIZER_VERTEX_IMPORT`/`RASTERIZER_FRAGMENT_IMPORT` once each,
 * both required to be zero-arg/zero-return functions over a shared
 * `env.memory` import.
 */
export function buildRasterizerModule(): Uint8Array {
  const voidToVoid = [WASM_FUNC, ...wasmVec([]), ...wasmVec([])];
  const typeSection = wasmSection(1, wasmVec([voidToVoid]));

  const vertexImport = [
    ...wasmStrBytes(RASTERIZER_VERTEX_IMPORT.module),
    ...wasmStrBytes(RASTERIZER_VERTEX_IMPORT.name),
    0x00, // func import
    ...wasmUleb128(0), // type 0: void -> void
  ];
  const fragmentImport = [
    ...wasmStrBytes(RASTERIZER_FRAGMENT_IMPORT.module),
    ...wasmStrBytes(RASTERIZER_FRAGMENT_IMPORT.name),
    0x00,
    ...wasmUleb128(0),
  ];
  const memoryImport = [
    ...wasmStrBytes("env"),
    ...wasmStrBytes("memory"),
    0x02, // memory import
    0x00, // limits: min only
    ...wasmUleb128(1),
  ];
  const importSection = wasmSection(2, wasmVec([vertexImport, fragmentImport, memoryImport]));

  const rasterizeFuncIndex = 2; // 0: vertex import, 1: fragment import
  const funcSection = wasmSection(3, wasmVec([[...wasmUleb128(0)]]));

  const exportSection = wasmSection(
    7,
    wasmVec([[...wasmStrBytes("rasterize"), 0x00, ...wasmUleb128(rasterizeFuncIndex)]]),
  );

  const body = [WASM_OP.call, ...wasmUleb128(0), WASM_OP.call, ...wasmUleb128(1), WASM_OP.end];
  const funcBody = [...wasmVec([]), ...body];
  const codeSection = wasmSection(10, wasmVec([[...wasmUleb128(funcBody.length), ...funcBody]]));

  return new Uint8Array([
    0x00,
    0x61,
    0x73,
    0x6d, // "\0asm"
    0x01,
    0x00,
    0x00,
    0x00, // version 1
    ...typeSection,
    ...importSection,
    ...funcSection,
    ...exportSection,
    ...codeSection,
  ]);
}

/**
 * Instantiates {@link buildRasterizerModule}'s module against a specific
 * vertex/fragment pair's exported `main` functions and the `memory` those
 * two were themselves instantiated with, returning its `rasterize` export.
 */
export function instantiateRasterizer(
  vertexMain: () => void,
  fragmentMain: () => void,
  memory: WebAssembly.Memory,
): { rasterize: () => void } {
  const instance = new WebAssembly.Instance(new WebAssembly.Module(buildRasterizerModule().buffer as ArrayBuffer), {
    vertex: { main: vertexMain },
    fragment: { main: fragmentMain },
    env: { memory },
  });
  return { rasterize: instance.exports.rasterize as () => void };
}
