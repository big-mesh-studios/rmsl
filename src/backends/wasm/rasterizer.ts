import { i32ConstBytes, WASM_FUNC, WASM_I32, WASM_OP, wasmSection, wasmStrBytes, wasmUleb128, wasmVec } from "./wasm";

/** Import name the rasterizer module expects for the vertex stage's exported function. */
export const RASTERIZER_VERTEX_IMPORT = { module: "vertex", name: "main" } as const;

/** Import name the rasterizer module expects for the fragment stage's exported function. */
export const RASTERIZER_FRAGMENT_IMPORT = { module: "fragment", name: "main" } as const;

/** `rasterize`'s own i32 params, in argument order. */
export const RASTERIZE_PARAMS = [
  "vertexCount",
  "attrSrcBase",
  "attrStrideBytes",
  "vertexAttrDestAddress",
  "vertexPositionAddress",
  "positionsOutBase",
] as const;

const POSITION_BYTES = 32; // vec4 of f64

const local = (index: number) => [WASM_OP.localGet, ...wasmUleb128(index)];

/**
 * A `block { loop { ... } }` copying `length` bytes one at a time from
 * `src(byteCounter)` to `dest(byteCounter)`, using `counterLocal` as its
 * own scratch index — the same block/loop/br_if shape `wasm.ts`'s own
 * `batch()` grid loop uses, generalized to a byte copy instead of a pixel
 * write.
 */
function emitByteCopyLoop(
  dest: (byteOffset: number[]) => number[],
  src: (byteOffset: number[]) => number[],
  length: number[],
  counterLocal: number,
): number[] {
  return [
    ...i32ConstBytes(0),
    WASM_OP.localSet,
    ...wasmUleb128(counterLocal),
    WASM_OP.block,
    0x40,
    WASM_OP.loop,
    0x40,
    ...local(counterLocal),
    ...length,
    WASM_OP.i32GeS,
    WASM_OP.brIf,
    ...wasmUleb128(1),
    ...dest(local(counterLocal)),
    ...src(local(counterLocal)),
    WASM_OP.i32Load8U,
    0x00,
    0x00,
    WASM_OP.i32Store8,
    0x00,
    0x00,
    ...local(counterLocal),
    ...i32ConstBytes(1),
    WASM_OP.i32Add,
    WASM_OP.localSet,
    ...wasmUleb128(counterLocal),
    WASM_OP.br,
    ...wasmUleb128(0),
    WASM_OP.end,
    WASM_OP.end,
  ];
}

/**
 * Builds the rasterizer module's bytes: a `"rasterize"` export
 * ({@link RASTERIZE_PARAMS}) looping over `vertexCount` vertices — for
 * each one, copying `attrStrideBytes` bytes from the attribute buffer at
 * `attrSrcBase` into {@link RASTERIZER_VERTEX_IMPORT}'s own attribute
 * address, calling it, then copying its `vec4` position out to
 * `positionsOutBase`. No triangle setup or fragment calls yet.
 */
export function buildRasterizerModule(): Uint8Array {
  const voidToVoid = [WASM_FUNC, ...wasmVec([]), ...wasmVec([])];
  const rasterizeType = [WASM_FUNC, ...wasmVec(RASTERIZE_PARAMS.map(() => [WASM_I32])), ...wasmVec([])];
  const typeSection = wasmSection(1, wasmVec([voidToVoid, rasterizeType]));

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
  const funcSection = wasmSection(3, wasmVec([[...wasmUleb128(1)]]));

  const exportSection = wasmSection(
    7,
    wasmVec([[...wasmStrBytes("rasterize"), 0x00, ...wasmUleb128(rasterizeFuncIndex)]]),
  );

  const [vertexCount, attrSrcBase, attrStrideBytes, vertexAttrDestAddress, vertexPositionAddress, positionsOutBase] =
    RASTERIZE_PARAMS.map((_, i) => i);
  const iIdx = RASTERIZE_PARAMS.length; // vertex loop counter
  const byteCounterIdx = iIdx + 1; // reused by both copy loops

  const copyAttributeIn = emitByteCopyLoop(
    (offset) => [...local(vertexAttrDestAddress), ...offset, WASM_OP.i32Add],
    (offset) => [
      ...local(attrSrcBase),
      ...local(iIdx),
      ...local(attrStrideBytes),
      WASM_OP.i32Mul,
      WASM_OP.i32Add,
      ...offset,
      WASM_OP.i32Add,
    ],
    [...local(attrStrideBytes)],
    byteCounterIdx,
  );

  const copyPositionOut = emitByteCopyLoop(
    (offset) => [
      ...local(positionsOutBase),
      ...local(iIdx),
      ...i32ConstBytes(POSITION_BYTES),
      WASM_OP.i32Mul,
      WASM_OP.i32Add,
      ...offset,
      WASM_OP.i32Add,
    ],
    (offset) => [...local(vertexPositionAddress), ...offset, WASM_OP.i32Add],
    [...i32ConstBytes(POSITION_BYTES)],
    byteCounterIdx,
  );

  const vertexLoopBody = [
    ...local(iIdx),
    ...local(vertexCount),
    WASM_OP.i32GeS,
    WASM_OP.brIf,
    ...wasmUleb128(1),
    ...copyAttributeIn,
    WASM_OP.call,
    ...wasmUleb128(0), // vertex.main
    ...copyPositionOut,
    ...local(iIdx),
    ...i32ConstBytes(1),
    WASM_OP.i32Add,
    WASM_OP.localSet,
    ...wasmUleb128(iIdx),
    WASM_OP.br,
    ...wasmUleb128(0),
  ];

  const code = [
    ...i32ConstBytes(0),
    WASM_OP.localSet,
    ...wasmUleb128(iIdx),
    WASM_OP.block,
    0x40,
    WASM_OP.loop,
    0x40,
    ...vertexLoopBody,
    WASM_OP.end,
    WASM_OP.end,
  ];

  const localsDecl = wasmVec([
    [...wasmUleb128(1), WASM_I32],
    [...wasmUleb128(1), WASM_I32],
  ]);
  const funcBody = [...localsDecl, ...code, WASM_OP.end];
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
): { rasterize: (...args: number[]) => void } {
  const instance = new WebAssembly.Instance(new WebAssembly.Module(buildRasterizerModule().buffer as ArrayBuffer), {
    vertex: { main: vertexMain },
    fragment: { main: fragmentMain },
    env: { memory },
  });
  return { rasterize: instance.exports.rasterize as (...args: number[]) => void };
}
