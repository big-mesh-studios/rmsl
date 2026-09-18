import {
  f64ConstBytes,
  i32ConstBytes,
  WASM_F64,
  WASM_FUNC,
  WASM_I32,
  WASM_OP,
  wasmSection,
  wasmStrBytes,
  wasmUleb128,
  wasmVec,
} from "./wasm";

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
  "width",
  "height",
  "fragmentValueAddress",
  "outputBase",
  "varyingBytes",
  "vertexVaryingAddress",
  "fragmentVaryingAddress",
  "varyingsOutBase",
] as const;

/** A `vec4` position or fragment color, stored as 4 f64 components. */
const VEC4_BYTES = 32;

const local = (index: number) => [WASM_OP.localGet, ...wasmUleb128(index)];
const localSet = (index: number) => [WASM_OP.localSet, ...wasmUleb128(index)];
const bin = (op: number) => (a: number[], b: number[]) => [...a, ...b, op];
const un = (op: number) => (a: number[]) => [...a, op];

const fAdd = bin(WASM_OP.f64Add);
const fSub = bin(WASM_OP.f64Sub);
const fMul = bin(WASM_OP.f64Mul);
const fDiv = bin(WASM_OP.f64Div);
const fMin = bin(WASM_OP.f64Min);
const fMax = bin(WASM_OP.f64Max);
const fGe = bin(WASM_OP.f64Ge);
const fLe = bin(WASM_OP.f64Le);
const fEq = bin(WASM_OP.f64Eq);
const fFloor = un(WASM_OP.f64Floor);
const fCeil = un(WASM_OP.f64Ceil);
const iAdd = bin(WASM_OP.i32Add);
const iMul = bin(WASM_OP.i32Mul);
const iGtS = bin(WASM_OP.i32GtS);
const iGeS = bin(WASM_OP.i32GeS);
const iAnd = bin(WASM_OP.i32And);
const iOr = bin(WASM_OP.i32Or);
const iDivS = bin(WASM_OP.i32DivS);
const toF64 = un(WASM_OP.f64ConvertI32S);
const toI32 = un(WASM_OP.i32TruncF64S);
const loadF64 = (addr: number[]) => [...addr, WASM_OP.f64Load, 0x00, 0x00];
const storeF64 = (addr: number[], value: number[]) => [...addr, ...value, WASM_OP.f64Store, 0x00, 0x00];

/** Assigns sequential local indices/types past a function's own params. */
class LocalAllocator {
  private types: number[] = [];
  constructor(private base: number) {}
  alloc(type: number): number {
    this.types.push(type);
    return this.base + this.types.length - 1;
  }
  declBytes(): number[] {
    return wasmVec(this.types.map((t) => [...wasmUleb128(1), t]));
  }
}

/**
 * A `block { loop { ... } }` copying `length` bytes one at a time from
 * `src(byteCounter)` to `dest(byteCounter)`. See rasterizer.md.
 */
function emitByteCopyLoop(
  dest: (byteOffset: number[]) => number[],
  src: (byteOffset: number[]) => number[],
  length: number[],
  counterLocal: number,
): number[] {
  return [
    ...i32ConstBytes(0),
    ...localSet(counterLocal),
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
    ...localSet(counterLocal),
    WASM_OP.br,
    ...wasmUleb128(0),
    WASM_OP.end,
    WASM_OP.end,
  ];
}

/**
 * Builds the rasterizer module's bytes — see rasterizer.md for the full
 * vertex-pass/triangle-pass design and its v1 scope.
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

  const [
    vertexCount,
    attrSrcBase,
    attrStrideBytes,
    vertexAttrDestAddress,
    vertexPositionAddress,
    positionsOutBase,
    width,
    height,
    fragmentValueAddress,
    outputBase,
    varyingBytes,
    vertexVaryingAddress,
    fragmentVaryingAddress,
    varyingsOutBase,
  ] = RASTERIZE_PARAMS.map((_, i) => i);

  const locals = new LocalAllocator(RASTERIZE_PARAMS.length);
  const iIdx = locals.alloc(WASM_I32); // vertex loop counter
  const byteCounterIdx = locals.alloc(WASM_I32); // shared by every byte-copy loop
  const tIdx = locals.alloc(WASM_I32); // triangle loop counter
  const widthFIdx = locals.alloc(WASM_F64);
  const heightFIdx = locals.alloc(WASM_F64);
  const s0xIdx = locals.alloc(WASM_F64);
  const s0yIdx = locals.alloc(WASM_F64);
  const s1xIdx = locals.alloc(WASM_F64);
  const s1yIdx = locals.alloc(WASM_F64);
  const s2xIdx = locals.alloc(WASM_F64);
  const s2yIdx = locals.alloc(WASM_F64);
  const areaIdx = locals.alloc(WASM_F64);
  const minXIdx = locals.alloc(WASM_I32);
  const maxXIdx = locals.alloc(WASM_I32);
  const minYIdx = locals.alloc(WASM_I32);
  const maxYIdx = locals.alloc(WASM_I32);
  const xIdx = locals.alloc(WASM_I32);
  const yIdx = locals.alloc(WASM_I32);
  const pxIdx = locals.alloc(WASM_F64);
  const pyIdx = locals.alloc(WASM_F64);
  const e0Idx = locals.alloc(WASM_F64);
  const e1Idx = locals.alloc(WASM_F64);
  const e2Idx = locals.alloc(WASM_F64);
  const w0Idx = locals.alloc(WASM_F64);
  const w1Idx = locals.alloc(WASM_F64);
  const w2Idx = locals.alloc(WASM_F64);
  const invW0Idx = locals.alloc(WASM_F64);
  const invW1Idx = locals.alloc(WASM_F64);
  const invW2Idx = locals.alloc(WASM_F64);
  const b0Idx = locals.alloc(WASM_F64);
  const b1Idx = locals.alloc(WASM_F64);
  const b2Idx = locals.alloc(WASM_F64);
  const invWIdx = locals.alloc(WASM_F64);
  const numVaryingComponentsIdx = locals.alloc(WASM_I32);
  const componentIdx = locals.alloc(WASM_I32);

  const copyAttributeIn = emitByteCopyLoop(
    (offset) => iAdd(local(vertexAttrDestAddress), offset),
    (offset) => iAdd(iAdd(local(attrSrcBase), iMul(local(iIdx), local(attrStrideBytes))), offset),
    local(attrStrideBytes),
    byteCounterIdx,
  );
  const copyPositionOut = emitByteCopyLoop(
    (offset) => iAdd(iAdd(local(positionsOutBase), iMul(local(iIdx), i32ConstBytes(VEC4_BYTES))), offset),
    (offset) => iAdd(local(vertexPositionAddress), offset),
    i32ConstBytes(VEC4_BYTES),
    byteCounterIdx,
  );
  const copyVaryingOut = emitByteCopyLoop(
    (offset) => iAdd(iAdd(local(varyingsOutBase), iMul(local(iIdx), local(varyingBytes))), offset),
    (offset) => iAdd(local(vertexVaryingAddress), offset),
    local(varyingBytes),
    byteCounterIdx,
  );

  const vertexLoop = [
    ...i32ConstBytes(0),
    ...localSet(iIdx),
    WASM_OP.block,
    0x40,
    WASM_OP.loop,
    0x40,
    ...iGeS(local(iIdx), local(vertexCount)),
    WASM_OP.brIf,
    ...wasmUleb128(1),
    ...copyAttributeIn,
    WASM_OP.call,
    ...wasmUleb128(0), // vertex.main
    ...copyPositionOut,
    ...copyVaryingOut,
    ...iAdd(local(iIdx), i32ConstBytes(1)),
    ...localSet(iIdx),
    WASM_OP.br,
    ...wasmUleb128(0),
    WASM_OP.end,
    WASM_OP.end,
  ];

  const posComponent = (vertexIndex: number[], comp: number) =>
    loadF64(iAdd(iAdd(local(positionsOutBase), iMul(vertexIndex, i32ConstBytes(VEC4_BYTES))), i32ConstBytes(comp * 8)));
  const screenX = (vertexIndex: number[]) =>
    fMul(
      fAdd(fMul(fDiv(posComponent(vertexIndex, 0), posComponent(vertexIndex, 3)), f64ConstBytes(0.5)), f64ConstBytes(0.5)),
      local(widthFIdx),
    );
  const screenY = (vertexIndex: number[]) =>
    fMul(
      fSub(
        f64ConstBytes(1),
        fAdd(fMul(fDiv(posComponent(vertexIndex, 1), posComponent(vertexIndex, 3)), f64ConstBytes(0.5)), f64ConstBytes(0.5)),
      ),
      local(heightFIdx),
    );

  const tExpr = local(tIdx);
  const t1Expr = iAdd(local(tIdx), i32ConstBytes(1));
  const t2Expr = iAdd(local(tIdx), i32ConstBytes(2));

  const computeScreenSpace = [
    ...screenX(tExpr),
    ...localSet(s0xIdx),
    ...screenY(tExpr),
    ...localSet(s0yIdx),
    ...screenX(t1Expr),
    ...localSet(s1xIdx),
    ...screenY(t1Expr),
    ...localSet(s1yIdx),
    ...screenX(t2Expr),
    ...localSet(s2xIdx),
    ...screenY(t2Expr),
    ...localSet(s2yIdx),
    ...posComponent(tExpr, 3),
    ...localSet(w0Idx),
    ...posComponent(t1Expr, 3),
    ...localSet(w1Idx),
    ...posComponent(t2Expr, 3),
    ...localSet(w2Idx),
    ...fDiv(f64ConstBytes(1), local(w0Idx)),
    ...localSet(invW0Idx),
    ...fDiv(f64ConstBytes(1), local(w1Idx)),
    ...localSet(invW1Idx),
    ...fDiv(f64ConstBytes(1), local(w2Idx)),
    ...localSet(invW2Idx),
    ...fSub(
      fMul(fSub(local(s1xIdx), local(s0xIdx)), fSub(local(s2yIdx), local(s0yIdx))),
      fMul(fSub(local(s1yIdx), local(s0yIdx)), fSub(local(s2xIdx), local(s0xIdx))),
    ),
    ...localSet(areaIdx),
  ];

  const min3 = (a: number[], b: number[], c: number[]) => fMin(fMin(a, b), c);
  const max3 = (a: number[], b: number[], c: number[]) => fMax(fMax(a, b), c);
  const computeBBox = [
    ...toI32(fMax(fFloor(min3(local(s0xIdx), local(s1xIdx), local(s2xIdx))), f64ConstBytes(0))),
    ...localSet(minXIdx),
    ...toI32(fMin(fCeil(max3(local(s0xIdx), local(s1xIdx), local(s2xIdx))), fSub(local(widthFIdx), f64ConstBytes(1)))),
    ...localSet(maxXIdx),
    ...toI32(fMax(fFloor(min3(local(s0yIdx), local(s1yIdx), local(s2yIdx))), f64ConstBytes(0))),
    ...localSet(minYIdx),
    ...toI32(fMin(fCeil(max3(local(s0yIdx), local(s1yIdx), local(s2yIdx))), fSub(local(heightFIdx), f64ConstBytes(1)))),
    ...localSet(maxYIdx),
  ];

  const copyFragmentOut = emitByteCopyLoop(
    (offset) =>
      iAdd(
        iAdd(local(outputBase), iMul(iAdd(iMul(local(yIdx), local(width)), local(xIdx)), i32ConstBytes(VEC4_BYTES))),
        offset,
      ),
    (offset) => iAdd(local(fragmentValueAddress), offset),
    i32ConstBytes(VEC4_BYTES),
    byteCounterIdx,
  );

  const edgeFn = (ax: number, ay: number, bx: number, by: number) =>
    fSub(
      fMul(fSub(local(ax), local(pxIdx)), fSub(local(by), local(pyIdx))),
      fMul(fSub(local(ay), local(pyIdx)), fSub(local(bx), local(pxIdx))),
    );

  const varyingComponent = (vertexIndex: number[], comp: number[]) =>
    loadF64(iAdd(iAdd(local(varyingsOutBase), iMul(vertexIndex, local(varyingBytes))), iMul(comp, i32ConstBytes(8))));

  const interpolateVaryings = [
    ...fDiv(local(e0Idx), local(areaIdx)),
    ...localSet(b0Idx),
    ...fDiv(local(e1Idx), local(areaIdx)),
    ...localSet(b1Idx),
    ...fDiv(local(e2Idx), local(areaIdx)),
    ...localSet(b2Idx),
    ...fAdd(
      fAdd(fMul(local(b0Idx), local(invW0Idx)), fMul(local(b1Idx), local(invW1Idx))),
      fMul(local(b2Idx), local(invW2Idx)),
    ),
    ...localSet(invWIdx),
    ...i32ConstBytes(0),
    ...localSet(componentIdx),
    WASM_OP.block,
    0x40,
    WASM_OP.loop,
    0x40,
    ...iGeS(local(componentIdx), local(numVaryingComponentsIdx)),
    WASM_OP.brIf,
    ...wasmUleb128(1),
    ...storeF64(
      iAdd(local(fragmentVaryingAddress), iMul(local(componentIdx), i32ConstBytes(8))),
      fDiv(
        fAdd(
          fAdd(
            fMul(fMul(local(b0Idx), local(invW0Idx)), varyingComponent(tExpr, local(componentIdx))),
            fMul(fMul(local(b1Idx), local(invW1Idx)), varyingComponent(t1Expr, local(componentIdx))),
          ),
          fMul(fMul(local(b2Idx), local(invW2Idx)), varyingComponent(t2Expr, local(componentIdx))),
        ),
        local(invWIdx),
      ),
    ),
    ...iAdd(local(componentIdx), i32ConstBytes(1)),
    ...localSet(componentIdx),
    WASM_OP.br,
    ...wasmUleb128(0),
    WASM_OP.end,
    WASM_OP.end,
  ];

  const pixelBody = [
    ...fAdd(toF64(local(xIdx)), f64ConstBytes(0.5)),
    ...localSet(pxIdx),
    ...fAdd(toF64(local(yIdx)), f64ConstBytes(0.5)),
    ...localSet(pyIdx),
    ...edgeFn(s1xIdx, s1yIdx, s2xIdx, s2yIdx),
    ...localSet(e0Idx),
    ...edgeFn(s2xIdx, s2yIdx, s0xIdx, s0yIdx),
    ...localSet(e1Idx),
    ...edgeFn(s0xIdx, s0yIdx, s1xIdx, s1yIdx),
    ...localSet(e2Idx),
    ...iOr(
      iAnd(
        iAnd(fGe(local(e0Idx), f64ConstBytes(0)), fGe(local(e1Idx), f64ConstBytes(0))),
        fGe(local(e2Idx), f64ConstBytes(0)),
      ),
      iAnd(
        iAnd(fLe(local(e0Idx), f64ConstBytes(0)), fLe(local(e1Idx), f64ConstBytes(0))),
        fLe(local(e2Idx), f64ConstBytes(0)),
      ),
    ),
    WASM_OP.if_,
    0x40,
    ...interpolateVaryings,
    WASM_OP.call,
    ...wasmUleb128(1), // fragment.main
    ...copyFragmentOut,
    WASM_OP.end,
  ];

  const xLoop = [
    ...local(minXIdx),
    ...localSet(xIdx),
    WASM_OP.block,
    0x40,
    WASM_OP.loop,
    0x40,
    ...iGtS(local(xIdx), local(maxXIdx)),
    WASM_OP.brIf,
    ...wasmUleb128(1),
    ...pixelBody,
    ...iAdd(local(xIdx), i32ConstBytes(1)),
    ...localSet(xIdx),
    WASM_OP.br,
    ...wasmUleb128(0),
    WASM_OP.end,
    WASM_OP.end,
  ];

  const yLoop = [
    ...local(minYIdx),
    ...localSet(yIdx),
    WASM_OP.block,
    0x40,
    WASM_OP.loop,
    0x40,
    ...iGtS(local(yIdx), local(maxYIdx)),
    WASM_OP.brIf,
    ...wasmUleb128(1),
    ...xLoop,
    ...iAdd(local(yIdx), i32ConstBytes(1)),
    ...localSet(yIdx),
    WASM_OP.br,
    ...wasmUleb128(0),
    WASM_OP.end,
    WASM_OP.end,
  ];

  const triangleBody = [
    WASM_OP.block,
    0x40,
    ...computeScreenSpace,
    ...fEq(local(areaIdx), f64ConstBytes(0)),
    WASM_OP.brIf,
    ...wasmUleb128(0),
    ...computeBBox,
    ...yLoop,
    WASM_OP.end,
  ];

  const triangleLoop = [
    ...i32ConstBytes(0),
    ...localSet(tIdx),
    WASM_OP.block,
    0x40,
    WASM_OP.loop,
    0x40,
    ...iGeS(t2Expr, local(vertexCount)),
    WASM_OP.brIf,
    ...wasmUleb128(1),
    ...triangleBody,
    ...iAdd(local(tIdx), i32ConstBytes(3)),
    ...localSet(tIdx),
    WASM_OP.br,
    ...wasmUleb128(0),
    WASM_OP.end,
    WASM_OP.end,
  ];

  const code = [
    ...toF64(local(width)),
    ...localSet(widthFIdx),
    ...toF64(local(height)),
    ...localSet(heightFIdx),
    ...iDivS(local(varyingBytes), i32ConstBytes(8)),
    ...localSet(numVaryingComponentsIdx),
    ...vertexLoop,
    ...triangleLoop,
  ];

  const funcBody = [...locals.declBytes(), ...code, WASM_OP.end];
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
