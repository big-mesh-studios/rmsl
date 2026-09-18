import {
  block,
  countingLoop,
  exitBlockIf,
  f64ConstBytes,
  fAdd,
  fCeil,
  fDiv,
  fEq,
  fFloor,
  fGe,
  fGt,
  fLe,
  fMax,
  fMin,
  fMul,
  fSub,
  i32ConstBytes,
  iAdd,
  iAnd,
  iDivS,
  iEq,
  iGeS,
  iGtS,
  iMul,
  iNe,
  iOr,
  ifThen,
  local,
  loadF64,
  loadI32,
  localSet,
  MEMARG_NATURAL,
  storeF64,
  toF64,
  toI32,
  WASM_F64,
  WASM_FUNC,
  WASM_I32,
  WASM_OP,
  wasmSection,
  wasmStrBytes,
  wasmUleb128,
  wasmVec,
} from "./utils";

/**
 * Import name the rasterizer module expects for the vertex stage's exported function.
 */
export const RASTERIZER_VERTEX_IMPORT = { module: "vertex", name: "main" } as const;

/**
 * Import name the rasterizer module expects for the fragment stage's exported function.
 */
export const RASTERIZER_FRAGMENT_IMPORT = { module: "fragment", name: "main" } as const;

/**
 * `rasterize`'s own i32 params, in argument order.
 */
export const RASTERIZE_PARAMS = [
  "vertexCount",
  "attrSrcBase",
  "attrStrideBytes",
  "attrDescBase",
  "attrDescCount",
  "vertexPositionAddress",
  "positionsOutBase",
  "width",
  "height",
  "fragmentValueAddress",
  "outputBase",
  "varyingBytes",
  "varyingDescBase",
  "varyingDescCount",
  "varyingsOutBase",
  "clipScratchBase",
  "clippedPositionsOutBase",
  "clippedVaryingsOutBase",
  "depthBufferBase",
] as const;

/**
 * Homogeneous-clip-space near-plane epsilon: a vertex with `w` at or below
 * this is treated as behind the eye. See rasterizer.md.
 */
const W_CLIP_EPS = 1e-5;

/**
 * A `vec4` position or fragment color, stored as 4 f64 components.
 */
const VEC4_BYTES = 32;

/**
 * Byte size of one `[srcOffset, destAddress, sizeBytes]` attribute descriptor entry.
 */
const ATTR_DESC_BYTES = 12;

/**
 * Byte size of one `[recordOffset, vertexSrcAddress, fragmentDestAddress,
 * sizeBytes]` varying descriptor entry.
 */
const VARYING_DESC_BYTES = 16;

/**
 * Assigns sequential local indices/types past a function's own params.
 */
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
 * Copies `length` bytes one at a time from `src(byteCounter)` to
 * `dest(byteCounter)`, using `counterLocal` as the byte counter. See
 * rasterizer.md.
 */
function emitByteCopyLoop(
  dest: (byteOffset: number[]) => number[],
  src: (byteOffset: number[]) => number[],
  length: number[],
  counterLocal: number,
): number[] {
  return countingLoop(
    counterLocal,
    i32ConstBytes(0),
    iGeS(local(counterLocal), length),
    [
      ...dest(local(counterLocal)),
      ...src(local(counterLocal)),
      WASM_OP.i32Load8U,
      ...MEMARG_NATURAL,
      WASM_OP.i32Store8,
      ...MEMARG_NATURAL,
    ],
    i32ConstBytes(1),
  );
}

/**
 * Builds the rasterizer module's bytes — see rasterizer.md for the full
 * vertex-pass/clip-pass/triangle-pass design and its v1 scope.
 */
export function buildRasterizerModule(): Uint8Array {
  const voidToVoid = [WASM_FUNC, ...wasmVec([]), ...wasmVec([])];
  const rasterizeType = [WASM_FUNC, ...wasmVec(RASTERIZE_PARAMS.map(() => [WASM_I32])), ...wasmVec([])];
  const typeSection = wasmSection(1, wasmVec([voidToVoid, rasterizeType]));

  const IMPORT_KIND_FUNC = 0x00;
  const IMPORT_KIND_MEMORY = 0x02;
  const LIMITS_MIN_ONLY = 0x00; // no declared maximum
  const VOID_TO_VOID_TYPE_INDEX = 0;

  const vertexImport = [
    ...wasmStrBytes(RASTERIZER_VERTEX_IMPORT.module),
    ...wasmStrBytes(RASTERIZER_VERTEX_IMPORT.name),
    IMPORT_KIND_FUNC,
    ...wasmUleb128(VOID_TO_VOID_TYPE_INDEX),
  ];
  const fragmentImport = [
    ...wasmStrBytes(RASTERIZER_FRAGMENT_IMPORT.module),
    ...wasmStrBytes(RASTERIZER_FRAGMENT_IMPORT.name),
    IMPORT_KIND_FUNC,
    ...wasmUleb128(VOID_TO_VOID_TYPE_INDEX),
  ];
  const memoryImport = [
    ...wasmStrBytes("env"),
    ...wasmStrBytes("memory"),
    IMPORT_KIND_MEMORY,
    LIMITS_MIN_ONLY,
    ...wasmUleb128(1), // 1 page (64KiB) to start; the host grows it as needed
  ];
  const importSection = wasmSection(2, wasmVec([vertexImport, fragmentImport, memoryImport]));

  const rasterizeFuncIndex = 2; // function index space: 0 = vertex import, 1 = fragment import, 2 = this module's own "rasterize"
  const funcSection = wasmSection(3, wasmVec([[...wasmUleb128(1)]])); // "rasterize" has type index 1 (rasterizeType)

  const EXPORT_KIND_FUNC = 0x00;
  const exportSection = wasmSection(
    7,
    wasmVec([[...wasmStrBytes("rasterize"), EXPORT_KIND_FUNC, ...wasmUleb128(rasterizeFuncIndex)]]),
  );

  const [
    vertexCount,
    attrSrcBase,
    attrStrideBytes,
    attrDescBase,
    attrDescCount,
    vertexPositionAddress,
    positionsOutBase,
    width,
    height,
    fragmentValueAddress,
    outputBase,
    varyingBytes,
    varyingDescBase,
    varyingDescCount,
    varyingsOutBase,
    clipScratchBase,
    clippedPositionsOutBase,
    clippedVaryingsOutBase,
    depthBufferBase,
  ] = RASTERIZE_PARAMS.map((_, i) => i);

  const locals = new LocalAllocator(RASTERIZE_PARAMS.length);
  const iIdx = locals.alloc(WASM_I32); // vertex loop counter
  const byteCounterIdx = locals.alloc(WASM_I32); // shared by every byte-copy loop
  const tIdx = locals.alloc(WASM_I32); // triangle loop counter (clip pass, then reused for the raster pass)
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
  const componentIdx = locals.alloc(WASM_I32);
  const dIdx = locals.alloc(WASM_I32); // descriptor loop counter, reused across every descriptor loop
  const descField0Idx = locals.alloc(WASM_I32);
  const descField1Idx = locals.alloc(WASM_I32);
  const descField2Idx = locals.alloc(WASM_I32);
  const descField3Idx = locals.alloc(WASM_I32);
  const numComponentsIdx = locals.alloc(WASM_I32);
  const wholeRecordComponentsIdx = locals.alloc(WASM_I32);
  const clippedVertexCountIdx = locals.alloc(WASM_I32);
  const outCountIdx = locals.alloc(WASM_I32); // clip-output scratch fill count, 0-4, reset per original triangle
  const clipTIdx = locals.alloc(WASM_F64); // clip-edge interpolation parameter
  const depth0Idx = locals.alloc(WASM_F64);
  const depth1Idx = locals.alloc(WASM_F64);
  const depth2Idx = locals.alloc(WASM_F64);
  const pixelDepthIdx = locals.alloc(WASM_F64);
  const pixelIndexIdx = locals.alloc(WASM_I32);

  const descAddr = (descBase: number, index: number, descBytes: number) =>
    iAdd(local(descBase), iMul(local(index), i32ConstBytes(descBytes)));

  const copyAttributeIn = countingLoop(
    dIdx,
    i32ConstBytes(0),
    iGeS(local(dIdx), local(attrDescCount)),
    [
      ...loadI32(descAddr(attrDescBase, dIdx, ATTR_DESC_BYTES)),
      ...localSet(descField0Idx), // srcOffset
      ...loadI32(iAdd(descAddr(attrDescBase, dIdx, ATTR_DESC_BYTES), i32ConstBytes(4))),
      ...localSet(descField1Idx), // destAddress
      ...loadI32(iAdd(descAddr(attrDescBase, dIdx, ATTR_DESC_BYTES), i32ConstBytes(8))),
      ...localSet(descField2Idx), // sizeBytes
      ...emitByteCopyLoop(
        (offset) => iAdd(local(descField1Idx), offset),
        (offset) =>
          iAdd(iAdd(iAdd(local(attrSrcBase), iMul(local(iIdx), local(attrStrideBytes))), local(descField0Idx)), offset),
        local(descField2Idx),
        byteCounterIdx,
      ),
    ],
    i32ConstBytes(1),
  );
  const copyPositionOut = emitByteCopyLoop(
    (offset) => iAdd(iAdd(local(positionsOutBase), iMul(local(iIdx), i32ConstBytes(VEC4_BYTES))), offset),
    (offset) => iAdd(local(vertexPositionAddress), offset),
    i32ConstBytes(VEC4_BYTES),
    byteCounterIdx,
  );
  const copyVaryingOut = countingLoop(
    dIdx,
    i32ConstBytes(0),
    iGeS(local(dIdx), local(varyingDescCount)),
    [
      ...loadI32(descAddr(varyingDescBase, dIdx, VARYING_DESC_BYTES)),
      ...localSet(descField0Idx), // recordOffset
      ...loadI32(iAdd(descAddr(varyingDescBase, dIdx, VARYING_DESC_BYTES), i32ConstBytes(4))),
      ...localSet(descField1Idx), // vertexSrcAddress
      ...loadI32(iAdd(descAddr(varyingDescBase, dIdx, VARYING_DESC_BYTES), i32ConstBytes(12))),
      ...localSet(descField3Idx), // sizeBytes
      ...emitByteCopyLoop(
        (offset) =>
          iAdd(
            iAdd(iAdd(local(varyingsOutBase), iMul(local(iIdx), local(varyingBytes))), local(descField0Idx)),
            offset,
          ),
        (offset) => iAdd(local(descField1Idx), offset),
        local(descField3Idx),
        byteCounterIdx,
      ),
    ],
    i32ConstBytes(1),
  );

  const vertexLoop = countingLoop(
    iIdx,
    i32ConstBytes(0),
    iGeS(local(iIdx), local(vertexCount)),
    [...copyAttributeIn, WASM_OP.call, ...wasmUleb128(0) /* vertex.main */, ...copyPositionOut, ...copyVaryingOut],
    i32ConstBytes(1),
  );

  const tExpr = local(tIdx);
  const t1Expr = iAdd(local(tIdx), i32ConstBytes(1));
  const t2Expr = iAdd(local(tIdx), i32ConstBytes(2));

  const posComponentAt = (base: number, vertexIndex: number[], comp: number) =>
    loadF64(iAdd(iAdd(local(base), iMul(vertexIndex, i32ConstBytes(VEC4_BYTES))), i32ConstBytes(comp * 8)));
  const varyingComponentAt = (base: number, vertexIndex: number[], recordOffset: number[], comp: number[]) =>
    loadF64(
      iAdd(iAdd(iAdd(local(base), iMul(vertexIndex, local(varyingBytes))), recordOffset), iMul(comp, i32ConstBytes(8))),
    );
  // clip pass reads the vertex pass's raw (pre-clip) output
  const rawPos = (vertexIndex: number[], comp: number) => posComponentAt(positionsOutBase, vertexIndex, comp);
  const rawVarying = (vertexIndex: number[], comp: number[]) =>
    varyingComponentAt(varyingsOutBase, vertexIndex, i32ConstBytes(0), comp);
  // the raster pass reads the clip pass's output instead of the raw vertices
  const posComponent = (vertexIndex: number[], comp: number) =>
    posComponentAt(clippedPositionsOutBase, vertexIndex, comp);
  const varyingComponent = (vertexIndex: number[], recordOffset: number[], comp: number[]) =>
    varyingComponentAt(clippedVaryingsOutBase, vertexIndex, recordOffset, comp);

  const scratchVertexAddr = (slot: number[]) =>
    iAdd(local(clipScratchBase), iMul(slot, iAdd(i32ConstBytes(VEC4_BYTES), local(varyingBytes))));
  const scratchPosAddr = scratchVertexAddr;
  const scratchVaryingAddr = (slot: number[]) => iAdd(scratchVertexAddr(slot), i32ConstBytes(VEC4_BYTES));

  const emitVertexToScratch = (vertexIndex: number[]) => [
    ...emitByteCopyLoop(
      (offset) => iAdd(scratchPosAddr(local(outCountIdx)), offset),
      (offset) => iAdd(iAdd(local(positionsOutBase), iMul(vertexIndex, i32ConstBytes(VEC4_BYTES))), offset),
      i32ConstBytes(VEC4_BYTES),
      byteCounterIdx,
    ),
    ...emitByteCopyLoop(
      (offset) => iAdd(scratchVaryingAddr(local(outCountIdx)), offset),
      (offset) => iAdd(iAdd(local(varyingsOutBase), iMul(vertexIndex, local(varyingBytes))), offset),
      local(varyingBytes),
      byteCounterIdx,
    ),
    ...iAdd(local(outCountIdx), i32ConstBytes(1)),
    ...localSet(outCountIdx),
  ];

  const lerp = (a: number[], b: number[], t: number[]) => fAdd(a, fMul(fSub(b, a), t));

  const emitInterpVertexToScratch = (a: number[], b: number[]) => [
    ...[0, 1, 2, 3].flatMap((c) =>
      storeF64(
        iAdd(scratchPosAddr(local(outCountIdx)), i32ConstBytes(c * 8)),
        lerp(rawPos(a, c), rawPos(b, c), local(clipTIdx)),
      ),
    ),
    ...countingLoop(
      componentIdx,
      i32ConstBytes(0),
      iGeS(local(componentIdx), local(wholeRecordComponentsIdx)),
      storeF64(
        iAdd(scratchVaryingAddr(local(outCountIdx)), iMul(local(componentIdx), i32ConstBytes(8))),
        lerp(rawVarying(a, local(componentIdx)), rawVarying(b, local(componentIdx)), local(clipTIdx)),
      ),
      i32ConstBytes(1),
    ),
    ...iAdd(local(outCountIdx), i32ConstBytes(1)),
    ...localSet(outCountIdx),
  ];

  /**
   * One edge (`a` -> `b`) of a triangle's single-plane Sutherland-Hodgman
   * clip against `w > W_CLIP_EPS`: keep `a` if it's on the inside, and
   * whenever the edge crosses the plane (`a`/`b` disagree), emit the cut
   * point. See rasterizer.md.
   */
  const emitClipEdge = (a: number[], b: number[]) => [
    ...ifThen(fGt(rawPos(a, 3), f64ConstBytes(W_CLIP_EPS)), emitVertexToScratch(a)),
    ...ifThen(iNe(fGt(rawPos(a, 3), f64ConstBytes(W_CLIP_EPS)), fGt(rawPos(b, 3), f64ConstBytes(W_CLIP_EPS))), [
      ...fDiv(fSub(f64ConstBytes(W_CLIP_EPS), rawPos(a, 3)), fSub(rawPos(b, 3), rawPos(a, 3))),
      ...localSet(clipTIdx),
      ...emitInterpVertexToScratch(a, b),
    ]),
  ];

  const emitTriangleFromScratch = (slots: readonly [number, number, number]) => [
    ...slots.flatMap((slot, i) => [
      ...emitByteCopyLoop(
        (offset) =>
          iAdd(
            iAdd(
              local(clippedPositionsOutBase),
              iMul(iAdd(local(clippedVertexCountIdx), i32ConstBytes(i)), i32ConstBytes(VEC4_BYTES)),
            ),
            offset,
          ),
        (offset) => iAdd(scratchPosAddr(i32ConstBytes(slot)), offset),
        i32ConstBytes(VEC4_BYTES),
        byteCounterIdx,
      ),
      ...emitByteCopyLoop(
        (offset) =>
          iAdd(
            iAdd(
              local(clippedVaryingsOutBase),
              iMul(iAdd(local(clippedVertexCountIdx), i32ConstBytes(i)), local(varyingBytes)),
            ),
            offset,
          ),
        (offset) => iAdd(scratchVaryingAddr(i32ConstBytes(slot)), offset),
        local(varyingBytes),
        byteCounterIdx,
      ),
    ]),
    ...iAdd(local(clippedVertexCountIdx), i32ConstBytes(3)),
    ...localSet(clippedVertexCountIdx),
  ];

  // A triangle clipped against one plane always comes out with 0, 3, or 4
  // vertices — never 1 or 2 — so only these two fan triangles are possible.
  const clipOneTriangle = [
    ...i32ConstBytes(0),
    ...localSet(outCountIdx),
    ...emitClipEdge(tExpr, t1Expr),
    ...emitClipEdge(t1Expr, t2Expr),
    ...emitClipEdge(t2Expr, tExpr),
    ...ifThen(iGeS(local(outCountIdx), i32ConstBytes(3)), emitTriangleFromScratch([0, 1, 2])),
    ...ifThen(iEq(local(outCountIdx), i32ConstBytes(4)), emitTriangleFromScratch([0, 2, 3])),
  ];

  const clipLoop = [
    ...i32ConstBytes(0),
    ...localSet(clippedVertexCountIdx),
    ...countingLoop(tIdx, i32ConstBytes(0), iGeS(t2Expr, local(vertexCount)), clipOneTriangle, i32ConstBytes(3)),
  ];

  const screenX = (vertexIndex: number[]) =>
    fMul(
      fAdd(
        fMul(fDiv(posComponent(vertexIndex, 0), posComponent(vertexIndex, 3)), f64ConstBytes(0.5)),
        f64ConstBytes(0.5),
      ),
      local(widthFIdx),
    );
  const screenY = (vertexIndex: number[]) =>
    fMul(
      fSub(
        f64ConstBytes(1),
        fAdd(
          fMul(fDiv(posComponent(vertexIndex, 1), posComponent(vertexIndex, 3)), f64ConstBytes(0.5)),
          f64ConstBytes(0.5),
        ),
      ),
      local(heightFIdx),
    );

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
    // NDC depth (z/w) is affine in screen space, like x/w and y/w above, so
    // it interpolates with plain barycentric weights — no invW needed.
    ...fDiv(posComponent(tExpr, 2), local(w0Idx)),
    ...localSet(depth0Idx),
    ...fDiv(posComponent(t1Expr, 2), local(w1Idx)),
    ...localSet(depth1Idx),
    ...fDiv(posComponent(t2Expr, 2), local(w2Idx)),
    ...localSet(depth2Idx),
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
    (offset) => iAdd(iAdd(local(outputBase), iMul(local(pixelIndexIdx), i32ConstBytes(VEC4_BYTES))), offset),
    (offset) => iAdd(local(fragmentValueAddress), offset),
    i32ConstBytes(VEC4_BYTES),
    byteCounterIdx,
  );

  const edgeFn = (ax: number, ay: number, bx: number, by: number) =>
    fSub(
      fMul(fSub(local(ax), local(pxIdx)), fSub(local(by), local(pyIdx))),
      fMul(fSub(local(ay), local(pyIdx)), fSub(local(bx), local(pxIdx))),
    );

  const interpolateOneDescriptor = [
    ...loadI32(descAddr(varyingDescBase, dIdx, VARYING_DESC_BYTES)),
    ...localSet(descField0Idx), // recordOffset
    ...loadI32(iAdd(descAddr(varyingDescBase, dIdx, VARYING_DESC_BYTES), i32ConstBytes(8))),
    ...localSet(descField2Idx), // fragmentDestAddress
    ...loadI32(iAdd(descAddr(varyingDescBase, dIdx, VARYING_DESC_BYTES), i32ConstBytes(12))),
    ...localSet(descField3Idx), // sizeBytes
    ...iDivS(local(descField3Idx), i32ConstBytes(8)),
    ...localSet(numComponentsIdx),
    ...countingLoop(
      componentIdx,
      i32ConstBytes(0),
      iGeS(local(componentIdx), local(numComponentsIdx)),
      storeF64(
        iAdd(local(descField2Idx), iMul(local(componentIdx), i32ConstBytes(8))),
        fDiv(
          fAdd(
            fAdd(
              fMul(
                fMul(local(b0Idx), local(invW0Idx)),
                varyingComponent(tExpr, local(descField0Idx), local(componentIdx)),
              ),
              fMul(
                fMul(local(b1Idx), local(invW1Idx)),
                varyingComponent(t1Expr, local(descField0Idx), local(componentIdx)),
              ),
            ),
            fMul(
              fMul(local(b2Idx), local(invW2Idx)),
              varyingComponent(t2Expr, local(descField0Idx), local(componentIdx)),
            ),
          ),
          local(invWIdx),
        ),
      ),
      i32ConstBytes(1),
    ),
  ];

  const computeBarycentricWeights = [
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
  ];

  const interpolateVaryings = countingLoop(
    dIdx,
    i32ConstBytes(0),
    iGeS(local(dIdx), local(varyingDescCount)),
    interpolateOneDescriptor,
    i32ConstBytes(1),
  );

  const depthBufferAddr = iAdd(local(depthBufferBase), iMul(local(pixelIndexIdx), i32ConstBytes(8)));

  const pixelBody = [
    ...fAdd(toF64(local(xIdx)), f64ConstBytes(0.5)),
    ...localSet(pxIdx),
    ...fAdd(toF64(local(yIdx)), f64ConstBytes(0.5)),
    ...localSet(pyIdx),
    ...iAdd(iMul(local(yIdx), local(width)), local(xIdx)),
    ...localSet(pixelIndexIdx),
    ...edgeFn(s1xIdx, s1yIdx, s2xIdx, s2yIdx),
    ...localSet(e0Idx),
    ...edgeFn(s2xIdx, s2yIdx, s0xIdx, s0yIdx),
    ...localSet(e1Idx),
    ...edgeFn(s0xIdx, s0yIdx, s1xIdx, s1yIdx),
    ...localSet(e2Idx),
    ...ifThen(
      // covered iff all three edge functions agree on sign (all >= 0, or all <= 0)
      iOr(
        iAnd(
          iAnd(fGe(local(e0Idx), f64ConstBytes(0)), fGe(local(e1Idx), f64ConstBytes(0))),
          fGe(local(e2Idx), f64ConstBytes(0)),
        ),
        iAnd(
          iAnd(fLe(local(e0Idx), f64ConstBytes(0)), fLe(local(e1Idx), f64ConstBytes(0))),
          fLe(local(e2Idx), f64ConstBytes(0)),
        ),
      ),
      [
        ...computeBarycentricWeights,
        ...fAdd(
          fAdd(fMul(local(b0Idx), local(depth0Idx)), fMul(local(b1Idx), local(depth1Idx))),
          fMul(local(b2Idx), local(depth2Idx)),
        ),
        ...localSet(pixelDepthIdx),
        // depth test: closer-or-equal wins (matches typical LEQUAL hardware
        // default) — the host must pre-clear depthBufferBase to a large
        // value so the first triangle over any pixel always passes.
        ...ifThen(fLe(local(pixelDepthIdx), loadF64(depthBufferAddr)), [
          ...storeF64(depthBufferAddr, local(pixelDepthIdx)),
          ...interpolateVaryings,
          WASM_OP.call,
          ...wasmUleb128(1), // fragment.main
          ...copyFragmentOut,
        ]),
      ],
    ),
  ];

  const xLoop = countingLoop(xIdx, local(minXIdx), iGtS(local(xIdx), local(maxXIdx)), pixelBody, i32ConstBytes(1));
  const yLoop = countingLoop(yIdx, local(minYIdx), iGtS(local(yIdx), local(maxYIdx)), xLoop, i32ConstBytes(1));

  const triangleBody = block([
    ...computeScreenSpace,
    ...exitBlockIf(fEq(local(areaIdx), f64ConstBytes(0))), // degenerate (zero-area) triangle: skip it
    ...computeBBox,
    ...yLoop,
  ]);

  const triangleLoop = countingLoop(
    tIdx,
    i32ConstBytes(0),
    iGeS(t2Expr, local(clippedVertexCountIdx)),
    triangleBody,
    i32ConstBytes(3),
  );

  const code = [
    ...toF64(local(width)),
    ...localSet(widthFIdx),
    ...toF64(local(height)),
    ...localSet(heightFIdx),
    ...iDivS(local(varyingBytes), i32ConstBytes(8)),
    ...localSet(wholeRecordComponentsIdx),
    ...vertexLoop,
    ...clipLoop,
    ...triangleLoop,
  ];

  const funcBody = [...locals.declBytes(), ...code, WASM_OP.end];
  const codeSection = wasmSection(10, wasmVec([[...wasmUleb128(funcBody.length), ...funcBody]]));

  // prettier-ignore
  return new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, // "\0asm"
    0x01, 0x00, 0x00, 0x00, // version 1
    ...typeSection,
    ...importSection,
    ...funcSection,
    ...exportSection,
    ...codeSection,
  ]);
}

/**
 * One attribute slot's copy: `sizeBytes` bytes starting at `srcOffset`
 * within each vertex's record in the source buffer, copied to
 * `destAddress` (the vertex module's own memory address for that slot).
 */
export interface AttributeDescriptor {
  srcOffset: number;
  destAddress: number;
  sizeBytes: number;
}

/**
 * Packs `descriptors` into `view` at `base`, in the layout
 * {@link buildRasterizerModule}'s attribute-copy loop reads.
 */
export function writeAttributeDescriptors(
  view: DataView,
  base: number,
  descriptors: readonly AttributeDescriptor[],
): void {
  descriptors.forEach((d, i) => {
    view.setInt32(base + i * ATTR_DESC_BYTES, d.srcOffset, true);
    view.setInt32(base + i * ATTR_DESC_BYTES + 4, d.destAddress, true);
    view.setInt32(base + i * ATTR_DESC_BYTES + 8, d.sizeBytes, true);
  });
}

/**
 * One varying slot's linkage: `sizeBytes` bytes read from
 * `vertexSrcAddress` (that slot's address in the vertex module) after
 * each vertex call, stored at `recordOffset` within the per-vertex
 * varying record, and later interpolated into `fragmentDestAddress`
 * (that slot's address in the fragment module) per covered pixel.
 */
export interface VaryingDescriptor {
  recordOffset: number;
  vertexSrcAddress: number;
  fragmentDestAddress: number;
  sizeBytes: number;
}

/**
 * Packs `descriptors` into `view` at `base`, in the layout
 * {@link buildRasterizerModule}'s varying copy/interpolation loops read.
 */
export function writeVaryingDescriptors(view: DataView, base: number, descriptors: readonly VaryingDescriptor[]): void {
  descriptors.forEach((d, i) => {
    view.setInt32(base + i * VARYING_DESC_BYTES, d.recordOffset, true);
    view.setInt32(base + i * VARYING_DESC_BYTES + 4, d.vertexSrcAddress, true);
    view.setInt32(base + i * VARYING_DESC_BYTES + 8, d.fragmentDestAddress, true);
    view.setInt32(base + i * VARYING_DESC_BYTES + 12, d.sizeBytes, true);
  });
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
