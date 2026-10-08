import { Node, ShaderType } from "../../core";
import { componentCountOf, CpuDrawBuffer, CpuShaderContext, CpuTextureData } from "../cpu";
import { DrawClearOptions, DrawCountOptions, TRANSPARENT_BLACK, TypedArray } from "../adapter";
import { compileWasmFn, createWasmInputMarshaller, WasmCompileFields, WasmFloatWidth, WasmParam } from "./wasm";
import { wasmUleb128 } from "./utils";
import RASTERIZER_WASM_BYTES from "./rasterizer.wat";

/**
 * Import name the rasterizer module expects for the vertex stage's exported function.
 */
export const RASTERIZER_VERTEX_IMPORT = { module: "vertex", name: "main" } as const;

/**
 * Import name the rasterizer module expects for the fragment stage's exported function.
 */
export const RASTERIZER_FRAGMENT_IMPORT = { module: "fragment", name: "main" } as const;

/**
 * `rasterize`'s own i32 params, in argument order — see rasterizer.wat's
 * `$rasterize` function for the same list.
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
  "writesColour",
  "fragCoordAddress",
  "writesFragCoord",
  "fragDepthAddress",
  "writesDepth",
  "discardAddress",
  "mayDiscard",
] as const;

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
 * The rasterizer module's bytes — see rasterizer.md for the full
 * vertex-pass/clip-pass/triangle-pass design and its v1 scope, and
 * rasterizer.wat for the module itself.
 */
export function buildRasterizerModule(shared?: { maximum: number }): Uint8Array {
  if (!shared) return RASTERIZER_WASM_BYTES;
  let variant = sharedVariants.get(shared.maximum);
  if (!variant)
    sharedVariants.set(shared.maximum, (variant = declareSharedMemory(RASTERIZER_WASM_BYTES, shared.maximum)));
  return variant;
}

/** The module with a shared memory import, by the maximum it declares. */
const sharedVariants = new Map<number, Uint8Array>();

/** Reads an unsigned LEB128 integer at `at`, and the offset after it. */
function readUleb(bytes: Uint8Array, at: number): [value: number, next: number] {
  let value = 0;
  let shift = 0;
  for (;;) {
    const byte = bytes[at++]!;
    value |= (byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) return [value >>> 0, at];
    shift += 7;
  }
}

/**
 * The module with its memory import declared shared and bounded by `maximum`
 * pages. A shared memory only links against an import declared shared, and the
 * rasterizer's own import is not, so the limits of that import are rewritten
 * and the import section's size follows them.
 */
function declareSharedMemory(bytes: Uint8Array, maximum: number): Uint8Array {
  let at = 8; // after the magic number and the version
  while (at < bytes.length) {
    const sectionStart = at;
    const id = bytes[at++]!;
    const [size, bodyStart] = readUleb(bytes, at);
    at = bodyStart;
    if (id !== 2) {
      at += size;
      continue;
    }
    let [count, cursor] = readUleb(bytes, at);
    for (; count > 0; count--) {
      for (let name = 0; name < 2; name++) {
        const [length, next] = readUleb(bytes, cursor);
        cursor = next + length;
      }
      const kind = bytes[cursor++]!;
      if (kind === 0) {
        cursor = readUleb(bytes, cursor)[1]; // the index of the function's type
        continue;
      }
      if (kind !== 2)
        throw new Error("[RMSL] the rasterizer module imports something other than functions and a memory");
      const limitsStart = cursor;
      const flags = bytes[cursor++]!;
      const [minimum, afterMinimum] = readUleb(bytes, cursor);
      cursor = afterMinimum;
      if (flags & 1) cursor = readUleb(bytes, cursor)[1];
      const limits = [0x03, ...wasmUleb128(minimum), ...wasmUleb128(maximum)];
      const body = [...bytes.subarray(bodyStart, limitsStart), ...limits, ...bytes.subarray(cursor, bodyStart + size)];
      return Uint8Array.from([
        ...bytes.subarray(0, sectionStart),
        id,
        ...wasmUleb128(body.length),
        ...body,
        ...bytes.subarray(bodyStart + size),
      ]);
    }
    return bytes;
  }
  return bytes;
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
  /** The maximum of a shared memory, which the module declares as its import. */
  shared?: { maximum: number },
): { rasterize: (...args: number[]) => void } {
  const instance = new WebAssembly.Instance(
    new WebAssembly.Module(buildRasterizerModule(shared).buffer as ArrayBuffer),
    {
      vertex: { main: vertexMain },
      fragment: { main: fragmentMain },
      env: { memory },
    },
  );
  return { rasterize: instance.exports.rasterize as (...args: number[]) => void };
}

/**
 * Options shared by both stages of a {@link compileWasm} pair — the same
 * fields `compileWasmFn` itself takes, minus `name`/`stage`/`params`/
 * `scalarsInMemory`/`memoryBase`, which `compileWasm` fixes itself (the
 * rasterizer's fixed-arity imports need a zero-arg `"main"` export from
 * each stage, and the two share one memory at non-overlapping bases).
 */
export type CompileWasmOptions = Pick<
  WasmCompileFields,
  "derivatives" | "reentrant" | "memory" | "sharedMemory" | "maxMemoryPages"
> &
  WasmFloatWidth;

/**
 * One draw call's inputs: per-vertex attribute buffers (one flat, planar
 * `TypedArray` per slot — `compileWasm` interleaves them internally, since
 * the rasterizer module's own attribute-copy loop expects one packed,
 * per-vertex-stride source region) plus the uniforms/textures shared
 * across the whole call.
 */
export interface WasmRasterContext {
  attributes: Record<string, TypedArray>;
  uniforms?: Record<string, number | number[]>;
  textures?: Record<string, CpuTextureData>;
}

/**
 * Both the output buffer's and the depth buffer's own addresses are
 * reused deterministically call to call, so a draw that passes
 * `clear: false` or `clearDepth: false` composes onto them exactly like
 * several draws into one real framebuffer would (occlusion included) —
 * matching how a WebGPU render pass declares `loadOp`/`depthLoadOp`
 * together, per pass, rather than clearing as some separate operation.
 * Both default to `true`, so each draw starts from a cleared buffer, as a
 * three.js render does.
 */
export interface WasmRasterDrawOptions extends DrawCountOptions, DrawClearOptions {
  width: number;
  height: number;
  out?: CpuDrawBuffer;
  clearDepth?: boolean;
}

/**
 * The callable a {@link compileWasm} pair produces — closes over the
 * vertex/fragment/rasterizer instances entirely; a caller never sees them.
 */
export interface WasmRasterRoutine {
  /**
   * Runs the vertex pass over `options.count` vertices (a non-indexed
   * triangle list, so a multiple of 3 — defaults to everything the first
   * attribute slot's data implies, like every other draw-capable
   * adapter's own `count`), then the clip and triangle passes into a
   * `width` x `height`, 4-components-per-pixel buffer, the same flat
   * row-major convention `CpuRoutine.draw()` uses. See
   * {@link WasmRasterDrawOptions} for `clear`/`clearDepth`.
   */
  draw(ctx: WasmRasterContext, options: WasmRasterDrawOptions): CpuDrawBuffer;
}

function align8(n: number): number {
  return Math.ceil(n / 8) * 8;
}

/**
 * Compiles a vertex/fragment `Fn` pair and links them against the shared
 * rasterizer module (see rasterizer.md) into one {@link WasmRasterRoutine}.
 *
 * Both stages compile with `scalarsInMemory: true` (the rasterizer's
 * imports must be zero-arg `"main"` exports — see rasterizer.md's v1
 * scope) and share one `WebAssembly.Memory`, the fragment stage's own
 * compile-time layout placed right after the vertex stage's via
 * `memoryBase` (see `CompileWasmFnOptions.memoryBase`) so neither's fixed
 * addresses collide.
 */
export function compileWasm(
  vertexFn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  fragmentFn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  options: CompileWasmOptions = {},
): WasmRasterRoutine {
  const vertexCompiled = compileWasmFn(vertexFn, {
    ...options,
    name: "main",
    params: [],
    stage: "vertex",
    scalarsInMemory: true,
  });
  const fragmentCompiled = compileWasmFn(fragmentFn, {
    ...options,
    name: "main",
    params: [],
    stage: "fragment",
    scalarsInMemory: true,
    memoryBase: align8(vertexCompiled.textureHeapBase),
  });

  const attrParams = vertexCompiled.params.filter(
    (p): p is Extract<WasmParam, { kind: "attributeMemory" }> => p.kind === "attributeMemory",
  );
  const vertexVaryingParams = vertexCompiled.params.filter(
    (p): p is Extract<WasmParam, { kind: "varyingOutputMemory" }> => p.kind === "varyingOutputMemory",
  );
  const fragmentVaryingParams = fragmentCompiled.params.filter(
    (p): p is Extract<WasmParam, { kind: "varyingMemory" }> => p.kind === "varyingMemory",
  );
  const positionParam = vertexCompiled.params.find(
    (p): p is Extract<WasmParam, { kind: "positionMemory" }> => p.kind === "positionMemory",
  );
  if (!positionParam) {
    throw new Error("[RMSL] compileWasm: vertexFn must produce a position (builtinPosition() or a bare vec4 return)");
  }
  const fragmentValueParam = fragmentCompiled.params.find(
    (p): p is Extract<WasmParam, { kind: "valueMemory" }> => p.kind === "valueMemory",
  );
  if (fragmentValueParam !== undefined && fragmentValueParam.shaderType !== "vec4") {
    throw new Error("[RMSL] compileWasm: fragmentFn must return a vec4 color, or nothing");
  }
  if (fragmentValueParam === undefined && fragmentCompiled.params.some((p) => p.kind === "outputMemory")) {
    throw new Error("[RMSL] compileWasm: the rasterizer draws a vec4 colour, so fragmentFn cannot declare outputs");
  }
  const positionAddress = positionParam.address;
  /** The address of what the fragment stage writes or reads, and 1 when it does, 0 when it does not. */
  const fragmentInput = (kind: "valueMemory" | "fragCoordMemory" | "fragDepthMemory" | "discardMemory") => {
    const param = fragmentCompiled.params.find((p) => p.kind === kind) as { address: number } | undefined;
    return [param?.address ?? 0, param ? 1 : 0] as const;
  };
  const [fragmentValueAddress, writesColour] = fragmentInput("valueMemory");
  const [fragCoordAddress, writesFragCoord] = fragmentInput("fragCoordMemory");
  const [fragDepthAddress, writesDepth] = fragmentInput("fragDepthMemory");
  const [discardAddress, mayDiscard] = fragmentInput("discardMemory");

  const missingInFragment = vertexVaryingParams.filter((v) => !fragmentVaryingParams.some((f) => f.slot === v.slot));
  if (missingInFragment.length > 0) {
    throw new Error(
      `[RMSL] compileWasm: varying(s) ${missingInFragment.map((v) => v.slot).join(", ")} written by vertexFn but never read by fragmentFn`,
    );
  }

  let varyingCursor = 0;
  const varyingLayout = vertexVaryingParams.map((v) => {
    const sizeBytes = componentCountOf(v.shaderType) * 8;
    const offset = varyingCursor;
    varyingCursor += sizeBytes;
    return {
      slot: v.slot,
      offset,
      sizeBytes,
      vertexSrcAddress: v.address,
      fragmentDestAddress: fragmentVaryingParams.find((f) => f.slot === v.slot)!.address,
    };
  });
  const varyingBytes = varyingCursor;

  let attrCursor = 0;
  const attrLayout = attrParams.map((p) => {
    const sizeBytes = componentCountOf(p.shaderType) * 8;
    const offset = attrCursor;
    attrCursor += sizeBytes;
    return { slot: p.slot, offset, sizeBytes, destAddress: p.address };
  });
  const attrStrideBytes = attrCursor;

  const initialPages = Math.max(vertexCompiled.memoryPages, fragmentCompiled.memoryPages, 1);
  const memory =
    options.memory ??
    new WebAssembly.Memory(
      vertexCompiled.sharedMemory
        ? { initial: initialPages, maximum: vertexCompiled.maxMemoryPages, shared: true }
        : { initial: initialPages },
    );
  const memoryIsShared = typeof SharedArrayBuffer !== "undefined" && memory.buffer instanceof SharedArrayBuffer;
  if (memoryIsShared !== vertexCompiled.sharedMemory) {
    throw new Error(
      `[RMSL] compileWasm: the memory is ${memoryIsShared ? "shared" : "not shared"}, but the stages were compiled with sharedMemory ${vertexCompiled.sharedMemory}`,
    );
  }

  const vertexInstance = new WebAssembly.Instance(new WebAssembly.Module(vertexCompiled.bytes.buffer as ArrayBuffer), {
    math: Math as unknown as WebAssembly.ModuleImports,
    env: { memory },
  });
  const fragmentInstance = new WebAssembly.Instance(
    new WebAssembly.Module(fragmentCompiled.bytes.buffer as ArrayBuffer),
    { math: Math as unknown as WebAssembly.ModuleImports, env: { memory } },
  );

  const { rasterize } = instantiateRasterizer(
    vertexInstance.exports.main as () => void,
    fragmentInstance.exports.main as () => void,
    memory,
    vertexCompiled.sharedMemory ? { maximum: vertexCompiled.maxMemoryPages } : undefined,
  );

  // Each stage's textures sit after both stages' fixed layouts, the vertex
  // stage's first, so neither heap lies over a layout or over the other heap.
  const heapStart = align8(fragmentCompiled.textureHeapBase);
  // Attributes and interpolated varyings are written by the rasterizer
  // itself, per vertex and per covered pixel, so the marshallers skip them.
  const vertexMarshaller = createWasmInputMarshaller(
    vertexCompiled.params.filter((p) => p.kind !== "attributeMemory"),
    heapStart,
    memory,
    vertexCompiled.float32,
  );
  const fragmentMarshaller = createWasmInputMarshaller(
    fragmentCompiled.params.filter((p) => p.kind !== "varyingMemory"),
    heapStart,
    memory,
    fragmentCompiled.float32,
  );

  let depthBufferBase: number | undefined;
  let depthCapacityPixels = 0;
  /** The size of the draw the depth buffer holds, whose pixels another size would read at the wrong places. */
  let depthWidth = 0;
  let depthHeight = 0;

  function clearDepthBuffer(): void {
    if (depthBufferBase === undefined) return;
    const view = new DataView(memory.buffer);
    for (let i = 0; i < depthCapacityPixels; i++) view.setFloat64(depthBufferBase + i * 8, Infinity, true);
  }

  function draw(ctx: WasmRasterContext, options: WasmRasterDrawOptions): CpuDrawBuffer {
    const { width, height, out } = options;
    const first = options.first ?? 0;
    const firstAttr = attrLayout[0];
    const inferredCount = firstAttr
      ? Math.floor(ctx.attributes[firstAttr.slot]!.length / (firstAttr.sizeBytes / 8)) - first
      : 0;
    const vertexCount = options.count ?? inferredCount;
    const sharedCtx = { uniforms: ctx.uniforms, textures: ctx.textures } as CpuShaderContext;
    // Every region is sized before anything is written, so the depth buffer can
    // be moved out of the way of the others first.
    const vertexHeapEnd = vertexMarshaller.footprint(sharedCtx, heapStart);
    const fragmentHeapStart = align8(vertexHeapEnd);
    const fragmentHeapEnd = fragmentMarshaller.footprint(sharedCtx, fragmentHeapStart);

    let cursor = align8(Math.max(vertexHeapEnd, fragmentHeapEnd));
    const attrSrcBase = cursor;
    cursor = align8(cursor + vertexCount * attrStrideBytes);
    const attrDescBase = cursor;
    cursor = align8(cursor + attrLayout.length * ATTR_DESC_BYTES);
    const positionsOutBase = cursor;
    cursor = align8(cursor + vertexCount * VEC4_BYTES);
    const varyingsOutBase = cursor;
    cursor = align8(cursor + vertexCount * varyingBytes);
    const varyingDescBase = cursor;
    cursor = align8(cursor + varyingLayout.length * VARYING_DESC_BYTES);
    const clipScratchBase = cursor;
    cursor = align8(cursor + 4 * (VEC4_BYTES + varyingBytes));
    // near-plane clipping fans each triangle into at most a quad (2 triangles).
    const maxClippedVertices = vertexCount * 2;
    const clippedPositionsOutBase = cursor;
    cursor = align8(cursor + maxClippedVertices * VEC4_BYTES);
    const clippedVaryingsOutBase = cursor;
    cursor = align8(cursor + maxClippedVertices * varyingBytes);
    const outputBase = cursor;
    cursor = align8(cursor + width * height * VEC4_BYTES);

    // The depth buffer stays where it is until a draw's regions reach it, and
    // moves above them then, so occlusion carries across draws that differ in
    // size without a copy on each draw.
    const neededDepthPixels = width * height;
    const outgrown = neededDepthPixels > depthCapacityPixels;
    const resized = width !== depthWidth || height !== depthHeight;
    const needsClear = depthBufferBase === undefined || outgrown || resized || options.clearDepth !== false;
    depthWidth = width;
    depthHeight = height;
    const movesTo = depthBufferBase === undefined || cursor > depthBufferBase ? cursor : undefined;
    const previous = { base: depthBufferBase, pixels: depthCapacityPixels };
    if (movesTo !== undefined) depthBufferBase = movesTo;
    if (outgrown) depthCapacityPixels = neededDepthPixels;
    const memoryEnd = depthBufferBase! + depthCapacityPixels * 8;

    if (memoryEnd > memory.buffer.byteLength) {
      memory.grow(Math.ceil((memoryEnd - memory.buffer.byteLength) / 65536));
    }
    if (needsClear) clearDepthBuffer();
    else if (movesTo !== undefined && previous.base !== undefined) {
      new Uint8Array(memory.buffer).copyWithin(movesTo, previous.base, previous.base + previous.pixels * 8);
    }

    vertexMarshaller.marshal(sharedCtx, heapStart);
    fragmentMarshaller.marshal(sharedCtx, fragmentHeapStart);

    const view = new DataView(memory.buffer);
    for (const a of attrLayout) {
      const src = ctx.attributes[a.slot];
      if (!src) throw new Error(`[RMSL] compileWasm: draw() is missing attribute "${a.slot}"`);
      const componentCount = a.sizeBytes / 8;
      for (let v = 0; v < vertexCount; v++) {
        const base = attrSrcBase + v * attrStrideBytes + a.offset;
        const srcIndex = (v + first) * componentCount;
        for (let c = 0; c < componentCount; c++) {
          view.setFloat64(base + c * 8, src[srcIndex + c] as number, true);
        }
      }
    }
    writeAttributeDescriptors(
      view,
      attrDescBase,
      attrLayout.map((a) => ({ srcOffset: a.offset, destAddress: a.destAddress, sizeBytes: a.sizeBytes })),
    );
    writeVaryingDescriptors(
      view,
      varyingDescBase,
      varyingLayout.map((v) => ({
        recordOffset: v.offset,
        vertexSrcAddress: v.vertexSrcAddress,
        fragmentDestAddress: v.fragmentDestAddress,
        sizeBytes: v.sizeBytes,
      })),
    );

    if (options.clear === false && out) {
      // A draw that keeps what lies under it, given a buffer, draws over what that buffer holds.
      new Float64Array(memory.buffer, outputBase, width * height * 4).set(out.subarray(0, width * height * 4));
    } else if (options.clear !== false) {
      const [r, g, b, a] = options.clearColor ?? TRANSPARENT_BLACK;
      const output = new Float64Array(memory.buffer, outputBase, width * height * 4);
      for (let i = 0; i < output.length; i += 4) {
        output[i] = r;
        output[i + 1] = g;
        output[i + 2] = b;
        output[i + 3] = a;
      }
    }

    rasterize(
      vertexCount,
      attrSrcBase,
      attrStrideBytes,
      attrDescBase,
      attrLayout.length,
      positionAddress,
      positionsOutBase,
      width,
      height,
      fragmentValueAddress,
      outputBase,
      varyingBytes,
      varyingDescBase,
      varyingLayout.length,
      varyingsOutBase,
      clipScratchBase,
      clippedPositionsOutBase,
      clippedVaryingsOutBase,
      depthBufferBase!,
      writesColour,
      fragCoordAddress,
      writesFragCoord,
      fragDepthAddress,
      writesDepth,
      discardAddress,
      mayDiscard,
    );

    const result = new Float64Array(memory.buffer, outputBase, width * height * 4);
    if (out) {
      out.set(result);
      return out;
    }
    return new Float64Array(result); // copy out — memory can grow (and detach `result`'s buffer) on a later draw()
  }

  return { draw };
}
