import {
  AttributeNode,
  Node,
  ShaderType,
  StorageBufferAttribute,
  UniformArrayNode,
  UniformNode,
  UniformValue,
} from "../../core";
import { compile, WgslResource } from "../../wgsl";
import {
  Adapter,
  DrawClearOptions,
  DrawCountOptions,
  requestedStorageSlots,
  slotOf,
  TRANSPARENT_BLACK,
  TypedArray,
} from "../adapter";
import { componentCountOf, componentKindOf, type CpuTextureData } from "../cpu";
import { textureImage } from "../texture-image";
import { CompileCtx, storageAttributes, storageNodesOf, VertexRoot } from "../shared";
import type { WgslContext } from "./context-wgsl";
import { spread, storageLayout, type StorageLayout } from "./storage-layout";
import {
  compileWGSLStage,
  compileWGSLWithStage,
  isWgslTexture,
  sharedSamplerDeclarations,
  typeToWGSL,
  WGSL_RENDER_STORAGE_GROUP,
  wgslMatrixColumns,
  wgslUniformLayout,
} from "./wgsl";

/**
 * A device on `adapter` that binds as many storage buffers in one shader
 * stage as the hardware can, rather than WebGPU's default of 8.
 */
export function requestComputeDevice(adapter: GPUAdapter): Promise<GPUDevice> {
  return adapter.requestDevice({
    requiredLimits: { maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage },
  });
}

/**
 * Throws unless `limits`, a device's or the adapter's it is requested from,
 * let one shader stage bind every storage buffer a compiled compute program
 * uses. Past the limit, the device refuses the pipeline, and the work
 * submitted with it is dropped without an error in JavaScript; this says so
 * before that happens.
 */
export function assertStorageBufferLimit(
  limits: GPUSupportedLimits,
  resources: readonly WgslResource[],
  where: string,
): void {
  const count = resources.filter((r) => r.kind === "storage").length;
  const limit = limits.maxStorageBuffersPerShaderStage;
  if (count > limit) {
    throw new Error(
      `[RMSL] ${where}: a compute program uses ${count} storage buffers, more than the ${limit} this device allows in one shader stage; split it into programs that each use fewer, or keep values that belong together in one buffer`,
    );
  }
}

/** One typed array per storage slot, keyed by name — read_write, so the same
 * record a caller passes into `compute` is the one read back out of. */
export type AdapterResult = Record<string, TypedArray>;

/**
 * Unlike GL's `drawArrays`, WebGPU bakes primitive topology into the
 * pipeline (`createWgsl`'s own `topology` option), not the draw call —
 * so there's no `mode` here, just `DrawCountOptions`'s own `count`/`first`
 * plus how many instances.
 */
export interface WgslDrawOptions extends DrawCountOptions, DrawClearOptions {
  /** Instances to draw. Defaults to 1. */
  instanceCount?: number;
}

export interface CreateWgslAdapterOptions {
  /** A traditional attribute()/uniform()/varying() vertex stage — requires `fragment` too. */
  vertex: VertexRoot;
  /** The fragment stage paired with `vertex`. */
  fragment: Node<ShaderType> | readonly Node<ShaderType>[];
  /** Fixed for the render pipeline's lifetime — see `WgslDrawOptions`. Defaults to "triangle-list". */
  topology?: GPUPrimitiveTopology;
  /**
   * A compute context to draw from: the adapter uses its device, and a
   * `storage()` node the stages read binds that context's buffer for the
   * node's attribute, so a draw reads what the context's programs wrote.
   * Without one, the adapter requests its own device and creates its own
   * storage buffers, filled with `setAttribute(node.name, data)`.
   */
  context?: WgslContext;
}

/** How a WGSL adapter configures the canvas it attaches to. */
export interface WgslAttachOptions {
  /**
   * Whether the canvas has an alpha channel, as three.js's `alpha` parameter.
   * `true`, the default, configures it `premultiplied`, so a pixel the draw
   * leaves transparent shows the page. `false` configures it `opaque`.
   */
  alpha?: boolean;
}

/** A WGSL adapter, plus the one thing the shared `Adapter` shape has no
 * generic name for: direct access to a vertex attribute's persistent GPU
 * buffer, so another adapter's `compute` pass sharing this one's device
 * could bind it without a readback ever happening. */
export interface WgslAdapter extends Adapter<AdapterResult, WgslDrawOptions> {
  // Narrower than the base Adapter's `void | Promise<void>` — requesting a
  // GPUAdapter/GPUDevice is always async, unlike GL's attach.
  attach(canvas?: HTMLCanvasElement, options?: WgslAttachOptions): Promise<void>;
  draw(options?: WgslDrawOptions): void;
  buffer(slot: string): GPUBuffer | undefined;
  /** The device backing this adapter, once `attach()` has resolved — a
   * `draw` pass sharing its buffers has to build its own pipeline against
   * this same device, since a GPUBuffer is only valid on the device that
   * created it. */
  device(): GPUDevice | undefined;
}

function freshCtx(shaderStage: CompileCtx["shaderStage"]): CompileCtx {
  return {
    nextId: 0,
    shaderStage,
    uniforms: new Map(),
    attributes: new Map(),
    varyings: new Map(),
    outputs: new Map(),
    wgslSamplers: new Map(),
    varDefs: new Map(),
    memo: new Map(),
    wgslHelpers: new Set(),
    positionWritten: false,
    inFn: false,
    fragDepthUsed: false,
    fragCoordUsed: false,
    jsParams: new Set(),
    jsHelpers: new Set(),
    outTarget: null,
    jsEpoch: 0,
    jsReadsSlot: false,
    derivatives: "throw",
    reentrant: false,
    jsNeedsRes: false,
  };
}

type ReflectedAttribute = { slot: string; type: string };
type ReflectedUniform = {
  slot: string;
  type: string;
  length?: number;
  node: UniformNode<ShaderType> | UniformArrayNode<ShaderType>;
};

/** What a vertex/fragment stage reads: its own attributes (vertex only,
 * in creation order) and uniforms — same ctx-walking trick `compile()`
 * uses for a compute program's storage()/uniform() resources. */
function reflectStage(
  root: Node<ShaderType> | readonly Node<ShaderType>[],
  shaderStage: "vertex" | "fragment",
): { attributes: ReflectedAttribute[]; uniforms: ReflectedUniform[] } {
  let ctx = freshCtx(shaderStage);
  let nodes = Array.isArray(root) ? root : root ? [root] : [];
  for (let n of nodes) compileWGSLStage(n as Node<ShaderType>, ctx);
  return {
    attributes: [...ctx.attributes.entries()]
      .sort((a, b) => a[1].id - b[1].id)
      .map(([, info]) => ({ slot: info.slot, type: info.type })),
    uniforms: [...ctx.uniforms.values()]
      .sort((a, b) => a.order! - b.order!)
      .map((u) => ({ slot: u.slot, type: u.type, length: u.length, node: u.node as ReflectedUniform["node"] })),
  };
}

/** The place of a uniform in the order the program created its uniforms. */
const creationOf = (node: ReflectedUniform["node"]) => (node as unknown as { value: { id: number } }).value.id;

/** Scalar/vector f32 only — a matrix attribute arrives as several columns
 * (see wgslMatrixColumns in wgsl.ts) with no single `GPUVertexFormat` of
 * its own, which this adapter doesn't build the multi-slot layout for. */
const VERTEX_FORMAT: Record<string, GPUVertexFormat> = {
  f32: "float32",
  "vec2<f32>": "float32x2",
  "vec3<f32>": "float32x3",
  "vec4<f32>": "float32x4",
};

function vertexComponentCount(type: string): number {
  let match = /^vec(\d)<f32>$/.exec(type);
  return match ? Number(match[1]) : 1;
}

/**
 * A uniform buffer's CPU-side copy, viewed as each 32-bit element type a
 * member can have, so an `int` member gets integer bits and not the bits of
 * the equivalent float.
 */
export type UniformScratch = { f32: Float32Array; i32: Int32Array; u32: Uint32Array };

export function uniformScratch(bytes: number): UniformScratch {
  let buffer = new ArrayBuffer(bytes);
  return { f32: new Float32Array(buffer), i32: new Int32Array(buffer), u32: new Uint32Array(buffer) };
}

/**
 * The typed array that holds one element of `type` exactly as the GPU reads it.
 * Takes an RMSL type name (`int`, `uvec2`) or a WGSL one (`i32`, `vec2<u32>`):
 * compute uniform resources are parsed back out of the generated WGSL, so they
 * carry the WGSL spelling.
 */
export function elementView(type: string, views: UniformScratch): Float32Array | Int32Array | Uint32Array {
  let kind = componentKindOf(type);
  if (kind === "int" || type.includes("i32")) return views.i32;
  if (kind === "uint" || type.includes("u32")) return views.u32;
  return views.f32;
}

/** Where one uniform lives in its buffer: as compiled, whether from a layout member or a reflected resource. */
export type UniformPlacement = { offset: number; size: number; type: string; length?: number };

/** A uniform resource of a compiled compute program as a placement, which names its type in the WGSL spelling. */
export function resourcePlacement(resource: Extract<WgslResource, { kind: "uniform" }>): UniformPlacement {
  return { ...resource, type: typeToWGSL[resource.shaderType] ?? resource.shaderType };
}

/** Number of 32-bit slots between a WGSL matrix's columns, and its column and row counts. */
function matrixShape(type: string): { columns: number; rows: number; columnStride: number } | undefined {
  let match = /^mat(\d)x(\d)<f32>$/.exec(type);
  if (!match) return undefined;
  let rows = Number(match[2]);
  return { columns: Number(match[1]), rows, columnStride: rows === 3 ? 4 : rows };
}

/** Writes one value — a scalar, a vector, or a column-major matrix — at slot `at`. */
function writeUniformElement(
  view: Float32Array | Int32Array | Uint32Array,
  at: number,
  type: string,
  value: number | ArrayLike<number>,
): void {
  if (typeof value !== "object") {
    view[at] = value;
    return;
  }
  let matrix = matrixShape(type);
  if (!matrix) {
    view.set(value, at);
    return;
  }
  // A value shorter than the matrix leaves the rest of it as it was.
  const length = Math.min(value.length, matrix.columns * matrix.rows);
  for (let k = 0; k < length; k++) {
    view[at + Math.floor(k / matrix.rows) * matrix.columnStride + (k % matrix.rows)] = value[k]!;
  }
}

/**
 * Writes a uniform's value into its place in the scratch buffer, laid out as
 * WGSL reads it: an array's elements `size / length` bytes apart (each in its
 * own 16-byte slot), and a `mat3x3`'s columns padded to four components.
 */
export function writeUniformMember(scratch: UniformScratch, placement: UniformPlacement, value: unknown): void {
  let view = elementView(placement.type, scratch);
  if (placement.length === undefined) {
    writeUniformElement(view, placement.offset / 4, placement.type, value as number | number[]);
    return;
  }
  let stride = placement.size / placement.length / 4;
  (value as (number | number[])[]).forEach((element, i) =>
    writeUniformElement(view, placement.offset / 4 + i * stride, placement.type, element),
  );
}

/** A uniform buffer's byte size: WGSL rounds a struct up to its 16-byte alignment. */
export function uniformBufferSize(end: number): number {
  return Math.max(16, Math.ceil(end / 16) * 16);
}

/**
 * Compiles a `vertex`+`fragment` pair into a real `GPURenderPipeline` —
 * the render-pipeline-shaped counterpart to {@link createWgslCompute}'s
 * compute-pipeline shape, and the WGSL-side sibling of `createWasm`/
 * `createJs`. There is no `getActiveAttrib` the way WebGL has, and a
 * vertex buffer's layout is fixed at pipeline-creation time, not
 * discoverable from a linked program afterward, so attributes/uniforms
 * are reflected by walking the vertex/fragment graphs directly
 * (`reflectStage`) rather than read back off the compiled program.
 */
export function createWgsl(options: CreateWgslAdapterOptions): WgslAdapter {
  let device: GPUDevice | null = null;

  let renderPipeline: GPURenderPipeline | null = null;
  let context: GPUCanvasContext | null = null;
  let vertexAttributes: ReflectedAttribute[] = [];
  let renderUniformLayout: ReturnType<typeof wgslUniformLayout> | null = null;
  let renderUniformBuffer: GPUBuffer | null = null;
  let renderUniformScratch: UniformScratch | null = null;
  let renderBindGroup0: GPUBindGroup | null = null;
  let vertexBuffers = new Map<string, { buffer: GPUBuffer; componentCount: number }>();
  let vertexCount = 0;
  let countSlot: string | undefined;
  /** The storage attributes the stages read, by slot, and the buffers they bind. */
  let storages = new Map<string, StorageBufferAttribute>();
  let ownStorageBuffers = new Map<StorageBufferAttribute, GPUBuffer>();
  /** Bind groups by group index; a gap below the storage group is an empty group. */
  let renderBindGroups: (GPUBindGroup | null)[] = [];

  function storageBuffer(attribute: StorageBufferAttribute): GPUBuffer {
    if (options.context) return options.context.buffer(attribute);
    let existing = ownStorageBuffers.get(attribute);
    if (existing) return existing;
    const layout = storageLayout(attribute);
    existing = device!.createBuffer({
      size: Math.max(4, attribute.count * layout.stride * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device!.queue.writeBuffer(existing, 0, spread(attribute.array, 0, layout).data as BufferSource);
    ownStorageBuffers.set(attribute, existing);
    return existing;
  }

  let pendingUniforms = new Map<string, number | number[]>();
  let pendingAttributes = new Map<string, TypedArray>();
  let pendingTextures = new Map<string, CpuTextureData>();
  /** The textures the program reads, in binding order. */
  let textureDeclarations: ReturnType<typeof sharedSamplerDeclarations> = [];
  /** The GPU texture, and sampler for a float one, of each sampler the host has set. */
  let gpuTextures = new Map<string, { texture: GPUTexture; sampler: GPUSampler | null; samplerKey: string }>();
  /** Whether a texture was set since the bind groups of groups 1 and 2 were built, or they were never built. */
  let texturesChanged = true;

  function setTexture(uniform: UniformNode<ShaderType> | string, data: CpuTextureData): void {
    const slot = slotOf(uniform);
    if (!device) {
      pendingTextures.set(slot, data);
      return;
    }
    const declaration = textureDeclarations.find((t) => t.slot === slot);
    // A texture the program does not read is ignored, as `setUniform` ignores a uniform it does not read.
    if (!declaration) return;
    const image = textureImage(data, declaration.shaderType);
    const volume = declaration.shaderType.endsWith("3D");
    const format: GPUTextureFormat = image.normalized
      ? "rgba8unorm"
      : (`rgba${image.bits}${image.signed ? "sint" : "uint"}` as GPUTextureFormat);
    const held = gpuTextures.get(slot);
    // A texture of the shape the sampler already holds is written in place.
    const reuse =
      held !== undefined &&
      held.texture.width === image.width &&
      held.texture.height === image.height &&
      held.texture.depthOrArrayLayers === image.depth &&
      held.texture.format === format;
    if (!reuse) held?.texture.destroy();
    const texture = reuse
      ? held!.texture
      : device.createTexture({
          size: [image.width, image.height, image.depth],
          dimension: volume ? "3d" : "2d",
          format,
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
    device.queue.writeTexture(
      { texture },
      image.texels as BufferSource,
      { bytesPerRow: image.width * 4 * (image.bits / 8), rowsPerImage: image.height },
      [image.width, image.height, image.depth],
    );
    const address = (wrap: CpuTextureData["wrapS"]): GPUAddressMode =>
      wrap === "repeat" ? "repeat" : wrap === "mirror" ? "mirror-repeat" : "clamp-to-edge";
    // An integer texture is read with textureLoad and takes no sampler.
    const filter: GPUFilterMode = data.magFilter === "linear" ? "linear" : "nearest";
    const samplerKey = `${filter},${address(data.wrapS)},${address(data.wrapT)},${address(data.wrapR)}`;
    const sampler = declaration.integer
      ? null
      : reuse && held!.samplerKey === samplerKey
        ? held!.sampler
        : device.createSampler({
            // A CPU target has no footprint to minify by, so both filters follow `magFilter`.
            magFilter: filter,
            minFilter: filter,
            addressModeU: address(data.wrapS),
            addressModeV: address(data.wrapT),
            addressModeW: address(data.wrapR),
          });
    // The bind groups name the texture and sampler, so they are built again only when either is new.
    if (!reuse || sampler !== held!.sampler) texturesChanged = true;
    gpuTextures.set(slot, { texture, sampler, samplerKey });
  }

  /** Builds the bind groups of the textures (group 1) and the samplers of the float ones (group 2). */
  function bindTextures(): void {
    if (!device || !renderPipeline) return;
    const entries = textureDeclarations.map((declaration) => {
      const held = gpuTextures.get(declaration.slot);
      if (!held) throw new Error(`[RMSL] createWgsl: setTexture was never called for "${declaration.slot}"`);
      return { declaration, held };
    });
    renderBindGroups[1] = device.createBindGroup({
      layout: renderPipeline.getBindGroupLayout(1),
      entries: entries.map(({ declaration, held }, binding) => ({
        binding,
        resource: held.texture.createView({ dimension: declaration.shaderType.endsWith("3D") ? "3d" : "2d" }),
      })),
    });
    const filtered = entries.filter(({ declaration }) => !declaration.integer);
    if (filtered.length > 0) {
      renderBindGroups[2] = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(2),
        entries: filtered.map(({ held }, binding) => ({ binding, resource: held.sampler! })),
      });
    }
    texturesChanged = false;
  }

  function setUniform<T extends ShaderType>(uniform: UniformNode<T>, value: UniformValue<T>): void;
  function setUniform<T extends ShaderType>(uniform: UniformArrayNode<T>, value: UniformValue<T>[]): void;
  function setUniform(slot: string, value: number | number[]): void;
  function setUniform(uniform: UniformNode<ShaderType> | UniformArrayNode<ShaderType> | string, _value: unknown): void {
    const slot = slotOf(uniform);
    const value = _value as number | number[];
    if (!device) {
      pendingUniforms.set(slot, value);
      return;
    }

    let renderMember = renderUniformLayout?.members.find((m) => m.name === slot);
    if (renderMember && renderUniformScratch && renderUniformBuffer) {
      writeUniformMember(renderUniformScratch, renderMember, value);
      device.queue.writeBuffer(renderUniformBuffer, 0, renderUniformScratch.f32 as BufferSource);
      return;
    }

    // A uniform the program doesn't read is ignored, as in TSL and every other adapter.
    if (!renderPipeline) pendingUniforms.set(slot, value);
  }

  function setAttribute<T extends ShaderType>(attribute: AttributeNode<T>, data: TypedArray): void;
  function setAttribute(slot: string, data: TypedArray): void;
  function setAttribute(attribute: AttributeNode<ShaderType> | string, data: TypedArray): void {
    const slot = slotOf(attribute);
    if (!device) {
      pendingAttributes.set(slot, data);
      return;
    }

    let attrInfo = vertexAttributes.find((a) => a.slot === slot);
    if (attrInfo) {
      let componentCount = vertexComponentCount(attrInfo.type);
      let bytes = data.length * 4;
      let existing = vertexBuffers.get(slot);
      if (!existing || existing.buffer.size < bytes) {
        existing?.buffer.destroy();
        let buffer = device.createBuffer({ size: bytes, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
        existing = { buffer, componentCount };
        vertexBuffers.set(slot, existing);
      }
      device.queue.writeBuffer(existing.buffer, 0, data as BufferSource);
      countSlot ??= slot;
      if (slot === countSlot) vertexCount = Math.floor(data.length / componentCount);
      return;
    }

    let storage = storages.get(slot);
    if (storage) {
      // The buffer is laid out as WGSL lays out a storage array, so a vec3 leaves a slot empty after each element.
      const { slot: first, data: slots } = spread(data, 0, storageLayout(storage));
      device.queue.writeBuffer(storageBuffer(storage), first * 4, slots as BufferSource);
      return;
    }

    if (!renderPipeline) {
      pendingAttributes.set(slot, data);
    }
    // Past attach, an attribute the program does not read is taken and counts for nothing, as an unread uniform is.
  }

  let adapter: WgslAdapter = {
    async attach(canvas, attachOptions) {
      if (options.context) {
        device = options.context.device;
      } else {
        let gpuAdapter = await navigator.gpu?.requestAdapter();
        if (!gpuAdapter) throw new Error("[RMSL] WebGPU is not available");
        device = await gpuAdapter.requestDevice();
      }

      let target = canvas ?? document.createElement("canvas");
      let glCanvasContext = target.getContext("webgpu");
      if (!glCanvasContext) throw new Error("[RMSL] WebGPU canvas context unavailable");
      context = glCanvasContext;
      let format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode: attachOptions?.alpha === false ? "opaque" : "premultiplied" });

      let vertexReflection = reflectStage(options.vertex, "vertex");
      let fragmentReflection = reflectStage(options.fragment, "fragment");
      vertexAttributes = vertexReflection.attributes;

      for (let attr of vertexAttributes) {
        if (wgslMatrixColumns(attr.type)) {
          throw new Error(
            `[RMSL] createWgsl's vertex attributes don't support matrix types ("${attr.slot}": ${attr.type})`,
          );
        }
      }

      // Vertex and fragment share one uniform struct — same reasoning
      // CompileWGSLOptions.uniforms documents on the compiler side: two
      // stages declaring only what they individually read would pack a
      // shared member at two different offsets.
      let sharedUniforms = new Map<string, ReflectedUniform>();
      let sharedTextures = new Map<string, ReflectedUniform>();
      for (let u of [...vertexReflection.uniforms, ...fragmentReflection.uniforms]) {
        (isWgslTexture(u.type) ? sharedTextures : sharedUniforms).set(u.slot, u);
      }
      let renderDeclarations = [...sharedUniforms.values()].sort((a, b) => creationOf(a.node) - creationOf(b.node));
      let renderUniforms = renderDeclarations.map((u) => u.node);
      // A texture takes a binding of its own, in the order of the whole program's textures.
      textureDeclarations = sharedSamplerDeclarations(
        [...sharedTextures.values()].map((u) => ({
          slot: u.slot,
          type: Object.keys(typeToWGSL).find((key) => /sampler/.test(key) && typeToWGSL[key] === u.type) ?? u.type,
        })),
      );
      storages = storageAttributes([options.vertex, options.fragment]);
      let storageNodes = storageNodesOf([options.vertex, options.fragment]);
      let storageOrder = storageNodes.map((node) => node.name);
      let stageOptions = {
        ...(renderUniforms.length > 0 ? { uniforms: renderUniforms } : {}),
        ...(storageNodes.length > 0 ? { storages: storageNodes } : {}),
        ...(textureDeclarations.length > 0
          ? { samplers: textureDeclarations.map((t) => ({ slot: t.slot, type: t.shaderType })) }
          : {}),
      };

      let vertexCode = compileWGSLWithStage(options.vertex as Node<ShaderType>, "vertex", stageOptions);
      let fragmentCode = compileWGSLWithStage(options.fragment, "fragment", stageOptions);
      let vertexModule = device.createShaderModule({ code: vertexCode });
      let fragmentModule = device.createShaderModule({ code: fragmentCode });

      let buffers: GPUVertexBufferLayout[] = vertexAttributes.map((attr, location) => {
        let gpuFormat = VERTEX_FORMAT[attr.type];
        if (!gpuFormat) {
          throw new Error(`[RMSL] unsupported vertex attribute type "${attr.type}" for slot "${attr.slot}"`);
        }
        return {
          arrayStride: vertexComponentCount(attr.type) * 4,
          attributes: [{ shaderLocation: location, offset: 0, format: gpuFormat }],
        };
      });

      renderPipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: { module: vertexModule, entryPoint: "main", buffers },
        fragment: { module: fragmentModule, entryPoint: "main", targets: [{ format }] },
        primitive: { topology: options.topology ?? "triangle-list" },
      });

      if (renderUniforms.length > 0) {
        renderUniformLayout = wgslUniformLayout(renderDeclarations);
        renderUniformBuffer = device.createBuffer({
          size: uniformBufferSize(renderUniformLayout.size),
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        renderUniformScratch = uniformScratch(renderUniformBuffer.size);
        renderBindGroup0 = device.createBindGroup({
          layout: renderPipeline.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: renderUniformBuffer } }],
        });
      }

      renderBindGroups = [renderBindGroup0];
      if (storageOrder.length > 0 || textureDeclarations.length > 0) {
        // The derived layout has a (possibly empty) group for every index up to the
        // last group used, and a draw needs one bound at each.
        const lastGroup =
          storageOrder.length > 0 ? WGSL_RENDER_STORAGE_GROUP : textureDeclarations.some((t) => !t.integer) ? 2 : 1;
        for (let group = 0; group < lastGroup; group++) {
          renderBindGroups[group] ??= device.createBindGroup({
            layout: renderPipeline.getBindGroupLayout(group),
            entries: [],
          });
        }
        if (storageOrder.length > 0) {
          renderBindGroups[WGSL_RENDER_STORAGE_GROUP] = device.createBindGroup({
            layout: renderPipeline.getBindGroupLayout(WGSL_RENDER_STORAGE_GROUP),
            entries: storageOrder.map((slot, binding) => ({
              binding,
              resource: { buffer: storageBuffer(storages.get(slot)!) },
            })),
          });
        }
      }

      for (let [slot, value] of pendingUniforms) adapter.setUniform(slot, value);
      for (let [slot, data] of pendingAttributes) adapter.setAttribute(slot, data);
      for (let [slot, data] of pendingTextures) setTexture(slot, data);
      pendingUniforms.clear();
      pendingAttributes.clear();
      pendingTextures.clear();
    },

    setUniform,
    setAttribute,
    setTexture,

    draw(drawOptions) {
      if (!device || !renderPipeline || !context) {
        throw new Error("[RMSL] createWgsl: attach() was never called");
      }
      if (textureDeclarations.length > 0 && texturesChanged) bindTextures();
      let encoder = device.createCommandEncoder();
      let view = context.getCurrentTexture().createView();
      let pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view,
            clearValue: drawOptions?.clearColor ?? TRANSPARENT_BLACK,
            loadOp: drawOptions?.clear === false ? "load" : "clear",
            storeOp: "store",
          },
        ],
      });
      pass.setPipeline(renderPipeline);
      renderBindGroups.forEach((group, index) => group && pass.setBindGroup(index, group));
      for (let i = 0; i < vertexAttributes.length; i++) {
        let vb = vertexBuffers.get(vertexAttributes[i].slot);
        if (vb) pass.setVertexBuffer(i, vb.buffer);
      }
      const first = drawOptions?.first ?? 0;
      pass.draw(drawOptions?.count ?? Math.max(0, vertexCount - first), drawOptions?.instanceCount ?? 1, first, 0);
      pass.end();
      device.queue.submit([encoder.finish()]);
    },

    buffer(slot) {
      return vertexBuffers.get(slot)?.buffer;
    },

    device() {
      return device ?? undefined;
    },

    destroy() {
      for (let vb of vertexBuffers.values()) vb.buffer.destroy();
      for (let buffer of ownStorageBuffers.values()) buffer.destroy();
      renderUniformBuffer?.destroy();
      for (let held of gpuTextures.values()) held.texture.destroy();
      if (!options.context) device?.destroy();
    },
  };

  return adapter;
}

export interface CreateWgslComputeOptions {
  /** Threads per workgroup, along the WGSL `@workgroup_size(n)` x-axis only — see `compile()` in `src/wgsl.ts`. Defaults to 64. */
  workgroupSize?: number;
}

/**
 * `setAttribute`/`setUniform` collect draw state the way {@link WgslAdapter}
 * does; `compute()` is the only invocation method, since a `storage()`/
 * `invocationIndex()` program has no `draw()` counterpart — unlike
 * `WgslAdapter`, that's not just unused here, it's not part of the type
 * at all. `attach()` takes no canvas: a compute pass has nothing to
 * present.
 */
export interface WgslComputeAdapter {
  attach(): Promise<void>;
  setUniform<T extends ShaderType>(uniform: UniformNode<T>, value: UniformValue<T>): void;
  setUniform<T extends ShaderType>(uniform: UniformArrayNode<T>, value: UniformValue<T>[]): void;
  setUniform(slot: string, value: number | number[]): void;
  setAttribute<T extends ShaderType>(attribute: AttributeNode<T>, data: TypedArray): void;
  setAttribute(slot: string, data: TypedArray): void;
  /**
   * With `out`, writes the result into it and returns it. Without it,
   * just dispatches — for keeping the result GPU-resident (a storage
   * buffer a `draw` pass reads directly via {@link buffer}), forcing a
   * readback on every call would take away the one advantage a compute
   * pass has over the CPU backends.
   *
   * `count` is the number of invocations to dispatch. Given none, it is one
   * per element of the first storage buffer `setAttribute` was given. A
   * dispatch covers whole workgroups, so the last one also runs the
   * invocations past the count.
   */
  compute(out?: AdapterResult, count?: number): Promise<AdapterResult | void>;
  /** A storage slot's persistent `GPUBuffer` — so a `draw` pass sharing
   * this adapter's device can bind it directly, no readback. A `vec3` or
   * `mat3` slot is laid out as WGSL lays out a storage array, 16 bytes for each
   * `vec3`, so a draw reads it with a stride of four slots for each. */
  buffer(slot: string): GPUBuffer | undefined;
  /** The device backing this adapter, once `attach()` has resolved. */
  device(): GPUDevice | undefined;
  destroy(): void;
}

/**
 * Compiles a `storage()`/`invocationIndex()` program into a real
 * `GPUComputePipeline` and wraps it in a {@link WgslComputeAdapter} — the
 * wgpu-compute-pipeline-shaped counterpart to {@link createWgsl}'s
 * render-pipeline shape, and the WGSL-side sibling of `createWasmCompute`/
 * `createJsCompute`. Unlike those two, this one dispatches on the GPU
 * exactly as a real compute pipeline would — no host-side per-element
 * loop, real `@workgroup_size` grouping — so it doesn't carry either gap
 * ROADMAP.md's "createWasmCompute's dispatch model" item records for the
 * CPU backends.
 */
export function createWgslCompute(
  compute: Node<ShaderType> | readonly Node<ShaderType>[],
  options: CreateWgslComputeOptions = {},
): WgslComputeAdapter {
  let device: GPUDevice | null = null;

  let computePipeline: GPUComputePipeline | null = null;
  let computeResources: WgslResource[] = [];
  /** Elements of the first storage buffer the host passed, the dispatch's
   *  count when the caller names none. */
  let firstStorageCount = 0;
  let firstStorageSeen = false;
  /** Each storage slot's buffer, sized to its own elements and laid out as WGSL lays out a storage array. */
  let storageSlots = new Map<
    string,
    { buffer: GPUBuffer; layout: StorageLayout; elements: number; itemSize: number }
  >();
  let computeBindGroup1: GPUBindGroup | null = null;
  let computeUniformBuffer: GPUBuffer | null = null;
  let computeUniformScratch: UniformScratch | null = null;
  let computeBindGroup0: GPUBindGroup | null = null;

  let pendingUniforms = new Map<string, number | number[]>();
  let pendingAttributes = new Map<string, TypedArray>();

  function computeStorageResources(): Extract<WgslResource, { kind: "storage" }>[] {
    return computeResources.filter((r): r is Extract<WgslResource, { kind: "storage" }> => r.kind === "storage");
  }
  function computeUniformResources(): Extract<WgslResource, { kind: "uniform" }>[] {
    return computeResources.filter((r): r is Extract<WgslResource, { kind: "uniform" }> => r.kind === "uniform");
  }

  /** Makes the buffer of `resource` hold `elements` elements, keeping every other slot's buffer and what it holds. */
  function sizeStorageSlot(resource: Extract<WgslResource, { kind: "storage" }>, elements: number) {
    const existing = storageSlots.get(resource.name);
    if (existing && existing.elements === elements) return existing;
    const itemSize = componentCountOf(resource.shaderType);
    const layout = storageLayout({ itemSize, elementType: resource.shaderType });
    const buffer = device!.createBuffer({
      size: Math.max(4, elements * layout.stride * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const slot = { buffer, layout, elements, itemSize };
    storageSlots.set(resource.name, slot);
    // The bind group needs every slot's buffer, so it is made once each has one.
    const resources = computeStorageResources();
    if (resources.every((r) => storageSlots.has(r.name))) {
      computeBindGroup1 = device!.createBindGroup({
        layout: computePipeline!.getBindGroupLayout(1),
        entries: resources.map((r) => ({ binding: r.binding, resource: { buffer: storageSlots.get(r.name)!.buffer } })),
      });
    }
    existing?.buffer.destroy();
    return slot;
  }

  function setUniform<T extends ShaderType>(uniform: UniformNode<T>, value: UniformValue<T>): void;
  function setUniform<T extends ShaderType>(uniform: UniformArrayNode<T>, value: UniformValue<T>[]): void;
  function setUniform(slot: string, value: number | number[]): void;
  function setUniform(uniform: UniformNode<ShaderType> | UniformArrayNode<ShaderType> | string, _value: unknown): void {
    const slot = slotOf(uniform);
    const value = _value as number | number[];
    if (!device) {
      pendingUniforms.set(slot, value);
      return;
    }

    let computeRes = computeUniformResources().find((r) => r.name === slot);
    if (computeRes && computeUniformScratch && computeUniformBuffer) {
      writeUniformMember(computeUniformScratch, resourcePlacement(computeRes), value);
      device.queue.writeBuffer(computeUniformBuffer, 0, computeUniformScratch.f32 as BufferSource);
      return;
    }

    // A uniform the program doesn't read is ignored, as in TSL and every other adapter.
    if (!computePipeline) pendingUniforms.set(slot, value);
  }

  // Named `setAttribute` by convention with the render-shaped adapters,
  // but it's a storage() slot's current value here — read_write, so the
  // same slot comes back out of `compute`'s result.
  function setAttribute<T extends ShaderType>(attribute: AttributeNode<T>, data: TypedArray): void;
  function setAttribute(slot: string, data: TypedArray): void;
  function setAttribute(attribute: AttributeNode<ShaderType> | string, data: TypedArray): void {
    const slot = slotOf(attribute);
    if (!device) {
      pendingAttributes.set(slot, data);
      return;
    }

    const resource = computeStorageResources().find((r) => r.name === slot);
    if (resource) {
      const elements = Math.ceil(data.length / componentCountOf(resource.shaderType));
      const state = sizeStorageSlot(resource, elements);
      // The dispatch covers the first buffer the host passed, in elements:
      // TSL's caller writes that count beside `instancedArray(count, type)`.
      if (!firstStorageSeen) {
        firstStorageSeen = true;
        firstStorageCount = Math.floor(data.length / componentCountOf(resource.shaderType));
      }
      const { slot: first, data: slots } = spread(data, 0, state.layout);
      device.queue.writeBuffer(state.buffer, first * 4, slots as BufferSource);
      return;
    }

    if (!computePipeline) {
      pendingAttributes.set(slot, data);
      return;
    }
    throw new Error(`[RMSL] unknown storage slot "${slot}"`);
  }

  let adapter: WgslComputeAdapter = {
    async attach() {
      let gpuAdapter = await navigator.gpu?.requestAdapter();
      if (!gpuAdapter) throw new Error("[RMSL] WebGPU is not available");
      // Compiled and checked before the device is requested, so a program that throws leaves no device behind.
      let program = compile({ stage: "compute", workgroupSize: options.workgroupSize ?? 64 }, compute);
      assertStorageBufferLimit(gpuAdapter.limits, program.resources, "createWgslCompute");
      device = await requestComputeDevice(gpuAdapter);
      let module = device.createShaderModule({ code: program.code });
      computePipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint: program.entryPoint },
      });
      computeResources = program.resources;

      let uniforms = computeUniformResources();
      if (uniforms.length > 0) {
        let size = uniformBufferSize(Math.max(...uniforms.map((u) => u.offset + u.size)));
        computeUniformBuffer = device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        computeUniformScratch = uniformScratch(size);
        computeBindGroup0 = device.createBindGroup({
          layout: computePipeline.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: computeUniformBuffer } }],
        });
      }

      // A slot the host never sets, such as one the program only writes, is as long as the program declared it:
      // `instancedArray(count, type)`, as TSL's buffer is. It has a buffer to bind either way.
      const declared = storageAttributes(compute);
      for (const resource of computeStorageResources())
        sizeStorageSlot(resource, declared.get(resource.name)?.count ?? 0);
      // A program with no storage binds an empty group, which its compute() still dispatches against.
      if (computeStorageResources().length === 0) {
        computeBindGroup1 = device.createBindGroup({ layout: computePipeline.getBindGroupLayout(1), entries: [] });
      }

      for (let [slot, value] of pendingUniforms) adapter.setUniform(slot, value);
      for (let [slot, data] of pendingAttributes) adapter.setAttribute(slot, data);
      pendingUniforms.clear();
      pendingAttributes.clear();
    },

    setUniform,
    setAttribute,

    async compute(out, count) {
      if (!device || !computePipeline || !computeBindGroup1) {
        throw new Error("[RMSL] createWgslCompute: attach() was never called");
      }
      let workgroupSize = options.workgroupSize ?? 64;
      let invocations = count ?? firstStorageCount;
      let encoder = device.createCommandEncoder();
      let pass = encoder.beginComputePass();
      pass.setPipeline(computePipeline);
      if (computeBindGroup0) pass.setBindGroup(0, computeBindGroup0);
      pass.setBindGroup(1, computeBindGroup1);
      // A dispatch covers whole workgroups, as WebGPU's own dispatch does, so
      // the last workgroup runs the invocations past the count too; the
      // program's own bounds check is what skips them.
      pass.dispatchWorkgroups(Math.ceil(invocations / workgroupSize));
      pass.end();
      device.queue.submit([encoder.finish()]);

      // No `out` means the caller means to keep the result GPU-resident —
      // read via `buffer(slot)` from a draw pass, never mapped back to the
      // CPU at all.
      if (!out) return;

      // Only the slots `out` names are copied and mapped: each one is a GPU
      // round trip.
      let requested = new Set(
        requestedStorageSlots(
          out,
          computeStorageResources().map((r) => r.name),
        ),
      );
      // Each call reads back through staging buffers of its own, so a call made
      // before another resolves maps none of the other's.
      let reads = computeStorageResources()
        .filter((r) => requested.has(r.name))
        .map((resource) => {
          const state = storageSlots.get(resource.name)!;
          const target = out[resource.name]!;
          const valueCount = state.elements * state.itemSize;
          // Checked before any buffer is copied, so a refusal reads nothing back.
          if (target.length < valueCount) {
            throw new RangeError(
              `[RMSL] out["${resource.name}"] holds ${target.length} values, and the slot has ${valueCount}`,
            );
          }
          return { resource, state, target, valueCount, bytes: Math.max(4, state.elements * state.layout.stride * 4) };
        });
      let readEncoder = device.createCommandEncoder();
      let stagings = reads.map(({ state, bytes }) => {
        let staging = device!.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        readEncoder.copyBufferToBuffer(state.buffer, 0, staging, 0, bytes);
        return staging;
      });
      device.queue.submit([readEncoder.finish()]);
      try {
        await Promise.all(stagings.map((staging) => staging.mapAsync(GPUMapMode.READ)));
        reads.forEach(({ resource, state, target, valueCount }, i) => {
          let range = stagings[i]!.getMappedRange();
          let values = elementView(resource.shaderType, {
            f32: new Float32Array(range),
            i32: new Int32Array(range),
            u32: new Uint32Array(range),
          });
          // The values come back out of the slots that WGSL's layout gave them, without the padding of a vec3.
          if (state.layout.identity) target.set(values.subarray(0, valueCount));
          else for (let k = 0; k < valueCount; k++) target[k] = values[state.layout.slot(k)]!;
        });
      } finally {
        for (let staging of stagings) staging.destroy();
      }
      return out;
    },

    buffer(slot) {
      return storageSlots.get(slot)?.buffer;
    },

    device() {
      return device ?? undefined;
    },

    destroy() {
      for (let state of storageSlots.values()) state.buffer.destroy();
      computeUniformBuffer?.destroy();
      device?.destroy();
    },
  };

  return adapter;
}
