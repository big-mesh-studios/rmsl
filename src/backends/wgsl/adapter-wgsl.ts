import { AttributeNode, Node, ShaderType, UniformArrayNode, UniformNode, UniformValue } from "../../core";
import { compile, WgslResource } from "../../wgsl";
import { Adapter, DrawCountOptions, slotOf, TypedArray } from "../adapter";
import { CompileCtx, VertexRoot } from "../shared";
import { compileWGSLStage, compileWGSLWithStage, wgslMatrixColumns, wgslUniformLayout } from "./wgsl";

/** One typed array per storage slot, keyed by name — read_write, so the same
 * record a caller passes into `compute` is the one read back out of. */
export type AdapterResult = Record<string, TypedArray>;

/**
 * Unlike GL's `drawArrays`, WebGPU bakes primitive topology into the
 * pipeline (`createWgsl`'s own `topology` option), not the draw call —
 * so there's no `mode` here, just `DrawCountOptions`'s own `count`/`first`
 * plus how many instances.
 */
export interface WgslDrawOptions extends DrawCountOptions {
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
}

/** A WGSL adapter, plus the one thing the shared `Adapter` shape has no
 * generic name for: direct access to a vertex attribute's persistent GPU
 * buffer, so another adapter's `compute` pass sharing this one's device
 * could bind it without a readback ever happening. */
export interface WgslAdapter extends Adapter<AdapterResult, WgslDrawOptions> {
  // Narrower than the base Adapter's `void | Promise<void>` — requesting a
  // GPUAdapter/GPUDevice is always async, unlike GL's attach.
  attach(canvas?: HTMLCanvasElement): Promise<void>;
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
    derivatives: "throw",
    reentrant: false,
    jsNeedsRes: false,
  };
}

type ReflectedAttribute = { slot: string; type: string };
type ReflectedUniform = { slot: string; type: string; length?: number };

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
      .sort((a, b) => a[0] - b[0])
      .map(([, info]) => ({ slot: info.slot, type: info.type })),
    uniforms: [...ctx.uniforms.values()]
      .sort((a, b) => a.slot.localeCompare(b.slot))
      .map((u) => ({ slot: u.slot, type: u.type, length: u.length })),
  };
}

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

function writeUniformScratch(scratch: Float32Array, offset: number, value: number | number[]): void {
  if (Array.isArray(value)) value.forEach((v, i) => (scratch[offset + i] = v));
  else scratch[offset] = value;
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
  let renderUniformScratch: Float32Array | null = null;
  let renderBindGroup0: GPUBindGroup | null = null;
  let vertexBuffers = new Map<string, { buffer: GPUBuffer; componentCount: number }>();
  let vertexCount = 0;

  let pendingUniforms = new Map<string, number | number[]>();
  let pendingAttributes = new Map<string, TypedArray>();

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
      writeUniformScratch(renderUniformScratch, renderMember.offset / 4, value);
      device.queue.writeBuffer(renderUniformBuffer, 0, renderUniformScratch as BufferSource);
      return;
    }

    if (!renderPipeline) {
      pendingUniforms.set(slot, value);
      return;
    }
    throw new Error(`[RMSL] unknown uniform "${slot}"`);
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
      vertexCount = Math.max(vertexCount, Math.floor(data.length / componentCount));
      return;
    }

    if (!renderPipeline) {
      pendingAttributes.set(slot, data);
      return;
    }
    throw new Error(`[RMSL] unknown attribute "${slot}"`);
  }

  let adapter: WgslAdapter = {
    async attach(canvas) {
      let gpuAdapter = await navigator.gpu?.requestAdapter();
      if (!gpuAdapter) throw new Error("[RMSL] WebGPU is not available");
      device = await gpuAdapter.requestDevice();

      let target = canvas ?? document.createElement("canvas");
      let glCanvasContext = target.getContext("webgpu");
      if (!glCanvasContext) throw new Error("[RMSL] WebGPU canvas context unavailable");
      context = glCanvasContext;
      let format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode: "opaque" });

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
      for (let u of [...vertexReflection.uniforms, ...fragmentReflection.uniforms]) sharedUniforms.set(u.slot, u);
      let renderUniforms = [...sharedUniforms.values()];
      let uniformOption = renderUniforms.length > 0 ? { uniforms: renderUniforms } : undefined;

      let vertexCode = compileWGSLWithStage(options.vertex as Node<ShaderType>, "vertex", uniformOption);
      let fragmentCode = compileWGSLWithStage(options.fragment, "fragment", uniformOption);
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
        renderUniformLayout = wgslUniformLayout(renderUniforms);
        renderUniformBuffer = device.createBuffer({
          size: Math.max(16, renderUniformLayout.size),
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        renderUniformScratch = new Float32Array(renderUniformBuffer.size / 4);
        renderBindGroup0 = device.createBindGroup({
          layout: renderPipeline.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: renderUniformBuffer } }],
        });
      }

      for (let [slot, value] of pendingUniforms) adapter.setUniform(slot, value);
      for (let [slot, data] of pendingAttributes) adapter.setAttribute(slot, data);
      pendingUniforms.clear();
      pendingAttributes.clear();
    },

    setUniform,
    setAttribute,

    draw(drawOptions) {
      if (!device || !renderPipeline || !context) {
        throw new Error("[RMSL] createWgsl: attach() was never called");
      }
      let encoder = device.createCommandEncoder();
      let view = context.getCurrentTexture().createView();
      let pass = encoder.beginRenderPass({
        colorAttachments: [{ view, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
      });
      pass.setPipeline(renderPipeline);
      if (renderBindGroup0) pass.setBindGroup(0, renderBindGroup0);
      for (let i = 0; i < vertexAttributes.length; i++) {
        let vb = vertexBuffers.get(vertexAttributes[i].slot);
        if (vb) pass.setVertexBuffer(i, vb.buffer);
      }
      pass.draw(drawOptions?.count ?? vertexCount, drawOptions?.instanceCount ?? 1, drawOptions?.first ?? 0, 0);
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
      renderUniformBuffer?.destroy();
      device?.destroy();
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
   */
  compute(out?: AdapterResult): Promise<AdapterResult | void>;
  /** A storage slot's persistent `GPUBuffer` — so a `draw` pass sharing
   * this adapter's device can bind it directly, no readback. */
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
  let n = 0;
  let storageBuffers = new Map<string, GPUBuffer>();
  let computeBindGroup1: GPUBindGroup | null = null;
  let computeUniformBuffer: GPUBuffer | null = null;
  let computeUniformScratch: Float32Array | null = null;
  let computeBindGroup0: GPUBindGroup | null = null;
  let staging: GPUBuffer | null = null;

  let pendingUniforms = new Map<string, number | number[]>();
  let pendingAttributes = new Map<string, TypedArray>();

  function computeStorageResources(): Extract<WgslResource, { kind: "storage" }>[] {
    return computeResources.filter((r): r is Extract<WgslResource, { kind: "storage" }> => r.kind === "storage");
  }
  function computeUniformResources(): Extract<WgslResource, { kind: "uniform" }>[] {
    return computeResources.filter((r): r is Extract<WgslResource, { kind: "uniform" }> => r.kind === "uniform");
  }

  function rebuildStorageBuffers() {
    if (!device || !computePipeline) return;
    for (let buf of storageBuffers.values()) buf.destroy();
    staging?.destroy();

    let usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    let bytes = Math.max(4, n * 4);
    storageBuffers = new Map(
      computeStorageResources().map((r) => [r.name, device!.createBuffer({ size: bytes, usage })]),
    );

    computeBindGroup1 = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(1),
      entries: computeStorageResources().map((r) => ({
        binding: r.binding,
        resource: { buffer: storageBuffers.get(r.name)! },
      })),
    });

    staging = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
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
      writeUniformScratch(computeUniformScratch, computeRes.offset / 4, value);
      device.queue.writeBuffer(computeUniformBuffer, 0, computeUniformScratch as BufferSource);
      return;
    }

    if (!computePipeline) {
      pendingUniforms.set(slot, value);
      return;
    }
    throw new Error(`[RMSL] unknown uniform "${slot}"`);
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

    if (computeStorageResources().some((r) => r.name === slot)) {
      if (data.length !== n) {
        n = data.length;
        rebuildStorageBuffers();
      }
      let buf = storageBuffers.get(slot);
      if (!buf) throw new Error(`[RMSL] unknown storage "${slot}"`);
      device.queue.writeBuffer(buf, 0, data as BufferSource);
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
      device = await gpuAdapter.requestDevice();

      let program = compile({ stage: "compute", workgroupSize: options.workgroupSize ?? 64 }, compute);
      let module = device.createShaderModule({ code: program.code });
      computePipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint: program.entryPoint },
      });
      computeResources = program.resources;

      let uniforms = computeUniformResources();
      if (uniforms.length > 0) {
        let size = Math.max(16, ...uniforms.map((u) => u.offset + u.size));
        computeUniformBuffer = device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        computeUniformScratch = new Float32Array(size / 4);
        computeBindGroup0 = device.createBindGroup({
          layout: computePipeline.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: computeUniformBuffer } }],
        });
      }

      // Storage buffers are sized to the first upload, so a pending value
      // has to set `n` before the (empty) default-sized buffers are built.
      for (let [, data] of pendingAttributes) if (n === 0) n = data.length;
      rebuildStorageBuffers();

      for (let [slot, value] of pendingUniforms) adapter.setUniform(slot, value);
      for (let [slot, data] of pendingAttributes) adapter.setAttribute(slot, data);
      pendingUniforms.clear();
      pendingAttributes.clear();
    },

    setUniform,
    setAttribute,

    async compute(out) {
      if (!device || !computePipeline || !computeBindGroup1 || !staging) {
        throw new Error("[RMSL] createWgslCompute: attach() was never called");
      }
      let encoder = device.createCommandEncoder();
      let pass = encoder.beginComputePass();
      pass.setPipeline(computePipeline);
      if (computeBindGroup0) pass.setBindGroup(0, computeBindGroup0);
      pass.setBindGroup(1, computeBindGroup1);
      pass.dispatchWorkgroups(Math.max(1, Math.ceil(n / (options.workgroupSize ?? 64))));
      pass.end();
      device.queue.submit([encoder.finish()]);

      // No `out` means the caller means to keep the result GPU-resident —
      // read via `buffer(slot)` from a draw pass, never mapped back to the
      // CPU at all.
      if (!out) return;

      let bytes = Math.max(4, n * 4);
      for (let [slot, buffer] of storageBuffers) {
        let readEncoder = device.createCommandEncoder();
        readEncoder.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
        device.queue.submit([readEncoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        (out[slot] as Float32Array).set(new Float32Array(staging.getMappedRange()).subarray(0, n));
        staging.unmap();
      }
      return out;
    },

    buffer(slot) {
      return storageBuffers.get(slot);
    },

    device() {
      return device ?? undefined;
    },

    destroy() {
      for (let buf of storageBuffers.values()) buf.destroy();
      computeUniformBuffer?.destroy();
      staging?.destroy();
      device?.destroy();
    },
  };

  return adapter;
}
