import {
  isSamplerType,
  MATRIX_DIMENSIONS,
  someNode,
  type ComputeNode,
  type StorageBufferAttribute,
  type Node,
  type ShaderType,
  type UniformArrayNode,
  type UniformNode,
  type UniformValue,
} from "../../core";
import { compile, type WgslResource } from "../../wgsl";
import { slotOf, type TypedArray } from "../adapter";
import { assertWriteFits } from "../shared";
import {
  assertStorageBufferLimit,
  requestComputeDevice,
  uniformBufferSize,
  uniformScratch,
  writeUniformMember,
  type UniformScratch,
} from "./adapter-wgsl";

/**
 * Several compute programs on one `GPUDevice`, sharing their storage buffers,
 * as TSL's renderer runs `ComputeNode`s. Each {@link StorageBufferAttribute}
 * a program references gets one resident `GPUBuffer`, so programs that
 * reference the same attribute read and write the same memory.
 */
export interface WgslContext {
  /** The device every program and buffer lives on. */
  readonly device: GPUDevice;
  /**
   * Records every dispatch into one compute pass and submits it once. They run
   * in order, each seeing the writes of the ones before it. A program is
   * compiled the first time it is dispatched.
   */
  compute(nodes: ComputeNode | readonly ComputeNode[]): void;
  /**
   * Sets a uniform for every program that reads it. A value applies to every
   * dispatch in the next `compute()`, as in TSL, not to individual dispatches.
   */
  setUniform<T extends ShaderType>(uniform: UniformNode<T>, value: UniformValue<T>): void;
  setUniform<T extends ShaderType>(uniform: UniformArrayNode<T>, value: UniformValue<T>[]): void;
  /** Writes `data` into the attribute's buffer, from element `offset` on. */
  write(attribute: StorageBufferAttribute, data: TypedArray, offset?: number): void;
  /**
   * The attribute's buffer contents, copied back from the GPU, as TSL's
   * `getArrayBufferAsync()`, laid out as the attribute holds them.
   */
  getArrayBufferAsync(attribute: StorageBufferAttribute): Promise<ArrayBuffer>;
  /**
   * The attribute's `GPUBuffer`, for binding in a render pipeline on the same
   * device. It is laid out as WGSL lays out a storage array: a `vec3`, and
   * each column of three in a matrix, takes 16 bytes.
   */
  buffer(attribute: StorageBufferAttribute): GPUBuffer;
  /** Destroys every buffer, and the device if the context created it. */
  destroy(): void;
}

export interface CreateWgslContextOptions {
  /** A device to use instead of requesting one. It is not destroyed with the context. */
  device?: GPUDevice;
}

type CompiledProgram = {
  pipeline: GPUComputePipeline;
  storageGroup: GPUBindGroup | null;
  uniforms: {
    buffer: GPUBuffer;
    scratch: UniformScratch;
    group: GPUBindGroup;
    resources: Extract<WgslResource, { kind: "uniform" }>[];
    /** The uniform version and count last uploaded, so an unchanged buffer isn't written again. */
    uploaded: { version: number; count: number } | null;
  } | null;
};

/** Whether any node reachable from `root` is a texture, which the context has no binding for. */
function samplesTextures(root: Node<ShaderType>): boolean {
  return someNode(root, (node) => typeof node._t === "string" && isSamplerType(node._t));
}

/**
 * Where an attribute's values sit in its GPU buffer, in 32-bit slots. WGSL
 * gives a `vec3`, and each column of three in a matrix, the room of four, so
 * those leave one slot empty after every three values; every other element
 * type is laid out as the attribute holds it.
 */
type StorageLayout = {
  /** Slots per element. */
  stride: number;
  /** The slot holding the attribute's value `k`. */
  slot(k: number): number;
};

function storageLayout(attribute: StorageBufferAttribute): StorageLayout {
  const { itemSize, elementType } = attribute;
  const rows = (elementType && MATRIX_DIMENSIONS[elementType]?.[1]) ?? itemSize;
  if (rows !== 3) return { stride: itemSize, slot: (k) => k };
  const stride = (itemSize / 3) * 4;
  return {
    stride,
    slot: (k) => Math.floor(k / itemSize) * stride + Math.floor((k % itemSize) / 3) * 4 + (k % 3),
  };
}

/** `values`, the attribute's values from value `first` on, spread into the slots `layout` gives them. */
function spread(values: TypedArray, first: number, layout: StorageLayout): { slot: number; data: TypedArray } {
  if (values.length === 0) return { slot: 0, data: values };
  const slot = layout.slot(first);
  const data = new (values.constructor as Float32ArrayConstructor)(layout.slot(first + values.length - 1) - slot + 1);
  for (let k = 0; k < values.length; k++) data[layout.slot(first + k) - slot] = values[k]!;
  return { slot, data };
}

/**
 * Creates a {@link WgslContext} on `options.device`, or on a new device that
 * binds as many storage buffers per shader stage as the hardware can.
 */
export async function createWgslContext(options: CreateWgslContextOptions = {}): Promise<WgslContext> {
  let device = options.device;
  if (!device) {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error("[RMSL] WebGPU is not available");
    device = await requestComputeDevice(adapter);
  }
  const gpu = device;

  /** Each attribute's buffer, and the layout it was given, which a program using it has to share. */
  const buffers = new Map<StorageBufferAttribute, { buffer: GPUBuffer; layout: StorageLayout }>();
  const programs = new Map<ComputeNode, CompiledProgram>();
  const uniformValues = new Map<string, number | number[]>();
  /** Bumped by every `setUniform()`, so a program knows whether its uniform buffer is stale. */
  let uniformVersion = 0;
  /** One staging buffer per attribute for reading it back, reused unless a read is still pending. */
  const stagingBuffers = new Map<StorageBufferAttribute, GPUBuffer>();

  function resident(attribute: StorageBufferAttribute): { buffer: GPUBuffer; layout: StorageLayout } {
    let existing = buffers.get(attribute);
    if (existing) return existing;
    const layout = storageLayout(attribute);
    existing = {
      buffer: gpu.createBuffer({
        size: Math.max(4, attribute.count * layout.stride * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
      }),
      layout,
    };
    buffers.set(attribute, existing);
    if (attribute.array)
      gpu.queue.writeBuffer(existing.buffer, 0, spread(attribute.array, 0, layout).data as BufferSource);
    return existing;
  }

  function buffer(attribute: StorageBufferAttribute): GPUBuffer {
    return resident(attribute).buffer;
  }

  function program(node: ComputeNode): CompiledProgram {
    let existing = programs.get(node);
    if (existing) return existing;
    if (samplesTextures(node.computeNode)) {
      throw new Error("[RMSL] createWgslContext: programs that sample textures aren't supported yet.");
    }
    const compiled = compile({ stage: "compute" }, node);
    assertStorageBufferLimit(gpu.limits, compiled.resources, "createWgslContext");
    const pipeline = gpu.createComputePipeline({
      layout: "auto",
      compute: { module: gpu.createShaderModule({ code: compiled.code }), entryPoint: compiled.entryPoint },
    });

    const storages = compiled.resources.filter((r) => r.kind === "storage");
    for (const r of storages) {
      if (resident(r.attribute).layout.stride !== storageLayout(r.attribute).stride) {
        throw new Error(
          `[RMSL] createWgslContext: a ${r.shaderType} attribute's buffer was laid out before a storage node named its type; create the node before writing or reading the attribute`,
        );
      }
    }
    const storageGroup =
      storages.length === 0
        ? null
        : gpu.createBindGroup({
            layout: pipeline.getBindGroupLayout(1),
            entries: storages.map((r) => ({
              binding: r.binding,
              resource: { buffer: buffer(r.attribute) },
            })),
          });

    const uniformResources = compiled.resources.filter(
      (r): r is Extract<WgslResource, { kind: "uniform" }> => r.kind === "uniform",
    );
    let uniforms: CompiledProgram["uniforms"] = null;
    if (uniformResources.length > 0) {
      const size = uniformBufferSize(Math.max(...uniformResources.map((u) => u.offset + u.size)));
      const uniformBuffer = gpu.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      uniforms = {
        buffer: uniformBuffer,
        scratch: uniformScratch(size),
        group: gpu.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
        }),
        resources: uniformResources,
        uploaded: null,
      };
    }

    existing = { pipeline, storageGroup, uniforms };
    programs.set(node, existing);
    return existing;
  }

  function uploadUniforms(node: ComputeNode, compiled: CompiledProgram): void {
    const uniforms = compiled.uniforms;
    if (!uniforms || (uniforms.uploaded?.version === uniformVersion && uniforms.uploaded.count === node.count)) return;
    uniforms.uploaded = { version: uniformVersion, count: node.count };
    for (const resource of uniforms.resources) {
      const value = resource.name === node.countNode.name ? node.count : uniformValues.get(resource.name);
      if (value !== undefined) writeUniformMember(uniforms.scratch, { ...resource, type: resource.shaderType }, value);
    }
    gpu.queue.writeBuffer(uniforms.buffer, 0, uniforms.scratch.f32 as BufferSource);
  }

  return {
    device: gpu,

    compute(nodes) {
      const list = (Array.isArray(nodes) ? nodes : [nodes]).filter((node: ComputeNode) => node.count > 0);
      const { maxComputeWorkgroupsPerDimension, maxComputeWorkgroupSizeX, maxComputeInvocationsPerWorkgroup } =
        gpu.limits;
      const maxWorkgroupSize = Math.min(maxComputeWorkgroupSizeX, maxComputeInvocationsPerWorkgroup);
      // Every program is checked and compiled before anything is uploaded or
      // encoded, so one that can't run leaves the others' state untouched.
      const compiled = list.map((node) => {
        if (node.workgroupSize > maxWorkgroupSize) {
          throw new Error(
            `[RMSL] createWgslContext: a workgroup size of ${node.workgroupSize} is past the device's limit of ${maxWorkgroupSize}`,
          );
        }
        const workgroups = Math.ceil(node.count / node.workgroupSize);
        if (workgroups > maxComputeWorkgroupsPerDimension) {
          throw new Error(
            `[RMSL] createWgslContext: a count of ${node.count} needs ${workgroups} workgroups of ${node.workgroupSize}, past the device's limit of ${maxComputeWorkgroupsPerDimension}; use a larger workgroup size, up to ${maxWorkgroupSize}`,
          );
        }
        return program(node);
      });
      const encoder = gpu.createCommandEncoder();
      const pass = encoder.beginComputePass();
      list.forEach((node, i) => {
        const { pipeline, uniforms, storageGroup } = compiled[i]!;
        uploadUniforms(node, compiled[i]!);
        pass.setPipeline(pipeline);
        if (uniforms) pass.setBindGroup(0, uniforms.group);
        if (storageGroup) pass.setBindGroup(1, storageGroup);
        pass.dispatchWorkgroups(Math.ceil(node.count / node.workgroupSize));
      });
      pass.end();
      gpu.queue.submit([encoder.finish()]);
    },

    setUniform(uniform: UniformNode<ShaderType> | UniformArrayNode<ShaderType>, value: unknown) {
      // Copied, so changing the array afterwards doesn't reach a program without another setUniform().
      uniformValues.set(slotOf(uniform), structuredClone(value) as number | number[]);
      uniformVersion++;
    },

    write(attribute, data, offset = 0) {
      assertWriteFits(attribute, data.length, offset);
      // Converted to the attribute's own array type, so values arrive as numbers, not reinterpreted bits.
      const values = data instanceof attribute.arrayClass ? data : attribute.arrayClass.from(data as ArrayLike<number>);
      const { buffer, layout } = resident(attribute);
      const { slot, data: slots } = spread(values, offset * attribute.itemSize, layout);
      gpu.queue.writeBuffer(buffer, slot * 4, slots as BufferSource);
    },

    async getArrayBufferAsync(attribute) {
      // The GPUBuffer is at least 4 bytes, but only the attribute's own elements are returned.
      const { buffer: source, layout } = resident(attribute);
      const size = attribute.count * layout.stride * 4;
      if (size === 0) return new ArrayBuffer(0);
      // A read still waiting on the cached staging buffer keeps it, and this one gets a buffer of its own.
      const cached = stagingBuffers.get(attribute);
      const staging =
        cached?.mapState === "unmapped"
          ? cached
          : gpu.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      if (!cached) stagingBuffers.set(attribute, staging);
      try {
        const encoder = gpu.createCommandEncoder();
        encoder.copyBufferToBuffer(source, 0, staging, 0, size);
        gpu.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const slots = new Uint32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        if (layout.stride === attribute.itemSize) return slots.buffer;
        // Padded: gathered back into the attribute's own layout, as the WASM context returns it.
        const values = new Uint32Array(attribute.count * attribute.itemSize);
        for (let k = 0; k < values.length; k++) values[k] = slots[layout.slot(k)]!;
        return values.buffer;
      } finally {
        if (stagingBuffers.get(attribute) !== staging) staging.destroy();
      }
    },

    buffer,

    destroy() {
      for (const b of buffers.values()) b.buffer.destroy();
      for (const b of stagingBuffers.values()) b.destroy();
      for (const p of programs.values()) p.uniforms?.buffer.destroy();
      if (!options.device) gpu.destroy();
    },
  };
}
