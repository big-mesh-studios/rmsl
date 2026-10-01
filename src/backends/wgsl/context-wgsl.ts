import {
  ComputeNode,
  StorageBufferAttribute,
  type Node,
  type ShaderType,
  type UniformArrayNode,
  type UniformNode,
  type UniformValue,
} from "../../core";
import { compile, type WgslResource } from "../../wgsl";
import { slotOf, type TypedArray } from "../adapter";
import { uniformBufferSize, uniformScratch, writeUniformMember, type UniformScratch } from "./adapter-wgsl";

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
  /** The attribute's buffer contents, copied back from the GPU, as TSL's `getArrayBufferAsync()`. */
  getArrayBufferAsync(attribute: StorageBufferAttribute): Promise<ArrayBuffer>;
  /** The attribute's `GPUBuffer`, for binding in a render pipeline on the same device. */
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
  } | null;
};

/** Every storage attribute reachable from `root`, keyed by the slot name its nodes compile to. */
function storageAttributes(root: Node<ShaderType>): Map<string, StorageBufferAttribute> {
  const attributes = new Map<string, StorageBufferAttribute>();
  const visited = new Set<unknown>();
  const walk = (node: any): void => {
    if (!node || typeof node !== "object" || visited.has(node)) return;
    visited.add(node);
    if (node.type === "storage") attributes.set(node.value.slot, node.value.attribute);
    if (Array.isArray(node.params)) for (const p of node.params) walk(p);
  };
  walk(root);
  return attributes;
}

/** Bytes per element of a WGSL storage array of `itemSize` 32-bit components. */
function elementStride(itemSize: number): number {
  if (itemSize === 3 || itemSize > 4) {
    throw new Error(
      `[RMSL] createWgslContext: storage elements of ${itemSize} components aren't supported yet; use 1, 2 or 4.`,
    );
  }
  return itemSize * 4;
}

/** Creates a {@link WgslContext} on a new device, or on `options.device`. */
export async function createWgslContext(options: CreateWgslContextOptions = {}): Promise<WgslContext> {
  let device = options.device;
  if (!device) {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error("[RMSL] WebGPU is not available");
    device = await adapter.requestDevice();
  }
  const gpu = device;

  const buffers = new Map<StorageBufferAttribute, GPUBuffer>();
  const programs = new Map<ComputeNode, CompiledProgram>();
  const uniformValues = new Map<string, number | number[]>();

  function buffer(attribute: StorageBufferAttribute): GPUBuffer {
    let existing = buffers.get(attribute);
    if (existing) return existing;
    const size = Math.max(4, attribute.count * elementStride(attribute.itemSize));
    existing = gpu.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
    });
    buffers.set(attribute, existing);
    if (attribute.array) gpu.queue.writeBuffer(existing, 0, attribute.array as BufferSource);
    return existing;
  }

  function program(node: ComputeNode): CompiledProgram {
    let existing = programs.get(node);
    if (existing) return existing;
    const compiled = compile({ stage: "compute" }, node);
    const pipeline = gpu.createComputePipeline({
      layout: "auto",
      compute: { module: gpu.createShaderModule({ code: compiled.code }), entryPoint: compiled.entryPoint },
    });

    const attributes = storageAttributes(node.computeNode);
    const storages = compiled.resources.filter((r) => r.kind === "storage");
    const storageGroup =
      storages.length === 0
        ? null
        : gpu.createBindGroup({
            layout: pipeline.getBindGroupLayout(1),
            entries: storages.map((r) => ({
              binding: r.binding,
              resource: { buffer: buffer(attributes.get(r.name)!) },
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
      };
    }

    existing = { pipeline, storageGroup, uniforms };
    programs.set(node, existing);
    return existing;
  }

  function uploadUniforms(node: ComputeNode, compiled: CompiledProgram): void {
    const uniforms = compiled.uniforms;
    if (!uniforms) return;
    for (const resource of uniforms.resources) {
      const value = resource.name === node.countNode.name ? node.count : uniformValues.get(resource.name);
      if (value !== undefined) writeUniformMember(uniforms.scratch, { ...resource, type: resource.shaderType }, value);
    }
    gpu.queue.writeBuffer(uniforms.buffer, 0, uniforms.scratch.f32 as BufferSource);
  }

  return {
    device: gpu,

    compute(nodes) {
      const list = nodes instanceof ComputeNode ? [nodes] : nodes;
      const encoder = gpu.createCommandEncoder();
      const pass = encoder.beginComputePass();
      for (const node of list) {
        if (node.count <= 0) continue;
        const compiled = program(node);
        uploadUniforms(node, compiled);
        pass.setPipeline(compiled.pipeline);
        if (compiled.uniforms) pass.setBindGroup(0, compiled.uniforms.group);
        if (compiled.storageGroup) pass.setBindGroup(1, compiled.storageGroup);
        pass.dispatchWorkgroups(Math.ceil(node.count / node.workgroupSize));
      }
      pass.end();
      gpu.queue.submit([encoder.finish()]);
    },

    setUniform(uniform: UniformNode<ShaderType> | UniformArrayNode<ShaderType>, value: unknown) {
      uniformValues.set(slotOf(uniform), value as number | number[]);
    },

    write(attribute, data, offset = 0) {
      gpu.queue.writeBuffer(buffer(attribute), offset * elementStride(attribute.itemSize), data as BufferSource);
    },

    async getArrayBufferAsync(attribute) {
      const source = buffer(attribute);
      const staging = gpu.createBuffer({ size: source.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      try {
        const encoder = gpu.createCommandEncoder();
        encoder.copyBufferToBuffer(source, 0, staging, 0, source.size);
        gpu.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        return staging.getMappedRange().slice(0);
      } finally {
        staging.destroy();
      }
    },

    buffer,

    destroy() {
      for (const b of buffers.values()) b.destroy();
      for (const p of programs.values()) p.uniforms?.buffer.destroy();
      if (!options.device) gpu.destroy();
    },
  };
}
