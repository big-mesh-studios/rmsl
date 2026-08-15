/// <reference types="@webgpu/types" />
import { compileWGSL, wgslUniformLayout } from "../../rmsl";
import { Color } from "../math/Color";
import type { Scene } from "../scenes/Scene";
import type { Camera } from "../cameras/Camera";
import type { Mesh } from "../objects/Mesh";
import type { BufferGeometry } from "../geometries/BufferGeometry";
import type { Texture } from "../textures/Texture";
import { DataTexture } from "../textures/DataTexture";
import type { NodeMaterial, MaterialProgram } from "../materials/NodeMaterial";
import { Side } from "../materials/Material";
import {
  cameraUniformValue, objectUniformValue, lightsSignature, wgslTypeName, toBufferView,
} from "./common";

interface PipelineEntry {
  signature: string;
  program: MaterialProgram;
  pipeline: GPURenderPipeline;
  bindGroup: GPUBindGroup;
  /** Ring of uniform slots so per-draw writes never race the previous draw. */
  ringBuffer: GPUBuffer;
  slotSize: number;
  slots: number;
  layoutMembers: { name: string; offset: number }[];
  vertexFormats: { name: string; shaderLocation: number; format: GPUVertexFormat }[];
}

interface GeometryBuffers {
  attributes: Map<string, GPUBuffer>;
  index: GPUBuffer | null;
  indexFormat: "uint16" | "uint32" | null;
  needsUpload: boolean;
}

const UNIFORM_SLOTS = 64;

/**
 * A WebGPU renderer for `@random-mesh/rmsl/scene`, mirroring the WebGL
 * renderer: material node graphs compile to WGSL, uniform values are packed
 * into per-program ring buffers, and `render(scene, camera)` draws everything.
 */
export class WebGPURenderer {
  readonly isWebGPURenderer = true;

  canvas: HTMLCanvasElement;
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;

  private pipelines = new Map<NodeMaterial, PipelineEntry>();
  private geometryBuffers = new Map<BufferGeometry, GeometryBuffers>();
  private textures = new Map<Texture, GPUTexture>();
  private samplers = new Map<Texture, GPUSampler>();
  private depthTexture: GPUTexture | null = null;
  private depthView: GPUTextureView | null = null;
  private clearColor = new Color(0, 0, 0);
  private clearAlpha = 1;
  private animationCallback: ((time: number) => void) | null = null;
  private animationHandle: number | null = null;
  private blankTextureObject: DataTexture | null = null;

  constructor(canvas: HTMLCanvasElement, device: GPUDevice) {
    this.canvas = canvas;
    this.device = device;
    const context = canvas.getContext("webgpu");
    if (!context) throw new Error("[RMSL/scene] WebGPU context unavailable");
    this.context = context as GPUCanvasContext;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format: this.format, alphaMode: "premultiplied" });
  }

  static async init(canvas?: HTMLCanvasElement): Promise<WebGPURenderer> {
    if (!navigator.gpu) throw new Error("[RMSL/scene] WebGPU is not supported by this browser");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("[RMSL/scene] no WebGPU adapter available");
    const device = await adapter.requestDevice();
    const c = canvas ?? document.createElement("canvas");
    return new WebGPURenderer(c, device);
  }

  setClearColor(color: Color | number, alpha = 1): void {
    if (typeof color === "number") this.clearColor.setHex(color);
    else this.clearColor.copy(color);
    this.clearAlpha = alpha;
  }

  setSize(width: number, height: number): void {
    this.canvas.width = width;
    this.canvas.height = height;
  }

  setAnimationLoop(callback: ((time: number) => void) | null): void {
    this.animationCallback = callback;
    if (callback && this.animationHandle === null) {
      const loop = (now: number): void => {
        if (!this.animationCallback) {
          this.animationHandle = null;
          return;
        }
        this.animationCallback(now);
        this.animationHandle = requestAnimationFrame(loop);
      };
      this.animationHandle = requestAnimationFrame(loop);
    }
  }

  render(scene: Scene, camera: Camera): void {
    scene.updateMatrixWorld(true);
    camera.updateMatrixWorld(true);

    this.ensureDepthTexture();
    const device = this.device;

    const encoder = device.createCommandEncoder();
    const colorView = this.context.getCurrentTexture().createView();

    let slotIndex = 0;
    let firstPass = true;
    scene.traverseVisible((object) => {
      if (!object.isMesh) return;
      const mesh = object as Mesh;
      const material = mesh.material;
      if (!(material as NodeMaterial).isNodeMaterial) return;
      const entry = this.ensurePipeline(material as NodeMaterial, scene);
      if (!entry) return;

      this.packUniforms(entry, mesh, camera, slotIndex);

      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: colorView,
          clearValue: { r: this.clearColor.r, g: this.clearColor.g, b: this.clearColor.b, a: this.clearAlpha },
          loadOp: firstPass ? "clear" : "load",
          storeOp: "store",
        }],
        depthStencilAttachment: {
          view: this.depthView!,
          depthClearValue: 1.0,
          depthLoadOp: firstPass ? "clear" : "load",
          depthStoreOp: "store",
        },
      });
      firstPass = false;

      pass.setPipeline(entry.pipeline);
      pass.setBindGroup(0, entry.bindGroup, [slotIndex * entry.slotSize]);
      this.setVertexBuffers(pass, entry, mesh.geometry);

      const geometry = mesh.geometry;
      if (geometry.index) {
        const buffers = this.ensureGeometryBuffers(geometry);
        pass.setIndexBuffer(buffers.index!, buffers.indexFormat as GPUIndexFormat, 0);
        pass.drawIndexed(geometry.index.count);
      } else {
        pass.draw(geometry.attributes.position?.count ?? 0);
      }
      pass.end();

      slotIndex = (slotIndex + 1) % entry.slots;
    });

    device.queue.submit([encoder.finish()]);
  }

  private packUniforms(entry: PipelineEntry, mesh: Mesh, camera: Camera, slotIndex: number): void {
    const floats = new Float32Array(entry.slotSize / 4);
    for (const binding of entry.program.uniforms) {
      const member = entry.layoutMembers.find((m) => m.name === binding.node.name);
      if (!member) continue;
      let value: number | number[] | Float32Array;
      if (binding.scope === "camera") {
        value = cameraUniformValue(binding.name, camera);
      } else if (binding.scope === "object") {
        value = objectUniformValue(binding.name, mesh);
      } else {
        value = binding.value?.({ camera, mesh }) ?? [];
      }
      const base = member.offset / 4;
      if (typeof value === "number") {
        floats[base] = value;
      } else {
        for (let i = 0; i < value.length; i++) {
          floats[base + i] = value[i];
        }
      }
    }
    this.device.queue.writeBuffer(entry.ringBuffer, slotIndex * entry.slotSize, floats, 0, entry.slotSize / 4);
  }

  private ensurePipeline(material: NodeMaterial, scene: Scene): PipelineEntry | null {
    let entry = this.pipelines.get(material);
    const signature = lightsSignature(scene);
    if (entry && entry.signature === signature && !material.needsUpdate) {
      return entry;
    }

    const program = material.build(scene);
    const device = this.device;

    const vertexModule = device.createShaderModule({ code: compileWGSL.vertex(program.vertexRoot) });
    const fragmentModule = device.createShaderModule({ code: compileWGSL.fragment(program.fragmentRoot) });

    // The uniform struct the compiler emits, member offsets included.
    const layout = wgslUniformLayout(
      program.uniforms.map((u) => ({ slot: u.node.name, type: wgslTypeName(u.node._t) })),
    );
    const layoutMembers = layout.members.map((m) => ({ name: m.name, offset: m.offset }));

    const slotSize = Math.max(256, Math.ceil(layout.size / 256) * 256);
    const slots = UNIFORM_SLOTS;
    const ringBuffer = device.createBuffer({
      size: slotSize * slots,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // The compiler numbers textures (group 1) by alphabetical slot name and
    // samplers (group 2) in the order the graph samples them.
    const textureBindings = program.samplers
      .slice()
      .sort((a, b) => a.name.localeCompare(b.name))
      .map((s, i) => ({ name: s.name, binding: i }));
    const samplerBindings = program.samplers.map((s, i) => ({ name: s.name, binding: i }));

    const bindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = [{
      binding: 0,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
      buffer: { type: "uniform", hasDynamicOffset: true },
    }];
    for (const t of textureBindings) {
      bindGroupLayoutEntries.push({
        binding: t.binding,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: "float", viewDimension: "2d" },
      });
    }
    for (const s of samplerBindings) {
      bindGroupLayoutEntries.push({
        binding: s.binding,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: { type: "filtering" },
      });
    }

    const bindGroupLayout = device.createBindGroupLayout({ entries: bindGroupLayoutEntries });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

    const bindGroupResources: GPUBindGroupEntry[] = [{
      binding: 0,
      resource: { buffer: ringBuffer, offset: 0, size: slotSize },
    }];
    for (const t of textureBindings) {
      const sampler = program.samplers.find((s) => s.name === t.name)!;
      bindGroupResources.push({ binding: t.binding, resource: this.ensureGpuTexture(sampler.texture()).view });
    }
    for (const s of samplerBindings) {
      const sampler = program.samplers.find((x) => x.name === s.name)!;
      bindGroupResources.push({ binding: s.binding, resource: this.ensureSampler(sampler.texture()) });
    }
    const bindGroup = device.createBindGroup({ layout: bindGroupLayout, entries: bindGroupResources });

    const vertexFormats: PipelineEntry["vertexFormats"] = program.attributes.map((attribute, i) => ({
      name: attribute.name,
      shaderLocation: i,
      format: vertexFormatFromType(attribute.node._t),
    }));

    const cullMode: GPUCullMode = material.side === Side.FrontSide
      ? "back"
      : material.side === Side.BackSide ? "front" : "none";

    const pipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: vertexModule,
        entryPoint: "main",
        buffers: vertexFormats.map((v) => ({
          arrayStride: strideForFormat(v.format),
          attributes: [{ shaderLocation: v.shaderLocation, offset: 0, format: v.format }],
        })),
      },
      fragment: {
        module: fragmentModule,
        entryPoint: "main",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list", cullMode },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });

    entry = {
      signature,
      program,
      pipeline,
      bindGroup,
      ringBuffer,
      slotSize,
      slots,
      layoutMembers,
      vertexFormats,
    };
    this.pipelines.set(material, entry);
    material.needsUpdate = false;
    return entry;
  }

  private ensureGeometryBuffers(geometry: BufferGeometry): GeometryBuffers {
    let buffers = this.geometryBuffers.get(geometry);
    if (!buffers) {
      buffers = { attributes: new Map(), index: null, indexFormat: null, needsUpload: true };
      this.geometryBuffers.set(geometry, buffers);
    }
    const needsUpload = buffers.needsUpload
      || Object.values(geometry.attributes).some((a) => a.needsUpdate);
    if (!needsUpload) return buffers;

    for (const [name, attribute] of Object.entries(geometry.attributes)) {
      let buffer = buffers.attributes.get(name);
      if (!buffer) {
        buffer = this.device.createBuffer({
          size: Math.max(toBufferView(attribute.array).byteLength, 4),
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        buffers.attributes.set(name, buffer);
      }
      this.device.queue.writeBuffer(buffer, 0, toBufferView(attribute.array));
      attribute.needsUpdate = false;
    }
    if (geometry.index) {
      if (!buffers.index) {
        buffers.index = this.device.createBuffer({
          size: Math.max(toBufferView(geometry.index.array, true).byteLength, 4),
          usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
      }
      this.device.queue.writeBuffer(buffers.index, 0, toBufferView(geometry.index.array, true));
      buffers.indexFormat = (toBufferView(geometry.index.array, true) as Uint16Array | Uint32Array).BYTES_PER_ELEMENT === 2 ? "uint16" : "uint32";
    }
    buffers.needsUpload = false;
    return buffers;
  }

  private setVertexBuffers(pass: GPURenderPassEncoder, entry: PipelineEntry, geometry: BufferGeometry): void {
    const buffers = this.ensureGeometryBuffers(geometry);
    for (const format of entry.vertexFormats) {
      const buffer = buffers.attributes.get(format.name);
      if (buffer) pass.setVertexBuffer(format.shaderLocation, buffer);
    }
  }

  private ensureGpuTexture(texture: Texture | null): { view: GPUTextureView } {
    const t = texture ?? this.blankTexture();
    let gpu = this.textures.get(t);
    if (!gpu || t.needsUpdate) {
      const width = ArrayBuffer.isView(t.image) ? (t as DataTexture).width ?? 1 : 1;
      const height = ArrayBuffer.isView(t.image) ? (t as DataTexture).height ?? 1 : 1;
      if (!gpu) {
        gpu = this.device.createTexture({
          size: [width, height],
          format: "rgba8unorm",
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.textures.set(t, gpu);
      }
      if (ArrayBuffer.isView(t.image)) {
        this.device.queue.writeTexture(
          { texture: gpu },
          t.image as unknown as ArrayBufferView<ArrayBuffer>,
          { bytesPerRow: width * 4 },
          [width, height],
        );
      }
      t.needsUpdate = false;
    }
    return { view: gpu.createView() };
  }

  private blankTexture(): DataTexture {
    if (!this.blankTextureObject) {
      this.blankTextureObject = new DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
    }
    return this.blankTextureObject;
  }

  private ensureSampler(texture: Texture | null): GPUSampler {
    const t = texture ?? this.blankTexture();
    let sampler = this.samplers.get(t);
    if (!sampler) {
      sampler = this.device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
      });
      this.samplers.set(t, sampler);
    }
    return sampler;
  }

  private ensureDepthTexture(): void {
    const width = this.canvas.width;
    const height = this.canvas.height;
    if (this.depthTexture && this.depthTexture.width === width && this.depthTexture.height === height) {
      return;
    }
    this.depthTexture?.destroy();
    this.depthTexture = this.device.createTexture({
      size: [width, height],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.depthView = this.depthTexture.createView();
  }

  dispose(): void {
    for (const entry of this.pipelines.values()) entry.ringBuffer.destroy();
    for (const buffers of this.geometryBuffers.values()) {
      for (const buffer of buffers.attributes.values()) buffer.destroy();
      buffers.index?.destroy();
    }
    for (const texture of this.textures.values()) texture.destroy();
    this.depthTexture?.destroy();
    this.pipelines.clear();
    this.geometryBuffers.clear();
    this.textures.clear();
    this.samplers.clear();
    this.depthTexture = null;
    this.depthView = null;
  }
}

function vertexFormatFromType(type: string): GPUVertexFormat {
  switch (type) {
    case "float": return "float32";
    case "vec2": return "float32x2";
    case "vec3": return "float32x3";
    case "vec4": return "float32x4";
    default: return "float32x3";
  }
}

function strideForFormat(format: GPUVertexFormat): number {
  switch (format) {
    case "float32": return 4;
    case "float32x2": return 8;
    case "float32x3": return 12;
    default: return 16;
  }
}
