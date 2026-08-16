/// <reference types="@webgpu/types" />
import { compileWGSL, wgslUniformLayout } from "../../rmsl";
import { Color } from "../math/Color";
import { Vector4 } from "../math/Vector4";
import type { Scene } from "../scenes/Scene";
import type { Camera } from "../cameras/Camera";
import type { Mesh } from "../objects/Mesh";
import type { InstancedMesh } from "../objects/InstancedMesh";
import type { BufferGeometry } from "../geometries/BufferGeometry";
import type { BufferAttribute } from "../geometries/BufferAttribute";
import type { Texture } from "../textures/Texture";
import { DataTexture } from "../textures/DataTexture";
import type { NodeMaterial, MaterialProgram } from "../materials/NodeMaterial";
import { Side } from "../materials/Material";
import {
  cameraUniformValue, isIntegerSampler, objectUniformValue, lightsSignature,
  samplerDimension, samplerSampleType, wgslTypeName, toBufferView,
  rendererUniformValue, programSignature, geometryAttribute,
} from "./common";

interface PipelineEntry {
  program: MaterialProgram;
  pipeline: GPURenderPipeline;
  bindGroup: GPUBindGroup;
  /** Ring of uniform slots so per-draw writes never race the previous draw. */
  ringBuffer: GPUBuffer;
  slotSize: number;
  slots: number;
  layoutMembers: { name: string; offset: number }[];
  vertexFormats: VertexBufferLayout[];
}

/**
 * One vertex buffer slot of a render pipeline, mirroring a
 * `GPUVertexBufferLayout`. A `mat4` attribute (an `InstancedMesh`'s
 * instanceMatrix) spans four consecutive shader locations from a single
 * 64-byte-strided buffer, so it is one slot with four entries — the WGSL
 * `mat4x4<f32>` input occupies locations `n..n+3`, each fed by one
 * `float32x4` column of the record.
 */
interface VertexBufferLayout {
  name: string;
  stepMode: GPUVertexStepMode;
  arrayStride: number;
  attributes: { shaderLocation: number; offset: number; format: GPUVertexFormat }[];
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

  private pipelines = new Map<NodeMaterial, Map<string, PipelineEntry>>();
  private geometryBuffers = new Map<BufferGeometry, GeometryBuffers>();
  /**
   * Buffers for attributes that live on the object rather than the geometry —
   * an `InstancedMesh`'s `instanceMatrix`/`instanceColor`. Keyed by the
   * attribute so two instanced meshes sharing a geometry keep separate buffers.
   */
  private attributeBuffers = new Map<BufferAttribute, GPUBuffer>();
  private textures = new Map<Texture, GPUTexture>();
  private samplers = new Map<Texture, GPUSampler>();
  private depthTexture: GPUTexture | null = null;
  private depthView: GPUTextureView | null = null;
  private clearColor = new Color(0, 0, 0);
  private clearAlpha = 1;
  private animationCallback: ((time: number) => void) | null = null;
  private animationHandle: number | null = null;
  private blankTextures = new Map<string, DataTexture>();

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

  /** The drawing surface viewport: `(x, y, width, height)` in device pixels. */
  getViewport(target = new Vector4()): Vector4 {
    target.set(0, 0, this.canvas.width, this.canvas.height);
    return target;
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
      const instancing = (mesh as InstancedMesh).isInstancedMesh === true;
      const instancingColor = instancing && (mesh as InstancedMesh).instanceColor !== null;
      const entry = this.ensurePipeline(material as NodeMaterial, scene, instancing, instancingColor);
      if (!entry) return;

      // Give objects a chance to update per-draw state (line resolution, ...).
      mesh.onBeforeRender?.(this, scene, camera);

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
      this.setVertexBuffers(pass, entry, mesh);

      const geometry = mesh.geometry;
      const instanceCount = instancing ? (mesh as InstancedMesh).count : geometry.instanceCount;
      if (geometry.index) {
        const buffers = this.ensureGeometryBuffers(geometry);
        pass.setIndexBuffer(buffers.index!, buffers.indexFormat as GPUIndexFormat, 0);
        pass.drawIndexed(geometry.index.count, instanceCount);
      } else {
        pass.draw(geometry.attributes.position?.count ?? 0, instanceCount);
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
      } else if (binding.scope === "renderer") {
        value = rendererUniformValue(binding.name, this.canvas.width, this.canvas.height);
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

  private ensurePipeline(
    material: NodeMaterial,
    scene: Scene,
    instancing: boolean,
    instancingColor: boolean,
  ): PipelineEntry | null {
    const signature = programSignature(lightsSignature(scene), instancing, instancingColor);
    let bySignature = this.pipelines.get(material);
    const entry = bySignature?.get(signature);
    if (entry && !material.needsUpdate) {
      return entry;
    }

    const program = material.build(scene, { instancing, instancingColor });
    const device = this.device;

    const vertexModule = device.createShaderModule({ code: compileWGSL.vertex(program.vertexRoot) });
    const fragmentModule = device.createShaderModule({ code: compileWGSL.fragment(program.fragmentRoot) });

    // The uniform struct the compiler emits, member offsets included. The
    // compiler lays out members from its own alphabetical sort of the slots,
    // so the same sorted order must be fed to `wgslUniformLayout` here or the
    // byte offsets drift from what the WGSL struct actually declares.
    const uniforms = [...program.uniforms].sort((a, b) => a.node.name.localeCompare(b.node.name));
    const layout = wgslUniformLayout(
      uniforms.map((u) => ({ slot: u.node.name, type: wgslTypeName(u.node._t) })),
    );
    const layoutMembers = layout.members.map((m) => ({ name: m.name, offset: m.offset }));

    const slotSize = Math.max(256, Math.ceil(layout.size / 256) * 256);
    const slots = UNIFORM_SLOTS;
    const ringBuffer = device.createBuffer({
      size: slotSize * slots,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // The compiler numbers textures (group 1) by alphabetical slot name and
    // samplers (group 2) in the order the graph samples them. Integer
    // textures are read with textureLoad, which takes no sampler, so the WGSL
    // declares no companion sampler for them — the bindings must mirror that.
    const textureBindings = program.samplers
      .slice()
      .sort((a, b) => a.name.localeCompare(b.name))
      .map((s, i) => ({ name: s.name, type: s.type, binding: i }));
    const samplerBindings = program.samplers
      .filter((s) => !isIntegerSampler(s.type))
      .map((s, i) => ({ name: s.name, binding: i }));

    const bindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = [{
      binding: 0,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
      buffer: { type: "uniform", hasDynamicOffset: true },
    }];
    for (const t of textureBindings) {
      bindGroupLayoutEntries.push({
        binding: t.binding,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {
          sampleType: samplerSampleType(t.type),
          viewDimension: samplerDimension(t.type),
        },
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
      bindGroupResources.push({ binding: t.binding, resource: this.ensureGpuTexture(sampler.texture(), t.type).view });
    }
    for (const s of samplerBindings) {
      const sampler = program.samplers.find((x) => x.name === s.name)!;
      bindGroupResources.push({ binding: s.binding, resource: this.ensureSampler(sampler.texture()) });
    }
    const bindGroup = device.createBindGroup({ layout: bindGroupLayout, entries: bindGroupResources });

    // The vertex buffer layout: one slot per shader attribute, in the same
    // order the WGSL `VertexInput` struct numbers its `@location`s (and with
    // the same spacing — a mat4 attribute spans four consecutive locations).
    // The compiler and this allocation must agree, or the pipeline layout
    // points at locations the shader put something else on.
    const vertexFormats: PipelineEntry["vertexFormats"] = [];
    let shaderLocation = 0;
    for (const attribute of program.attributes) {
      if (attribute.node._t === "mat4") {
        // An InstancedMesh's instanceMatrix is a 64-byte record per instance;
        // its four columns feed locations n..n+3 as float32x4.
        const columns: VertexBufferLayout["attributes"] = [];
        for (let i = 0; i < 4; i++) {
          columns.push({ shaderLocation: shaderLocation + i, offset: i * 16, format: "float32x4" });
        }
        vertexFormats.push({ name: attribute.name, stepMode: attribute.stepMode, arrayStride: 64, attributes: columns });
        shaderLocation += 4;
      } else {
        const format = vertexFormatFromType(attribute.node._t);
        vertexFormats.push({
          name: attribute.name,
          stepMode: attribute.stepMode,
          arrayStride: strideForFormat(format),
          attributes: [{ shaderLocation, offset: 0, format }],
        });
        shaderLocation += 1;
      }
    }

    const cullMode: GPUCullMode = material.side === Side.FrontSide
      ? "back"
      : material.side === Side.BackSide ? "front" : "none";

    const pipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: vertexModule,
        entryPoint: "main",
        buffers: vertexFormats.map((v) => ({
          arrayStride: v.arrayStride,
          stepMode: v.stepMode,
          attributes: v.attributes,
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

    const built: PipelineEntry = {
      program,
      pipeline,
      bindGroup,
      ringBuffer,
      slotSize,
      slots,
      layoutMembers,
      vertexFormats,
    };
    if (!bySignature) {
      bySignature = new Map();
      this.pipelines.set(material, bySignature);
    }
    bySignature.set(signature, built);
    material.needsUpdate = false;
    return built;
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

  private setVertexBuffers(pass: GPURenderPassEncoder, entry: PipelineEntry, mesh: Mesh): void {
    const buffers = this.ensureGeometryBuffers(mesh.geometry);
    // Each `vertexFormats` entry is one vertex buffer slot, so the buffer is
    // bound at its slot index (the loop position) rather than any shader
    // location — a mat4 entry spans several locations from a single buffer.
    for (let slot = 0; slot < entry.vertexFormats.length; slot++) {
      const layout = entry.vertexFormats[slot];
      const buffer = this.attributeBuffer(mesh, layout.name, buffers);
      if (buffer) pass.setVertexBuffer(slot, buffer);
    }
  }

  /**
   * The GPU buffer a shader attribute reads, uploading it when first created
   * or when the attribute asks for an update. Geometry attributes come from
   * the per-geometry cache; an `InstancedMesh`'s `instanceMatrix`/
   * `instanceColor` live on the object, so those use a per-attribute cache.
   */
  private attributeBuffer(mesh: Mesh, name: string, buffers: GeometryBuffers): GPUBuffer | null {
    const attr = geometryAttribute(mesh, mesh.geometry, name);
    if (!attr) return null;
    if (mesh.geometry.attributes[name] !== undefined) {
      return buffers.attributes.get(name) ?? null;
    }
    let buffer = this.attributeBuffers.get(attr);
    const isNew = buffer === undefined;
    if (!buffer) {
      buffer = this.device.createBuffer({
        size: Math.max(toBufferView(attr.array).byteLength, 4),
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      this.attributeBuffers.set(attr, buffer);
    }
    if (isNew || attr.needsUpdate) {
      this.device.queue.writeBuffer(buffer, 0, toBufferView(attr.array));
      attr.needsUpdate = false;
    }
    return buffer;
  }

  private ensureGpuTexture(texture: Texture | null, samplerType: string): { view: GPUTextureView } {
    const t = texture ?? this.blankTexture(samplerType);
    const integer = isIntegerSampler(samplerType);
    const dimension = samplerDimension(samplerType);
    let gpu = this.textures.get(t);
    if (!gpu || t.needsUpdate) {
      const width = ArrayBuffer.isView(t.image) ? (t as DataTexture).width ?? 1 : 1;
      const height = ArrayBuffer.isView(t.image) ? (t as DataTexture).height ?? 1 : 1;
      const depth = dimension === "3d" ? (t as DataTexture).depth ?? 1 : 1;
      const format = integer ? integerGpuFormat(samplerType, ArrayBuffer.isView(t.image) ? t.image : null) : "rgba8unorm";
      if (!gpu) {
        gpu = this.device.createTexture({
          size: [width, height, depth],
          format,
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.textures.set(t, gpu);
      }
      if (ArrayBuffer.isView(t.image)) {
        this.writeTexture(gpu, t.image as unknown as ArrayBufferView<ArrayBuffer>, width, height, depth, format);
      }
      t.needsUpdate = false;
    }
    return { view: gpu.createView() };
  }

  /**
   * Copy texture data to the GPU. A 3D write needs a 256-aligned row stride —
   * the natural `width * bytesPerTexel` rarely is one — so the data is
   * repacked into a padded buffer first.
   */
  private writeTexture(
    texture: GPUTexture,
    image: ArrayBufferView<ArrayBuffer>,
    width: number,
    height: number,
    depth: number,
    format: GPUTextureFormat,
  ): void {
    const bytesPerTexel = format === "rgba32uint" || format === "rgba32sint" ? 16
      : format === "rgba16uint" || format === "rgba16sint" ? 8
      : 4;
    const bytesPerRow = width * bytesPerTexel;
    if (depth === 1) {
      this.device.queue.writeTexture(
        { texture },
        image,
        { bytesPerRow },
        [width, height, 1],
      );
      return;
    }
    const paddedBytesPerRow = Math.ceil(bytesPerRow / 256) * 256;
    const padded = new Uint8Array(paddedBytesPerRow * height * depth);
    const src = new Uint8Array(image.buffer, image.byteOffset, image.byteLength);
    for (let z = 0; z < depth; z++) {
      for (let y = 0; y < height; y++) {
        const row = (z * height + y) * bytesPerRow;
        padded.set(src.subarray(row, row + bytesPerRow), (z * height + y) * paddedBytesPerRow);
      }
    }
    this.device.queue.writeTexture(
      { texture },
      padded,
      { bytesPerRow: paddedBytesPerRow, rowsPerImage: height },
      [width, height, depth],
    );
  }

  /**
   * A 1×1 fallback texture, created per format and dimension so an integer or
   * 3D sampler with no texture still gets a valid resource.
   */
  private blankTexture(samplerType = "sampler2D"): DataTexture {
    const key = `${samplerDimension(samplerType)}:${samplerSampleType(samplerType)}`;
    let blank = this.blankTextures.get(key);
    if (!blank) {
      const sampleType = samplerSampleType(samplerType);
      const data = sampleType === "sint" ? new Int8Array([0, 0, 0, -1]) : new Uint8Array([0, 0, 0, 255]);
      blank = new DataTexture(data, 1, 1, 1);
      this.blankTextures.set(key, blank);
    }
    return blank;
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
    for (const bySignature of this.pipelines.values()) {
      for (const entry of bySignature.values()) entry.ringBuffer.destroy();
    }
    for (const buffers of this.geometryBuffers.values()) {
      for (const buffer of buffers.attributes.values()) buffer.destroy();
      buffers.index?.destroy();
    }
    for (const buffer of this.attributeBuffers.values()) buffer.destroy();
    for (const texture of this.textures.values()) texture.destroy();
    this.depthTexture?.destroy();
    this.pipelines.clear();
    this.geometryBuffers.clear();
    this.attributeBuffers.clear();
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

/**
 * The WebGPU format for an integer RGBA texture, from the bit depth of its
 * data view and the sampler's signedness.
 */
function integerGpuFormat(samplerType: string, view: ArrayBufferView | null): GPUTextureFormat {
  const signed = samplerType.startsWith("isampler");
  const bytes = (view as { BYTES_PER_ELEMENT?: number } | null)?.BYTES_PER_ELEMENT ?? 1;
  if (signed) {
    if (bytes === 1) return "rgba8sint";
    if (bytes === 2) return "rgba16sint";
    return "rgba32sint";
  }
  if (bytes === 1) return "rgba8uint";
  if (bytes === 2) return "rgba16uint";
  return "rgba32uint";
}
