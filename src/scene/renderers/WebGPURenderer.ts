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
import type { SamplerShaderType } from "../materials/nodes/Builder";
import { Side } from "../materials/Material";
import {
  cameraUniformValue,
  isIntegerSampler,
  objectUniformValue,
  lightsSignature,
  samplerDimension,
  samplerSampleType,
  wgslTypeName,
  toBufferView,
  rendererUniformValue,
  programSignature,
  geometryAttribute,
  samplerState,
  textureChannels,
  type SamplerState,
  type TextureWrap,
} from "./common";

interface PipelineEntry {
  program: MaterialProgram;
  pipeline: GPURenderPipeline;
  /** The uniform buffer's group, which nothing invalidates. */
  bindGroup: GPUBindGroup;
  /**
   * The texture and sampler groups, in the groups the compiled WGSL declares
   * them in. Null once what they hold has gone — a disposed texture, one
   * re-created at a new size, one that changed how it is sampled — so
   * `ensurePipeline` builds them again before the next draw. Null too when the
   * program samples nothing.
   */
  textureBindGroup: GPUBindGroup | null;
  samplerBindGroup: GPUBindGroup | null;
  bindGroupLayouts: {
    uniforms: GPUBindGroupLayout;
    textures: GPUBindGroupLayout | null;
    samplers: GPUBindGroupLayout | null;
  };
  /** One entry per texture binding, and per sampler binding, of those groups. */
  textureBindings: { name: string; type: SamplerShaderType; binding: number }[];
  samplerBindings: { name: string; binding: number }[];
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
  /**
   * Samplers by the state they were made for, not by texture: a sampler holds
   * no image, so every texture filtered and wrapped the same way shares one.
   */
  private samplers = new Map<string, GPUSampler>();
  /** The sampler state each texture was last bound with, to notice a change. */
  private samplerKeys = new Map<Texture, string>();
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
        colorAttachments: [
          {
            view: colorView,
            clearValue: { r: this.clearColor.r, g: this.clearColor.g, b: this.clearColor.b, a: this.clearAlpha },
            loadOp: firstPass ? "clear" : "load",
            storeOp: "store",
          },
        ],
        depthStencilAttachment: {
          view: this.depthView!,
          depthClearValue: 1.0,
          depthLoadOp: firstPass ? "clear" : "load",
          depthStoreOp: "store",
        },
      });
      firstPass = false;

      pass.setPipeline(entry.pipeline);
      // The compiler puts the uniform struct in group 0, textures in group 1
      // and samplers in group 2, so a draw sets one group per kind.
      pass.setBindGroup(0, entry.bindGroup, [slotIndex * entry.slotSize]);
      if (entry.textureBindGroup) pass.setBindGroup(1, entry.textureBindGroup);
      if (entry.samplerBindGroup) pass.setBindGroup(2, entry.samplerBindGroup);
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
      this.refreshTextures(entry);
      // A texture disposed since the last draw took this entry's texture and
      // sampler groups with it, and so does one re-created at a new size;
      // build them again from the textures the material points at now.
      this.bindTextures(entry);
      return entry;
    }

    const program = material.build(scene, { instancing, instancingColor });
    const device = this.device;

    // The uniform struct the compiler emits, member offsets included. The
    // compiler lays out members from its own alphabetical sort of the slots,
    // so the same sorted order must be fed to `wgslUniformLayout` here or the
    // byte offsets drift from what the WGSL struct actually declares.
    //
    // The same list goes to both stages. Left to itself each stage declares
    // only the uniforms it reads, which are not the same two sets — a material
    // colour is read by the fragment stage alone, the matrices by the vertex
    // stage alone — so the one buffer bound to both would mean something
    // different in each, and different again from what is packed here.
    const uniforms = [...program.uniforms].sort((a, b) => a.node.name.localeCompare(b.node.name));
    const declaredUniforms = uniforms.map((u) => ({
      slot: u.node.name,
      type: wgslTypeName(u.node._t),
    }));
    const layout = wgslUniformLayout(declaredUniforms);

    const vertexModule = device.createShaderModule({
      code: compileWGSL.vertex(program.vertexRoot, { uniforms: declaredUniforms }),
    });
    const fragmentModule = device.createShaderModule({
      code: compileWGSL.fragment(program.fragmentRoot, { uniforms: declaredUniforms }),
    });
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

    // One layout per group, in the groups the WGSL declares: the uniform
    // buffer alone in group 0, the textures in group 1, the samplers in group
    // 2. Numbering them all into one group would leave several bindings
    // claiming binding 0 and a pipeline layout that reaches neither of the
    // groups the shader reads.
    const uniformLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform", hasDynamicOffset: true },
        },
      ],
    });
    const textureLayout =
      textureBindings.length === 0
        ? null
        : device.createBindGroupLayout({
            entries: textureBindings.map((t) => ({
              binding: t.binding,
              visibility: GPUShaderStage.FRAGMENT,
              texture: {
                sampleType: samplerSampleType(t.type),
                viewDimension: samplerDimension(t.type),
              },
            })),
          });
    const samplerLayout =
      samplerBindings.length === 0
        ? null
        : device.createBindGroupLayout({
            entries: samplerBindings.map((s) => ({
              binding: s.binding,
              visibility: GPUShaderStage.FRAGMENT,
              sampler: { type: "filtering" },
            })),
          });
    const groupLayouts = [uniformLayout];
    if (textureLayout) groupLayouts.push(textureLayout);
    if (samplerLayout) groupLayouts.push(samplerLayout);
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: groupLayouts });

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
        vertexFormats.push({
          name: attribute.name,
          stepMode: attribute.stepMode,
          arrayStride: 64,
          attributes: columns,
        });
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

    const cullMode: GPUCullMode =
      material.side === Side.FrontSide ? "back" : material.side === Side.BackSide ? "front" : "none";

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
      bindGroup: device.createBindGroup({
        layout: uniformLayout,
        entries: [{ binding: 0, resource: { buffer: ringBuffer, offset: 0, size: slotSize } }],
      }),
      textureBindGroup: null,
      samplerBindGroup: null,
      bindGroupLayouts: { uniforms: uniformLayout, textures: textureLayout, samplers: samplerLayout },
      textureBindings,
      samplerBindings,
      ringBuffer,
      slotSize,
      slots,
      layoutMembers,
      vertexFormats,
    };
    this.bindTextures(built);
    if (!bySignature) {
      bySignature = new Map();
      this.pipelines.set(material, bySignature);
    }
    bySignature.set(signature, built);
    material.needsUpdate = false;
    return built;
  }

  /**
   * Fill in this entry's texture and sampler groups, if it is missing them: a
   * view per texture binding, and a sampler per filterable one, in the
   * numbering the compiled WGSL declares.
   *
   * They are built apart from the pipeline because a texture can be disposed
   * under a pipeline that outlives it, and a group — not the pipeline — is what
   * holds the view of the texture that went away.
   */
  private bindTextures(entry: PipelineEntry): void {
    const { textures, samplers } = entry.bindGroupLayouts;
    if (textures && !entry.textureBindGroup) {
      entry.textureBindGroup = this.device.createBindGroup({
        layout: textures,
        entries: entry.textureBindings.map((t) => ({
          binding: t.binding,
          resource: this.ensureGpuTexture(this.samplerBinding(entry, t.name).texture(), t.type).createView(),
        })),
      });
    }
    if (samplers && !entry.samplerBindGroup) {
      entry.samplerBindGroup = this.device.createBindGroup({
        layout: samplers,
        entries: entry.samplerBindings.map((s) => {
          const binding = this.samplerBinding(entry, s.name);
          return { binding: s.binding, resource: this.ensureSampler(binding.texture(), binding.type) };
        }),
      });
    }
  }

  /** The program's sampler of that name — what a binding number stands for. */
  private samplerBinding(entry: PipelineEntry, name: string) {
    return entry.program.samplers.find((s) => s.name === name)!;
  }

  /**
   * Upload again the textures of this entry whose `needsUpdate` is set, so a
   * texture whose image changed reaches the GPU on the next draw.
   *
   * A bind group binds the *texture*, not its contents, so an image rewritten
   * at the same size needs nothing else. One that changed size or format is a
   * new GPU texture, and `ensureGpuTexture` drops the bind groups that named
   * the old one.
   */
  private refreshTextures(entry: PipelineEntry): void {
    for (const t of entry.textureBindings) {
      const texture = entry.program.samplers.find((s) => s.name === t.name)!.texture();
      if (!texture?.needsUpdate) continue;
      // Filtering or wrapping changed with it means a different sampler, and
      // this bind group holds the old one.
      const key = samplerKey(samplerState(texture, t.type));
      if (this.samplerKeys.has(texture) && this.samplerKeys.get(texture) !== key) {
        this.invalidateBindGroups(texture);
      }
      this.ensureGpuTexture(texture, t.type);
    }
  }

  /**
   * Destroy the GPU texture a disposed `Texture` owns, drop its sampler, and
   * stop listening to it. Every bind group that binds it is dropped too, since
   * a bind group holding a view of a destroyed texture cannot be drawn with;
   * `ensurePipeline` builds a replacement, which re-uploads the image if the
   * material still points at the texture.
   */
  private onTextureDispose = (event: unknown): void => {
    const texture = (event as { target: Texture }).target;
    this.textures.get(texture)?.destroy();
    this.textures.delete(texture);
    // The sampler is shared with every other texture filtered and wrapped the
    // same way, so it stays; only this texture's claim on one goes.
    this.samplerKeys.delete(texture);
    texture.removeEventListener("dispose", this.onTextureDispose);
    this.invalidateBindGroups(texture);
  };

  /**
   * Destroy the vertex and index buffers a disposed `BufferGeometry` owns, and
   * stop listening to it. Drawing with the geometry again is allowed:
   * `ensureGeometryBuffers` finds no buffers for it and uploads its attributes
   * into new ones. No bind group names a vertex buffer, so none is invalidated.
   */
  private onGeometryDispose = (event: unknown): void => {
    const geometry = (event as { target: BufferGeometry }).target;
    const buffers = this.geometryBuffers.get(geometry);
    if (buffers) {
      for (const buffer of buffers.attributes.values()) buffer.destroy();
      buffers.index?.destroy();
    }
    this.geometryBuffers.delete(geometry);
    geometry.removeEventListener("dispose", this.onGeometryDispose);
  };

  /**
   * Drop the bind group of every cached pipeline that binds this texture, so
   * the next `ensurePipeline` builds one that names whatever GPU texture the
   * `Texture` has now — or none at all, if it was disposed.
   */
  private invalidateBindGroups(texture: Texture): void {
    for (const bySignature of this.pipelines.values()) {
      for (const entry of bySignature.values()) {
        if (!entry.program.samplers.some((s) => s.texture() === texture)) continue;
        entry.textureBindGroup = null;
        entry.samplerBindGroup = null;
      }
    }
  }

  private ensureGeometryBuffers(geometry: BufferGeometry): GeometryBuffers {
    let buffers = this.geometryBuffers.get(geometry);
    if (!buffers) {
      buffers = { attributes: new Map(), index: null, indexFormat: null, needsUpload: true };
      this.geometryBuffers.set(geometry, buffers);
      geometry.addEventListener("dispose", this.onGeometryDispose);
    }
    const needsUpload = buffers.needsUpload || Object.values(geometry.attributes).some((a) => a.needsUpdate);
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
      buffers.indexFormat =
        (toBufferView(geometry.index.array, true) as Uint16Array | Uint32Array).BYTES_PER_ELEMENT === 2
          ? "uint16"
          : "uint32";
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

  /**
   * The GPU texture holding this `Texture`'s image, created on first use and
   * written again whenever `needsUpdate` says the image changed.
   */
  private ensureGpuTexture(texture: Texture | null, samplerType: string): GPUTexture {
    const t = texture ?? this.blankTexture(samplerType);
    const integer = isIntegerSampler(samplerType);
    const dimension = samplerDimension(samplerType);
    let gpu = this.textures.get(t);
    if (!gpu || t.needsUpdate) {
      const width = ArrayBuffer.isView(t.image) ? ((t as DataTexture).width ?? 1) : 1;
      const height = ArrayBuffer.isView(t.image) ? ((t as DataTexture).height ?? 1) : 1;
      const depth = dimension === "3d" ? ((t as DataTexture).depth ?? 1) : 1;
      const format = integer
        ? textureChannels(t) === 1
          ? samplerType.startsWith("isampler")
            ? "r8sint"
            : "r8uint"
          : integerGpuFormat(samplerType, ArrayBuffer.isView(t.image) ? t.image : null)
        : "rgba8unorm";
      // A WebGPU texture's size and format are fixed when it is created, so an
      // image that changed shape cannot be written into the texture it had
      // before: that one is destroyed and replaced. Whatever bound it has to
      // be rebound, since a bind group names a texture that no longer exists.
      if (
        gpu &&
        (gpu.width !== width || gpu.height !== height || gpu.depthOrArrayLayers !== depth || gpu.format !== format)
      ) {
        gpu.destroy();
        this.textures.delete(t);
        this.invalidateBindGroups(t);
        gpu = undefined;
      }
      if (!gpu) {
        gpu = this.device.createTexture({
          size: [width, height, depth],
          format,
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.textures.set(t, gpu);
        t.addEventListener("dispose", this.onTextureDispose);
      }
      if (ArrayBuffer.isView(t.image)) {
        this.writeTexture(gpu, t.image as unknown as ArrayBufferView<ArrayBuffer>, width, height, depth, format);
      }
      t.needsUpdate = false;
    }
    return gpu;
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
    const bytesPerTexel =
      format === "rgba32uint" || format === "rgba32sint"
        ? 16
        : format === "rgba16uint" || format === "rgba16sint"
          ? 8
          : format === "r8uint" || format === "r8sint"
            ? 1
            : 4;
    const bytesPerRow = width * bytesPerTexel;
    if (depth === 1) {
      this.device.queue.writeTexture({ texture }, image, { bytesPerRow }, [width, height, 1]);
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
    this.device.queue.writeTexture({ texture }, padded, { bytesPerRow: paddedBytesPerRow, rowsPerImage: height }, [
      width,
      height,
      depth,
    ]);
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

  /** The sampler that reads this texture the way the texture asks to be read. */
  private ensureSampler(texture: Texture | null, samplerType = "sampler2D"): GPUSampler {
    const t = texture ?? this.blankTexture();
    const state = samplerState(t, samplerType);
    const key = samplerKey(state);
    this.samplerKeys.set(t, key);
    let sampler = this.samplers.get(key);
    if (!sampler) {
      sampler = this.device.createSampler({
        magFilter: state.magFilter,
        minFilter: state.minFilter,
        addressModeU: gpuAddressMode(state.wrapS),
        addressModeV: gpuAddressMode(state.wrapT),
        addressModeW: gpuAddressMode(state.wrapR),
      });
      this.samplers.set(key, sampler);
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
    for (const [geometry, buffers] of this.geometryBuffers) {
      for (const buffer of buffers.attributes.values()) buffer.destroy();
      buffers.index?.destroy();
      geometry.removeEventListener("dispose", this.onGeometryDispose);
    }
    for (const buffer of this.attributeBuffers.values()) buffer.destroy();
    for (const [texture, gpu] of this.textures) {
      gpu.destroy();
      texture.removeEventListener("dispose", this.onTextureDispose);
    }
    this.depthTexture?.destroy();
    this.pipelines.clear();
    this.geometryBuffers.clear();
    this.attributeBuffers.clear();
    this.textures.clear();
    this.samplers.clear();
    this.samplerKeys.clear();
    this.depthTexture = null;
    this.depthView = null;
  }
}

/** A wrapping mode as the sampler descriptor's spelling of it. */
function gpuAddressMode(wrap: TextureWrap): GPUAddressMode {
  switch (wrap) {
    case "repeat":
      return "repeat";
    case "mirror":
      return "mirror-repeat";
    default:
      return "clamp-to-edge";
  }
}

/** One sampler state as a string, so two of them can share a sampler. */
function samplerKey(state: SamplerState): string {
  return `${state.magFilter}|${state.minFilter}|${state.wrapS}|${state.wrapT}|${state.wrapR}`;
}

function vertexFormatFromType(type: string): GPUVertexFormat {
  switch (type) {
    case "float":
      return "float32";
    case "vec2":
      return "float32x2";
    case "vec3":
      return "float32x3";
    case "vec4":
      return "float32x4";
    default:
      return "float32x3";
  }
}

function strideForFormat(format: GPUVertexFormat): number {
  switch (format) {
    case "float32":
      return 4;
    case "float32x2":
      return 8;
    case "float32x3":
      return 12;
    default:
      return 16;
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
