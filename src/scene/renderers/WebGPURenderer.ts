/// <reference types="@webgpu/types" />
import { compileWgsl, wgslUniformLayout } from "../../wgsl";
import { sharedSamplerDeclarations } from "../../backends/wgsl/wgsl";
import { uniformScratch, writeUniformMember, type UniformScratch } from "../../backends/wgsl/adapter-wgsl";
import { Color } from "../math/Color";
import { Vector4 } from "../math/Vector4";
import type { Scene } from "../scenes/Scene";
import type { Camera } from "../cameras/Camera";
import type { Mesh } from "../objects/Mesh";
import type { WebGLRenderTarget } from "./WebGLRenderTarget";
import type { Object3D } from "../core/Object3D";
import type { InstancedMesh } from "../objects/InstancedMesh";
import type { BufferGeometry } from "../geometries/BufferGeometry";
import type { BufferAttribute } from "../geometries/BufferAttribute";
import type { Texture } from "../textures/Texture";
import { DataTexture } from "../textures/DataTexture";
import type { NodeMaterial, MaterialProgram } from "../materials/NodeMaterial";
import type { SamplerShaderType } from "../materials/nodes/Builder";
import { Blending, Side } from "../materials/Material";
import {
  blankTexture,
  cameraUniformValue,
  clearColourOf,
  FrameOrder,
  isFloatTexture,
  isIntegerSampler,
  mergedUpdateRanges,
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
  vertexFormatOf,
  type VertexFormat,
  VERTEX_FORMATS,
  textureChannels,
  type SamplerState,
  type TextureWrap,
} from "./common";

interface PipelineEntry {
  program: MaterialProgram;
  /** The `version` of the material the pipeline was built from. */
  version: number;
  /** Which samplers, by bit, read a float texture the device cannot filter, as the layout was built for. */
  unfilterable: number;
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
  /** Frames in a row the ring has held more than four times the slots the frame needed. */
  oversizedFrames: number;
  layoutMembers: { name: string; type: string; offset: number; size: number; length?: number }[];
  /** Where `packUniforms` lays one draw's uniforms out before it writes them to the ring. */
  scratch: UniformScratch;
  /** What a render pipeline of this program is built from. */
  pipelineDescriptor: Omit<GPURenderPipelineDescriptor, "vertex"> & { vertexModule: GPUShaderModule };
  /** One render pipeline for each set of vertex formats the meshes drawn with this program hold. */
  variants: Map<string, PipelineVariant>;
}

/** A render pipeline and the vertex buffer layout it was built with. */
interface PipelineVariant {
  pipeline: GPURenderPipeline;
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
}

const UNIFORM_SLOTS = 64;

/** Frames a uniform ring stays oversized before it shrinks, so a scene that comes and goes does not reallocate it. */
const RING_SHRINK_FRAMES = 60;

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
  /** The draws of the frame being recorded, kept between frames so recording one allocates nothing. */
  private frameMeshes: Mesh[] = [];
  private frameEntries: PipelineEntry[] = [];
  private frameVariants: PipelineVariant[] = [];
  /** How many draws of the frame each program has, and how many of them are recorded so far. */
  private frameDrawCounts = new Map<PipelineEntry, number>();
  private frameSlots = new Map<PipelineEntry, number>();
  private geometryBuffers = new Map<BufferGeometry, GeometryBuffers>();
  /**
   * Buffers for attributes that live on the object rather than the geometry —
   * an `InstancedMesh`'s `instanceMatrix`/`instanceColor`. Keyed by the
   * attribute so two instanced meshes sharing a geometry keep separate buffers.
   */
  private attributeBuffers = new Map<BufferAttribute, GPUBuffer>();
  private textures = new Map<Texture, GPUTexture>();
  /**
   * The `version` of each texture and attribute this renderer last uploaded.
   * Each renderer keeps its own, so a change reaches every renderer that draws
   * the object, as three.js keeps it per renderer.
   */
  private uploadedVersions = new WeakMap<Texture | BufferAttribute, number>();
  /**
   * The attribute each buffer holds, and the version of it. A buffer belongs to
   * one geometry, so an attribute two geometries share uploads into each.
   */
  private heldAttributes = new WeakMap<GPUBuffer, BufferAttribute>();
  private heldVersions = new WeakMap<GPUBuffer, number>();
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
    // A float texture filters linearly where the adapter can, as in three.js; elsewhere it reads its nearest texel.
    const device = await adapter.requestDevice({
      requiredFeatures: adapter.features.has("float32-filterable") ? ["float32-filterable"] : [],
    });
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

  /**
   * Draws `scene` from `camera` into the canvas, or into `target` when one is
   * given, which `readPixels(target)` then reads back.
   */
  render(scene: Scene, camera: Camera, target: WebGLRenderTarget | null = null): void {
    scene.updateMatrixWorld(true);
    camera.updateMatrixWorld(true);

    this.ensureDepthTexture();
    const surface = target === null ? null : this.renderTargetTextures(target);
    // The surface the frame draws into, whose size a renderer-scoped uniform such as `resolution` gives.
    this.frameWidth = target?.width ?? this.canvas.width;
    this.frameHeight = target?.height ?? this.canvas.height;
    const depthView = surface === null ? this.depthView! : surface.depthView;
    const device = this.device;

    const { frameMeshes, frameEntries, frameVariants, frameDrawCounts, frameSlots } = this;
    frameMeshes.length = frameEntries.length = frameVariants.length = 0;
    frameDrawCounts.clear();
    frameSlots.clear();
    const order = this.frameOrder;
    order.clear();
    try {
      scene.traverseVisible(this.collectVisible);
      order.sort(camera);
      for (const mesh of order.meshes) {
        const material = mesh.material;
        if (!(material as NodeMaterial).isNodeMaterial) continue;
        const instancing = (mesh as InstancedMesh).isInstancedMesh === true;
        const instancingColor = instancing && (mesh as InstancedMesh).instanceColor !== null;
        const entry = this.ensurePipeline(material as NodeMaterial, scene, instancing, instancingColor);
        if (!entry) continue;
        frameMeshes.push(mesh);
        frameEntries.push(entry);
        frameVariants.push(this.pipelineVariant(entry, mesh));
        frameDrawCounts.set(entry, (frameDrawCounts.get(entry) ?? 0) + 1);
      }
      // Every draw's uniforms are written before the frame is submitted, so each draw needs a slot of its own.
      for (const bySignature of this.pipelines.values()) {
        for (const entry of bySignature.values()) this.fitRing(entry, frameDrawCounts.get(entry) ?? 0);
      }

      // After the draws are collected, so an attribute that is refused leaves no half-recorded frame.
      const encoder = device.createCommandEncoder();
      const colorView = surface === null ? this.context.getCurrentTexture().createView() : surface.colorView;
      const clear = clearColourOf(scene, this.clearColor, this.clearAlpha);
      // A frame with nothing to draw still clears, in a pass of its own.
      if (frameMeshes.length === 0) {
        encoder
          .beginRenderPass({
            colorAttachments: [{ view: colorView, clearValue: clear, loadOp: "clear", storeOp: "store" }],
            depthStencilAttachment: {
              view: depthView,
              depthClearValue: 1.0,
              depthLoadOp: "clear",
              depthStoreOp: "store",
            },
          })
          .end();
      }

      let firstPass = true;
      for (let draw = 0; draw < frameMeshes.length; draw++) {
        const mesh = frameMeshes[draw];
        const entry = frameEntries[draw];
        const variant = frameVariants[draw];
        const instancing = (mesh as InstancedMesh).isInstancedMesh === true;
        const slotIndex = frameSlots.get(entry) ?? 0;
        frameSlots.set(entry, slotIndex + 1);

        // Give objects a chance to update per-draw state (line resolution, ...).
        mesh.onBeforeRender?.(this, scene, camera);

        this.packUniforms(entry, mesh, camera, slotIndex);

        const pass = encoder.beginRenderPass({
          colorAttachments: [
            {
              view: colorView,
              clearValue: clear,
              loadOp: firstPass ? "clear" : "load",
              storeOp: "store",
            },
          ],
          depthStencilAttachment: {
            view: depthView,
            depthClearValue: 1.0,
            depthLoadOp: firstPass ? "clear" : "load",
            depthStoreOp: "store",
          },
        });
        firstPass = false;

        pass.setPipeline(variant.pipeline);
        // The compiler puts the uniform struct in group 0, textures in group 1
        // and samplers in group 2, so a draw sets one group per kind.
        pass.setBindGroup(0, entry.bindGroup, [slotIndex * entry.slotSize]);
        if (entry.textureBindGroup) pass.setBindGroup(1, entry.textureBindGroup);
        if (entry.samplerBindGroup) pass.setBindGroup(2, entry.samplerBindGroup);
        this.setVertexBuffers(pass, variant, mesh);

        const geometry = mesh.geometry;
        const instanceCount = instancing ? (mesh as InstancedMesh).count : geometry.instanceCount;
        // A mesh can draw a slice of its geometry; an infinite count draws the rest of it.
        const range = mesh.drawRange;
        if (geometry.index) {
          const buffers = this.ensureGeometryBuffers(geometry);
          pass.setIndexBuffer(buffers.index!, buffers.indexFormat as GPUIndexFormat, 0);
          const count = Number.isFinite(range.count) ? range.count : geometry.index.count - range.start;
          pass.drawIndexed(count, instanceCount, range.start);
        } else {
          const vertices = geometry.attributes.position?.count ?? 0;
          const count = Number.isFinite(range.count) ? range.count : vertices - range.start;
          pass.draw(count, instanceCount, range.start);
        }
        pass.end();
      }

      device.queue.submit([encoder.finish()]);
    } finally {
      // Holding the meshes between frames would keep a removed mesh alive.
      frameMeshes.length = frameEntries.length = frameVariants.length = 0;
      order.clear();
    }
  }

  /** The size of the surface the frame draws into: a render target's, or the canvas's. */
  private frameWidth = 0;
  private frameHeight = 0;

  /** The meshes of the frame `render` is drawing, opaque ones first and transparent ones back to front. */
  private frameOrder = new FrameOrder();

  /** Adds a mesh of the frame to `frameOrder`; made once, so a frame allocates no callback. */
  private collectVisible = (object: Object3D): void => {
    if (object.isMesh) this.frameOrder.add(object as Mesh);
  };

  /**
   * Makes the uniform ring of `entry` hold `draws` slots of a frame: it grows
   * to at least twice its size when too small, and shrinks to twice the draws
   * once it has held more than four times as many as a frame needs for
   * `RING_SHRINK_FRAMES` frames in a row.
   */
  private fitRing(entry: PipelineEntry, draws: number): void {
    let slots = entry.slots;
    const oversized = slots > UNIFORM_SLOTS && slots > draws * 4;
    entry.oversizedFrames = oversized ? entry.oversizedFrames + 1 : 0;
    if (draws > slots) slots = Math.max(draws, slots * 2);
    else if (oversized && entry.oversizedFrames >= RING_SHRINK_FRAMES) slots = Math.max(UNIFORM_SLOTS, draws * 2);
    if (slots === entry.slots) return;
    entry.oversizedFrames = 0;
    entry.ringBuffer.destroy();
    entry.ringBuffer = this.device.createBuffer({
      size: entry.slotSize * slots,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    entry.bindGroup = this.device.createBindGroup({
      layout: entry.bindGroupLayouts.uniforms,
      entries: [{ binding: 0, resource: { buffer: entry.ringBuffer, offset: 0, size: entry.slotSize } }],
    });
    entry.slots = slots;
  }

  private packUniforms(entry: PipelineEntry, mesh: Mesh, camera: Camera, slotIndex: number): void {
    const { scratch } = entry;
    scratch.f32.fill(0);
    for (const binding of entry.program.uniforms) {
      const member = entry.layoutMembers.find((m) => m.name === binding.node.name);
      if (!member) continue;
      let value: number | ArrayLike<number>;
      if (binding.scope === "camera") {
        value = cameraUniformValue(binding.name, camera);
      } else if (binding.scope === "object") {
        value = objectUniformValue(binding.name, mesh);
      } else if (binding.scope === "renderer") {
        value = rendererUniformValue(binding.name, this.frameWidth, this.frameHeight);
      } else {
        value = binding.value?.({ camera, mesh }) ?? [];
      }
      writeUniformMember(scratch, member, value);
    }
    this.device.queue.writeBuffer(
      entry.ringBuffer,
      slotIndex * entry.slotSize,
      scratch.f32 as BufferSource,
      0,
      entry.slotSize / 4,
    );
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
    if (
      entry &&
      entry.version === material.version &&
      entry.unfilterable === this.unfilterableSamplers(entry.program)
    ) {
      this.refreshTextures(entry);
      // A texture disposed since the last draw took this entry's texture and
      // sampler groups with it, and so does one re-created at a new size;
      // build them again from the textures the material points at now.
      this.bindTextures(entry);
      return entry;
    }

    const program = material.build(scene, { instancing, instancingColor });
    const device = this.device;

    // Must match the compiler's own alphabetical member sort, or byte offsets
    // drift from the WGSL struct. The same sorted list goes to both stages,
    // since each stage alone reads a different subset of the uniforms.
    const uniforms = [...program.uniforms].sort((a, b) => a.node.name.localeCompare(b.node.name));
    const declaredUniforms = uniforms.map((u) => ({
      slot: u.node.name,
      type: wgslTypeName(u.node._t),
    }));
    const layout = wgslUniformLayout(declaredUniforms);
    const declaredSamplers = program.samplers.map((s) => ({ slot: s.name, type: s.type }));

    const vertexModule = device.createShaderModule({
      code: compileWgsl.vertex(program.vertexRoot, { uniforms: declaredUniforms, samplers: declaredSamplers }),
    });
    const fragmentModule = device.createShaderModule({
      code: compileWgsl.fragment(program.fragmentRoot, { uniforms: declaredUniforms, samplers: declaredSamplers }),
    });
    const layoutMembers = layout.members;

    const slotSize = Math.max(256, Math.ceil(layout.size / 256) * 256);
    const slots = UNIFORM_SLOTS;
    const ringBuffer = device.createBuffer({
      size: slotSize * slots,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Both stages number the textures (group 1) and the samplers of the float
    // ones (group 2) from the program's whole set, as the compiler does.
    const sharedSamplers = sharedSamplerDeclarations(declaredSamplers);
    const textureBindings = sharedSamplers.map((t, binding) => ({
      name: t.slot,
      type: t.shaderType as SamplerShaderType,
      binding,
    }));
    const samplerBindings = sharedSamplers.filter((t) => !t.integer).map((t, binding) => ({ name: t.slot, binding }));

    // One layout per group the WGSL declares: uniforms in group 0, textures
    // in group 1, samplers in group 2.
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
              visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
              texture: {
                sampleType: this.unfilterable(entryTexture(program, t.name))
                  ? "unfilterable-float"
                  : samplerSampleType(t.type),
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
              visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
              sampler: { type: this.unfilterable(entryTexture(program, s.name)) ? "non-filtering" : "filtering" },
            })),
          });
    const groupLayouts = [uniformLayout];
    if (textureLayout) groupLayouts.push(textureLayout);
    if (samplerLayout) groupLayouts.push(samplerLayout);
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: groupLayouts });

    const cullMode: GPUCullMode =
      material.side === Side.FrontSide ? "back" : material.side === Side.BackSide ? "front" : "none";

    const built: PipelineEntry = {
      program,
      version: material.version,
      unfilterable: this.unfilterableSamplers(program),
      pipelineDescriptor: {
        layout: pipelineLayout,
        vertexModule,
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
      },
      variants: new Map(),
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
      oversizedFrames: 0,
      layoutMembers,
      scratch: uniformScratch(slotSize),
    };
    this.bindTextures(built);
    if (!bySignature) {
      bySignature = new Map();
      this.pipelines.set(material, bySignature);
    }
    // The entry this one replaces is never drawn again, so its ring goes with it.
    bySignature.get(signature)?.ringBuffer.destroy();
    bySignature.set(signature, built);
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
   * Upload again the textures of this entry whose `version` passed the one this
   * renderer uploaded, so a texture whose image changed reaches the GPU on the
   * next draw.
   *
   * A bind group binds the *texture*, not its contents, so an image rewritten
   * at the same size needs nothing else. One that changed size or format is a
   * new GPU texture, and `ensureGpuTexture` drops the bind groups that named
   * the old one.
   */
  private refreshTextures(entry: PipelineEntry): void {
    for (const t of entry.textureBindings) {
      const texture = entry.program.samplers.find((s) => s.name === t.name)!.texture();
      if (!texture || this.uploadedVersions.get(texture) === texture.version) continue;
      // Filtering or wrapping changed with it means a different sampler, and
      // this bind group holds the old one.
      const key = samplerKey(this.samplingOf(texture, t.type));
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
      buffers = { attributes: new Map(), index: null, indexFormat: null };
      this.geometryBuffers.set(geometry, buffers);
      geometry.addEventListener("dispose", this.onGeometryDispose);
    }
    for (const name in geometry.attributes) {
      const attribute = geometry.attributes[name]!;
      const previous = buffers.attributes.get(name);
      if (previous && this.holds(previous, attribute)) continue;
      const data = toBufferView(attribute.array);
      const buffer = this.bufferFitting(previous, data, GPUBufferUsage.VERTEX);
      buffers.attributes.set(name, buffer);
      this.writeAttribute(buffer, data, attribute, this.heldAttributes.get(buffer) !== attribute);
    }
    const index = geometry.index;
    if (index && !(buffers.index && this.holds(buffers.index, index))) {
      const data = toBufferView(index.array, true);
      buffers.index = this.bufferFitting(buffers.index ?? undefined, data, GPUBufferUsage.INDEX);
      this.writeAttribute(buffers.index, data, index, this.heldAttributes.get(buffers.index) !== index);
      buffers.indexFormat = (data as Uint16Array | Uint32Array).BYTES_PER_ELEMENT === 2 ? "uint16" : "uint32";
    }
    return buffers;
  }

  /**
   * Writes `attribute`'s data into `buffer`: whole when the buffer is new or
   * when no range is marked, and otherwise each range `addUpdateRange` marked,
   * merged as three.js merges them. The ranges are cleared afterwards, as
   * three.js clears them.
   */
  private writeAttribute(
    buffer: GPUBuffer,
    data: ArrayBufferView<ArrayBuffer>,
    attribute: BufferAttribute,
    whole: boolean,
  ): void {
    const ranges = mergedUpdateRanges(attribute);
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    if (whole || ranges.length === 0) {
      this.writeBytes(buffer, bytes, 0, bytes.length);
    } else {
      const element = (data as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
      const length = data.byteLength / element;
      for (const range of ranges) {
        const start = Math.min(length, Math.max(0, range.start));
        const count = Math.min(length - start, Math.max(0, range.count));
        if (count > 0) this.writeBytes(buffer, bytes, start * element, (start + count) * element);
      }
    }
    attribute.clearUpdateRanges();
    this.heldAttributes.set(buffer, attribute);
    this.heldVersions.set(buffer, attribute.version);
  }

  /** Whether `buffer` holds `attribute` at its current version. */
  private holds(buffer: GPUBuffer, attribute: BufferAttribute): boolean {
    return this.heldAttributes.get(buffer) === attribute && this.heldVersions.get(buffer) === attribute.version;
  }

  /** The last word of a write that runs past its data, padded with zeros; `writeBuffer` copies it at once. */
  private readonly tailWord = new Uint8Array(4);

  /**
   * Writes bytes `from` to `to` of `bytes` into `buffer` at the same offset,
   * widened to whole 4-byte words, since WebGPU writes nothing smaller. A last
   * word that runs past the data goes up padded with zeros, which the buffer,
   * sized to whole words, holds.
   */
  private writeBytes(buffer: GPUBuffer, bytes: Uint8Array<ArrayBuffer>, from: number, to: number): void {
    const start = from & ~3;
    const end = Math.ceil(to / 4) * 4;
    if (end <= bytes.length) {
      this.device.queue.writeBuffer(buffer, start, bytes, start, end - start);
      return;
    }
    const words = Math.max(start, bytes.length & ~3);
    if (words > start) this.device.queue.writeBuffer(buffer, start, bytes, start, words - start);
    this.tailWord.fill(0);
    this.tailWord.set(bytes.subarray(words));
    this.device.queue.writeBuffer(buffer, words, this.tailWord, 0, 4);
  }

  /**
   * `buffer` when it holds `data`, or a new buffer of `usage` that does, the
   * old one destroyed: a buffer's size is fixed when it is made, so an
   * attribute whose array grew needs a bigger one.
   */
  private bufferFitting(buffer: GPUBuffer | undefined, data: ArrayBufferView, usage: number): GPUBuffer {
    if (buffer && buffer.size >= data.byteLength) return buffer;
    buffer?.destroy();
    return this.device.createBuffer({
      size: Math.max(Math.ceil(data.byteLength / 4) * 4, 4),
      usage: usage | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * The render pipeline of `entry` for the vertex formats `mesh` holds its
   * attributes in, made on first use. One slot per shader attribute, matching
   * the WGSL `VertexInput` struct's `@location` numbering, with the formats
   * resolved as `WebGLRenderer` resolves them.
   */
  private pipelineVariant(entry: PipelineEntry, mesh: Mesh): PipelineVariant {
    const material = mesh.material as RenderStateMaterial;
    // The material's blend and depth state is part of the pipeline, read at each draw as WebGL reads it.
    let key = `${material.transparent},${material.blending},${material.depthTest},${material.depthWrite};`;
    for (const attribute of entry.program.attributes) {
      key += `${this.formatOf(attribute, mesh)},`;
    }
    const cached = entry.variants.get(key);
    if (cached) return cached;
    const vertexFormats: VertexBufferLayout[] = [];
    let shaderLocation = 0;
    for (const attribute of entry.program.attributes) {
      const columns = attribute.node._t === "mat4" ? 4 : 1;
      const format = this.formatOf(attribute, mesh);
      const { count, bytes } = VERTEX_FORMATS[format];
      const locations: VertexBufferLayout["attributes"] = [];
      for (let i = 0; i < columns; i++) {
        locations.push({ shaderLocation: shaderLocation + i, offset: i * count * bytes, format });
      }
      vertexFormats.push({
        name: attribute.name,
        stepMode: attribute.stepMode,
        arrayStride: columns * count * bytes,
        attributes: locations,
      });
      shaderLocation += columns;
    }
    const { vertexModule, ...descriptor } = entry.pipelineDescriptor;
    const variant = {
      pipeline: this.device.createRenderPipeline({
        ...descriptor,
        fragment: {
          ...descriptor.fragment!,
          targets: [{ format: this.format, blend: blendState(material) }],
        },
        depthStencil: {
          format: "depth24plus",
          depthWriteEnabled: material.depthWrite,
          depthCompare: material.depthTest ? "less" : "always",
        },
        vertex: {
          module: vertexModule,
          entryPoint: "main",
          buffers: vertexFormats.map((layout) => ({
            arrayStride: layout.arrayStride,
            stepMode: layout.stepMode,
            attributes: layout.attributes,
          })),
        },
      }),
      vertexFormats,
    };
    entry.variants.set(key, variant);
    return variant;
  }

  /** The vertex format `mesh` holds the attribute of the program in, or the one its shader type implies. */
  private formatOf(attribute: MaterialProgram["attributes"][number], mesh: Mesh): VertexFormat {
    const columns = attribute.node._t === "mat4" ? 4 : 1;
    const attr = geometryAttribute(mesh, mesh.geometry, attribute.name);
    return attr
      ? vertexFormatOf(attr, attr.itemSize / columns)
      : vertexFormatFromType(columns === 4 ? "vec4" : attribute.node._t);
  }

  private setVertexBuffers(pass: GPURenderPassEncoder, variant: PipelineVariant, mesh: Mesh): void {
    const buffers = this.ensureGeometryBuffers(mesh.geometry);
    // Each `vertexFormats` entry is one vertex buffer slot, so the buffer is
    // bound at its slot index (the loop position) rather than any shader
    // location — a mat4 entry spans several locations from a single buffer.
    for (let slot = 0; slot < variant.vertexFormats.length; slot++) {
      const layout = variant.vertexFormats[slot];
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
    if (!buffer || this.uploadedVersions.get(attr) !== attr.version) {
      const data = toBufferView(attr.array);
      const previous = buffer;
      buffer = this.bufferFitting(buffer, data, GPUBufferUsage.VERTEX);
      this.attributeBuffers.set(attr, buffer);
      this.writeAttribute(buffer, data, attr, buffer !== previous);
      this.uploadedVersions.set(attr, attr.version);
    }
    return buffer;
  }

  /**
   * The GPU texture holding this `Texture`'s image, created on first use and
   * written again whenever its `version` passes the one this renderer uploaded.
   */
  private ensureGpuTexture(texture: Texture | null, samplerType: string): GPUTexture {
    const t = texture ?? this.blankTexture(samplerType);
    const integer = isIntegerSampler(samplerType);
    const dimension = samplerDimension(samplerType);
    let gpu = this.textures.get(t);
    if (!gpu || this.uploadedVersions.get(t) !== t.version) {
      // An image element, bitmap or canvas has a size of its own; a data texture names its size.
      const source = imageSource(t.image);
      const width = source ? source.width : ArrayBuffer.isView(t.image) ? ((t as DataTexture).width ?? 1) : 1;
      const height = source ? source.height : ArrayBuffer.isView(t.image) ? ((t as DataTexture).height ?? 1) : 1;
      const depth = dimension === "3d" ? ((t as DataTexture).depth ?? 1) : 1;
      const format: GPUTextureFormat = integer
        ? textureChannels(t) === 1
          ? samplerType.startsWith("isampler")
            ? "r8sint"
            : "r8uint"
          : integerGpuFormat(samplerType, ArrayBuffer.isView(t.image) ? t.image : null)
        : isFloatTexture(t)
          ? "rgba32float"
          : "rgba8unorm";
      // A WebGPU texture's size/format is fixed at creation, so a reshaped
      // image gets a new texture — whatever bound the old one must rebind.
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
          // Copying an image source in renders into the texture.
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            (source ? GPUTextureUsage.RENDER_ATTACHMENT : 0),
        });
        this.textures.set(t, gpu);
        t.addEventListener("dispose", this.onTextureDispose);
      }
      if (ArrayBuffer.isView(t.image)) {
        this.writeTexture(gpu, t.image as unknown as ArrayBufferView<ArrayBuffer>, width, height, depth, format);
      } else if (source) {
        this.device.queue.copyExternalImageToTexture(
          { source: source as GPUCopyExternalImageSource },
          { texture: gpu },
          [width, height],
        );
      }
      this.uploadedVersions.set(t, t.version);
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
      format === "rgba32uint" || format === "rgba32sint" || format === "rgba32float"
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

  private blankTexture(samplerType = "sampler2D"): DataTexture {
    return blankTexture(this.blankTextures, samplerType);
  }

  /**
   * Whether `texture` holds 32-bit floats the device cannot filter, so it binds
   * as an unfilterable texture and reads through a sampler that takes the
   * nearest texel.
   */
  private unfilterable(texture: Texture | null): boolean {
    return texture !== null && isFloatTexture(texture) && !this.device.features?.has("float32-filterable");
  }

  /**
   * The bits of the samplers of `program` that read a float texture the device
   * cannot filter. A layout built for other textures cannot bind it.
   */
  private unfilterableSamplers(program: MaterialProgram): number {
    let bits = 0;
    for (let i = 0; i < program.samplers.length; i++) {
      if (this.unfilterable(program.samplers[i]!.texture())) bits |= 1 << i;
    }
    return bits;
  }

  /** How a sampler reads `texture`: as it asks, or nearest where the device cannot filter it. */
  private samplingOf(texture: Texture, samplerType: string) {
    const asked = samplerState(texture, samplerType);
    return this.unfilterable(texture)
      ? { ...asked, magFilter: "nearest" as const, minFilter: "nearest" as const }
      : asked;
  }

  /** The sampler that reads this texture the way the texture asks to be read. */
  private ensureSampler(texture: Texture | null, samplerType = "sampler2D"): GPUSampler {
    const t = texture ?? this.blankTexture();
    const state = this.samplingOf(t, samplerType);
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

  /** The colour and depth textures behind each render target, at its size when it was last drawn into. */
  private renderTargets = new Map<
    WebGLRenderTarget,
    { color: GPUTexture; depth: GPUTexture; colorView: GPUTextureView; depthView: GPUTextureView }
  >();

  /**
   * The textures behind `target`, made on first use and made again at the
   * target's new size when it changed, so a target is resized by setting its
   * `width` and `height` before the next render.
   */
  private renderTargetTextures(target: WebGLRenderTarget) {
    const existing = this.renderTargets.get(target);
    if (existing && existing.color.width === target.width && existing.color.height === target.height) return existing;
    existing?.color.destroy();
    existing?.depth.destroy();
    const color = this.device.createTexture({
      size: [target.width, target.height],
      format: this.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const depth = this.device.createTexture({
      size: [target.width, target.height],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const textures = { color, depth, colorView: color.createView(), depthView: depth.createView() };
    this.renderTargets.set(target, textures);
    return textures;
  }

  /**
   * Reads a render target's colour back as RGBA bytes, row by row from the
   * bottom as the WebGL renderer's `readPixels` gives them, into `out` or a new
   * array, so the same frame gives the same bytes on both. WebGPU reads a texture back only
   * asynchronously, so this gives a promise; it resolves once the copy has
   * landed, typically a frame or a few after the call.
   */
  async readPixels(target: WebGLRenderTarget, out?: Uint8Array): Promise<Uint8Array> {
    const { color } = this.renderTargetTextures(target);
    const { width, height } = target;
    const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
    const staging = this.device.createBuffer({
      size: bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyTextureToBuffer({ texture: color }, { buffer: staging, bytesPerRow }, [width, height]);
      this.device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const rows = new Uint8Array(staging.getMappedRange());
      const pixels = out ?? new Uint8Array(width * height * 4);
      // The texture is in the canvas's format, BGRA on some platforms; the bytes come back as RGBA.
      const bgra = this.format === "bgra8unorm";
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const from = y * bytesPerRow + x * 4;
          const to = ((height - 1 - y) * width + x) * 4;
          pixels[to] = rows[from + (bgra ? 2 : 0)]!;
          pixels[to + 1] = rows[from + 1]!;
          pixels[to + 2] = rows[from + (bgra ? 0 : 2)]!;
          pixels[to + 3] = rows[from + 3]!;
        }
      }
      staging.unmap();
      return pixels;
    } finally {
      staging.destroy();
    }
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
    for (const { color, depth } of this.renderTargets.values()) {
      color.destroy();
      depth.destroy();
    }
    this.renderTargets.clear();
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

function vertexFormatFromType(type: string): VertexFormat {
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

/**
 * The WebGPU format for an integer RGBA texture, from the bit depth of its
 * data view and the sampler's signedness.
 */
/** The texture a program's sampler of that name reads now, or null. */
function entryTexture(program: MaterialProgram, name: string): Texture | null {
  return program.samplers.find((s) => s.name === name)?.texture() ?? null;
}

/** `image` when it is an image element, bitmap or canvas, which has a size of its own, or null for data or nothing. */
function imageSource(image: Texture["image"]): { width: number; height: number } | null {
  if (image === null || ArrayBuffer.isView(image)) return null;
  const { width, height } = image as { width?: number; height?: number };
  return typeof width === "number" && typeof height === "number" ? { width, height } : null;
}

/** What a draw's blend and depth state is read from. */
type RenderStateMaterial = { transparent: boolean; blending: Blending; depthTest: boolean; depthWrite: boolean };

/**
 * The blend state a material asks for, as the WebGL renderer sets it: none for
 * an opaque material of normal blending, additive for additive blending, and
 * over the destination by the source alpha otherwise.
 */
function blendState(material: RenderStateMaterial): GPUBlendState | undefined {
  if (!material.transparent && material.blending === Blending.NormalBlending) return undefined;
  const dstFactor: GPUBlendFactor = material.blending === Blending.AdditiveBlending ? "one" : "one-minus-src-alpha";
  return {
    color: { srcFactor: "src-alpha", dstFactor, operation: "add" },
    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
  };
}

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
