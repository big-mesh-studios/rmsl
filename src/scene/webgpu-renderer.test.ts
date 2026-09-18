import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { vec2 } from "../rmsl";
import {
  WebGPURenderer,
  Scene,
  MeshBasicMaterial,
  DataTexture,
  PlaneGeometry,
  NearestFilter,
  RepeatWrapping,
  MirroredRepeatWrapping,
} from "./index";

/** A GPUTexture stand-in: its size and format, and whether it was destroyed. */
interface StubTexture {
  width: number;
  height: number;
  depthOrArrayLayers: number;
  format: string;
  destroyed: boolean;
  createView: () => { texture: StubTexture };
  destroy: () => void;
}

/** A GPUBuffer stand-in: the size it was asked for, and whether it was destroyed. */
interface StubBuffer {
  size: number;
  destroyed: boolean;
  destroy: () => void;
}

interface StubDevice {
  device: any;
  textures: StubTexture[];
  buffers: StubBuffer[];
  bindGroups: unknown[];
  /** One entry per `createSampler`: the descriptor it was asked for. */
  samplers: any[];
  /** One entry per `queue.writeTexture`: which texture, and the bytes given. */
  writes: { texture: StubTexture; data: ArrayBufferView }[];
}

function stubDevice(): StubDevice {
  const textures: StubTexture[] = [];
  const buffers: StubBuffer[] = [];
  const bindGroups: unknown[] = [];
  const samplers: any[] = [];
  const writes: { texture: StubTexture; data: ArrayBufferView }[] = [];
  const device = {
    createShaderModule: () => ({}),
    createBuffer: (descriptor: any = {}) => {
      const buffer: StubBuffer = {
        size: descriptor.size ?? 0,
        destroyed: false,
        destroy: () => {
          buffer.destroyed = true;
        },
      };
      buffers.push(buffer);
      return buffer;
    },
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    createRenderPipeline: () => ({}),
    createSampler: (descriptor: unknown) => {
      samplers.push(descriptor);
      return descriptor;
    },
    createBindGroup: (descriptor: unknown) => {
      const group = { descriptor };
      bindGroups.push(group);
      return group;
    },
    createTexture: (descriptor: any) => {
      const [width, height, depth] = descriptor.size;
      const texture: StubTexture = {
        width,
        height,
        depthOrArrayLayers: depth,
        format: descriptor.format,
        destroyed: false,
        createView: () => ({ texture }),
        destroy: () => {
          texture.destroyed = true;
        },
      };
      textures.push(texture);
      return texture;
    },
    queue: {
      writeTexture: (destination: any, data: ArrayBufferView) => {
        writes.push({ texture: destination.texture, data });
      },
      writeBuffer: () => {},
    },
  };
  return { device, textures, buffers, bindGroups, samplers, writes };
}

function stubCanvas(): any {
  return {
    width: 16,
    height: 16,
    getContext: () => ({ configure: () => {} }),
  };
}

/** A material whose only fragment work is sampling `texture`. */
function texturedMaterial(texture: DataTexture): MeshBasicMaterial {
  const material = new MeshBasicMaterial();
  material.fragmentNode = (b) => b.sampler("map", () => texture).texture(vec2(0.5, 0.5));
  return material;
}

beforeEach(() => {
  vi.stubGlobal("navigator", { gpu: { getPreferredCanvasFormat: () => "bgra8unorm" } });
  vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1, COPY_DST: 2, VERTEX: 4, INDEX: 8 });
  vi.stubGlobal("GPUTextureUsage", { TEXTURE_BINDING: 1, COPY_DST: 2, RENDER_ATTACHMENT: 4 });
  vi.stubGlobal("GPUShaderStage", { VERTEX: 1, FRAGMENT: 2 });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("WebGPURenderer samplers", () => {
  it("makes one sampler per filtering and wrapping combination", () => {
    const { device, samplers } = stubDevice();
    const renderer = new WebGPURenderer(stubCanvas(), device) as any;
    const scene = new Scene();

    const clamped = new DataTexture(new Uint8Array([0, 0, 220, 255]), 1, 1);
    const alsoClamped = new DataTexture(new Uint8Array([220, 0, 0, 255]), 1, 1);
    renderer.ensurePipeline(texturedMaterial(clamped), scene, false, false);
    renderer.ensurePipeline(texturedMaterial(alsoClamped), scene, false, false);
    // Two textures, sampled the same way: a sampler holds no image, so one does.
    expect(samplers).toHaveLength(1);

    const tiled = new DataTexture(new Uint8Array([0, 220, 0, 255]), 1, 1);
    tiled.wrapS = RepeatWrapping;
    renderer.ensurePipeline(texturedMaterial(tiled), scene, false, false);
    expect(samplers).toHaveLength(2);
    expect(samplers[1]).toMatchObject({ addressModeU: "repeat", addressModeV: "clamp-to-edge" });
  });

  it("describes the sampler the way the texture asked", () => {
    const { device, samplers } = stubDevice();
    const renderer = new WebGPURenderer(stubCanvas(), device) as any;
    const texture = new DataTexture(new Uint8Array([0, 0, 220, 255]), 1, 1);
    texture.magFilter = NearestFilter;
    texture.minFilter = NearestFilter;
    texture.wrapS = RepeatWrapping;
    texture.wrapT = MirroredRepeatWrapping;
    renderer.ensurePipeline(texturedMaterial(texture), new Scene(), false, false);
    expect(samplers[0]).toEqual({
      magFilter: "nearest",
      minFilter: "nearest",
      addressModeU: "repeat",
      addressModeV: "mirror-repeat",
      addressModeW: "clamp-to-edge",
    });
  });

  it("rebinds when a texture is updated with a different sampler state", () => {
    const { device, samplers } = stubDevice();
    const renderer = new WebGPURenderer(stubCanvas(), device) as any;
    const texture = new DataTexture(new Uint8Array([0, 0, 220, 255]), 1, 1);
    const material = texturedMaterial(texture);
    const scene = new Scene();
    const entry = renderer.ensurePipeline(material, scene, false, false);
    const first = entry.samplerBindGroup;

    texture.wrapS = RepeatWrapping;
    texture.needsUpdate = true;
    const again = renderer.ensurePipeline(material, scene, false, false);

    // The bind group held the clamping sampler, so it cannot stand.
    expect(samplers).toHaveLength(2);
    expect(again.samplerBindGroup).not.toBe(first);
  });

  it("leaves the bind group alone when an update changes nothing about sampling", () => {
    const { device, samplers } = stubDevice();
    const renderer = new WebGPURenderer(stubCanvas(), device) as any;
    const texture = new DataTexture(new Uint8Array([0, 0, 220, 255]), 1, 1);
    const material = texturedMaterial(texture);
    const scene = new Scene();
    const entry = renderer.ensurePipeline(material, scene, false, false);
    const first = entry.samplerBindGroup;

    texture.image = new Uint8Array([220, 0, 0, 255]);
    texture.needsUpdate = true;
    const again = renderer.ensurePipeline(material, scene, false, false);

    expect(samplers).toHaveLength(1);
    expect(again.samplerBindGroup).toBe(first);
  });
});

describe("WebGPURenderer texture updates", () => {
  it("uploads a changed image again on the next render", () => {
    const { device, textures, bindGroups, writes } = stubDevice();
    const renderer = new WebGPURenderer(stubCanvas(), device) as any;
    const texture = new DataTexture(new Uint8Array([0, 0, 220, 255]), 1, 1);
    const material = texturedMaterial(texture);
    const scene = new Scene();

    const entry = renderer.ensurePipeline(material, scene, false, false);
    expect(writes).toHaveLength(1);
    expect(texture.needsUpdate).toBe(false);
    const boundTexture = entry.textureBindGroup;

    texture.image = new Uint8Array([220, 0, 0, 255]);
    texture.needsUpdate = true;
    renderer.ensurePipeline(material, scene, false, false);

    expect(writes).toHaveLength(2);
    expect([...(writes[1].data as Uint8Array)]).toEqual([220, 0, 0, 255]);
    // The image is written into the texture that is already bound, so nothing
    // was created and the bind group still holds.
    expect(writes[1].texture).toBe(textures[0]);
    expect(textures).toHaveLength(1);
    expect(entry.textureBindGroup).toBe(boundTexture);
  });

  it("leaves an unchanged texture alone", () => {
    const { device, writes } = stubDevice();
    const renderer = new WebGPURenderer(stubCanvas(), device) as any;
    const texture = new DataTexture(new Uint8Array([0, 0, 220, 255]), 1, 1);
    const material = texturedMaterial(texture);
    const scene = new Scene();

    renderer.ensurePipeline(material, scene, false, false);
    renderer.ensurePipeline(material, scene, false, false);
    expect(writes).toHaveLength(1);
  });

  it("replaces and rebinds a texture whose image changed size", () => {
    const { device, textures, bindGroups, writes } = stubDevice();
    const renderer = new WebGPURenderer(stubCanvas(), device) as any;
    const texture = new DataTexture(new Uint8Array([0, 0, 220, 255]), 1, 1);
    const material = texturedMaterial(texture);
    const scene = new Scene();

    const entry = renderer.ensurePipeline(material, scene, false, false);
    const first = entry.textureBindGroup;

    texture.image = new Uint8Array(2 * 2 * 4).fill(220);
    texture.width = 2;
    texture.height = 2;
    texture.needsUpdate = true;
    const again = renderer.ensurePipeline(material, scene, false, false);

    // A GPU texture is fixed at the size it was created with, so the bigger
    // image needs a new one — and a bind group naming the old one is no good.
    expect(textures).toHaveLength(2);
    expect(textures[0].destroyed).toBe(true);
    expect(textures[1].width).toBe(2);
    expect(writes[1].texture).toBe(textures[1]);
    expect(renderer.textures.get(texture)).toBe(textures[1]);
    expect(again.textureBindGroup).not.toBe(first);
  });
});

describe("WebGPURenderer texture disposal", () => {
  it("destroys the GPU texture and rebuilds the bind groups that named it", () => {
    const { device, textures, bindGroups } = stubDevice();
    const renderer = new WebGPURenderer(stubCanvas(), device) as any;
    const texture = new DataTexture(new Uint8Array([0, 0, 220, 255]), 1, 1);
    const material = texturedMaterial(texture);
    const scene = new Scene();

    const entry = renderer.ensurePipeline(material, scene, false, false);
    expect(textures).toHaveLength(1);
    // One group per kind: the uniform buffer, the texture, the sampler.
    expect(bindGroups).toHaveLength(3);
    const first = entry.textureBindGroup;

    texture.dispose();
    expect(textures[0].destroyed).toBe(true);
    expect(renderer.textures.size).toBe(0);
    // The sampler is shared with every texture filtered and wrapped the same
    // way, so it stays; what goes is this texture's claim on one.
    expect(renderer.samplerKeys.size).toBe(0);
    // The group still holding a view of the destroyed texture is gone, so no
    // draw can reach it. The uniform group holds nothing of the texture and
    // stays as it was.
    expect(entry.textureBindGroup).toBe(null);
    expect(entry.samplerBindGroup).toBe(null);
    expect(entry.bindGroup).not.toBe(null);

    // Looking the pipeline up again re-uploads the texture and binds it anew,
    // without recompiling the shaders.
    const again = renderer.ensurePipeline(material, scene, false, false);
    expect(again).toBe(entry);
    expect(textures).toHaveLength(2);
    expect(again.textureBindGroup).not.toBe(first);
    expect(renderer.textures.get(texture)).toBe(textures[1]);
  });

  it("leaves pipelines that do not sample the disposed texture alone", () => {
    const { device } = stubDevice();
    const renderer = new WebGPURenderer(stubCanvas(), device) as any;
    const used = new DataTexture(new Uint8Array([0, 0, 220, 255]), 1, 1);
    const unused = new DataTexture(new Uint8Array([220, 0, 0, 255]), 1, 1);
    const scene = new Scene();
    const entry = renderer.ensurePipeline(texturedMaterial(used), scene, false, false);
    const bindGroup = entry.textureBindGroup;

    unused.dispose();
    expect(entry.textureBindGroup).toBe(bindGroup);
  });

  it("stops listening to the textures it frees when the renderer is disposed", () => {
    const { device, textures } = stubDevice();
    const renderer = new WebGPURenderer(stubCanvas(), device) as any;
    const texture = new DataTexture(new Uint8Array([0, 0, 220, 255]), 1, 1);
    renderer.ensurePipeline(texturedMaterial(texture), new Scene(), false, false);

    renderer.dispose();
    expect(textures[0].destroyed).toBe(true);
    expect(texture.hasEventListener("dispose", renderer.onTextureDispose)).toBe(false);
  });
});

describe("WebGPURenderer geometry disposal", () => {
  it("destroys the vertex and index buffers and uploads again on the next draw", () => {
    const { device, buffers } = stubDevice();
    const renderer = new WebGPURenderer(stubCanvas(), device) as any;
    const geometry = new PlaneGeometry(2, 2);

    const uploaded = renderer.ensureGeometryBuffers(geometry);
    // One buffer per attribute, plus the index.
    const held = [...uploaded.attributes.values(), uploaded.index];
    expect(held).toHaveLength(Object.keys(geometry.attributes).length + 1);
    expect(held.every((buffer: StubBuffer) => !buffer.destroyed)).toBe(true);

    geometry.dispose();
    expect(held.every((buffer: StubBuffer) => buffer.destroyed)).toBe(true);
    expect(renderer.geometryBuffers.size).toBe(0);

    // Drawing with the same geometry again fills fresh buffers.
    const madeBefore = buffers.length;
    const after = renderer.ensureGeometryBuffers(geometry);
    expect(renderer.geometryBuffers.size).toBe(1);
    expect(buffers.length).toBe(madeBefore * 2);
    expect(after.index).not.toBe(uploaded.index);
  });

  it("stops listening to the geometries it frees when the renderer is disposed", () => {
    const { device } = stubDevice();
    const renderer = new WebGPURenderer(stubCanvas(), device) as any;
    const geometry = new PlaneGeometry(2, 2);
    renderer.ensureGeometryBuffers(geometry);

    renderer.dispose();
    expect(geometry.hasEventListener("dispose", renderer.onGeometryDispose)).toBe(false);
  });
});
