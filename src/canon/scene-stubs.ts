import { MeshBasicMaterial, PerspectiveCamera, WebGLRenderer, type Texture } from "../scene";
import { vec2 } from "../rmsl";

/**
 * Stand-ins for a GPU device and a WebGL context that record what a scene
 * renderer asks of them, shared by the scene tests.
 */

export interface Call {
  name: string;
  args: any[];
}

/**
 * A `GPUDevice` and canvas context that record what the renderer asks of them:
 * buffer and texture writes kept as bytes, pipelines, samplers and layouts by descriptor,
 * and every call a render pass receives.
 */
export function stubDevice() {
  const contents = new Map<object, Uint8Array>();
  const textures: any[] = [];
  const pipelines: any[] = [];
  const samplers: any[] = [];
  const layouts: any[] = [];
  const textureWrites: { texture: any; data: ArrayBufferView }[] = [];
  const bufferWrites: { buffer: any; offset: number }[] = [];
  const passes: { descriptor: any; calls: Call[] }[] = [];
  const queue: Call[] = [];
  const destroyed = new Set<object>();
  const modules: { code: string }[] = [];
  const device = {
    createShaderModule: (descriptor: any) => {
      modules.push(descriptor);
      return descriptor;
    },
    createBuffer: (descriptor: any) => {
      const buffer: any = { size: descriptor.size, usage: descriptor.usage, destroy: () => destroyed.add(buffer) };
      contents.set(buffer, new Uint8Array(descriptor.size));
      return buffer;
    },
    createBindGroupLayout: (descriptor: any) => {
      layouts.push(descriptor);
      return descriptor;
    },
    createPipelineLayout: (descriptor: any) => descriptor,
    createRenderPipeline: (descriptor: any) => {
      pipelines.push(descriptor);
      return descriptor;
    },
    createSampler: (descriptor: any) => {
      samplers.push(descriptor);
      return descriptor;
    },
    createBindGroup: (descriptor: any) => descriptor,
    createTexture: (descriptor: any) => {
      const [width, height, depth] = descriptor.size;
      const texture = {
        width,
        height,
        depthOrArrayLayers: depth ?? 1,
        format: descriptor.format,
        createView: () => ({ texture }),
        destroy: () => {},
      };
      textures.push(texture);
      return texture;
    },
    createCommandEncoder: () => ({
      beginRenderPass: (descriptor: any) => {
        const calls: Call[] = [];
        passes.push({ descriptor, calls });
        return new Proxy(
          {},
          {
            get:
              (_target, name: string) =>
              (...args: any[]) =>
                calls.push({ name, args }),
          },
        );
      },
      finish: () => ({}),
    }),
    queue: {
      writeBuffer: (buffer: any, offset: number, data: ArrayBufferView, dataOffset = 0, size?: number) => {
        const element = (data as any).BYTES_PER_ELEMENT ?? 1;
        const bytes = new Uint8Array(data.buffer, data.byteOffset + dataOffset * element);
        const length = size === undefined ? bytes.length : size * element;
        if (offset % 4 !== 0 || length % 4 !== 0) throw new Error("writeBuffer of an offset or size not a multiple of 4");
        const target = contents.get(buffer)!;
        if (offset + length > target.length) throw new Error("writeBuffer past the end of the buffer");
        target.set(bytes.subarray(0, length), offset);
        bufferWrites.push({ buffer, offset });
      },
      writeTexture: (destination: any, data: ArrayBufferView) => {
        textureWrites.push({ texture: destination.texture, data });
      },
      copyExternalImageToTexture: (_source: any, destination: any) => {
        queue.push({ name: "copyExternalImageToTexture", args: [destination] });
      },
      submit: () => {},
    },
  };
  const canvas: any = {
    width: 16,
    height: 16,
    getContext: () => ({ configure: () => {}, getCurrentTexture: () => ({ createView: () => ({}) }) }),
  };
  /** The bytes a buffer holds once everything written to it has landed. */
  const bytesOf = (buffer: object) => contents.get(buffer)!;
  return {
    device,
    canvas,
    modules,
    textures,
    pipelines,
    samplers,
    layouts,
    textureWrites,
    bufferWrites,
    passes,
    queue,
    bytesOf,
    destroyed,
  };
}

/**
 * A `WebGL2RenderingContext` that accepts every call and records it. Constants
 * read as distinct numbers, shaders compile and programs link, and each
 * uniform location is an object naming its uniform.
 */
export function stubGl(
  canvas: { width: number; height: number },
  overrides: Record<string, (...args: any[]) => unknown> = {},
) {
  const calls: Call[] = [];
  const constants = new Map<string, number>();
  let location = 0;
  const answers: Record<string, (...args: any[]) => unknown> = {
    getShaderParameter: () => true,
    getProgramParameter: () => true,
    getParameter: () => 16,
    getAttribLocation: () => (location += 4),
    getUniformLocation: (_program: unknown, name: string) => ({ name }),
    ...overrides,
  };
  const gl = new Proxy(
    {},
    {
      get: (_target, name: string) => {
        if (name === "drawingBufferWidth") return canvas.width;
        if (name === "drawingBufferHeight") return canvas.height;
        if (/^[A-Z0-9_]+$/.test(name)) {
          if (!constants.has(name)) constants.set(name, 0x1000 + constants.size);
          return constants.get(name);
        }
        return (...args: any[]) => {
          calls.push({ name, args });
          return name in answers ? answers[name]!(...args) : {};
        };
      },
    },
  ) as any;
  return { gl, calls };
}

/** A WebGL renderer drawing through `stubGl`, on a 32×32 canvas. */
export function stubWebGl(
  overrides: Record<string, (...args: any[]) => unknown> = {},
  options: ConstructorParameters<typeof WebGLRenderer>[1] = {},
) {
  const canvas: any = { width: 32, height: 32 };
  const { gl, calls } = stubGl(canvas, overrides);
  canvas.getContext = () => gl;
  return { renderer: new WebGLRenderer(canvas, options) as any, gl, calls };
}

export function camera(): PerspectiveCamera {
  const c = new PerspectiveCamera(50, 1, 0.1, 100);
  c.position.set(0, 0, 4);
  c.lookAt(0, 0, 0);
  return c;
}

/** A material whose fragment stage samples `texture` and nothing else. */
export function sampling(texture: Texture): MeshBasicMaterial {
  const material = new MeshBasicMaterial();
  material.fragmentNode = (b) => b.sampler("map", () => texture).texture(vec2(0.5, 0.5));
  return material;
}

/** The byte offset the uniform layout gives a uniform of an entry's program. */
export function offsetOf(entry: any, slot: string): number {
  return entry.layoutMembers.find((m: any) => m.name === slot)!.offset;
}
