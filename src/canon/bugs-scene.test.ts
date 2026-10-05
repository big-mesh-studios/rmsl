import { afterAll, afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { float, vec2, vec4 } from "../rmsl";
import { compileWgsl } from "../wgsl";
import {
  BufferAttribute,
  BufferGeometry,
  Color,
  DataTexture,
  InstancedMesh,
  Line2NodeMaterial,
  LineSegments2,
  LineSegmentsGeometry,
  Matrix3,
  Mesh,
  MeshBasicMaterial,
  PerspectiveCamera,
  PlaneGeometry,
  RGBAFormat,
  Scene,
  Texture,
  WebGLRenderer,
  WebGLRenderTarget,
  WebGPURenderer,
} from "../scene";
import { collectNodes } from "../scene/materials/nodes/graph";
import { GPU_ENABLED, releaseGpu } from "../testing/gpu";
import { runInGpuPage } from "../testing/browser";

/** three.js's `FloatType`, which `./scene` does not export. */
const FloatType = 1015;

interface Call {
  name: string;
  args: any[];
}

/**
 * A `GPUDevice` and canvas context that record what the renderer asks of them:
 * buffer and texture writes kept as bytes, pipelines and layouts by descriptor,
 * and every call a render pass receives.
 */
function stubDevice() {
  const contents = new Map<object, Uint8Array>();
  const textures: any[] = [];
  const pipelines: any[] = [];
  const layouts: any[] = [];
  const textureWrites: { texture: any; data: ArrayBufferView }[] = [];
  const bufferWrites: { buffer: any; offset: number }[] = [];
  const passes: { descriptor: any; calls: Call[] }[] = [];
  const queue: Call[] = [];
  const destroyed = new Set<object>();
  const device = {
    createShaderModule: (descriptor: any) => descriptor,
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
    createSampler: (descriptor: any) => descriptor,
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
    textures,
    pipelines,
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
function stubGl(canvas: { width: number; height: number }) {
  const calls: Call[] = [];
  const constants = new Map<string, number>();
  let location = 0;
  const answers: Record<string, (...args: any[]) => unknown> = {
    getShaderParameter: () => true,
    getProgramParameter: () => true,
    getParameter: () => 16,
    getAttribLocation: () => (location += 4),
    getUniformLocation: (_program: unknown, name: string) => ({ name }),
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
          return answers[name]?.(...args) ?? {};
        };
      },
    },
  ) as any;
  return { gl, calls };
}

/** A WebGL renderer drawing through `stubGl`, on a 32×32 canvas. */
function stubWebGl() {
  const canvas: any = { width: 32, height: 32 };
  const { gl, calls } = stubGl(canvas);
  canvas.getContext = () => gl;
  return { renderer: new WebGLRenderer(canvas) as any, gl, calls };
}

function camera(): PerspectiveCamera {
  const c = new PerspectiveCamera(50, 1, 0.1, 100);
  c.position.set(0, 0, 4);
  c.lookAt(0, 0, 0);
  return c;
}

/** A material whose fragment stage samples `texture` and nothing else. */
function sampling(texture: Texture): MeshBasicMaterial {
  const material = new MeshBasicMaterial();
  material.fragmentNode = (b) => b.sampler("map", () => texture).texture(vec2(0.5, 0.5));
  return material;
}

/** The byte offset the uniform layout gives a uniform of an entry's program. */
function offsetOf(entry: any, slot: string): number {
  return entry.layoutMembers.find((m: any) => m.name === slot)!.offset;
}

beforeEach(() => {
  vi.stubGlobal("navigator", { gpu: { getPreferredCanvasFormat: () => "bgra8unorm" } });
  vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1, COPY_DST: 2, VERTEX: 4, INDEX: 8 });
  vi.stubGlobal("GPUTextureUsage", { TEXTURE_BINDING: 1, COPY_DST: 2, RENDER_ATTACHMENT: 4 });
  vi.stubGlobal("GPUShaderStage", { VERTEX: 1, FRAGMENT: 2 });
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("known bugs of the scene library, each failing until its fix", () => {
  /**
   * The WebGPU renderer writes a `mat3` uniform as nine packed floats, where
   * WGSL pads each column to 16 bytes, so the shader reads the normal matrix
   * shifted from its second column on.
   *
   * @canon bug-webgpu-uploads-a-mat3-without-column-padding
   */
  it.fails("uploads the normal matrix with each column padded to 16 bytes on WebGPU", () => {
    const { device, canvas, bytesOf } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const mesh = new Mesh(new PlaneGeometry(), new MeshBasicMaterial());
    mesh.rotation.set(0.3, 0.5, 0.7);
    mesh.scale.set(1, 2, 3);
    mesh.updateMatrixWorld(true);
    const entry = renderer.ensurePipeline(mesh.material, new Scene(), false, false);
    renderer.packUniforms(entry, mesh, camera(), 0);

    const floats = new Float32Array(bytesOf(entry.ringBuffer).buffer);
    const base = offsetOf(entry, "normalMatrix") / 4;
    const expected = new Matrix3().getNormalMatrix(mesh.matrixWorld).elements;
    for (let column = 0; column < 3; column++) {
      for (let row = 0; row < 3; row++) {
        expect(floats[base + column * 4 + row]).toBeCloseTo(expected[column * 3 + row], 5);
      }
    }
  });

  /**
   * The WebGPU renderer writes every uniform through a `Float32Array`, so an
   * `int` or `uint` uniform reaches the shader as the bits of a float.
   *
   * @canon bug-webgpu-uploads-an-integer-uniform-as-float-bits
   */
  it.fails("uploads an int uniform as an integer on WebGPU", () => {
    const { device, canvas, bytesOf } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const material = new MeshBasicMaterial();
    material.fragmentNode = (b) => vec4(b.materialUniform("count", "int", () => 7).toFloat(), 0, 0, 1);
    const mesh = new Mesh(new PlaneGeometry(), material);
    mesh.updateMatrixWorld(true);
    const entry = renderer.ensurePipeline(material, new Scene(), false, false);
    renderer.packUniforms(entry, mesh, camera(), 0);

    const ints = new Int32Array(bytesOf(entry.ringBuffer).buffer);
    expect(ints[offsetOf(entry, "count") / 4]).toBe(7);
  });

  /**
   * The WebGL renderer has no case for `uint` or the `uvec` types, so it never
   * uploads them and the shader reads zero.
   *
   * @canon bug-webgl-never-uploads-an-unsigned-uniform
   */
  it.fails("uploads a uint and a uvec2 uniform on WebGL", () => {
    const { renderer, calls } = stubWebGl();
    renderer.setUniform({ name: "a" }, "uint", 5);
    renderer.setUniform({ name: "b" }, "uvec2", [5, 6]);
    expect(calls.map((c) => c.name)).toEqual(["uniform1ui", "uniform2ui"]);
  });

  /**
   * The WebGPU renderer reads a normalized `Uint8Array` `vec4` as `unorm8x4`
   * with a 4-byte stride.
   *
   * @canon spec-a-vertex-attribute-reaches-the-shader-as-its-declared-type
   */
  it("reads a normalized byte attribute as unorm8x4 on WebGPU", () => {
    const { device, canvas, pipelines } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any);
    const geometry = new PlaneGeometry();
    geometry.setAttribute("tint", new BufferAttribute(new Uint8Array(4 * 4).fill(255), 4, true));
    const material = new MeshBasicMaterial();
    material.fragmentNode = (b) => {
      const tint = b.varying("tint", "vec4");
      return tint;
    };
    material.vertexNode = (b) => {
      b.varying("tint", "vec4").assign(b.attribute("tint", "vec4"));
      return b.projectionMatrix.mul(b.viewMatrix.mul(b.modelMatrix.mul(vec4(b.position, 1))));
    };
    const scene = new Scene();
    scene.add(new Mesh(geometry, material));
    renderer.render(scene, camera());

    const tint = pipelines[0].vertex.buffers.find((buffer: any) =>
      buffer.attributes.some((a: any) => a.format === "unorm8x4" || a.format === "float32x4"),
    );
    expect(tint).toMatchObject({ arrayStride: 4, attributes: [{ format: "unorm8x4" }] });
  });

  /**
   * Meshes that share a material and hold an attribute in different formats
   * each draw with a pipeline that reads it in its own, whether the formats
   * differ by array type or by width, and a mesh's attributes that the
   * program never reads do not make another pipeline.
   *
   * @canon spec-a-vertex-attribute-reaches-the-shader-as-its-declared-type
   */
  it("draws meshes that hold an attribute in different formats with a pipeline each on WebGPU", () => {
    const { device, canvas, pipelines, passes } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any);
    const material = new MeshBasicMaterial();
    material.fragmentNode = (b) => b.varying("tint", "vec4");
    material.vertexNode = (b) => {
      b.varying("tint", "vec4").assign(b.attribute("tint", "vec4"));
      return b.projectionMatrix.mul(b.viewMatrix.mul(b.modelMatrix.mul(vec4(b.position, 1))));
    };
    const tints = [
      new BufferAttribute(new Uint8Array(16).fill(255), 4, true),
      new BufferAttribute(new Float32Array(16), 4),
      new BufferAttribute(new Float32Array(12), 3),
      new BufferAttribute(new Float32Array(16), 4),
    ];
    const scene = new Scene();
    tints.forEach((tint, i) => {
      const geometry = new PlaneGeometry();
      geometry.setAttribute("tint", tint);
      // No program reads this one, and it differs between meshes.
      geometry.setAttribute(
        "unused",
        new BufferAttribute(i % 2 ? new Uint8Array(4) : new Float32Array(4), 4, i % 2 === 1),
      );
      scene.add(new Mesh(geometry, material));
    });
    renderer.render(scene, camera());

    const formatOf = (pipeline: any) =>
      pipeline.vertex.buffers.flatMap((b: any) => b.attributes.map((a: any) => a.format));
    const used = passes.map((pass) => pass.calls.find((c) => c.name === "setPipeline")!.args[0]);
    expect(used.map(formatOf)).toEqual([
      ["unorm8x4", "float32x3"],
      ["float32x4", "float32x3"],
      ["float32x3", "float32x3"],
      ["float32x4", "float32x3"],
    ]);
    expect(used[3]).toBe(used[1]);
    expect(pipelines).toHaveLength(3);
  });

  /**
   * The WebGPU renderer draws a mesh's whole geometry, ignoring the slice its
   * `drawRange` selects.
   *
   * @canon bug-webgpu-ignores-the-draw-range
   */
  it.fails("draws only the slice a mesh's drawRange selects on WebGPU", () => {
    const { device, canvas, passes } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any);
    const mesh = new Mesh(new PlaneGeometry(), new MeshBasicMaterial());
    mesh.drawRange = { start: 3, count: 3 };
    const scene = new Scene();
    scene.add(mesh);
    renderer.render(scene, camera());

    const draw = passes[0].calls.find((c) => c.name === "drawIndexed")!;
    expect(draw.args[0]).toBe(3);
    expect(draw.args[2]).toBe(3);
  });

  /**
   * The WebGPU renderer takes no render target and has no `readPixels`, so it
   * can only draw to its canvas.
   *
   * @canon bug-webgpu-has-no-render-target
   */
  it.fails("draws into a render target and reads its pixels back on WebGPU", () => {
    const { device, canvas } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    expect(renderer.render.length).toBeGreaterThanOrEqual(3);
    expect(typeof renderer.readPixels).toBe("function");
  });

  /**
   * Each of 65 draws of one material reads its own model matrix, so the
   * frame holds more draws than the ring held at first.
   *
   * @canon spec-a-webgpu-draw-keeps-its-own-uniforms-however-many-draws-a-frame-has
   */
  it("draws each of 65 meshes with its own model matrix on WebGPU", () => {
    const { device, canvas, passes, bytesOf } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const material = new MeshBasicMaterial();
    const geometry = new PlaneGeometry();
    const scene = new Scene();
    for (let i = 0; i < 65; i++) {
      const mesh = new Mesh(geometry, material);
      mesh.position.x = i;
      scene.add(mesh);
    }
    renderer.render(scene, camera());

    const entry = [...renderer.pipelines.get(material).values()][0];
    const floats = new Float32Array(bytesOf(entry.ringBuffer).buffer);
    const translation = offsetOf(entry, "modelMatrix") / 4 + 12;
    const seen = passes.map((pass) => {
      const [offset] = pass.calls.find((c) => c.name === "setBindGroup" && c.args[0] === 0)!.args[2];
      return floats[offset / 4 + translation];
    });
    expect(seen).toEqual([...Array(65).keys()]);
  });

  /**
   * A program's uniform ring shrinks again once a frame needs far fewer
   * slots, and the ring of an entry a rebuild replaces is freed.
   *
   * @canon spec-a-webgpu-renderer-frees-the-uniform-buffers-it-no-longer-uses
   */
  it("frees a uniform ring that a frame no longer fills, and one a rebuild replaces on WebGPU", () => {
    const { device, canvas, destroyed } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const material = new MeshBasicMaterial();
    const geometry = new PlaneGeometry();
    const scene = new Scene();
    const meshes = Array.from({ length: 1000 }, () => new Mesh(geometry, material));
    for (const mesh of meshes) scene.add(mesh);
    renderer.render(scene, camera());
    const entry = [...renderer.pipelines.get(material).values()][0];
    const full = entry.ringBuffer;
    expect(entry.slots).toBeGreaterThanOrEqual(1000);

    for (const mesh of meshes.slice(3)) scene.remove(mesh);
    renderer.render(scene, camera());
    expect(entry.slots).toBeLessThan(1000);
    expect(destroyed.has(full)).toBe(true);

    const small = entry.ringBuffer;
    material.needsUpdate = true;
    renderer.render(scene, camera());
    expect(destroyed.has(small)).toBe(true);
  });

  /**
   * The WebGPU renderer numbers sampler bindings in the order the material
   * registers samplers, counting ones it never samples, where the compiler
   * numbers them in the order the graph samples them, so a draw reads its
   * textures through the wrong samplers or fails to bind.
   *
   * @canon bug-webgpu-numbers-samplers-unlike-the-compiler
   */
  it.fails("numbers texture and sampler bindings as the compiled WGSL does on WebGPU", () => {
    const { device, canvas } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const material = new MeshBasicMaterial();
    const texture = () => new DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
    material.fragmentNode = (b) => {
      b.sampler("unused", texture);
      const first = b.sampler("first", texture);
      const second = b.sampler("second", texture);
      return second.texture(vec2(0.5, 0.5)).add(first.texture(vec2(0.5, 0.5)));
    };
    const entry = renderer.ensurePipeline(material, new Scene(), false, false);
    const wgsl = compileWgsl.fragment(entry.program.fragmentRoot);

    const declared = (group: number) =>
      [...wgsl.matchAll(new RegExp(`@group\\(${group}\\) @binding\\((\\d+)\\) var (\\w+)`, "g"))].map(
        (m) => `${m[2]}@${m[1]}`,
      );
    const bound = (bindings: { name: string; binding: number }[], suffix: string) =>
      bindings.map((b) => `${b.name}${suffix}@${b.binding}`);
    const samplerSuffix = declared(2)[0].replace(/^second|@\d+$/g, "");
    expect(bound(entry.textureBindings, "").sort()).toEqual(declared(1).sort());
    expect(bound(entry.samplerBindings, samplerSuffix).sort()).toEqual(declared(2).sort());
  });

  /**
   * The WebGPU renderer makes every texture and sampler binding visible to the
   * fragment stage alone, so a texture sampled in `positionNode` cannot bind.
   *
   * @canon bug-webgpu-binds-textures-to-the-fragment-stage-only
   */
  it.fails("binds a texture sampled in the vertex stage to the vertex stage on WebGPU", () => {
    const { device, canvas, layouts } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const height = new DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
    const material = new MeshBasicMaterial();
    material.positionNode = (b) => b.position.add(b.sampler("height", () => height).textureLod(b.uv, float(0)).xyz);
    renderer.ensurePipeline(material, new Scene(), false, false);

    const textureLayout = layouts.find((l) => l.entries.some((e: any) => e.texture))!;
    expect(textureLayout.entries[0].visibility & 1).toBe(1);
  });

  /**
   * The WebGPU renderer uploads only an `ArrayBufferView` image: a texture
   * holding an image element or bitmap becomes a 1×1 texture with nothing
   * written to it.
   *
   * @canon bug-webgpu-never-uploads-an-image-source
   */
  it.fails("uploads a texture whose image is an image source on WebGPU", () => {
    const { device, canvas, textures, queue } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const bitmap = { width: 2, height: 2 } as unknown as ImageBitmap;
    renderer.ensurePipeline(sampling(new Texture(bitmap)), new Scene(), false, false);

    expect(textures[0]).toMatchObject({ width: 2, height: 2 });
    expect(queue.map((c) => c.name)).toContain("copyExternalImageToTexture");
  });

  /**
   * The WebGPU renderer sizes a geometry's vertex buffer at its first upload
   * and writes a grown attribute into it unchanged, past its end.
   *
   * @canon bug-webgpu-never-grows-a-geometry-buffer
   */
  it.fails("grows a vertex buffer for an attribute whose array grew on WebGPU", () => {
    const { device, canvas } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(new Float32Array(9), 3));
    renderer.ensureGeometryBuffers(geometry);
    geometry.attributes.position.setArray(new Float32Array(18));

    expect(() => renderer.ensureGeometryBuffers(geometry)).not.toThrow();
  });

  /**
   * The WebGPU renderer never reads `geometry.index.needsUpdate`, so changed
   * indices are not uploaded unless a vertex attribute changed too.
   *
   * @canon bug-webgpu-ignores-a-changed-index
   */
  it.fails("uploads a changed index on the next render on WebGPU", () => {
    const { device, canvas, bufferWrites } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const geometry = new PlaneGeometry();
    const buffers = renderer.ensureGeometryBuffers(geometry);
    const before = bufferWrites.filter((w) => w.buffer === buffers.index).length;
    (geometry.index!.array as Uint16Array).reverse();
    geometry.index!.needsUpdate = true;
    renderer.ensureGeometryBuffers(geometry);

    expect(bufferWrites.filter((w) => w.buffer === buffers.index).length).toBe(before + 1);
  });

  /**
   * Both renderers cache a geometry's buffers by attribute name, so an
   * attribute replaced by a new object after the first render — what
   * `LineSegmentsGeometry.setPositions` does — is never uploaded.
   *
   * @canon bug-a-replaced-attribute-keeps-its-old-data
   */
  it.fails("uploads an attribute replaced after the first render", () => {
    const { device, canvas, bufferWrites } = stubDevice();
    const gpu = new WebGPURenderer(canvas, device as any) as any;
    const { renderer: gl, calls } = stubWebGl();
    const geometry = new LineSegmentsGeometry();
    geometry.setPositions([0, 0, 0, 1, 0, 0]);
    const scene = new Scene();
    scene.add(new LineSegments2(geometry, new Line2NodeMaterial()));
    gpu.render(scene, camera());
    gl.render(scene, camera());

    geometry.setPositions([0, 0, 0, 0, 1, 0]);
    const gpuWrites = bufferWrites.length;
    const glUploads = calls.filter((c) => c.name === "bufferData" || c.name === "bufferSubData").length;
    gpu.render(scene, camera());
    gl.render(scene, camera());

    expect(
      bufferWrites
        .slice(gpuWrites)
        .some((w) => w.buffer === gpu.geometryBuffers.get(geometry).attributes.get("instanceEnd")),
    ).toBe(true);
    expect(calls.filter((c) => c.name === "bufferData" || c.name === "bufferSubData").length).toBeGreaterThan(
      glUploads,
    );
  });

  /**
   * A renderer clears `needsUpdate` once it uploads a texture, so a second
   * renderer drawing the same texture never sees the change.
   *
   * @canon bug-the-first-renderer-consumes-needs-update
   */
  it.fails("shows a changed texture in every renderer that draws it", () => {
    const first = stubDevice();
    const second = stubDevice();
    const a = new WebGPURenderer(first.canvas, first.device as any) as any;
    const b = new WebGPURenderer(second.canvas, second.device as any) as any;
    const texture = new DataTexture(new Uint8Array([0, 0, 220, 255]), 1, 1);
    const material = sampling(texture);
    const scene = new Scene();
    a.ensurePipeline(material, scene, false, false);
    b.ensurePipeline(material, scene, false, false);

    texture.image = new Uint8Array([220, 0, 0, 255]);
    texture.needsUpdate = true;
    a.ensurePipeline(material, scene, false, false);
    b.ensurePipeline(material, scene, false, false);

    expect(second.textureWrites).toHaveLength(2);
  });

  /**
   * A rebuild flagged by `needsUpdate` rebuilds only the program of the first
   * kind of mesh drawn after it, and clears the flag, so a material shared by
   * a `Mesh` and an `InstancedMesh` keeps the stale program for the other.
   *
   * @canon bug-a-rebuild-reaches-one-signature-of-a-shared-material
   */
  it.fails("rebuilds the program of every kind of mesh after a precision change on WebGL", () => {
    const { renderer, calls } = stubWebGl();
    const material = new MeshBasicMaterial();
    const scene = new Scene();
    scene.add(new Mesh(new PlaneGeometry(), material));
    scene.add(new InstancedMesh(new PlaneGeometry(), material, 1));
    renderer.render(scene, camera());

    material.precision = "mediump";
    const before = calls.length;
    renderer.render(scene, camera());
    const sources = calls
      .slice(before)
      .filter((c) => c.name === "shaderSource")
      .map((c) => c.args[1] as string);
    expect(sources.filter((s) => s.includes("precision mediump float"))).toHaveLength(4);
  });

  /**
   * The WebGL renderer gives the `resolution` uniform the canvas's drawing
   * buffer size even while it draws into a smaller render target, so a line
   * drawn there is the wrong width.
   *
   * @canon bug-line-resolution-ignores-the-render-target
   */
  it.fails("gives a line the render target's resolution when drawing into one on WebGL", () => {
    const { renderer, calls } = stubWebGl();
    const geometry = new LineSegmentsGeometry();
    geometry.setPositions([0, 0, 0, 1, 0, 0]);
    const scene = new Scene();
    scene.add(new LineSegments2(geometry, new Line2NodeMaterial()));
    renderer.render(scene, camera(), new WebGLRenderTarget(8, 4));

    const resolution = calls.find((c) => c.name === "uniform2f" && /resolution/.test(c.args[0].name))!;
    expect(resolution.args.slice(1)).toEqual([8, 4]);
  });

  /**
   * `Line2NodeMaterial` compiles its `opacity` as a literal, so changing it
   * after the first render has no effect.
   *
   * @canon bug-line-opacity-is-compiled-as-a-literal
   */
  it.fails("shows a line's opacity changed after its first render", () => {
    const material = new Line2NodeMaterial({ opacity: 0.5, transparent: true });
    const program = material.build(new Scene());
    material.needsUpdate = false;
    material.opacity = 0.25;

    const live = program.uniforms.some((u) => u.value?.({} as any) === 0.25);
    expect(live || material.needsUpdate).toBe(true);
  });

  /**
   * The WebGPU renderer builds every pipeline with no blend state, a depth
   * test and depth writes, whatever the material's `transparent`, `blending`,
   * `depthTest` and `depthWrite` ask for.
   *
   * @canon bug-webgpu-ignores-the-material-blend-and-depth-state
   */
  it.fails("blends a transparent material and honours depthTest and depthWrite on WebGPU", () => {
    const { device, canvas, pipelines } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const material = new MeshBasicMaterial({ transparent: true, opacity: 0.5 });
    material.depthTest = false;
    material.depthWrite = false;
    renderer.ensurePipeline(material, new Scene(), false, false);

    expect(pipelines[0].fragment.targets[0].blend).toBeDefined();
    expect(pipelines[0].depthStencil).toMatchObject({ depthWriteEnabled: false, depthCompare: "always" });
  });

  /**
   * The WebGPU renderer clears in the first draw's render pass, so a scene
   * with nothing to draw leaves the canvas as it was.
   *
   * @canon bug-webgpu-leaves-an-empty-scene-uncleared
   */
  it.fails("clears the canvas when the scene draws nothing on WebGPU", () => {
    const { device, canvas, passes } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any);
    renderer.render(new Scene(), camera());

    expect(passes.map((p) => p.descriptor.colorAttachments[0].loadOp)).toEqual(["clear"]);
  });

  /**
   * Both renderers ignore `DataTexture.type`, so a `Float32Array` image is
   * uploaded as unsigned bytes.
   *
   * @canon bug-a-float-texture-is-uploaded-as-bytes
   */
  it.fails("uploads a float data texture as floats", () => {
    const { device, canvas, textures } = stubDevice();
    const gpu = new WebGPURenderer(canvas, device as any) as any;
    const { renderer: gl, gl: context, calls } = stubWebGl();
    const texture = () => new DataTexture(new Float32Array([1, 0.5, 0.25, 1]), 1, 1, 1, RGBAFormat, FloatType);
    gpu.ensurePipeline(sampling(texture()), new Scene(), false, false);
    const scene = new Scene();
    scene.add(new Mesh(new PlaneGeometry(), sampling(texture())));
    gl.render(scene, camera());

    expect(textures[0].format).toMatch(/float$/);
    expect(calls.find((c) => c.name === "texImage2D")!.args[7]).toBe(context.FLOAT);
  });

  /**
   * The WebGL renderer never sets the unit of a sampler that has no texture,
   * so it reads unit 0, the texture of another sampler.
   *
   * @canon bug-webgl-leaves-a-textureless-sampler-on-unit-0
   */
  it.fails("gives a sampler with no texture a unit of its own on WebGL", () => {
    const { renderer, calls } = stubWebGl();
    const texture = new DataTexture(new Uint8Array([220, 0, 0, 255]), 1, 1);
    const material = new MeshBasicMaterial();
    material.fragmentNode = (b) => {
      const present = b.sampler("present", () => texture);
      const missing = b.sampler("missing", () => null);
      return present.texture(vec2(0.5, 0.5)).add(missing.texture(vec2(0.5, 0.5)));
    };
    const scene = new Scene();
    scene.add(new Mesh(new PlaneGeometry(), material));
    renderer.render(scene, camera());

    const unitOf = (name: string) => calls.find((c) => c.name === "uniform1i" && c.args[0].name === name)?.args[1];
    expect(unitOf("missing")).toBeDefined();
    expect(unitOf("missing")).not.toBe(unitOf("present"));
  });

  /**
   * Both renderers ignore `scene.background` and clear to the renderer's clear
   * colour.
   *
   * @canon bug-render-ignores-the-scene-background
   */
  it.fails("clears to the scene's background colour", () => {
    const { device, canvas, passes } = stubDevice();
    const gpu = new WebGPURenderer(canvas, device as any);
    const { renderer: gl, calls } = stubWebGl();
    const scene = new Scene();
    scene.background = new Color(1, 0, 0);
    scene.add(new Mesh(new PlaneGeometry(), new MeshBasicMaterial()));
    gpu.render(scene, camera());
    gl.render(scene, camera());

    expect(passes[0].descriptor.colorAttachments[0].clearValue).toMatchObject({ r: 1, g: 0, b: 0 });
    expect(calls.find((c) => c.name === "clearColor")!.args.slice(0, 3)).toEqual([1, 0, 0]);
  });

  /**
   * Both renderers draw meshes in scene-graph order, so a transparent mesh
   * drawn before a farther one hides it instead of blending over it.
   *
   * @canon bug-transparent-meshes-draw-in-scene-graph-order
   */
  it.fails("draws transparent meshes back to front", () => {
    const { device, canvas, passes, bytesOf } = stubDevice();
    const gpu = new WebGPURenderer(canvas, device as any) as any;
    const { renderer: gl, calls } = stubWebGl();
    const material = new MeshBasicMaterial({ transparent: true, opacity: 0.5 });
    const geometry = new PlaneGeometry();
    const scene = new Scene();
    for (const z of [1, -1]) {
      const mesh = new Mesh(geometry, material);
      mesh.position.z = z;
      scene.add(mesh);
    }
    gpu.render(scene, camera());
    gl.render(scene, camera());

    const entry = [...gpu.pipelines.get(material).values()][0];
    const floats = new Float32Array(bytesOf(entry.ringBuffer).buffer);
    const depth = offsetOf(entry, "modelMatrix") / 4 + 14;
    const gpuOrder = passes.map((pass) => {
      const [offset] = pass.calls.find((c) => c.name === "setBindGroup" && c.args[0] === 0)!.args[2];
      return floats[offset / 4 + depth];
    });
    const glOrder = calls
      .filter((c) => c.name === "uniformMatrix4fv" && /modelMatrix/.test(c.args[0].name))
      .map((c) => c.args[2][14]);
    expect(gpuOrder).toEqual([-1, 1]);
    expect(glOrder).toEqual([-1, 1]);
  });

  /**
   * The WebGPU renderer writes a changed attribute whole, from byte 0, ignoring
   * the slice its `updateRange` selects.
   *
   * @canon bug-webgpu-ignores-an-attribute-update-range
   */
  it.fails("uploads only the updateRange of a changed attribute on WebGPU", () => {
    const { device, canvas, bufferWrites } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(new Float32Array(9), 3));
    const buffers = renderer.ensureGeometryBuffers(geometry);
    const before = bufferWrites.length;
    geometry.attributes.position.updateRange = { offset: 3, count: 3 };
    geometry.attributes.position.needsUpdate = true;
    renderer.ensureGeometryBuffers(geometry);

    const writes = bufferWrites.slice(before).filter((w) => w.buffer === buffers.attributes.get("position"));
    expect(writes.map((w) => w.offset)).toEqual([12]);
  });

  /**
   * The builder's `position` and `normal` read the attributes in object space
   * in the vertex stage, but the world-space `positionWorld` and `normalWorld`
   * varyings in the fragment stage.
   *
   * @canon bug-position-and-normal-read-world-space-in-the-fragment-stage
   */
  it.fails("reads position and normal in object space in the fragment stage", () => {
    const material = new MeshBasicMaterial();
    material.fragmentNode = (b) => vec4(b.position.add(b.normal), 1);
    const program = material.build(new Scene());

    const read = collectNodes(program.fragmentRoot).varyings;
    const names = program.varyings.filter((v) => read.has(v.node)).map((v) => v.name);
    expect(names).toHaveLength(2);
    expect(names).not.toContain("positionWorld");
    expect(names).not.toContain("normalWorld");
  });

  /**
   * The WebGL renderer reads the clear colour into a new array on every
   * frame, with `Color.toArray()`.
   *
   * @canon bug-webgl-render-allocates-the-clear-colour-per-frame
   */
  it.fails("renders a frame without allocating the clear colour on WebGL", () => {
    const { renderer } = stubWebGl();
    const toArray = vi.spyOn(Color.prototype, "toArray");
    renderer.render(new Scene(), camera());
    expect(toArray).not.toHaveBeenCalled();
  });

  /**
   * The WebGL renderer builds a new callback for `traverseVisible` on every
   * frame.
   *
   * @canon bug-webgl-render-allocates-a-traversal-closure-per-frame
   */
  it.fails("renders every frame with one traversal callback on WebGL", () => {
    const { renderer } = stubWebGl();
    const scene = new Scene();
    const callbacks: unknown[] = [];
    const traverse = scene.traverseVisible.bind(scene);
    scene.traverseVisible = (callback) => {
      callbacks.push(callback);
      traverse(callback);
    };
    renderer.render(scene, camera());
    renderer.render(scene, camera());
    expect(callbacks[1]).toBe(callbacks[0]);
  });

  /**
   * The WebGL renderer lists a geometry's attributes with `Object.values` on
   * every draw, to ask whether any needs an update.
   *
   * @canon bug-webgl-draw-allocates-the-attribute-list-per-draw
   */
  it.fails("draws a mesh without listing its attributes on WebGL", () => {
    const { renderer } = stubWebGl();
    const scene = new Scene();
    scene.add(new Mesh(new PlaneGeometry(), new MeshBasicMaterial()));
    renderer.render(scene, camera());
    const values = vi.spyOn(Object, "values");
    renderer.render(scene, camera());
    expect(values).not.toHaveBeenCalled();
  });
});

// A 3×2 single-channel texture: its rows are three bytes long, so the second
// row starts at byte 3, not at the next multiple of four.
const ENTRY_R8UI_ROWS = `
import { WebGLRenderer, Scene, Mesh, PerspectiveCamera, PlaneGeometry,
  MeshBasicMaterial, DataTexture, RedIntegerFormat, UnsignedByteType } from "../scene";
import { float, uvec2, vec4 } from "../rmsl";
globalThis.__rmslR8UIRowsRun = () => {
  const canvas = document.createElement("canvas");
  canvas.width = 16;
  canvas.height = 16;
  const renderer = new WebGLRenderer(canvas, { antialias: false });
  renderer.setClearColor(0x000000);
  const texture = new DataTexture(new Uint8Array([10, 20, 30, 40, 50, 60]), 3, 2, 1, RedIntegerFormat, UnsignedByteType);
  const material = new MeshBasicMaterial();
  material.fragmentNode = (b) => {
    const data = b.sampler("data", "usampler2D", () => texture);
    return vec4(data.texture(uvec2(0, 1)).r.toFloat().div(float(255)), 0, 0, 1);
  };
  const scene = new Scene();
  scene.add(new Mesh(new PlaneGeometry(2, 2), material));
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);
  renderer.render(scene, camera);
  const gl = renderer.gl;
  const pixels = new Uint8Array(4);
  gl.readPixels(8, 8, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  return { r: pixels[0], error: gl.getError() };
};
`;

describe.skipIf(!GPU_ENABLED)("known bugs of the scene library on a real driver", () => {
  /**
   * The WebGL renderer uploads a single-channel integer texture under the
   * default unpack alignment of four, so a tightly packed image whose width is
   * not a multiple of four is rejected and the texture reads zero.
   *
   * @canon bug-webgl-rejects-a-narrow-r8ui-texture
   */
  it.fails(
    "reads the second row of a three-texel-wide R8UI texture on WebGL",
    async () => {
      const result = await runInGpuPage(ENTRY_R8UI_ROWS, "__rmslR8UIRowsRun", new URL(".", import.meta.url).pathname);
      expect(result.error).toBe(0);
      expect(result.r).toBe(40);
    },
    60_000,
  );
});

afterAll(async () => {
  await releaseGpu();
});
