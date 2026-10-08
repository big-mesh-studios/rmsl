import { afterAll, afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { vec4 } from "../rmsl";
import {
  AmbientLight,
  BufferAttribute,
  BufferGeometry,
  Color,
  DataTexture,
  DirectionalLight,
  FloatType,
  InstancedMesh,
  Line2NodeMaterial,
  LineSegments2,
  LineSegmentsGeometry,
  Mesh,
  MeshBasicMaterial,
  MeshLambertMaterial,
  PerspectiveCamera,
  PlaneGeometry,
  PointLight,
  RGBAFormat,
  Scene,
  Texture,
  WebGLRenderer,
  WebGLRenderTarget,
  WebGPURenderer,
} from "../scene";
import { GPU_ENABLED, releaseGpu } from "../testing/gpu";
import { runInGpuPage } from "../testing/browser";
import { camera, offsetOf, sampling, stubDevice, stubWebGl } from "./scene-stubs";

/** The value a built program's uniform of that name holds now. */
function uniformValue(material: MeshLambertMaterial, scene: Scene, name: string): unknown {
  const binding = material.build(scene).uniforms.find((u) => u.name === name)!;
  return binding.value!({} as any);
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

// A three-texel-wide single-channel integer texture: its second row starts three bytes in, which the default unpack alignment of four misreads.
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

describe("a scene renderer manages what it uploads", () => {
  /**
   * @canon spec-render-clears-to-the-scene-background
   */
  it("clears to the scene's background colour", () => {
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
   * @canon spec-a-renderer-draws-transparent-meshes-back-to-front
   */
  it("draws transparent meshes back to front", () => {
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
   * @canon spec-a-scene-that-draws-nothing-still-clears
   */
  it("clears the canvas when the scene draws nothing on WebGPU", () => {
    const { device, canvas, passes } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any);
    renderer.render(new Scene(), camera());

    expect(passes.map((p) => p.descriptor.colorAttachments[0].loadOp)).toEqual(["clear"]);
  });

  /**
   * @canon spec-a-webgl-renderer-allocates-nothing-per-frame
   */
  it("renders a frame without allocating the clear colour on WebGL", () => {
    const { renderer } = stubWebGl();
    const toArray = vi.spyOn(Color.prototype, "toArray");
    renderer.render(new Scene(), camera());
    expect(toArray).not.toHaveBeenCalled();
  });

  /**
   * @canon spec-a-webgl-renderer-allocates-nothing-per-frame
   */
  it("renders every frame with one traversal callback on WebGL", () => {
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
   * @canon spec-a-texture-uploads-whatever-holds-its-image
   */
  it("uploads a texture whose image is an image source on WebGPU", () => {
    const { device, canvas, textures, queue } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const bitmap = { width: 2, height: 2 } as unknown as ImageBitmap;
    renderer.ensurePipeline(sampling(new Texture(bitmap)), new Scene(), false, false);

    expect(textures[0]).toMatchObject({ width: 2, height: 2 });
    expect(queue.map((c) => c.name)).toContain("copyExternalImageToTexture");
  });

  /**
   * @canon spec-a-grown-attribute-gets-a-buffer-that-holds-it
   */
  it("grows a vertex buffer for an attribute whose array grew on WebGPU", () => {
    const { device, canvas } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(new Float32Array(9), 3));
    renderer.ensureGeometryBuffers(geometry);
    geometry.attributes.position.setArray(new Float32Array(18));

    expect(() => renderer.ensureGeometryBuffers(geometry)).not.toThrow();
  });

  /**
   * @canon spec-a-changed-index-uploads-on-the-next-render
   */
  it("uploads a changed index on the next render on WebGPU", () => {
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
   * @canon spec-a-data-texture-uploads-in-the-type-it-names
   */
  it("uploads a float data texture as floats", () => {
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
   * @canon spec-a-data-texture-uploads-in-the-type-it-names
   */
  it("reads a float texture's nearest texel on WebGL when the context cannot filter floats", () => {
    const { renderer, gl, calls } = stubWebGl({ getExtension: () => null });
    const scene = new Scene();
    const texture = new DataTexture(new Float32Array([1, 0.5, 0.25, 1]), 1, 1, 1, RGBAFormat, FloatType);
    scene.add(new Mesh(new PlaneGeometry(), sampling(texture)));
    renderer.render(scene, camera());

    const filters = calls.filter(
      (c) => c.name === "texParameteri" && (c.args[1] === gl.TEXTURE_MIN_FILTER || c.args[1] === gl.TEXTURE_MAG_FILTER),
    );
    const named = (value: number) => (value === gl.NEAREST ? "nearest" : value === gl.LINEAR ? "linear" : value);
    expect(new Set(filters.map((c) => named(c.args[2])))).toEqual(new Set(["nearest"]));
  });

  /**
   * @canon spec-a-material-reads-any-sampler-type
   */
  it("reads the second row of a three-texel-wide R8UI texture on WebGL", async () => {
    const result = await runInGpuPage(ENTRY_R8UI_ROWS, "__rmslR8UIRowsRun", new URL(".", import.meta.url).pathname);
    expect(result.error).toBe(0);
    expect(result.r).toBe(40);
  }, 60_000);

  /**
   * @canon spec-a-change-raises-a-version-every-renderer-reads
   */
  it("shows a changed texture in every renderer that draws it", () => {
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
   * @canon spec-a-change-raises-a-version-every-renderer-reads
   */
  it("rebuilds the program of every kind of mesh after a precision change on WebGL", () => {
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
   * @canon spec-a-replaced-attribute-uploads-again
   */
  it("uploads an attribute replaced after the first render", () => {
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
   * @canon spec-a-webgl-renderer-allocates-nothing-per-frame
   */
  it("draws a mesh without listing its attributes on WebGL", () => {
    const { renderer } = stubWebGl();
    const scene = new Scene();
    scene.add(new Mesh(new PlaneGeometry(), new MeshBasicMaterial()));
    renderer.render(scene, camera());
    const values = vi.spyOn(Object, "values");
    renderer.render(scene, camera());
    expect(values).not.toHaveBeenCalled();
  });

  /**
   * The vertex stage reads only the matrices and the fragment stage only the
   * material's tint, yet both declare every uniform at the same offsets, so
   * one buffer serves both.
   *
   * @canon spec-the-webgpu-renderer-declares-one-uniform-struct-in-both-stages
   */
  it("declares the same uniform struct in the vertex and fragment WGSL", () => {
    const { device, canvas, modules } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const material = new MeshBasicMaterial();
    material.fragmentNode = (b) =>
      vec4(
        b.materialUniform("tint", "vec3", () => [1, 0, 0]),
        1,
      );
    renderer.ensurePipeline(material, new Scene(), false, false);

    const struct = (code: string) => /struct \w+ \{[^}]*\}/.exec(code)![0];
    expect(modules).toHaveLength(2);
    expect(struct(modules[0].code)).toContain("tint");
    expect(struct(modules[1].code)).toContain("modelMatrix");
    expect(struct(modules[0].code)).toBe(struct(modules[1].code));
  });

  /**
   * Growing a target between two renders frees its old framebuffer and
   * allocates colour and depth storage at the new size.
   *
   * @canon spec-a-render-target-takes-its-new-size-on-the-next-render
   */
  it("redraws a render target at the size it was given since the last render", () => {
    const { renderer, gl, calls } = stubWebGl();
    const target = new WebGLRenderTarget(8, 4);
    renderer.render(new Scene(), camera(), target);
    target.width = 16;
    target.height = 8;
    const before = calls.length;
    renderer.render(new Scene(), camera(), target);

    const after = calls.slice(before);
    expect(after.map((c) => c.name)).toContain("deleteFramebuffer");
    expect(after.find((c) => c.name === "texImage2D")!.args.slice(3, 5)).toEqual([16, 8]);
    expect(after.find((c) => c.name === "renderbufferStorage")!.args.slice(1)).toEqual([gl.DEPTH_COMPONENT24, 16, 8]);
    expect(after.find((c) => c.name === "viewport")!.args).toEqual([0, 0, 16, 8]);
  });
});

describe("a node material lights as three.js does", () => {
  /** @canon spec-ambient-lights-sum-into-one-colour */
  it("folds two ambient lights into one uniform holding their sum", () => {
    const scene = new Scene();
    scene.add(new AmbientLight(0xff0000, 0.5));
    scene.add(new AmbientLight(0x0000ff, 2));
    const material = new MeshLambertMaterial();
    const program = material.build(scene);

    expect(program.uniforms.filter((u) => /ambient/i.test(u.name)).map((u) => u.name)).toEqual(["ambientColor"]);
    expect(uniformValue(material, scene, "ambientColor")).toEqual([0.5, 0, 2]);
  });

  /**
   * A directional and a point light each hand the shader their colour already
   * multiplied by their intensity.
   *
   * @canon spec-a-light-uniform-carries-its-colour-times-its-intensity
   */
  it.each([
    ["directional", () => new DirectionalLight(0xff8000, 3), "directionalColor0"],
    ["point", () => new PointLight(0xff8000, 3), "pointColor0"],
  ])("scales a %s light's colour by its intensity", (_kind, light, name) => {
    const scene = new Scene();
    const source = light();
    scene.add(source);
    const [r, g, b] = source.color.toArray();

    const value = uniformValue(new MeshLambertMaterial(), scene, name) as number[];
    expect(value[0]).toBeCloseTo(r * 3, 6);
    expect(value[1]).toBeCloseTo(g * 3, 6);
    expect(value[2]).toBeCloseTo(b * 3, 6);
  });
});

// A frame whose last draw has `depthWrite: false`, then a frame of one far
// plane: the second frame must read the same pixel as the same frame on a
// renderer that drew nothing before it.
const ENTRY_DEPTH_MASK = `
import { WebGLRenderer, Scene, Mesh, PerspectiveCamera, PlaneGeometry, MeshBasicMaterial } from "../scene";
globalThis.__rmslDepthMaskRun = () => {
  const canvas = document.createElement("canvas");
  canvas.width = 16;
  canvas.height = 16;
  const renderer = new WebGLRenderer(canvas, { antialias: false });
  renderer.setClearColor(0x000000);
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 4);
  camera.lookAt(0, 0, 0);
  const plane = (z, color, depthWrite) => {
    const material = new MeshBasicMaterial({ color });
    material.depthWrite = depthWrite;
    const mesh = new Mesh(new PlaneGeometry(2, 2), material);
    mesh.position.z = z;
    return mesh;
  };
  const centre = () => {
    const gl = renderer.gl;
    const pixels = new Uint8Array(4);
    gl.readPixels(8, 8, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    return [pixels[0], pixels[1], pixels[2]];
  };
  const far = new Scene();
  far.add(plane(-2, 0x00ff00, true));
  renderer.render(far, camera);
  const fresh = centre();

  const earlier = new Scene();
  earlier.add(plane(2, 0xff0000, true), plane(3, 0x0000ff, false));
  renderer.render(earlier, camera);
  renderer.render(far, camera);
  return { fresh, afterMaskedDraw: centre() };
};
`;

// A render whose sampler has no texture, after a render whose sampler had one:
// it must read what it reads on a renderer that drew nothing before it.
const ENTRY_TEXTURELESS = `
import { WebGLRenderer, Scene, Mesh, PerspectiveCamera, PlaneGeometry, MeshBasicMaterial, DataTexture } from "../scene";
import { vec2 } from "../rmsl";
globalThis.__rmslTexturelessRun = () => {
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);
  const make = () => {
    const canvas = document.createElement("canvas");
    canvas.width = 16;
    canvas.height = 16;
    const renderer = new WebGLRenderer(canvas, { antialias: false });
    renderer.setClearColor(0x000000);
    return renderer;
  };
  const sampling = (texture) => {
    const material = new MeshBasicMaterial();
    material.fragmentNode = (b) => b.sampler("map", "sampler2D", () => texture).texture(vec2(0.5, 0.5));
    const scene = new Scene();
    scene.add(new Mesh(new PlaneGeometry(2, 2), material));
    return scene;
  };
  const centre = (renderer) => {
    const gl = renderer.gl;
    const pixels = new Uint8Array(4);
    gl.readPixels(8, 8, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    return [pixels[0], pixels[1], pixels[2], pixels[3]];
  };
  const renderer = make();
  renderer.render(sampling(new DataTexture(new Uint8Array([200, 0, 0, 255]), 1, 1)), camera);
  renderer.render(sampling(null), camera);
  const fresh = make();
  fresh.render(sampling(null), camera);
  return { afterTextured: centre(renderer), fresh: centre(fresh) };
};
`;

describe.skipIf(!GPU_ENABLED)("a render depends only on what it is given, on a real driver", () => {
  /**
   * @canon spec-a-render-clears-the-depth-buffer-whatever-the-last-draw-masked
   */
  it("draws a far plane after a frame whose last draw wrote no depth on WebGL", async () => {
    const result = await runInGpuPage(ENTRY_DEPTH_MASK, "__rmslDepthMaskRun", new URL(".", import.meta.url).pathname);
    expect(result.afterMaskedDraw).toEqual(result.fresh);
  }, 60_000);

  /**
   * @canon spec-a-sampler-without-a-texture-reads-black
   */
  it("reads opaque black from a sampler with no texture, after a render that bound one on WebGL", async () => {
    const result = await runInGpuPage(
      ENTRY_TEXTURELESS,
      "__rmslTexturelessRun",
      new URL(".", import.meta.url).pathname,
    );
    expect(result.fresh).toEqual([0, 0, 0, 255]);
    expect(result.afterTextured).toEqual(result.fresh);
  }, 60_000);
});

afterAll(async () => {
  await releaseGpu();
});
