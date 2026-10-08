import { afterAll, afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { vec2, vec4 } from "../rmsl";
import {
  AmbientLight,
  Blending,
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
  NearestFilter,
  PerspectiveCamera,
  PlaneGeometry,
  PointLight,
  RGBAFormat,
  RepeatWrapping,
  Scene,
  Texture,
  WebGLRenderer,
  WebGLRenderTarget,
  WebGPURenderer,
} from "../scene";
import { GPU_ENABLED, releaseGpu } from "../testing/gpu";
import { GL_STATE, runInGpuPage } from "../testing/browser";
import { camera, offsetOf, sampling, stubDevice, stubGl, stubWebGl } from "./scene-stubs";
import { drawToCanvas, GlStateKeeper } from "../backends/glsl/gl-state";

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
   * @canon spec-a-changed-attribute-uploads-only-its-update-range
   */
  it("uploads only the range addUpdateRange marked of a changed attribute on WebGPU", () => {
    const { device, canvas, bufferWrites } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(new Float32Array(9), 3));
    const buffers = renderer.ensureGeometryBuffers(geometry);
    const before = bufferWrites.length;
    geometry.attributes.position.addUpdateRange(3, 3);
    geometry.attributes.position.needsUpdate = true;
    renderer.ensureGeometryBuffers(geometry);

    const writes = bufferWrites.slice(before).filter((w) => w.buffer === buffers.attributes.get("position"));
    expect(writes.map((w) => w.offset)).toEqual([12]);
  });

  /**
   * @canon spec-a-changed-attribute-uploads-only-its-update-range
   */
  it("writes a 16-bit index of a length and a range that are not whole words on WebGPU", () => {
    const { device, canvas, bytesOf } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(new Float32Array(9), 3));
    geometry.setIndex(new BufferAttribute(new Uint16Array([0, 1, 2]), 1));
    const buffers = renderer.ensureGeometryBuffers(geometry);
    expect(Array.from(new Uint16Array(bytesOf(buffers.index).buffer, 0, 3))).toEqual([0, 1, 2]);

    (geometry.index!.array as Uint16Array)[1] = 7;
    geometry.index!.addUpdateRange(1, 1);
    geometry.index!.needsUpdate = true;
    renderer.ensureGeometryBuffers(geometry);
    expect(Array.from(new Uint16Array(bytesOf(buffers.index).buffer, 0, 3))).toEqual([0, 7, 2]);
  });

  /**
   * @canon spec-an-attribute-two-geometries-share-uploads-into-each
   */
  it("uploads an attribute two geometries share once, into the buffer both draw from, on WebGPU", () => {
    const { device, canvas, bufferWrites } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const position = new BufferAttribute(new Float32Array(9), 3);
    const [first, second] = [0, 1].map(() => new BufferGeometry().setAttribute("position", position));
    const firstBuffer = renderer.ensureGeometryBuffers(first).attributes.get("position");
    expect(renderer.ensureGeometryBuffers(second).attributes.get("position")).toBe(firstBuffer);

    (position.array as Float32Array)[3] = 5;
    position.addUpdateRange(3, 3);
    position.needsUpdate = true;
    const before = bufferWrites.length;
    renderer.ensureGeometryBuffers(first);
    renderer.ensureGeometryBuffers(second);
    expect(bufferWrites.slice(before).map((w) => w.offset)).toEqual([12]);
  });

  /**
   * @canon spec-an-attribute-two-geometries-share-uploads-into-each
   */
  it("uploads an attribute two geometries share once, into the buffer both draw from, on WebGL", () => {
    const { renderer, calls } = stubWebGl();
    const position = new BufferAttribute(new Float32Array(9), 3);
    const scene = new Scene();
    for (let i = 0; i < 2; i++) {
      scene.add(new Mesh(new BufferGeometry().setAttribute("position", position), new MeshBasicMaterial()));
    }
    renderer.render(scene, camera());
    (position.array as Float32Array)[3] = 5;
    position.addUpdateRange(3, 3);
    position.needsUpdate = true;
    const before = calls.length;
    renderer.render(scene, camera());

    const frame = calls.slice(before);
    const uploads = frame.filter((c) => c.name === "bufferData" || c.name === "bufferSubData");
    expect(uploads.map((c) => [c.name, c.args[1]])).toEqual([["bufferSubData", 12]]);
    const bound = frame.filter((c) => c.name === "bindBuffer" && c.args[0] === renderer.gl.ARRAY_BUFFER);
    expect(new Set(bound.map((c) => c.args[1])).size).toBe(1);
  });

  /**
   * @canon spec-a-changed-attribute-uploads-only-its-update-range
   */
  it("uploads whole into a renderer that missed ranges another renderer cleared on WebGPU", () => {
    const { device, canvas, bytesOf } = stubDevice();
    const [first, second] = [0, 1].map(() => new WebGPURenderer(canvas, device as any) as any);
    const position = new BufferAttribute(new Float32Array(12), 3);
    const geometry = new BufferGeometry().setAttribute("position", position);
    first.ensureGeometryBuffers(geometry);
    const buffers = second.ensureGeometryBuffers(geometry);
    const array = position.array as Float32Array;
    array[0] = 1;
    position.addUpdateRange(0, 3);
    position.needsUpdate = true;
    first.ensureGeometryBuffers(geometry);
    array[9] = 2;
    position.addUpdateRange(9, 3);
    position.needsUpdate = true;
    second.ensureGeometryBuffers(geometry);

    const held = new Float32Array(bytesOf(buffers.attributes.get("position")).buffer);
    expect([held[0], held[9]]).toEqual([1, 2]);
  });

  /**
   * @canon spec-a-changed-attribute-uploads-only-its-update-range
   */
  it("uploads whole into a renderer that missed ranges another renderer cleared on WebGL", () => {
    const [first, second] = [0, 1].map(() => stubWebGl());
    const position = new BufferAttribute(new Float32Array(12), 3);
    const scene = new Scene();
    scene.add(new Mesh(new BufferGeometry().setAttribute("position", position), new MeshBasicMaterial()));
    first!.renderer.render(scene, camera());
    second!.renderer.render(scene, camera());
    position.addUpdateRange(0, 3);
    position.needsUpdate = true;
    first!.renderer.render(scene, camera());
    position.addUpdateRange(9, 3);
    position.needsUpdate = true;
    const before = second!.calls.length;
    second!.renderer.render(scene, camera());

    const uploads = second!.calls.slice(before).filter((c) => c.name === "bufferData" || c.name === "bufferSubData");
    expect(uploads.map((c) => [c.args[1], c.args[2].length])).toEqual([[0, 12]]);
  });

  /**
   * @canon spec-a-replaced-attribute-uploads-again
   */
  it("uploads whole an attribute that replaced another of its size on WebGPU", () => {
    const { device, canvas, bytesOf } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const geometry = new BufferGeometry().setAttribute("position", new BufferAttribute(new Float32Array(9), 3));
    const buffers = renderer.ensureGeometryBuffers(geometry);
    const replacement = new BufferAttribute(new Float32Array(9).fill(2), 3);
    replacement.addUpdateRange(0, 3);
    geometry.setAttribute("position", replacement);
    renderer.ensureGeometryBuffers(geometry);

    expect(Array.from(new Float32Array(bytesOf(buffers.attributes.get("position")).buffer))).toEqual(Array(9).fill(2));
  });

  /**
   * @canon spec-a-renderer-supplies-the-camera-and-object-uniforms
   */
  it("gives a line the render target's resolution when drawing into one on WebGL", () => {
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
   * @canon spec-wide-lines-follow-three-js
   */
  it("shows a line's opacity changed after its first render", () => {
    const material = new Line2NodeMaterial({ opacity: 0.5, transparent: true });
    const program = material.build(new Scene());
    const version = material.version;
    material.opacity = 0.25;

    const live = program.uniforms.some((u) => u.value?.({} as any) === 0.25);
    expect(live || material.version > version).toBe(true);
  });

  /**
   * @canon spec-a-mesh-draws-the-slice-its-draw-range-selects
   */
  it("draws only the slice a mesh's drawRange selects on WebGPU", () => {
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
   * @canon spec-a-mesh-draws-the-slice-its-draw-range-selects
   */
  it("cuts a draw range to its geometry, and draws nothing for one past its end", () => {
    const { device, canvas, passes } = stubDevice();
    const gpu = new WebGPURenderer(canvas, device as any);
    const { renderer: gl, calls } = stubWebGl();
    const mesh = new Mesh(new PlaneGeometry(), new MeshBasicMaterial());
    const scene = new Scene();
    scene.add(mesh);
    const drawn = () => [
      passes
        .at(-1)!
        .calls.find((c) => c.name === "drawIndexed")
        ?.args.slice(0, 1),
      calls.filter((c) => c.name === "drawElementsInstanced").at(-1)?.args[1],
    ];
    mesh.drawRange = { start: 0, count: 1000 };
    gpu.render(scene, camera());
    gl.render(scene, camera());
    expect(drawn()).toEqual([[6], 6]);

    mesh.drawRange = { start: 400, count: Infinity };
    const before = calls.length;
    gpu.render(scene, camera());
    gl.render(scene, camera());
    expect(passes.at(-1)!.calls.some((c) => c.name === "drawIndexed")).toBe(false);
    expect(calls.slice(before).some((c) => c.name === "drawElementsInstanced")).toBe(false);
  });

  /**
   * @canon spec-a-mesh-draws-the-slice-its-draw-range-selects
   */
  it("draws the range of a geometry with no index and no position uncut", () => {
    const { device, canvas, passes } = stubDevice();
    const gpu = new WebGPURenderer(canvas, device as any);
    const { renderer: gl, calls } = stubWebGl();
    const mesh = new Mesh(new BufferGeometry(), new MeshBasicMaterial());
    mesh.drawRange = { start: 0, count: 3 };
    const scene = new Scene();
    scene.add(mesh);
    gpu.render(scene, camera());
    gl.render(scene, camera());

    expect(passes.at(-1)!.calls.find((c) => c.name === "draw")?.args[0]).toBe(3);
    expect(calls.find((c) => c.name === "drawArraysInstanced")?.args[2]).toBe(3);
  });

  /**
   * @canon spec-an-image-uploads-once-it-has-loaded
   */
  it("takes a VideoFrame's display size as its size", () => {
    const { device, canvas, textures } = stubDevice();
    const gpu = new WebGPURenderer(canvas, device as any);
    const scene = new Scene();
    scene.add(new Mesh(new PlaneGeometry(), sampling(new Texture({ displayWidth: 4, displayHeight: 2 } as any))));
    gpu.render(scene, camera());

    expect(textures.some((t) => t.width === 4 && t.height === 2)).toBe(true);
  });

  /**
   * @canon spec-a-draw-takes-its-material-blend-and-depth-state
   */
  it("blends colour and alpha with the factors three.js uses on both renderers", () => {
    const { device, canvas, pipelines } = stubDevice();
    const gpu = new WebGPURenderer(canvas, device as any);
    const { renderer: gl, gl: context, calls } = stubWebGl();
    const normal = new MeshBasicMaterial({ transparent: true, opacity: 0.5 });
    const additive = new MeshBasicMaterial({ transparent: true, opacity: 0.5 });
    additive.blending = Blending.AdditiveBlending;
    const scene = new Scene();
    scene.add(new Mesh(new PlaneGeometry(), normal), new Mesh(new PlaneGeometry(), additive));
    gpu.render(scene, camera());
    gl.render(scene, camera());

    const blends = pipelines.map((p) => p.fragment.targets[0].blend);
    expect(blends).toContainEqual({
      color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
      alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
    });
    expect(blends).toContainEqual({
      color: { srcFactor: "src-alpha", dstFactor: "one", operation: "add" },
      alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
    });
    const { SRC_ALPHA, ONE_MINUS_SRC_ALPHA, ONE } = context;
    const functions = calls.filter((c) => c.name === "blendFuncSeparate").map((c) => c.args);
    expect(functions).toContainEqual([SRC_ALPHA, ONE_MINUS_SRC_ALPHA, ONE, ONE_MINUS_SRC_ALPHA]);
    expect(functions).toContainEqual([SRC_ALPHA, ONE, ONE, ONE]);
  });

  /**
   * @canon spec-a-draw-takes-its-material-blend-and-depth-state
   */
  it("blends a transparent material and honours depthTest and depthWrite on WebGPU", () => {
    const { device, canvas, pipelines } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any);
    const material = new MeshBasicMaterial({ transparent: true, opacity: 0.5 });
    material.depthTest = false;
    material.depthWrite = false;
    const scene = new Scene();
    scene.add(new Mesh(new PlaneGeometry(), material));
    renderer.render(scene, camera());

    expect(pipelines[0].fragment.targets[0].blend).toBeDefined();
    expect(pipelines[0].depthStencil).toMatchObject({ depthWriteEnabled: false, depthCompare: "always" });

    // A change to the depth state after the first draw is read at the next one, without needsUpdate.
    material.depthTest = true;
    renderer.render(scene, camera());
    expect(pipelines.at(-1).depthStencil).toMatchObject({ depthCompare: "less" });
  });

  /**
   * @canon spec-a-mirrored-mesh-shows-the-faces-its-material-asks-for
   */
  it("winds the front face of a mirrored mesh clockwise on WebGL", () => {
    const { renderer, gl, calls } = stubWebGl();
    const mirrored = new Mesh(new PlaneGeometry(), new MeshBasicMaterial());
    mirrored.scale.x = -1;
    const scene = new Scene();
    scene.add(mirrored, new Mesh(new PlaneGeometry(), new MeshBasicMaterial()));
    renderer.render(scene, camera());

    const draws = calls.filter((c) => c.name === "frontFace" || c.name === "drawElementsInstanced");
    const facesAtDraws = draws.flatMap((c, i) => (c.name === "frontFace" ? [] : [draws[i - 1]!.args[0]]));
    expect(facesAtDraws).toEqual([gl.CW, gl.CCW]);
  });

  /**
   * @canon spec-a-mirrored-mesh-shows-the-faces-its-material-asks-for
   */
  it("winds the front face of a mirrored mesh clockwise on WebGPU", () => {
    const { device, canvas, passes } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any);
    const mirrored = new Mesh(new PlaneGeometry(), new MeshBasicMaterial());
    mirrored.scale.x = -1;
    const scene = new Scene();
    scene.add(mirrored, new Mesh(new PlaneGeometry(), new MeshBasicMaterial()));
    renderer.render(scene, camera());

    const faces = passes.map((p) => p.calls.find((c) => c.name === "setPipeline")!.args[0].primitive.frontFace);
    expect(faces).toEqual(["cw", "ccw"]);
  });

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
   * @canon spec-a-loaded-image-uploads-at-its-own-size
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
   * @canon spec-an-attribute-whose-elements-changed-size-uploads-whole
   */
  it("uploads whole an index whose values passed 16 bits, ranges or not, on WebGPU", () => {
    const { device, canvas, bytesOf } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const geometry = new BufferGeometry().setAttribute("position", new BufferAttribute(new Float32Array(12), 3));
    geometry.setIndex([0, 1, 2, 3, 0, 1, 2, 3]);
    renderer.ensureGeometryBuffers(geometry);
    geometry.index!.setArray([0, 1, 2, 70000]);
    geometry.index!.addUpdateRange(3, 1);
    const buffers = renderer.ensureGeometryBuffers(geometry);

    expect(buffers.indexFormat).toBe("uint32");
    expect(Array.from(new Uint32Array(bytesOf(buffers.index).buffer, 0, 4))).toEqual([0, 1, 2, 70000]);
  });

  /**
   * @canon spec-an-attribute-whose-elements-changed-size-uploads-whole
   */
  it("uploads whole an index whose values passed 16 bits, ranges or not, on WebGL", () => {
    const { renderer, calls } = stubWebGl();
    const geometry = new BufferGeometry().setAttribute("position", new BufferAttribute(new Float32Array(12), 3));
    geometry.setIndex([0, 1, 2, 3, 0, 1, 2, 3]);
    const scene = new Scene();
    scene.add(new Mesh(geometry, new MeshBasicMaterial()));
    renderer.render(scene, camera());
    geometry.index!.setArray([0, 1, 2, 70000]);
    geometry.index!.addUpdateRange(3, 1);
    const before = calls.length;
    renderer.render(scene, camera());

    const uploads = calls.slice(before).filter((c) => c.name === "bufferData" || c.name === "bufferSubData");
    expect(uploads.map((c) => [c.args[1], Array.from(c.args[2])])).toEqual([[0, [0, 1, 2, 70000]]]);
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
   * @canon spec-an-image-uploads-once-it-has-loaded
   */
  it("uploads an image at the first render after it loads", () => {
    const { device, canvas, queue } = stubDevice();
    const gpu = new WebGPURenderer(canvas, device as any);
    const { renderer: gl, calls } = stubWebGl();
    const image: any = { complete: false, width: 0, height: 0 };
    const texture = new Texture(image);
    const scene = new Scene();
    scene.add(new Mesh(new PlaneGeometry(), sampling(texture)));
    const copies = () => queue.filter((c) => c.name === "copyExternalImageToTexture").length;
    const uploads = () => calls.filter((c) => c.name === "texImage2D").length;
    gpu.render(scene, camera());
    gl.render(scene, camera());
    expect([copies(), uploads()]).toEqual([0, 0]);

    Object.assign(image, { complete: true, width: 2, height: 2 });
    gpu.render(scene, camera());
    gl.render(scene, camera());
    expect([copies(), uploads()]).toEqual([1, 1]);
  });

  /**
   * @canon spec-a-loaded-image-uploads-at-its-own-size
   */
  it("makes a texture an image can be copied into when an image replaces data of its size on WebGPU", () => {
    const { device, canvas, textures } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const texture = new DataTexture(new Uint8Array(16), 2, 2);
    const material = sampling(texture);
    renderer.ensurePipeline(material, new Scene(), false, false);
    texture.image = { width: 2, height: 2 } as any;
    texture.needsUpdate = true;
    renderer.ensurePipeline(material, new Scene(), false, false);

    expect(textures.at(-1).usage & GPUTextureUsage.RENDER_ATTACHMENT).toBeTruthy();
  });

  /**
   * @canon spec-a-data-texture-uploads-in-the-type-it-names
   */
  it("lays a sampler out for the float texture put in it after the first draw on WebGPU", () => {
    const { device, canvas, layouts } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    let current = new DataTexture(new Uint8Array([0, 0, 255, 255]), 1, 1);
    const material = new MeshBasicMaterial();
    material.fragmentNode = (b) => b.sampler("map", () => current).texture(vec2(0.5, 0.5));
    renderer.ensurePipeline(material, new Scene(), false, false);
    current = new DataTexture(new Float32Array([1, 0.5, 0.25, 1]), 1, 1, 1, RGBAFormat, FloatType);
    renderer.ensurePipeline(material, new Scene(), false, false);

    const entries = layouts.flatMap((l) => l.entries);
    expect(entries.filter((e) => e.texture).at(-1).texture.sampleType).toBe("unfilterable-float");
    expect(entries.filter((e) => e.sampler).at(-1).sampler.type).toBe("non-filtering");
  });

  /**
   * @canon spec-a-data-texture-uploads-in-the-type-it-names
   */
  it("lays a sampler out again for a float texture without compiling the material again on WebGPU", () => {
    const { device, canvas, modules } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    let current = new DataTexture(new Uint8Array([0, 0, 255, 255]), 1, 1);
    const material = new MeshBasicMaterial();
    material.fragmentNode = (b) => b.sampler("map", () => current).texture(vec2(0.5, 0.5));
    const first = renderer.ensurePipeline(material, new Scene(), false, false);
    const compiled = modules.length;
    current = new DataTexture(new Float32Array([1, 0.5, 0.25, 1]), 1, 1, 1, RGBAFormat, FloatType);

    expect(renderer.ensurePipeline(material, new Scene(), false, false)).toBe(first);
    expect(modules).toHaveLength(compiled);
  });

  /**
   * @canon spec-a-webgpu-draw-binds-groups-of-the-layout-its-pipeline-has
   */
  it("binds each draw's groups through the layout of its pipeline when a texture changes mid-frame on WebGPU", () => {
    const { device, canvas, passes } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    let current = new DataTexture(new Uint8Array([0, 0, 255, 255]), 1, 1);
    const material = new MeshBasicMaterial();
    material.fragmentNode = (b) => b.sampler("map", () => current).texture(vec2(0.5, 0.5));
    const scene = new Scene();
    scene.add(new Mesh(new PlaneGeometry(), material), new Mesh(new PlaneGeometry(), material));
    // The texture becomes one the device cannot filter between the two meshes' pipelines.
    const ensurePipeline = renderer.ensurePipeline.bind(renderer);
    let calls = 0;
    renderer.ensurePipeline = (...args: unknown[]) => {
      if (calls++ === 1) current = new DataTexture(new Float32Array([1, 0.5, 0.25, 1]), 1, 1, 1, RGBAFormat, FloatType);
      return ensurePipeline(...args);
    };
    renderer.render(scene, camera());

    expect(passes).toHaveLength(2);
    for (const { calls } of passes) {
      const pipeline = calls.find((c) => c.name === "setPipeline")!.args[0];
      const textures = calls.find((c) => c.name === "setBindGroup" && c.args[0] === 1)!.args[1];
      expect(textures.layout).toBe(pipeline.layout.bindGroupLayouts[1]);
    }
  });

  /**
   * @canon spec-the-webgpu-renderer-shares-one-sampler-per-state
   */
  it("keeps the bind groups of a float texture it cannot filter when the texture updates on WebGPU", () => {
    const { device, canvas } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const texture = new DataTexture(new Float32Array([1, 0.5, 0.25, 1]), 1, 1, 1, RGBAFormat, FloatType);
    const material = sampling(texture);
    const first = renderer.ensurePipeline(material, new Scene(), false, false).samplerBindGroup;
    texture.needsUpdate = true;

    expect(renderer.ensurePipeline(material, new Scene(), false, false).samplerBindGroup).toBe(first);
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
   * @canon spec-a-sampler-change-takes-effect-after-needs-update
   */
  it("writes a changed filter on WebGL only after needsUpdate", () => {
    const { renderer, gl, calls } = stubWebGl();
    const scene = new Scene();
    const texture = new DataTexture(new Uint8Array([0, 0, 255, 255]), 1, 1);
    scene.add(new Mesh(new PlaneGeometry(), sampling(texture)));
    const minFilters = (from: number) =>
      calls.slice(from).filter((c) => c.name === "texParameteri" && c.args[1] === gl.TEXTURE_MIN_FILTER);
    renderer.render(scene, camera());

    texture.minFilter = NearestFilter;
    let before = calls.length;
    renderer.render(scene, camera());
    expect(minFilters(before)).toHaveLength(0);

    texture.needsUpdate = true;
    before = calls.length;
    renderer.render(scene, camera());
    expect(minFilters(before).map((c) => c.args[2])).toEqual([gl.NEAREST]);
  });

  /**
   * @canon spec-a-sampler-change-takes-effect-after-needs-update
   */
  it("binds a changed wrap on WebGPU only after needsUpdate", () => {
    const { device, canvas, samplers } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const texture = new DataTexture(new Uint8Array([0, 0, 255, 255]), 1, 1);
    const material = sampling(texture);
    const scene = new Scene();
    const first = renderer.ensurePipeline(material, scene, false, false).samplerBindGroup;

    texture.wrapS = RepeatWrapping;
    expect(renderer.ensurePipeline(material, scene, false, false).samplerBindGroup).toBe(first);
    expect(samplers).toHaveLength(1);

    texture.needsUpdate = true;
    expect(renderer.ensurePipeline(material, scene, false, false).samplerBindGroup).not.toBe(first);
    expect(samplers[1]).toMatchObject({ addressModeU: "repeat" });
  });

  /**
   * @canon spec-the-webgl-renderer-draws-from-its-own-vertex-array
   */
  it("draws from a vertex array of its own after it was disposed on WebGL", () => {
    let made = 0;
    const { renderer, calls } = stubWebGl({ createVertexArray: () => ({ made: ++made }) });
    const scene = new Scene();
    scene.add(new Mesh(new PlaneGeometry(), new MeshBasicMaterial()));
    renderer.render(scene, camera());
    renderer.dispose();
    const before = calls.length;
    renderer.render(scene, camera());

    const bound = calls.slice(before).find((c) => c.name === "bindVertexArray")!.args[0];
    const deleted = calls.filter((c) => c.name === "deleteVertexArray").map((c) => c.args[0]);
    expect(bound).not.toBeNull();
    expect(deleted).not.toContain(bound);
  });

  /**
   * @canon spec-a-webgl-draw-gives-an-attribute-with-no-data-a-fresh-value
   */
  it("gives an attribute its geometry lacks the value 0, 0, 0, 1 on WebGL", () => {
    const { renderer, calls } = stubWebGl();
    const material = new MeshBasicMaterial();
    material.positionNode = (b) => b.position.add(b.attribute("offset", "vec4").xyz);
    const scene = new Scene();
    scene.add(new Mesh(new PlaneGeometry(), material));
    renderer.render(scene, camera());

    const values = calls.filter((c) => c.name === "vertexAttrib4f");
    expect(values).toHaveLength(1);
    expect(values[0]!.args.slice(1)).toEqual([0, 0, 0, 1]);
  });

  /**
   * @canon spec-a-webgl-renderer-asked-to-preserve-state-puts-it-back
   * @canon spec-a-glsl-adapter-asked-to-preserve-state-puts-it-back
   */
  it("puts back the framebuffer a draw to the canvas replaced, whatever else the call keeps", () => {
    const framebuffer = {};
    const { gl, calls } = stubGl(
      { width: 4, height: 4 },
      { getParameter: (name: number) => (name === gl.DRAW_FRAMEBUFFER_BINDING ? framebuffer : 0) },
    );
    const state = new GlStateKeeper(gl);
    state.begin(0);
    drawToCanvas(gl, state);
    state.end();

    const draws = calls.filter(
      (c) => c.name === "bindFramebuffer" && (c.args[0] === gl.DRAW_FRAMEBUFFER || c.args[0] === gl.FRAMEBUFFER),
    );
    expect(draws.at(-1)!.args[1]).toBe(framebuffer);
  });

  /**
   * @canon exception-preserving-webgl-state-allocates-on-each-call
   */
  it("reads the viewport, the clear colour, the colour mask and the depth range on each render with preserveState", () => {
    let vectors: Record<number, unknown> = {};
    const { renderer, gl, calls } = stubWebGl(
      { getParameter: (name: number) => vectors[name] ?? 16 },
      { preserveState: true },
    );
    vectors = {
      [gl.VIEWPORT]: new Int32Array(4),
      [gl.COLOR_CLEAR_VALUE]: new Float32Array(4),
      [gl.COLOR_WRITEMASK]: [true, true, true, true],
      [gl.DEPTH_RANGE]: new Float32Array(2),
    };
    const scene = new Scene();
    scene.add(new Mesh(new PlaneGeometry(), new MeshBasicMaterial()));
    const reads = () =>
      calls.filter(
        (c) =>
          c.name === "getParameter" &&
          [gl.VIEWPORT, gl.COLOR_CLEAR_VALUE, gl.COLOR_WRITEMASK, gl.DEPTH_RANGE].includes(c.args[0]),
      ).length;
    renderer.render(scene, camera());
    const first = reads();
    renderer.render(scene, camera());
    expect(first).toBe(4);
    expect(reads()).toBe(8);
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
        .some((w) => w.buffer === gpu.ensureGeometryBuffers(geometry).attributes.get("instanceEnd")),
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

// One texture read through a float sampler and then through an integer one. The
// second read must give what it gives on a renderer that never read the texture
// as a float.
const ENTRY_FLOAT_THEN_INTEGER = `
import { WebGLRenderer, Scene, Mesh, PerspectiveCamera, PlaneGeometry, MeshBasicMaterial, DataTexture } from "../scene";
import { float, uvec2, vec2 } from "../rmsl";
globalThis.__rmslFloatThenIntegerRun = () => {
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
  const reading = (texture, type) => {
    const material = new MeshBasicMaterial();
    material.fragmentNode = (b) => {
      const sampler = b.sampler("map", type, () => texture);
      return type === "sampler2D"
        ? sampler.texture(vec2(0.5, 0.5))
        : sampler.texture(uvec2(0, 0)).toVec4().div(float(255));
    };
    const scene = new Scene();
    scene.add(new Mesh(new PlaneGeometry(2, 2), material));
    return scene;
  };
  const centre = (renderer) => {
    const gl = renderer.gl;
    const pixels = new Uint8Array(4);
    gl.readPixels(8, 8, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    return [pixels[0], pixels[1], pixels[2]];
  };
  const texture = () => new DataTexture(new Uint8Array([0, 0, 255, 255]), 1, 1);

  const shared = texture();
  const renderer = make();
  renderer.render(reading(shared, "sampler2D"), camera);
  renderer.render(reading(shared, "usampler2D"), camera);
  const fresh = make();
  fresh.render(reading(texture(), "usampler2D"), camera);
  return { afterFloat: centre(renderer), fresh: centre(fresh) };
};
`;

// A float and an integer texture, a transparent mesh, a render target and both
// readbacks, each run over state the page set: every piece of state that a
// call left changed.
const ENTRY_PRESERVE_STATE = `
import { Color, WebGLRenderer, WebGLRenderTarget, Scene, Mesh, PerspectiveCamera, PlaneGeometry, MeshBasicMaterial, DataTexture } from "../scene";
import { float, uvec2, vec2 } from "../rmsl";
${GL_STATE}
const preserveStateRun = async (preserveState) => {
  const canvas = document.createElement("canvas");
  canvas.width = 16;
  canvas.height = 16;
  const renderer = new WebGLRenderer(canvas, { antialias: false, preserveState });
  const gl = renderer.gl;
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);
  const texel = () => new DataTexture(new Uint8Array([0, 0, 255, 255]), 1, 1);
  const floats = new MeshBasicMaterial();
  floats.transparent = true;
  floats.fragmentNode = (b) => b.sampler("map", "sampler2D", texel).texture(vec2(0.5, 0.5));
  const integers = new MeshBasicMaterial();
  integers.fragmentNode = (b) => b.sampler("map", "usampler2D", texel).texture(uvec2(0, 0)).toVec4().div(float(255));
  integers.positionNode = (b) => b.position.add(b.attribute("offset", "vec4").xyz);
  const scene = new Scene();
  scene.add(new Mesh(new PlaneGeometry(2, 2), integers), new Mesh(new PlaneGeometry(2, 2), floats));
  const target = new WebGLRenderTarget(4, 4);
  dirtyGlState(gl);
  const before = glState(gl);
  const changed = new Set();
  const check = () => changedGlState(before, glState(gl)).forEach((name) => changed.add(name));
  renderer.render(scene, camera);
  check();
  renderer.render(scene, camera, target);
  check();
  renderer.readPixels(target);
  check();
  await renderer.readPixelsAsync(target);
  check();
  return [...changed].sort();
};
globalThis.__rmslPreserveStateOff = () => preserveStateRun(false);
globalThis.__rmslPreserveStateOn = () => preserveStateRun(true);
const ownVertexArrayRun = (vertexArray) => {
  const canvas = document.createElement("canvas");
  const renderer = new WebGLRenderer(canvas, { antialias: false });
  const gl = renderer.gl;
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 1);
  dirtyGlState(gl);
  if (vertexArray === "default") {
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.enableVertexAttribArray(1);
  }
  const before = glState(gl);
  const scene = new Scene();
  scene.add(new Mesh(new PlaneGeometry(2, 2), new MeshBasicMaterial()));
  renderer.render(scene, camera);
  gl.bindVertexArray(before.VERTEX_ARRAY_BINDING);
  return changedGlState(before, glState(gl)).filter((name) => /^(ELEMENT_ARRAY|VERTEX_ATTRIB)/.test(name));
};
// One frame, a transparent textured mesh in front of an opaque one over a
// background, read back from a render target, drawn over state the page set and
// over a fresh context. The transparent mesh reads an attribute its geometry
// lacks, which moves it out of view unless it holds 0.
const pixelsOver = (dirty) => {
  const canvas = document.createElement("canvas");
  canvas.width = 16;
  canvas.height = 16;
  const renderer = new WebGLRenderer(canvas, { antialias: false });
  const gl = renderer.gl;
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);
  const texture = new DataTexture(Uint8Array.of(255, 0, 0, 128, 0, 255, 0, 128, 0, 0, 255, 128, 255, 255, 255, 128), 2, 2);
  const material = new MeshBasicMaterial();
  material.transparent = true;
  material.fragmentNode = (b) => b.sampler("map", "sampler2D", () => texture).texture(vec2(0.25, 0.75));
  material.positionNode = (b) => b.position.add(b.attribute("offset", "vec4").xyz.mul(8));
  const behind = new Mesh(new PlaneGeometry(2, 2), new MeshBasicMaterial({ color: 0xff00ff }));
  behind.position.z = -0.5;
  const scene = new Scene();
  scene.background = new Color(0.2, 0.4, 0.6);
  scene.add(behind, new Mesh(new PlaneGeometry(2, 2), material));
  if (dirty) dirtyGlState(gl);
  const target = new WebGLRenderTarget(4, 4);
  renderer.render(scene, camera, target);
  return Array.from(renderer.readPixels(target));
};
globalThis.__rmslDrawsOverState = () => ({ dirty: pixelsOver(true), clean: pixelsOver(false) });
globalThis.__rmslDefaultVertexArray = () => ownVertexArrayRun("default");
globalThis.__rmslApplicationVertexArray = () => ownVertexArrayRun("application");
`;

describe.skipIf(!GPU_ENABLED)("a render depends only on what it is given, on a real driver", () => {
  /**
   * @canon spec-a-webgl-call-sets-the-state-it-reads
   * @canon spec-a-webgl-draw-gives-an-attribute-with-no-data-a-fresh-value
   */
  it("draws over state the page set as it draws on a fresh context on WebGL", async () => {
    const { dirty, clean } = await runInGpuPage(
      ENTRY_PRESERVE_STATE,
      "__rmslDrawsOverState",
      new URL(".", import.meta.url).pathname,
    );
    expect(dirty).toEqual(clean);
  }, 60_000);

  /**
   * @canon spec-a-webgl-call-leaves-the-state-it-set
   */
  it("leaves the unpack alignment and its program as it set them on WebGL", async () => {
    const changed = await runInGpuPage(
      ENTRY_PRESERVE_STATE,
      "__rmslPreserveStateOff",
      new URL(".", import.meta.url).pathname,
    );
    expect(changed).toEqual(expect.arrayContaining(["UNPACK_ALIGNMENT", "CURRENT_PROGRAM"]));
  }, 60_000);

  /**
   * @canon spec-a-webgl-call-sets-the-state-it-reads
   */
  it("turns dithering off over a page that turned it on on WebGL", async () => {
    const changed = await runInGpuPage(
      ENTRY_PRESERVE_STATE,
      "__rmslPreserveStateOff",
      new URL(".", import.meta.url).pathname,
    );
    expect(changed).toContain("DITHER");
  }, 60_000);

  /**
   * @canon spec-a-webgl-renderer-asked-to-preserve-state-puts-it-back
   */
  it("puts back every piece of state it changed with preserveState on WebGL", async () => {
    const changed = await runInGpuPage(
      ENTRY_PRESERVE_STATE,
      "__rmslPreserveStateOn",
      new URL(".", import.meta.url).pathname,
    );
    expect(changed).toEqual([]);
  }, 60_000);

  /**
   * @canon spec-the-webgl-renderer-draws-from-its-own-vertex-array
   */
  it.each(["__rmslDefaultVertexArray", "__rmslApplicationVertexArray"])(
    "leaves a vertex array of the application's as it found it on WebGL (%s)",
    async (entryPoint) => {
      const changed = await runInGpuPage(ENTRY_PRESERVE_STATE, entryPoint, new URL(".", import.meta.url).pathname);
      expect(changed).toEqual([]);
    },
    60_000,
  );

  /**
   * @canon spec-a-texture-reads-as-its-sampler-asks-whichever-sampler-uploaded-it
   */
  it("reads a texture as an integer after reading it as a float on WebGL", async () => {
    const result = await runInGpuPage(
      ENTRY_FLOAT_THEN_INTEGER,
      "__rmslFloatThenIntegerRun",
      new URL(".", import.meta.url).pathname,
    );
    expect(result.afterFloat).toEqual(result.fresh);
  }, 60_000);
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
