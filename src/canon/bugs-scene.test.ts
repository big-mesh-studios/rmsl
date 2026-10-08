import { afterAll, afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { float, vec2, vec4 } from "../rmsl";
import { compileWgsl } from "../wgsl";
import {
  BufferAttribute,
  BufferGeometry,
  Color,
  DataTexture,
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
import { camera, offsetOf, sampling, stubDevice, stubWebGl } from "./scene-stubs";
import { GPU_ENABLED, releaseGpu } from "../testing/gpu";
import { runInGpuPage } from "../testing/browser";

/** three.js's `FloatType`, which `./scene` does not export. */
const FloatType = 1015;

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
   * @canon spec-a-webgpu-render-records-what-a-fresh-renderer-records
   */
  it("records the pass a fresh renderer records after drawing another scene first", () => {
    const sceneOf = (z: number, color: number) => {
      const scene = new Scene();
      const mesh = new Mesh(new PlaneGeometry(), new MeshBasicMaterial({ color }));
      mesh.position.z = z;
      scene.add(mesh);
      return scene;
    };
    const recorded = (render: (renderer: WebGPURenderer) => void) => {
      const { device, canvas, passes } = stubDevice();
      render(new WebGPURenderer(canvas, device as any));
      const last = passes[passes.length - 1];
      return JSON.stringify([
        last.descriptor.colorAttachments[0].clearValue,
        last.calls.map((c) => [c.name, c.args.filter((a) => typeof a === "number")]),
      ]);
    };
    const later = sceneOf(-1, 0x00ff00);

    const afterOther = recorded((renderer) => {
      renderer.render(sceneOf(1, 0xff0000), camera());
      renderer.render(later, camera());
    });
    const fresh = recorded((renderer) => renderer.render(later, camera()));

    expect(afterOther).toBe(fresh);
  });

  /**
   * The WebGPU renderer pads each column of a `mat3` uniform to 16 bytes, as WGSL
   * reads it.
   *
   * @canon spec-a-uniform-uploads-in-the-shape-its-type-has
   */
  it("uploads the normal matrix with each column padded to 16 bytes on WebGPU", () => {
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
   * The WebGPU renderer writes an `int` uniform as an integer.
   *
   * @canon spec-a-uniform-uploads-in-the-shape-its-type-has
   */
  it("uploads an int uniform as an integer on WebGPU", () => {
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
   * The WebGPU renderer leaves a `mat3` uniform at zero when the value it
   * takes is an empty array, as it does for a name it does not know.
   *
   * @canon spec-a-uniform-uploads-in-the-shape-its-type-has
   */
  it("writes nothing for an empty matrix value on WebGPU", () => {
    const { device, canvas, bytesOf } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const material = new MeshBasicMaterial();
    material.fragmentNode = (b) =>
      vec4(
        b
          .materialUniform("m", "mat3", () => [])
          .element(0)
          .element(0),
        0,
        0,
        1,
      );
    const mesh = new Mesh(new PlaneGeometry(), material);
    mesh.updateMatrixWorld(true);
    const entry = renderer.ensurePipeline(material, new Scene(), false, false);
    renderer.packUniforms(entry, mesh, camera(), 0);

    const floats = new Float32Array(bytesOf(entry.ringBuffer).buffer);
    const base = offsetOf(entry, "m") / 4;
    expect(Array.from(floats.subarray(base, base + 12))).toEqual(new Array(12).fill(0));
  });

  /**
   * The WebGL renderer uploads a `uint`, a `uvec2` and a `bvec3` uniform.
   *
   * @canon spec-a-uniform-uploads-in-the-shape-its-type-has
   */
  it("uploads a uint, a uvec2 and a bvec3 uniform on WebGL", () => {
    const { renderer, calls } = stubWebGl();
    renderer.setUniform({ name: "a" }, "uint", 5);
    renderer.setUniform({ name: "b" }, "uvec2", [5, 6]);
    renderer.setUniform({ name: "c" }, "bvec3", [1, 0, 1]);
    expect(calls.map((c) => c.name)).toEqual(["uniform1ui", "uniform2ui", "uniform3i"]);
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
    // A ring that is oversized for a frame or two is kept.
    expect(entry.ringBuffer).toBe(full);
    for (let frame = 0; frame < 60; frame++) renderer.render(scene, camera());
    expect(entry.slots).toBeLessThan(1000);
    expect(destroyed.has(full)).toBe(true);

    // A program with no draw at all shrinks too.
    scene.clear();
    for (let frame = 0; frame < 60; frame++) renderer.render(scene, camera());
    expect(entry.slots).toBe(64);

    const small = entry.ringBuffer;
    scene.add(meshes[0]);
    material.needsUpdate = true;
    renderer.render(scene, camera());
    expect(destroyed.has(small)).toBe(true);
  });

  /**
   * The WebGPU renderer binds each texture and sampler at the binding the
   * compiled WGSL declares it at, in both stages.
   *
   * @canon spec-each-sampler-gets-its-own-texture
   */
  it("numbers texture and sampler bindings as the compiled WGSL does on WebGPU", () => {
    const { device, canvas, pipelines } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const material = new MeshBasicMaterial();
    const texture = () => new DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
    material.fragmentNode = (b) => {
      b.sampler("unused", texture);
      const first = b.sampler("first", texture);
      const second = b.sampler("second", texture);
      return second.texture(vec2(0.5, 0.5)).add(first.texture(vec2(0.5, 0.5)));
    };
    const scene = new Scene();
    scene.add(new Mesh(new PlaneGeometry(), material));
    renderer.render(scene, camera());
    const entry = [...renderer.pipelines.get(material).values()][0];
    const wgsl = pipelines[0].fragment.module.code;

    const declared = (group: number) =>
      [...wgsl.matchAll(new RegExp(`@group\\(${group}\\) @binding\\((\\d+)\\) var (\\w+)`, "g"))].map(
        (m) => `${m[2]}@${m[1]}`,
      );
    const bound = (bindings: { name: string; binding: number }[], suffix: string) =>
      bindings.map((b) => `${b.name}${suffix}@${b.binding}`);
    const samplerSuffix = "_s";
    expect(bound(entry.textureBindings, "").sort()).toEqual(declared(1).sort());
    expect(bound(entry.samplerBindings, samplerSuffix).sort()).toEqual(declared(2).sort());
  });

  /**
   * The WebGPU renderer makes a texture and its sampler visible to the vertex
   * stage as well as the fragment stage, so a texture sampled in
   * `positionNode` binds.
   *
   * @canon spec-a-texture-is-bound-to-every-stage-that-samples-it
   */
  it("binds a texture sampled in the vertex stage to the vertex stage on WebGPU", () => {
    const { device, canvas, layouts } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const height = new DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
    const material = new MeshBasicMaterial();
    material.positionNode = (b) => b.position.add(b.sampler("height", () => height).textureLod(b.uv, float(0)).xyz);
    renderer.ensurePipeline(material, new Scene(), false, false);

    const textureLayout = layouts.find((l) => l.entries.some((e: any) => e.texture))!;
    expect(textureLayout.entries[0].visibility & 1).toBe(1);
    expect(textureLayout.entries[0].visibility & 2).toBe(2);
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
    const version = material.version;
    material.opacity = 0.25;

    const live = program.uniforms.some((u) => u.value?.({} as any) === 0.25);
    expect(live || material.version > version).toBe(true);
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

  /**
   * The WebGL renderer writes a texture's filters once, from the sampler type
   * that uploaded it, so an integer read after a float read meets linear
   * filters and reads zero.
   *
   * @canon bug-webgl-keeps-the-sampler-state-of-the-first-sampler-that-uploaded-a-texture
   */
  it.fails(
    "reads a texture as an integer after reading it as a float on WebGL",
    async () => {
      const result = await runInGpuPage(
        ENTRY_FLOAT_THEN_INTEGER,
        "__rmslFloatThenIntegerRun",
        new URL(".", import.meta.url).pathname,
      );
      expect(result.afterFloat).toEqual(result.fresh);
    },
    60_000,
  );
});

afterAll(async () => {
  await releaseGpu();
});
