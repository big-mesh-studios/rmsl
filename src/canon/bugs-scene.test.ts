import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { float, vec2, vec4 } from "../rmsl";
import { compileWgsl } from "../wgsl";
import {
  BufferAttribute,
  BufferGeometry,
  DataTexture,
  Matrix3,
  Mesh,
  MeshBasicMaterial,
  PerspectiveCamera,
  PlaneGeometry,
  Scene,
  WebGLRenderer,
  WebGPURenderer,
} from "../scene";
import { collectNodes } from "../scene/materials/nodes/graph";
import { camera, offsetOf, sampling, stubDevice, stubWebGl } from "./scene-stubs";

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
    const before = calls.length;
    renderer.setUniform({ name: "a" }, "uint", 5);
    renderer.setUniform({ name: "b" }, "uvec2", [5, 6]);
    renderer.setUniform({ name: "c" }, "bvec3", [1, 0, 1]);
    expect(calls.slice(before).map((c) => c.name)).toEqual(["uniform1ui", "uniform2ui", "uniform3i"]);
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
});
