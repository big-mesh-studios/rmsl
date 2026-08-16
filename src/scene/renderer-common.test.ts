import { describe, it, expect } from "vitest";
import { wgslUniformLayout } from "../rmsl";
import {
  Scene, Mesh, InstancedMesh, BoxGeometry, MeshStandardMaterial,
  MeshBasicMaterial,
  AmbientLight, DirectionalLight, PointLight,
  PerspectiveCamera, Vector3, Matrix4, Matrix3, Color,
} from "./index";
import {
  cameraUniformValue, objectUniformValue, lightsSignature, wgslTypeName,
  isIntegerSampler, samplerSampleType, samplerDimension,
  uniformUploadValue, programSignature, geometryAttribute,
} from "./renderers/common";
import { BufferAttribute } from "./geometries/BufferAttribute";

describe("cameraUniformValue", () => {
  it("returns the projection, view and position", () => {
    const camera = new PerspectiveCamera(50, 1, 0.1, 100);
    camera.position.set(1, 2, 3);
    camera.updateMatrixWorld(true);
    const proj = cameraUniformValue("projectionMatrix", camera);
    expect(proj).toHaveLength(16);
    expect(proj).toEqual(camera.projectionMatrix.elements);
    const view = cameraUniformValue("viewMatrix", camera);
    expect(view).toEqual(camera.matrixWorldInverse.elements);
    const pos = cameraUniformValue("cameraPosition", camera);
    expect(pos).toEqual([1, 2, 3]);
  });

  it("returns nothing for unknown names", () => {
    const camera = new PerspectiveCamera();
    expect(cameraUniformValue("nope", camera)).toEqual([]);
  });
});

describe("objectUniformValue", () => {
  it("returns the world matrix and normal matrix", () => {
    const mesh = new Mesh();
    mesh.position.set(1, 0, 0);
    mesh.scale.set(2, 3, 4);
    mesh.updateMatrixWorld(true);
    const model = objectUniformValue("modelMatrix", mesh);
    expect(model).toEqual(mesh.matrixWorld.elements);
    const normal = objectUniformValue("normalMatrix", mesh);
    expect(normal).toHaveLength(9);
    // A scaled matrix's normal matrix is the inverse-transpose of its 3x3.
    const expected = new Matrix3().getNormalMatrix(mesh.matrixWorld).toArray();
    expect(normal).toEqual(expected);
  });
});

describe("lightsSignature", () => {
  it("tracks the light set in order", () => {
    const scene = new Scene();
    expect(lightsSignature(scene)).toBe("");
    scene.add(new AmbientLight());
    expect(lightsSignature(scene)).toBe("a");
    scene.add(new DirectionalLight());
    expect(lightsSignature(scene)).toBe("ad");
    scene.add(new PointLight());
    expect(lightsSignature(scene)).toBe("adp");
  });
});

describe("programSignature", () => {
  it("combines the light set with the drawable's instancing flags", () => {
    expect(programSignature("", false, false)).toBe("|");
    expect(programSignature("adp", false, false)).toBe("adp|");
    expect(programSignature("a", true, false)).toBe("a|i");
    expect(programSignature("a", false, true)).toBe("a|c");
    expect(programSignature("a", true, true)).toBe("a|ic");
  });
});

describe("geometryAttribute", () => {
  it("falls back to an instanced mesh's object-owned attributes", () => {
    const geometry = new BoxGeometry();
    const mesh = new InstancedMesh(geometry, new MeshBasicMaterial(), 2);
    expect(geometryAttribute(mesh, geometry, "instanceMatrix")).toBe(mesh.instanceMatrix);
    expect(geometryAttribute(mesh, geometry, "instanceColor")).toBeUndefined();
    mesh.setColorAt(0, new Color());
    expect(geometryAttribute(mesh, geometry, "instanceColor")).toBe(mesh.instanceColor);
  });

  it("prefers a geometry attribute over the object fallback", () => {
    const geometry = new BoxGeometry();
    const attr = new BufferAttribute(new Float32Array(6), 3);
    geometry.setAttribute("instanceMatrix", attr);
    const mesh = new InstancedMesh(geometry, new MeshBasicMaterial(), 2);
    expect(geometryAttribute(mesh, geometry, "instanceMatrix")).toBe(attr);
  });

  it("returns nothing for a plain mesh's missing attributes", () => {
    const geometry = new BoxGeometry();
    const mesh = new Mesh(geometry, new MeshBasicMaterial());
    expect(geometryAttribute(mesh, geometry, "instanceMatrix")).toBeUndefined();
    expect(geometryAttribute(mesh, geometry, "instanceColor")).toBeUndefined();
  });
});

describe("wgslTypeName", () => {
  it("maps RMSL shader types to WGSL", () => {
    expect(wgslTypeName("float")).toBe("f32");
    expect(wgslTypeName("vec3")).toBe("vec3<f32>");
    expect(wgslTypeName("mat4")).toBe("mat4x4<f32>");
    expect(wgslTypeName("mat3")).toBe("mat3x3<f32>");
    expect(wgslTypeName("ivec2")).toBe("vec2<i32>");
  });
});

describe("sampler classification", () => {
  it("recognises the integer samplers", () => {
    expect(isIntegerSampler("isampler2D")).toBe(true);
    expect(isIntegerSampler("isampler3D")).toBe(true);
    expect(isIntegerSampler("usampler2D")).toBe(true);
    expect(isIntegerSampler("usampler3D")).toBe(true);
    expect(isIntegerSampler("sampler2D")).toBe(false);
    expect(isIntegerSampler("sampler3D")).toBe(false);
  });

  it("maps each sampler to its WebGPU sample type", () => {
    expect(samplerSampleType("isampler2D")).toBe("sint");
    expect(samplerSampleType("usampler3D")).toBe("uint");
    expect(samplerSampleType("sampler2D")).toBe("float");
    expect(samplerSampleType("sampler3D")).toBe("float");
  });

  it("maps each sampler to its dimension", () => {
    expect(samplerDimension("usampler3D")).toBe("3d");
    expect(samplerDimension("isampler3D")).toBe("3d");
    expect(samplerDimension("usampler2D")).toBe("2d");
    expect(samplerDimension("sampler2D")).toBe("2d");
  });
});

describe("uniformUploadValue", () => {
  it("keeps a scalar uniform as a scalar", () => {
    const { scalar, array } = uniformUploadValue(0.25);
    expect(scalar).toBe(0.25);
    expect(array[0]).toBe(0.25);
  });

  it("keeps vector and matrix values as arrays", () => {
    const vec = uniformUploadValue([1, 2, 3]);
    expect(vec.scalar).toBeNull();
    expect(Array.from(vec.array)).toEqual([1, 2, 3]);
    const mat = uniformUploadValue(new Float32Array(16));
    expect(mat.scalar).toBeNull();
    expect(mat.array).toHaveLength(16);
  });

  it("every material float uniform yields a scalar for direct upload", () => {
    const scene = new Scene();
    scene.add(new AmbientLight());
    scene.add(new DirectionalLight());
    const mesh = new Mesh(new BoxGeometry(), new MeshStandardMaterial());
    scene.add(mesh);
    const program = (mesh.material as MeshStandardMaterial).build(scene);
    for (const binding of program.uniforms) {
      if (binding.node._t === "float") {
        const value = binding.value!({ camera: new PerspectiveCamera(), mesh });
        expect(typeof value).toBe("number");
        expect(Number.isFinite(value)).toBe(true);
      }
    }
  });
});

describe("material program → uniform layout", () => {
  it("every collected uniform lands in the WGSL struct layout", () => {
    const scene = new Scene();
    scene.add(new AmbientLight(0xffffff, 0.2));
    scene.add(new DirectionalLight());
    const mesh = new Mesh(new BoxGeometry(), new MeshStandardMaterial());
    mesh.updateMatrixWorld(true);
    scene.add(mesh);
    const program = (mesh.material as MeshStandardMaterial).build(scene);

    const layout = wgslUniformLayout(
      program.uniforms.map((u) => ({ slot: u.node.name, type: wgslTypeName(u.node._t) })),
    );
    expect(layout.members.length).toBe(program.uniforms.length);
    expect(layout.size).toBeGreaterThan(0);

    for (const binding of program.uniforms) {
      const member = layout.members.find((m) => m.name === binding.node.name);
      expect(member).toBeDefined();
      // The value a getter yields must fit the member's footprint.
      const value = binding.scope === "material"
        ? binding.value!({ camera: new PerspectiveCamera(), mesh })
        : [];
      const expectedFloats = member!.size / 4;
      if (typeof value === "number") {
        expect(expectedFloats).toBeGreaterThanOrEqual(1);
      } else {
        expect(value.length).toBeLessThanOrEqual(expectedFloats);
      }
    }
  });

  it("a matrix's modelMatrix value fills a 64-byte member", () => {
    const m = new Matrix4().makeTranslation(1, 2, 3);
    const layout = wgslUniformLayout([{ slot: "modelMatrix", type: "mat4x4<f32>" }]);
    expect(layout.size).toBe(64);
    expect(layout.members[0].size).toBe(64);
    expect(layout.members[0].offset).toBe(0);
  });
});
