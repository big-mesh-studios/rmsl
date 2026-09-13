import { describe, it, expect } from "vitest";
import { wgslUniformLayout } from "../rmsl";
import {
  Scene,
  Mesh,
  InstancedMesh,
  BoxGeometry,
  MeshStandardMaterial,
  MeshBasicMaterial,
  Texture,
  DataTexture,
  RedIntegerFormat,
  RGBAFormat,
  AmbientLight,
  DirectionalLight,
  PointLight,
  PerspectiveCamera,
  Vector3,
  Matrix4,
  Matrix3,
  Color,
  NearestFilter,
  LinearFilter,
  RepeatWrapping,
  MirroredRepeatWrapping,
  LinearMipmapLinearFilter,
  LinearMipmapNearestFilter,
  NearestMipmapLinearFilter,
  NearestMipmapNearestFilter,
} from "./index";
import {
  cameraUniformValue,
  objectUniformValue,
  lightsSignature,
  wgslTypeName,
  isIntegerSampler,
  samplerSampleType,
  samplerDimension,
  samplerState,
  textureChannels,
  uniformUploadValue,
  programSignature,
  geometryAttribute,
  VERTEX_FORMATS,
  vertexFormatOf,
  type VertexFormatSpec,
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

describe("samplerState", () => {
  it("defaults to linear filtering and clamped edges", () => {
    expect(samplerState(new Texture(), "sampler2D")).toEqual({
      magFilter: "linear",
      minFilter: "linear",
      wrapS: "clamp",
      wrapT: "clamp",
      wrapR: "clamp",
    });
  });

  it("carries what the texture asks for", () => {
    const texture = new Texture();
    texture.magFilter = NearestFilter;
    texture.minFilter = NearestFilter;
    texture.wrapS = RepeatWrapping;
    texture.wrapT = MirroredRepeatWrapping;
    const state = samplerState(texture, "sampler2D");
    expect(state.magFilter).toBe("nearest");
    expect(state.minFilter).toBe("nearest");
    expect(state.wrapS).toBe("repeat");
    expect(state.wrapT).toBe("mirror");
  });

  it("reads a mipmapped filter as its base filter", () => {
    const texture = new Texture();
    // No renderer builds a mip chain yet, and honouring these literally leaves
    // WebGL with an incomplete texture, which samples as black.
    texture.minFilter = LinearMipmapLinearFilter;
    expect(samplerState(texture, "sampler2D").minFilter).toBe("linear");
    texture.minFilter = NearestMipmapLinearFilter;
    expect(samplerState(texture, "sampler2D").minFilter).toBe("nearest");
    texture.minFilter = NearestMipmapNearestFilter;
    expect(samplerState(texture, "sampler2D").minFilter).toBe("nearest");
    texture.minFilter = LinearMipmapNearestFilter;
    expect(samplerState(texture, "sampler2D").minFilter).toBe("linear");
  });

  it("holds an integer texture to nearest, whatever it asked for", () => {
    const texture = new Texture();
    texture.magFilter = LinearFilter;
    texture.minFilter = LinearFilter;
    texture.wrapS = RepeatWrapping;
    // An integer texture is not filterable in either language — but it still
    // wraps, which is a property of the coordinate, not of the filter.
    const state = samplerState(texture, "usampler2D");
    expect(state.magFilter).toBe("nearest");
    expect(state.minFilter).toBe("nearest");
    expect(state.wrapS).toBe("repeat");
  });

  it("treats a wrapping mode it does not know as clamped", () => {
    const texture = new Texture();
    texture.wrapS = 99999;
    expect(samplerState(texture, "sampler2D").wrapS).toBe("clamp");
  });
});

describe("textureChannels", () => {
  it("reads a single-channel format as one channel and everything else as four", () => {
    const data = new Uint8Array([1, 2, 3, 4]);
    expect(textureChannels(new DataTexture(data, 2, 2, 1, RedIntegerFormat))).toBe(1);
    expect(textureChannels(new DataTexture(data, 1, 1, 1, RGBAFormat))).toBe(4);
    // A texture holding an image rather than a data view has no format at all.
    expect(textureChannels(new Texture())).toBe(4);
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
      const value = binding.scope === "material" ? binding.value!({ camera: new PerspectiveCamera(), mesh }) : [];
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

describe("VERTEX_FORMATS", () => {
  it("holds only formats a buffer of its own can carry", () => {
    for (const [name, spec] of Object.entries(VERTEX_FORMATS)) {
      // A buffer holding one attribute has this stride, and WebGPU takes only a
      // multiple of four. A format failing this could not be bound alone.
      const stride = spec.count * spec.bytes;
      expect(`${name} stride ${stride}`).toBe(`${name} stride ${Math.ceil(stride / 4) * 4}`);
      // Below 32 bits WebGPU has no three-component format, which the stride
      // rule alone would not forbid at every width.
      if (spec.bytes < 4) {
        expect(`${name} count ${spec.count}`).not.toBe(`${name} count 3`);
      }
    }
  });

  it("gives each format the WebGL type its own name describes", () => {
    const expected: Record<string, VertexFormatSpec["gl"]> = {
      float32: "FLOAT",
      float16: "HALF_FLOAT",
      snorm8: "BYTE",
      unorm8: "UNSIGNED_BYTE",
      snorm16: "SHORT",
      unorm16: "UNSIGNED_SHORT",
    };
    for (const [name, spec] of Object.entries(VERTEX_FORMATS)) {
      const prefix = name.split("x")[0];
      expect(`${name} -> ${spec.gl}`).toBe(`${name} -> ${expected[prefix]}`);
      // The bit width in the name is the byte width in the row.
      const bits = Number(prefix.replace(/^[a-z]+/, ""));
      expect(`${name} bytes ${spec.bytes}`).toBe(`${name} bytes ${bits / 8}`);
      // Only the norm formats scale on the way in.
      expect(`${name} normalized ${spec.normalized}`).toBe(`${name} normalized ${/^[su]norm/.test(prefix)}`);
    }
  });
});

describe("vertexFormatOf", () => {
  it("reads a float attribute's format off its width", () => {
    const of = (itemSize: number): string =>
      vertexFormatOf(new BufferAttribute(new Float32Array(itemSize * 2), itemSize));
    expect(of(1)).toBe("float32");
    expect(of(2)).toBe("float32x2");
    expect(of(3)).toBe("float32x3");
    expect(of(4)).toBe("float32x4");
  });

  it("treats a plain number array as the floats it uploads as", () => {
    expect(vertexFormatOf(new BufferAttribute([0, 1, 2], 3))).toBe("float32x3");
  });

  it("reads a normalized integer attribute as its norm format", () => {
    expect(vertexFormatOf(new BufferAttribute(new Uint8Array(8), 4, true))).toBe("unorm8x4");
    expect(vertexFormatOf(new BufferAttribute(new Int8Array(8), 4, true))).toBe("snorm8x4");
    expect(vertexFormatOf(new BufferAttribute(new Uint16Array(4), 2, true))).toBe("unorm16x2");
    expect(vertexFormatOf(new BufferAttribute(new Int16Array(4), 2, true))).toBe("snorm16x2");
  });

  it("takes a mat4's column width rather than its whole item size", () => {
    const attr = new BufferAttribute(new Float32Array(32), 16, false, "instance");
    expect(vertexFormatOf(attr, 4)).toBe("float32x4");
  });

  it("lets an attribute declare a format its array type cannot say", () => {
    // Half floats held in a Uint16Array are indistinguishable from normalized
    // integers, which is the case the field exists for.
    const attr = new BufferAttribute(new Uint16Array([0x3c00, 0x3400]), 2, true);
    expect(vertexFormatOf(attr)).toBe("unorm16x2");
    attr.format = "float16x2";
    expect(vertexFormatOf(attr)).toBe("float16x2");
  });

  it("carries a declared format through a clone", () => {
    const attr = new BufferAttribute(new Uint16Array([0x3c00, 0x3400]), 2, true);
    attr.format = "float16x2";
    expect(attr.clone().format).toBe("float16x2");
  });

  it("refuses a raw integer array, which would not be floats in the shader", () => {
    expect(() => vertexFormatOf(new BufferAttribute(new Uint8Array(8), 4))).toThrow(/normalized/);
  });

  it("refuses a width no format covers", () => {
    expect(() => vertexFormatOf(new BufferAttribute(new Uint8Array(6), 3, true))).toThrow(/unorm8x3/);
  });
});
