import { describe, it, expect, afterAll } from "vitest";
import {
  recordingGLSL as compileGLSL,
  recordingWGSL as compileWGSL,
  assertRecordedShadersValid,
} from "../testing/shader-validity";
import {
  Scene, Group,
  InstancedMesh, Mesh,
  BoxGeometry,
  MeshBasicMaterial, MeshStandardMaterial,
  NodeMaterial, Builder,
  PerspectiveCamera,
  Color, Vector3, Matrix4,
  type MaterialProgram,
} from "./index";
import { BufferAttribute } from "./geometries/BufferAttribute";

afterAll(async () => {
  await assertRecordedShadersValid();
}, 120_000);

function compileMaterial(program: { vertexRoot: any; fragmentRoot: any }): { glsl: string; wgsl: string } {
  const glsl = compileGLSL.vertex(program.vertexRoot)
    + "\n---\n"
    + compileGLSL.fragment(program.fragmentRoot);
  const wgsl = compileWGSL.vertex(program.vertexRoot)
    + "\n---\n"
    + compileWGSL.fragment(program.fragmentRoot);
  return { glsl, wgsl };
}

/** The `@location(n) name: type` pairs of a WGSL VertexInput struct. */
function vertexLocations(wgsl: string): { location: number; name: string; type: string }[] {
  const vertex = wgsl.split("\n---\n")[0];
  const start = vertex.indexOf("struct VertexInput");
  const end = vertex.indexOf("};", start);
  const struct = vertex.slice(start, end);
  return [...struct.matchAll(/@location\((\d+)\) (\w+): ([^,]+),/g)].map((m) => ({
    location: Number(m[1]),
    name: m[2],
    type: m[3].trim(),
  }));
}

describe("InstancedMesh", () => {
  it("initializes count instances to the identity matrix", () => {
    const mesh = new InstancedMesh(new BoxGeometry(), new MeshBasicMaterial(), 3);
    expect(mesh.isInstancedMesh).toBe(true);
    expect(mesh.count).toBe(3);
    const matrix = new Matrix4();
    for (let i = 0; i < 3; i++) {
      mesh.getMatrixAt(i, matrix);
      expect(matrix.equals(new Matrix4())).toBe(true);
    }
  });

  it("is a mesh with an instanced matrix attribute", () => {
    const mesh = new InstancedMesh(new BoxGeometry(), new MeshBasicMaterial(), 2);
    expect(mesh.isMesh).toBe(true);
    expect(mesh.instanceMatrix).toBeInstanceOf(BufferAttribute);
    expect(mesh.instanceMatrix.itemSize).toBe(16);
    expect(mesh.instanceMatrix.stepMode).toBe("instance");
    expect(mesh.instanceMatrix.count).toBe(2);
  });

  it("round-trips instance transforms", () => {
    const mesh = new InstancedMesh(new BoxGeometry(), new MeshBasicMaterial(), 2);
    const matrix = new Matrix4().makeTranslation(1, 2, 3);
    const out = new Matrix4();
    mesh.setMatrixAt(0, matrix);
    mesh.getMatrixAt(0, out);
    expect(out.equals(matrix)).toBe(true);
    // The untouched instance stays the identity.
    mesh.getMatrixAt(1, out);
    expect(out.equals(new Matrix4())).toBe(true);
  });

  it("lazily creates instanceColor and round-trips colors", () => {
    const mesh = new InstancedMesh(new BoxGeometry(), new MeshBasicMaterial(), 2);
    const out = new Color();
    // No colors set yet: reads back white.
    mesh.getColorAt(0, out);
    expect(out.getHex()).toBe(0xffffff);
    expect(mesh.instanceColor).toBeNull();

    mesh.setColorAt(0, new Color().setHex(0xff0000));
    mesh.setColorAt(1, new Color().setRGB(0, 0.5, 1));
    expect(mesh.instanceColor).toBeInstanceOf(BufferAttribute);
    expect(mesh.instanceColor!.stepMode).toBe("instance");
    expect(mesh.instanceColor!.itemSize).toBe(3);

    mesh.getColorAt(0, out);
    expect(out.getHex()).toBe(0xff0000);
    mesh.getColorAt(1, out);
    expect(out.r).toBeCloseTo(0);
    expect(out.g).toBeCloseTo(0.5);
    expect(out.b).toBeCloseTo(1);
  });

  it("copies geometry, material, transforms and instance data", () => {
    const source = new InstancedMesh(new BoxGeometry(), new MeshBasicMaterial({ color: 0x123456 }), 2);
    source.setMatrixAt(0, new Matrix4().makeTranslation(5, 6, 7));
    source.setColorAt(1, new Color().setHex(0x00ff00));
    source.position.set(1, 1, 1);

    const copy = new InstancedMesh().copy(source);
    expect(copy.count).toBe(2);
    expect(copy.geometry).toBe(source.geometry);
    expect(copy.material).toBe(source.material);
    expect(copy.position.equals(source.position)).toBe(true);
    const out = new Matrix4();
    copy.getMatrixAt(0, out);
    expect(out.equals(new Matrix4().makeTranslation(5, 6, 7))).toBe(true);
    const color = new Color();
    copy.getColorAt(1, color);
    expect(color.getHex()).toBe(0x00ff00);
  });
});

describe("instanced materials", () => {
  it("declares instanceMatrix only when built for an instanced mesh", () => {
    const scene = new Scene();
    const plain = new MeshBasicMaterial().build(scene);
    expect(plain.attributes.find((a) => a.name === "instanceMatrix")).toBeUndefined();

    const instanced = new MeshBasicMaterial().build(scene, { instancing: true });
    const im = instanced.attributes.find((a) => a.name === "instanceMatrix");
    expect(im).toBeDefined();
    expect(im!.stepMode).toBe("instance");
    expect(im!.node._t).toBe("mat4");
  });

  it("declares instanceColor and its varying when the mesh carries colors", () => {
    const scene = new Scene();
    const program = new MeshBasicMaterial().build(scene, { instancing: true, instancingColor: true });
    expect(program.attributes.map((a) => [a.name, a.stepMode])).toEqual(
      expect.arrayContaining([
        ["instanceMatrix", "instance"],
        ["instanceColor", "instance"],
      ]),
    );
    expect(program.varyings.map((v) => v.name)).toContain("instanceColor");
  });

  it("applies the instance transform to position and normal in the vertex shader", () => {
    const scene = new Scene();
    const material = new MeshBasicMaterial();
    const program = material.build(scene, { instancing: true });
    const { glsl, wgsl } = compileMaterial(program);

    // The vertex stage multiplies by the instance matrix and truncates it for
    // the normal — GLSL and WGSL both spell the cast `mat3x3<f32>(...)`.
    expect(glsl.split("\n---\n")[0]).toContain("in mat4 ");
    expect(glsl.split("\n---\n")[0]).toMatch(/\* vec4\(/);
    expect(glsl.split("\n---\n")[0]).toContain("mat3(");
    expect(wgsl.split("\n---\n")[0]).toContain("mat4x4<f32>");
    expect(wgsl.split("\n---\n")[0]).toContain("mat3x3<f32>(");
  });

  it("tints the fragment color with the instance color varying", () => {
    const scene = new Scene();
    const material = new MeshBasicMaterial();
    const program = material.build(scene, { instancing: true, instancingColor: true });
    const { glsl } = compileMaterial(program);
    expect(glsl.split("\n---\n")[1]).toMatch(/\* _rmsl_v\d+/);
  });

  it("advances the WGSL attribute location by four for a mat4", () => {
    const scene = new Scene();
    const program = new MeshStandardMaterial().build(scene, { instancing: true });
    const { wgsl } = compileMaterial(program);

    const locations = vertexLocations(wgsl);
    const im = locations.find((l) => l.type === "mat4x4<f32>");
    expect(im).toBeDefined();
    const index = locations.indexOf(im!);
    // The next vertex input starts four locations after the matrix.
    expect(locations[index + 1].location).toBe(im!.location + 4);
  });

  it("keeps the matrix's stride implicit in the WGSL struct", () => {
    // The WGSL struct declares the matrix once; its four locations and 64-byte
    // stride are supplied by the pipeline's vertex layout, which the renderer
    // must expand to four float32x4 columns. This test pins the compiler side:
    // the single mat4 member, with the next attribute four locations later.
    const scene = new Scene();
    const program = new MeshBasicMaterial().build(scene, { instancing: true });
    const { wgsl } = compileMaterial(program);
    const locations = vertexLocations(wgsl);
    const im = locations.find((l) => l.type === "mat4x4<f32>");
    expect(im).toBeDefined();
    expect(locations.filter((l) => l.type === "mat4x4<f32>")).toHaveLength(1);
  });

  it("the renderer's vertex-layout rule matches the compiler's locations", () => {
    // The WebGPU renderer numbers its pipeline's vertex buffers from
    // `program.attributes`, advancing four locations per mat4 — exactly the
    // rule the WGSL compiler uses for its VertexInput struct. The two must
    // agree or the pipeline points at locations the shader put something else
    // on, so this test replays the renderer's allocation against the real
    // compiled locations.
    const scene = new Scene();
    const program = new MeshStandardMaterial().build(scene, { instancing: true, instancingColor: true });
    const { wgsl } = compileMaterial(program);

    let location = 0;
    const rendered = program.attributes.map((a) => {
      const start = location;
      location += a.node._t === "mat4" ? 4 : 1;
      return { name: a.node.name, location: start, type: a.node._t };
    });

    const compiled = vertexLocations(wgsl).map((l) => ({ name: l.name, location: l.location, type: l.type }));
    expect(compiled.map((c) => c.location)).toEqual(rendered.map((r) => r.location));
    expect(compiled.map((c) => c.name)).toEqual(rendered.map((r) => r.name));
  });

  it("exposes the instanced accessors on the builder", () => {
    const b = new Builder();
    b.instancing = true;
    b.instancingColor = true;
    expect(b.instanceMatrix._t).toBe("mat4");
    expect(b.instanceColor._t).toBe("vec3");
    expect(b.attributes.get("instanceMatrix")?.stepMode).toBe("instance");
    expect(b.attributes.get("instanceColor")?.stepMode).toBe("instance");
    expect(b.instanceColorVarying._t).toBe("vec3");
    expect(b.varyings.get("instanceColor")?.name).toBe("instanceColor");
  });

  it("compiles the shared material to both plain and instanced programs", () => {
    // A material used by a plain mesh and an InstancedMesh in one scene keeps
    // two programs; each is the one that matches its object.
    const scene = new Scene();
    const material = new MeshStandardMaterial();
    const plain = material.build(scene);
    const instanced = material.build(scene, { instancing: true });
    const colored = material.build(scene, { instancing: true, instancingColor: true });

    expect(plain.attributes.find((a) => a.name === "instanceMatrix")).toBeUndefined();
    expect(instanced.attributes.find((a) => a.name === "instanceMatrix")).toBeDefined();
    expect(instanced.attributes.find((a) => a.name === "instanceColor")).toBeUndefined();
    expect(colored.attributes.find((a) => a.name === "instanceColor")).toBeDefined();
  });
});
