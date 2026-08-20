import { describe, it, expect, afterAll } from "vitest";
import { Fn, vec3, vec4, mix, compileJS, ivec2, ivec3, uvec2, uvec3 } from "../rmsl";
import {
  recordingGLSL as compileGLSL,
  recordingWGSL as compileWGSL,
  assertRecordedShadersValid,
} from "../testing/shader-validity";
import {
  Scene, Group, Mesh,
  BoxGeometry, PlaneGeometry,
  AmbientLight, DirectionalLight, PointLight,
  MeshBasicMaterial, MeshLambertMaterial, MeshStandardMaterial,
  NodeMaterial, Builder,
  PerspectiveCamera,
  Color, Vector3,
  DataTexture,
  cameraUniformValue, objectUniformValue,
  shaderPrecision,
  type MaterialProgram,
} from "./index";

afterAll(async () => {
  await assertRecordedShadersValid();
}, 120_000);

function litScene(): Scene {
  const scene = new Scene();
  scene.background = new Color().setHex(0x222222);
  scene.add(new AmbientLight(0xffffff, 0.2));
  const sun = new DirectionalLight(0xffeedd, 1);
  sun.position.set(5, 10, 7);
  scene.add(sun);
  const point = new PointLight(0x4477ff, 2, 10, 2);
  point.position.set(-3, 1, -2);
  scene.add(point);
  const mesh = new Mesh(new BoxGeometry(), new MeshBasicMaterial());
  mesh.position.set(0, 0, 0);
  scene.add(mesh);
  return scene;
}

function compileMaterial(program: { vertexRoot: any; fragmentRoot: any }): { glsl: string; wgsl: string } {
  const glsl = compileGLSL.vertex(program.vertexRoot)
    + "\n---\n"
    + compileGLSL.fragment(program.fragmentRoot);
  const wgsl = compileWGSL.vertex(program.vertexRoot)
    + "\n---\n"
    + compileWGSL.fragment(program.fragmentRoot);
  return { glsl, wgsl };
}

describe("node materials", () => {
  it("MeshBasicMaterial declares its uniforms and varyings", () => {
    const scene = litScene();
    const material = new MeshBasicMaterial({ color: 0xff0000 });
    const program = material.build(scene);
    const { glsl, wgsl } = compileMaterial(program);

    expect(glsl).toContain("uniform mat4 projectionMatrix");
    expect(glsl).toContain("uniform mat4 viewMatrix");
    expect(glsl).toContain("uniform mat4 modelMatrix");
    expect(glsl).toContain("uniform mat3 normalMatrix");
    expect(glsl).toContain("uniform vec3 materialColor");
    const posAttr = program.attributes.find((a) => a.name === "position")!.node.name;
    expect(glsl).toContain(`in vec3 ${posAttr}`);
    expect(glsl).toContain("layout(location=0) out vec4 _rmsl_o0");
    expect(wgsl).toContain("modelMatrix: mat4x4<f32>");
    expect(wgsl).toContain("@builtin(position) position: vec4<f32>");

    // The vertex stage transforms position and assigns the world varyings.
    expect(glsl).toContain("gl_Position");
  });

  it("MeshBasicMaterial honors a custom colorNode", () => {
    const scene = litScene();
    const material = new MeshBasicMaterial();
    material.colorNode = (b) => mix(vec3(1, 0, 0), vec3(0, 0, 1), b.uv.x);
    const { glsl } = compileMaterial(material.build(scene));
    expect(glsl).toContain("mix(");
  });

  it("MeshLambertMaterial folds lights into the fragment", () => {
    const scene = litScene();
    const material = new MeshLambertMaterial({ color: 0x888888 });
    const { glsl, wgsl } = compileMaterial(material.build(scene));

    expect(glsl).toContain("uniform vec3 ambientColor");
    expect(glsl).toContain("uniform vec3 directionalColor0");
    expect(glsl).toContain("uniform vec3 pointPosition0");
    expect(glsl).toContain("max(dot(");
    expect(wgsl).toContain("directionalDirection0");
  });

  it("MeshStandardMaterial computes a GGX specular", () => {
    const scene = litScene();
    const material = new MeshStandardMaterial({ color: 0xeeeeee, roughness: 0.4, metalness: 0.2 });
    const { glsl, wgsl } = compileMaterial(material.build(scene));

    expect(glsl).toContain("uniform float materialRoughness");
    expect(glsl).toContain("uniform float materialMetalness");
    expect(glsl).toContain("mix(vec3(0.04");
    expect(wgsl).toContain("materialRoughness");
  });

  it("a no-lights scene declares no light uniforms", () => {
    const scene = new Scene();
    const mesh = new Mesh(new BoxGeometry(), new MeshStandardMaterial());
    scene.add(mesh);
    const program = (mesh.material as MeshStandardMaterial).build(scene);
    const { glsl } = compileMaterial(program);
    // No lights means no per-light uniforms are declared (the ambient slot
    // still exists, contributing zero).
    expect(glsl).not.toContain("directionalColor");
    expect(glsl).not.toContain("pointColor");
  });

  it("collects only the uniforms the graph actually references", () => {
    const scene = litScene();
    const material = new MeshStandardMaterial();
    const program = material.build(scene);
    const names = program.uniforms.map((u) => u.name);
    expect(names).toContain("modelMatrix");
    expect(names).toContain("normalMatrix");
    expect(names).toContain("materialColor");
    expect(names).toContain("ambientColor");
    // The standard material shades via the view direction, so the camera
    // position is genuinely referenced.
    expect(names).toContain("cameraPosition");
    // No sampler was registered, so no sampler binding leaks through.
    expect(program.samplers).toEqual([]);
  });

  it("world transform is camera-relative and object uniforms carry scope", () => {
    const scene = litScene();
    const material = new MeshBasicMaterial();
    const program = material.build(scene);
    const model = program.uniforms.find((u) => u.name === "modelMatrix")!;
    const proj = program.uniforms.find((u) => u.name === "projectionMatrix")!;
    const color = program.uniforms.find((u) => u.name === "materialColor")!;
    expect(model.scope).toBe("object");
    expect(proj.scope).toBe("camera");
    expect(color.scope).toBe("material");
  });

  it("supports the vertexNode / fragmentNode escape hatches", () => {
    const scene = litScene();
    const material = new MeshBasicMaterial();
    material.fragmentNode = () => Fn(() => vec4(0.1, 0.2, 0.3, 1))();
    const program = material.build(scene);
    const { glsl } = compileMaterial(program);
    expect(glsl).toContain("0.3");
  });

  it("material uniforms expose live values through getters", () => {
    const scene = litScene();
    const material = new MeshStandardMaterial({ color: 0x123456, roughness: 0.7 });
    const program = material.build(scene);
    const color = program.uniforms.find((u) => u.name === "materialColor")!;
    const roughness = program.uniforms.find((u) => u.name === "materialRoughness")!;
    expect(color.value!(null as never)).toEqual([0x12 / 255, 0x34 / 255, 0x56 / 255]);
    expect(roughness.value!(null as never)).toBe(0.7);
  });

  it("MaterialProgram carries the referenced attributes and varyings", () => {
    const scene = litScene();
    const material = new MeshLambertMaterial();
    const program = material.build(scene);
    expect(program.attributes.map((a) => a.name)).toEqual(
      expect.arrayContaining(["position", "normal"]),
    );
    expect(program.varyings.map((v) => v.name)).toEqual(
      expect.arrayContaining(["positionWorld", "normalWorld"]),
    );
  });

  it("the standard material evaluates to a finite color on the CPU", () => {
    // A lit surface must not produce NaN: `pow(1 - dot, 5)` blackens the whole
    // object when a dot product creeps past 1.0, and `1 - vec3` must stay a
    // vec3 for the JS target (an array times a scalar is NaN).
    const scene = litScene();
    const camera = new PerspectiveCamera(50, 1, 0.1, 100);
    camera.position.set(7, 5, 9);
    camera.lookAt(0, 0.5, 0);
    scene.updateMatrixWorld(true);
    camera.updateMatrixWorld(true);

    const mesh = new Mesh(new BoxGeometry(), new MeshStandardMaterial({ color: 0xff5533, roughness: 0.25, metalness: 0.6 }));
    mesh.position.set(0, 1.4, 0);
    scene.add(mesh);

    const program = (mesh.material as MeshStandardMaterial).build(scene);
    const uniforms: Record<string, number | number[]> = {};
    for (const binding of program.uniforms) {
      if (binding.scope === "camera") uniforms[binding.node.name] = cameraUniformValue(binding.name, camera);
      else if (binding.scope === "object") uniforms[binding.node.name] = objectUniformValue(binding.name, mesh);
      else uniforms[binding.node.name] = binding.value!({ camera, mesh });
    }
    const varyings: Record<string, number[]> = {
      [program.varyings.find((v) => v.name === "positionWorld")!.node.name]: [0, 2.2, 0],
      [program.varyings.find((v) => v.name === "normalWorld")!.node.name]: [0, 1, 0],
    };

    const fn = compileJS(() => program.fragmentRoot, { stage: "fragment", params: [], name: "pick" });
    const result = fn({ uniforms, varyings }) as { outputs: Record<string, number[]> };
    const color = result.outputs[Object.keys(result.outputs)[0]];
    expect(color).toBeDefined();
    for (const channel of color) expect(Number.isFinite(channel)).toBe(true);
    // The sunlit top face must be visibly lit, not black.
    expect(color[0]).toBeGreaterThan(0.1);
  });

  it("registers and compiles the integer and 3D samplers", () => {
    const texture = () => new DataTexture(new Uint8Array([1, 2, 3, 4]), 1, 1);
    const cases: {
      type: "isampler2D" | "isampler3D" | "usampler2D" | "usampler3D";
      coords: ReturnType<typeof ivec2 | typeof ivec3 | typeof uvec2 | typeof uvec3>;
      wgsl: string;
    }[] = [
      { type: "isampler2D", coords: ivec2(0, 0), wgsl: "texture_2d<i32>" },
      { type: "isampler3D", coords: ivec3(0, 0, 0), wgsl: "texture_3d<i32>" },
      { type: "usampler2D", coords: uvec2(0, 0), wgsl: "texture_2d<u32>" },
      { type: "usampler3D", coords: uvec3(0, 0, 0), wgsl: "texture_3d<u32>" },
    ];
    for (const { type, coords, wgsl } of cases) {
      const material = new MeshBasicMaterial();
      material.fragmentNode = (b) => {
        const tex = b.sampler("data", type, texture);
        return tex.texture(coords as never).toVec4();
      };
      const program = material.build(new Scene());
      const binding = program.samplers.find((s) => s.name === "data")!;
      expect(binding.type).toBe(type);
      expect(binding.node._t).toBe(type);

      const { glsl, wgsl: compiled } = compileMaterial(program);
      expect(glsl).toContain(`uniform ${type} data;`);
      expect(glsl).toContain("texelFetch(");
      expect(compiled).toContain(wgsl);
      // Integer textures are read with textureLoad, so no companion sampler
      // binding is declared for them.
      expect(compiled).not.toContain("data_s: sampler");
      expect(compiled).not.toMatch(/@group\(2\)/);
    }
  });

  it("keeps the two-argument sampler form as a sampler2D and supports sampler3D", () => {
    const material = new MeshBasicMaterial();
    material.fragmentNode = (b) => {
      // Named `diffuse`, not `flat` — the latter is a GLSL ES interpolation
      // qualifier and reserved.
      const flat = b.sampler("diffuse", () => new DataTexture(new Uint8Array(4), 1, 1));
      const volume = b.sampler("volume", "sampler3D", () => new DataTexture(new Uint8Array(8), 2, 2, 2));
      return flat.texture(b.uv).add(volume.texture(vec3(0, 0, 0)));
    };
    const program = material.build(new Scene());
    const flat = program.samplers.find((s) => s.name === "diffuse")!;
    const volume = program.samplers.find((s) => s.name === "volume")!;
    expect(flat.type).toBe("sampler2D");
    expect(flat.node._t).toBe("sampler2D");
    expect(volume.type).toBe("sampler3D");
    expect(volume.node._t).toBe("sampler3D");

    const { glsl, wgsl } = compileMaterial(program);
    expect(glsl).toContain("uniform sampler2D diffuse;");
    expect(glsl).toContain("uniform sampler3D volume;");
    expect(wgsl).toContain("texture_2d<f32>");
    expect(wgsl).toContain("texture_3d<f32>");
    expect(wgsl).toContain("volume_s: sampler");
  });
});

describe("shader precision", () => {
  // Mirror of three.js: the renderer supplies the default precision and a
  // material's `precision` overrides it. WGSL has no precision qualifiers, so
  // the material compile options only reach the GLSL side.
  it("goes with the renderer default when the material sets none", () => {
    const material = new MeshStandardMaterial();
    expect(shaderPrecision(material, "highp")).toBe("highp");
    expect(shaderPrecision(material, "mediump")).toBe("mediump");
    expect(shaderPrecision(material, "lowp")).toBe("lowp");
  });

  it("lets a material override the renderer default", () => {
    const material = new MeshStandardMaterial({ precision: "lowp" });
    expect(material.precision).toBe("lowp");
    expect(shaderPrecision(material, "highp")).toBe("lowp");

    material.precision = null;
    expect(shaderPrecision(material, "mediump")).toBe("mediump");
  });

  it("accepts a precision in every material constructor", () => {
    const basic = new MeshBasicMaterial({ precision: "mediump" });
    const lambert = new MeshLambertMaterial({ precision: "mediump" });
    const standard = new MeshStandardMaterial({ precision: "mediump" });
    expect(basic.precision).toBe("mediump");
    expect(lambert.precision).toBe("mediump");
    expect(standard.precision).toBe("mediump");
  });

  it("flags a rebuild when the precision changes", () => {
    const material = new MeshBasicMaterial();
    expect(material.needsUpdate).toBe(false);
    material.precision = "lowp";
    expect(material.needsUpdate).toBe(true);
  });
});
