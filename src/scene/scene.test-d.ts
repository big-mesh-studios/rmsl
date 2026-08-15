/**
 * Type-level tests for `@random-mesh/rmsl/scene`: the material slot shapes,
 * the builder accessor types, and the math class signatures.
 */

import { describe, it, expectTypeOf } from "vitest";
import {
  Scene, Group, Mesh, PerspectiveCamera, OrthographicCamera,
  AmbientLight, DirectionalLight, PointLight,
  MeshBasicMaterial, MeshStandardMaterial, NodeMaterial,
  Builder, Color,
  WebGLRenderer, WebGPURenderer,
  Vector3, Matrix4, Quaternion, Euler,
  type MaterialProgram,
} from "./index";
import type { Node, UniformNode, VaryingNode, AttributeNode, ShaderType } from "../rmsl";
import type { BufferGeometry } from "./geometries/BufferGeometry";
import { float } from "../rmsl";

describe("scene graph classes", () => {
  it("builds the expected hierarchy", () => {
    const scene = new Scene();
    const group = new Group();
    scene.add(group);
    const mesh = new Mesh();
    group.add(mesh);
    expectTypeOf(scene.isScene).toBeBoolean();
    expectTypeOf(group.isGroup).toBeBoolean();
    expectTypeOf(mesh.isMesh).toBeBoolean();
    expectTypeOf(mesh.geometry).toMatchTypeOf<BufferGeometry>();
  });

  it("camera kinds narrow the projection matrix", () => {
    const perspective = new PerspectiveCamera();
    const orthographic = new OrthographicCamera();
    expectTypeOf(perspective.isPerspectiveCamera).toBeBoolean();
    expectTypeOf(orthographic.isOrthographicCamera).toBeBoolean();
    expectTypeOf(perspective.projectionMatrix).toMatchTypeOf<Matrix4>();
  });

  it("lights carry a color and an intensity", () => {
    const ambient = new AmbientLight(0xffffff, 1);
    expectTypeOf(ambient.color).toMatchTypeOf<Color>();
    expectTypeOf(ambient.intensity).toEqualTypeOf<number>();
    expectTypeOf(new DirectionalLight().target.isObject3D).toBeBoolean();
    expectTypeOf(new PointLight().decay).toEqualTypeOf<number>();
  });
});

describe("math classes", () => {
  it("transforms compose with the documented return types", () => {
    const m = new Matrix4().compose(new Vector3(), new Quaternion(), new Vector3(1, 1, 1));
    expectTypeOf(m).toEqualTypeOf<Matrix4>();
    const v = new Vector3(1, 2, 3).applyMatrix4(m);
    expectTypeOf(v).toEqualTypeOf<Vector3>();
    const q = new Quaternion().setFromEuler(new Euler(0, 0, 0));
    expectTypeOf(q).toEqualTypeOf<Quaternion>();
  });
});

describe("node materials", () => {
  it("builds a MaterialProgram", () => {
    const scene = new Scene();
    const material = new MeshStandardMaterial();
    expectTypeOf(material.build(scene)).toEqualTypeOf<MaterialProgram>();
  });

  it("slots accept a node or a builder function", () => {
    const material = new MeshStandardMaterial();
    material.colorNode = (b) => b.uv.x.toVec3();
    material.roughnessNode = () => float(0.5);
    material.normalNode = (b) => b.normalWorld;
    material.opacityNode = () => float(1);
    expectTypeOf(material).toMatchTypeOf<NodeMaterial>();
  });

  it("exposes typed accessors on the builder", () => {
    const b = new Builder();
    expectTypeOf(b.position).toEqualTypeOf<AttributeNode<"vec3">>();
    expectTypeOf(b.normal).toEqualTypeOf<AttributeNode<"vec3">>();
    expectTypeOf(b.uv).toEqualTypeOf<AttributeNode<"vec2">>();
    expectTypeOf(b.positionWorld).toEqualTypeOf<VaryingNode<"vec3">>();
    expectTypeOf(b.normalWorld).toEqualTypeOf<VaryingNode<"vec3">>();
    expectTypeOf(b.cameraPosition).toEqualTypeOf<UniformNode<"vec3">>();
    expectTypeOf(b.projectionMatrix).toEqualTypeOf<UniformNode<"mat4">>();
    expectTypeOf(b.viewMatrix).toEqualTypeOf<UniformNode<"mat4">>();
    expectTypeOf(b.modelMatrix).toEqualTypeOf<UniformNode<"mat4">>();
    expectTypeOf(b.normalMatrix).toEqualTypeOf<UniformNode<"mat3">>();
  });

  it("color slots are vec3-typed", () => {
    const material = new MeshBasicMaterial();
    // @ts-expect-error a vec4 is not a valid color slot
    material.colorNode = (b) => b.uv.x.toVec4();
    // @ts-expect-error a raw number is not a color slot value
    material.colorNode = 0xff0000;
  });

  it("renderer constructors are typed", () => {
    expectTypeOf(WebGLRenderer).toMatchTypeOf<new () => WebGLRenderer>();
    expectTypeOf(WebGPURenderer.init).toBeFunction();
    expectTypeOf(WebGPURenderer.init).toMatchTypeOf<() => Promise<WebGPURenderer>>();
  });

  it("material program bindings expose the uniform scope", () => {
    const scene = new Scene();
    const program = new MeshBasicMaterial().build(scene);
    const scopes = program.uniforms.map((u) => u.scope) as string[];
    expectTypeOf(scopes).toEqualTypeOf<string[]>();
  });
});
