import {
  attribute, uniformRaw, varying,
  type AttributeNode, type ShaderType, type UniformNode, type VaryingNode,
} from "../../../rmsl";
import type { Camera } from "../../cameras/Camera";
import type { Mesh } from "../../objects/Mesh";
import type { Texture } from "../../textures/Texture";

export type UniformScope = "camera" | "object" | "material" | "renderer";

export interface UniformContext {
  camera: Camera;
  mesh: Mesh;
}

export interface UniformBinding {
  node: UniformNode<ShaderType>;
  /** Logical name the renderer uses to derive camera/object values. */
  name: string;
  scope: UniformScope;
  /** Value source; material-scope uniforms always carry one. */
  value?: (ctx: UniformContext) => number | number[];
}

export interface AttributeBinding {
  node: AttributeNode<ShaderType>;
  name: string;
  /**
   * How the GPU advances the buffer: `"instance"` fetches one element per
   * instance (three.js's `InstancedBufferAttribute`), `"vertex"` one per
   * vertex. The material declares this — the renderer's pipeline layout and
   * draw calls depend on it.
   */
  stepMode: "vertex" | "instance";
}

export interface VaryingBinding {
  node: VaryingNode<ShaderType>;
  name: string;
}

/**
 * The sampler shader types the scene graph can bind a texture to. Integer
 * samplers read unfiltered texels and return integer vectors; 3D samplers
 * take a volume coordinate.
 */
export type SamplerShaderType =
  | "sampler2D" | "sampler3D"
  | "isampler2D" | "isampler3D"
  | "usampler2D" | "usampler3D";

/** A sampler binding narrowed to no particular type, for whole-program lists. */
export type AnySamplerBinding = SamplerBinding<SamplerShaderType>;

export interface SamplerBinding<T extends SamplerShaderType = "sampler2D"> {
  node: UniformNode<T>;
  name: string;
  type: T;
  texture: () => Texture | null;
}

/**
 * One material's compile pass. Resolves the built-in accessors (`position`,
 * `normal`, `uv`, `cameraPosition`, the matrices, ...) to ordinary RMSL
 * attribute/uniform/varying nodes, lazily and exactly once per material, and
 * records what was created so a renderer knows what to bind.
 */
export class Builder {
  readonly uniforms = new Map<string, UniformBinding>();
  readonly attributes = new Map<string, AttributeBinding>();
  readonly varyings = new Map<string, VaryingBinding>();
  readonly samplers = new Map<string, AnySamplerBinding>();

  /**
   * The shader stage currently being built. The built-in geometry accessors
   * resolve differently per stage: a vertex reads the raw attribute, while a
   * fragment can only read the interpolated varying the vertex stage wrote.
   */
  stage: "vertex" | "fragment" = "vertex";

  /**
   * Whether the mesh this material draws is an `InstancedMesh`. Set by the
   * material build from the renderer's per-object flags; the vertex body uses
   * it to multiply by `instanceMatrix` and the accessors below only resolve
   * to real attributes when it is true.
   */
  instancing = false;

  /** Whether the instanced mesh carries a per-instance color. */
  instancingColor = false;

  attribute<T extends ShaderType>(name: string, type: T, stepMode: "vertex" | "instance" = "vertex"): AttributeNode<T> {
    let existing = this.attributes.get(name);
    if (!existing) {
      const node = attribute(type);
      this.attributes.set(name, { node, name, stepMode });
      existing = this.attributes.get(name)!;
    }
    return existing.node as AttributeNode<T>;
  }

  varying<T extends ShaderType>(name: string, type: T): VaryingNode<T> {
    let existing = this.varyings.get(name);
    if (!existing) {
      const node = varying(type);
      this.varyings.set(name, { node, name });
      existing = this.varyings.get(name)!;
    }
    return existing.node as VaryingNode<T>;
  }

  uniform<T extends ShaderType>(
    name: string,
    type: T,
    scope: UniformScope,
    value?: (ctx: UniformContext) => number | number[],
  ): UniformNode<T> {
    let existing = this.uniforms.get(name) as UniformBinding | undefined;
    if (!existing) {
      const node = uniformRaw(name, type);
      this.uniforms.set(name, { node, name, scope, value });
      existing = this.uniforms.get(name)!;
    }
    return existing.node as UniformNode<T>;
  }

  /** A material-scoped uniform whose current value comes from `value`. */
  materialUniform<T extends ShaderType>(
    name: string,
    type: T,
    value: (ctx: UniformContext) => number | number[],
  ): UniformNode<T> {
    return this.uniform(name, type, "material", value);
  }

  /**
   * A renderer-scoped uniform (canvas resolution, device pixel ratio, ...).
   * The renderer supplies the value by name at draw time; no per-material
   * value function is involved.
   */
  rendererUniform<T extends ShaderType>(name: string, type: T): UniformNode<T> {
    return this.uniform(name, type, "renderer");
  }

  /** A `sampler2D` uniform bound to a texture supplied by `texture`. */
  sampler(
    name: string,
    texture: () => Texture | null,
  ): UniformNode<"sampler2D">;
  /**
   * A sampler uniform of the given type (integer or 3D) bound to a texture
   * supplied by `texture`.
   */
  sampler<T extends SamplerShaderType>(
    name: string,
    type: T,
    texture: () => Texture | null,
  ): UniformNode<T>;
  sampler<T extends SamplerShaderType>(
    name: string,
    typeOrTexture: T | (() => Texture | null),
    texture?: () => Texture | null,
  ): UniformNode<T> {
    const type: SamplerShaderType = typeof typeOrTexture === "string" ? typeOrTexture : "sampler2D";
    const textureFn = typeof typeOrTexture === "string"
      ? texture!
      : typeOrTexture;
    let existing = this.samplers.get(name);
    if (!existing) {
      const node = uniformRaw(name, type);
      this.samplers.set(name, { node, name, type, texture: textureFn });
      existing = this.samplers.get(name)!;
    }
    return existing.node as UniformNode<T>;
  }

  // === Built-in attribute accessors ===
  //
  // Each resolves to the raw attribute in the vertex stage and to the varying
  // the vertex writes in the fragment stage — the fragment cannot read a vertex
  // input, so `b.uv.x` inside a `colorNode` or `fragmentNode` must mean the
  // interpolated uv, not the geometry attribute.

  get position(): AttributeNode<"vec3"> {
    return this.stage === "vertex"
      ? this.attribute("position", "vec3")
      : this.varying("positionWorld", "vec3");
  }

  get normal(): AttributeNode<"vec3"> {
    return this.stage === "vertex"
      ? this.attribute("normal", "vec3")
      : this.varying("normalWorld", "vec3");
  }

  get uv(): AttributeNode<"vec2"> {
    return this.stage === "vertex"
      ? this.attribute("uv", "vec2")
      : this.varying("uv", "vec2");
  }

  /**
   * The per-instance transform of an `InstancedMesh`, an instanced `mat4`
   * attribute. Only meaningful when the mesh is instanced (`this.instancing`).
   */
  get instanceMatrix(): AttributeNode<"mat4"> {
    return this.attribute("instanceMatrix", "mat4", "instance");
  }

  /**
   * The per-instance color of an `InstancedMesh`, an instanced `vec3`
   * attribute. Only meaningful when the mesh carries instance colors
   * (`this.instancingColor`).
   */
  get instanceColor(): AttributeNode<"vec3"> {
    return this.attribute("instanceColor", "vec3", "instance");
  }

  // === Built-in varying accessors (vertex writes, fragment reads) ===
  get positionWorld(): VaryingNode<"vec3"> {
    return this.varying("positionWorld", "vec3");
  }

  get normalWorld(): VaryingNode<"vec3"> {
    return this.varying("normalWorld", "vec3");
  }

  get uvVarying(): VaryingNode<"vec2"> {
    return this.varying("uv", "vec2");
  }

  /**
   * The per-instance color carried from the vertex stage, where the instanced
   * `instanceColor` attribute lives, into the fragment stage. Only written
   * when the mesh carries instance colors (`this.instancingColor`).
   */
  get instanceColorVarying(): VaryingNode<"vec3"> {
    return this.varying("instanceColor", "vec3");
  }

  // === Built-in uniform accessors ===
  get cameraPosition(): UniformNode<"vec3"> {
    return this.uniform("cameraPosition", "vec3", "camera");
  }

  get viewMatrix(): UniformNode<"mat4"> {
    return this.uniform("viewMatrix", "mat4", "camera");
  }

  get projectionMatrix(): UniformNode<"mat4"> {
    return this.uniform("projectionMatrix", "mat4", "camera");
  }

  get modelMatrix(): UniformNode<"mat4"> {
    return this.uniform("modelMatrix", "mat4", "object");
  }

  get normalMatrix(): UniformNode<"mat3"> {
    return this.uniform("normalMatrix", "mat3", "object");
  }

  /** The camera-relative transform, composed in the shader. */
  get modelViewMatrix(): any {
    return this.viewMatrix.mul(this.modelMatrix);
  }

  /** World direction from the current fragment toward the camera. */
  get viewDirection(): any {
    return this.cameraPosition.sub(this.positionWorld).normalize();
  }
}
