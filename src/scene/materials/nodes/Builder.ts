import {
  attribute, uniformRaw, varying,
  type AttributeNode, type ShaderType, type UniformNode, type VaryingNode,
} from "../../../rmsl";
import type { Camera } from "../../cameras/Camera";
import type { Mesh } from "../../objects/Mesh";
import type { Texture } from "../../textures/Texture";

export type UniformScope = "camera" | "object" | "material";

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
}

export interface VaryingBinding {
  node: VaryingNode<ShaderType>;
  name: string;
}

export interface SamplerBinding {
  node: UniformNode<"sampler2D">;
  name: string;
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
  readonly samplers = new Map<string, SamplerBinding>();

  attribute<T extends ShaderType>(name: string, type: T): AttributeNode<T> {
    let existing = this.attributes.get(name);
    if (!existing) {
      const node = attribute(type);
      this.attributes.set(name, { node, name });
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

  /** A sampler uniform bound to a texture supplied by `texture`. */
  sampler(
    name: string,
    texture: () => Texture | null,
  ): UniformNode<"sampler2D"> {
    let existing = this.samplers.get(name);
    if (!existing) {
      const node = uniformRaw(name, "sampler2D");
      this.samplers.set(name, { node, name, texture });
      existing = this.samplers.get(name)!;
    }
    return existing.node;
  }

  // === Built-in attribute accessors ===
  get position(): AttributeNode<"vec3"> {
    return this.attribute("position", "vec3");
  }

  get normal(): AttributeNode<"vec3"> {
    return this.attribute("normal", "vec3");
  }

  get uv(): AttributeNode<"vec2"> {
    return this.attribute("uv", "vec2");
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
