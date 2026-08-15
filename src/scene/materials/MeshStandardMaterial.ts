import {
  vec3, vec4,
  type Node, type UniformNode,
} from "../../rmsl";
import { NodeMaterial, resolveSlot } from "./NodeMaterial";
import { Builder } from "./nodes/Builder";
import {
  collectLights, type LightUniforms,
} from "./nodes/lights";
import { pointLightAttenuation, standardLight } from "./nodes/lighting";
import { Color } from "../math/Color";
import { Side } from "./Material";
import type { Scene } from "../scenes/Scene";

/**
 * A physically-based-ish lit material in the metalness workflow: a Lambert
 * diffuse plus a simplified GGX specular, with ambient, directional and point
 * lights. Like a trimmed-down three.js `MeshStandardMaterial`.
 */
export class MeshStandardMaterial extends NodeMaterial {
  readonly isMeshStandardMaterial = true;

  color = new Color(1, 1, 1);
  roughness = 1;
  metalness = 0;
  emissive = new Color(0, 0, 0);

  protected colorUniform?: UniformNode<"vec3">;
  protected opacityUniform?: UniformNode<"float">;
  protected roughnessUniform?: UniformNode<"float">;
  protected metalnessUniform?: UniformNode<"float">;
  protected emissiveUniform?: UniformNode<"vec3">;
  protected lights?: LightUniforms;

  constructor(parameters: {
    color?: Color | number;
    roughness?: number;
    metalness?: number;
    emissive?: Color | number;
    opacity?: number;
    transparent?: boolean;
    side?: Side;
  } = {}) {
    super();
    if (parameters.color !== undefined) {
      this.color = typeof parameters.color === "number"
        ? new Color().setHex(parameters.color)
        : parameters.color.clone();
    }
    if (parameters.roughness !== undefined) this.roughness = parameters.roughness;
    if (parameters.metalness !== undefined) this.metalness = parameters.metalness;
    if (parameters.emissive !== undefined) {
      this.emissive = typeof parameters.emissive === "number"
        ? new Color().setHex(parameters.emissive)
        : parameters.emissive.clone();
    }
    if (parameters.opacity !== undefined) this.opacity = parameters.opacity;
    if (parameters.transparent !== undefined) this.transparent = parameters.transparent;
    if (parameters.side !== undefined) this.side = parameters.side;
  }

  protected setup(b: Builder, scene: Scene): void {
    this.colorUniform = b.materialUniform("materialColor", "vec3", () => this.color.toArray());
    this.opacityUniform = b.materialUniform("materialOpacity", "float", () => this.opacity);
    this.roughnessUniform = b.materialUniform("materialRoughness", "float", () => this.roughness);
    this.metalnessUniform = b.materialUniform("materialMetalness", "float", () => this.metalness);
    this.emissiveUniform = b.materialUniform("materialEmissive", "vec3", () => this.emissive.toArray());
    this.lights = collectLights(b, scene);
  }

  protected buildFragmentBody(b: Builder): Node<"vec4"> {
    const albedo = resolveSlot(this.colorNode, b) ?? this.colorUniform!;
    const opacity = resolveSlot(this.opacityNode, b) ?? this.opacityUniform!;
    const roughness = resolveSlot(this.roughnessNode, b) ?? this.roughnessUniform!;
    const metalness = resolveSlot(this.metalnessNode, b) ?? this.metalnessUniform!;
    const emissive = resolveSlot(this.emissiveNode, b) ?? this.emissiveUniform!;

    const normal = (resolveSlot(this.normalNode, b) ?? b.normalWorld).normalize().toVar();
    const viewDir = b.viewDirection.normalize().toVar();
    const f0 = vec3(0.04).mix(albedo, metalness).toVar();

    const color = vec3(0).toVar();
    color.addAssign(emissive);
    color.addAssign(albedo.mul(this.lights!.ambient));

    for (const light of this.lights!.directionals) {
      const dir = light.direction.normalize();
      color.addAssign(standardLight(albedo, normal, viewDir, dir, light.color, roughness, metalness, f0));
    }
    for (const light of this.lights!.points) {
      const dir = light.position.sub(b.positionWorld).normalize();
      const attenuation = pointLightAttenuation(light.position, b.positionWorld, light.distance, light.decay);
      color.addAssign(standardLight(albedo, normal, viewDir, dir, light.color, roughness, metalness, f0).mul(attenuation));
    }

    return vec4(color, opacity);
  }
}
