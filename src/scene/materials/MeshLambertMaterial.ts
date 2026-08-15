import {
  vec3, vec4,
  type Node, type UniformNode,
} from "../../rmsl";
import { NodeMaterial, resolveSlot } from "./NodeMaterial";
import { Builder } from "./nodes/Builder";
import { collectLights, type LightUniforms } from "./nodes/lights";
import { pointLightAttenuation, lambertDiffuse } from "./nodes/lighting";
import { Color } from "../math/Color";
import { Side } from "./Material";
import type { Scene } from "../scenes/Scene";

/**
 * A diffuse-only lit material with ambient, directional and point lights, like
 * three.js's `MeshLambertMaterial`.
 */
export class MeshLambertMaterial extends NodeMaterial {
  readonly isMeshLambertMaterial = true;

  color = new Color(1, 1, 1);
  emissive = new Color(0, 0, 0);

  protected colorUniform?: UniformNode<"vec3">;
  protected opacityUniform?: UniformNode<"float">;
  protected emissiveUniform?: UniformNode<"vec3">;
  protected lights?: LightUniforms;

  constructor(parameters: {
    color?: Color | number;
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
    this.emissiveUniform = b.materialUniform("materialEmissive", "vec3", () => this.emissive.toArray());
    this.lights = collectLights(b, scene);
  }

  protected buildFragmentBody(b: Builder): Node<"vec4"> {
    const albedo = resolveSlot(this.colorNode, b) ?? this.colorUniform!;
    const opacity = resolveSlot(this.opacityNode, b) ?? this.opacityUniform!;
    const emissive = resolveSlot(this.emissiveNode, b) ?? this.emissiveUniform!;
    const normal = (resolveSlot(this.normalNode, b) ?? b.normalWorld).normalize().toVar();

    const color = vec3(0).toVar();
    color.addAssign(emissive);
    color.addAssign(albedo.mul(this.lights!.ambient));

    for (const light of this.lights!.directionals) {
      color.addAssign(lambertDiffuse(albedo, normal, light.direction.normalize(), light.color));
    }
    for (const light of this.lights!.points) {
      const dir = light.position.sub(b.positionWorld).normalize();
      const attenuation = pointLightAttenuation(light.position, b.positionWorld, light.distance, light.decay);
      color.addAssign(lambertDiffuse(albedo, normal, dir, light.color).mul(attenuation));
    }

    return vec4(color, opacity);
  }
}
