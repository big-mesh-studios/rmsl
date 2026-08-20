import {
  vec4,
  type Node, type UniformNode, type GLSLPrecision,
} from "../../rmsl";
import { NodeMaterial, resolveSlot, type SlotValue } from "./NodeMaterial";
import { Builder } from "./nodes/Builder";
import { Color } from "../math/Color";
import { Side } from "./Material";
import type { Scene } from "../scenes/Scene";
import type { Texture } from "../textures/Texture";

/**
 * An unlit material: the color (optionally multiplied by a texture map),
 * written straight to the output. Like three.js's `MeshBasicMaterial`.
 */
export class MeshBasicMaterial extends NodeMaterial {
  readonly isMeshBasicMaterial = true;

  color = new Color(1, 1, 1);
  map: Texture | null = null;

  protected colorUniform?: UniformNode<"vec3">;
  protected opacityUniform?: UniformNode<"float">;
  protected mapUniform?: UniformNode<"sampler2D">;

  constructor(parameters: {
    color?: Color | number;
    map?: Texture | null;
    opacity?: number;
    transparent?: boolean;
    side?: Side;
    precision?: GLSLPrecision;
  } = {}) {
    super();
    if (parameters.color !== undefined) {
      this.color = typeof parameters.color === "number"
        ? new Color().setHex(parameters.color)
        : parameters.color.clone();
    }
    if (parameters.map !== undefined) this.map = parameters.map;
    if (parameters.opacity !== undefined) this.opacity = parameters.opacity;
    if (parameters.transparent !== undefined) this.transparent = parameters.transparent;
    if (parameters.side !== undefined) this.side = parameters.side;
    if (parameters.precision !== undefined) this.precision = parameters.precision;
  }

  protected setup(b: Builder, _scene: Scene): void {
    this.colorUniform = b.materialUniform("materialColor", "vec3", () => this.color.toArray());
    this.opacityUniform = b.materialUniform("materialOpacity", "float", () => this.opacity);
    if (this.map) this.mapUniform = b.sampler("map", () => this.map);
  }

  protected buildFragmentBody(b: Builder): Node<"vec4"> {
    const color = resolveSlot(this.colorNode, b) ?? this.colorUniform!;
    const opacity = resolveSlot(this.opacityNode, b) ?? this.opacityUniform!;
    const uv = resolveSlot(this.uvNode, b) ?? b.uvVarying;
    const finalColor = this.mapUniform
      ? color.mul(this.mapUniform.texture(uv).rgb)
      : color;
    return vec4(finalColor, opacity);
  }
}
