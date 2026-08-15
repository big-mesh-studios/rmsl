import { Color } from "../math/Color";
import { Object3D } from "../core/Object3D";

/**
 * The base of every light, like three.js's `Light`: a color and an intensity.
 * Lights are ordinary scene-graph nodes, so a directional or point light has
 * a position via `Object3D.position`.
 */
export class Light extends Object3D {
  readonly isLight = true;

  color: Color;
  intensity: number;

  constructor(color: Color | number = 0xffffff, intensity = 1) {
    super();
    this.color = new Color();
    if (typeof color === "number") this.color.setHex(color);
    else this.color.copy(color);
    this.intensity = intensity;
  }
}
