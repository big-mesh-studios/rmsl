import { Light } from "./Light";
import { Color } from "../math/Color";

/**
 * Light radiating from a point in all directions, with optional distance
 * falloff, like three.js's `PointLight(color, intensity, distance, decay)`.
 */
export class PointLight extends Light {
  readonly isPointLight = true;

  distance: number;
  decay: number;

  constructor(color?: Color | number, intensity?: number, distance = 0, decay = 2) {
    super(color, intensity);
    this.distance = distance;
    this.decay = decay;
  }
}
