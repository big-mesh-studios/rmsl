import { Light } from "./Light";
import { Color } from "../math/Color";

/**
 * Light that illuminates every surface equally, regardless of direction.
 * Equivalent to three.js's `AmbientLight(color, intensity)`.
 */
export class AmbientLight extends Light {
  readonly isAmbientLight = true;

  constructor(color?: Color | number, intensity?: number) {
    super(color, intensity);
  }
}
