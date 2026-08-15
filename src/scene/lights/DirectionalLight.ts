import { Light } from "./Light";
import { Object3D } from "../core/Object3D";
import { Color } from "../math/Color";

/**
 * Light emitted along a single direction, with its position used only to
 * determine the direction it points (at the origin by default), like
 * three.js's `DirectionalLight`.
 */
export class DirectionalLight extends Light {
  readonly isDirectionalLight = true;

  /** The position the light points at; the light sits at `Object3D.position`. */
  target = new Object3D();

  constructor(color?: Color | number, intensity?: number) {
    super(color, intensity);
  }
}
