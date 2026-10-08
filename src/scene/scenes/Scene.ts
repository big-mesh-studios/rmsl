import type { Color } from "../math/Color";
import { Object3D } from "../core/Object3D";

/**
 * The root of a scene graph. Holds the background color and provides the
 * tree that a renderer traverses.
 */
export class Scene extends Object3D {
  readonly isScene = true;

  /**
   * The colour `render` clears to, or `null` (the default, as in three.js) to
   * clear to the renderer's clear colour.
   */
  background: Color | null = null;
}
