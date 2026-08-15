import { Color } from "../math/Color";
import { Object3D } from "../core/Object3D";

/**
 * The root of a scene graph. Holds the background color and provides the
 * tree that a renderer traverses.
 */
export class Scene extends Object3D {
  readonly isScene = true;

  background: Color | null = new Color(0, 0, 0);
}
