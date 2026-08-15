import { Matrix4 } from "../math/Matrix4";
import { Vector3 } from "../math/Vector3";
import { Object3D } from "../core/Object3D";

/**
 * The base camera. Holds the projection matrix and the inverse of the world
 * matrix (the view matrix), like three.js's `Camera`.
 */
export class Camera extends Object3D {
  readonly isCamera = true;

  readonly matrixWorldInverse = new Matrix4();
  readonly projectionMatrix = new Matrix4();
  readonly projectionMatrixInverse = new Matrix4();

  updateMatrixWorld(force = false): void {
    super.updateMatrixWorld(force);
    this.matrixWorldInverse.copy(this.matrixWorld).invert();
  }

  getWorldDirection(target = new Vector3()): Vector3 {
    return super.getWorldDirection(target);
  }
}
