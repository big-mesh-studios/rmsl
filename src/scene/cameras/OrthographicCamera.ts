import { Camera } from "./Camera";

/**
 * An orthographic camera, like three.js's
 * `OrthographicCamera(left, right, top, bottom, near, far)`.
 */
export class OrthographicCamera extends Camera {
  readonly isOrthographicCamera = true;

  left: number;
  right: number;
  top: number;
  bottom: number;
  near: number;
  far: number;

  constructor(left = -1, right = 1, top = 1, bottom = -1, near = 0.1, far = 2000) {
    super();
    this.left = left;
    this.right = right;
    this.top = top;
    this.bottom = bottom;
    this.near = near;
    this.far = far;
    this.updateProjectionMatrix();
  }

  updateProjectionMatrix(): void {
    this.projectionMatrix.makeOrthographic(
      this.left, this.right, this.top, this.bottom, this.near, this.far,
    );
    this.projectionMatrixInverse.copy(this.projectionMatrix).invert();
  }
}
