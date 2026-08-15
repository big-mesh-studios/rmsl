import { Camera } from "./Camera";
import { degToRad } from "../math/MathUtils";

/**
 * A perspective camera with a vertical field of view, like three.js's
 * `PerspectiveCamera(fov, aspect, near, far)`.
 */
export class PerspectiveCamera extends Camera {
  readonly isPerspectiveCamera = true;

  fov: number;
  aspect: number;
  near: number;
  far: number;

  constructor(fov = 50, aspect = 1, near = 0.1, far = 2000) {
    super();
    this.fov = fov;
    this.aspect = aspect;
    this.near = near;
    this.far = far;
    this.updateProjectionMatrix();
  }

  updateProjectionMatrix(): void {
    const near = this.near;
    const top = near * Math.tan(degToRad(0.5 * this.fov));
    const height = 2 * top;
    const width = this.aspect * height;
    const left = -0.5 * width;
    this.projectionMatrix.makePerspective(left, left + width, top, top - height, near, this.far);
    this.projectionMatrixInverse.copy(this.projectionMatrix).invert();
  }
}
