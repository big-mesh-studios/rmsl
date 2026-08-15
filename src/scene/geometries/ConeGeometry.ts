import { BufferGeometry } from "./BufferGeometry";
import { CylinderGeometry } from "./CylinderGeometry";

/** A cone on the y axis, like three.js's `ConeGeometry`. */
export class ConeGeometry extends BufferGeometry {
  constructor(radius = 1, height = 1, radialSegments = 32, heightSegments = 1, openEnded = false, thetaStart = 0, thetaLength = Math.PI * 2) {
    super();
    // A cone is a cylinder with a top radius of zero.
    const cylinder = new CylinderGeometry(
      0, radius, height, radialSegments, heightSegments, openEnded, thetaStart, thetaLength,
    );
    this.attributes = cylinder.attributes;
    this.index = cylinder.index;
  }
}
