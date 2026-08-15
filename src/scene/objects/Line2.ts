import { LineSegments2 } from "./LineSegments2";
import { LineGeometry } from "../geometries/LineGeometry";
import { Line2NodeMaterial } from "../materials/Line2NodeMaterial";

/**
 * A polyline drawn between consecutive vertices, like three.js's `Line2`:
 * extends {@link LineSegments2} and builds its segments from a chain of points
 * via {@link LineGeometry}.
 */
export class Line2 extends LineSegments2 {
  readonly isLine2 = true;

  type = "Line2";

  declare geometry: LineGeometry;

  constructor(
    geometry: LineGeometry = new LineGeometry(),
    material: Line2NodeMaterial = new Line2NodeMaterial(),
  ) {
    super(geometry, material);
  }
}
