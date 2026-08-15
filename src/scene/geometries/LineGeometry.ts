import { LineSegmentsGeometry } from "./LineSegmentsGeometry";
import { Vector3 } from "../math/Vector3";

/**
 * A polyline drawn between a chain of vertices, like three.js's `LineGeometry`:
 * consecutive points become line segments, so `n` points yield `n - 1`
 * segments. Points can be passed as a flat array (`xyz xyz …`), a typed array,
 * or `Vector3` instances.
 */
export class LineGeometry extends LineSegmentsGeometry {
  readonly isLineGeometry = true;

  type = "LineGeometry";

  constructor(points?: number[] | Float32Array | Vector3[]) {
    super();
    if (points === undefined || points.length === 0) return;

    let flat: Float32Array;
    if (Array.isArray(points) && typeof (points as Vector3[])[0] === "object" && "x" in (points as Vector3[])[0]) {
      flat = new Float32Array((points as Vector3[]).flatMap((p) => [p.x, p.y, p.z]));
    } else {
      flat = points instanceof Float32Array ? points : new Float32Array(points as number[]);
    }

    const pointCount = Math.floor(flat.length / 3);
    const segmentCount = Math.max(pointCount - 1, 0);
    const segments = new Float32Array(segmentCount * 6);
    for (let i = 0; i < segmentCount; i++) {
      segments[i * 6] = flat[i * 3];
      segments[i * 6 + 1] = flat[i * 3 + 1];
      segments[i * 6 + 2] = flat[i * 3 + 2];
      segments[i * 6 + 3] = flat[i * 3 + 3];
      segments[i * 6 + 4] = flat[i * 3 + 4];
      segments[i * 6 + 5] = flat[i * 3 + 5];
    }
    this.setPositions(segments);
  }
}
