import { BufferGeometry } from "./BufferGeometry";
import { BufferAttribute } from "./BufferAttribute";
import { Matrix4 } from "../math/Matrix4";
import { Vector3 } from "../math/Vector3";

// A unit quad strip that the line material expands into a wide ribbon: two
// middle quads (the "body" of the segment) plus an endcap quad at each end.
// `position` y runs [-1, 2], `uv` mirrors it, and the index fans the strip
// into six triangles — matching three.js's `LineSegmentsGeometry`.
const _positions = [-1, 2, 0, 1, 2, 0, -1, 1, 0, 1, 1, 0, -1, 0, 0, 1, 0, 0, -1, -1, 0, 1, -1, 0];
const _uvs = [-1, 2, 1, 2, -1, 1, 1, 1, -1, -1, 1, -1, -1, -2, 1, -2];
const _index = [0, 2, 1, 2, 3, 1, 2, 4, 3, 4, 5, 3, 4, 6, 5, 6, 7, 5];

/**
 * A series of line segments drawn between pairs of vertices, like three.js's
 * `LineSegmentsGeometry`. Each segment lives in the per-instance attributes
 * `instanceStart`/`instanceEnd` (and optionally `instanceDistanceStart`/
 * `instanceDistanceEnd` for dashes, `instanceColorStart`/`instanceColorEnd`
 * for vertex colors); the quad strip is expanded by {@link Line2NodeMaterial}
 * into a screen- or world-width ribbon.
 */
export class LineSegmentsGeometry extends BufferGeometry {
  readonly isLineSegmentsGeometry = true;

  type = "LineSegmentsGeometry";

  constructor() {
    super();
    this.setIndex(_index);
    this.setAttribute("position", new BufferAttribute(new Float32Array(_positions), 3));
    this.setAttribute("uv", new BufferAttribute(new Float32Array(_uvs), 2));
  }

  /**
   * Sets the segment positions. The array length must be a multiple of six:
   * each segment is a `(xyz xyz)` start/end pair.
   */
  setPositions(array: Float32Array | number[]): this {
    const positions = array instanceof Float32Array ? array : new Float32Array(array);
    const segmentCount = Math.floor(positions.length / 6);
    const start = new Float32Array(segmentCount * 3);
    const end = new Float32Array(segmentCount * 3);
    for (let i = 0; i < segmentCount; i++) {
      const j = i * 6;
      start[i * 3] = positions[j];
      start[i * 3 + 1] = positions[j + 1];
      start[i * 3 + 2] = positions[j + 2];
      end[i * 3] = positions[j + 3];
      end[i * 3 + 1] = positions[j + 4];
      end[i * 3 + 2] = positions[j + 5];
    }
    this.setAttribute("instanceStart", new BufferAttribute(start, 3).setStepMode("instance"));
    this.setAttribute("instanceEnd", new BufferAttribute(end, 3).setStepMode("instance"));
    this.instanceCount = segmentCount;
    return this;
  }

  /**
   * Sets the per-segment vertex colors: one `(rgb rgb)` start/end pair per
   * segment. Requires the material's `vertexColors` to be enabled.
   */
  setColors(array: Float32Array | number[]): this {
    const colors = array instanceof Float32Array ? array : new Float32Array(array);
    const segmentCount = Math.floor(colors.length / 6);
    const start = new Float32Array(segmentCount * 3);
    const end = new Float32Array(segmentCount * 3);
    for (let i = 0; i < segmentCount; i++) {
      const j = i * 6;
      start[i * 3] = colors[j];
      start[i * 3 + 1] = colors[j + 1];
      start[i * 3 + 2] = colors[j + 2];
      end[i * 3] = colors[j + 3];
      end[i * 3 + 1] = colors[j + 4];
      end[i * 3 + 2] = colors[j + 5];
    }
    this.setAttribute("instanceColorStart", new BufferAttribute(start, 3).setStepMode("instance"));
    this.setAttribute("instanceColorEnd", new BufferAttribute(end, 3).setStepMode("instance"));
    return this;
  }

  /**
   * Computes the cumulative length along the line for each segment, stored in
   * `instanceDistanceStart`/`instanceDistanceEnd` — the data the dash shader
   * needs to place gaps in world-length units.
   */
  computeLineDistances(): this {
    const instanceStart = this.attributes.instanceStart;
    const instanceEnd = this.attributes.instanceEnd;
    if (!instanceStart || !instanceEnd) return this;
    const count = instanceStart.count;
    const distanceStart = new Float32Array(count);
    const distanceEnd = new Float32Array(count);
    let total = 0;
    for (let i = 0; i < count; i++) {
      distanceStart[i] = total;
      const dx = instanceEnd.getX(i) - instanceStart.getX(i);
      const dy = instanceEnd.getY(i) - instanceStart.getY(i);
      const dz = instanceEnd.getZ(i) - instanceStart.getZ(i);
      total += Math.sqrt(dx * dx + dy * dy + dz * dz);
      distanceEnd[i] = total;
    }
    this.setAttribute("instanceDistanceStart", new BufferAttribute(distanceStart, 1).setStepMode("instance"));
    this.setAttribute("instanceDistanceEnd", new BufferAttribute(distanceEnd, 1).setStepMode("instance"));
    return this;
  }

  /** Transforms the stored segments in place. */
  applyMatrix4(matrix: Matrix4): this {
    const start = this.attributes.instanceStart;
    const end = this.attributes.instanceEnd;
    if (start && end) {
      const sa = start.array as Float32Array;
      const ea = end.array as Float32Array;
      for (let i = 0; i < start.count; i++) {
        _v1.set(sa[i * 3], sa[i * 3 + 1], sa[i * 3 + 2]).applyMatrix4(matrix);
        sa[i * 3] = _v1.x;
        sa[i * 3 + 1] = _v1.y;
        sa[i * 3 + 2] = _v1.z;
        _v1.set(ea[i * 3], ea[i * 3 + 1], ea[i * 3 + 2]).applyMatrix4(matrix);
        ea[i * 3] = _v1.x;
        ea[i * 3 + 1] = _v1.y;
        ea[i * 3 + 2] = _v1.z;
      }
      start.needsUpdate = true;
      end.needsUpdate = true;
    }
    return this;
  }
}

const _v1 = new Vector3();
