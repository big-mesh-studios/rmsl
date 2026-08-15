import { BufferGeometry } from "./BufferGeometry";
import { BufferAttribute } from "./BufferAttribute";

/** A flat circle in the XY plane, like three.js's `CircleGeometry`. */
export class CircleGeometry extends BufferGeometry {
  constructor(radius = 1, segments = 32, thetaStart = 0, thetaLength = Math.PI * 2) {
    super();

    segments = Math.max(3, segments);

    const indices: number[] = [];
    const vertices: number[] = [];
    const normals: number[] = [];
    const uvs: number[] = [];

    vertices.push(0, 0, 0);
    normals.push(0, 0, 1);
    uvs.push(0.5, 0.5);

    for (let s = 0; s <= segments; s++) {
      const segment = thetaStart + s / segments * thetaLength;
      vertices.push(radius * Math.cos(segment), radius * Math.sin(segment), 0);
      normals.push(0, 0, 1);
      uvs.push(Math.cos(segment) * 0.5 + 0.5, Math.sin(segment) * 0.5 + 0.5);
    }

    for (let i = 1; i <= segments; i++) {
      indices.push(i, i + 1, 0);
    }

    this.setAttribute("position", new BufferAttribute(new Float32Array(vertices), 3));
    this.setAttribute("normal", new BufferAttribute(new Float32Array(normals), 3));
    this.setAttribute("uv", new BufferAttribute(new Float32Array(uvs), 2));
    this.setIndex(indices);
  }
}
