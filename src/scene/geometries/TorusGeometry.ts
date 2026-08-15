import { BufferGeometry } from "./BufferGeometry";
import { BufferAttribute } from "./BufferAttribute";
import { Vector3 } from "../math/Vector3";

/** A torus around the y axis, like three.js's `TorusGeometry`. */
export class TorusGeometry extends BufferGeometry {
  constructor(radius = 1, tube = 0.4, radialSegments = 12, tubularSegments = 48, arc = Math.PI * 2) {
    super();

    radialSegments = Math.floor(radialSegments);
    tubularSegments = Math.floor(tubularSegments);

    const indices: number[] = [];
    const vertices: number[] = [];
    const normals: number[] = [];
    const uvs: number[] = [];

    let index = 0;
    const vertex = new Vector3();
    const normal = new Vector3();
    const center = new Vector3();

    for (let j = 0; j <= radialSegments; j++) {
      for (let i = 0; i <= tubularSegments; i++) {
        const u = i / tubularSegments * arc;
        const v = j / radialSegments * Math.PI * 2;

        center.x = radius * Math.cos(u);
        center.y = radius * Math.sin(u);

        vertex.x = (radius + tube * Math.cos(v)) * Math.cos(u);
        vertex.y = (radius + tube * Math.cos(v)) * Math.sin(u);
        vertex.z = tube * Math.sin(v);

        normal.x = Math.cos(v) * Math.cos(u);
        normal.y = Math.cos(v) * Math.sin(u);
        normal.z = Math.sin(v);

        vertices.push(vertex.x, vertex.y, vertex.z);
        normals.push(normal.x, normal.y, normal.z);
        uvs.push(i / tubularSegments, j / radialSegments);
        index++;
      }
    }

    for (let j = 1; j <= radialSegments; j++) {
      for (let i = 1; i <= tubularSegments; i++) {
        const a = (tubularSegments + 1) * j + i - 1;
        const b = (tubularSegments + 1) * (j - 1) + i - 1;
        const c = (tubularSegments + 1) * (j - 1) + i;
        const d = (tubularSegments + 1) * j + i;
        indices.push(a, b, d, b, c, d);
      }
    }

    this.setAttribute("position", new BufferAttribute(new Float32Array(vertices), 3));
    this.setAttribute("normal", new BufferAttribute(new Float32Array(normals), 3));
    this.setAttribute("uv", new BufferAttribute(new Float32Array(uvs), 2));
    this.setIndex(indices);
  }
}
