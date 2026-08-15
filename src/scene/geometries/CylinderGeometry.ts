import { BufferGeometry } from "./BufferGeometry";
import { BufferAttribute } from "./BufferAttribute";

/**
 * A cylinder (or, with one radius zero, a cone) centered on the y axis, like
 * three.js's `CylinderGeometry`.
 */
export class CylinderGeometry extends BufferGeometry {
  constructor(radiusTop = 1, radiusBottom = 1, height = 1, radialSegments = 32, heightSegments = 1, openEnded = false, thetaStart = 0, thetaLength = Math.PI * 2) {
    super();

    radialSegments = Math.floor(radialSegments);
    heightSegments = Math.floor(heightSegments);

    const indices: number[] = [];
    const vertices: number[] = [];
    const normals: number[] = [];
    const uvs: number[] = [];

    let index = 0;
    const indexArray: number[][] = [];

    generateTorso();

    if (openEnded === false) {
      if (radiusTop > 0) generateCap(true);
      if (radiusBottom > 0) generateCap(false);
    }

    function generateTorso(): void {
      const normal = { x: 0, y: 0, z: 0 };

      for (let y = 0; y <= heightSegments; y++) {
        const indexRow: number[] = [];
        const v = y / heightSegments;
        const radius = v * (radiusBottom - radiusTop) + radiusTop;

        for (let x = 0; x <= radialSegments; x++) {
          const u = x / radialSegments;
          const theta = u * thetaLength + thetaStart;

          const sinTheta = Math.sin(theta);
          const cosTheta = Math.cos(theta);

          vertices.push(sinTheta * radius, -v * height + height / 2, cosTheta * radius);
          normal.x = sinTheta;
          normal.y = 0;
          normal.z = cosTheta;
          normals.push(normal.x, normal.y, normal.z);
          uvs.push(u, 1 - v);
          indexRow.push(index++);
        }
        indexArray.push(indexRow);
      }

      for (let x = 0; x < radialSegments; x++) {
        for (let y = 0; y < heightSegments; y++) {
          const a = indexArray[y][x];
          const b = indexArray[y + 1][x];
          const c = indexArray[y + 1][x + 1];
          const d = indexArray[y][x + 1];
          indices.push(a, b, d, b, c, d);
        }
      }
    }

    function generateCap(top: boolean): void {
      const radius = top ? radiusTop : radiusBottom;
      const sign = top ? 1 : -1;
      const centerIndexStart = index;

      for (let x = 1; x <= radialSegments; x++) {
        vertices.push(0, sign * height / 2, 0);
        normals.push(0, sign, 0);
        uvs.push(0.5, 0.5);
        index++;
      }

      const centerIndexEnd = index;

      for (let x = 0; x <= radialSegments; x++) {
        const u = x / radialSegments;
        const theta = u * thetaLength + thetaStart;
        const cosTheta = Math.cos(theta);
        const sinTheta = Math.sin(theta);
        vertices.push(sinTheta * radius, sign * height / 2, cosTheta * radius);
        normals.push(0, sign, 0);
        uvs.push(cosTheta * 0.5 + 0.5, sinTheta * 0.5 * sign + 0.5);
        index++;
      }

      for (let x = 0; x < radialSegments; x++) {
        const c = centerIndexStart + x;
        const i = centerIndexEnd + x;
        if (top) indices.push(i, i + 1, c);
        else indices.push(i + 1, i, c);
      }
    }

    this.setAttribute("position", new BufferAttribute(new Float32Array(vertices), 3));
    this.setAttribute("normal", new BufferAttribute(new Float32Array(normals), 3));
    this.setAttribute("uv", new BufferAttribute(new Float32Array(uvs), 2));
    this.setIndex(indices);
  }
}
