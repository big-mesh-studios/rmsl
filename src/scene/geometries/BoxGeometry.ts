import { BufferGeometry } from "./BufferGeometry";
import { BufferAttribute } from "./BufferAttribute";

/** A box of the given size centered at the origin, like three.js's `BoxGeometry`. */
export class BoxGeometry extends BufferGeometry {
  constructor(width = 1, height = 1, depth = 1) {
    super();

    const widthSegments = 1;
    const heightSegments = 1;
    const depthSegments = 1;

    const indices: number[] = [];
    const vertices: number[] = [];
    const normals: number[] = [];
    const uvs: number[] = [];

    let numberOfVertices = 0;
    buildPlane("z", "y", "x", -1, -1, depth, height, width, depthSegments, heightSegments);
    buildPlane("z", "y", "x", 1, -1, depth, height, -width, depthSegments, heightSegments);
    buildPlane("x", "z", "y", 1, 1, width, depth, height, widthSegments, depthSegments);
    buildPlane("x", "z", "y", 1, -1, width, depth, -height, widthSegments, depthSegments);
    buildPlane("x", "y", "z", 1, -1, width, height, depth, widthSegments, heightSegments);
    buildPlane("x", "y", "z", -1, -1, width, height, -depth, widthSegments, heightSegments);

    function buildPlane(
      u: string, v: string, w: string,
      udir: number, vdir: number,
      width: number, height: number, depth: number,
      gridX: number, gridY: number,
    ): void {
      const segmentWidth = width / gridX;
      const segmentHeight = height / gridY;

      const widthHalf = width / 2;
      const heightHalf = height / 2;
      const depthHalf = depth / 2;

      const wi = u === "x" ? 0 : u === "y" ? 1 : 2;
      const hi = v === "x" ? 0 : v === "y" ? 1 : 2;
      const di = w === "x" ? 0 : w === "y" ? 1 : 2;

      const gridX1 = gridX + 1;
      const gridY1 = gridY + 1;

      const start = numberOfVertices;
      for (let iy = 0; iy < gridY1; iy++) {
        const y = iy * segmentHeight - heightHalf;
        for (let ix = 0; ix < gridX1; ix++) {
          const x = ix * segmentWidth - widthHalf;

          const px = wi === 0 ? x * udir : hi === 0 ? y * vdir : depthHalf;
          const py = wi === 1 ? x * udir : hi === 1 ? y * vdir : depthHalf;
          const pz = wi === 2 ? x * udir : hi === 2 ? y * vdir : depthHalf;

          const nx = wi === 0 ? 0 : hi === 0 ? 0 : depth > 0 ? 1 : -1;
          const ny = wi === 1 ? 0 : hi === 1 ? 0 : depth > 0 ? 1 : -1;
          const nz = wi === 2 ? 0 : hi === 2 ? 0 : depth > 0 ? 1 : -1;

          vertices.push(px, py, pz);
          normals.push(nx, ny, nz);
          uvs.push(ix / gridX, 1 - iy / gridY);
          numberOfVertices++;
        }
      }

      for (let iy = 0; iy < gridY; iy++) {
        for (let ix = 0; ix < gridX; ix++) {
          const a = start + ix + gridX1 * iy;
          const b = start + ix + gridX1 * (iy + 1);
          const c = start + (ix + 1) + gridX1 * (iy + 1);
          const d = start + (ix + 1) + gridX1 * iy;
          indices.push(a, b, d, b, c, d);
        }
      }
    }

    this.setAttribute("position", new BufferAttribute(new Float32Array(vertices), 3));
    this.setAttribute("normal", new BufferAttribute(new Float32Array(normals), 3));
    this.setAttribute("uv", new BufferAttribute(new Float32Array(uvs), 2));
    this.setIndex(indices);
  }
}
