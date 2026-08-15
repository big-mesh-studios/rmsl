// Column-major mat4 helpers. Perspective / lookAt / inverse and the fullscreen
// quad verts are shared with the other apps (`apps/shared/shader.ts`); the
// composition and rigid-body builders live here.
import {
  mat4Perspective, mat4LookAt, mat4Inverse, quadVerts,
} from "../../shared/shader";

export { mat4Perspective, mat4LookAt, mat4Inverse, quadVerts };

/** `a · b` — column-major matrix product (apply `b` first, then `a`). */
export function mat4Multiply(a: Float32Array, b: Float32Array): Float32Array<ArrayBuffer> {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    const bc0 = b[c * 4 + 0], bc1 = b[c * 4 + 1], bc2 = b[c * 4 + 2], bc3 = b[c * 4 + 3];
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] = a[r] * bc0 + a[4 + r] * bc1 + a[8 + r] * bc2 + a[12 + r] * bc3;
    }
  }
  return out;
}

export function mat4Translation(x: number, y: number, z: number): Float32Array<ArrayBuffer> {
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    x, y, z, 1,
  ]);
}

export function mat4RotationZ(angle: number): Float32Array<ArrayBuffer> {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return new Float32Array([
    c, s, 0, 0,
    -s, c, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
}

export function mat4Identity(): Float32Array<ArrayBuffer> {
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
}

/** Transform a direction vector by a matrix's upper-left 3x3 (no translation). */
export function mat3TransformDirection(m: Float32Array, v: [number, number, number]): Float32Array<ArrayBuffer> {
  return new Float32Array([
    m[0] * v[0] + m[4] * v[1] + m[8] * v[2],
    m[1] * v[0] + m[5] * v[1] + m[9] * v[2],
    m[2] * v[0] + m[6] * v[1] + m[10] * v[2],
  ]);
}
