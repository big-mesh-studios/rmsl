export function mat4Perspective(fovY: number, aspect: number, near: number, far: number): Float32Array<ArrayBuffer> {
  const f = 1 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
}

export function mat4LookAt(
  eyeX: number,
  eyeY: number,
  eyeZ: number,
  cx: number,
  cy: number,
  cz: number,
  upX: number,
  upY: number,
  upZ: number,
): Float32Array<ArrayBuffer> {
  let zx = eyeX - cx,
    zy = eyeY - cy,
    zz = eyeZ - cz;
  const zl = Math.sqrt(zx * zx + zy * zy + zz * zz);
  zx /= zl;
  zy /= zl;
  zz /= zl;
  let xx = upY * zz - upZ * zy;
  let xy = upZ * zx - upX * zz;
  let xz = upX * zy - upY * zx;
  const xl = Math.sqrt(xx * xx + xy * xy + xz * xz);
  xx /= xl;
  xy /= xl;
  xz /= xl;
  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;
  return new Float32Array([
    xx,
    yx,
    zx,
    0,
    xy,
    yy,
    zy,
    0,
    xz,
    yz,
    zz,
    0,
    -(xx * eyeX + xy * eyeY + xz * eyeZ),
    -(yx * eyeX + yy * eyeY + yz * eyeZ),
    -(zx * eyeX + zy * eyeY + zz * eyeZ),
    1,
  ]);
}

export function mat4Inverse(m: Float32Array): Float32Array<ArrayBuffer> {
  const a00 = m[0],
    a01 = m[1],
    a02 = m[2],
    a03 = m[3];
  const a10 = m[4],
    a11 = m[5],
    a12 = m[6],
    a13 = m[7];
  const a20 = m[8],
    a21 = m[9],
    a22 = m[10],
    a23 = m[11];
  const a30 = m[12],
    a31 = m[13],
    a32 = m[14],
    a33 = m[15];
  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return new Float32Array(16);
  const id = 1 / det;
  const out = new Float32Array(16);
  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * id;
  out[1] = (-a01 * b11 + a02 * b10 - a03 * b09) * id;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * id;
  out[3] = (-a21 * b05 + a22 * b04 - a23 * b03) * id;
  out[4] = (-a10 * b11 + a12 * b08 - a13 * b07) * id;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * id;
  out[6] = (-a30 * b05 + a32 * b02 - a33 * b01) * id;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * id;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * id;
  out[9] = (-a00 * b10 + a01 * b08 - a03 * b06) * id;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * id;
  out[11] = (-a20 * b04 + a21 * b02 - a23 * b00) * id;
  out[12] = (-a10 * b09 + a11 * b07 - a12 * b06) * id;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * id;
  out[14] = (-a30 * b03 + a31 * b01 - a32 * b00) * id;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * id;
  return out;
}

export const quadVerts = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);

/** `a · b` — column-major matrix product (apply `b` first, then `a`). */
export function mat4Multiply(a: Float32Array, b: Float32Array): Float32Array<ArrayBuffer> {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    const bc0 = b[c * 4 + 0],
      bc1 = b[c * 4 + 1],
      bc2 = b[c * 4 + 2],
      bc3 = b[c * 4 + 3];
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] = a[r] * bc0 + a[4 + r] * bc1 + a[8 + r] * bc2 + a[12 + r] * bc3;
    }
  }
  return out;
}

export function mat4Translation(x: number, y: number, z: number): Float32Array<ArrayBuffer> {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
}

export function mat4RotationZ(angle: number): Float32Array<ArrayBuffer> {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return new Float32Array([c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

export function mat4Identity(): Float32Array<ArrayBuffer> {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

/** Transform a direction vector by a matrix's upper-left 3x3 (no translation). */
export function mat3TransformDirection(m: Float32Array, v: [number, number, number]): Float32Array<ArrayBuffer> {
  return new Float32Array([
    m[0] * v[0] + m[4] * v[1] + m[8] * v[2],
    m[1] * v[0] + m[5] * v[1] + m[9] * v[2],
    m[2] * v[0] + m[6] * v[1] + m[10] * v[2],
  ]);
}
