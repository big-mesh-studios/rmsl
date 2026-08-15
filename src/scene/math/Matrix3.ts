import { Matrix4 } from "./Matrix4";

/**
 * A 3x3 matrix, column-major in a nine-element array, following the same
 * storage convention as `Matrix4`. Used for transforming normals and
 * directions where the translation row is not wanted.
 */
export class Matrix3 {
  readonly elements: number[];

  constructor(n11 = 1, n21 = 0, n31 = 0, n12 = 0, n22 = 1, n32 = 0, n13 = 0, n23 = 0, n33 = 1) {
    this.elements = [n11, n21, n31, n12, n22, n32, n13, n23, n33];
  }

  identity(): this {
    this.set(1, 0, 0, 0, 1, 0, 0, 0, 1);
    return this;
  }

  clone(): Matrix3 {
    return new Matrix3().fromArray(this.elements);
  }

  copy(m: Matrix3): this {
    this.fromArray(m.elements);
    return this;
  }

  set(
    n11: number, n21: number, n31: number,
    n12: number, n22: number, n32: number,
    n13: number, n23: number, n33: number,
  ): this {
    const te = this.elements;
    te[0] = n11; te[3] = n12; te[6] = n13;
    te[1] = n21; te[4] = n22; te[7] = n23;
    te[2] = n31; te[5] = n32; te[8] = n33;
    return this;
  }

  multiply(m: Matrix3): this {
    return this.multiplyMatrices(this, m);
  }

  premultiply(m: Matrix3): this {
    return this.multiplyMatrices(m, this);
  }

  multiplyMatrices(a: Matrix3, b: Matrix3): this {
    const ae = a.elements;
    const be = b.elements;
    const te = this.elements;

    const a11 = ae[0], a12 = ae[3], a13 = ae[6];
    const a21 = ae[1], a22 = ae[4], a23 = ae[7];
    const a31 = ae[2], a32 = ae[5], a33 = ae[8];

    const b11 = be[0], b12 = be[3], b13 = be[6];
    const b21 = be[1], b22 = be[4], b23 = be[7];
    const b31 = be[2], b32 = be[5], b33 = be[8];

    te[0] = a11 * b11 + a12 * b21 + a13 * b31;
    te[3] = a11 * b12 + a12 * b22 + a13 * b32;
    te[6] = a11 * b13 + a12 * b23 + a13 * b33;

    te[1] = a21 * b11 + a22 * b21 + a23 * b31;
    te[4] = a21 * b12 + a22 * b22 + a23 * b32;
    te[7] = a21 * b13 + a22 * b23 + a23 * b33;

    te[2] = a31 * b11 + a32 * b21 + a33 * b31;
    te[5] = a31 * b12 + a32 * b22 + a33 * b32;
    te[8] = a31 * b13 + a32 * b23 + a33 * b33;
    return this;
  }

  determinant(): number {
    const te = this.elements;
    const a = te[0], b = te[1], c = te[2];
    const d = te[3], e = te[4], f = te[5];
    const g = te[6], h = te[7], i = te[8];
    return a * e * i - a * f * h - b * d * i + b * f * g + c * d * h - c * e * g;
  }

  invert(): this {
    const te = this.elements;
    const a = te[0], b = te[1], c = te[2];
    const d = te[3], e = te[4], f = te[5];
    const g = te[6], h = te[7], i = te[8];
    const n11 = e * i - f * h;
    const n12 = f * g - d * i;
    const n13 = d * h - e * g;
    const det = a * n11 + b * n12 + c * n13;
    if (det === 0) return this.set(0, 0, 0, 0, 0, 0, 0, 0, 0);
    const idet = 1 / det;
    te[0] = n11 * idet;
    te[1] = (c * h - b * i) * idet;
    te[2] = (b * f - c * e) * idet;
    te[3] = n12 * idet;
    te[4] = (a * i - c * g) * idet;
    te[5] = (c * d - a * f) * idet;
    te[6] = n13 * idet;
    te[7] = (b * g - a * h) * idet;
    te[8] = (a * e - b * d) * idet;
    return this;
  }

  transpose(): this {
    let tmp;
    const m = this.elements;
    tmp = m[1]; m[1] = m[3]; m[3] = tmp;
    tmp = m[2]; m[2] = m[6]; m[6] = tmp;
    tmp = m[5]; m[5] = m[7]; m[7] = tmp;
    return this;
  }

  setFromMatrix4(m: Matrix4): this {
    const me = m.elements;
    this.set(
      me[0], me[4], me[8],
      me[1], me[5], me[9],
      me[2], me[6], me[10],
    );
    return this;
  }

  /** The upper-left 3x3 of a matrix's inverse — the normal transform. */
  getNormalMatrix(matrix: Matrix4): this {
    return this.setFromMatrix4(matrix).invert().transpose();
  }

  fromArray(array: ArrayLike<number>, offset = 0): this {
    for (let i = 0; i < 9; i++) this.elements[i] = array[offset + i];
    return this;
  }

  toArray(array: number[] = [], offset = 0): number[] {
    for (let i = 0; i < 9; i++) array[offset + i] = this.elements[i];
    return array;
  }

  equals(m: Matrix3): boolean {
    const te = this.elements;
    const me = m.elements;
    for (let i = 0; i < 9; i++) {
      if (te[i] !== me[i]) return false;
    }
    return true;
  }
}
