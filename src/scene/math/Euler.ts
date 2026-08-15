import { Matrix4 } from "./Matrix4";
import { Quaternion } from "./Quaternion";
import { Vector3 } from "./Vector3";

export type EulerOrder = "XYZ" | "YXZ" | "ZXY" | "ZYX" | "YZX" | "XZY";

export class Euler {
  private _x: number;
  private _y: number;
  private _z: number;
  private _onChangeCallback: (() => void) | null = null;
  order: EulerOrder;

  constructor(x = 0, y = 0, z = 0, order: EulerOrder = "XYZ") {
    this._x = x;
    this._y = y;
    this._z = z;
    this.order = order;
  }

  get x(): number {
    return this._x;
  }

  set x(value: number) {
    this._x = value;
    this._onChangeCallback?.();
  }

  get y(): number {
    return this._y;
  }

  set y(value: number) {
    this._y = value;
    this._onChangeCallback?.();
  }

  get z(): number {
    return this._z;
  }

  set z(value: number) {
    this._z = value;
    this._onChangeCallback?.();
  }

  /** Registers a callback fired on every mutation (three.js's `_onChange`). */
  _onChange(callback: () => void): void {
    this._onChangeCallback = callback;
  }

  setXYZ(x: number, y: number, z: number): this {
    this._x = x;
    this._y = y;
    this._z = z;
    this._onChangeCallback?.();
    return this;
  }

  set(x: number, y: number, z: number, order: EulerOrder = this.order): this {
    this._x = x;
    this._y = y;
    this._z = z;
    this.order = order;
    this._onChangeCallback?.();
    return this;
  }

  copy(euler: Euler): this {
    this._x = euler._x;
    this._y = euler._y;
    this._z = euler._z;
    this.order = euler.order;
    this._onChangeCallback?.();
    return this;
  }

  clone(): Euler {
    return new Euler(this.x, this.y, this.z, this.order);
  }

  setFromRotationMatrix(m: Matrix4, order: EulerOrder = this.order): this {
    const te = m.elements;
    const m11 = te[0], m12 = te[4], m13 = te[8];
    const m21 = te[1], m22 = te[5], m23 = te[9];
    const m31 = te[2], m32 = te[6], m33 = te[10];

    switch (order) {
      case "XYZ":
        this.y = Math.asin(Math.min(1, Math.max(-1, m13)));
        if (Math.abs(m13) < 0.9999999) {
          this.x = Math.atan2(-m23, m33);
          this.z = Math.atan2(-m12, m11);
        } else {
          this.x = Math.atan2(m32, m22);
          this.z = 0;
        }
        break;
      case "YXZ":
        this.x = Math.asin(-Math.min(1, Math.max(-1, m23)));
        if (Math.abs(m23) < 0.9999999) {
          this.y = Math.atan2(m13, m33);
          this.z = Math.atan2(m21, m22);
        } else {
          this.y = Math.atan2(-m31, m11);
          this.z = 0;
        }
        break;
      case "ZXY":
        this.x = Math.asin(Math.min(1, Math.max(-1, m32)));
        if (Math.abs(m32) < 0.9999999) {
          this.y = Math.atan2(-m31, m33);
          this.z = Math.atan2(-m12, m22);
        } else {
          this.y = 0;
          this.z = Math.atan2(m21, m11);
        }
        break;
      case "ZYX":
        this.y = Math.asin(-Math.min(1, Math.max(-1, m31)));
        if (Math.abs(m31) < 0.9999999) {
          this.x = Math.atan2(m32, m33);
          this.z = Math.atan2(m21, m11);
        } else {
          this.x = 0;
          this.z = Math.atan2(-m12, m22);
        }
        break;
      case "YZX":
        this.z = Math.asin(Math.min(1, Math.max(-1, m21)));
        if (Math.abs(m21) < 0.9999999) {
          this.x = Math.atan2(-m23, m22);
          this.y = Math.atan2(-m31, m11);
        } else {
          this.x = 0;
          this.y = Math.atan2(m13, m33);
        }
        break;
      case "XZY":
        this.z = Math.asin(-Math.min(1, Math.max(-1, m12)));
        if (Math.abs(m12) < 0.9999999) {
          this.x = Math.atan2(m32, m22);
          this.y = Math.atan2(m13, m11);
        } else {
          this.x = Math.atan2(-m23, m33);
          this.y = 0;
        }
        break;
      default:
        throw new Error(`unsupported euler order: ${order}`);
    }

    this.order = order;
    this._onChangeCallback?.();
    return this;
  }

  setFromQuaternion(q: Quaternion, order: EulerOrder = this.order): this {
    _matrix.makeRotationFromQuaternion(q);
    return this.setFromRotationMatrix(_matrix, order);
  }

  reorder(newOrder: EulerOrder): this {
    _quat.setFromEuler(this);
    return this.setFromQuaternion(_quat, newOrder);
  }

  equals(euler: Euler): boolean {
    return euler._x === this._x && euler._y === this._y && euler._z === this._z && euler.order === this.order;
  }

  fromArray(array: ArrayLike<number | string>): this {
    this.x = array[0] as number;
    this.y = array[1] as number;
    this.z = array[2] as number;
    if (array[3] !== undefined) this.order = array[3] as unknown as EulerOrder;
    return this;
  }

  toArray(array: (number | string)[] = [], offset = 0): (number | string)[] {
    array[offset] = this.x;
    array[offset + 1] = this.y;
    array[offset + 2] = this.z;
    array[offset + 3] = this.order;
    return array;
  }
}

const _matrix = new Matrix4();
const _quat = new Quaternion();
