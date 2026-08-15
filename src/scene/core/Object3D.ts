import { Euler } from "../math/Euler";
import { Matrix4 } from "../math/Matrix4";
import { Quaternion } from "../math/Quaternion";
import { Vector3 } from "../math/Vector3";
import { EventDispatcher } from "./EventDispatcher";

/**
 * The base scene-graph node, mirroring three.js's `Object3D`: a local
 * transform (position / quaternion / scale, with `rotation` as a syncing
 * Euler), a parent/children hierarchy, and a world matrix computed by
 * `updateMatrixWorld`.
 */
export class Object3D extends EventDispatcher {
  readonly isObject3D = true;
  readonly isCamera: boolean = false;
  readonly isLight: boolean = false;
  readonly isMesh: boolean = false;

  name = "";
  parent: Object3D | null = null;
  children: Object3D[] = [];
  visible = true;

  readonly position = new Vector3();
  readonly quaternion = new Quaternion();
  readonly rotation = new Euler();
  readonly scale = new Vector3(1, 1, 1);
  readonly up = new Vector3(0, 1, 0);

  readonly matrix = new Matrix4();
  readonly matrixWorld = new Matrix4();
  matrixAutoUpdate = true;
  matrixWorldNeedsUpdate = true;

  constructor() {
    super();
    this.rotation._onChange(() => {
      this.quaternion.setFromEuler(this.rotation);
    });
  }

  onBeforeRender?: (renderer: unknown, scene: Object3D, camera: Object3D) => void;

  add(...objects: Object3D[]): this {
    for (const object of objects) {
      if (object === this) {
        throw new Error("[RMSL/scene] an object cannot be added to itself");
      }
      if (object.parent !== null) {
        object.parent.remove(object);
      }
      object.parent = this;
      this.children.push(object);
      object.dispatchEvent({ type: "added", object });
    }
    return this;
  }

  remove(...objects: Object3D[]): this {
    for (const object of objects) {
      const index = this.children.indexOf(object);
      if (index !== -1) {
        object.parent = null;
        this.children.splice(index, 1);
        object.dispatchEvent({ type: "removed", object });
      }
    }
    return this;
  }

  clear(): this {
    for (const child of this.children) {
      child.parent = null;
      child.dispatchEvent({ type: "removed", child });
    }
    this.children.length = 0;
    return this;
  }

  getObjectByName(name: string): Object3D | undefined {
    if (this.name === name) return this;
    for (const child of this.children) {
      const match = child.getObjectByName(name);
      if (match !== undefined) return match;
    }
    return undefined;
  }

  traverse(callback: (object: Object3D) => void): void {
    callback(this);
    for (const child of this.children) child.traverse(callback);
  }

  traverseVisible(callback: (object: Object3D) => void): void {
    if (this.visible) {
      callback(this);
      for (const child of this.children) child.traverseVisible(callback);
    }
  }

  updateMatrix(): void {
    this.matrix.compose(this.position, this.quaternion, this.scale);
    this.matrixWorldNeedsUpdate = true;
  }

  updateMatrixWorld(force = false): void {
    if (this.matrixAutoUpdate) this.updateMatrix();
    if (this.matrixWorldNeedsUpdate || force) {
      if (this.parent === null) {
        this.matrixWorld.copy(this.matrix);
      } else {
        this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix);
      }
      this.matrixWorldNeedsUpdate = false;
      force = true;
    }
    for (const child of this.children) child.updateMatrixWorld(force);
  }

  updateWorldMatrix(updateParents: boolean, updateChildren: boolean): void {
    const parent = this.parent;
    if (updateParents && parent !== null) parent.updateWorldMatrix(true, false);
    if (this.matrixAutoUpdate) this.updateMatrix();
    if (this.parent === null) {
      this.matrixWorld.copy(this.matrix);
    } else {
      this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix);
    }
    this.matrixWorldNeedsUpdate = false;
    if (updateChildren) {
      for (const child of this.children) child.updateWorldMatrix(false, true);
    }
  }

  applyMatrix4(matrix: Matrix4): this {
    if (this.matrixAutoUpdate) this.updateMatrix();
    this.matrix.premultiply(matrix);
    this.matrix.decompose(this.position, this.quaternion, this.scale);
    return this;
  }

  getWorldPosition(target = new Vector3()): Vector3 {
    this.updateWorldMatrix(true, false);
    return target.setFromMatrixPosition(this.matrixWorld);
  }

  getWorldQuaternion(target = new Quaternion()): Quaternion {
    this.updateWorldMatrix(true, false);
    this.matrixWorld.decompose(_position, target, _scale);
    return target;
  }

  getWorldScale(target = new Vector3()): Vector3 {
    this.updateWorldMatrix(true, false);
    this.matrixWorld.decompose(_position, _quaternion, target);
    return target;
  }

  getWorldDirection(target = new Vector3()): Vector3 {
    this.updateWorldMatrix(true, false);
    const e = this.matrixWorld.elements;
    return target.set(-e[8], -e[9], -e[10]).normalize();
  }

  /** Rotates this object's local axes so its forward points at the target. */
  lookAt(target: Vector3 | number, y?: number, z?: number): this {
    if (target instanceof Vector3) {
      _target.copy(target);
    } else {
      _target.set(target, y as number, z as number);
    }

    this.updateWorldMatrix(true, false);
    _position.setFromMatrixPosition(this.matrixWorld);

    if (this.isCamera || this.isLight) {
      _m1.lookAt(_position, _target, this.up);
    } else {
      _m1.lookAt(_target, _position, this.up);
    }

    this.quaternion.setFromRotationMatrix(_m1);

    const parent = this.parent;
    if (parent !== null) {
      _m2.extractRotation(parent.matrixWorld);
      _m1.premultiply(_m2.invert());
      this.quaternion.setFromRotationMatrix(_m1);
    }
    return this;
  }
}

const _position = new Vector3();
const _target = new Vector3();
const _quaternion = new Quaternion();
const _scale = new Vector3();
const _m1 = new Matrix4();
const _m2 = new Matrix4();
