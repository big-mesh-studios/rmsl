import { Mesh } from "./Mesh";
import { BufferGeometry } from "../geometries/BufferGeometry";
import { BufferAttribute } from "../geometries/BufferAttribute";
import { Matrix4 } from "../math/Matrix4";
import { Color } from "../math/Color";
import { Material } from "../materials/Material";

const _identity = new Matrix4();

/**
 * A mesh drawn `count` times, once per instance, each with its own transform
 * and (optionally) color — like three.js's `InstancedMesh`. The per-instance
 * matrix and color live on the object (not the geometry): the vertex shader
 * multiplies the local position by `instanceMatrix` before the mesh's own
 * world matrix, and `instanceColor` tints the material's color in the
 * fragment stage. The renderers resolve these attributes from the object and
 * draw `count` instances in a single instanced draw.
 */
export class InstancedMesh extends Mesh {
  readonly isInstancedMesh = true;

  count: number;
  instanceMatrix: BufferAttribute;
  instanceColor: BufferAttribute | null = null;

  constructor(geometry: BufferGeometry = new BufferGeometry(), material: Material = new Material(), count = 0) {
    super(geometry, material);
    this.count = count;
    this.instanceMatrix = new BufferAttribute(new Float32Array(count * 16), 16, false, "instance");
    for (let i = 0; i < count; i++) this.setMatrixAt(i, _identity);
  }

  /** Copies the local transform of the given instance into `matrix`. */
  getMatrixAt(index: number, matrix: Matrix4): Matrix4 {
    return matrix.fromArray(this.instanceMatrix.array, index * 16);
  }

  /**
   * Sets the local transform of the given instance. Call
   * `this.instanceMatrix.needsUpdate = true` after updating, like three.js.
   */
  setMatrixAt(index: number, matrix: Matrix4): this {
    (this.instanceMatrix.array as Float32Array).set(matrix.elements, index * 16);
    return this;
  }

  /** Copies the color of the given instance into `color`; white when none was set. */
  getColorAt(index: number, color: Color): Color {
    if (this.instanceColor === null) return color.setRGB(1, 1, 1);
    const array = this.instanceColor.array as Float32Array;
    return color.setRGB(array[index * 3], array[index * 3 + 1], array[index * 3 + 2]);
  }

  /**
   * Sets the color of the given instance, creating the `instanceColor`
   * attribute on first use. Call `this.instanceColor.needsUpdate = true` after
   * updating, like three.js.
   */
  setColorAt(index: number, color: Color): this {
    if (this.instanceColor === null) {
      this.instanceColor = new BufferAttribute(new Float32Array(this.count * 3).fill(1), 3, false, "instance");
    }
    const array = this.instanceColor.array as Float32Array;
    array[index * 3] = color.r;
    array[index * 3 + 1] = color.g;
    array[index * 3 + 2] = color.b;
    return this;
  }

  copy(source: InstancedMesh): this {
    this.name = source.name;
    this.visible = source.visible;
    this.position.copy(source.position);
    this.quaternion.copy(source.quaternion);
    this.scale.copy(source.scale);
    this.geometry = source.geometry;
    this.material = source.material;
    this.count = source.count;
    this.instanceMatrix = new BufferAttribute(new Float32Array(source.instanceMatrix.array), 16, false, "instance");
    this.instanceColor = source.instanceColor
      ? new BufferAttribute(new Float32Array(source.instanceColor.array), 3, false, "instance")
      : null;
    return this;
  }
}
