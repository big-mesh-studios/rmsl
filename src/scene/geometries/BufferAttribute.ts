import type { VertexFormat } from "../renderers/common";

/**
 * One interleaved-or-flat attribute of a geometry: a typed array of `itemSize`
 * components per vertex. Mirrors three.js's `BufferAttribute`.
 *
 * `stepMode` says how the GPU advances between elements: `"vertex"` consumes
 * one element per vertex while `"instance"` consumes one per instance (three.js's
 * `InstancedBufferAttribute`). Drawables like wide lines read per-instance data
 * (`instanceStart`, `instanceEnd`, ...) from instanced attributes.
 */
export class BufferAttribute {
  readonly isBufferAttribute = true;

  array: ArrayLike<number>;
  itemSize: number;
  normalized: boolean;
  count: number;
  stepMode: "vertex" | "instance";
  needsUpdate = false;
  /**
   * The vertex format these bytes are in, for the cases the array's own type
   * cannot say. A `Uint16Array` of half floats is the one that needs it: it is
   * indistinguishable from a `Uint16Array` of normalized integers, and only the
   * author knows which. Left undefined, `vertexFormatOf` reads the format off
   * the array type, the item size and `normalized`.
   */
  format?: VertexFormat;
  /**
   * The slice of `array` to re-upload when `needsUpdate` is set, in array
   * elements: `offset` is where the new data begins and `count` how many
   * elements of it. `count` of `-1` (the default) uploads the whole array.
   * The renderer grows the GPU buffer as needed, so an attribute whose array
   * only grew can re-send just its tail; the caller promises the leading
   * elements still hold what the GPU already has.
   */
  updateRange = { offset: 0, count: -1 };

  constructor(
    array: ArrayLike<number>,
    itemSize: number,
    normalized = false,
    stepMode: "vertex" | "instance" = "vertex",
  ) {
    this.array = array;
    this.itemSize = itemSize;
    this.normalized = normalized;
    this.stepMode = stepMode;
    this.count = array !== undefined ? array.length / itemSize : 0;
  }

  setStepMode(stepMode: "vertex" | "instance"): this {
    this.stepMode = stepMode;
    return this;
  }

  setArray(array: ArrayLike<number>): this {
    this.array = array;
    this.count = array.length / this.itemSize;
    this.needsUpdate = true;
    return this;
  }

  getX(index: number): number {
    return this.array[index * this.itemSize];
  }
  getY(index: number): number {
    return this.array[index * this.itemSize + 1];
  }
  getZ(index: number): number {
    return this.array[index * this.itemSize + 2];
  }
  getW(index: number): number {
    return this.array[index * this.itemSize + 3];
  }

  clone(): BufferAttribute {
    const array = (this.array as number[]).slice ? (this.array as number[]).slice() : Array.from(this.array);
    const clone = new BufferAttribute(array, this.itemSize, this.normalized, this.stepMode);
    clone.format = this.format;
    return clone;
  }
}
