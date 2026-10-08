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
  /** How many times the attribute was marked changed, as three.js counts it: a renderer uploads it again when this passes the version it uploaded. */
  version = 0;

  /** Marks the attribute changed, raising `version`, as in three.js. Reading it gives `undefined`, as in three.js. */
  set needsUpdate(value: boolean) {
    if (value) this.version++;
  }
  /**
   * The vertex format these bytes are in, for the cases the array's own type
   * cannot say. A `Uint16Array` of half floats is the one that needs it: it is
   * indistinguishable from a `Uint16Array` of normalized integers, and only the
   * author knows which. Left undefined, `vertexFormatOf` reads the format off
   * the array type, the item size and `normalized`.
   */
  format?: VertexFormat;
  /**
   * The slices of `array` to upload when `needsUpdate` is set, in array
   * elements, as three.js keeps them: each from `start`, `count` elements
   * long. Empty (the default) uploads the whole array. A renderer merges the
   * ranges that touch, uploads them, and clears the list. The renderer grows
   * the GPU buffer as needed, so an attribute whose array only grew can mark
   * just its tail; the caller promises the rest still holds what the GPU has.
   */
  readonly updateRanges: { start: number; count: number }[] = [];

  /** Marks `count` elements from `start` as changed, as three.js's `addUpdateRange` does. */
  addUpdateRange(start: number, count: number): void {
    this.updateRanges.push({ start, count });
  }

  /** Forgets the marked ranges, as three.js's `clearUpdateRanges` does. */
  clearUpdateRanges(): void {
    this.updateRanges.length = 0;
  }

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
