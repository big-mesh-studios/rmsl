/**
 * One interleaved-or-flat attribute of a geometry: a typed array of `itemSize`
 * components per vertex. Mirrors three.js's `BufferAttribute`.
 */
export class BufferAttribute {
  readonly isBufferAttribute = true;

  array: ArrayLike<number>;
  itemSize: number;
  normalized: boolean;
  count: number;
  needsUpdate = false;

  constructor(array: ArrayLike<number>, itemSize: number, normalized = false) {
    this.array = array;
    this.itemSize = itemSize;
    this.normalized = normalized;
    this.count = array !== undefined ? array.length / itemSize : 0;
  }

  setArray(array: ArrayLike<number>): this {
    this.array = array;
    this.count = array.length / this.itemSize;
    this.needsUpdate = true;
    return this;
  }

  getX(index: number): number { return this.array[index * this.itemSize]; }
  getY(index: number): number { return this.array[index * this.itemSize + 1]; }
  getZ(index: number): number { return this.array[index * this.itemSize + 2]; }
  getW(index: number): number { return this.array[index * this.itemSize + 3]; }

  clone(): BufferAttribute {
    const array = (this.array as number[]).slice
      ? (this.array as number[]).slice()
      : Array.from(this.array);
    return new BufferAttribute(array, this.itemSize, this.normalized);
  }
}
