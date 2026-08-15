import { BufferAttribute } from "./BufferAttribute";

function maxOf(values: ArrayLike<number>): number {
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v > max) max = v;
  }
  return max;
}

/**
 * Vertex data for a drawable: named `BufferAttribute`s (position, normal, uv,
 * ...) plus an optional index. Mirrors three.js's `BufferGeometry`.
 */
export class BufferGeometry {
  readonly isBufferGeometry = true;

  attributes: Record<string, BufferAttribute> = {};
  index: BufferAttribute | null = null;
  name = "";
  /**
   * How many times the vertex data is drawn, one vertex pass per instance.
   * Instanced attributes (`stepMode === "instance"`) advance once per
   * instance; the default of 1 is an ordinary non-instanced draw.
   */
  instanceCount = 1;

  setAttribute(name: string, attribute: BufferAttribute): this {
    this.attributes[name] = attribute;
    return this;
  }

  getAttribute(name: string): BufferAttribute | undefined {
    return this.attributes[name];
  }

  hasAttribute(name: string): boolean {
    return this.attributes[name] !== undefined;
  }

  deleteAttribute(name: string): this {
    delete this.attributes[name];
    return this;
  }

  setIndex(index: BufferAttribute | ArrayLike<number> | null): this {
    if (index === null) {
      this.index = null;
    } else if (index instanceof BufferAttribute) {
      this.index = index;
    } else {
      // WebGL (and WebGPU) upload element buffers from a typed array: the
      // byte size of the upload follows the view, and the draw call picks
      // UNSIGNED_SHORT or UNSIGNED_INT from its type. A plain number array
      // carries neither, so it is converted up front.
      const needsUint32 = index.length > 0 && maxOf(index) > 65535;
      this.index = new BufferAttribute(
        needsUint32 ? new Uint32Array(index) : new Uint16Array(index),
        1,
      );
    }
    return this;
  }

  get position(): BufferAttribute | undefined {
    return this.attributes.position;
  }

  set position(attribute: BufferAttribute | undefined) {
    if (attribute === undefined) delete this.attributes.position;
    else this.attributes.position = attribute;
  }

  get normal(): BufferAttribute | undefined {
    return this.attributes.normal;
  }

  set normal(attribute: BufferAttribute | undefined) {
    if (attribute === undefined) delete this.attributes.normal;
    else this.attributes.normal = attribute;
  }

  get uv(): BufferAttribute | undefined {
    return this.attributes.uv;
  }

  set uv(attribute: BufferAttribute | undefined) {
    if (attribute === undefined) delete this.attributes.uv;
    else this.attributes.uv = attribute;
  }

  get vertexCount(): number {
    if (this.index) return this.index.count;
    return this.attributes.position?.count ?? 0;
  }

  get drawCount(): number {
    if (this.index) return this.index.count;
    return this.attributes.position?.count ?? 0;
  }
}
