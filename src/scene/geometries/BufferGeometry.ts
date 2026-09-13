import { EventDispatcher } from "../core/EventDispatcher";
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
export class BufferGeometry extends EventDispatcher {
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
      this.index = new BufferAttribute(needsUint32 ? new Uint32Array(index) : new Uint16Array(index), 1);
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

  /**
   * Release the vertex and index buffers every renderer holds for this
   * geometry, like three.js's `BufferGeometry.dispose()`. Renderers listen for
   * the `dispose` event and delete their own buffers, so a geometry drawn by
   * two renderers frees both.
   *
   * The geometry object itself stays usable: drawing with it again uploads its
   * attributes to fresh buffers. Call this when a geometry leaves the scene for
   * good, rather than waiting for `renderer.dispose()`, which frees everything
   * the renderer holds at once. A renderer keys its buffers by geometry object,
   * so a geometry dropped without this is held by the renderer — with its
   * buffers and the arrays its attributes point at — for as long as the
   * renderer lives.
   */
  dispose(): void {
    this.dispatchEvent({ type: "dispose" });
  }
}
