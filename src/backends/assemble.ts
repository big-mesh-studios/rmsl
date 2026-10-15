import { UsageError, type Origin } from "../errors";
import { withArticle } from "../article";

/** How a CPU rasterizer draw makes triangles of its vertices: WebGPU's two topologies of triangles. */
export type CpuTopology = "triangle-list" | "triangle-strip";

/** Which faces a CPU rasterizer draw discards, as WebGPU's `GPUCullMode` names them. */
export type CpuCullMode = "none" | "front" | "back";

/** Which winding a CPU rasterizer draw takes as front-facing, as WebGPU's `GPUFrontFace` names it. */
export type CpuFrontFace = "ccw" | "cw";

/** The indices a CPU rasterizer draw takes its vertices by, in one of WebGPU's two index formats. */
export type CpuIndexArray = Uint16Array | Uint32Array;

/** Refuses `data` as the indices of a draw unless it is a `Uint16Array` or a `Uint32Array`, as WebGPU's index formats hold. */
export function assertIndexArray(origin: Origin, data: unknown): asserts data is CpuIndexArray {
  if (data instanceof Uint16Array || data instanceof Uint32Array) return;
  const given = ArrayBuffer.isView(data) ? data.constructor.name : Array.isArray(data) ? "number[]" : typeof data;
  throw new UsageError(
    "host-array-of-another-kind",
    `the index array takes a Uint16Array or a Uint32Array, not ${withArticle(given)}.`,
    origin,
  );
}

/** The state of a CPU rasterizer draw that decides which triangles it makes and which it culls. */
export interface DrawAssemblyOptions {
  /** How the draw makes triangles of its vertices. Defaults to `"triangle-list"`. */
  topology?: CpuTopology;
  /** Which faces the draw discards. Defaults to `"none"`. */
  cullMode?: CpuCullMode;
  /** Which winding is front-facing. Defaults to `"ccw"`. */
  frontFace?: CpuFrontFace;
}

/**
 * The sign a triangle's screen area takes when the draw culls it, or 0 when
 * it culls nothing. Screen y runs down, so a triangle counter-clockwise in
 * normalized device coordinates has a negative screen area.
 */
export function cullSign(options: DrawAssemblyOptions): number {
  const mode = options.cullMode ?? "none";
  if (mode === "none") return 0;
  const back = mode === "back" ? 1 : -1;
  return options.frontFace === "cw" ? -back : back;
}

/**
 * The vertices a draw shades and the triangles it makes of them, in buffers
 * kept across draws. Each vertex the draw names once, whatever the number of
 * indices that name it, has a slot; each triangle is three slots, wound as
 * WebGPU winds it, its first slot the one that gives its flat varyings.
 */
export class TriangleAssembly {
  /** The vertex, counted from the start of the attributes, that each slot shades. */
  vertexOfSlot = new Uint32Array(0);
  /** The slots that the draw shades. */
  slotCount = 0;
  /** Three slots for each triangle. */
  triangles = new Int32Array(0);
  /** The triangles the draw makes. */
  triangleCount = 0;
  /** The slot of each vertex below the dense limit, or -1, reset after each draw. */
  private slotOfVertex = new Int32Array(0);
  /** The slot of each vertex at or past the dense limit, which only an index past the attributes names. */
  private sparse = new Map<number, number>();

  /**
   * Makes the slots and triangles of a draw of `count` vertices from `first`,
   * by `index` when it is given, and refuses an index range past its end.
   * `vertices` is the number of vertices the longest attribute holds.
   */
  assemble(
    origin: Origin,
    index: CpuIndexArray | undefined,
    first: number,
    count: number | undefined,
    topology: CpuTopology | undefined,
    vertices: number,
  ): void {
    const length = index ? index.length : 0;
    const n = index ? (count ?? length - first) : (count ?? 0);
    if (index && (first > length || first + n > length)) {
      throw new UsageError(
        "index-range",
        `${origin.caller}: draw() reads ${n} indices from index ${first}, and the index array holds ${length}`,
        origin,
      );
    }
    this.slotCount = 0;
    this.triangleCount = 0;
    if (n <= 0) return;
    this.reserve(n, vertices);
    const strip = topology === "triangle-strip";
    const restart = index instanceof Uint16Array ? 0xffff : 0xffffffff;
    // The slots of the two vertices before this one: strip vertices k and k + 1 of triangle k, or the list's triangle so far.
    let a = -1;
    let b = -1;
    let run = 0;
    for (let p = 0; p < n; p++) {
      const vertex = index ? index[first + p]! : first + p;
      if (index && strip && vertex === restart) {
        run = 0;
        continue;
      }
      const slot = index ? this.slotOf(vertex, vertices) : this.newSlot(vertex);
      if (run >= 2) {
        if (!strip) this.addTriangle(a, b, slot);
        // An odd strip triangle winds in reverse. Its first vertex is the strip's vertex k, as WebGPU's implementations take it.
        else if (run % 2 === 0) this.addTriangle(a, b, slot);
        else this.addTriangle(a, slot, b);
      }
      if (strip) {
        a = b;
        b = slot;
        run++;
      } else if (run === 2) {
        run = 0;
      } else {
        if (run === 0) a = slot;
        else b = slot;
        run++;
      }
    }
    this.resetSlots(vertices);
  }

  /** Grows the buffers to hold a draw of `n` vertices and `vertices` dense slots. */
  private reserve(n: number, vertices: number): void {
    if (this.vertexOfSlot.length < n) this.vertexOfSlot = new Uint32Array(n);
    if (this.triangles.length < n * 3) this.triangles = new Int32Array(n * 3);
    if (this.slotOfVertex.length < vertices) {
      this.slotOfVertex = new Int32Array(vertices).fill(-1);
    }
  }

  private newSlot(vertex: number): number {
    this.vertexOfSlot[this.slotCount] = vertex;
    return this.slotCount++;
  }

  private slotOf(vertex: number, vertices: number): number {
    if (vertex < vertices) {
      const known = this.slotOfVertex[vertex]!;
      if (known >= 0) return known;
      return (this.slotOfVertex[vertex] = this.newSlot(vertex));
    }
    const known = this.sparse.get(vertex);
    if (known !== undefined) return known;
    const slot = this.newSlot(vertex);
    this.sparse.set(vertex, slot);
    return slot;
  }

  private addTriangle(s0: number, s1: number, s2: number): void {
    const t = this.triangleCount++ * 3;
    this.triangles[t] = s0;
    this.triangles[t + 1] = s1;
    this.triangles[t + 2] = s2;
  }

  /** Clears the dense slots this draw set, so the next draw starts from none. */
  private resetSlots(vertices: number): void {
    for (let s = 0; s < this.slotCount; s++) {
      const vertex = this.vertexOfSlot[s]!;
      if (vertex < vertices) this.slotOfVertex[vertex] = -1;
    }
    if (this.sparse.size > 0) this.sparse.clear();
  }
}
