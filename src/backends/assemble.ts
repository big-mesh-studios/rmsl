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

/** The values each option of a draw takes, which a draw refuses any other of. */
const OPTION_VALUES = {
  topology: ["triangle-list", "triangle-strip"],
  cullMode: ["none", "front", "back"],
  frontFace: ["ccw", "cw"],
} as const;

/** The largest `first` or `count` a draw takes, as WebGPU's `GPUSize32` holds it. */
const LARGEST_RANGE = 0xffffffff;

/**
 * The vertices below which an index draws through the dense slot table. A
 * vertex the attributes hold always does; one past them only up to this, as
 * the table grows to the largest such vertex a draw names.
 */
const DENSE_VERTICES = 1 << 20;

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

/**
 * The indices an adapter keeps between draws: `set` takes an index array,
 * refused where the host gives one of another kind, or `null` for none.
 */
export function indexHolder(origin: Origin): {
  set(index: CpuIndexArray | null): void;
  readonly index: CpuIndexArray | undefined;
} {
  let index: CpuIndexArray | undefined;
  return {
    set(given) {
      if (given === null) index = undefined;
      else {
        assertIndexArray(origin, given);
        index = given;
      }
    },
    get index() {
      return index;
    },
  };
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

/** What a CPU rasterizer draw is asked to draw: its range and its state, as a draw's options give them. */
export interface DrawRequest extends DrawAssemblyOptions {
  first?: number;
  count?: number;
}

/**
 * The attributes the vertex stage of a draw reads: each slot, and the
 * components of one vertex of it.
 */
export interface DrawAttributes {
  readonly slots: readonly string[];
  readonly widths: readonly number[];
}

/**
 * The vertices a draw shades and the triangles it makes of them, in tables a
 * routine keeps across draws. Each vertex the draw names once, whatever the
 * number of indices that name it, has a slot; each triangle is three slots,
 * wound as WebGPU winds it, its first slot the one that gives its flat varyings.
 */
export interface DrawAssembly {
  /** The vertex, counted from the start of the attributes, that each slot shades. */
  readonly vertexOfSlot: Uint32Array;
  /** The slots that the draw shades. */
  readonly slotCount: number;
  /** Three slots for each triangle. */
  readonly triangles: Int32Array;
  /** The triangles the draw makes. */
  readonly triangleCount: number;
  /** Whether the triangles are a list of the slots in order, which a draw need not read from `triangles`. */
  readonly inOrder: boolean;
  /** The length of the table that maps a vertex to its slot, which a draw without indices leaves at 0. */
  readonly slotTableLength: number;
  /**
   * Refuses a draw's options that WebGPU would refuse, and makes the vertices
   * and triangles it draws. The attributes are those the host passed, which
   * hold every slot the routine reads.
   */
  draw(
    attributes: Readonly<Record<string, ArrayLike<number>>>,
    index: CpuIndexArray | undefined,
    request: DrawRequest,
  ): void;
}

/**
 * The {@link DrawAssembly} of one routine, which reads the attributes `read`
 * names. Each routine makes its own, so that the engine optimizes its code
 * for that routine's inputs alone: code that one routine's attributes, indices
 * and options all reach would see the shapes of every routine's, and be
 * thrown away whenever an earlier routine's objects are collected.
 */
export function createDrawAssembly(origin: Origin, read: DrawAttributes): DrawAssembly {
  let vertexOfSlot = new Uint32Array(0);
  let slotCount = 0;
  let triangles = new Int32Array(0);
  let triangleCount = 0;
  let inOrder = false;
  /** The slot of each vertex below the dense limit, or -1, reset after each draw by indices. */
  let slotOfVertex = new Int32Array(0);
  /** The slot of each vertex at or past the dense limit, made for the draw that names it. */
  const sparse = new Map<number, number>();

  /** Refuses a value of the draw option `option` that is not one the option takes. */
  function checkOption(option: keyof typeof OPTION_VALUES, value: string | undefined): void {
    if (value === undefined) return;
    const values: readonly string[] = OPTION_VALUES[option];
    if (values.includes(value)) return;
    throw new UsageError(
      "unknown-draw-option",
      `${origin.caller}: draw() takes no ${option} "${String(value)}"; it takes ${values.map((v) => `"${v}"`).join(", ")}`,
      origin,
    );
  }

  /** `value` as the `first` or `count` of a draw: its fraction dropped, refused when it is not a whole number WebGPU takes. */
  function drawRange(name: "first" | "count", value: number): number {
    const whole = Math.trunc(value);
    if (!(whole >= 0 && whole <= LARGEST_RANGE)) {
      throw new UsageError(
        "draw-range",
        `${origin.caller}: draw() takes a ${name} from 0 to ${LARGEST_RANGE}, not ${value}`,
        origin,
      );
    }
    return whole;
  }

  /**
   * The vertices the first attribute the program reads holds, in the order the
   * host passed them. An attribute the program does not read counts for nothing,
   * as a WebGPU draw reads no buffer its shader does not declare.
   */
  function inferredCount(attributes: Readonly<Record<string, ArrayLike<number>>>): number {
    for (const slot in attributes) {
      const a = read.slots.indexOf(slot);
      if (a >= 0) return Math.floor(attributes[slot]!.length / read.widths[a]!);
    }
    return 0;
  }

  function newSlot(vertex: number): number {
    vertexOfSlot[slotCount] = vertex;
    return slotCount++;
  }

  function slotOf(vertex: number, dense: number): number {
    if (vertex < dense) {
      const known = slotOfVertex[vertex]!;
      if (known >= 0) return known;
      return (slotOfVertex[vertex] = newSlot(vertex));
    }
    const known = sparse.get(vertex);
    if (known !== undefined) return known;
    const slot = newSlot(vertex);
    sparse.set(vertex, slot);
    return slot;
  }

  function addTriangle(s0: number, s1: number, s2: number): void {
    const t = triangleCount++ * 3;
    triangles[t] = s0;
    triangles[t + 1] = s1;
    triangles[t + 2] = s2;
  }

  /** Clears the dense slots a draw by indices set, so the next draw starts from none. */
  function resetSlots(dense: number): void {
    for (let s = 0; s < slotCount; s++) {
      const vertex = vertexOfSlot[s]!;
      if (vertex < dense) slotOfVertex[vertex] = -1;
    }
    if (sparse.size > 0) sparse.clear();
  }

  /**
   * Makes the slots and triangles of a draw of `count` vertices from `first`,
   * by `index` when it is given, and refuses an index range past its end.
   * `vertices` is the number of vertices the longest attribute holds.
   */
  function assemble(
    index: CpuIndexArray | undefined,
    first: number,
    count: number | undefined,
    topology: CpuTopology | undefined,
    vertices: number,
  ): void {
    const length = index ? index.length : 0;
    if (index && (first > length || (count !== undefined && first + count > length))) {
      const reads =
        count === undefined ? `the indices from index ${first} to the end` : `${count} indices from index ${first}`;
      throw new UsageError(
        "index-range",
        `${origin.caller}: draw() reads ${reads}, and the index array holds ${length}`,
        origin,
      );
    }
    const n = index ? (count ?? length - first) : (count ?? 0);
    const strip = topology === "triangle-strip";
    slotCount = 0;
    triangleCount = 0;
    inOrder = !index && !strip;
    if (n <= 0) return;
    if (vertexOfSlot.length < n) vertexOfSlot = new Uint32Array(n);
    if (triangles.length < n * 3) triangles = new Int32Array(n * 3);
    const restart = index instanceof Uint16Array ? 0xffff : 0xffffffff;
    let dense = 0;
    if (index) {
      // The dense table covers the attributes, and every vertex past them a draw names up to its limit.
      let largest = -1;
      for (let p = 0; p < n; p++) {
        const vertex = index[first + p]!;
        if (!(strip && vertex === restart) && vertex > largest) largest = vertex;
      }
      dense = Math.max(vertices, Math.min(largest + 1, DENSE_VERTICES));
      if (slotOfVertex.length < dense) slotOfVertex = new Int32Array(dense).fill(-1);
    }
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
      const slot = index ? slotOf(vertex, dense) : newSlot(vertex);
      if (run >= 2) {
        if (!strip) addTriangle(a, b, slot);
        // An odd strip triangle winds in reverse. Its first vertex is the strip's vertex k, as WebGPU's implementations take it.
        else if (run % 2 === 0) addTriangle(a, b, slot);
        else addTriangle(a, slot, b);
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
    if (index) resetSlots(dense);
  }

  function draw(
    attributes: Readonly<Record<string, ArrayLike<number>>>,
    index: CpuIndexArray | undefined,
    request: DrawRequest,
  ): void {
    checkOption("topology", request.topology);
    checkOption("cullMode", request.cullMode);
    checkOption("frontFace", request.frontFace);
    const first = request.first === undefined ? 0 : drawRange("first", request.first);
    const count = request.count === undefined ? undefined : drawRange("count", request.count);
    let held = 0;
    for (let a = 0; a < read.slots.length; a++) {
      held = Math.max(held, Math.floor(attributes[read.slots[a]!]!.length / read.widths[a]!));
    }
    if (index) {
      assemble(index, first, count, request.topology, held);
      return;
    }
    const n = count ?? Math.max(0, inferredCount(attributes) - first);
    // Each attribute holds every vertex up to the end of a draw without indices, of no vertex too, as WebGPU requires.
    for (let a = 0; a < read.slots.length; a++) {
      const slot = read.slots[a]!;
      const holds = Math.floor(attributes[slot]!.length / read.widths[a]!);
      if (first + n > holds) {
        throw new UsageError(
          "attribute-range",
          `${origin.caller}: draw() reads ${n} vertices from vertex ${first}, and the attribute "${slot}" holds ${holds}`,
          origin,
        );
      }
    }
    assemble(undefined, first, n, request.topology, held);
  }

  return {
    get vertexOfSlot() {
      return vertexOfSlot;
    },
    get slotCount() {
      return slotCount;
    },
    get triangles() {
      return triangles;
    },
    get triangleCount() {
      return triangleCount;
    },
    get inOrder() {
      return inOrder;
    },
    get slotTableLength() {
      return slotOfVertex.length;
    },
    draw,
  };
}
