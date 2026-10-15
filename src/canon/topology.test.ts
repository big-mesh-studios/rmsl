import { describe, expect, it } from "vitest";
import { attribute, builtinPosition, Fn, uniform, varying, vec4, vertexIndex } from "../rmsl";
import { UsageError } from "../errors";
import { compileJS, createJs } from "../js";
import { compileWasm, createWasm } from "../wasm";

const rasterizers: [string, typeof compileJS][] = [
  ["JS", compileJS],
  ["WASM", compileWasm as unknown as typeof compileJS],
];

const adapters: [string, typeof createJs][] = [
  ["JS", createJs],
  ["WASM", createWasm as unknown as typeof createJs],
];

const WIDTH = 64;
const HEIGHT = 16;

type DrawOptions = Parameters<ReturnType<typeof compileJS>["draw"]>[1];
type Index = Uint16Array | Uint32Array;

/** Clip-space positions of `xy` pairs, each at depth 0 and `w` 1. */
const at = (...xy: number[]) => xy.flatMap((c, i) => (i % 2 ? [c, 0, 1] : [c]));

/** A zigzag strip of `n` vertices across the target, starting at the top, so its first triangle runs counter-clockwise. */
function zigzag(n: number): number[] {
  const out: number[] = [];
  for (let k = 0; k < n; k++) out.push(-0.9 + (k * 1.8) / (n - 1), k % 2 === 0 ? 0.8 : -0.8, 0, 1);
  return out;
}

/** The pixel of the centroid of vertices `a`, `b` and `c` of `positions`. */
function centroid(positions: number[], a: number, b: number, c: number): [number, number] {
  let x = 0;
  let y = 0;
  for (const v of [a, b, c]) {
    x += positions[v * 4]! / positions[v * 4 + 3]! / 3;
    y += positions[v * 4 + 1]! / positions[v * 4 + 3]! / 3;
  }
  return [Math.floor(((x + 1) / 2) * WIDTH), Math.floor(((1 - y) / 2) * HEIGHT)];
}

/** What a pixel shows: `null` where nothing drew, else the flat id and the vertex index of the fragment. */
type Shown = { id: number; vertexIndex: number } | null;

/**
 * A rasterizer whose every vertex writes its own id, given as an attribute,
 * and its `vertexIndex()` to flat varyings, which the fragment writes to red
 * and green. Each vertex has the id `10 + k` unless the draw gives others.
 */
function idProgram(compile: typeof compileJS) {
  const position = attribute("vec4");
  const id = attribute("int");
  const k = varying("int");
  const vi = varying("int");
  const routine = compile(
    () =>
      Fn(() => {
        k.assign(id);
        vi.assign(vertexIndex().toInt());
        builtinPosition().assign(position);
      })() as any,
    () => Fn(() => vec4(k.toFloat(), vi.toFloat(), 1, 1))() as any,
  );
  return (
    positions: number[],
    options: Partial<DrawOptions> = {},
    index?: Index,
    ids?: number[],
  ): ((pixel: [number, number]) => Shown) => {
    const n = positions.length / 4;
    // A routine draws into an image it keeps for the next draw, so the result is copied.
    const image = routine
      .draw(
        {
          attributes: {
            [position.name]: Float64Array.from(positions),
            [id.name]: Int32Array.from(ids ?? Array.from({ length: n }, (_, v) => 10 + v)),
          },
          ...(index ? { index } : {}),
        },
        { width: WIDTH, height: HEIGHT, ...options },
      )
      .slice();
    return ([x, y]) => {
      const o = (y * WIDTH + x) * 4;
      return image[o + 3] === 0 ? null : { id: image[o]!, vertexIndex: image[o + 1]! };
    };
  };
}

/** The ids each triangle `k`, `k + 1`, `k + 2` of `positions` shows at its centroid, for `count` triangles. */
const stripIds = (show: (pixel: [number, number]) => Shown, positions: number[], count: number) =>
  Array.from({ length: count }, (_, k) => show(centroid(positions, k, k + 1, k + 2))?.id ?? null);

describe.each(rasterizers)("a CPU rasterizer on %s assembles triangles as WebGPU does", (_, compile) => {
  /**
   * Six zigzag vertices: as a list they make the triangles 0, 1, 2 and 3, 4, 5;
   * as a strip they would also cover the triangles between.
   *
   * @canon spec-a-cpu-draw-is-a-triangle-list-unless-it-names-another-topology
   */
  it("draws each three vertices as a triangle when it names no topology or a list", () => {
    const draw = idProgram(compile);
    const positions = zigzag(6);
    for (const options of [{}, { topology: "triangle-list" as const }]) {
      const show = draw(positions, options);
      expect(stripIds(show, positions, 4)).toEqual([10, null, null, 13]);
    }
  });

  /**
   * @canon spec-a-cpu-strip-makes-a-triangle-of-each-vertex-after-the-second
   */
  it("draws a triangle of each vertex after the second of a strip", () => {
    const positions = zigzag(6);
    const show = idProgram(compile)(positions, { topology: "triangle-strip" });
    expect(stripIds(show, positions, 4).every((id) => id !== null)).toBe(true);
  });

  /**
   * @canon spec-a-cpu-list-draws-nothing-for-vertices-that-complete-no-triangle
   */
  it("draws nothing for the last one or two vertices of a list", () => {
    const positions = zigzag(6);
    const draw = idProgram(compile);
    for (const count of [4, 5])
      expect(stripIds(draw(positions, { count }), positions, 4)).toEqual([10, null, null, null]);
  });

  /**
   * @canon spec-a-cpu-strip-of-fewer-than-three-vertices-draws-nothing
   */
  it("draws nothing for a strip of one or two vertices", () => {
    const positions = zigzag(6);
    const draw = idProgram(compile);
    for (const count of [1, 2])
      expect(stripIds(draw(positions, { topology: "triangle-strip", count }), positions, 4)).toEqual([
        null,
        null,
        null,
        null,
      ]);
  });

  /**
   * Strip vertex `k` has the id `10 + k`, so vertex `k + 1` or `k + 2` would show another.
   *
   * @canon spec-a-flat-varying-of-an-even-strip-triangle-takes-strip-vertex-k
   */
  it("reads an integer varying of an even strip triangle as strip vertex k wrote it", () => {
    const positions = zigzag(6);
    const show = idProgram(compile)(positions, { topology: "triangle-strip" });
    expect([0, 2].map((k) => show(centroid(positions, k, k + 1, k + 2))?.id)).toEqual([10, 12]);
  });

  /**
   * @canon spec-a-flat-varying-of-an-odd-strip-triangle-takes-strip-vertex-k
   */
  it("reads an integer varying of an odd strip triangle as strip vertex k wrote it", () => {
    const positions = zigzag(6);
    const show = idProgram(compile)(positions, { topology: "triangle-strip" });
    expect([1, 3].map((k) => show(centroid(positions, k, k + 1, k + 2))?.id)).toEqual([11, 13]);
  });

  /**
   * Its third triangle doubles back, so it faces the other way, and a cull of
   * back faces drops it alone; triangle 1 lies under it.
   *
   * @canon spec-a-cpu-strip-triangle-faces-by-its-own-winding
   */
  it("culls the triangles of a folded strip by their own winding", () => {
    const folded = at(-0.9, 0.9, -0.9, -0.9, -0.3, 0.9, 0.6, -0.9, -0.6, 0.9);
    const draw = idProgram(compile);
    const ids = (cullMode: "none" | "back" | "front") =>
      stripIds(draw(folded, { topology: "triangle-strip", cullMode }), folded, 3);
    expect(ids("none")).toEqual([10, 11, 12]);
    expect(ids("back")).toEqual([10, 11, 11]);
    expect(ids("front")).toEqual([null, null, 12]);
  });

  /**
   * Its first two vertices coincide, so its first triangle has zero area, and
   * its second runs counter-clockwise.
   *
   * @canon spec-a-cpu-strip-triangle-faces-by-its-own-winding
   */
  it("faces a strip triangle by its own winding after a first triangle of zero area", () => {
    const flat = at(-0.9, -0.9, -0.9, -0.9, -0.5, 0.9, 0.5, -0.9);
    const draw = idProgram(compile);
    const shown = (cullMode: "back" | "front") =>
      draw(flat, { topology: "triangle-strip", cullMode })(centroid(flat, 1, 2, 3))?.id ?? null;
    expect(shown("back")).toBe(11);
    expect(shown("front")).toBe(null);
  });

  /**
   * Two triangles over the same pixels at the same depth: the one drawn second
   * shows, in either order, as a list and through indices.
   *
   * @canon spec-a-cpu-draw-draws-its-triangles-in-the-order-it-makes-them
   */
  it("keeps the fragment of the triangle it draws later at the same depth", () => {
    const tri = at(-0.8, -0.8, 0.8, -0.8, 0, 0.8);
    const positions = [...tri, ...tri];
    const draw = idProgram(compile);
    const centre = centroid(positions, 0, 1, 2);
    expect(draw(positions)(centre)?.id).toBe(13);
    expect(draw(positions, {}, Uint16Array.of(3, 4, 5, 0, 1, 2))(centre)?.id).toBe(10);
  });

  /**
   * A float attribute of 3 vertices, under a draw of 6: the second triangle's
   * vertices all lie past its end.
   *
   * The draw reads the attribute as zero instead.
   *
   * @canon bug-a-cpu-draw-whose-count-passes-an-attribute-reads-it-as-zero
   */
  it.fails("refuses a draw whose count reaches past the end of an attribute", () => {
    const position = attribute("vec4");
    const shade = attribute("float");
    const s = varying("float");
    const routine = compile(
      () =>
        Fn(() => {
          s.assign(shade);
          builtinPosition().assign(position);
        })() as any,
      () => Fn(() => vec4(s, 1, 0, 1))() as any,
    );
    const positions = zigzag(6);
    expect(() =>
      routine.draw(
        { attributes: { [position.name]: Float64Array.from(positions), [shade.name]: Float64Array.of(0.5, 0.5, 0.5) } },
        { width: WIDTH, height: HEIGHT, count: 6 },
      ),
    ).toThrowRmsl(UsageError, "attribute-range");
  });

  /**
   * @canon spec-a-cpu-draw-has-no-fan-topology
   * @canon spec-a-cpu-draw-has-no-point-list-topology
   * @canon spec-a-cpu-draw-has-no-line-topology
   */
  it("admits no fan, point or line topology in its type", () => {
    const options = (topology: DrawOptions["topology"]) => ({ topology });
    // @ts-expect-error A CPU draw has no fan.
    options("triangle-fan");
    // @ts-expect-error A CPU draw has no points.
    options("point-list");
    // @ts-expect-error A CPU draw has no lines.
    options("line-list");
    // @ts-expect-error A CPU draw has no lines.
    options("line-strip");
    expect(options("triangle-strip").topology).toBe("triangle-strip");
  });
});

describe.each(rasterizers)("a CPU rasterizer on %s draws by an index array", (_, compile) => {
  /**
   * The indices share vertices 1 and 2 between two triangles, and skip vertex 4.
   *
   * @canon spec-an-indexed-cpu-draw-takes-the-vertex-each-index-names
   */
  it("takes the vertex each index names", () => {
    const positions = zigzag(8);
    const show = idProgram(compile)(positions, {}, Uint16Array.of(0, 1, 2, 2, 1, 3, 5, 6, 7));
    expect(show(centroid(positions, 0, 1, 2))?.id).toBe(10);
    expect(show(centroid(positions, 1, 2, 3))?.id).toBe(12);
    expect(show(centroid(positions, 5, 6, 7))?.id).toBe(15);
    expect(show(centroid(positions, 3, 4, 5))).toBe(null);
  });

  /**
   * The draw names vertices 5, 6 and 7 from the third index on, so their
   * indices, not their places in the draw, are what `vertexIndex()` reads.
   *
   * @canon spec-an-indexed-cpu-draw-reads-the-named-vertex-as-the-vertex-index
   */
  it("reads the index of the vertex an index names as vertexIndex()", () => {
    const positions = zigzag(8);
    const show = idProgram(compile)(positions, { first: 3 }, Uint16Array.of(0, 1, 2, 5, 6, 7));
    expect(show(centroid(positions, 5, 6, 7))).toEqual({ id: 15, vertexIndex: 5 });
  });

  /**
   * @canon spec-an-index-of-a-draw-with-no-base-vertex-names-the-vertex-at-its-value
   */
  it("names by each index the vertex at that index, counted from the start of the attributes", () => {
    const positions = zigzag(8);
    const show = idProgram(compile)(positions, {}, Uint32Array.of(7, 6, 5));
    expect(show(centroid(positions, 5, 6, 7))).toEqual({ id: 17, vertexIndex: 7 });
  });

  /**
   * @canon spec-an-indexed-cpu-draw-counts-its-indices
   */
  it("counts first and count in indices, and draws to the end of the array when it names no count", () => {
    const positions = zigzag(6);
    const draw = idProgram(compile);
    // The indices name the second triangle first, so counting vertices instead of indices draws the other one.
    const index = Uint16Array.of(3, 4, 5, 0, 1, 2);
    const both = draw(positions, {}, index);
    const fromThird = draw(positions, { first: 3 }, index);
    const firstThree = draw(positions, { first: 0, count: 3 }, index);
    const ids = (show: (p: [number, number]) => Shown) =>
      [centroid(positions, 0, 1, 2), centroid(positions, 3, 4, 5)].map((p) => show(p)?.id ?? null);
    expect(ids(both)).toEqual([10, 13]);
    expect(ids(fromThird)).toEqual([10, null]);
    expect(ids(firstThree)).toEqual([null, 13]);
  });

  /**
   * Six indices: from 6 for 0 draws nothing and is no mistake; from 4 for 3,
   * from 0 for 7, and from 7 with no count each reach past the end.
   *
   * @canon spec-an-indexed-cpu-draw-past-its-index-array-is-refused
   */
  it("refuses a draw whose first index or count passes the end of its index array", () => {
    const positions = zigzag(6);
    const draw = idProgram(compile);
    const index = Uint16Array.of(0, 1, 2, 3, 4, 5);
    expect(draw(positions, { first: 6, count: 0 }, index)(centroid(positions, 0, 1, 2))).toBe(null);
    for (const options of [{ first: 4, count: 3 }, { first: 0, count: 7 }, { first: 7 }])
      expect(() => draw(positions, options, index)).toThrowRmsl(UsageError, "index-range");
  });

  /**
   * @canon spec-an-index-array-is-a-uint16-or-uint32-array
   */
  it("admits only a Uint16Array or a Uint32Array as its indices in its type", () => {
    type Context = Parameters<ReturnType<typeof compileJS>["draw"]>[0];
    const context = (index: Context["index"]) => ({ attributes: {}, index });
    context(Uint16Array.of(0));
    context(Uint32Array.of(0));
    // @ts-expect-error An index array is a Uint16Array or a Uint32Array.
    context(Int32Array.of(0));
    // @ts-expect-error An index array is a Uint16Array or a Uint32Array.
    context(Uint8Array.of(0));
    expect(context(Uint16Array.of(0)).index).toBeInstanceOf(Uint16Array);
  });

  /**
   * @canon spec-an-indexed-cpu-strip-restarts-at-the-largest-index-of-its-array
   */
  it.each([
    ["Uint16Array", (values: number[]) => Uint16Array.from(values), 0xffff],
    ["Uint32Array", (values: number[]) => Uint32Array.from(values), 0xffffffff],
  ] as const)("restarts an indexed strip at the largest index of a %s", (_, make, max) => {
    const positions = zigzag(8);
    const show = idProgram(compile)(positions, { topology: "triangle-strip" }, make([0, 1, 2, 3, max, 4, 5, 6, 7]));
    expect(stripIds(show, positions, 6)).toEqual([10, 11, null, null, 14, 15]);
  });

  /**
   * Parts of one and two indices before the first restart, between two, and
   * after the last; and a restart first, last, and twice in a row.
   *
   * @canon spec-a-part-of-an-indexed-cpu-strip-under-three-vertices-draws-nothing
   */
  it("draws nothing for a part of an indexed strip of fewer than three indices", () => {
    const R = 0xffff;
    const positions = zigzag(8);
    const draw = idProgram(compile);
    const strip = (values: number[]) =>
      stripIds(draw(positions, { topology: "triangle-strip" }, Uint16Array.from(values)), positions, 6);
    expect(strip([0, R, 1, 2, 3])).toEqual([null, 11, null, null, null, null]);
    expect(strip([0, 1, R, 2, 3, 4])).toEqual([null, null, 12, null, null, null]);
    expect(strip([0, 1, 2, R, 3, 4, R, 5, 6, 7])).toEqual([10, null, null, null, null, 15]);
    expect(strip([0, 1, 2, R, 3])).toEqual([10, null, null, null, null, null]);
    expect(strip([R, 0, 1, 2, R])).toEqual([10, null, null, null, null, null]);
    expect(strip([0, 1, 2, R, R, 3, 4, 5])).toEqual([10, null, null, 13, null, null]);
  });

  /**
   * 65,537 vertices, so vertex 0xFFFF exists: a list draws its triangle through it.
   *
   * @canon spec-an-indexed-cpu-list-takes-the-largest-index-as-a-vertex
   */
  it("takes the largest index of a Uint16Array as a vertex in a list", () => {
    const n = 65537;
    const positions = new Array<number>(n * 4).fill(5);
    const ids = new Array<number>(n).fill(0);
    positions.splice(0, 8, -0.9, -0.9, 0, 1, 0.9, -0.9, 0, 1);
    positions.splice(0xffff * 4, 4, 0, 0.9, 0, 1);
    ids[0xffff] = 200;
    const show = idProgram(compile)(positions, {}, Uint16Array.of(0xffff, 0, 1), ids);
    expect(show([32, 10])).toEqual({ id: 200, vertexIndex: 0xffff });
  });

  /**
   * The second draw names the same vertices with other attributes: a vertex
   * kept from the first draw would show its old id.
   *
   * @canon spec-an-indexed-cpu-draw-shades-with-the-inputs-of-that-draw
   */
  it("shades every vertex it names with the inputs of that draw", () => {
    const positions = zigzag(6);
    const draw = idProgram(compile);
    const index = Uint16Array.of(0, 1, 2);
    draw(positions, {}, index);
    const show = draw(positions, {}, index, [40, 41, 42, 43, 44, 45]);
    expect(show(centroid(positions, 0, 1, 2))?.id).toBe(40);
  });

  /**
   * The id attribute holds 6 vertices and the position attribute 7, so index 6
   * reaches a position and no id: the id reads 0, where its last element is 15.
   *
   * @canon spec-a-cpu-indexed-draw-reads-an-attribute-past-its-end-as-zero
   */
  it("reads an attribute of a vertex past its end as zero", () => {
    const positions = [...zigzag(6), 0, 0.9, 0, 1];
    const show = idProgram(compile)(positions, {}, Uint16Array.of(6, 0, 1), [10, 11, 12, 13, 14, 15]);
    expect(show(centroid(positions, 6, 0, 1))).toEqual({ id: 0, vertexIndex: 6 });
  });
});

describe("a strip whose every triangle crosses the near plane", () => {
  /**
   * Its odd vertices lie behind the eye, so the clip cuts every triangle, and
   * makes up to six vertices of each: more than a list of the same vertices.
   *
   * @canon spec-the-wasm-rasterizer-draws-what-the-js-rasterizer-draws
   */
  it("draws the same pixels on WASM as on JS", () => {
    const positions: number[] = [];
    for (let k = 0; k < 8; k++)
      positions.push(-0.9 + (k * 1.8) / 7, k % 2 === 0 ? 0.6 : -0.6, 0, k % 2 === 0 ? 1 : -0.5);
    const images = rasterizers.map(([, compile]) => {
      const position = attribute("vec4");
      const routine = compile(
        () => Fn(() => builtinPosition().assign(position))() as any,
        () => Fn(() => vec4(1, 0, 0, 1))() as any,
      );
      return Array.from(
        routine.draw(
          { attributes: { [position.name]: Float64Array.from(positions) } },
          { width: WIDTH, height: HEIGHT, topology: "triangle-strip" },
        ),
      );
    });
    expect(images[0]!.filter((_, i) => i % 4 === 3 && images[0]![i] === 1).length).toBeGreaterThan(0);
    expect(images[1]).toEqual(images[0]);
  });
});

describe("an indexed CPU draw runs the vertex stage once for each vertex it names", () => {
  /**
   * Six indices that name four of eight vertices, two of them twice: a run for
   * each index would make six runs, and a run for each vertex eight.
   */
  const index = Uint16Array.of(0, 1, 2, 2, 1, 3);
  const positions = zigzag(8);

  /**
   * A uniform the vertex stage reads is read once for each run of the stage.
   *
   * @canon spec-an-indexed-cpu-draw-shades-each-vertex-it-names-once
   */
  it("runs it four times for six indices that name four vertices on JS", () => {
    const position = attribute("vec4");
    const scale = uniform("float");
    const routine = compileJS(
      () => Fn(() => builtinPosition().assign(position.mul(scale)))() as any,
      () => Fn(() => vec4(1))() as any,
    );
    let reads = 0;
    const uniforms = {
      get [scale.name]() {
        reads++;
        return 1;
      },
    };
    routine.draw(
      { attributes: { [position.name]: Float64Array.from(positions) }, uniforms, index },
      { width: 4, height: 4 },
    );
    expect(reads).toBe(4);
  });

  /**
   * The rasterizer module calls the vertex module's `main` once for each run.
   *
   * @canon spec-an-indexed-cpu-draw-shades-each-vertex-it-names-once
   */
  it("runs it four times for six indices that name four vertices on WASM", () => {
    const Instance = WebAssembly.Instance;
    let runs = 0;
    (WebAssembly as any).Instance = function (module: WebAssembly.Module, imports?: any) {
      const main = imports?.vertex?.main;
      if (typeof main === "function") {
        imports = { ...imports, vertex: { main: () => (runs++, main()) } };
      }
      return new Instance(module, imports);
    };
    try {
      const position = attribute("vec4");
      const routine = compileWasm(
        () => Fn(() => builtinPosition().assign(position))() as any,
        () => Fn(() => vec4(1))() as any,
      );
      routine.draw({ attributes: { [position.name]: Float64Array.from(positions) }, index }, { width: 4, height: 4 });
    } finally {
      (WebAssembly as any).Instance = Instance;
    }
    expect(runs).toBe(4);
  });
});

describe.each(rasterizers)("a CPU rasterizer on %s culls the faces a draw asks it to", (_, compile) => {
  const ccw = at(-0.8, -0.8, 0.8, -0.8, 0, 0.8);
  const cw = at(-0.8, -0.8, 0, 0.8, 0.8, -0.8);
  const centre = centroid(ccw, 0, 1, 2);
  const drawn = (positions: number[], options: Partial<DrawOptions>) =>
    idProgram(compile)(positions, options)(centre) !== null;

  /**
   * @canon spec-a-cpu-draw-culling-back-draws-only-front-facing-triangles
   */
  it("draws only front-facing triangles with cullMode back", () => {
    expect([drawn(ccw, { cullMode: "back" }), drawn(cw, { cullMode: "back" })]).toEqual([true, false]);
  });

  /**
   * @canon spec-a-cpu-draw-culling-front-draws-only-back-facing-triangles
   */
  it("draws only back-facing triangles with cullMode front", () => {
    expect([drawn(ccw, { cullMode: "front" }), drawn(cw, { cullMode: "front" })]).toEqual([false, true]);
  });

  /**
   * @canon spec-a-cpu-triangle-counter-clockwise-on-screen-faces-front-by-default
   */
  it("takes a counter-clockwise triangle as front-facing when it names no front face, or ccw", () => {
    for (const options of [{ cullMode: "back" as const }, { cullMode: "back" as const, frontFace: "ccw" as const }])
      expect([drawn(ccw, options), drawn(cw, options)]).toEqual([true, false]);
  });

  /**
   * @canon spec-a-cpu-draw-with-front-face-cw-takes-a-clockwise-triangle-as-front-facing
   */
  it("takes a clockwise triangle as front-facing with frontFace cw", () => {
    const options = { cullMode: "back" as const, frontFace: "cw" as const };
    expect([drawn(ccw, options), drawn(cw, options)]).toEqual([false, true]);
  });

  /**
   * Its third vertex lies behind the eye: projected, the triangle runs
   * clockwise, and its part in front of the eye runs counter-clockwise. Its
   * mirror image runs the other way in both.
   *
   * @canon spec-a-cpu-triangle-across-the-near-plane-faces-as-its-part-in-front-winds
   */
  it("faces a triangle across the near plane as its part in front of the eye winds", () => {
    const behind = [-0.5, -0.5, 0, 1, 0.5, -0.5, 0, 1, 0, 0.5, 0, -0.5];
    const mirrored = [0.5, -0.5, 0, 1, -0.5, -0.5, 0, 1, 0, 0.5, 0, -0.5];
    const shown = (positions: number[], cullMode: "back" | "front") =>
      idProgram(compile)(positions, { cullMode })([32, 10]) !== null;
    expect([shown(behind, "back"), shown(behind, "front")]).toEqual([true, false]);
    expect([shown(mirrored, "back"), shown(mirrored, "front")]).toEqual([false, true]);
  });
});

describe.each(adapters)("the %s adapter draws with indices, a topology and a cull state", (_, create) => {
  /** An adapter of a flat id program on a fake 64 by 16 canvas, and the bytes its last draw put there. */
  function adapter() {
    const position = attribute("vec4");
    const id = attribute("int");
    const k = varying("int");
    const drawer = create(
      () =>
        Fn(() => {
          k.assign(id);
          builtinPosition().assign(position);
        })(),
      () => Fn(() => vec4(k.toFloat().div(255), 0, 0, 1))(),
    );
    let shown: Uint8ClampedArray | undefined;
    const hadImageData = "ImageData" in globalThis;
    if (!hadImageData)
      (globalThis as any).ImageData = class {
        data: Uint8ClampedArray;
        constructor(
          public width: number,
          public height: number,
        ) {
          this.data = new Uint8ClampedArray(width * height * 4);
        }
      };
    const canvas = {
      width: WIDTH,
      height: HEIGHT,
      getContext: () => ({ putImageData: (image: ImageData) => (shown = image.data) }),
    };
    drawer.attach(canvas as unknown as HTMLCanvasElement);
    const restore = () => {
      if (!hadImageData) delete (globalThis as any).ImageData;
    };
    const red = ([x, y]: [number, number]) =>
      shown![(y * WIDTH + x) * 4 + 3] === 0 ? null : shown![(y * WIDTH + x) * 4]!;
    return { drawer, position, id, red, restore };
  }

  /**
   * @canon spec-set-index-gives-a-cpu-adapter-the-indices-it-draws-by
   */
  it("draws by the indices setIndex gave it, until setIndex(null)", () => {
    const { drawer, position, id, red, restore } = adapter();
    try {
      const positions = zigzag(6);
      drawer.setAttribute(position, Float32Array.from(positions));
      drawer.setAttribute(id, Int32Array.from([10, 11, 12, 13, 14, 15]));
      drawer.setIndex(Uint16Array.of(3, 4, 5));
      drawer.draw();
      const triangles = () => [red(centroid(positions, 0, 1, 2)), red(centroid(positions, 3, 4, 5))];
      expect(triangles()).toEqual([null, 13]);
      drawer.setIndex(null);
      drawer.draw();
      expect(triangles()).toEqual([10, 13]);
    } finally {
      restore();
    }
  });

  /**
   * @canon spec-set-index-refuses-an-array-of-another-kind
   */
  it("refuses an index array of another kind where the host sets it", () => {
    const { drawer, restore } = adapter();
    try {
      expect(() => drawer.setIndex(Int32Array.of(0, 1, 2) as any)).toThrowRmsl(
        UsageError,
        "host-array-of-another-kind",
      );
      expect(() => drawer.setIndex(Uint8Array.of(0, 1, 2) as any)).toThrowRmsl(
        UsageError,
        "host-array-of-another-kind",
      );
    } finally {
      restore();
    }
  });

  /**
   * @canon spec-a-cpu-adapter-draw-takes-a-topology-and-a-cull-state
   */
  it("takes a topology and a cull state among the options of each draw", () => {
    const { drawer, position, id, red, restore } = adapter();
    try {
      const positions = zigzag(4);
      drawer.setAttribute(position, Float32Array.from(positions));
      drawer.setAttribute(id, Int32Array.from([10, 11, 12, 13]));
      drawer.draw({ topology: "triangle-strip" });
      expect(red(centroid(positions, 1, 2, 3))).toBe(11);
      drawer.draw({ topology: "triangle-strip", cullMode: "back", frontFace: "cw" });
      expect(red(centroid(positions, 1, 2, 3))).toBe(null);
      drawer.draw();
      expect(red(centroid(positions, 1, 2, 3))).toBe(null);
    } finally {
      restore();
    }
  });
});
