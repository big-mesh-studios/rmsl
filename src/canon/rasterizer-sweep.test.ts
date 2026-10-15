import { describe, expect, it } from "vitest";
import { Fn, attribute, builtinFragDepth, varying, vec4 } from "../rmsl";
import { compileJS } from "../js";
import { compileWasm } from "../wasm";
import { exactlyOnLine, random, referenceFragments, windingInClipVolume } from "../testing/rasterizer-reference";

/**
 * Draws triangles from a seeded generator on both CPU rasterizers and holds
 * each to an exact computation of what the canon says it draws, and the two
 * rasterizers to the same bits. A vertex has the colour red, green or blue,
 * so a fragment's colour is the weight of each vertex, and the fragment
 * writes its depth as alpha, over a target cleared to an alpha of -1.
 */
type Draw = (positions: readonly number[], width: number, height: number) => Float64Array;

function rasterizer(compile: typeof compileJS): Draw {
  const position = attribute("vec4");
  const colour = attribute("vec3");
  const shade = varying("vec3");
  const routine = compile(
    () =>
      Fn(() => {
        shade.assign(colour);
        return vec4(position);
      })() as any,
    () => Fn(() => vec4(shade, builtinFragDepth()))() as any,
  );
  const colours = Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);
  return (positions, width, height) =>
    Float64Array.from(
      routine.draw(
        { attributes: { [position.name]: Float64Array.from(positions), [colour.name]: colours } },
        { width, height, clearColor: [0, 0, 0, -1] },
      ),
    ).slice(0, width * height * 4);
}

const js = rasterizer(compileJS);
const wasm = rasterizer(compileWasm as unknown as typeof compileJS);

const SIZES: [number, number][] = [
  [16, 16],
  [15, 15],
  [30, 15],
  [24, 12],
  [7, 5],
];

/** A coordinate on a grid of 1/16 from -range to range, so many pixel centres lie exactly on an edge. */
const onGrid = (next: () => number, range: number) => Math.round((next() * 2 - 1) * range * 16) / 16;

/**
 * A vertex in front of the eye, or behind it, on the grid: x and y up to 1.25 times its w, so some lie
 * outside the clip volume, and z from 0 to w in front. One in four sits exactly on a plane of the clip volume.
 */
function gridVertex(next: () => number, behind: boolean): number[] {
  const w = (Math.abs(onGrid(next, 2)) + 0.0625) * (behind ? -1 : 1);
  const v = [onGrid(next, Math.abs(w) * 1.25), onGrid(next, Math.abs(w) * 1.25), 0, w];
  v[2] = behind ? onGrid(next, Math.abs(w)) : Math.abs(onGrid(next, w)) % w;
  if (next() < 0.25) {
    const plane = Math.floor(next() * 6);
    if (plane < 2) v[0] = plane ? w : -w;
    else if (plane < 4) v[1] = plane === 3 ? w : -w;
    else v[2] = plane === 4 ? 0 : w;
  }
  return v;
}

/** `count` triangles from `seed`, `behind` of their vertices behind the eye, given as twelve components each. */
function gridTriangles(seed: number, count: number, behind: number): number[][] {
  const next = random(seed);
  return Array.from({ length: count }, () =>
    [0, 1, 2].flatMap((k) => gridVertex(next, k < behind && (k === 1 || behind > 1))),
  );
}

/** A factor from 1 to 2 of 46 bits, which times a coordinate of 7 bits on the grid of 1/16 stays exact. */
const messyFactor = (next: () => number) => 1 + Math.floor(next() * 2 ** 45) / 2 ** 45;

/**
 * Grid triangles with each vertex scaled by a factor of its own with many digits: each lies where it
 * lay, so its centres on an edge stay on it, but the products of its coordinates round. Some have
 * z = x, whose depth is exactly 0 down the column at x 0 of a target of odd width, and some z = w + x,
 * whose depth is exactly 1 there.
 */
function messyTriangles(seed: number, count: number, behind: number): number[][] {
  const next = random(seed);
  return gridTriangles(seed + 1000, count, behind).map((t) => {
    const depth = next();
    for (let k = 0; k < 3; k++) {
      const o = 4 * k;
      if (depth < 0.25) t[o + 2] = t[o]!;
      else if (depth < 0.5) t[o + 2] = t[o + 3]! + t[o]!;
      const f = messyFactor(next);
      for (let c = 0; c < 4; c++) t[o + c] = t[o + c]! * f;
    }
    return t;
  });
}

/**
 * Triangles in front of the eye with an edge at depth 0 or 1: two vertices with z 0, or z = w, and the
 * third between. Each vertex is scaled by a factor of many digits, so the depth along that edge,
 * exactly 0 or 1, rounds when computed.
 */
function edgeAtDepthTriangles(seed: number, count: number): number[][] {
  const next = random(seed);
  return Array.from({ length: count }, (_, n) =>
    [0, 1, 2].flatMap((k) => {
      const w = Math.abs(onGrid(next, 2)) + 0.0625;
      const x = onGrid(next, w),
        y = onGrid(next, w);
      const z = k === 2 ? Math.abs(onGrid(next, w)) % w : n % 2 ? w : 0;
      const f = messyFactor(next);
      return [x * f, y * f, z * f, w * f];
    }),
  );
}

/**
 * Draws `positions` on both rasterizers and asserts the JS and WASM images are the same bits, and
 * that each lights the pixels the reference lights, with its weights, and its depth, from 0 to 1.
 * Gives the image.
 */
function drawsAsTheReference(positions: readonly number[], width: number, height: number): Float64Array {
  const image = js(positions, width, height);
  const other = wasm(positions, width, height);
  const label = `${JSON.stringify(positions)} on ${width}x${height}`;
  for (let i = 0; i < image.length; i++)
    if (!Object.is(image[i], other[i])) expect.fail(`JS and WASM differ at ${i}: ${image[i]} ${other[i]}, ${label}`);
  const reference = referenceFragments(positions, width, height)!;
  expect(reference, label).not.toBeNull();
  for (let p = 0; p < width * height; p++) {
    const lit = image[p * 4 + 3] !== -1;
    if (lit !== (reference.covered[p] === 1))
      expect.fail(`pixel ${p % width},${Math.floor(p / width)} lit ${lit}, ${label}`);
    if (!lit) continue;
    for (let e = 0; e < 3; e++)
      if (!(Math.abs(image[p * 4 + e]! - reference.weights[p * 3 + e]!) <= 1e-9))
        expect.fail(`weight ${e} at ${p}: ${image[p * 4 + e]} for ${reference.weights[p * 3 + e]}, ${label}`);
    const depth = image[p * 4 + 3]!;
    if (!(depth >= 0 && depth <= 1 && Math.abs(depth - reference.depth[p]!) <= 1e-9))
      expect.fail(`depth at ${p}: ${depth} for ${reference.depth[p]}, ${label}`);
  }
  return image;
}

/** Each vertex of `positions` times `scales[k]`, all four components alike. */
const scaled = (positions: readonly number[], scales: readonly number[]) =>
  positions.map((c, i) => c * scales[Math.floor(i / 4)]!);

/** Whether the triangle at `positions` lies in one plane with the eye, as the reference has no lines for it. */
const inPlaneWithTheEye = (positions: readonly number[]) => referenceFragments(positions, 2, 2) === null;

describe("a sweep of triangles on the CPU rasterizers against an exact reference", { timeout: 60_000 }, () => {
  /**
   * @canon spec-a-rasterizer-shades-a-pixel-centre-on-a-top-or-left-edge
   */
  it("lights the pixels a triangle in front of the eye covers, on its edges by the top-left rule", () => {
    for (const t of gridTriangles(1, 120, 0)) if (!inPlaneWithTheEye(t)) drawsAsTheReference(t, 16, 16);
  });

  /**
   * @canon spec-a-rasterizer-decides-a-pixel-centre-on-a-clipped-edge-as-its-triangle-places-it
   */
  it("lights the pixels a triangle with a vertex behind the eye covers, its own edges deciding", () => {
    for (const behind of [1, 2])
      for (const t of gridTriangles(2 + behind, 120, behind)) if (!inPlaneWithTheEye(t)) drawsAsTheReference(t, 16, 16);
  });

  /**
   * @canon spec-a-rasterizer-decides-a-pixel-centre-on-an-edge-on-a-target-of-any-size
   */
  it("decides a pixel centre on an edge alike on a target of any size", () => {
    for (const [width, height] of SIZES.slice(1))
      for (const behind of [0, 1])
        for (const t of gridTriangles(10 + width + behind, 40, behind))
          if (!inPlaneWithTheEye(t)) drawsAsTheReference(t, width, height);
  });

  /**
   * @canon spec-a-rasterizer-shades-a-pixel-centre-on-a-top-or-left-edge
   */
  it("decides a centre on an edge, or at depth 0 or 1, exactly where products of the coordinates round", () => {
    for (const [width, height] of [SIZES[0]!, SIZES[1]!])
      for (const t of messyTriangles(80 + width, 80, 0))
        if (!inPlaneWithTheEye(t)) drawsAsTheReference(t, width, height);
  });

  /**
   * A centre on an edge at depth 0 or 1 lies at that depth exactly, and the depth computed there rounds.
   *
   * @canon spec-a-rasterizer-clips-outside-the-depth-range
   */
  it("gives a depth from 0 to 1 at a centre on an edge at depth 0 or 1", () => {
    for (const [width, height] of [SIZES[0]!, SIZES[1]!])
      for (const t of edgeAtDepthTriangles(100 + width, 400))
        if (!inPlaneWithTheEye(t)) drawsAsTheReference(t, width, height);
  });

  /**
   * @canon spec-a-rasterizer-decides-a-pixel-centre-on-a-clipped-edge-as-its-triangle-places-it
   */
  it("decides a centre on an edge the clip cuts exactly where products of the coordinates round", () => {
    for (const [width, height] of [SIZES[0]!, SIZES[1]!])
      for (const t of messyTriangles(90 + width, 80, 1))
        if (!inPlaneWithTheEye(t)) drawsAsTheReference(t, width, height);
  });

  /**
   * A factor that is a power of two keeps every component exact, down to the subnormals.
   *
   * @canon spec-a-rasterizer-weighs-a-vertex-by-the-inverse-of-its-w-at-any-scale
   */
  it("draws the same fragments with every position scaled by one factor, however large or small", () => {
    for (const behind of [0, 1])
      for (const t of gridTriangles(20 + behind, 60, behind)) {
        if (inPlaneWithTheEye(t)) continue;
        const unit = drawsAsTheReference(t, 16, 16);
        for (const factor of [2 ** 600, 2 ** -600, 2 ** -1030]) {
          const image = drawsAsTheReference(scaled(t, [factor, factor, factor]), 16, 16);
          for (let i = 0; i < image.length; i++) expect(Math.abs(image[i]! - unit[i]!)).toBeLessThanOrEqual(1e-12);
        }
      }
  });

  /**
   * Each vertex scaled by a power of two of its own lies where it lay, and weighs by its w: a w
   * of 2^-1060 against one of 2^900 weighs as an ordinary number would.
   *
   * @canon spec-a-rasterizer-weighs-a-vertex-by-the-inverse-of-its-w-at-any-scale
   */
  it("weighs each vertex by the inverse of its w, whatever power of two scales it", () => {
    const next = random(30);
    const powers = [0, 0, 1, -1, 300, -300, 900, -1000, -1060];
    for (const behind of [0, 1])
      for (const t of gridTriangles(31 + behind, 60, behind)) {
        if (inPlaneWithTheEye(t)) continue;
        const scales = [0, 1, 2].map(() => 2 ** powers[Math.floor(next() * powers.length)]!);
        // Two vertices of one w, however small, weigh alike.
        if (next() < 0.3) scales[2] = scales[1]!;
        drawsAsTheReference(scaled(t, scales), 16, 16);
      }
  });

  /**
   * A component of 2^-310 beside one of order 1 counts as 0; one of 2^-290 counts as it is. On a
   * target of odd size, the column of pixel centres at x 0 lies on an edge at x 0.
   *
   * @canon spec-a-rasterizer-takes-a-component-far-smaller-than-its-vertex-for-0
   */
  it("takes a component far smaller than its vertex for 0", () => {
    const next = random(40);
    for (const tiny of [2 ** -310, -(2 ** -310), 2 ** -290, -(2 ** -290)])
      for (let n = 0; n < 20; n++) {
        const t = gridTriangles(41 + n, 1, n % 2)[0]!;
        // An edge through x 0 at the first two vertices, one of them a tiny way off it.
        t[0] = tiny * t[3]!;
        t[4] = 0;
        if (next() < 0.5) t[5] = tiny * t[7]!;
        if (!inPlaneWithTheEye(t)) drawsAsTheReference(t, 15, 15);
      }
  });

  /**
   * The first two vertices, 2^301 to either side, take their y, z and w for 0 and become the
   * directions (∓1, 0, 0, 0), whose edge passes through vec4(0): a wedge to the centre, where the
   * triangle as its vertices place it covers every pixel. At 2^299 it covers every pixel.
   *
   * @canon spec-a-rasterizer-takes-a-component-far-smaller-than-its-vertex-for-0
   */
  it("can turn a triangle into one in a plane with the eye by taking a component for 0", () => {
    const lit = (image: Float64Array) => image.filter((_, i) => i % 4 === 3 && image[i] !== -1).length;
    for (const draw of [js, wasm]) {
      expect(lit(draw([-(2 ** 301), -1, 0.5, 1, 2 ** 301, -1, 0.5, 1, 0, 1, 0.5, 1], 16, 16))).toBe(36);
      expect(lit(draw([-(2 ** 299), -1, 0.5, 1, 2 ** 299, -1, 0.5, 1, 0, 1, 0.5, 1], 16, 16))).toBe(256);
    }
  });

  /**
   * The third vertex is minus a power of two times the first, plus a hair, so an edge passes a
   * hair from vec4(0), and a vertex the clip cuts there has a w of that hair: its projection is
   * rounding, and the pixels to test are all of the target's. The lines of such a triangle are
   * nearly parallel, and near a tie a line's value comes from its exact sum, so its weights and
   * depth are held to the reference too.
   *
   * @canon spec-a-rasterizer-clips-a-triangle-to-the-clip-volume
   * @canon spec-a-rasterizer-weighs-a-vertex-by-the-inverse-of-its-w-at-any-scale
   */
  it("draws a triangle whose edge passes a hair from vec4(0)", () => {
    const next = random(55);
    for (let n = 0; n < 480; n++) {
      const p0 = gridVertex(next, false),
        p1 = gridVertex(next, false);
      const hair = 2 ** -(36 + (n % 24));
      const k = [0.5, 1, 2, 4][n % 4]!;
      const p2 = p0.map((c) => -k * c + hair * (Math.round(next() * 64) - 32));
      const t = n % 2 ? [...p0, ...p1, ...p2] : [...p2, ...p0, ...p1];
      if (!inPlaneWithTheEye(t)) drawsAsTheReference(t, 16, 16);
    }
  });

  /**
   * @canon spec-a-triangle-in-one-plane-with-the-eye-covers-no-pixel-centre
   */
  it("lights nothing for a triangle in one plane with the eye that does not reach vec4(0)", () => {
    const next = random(50);
    for (let n = 0; n < 200; n++) {
      const behind = n % 2 === 1;
      const p0 = gridVertex(next, false),
        p1 = gridVertex(next, false);
      const k = behind ? -2 : 1;
      const p2 = p0.map((c, i) => c + k * p1[i]!);
      if (p2.every((c) => c === 0)) continue;
      const positions = [...p0, ...p1, ...p2];
      for (const image of [js(positions, 16, 16), wasm(positions, 16, 16)])
        for (let p = 0; p < 256; p++) expect(image[p * 4 + 3], JSON.stringify(positions)).toBe(-1);
    }
  });

  /**
   * @canon spec-a-rasterizer-draws-a-vertex-at-vec4-0-at-the-centre-with-its-varyings
   */
  it("draws a vertex at vec4(0) at the centre with its varyings, the other two in the clip volume", () => {
    const next = random(60);
    let drawn = 0;
    while (drawn < 80) {
      const v = [gridVertex(next, false), gridVertex(next, false)];
      if (!v.every((p) => Math.abs(p[0]!) <= p[3]! && Math.abs(p[1]!) <= p[3]! && p[2]! <= p[3]!)) continue;
      v.splice(Math.floor(next() * 3), 0, [0, 0, 0, 0]);
      drawsAsTheReference(v.flat(), drawn % 2 ? 15 : 16, 16);
      drawn++;
    }
  });
});

describe("a sweep of edges two triangles share on the CPU rasterizers", { timeout: 60_000 }, () => {
  /**
   * Each quad is two triangles that share an edge and do not overlap, one or both with a vertex
   * behind the eye. A centre exactly on the shared edge that either covers is lit by exactly one.
   *
   * @canon spec-a-pixel-on-a-shared-edge-is-shaded-once
   */
  it("shades a pixel centre on an edge two triangles share once, clipped or not", () => {
    const next = random(70);
    let quads = 0;
    while (quads < 150) {
      const both = quads % 2 === 1;
      const p0 = gridVertex(next, false),
        p1 = gridVertex(next, false);
      const p2 = gridVertex(next, both);
      const k = 0.25 + Math.round(next() * 16) / 16;
      // The fourth vertex lies on the other side of the plane through the eye, p1 and p2, from p0.
      const a = Math.round((next() - 0.5) * 16) / 16,
        b = Math.round((next() - 0.5) * 16) / 16;
      const p3 = p0.map((c, i) => -k * c + a * p1[i]! + b * p2[i]!);
      if (!(p3[3]! < -0.0625) && !both) continue;
      const first = [...p0, ...p1, ...p2],
        second = [...p3, ...p2, ...p1];
      if (inPlaneWithTheEye(first) || inPlaneWithTheEye(second)) continue;
      const side = (p: number[]) =>
        (p1[1]! * p2[3]! - p1[3]! * p2[1]!) * p[0]! +
        (p1[3]! * p2[0]! - p1[0]! * p2[3]!) * p[1]! +
        (p1[0]! * p2[1]! - p1[1]! * p2[0]!) * p[3]!;
      if (!(Math.sign(side(p3)) === -Math.sign(side(p0)))) continue;
      quads++;
      const onEdge = exactlyOnLine(p1, p2, 32, 32);
      for (const draw of [js, wasm]) {
        const one = draw(first, 32, 32),
          other = draw(second, 32, 32);
        for (let p = 0; p < 32 * 32; p++) {
          const litOne = one[p * 4 + 3] !== -1,
            litOther = other[p * 4 + 3] !== -1;
          if (litOne && litOther) expect.fail(`pixel ${p} shaded twice, ${JSON.stringify([first, second])}`);
          if (onEdge[p] && !litOne && !litOther) {
            const reference = [first, second].some((t) => referenceFragments(t, 32, 32)!.covered[p]);
            if (reference)
              expect.fail(`pixel ${p} on the shared edge shaded by neither, ${JSON.stringify([first, second])}`);
          }
        }
      }
    }
  });
});

/** What a draw by {@link assembler} is asked: its vertices' positions and colours, indices, topology and cull state. */
interface AssembledDraw {
  positions: number[];
  colours: number[];
  index?: Uint16Array | Uint32Array;
  topology?: "triangle-list" | "triangle-strip";
  cullMode?: "none" | "front" | "back";
  frontFace?: "ccw" | "cw";
}

/** The rasterizer of {@link rasterizer}, with each vertex's colour its own and the draw's options passed on. */
function assembler(compile: typeof compileJS): (draw: AssembledDraw, width: number, height: number) => Float64Array {
  const position = attribute("vec4");
  const colour = attribute("vec3");
  const shade = varying("vec3");
  const routine = compile(
    () =>
      Fn(() => {
        shade.assign(colour);
        return vec4(position);
      })() as any,
    () => Fn(() => vec4(shade, builtinFragDepth()))() as any,
  );
  return (draw, width, height) =>
    Float64Array.from(
      routine.draw(
        {
          attributes: {
            [position.name]: Float64Array.from(draw.positions),
            [colour.name]: Float64Array.from(draw.colours),
          },
          index: draw.index,
        },
        {
          width,
          height,
          clearColor: [0, 0, 0, -1],
          topology: draw.topology,
          cullMode: draw.cullMode,
          frontFace: draw.frontFace,
        },
      ),
    ).slice(0, width * height * 4);
}

const assembledJs = assembler(compileJS);
const assembledWasm = assembler(compileWasm as unknown as typeof compileJS);

/**
 * The triangles a draw makes, each as its three vertex numbers, written from WebGPU's rules apart
 * from the rasterizers' own assembly: a list takes each three in turn, a strip makes triangle `k`
 * of `k`, `k + 1`, `k + 2`, an odd one wound `k + 1`, `k`, `k + 2`, and the largest index of the
 * array starts a strip again.
 */
function trianglesOf(draw: AssembledDraw, count: number): number[][] {
  const vertices = draw.index ? Array.from(draw.index) : Array.from({ length: count }, (_, i) => i);
  const restart = draw.index instanceof Uint16Array ? 0xffff : 0xffffffff;
  const parts: number[][] = [[]];
  for (const v of vertices)
    if (draw.topology === "triangle-strip" && draw.index && v === restart) parts.push([]);
    else parts[parts.length - 1]!.push(v);
  const out: number[][] = [];
  for (const part of parts) {
    if (draw.topology === "triangle-strip")
      for (let k = 0; k + 2 < part.length; k++)
        out.push(k % 2 ? [part[k + 1]!, part[k]!, part[k + 2]!] : [part[k]!, part[k + 1]!, part[k + 2]!]);
    else for (let k = 0; k + 2 < part.length; k += 3) out.push([part[k]!, part[k + 1]!, part[k + 2]!]);
  }
  return out;
}

/** Whether a draw with this cull state draws a triangle of winding `winding`, as WebGPU decides it. */
function drawsFace(draw: AssembledDraw, winding: number): boolean {
  const mode = draw.cullMode ?? "none";
  if (mode === "none") return true;
  const front = winding !== 0 && (draw.frontFace === "cw" ? winding < 0 : winding > 0);
  return mode === "back" ? front : !front;
}

/** The positions of the triangle of vertices `t` of `draw`, twelve components. */
const positionsOf = (draw: AssembledDraw, t: readonly number[]) =>
  t.flatMap((v) => draw.positions.slice(4 * v, 4 * v + 4));

/** The vertices of an assembled draw: in front of the eye, behind it, on a plane of the clip volume, and at `vec4(0)`. */
function assembledVertex(next: () => number): number[] {
  const r = next();
  if (r < 0.08) return [0, 0, 0, 0];
  return gridVertex(next, r < 0.3);
}

describe("a sweep of indexed, stripped and culled draws on the CPU rasterizers", { timeout: 60_000 }, () => {
  /**
   * A draw by indices, as a list or a strip with restarts, and culled or not, draws the image a
   * draw of the triangles it makes, each as a list of its own vertices, draws with the culled
   * ones left out; and each of those triangles draws what the exact reference draws. A triangle
   * with a vertex behind the eye or at `vec4(0)` faces as the part of it the clip leaves winds,
   * which the reference computes exactly.
   *
   * @canon spec-an-indexed-cpu-draw-takes-the-vertex-each-index-names
   * @canon spec-a-cpu-strip-makes-a-triangle-of-each-vertex-after-the-second
   * @canon spec-an-indexed-cpu-strip-restarts-at-the-largest-index-of-its-array
   * @canon spec-a-cpu-triangle-across-the-near-plane-faces-as-its-part-in-front-winds
   * @canon spec-a-cpu-draw-culling-back-draws-only-front-facing-triangles
   * @canon spec-a-cpu-draw-culling-front-draws-only-back-facing-triangles
   * @canon spec-a-cpu-draw-with-front-face-cw-takes-a-clockwise-triangle-as-front-facing
   * @canon spec-a-cpu-strip-triangle-faces-by-its-own-winding
   * @canon spec-a-cpu-triangle-through-vec4-0-faces-as-the-part-it-draws-winds
   */
  it("draws the triangles an indexed or stripped draw makes, culled by the winding of their clipped part", () => {
    const next = random(400);
    const culls: AssembledDraw["cullMode"][] = ["none", "back", "front"];
    let culledClipped = 0;
    for (let n = 0; n < 240; n++) {
      const count = 6;
      const positions = Array.from({ length: count }, () => assembledVertex(next)).flat();
      const colours = Array.from({ length: count * 3 }, () => Math.round(next() * 8) / 8);
      const strip = n % 2 === 1;
      const wide = n % 4 >= 2;
      const indices: number[] = [];
      for (let i = 0; i < 9; i++)
        indices.push(strip && next() < 0.12 ? (wide ? 0xffffffff : 0xffff) : Math.floor(next() * count));
      const draw: AssembledDraw = {
        positions,
        colours,
        index: n % 8 === 7 ? undefined : wide ? Uint32Array.from(indices) : Uint16Array.from(indices),
        topology: strip ? "triangle-strip" : "triangle-list",
        cullMode: culls[n % 3],
        frontFace: n % 5 === 0 ? "cw" : "ccw",
      };
      const kept = trianglesOf(draw, count).filter((t) => drawsFace(draw, windingInClipVolume(positionsOf(draw, t))));
      const list: AssembledDraw = {
        positions: kept.flatMap((t) => positionsOf(draw, t)),
        colours: kept.flatMap((t) => t.flatMap((v) => colours.slice(3 * v, 3 * v + 3))),
      };
      const width = n % 3 === 1 ? 15 : 16;
      const label = JSON.stringify(draw, (_, v) => (ArrayBuffer.isView(v) ? Array.from(v as Uint32Array) : v));
      const image = assembledJs(draw, width, 16);
      const other = assembledWasm(draw, width, 16);
      const expected = assembledJs(list, width, 16);
      for (let i = 0; i < image.length; i++) {
        if (!Object.is(image[i], other[i]))
          expect.fail(`JS and WASM differ at ${i}: ${image[i]} ${other[i]}, ${label}`);
        if (!Object.is(image[i], expected[i]))
          expect.fail(`the draw and its list differ at ${i}: ${image[i]} ${expected[i]}, ${label}`);
      }
      for (const t of trianglesOf(draw, count)) {
        const p = positionsOf(draw, t);
        if (!inPlaneWithTheEye(p)) drawsAsTheReference(p, width, 16);
        const clipped = t.some((v) => !(positions[4 * v + 3]! > 0));
        if (!clipped || (draw.cullMode ?? "none") === "none") continue;
        const alone = assembledJs(
          { positions: p, colours: t.flatMap((v) => colours.slice(3 * v, 3 * v + 3)) },
          width,
          16,
        );
        if (alone.some((c, i) => i % 4 === 3 && c !== -1)) culledClipped++;
      }
    }
    // The sweep reaches what it claims: triangles with a vertex behind the eye or at vec4(0) that light pixels, in draws that cull.
    expect(culledClipped).toBeGreaterThan(100);
  });
});
