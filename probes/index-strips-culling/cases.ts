/**
 * The draws both probes make, described once so that WebGPU and WebGL 2 draw the same geometry.
 *
 * Each vertex carries a clip-space position and an id of `10 + k`, so a fragment's flat id names
 * the vertex it came from apart from its vertex index `k`. Each fragment writes its flat id to red,
 * its vertex index to green, 255 to blue when it is front-facing and 128 when it is back-facing,
 * and 255 to alpha. The target is cleared to transparent black, so a pixel no fragment wrote has
 * alpha 0, and a pixel written black still has alpha 255.
 */

export const WIDTH = 64;
export const HEIGHT = 16;

export interface DrawCase {
  name: string;
  /** Clip-space positions, four numbers per vertex. */
  positions: number[];
  /** The id of each vertex, `10 + k` unless the case gives its own. */
  ids?: number[];
  topology: "triangle-list" | "triangle-strip" | "triangle-fan";
  /** Indices, with the array kind that holds them, or none for a draw that is not indexed. */
  index?: { kind: "uint8" | "uint16" | "uint32"; values: number[] };
  /** For WebGPU only: `firstIndex` and `baseVertex` of `drawIndexed`, or `firstVertex` of `draw`. */
  firstIndex?: number;
  baseVertex?: number;
  firstVertex?: number;
  count: number;
  cullMode: "none" | "front" | "back";
  frontFace: "ccw" | "cw";
  /** Pixels to read back, each named, in framebuffer coordinates from the top left. */
  probes: { name: string; x: number; y: number }[];
}

/** A zigzag strip of `n` vertices across the target, starting at the top so its first triangle runs counter-clockwise in NDC. */
export function zigzag(n: number, flip = false): number[] {
  const out: number[] = [];
  for (let k = 0; k < n; k++) {
    const x = -0.9 + (k * 1.8) / (n - 1);
    const top = k % 2 === 0 ? 0.8 : -0.8;
    out.push(x, flip ? -top : top, 0, 1);
  }
  return out;
}

/** The pixel holding the centroid of the triangle `a`, `b`, `c` of `positions`, after the perspective divide. */
export function centroid(positions: number[], a: number, b: number, c: number): { x: number; y: number } {
  let x = 0;
  let y = 0;
  for (const v of [a, b, c]) {
    const w = positions[v * 4 + 3]!;
    x += positions[v * 4]! / w / 3;
    y += positions[v * 4 + 1]! / w / 3;
  }
  return { x: Math.floor(((x + 1) / 2) * WIDTH), y: Math.floor(((1 - y) / 2) * HEIGHT) };
}

/** Probes at the centroid of each triangle `k`, `k + 1`, `k + 2` of a strip of `n` vertices. */
function stripProbes(positions: number[], n: number, offset = 0): DrawCase["probes"] {
  return Array.from({ length: n - 2 }, (_, k) => ({
    name: `strip triangle ${k + offset}`,
    ...centroid(positions, k, k + 1, k + 2),
  }));
}

const CULLS: DrawCase["cullMode"][] = ["none", "back", "front"];
const FACES: DrawCase["frontFace"][] = ["ccw", "cw"];

export function cases(): DrawCase[] {
  const out: DrawCase[] = [];

  // One triangle each way round, under every cull mode and front face.
  const ccw = [-0.8, -0.8, 0, 1, 0.8, -0.8, 0, 1, 0, 0.8, 0, 1];
  const cw = [-0.8, -0.8, 0, 1, 0, 0.8, 0, 1, 0.8, -0.8, 0, 1];
  for (const [name, positions] of [
    ["ndc-ccw triangle", ccw],
    ["ndc-cw triangle", cw],
  ] as const) {
    for (const cullMode of CULLS)
      for (const frontFace of FACES)
        out.push({
          name: `${name} cull=${cullMode} front=${frontFace}`,
          positions: [...positions],
          topology: "triangle-list",
          count: 3,
          cullMode,
          frontFace,
          probes: [{ name: "centre", ...centroid(positions as number[], 0, 1, 2) }],
        });
  }

  // Strips of odd and even length, starting either way round, under every cull mode and front face.
  for (const n of [3, 4, 5, 6])
    for (const flip of [false, true])
      for (const cullMode of CULLS)
        for (const frontFace of FACES) {
          const positions = zigzag(n, flip);
          out.push({
            name: `strip n=${n} first=${flip ? "cw" : "ccw"} cull=${cullMode} front=${frontFace}`,
            positions,
            topology: "triangle-strip",
            count: n,
            cullMode,
            frontFace,
            probes: stripProbes(positions, n),
          });
        }

  // A strip of fewer than three vertices draws nothing.
  {
    const positions = zigzag(6);
    out.push({
      name: "strip count=2",
      positions,
      topology: "triangle-strip",
      count: 2,
      cullMode: "none",
      frontFace: "ccw",
      probes: stripProbes(positions, 6),
    });
  }

  // A list whose count is not a multiple of three.
  {
    const positions = zigzag(6);
    for (const count of [4, 5])
      out.push({
        name: `list count=${count}`,
        positions,
        topology: "triangle-list",
        count,
        cullMode: "none",
        frontFace: "ccw",
        probes: [
          { name: "triangle 0,1,2", ...centroid(positions, 0, 1, 2) },
          { name: "triangle 3,4,5", ...centroid(positions, 3, 4, 5) },
          { name: "triangle 2,3,4", ...centroid(positions, 2, 3, 4) },
        ],
      });
  }

  // Indexed strips with a restart, in both index kinds, after an even and an odd run.
  for (const kind of ["uint16", "uint32"] as const) {
    const restart = kind === "uint16" ? 0xffff : 0xffffffff;
    const positions = zigzag(8);
    out.push({
      name: `indexed strip ${kind} restart after 4`,
      positions,
      topology: "triangle-strip",
      index: { kind, values: [0, 1, 2, 3, restart, 4, 5, 6, 7] },
      count: 9,
      cullMode: "back",
      frontFace: "ccw",
      probes: [
        { name: "0,1,2", ...centroid(positions, 0, 1, 2) },
        { name: "1,2,3", ...centroid(positions, 1, 2, 3) },
        { name: "2,3,4 (spans the restart)", ...centroid(positions, 2, 3, 4) },
        { name: "3,4,5 (spans the restart)", ...centroid(positions, 3, 4, 5) },
        { name: "4,5,6", ...centroid(positions, 4, 5, 6) },
        { name: "5,6,7", ...centroid(positions, 5, 6, 7) },
      ],
    });
    out.push({
      name: `indexed strip ${kind} restart after 3, cull none`,
      positions,
      topology: "triangle-strip",
      index: { kind, values: [0, 1, 2, restart, 3, 4, 5, 6] },
      count: 8,
      cullMode: "none",
      frontFace: "ccw",
      probes: [
        { name: "0,1,2", ...centroid(positions, 0, 1, 2) },
        { name: "1,2,3 (spans the restart)", ...centroid(positions, 1, 2, 3) },
        { name: "3,4,5", ...centroid(positions, 3, 4, 5) },
        { name: "4,5,6", ...centroid(positions, 4, 5, 6) },
      ],
    });
  }

  // An indexed list: a reused vertex, the vertex index an index gives, firstIndex and baseVertex.
  {
    const positions = zigzag(8);
    out.push({
      name: "indexed list reusing vertices",
      positions,
      topology: "triangle-list",
      index: { kind: "uint16", values: [0, 1, 2, 2, 1, 3, 5, 6, 7] },
      count: 9,
      cullMode: "none",
      frontFace: "ccw",
      probes: [
        { name: "0,1,2", ...centroid(positions, 0, 1, 2) },
        { name: "2,1,3", ...centroid(positions, 2, 1, 3) },
        { name: "5,6,7", ...centroid(positions, 5, 6, 7) },
        { name: "3,4,5 (not named)", ...centroid(positions, 3, 4, 5) },
      ],
    });
    out.push({
      name: "indexed list firstIndex=3 baseVertex=2",
      positions,
      topology: "triangle-list",
      index: { kind: "uint16", values: [0, 1, 2, 3, 4, 5] },
      firstIndex: 3,
      baseVertex: 2,
      count: 3,
      cullMode: "none",
      frontFace: "ccw",
      probes: [
        { name: "5,6,7", ...centroid(positions, 5, 6, 7) },
        { name: "3,4,5", ...centroid(positions, 3, 4, 5) },
        { name: "0,1,2", ...centroid(positions, 0, 1, 2) },
      ],
    });
  }

  // The largest index of each kind in a list, and an index past the vertices.
  {
    const positions = zigzag(6);
    for (const kind of ["uint8", "uint16", "uint32"] as const) {
      const max = kind === "uint8" ? 0xff : kind === "uint16" ? 0xffff : 0xffffffff;
      out.push({
        name: `indexed list ${kind} with its largest value`,
        positions,
        topology: "triangle-list",
        index: { kind, values: [0, 1, 2, max, 3, 4, 5, 3, 4] },
        count: 9,
        cullMode: "none",
        frontFace: "ccw",
        probes: [
          { name: "0,1,2", ...centroid(positions, 0, 1, 2) },
          { name: "3,4,5", ...centroid(positions, 3, 4, 5) },
          { name: "2,3,4", ...centroid(positions, 2, 3, 4) },
        ],
      });
    }
    for (const past of [6, 1000]) {
      out.push({
        name: `indexed list with index ${past} first in a triangle`,
        positions,
        topology: "triangle-list",
        index: { kind: "uint16", values: [past, 4, 5] },
        count: 3,
        cullMode: "none",
        frontFace: "ccw",
        probes: [{ name: "3,4,5", ...centroid(positions, 3, 4, 5) }],
      });
      out.push({
        name: `indexed list with index ${past} past 6 vertices`,
        positions,
        topology: "triangle-list",
        index: { kind: "uint16", values: [0, 1, 2, 3, 4, past] },
        count: 6,
        cullMode: "none",
        frontFace: "ccw",
        probes: [
          { name: "0,1,2", ...centroid(positions, 0, 1, 2) },
          { name: "3,4,5", ...centroid(positions, 3, 4, 5) },
        ],
      });
    }
  }

  // A triangle with a vertex behind the eye: its clipped polygon winds one way, its naive projection the other.
  {
    // The same triangle with its first two vertices swapped winds the other way, before and after the clip.
    const geometries = [
      ["", [-0.5, -0.5, 0, 1, 0.5, -0.5, 0, 1, 0, 0.5, 0, -0.5]],
      [" mirrored", [0.5, -0.5, 0, 1, -0.5, -0.5, 0, 1, 0, 0.5, 0, -0.5]],
    ] as const;
    for (const [label, positions] of geometries)
      for (const cullMode of CULLS)
        for (const frontFace of FACES)
          out.push({
            name: `near-plane triangle${label} cull=${cullMode} front=${frontFace}`,
            positions: [...positions],
            topology: "triangle-list",
            count: 3,
            cullMode,
            frontFace,
            probes: [
              { name: "inside the clipped polygon", x: 32, y: 10 },
              { name: "row above", x: 32, y: 8 },
            ],
          });
  }

  // Two triangles over the same pixels, in either order: the later one's fragment is the one kept.
  {
    const tri = [-0.8, -0.8, 0, 1, 0.8, -0.8, 0, 1, 0, 0.8, 0, 1];
    const positions = [...tri, ...tri];
    for (const values of [
      [0, 1, 2, 3, 4, 5],
      [3, 4, 5, 0, 1, 2],
    ])
      out.push({
        name: `overlapping triangles indexed ${values.join(",")}`,
        positions,
        topology: "triangle-list",
        index: { kind: "uint16", values },
        count: 6,
        cullMode: "none",
        frontFace: "ccw",
        probes: [{ name: "centre", ...centroid(positions, 0, 1, 2) }],
      });
  }

  // Strips that fold on screen, and a strip whose first triangle has zero area: does each triangle face by its own winding?
  {
    const xy = (...c: number[]) => c.flatMap((v, i) => (i % 2 ? [v, 0, 1] : [v]));
    // Triangle 0 runs counter-clockwise, triangle 1 too, and triangle 2 doubles back and runs clockwise.
    const folded = xy(-0.9, 0.9, -0.9, -0.9, -0.3, 0.9, 0.6, -0.9, -0.6, 0.9);
    // The first two vertices coincide, so triangle 0 has zero area; triangle 1 runs counter-clockwise.
    const flat = xy(-0.9, -0.9, -0.9, -0.9, -0.5, 0.9, 0.5, -0.9);
    for (const cullMode of CULLS)
      for (const frontFace of FACES) {
        out.push({
          name: `folded strip cull=${cullMode} front=${frontFace}`,
          positions: folded,
          topology: "triangle-strip",
          count: 5,
          cullMode,
          frontFace,
          probes: [
            { name: "triangle 0", ...centroid(folded, 0, 1, 2) },
            { name: "triangle 1", ...centroid(folded, 1, 2, 3) },
            { name: "triangle 2", ...centroid(folded, 2, 3, 4) },
          ],
        });
        out.push({
          name: `strip with a zero-area first triangle cull=${cullMode} front=${frontFace}`,
          positions: flat,
          topology: "triangle-strip",
          count: 4,
          cullMode,
          frontFace,
          probes: [{ name: "triangle 1", ...centroid(flat, 1, 2, 3) }],
        });
      }
  }

  // The boundaries of a restart: first, twice in a row, last, and around a part of fewer than three vertices.
  for (const kind of ["uint16", "uint32"] as const) {
    const R = kind === "uint16" ? 0xffff : 0xffffffff;
    const positions = zigzag(8);
    const all = (values: number[]) => ({ values, probes: [] as DrawCase["probes"] });
    for (const [label, values] of [
      ["restart first", [R, 0, 1, 2, 3]],
      ["two restarts in a row", [0, 1, 2, R, R, 3, 4, 5]],
      ["restart last", [0, 1, 2, 3, R]],
      ["a part of two vertices between restarts", [0, 1, 2, R, 3, 4, R, 5, 6, 7]],
      ["a part of one vertex between restarts", [0, 1, 2, R, 3, R, 5, 6, 7]],
    ] as const) {
      const { probes } = all([...values]);
      for (let k = 0; k + 2 < 8; k++)
        probes.push({ name: `${k},${k + 1},${k + 2}`, ...centroid(positions, k, k + 1, k + 2) });
      out.push({
        name: `indexed strip ${kind} ${label}`,
        positions,
        topology: "triangle-strip",
        index: { kind, values: [...values] },
        count: values.length,
        cullMode: "none",
        frontFace: "ccw",
        probes,
      });
    }
  }

  // A uint16 list through vertex 0xFFFF where 65,537 vertices exist, so that vertex is no index past the buffer.
  {
    const n = 65537;
    const positions: number[] = [];
    const ids: number[] = [];
    for (let k = 0; k < n; k++) {
      positions.push(5, 5, 0, 1);
      ids.push(k === 0xffff ? 200 : k === 0 ? 50 : k === 1 ? 51 : 0);
    }
    positions.splice(0, 8, -0.9, -0.9, 0, 1, 0.9, -0.9, 0, 1);
    positions.splice(0xffff * 4, 4, 0, 0.9, 0, 1);
    out.push({
      name: "indexed list uint16 through vertex 0xFFFF of 65537",
      positions,
      ids,
      topology: "triangle-list",
      index: { kind: "uint16", values: [0xffff, 0, 1] },
      count: 3,
      cullMode: "none",
      frontFace: "ccw",
      probes: [{ name: "centre", x: 32, y: 10 }],
    });
  }

  // An indexed draw whose first index and count run past the end of its index array.
  {
    const positions = zigzag(6);
    for (const [firstIndex, count] of [
      [0, 7],
      [4, 3],
    ])
      out.push({
        name: `indexed list of 6 indices drawn from ${firstIndex} for ${count}`,
        positions,
        topology: "triangle-list",
        index: { kind: "uint16", values: [0, 1, 2, 3, 4, 5] },
        firstIndex,
        count,
        cullMode: "none",
        frontFace: "ccw",
        probes: [
          { name: "0,1,2", ...centroid(positions, 0, 1, 2) },
          { name: "3,4,5", ...centroid(positions, 3, 4, 5) },
        ],
      });
  }

  // A fan, which only WebGL 2 has.
  {
    const positions = [0, 0, 0, 1, 0.8, -0.8, 0, 1, 0.8, 0.8, 0, 1, -0.8, 0.8, 0, 1];
    for (const cullMode of CULLS)
      out.push({
        name: `fan n=4 cull=${cullMode}`,
        positions,
        topology: "triangle-fan",
        count: 4,
        cullMode,
        frontFace: "ccw",
        probes: [
          { name: "0,1,2", ...centroid(positions, 0, 1, 2) },
          { name: "0,2,3", ...centroid(positions, 0, 2, 3) },
        ],
      });
  }
  return out;
}

/** Counts the pixels of an RGBA8 image, `bytesPerRow` apart, by what {@link decode} makes of each. */
export function histogram(bytes: ArrayLike<number>, bytesPerRow: number): Record<string, number> {
  const counts: Record<string, number> = {};
  for (let y = 0; y < HEIGHT; y++)
    for (let x = 0; x < WIDTH; x++) {
      const o = y * bytesPerRow + x * 4;
      const key = decode([bytes[o]!, bytes[o + 1]!, bytes[o + 2]!, bytes[o + 3]!]);
      counts[key] = (counts[key] ?? 0) + 1;
    }
  return counts;
}

/** What a probe pixel shows: nothing drawn, or the flat id, vertex index and facing of the fragment that drew it. */
export function decode(rgba: number[]): string {
  const [r, g, b, a] = rgba;
  if (a === 0) return "nothing";
  const facing = b === 255 ? "front" : b === 128 ? "back" : `facing?${b}`;
  return `id=${r} vi=${g} ${facing}${a === 255 ? "" : ` a=${a}`}`;
}
