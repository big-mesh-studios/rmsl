import { random } from "../../src/testing/rasterizer-reference";

/**
 * 1500 triangles with coordinates on a grid of 1/16, from a seeded generator. In
 * `behind`, the second vertex lies behind the eye; `front` is the same set with
 * that vertex's w and z turned to lie in front of it. The z of each vertex in
 * front lies from 0 to its w.
 */
export function triangles(set: "front" | "behind", count = 1500, seed = 302): number[][] {
  const next = random(seed);
  const grid = (range: number) => Math.round((next() * 2 - 1) * range * 16) / 16;
  const out: number[][] = [];
  while (out.length < count) {
    const t: number[] = [];
    for (let k = 0; k < 3; k++) {
      const w = k === 1 ? -Math.abs(grid(2)) - 0.0625 : Math.abs(grid(2)) + 0.0625;
      t.push(grid(Math.abs(w) * 1.2), grid(Math.abs(w) * 1.2), grid(Math.abs(w) * 1.2), w);
    }
    for (const k of [0, 2]) t[4 * k + 2] = Math.abs(t[4 * k + 2]!) % t[4 * k + 3]!;
    out.push(set === "behind" ? t : t.map((c, i) => (i === 7 ? -c : i === 6 ? Math.abs(c) % -t[7]! : c)));
  }
  return out;
}
