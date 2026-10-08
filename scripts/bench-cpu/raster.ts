/**
 * Draws the Lambert and Standard materials of `rmsl/scene` on a sphere through
 * the JS rasterizer, `compileJS`, and reports the time of a draw and what one
 * draw allocates. raster-browsers.ts times the same draws in Firefox and
 * Chromium.
 *
 * Usage: node --expose-gc scripts/bench-cpu/raster.ts [dist ...]
 * Each argument is a `dist/` directory to load the library from, the
 * repository's own by default. Compare branches by building each and passing
 * both directories.
 */
import { Session } from "node:inspector/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  DRAWS,
  HEIGHT,
  MATERIALS,
  BATCHES,
  WARMUP,
  WIDTH,
  rasterDraw,
  samePixels,
  timeDraw,
} from "./raster-workload.ts";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const dists = process.argv.slice(2).map((d) => resolve(d));
if (dists.length === 0) dists.push(join(root, "dist"));

/** Kilobytes one draw allocates, by V8's sampling heap profiler, counting only the library's own code. */
async function allocated(dist: string, draw: () => unknown): Promise<number> {
  // Connected before the draws warm up: connecting deoptimizes the code it finds running.
  const session = new Session();
  session.connect();
  await session.post("HeapProfiler.enable");
  for (let i = 0; i < WARMUP; i++) draw();
  await session.post("HeapProfiler.startSampling", {
    samplingInterval: 4096,
    includeObjectsCollectedByMajorGC: true,
    includeObjectsCollectedByMinorGC: true,
  });
  for (let i = 0; i < DRAWS; i++) draw();
  const { profile } = await session.post("HeapProfiler.stopSampling");
  session.disconnect();
  // A compiled program has no source URL; the rest of the library is in `dist`.
  const library = pathToFileURL(dist).href;
  let bytes = 0;
  const walk = (node: any): void => {
    const url: string = node.callFrame.url;
    if (url === "" || url.startsWith(library)) bytes += node.selfSize;
    for (const child of node.children) walk(child);
  };
  walk(profile.head);
  return bytes / DRAWS / 1024;
}

const columns: string[] = [];
const rows = new Map<string, string[]>();
/** The pixels each program drew with the first library, to compare the others' with. */
const firstPixels = new Map<string, Float64Array>();
const differing: string[] = [];
for (const dist of dists) {
  columns.push(`${dist.replace(`${root}/`, "")} ms`, "KB/draw");
  const library = {
    scene: await import(pathToFileURL(join(dist, "scene.js")).href),
    js: await import(pathToFileURL(join(dist, "js.js")).href),
  };
  for (const [name, make] of MATERIALS) {
    const draw = rasterDraw(library, make);
    const pixels = draw().slice();
    const reference = firstPixels.get(name);
    if (!reference) firstPixels.set(name, pixels);
    else if (!samePixels(pixels, reference)) differing.push(`${name}: ${dist}`);
    const ms = timeDraw(draw);
    const kb = await allocated(dist, draw);
    rows.set(name, [...(rows.get(name) ?? []), ms.toFixed(2), kb.toFixed(1)]);
  }
}
console.log(
  `== node ${process.versions.node}, a ${WIDTH}x${HEIGHT} draw of a sphere, median of ${BATCHES} batches of ${DRAWS}`,
);
console.log(["program".padEnd(10), ...columns.map((c) => c.padStart(16))].join(""));
for (const [name, cells] of rows) console.log([name.padEnd(10), ...cells.map((c) => c.padStart(16))].join(""));
if (dists.length > 1) {
  console.log(
    differing.length
      ? `pixels differ from the first library:\n${differing.join("\n")}`
      : "every library draws the same pixels",
  );
}
