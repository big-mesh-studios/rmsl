/**
 * Draws the Lambert and Standard materials of `rmsl/scene` on a sphere through
 * the JS rasterizer, `compileJS`, and reports the time of a draw and what one
 * draw allocates.
 *
 * Usage: node --expose-gc scripts/bench-cpu/raster.ts [dist ...]
 * Each argument is a `dist/` directory to load the library from, the
 * repository's own by default. Compare branches by building each and passing
 * both directories.
 */
import { Session } from "node:inspector/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const dists = process.argv.slice(2).map((d) => resolve(d));
if (dists.length === 0) dists.push(join(root, "dist"));

const WIDTH = 128;
const HEIGHT = 128;
const BATCHES = 9;
const DRAWS = 4;
const WARMUP = 20;

/** A material on a sphere in a lit scene, compiled as both stages, with the draw that rasterizes it. */
async function workload(dist: string, name: string, make: (scene: any) => any) {
  const scene = await import(pathToFileURL(join(dist, "scene.js")).href);
  const { compileJS } = await import(pathToFileURL(join(dist, "js.js")).href);
  const material = make(scene);
  const world = new scene.Scene();
  world.add(new scene.AmbientLight(0xffffff, 0.2));
  const sun = new scene.DirectionalLight(0xffeedd, 1);
  sun.position.set(5, 10, 7);
  world.add(sun);
  const camera = new scene.PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 4);
  camera.lookAt(0, 0, 0);
  const geometry = new scene.SphereGeometry(1, 32, 16);
  const mesh = new scene.Mesh(geometry, material);
  world.add(mesh);
  world.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);
  const program = material.build(world);

  const uniforms: Record<string, unknown> = {};
  for (const binding of program.uniforms) {
    if (binding.scope === "camera") uniforms[binding.node.name] = scene.cameraUniformValue(binding.name, camera);
    else if (binding.scope === "object") uniforms[binding.node.name] = scene.objectUniformValue(binding.name, mesh);
    else uniforms[binding.node.name] = binding.value({ camera, mesh });
  }
  // The rasterizer draws a triangle from each three vertices in turn, so the indexed sphere is laid out flat.
  const index: ArrayLike<number> = geometry.index.array;
  const attributes: Record<string, Float64Array> = {};
  const attributeTypes: Record<string, string> = {};
  for (const binding of program.attributes) {
    const source = geometry.attributes[binding.name];
    const flat = new Float64Array(index.length * source.itemSize);
    for (let i = 0; i < index.length; i++) {
      for (let k = 0; k < source.itemSize; k++)
        flat[i * source.itemSize + k] = source.array[index[i]! * source.itemSize + k];
    }
    attributes[binding.node.name] = flat;
    attributeTypes[binding.node.name] = binding.node._t;
  }
  const raster = compileJS(
    () => program.vertexRoot,
    () => program.fragmentRoot,
    { attributeTypes },
  );
  const ctx = { attributes, uniforms };
  const options = { width: WIDTH, height: HEIGHT, clear: true, clearDepth: true };
  return { name, draw: () => raster.draw(ctx, options) as Float64Array };
}

/** The median of the batches' times for one draw, in milliseconds. */
function time(draw: () => unknown): number {
  for (let i = 0; i < WARMUP; i++) draw();
  const batches: number[] = [];
  for (let b = 0; b < BATCHES; b++) {
    const start = performance.now();
    for (let i = 0; i < DRAWS; i++) draw();
    batches.push((performance.now() - start) / DRAWS);
  }
  return batches.sort((a, b) => a - b)[BATCHES >> 1]!;
}

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
  for (const [name, make] of [
    ["lambert", (s: any) => new s.MeshLambertMaterial({ color: 0xff5533 })],
    ["standard", (s: any) => new s.MeshStandardMaterial({ color: 0xff5533, roughness: 0.25, metalness: 0.6 })],
  ] as const) {
    const { draw } = await workload(dist, name, make);
    const pixels = draw().slice();
    const reference = firstPixels.get(name);
    if (!reference) firstPixels.set(name, pixels);
    else if (pixels.some((value, i) => !Object.is(value, reference[i]))) differing.push(`${name}: ${dist}`);
    const ms = time(draw);
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
