/**
 * Builds the benchmark workloads from the library in `dist/` and writes, for
 * each, the JS source its compile gives at each float width, with what a run
 * needs: uniform values, texture data, and which varyings carry the surface.
 *
 * Usage: node scripts/bench-cpu/generate.ts <label>
 * Writes reports/bench-cpu/<label>.json. Compare branches by generating each
 * under its own label, then running scripts/bench-cpu/run.ts over them.
 */
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { Program, TextureJson } from "./run.ts";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
execFileSync("node", [join(root, "scripts", "ensure-build.mjs")], { stdio: "inherit" });

const rmsl = await import(join(root, "dist", "rmsl.js"));
const { compileJSFn } = await import(join(root, "dist", "js.js"));
const effects = await import(join(root, "dist", "effects.js"));
const scene = await import(join(root, "dist", "scene.js"));

const label = process.argv[2];
if (!label) throw new Error("usage: node scripts/bench-cpu/generate.ts <label>");

const WIDTH = 128;
const HEIGHT = 128;
const WIDTHS = ["f64", "f32"];

/** A program built from the library, before it is compiled, with what a run passes it. */
type Workload = Omit<Program, "width" | "height" | "sources"> & { root: unknown };

/** An RGBA image of 8-bit channels with smooth gradients and hard edges, so every filter has work to do. */
function image(width: number, height: number): TextureJson {
  const data: number[] = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const checker = ((x >> 4) + (y >> 4)) % 2;
      data.push((x * 255) / width, (y * 255) / height, checker * 200 + 30, 255);
    }
  }
  return { type: "Uint8Array", data, width, height, channels: 4, magFilter: "linear", wrapS: "clamp", wrapT: "clamp" };
}

/** Every uniform node a graph reads, by slot. */
function uniformsOf(root: any): Map<string, string> {
  const found = new Map<string, string>();
  const seen = new Set<unknown>();
  const walk = (node: any): void => {
    if (!node || typeof node !== "object" || seen.has(node)) return;
    seen.add(node);
    if (node.type === "uniform" && node.value?.slot) found.set(node.value.slot, node._t);
    for (const p of node.params ?? []) walk(p);
  };
  walk(root);
  return found;
}

/** An effect evaluated once for each pixel of an image: the texture is the image, and a `vec2` uniform is the screen size. */
function effectWorkload(name: string, build: (tex: any) => any): Workload {
  const tex = rmsl.uniform("sampler2D");
  const root = build(tex);
  const uniforms: Record<string, unknown> = {};
  const textures: Record<string, TextureJson> = {};
  for (const [slot, type] of uniformsOf(root)) {
    if (type === "sampler2D") textures[slot] = image(WIDTH, HEIGHT);
    else if (type === "vec2") uniforms[slot] = [WIDTH, HEIGHT];
    else if (type === "float") uniforms[slot] = 0.5;
    else throw new Error(`${name}: no value for a ${type} uniform`);
  }
  return { name, kind: "grid", root, ctx: { uniforms, textures }, surface: null };
}

/** A material shading each pixel of a sphere in a lit scene, through its fragment stage. */
function materialWorkload(name: string, material: any): Workload {
  const world = new scene.Scene();
  world.add(new scene.AmbientLight(0xffffff, 0.2));
  const sun = new scene.DirectionalLight(0xffeedd, 1);
  sun.position.set(5, 10, 7);
  world.add(sun);
  const point = new scene.PointLight(0x4477ff, 2, 10, 2);
  point.position.set(-3, 1, -2);
  world.add(point);
  const camera = new scene.PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 0, 4);
  camera.lookAt(0, 0, 0);
  const mesh = new scene.Mesh(new scene.SphereGeometry(1, 32, 16), material);
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
  const slot = (role: string): string | undefined => program.varyings.find((v: any) => v.name === role)?.node.name;
  const varyings: Record<string, number[]> = {};
  for (const v of program.varyings) varyings[v.node.name] = new Array(v.node._t === "vec2" ? 2 : 3).fill(0);
  return {
    name,
    kind: "fragment",
    root: program.fragmentRoot,
    ctx: { uniforms, textures: {}, varyings },
    surface: { position: slot("positionWorld"), normal: slot("normalWorld") },
  };
}

const { uv } = rmsl;
const sample = (tex: any) => texture(tex, uv());
const workloads = [
  effectWorkload("sepia", (tex) => effects.sepia(sample(tex))),
  effectWorkload("bleach", (tex) => effects.bleach(sample(tex))),
  effectWorkload("dotScreen", (tex) => effects.dotScreen(sample(tex))),
  effectWorkload("rgbShift", (tex) => effects.rgbShift(tex)),
  effectWorkload("sobel", (tex) => effects.sobel(tex)),
  effectWorkload("sharpen", (tex) => effects.sharpen(tex)),
  effectWorkload("fxaa", (tex) => effects.fxaa(tex)),
  effectWorkload("boxBlur", (tex) => effects.boxBlur(tex, { size: 2 })),
  effectWorkload("radialBlur", (tex) => effects.radialBlur(tex, { count: 16 })),
  materialWorkload("lambert", new scene.MeshLambertMaterial({ color: 0xff5533 })),
  materialWorkload("standard", new scene.MeshStandardMaterial({ color: 0xff5533, roughness: 0.25, metalness: 0.6 })),
];

const programs: Program[] = workloads.map(({ name, kind, root, ctx, surface }) => {
  const sources: Record<string, string> = {};
  for (const float of WIDTHS) {
    sources[float] = compileJSFn(() => root, { name: "main", params: [], stage: "fragment", float });
  }
  return { name, kind, width: WIDTH, height: HEIGHT, sources, ctx, surface };
});

const outDir = join(root, "reports", "bench-cpu");
mkdirSync(outDir, { recursive: true });
const file = join(outDir, `${label}.json`);
writeFileSync(file, JSON.stringify({ label, programs }));
console.log(`wrote ${file}: ${programs.map((p) => p.name).join(", ")}`);
