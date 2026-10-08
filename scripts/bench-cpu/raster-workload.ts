/**
 * The draws scripts/bench-cpu/raster.ts and raster-browsers.ts time: the
 * Lambert and Standard materials of `rmsl/scene` on a sphere, drawn through the
 * JS rasterizer, `compileJS`. It takes the library's modules as it is given
 * them, so it runs in Node and in a browser page alike.
 */

export const WIDTH = 128;
export const HEIGHT = 128;
export const BATCHES = 9;
export const DRAWS = 4;
export const WARMUP = 20;

/** The library's modules a draw is built with: `rmsl/scene` and `rmsl/js`. */
export type Library = { scene: any; js: any };

/** Each material drawn, by name. */
export const MATERIALS: [string, (scene: any) => any][] = [
  ["lambert", (s) => new s.MeshLambertMaterial({ color: 0xff5533 })],
  ["standard", (s) => new s.MeshStandardMaterial({ color: 0xff5533, roughness: 0.25, metalness: 0.6 })],
];

/** A material on a sphere in a lit scene, compiled as both stages, as a function that draws it once. */
export function rasterDraw({ scene, js }: Library, make: (scene: any) => any): () => Float64Array {
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
  const raster = js.compileJS(
    () => program.vertexRoot,
    () => program.fragmentRoot,
    { attributeTypes },
  );
  const ctx = { attributes, uniforms };
  const options = { width: WIDTH, height: HEIGHT, clear: true, clearDepth: true };
  return () => raster.draw(ctx, options) as Float64Array;
}

/** The median of the batches' times for one draw, in milliseconds. */
export function timeDraw(draw: () => unknown): number {
  for (let i = 0; i < WARMUP; i++) draw();
  const batches: number[] = [];
  for (let b = 0; b < BATCHES; b++) {
    const start = performance.now();
    for (let i = 0; i < DRAWS; i++) draw();
    batches.push((performance.now() - start) / DRAWS);
  }
  return batches.sort((a, b) => a - b)[BATCHES >> 1]!;
}

/** Whether two draws gave the same pixels, bit for bit. */
export function samePixels(a: Float64Array, b: Float64Array): boolean {
  return a.length === b.length && a.every((value, i) => Object.is(value, b[i]));
}
