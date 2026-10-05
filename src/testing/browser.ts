import { readFile } from "node:fs/promises";
import { build, type Plugin } from "esbuild";
import wabtInit from "wabt";
import { gpuPage, webgpuPage } from "./gpu";

/**
 * A `readPixel(canvas, x, y)` for an entry to read its result with, through a
 * 2D canvas rather than copying the WebGPU texture: the drawing surface is
 * `bgra8unorm` in one browser and `rgba8unorm` in another, and `getImageData`
 * is in the same channel order either way.
 */
export const READ_PIXEL = `
const readPixel = (canvas, x, y) => {
  const flat = document.createElement("canvas");
  flat.width = canvas.width;
  flat.height = canvas.height;
  const context = flat.getContext("2d");
  context.drawImage(canvas, 0, 0);
  const [r, g, b, a] = context.getImageData(x, y, 1, 1).data;
  return { r, g, b, a };
};
`;

/**
 * Compiles an imported `.wat` file to the bytes of its module, as the
 * `compileWat` Vite plugin does for the library build, so an entry can bundle
 * the WASM rasterizer.
 */
const watLoader: Plugin = {
  name: "rmsl:wat",
  setup(esbuild) {
    let wabt: Awaited<ReturnType<typeof wabtInit>> | undefined;
    esbuild.onLoad({ filter: /\.wat$/ }, async ({ path }) => {
      wabt ??= await wabtInit();
      const bytes = new Uint8Array(wabt.parseWat(path, await readFile(path, "utf8")).toBinary({}).buffer);
      return { contents: `export default new Uint8Array([${bytes.join(",")}]);`, loader: "js" };
    });
  },
};

/** Bundles a TypeScript entry for the browser, resolving its imports from `resolveDir`. */
async function bundleEntry(source: string, resolveDir: string): Promise<string> {
  const result = await build({
    stdin: { contents: source, resolveDir, loader: "ts" },
    bundle: true,
    write: false,
    format: "iife",
    platform: "browser",
    plugins: [watLoader],
    logLevel: "silent",
  });
  return result.outputFiles[0]!.text;
}

/**
 * Bundles an entry, runs it in `page`, and hands back what the async function
 * it assigned to `globalThis[entryPoint]` resolved to.
 */
async function runInPage(page: any, source: string, entryPoint: string, resolveDir: string): Promise<any> {
  const code = await bundleEntry(source, resolveDir);
  return await page.evaluate(
    async ([bundle, name]: [string, string]) => {
      // eslint-disable-next-line no-new-func
      new Function(bundle)();
      return await (globalThis as any)[name]();
    },
    [code, entryPoint] as [string, string],
  );
}

/** Runs an entry in the shared WebGPU page. */
export async function runInWebGpuPage(source: string, entryPoint: string, resolveDir: string): Promise<any> {
  return runInPage(await webgpuPage(), source, entryPoint, resolveDir);
}

/** Runs an entry in the shared WebGL page, which also serves a 2D canvas to the CPU adapters. */
export async function runInGpuPage(source: string, entryPoint: string, resolveDir: string): Promise<any> {
  return runInPage(await gpuPage(), source, entryPoint, resolveDir);
}
