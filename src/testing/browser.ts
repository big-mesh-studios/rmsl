import { build } from "esbuild";
import { webgpuPage } from "./gpu";

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

/** Bundles a TypeScript entry for the browser, resolving its imports from `resolveDir`. */
async function bundleEntry(source: string, resolveDir: string): Promise<string> {
  const result = await build({
    stdin: { contents: source, resolveDir, loader: "ts" },
    bundle: true,
    write: false,
    format: "iife",
    platform: "browser",
    logLevel: "silent",
  });
  return result.outputFiles[0]!.text;
}

/**
 * Bundles an entry, runs it in the shared WebGPU page, and hands back what the
 * async function it assigned to `globalThis[entryPoint]` resolved to.
 */
export async function runInWebGpuPage(source: string, entryPoint: string, resolveDir: string): Promise<any> {
  const page = await webgpuPage();
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
