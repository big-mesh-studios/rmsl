/** What every probe shares: the environment it ran in, and the pages it runs entries in. */
import { gpuBrowser, gpuPage, releaseGpu, webgpuBrowser, webgpuPage } from "../../src/testing/gpu";

export { runInGpuPage, runInWebGpuPage } from "../../src/testing/browser";
export { releaseGpu };

/** The directory entries resolve their imports from, so `../src/...` reaches the library. */
export const PROBES = new URL(".", import.meta.url).pathname.replace(/\.out\/$/, "");

/**
 * The browser, WebGL renderer and WebGPU adapter of each page: the WebGL page
 * renders through SwiftShader, the WebGPU page through the machine's GPU.
 */
export async function environment(): Promise<Record<string, unknown>> {
  const describe = async (page: any) =>
    page.evaluate(async () => {
      const gl = document.createElement("canvas").getContext("webgl2");
      const info = gl?.getExtension("WEBGL_debug_renderer_info");
      const adapter = await (navigator as any).gpu?.requestAdapter?.().catch(() => null);
      return {
        userAgent: navigator.userAgent,
        webgl: gl
          ? {
              renderer: gl.getParameter(info ? info.UNMASKED_RENDERER_WEBGL : gl.RENDERER),
              version: gl.getParameter(gl.VERSION),
            }
          : null,
        webgpu: adapter
          ? {
              vendor: adapter.info?.vendor,
              architecture: adapter.info?.architecture,
              description: adapter.info?.description,
            }
          : null,
      };
    });
  return {
    node: process.version,
    platform: `${process.platform} ${process.arch}`,
    webglPage: { browser: (await gpuBrowser()).version(), ...(await describe(await gpuPage())) },
    webgpuPage: { browser: (await webgpuBrowser()).version(), ...(await describe(await webgpuPage())) },
  };
}

/** Prints the environment and the result as one JSON document, and releases the browsers. */
export async function report(name: string, result: () => Promise<unknown>): Promise<void> {
  try {
    console.log(JSON.stringify({ probe: name, environment: await environment(), result: await result() }, null, 2));
  } finally {
    await releaseGpu();
  }
}
