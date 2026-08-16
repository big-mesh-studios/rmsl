/**
 * The browser and graphics device the test harnesses share.
 *
 * Both harnesses need the same two things — a Chromium with a software GL
 * driver, for GLSL, and a WebGPU device, for WGSL — and each used to stand up
 * its own. That is slow, and it is two places to change: releasing the device
 * had to be fixed in both, and one of them could easily have been missed.
 *
 * Everything here is created once and reused. What is memoised is the
 * *promise*, not the result: it is assigned before the first await, so two
 * callers arriving together share one launch instead of each starting their own
 * and one being dropped on the floor still running.
 */

declare const process: { env: Record<string, string | undefined> };

let browserPromise: Promise<any> | undefined;
let pagePromise: Promise<any> | undefined;
let devicePromise: Promise<any> | undefined;

/** A Chromium rendering through SwiftShader, so no real GPU is needed. */
export function gpuBrowser(): Promise<any> {
  browserPromise ??= (async () => {
    const { chromium } = await import("playwright");
    return chromium.launch({
      args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"],
    });
  })();
  return browserPromise;
}

/**
 * One blank page for the whole run.
 *
 * Opening a page per call meant navigating and tearing down a rendering context
 * every time, which cost far more than the work being measured.
 */
export function gpuPage(): Promise<any> {
  pagePromise ??= (async () => {
    const page = await (await gpuBrowser()).newPage();
    await page.goto("about:blank");
    return page;
  })();
  return pagePromise;
}

/** A WebGPU device, for handing WGSL to Dawn. */
export function gpuDevice(): Promise<any> {
  devicePromise ??= (async () => {
    let gpu;
    try {
      gpu = await import("@kmamal/gpu");
    } catch {
      throw new Error(
        `@kmamal/gpu is not installed, so WGSL cannot be handed to a real compiler. Set RMSL_SKIP_GPU=1 to run without the layers that need a graphics device.`
      );
    }
    const adapter = await gpu.create([]).requestAdapter();
    if (!adapter) throw new Error("No WebGPU adapter available");
    return adapter.requestDevice();
  })();
  return devicePromise;
}

/**
 * Compile GLSL shaders in the shared page and report what the driver said.
 *
 * Null means the shader compiled. Every other path returns a non-empty string,
 * including the ones where the driver says nothing: an empty message would be
 * falsy, and a caller testing for truth would read a failure as a success.
 */
export async function compileGLSLInPage(
  items: { src: string; stage: string }[],
): Promise<(string | null)[]> {
  if (items.length === 0) return [];
  const page = await gpuPage();
  return await page.evaluate((list: { src: string; stage: string }[]) => {
    const gl = document.createElement("canvas").getContext("webgl2");
    if (!gl) throw new Error("WebGL2 unavailable in the validation browser");
    return list.map(({ src, stage }) => {
      // A lost context makes every query afterwards return null, which would
      // otherwise read as a clean compile for this shader and every one after.
      if (gl.isContextLost()) {
        throw new Error("WebGL2 context was lost during validation");
      }
      const shader = gl.createShader(
        stage === "vertex" ? gl.VERTEX_SHADER : gl.FRAGMENT_SHADER,
      );
      if (!shader) return "could not create a shader object";
      gl.shaderSource(shader, src);
      gl.compileShader(shader);
      if (gl.getShaderParameter(shader, gl.COMPILE_STATUS)) return null;
      const log = (gl.getShaderInfoLog(shader) ?? "").trim().split("\n")[0];
      return log || "failed to compile, and the driver gave no message";
    });
  }, items);
}

/** Release everything held open. Safe to call when nothing was created. */
export async function releaseGpu(): Promise<void> {
  const page = pagePromise;
  const browser = browserPromise;
  const device = devicePromise;
  pagePromise = browserPromise = devicePromise = undefined;

  if (page) await (await page).close().catch(() => {});
  if (browser) await (await browser).close().catch(() => {});
  // The device holds native resources, so it is released rather than left for
  // the process to clean up.
  if (device) (await device).destroy?.();
}

/**
 * Whether the layers that need a graphics device run.
 *
 * On unless `RMSL_SKIP_GPU` says otherwise. It used to be off unless
 * `RMSL_GPU=1` said otherwise, which meant a plain `vitest run` — what an
 * editor runs, and what anyone gets who has not read the scripts — silently
 * skipped every test that hands a shader to a real compiler. Two renderer tests
 * reached main broken because of it: both failed the moment they were actually
 * run, and neither had been.
 *
 * A machine without a graphics device sets `RMSL_SKIP_GPU=1` and gets the old
 * behaviour. That is the right way round: skipping is a choice someone makes
 * about their machine, not the default everyone inherits.
 */
export const GPU_ENABLED = !process.env.RMSL_SKIP_GPU;
