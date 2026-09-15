// === CPU adapter loop (shared by the JS and WASM adapters) ===
// compileJS and compileWasm hand back a CpuRenderer, which is really two
// capabilities in one callable: call it once per entity with `storages`/
// `index` set (a storage()/invocationIndex() program, mutating shared
// arrays in place — WGSL's own compute contract, just host-driven), or
// call its `.draw()` once per pixel over a whole image (a fragCoord()
// program returning a color — the same "return a value" contract WGSL's
// own fragment stage has). One compiled program is only ever one or the
// other; `compute`/`draw` here are two independently optional
// CpuRenderers for exactly that reason.
//
// Not exported publicly: createJs (adapter-js.ts) and createWasm
// (adapter-wasm.ts) each wrap this around their own compile() calls, so a
// caller's constructor always takes root graphs, the same contract
// createGlsl/createWgsl have, instead of already-compiled callables only
// this generic version needed.
import { Adapter, TypedArray } from "./adapter";
import { CpuDrawBuffer, CpuRenderer } from "./cpu";

/** One typed array per storage slot, keyed by name. */
export type AdapterResult = Record<string, TypedArray>;

export interface CpuAdapterPrograms {
  compute?: CpuRenderer;
  draw?: CpuRenderer;
}

/** `compute`/`draw` here are each required — unlike the base Adapter's
 * optional, possibly-async versions — for whichever of the two this
 * adapter was actually built with; createJs/createWasm throw at
 * construction time otherwise. Both are synchronous: this loop never
 * awaits anything. */
export interface CpuAdapter extends Adapter<AdapterResult> {
  compute(out?: AdapterResult): AdapterResult | void;
  draw(): void;
}

function clamp255(v: number): number {
  return Math.max(0, Math.min(255, Math.round(v * 255)));
}

/** `draw()`'s flat row-major buffer, one program-defined channel count per
 * pixel, read back as 0..1 float color the same convention GLSL/WGSL
 * fragment output uses — into a `CanvasRenderingContext2D`'s ImageData,
 * the closest a CPU target has to a GPU canvas surface. */
function bufferToImageData(buffer: CpuDrawBuffer, width: number, height: number): ImageData {
  const componentCount = buffer.length / (width * height);
  const imageData = new ImageData(width, height);
  const rgba = imageData.data;
  for (let i = 0; i < width * height; i++) {
    const base = i * componentCount;
    const r = clamp255(buffer[base] as number);
    rgba[i * 4] = r;
    rgba[i * 4 + 1] = componentCount > 1 ? clamp255(buffer[base + 1] as number) : r;
    rgba[i * 4 + 2] = componentCount > 2 ? clamp255(buffer[base + 2] as number) : r;
    rgba[i * 4 + 3] = componentCount > 3 ? clamp255(buffer[base + 3] as number) : 255;
  }
  return imageData;
}

export function createCpuAdapter(programs: CpuAdapterPrograms): CpuAdapter {
  // Named for what each actually is, not restated from `programs` — the
  // call site below is `perPixel.draw(...)`, not the `.draw.draw(...)`
  // `programs.draw.draw(...)` would read as.
  const computeStep = programs.compute;
  const perPixel = programs.draw;

  let n = 0;
  const storages: Record<string, TypedArray> = {};
  const uniforms: Record<string, number | number[]> = {};
  let canvas: HTMLCanvasElement | null = null;
  let ctx2d: CanvasRenderingContext2D | null = null;

  return {
    attach(givenCanvas) {
      if (!perPixel) return;
      canvas = givenCanvas ?? document.createElement("canvas");
      const context = canvas.getContext("2d");
      if (!context) throw new Error("[RMSL] 2D canvas context unavailable");
      ctx2d = context;
    },

    setUniform(slot, value) {
      uniforms[slot] = value;
    },

    setAttribute(slot, data) {
      storages[slot] = data;
      n = Math.max(n, data.length);
    },

    compute(out) {
      if (!computeStep) throw new Error("[RMSL] this adapter has no `compute` program");
      for (let i = 0; i < n; i++) {
        computeStep({ storages, uniforms, index: i } as any);
      }
      // storages already holds the caller's own arrays, mutated in place —
      // `out` is only for callers that want the WGSL adapter's optional-out
      // shape too, not something this loop needs to do its job.
      if (!out) return;
      for (const slot in storages) (out[slot] as TypedArray).set(storages[slot]);
      return out;
    },

    draw() {
      if (!perPixel || !canvas || !ctx2d) {
        throw new Error("[RMSL] this adapter has no `draw` program, or attach() was never called");
      }
      const buffer = perPixel.draw({ uniforms } as any, canvas.width, canvas.height);
      ctx2d.putImageData(bufferToImageData(buffer, canvas.width, canvas.height), 0, 0);
    },

    destroy() {},
  };
}
