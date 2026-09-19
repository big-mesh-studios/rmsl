import { AttributeNode, ShaderType, UniformArrayNode, UniformNode, UniformValue } from "../core";
import { Adapter, slotOf, TypedArray } from "./adapter";
import { CpuDrawBuffer, CpuRoutine } from "./cpu";

/** One typed array per storage slot, keyed by name. */
export type AdapterResult = Record<string, TypedArray>;

/**
 * `compute`/`batch` here are two independently optional {@link CpuRoutine}s —
 * `invoke()`d once per entity for a `compute` program, or `batch()`d once per
 * pixel for a `batch` program (named for the `CpuRoutine` method it's run
 * through, not the `Adapter.draw()` it's wired into below — those are two
 * different things sharing a canvas-render step, not one). Not exported
 * publicly: {@link createCpuAdapter} is wrapped by
 * `createJsCompute`/`createWasmCompute` (`compute` only) and
 * `createJsRoutine`/`createWasmRoutine` (`batch` only) — each passing a
 * single already-compiled routine under its own field, never both, now
 * that those are separate entry points rather than one options bag.
 */
export interface CpuAdapterPrograms {
  compute?: CpuRoutine;
  batch?: CpuRoutine;
}

/** `compute`/`draw` here are each required — unlike the base Adapter's
 * optional, possibly-async versions — even though `createJsRoutine`/
 * `createWasmRoutine` only ever build the `batch` half now (`compute()`
 * throws on the result). `createJsCompute`/`createWasmCompute` build the
 * `compute` half instead, but expose it through their own narrower
 * `JsComputeAdapter`/`WasmComputeAdapter` types rather than this one, so
 * their callers never see the always-throwing `draw()` this interface
 * still carries. Both are synchronous: this loop never awaits anything. */
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
 * the closest a CPU target has to a GPU canvas surface. Exported for
 * `createWasm`'s own rasterizer-backed adapter (`adapter-wasm.ts`), which
 * needs the same conversion for `WasmRasterRoutine.draw()`'s output. */
export function bufferToImageData(buffer: CpuDrawBuffer, width: number, height: number): ImageData {
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
  // call site below is `perPixel.batch(...)`, not `programs.batch.batch(...)`.
  const computeStep = programs.compute;
  const perPixel = programs.batch;

  let n = 0;
  const storages: Record<string, TypedArray> = {};
  const uniforms: Record<string, number | number[]> = {};
  let canvas: HTMLCanvasElement | null = null;
  let ctx2d: CanvasRenderingContext2D | null = null;

  function setUniform<T extends ShaderType>(uniform: UniformNode<T>, value: UniformValue<T>): void;
  function setUniform<T extends ShaderType>(uniform: UniformArrayNode<T>, value: UniformValue<T>[]): void;
  function setUniform(slot: string, value: number | number[]): void;
  function setUniform(uniform: UniformNode<ShaderType> | UniformArrayNode<ShaderType> | string, _value: unknown): void {
    uniforms[slotOf(uniform)] = _value as number | number[];
  }

  function setAttribute<T extends ShaderType>(attribute: AttributeNode<T>, data: TypedArray): void;
  function setAttribute(slot: string, data: TypedArray): void;
  function setAttribute(attribute: AttributeNode<ShaderType> | string, data: TypedArray): void {
    const slot = slotOf(attribute);
    storages[slot] = data;
    n = Math.max(n, data.length);
  }

  return {
    attach(givenCanvas) {
      if (!perPixel) return;
      canvas = givenCanvas ?? document.createElement("canvas");
      const context = canvas.getContext("2d");
      if (!context) throw new Error("[RMSL] 2D canvas context unavailable");
      ctx2d = context;
    },

    setUniform,
    setAttribute,

    compute(out) {
      if (!computeStep) throw new Error("[RMSL] this adapter has no `compute` program");
      for (let i = 0; i < n; i++) {
        computeStep.invoke({ storages, uniforms, index: i } as any);
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
        throw new Error("[RMSL] this adapter has no `batch` program, or attach() was never called");
      }
      const buffer = perPixel.batch({ uniforms } as any, canvas.width, canvas.height);
      ctx2d.putImageData(bufferToImageData(buffer, canvas.width, canvas.height), 0, 0);
    },

    destroy() {},
  };
}
