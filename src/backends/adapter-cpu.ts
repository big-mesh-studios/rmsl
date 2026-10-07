import { AttributeNode, ShaderType, UniformArrayNode, UniformNode, UniformValue } from "../core";
import { Adapter, requestedStorageSlots, slotOf, TypedArray } from "./adapter";
import {
  CpuDrawBuffer,
  componentCountOf,
  ComputeStage,
  CpuGrid,
  CpuRoutine,
  CpuShaderContext,
  CpuTextureData,
  CpuValue,
} from "./cpu";

/** One typed array per storage slot, keyed by name. */
export type AdapterResult = Record<string, TypedArray>;

/**
 * `compute`/`draw` here are two independently optional stages: a
 * {@link ComputeStage} that `compute()` dispatches once per entity of a
 * `storage()` program, and a {@link CpuGrid} of `vec4` colours that `draw()`
 * fills once for each canvas pixel. Not
 * exported publicly: {@link createCpuAdapter} is wrapped by
 * `createJsCompute`/`createWasmCompute` (`compute` only) and
 * `createJsGrid`/`createWasmGrid` (`draw` only) — each passing a
 * single already-compiled program under its own field, never both.
 */
export interface CpuAdapterPrograms {
  compute?: ComputeStage;
  draw?: CpuGrid<"vec4">;
  /**
   * Keep the uniforms in a dictionary-mode object, whose values V8 stores
   * boxed, so a stage that reads a float uniform by a variable key reads it
   * without boxing it again. A stage that reads each uniform by name, once per
   * invocation, is faster without it.
   */
  boxedUniforms?: boolean;
}

/** `compute`/`draw` here are each required — unlike the base Adapter's
 * optional, possibly-async versions — even though `createJsGrid`/
 * `createWasmGrid` only ever build the `draw` half now (`compute()`
 * throws on the result). `createJsCompute`/`createWasmCompute` build the
 * `compute` half instead, but expose it through their own narrower
 * `JsComputeAdapter`/`WasmComputeAdapter` types rather than this one, so
 * their callers never see the always-throwing `draw()` this interface
 * still carries. Both are synchronous: this loop never awaits anything. */
export interface CpuAdapter extends Adapter<AdapterResult> {
  compute(out?: AdapterResult, count?: number): AdapterResult | void;
  draw(): void;
}

function clamp255(v: number): number {
  return Math.max(0, Math.min(255, Math.round(v * 255)));
}

/** `draw()`'s flat row-major buffer of four channels per pixel, read back as
 * 0..1 float colour the same convention GLSL/WGSL fragment output uses — into
 * a `CanvasRenderingContext2D`'s ImageData, the closest a CPU target has to a
 * GPU canvas surface. Exported for `createWasm`'s own rasterizer-backed
 * adapter (`adapter-wasm.ts`), which needs the same conversion for
 * `WasmRasterRoutine.draw()`'s output. */
export function bufferToImageData(buffer: CpuDrawBuffer, width: number, height: number): ImageData {
  const imageData = new ImageData(width, height);
  const rgba = imageData.data;
  for (let i = 0; i < width * height * 4; i++) rgba[i] = clamp255(buffer[i] as number);
  return imageData;
}

export function createCpuAdapter(programs: CpuAdapterPrograms): CpuAdapter {
  // Named for what each actually is, not restated from `programs` — the
  // call site below is `perPixel.draw(...)`, not `programs.draw.draw(...)`.
  const computeStep = programs.compute;
  const perPixel = programs.draw;
  const storageTypes = programs.compute?.storageTypes ?? {};

  let firstStorage: string | undefined;
  /** Components per element of each slot the host passed by node, from the
   *  node's own type; a slot passed by name falls back to the program's. */
  const elementWidths = new Map<string, number>();
  const storages: Record<string, TypedArray> = {};
  /** The slots of `storages`, in the order the host first passed them. */
  const storageSlots: string[] = [];
  const uniforms: Record<string, number | number[]> = programs.boxedUniforms ? Object.create(null) : {};
  const textures: Record<string, CpuTextureData> = {};
  /** The stage takes the host's own flat arrays, so one context serves every dispatch. */
  const stepContext = { storages, uniforms, textures } as unknown as CpuShaderContext;
  let canvas: HTMLCanvasElement | null = null;
  let ctx2d: CanvasRenderingContext2D | null = null;

  /** One invocation per element of the first storage buffer the host passed:
   *  the count TSL's caller would have written beside `instancedArray(count,
   *  type)`. A flat array holds one entry per component, so an array of `vec4`
   *  holds a quarter as many elements as it has components. */
  function elementCount(): number {
    if (firstStorage === undefined) return 0;
    return Math.floor(storages[firstStorage]!.length / elementWidths.get(firstStorage)!);
  }

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
    if (!(slot in storages)) storageSlots.push(slot);
    storages[slot] = data;
    elementWidths.set(
      slot,
      typeof attribute === "string"
        ? componentCountOf(storageTypes[slot] ?? "float")
        : componentCountOf(attribute._t ?? "float"),
    );
    firstStorage ??= slot;
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

    setTexture(sampler, texture) {
      textures[slotOf(sampler)] = texture;
    },

    compute(out, count) {
      if (!computeStep) throw new Error("[RMSL] this adapter has no `compute` program");
      computeStep(stepContext, count ?? elementCount());
      // storages already holds the caller's own arrays, mutated in place —
      // `out` is only for callers that want the WGSL adapter's optional-out
      // shape too, not something this loop needs to do its job.
      if (!out) return;
      for (const slot in out) if (!(slot in storages)) requestedStorageSlots(out, storageSlots);
      for (let i = 0; i < storageSlots.length; i++) {
        const slot = storageSlots[i]!;
        out[slot]?.set(storages[slot]!);
      }
      return out;
    },

    draw() {
      if (!perPixel || !canvas || !ctx2d) {
        throw new Error("[RMSL] this adapter has no `draw` program, or attach() was never called");
      }
      const buffer = perPixel({ uniforms, textures } as any, canvas.width, canvas.height);
      if (buffer.length !== canvas.width * canvas.height * 4) {
        throw new Error(
          "[RMSL] the program of a draw adapter has to return a colour: a vec4, or a value that converts to one.",
        );
      }
      ctx2d.putImageData(bufferToImageData(buffer, canvas.width, canvas.height), 0, 0);
    },

    destroy() {},
  };
}

/**
 * An adapter of a routine: it keeps the uniforms and the textures the host
 * sets, and `run` calls the routine with them and the parameters it is given,
 * so the host does not build a context for each call. `run` answers at once.
 */
export interface CpuRoutineAdapter<A extends ShaderType = ShaderType> {
  setUniform<T extends ShaderType>(uniform: UniformNode<T>, value: UniformValue<T>): void;
  setUniform<T extends ShaderType>(uniform: UniformArrayNode<T>, value: UniformValue<T>[]): void;
  setUniform(slot: string, value: number | number[]): void;
  setTexture(sampler: UniformNode<ShaderType> | string, texture: CpuTextureData): void;
  /** Calls the routine with the uniforms and textures set so far, and these parameters by name. */
  run(params?: Record<string, number | number[]>): CpuValue<A>;
  destroy(): void;
}

/** Wraps a routine in a {@link CpuRoutineAdapter}. */
export function createCpuRoutineAdapter<A extends ShaderType>(routine: CpuRoutine<A>): CpuRoutineAdapter<A> {
  const uniforms: Record<string, number | number[]> = {};
  const textures: Record<string, CpuTextureData> = {};

  function setUniform(uniform: UniformNode<ShaderType> | UniformArrayNode<ShaderType> | string, value: unknown): void {
    uniforms[slotOf(uniform)] = value as number | number[];
  }

  return {
    setUniform: setUniform as CpuRoutineAdapter<A>["setUniform"],
    setTexture(sampler, texture) {
      textures[slotOf(sampler)] = texture;
    },
    run: (params) => routine({ params, uniforms, textures }),
    destroy() {},
  };
}
