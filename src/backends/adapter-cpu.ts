import { RoutineParam, RoutineParams, ShaderType, UniformNode } from "../core";
import {
  Adapter,
  AttributeOrSlot,
  AttributeOrStorage,
  assertReadbackFits,
  HostInputGuard,
  SetUniform,
  slotOf,
  TypedArray,
  UniformOrSlot,
  unknownStorageSlots,
} from "./adapter";
import { assertHostArray } from "./vertex-format";
import { assertHostShape } from "./host-shape";
import {
  CpuDrawBuffer,
  componentCountOf,
  ComputeStage,
  CpuGrid,
  CpuRoutine,
  CpuShaderContext,
  CpuTextureData,
  CpuValue,
  FloatWidth,
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
  /** What a uniform or an array the host sets is checked against: the guard of the program's stages. */
  guard: HostInputGuard;
  /** The width the stages compute floats in, and so the width of a float storage buffer the host never sets. */
  float?: FloatWidth;
}

/** `compute`/`draw` here are each required — unlike the base Adapter's
 * optional, possibly-async versions — even though `createJsGrid`/
 * `createWasmGrid` only ever build the `draw` half now (`compute()`
 * throws on the result). `createJsCompute`/`createWasmCompute` build the
 * `compute` half instead, but expose it through their own narrower
 * `JsComputeAdapter`/`WasmComputeAdapter` types rather than this one, so
 * their callers never see the always-throwing `draw()` this interface
 * still carries. Both are synchronous: this loop never awaits anything. */
export interface CpuAdapter extends Adapter<AdapterResult, void, Float32Array | Float64Array> {
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
  /** Components per element of each slot, from the type of the program's node
   *  or of the node the host passed; a slot passed by name falls back to the program's. */
  const elementWidths = new Map<string, number>();
  /** Each storage buffer by slot: the host's own array, or, at the first dispatch, a copy for one it never set. */
  const storages: Record<string, TypedArray> = {};
  /** The slots of the storage buffers: the program's, in the order it created them, then any other the host passed. */
  const storageSlots: string[] = [...programs.guard.storages.keys()];
  for (const [slot, node] of programs.guard.storages) elementWidths.set(slot, componentCountOf(node._t));
  let unsetStorages = programs.guard.storages.size > 0;
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
    const slot = firstStorage ?? storageSlots[0];
    if (slot === undefined) return 0;
    return Math.floor(storages[slot]!.length / elementWidths.get(slot)!);
  }

  /**
   * Starts each storage buffer the host never set as a copy of its
   * attribute's array, as a GPU buffer does. A float buffer takes the width
   * the stages compute in, as an array the host sets may.
   */
  function startUnsetStorages(): void {
    if (!unsetStorages) return;
    unsetStorages = false;
    for (const [slot, node] of programs.guard.storages) {
      if (Object.hasOwn(storages, slot)) continue;
      const array = node.attribute.array;
      storages[slot] =
        array instanceof Float32Array && programs.float !== "f32" ? Float64Array.from(array) : array.slice();
    }
  }

  function setUniform(uniform: UniformOrSlot, value: unknown): void {
    programs.guard.uniform(uniform, value);
    uniforms[slotOf(uniform)] = value as number | number[];
  }

  function setAttribute<T extends ShaderType>(attribute: AttributeOrStorage<T>, data: TypedArray): void;
  function setAttribute(slot: string, data: TypedArray): void;
  function setAttribute(attribute: AttributeOrSlot, data: TypedArray): void {
    const slot = slotOf(attribute);
    programs.guard.array(attribute, data);
    if (!storageSlots.includes(slot)) storageSlots.push(slot);
    storages[slot] = data;
    const elementType =
      typeof attribute === "string"
        ? (storageTypes[slot] ?? programs.guard.storages.get(slot)?._t)
        : "attribute" in attribute
          ? attribute.attribute.elementType
          : attribute._t;
    elementWidths.set(slot, componentCountOf(elementType ?? "float"));
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
      startUnsetStorages();
      // An out array is checked before the dispatch, so a refusal leaves the storages as they were.
      if (out) {
        for (const slot in out) if (!Object.hasOwn(storages, slot)) throw unknownStorageSlots(out, storageSlots);
        for (let i = 0; i < storageSlots.length; i++) {
          const slot = storageSlots[i]!;
          if (!Object.hasOwn(out, slot)) continue;
          const slotType = storageTypes[slot];
          if (slotType !== undefined) assertHostArray("storage buffer", slot, slotType, out[slot]!, true);
          assertReadbackFits(slot, out[slot]!, storages[slot]!.length);
        }
      }
      computeStep(stepContext, count ?? elementCount());
      // storages already holds the caller's own arrays, mutated in place —
      // `out` is only for callers that want the WGSL adapter's optional-out
      // shape too, not something this loop needs to do its job.
      if (!out) return;
      for (let i = 0; i < storageSlots.length; i++) {
        const slot = storageSlots[i]!;
        if (Object.hasOwn(out, slot)) out[slot]!.set(storages[slot]!);
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
 * `setUniform` refuses a value of another shape than its uniform's type, and
 * `run` a parameter of another shape than its type.
 */
export interface CpuRoutineAdapter<
  A extends ShaderType = ShaderType,
  W extends FloatWidth = "f64",
  P extends readonly RoutineParam[] = readonly RoutineParam[],
> {
  setUniform: SetUniform;
  setTexture(sampler: UniformNode<ShaderType> | string, texture: CpuTextureData): void;
  /** Calls the routine with the uniforms and textures set so far, and these parameters by name. */
  run(params?: RoutineParams<P>): CpuValue<A, W>;
  destroy(): void;
}

/**
 * Wraps a routine in a {@link CpuRoutineAdapter}, which checks a uniform
 * with `guard` where it is set, and each of `params` where `run` is given it.
 */
export function createCpuRoutineAdapter<
  A extends ShaderType,
  W extends FloatWidth = "f64",
  P extends readonly RoutineParam[] = readonly RoutineParam[],
>(routine: CpuRoutine<A, W>, params: P, guard: HostInputGuard): CpuRoutineAdapter<A, W, P> {
  const uniforms: Record<string, number | number[]> = {};
  const textures: Record<string, CpuTextureData> = {};

  function setUniform(uniform: UniformOrSlot, value: unknown): void {
    guard.uniform(uniform, value);
    uniforms[slotOf(uniform)] = value as number | number[];
  }

  return {
    setUniform,
    setTexture(sampler, texture) {
      textures[slotOf(sampler)] = texture;
    },
    run(given) {
      for (let i = 0; i < params.length; i++) {
        const param = params[i]!;
        assertHostShape("parameter", param.name, param.type, (given as Record<string, unknown>)?.[param.name]);
      }
      return routine({ params: given, uniforms, textures });
    },
    destroy() {},
  };
}
