import { AttributeNode, Node, ShaderType, UniformArrayNode, UniformNode, UniformValue } from "../../core";
import {
  Adapter,
  AttributeOrSlot,
  DrawClearOptions,
  DrawCountOptions,
  slotOf,
  TypedArray,
  UniformOrSlot,
} from "../adapter";
import {
  AdapterResult,
  bufferToImageData,
  CpuAdapter,
  CpuRoutineAdapter,
  createCpuAdapter,
  createCpuRoutineAdapter,
} from "../adapter-cpu";
import type { CpuTextureData, FloatWidth } from "../cpu";
import { compileWasm, CompileWasmOptions, WasmRasterContext } from "./rasterizer";
import { fragmentColour } from "../shared";
import {
  compileWasmCompute,
  compileWasmGrid,
  compileWasmRoutine,
  CompileWasmFnOptions,
  floatWidthOf,
  WasmFloatWidth,
} from "./wasm";

export type CreateWasmGridOptions = WasmFloatWidth & {
  /** A fragCoord() program that returns a colour, evaluated once per canvas pixel by `draw()`. */
  draw: Node<ShaderType>;
  name?: string;
  params?: CompileWasmFnOptions["params"];
  derivatives?: CompileWasmFnOptions["derivatives"];
  reentrant?: CompileWasmFnOptions["reentrant"];
  memory?: CompileWasmFnOptions["memory"];
  sharedMemory?: CompileWasmFnOptions["sharedMemory"];
  maxMemoryPages?: CompileWasmFnOptions["maxMemoryPages"];
};

/**
 * Compiles a `fragCoord()` program with {@link compileWasmGrid} and
 * wraps it in a {@link createCpuAdapter} — a plain CPU-callable evaluated
 * once per pixel/sample via its grid's in-WASM loop (`docs/wasm.md`'s
 * screen-pick/ray-march niche, or a `width x 1` per-sample audio-DSP
 * buffer), not a wgpu pipeline shape. See {@link createWasmCompute} for
 * the `storage()`/`invocationIndex()` shape and {@link createWasm} for
 * the vertex/fragment render shape — those each got their own dedicated
 * entry point rather than living as options here for the same reason.
 */
export function createWasmGrid(options: CreateWasmGridOptions): CpuAdapter {
  const draw = compileWasmGrid(() => fragmentColour([options.draw])[0] as Node<"vec4">, {
    name: options.name ?? "draw",
    params: options.params ?? [],
    derivatives: options.derivatives,
    reentrant: options.reentrant,
    memory: options.memory,
    sharedMemory: options.sharedMemory,
    maxMemoryPages: options.maxMemoryPages,
    ...floatWidthOf(options),
  });

  return createCpuAdapter({ draw });
}

export type CreateWasmRoutineOptions = WasmFloatWidth & {
  name?: string;
  params?: CompileWasmFnOptions["params"];
  derivatives?: CompileWasmFnOptions["derivatives"];
  reentrant?: CompileWasmFnOptions["reentrant"];
  memory?: CompileWasmFnOptions["memory"];
  sharedMemory?: CompileWasmFnOptions["sharedMemory"];
  maxMemoryPages?: CompileWasmFnOptions["maxMemoryPages"];
};

/**
 * Compiles a function of parameters and uniforms with {@link compileWasmRoutine}
 * and wraps it in a {@link CpuRoutineAdapter}: `setUniform` and `setTexture`
 * keep what the host gives it, and `run(params)` calls the routine with them.
 */
export function createWasmRoutine<A extends ShaderType, W extends FloatWidth = "f64">(
  fn: Node<A>,
  options: CreateWasmRoutineOptions & { float?: W } = {},
): CpuRoutineAdapter<A, W> {
  return createCpuRoutineAdapter<A, W>(
    compileWasmRoutine(() => fn, {
      name: options.name ?? "routine",
      params: options.params ?? [],
      derivatives: options.derivatives,
      reentrant: options.reentrant,
      memory: options.memory,
      sharedMemory: options.sharedMemory,
      maxMemoryPages: options.maxMemoryPages,
      ...floatWidthOf(options),
    }),
  );
}

export type CreateWasmComputeOptions = WasmFloatWidth & {
  name?: string;
  params?: CompileWasmFnOptions["params"];
  derivatives?: CompileWasmFnOptions["derivatives"];
  reentrant?: CompileWasmFnOptions["reentrant"];
  memory?: CompileWasmFnOptions["memory"];
  sharedMemory?: CompileWasmFnOptions["sharedMemory"];
  maxMemoryPages?: CompileWasmFnOptions["maxMemoryPages"];
};

/**
 * `setAttribute`/`setUniform` collect draw state the way {@link CpuAdapter}
 * does; `compute()` is the only invocation method, since a `storage()`/
 * `invocationIndex()` program has no `draw()`/`attach()` counterpart —
 * unlike `CpuAdapter`, that's not just unused here, it's not part of the
 * type at all.
 *
 * `compute`'s `count` is the number of invocations to run. Given none, it is
 * one per element of the first storage buffer `setAttribute` was given.
 */
export interface WasmComputeAdapter {
  setUniform<T extends ShaderType>(uniform: UniformNode<T>, value: UniformValue<T>): void;
  setUniform<T extends ShaderType>(uniform: UniformArrayNode<T>, value: UniformValue<T>[]): void;
  setUniform(slot: string, value: number | number[]): void;
  setAttribute<T extends ShaderType>(attribute: AttributeNode<T>, data: TypedArray): void;
  setAttribute(slot: string, data: TypedArray): void;
  compute(out?: AdapterResult, count?: number): AdapterResult | void;
}

/**
 * Compiles a `storage()`/`invocationIndex()` program with
 * {@link compileWasmCompute} and wraps it in a {@link createCpuAdapter} —
 * the wgpu-compute-pipeline-shaped counterpart to {@link createWasm}'s
 * render-pipeline shape. `compute()` copies each storage buffer into WASM
 * memory once and runs every invocation in one call to the module's own
 * dispatch loop. Invocations still run one after another, with no workgroup
 * model (issue #8). This only narrows the *type*, matching
 * `storage()`/`invocationIndex()`'s own shape instead of reusing
 * `createWasmGrid`'s `draw`-shaped, non-pipeline option bag.
 */
export function createWasmCompute(
  compute: Node<ShaderType> | readonly Node<ShaderType>[],
  options: CreateWasmComputeOptions = {},
): WasmComputeAdapter {
  const computeRoutine = compileWasmCompute(() => compute, {
    name: options.name ?? "compute",
    params: options.params ?? [],
    derivatives: options.derivatives,
    reentrant: options.reentrant,
    memory: options.memory,
    sharedMemory: options.sharedMemory,
    maxMemoryPages: options.maxMemoryPages,
    ...floatWidthOf(options),
    // A float passed as an argument to a module is boxed on every call; in memory it allocates nothing.
    scalarsInMemory: true,
  });

  // The module reads each uniform once per dispatch, by its slot.
  return createCpuAdapter({ compute: computeRoutine, boxedUniforms: true });
}

/**
 * `draw()`'s own options — `count`/`first` from `DrawCountOptions`, same
 * as GL/WGSL: unset `count` defaults to the first attribute
 * the host passed. `clear` and `clearDepth` default to `true`: the common case
 * for a single-material adapter is one `draw()` call, one whole frame —
 * mirroring how a WebGPU render pass declares `loadOp`/`depthLoadOp`
 * together, per pass, rather than clearing as a separate operation. Pass
 * `false` to composite several `draw()` calls into one frame instead
 * (several materials sharing a framebuffer/depth buffer); `compileWasm`'s
 * own `WasmRasterRoutine` is the lower-level primitive that composability
 * is built on.
 */
export interface WasmDrawOptions extends DrawCountOptions, DrawClearOptions {
  clearDepth?: boolean;
}

/**
 * Compiles a vertex/fragment `Fn` pair with {@link compileWasm} and wraps
 * the resulting `WasmRasterRoutine` in the uniform `Adapter` interface —
 * `setAttribute`/`setUniform` collect draw state, `attach` opens a 2D
 * canvas context, and `draw()` runs one frame into it. See
 * {@link WasmDrawOptions} for `count`/`first`/`clear`/`clearDepth`.
 */
export interface WasmAdapter extends Adapter<void, WasmDrawOptions> {
  attach(canvas?: HTMLCanvasElement): void;
  draw(options?: WasmDrawOptions): void;
}

export function createWasm(
  vertexFn: (...args: any[]) => any,
  fragmentFn: (...args: any[]) => any,
  options: CompileWasmOptions = {},
): WasmAdapter {
  const routine = compileWasm(vertexFn, fragmentFn, options);

  const uniforms: Record<string, number | number[]> = {};
  const attributes: Record<string, TypedArray> = {};
  const textures: Record<string, CpuTextureData> = {};
  let canvas: HTMLCanvasElement | null = null;
  let ctx2d: CanvasRenderingContext2D | null = null;

  function setUniform<T extends ShaderType>(uniform: UniformNode<T>, value: UniformValue<T>): void;
  function setUniform<T extends ShaderType>(uniform: UniformArrayNode<T>, value: UniformValue<T>[]): void;
  function setUniform(slot: string, value: number | number[]): void;
  function setUniform(uniform: UniformOrSlot, value: unknown): void {
    uniforms[slotOf(uniform)] = value as number | number[];
  }

  function setAttribute<T extends ShaderType>(attribute: AttributeNode<T>, data: TypedArray): void;
  function setAttribute(slot: string, data: TypedArray): void;
  function setAttribute(attribute: AttributeOrSlot, data: TypedArray): void {
    attributes[slotOf(attribute)] = data;
  }

  return {
    attach(givenCanvas) {
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

    draw(drawOptions) {
      if (!canvas || !ctx2d) throw new Error("[RMSL] createWasm: attach() was never called");
      const ctx: WasmRasterContext = { attributes, uniforms, textures };
      const buffer = routine.draw(ctx, {
        count: drawOptions?.count,
        first: drawOptions?.first,
        width: canvas.width,
        height: canvas.height,
        clear: drawOptions?.clear,
        clearColor: drawOptions?.clearColor,
        clearDepth: drawOptions?.clearDepth,
      });
      ctx2d.putImageData(bufferToImageData(buffer, canvas.width, canvas.height), 0, 0);
    },

    destroy() {},
  };
}
