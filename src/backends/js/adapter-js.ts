import { Node, RoutineParam, ShaderType } from "../../core";
import {
  Adapter,
  AttributeOrSlot,
  AttributeOrStorage,
  DrawClearOptions,
  DrawCountOptions,
  hostInputGuard,
  HostArray,
  SetUniform,
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
import { compileJS, compileJSRasterizer, CompileJSRasterOptions, JsRasterContext } from "./rasterizer";
import { fragmentColour } from "../shared";
import { compileJSCompute, compileJSGrid, compileJSRoutine, CompileJSOptions } from "./js";

export interface CreateJsGridOptions {
  /** A fragCoord() program that returns a colour, evaluated once per canvas pixel by `draw()`. */
  draw: Node<ShaderType>;
  name?: string;
  params?: CompileJSOptions["params"];
  derivatives?: CompileJSOptions["derivatives"];
  reentrant?: CompileJSOptions["reentrant"];
  float?: CompileJSOptions["float"];
}

/**
 * Compiles a `fragCoord()` program with {@link compileJSGrid} and
 * wraps it in a {@link createCpuAdapter} — a plain CPU-callable evaluated
 * once per pixel with its grid, not a wgpu pipeline shape. See
 * {@link createJsCompute} for the `storage()`/`invocationIndex()` shape
 * and {@link createJs} for the vertex/fragment render shape — those each
 * got their own dedicated entry point rather than living as options here
 * for the same reason (see `src/backends/wasm/adapter-wasm.ts`'s own
 * split, which this mirrors).
 */
export function createJsGrid(options: CreateJsGridOptions): CpuAdapter {
  const draw = compileJSGrid(() => fragmentColour([options.draw])[0] as Node<"vec4">, {
    name: options.name ?? "draw",
    params: options.params ?? [],
    derivatives: options.derivatives,
    reentrant: options.reentrant,
    float: options.float,
  });

  return createCpuAdapter({ draw, guard: hostInputGuard(options.draw, true) });
}

export interface CreateJsRoutineOptions {
  name?: string;
  params?: readonly RoutineParam[];
  derivatives?: CompileJSOptions["derivatives"];
  reentrant?: CompileJSOptions["reentrant"];
  float?: CompileJSOptions["float"];
}

/**
 * Compiles a function of parameters and uniforms with {@link compileJSRoutine}
 * and wraps it in a {@link CpuRoutineAdapter}: `setUniform` refuses a value of
 * another shape than its uniform's type and keeps the rest, `setTexture` keeps
 * what the host gives it, and `run(params)` refuses a parameter of another
 * shape than its type and calls the routine with them.
 */
export function createJsRoutine<
  A extends ShaderType,
  W extends FloatWidth = "f64",
  const P extends readonly RoutineParam[] = readonly [],
>(fn: Node<A>, options: CreateJsRoutineOptions & { float?: W; params?: P } = {}): CpuRoutineAdapter<A, W, P> {
  return createCpuRoutineAdapter<A, W, P>(
    compileJSRoutine(() => fn, {
      name: options.name ?? "routine",
      params: [...(options.params ?? [])],
      derivatives: options.derivatives,
      reentrant: options.reentrant,
      float: options.float,
    }),
    options.params ?? ([] as readonly RoutineParam[] as P),
    hostInputGuard(fn, true),
  );
}

export interface CreateJsComputeOptions {
  name?: string;
  params?: CompileJSOptions["params"];
  derivatives?: CompileJSOptions["derivatives"];
  reentrant?: CompileJSOptions["reentrant"];
  float?: CompileJSOptions["float"];
}

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
export interface JsComputeAdapter {
  setUniform: SetUniform;
  setAttribute<T extends ShaderType>(
    attribute: AttributeOrStorage<T>,
    data: HostArray<T, Float32Array | Float64Array>,
  ): void;
  setAttribute(slot: string, data: TypedArray): void;
  compute(out?: AdapterResult, count?: number): AdapterResult | void;
}

/**
 * Compiles a `storage()`/`invocationIndex()` program with
 * {@link compileJSCompute} and wraps it in a {@link createCpuAdapter} —
 * the wgpu-compute-pipeline-shaped counterpart to {@link createJs}'s
 * render-pipeline shape, and the JS-side sibling of `createWasmCompute`.
 */
export function createJsCompute(
  compute: Node<ShaderType> | readonly Node<ShaderType>[],
  options: CreateJsComputeOptions = {},
): JsComputeAdapter {
  const computeRoutine = compileJSCompute(() => compute, {
    name: options.name ?? "compute",
    params: options.params ?? [],
    derivatives: options.derivatives,
    reentrant: options.reentrant,
    float: options.float,
  });

  return createCpuAdapter({ compute: computeRoutine, guard: hostInputGuard(compute, true), float: options.float });
}

/**
 * `draw()`'s own options — `count`/`first` from `DrawCountOptions`, same
 * as GL/WGSL: unset `count` defaults to the first attribute the program
 * reads, in the order the host passed them. `clear` and `clearDepth` default to `true`: the common case
 * for a single-material adapter is one `draw()` call, one whole frame —
 * mirroring how a WebGPU render pass declares `loadOp`/`depthLoadOp`
 * together, per pass, rather than clearing as a separate operation. Pass
 * `false` to composite several `draw()` calls into one frame instead
 * (several materials sharing a framebuffer/depth buffer); `compileJS`'s
 * own `JsRasterRoutine` is the lower-level primitive that composability
 * is built on.
 */
export interface JsDrawOptions extends DrawCountOptions, DrawClearOptions {
  clearDepth?: boolean;
}

/**
 * Compiles a vertex/fragment `Fn` pair with {@link compileJS} and wraps
 * the resulting `JsRasterRoutine` in the uniform `Adapter` interface — the
 * JS-side counterpart to `createWasm`. `setAttribute`/`setUniform` collect
 * draw state, `attach` opens a 2D canvas context, and `draw()` runs one
 * frame into it. See {@link JsDrawOptions} for `count`/`first`/`clear`/`clearDepth`.
 */
export interface JsAdapter extends Adapter<void, JsDrawOptions, Float32Array | Float64Array> {
  attach(canvas?: HTMLCanvasElement): void;
  draw(options?: JsDrawOptions): void;
}

export function createJs(
  vertexFn: (...args: any[]) => any,
  fragmentFn: (...args: any[]) => any,
  options: CompileJSRasterOptions = {},
): JsAdapter {
  const vertexRoots = vertexFn();
  const fragmentRoots = fragmentFn();
  const routine = compileJSRasterizer(
    "createJs",
    () => vertexRoots,
    () => fragmentRoots,
    options,
  );
  const guard = hostInputGuard([vertexRoots, fragmentRoots], true);

  const uniforms: Record<string, unknown> = {};
  const attributes: Record<string, TypedArray> = {};
  const textures: Record<string, CpuTextureData> = {};
  let canvas: HTMLCanvasElement | null = null;
  let ctx2d: CanvasRenderingContext2D | null = null;

  function setUniform(uniform: UniformOrSlot, value: unknown): void {
    guard.uniform(uniform, value);
    uniforms[slotOf(uniform)] = value;
  }

  function setAttribute<T extends ShaderType>(attribute: AttributeOrStorage<T>, data: TypedArray): void;
  function setAttribute(slot: string, data: TypedArray): void;
  function setAttribute(attribute: AttributeOrSlot, data: TypedArray): void {
    guard.array(attribute, data);
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
      if (!canvas || !ctx2d) throw new Error("[RMSL] createJs: attach() was never called");
      const ctx: JsRasterContext = { attributes, uniforms, textures };
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
