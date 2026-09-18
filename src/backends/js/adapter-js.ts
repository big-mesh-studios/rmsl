import { AttributeNode, Node, ShaderType, UniformArrayNode, UniformNode, UniformValue } from "../../core";
import { Adapter, AttributeOrSlot, slotOf, TypedArray, UniformOrSlot } from "../adapter";
import { bufferToImageData, CpuAdapter, createCpuAdapter } from "../adapter-cpu";
import { compileJS, CompileJSRasterOptions, JsRasterContext } from "./rasterizer";
import { compileJSRoutine, CompileJSOptions } from "./js";

export interface CreateJsRoutineOptions {
  /** A storage()/invocationIndex() program to run as `compute()`. */
  compute?: Node<ShaderType> | readonly Node<ShaderType>[];
  /** A fragCoord() program, evaluated once per pixel by `draw()`'s `.batch()` call. */
  batch?: Node<ShaderType> | readonly Node<ShaderType>[];
  computeName?: string;
  batchName?: string;
  params?: CompileJSOptions["params"];
  derivatives?: CompileJSOptions["derivatives"];
  reentrant?: CompileJSOptions["reentrant"];
}

/** Compiles `compute`/`batch` with {@link compileJSRoutine} and wraps them in a {@link createCpuAdapter}. */
export function createJsRoutine(options: CreateJsRoutineOptions): CpuAdapter {
  if (!options.compute && !options.batch) {
    throw new Error("[RMSL] createJsRoutine needs a `compute` program, a `batch` program, or both");
  }

  const compute = options.compute
    ? compileJSRoutine(() => options.compute!, {
        name: options.computeName ?? "compute",
        params: options.params ?? [],
        derivatives: options.derivatives,
        reentrant: options.reentrant,
      })
    : undefined;

  const batch = options.batch
    ? compileJSRoutine(() => options.batch!, {
        name: options.batchName ?? "batch",
        params: options.params ?? [],
        stage: "fragment",
        derivatives: options.derivatives,
        reentrant: options.reentrant,
      })
    : undefined;

  return createCpuAdapter({ compute, batch });
}

/** `draw()`'s vertex count — a real rasterizer draw call, unlike `CreateJsRoutineOptions`'s
 * per-pixel `batch` program, actually runs over geometry, so it needs one. */
export interface JsDrawOptions {
  /** Vertex count for this draw — a non-indexed triangle list, so a multiple of 3. */
  vertexCount: number;
}

/**
 * Compiles a vertex/fragment `Fn` pair with {@link compileJS} and wraps
 * the resulting `JsRasterRoutine` in the uniform `Adapter` interface — the
 * JS-side counterpart to `createWasm`. `setAttribute`/`setUniform` collect
 * draw state, `attach` opens a 2D canvas context, and
 * `draw({ vertexCount })` auto-clears both depth and the output buffer
 * (one call, one frame) before rendering into it.
 *
 * A caller composing several materials into one shared framebuffer/depth
 * buffer needs `compileJS`'s own `JsRasterRoutine` directly, where both
 * clears are explicit instead (`clearDepth()`, and `draw()`'s own `clear`
 * argument).
 */
export function createJs(
  vertexFn: (...args: any[]) => any,
  fragmentFn: (...args: any[]) => any,
  options: CompileJSRasterOptions,
): Adapter<void, JsDrawOptions> {
  const routine = compileJS(vertexFn, fragmentFn, options);

  const uniforms: Record<string, unknown> = {};
  const attributes: Record<string, TypedArray> = {};
  let canvas: HTMLCanvasElement | null = null;
  let ctx2d: CanvasRenderingContext2D | null = null;

  function setUniform<T extends ShaderType>(uniform: UniformNode<T>, value: UniformValue<T>): void;
  function setUniform<T extends ShaderType>(uniform: UniformArrayNode<T>, value: UniformValue<T>[]): void;
  function setUniform(slot: string, value: number | number[]): void;
  function setUniform(uniform: UniformOrSlot, value: unknown): void {
    uniforms[slotOf(uniform)] = value;
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

    draw(drawOptions) {
      if (!canvas || !ctx2d) throw new Error("[RMSL] createJs: attach() was never called");
      if (!drawOptions) throw new Error("[RMSL] createJs: draw() needs a { vertexCount }");
      routine.clearDepth();
      const ctx: JsRasterContext = { attributes, uniforms };
      const buffer = routine.draw(ctx, drawOptions.vertexCount, canvas.width, canvas.height, undefined, true);
      ctx2d.putImageData(bufferToImageData(buffer, canvas.width, canvas.height), 0, 0);
    },

    destroy() {},
  };
}
