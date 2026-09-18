import { type Node, type ShaderType, type VariableNode } from "../rmsl";
import { compileJSRoutine, compileJSFn, type CpuShaderContext, type CpuTextureData } from "../js";
// How a texture asks to be read is the renderers' question too, and they
// already answer it without a device — so a shader tested here samples by the
// same reading, not by a second one written for the CPU.
import { samplerState, textureChannels } from "../scene/renderers/common";
import type { Texture } from "../scene/textures/Texture";

// === Values ===

/**
 * The JavaScript value a shader type carries on the CPU: scalars are numbers
 * (or booleans), vectors and matrices are flat arrays — matrices in the
 * column-major order the rest of the library uses.
 */
export type ShaderValue<A extends ShaderType> = A extends "float" | "int" | "uint"
  ? number
  : A extends "bool"
    ? boolean
    : A extends "bvec2" | "bvec3" | "bvec4"
      ? boolean[]
      : A extends `${string}sampler${string}`
        ? TextureData
        : A extends "void"
          ? never
          : number[];

/**
 * Texture data to sample from. A scene `DataTexture` fits as it stands: its
 * `image` is read when `data` is absent, and its three.js format, filtering
 * and wrapping constants are read as the CPU target's names for them.
 */
export type TextureData =
  | CpuTextureData
  | {
      /**
       * Pixels as a scene texture carries them, which is where a `DataTexture`
       * keeps its data. The sources only a browser could decode — an
       * `HTMLImageElement`, a canvas — are part of that type and are refused when
       * the texture is bound, since there is nothing here to read them with.
       */
      image: Texture["image"] | ArrayLike<number>;
      width: number;
      height: number;
      depth?: number;
      format?: number;
      magFilter?: number;
      minFilter?: number;
      wrapS?: number;
      wrapT?: number;
      wrapR?: number;
    };

/**
 * A value for one uniform, varying or attribute, given as the node itself —
 * `[uTint, [1, 0, 0]]`. The node knows which slot it is; nothing here needs the
 * generated `_rmsl_u0` names.
 */
export type ValueBinding = {
  [A in ShaderType]: readonly [VariableNode<A>, ShaderValue<A>];
}[ShaderType];

/** A texture for one sampler node. */
export type TextureBinding = {
  [A in Extract<ShaderType, `${string}sampler${string}`>]: readonly [VariableNode<A>, TextureData];
}[Extract<ShaderType, `${string}sampler${string}`>];

// === Inputs ===

/**
 * Values by name rather than by node. The names are a program's own — `uv`,
 * `materialColor`, the ones `fromProgram` and `fromPass` know — falling back to
 * a node's `.name` for a graph that has no program behind it.
 */
export type NamedValues = Record<string, unknown>;

/** What a shader reads when it is evaluated. */
export interface ShaderInputs {
  uniforms?: readonly ValueBinding[] | NamedValues;
  varyings?: readonly ValueBinding[] | NamedValues;
  attributes?: readonly ValueBinding[] | NamedValues;
  textures?: readonly TextureBinding[] | Record<string, TextureData>;
  /** The pixel `fragCoord()` reports, as `[x, y]`. */
  fragCoord?: readonly [number, number];
}

export interface RunnerOptions extends ShaderInputs {
  /** Which stage the graph is compiled as. Default `"fragment"`. */
  stage?: "vertex" | "fragment";
  /**
   * What `fwidth`/`dFdx`/`dFdy` do. A derivative is the difference between
   * neighbouring pixels, which a single CPU evaluation has no notion of, so
   * they are zero here by default rather than a compile error — an anti-aliased
   * edge reads as a hard one. Pass `"throw"` to be told instead.
   */
  derivatives?: "zero" | "throw";
  /** Give each call its own scratch variables. Default `false`. */
  reentrant?: boolean;
}

/** What a shader produced for one fragment. */
export interface EvaluationResult<A extends ShaderType = ShaderType> {
  /** The graph's own value, or `null` for a fragment that discarded. */
  value: ShaderValue<A> | null;
  /** Whether the fragment ran into `Discard()`. */
  discarded: boolean;
  /** Values written with `output()`, by slot name. */
  outputs: Record<string, unknown>;
  /**
   * Values written with `varying()` in a vertex stage, keyed the way they are
   * passed in: by the program's own name under `fromProgram`, by the node's
   * name otherwise.
   */
  varyings: Record<string, unknown>;
  /** Written with `builtinPosition()`. */
  position?: number[];
  /** Written with `builtinFragDepth()`. */
  fragDepth?: number;
}

/** A compiled shader, ready to be called once per fragment. */
export interface ShaderRunner<A extends ShaderType = ShaderType> {
  (inputs?: ShaderInputs): EvaluationResult<A>;
  /** The generated JavaScript, for when a result needs explaining. */
  readonly source: string;
}

const RUNNER = Symbol.for("rmsl.test.runner");

/**
 * Compile a shader graph once, to be evaluated many times.
 *
 * `evaluate` and `render` compile on every call, which is what you want for a
 * single assertion. Reach for this when the same graph is measured across many
 * inputs — a sweep over a uniform, a search for where a distance field crosses
 * zero — so the compile is not repeated per sample.
 *
 * The graph is given as a thunk, so a function taking arguments is tested by
 * calling it: `runner(() => myFn(float(2), vec2(1, 0)))`.
 */
export function runner<A extends ShaderType>(graph: () => Node<A>, options: RunnerOptions = {}): ShaderRunner<A> {
  return compileRunner(graph, options);
}

/**
 * The name a program gives each of its slots, per kind. Kept apart rather than
 * flattened because one program can call an attribute and a varying the same
 * thing — a material's `uv` is both.
 */
interface NameMaps {
  uniforms?: Map<string, string>;
  varyings?: Map<string, string>;
  attributes?: Map<string, string>;
  textures?: Map<string, string>;
}

/** `runner`, plus the names a program brings with it. */
function compileRunner<A extends ShaderType>(
  graph: () => Node<A>,
  options: RunnerOptions,
  names?: NameMaps,
): ShaderRunner<A> {
  const compileOptions = {
    name: "rmslTestShader",
    params: [],
    stage: options.stage ?? "fragment",
    derivatives: options.derivatives ?? "zero",
    reentrant: options.reentrant ?? false,
  };
  const callable = compileJSRoutine(graph, compileOptions);
  const source = compileJSFn(graph, compileOptions);

  // The program's names, the other way round. A slot goes in under the name the
  // program calls it, so that is the name a result comes back under, and the
  // name to say out loud when nothing bound it.
  const called: NameMaps = {
    uniforms: reverseNames(names?.uniforms),
    varyings: reverseNames(names?.varyings),
    attributes: reverseNames(names?.attributes),
    textures: reverseNames(names?.textures),
  };
  const reads = { stage: compileOptions.stage, named: names !== undefined, names: called };
  const run = (inputs: ShaderInputs = {}): EvaluationResult<A> =>
    readResult<A>(callable.invoke(mergeContext(options, inputs, names, reads)), called.varyings);
  return Object.defineProperties(run, {
    source: { value: source, enumerable: true },
    [RUNNER]: { value: true },
  }) as ShaderRunner<A>;
}

/**
 * Evaluate a shader graph for one fragment.
 *
 * ```typescript
 * const result = evaluate(() => tint(), {
 *   uniforms: [[uTint, [1, 0, 0]]],
 *   varyings: [[vUv, [0.5, 0.5]]],
 * });
 * ```
 */
export function evaluate<A extends ShaderType>(
  graph: (() => Node<A>) | ShaderRunner<A>,
  inputs: ShaderInputs & RunnerOptions = {},
): EvaluationResult<A> {
  return asRunner(graph, inputs)(inputs);
}

// === Rendering a grid ===

/** Which fragment of a grid render is being evaluated. */
export interface Fragment {
  /** Column, `0` to `width - 1`. */
  x: number;
  /** Row, `0` to `height - 1`, counting up from the bottom. */
  y: number;
  /** `x` and `y` mapped to the middle of the pixel, in `0..1`. */
  u: number;
  v: number;
  /** What `fragCoord()` reports here: the pixel centre. */
  fragCoord: [number, number];
}

export interface RenderOptions extends ShaderInputs, RunnerOptions {
  width: number;
  height: number;
  /**
   * Inputs that differ per fragment, merged over the shared ones — the usual
   * home for a `uv` varying:
   *
   * ```typescript
   * inputs: ({ u, v }) => ({ varyings: [[vUv, [u, v]]] })
   * ```
   */
  inputs?: (fragment: Fragment) => ShaderInputs | void;
}

/** A grid of evaluated fragments. */
export interface RenderedImage {
  width: number;
  height: number;
  /**
   * Four channels per fragment, row-major from `y = 0` — the bottom row, as
   * `fragCoord` counts. A fragment that discarded is four zeroes.
   */
  pixels: Float32Array;
  /** The four channels at one fragment. */
  at(x: number, y: number): [number, number, number, number];
  /** Whether the fragment at `x, y` discarded. */
  isDiscarded(x: number, y: number): boolean;
  /** The result the fragment at `x, y` produced, in full. */
  resultAt(x: number, y: number): EvaluationResult;
  /**
   * The image as clamped 8-bit RGBA, for a snapshot or an image file. Rows run
   * bottom-up like `pixels`; pass `{ flipY: true }` for the top-down order most
   * image formats want.
   */
  toUint8(options?: { flipY?: boolean }): Uint8ClampedArray;
  /**
   * The image as text, one character per fragment by brightness — a shader
   * snapshot that reads in a terminal diff. The top line is the top row of the
   * image; pass `{ flipY: true }` to print from `y = 0` instead.
   *
   * Brightness is the fragment composited over black, so a shape that lives in
   * alpha — a particle, a billboard, anything the blender fades out — reads as
   * the shape it is rather than a flat block. Pass `{ alpha: false }` to weigh
   * the colour channels alone.
   */
  toAscii(options?: { ramp?: string; flipY?: boolean; alpha?: boolean }): string;
}

const DEFAULT_RAMP = " .:-=+*#%@";

/**
 * Evaluate a shader across a grid of fragments.
 *
 * Nothing is rasterized: every fragment in the rectangle is evaluated, in
 * reading order. That is what makes an assertion about a *shape* possible —
 * where an edge falls, whether a ramp is monotonic, which corner is brightest —
 * without a canvas to read pixels back from.
 */
export function render<A extends ShaderType>(
  graph: (() => Node<A>) | ShaderRunner<A>,
  options: RenderOptions,
): RenderedImage {
  const { width, height } = options;
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
    throw new Error(`[RMSL/test] render needs a whole width and height of at least 1, got ${width}x${height}`);
  }
  const run = asRunner(graph, options);
  const pixels = new Float32Array(width * height * 4);
  const results: EvaluationResult<A>[] = new Array(width * height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const fragCoord: [number, number] = [x + 0.5, y + 0.5];
      const perFragment =
        options.inputs?.({
          x,
          y,
          u: fragCoord[0] / width,
          v: fragCoord[1] / height,
          fragCoord,
        }) ?? {};
      const result = run({ fragCoord, ...perFragment });
      const index = y * width + x;
      results[index] = result;
      if (!result.discarded) pixels.set(toRGBA(result), index * 4);
    }
  }
  return image(width, height, pixels, results);
}

function image(
  width: number,
  height: number,
  pixels: Float32Array,
  results: EvaluationResult<ShaderType>[],
): RenderedImage {
  const indexOf = (x: number, y: number): number => {
    if (!Number.isInteger(x) || !Number.isInteger(y) || x < 0 || y < 0 || x >= width || y >= height) {
      throw new Error(`[RMSL/test] (${x}, ${y}) is outside the ${width}x${height} image`);
    }
    return y * width + x;
  };
  return {
    width,
    height,
    pixels,
    at(x, y) {
      const base = indexOf(x, y) * 4;
      return [pixels[base], pixels[base + 1], pixels[base + 2], pixels[base + 3]];
    },
    isDiscarded(x, y) {
      return results[indexOf(x, y)].discarded;
    },
    resultAt(x, y) {
      return results[indexOf(x, y)];
    },
    toUint8({ flipY = false } = {}) {
      const out = new Uint8ClampedArray(width * height * 4);
      for (let y = 0; y < height; y++) {
        const row = flipY ? height - 1 - y : y;
        for (let i = 0; i < width * 4; i++) {
          out[y * width * 4 + i] = pixels[row * width * 4 + i] * 255;
        }
      }
      return out;
    },
    toAscii({ ramp = DEFAULT_RAMP, flipY = false, alpha = true } = {}) {
      const lines: string[] = [];
      for (let y = 0; y < height; y++) {
        const row = flipY ? y : height - 1 - y;
        let line = "";
        for (let x = 0; x < width; x++) {
          const base = (row * width + x) * 4;
          // Rec. 709 luminance, the weighting a monitor's channels carry, over
          // a black background — which is what makes a shape drawn in alpha,
          // and nothing else, visible here at all.
          const luma = 0.2126 * pixels[base] + 0.7152 * pixels[base + 1] + 0.0722 * pixels[base + 2];
          const coverage = alpha ? Math.min(1, Math.max(0, pixels[base + 3])) : 1;
          const step = Math.round(Math.min(1, Math.max(0, luma)) * coverage * (ramp.length - 1));
          line += ramp[step];
        }
        lines.push(line);
      }
      return lines.join("\n");
    },
  };
}

// === Programs and passes ===

/**
 * A compiled material, in the shape `NodeMaterial.build()` returns. Described
 * structurally rather than imported, so this module never pulls the scene graph
 * into a test that only wanted a shader.
 */
export interface ProgramLike {
  fragmentRoot: Node<"vec4">;
  vertexRoot?: Node<"vec4">;
  uniforms?: readonly ProgramUniform[];
  varyings?: readonly ProgramSlot[];
  attributes?: readonly ProgramSlot[];
  samplers?: readonly ProgramSampler[];
}

export interface ProgramSlot {
  /** The name the program calls it — `uv`, `materialColor`. */
  name: string;
  node: { name: string };
}

export interface ProgramUniform extends ProgramSlot {
  scope?: string;
  /** Where a material-scope uniform gets its value. */
  value?: (context: any) => number | number[];
}

export interface ProgramSampler extends ProgramSlot {
  type?: string;
  /** The texture the material points at, if it points at one. */
  texture?: () => unknown;
}

/** One fullscreen pass, in the shape `@random-mesh/rmsl/effects` describes. */
export interface PassLike {
  color: Node<"vec4">;
  /** The samplers the pass reads, by the name the caller gave them. */
  inputs?: Record<string, { name: string }>;
}

export interface ProgramOptions extends RunnerOptions {
  /**
   * What a material-scope uniform's own value function is called with. A
   * material that reads the camera or the mesh to compute a uniform needs them;
   * one that does not, does not.
   */
  context?: unknown;
  /** What the renderer-scope `resolution` uniform holds. Default `[1, 1]`. */
  resolution?: readonly [number, number];
}

/** A shader compiled from a program, with the names it was built under. */
export interface ProgramRunner extends ShaderRunner<"vec4"> {
  /**
   * Uniforms — and samplers — nothing filled in: neither the program, nor a
   * built-in default, nor the options. These are the inputs a program is
   * expected to bring with it, so a name here is usually a missing piece of the
   * material rather than something the caller forgot; an attribute or a varying
   * belongs to the call instead, and is reported when the shader reads it.
   *
   * A shader that reads one of these throws rather than shading with
   * `undefined`, so this is the list to check before calling.
   */
  readonly unbound: string[];
}

/**
 * Run a built material — or anything else shaped like a compiled program — on
 * the CPU, addressed by the names the program uses rather than by node.
 *
 * ```typescript
 * const shade = fromProgram(material.build(scene, {}));
 * shade({ varyings: { uv: [0.5, 0.5], normalWorld: [0, 1, 0] } });
 * ```
 *
 * Material-scope uniforms fill themselves in from the program — a colour, a
 * roughness, the light set — and a `DataTexture` bound to a sampler is read as
 * it stands. The matrices a renderer would supply default to the identity, so a
 * material that only shades a surface needs nothing else; pass your own values
 * by name where they matter.
 */
export function fromProgram(program: ProgramLike, options: ProgramOptions = {}): ProgramRunner {
  const stage = options.stage ?? "fragment";
  const root = stage === "vertex" ? program.vertexRoot : program.fragmentRoot;
  if (!root) throw new Error(`[RMSL/test] the program has no ${stage} root to run`);

  const names: NameMaps = {
    uniforms: slotNames(program.uniforms),
    varyings: slotNames(program.varyings),
    attributes: slotNames(program.attributes),
    textures: slotNames(program.samplers),
  };

  const given = slots(names.uniforms, options.uniforms);
  const uniforms: Record<string, unknown> = {};
  const unbound: string[] = [];
  for (const binding of program.uniforms ?? []) {
    if (binding.node.name in given) continue;
    const value = programValue(binding, options);
    if (value === undefined) unbound.push(binding.name);
    else uniforms[binding.node.name] = value;
  }

  const passed = textureSlots(names.textures, options.textures);
  const textures: Record<string, TextureData> = {};
  for (const sampler of program.samplers ?? []) {
    if (sampler.node.name in passed) continue;
    const texture = toTextureData(sampler.texture?.(), sampler.type);
    // A sampler pointing at nothing, or at an image only a browser could read.
    if (texture) textures[sampler.node.name] = texture;
    else unbound.push(sampler.name);
  }

  const run = compileRunner<"vec4">(
    () => root,
    {
      ...options,
      uniforms: { ...uniforms, ...given },
      textures: { ...textures, ...passed },
    },
    names,
  );

  return Object.defineProperties(run, {
    unbound: { value: unbound, enumerable: true },
  }) as ProgramRunner;
}

/**
 * Run one fullscreen pass of an effect on the CPU, its input textures given by
 * the names the pass calls them.
 *
 * ```typescript
 * const run = fromPass(pass, { textures: { source: { data, width: 4, height: 4 } } });
 * render(run, { width: 4, height: 4 });
 * ```
 *
 * A pass that reads `uv()` also holds a screen-size uniform it never named;
 * bind it with `uniformsIn(pass.color, "vec2")`.
 */
export function fromPass(pass: PassLike, options: RunnerOptions = {}): ShaderRunner<"vec4"> {
  const textures = new Map<string, string>();
  for (const [name, node] of Object.entries(pass.inputs ?? {})) textures.set(name, node.name);
  return compileRunner<"vec4">(() => pass.color, options, { textures });
}

/** Which slot each of a program's names points at. */
function slotNames(bindings: readonly ProgramSlot[] | undefined): Map<string, string> {
  return new Map((bindings ?? []).map((binding) => [binding.name, binding.node.name]));
}

/** What a program can say a uniform holds, or a sensible stand-in. */
function programValue(binding: ProgramUniform, options: ProgramOptions): unknown {
  try {
    const own = binding.value?.(options.context);
    if (own !== undefined && !(Array.isArray(own) && own.length === 0)) return own;
  } catch {
    // A value function that reaches for a camera or a mesh it was not given.
    // The name falls through to a default, or to whatever the caller passes.
  }
  if (binding.name === "resolution") return [...(options.resolution ?? [1, 1])];
  return RENDERER_DEFAULTS[binding.name];
}

const IDENTITY_MAT4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const IDENTITY_MAT3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

/**
 * What the matrices a renderer would compute hold when no renderer is running.
 * The identity puts the surface in world space at the origin, which is the
 * frame a shading term is easiest to reason about.
 */
const RENDERER_DEFAULTS: Record<string, unknown> = {
  modelMatrix: IDENTITY_MAT4,
  viewMatrix: IDENTITY_MAT4,
  projectionMatrix: IDENTITY_MAT4,
  normalMatrix: IDENTITY_MAT3,
  cameraPosition: [0, 0, 0],
};

/**
 * A texture the CPU target can sample, from whatever it was handed. A scene
 * texture fits as it stands — its pixels sit under `image` rather than `data`,
 * and its format, filtering and wrapping are the constants the target samples
 * by — so nothing here restates what the texture already says.
 *
 * Null when there are no pixels to read — an `HTMLImageElement` or a canvas is
 * nothing without a browser, and a sampler bound to one is left unbound rather
 * than bound to something empty.
 */
function toTextureData(texture: unknown, samplerType = "sampler2D"): CpuTextureData | null {
  if (typeof texture !== "object" || texture === null) return null;
  const candidate = texture as Record<string, unknown>;
  const pixels = candidate.data ?? candidate.image;
  if (!ArrayBuffer.isView(pixels) && !Array.isArray(pixels)) return null;
  if (typeof candidate.width !== "number" || typeof candidate.height !== "number") return null;
  const image = { data: pixels as ArrayLike<number>, width: candidate.width, height: candidate.height };
  if (!statesTextureConstants(candidate)) {
    return { ...(candidate as unknown as CpuTextureData), ...image };
  }
  const scene = texture as Texture;
  return {
    ...image,
    depth: candidate.depth as number | undefined,
    channels: textureChannels(scene),
    ...samplerState(scene, samplerType),
  };
}

/**
 * Whether a texture states how it is read in three.js constants — what a scene
 * texture carries — rather than in the words the CPU target samples by.
 */
function statesTextureConstants(texture: Record<string, unknown>): boolean {
  return (
    texture.isTexture === true ||
    typeof texture.format === "number" ||
    typeof texture.magFilter === "number" ||
    typeof texture.wrapS === "number"
  );
}

/**
 * The uniform nodes a graph reads, in the order they are reached, each name
 * appearing once. With a shader type, only the uniforms of that type — which
 * is the form that can be bound, since a binding has to know what it holds.
 *
 * Most uniforms are held by the test that made them, and binding is
 * `[uTint, [1, 0, 0]]`. Some are not: `uv()` mints a screen-size uniform inside
 * the graph, and a caller who never saw that node has no other way to reach it:
 *
 * ```typescript
 * const [resolution] = uniformsIn(graph, "vec2");
 * render(() => graph, { width: 8, height: 8, uniforms: [[resolution, [8, 8]]] });
 * ```
 */
export function uniformsIn<A extends ShaderType>(graph: Node<ShaderType>, type: A): VariableNode<A>[];
export function uniformsIn(graph: Node<ShaderType>): VariableNode<ShaderType>[];
export function uniformsIn(graph: Node<ShaderType>, type?: ShaderType): VariableNode<ShaderType>[] {
  const found: VariableNode<ShaderType>[] = [];
  const named = new Set<string>();
  const seen = new Set<unknown>();
  const walk = (node: unknown): void => {
    if (typeof node !== "object" || node === null || seen.has(node)) return;
    seen.add(node);
    const current = node as { type?: string; name?: string; params?: unknown[]; value?: unknown };
    if ((current.type === "uniform" || current.type === "uniformArray") && current.name && !named.has(current.name)) {
      named.add(current.name);
      found.push(node as VariableNode<ShaderType>);
    }
    for (const param of current.params ?? []) walk(param);
    // A uniform array element carries its array in `value`, not in `params`.
    if (current.value && typeof current.value === "object") walk(current.value);
  };
  walk(graph);
  return type === undefined ? found : found.filter((node) => node._t === type);
}

// === Comparing ===

/**
 * How far apart two results of the same magnitude may be.
 *
 * A shader's arithmetic is not reproduced exactly by the arithmetic a test
 * writes out by hand — an operation may be evaluated in another order, or
 * through another identity — and the gap grows with the numbers involved. The
 * allowance scales with magnitude, with a floor for values near zero where a
 * relative gap collapses to nothing.
 */
export function tolerance(magnitude: number): number {
  return Math.max(1e-6, Math.abs(magnitude) * 1e-6);
}

/** Whether two shader values agree, within `options.tolerance`. */
export function closeTo(actual: unknown, expected: unknown, options: { tolerance?: number } = {}): boolean {
  return difference(actual, expected, options.tolerance) === null;
}

/**
 * A predicate for one expected value, for a matcher that takes one:
 * `expect(result.value).toSatisfy(approx([1, 0, 0, 1]))`.
 */
export function approx(expected: unknown, options: { tolerance?: number } = {}): (actual: unknown) => boolean {
  return (actual: unknown) => closeTo(actual, expected, options);
}

/**
 * Throw unless two shader values agree, with a message naming the component
 * that differs. Framework-agnostic: it needs no matcher and no test runner.
 */
export function assertClose(
  actual: unknown,
  expected: unknown,
  options: { tolerance?: number; message?: string } = {},
): void {
  const reason = difference(actual, expected, options.tolerance);
  if (reason === null) return;
  const prefix = options.message ? `${options.message}: ` : "";
  throw new Error(
    `[RMSL/test] ${prefix}${reason}` + `\n  expected: ${format(expected)}` + `\n  actual:   ${format(actual)}`,
  );
}

/** What differs between two values, or `null` when nothing does. */
function difference(actual: unknown, expected: unknown, allowed?: number): string | null {
  if (typeof expected === "boolean" || typeof actual === "boolean") {
    return actual === expected ? null : `expected ${format(expected)}, got ${format(actual)}`;
  }
  if (Array.isArray(expected)) {
    if (!Array.isArray(actual)) return `expected an array of ${expected.length}, got ${format(actual)}`;
    if (actual.length !== expected.length) {
      return `expected ${expected.length} components, got ${actual.length}`;
    }
    for (let i = 0; i < expected.length; i++) {
      const reason = difference(actual[i], expected[i], allowed);
      if (reason !== null) return `component ${i}: ${reason}`;
    }
    return null;
  }
  if (typeof expected !== "number" || typeof actual !== "number") {
    return actual === expected ? null : `expected ${format(expected)}, got ${format(actual)}`;
  }
  if (Number.isNaN(expected) && Number.isNaN(actual)) return null;
  const gap = Math.abs(actual - expected);
  const limit = allowed ?? tolerance(expected);
  return gap <= limit ? null : `expected ${expected}, got ${actual} (off by ${gap}, allowed ${limit})`;
}

function format(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(format).join(", ")}]`;
  return String(value);
}

// === Plumbing ===

function asRunner<A extends ShaderType>(
  graph: (() => Node<A>) | ShaderRunner<A>,
  options: RunnerOptions,
): ShaderRunner<A> {
  return (graph as unknown as Record<symbol, unknown>)[RUNNER]
    ? (graph as ShaderRunner<A>)
    : runner(graph as () => Node<A>, options);
}

/** What a shader is, for the sake of saying which input it wanted. */
interface ReadContext {
  stage: string;
  /** Whether a program's names are in play, and so whether `unbound` exists. */
  named: boolean;
  /** The name to report each slot under. */
  names: NameMaps;
}

/** Slot-keyed context for the compiled callable, defaults under the call's own. */
function mergeContext(
  defaults: ShaderInputs,
  inputs: ShaderInputs,
  names: NameMaps | undefined,
  reads: ReadContext,
): CpuShaderContext {
  return {
    uniforms: bound("uniform", "uniforms", slots(names?.uniforms, defaults.uniforms, inputs.uniforms), reads),
    varyings: bound("varying", "varyings", slots(names?.varyings, defaults.varyings, inputs.varyings), reads),
    attributes: bound(
      "attribute",
      "attributes",
      slots(names?.attributes, defaults.attributes, inputs.attributes),
      reads,
    ),
    textures: bound("texture", "textures", textureSlots(names?.textures, defaults.textures, inputs.textures), reads),
    fragCoord: (inputs.fragCoord ?? defaults.fragCoord) as [number, number] | undefined,
  };
}

/**
 * The bound values, with a missing one reported where the shader reads it.
 *
 * Nothing here can know in advance which slots a shader reads: a read can sit
 * inside an `If` whose branch this fragment does not take, so a list drawn up
 * before the call would name inputs that are never wanted. Reporting at the
 * read itself names exactly the input this evaluation asked for — and it lands
 * on the missing value rather than three frames later on the `NaN` or the
 * `undefined[0]` it turns into.
 */
function bound<A extends object>(kind: string, field: keyof ShaderInputs, values: A, reads: ReadContext): A {
  return new Proxy(values, {
    get(target, key) {
      // `in`, not `undefined`: an inherited `toString` is how a debugger or a
      // test reporter looks at this object, and it is not a missing input.
      if (typeof key === "string" && !(key in target)) {
        const name = reads.names[field as keyof NameMaps]?.get(key);
        // Without a program there is no name but the generated slot, and a
        // generated name is no use to bind by: point at the node instead.
        const wanted = name
          ? `reads the ${kind} "${name}", which nothing bound. ` + `Pass it under \`${field}\`.`
          : `reads a ${kind} nothing bound, in the slot the compiler generated ` +
            `as "${key}". Pass its node under \`${field}\`, as [node, value].`;
        const listed = reads.named && field === "uniforms" ? " It is listed in the runner's `unbound`." : "";
        throw new Error(`[RMSL/test] the ${reads.stage} stage ${wanted}${listed}`);
      }
      return (target as Record<string, unknown>)[key as string];
    },
  });
}

function slots(
  names: Map<string, string> | undefined,
  ...given: (readonly ValueBinding[] | NamedValues | undefined)[]
): Record<string, unknown> {
  const record: Record<string, unknown> = {};
  for (const values of given) {
    if (values === undefined) continue;
    if (Array.isArray(values)) {
      for (const [node, value] of values as readonly ValueBinding[]) record[node.name] = value;
    } else {
      for (const [name, value] of Object.entries(values as NamedValues)) {
        record[names?.get(name) ?? name] = value;
      }
    }
  }
  return record;
}

function textureSlots(
  names: Map<string, string> | undefined,
  ...given: (readonly TextureBinding[] | Record<string, TextureData> | undefined)[]
): Record<string, CpuTextureData> {
  const record: Record<string, CpuTextureData> = {};
  for (const values of given) {
    if (values === undefined) continue;
    const pairs: [string, TextureData][] = Array.isArray(values)
      ? (values as readonly TextureBinding[]).map(([node, texture]) => [node.name, texture])
      : Object.entries(values as Record<string, TextureData>);
    for (const [name, texture] of pairs) {
      const data = toTextureData(texture);
      if (data) record[names?.get(name) ?? name] = data;
    }
  }
  return record;
}

/** A program's names by the slot each points at, for reading a result back. */
function reverseNames(names: Map<string, string> | undefined): Map<string, string> | undefined {
  if (!names) return undefined;
  const back = new Map<string, string>();
  // First name wins: a slot two names point at is read under the first one.
  for (const [name, slot] of names) if (!back.has(slot)) back.set(slot, name);
  return back;
}

/** Values re-keyed from generated slot names to the program's own names. */
function named(
  values: Record<string, unknown> | undefined,
  names: Map<string, string> | undefined,
): Record<string, unknown> {
  if (!values) return {};
  if (!names) return values;
  const record: Record<string, unknown> = {};
  for (const [slot, value] of Object.entries(values)) record[names.get(slot) ?? slot] = value;
  return record;
}

/**
 * The compiled callable's return value in one shape. It returns the graph's
 * value directly unless the program writes an output, a position or a fragment
 * depth, and `null` for a discarded fragment.
 */
function readResult<A extends ShaderType>(raw: unknown, varyingNames?: Map<string, string>): EvaluationResult<A> {
  if (raw === null) {
    return { value: null, discarded: true, outputs: {}, varyings: {} };
  }
  if (isProgramResult(raw)) {
    return {
      value: (raw.value ?? null) as ShaderValue<A> | null,
      discarded: false,
      outputs: raw.outputs ?? {},
      varyings: named(raw.varyings, varyingNames),
      position: raw.position,
      fragDepth: raw.fragDepth,
    };
  }
  return { value: raw as ShaderValue<A>, discarded: false, outputs: {}, varyings: {} };
}

interface ProgramResult {
  value?: unknown;
  outputs?: Record<string, unknown>;
  varyings?: Record<string, unknown>;
  position?: number[];
  fragDepth?: number;
}

function isProgramResult(raw: unknown): raw is ProgramResult {
  return typeof raw === "object" && raw !== null && !Array.isArray(raw);
}

/**
 * A fragment's four channels. A scalar lights all three colour channels, the
 * way a greyscale value reads; a shorter vector leaves the rest at zero, with
 * an opaque alpha, as a shader writing `vec4(rgb, 1)` would.
 */
function toRGBA(result: EvaluationResult<ShaderType>): [number, number, number, number] {
  const value = result.value ?? onlyOutput(result.outputs);
  if (typeof value === "number") return [value, value, value, 1];
  if (typeof value === "boolean") return value ? [1, 1, 1, 1] : [0, 0, 0, 1];
  if (Array.isArray(value)) {
    const channel = (i: number): number => {
      const component = value[i];
      return typeof component === "boolean" ? (component ? 1 : 0) : ((component ?? 0) as number);
    };
    return [channel(0), channel(1), channel(2), value.length > 3 ? channel(3) : 1];
  }
  return [0, 0, 0, 1];
}

/** The written output, when a program wrote exactly one and returned nothing. */
function onlyOutput(outputs: Record<string, unknown>): unknown {
  const written = Object.values(outputs);
  return written.length === 1 ? written[0] : null;
}
