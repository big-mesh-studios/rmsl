import type { BackendOrigin } from "../../errors";
import { Node, ShaderType, componentCountOf, componentKindOf, type ScalarKind } from "../../core";
import { DrawClearOptions, DrawCountOptions, TRANSPARENT_BLACK, TypedArray, missingAttributeError } from "../adapter";
import { CpuDrawBuffer, CpuShaderContext, isResultObject, vertexPosition } from "../cpu";
import { linkStages } from "../shared";
import { createDrawAssembly, CpuIndexArray, cullSign, DrawAssemblyOptions, DrawAttributes } from "../assemble";
import { attributeKindOf } from "../vertex-format";
import { compileJSProgram, CompileJSOptions } from "./js";
import { depthAt, liesInDepthRange, lineSign, makeLines, pointSign, prepareLines, prepareRow } from "./lines";

/**
 * How many times the rounding of a cut a vertex's components may be and still
 * be taken for 0: 16 units in the last place, `2 ** -48`, as `rasterizer.wat` takes it.
 */
const CUT_ROUNDING = 2 ** -48;

/**
 * The least w of a vertex of the polygon a clip leaves whose projection bounds the pixels to test:
 * a smaller one, so near the eye, could lie anywhere on the target by the rounding of its cut.
 */
const BOUNDED_W = 2 ** -24;

/**
 * Options shared by both stages of a {@link compileJS} pair — the same
 * fields `compileJSRoutine` itself takes, minus `name`/`stage`/`params`,
 * which `compileJS` fixes itself.
 */
export interface CompileJSRasterOptions {
  derivatives?: CompileJSOptions["derivatives"];
  reentrant?: CompileJSOptions["reentrant"];
  float?: CompileJSOptions["float"];
}

/**
 * One draw call's inputs — mirrors `WasmRasterContext`, minus `textures`'
 * WASM-specific shape (`CpuShaderContext["textures"]` here instead).
 */
export interface JsRasterContext {
  attributes: Record<string, TypedArray>;
  /** The indices the draw takes its vertices by, or none to take them in order. */
  index?: CpuIndexArray;
  uniforms?: Record<string, unknown>;
  textures?: CpuShaderContext["textures"];
}

/**
 * Same as `WasmRasterDrawOptions`: `clear` and `clearDepth` default to
 * `true`, so each `draw()` starts from a cleared colour and depth buffer,
 * as a three.js render does. Pass `false` to compose several draws onto
 * both buffers, occlusion included — mirroring how a WebGPU render pass
 * declares `loadOp`/`depthLoadOp` together, per pass, rather than clearing
 * as a separate operation.
 */
export interface JsRasterDrawOptions extends DrawCountOptions, DrawClearOptions, DrawAssemblyOptions {
  width: number;
  height: number;
  out?: CpuDrawBuffer;
  clearDepth?: boolean;
}

/**
 * The JS-side counterpart to `WasmRasterRoutine` — same shape, same
 * semantics (clipping to the clip volume, a persistent LEQUAL depth buffer, both
 * clears declared per `draw()` call via {@link JsRasterDrawOptions}), a
 * plain-JS implementation of the same algorithm `rasterizer.wat` compiles
 * to WASM bytecode. Always produces `vec4` output, matching
 * `compileWasm`'s own fragment-result requirement.
 */
export interface JsRasterRoutine {
  draw(ctx: JsRasterContext, options: JsRasterDrawOptions): CpuDrawBuffer;
  /** The type of each attribute the vertex stage reads, by slot. */
  readonly attributeTypes: Readonly<Record<string, string>>;
}

/**
 * A vertex as the rasterizer keeps it: its clip-space position, and each
 * varying in the order of the draw's varying slots, a scalar as one component.
 */
interface ClipVertex {
  position: Float64Array;
  varyings: Float64Array[];
}

/** The target of a rasterizer between draws. */
const NO_TARGET = new Float64Array(0);

/** The inputs a rasterizer's contexts hold between draws: none. */
const NO_INPUTS: Pick<JsRasterContext, "uniforms" | "textures"> = Object.freeze({});

/**
 * Whether an edge running `(dx, dy)`, in a triangle wound so that its inside
 * lies where every edge function is positive, owns the pixel centres on it:
 * a left edge, running up the screen, or a top edge, running right, by
 * WebGPU's top-left rule. The two triangles sharing an edge run it in opposite
 * directions, so exactly one owns it.
 */
function ownsEdge(dx: number, dy: number): boolean {
  return dy < 0 || (dy === 0 && dx > 0);
}

/**
 * A host's value as an attribute component of `kind` reads it: an integer one
 * as a typed array of 32-bit integers of its type stores it, as the WASM rasterizer writes it.
 */
function attributeComponent(kind: ScalarKind, value: number): number {
  return kind === "int" ? value | 0 : kind === "uint" ? value >>> 0 : value;
}

/**
 * Which of the reads in {@link compileJSRasterizer}'s loop over vertices reads
 * `array`: one for each kind of typed array, and one for any other array. Each
 * read is a place of its own in the code, so each sees one kind of array and
 * V8 keeps it from boxing what it reads, however many kinds the host passes.
 */
function readerOf(array: ArrayLike<number>): number {
  if (array instanceof Float64Array) return 0;
  if (array instanceof Float32Array) return 1;
  if (array instanceof Int32Array) return 2;
  if (array instanceof Uint32Array) return 3;
  if (array instanceof Int16Array) return 4;
  if (array instanceof Uint16Array) return 5;
  if (array instanceof Int8Array) return 6;
  if (array instanceof Uint8Array) return 7;
  return 8;
}

/** A vertex with room for a position and varyings of these widths. */
function makeVertex(widths: readonly number[]): ClipVertex {
  return { position: new Float64Array(4), varyings: widths.map((w) => new Float64Array(w)) };
}

/** Whether a position is `vec4(0)`, each component 0 or -0. */
function isOrigin(p: Float64Array): boolean {
  return p[0] === 0 && p[1] === 0 && p[2] === 0 && p[3] === 0;
}

/** Whether a position lies in WebGPU's clip volume: x and y from -w to w, and z from 0 to w. */
function insideClipVolume(p: Float64Array): boolean {
  const w = p[3]!;
  return Math.abs(p[0]!) <= w && Math.abs(p[1]!) <= w && p[2]! >= 0 && p[2]! <= w;
}

/** A clip coordinate over `w`, the coordinate of normalized device space; 0 for a vertex at `vec4(0)`, as WebGPU places it. */
function ndc(c: number, w: number): number {
  return w === 0 ? 0 : c / w;
}

/** Copies the position and varyings of `from` into `to`. */
function copyVertex(from: ClipVertex, to: ClipVertex): void {
  for (let i = 0; i < 4; i++) to.position[i] = from.position[i]!;
  for (let k = 0; k < from.varyings.length; k++) copyComponents(from.varyings[k]!, to.varyings[k]!);
}

/** Copies the components of `from` into `to`, element by element, where `set` can allocate. */
function copyComponents(from: Float64Array, to: Float64Array): void {
  for (let i = 0; i < from.length; i++) to[i] = from[i]!;
}

/**
 * Writes the vertex where plane `plane` cuts the edge from `a` to `b`, one
 * inside it and the other outside, into `out`. The cut takes the plane rather than its
 * fraction, as a float passed to a call V8 does not inline is boxed.
 *
 * A cut vertex whose four components all lie within the rounding of the cut
 * lies within that rounding of `vec4(0)`, and is that point, as its x, y and
 * z over w are rounding alone. One whose w alone is 0, on one plane and
 * outside another, keeps its x, y and z for the next plane to cut.
 */
function cutVertex(a: ClipVertex, b: ClipVertex, plane: number, out: ClipVertex): void {
  const da = planeDistance(a.position, plane);
  const t = da / (da - planeDistance(b.position, plane));
  let rounding = 0;
  let size = 0;
  for (let i = 0; i < 4; i++) {
    const step = (b.position[i]! - a.position[i]!) * t;
    out.position[i] = a.position[i]! + step;
    rounding = Math.max(rounding, Math.abs(a.position[i]!) + Math.abs(step));
    size = Math.max(size, Math.abs(out.position[i]!));
  }
  if (size <= rounding * CUT_ROUNDING) for (let i = 0; i < 4; i++) out.position[i] = 0;
  for (let k = 0; k < a.varyings.length; k++) {
    const x = a.varyings[k]!;
    const y = b.varyings[k]!;
    const into = out.varyings[k]!;
    for (let i = 0; i < x.length; i++) into[i] = x[i]! + (y[i]! - x[i]!) * t;
  }
}

/**
 * How far a clip position lies inside plane `plane` of WebGPU's clip volume,
 * negative outside it: `w + x`, `w - x`, `w + y`, `w - y`, `z` and `w - z`,
 * in that order, as `rasterizer.wat`'s `$planeDistance` takes them.
 */
function planeDistance(p: Float64Array, plane: number): number {
  const w = p[3]!;
  switch (plane) {
    case 0:
      return w + p[0]!;
    case 1:
      return w - p[0]!;
    case 2:
      return w + p[1]!;
    case 3:
      return w - p[1]!;
    case 4:
      return p[2]!;
    default:
      return w - p[2]!;
  }
}

// === Exported functions ===

const COMPILE_JS: BackendOrigin = { target: "js", caller: "compileJS" };

/**
 * Compiles a vertex/fragment `Fn` pair and rasterizes them exactly like
 * `compileWasm`'s `WasmRasterRoutine` does, in plain JS instead of a
 * shared WASM module — see rasterizer.md for the algorithm both share.
 *
 * A draw allocates nothing for each vertex or fragment: the vertices, those a
 * clip writes, the varyings a fragment reads and the context each stage is
 * called with are made once and filled again, and a stage's result is read
 * where it left it.
 */
export function compileJS(
  vertexFn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  fragmentFn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  options: CompileJSRasterOptions = {},
): JsRasterRoutine {
  return compileJSRasterizer(COMPILE_JS, vertexFn, fragmentFn, options);
}

/** {@link compileJS} for `origin`, whose caller the errors of a draw name: `compileJS` itself, or an adapter that wraps it. */
export function compileJSRasterizer(
  origin: BackendOrigin,
  vertexFn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  fragmentFn: (...args: any[]) => Node<ShaderType> | readonly Node<ShaderType>[],
  options: CompileJSRasterOptions,
): JsRasterRoutine {
  const stageOptions = {
    params: [],
    derivatives: options.derivatives,
    reentrant: options.reentrant,
    float: options.float,
  };
  const linked = linkStages(origin, vertexFn(), fragmentFn(), true);
  const vertexStage = compileJSProgram(origin, () => linked.vertex, {
    ...stageOptions,
    name: "vtx",
    stage: "vertex",
    rasterized: true,
  });
  const fragmentStage = compileJSProgram(origin, () => linked.fragment, {
    ...stageOptions,
    name: "frag",
    stage: "fragment",
    rasterized: true,
  });
  const writesDepth = fragmentStage.writesDepth;

  /** The attribute slots the vertex stage reads, as its program declares them. */
  const attributeSlots = Object.keys(vertexStage.attributeTypes);
  const attributeWidths = attributeSlots.map((slot) => componentCountOf(vertexStage.attributeTypes[slot]!));
  // A bool attribute reaches the vertex stage as the unsigned integers it is lowered to.
  const attributeKinds = attributeSlots.map((slot) => attributeKindOf(vertexStage.attributeTypes[slot]!));
  /** The attributes the vertex stage reads, as a draw checks and assembles them. */
  const read: DrawAttributes = { slots: attributeSlots, widths: attributeWidths };
  /**
   * The value of each attribute for the vertex being shaded, in the order of the
   * slots: an array of its own filled for each vertex, a scalar as one element.
   */
  const attributeArrays = attributeWidths.map((w) => new Float64Array(w));
  /** The same arrays under their slots, as the vertex stage reads them; the loop over vertices reads the list. */
  const vertexAttributes = Object.fromEntries(attributeSlots.map((slot, i) => [slot, attributeArrays[i]!]));
  /** The host's array of each attribute during a draw, and how {@link readerOf} reads it. */
  const hostArrays: ArrayLike<number>[] = attributeSlots.map(() => NO_TARGET);
  const hostReaders = attributeSlots.map(() => 0);
  const vertexCtx: CpuShaderContext = { attributes: vertexAttributes };

  /** The varying slots the vertex stage writes, and their widths, as its program declares them. */
  const varyingSlots = Object.keys(vertexStage.varyingTypes);
  const varyingWidths = varyingSlots.map((slot) => componentCountOf(vertexStage.varyingTypes[slot]!));
  /** Whether each varying is an integer one, which a fragment reads flat, as its triangle's first vertex wrote it. */
  const varyingFlat = varyingSlots.map((slot) => componentKindOf(vertexStage.varyingTypes[slot]!) !== "float");
  /** The varyings of the fragment being shaded, each in an array of its own filled for each fragment, a scalar as one element. */
  const fragmentArrays = varyingWidths.map((w) => new Float64Array(w));
  const fragmentVaryings: Record<string, Float64Array> = {};
  varyingSlots.forEach((slot, k) => {
    fragmentVaryings[slot] = fragmentArrays[k]!;
  });
  const fragCoord = new Float64Array(2);
  /** The depth of the fragment being shaded, which the stage reads and may write over. */
  const depthSlot = new Float64Array(1);
  const fragmentCtx: CpuShaderContext & { depthSlot: Float64Array } = {
    varyings: fragmentVaryings,
    fragCoord,
    depthSlot,
  };
  /** One vertex for each the draw shades, made the first time a draw has that many and kept. */
  const vertices: ClipVertex[] = [];
  /** The vertices each draw shades and the triangles it makes of them. */
  const assembly = createDrawAssembly(origin, read);
  /** The sign of the screen area of a triangle the draw being made culls, or 0. */
  let cull = 0;

  // Persists across draw() calls of one size, like WasmRasterRoutine's depth
  // buffer: a draw of another width or height clears it, as its pixels lie elsewhere.
  let depthBuffer: Float64Array | null = null;
  let depthWidth = 0;
  let depthHeight = 0;
  // Persists across draw() calls of one size too, as WasmRasterRoutine's image
  // does — a draw that passes `clear: false` composes onto it.
  let colorBuffer: Float64Array | null = null;
  let imageWidth = 0;
  let imageHeight = 0;

  /** What the triangle being rasterized writes into, set by each draw. */
  let target: CpuDrawBuffer = NO_TARGET;
  let targetWidth = 0;
  let targetHeight = 0;

  /**
   * Two halves of 9 vertices, which the planes of a clip take turns writing,
   * as `rasterizer.wat`'s clip scratch does: a clipped polygon has 9 at most.
   */
  const scratch = Array.from({ length: 18 }, () => makeVertex(varyingWidths));
  /** The lines of the triangle being drawn, and the exact sign of each at a pixel; see `lines.ts`. */
  const lines = makeLines();
  /** The weight of each vertex of the fragment being shaded, their sum, and its depth. */
  const fragment = new Float64Array(5);
  /** Where in `scratch` the polygon the last clip left starts. */
  let clipped = 0;
  /** The pixels a triangle drawn by its lines tests: the least and greatest x, then y. */
  const box = new Int32Array(4);

  /**
   * Clips the triangle in `scratch[0..2]` to WebGPU's clip volume, one plane after another
   * (Sutherland-Hodgman), in the order `rasterizer.wat`'s clip pass takes them, and gives how many
   * vertices the polygon left has; it starts at `scratch[clipped]`.
   */
  function clip(): number {
    let src = 0;
    let count = 3;
    for (let plane = 0; plane < 6 && count >= 3; plane++) {
      const dst = 9 - src;
      let out = 0;
      for (let i = 0; i < count; i++) {
        const a = scratch[src + i]!;
        const b = scratch[src + ((i + 1) % count)]!;
        const da = planeDistance(a.position, plane);
        const db = planeDistance(b.position, plane);
        // Rounding can make a cut polygon slightly concave; it keeps 9 vertices at most, the slots of a half.
        if (da >= 0 && out < 9) copyVertex(a, scratch[dst + out++]!);
        if (da >= 0 !== db >= 0 && out < 9) cutVertex(a, b, plane, scratch[dst + out++]!);
      }
      src = dst;
      count = out;
    }
    clipped = src;
    return count;
  }

  /**
   * Draws one triangle of the draw. One whose vertices do not lie in one plane with the eye is tested pixel by pixel against its
   * own lines, and so is one with a vertex at `vec4(0)` and the other two in the clip volume, that
   * vertex at the centre. Any other is clipped, and draws only through a vertex at `vec4(0)`.
   */
  function drawTriangle(v0: ClipVertex, v1: ClipVertex, v2: ClipVertex): void {
    const a = v0.position,
      b = v1.position,
      c = v2.position;
    const zeros = (isOrigin(a) ? 1 : 0) + (isOrigin(b) ? 1 : 0) + (isOrigin(c) ? 1 : 0);
    let centre = -1;
    if (zeros === 1) {
      centre = isOrigin(a) ? 0 : isOrigin(b) ? 1 : 2;
      const inside =
        (centre === 0 || insideClipVolume(a)) &&
        (centre === 1 || insideClipVolume(b)) &&
        (centre === 2 || insideClipVolume(c));
      if (!inside) centre = -1;
    }
    if (zeros === 0 || centre !== -1) {
      const orientation = prepareLines(lines, a, b, c, centre);
      if (orientation !== 0) {
        // The sign of a triangle is its winding in normalized device coordinates, that of its part in front of the eye too.
        if (orientation * cull < 0) return;
        rasterizeByLines(v0, v1, v2, centre, orientation);
        return;
      }
      if (centre !== -1) return;
    }
    clipAndRasterize(v0, v1, v2);
  }

  /**
   * Clips a triangle whose vertices lie in one plane with the eye, and rasterizes the fan of
   * triangles the polygon left makes from a vertex of it at `vec4(0)`, which lies at the centre and
   * gives the polygon an area. Without one, the polygon lies on a line and covers nothing. The
   * triangle's first vertex gives the flat varyings.
   */
  function clipAndRasterize(v0: ClipVertex, v1: ClipVertex, v2: ClipVertex): void {
    copyVertex(v0, scratch[0]!);
    copyVertex(v1, scratch[1]!);
    copyVertex(v2, scratch[2]!);
    const count = clip();
    const src = clipped;
    let origin = -1;
    for (let i = 0; i < count; i++) if (origin === -1 && isOrigin(scratch[src + i]!.position)) origin = i;
    if (origin === -1) return;
    for (let i = 1; i + 1 < count; i++)
      rasterize(
        scratch[src + origin]!,
        scratch[src + ((origin + i) % count)]!,
        scratch[src + ((origin + i + 1) % count)]!,
        v0,
      );
  }

  /**
   * Rasterizes a triangle of the fan of a clipped polygon by its vertices' projections. Its first
   * vertex, at `vec4(0)`, lies at the centre and gives every fragment its varyings, and `first`
   * gives the flat ones.
   */
  function rasterize(v0: ClipVertex, v1: ClipVertex, v2: ClipVertex, first: ClipVertex): void {
    const width = targetWidth;
    const height = targetHeight;
    const w0 = v0.position[3]!,
      w1 = v1.position[3]!,
      w2 = v2.position[3]!;

    // Clip space -> NDC (perspective divide) -> screen space, y flipped
    // to match a canvas's top-down row order.
    const s0x = (ndc(v0.position[0]!, w0) * 0.5 + 0.5) * width;
    const s0y = (1 - (ndc(v0.position[1]!, w0) * 0.5 + 0.5)) * height;
    const s1x = (ndc(v1.position[0]!, w1) * 0.5 + 0.5) * width;
    const s1y = (1 - (ndc(v1.position[1]!, w1) * 0.5 + 0.5)) * height;
    const s2x = (ndc(v2.position[0]!, w2) * 0.5 + 0.5) * width;
    const s2y = (1 - (ndc(v2.position[1]!, w2) * 0.5 + 0.5)) * height;

    const area = (s1x - s0x) * (s2y - s0y) - (s1y - s0y) * (s2x - s0x);
    if (area === 0) return; // degenerate (zero-area) triangle: skip it
    if (area * cull > 0) return; // a face the draw culls
    // Each edge function, times `wind`, is positive inside whichever way the triangle winds.
    const wind = area > 0 ? 1 : -1;
    const owns0 = ownsEdge((s2x - s1x) * wind, (s2y - s1y) * wind);
    const owns1 = ownsEdge((s0x - s2x) * wind, (s0y - s2y) * wind);
    const owns2 = ownsEdge((s1x - s0x) * wind, (s1y - s0y) * wind);

    const minX = Math.max(0, Math.floor(Math.min(s0x, s1x, s2x)));
    const maxX = Math.min(width - 1, Math.ceil(Math.max(s0x, s1x, s2x)));
    const minY = Math.max(0, Math.floor(Math.min(s0y, s1y, s2y)));
    const maxY = Math.min(height - 1, Math.ceil(Math.max(s0y, s1y, s2y)));

    // NDC depth (z/w) is affine in screen space, like x/w and y/w above,
    // so it interpolates with plain barycentric weights — no invW needed.
    const depth0 = ndc(v0.position[2]!, w0),
      depth1 = ndc(v1.position[2]!, w1),
      depth2 = ndc(v2.position[2]!, w2);
    // A triangle wholly outside depth 0 to 1 draws nothing. One that crosses it drops each
    // pixel outside it, which is what clipping there drops, as depth over w is affine in
    // screen space; one inside it draws every pixel, whatever the rounding of its depth.
    const nearest = Math.min(depth0, depth1, depth2);
    const farthest = Math.max(depth0, depth1, depth2);
    if (farthest < 0 || nearest > 1) return;
    const crossesDepthRange = nearest < 0 || farthest > 1;

    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const px = x + 0.5,
          py = y + 0.5;
        const e0 = (s1x - px) * (s2y - py) - (s1y - py) * (s2x - px);
        const e1 = (s2x - px) * (s0y - py) - (s2y - py) * (s0x - px);
        const e2 = (s0x - px) * (s1y - py) - (s0y - py) * (s1x - px);
        const f0 = e0 * wind,
          f1 = e1 * wind,
          f2 = e2 * wind;
        if (f0 < 0 || f1 < 0 || f2 < 0) continue;
        if ((f0 === 0 && !owns0) || (f1 === 0 && !owns1) || (f2 === 0 && !owns2)) continue;

        const b0 = e0 / area,
          b1 = e1 / area,
          b2 = e2 / area;
        const pixelDepth = b0 * depth0 + b1 * depth1 + b2 * depth2;
        if (crossesDepthRange && !(pixelDepth >= 0 && pixelDepth <= 1)) continue;

        // Perspective weighs a vertex by the inverse of its w, so the one at vec4(0) takes all of it.
        fragment[0] = 1;
        fragment[1] = 0;
        fragment[2] = 0;
        fragment[3] = 1;
        fragment[4] = pixelDepth;
        shade(x, y, v0, v1, v2, first);
      }
    }
  }

  /**
   * Rasterizes a triangle by its own lines, which {@link prepareLines} has measured: a pixel is
   * covered where each line's exact sign puts its centre inside, or on an edge the triangle owns,
   * and where the point there lies at a depth from 0 to 1. The weight of each
   * vertex is its line there, scaled; one at `centre` takes all of it. The depth is the
   * triangle's plane of depth there, which {@link prepareLines} summed exactly once, or near a tie of
   * its terms the point's z summed exactly. A triangle whose
   * vertices all lie at a depth from 0 to 1 needs no test of depth. The pixels tested are those
   * of the projections of the vertices, or of the polygon the clip leaves of a triangle with a
   * vertex at or behind the eye.
   */
  function rasterizeByLines(v0: ClipVertex, v1: ClipVertex, v2: ClipVertex, centre: number, orientation: number): void {
    const width = targetWidth;
    const height = targetHeight;
    const points = lines.points;
    box[0] = 0;
    box[1] = width - 1;
    box[2] = 0;
    box[3] = height - 1;
    if (points[3]! > 0 && points[7]! > 0 && points[11]! > 0) {
      let left = Infinity,
        right = -Infinity,
        top = Infinity,
        bottom = -Infinity;
      for (let k = 0; k < 3; k++) {
        const sx = ((points[4 * k]! / points[4 * k + 3]!) * 0.5 + 0.5) * width;
        const sy = (1 - ((points[4 * k + 1]! / points[4 * k + 3]!) * 0.5 + 0.5)) * height;
        left = Math.min(left, sx);
        right = Math.max(right, sx);
        top = Math.min(top, sy);
        bottom = Math.max(bottom, sy);
      }
      box[0] = Math.max(0, Math.floor(left));
      box[1] = Math.min(width - 1, Math.ceil(right));
      box[2] = Math.max(0, Math.floor(top));
      box[3] = Math.min(height - 1, Math.ceil(bottom));
    } else {
      for (let k = 0; k < 3; k++) for (let c = 0; c < 4; c++) scratch[k]!.position[c] = points[4 * k + c]!;
      const count = clip();
      if (count < 3) return;
      let left = Infinity,
        right = -Infinity,
        top = Infinity,
        bottom = -Infinity,
        bounded = true;
      for (let i = 0; i < count; i++) {
        const p = scratch[clipped + i]!.position;
        if (!(p[3]! >= BOUNDED_W)) bounded = false;
        const sx = ((p[0]! / p[3]!) * 0.5 + 0.5) * width;
        const sy = (1 - ((p[1]! / p[3]!) * 0.5 + 0.5)) * height;
        left = Math.min(left, sx);
        right = Math.max(right, sx);
        top = Math.min(top, sy);
        bottom = Math.max(bottom, sy);
      }
      // The rounding of a cut moves a vertex by far less than half a pixel, which the floor and ceiling take in.
      if (bounded) {
        box[0] = Math.max(0, Math.floor(left));
        box[1] = Math.min(width - 1, Math.ceil(right));
        box[2] = Math.max(0, Math.floor(top));
        box[3] = Math.min(height - 1, Math.ceil(bottom));
      }
    }
    const minX = box[0]!,
      maxX = box[1]!,
      minY = box[2]!,
      maxY = box[3]!;
    const owns = lines.owns;
    const values = lines.values;
    const ratios = lines.ratios;
    const whole = width * height;
    const inDepthRange = liesInDepthRange(lines);
    // Local names for the imported functions: a module runner, as vitest's, may read an import
    // through a getter at each call, which costs the loop more than the call.
    const testLine = lineSign,
      testPoint = pointSign,
      depthThere = depthAt,
      row = prepareRow;
    for (let y = minY; y <= maxY; y++) {
      const Y = (height - 2 * y - 1) * width;
      row(lines, Y, whole);
      for (let x = minX; x <= maxX; x++) {
        const X = (2 * x + 1 - width) * height;
        const s0 = testLine(lines, 0, X, Y, whole) * orientation;
        if (s0 < 0 || (s0 === 0 && owns[0] === 0)) continue;
        const s1 = testLine(lines, 1, X, Y, whole) * orientation;
        if (s1 < 0 || (s1 === 0 && owns[1] === 0)) continue;
        const s2 = testLine(lines, 2, X, Y, whole) * orientation;
        if (s2 < 0 || (s2 === 0 && owns[2] === 0)) continue;
        // At a depth from 0 to 1: z >= 0 and w - z >= 0. Inside the three lines, w is the
        // triangle's sign times W, so the point lies in front of the eye.
        if (!inDepthRange) {
          if (testPoint(lines, 2, X, Y, whole) * orientation < 0) continue;
          if (testPoint(lines, -1, X, Y, whole) * orientation < 0) continue;
        }
        const l0 = values[0]!,
          l1 = values[1]!,
          l2 = values[2]!;
        fragment[4] = depthThere(lines, X, Y, whole);
        if (centre !== -1) {
          fragment[0] = centre === 0 ? 1 : 0;
          fragment[1] = centre === 1 ? 1 : 0;
          fragment[2] = centre === 2 ? 1 : 0;
        } else {
          // A line that is 0 there has a value of 0, so the mask only picks the ratios.
          const mask = (s0 !== 0 ? 1 : 0) | (s1 !== 0 ? 2 : 0) | (s2 !== 0 ? 4 : 0);
          fragment[0] = ratios[mask * 3]! * Math.abs(l0);
          fragment[1] = ratios[mask * 3 + 1]! * Math.abs(l1);
          fragment[2] = ratios[mask * 3 + 2]! * Math.abs(l2);
        }
        fragment[3] = fragment[0]! + fragment[1]! + fragment[2]!;
        shade(x, y, v0, v1, v2, v0);
      }
    }
  }

  /**
   * Shades the fragment at pixel `(x, y)` of a triangle of `v0`, `v1` and `v2`, by the weights,
   * their sum and the depth in `fragment`: tests its depth, interpolates its varyings, runs the
   * fragment stage and writes what it gives. `first`, the triangle's first vertex, gives the flat varyings.
   */
  function shade(x: number, y: number, v0: ClipVertex, v1: ClipVertex, v2: ClipVertex, first: ClipVertex): void {
    const depths = depthBuffer!;
    const pixelIndex = y * targetWidth + x;
    const pixelDepth = fragment[4]!;
    // LEQUAL depth test: closer-or-equal wins. A stage that writes its depth is tested once it has run.
    if (!writesDepth && pixelDepth > depths[pixelIndex]!) return;
    const p0 = fragment[0]!,
      p1 = fragment[1]!,
      p2 = fragment[2]!,
      sum = fragment[3]!;
    const slots = varyingSlots;
    for (let k = 0; k < slots.length; k++) {
      const a = v0.varyings[k]!,
        b = v1.varyings[k]!,
        c = v2.varyings[k]!,
        out = fragmentArrays[k]!;
      if (varyingFlat[k]) out.set(first.varyings[k]!);
      else for (let i = 0; i < out.length; i++) out[i] = (a[i]! * p0 + b[i]! * p1 + c[i]! * p2) / sum;
    }
    fragCoord[0] = x + 0.5;
    fragCoord[1] = y + 0.5;
    depthSlot[0] = pixelDepth;

    const raw = fragmentStage.runInPlace(fragmentCtx);
    // A fragment that discards leaves the pixel and its depth as they were.
    if (raw === null) return;
    // A written depth is clamped to the depth range, as WebGPU clamps it. A path that writes none keeps the fragment's own.
    const depth = writesDepth ? Math.min(Math.max(depthSlot[0]!, 0), 1) : pixelDepth;
    if (writesDepth && depth > depths[pixelIndex]!) return;
    depths[pixelIndex] = depth;
    // A fragment that writes no colour leaves the pixel as it was.
    const color = ArrayBuffer.isView(raw)
      ? (raw as Float64Array)
      : ((isResultObject(raw) ? raw.value : undefined) as Float64Array);
    if (!color) return;
    const base = pixelIndex * 4;
    for (let c = 0; c < 4; c++) target[base + c] = color[c] ?? 0;
  }

  /**
   * Points both stages' contexts at the inputs of `from` that every vertex and
   * fragment shares. The contexts own their attributes and varyings.
   */
  function shareInputs(from: Pick<JsRasterContext, "uniforms" | "textures">): void {
    vertexCtx.uniforms = fragmentCtx.uniforms = from.uniforms;
    vertexCtx.textures = fragmentCtx.textures = from.textures;
  }

  function draw(ctx: JsRasterContext, options: JsRasterDrawOptions): CpuDrawBuffer {
    shareInputs(ctx);
    try {
      return drawShaded(ctx, options);
    } finally {
      // The contexts are kept for the next draw, so they let go of this one's inputs.
      shareInputs(NO_INPUTS);
      hostArrays.fill(NO_TARGET);
      target = NO_TARGET;
    }
  }

  /**
   * Runs the vertex stage for each of the first `slotCount` slots, on the
   * vertex `vertexOfSlot` gives it, and keeps each one's position and varyings.
   * A function of its own, so the per-draw work before it, which reads the
   * host's objects by many names, cannot deoptimize the work it does for each
   * vertex.
   */
  function shadeVertices(vertexOfSlot: Uint32Array, slotCount: number): void {
    for (let i = 0; i < slotCount; i++) {
      const v = vertexOfSlot[i]!;
      for (let a = 0; a < attributeSlots.length; a++) {
        const host = hostArrays[a]!;
        const reader = hostReaders[a]!;
        const w = attributeWidths[a]!;
        const kind = attributeKinds[a]!;
        const into = attributeArrays[a]!;
        // A vertex past the end of an attribute reads it as zero.
        if ((v + 1) * w > host.length) {
          into.fill(0);
          continue;
        }
        for (let k = 0, index = v * w; k < w; k++, index++) {
          switch (reader) {
            case 0:
              into[k] = attributeComponent(kind, (host as Float64Array)[index]!);
              break;
            case 1:
              into[k] = attributeComponent(kind, (host as Float32Array)[index]!);
              break;
            case 2:
              into[k] = attributeComponent(kind, (host as Int32Array)[index]!);
              break;
            case 3:
              into[k] = attributeComponent(kind, (host as Uint32Array)[index]!);
              break;
            case 4:
              into[k] = attributeComponent(kind, (host as Int16Array)[index]!);
              break;
            case 5:
              into[k] = attributeComponent(kind, (host as Uint16Array)[index]!);
              break;
            case 6:
              into[k] = attributeComponent(kind, (host as Int8Array)[index]!);
              break;
            case 7:
              into[k] = attributeComponent(kind, (host as Uint8Array)[index]!);
              break;
            default:
              into[k] = attributeComponent(kind, host[index]!);
          }
        }
      }
      vertexCtx.vertexIndex = v;
      const raw = vertexStage.runInPlace(vertexCtx);
      const position = vertexPosition(raw);
      const varyings = isResultObject(raw) ? raw.varyings : undefined;
      const vertex = (vertices[i] ??= makeVertex(varyingWidths));
      for (let c = 0; c < 4; c++) vertex.position[c] = position[c]!;
      for (let k = 0; k < varyingSlots.length; k++) {
        // The stage hands back a bool as true or false, and any other varying in an array, a scalar as one element.
        const value = varyings![varyingSlots[k]!];
        const into = vertex.varyings[k]!;
        if (typeof value === "boolean") into[0] = value ? 1 : 0;
        else for (let c = 0; c < into.length; c++) into[c] = (value as ArrayLike<number>)[c]!;
      }
    }
  }

  function drawShaded(ctx: JsRasterContext, options: JsRasterDrawOptions): CpuDrawBuffer {
    const { width, height, out } = options;
    const { attributes } = ctx;
    for (let a = 0; a < attributeSlots.length; a++) {
      const slot = attributeSlots[a]!;
      if (!attributes[slot]) throw missingAttributeError(origin, slot);
    }
    assembly.draw(attributes, ctx.index, options);

    for (let a = 0; a < attributeSlots.length; a++) {
      const host = attributes[attributeSlots[a]!]!;
      hostArrays[a] = host;
      hostReaders[a] = readerOf(host);
    }

    shadeVertices(assembly.vertexOfSlot, assembly.slotCount);

    const pixelCount = width * height;
    if (!depthBuffer || depthBuffer.length < pixelCount) {
      depthBuffer = new Float64Array(pixelCount).fill(Infinity);
    } else if (options.clearDepth !== false || width !== depthWidth || height !== depthHeight) {
      depthBuffer.fill(Infinity);
    }
    depthWidth = width;
    depthHeight = height;

    // An image of another size lies at other pixels, so a draw of that size starts from a transparent one.
    if (!out && (width !== imageWidth || height !== imageHeight)) {
      if (!colorBuffer || colorBuffer.length < pixelCount * 4) colorBuffer = new Float64Array(pixelCount * 4);
      else colorBuffer.fill(0, 0, pixelCount * 4);
      imageWidth = width;
      imageHeight = height;
    }
    const result = out ?? colorBuffer!;
    if (options.clear !== false) {
      const clearColor = options.clearColor ?? TRANSPARENT_BLACK;
      for (let i = 0; i < pixelCount * 4; i += 4) {
        result[i] = clearColor[0]!;
        result[i + 1] = clearColor[1]!;
        result[i + 2] = clearColor[2]!;
        result[i + 3] = clearColor[3]!;
      }
    }

    target = result;
    targetWidth = width;
    targetHeight = height;
    cull = cullSign(options);
    const triangles = assembly.triangles;
    for (let t = 0; t < assembly.triangleCount; t++) {
      drawTriangle(vertices[triangles[t * 3]!]!, vertices[triangles[t * 3 + 1]!]!, vertices[triangles[t * 3 + 2]!]!);
    }

    return result;
  }

  return { draw, attributeTypes: vertexStage.attributeTypes };
}
