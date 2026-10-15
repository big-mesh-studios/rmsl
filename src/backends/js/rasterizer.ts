import type { BackendOrigin } from "../../errors";
import { Node, ShaderType, componentCountOf, componentKindOf, type ScalarKind } from "../../core";
import { DrawClearOptions, DrawCountOptions, TRANSPARENT_BLACK, TypedArray, missingAttributeError } from "../adapter";
import { CpuDrawBuffer, CpuShaderContext, isResultObject, vertexPosition } from "../cpu";
import { linkStages } from "../shared";
import { compileJSProgram, CompileJSOptions } from "./js";

/**
 * How many times the rounding of a cut a vertex's components may be and still
 * be taken for 0: 16 units in the last place, `2 ** -48`, as `rasterizer.wat` takes it.
 */
const CUT_ROUNDING = 2 ** -48;

/** The w above which every vertex of a triangle lets it skip the clip — see rasterizer.md's clip pass (`rasterizer.wat` uses the same 1e-5). */
const W_CLIP_EPS = 1e-5;

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
export interface JsRasterDrawOptions extends DrawCountOptions, DrawClearOptions {
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

/** The inputs both stages share between draws: none, in one object, so a draw makes none. */
const NO_INPUTS: Pick<JsRasterContext, "uniforms" | "textures"> = {};

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

/** A vertex with room for a position and varyings of these widths. */
function makeVertex(widths: readonly number[]): ClipVertex {
  return { position: new Float64Array(4), varyings: widths.map((w) => new Float64Array(w)) };
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
 * Writes the vertex where plane `plane` cuts the edge from `a`, inside it, to
 * `b`, outside it, into `out`. A varying `flat` marks is copied from `first`,
 * the first vertex of the triangle being clipped, as no value lies between
 * two integers. The cut takes the plane rather than its fraction, as a float
 * passed to a call V8 does not inline is boxed.
 *
 * A cut vertex whose four components all lie within the rounding of the cut
 * lies within that rounding of `vec4(0)`, and is that point, as its x, y and
 * z over w are rounding alone. One whose w alone is 0, on one plane and
 * outside another, keeps its x, y and z for the next plane to cut.
 */
function cutVertex(
  a: ClipVertex,
  b: ClipVertex,
  plane: number,
  flat: readonly boolean[],
  first: ClipVertex,
  out: ClipVertex,
): void {
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
    if (flat[k]) copyComponents(first.varyings[k]!, into);
    else for (let i = 0; i < x.length; i++) into[i] = x[i]! + (y[i]! - x[i]!) * t;
  }
}

/**
 * Writes the edge functions of the triangle in `scratch[0..2]` on a target of
 * `width` by `height` into `lines`, and whether it owns the pixel centres on
 * each into `owns`, and gives whether it could. The function of the edge
 * opposite vertex `e` is the line through the other two vertices in
 * homogeneous coordinates, `(x, y, w)`, the cross product of the two, turned
 * into screen space. It is exactly 0 for a pixel centre the projections of
 * those vertices place on the edge, wherever a clip cuts it, and positive
 * inside the part of the triangle in front of the eye. Each vertex is first
 * scaled by a power of two that brings its largest of x, y and w near 1,
 * which moves neither its projection nor a bit of the products, at any scale
 * of w. A triangle whose three vertices lie in one plane with the eye, as one
 * at `vec4(0)` does, or with a vertex of no size, has none. `points` holds the
 * scaled vertices, four components each.
 */
function measureEdges(
  scratch: readonly ClipVertex[],
  width: number,
  height: number,
  points: Float64Array,
  lines: Float64Array,
  owns: Uint8Array,
): boolean {
  for (let k = 0; k < 3; k++) {
    const p = scratch[k]!.position;
    const size = Math.max(Math.abs(p[0]!), Math.abs(p[1]!), Math.abs(p[3]!));
    if (!(size >= 2 ** -500 && size <= 2 ** 500)) return false;
    const scale = 2 ** -Math.floor(Math.log2(size));
    points[4 * k] = p[0]! * scale;
    points[4 * k + 1] = p[1]! * scale;
    points[4 * k + 2] = p[2]! * scale;
    points[4 * k + 3] = p[3]! * scale;
  }
  // The line through vertices 1 and 2, taken at vertex 0, is the determinant of the three.
  const det =
    (points[5]! * points[11]! - points[7]! * points[9]!) * points[0]! +
    (points[7]! * points[8]! - points[4]! * points[11]!) * points[1]! +
    (points[4]! * points[9]! - points[5]! * points[8]!) * points[3]!;
  if (!(det > 0 || det < 0)) return false;
  const sign = det > 0 ? 1 : -1;
  for (let e = 0; e < 3; e++) {
    const i = 4 * ((e + 1) % 3),
      j = 4 * ((e + 2) % 3);
    const a = points[i + 1]! * points[j + 3]! - points[i + 3]! * points[j + 1]!,
      b = points[i + 3]! * points[j]! - points[i]! * points[j + 3]!,
      c = points[i]! * points[j + 1]! - points[i + 1]! * points[j]!;
    // a * ndcX + b * ndcY + c, where ndcX = 2x / width - 1 and ndcY = 1 - 2y / height.
    const ax = ((2 * a) / width) * sign,
      by = ((-2 * b) / height) * sign;
    lines[3 * e] = ax;
    lines[3 * e + 1] = by;
    lines[3 * e + 2] = (c - a + b) * sign;
    if (ax === 0 && by === 0) return false;
    owns[e] = ax > 0 || (ax === 0 && by > 0) ? 1 : 0;
  }
  return true;
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
  const vertexStage = compileJSProgram(origin, () => linked.vertex, { ...stageOptions, name: "vtx", stage: "vertex" });
  const fragmentStage = compileJSProgram(origin, () => linked.fragment, {
    ...stageOptions,
    name: "frag",
    stage: "fragment",
  });
  const writesDepth = fragmentStage.writesDepth;

  /** The attribute slots the vertex stage reads, as its program declares them. */
  const attributeSlots = Object.keys(vertexStage.attributeTypes);
  const attributeWidths = attributeSlots.map((slot) => componentCountOf(vertexStage.attributeTypes[slot]!));
  const attributeKinds = attributeSlots.map((slot) => componentKindOf(vertexStage.attributeTypes[slot]!));
  /**
   * The width of each attribute slot the vertex stage reads, by slot. A draw
   * looks up the host's slots in a map rather than an object: every
   * rasterizer runs one copy of the code below, and V8 deoptimizes a keyed
   * load in a `for-in` loop each time it meets slot names other than those it
   * was optimized for.
   */
  const widths = new Map(attributeSlots.map((slot, i) => [slot, attributeWidths[i]!]));
  /** The attributes of the vertex being shaded: a vector in an array of its own, filled for each vertex. */
  const vertexAttributes: Record<string, number | Float64Array | undefined> = {};
  attributeSlots.forEach((slot, i) => {
    vertexAttributes[slot] = attributeWidths[i]! > 1 ? new Float64Array(attributeWidths[i]!) : 0;
  });
  const vertexCtx: CpuShaderContext = { attributes: vertexAttributes };

  /** The varying slots the vertex stage writes, and their widths, as its program declares them. */
  const varyingSlots = Object.keys(vertexStage.varyingTypes);
  const varyingWidths = varyingSlots.map((slot) => componentCountOf(vertexStage.varyingTypes[slot]!));
  /** Whether each varying is an integer one, which a fragment reads flat, as its triangle's first vertex wrote it. */
  const varyingFlat = varyingSlots.map((slot) => componentKindOf(vertexStage.varyingTypes[slot]!) !== "float");
  /** The varyings of the fragment being shaded: a vector in an array of its own, filled for each fragment. */
  const fragmentArrays = varyingWidths.map((w) => new Float64Array(w));
  const fragmentVaryings: Record<string, number | Float64Array> = {};
  varyingSlots.forEach((slot, k) => {
    fragmentVaryings[slot] = varyingWidths[k]! > 1 ? fragmentArrays[k]! : 0;
  });
  const fragCoord = new Float64Array(2);
  const fragmentCtx: CpuShaderContext = { varyings: fragmentVaryings, fragCoord };
  /** One vertex for each the draw shades, made the first time a draw has that many and kept. */
  const vertices: ClipVertex[] = [];

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
  /**
   * The edge functions of the triangle being clipped, in screen space, three
   * numbers for each edge `e`: `lines[3e] * x + lines[3e + 1] * y + lines[3e + 2]`
   * is positive inside, and 0 exactly on the edge as its vertices place it.
   */
  const lines = new Float64Array(9);
  /** The vertices of the triangle being clipped, scaled; see {@link measureEdges}. */
  const points = new Float64Array(12);
  /** Whether the triangle being clipped owns the pixel centres on each of its edges, by the top-left rule. */
  const linesOwn = new Uint8Array(3);

  /**
   * Clips the triangle `v0`, `v1`, `v2` to WebGPU's clip volume, one plane after
   * another (Sutherland-Hodgman), in `scratch`, and rasterizes the fan of
   * triangles the polygon left makes, in the order `rasterizer.wat`'s clip pass
   * makes them. The first vertex of the polygon is `v0` or a vertex the clip
   * cut, which holds the flat varyings of `v0`.
   */
  function clipAndRasterize(v0: ClipVertex, v1: ClipVertex, v2: ClipVertex): void {
    copyVertex(v0, scratch[0]!);
    copyVertex(v1, scratch[1]!);
    copyVertex(v2, scratch[2]!);
    const measured = measureEdges(scratch, targetWidth, targetHeight, points, lines, linesOwn);
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
        // A cut runs from the vertex inside, so a w far smaller than the other's keeps its digits.
        if (da >= 0 !== db >= 0 && out < 9) {
          if (da >= 0) cutVertex(a, b, plane, varyingFlat, v0, scratch[dst + out++]!);
          else cutVertex(b, a, plane, varyingFlat, v0, scratch[dst + out++]!);
        }
      }
      src = dst;
      count = out;
    }
    for (let i = 1; i + 1 < count; i++) {
      // The edge opposite the first vertex lies on the polygon's edge; the one opposite the
      // second does for the last triangle, and the one opposite the third for the first.
      const outer = measured ? 1 | (i + 2 === count ? 2 : 0) | (i === 1 ? 4 : 0) : 0;
      rasterize(scratch[src]!, scratch[src + i]!, scratch[src + i + 1]!, outer);
    }
  }

  /**
   * Rasterizes one triangle, already clipped to the clip volume when it needed it, into `target`.
   * Its first vertex gives the flat varyings. Bit `k` of `outer` marks its edge opposite vertex
   * `k` as an edge of the polygon the clip left, in a fan cut from a triangle whose edges are
   * measured in `lines`. Then that triangle's edge functions and depth decide which pixels
   * the polygon covers, which no rounding of a cut moves, and the fan's own edges only part
   * the polygon's pixels between its triangles.
   */
  function rasterize(v0: ClipVertex, v1: ClipVertex, v2: ClipVertex, outer: number): void {
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
    // Each edge function, times `wind`, is positive inside whichever way the triangle winds.
    const wind = area > 0 ? 1 : -1;
    const owns0 = ownsEdge((s2x - s1x) * wind, (s2y - s1y) * wind);
    const owns1 = ownsEdge((s0x - s2x) * wind, (s0y - s2y) * wind);
    const owns2 = ownsEdge((s1x - s0x) * wind, (s1y - s0y) * wind);

    const minX = Math.max(0, Math.floor(Math.min(s0x, s1x, s2x)));
    const maxX = Math.min(width - 1, Math.ceil(Math.max(s0x, s1x, s2x)));
    const minY = Math.max(0, Math.floor(Math.min(s0y, s1y, s2y)));
    const maxY = Math.min(height - 1, Math.ceil(Math.max(s0y, s1y, s2y)));

    // Perspective weighs vertex i by its barycentric over its w, here times the least w, so
    // no weight overflows or underflows at any scale. A least w whose inverse is not
    // finite, as at vec4(0), takes all the weight.
    let near = 0,
      wNear = w0;
    if (w1 < wNear) {
      near = 1;
      wNear = w1;
    }
    if (w2 < wNear) {
      near = 2;
      wNear = w2;
    }
    const inverse = 1 / wNear;
    const finite = inverse > 0 && inverse < Infinity;
    const r0 = finite ? wNear / w0 : near === 0 ? 1 : 0,
      r1 = finite ? wNear / w1 : near === 1 ? 1 : 0,
      r2 = finite ? wNear / w2 : near === 2 ? 1 : 0;
    // NDC depth (z/w) is affine in screen space, like x/w and y/w above,
    // so it interpolates with plain barycentric weights — no invW needed.
    const depth0 = ndc(v0.position[2]!, w0),
      depth1 = ndc(v1.position[2]!, w1),
      depth2 = ndc(v2.position[2]!, w2);
    const slots = varyingSlots;
    const depths = depthBuffer!;
    // A triangle wholly outside depth 0 to 1 draws nothing. One that crosses it drops each
    // pixel outside it, which is what clipping there drops, as depth over w is affine in
    // screen space; one inside it draws every pixel, whatever the rounding of its depth.
    const nearest = Math.min(depth0, depth1, depth2);
    const farthest = Math.max(depth0, depth1, depth2);
    const measured = outer !== 0;
    if (!measured && (farthest < 0 || nearest > 1)) return;
    const crossesDepthRange = nearest < 0 || farthest > 1;

    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const px = x + 0.5,
          py = y + 0.5;
        const e0 = (s1x - px) * (s2y - py) - (s1y - py) * (s2x - px);
        const e1 = (s2x - px) * (s0y - py) - (s2y - py) * (s0x - px);
        const e2 = (s0x - px) * (s1y - py) - (s0y - py) * (s1x - px);
        // An edge of the polygon passes here; the measured triangle's own edges test the pixel below.
        const f0 = outer & 1 ? 1 : e0 * wind,
          f1 = outer & 2 ? 1 : e1 * wind,
          f2 = outer & 4 ? 1 : e2 * wind;
        if (f0 < 0 || f1 < 0 || f2 < 0) continue;
        if ((f0 === 0 && !owns0) || (f1 === 0 && !owns1) || (f2 === 0 && !owns2)) continue;
        // The functions of the edges of the triangle this one was cut from, which no rounding of a cut moves.
        const l0 = measured ? lines[0]! * px + lines[1]! * py + lines[2]! : 0,
          l1 = measured ? lines[3]! * px + lines[4]! * py + lines[5]! : 0,
          l2 = measured ? lines[6]! * px + lines[7]! * py + lines[8]! : 0;
        if (
          measured &&
          !(
            (l0 > 0 || (l0 === 0 && linesOwn[0])) &&
            (l1 > 0 || (l1 === 0 && linesOwn[1])) &&
            (l2 > 0 || (l2 === 0 && linesOwn[2]))
          )
        )
          continue;

        const b0 = e0 / area,
          b1 = e1 / area,
          b2 = e2 / area;
        // Its z over its w, from its vertices weighted by the functions of the edges opposite them.
        const pixelDepth = measured
          ? (l0 * points[2]! + l1 * points[6]! + l2 * points[10]!) /
            (l0 * points[3]! + l1 * points[7]! + l2 * points[11]!)
          : b0 * depth0 + b1 * depth1 + b2 * depth2;

        if ((measured || crossesDepthRange) && !(pixelDepth >= 0 && pixelDepth <= 1)) continue;
        const pixelIndex = y * width + x;
        // LEQUAL depth test: closer-or-equal wins. A stage that writes its depth is tested once it has run.
        if (!writesDepth && pixelDepth > depths[pixelIndex]!) continue;

        // Perspective-correct: each vertex's varying by its weight, over the sum of the weights.
        let p0 = b0 * r0,
          p1 = b1 * r1,
          p2 = b2 * r2,
          sum = p0 + p1 + p2;
        // On the edge opposite the vertex that takes all the weight, that vertex's barycentric is 0 too.
        if (sum === 0) {
          p0 = near === 0 ? 1 : 0;
          p1 = near === 1 ? 1 : 0;
          p2 = near === 2 ? 1 : 0;
          sum = 1;
        }
        const perspective = 1 / sum;
        for (let k = 0; k < slots.length; k++) {
          const a = v0.varyings[k]!,
            b = v1.varyings[k]!,
            c = v2.varyings[k]!,
            out = fragmentArrays[k]!;
          if (varyingFlat[k]) out.set(a);
          else for (let i = 0; i < out.length; i++) out[i] = (a[i]! * p0 + b[i]! * p1 + c[i]! * p2) * perspective;
          if (out.length === 1) fragmentVaryings[slots[k]!] = out[0]!;
        }
        fragCoord[0] = px;
        fragCoord[1] = py;
        fragmentCtx.fragDepth = pixelDepth;

        const raw = fragmentStage.runInPlace(fragmentCtx);
        // A fragment that discards leaves the pixel and its depth as they were.
        if (raw === null) continue;
        const written = isResultObject(raw) ? raw.fragDepth : undefined;
        // A written depth is clamped to the depth range, as WebGPU clamps it.
        const depth = typeof written === "number" ? Math.min(Math.max(written, 0), 1) : pixelDepth;
        if (writesDepth && depth > depths[pixelIndex]!) continue;
        depths[pixelIndex] = depth;
        // A fragment that writes no colour leaves the pixel as it was.
        const color = ArrayBuffer.isView(raw)
          ? (raw as Float64Array)
          : ((isResultObject(raw) ? raw.value : undefined) as Float64Array);
        if (!color) continue;
        const base = pixelIndex * 4;
        for (let c = 0; c < 4; c++) target[base + c] = color[c] ?? 0;
      }
    }
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
      target = NO_TARGET;
    }
  }

  function drawShaded(ctx: JsRasterContext, options: JsRasterDrawOptions): CpuDrawBuffer {
    const { width, height, out } = options;
    const { attributes } = ctx;
    const first = options.first ?? 0;
    for (let a = 0; a < attributeSlots.length; a++) {
      const slot = attributeSlots[a]!;
      if (!attributes[slot]) throw missingAttributeError(origin, slot);
    }
    // An attribute the program does not read counts for nothing, as a WebGPU
    // draw reads no buffer its shader does not declare.
    let firstSlot: string | undefined;
    for (const slot in attributes) {
      if (!widths.has(slot)) continue;
      firstSlot = slot;
      break;
    }
    const inferredCount = firstSlot ? Math.floor(attributes[firstSlot]!.length / widths.get(firstSlot)!) - first : 0;
    const vertexCount = options.count ?? inferredCount;

    // vertex pass
    for (let i = 0; i < vertexCount; i++) {
      for (let a = 0; a < attributeSlots.length; a++) {
        const slot = attributeSlots[a]!;
        const buffer = attributes[slot];
        const w = attributeWidths[a]!;
        const kind = attributeKinds[a]!;
        if (w === 1) vertexAttributes[slot] = attributeComponent(kind, buffer[i + first]!);
        else {
          let into = vertexAttributes[slot];
          if (!(into instanceof Float64Array)) into = vertexAttributes[slot] = new Float64Array(w);
          for (let k = 0; k < w; k++) into[k] = attributeComponent(kind, buffer[(i + first) * w + k]!);
        }
      }
      vertexCtx.vertexIndex = i + first;
      const raw = vertexStage.runInPlace(vertexCtx);
      const position = vertexPosition(raw);
      const varyings = isResultObject(raw) ? raw.varyings : undefined;
      const vertex = (vertices[i] ??= makeVertex(varyingWidths));
      for (let c = 0; c < 4; c++) vertex.position[c] = position[c]!;
      for (let k = 0; k < varyingSlots.length; k++) {
        // The stage gives 0 for a varying this vertex does not write, and a bool as true or false.
        const value = varyings![varyingSlots[k]!];
        const into = vertex.varyings[k]!;
        if (typeof value === "number") into[0] = value;
        else if (typeof value === "boolean") into[0] = value ? 1 : 0;
        else for (let c = 0; c < into.length; c++) into[c] = (value as ArrayLike<number>)[c]!;
      }
    }

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
    for (let t = 0; t + 2 < vertexCount; t += 3) {
      const v0 = vertices[t]!,
        v1 = vertices[t + 1]!,
        v2 = vertices[t + 2]!;
      // A triangle wholly in front of the eye is rasterized as it is; only one with a vertex near or behind it is clipped.
      if (v0.position[3]! > W_CLIP_EPS && v1.position[3]! > W_CLIP_EPS && v2.position[3]! > W_CLIP_EPS) {
        rasterize(v0, v1, v2, 0);
        continue;
      }
      clipAndRasterize(v0, v1, v2);
    }

    return result;
  }

  return { draw, attributeTypes: vertexStage.attributeTypes };
}
