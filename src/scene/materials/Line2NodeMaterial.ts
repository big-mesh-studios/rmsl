import {
  float, vec2, vec3, vec4, element, If, Discard, mix,
  type Node, type UniformNode, type GLSLPrecision,
} from "../../rmsl";
import { NodeMaterial, resolveSlot, type SlotValue } from "./NodeMaterial";
import { Builder } from "./nodes/Builder";
import { Color } from "../math/Color";
import { Side } from "./Material";
import type { Scene } from "../scenes/Scene";

/**
 * A node material that renders wide lines by expanding each segment into a
 * ribbon of triangles, following three.js's `Line2NodeMaterial` (the webgpu/
 * TSL implementation, which this library's node graph mirrors).
 *
 * When `worldUnits` is `false` (the default) `linewidth` is measured in device
 * pixels and the ribbon is expanded in screen space; when `true` it is measured
 * in world units and the ribbon is expanded in camera space. `dashed` renders
 * dash/gap patterns from the geometry's accumulated line distances (see
 * `LineSegmentsGeometry.computeLineDistances`), and `vertexColors` multiplies
 * the color by per-segment start/end colors (see `LineSegmentsGeometry.setColors`).
 *
 * The line shader runs on both the WebGL2 and WebGPU renderers. `alphaToCoverage`
 * is accepted for three.js parity but currently selects the MSAA-less discard
 * variant (the RMSL renderers expose no sample count yet).
 */
export class Line2NodeMaterial extends NodeMaterial {
  readonly isLine2NodeMaterial = true;

  color = new Color(1, 1, 1);
  linewidth = 1;
  dashSize = 1;
  gapSize = 1;
  dashOffset = 0;
  dashScale = 1;

  lineWidthNode?: SlotValue<Node<"float">>;
  dashSizeNode?: SlotValue<Node<"float">>;
  gapSizeNode?: SlotValue<Node<"float">>;
  dashOffsetNode?: SlotValue<Node<"float">>;
  dashScaleNode?: SlotValue<Node<"float">>;
  offsetNode?: SlotValue<Node<"float">>;

  protected colorUniform?: UniformNode<"vec3">;
  protected lineWidthUniform?: UniformNode<"float">;
  protected resolutionUniform?: UniformNode<"vec2">;
  protected dashScaleUniform?: UniformNode<"float">;
  protected dashSizeUniform?: UniformNode<"float">;
  protected gapSizeUniform?: UniformNode<"float">;
  protected dashOffsetUniform?: UniformNode<"float">;

  private _worldUnits = false;
  private _dashed = false;
  private _vertexColors = false;
  private _alphaToCoverage = true;

  constructor(parameters: {
    color?: Color | number;
    linewidth?: number;
    worldUnits?: boolean;
    dashed?: boolean;
    dashSize?: number;
    gapSize?: number;
    dashOffset?: number;
    dashScale?: number;
    vertexColors?: boolean;
    alphaToCoverage?: boolean;
    opacity?: number;
    transparent?: boolean;
    precision?: GLSLPrecision;
  } = {}) {
    super();
    this.side = Side.DoubleSide;
    if (parameters.color !== undefined) {
      this.color = typeof parameters.color === "number"
        ? new Color().setHex(parameters.color)
        : parameters.color.clone();
    }
    if (parameters.linewidth !== undefined) this.linewidth = parameters.linewidth;
    if (parameters.dashSize !== undefined) this.dashSize = parameters.dashSize;
    if (parameters.gapSize !== undefined) this.gapSize = parameters.gapSize;
    if (parameters.dashOffset !== undefined) this.dashOffset = parameters.dashOffset;
    if (parameters.dashScale !== undefined) this.dashScale = parameters.dashScale;
    if (parameters.worldUnits !== undefined) this._worldUnits = parameters.worldUnits;
    if (parameters.dashed !== undefined) this._dashed = parameters.dashed;
    if (parameters.vertexColors !== undefined) this._vertexColors = parameters.vertexColors;
    if (parameters.alphaToCoverage !== undefined) this._alphaToCoverage = parameters.alphaToCoverage;
    if (parameters.opacity !== undefined) this.opacity = parameters.opacity;
    if (parameters.transparent !== undefined) this.transparent = parameters.transparent;
    if (parameters.precision !== undefined) this.precision = parameters.precision;
  }

  /** Whether the lines are sized in world units (`true`) or pixels (`false`). */
  get worldUnits(): boolean { return this._worldUnits; }
  set worldUnits(value: boolean) {
    if (this._worldUnits !== value) { this._worldUnits = value; this.needsUpdate = true; }
  }

  /** Whether dashed line rendering is enabled. */
  get dashed(): boolean { return this._dashed; }
  set dashed(value: boolean) {
    if (this._dashed !== value) { this._dashed = value; this.needsUpdate = true; }
  }

  /** Whether the per-segment colors from `instanceColorStart/End` are used. */
  get vertexColors(): boolean { return this._vertexColors; }
  set vertexColors(value: boolean) {
    if (this._vertexColors !== value) { this._vertexColors = value; this.needsUpdate = true; }
  }

  /**
   * Whether alpha-to-coverage smoothing is requested. RMSL's renderers expose
   * no MSAA sample count, so this currently compiles to the hard-discard
   * variant (the `samples === 0` behaviour in three.js).
   */
  get alphaToCoverage(): boolean { return this._alphaToCoverage; }
  set alphaToCoverage(value: boolean) {
    if (this._alphaToCoverage !== value) { this._alphaToCoverage = value; this.needsUpdate = true; }
  }

  protected setup(b: Builder, _scene: Scene): void {
    this.colorUniform = b.materialUniform("materialColor", "vec3", () => this.color.toArray());
    this.lineWidthUniform = b.materialUniform("materialLineWidth", "float", () => this.linewidth);
    this.resolutionUniform = b.rendererUniform("resolution", "vec2");
    if (this._dashed) {
      this.dashScaleUniform = b.materialUniform("materialLineScale", "float", () => this.dashScale);
      this.dashSizeUniform = b.materialUniform("materialLineDashSize", "float", () => this.dashSize);
      this.gapSizeUniform = b.materialUniform("materialLineGapSize", "float", () => this.gapSize);
      this.dashOffsetUniform = b.materialUniform("materialLineDashOffset", "float", () => this.dashOffset);
    }
  }

  protected buildVertexBody(b: Builder): Node<"vec4"> {
    const posGeo = b.position;

    // The first statements reference the geometry attributes in the same order
    // they are registered (position, uv, instanceStart, instanceEnd, ...) so
    // the WebGPU attribute locations the compiler assigns by traversal order
    // line up with the vertex buffer slots the renderer binds.
    const quadY = posGeo.y.toVar("quadY");
    const quadX = posGeo.x.toVar("quadX");
    b.uvVarying.assign(b.uv);

    const instanceStart = b.attribute("instanceStart", "vec3", "instance");
    const start = vec4(b.modelViewMatrix.mul(vec4(instanceStart, 1))).toVar("start");

    const instanceEnd = b.attribute("instanceEnd", "vec3", "instance");
    const end = vec4(b.modelViewMatrix.mul(vec4(instanceEnd, 1))).toVar("end");

    let distanceStart: Node<"float"> | undefined;
    let distanceEnd: Node<"float"> | undefined;
    if (this._dashed) {
      distanceStart = b.attribute("instanceDistanceStart", "float", "instance").toVar("distanceStart");
      distanceEnd = b.attribute("instanceDistanceEnd", "float", "instance").toVar("distanceEnd");
    }

    if (this._worldUnits) {
      b.varying("worldStart", "vec3").assign(start.xyz);
      b.varying("worldEnd", "vec3").assign(end.xyz);
    }

    // Trim segments that cross the camera near plane (perspective only).
    const perspective = element(element(b.projectionMatrix, 2), 3).equal(-1.0);
    If(perspective, () => {
      If(start.z.lessThan(0.0).and(end.z.greaterThan(0.0)), () => {
        const alpha = trimSegmentAlpha(b, start, end);
        end.assign(vec4(mix(start.xyz, end.xyz, alpha), end.w));
        if (distanceStart && distanceEnd) distanceEnd.assign(mix(distanceStart, distanceEnd, alpha));
      }).ElseIf(end.z.lessThan(0.0).and(start.z.greaterThanEqual(0.0)), () => {
        const alpha = trimSegmentAlpha(b, end, start);
        start.assign(vec4(mix(end.xyz, start.xyz, alpha), start.w));
        if (distanceStart && distanceEnd) distanceStart.assign(mix(distanceEnd, distanceStart, alpha));
      });
    });

    if (this._dashed) {
      const dashScale = resolveSlot(this.dashScaleNode, b) ?? this.dashScaleUniform!;
      const offset = resolveSlot(this.offsetNode, b) ?? this.dashOffsetUniform!;
      const lineDist = quadY.lessThan(0.5)
        .select(dashScale.mul(distanceStart!), dashScale.mul(distanceEnd!))
        .add(offset);
      b.varying("lineDistance", "float").assign(lineDist);
    }

    // clip space
    const clipStart = b.projectionMatrix.mul(start);
    const clipEnd = b.projectionMatrix.mul(end);

    // ndc space [ - 1.0, 1.0 ]
    const ndcStart = clipStart.xyz.div(clipStart.w);
    const ndcEnd = clipEnd.xyz.div(clipEnd.w);

    const lineWidth = resolveSlot(this.lineWidthNode, b) ?? this.lineWidthUniform!;
    const clip = vec4().toVar("clip");

    if (this._worldUnits) {
      // get the offset direction as perpendicular to the view vector
      const worldDir = end.xyz.sub(start.xyz).normalize();
      const tmpFwd = mix(start.xyz, end.xyz, 0.5).normalize();
      const worldUp = worldDir.cross(tmpFwd).normalize();
      const worldFwd = worldDir.cross(worldUp);

      const worldPos = b.varying("worldPos", "vec4");
      worldPos.assign(quadY.lessThan(0.5).select(start, end));

      // height offset
      const halfWidth = lineWidth.mul(0.5);
      worldPos.assign(worldPos.add(vec4(quadX.lessThan(0.0)
        .select(worldUp.mul(halfWidth), worldUp.mul(halfWidth).negate()), 0)));

      // don't extend the line when rendering dashes — the endcaps are dropped
      if (!this._dashed) {
        // cap extension
        worldPos.assign(worldPos.add(vec4(quadY.lessThan(0.5)
          .select(worldDir.mul(halfWidth).negate(), worldDir.mul(halfWidth)), 0)));
        // add width to the box
        worldPos.assign(worldPos.add(vec4(worldFwd.mul(halfWidth), 0)));
        // endcaps
        If(quadY.greaterThan(1.0).or(quadY.lessThan(0.0)), () => {
          worldPos.assign(worldPos.sub(vec4(worldFwd.mul(2.0).mul(halfWidth), 0)));
        });
      }

      // project the worldpos
      clip.assign(b.projectionMatrix.mul(worldPos));

      // shift the depth of the projected points so the line segments overlap neatly
      const clipPose = quadY.lessThan(0.5).select(ndcStart, ndcEnd).toVar("clipPose");
      clip.z.assign(clipPose.z.mul(clip.w));
    } else {
      const aspect = this.resolutionUniform!.x.div(this.resolutionUniform!.y);

      // direction, accounting for clip-space aspect ratio
      const dir = ndcEnd.xy.sub(ndcStart.xy).toVar("dir");
      dir.x.assign(dir.x.mul(aspect));
      dir.assign(dir.normalize());

      const offset = vec2(dir.y, dir.x.negate()).toVar("offset");

      // undo aspect ratio adjustment
      dir.x.assign(dir.x.div(aspect));
      offset.x.assign(offset.x.div(aspect));

      // sign flip
      offset.assign(quadX.lessThan(0.0).select(offset.negate(), offset));

      // endcaps
      If(quadY.lessThan(0.0), () => {
        offset.assign(offset.sub(dir));
      }).ElseIf(quadY.greaterThan(1.0), () => {
        offset.assign(offset.add(dir));
      });

      // adjust for linewidth, then for clip-space to screen-space conversion
      offset.assign(offset.mul(lineWidth));
      offset.assign(offset.div(this.resolutionUniform!.y));

      // select end, back to clip space
      clip.assign(quadY.lessThan(0.5).select(clipStart, clipEnd));
      offset.assign(offset.mul(clip.w));
      clip.assign(clip.add(vec4(offset, 0, 0)));
    }

    if (this._vertexColors) {
      const colorStart = b.attribute("instanceColorStart", "vec3", "instance");
      const colorEnd = b.attribute("instanceColorEnd", "vec3", "instance");
      b.varying("instanceColor", "vec3")
        .assign(quadY.lessThan(0.5).select(colorStart, colorEnd));
    }

    return clip;
  }

  protected buildFragmentBody(b: Builder): Node<"vec4"> {
    const vUv = b.uv;

    if (this._dashed) {
      const dashSize = resolveSlot(this.dashSizeNode, b) ?? this.dashSizeUniform!;
      const gapSize = resolveSlot(this.gapSizeNode, b) ?? this.gapSizeUniform!;
      const lineDistance = b.varying("lineDistance", "float");

      // discard the endcaps, then the gaps
      If(vUv.y.lessThan(-1.0).or(vUv.y.greaterThan(1.0)), () => { Discard(); });
      If(lineDistance.mod(dashSize.add(gapSize)).greaterThan(dashSize), () => { Discard(); });
    }

    const lineWidth = resolveSlot(this.lineWidthNode, b) ?? this.lineWidthUniform!;

    if (this._worldUnits) {
      const worldStart = b.varying("worldStart", "vec3");
      const worldEnd = b.varying("worldEnd", "vec3");
      const worldPos = b.varying("worldPos", "vec4");

      // Find the closest points on the view ray and the line segment.
      const rayEnd = worldPos.xyz.normalize().mul(1e5);
      const lineDir = worldEnd.sub(worldStart);
      const params = closestLineToLine(worldStart, worldEnd, vec3(0, 0, 0), rayEnd);

      const p1 = worldStart.add(lineDir.mul(params.x));
      const p2 = rayEnd.mul(params.y);
      const norm = p1.sub(p2).length().div(lineWidth);

      // Alpha-to-coverage would smooth the edge under MSAA; without a sample
      // count the hard discard variant is used (three.js with samples === 0).
      If(norm.greaterThan(0.5), () => { Discard(); });
    } else {
      // round endcaps
      If(vUv.y.abs().greaterThan(1.0), () => {
        const a = vUv.x;
        const bb = vUv.y.greaterThan(0.0).select(vUv.y.sub(1.0), vUv.y.add(1.0));
        const len2 = a.mul(a).add(bb.mul(bb));
        If(len2.greaterThan(1.0), () => { Discard(); });
      });
    }

    let color: Node<"vec3"> = resolveSlot(this.colorNode, b) ?? this.colorUniform!;
    if (this._vertexColors) {
      color = color.mul(b.varying("instanceColor", "vec3"));
    }
    const opacity = resolveSlot(this.opacityNode, b) ?? float(this.opacity);
    return vec4(color, opacity);
  }
}

/**
 * How far along the view-space segment the near plane cuts it, from the
 * projection matrix's near estimate — three.js's `trimSegmentAlpha`.
 */
function trimSegmentAlpha(b: Builder, start: Node<"vec4">, end: Node<"vec4">): Node<"float"> {
  const a = element(element(b.projectionMatrix, 2), 2);
  const bb = element(element(b.projectionMatrix, 3), 2);
  // `a` is positive with a reversed depth buffer, so it picks the branch.
  const nearEstimate = a.greaterThan(0)
    .select(bb.negate().div(a.add(1)), bb.mul(-0.5).div(a));
  return nearEstimate.sub(start.z).div(end.z.sub(start.z));
}

/**
 * The closest points on two 3D lines, as parametric coordinates — three.js's
 * `closestLineToLine`. Used for the world-units distance check.
 */
function closestLineToLine(
  p1: Node<"vec3">, p2: Node<"vec3">,
  p3: Node<"vec3">, p4: Node<"vec3">,
): Node<"vec2"> {
  const p13 = p1.sub(p3);
  const p43 = p4.sub(p3);
  const p21 = p2.sub(p1);

  const d1343 = p13.dot(p43);
  const d4321 = p43.dot(p21);
  const d1321 = p13.dot(p21);
  const d4343 = p43.dot(p43);
  const d2121 = p21.dot(p21);

  const denom = d2121.mul(d4343).sub(d4321.mul(d4321));
  const numer = d1343.mul(d4321).sub(d1321.mul(d4343));

  const mua = numer.div(denom).clamp(float(0), float(1));
  const mub = d1343.add(d4321.mul(mua)).div(d4343).clamp(float(0), float(1));

  return vec2(mua, mub);
}
