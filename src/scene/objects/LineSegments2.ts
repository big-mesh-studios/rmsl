import { Mesh } from "./Mesh";
import { LineSegmentsGeometry } from "../geometries/LineSegmentsGeometry";
import { Line2NodeMaterial } from "../materials/Line2NodeMaterial";
import { Vector2 } from "../math/Vector2";
import { Vector4 } from "../math/Vector4";

/**
 * A series of lines drawn between pairs of vertices, like three.js's
 * `LineSegments2`: arbitrary line width, optionally in world units, via a
 * {@link LineSegmentsGeometry} fed to a {@link Line2NodeMaterial}. Renders on
 * both the WebGL2 and WebGPU renderers.
 */
export class LineSegments2 extends Mesh {
  readonly isLineSegments2 = true;

  type = "LineSegments2";

  declare geometry: LineSegmentsGeometry;
  declare material: Line2NodeMaterial;

  /** The drawing surface resolution, updated before every render. */
  readonly resolution = new Vector2();

  constructor(
    geometry: LineSegmentsGeometry = new LineSegmentsGeometry(),
    material: Line2NodeMaterial = new Line2NodeMaterial(),
  ) {
    super(geometry, material);
  }

  /**
   * Computes the accumulated length along the line and stores it in the
   * geometry's `instanceDistanceStart`/`instanceDistanceEnd` attributes — the
   * data the dash shader needs. Required before `material.dashed = true`.
   */
  computeLineDistances(): this {
    this.geometry.computeLineDistances();
    return this;
  }

  /** Called by the renderers before each draw; records the viewport size. */
  onBeforeRender = (renderer: unknown): void => {
    const viewport = (renderer as { getViewport(target?: Vector4): Vector4 }).getViewport(_viewport);
    this.resolution.set(viewport.z, viewport.w);
  };
}

const _viewport = new Vector4();
