import type { GLSLPrecision } from "../../glsl";

/** Which faces a material draws. */
export enum Side {
  FrontSide,
  BackSide,
  DoubleSide,
}

/** How a material blends with what is already in the framebuffer. */
export enum Blending {
  NoBlending,
  NormalBlending,
  AdditiveBlending,
}

/**
 * The render-state base of a material, mirroring three.js: the flags a
 * renderer reads to set blending, depth and culling, plus a shared type
 * marker so renderers can tell materials apart.
 */
export class Material {
  name = "";
  side: Side = Side.FrontSide;
  transparent = false;
  opacity = 1;
  depthTest = true;
  depthWrite = true;
  blending: Blending = Blending.NormalBlending;
  /**
   * Overrides the renderer's default shader precision for this material, as in
   * three.js (`null` = use the renderer's). `null` rather than `undefined` so
   * the rendering code can tell "not set" from the property being absent.
   * Assigning a value flags the program for a rebuild.
   */
  get precision(): GLSLPrecision | null {
    return this._precision;
  }
  set precision(value: GLSLPrecision | null) {
    if (this._precision !== value) {
      this._precision = value;
      this.needsUpdate = true;
    }
  }

  private _precision: GLSLPrecision | null = null;
  /** How many times the material was marked changed, as three.js counts it: a renderer rebuilds a program built from an earlier version. */
  version = 0;

  /** Marks the material changed, raising `version`, as in three.js. Reading it gives `undefined`, as in three.js. */
  set needsUpdate(value: boolean) {
    if (value) this.version++;
  }

  readonly isMaterial = true;
}
