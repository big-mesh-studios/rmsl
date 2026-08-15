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
  /** Whether the renderer should rebuild this material's program. */
  needsUpdate = false;

  readonly isMaterial = true;
}
