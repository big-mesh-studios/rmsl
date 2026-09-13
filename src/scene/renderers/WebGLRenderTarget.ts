/**
 * An offscreen drawing surface for `WebGLRenderer`, sized independently of its
 * canvas: a color texture (RGBA8) plus a depth renderbuffer, both created on
 * the renderer's context the first time the target is bound. Passing one to
 * `render(scene, camera, target)` redirects a draw away from the canvas, and
 * `readPixels(target)` pulls its color buffer back to the host — so a pass
 * nobody sees (a low-resolution colour-coded depth test whose readback decides
 * what the real pass draws) runs on the same context, sharing its programs,
 * buffers and textures instead of paying for a second renderer's worth.
 */
export class WebGLRenderTarget {
  readonly isRenderTarget = true;
  width: number;
  height: number;

  constructor(width = 1, height = 1) {
    this.width = width;
    this.height = height;
  }
}
