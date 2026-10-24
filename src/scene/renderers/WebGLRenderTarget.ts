import { RenderTargetTexture } from "../textures/RenderTargetTexture";

/**
 * An offscreen drawing surface for `WebGLRenderer`, sized independently of its
 * canvas: a color texture (RGBA8) plus a depth renderbuffer, both created on
 * the renderer's context the first time the target is bound. Passing one to
 * `render(scene, camera, target)` redirects a draw away from the canvas, and
 * `readPixels(target)` pulls its color buffer back to the host — so a pass
 * nobody sees (a low-resolution colour-coded depth test whose readback decides
 * what the real pass draws) runs on the same context, sharing its programs,
 * buffers and textures instead of paying for a second renderer's worth.
 *
 * ## `texture`: drawing from a surface you drew into
 *
 * A readback brings pixels to the host, which is the right thing for a photo and the wrong
 * thing for a monitor: a 320×240 readback is 300 KB a frame and lands a frame or two late.
 * So the target also publishes its colour attachment as `texture`, which a material samples
 * directly — `new MeshBasicMaterial({ map: target.texture })` is the whole of it. The pass
 * into the target must finish before the pass that samples it, since sampling a texture that
 * is still attached to the bound framebuffer is undefined; `Renderer.render` restores the
 * canvas framebuffer on return, so ordering two `render` calls is enough.
 *
 * Resizing is by assigning `width`/`height`: the attachment is remade at the new size on the
 * next bind, and `texture` stays the same object, so a material that names it keeps working.
 */
export class WebGLRenderTarget {
  readonly isRenderTarget = true;
  width: number;
  height: number;
  /**
   * The colour attachment, for a material to sample. Contents are whatever the last pass into
   * this target drew, and are undefined before one has.
   */
  readonly texture: RenderTargetTexture;

  constructor(width = 1, height = 1) {
    this.width = width;
    this.height = height;
    this.texture = new RenderTargetTexture(this);
  }
}
