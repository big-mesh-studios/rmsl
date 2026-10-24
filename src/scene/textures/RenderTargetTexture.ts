import { Texture } from "./Texture";
import type { WebGLRenderTarget } from "../renderers/WebGLRenderTarget";

/**
 * A texture whose image is a render target's colour attachment.
 *
 * ## Why it exists
 *
 * A render target is a surface a pass draws into, and the interesting thing about a surface
 * is that something can then be drawn *from* it: a monitor showing a camera, a mirror, a
 * thumbnail of the frame before last. Every renderer here keeps a target's colour attachment
 * as a GPU texture it made itself, so the only missing piece was a `Texture` a material could
 * name that says "bind the attachment, do not upload an image".
 *
 * ## It has no image and never raises `version`
 *
 * **Because its pixels do not arrive, they are drawn.** `image` is `null` and `version` stays
 * at zero: a renderer that did upload from it would upload nothing, and one that compared
 * versions would draw a stale frame. A renderer recognises it by `isRenderTargetTexture` and
 * binds `target`'s colour attachment instead — which is also why the filtering and wrapping
 * fields on this class are ignored: the attachment's own state was set when it was made.
 *
 * **The target outlives this texture and owns the GL texture**, so disposing this one must not
 * free anything. `WebGLRenderer.onTextureDispose` finds no entry for it and does nothing, which
 * is what makes `RenderTargetTexture` safe to share among materials and to drop without a
 * `dispose` call.
 */
export class RenderTargetTexture extends Texture {
  readonly isRenderTargetTexture = true;

  constructor(readonly target: WebGLRenderTarget) {
    super(null);
  }
}
