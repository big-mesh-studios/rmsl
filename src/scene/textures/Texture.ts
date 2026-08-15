/**
 * A texture to sample in a material, like three.js's `Texture`. The GPU object
 * is owned by the renderer and created when `needsUpdate` is set.
 */
export class Texture {
  readonly isTexture = true;

  name = "";
  image: TexImageSource | ArrayBufferView | null = null;
  needsUpdate = true;

  /** Set by the renderer to create or refresh the GPU resource. */
  userData: Record<string, unknown> = {};

  constructor(image: TexImageSource | ArrayBufferView | null = null) {
    this.image = image;
  }
}
