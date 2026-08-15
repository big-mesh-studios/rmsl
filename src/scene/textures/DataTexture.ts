import { Texture } from "./Texture";

/** A texture built from an in-memory pixel buffer, like three.js's `DataTexture`. */
export class DataTexture extends Texture {
  width: number;
  height: number;

  constructor(data: ArrayBufferView | null = null, width = 1, height = 1) {
    super();
    this.image = data;
    this.width = width;
    this.height = height;
  }
}
