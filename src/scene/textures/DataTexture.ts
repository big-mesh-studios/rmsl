import { Texture } from "./Texture";
import { RGBAFormat, UnsignedByteType } from "./constants";

/** A texture built from an in-memory pixel buffer, like three.js's `DataTexture`. */
export class DataTexture extends Texture {
  width: number;
  height: number;
  /** Depth for a 3D texture; 1 for a 2D one. */
  depth: number;
  /** Texel layout of the image data, like three.js's `DataTexture` format. */
  format: number;
  /** Element type of the image data, like three.js's `DataTexture` type. */
  type: number;

  constructor(
    data: ArrayBufferView | null = null,
    width = 1,
    height = 1,
    depth = 1,
    format = RGBAFormat,
    type = UnsignedByteType,
  ) {
    super();
    this.image = data;
    this.width = width;
    this.height = height;
    this.depth = depth;
    this.format = format;
    this.type = type;
  }
}

