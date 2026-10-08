import type { CpuTextureData } from "./cpu";

/** A texture's texels as a GPU target uploads them, four channels each. */
export interface TextureImage {
  width: number;
  height: number;
  /** Layers of a 3D texture, and one for a 2D one. */
  depth: number;
  texels: Uint8Array | Int8Array | Uint16Array | Int16Array | Uint32Array | Int32Array;
  /** Bits in one channel. */
  bits: 8 | 16 | 32;
  signed: boolean;
  /** Whether a channel is read as 0 to 1, which is how a float sampler reads 8-bit data. */
  normalized: boolean;
}

const INTEGER_ARRAYS = [Int8Array, Uint8Array, Int16Array, Uint16Array, Int32Array, Uint32Array] as const;

/**
 * The texels of `texture` for a GPU target to upload to a sampler of
 * `samplerType`, expanded to four channels.
 *
 * A float sampler reads 8-bit data as 0 to 1, as the CPU targets do, so its
 * texture holds a `Uint8Array` or `Uint8ClampedArray`. An integer sampler holds
 * an integer array of the signedness it names. A channel the texture does not
 * store reads as a device reads it: zero for the second and third, one for the
 * fourth.
 *
 * @throws When the data is not an array that sampler type can read.
 */
export function textureImage(texture: CpuTextureData, samplerType: string): TextureImage {
  if (samplerType.endsWith("Cube")) throw new Error(`[RMSL] setTexture: a ${samplerType} texture is not supported`);
  const data = texture.data as ArrayBufferView & ArrayLike<number>;
  const integer = samplerType.startsWith("isampler") || samplerType.startsWith("usampler");
  const signed = samplerType.startsWith("isampler");
  let bits: 8 | 16 | 32;
  let normalized = false;
  if (!integer) {
    if (!(data instanceof Uint8Array || data instanceof Uint8ClampedArray)) {
      throw new Error(`[RMSL] setTexture: a ${samplerType} texture takes Uint8Array data on a GPU target`);
    }
    bits = 8;
    normalized = true;
  } else {
    const kind = INTEGER_ARRAYS.find((array) => data instanceof array);
    if (!kind || (kind.name.startsWith("Int") ? !signed : signed)) {
      throw new Error(
        `[RMSL] setTexture: a ${samplerType} texture takes ${signed ? "Int8Array, Int16Array or Int32Array" : "Uint8Array, Uint16Array or Uint32Array"} data`,
      );
    }
    bits = (kind.BYTES_PER_ELEMENT * 8) as 8 | 16 | 32;
  }
  const depth = texture.depth ?? 1;
  const count = texture.width * texture.height * depth;
  const channels = texture.channels ?? 4;
  if (data.length < count * channels) {
    throw new Error(
      `[RMSL] setTexture: a ${texture.width}x${texture.height}x${depth} texture of ${channels} channels needs ${count * channels} values, and got ${data.length}`,
    );
  }
  const Texels = (data instanceof Uint8ClampedArray ? Uint8Array : data.constructor) as new (
    ...args: [number] | [ArrayBufferLike, number, number]
  ) => TextureImage["texels"];
  // Four-channel data is uploaded as it is: a view of the same bytes, not a copy.
  if (channels === 4) {
    return {
      width: texture.width,
      height: texture.height,
      depth,
      texels: new Texels(data.buffer, data.byteOffset, count * 4),
      bits,
      signed,
      normalized,
    };
  }
  const texels = new Texels(count * 4);
  const alpha = normalized ? 255 : 1;
  for (let i = 0; i < count; i++) {
    for (let c = 0; c < 4; c++) {
      texels[i * 4 + c] = c < channels ? (data[i * channels + c] ?? 0) : c === 3 ? alpha : 0;
    }
  }
  return { width: texture.width, height: texture.height, depth, texels, bits, signed, normalized };
}
