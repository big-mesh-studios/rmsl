import { isIntegerSamplerType } from "../../core";
import type { Texture } from "../textures/Texture";
import {
  MirroredRepeatWrapping,
  NearestFilter,
  NearestMipmapLinearFilter,
  NearestMipmapNearestFilter,
  RedIntegerFormat,
  RepeatWrapping,
} from "../textures/constants";

/**
 * How a texture is sampled, in terms neither backend's spelling: what to do
 * between texels, and what to do outside the image.
 *
 * Both renderers read the same three.js-style fields off a `Texture` and then
 * spell the answer their own way — `texParameteri` constants in WebGL, sampler
 * descriptor strings in WebGPU — so the rule for turning one into the other
 * lives here once, where it can be tested without a graphics device.
 */
export interface SamplerState {
  magFilter: "nearest" | "linear";
  minFilter: "nearest" | "linear";
  wrapS: TextureWrap;
  wrapT: TextureWrap;
  wrapR: TextureWrap;
}

export type TextureWrap = "clamp" | "repeat" | "mirror";

/**
 * The sampler state a texture asks for, as a sampler of `samplerType` can
 * honour it.
 *
 * An integer texture is not filterable in either language, so it reads with
 * nearest whatever it asked for. A mipmapped minification filter is treated as
 * its base filter, because no renderer builds a mip chain: honouring it
 * literally would leave WebGL with an incomplete texture, which samples as
 * black — see https://github.com/big-mesh-studios/rmsl/issues/3.
 */
export function samplerState(texture: Texture, samplerType: string): SamplerState {
  const filterable = !isIntegerSampler(samplerType);
  return {
    magFilter: filterable ? textureFilter(texture.magFilter) : "nearest",
    minFilter: filterable ? textureFilter(texture.minFilter) : "nearest",
    wrapS: textureWrap(texture.wrapS),
    wrapT: textureWrap(texture.wrapT),
    wrapR: textureWrap(texture.wrapR),
  };
}

/**
 * How many channels a texel of `texture` holds, which is what says where one
 * texel ends and the next begins.
 *
 * A `RedIntegerFormat` view is single-channel — `R8UI` in WebGL, `r8uint` in
 * WebGPU, one byte a texel on the CPU — and every other format is four. The
 * channels a texel does not store read as a sampler reports them: zero for
 * green and blue, one for alpha.
 */
export function textureChannels(texture: Texture): 1 | 4 {
  return (texture as { format?: number }).format === RedIntegerFormat ? 1 : 4;
}

/** A `Texture` filter constant as the choice between texels it stands for. */
function textureFilter(filter: number): "nearest" | "linear" {
  switch (filter) {
    case NearestFilter:
    case NearestMipmapNearestFilter:
    case NearestMipmapLinearFilter:
      return "nearest";
    default:
      return "linear";
  }
}

/** A `Texture` wrapping constant as what it does outside the image. */
function textureWrap(wrap: number): TextureWrap {
  switch (wrap) {
    case RepeatWrapping:
      return "repeat";
    case MirroredRepeatWrapping:
      return "mirror";
    default:
      return "clamp";
  }
}

/** Whether a sampler type reads an integer texture (unfiltered texels). */
export function isIntegerSampler(type: string): boolean {
  return isIntegerSamplerType(type);
}
