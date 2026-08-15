import {Fn, Loop, float, interleavedGradientNoise, mix, premultiplyAlpha, screenCoordinate, unpremultiplyAlpha, uv, vec2, vec4, type Node} from "../rmsl";
import { f, type FloatIn, type IntIn, type Vec2In, type Sampler2D, type Sampler3D } from "./util";

export interface RadialBlurOptions {
  /** The center of the light in screen UVs. */
  center?: [number, number] | Node<"vec2">;
  /** Base weight factor for each sample in `[0, 1]`. */
  weight?: FloatIn;
  /** Decreases the weight each iteration; must stay in `[0, 1]`. */
  decay?: FloatIn;
  /** The number of iterations; should be in `[16, 64]`. */
  count?: IntIn;
  /** Exposure control of the blur. */
  exposure?: FloatIn;
  /** Whether to blur in premultiplied alpha. */
  premultipliedAlpha?: boolean;
}

/**
 * Blurs an image in a circular pattern radiating from a configurable center,
 * in screen space. Useful for faking light shafts.
 *
 * Ported from three.js's `examples/jsm/tsl/display/radialBlur.js`.
 *
 * @param textureNode - The texture node to blur.
 * @param options - Options for the effect.
 * @return The blurred color.
 */
export const radialBlur = (
  textureNode: Sampler2D,
  options: RadialBlurOptions = {},
): Node<"vec4"> => {
  const center = options.center ?? [0.5, 0.5];
  const weight = options.weight ?? 0.9;
  const decay = options.decay ?? 0.95;
  const count = options.count ?? 32;
  const exposure = options.exposure ?? 5;
  const premultipliedAlpha = options.premultipliedAlpha ?? false;

  const tap = (u: Node<"vec2">): Node<"vec4"> => {
    const sample = textureNode.texture(u);
    return premultipliedAlpha ? premultiplyAlpha(sample) : sample;
  };

  const centerNode = Array.isArray(center) ? vec2(center[0], center[1]) : center;

  return Fn(() => {
    const sampleUv = uv().toVar();
    const base = tap(sampleUv);
    const blur = vec4().toVar();
    const offset = centerNode.sub(sampleUv).div(float(count));
    const w = f(weight).toVar();

    const noise = interleavedGradientNoise(screenCoordinate());
    sampleUv.addAssign(offset.mul(noise)); // mitigate banding

    Loop(count, () => {
      sampleUv.addAssign(offset);
      blur.addAssign(tap(sampleUv).mul(w));
      w.mulAssign(decay);
    });

    blur.divAssign(float(count));
    blur.mulAssign(exposure);

    const color = mix(blur, base.mul(2), 0.5);
    return premultipliedAlpha ? unpremultiplyAlpha(color) : color;
  })();
};
