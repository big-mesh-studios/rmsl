import {For, Fn, cos, degrees, div, float, mul, premultiplyAlpha, rand, sin, unpremultiplyAlpha, uv, vec2, vec4, type Node} from "../rmsl";
import { f, type FloatIn, type IntIn, type Vec2In, type Sampler2D, type Sampler3D } from "./util";

export interface HashBlurOptions {
  /** The number of iterations for the blur effect. */
  repeats?: FloatIn;
  /** Whether to blur in premultiplied alpha. */
  premultipliedAlpha?: boolean;
}

/**
 * Applies a hash blur to the given texture node.
 *
 * Unlike a kernel-based blur, the base texture is read multiple times in a
 * random pattern and the samples averaged. A slightly noisy look is typical
 * and can be mitigated by raising `repeats`. Requires a single pass.
 *
 * Ported from three.js's `examples/jsm/tsl/display/hashBlur.js`.
 *
 * @param textureNode - The texture node to blur.
 * @param bluramount - The amount of blur.
 * @param options - Options for the effect.
 * @return The blurred color.
 */
export const hashBlur = (
  textureNode: Sampler2D,
  bluramount: FloatIn = 0.1,
  options: HashBlurOptions = {},
): Node<"vec4"> => {
  const repeats = options.repeats ?? 45;
  const premultipliedAlpha = options.premultipliedAlpha ?? false;

  const tap = (u: Node<"vec2">): Node<"vec4"> => {
    const sample = textureNode.texture(u);
    return premultipliedAlpha ? premultiplyAlpha(sample) : sample;
  };

  return Fn(() => {
    const targetUV = uv();
    const blurred = vec4(0).toVar();
    const repeatsF = f(repeats);

    For(
      () => float(0).toVar(),
      (i) => i.lessThan(repeatsF),
      (i) => { i.assign(i.add(1)); },
      (i) => {
        const angle = degrees(mul(div(i, repeatsF), 360));
        const dir = vec2(cos(angle), sin(angle));
        const q = dir.mul(rand(vec2(i, targetUV.x.add(targetUV.y))).add(bluramount));
        blurred.addAssign(tap(targetUV.add(q.mul(bluramount))));
      },
    );

    blurred.divAssign(repeatsF);
    return premultipliedAlpha ? unpremultiplyAlpha(blurred) : blurred;
  })();
};
