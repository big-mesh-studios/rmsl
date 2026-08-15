import {For, Fn, float, int, max, premultiplyAlpha, textureSize, unpremultiplyAlpha, uv, vec2, vec4, type Node} from "../rmsl";
import { f, type FloatIn, type IntIn, type Vec2In, type Sampler2D, type Sampler3D } from "./util";

export interface BoxBlurOptions {
  /** Controls the blur's kernel; keep within [1, 3] for performance. */
  size?: IntIn;
  /** Spreads the blur out without extra samples. */
  separation?: IntIn;
  /** Whether to blur in premultiplied alpha. */
  premultipliedAlpha?: boolean;
}

/**
 * Applies a box blur to the given texture node.
 *
 * Compared to a gaussian blur, a box blur is more blocky but cheaper when
 * correctly configured — the kernel is (size * 2 + 1)² samples in a single
 * pass. Widen the effect with `separation` instead of `size`, since it does
 * not cost extra samples.
 *
 * Ported from three.js's `examples/jsm/tsl/display/boxBlur.js`.
 *
 * @param textureNode - The texture node to blur.
 * @param options - Options for the effect.
 * @return The blurred color.
 */
export const boxBlur = (
  textureNode: Sampler2D,
  options: BoxBlurOptions = {},
): Node<"vec4"> => {
  const size = typeof options.size === "number" ? int(options.size) : (options.size ?? int(1));
  const separation = options.separation ?? 1;
  const premultipliedAlpha = options.premultipliedAlpha ?? false;

  const tap = (u: Node<"vec2">): Node<"vec4"> => {
    const sample = textureNode.texture(u);
    return premultipliedAlpha ? premultiplyAlpha(sample) : sample;
  };

  return Fn(() => {
    const targetUV = uv();
    const result = vec4(0).toVar();
    const sep = float(max(separation, 1));
    const count = int(0).toVar();
    const pixelStep = vec2(1).div(textureSize(textureNode).toVec2());

    For(
      () => size.negate().toVar(),
      (i) => i.lessThanEqual(size),
      (i) => { i.assign(i.add(1)); },
      (i) => {
        For(
          () => size.negate().toVar(),
          (j) => j.lessThanEqual(size),
          (j) => { j.assign(j.add(1)); },
          (j) => {
            const uvs = targetUV.add(vec2(i.toFloat(), j.toFloat()).mul(pixelStep).mul(sep));
            result.addAssign(tap(uvs));
            count.addAssign(1);
          },
        );
      },
    );

    result.divAssign(float(count));
    return premultipliedAlpha ? unpremultiplyAlpha(result) : result;
  })();
};
