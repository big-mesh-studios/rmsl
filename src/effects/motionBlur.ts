import { f, type FloatIn, type IntIn, type Vec2In, type Sampler2D, type Sampler3D } from "./util";
import {For, Fn, float, int, uv, type Node} from "../rmsl";

/**
 * Applies a motion blur effect to the given texture node.
 *
 * Ported from three.js's `examples/jsm/tsl/display/MotionBlur.js`. The motion
 * vector for the current pixel is a plain node — e.g. sampled from a velocity
 * texture by the caller.
 *
 * @param inputNode - The texture node to apply the motion blur for.
 * @param velocity - The motion vector of the beauty pass at this pixel.
 * @param numSamples - How many samples the effect should use. A higher value
 *                     results in better quality but is also more expensive.
 * @return The blurred color.
 */
export const motionBlur = (
  inputNode: Sampler2D,
  velocity: Node<"vec2">,
  numSamples: IntIn = 16,
): Node<"vec4"> => {
  const n = typeof numSamples === "number" ? int(numSamples) : numSamples;
  return Fn(() => {
    const uvs = uv();
    const colorResult = inputNode.texture(uvs).toVar();
    const fSamples = float(n);

    For(
      () => int(1).toVar(),
      (i) => i.lessThanEqual(n),
      (i) => { i.assign(i.add(1)); },
      (i) => {
        const offset = velocity.mul(float(i).div(fSamples.sub(1)).sub(0.5));
        colorResult.addAssign(inputNode.texture(uvs.add(offset)));
      },
    );

    colorResult.divAssign(fSamples);
    return colorResult;
  })();
};
