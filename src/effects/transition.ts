import {Fn, If, add, clamp, div, equal, int, mix, mul, sub, uv, vec4, type Node} from "../rmsl";
import { f, type FloatIn, type IntIn, type Vec2In, type Sampler2D, type Sampler3D } from "./util";

/**
 * Post-processing effect creating a transition between two scenes.
 *
 * Ported from three.js's `examples/jsm/tsl/display/TransitionNode.js`.
 *
 * @param textureNodeA - The beauty pass of the first scene.
 * @param textureNodeB - The beauty pass of the second scene.
 * @param mixTextureNode - A texture that defines how the transition looks; pass
 *                         `null` to get a plain cross-fade.
 * @param mixRatio - The interpolation factor that controls the mix.
 * @param threshold - Tweaks the linear interpolation.
 * @param useTexture - Whether `mixTextureNode` should influence the transition.
 * @return The mixed color.
 */
export const transition = (
  textureNodeA: Sampler2D,
  textureNodeB: Sampler2D,
  mixTextureNode: Sampler2D | null,
  mixRatio: FloatIn,
  threshold: FloatIn,
  useTexture: FloatIn,
): Node<"vec4"> => {
  const useMixTexture = equal(useTexture, int(1));
  return Fn(() => {
    const uvNode = uv();
    const texelOne = textureNodeA.texture(uvNode);
    const texelTwo = textureNodeB.texture(uvNode);
    const color = vec4().toVar();

    If(useMixTexture, () => {
      const transitionTexel = mixTextureNode!.texture(uvNode);
      const r = sub(mul(mixRatio, add(mul(threshold, 2.0), 1.0)), threshold);
      const mixf = clamp(mul(transitionTexel.r.sub(r), div(1.0, threshold)), 0.0, 1.0);
      color.assign(mix(texelOne, texelTwo, mixf));
    }).Else(() => {
      color.assign(mix(texelTwo, texelOne, mixRatio));
    });

    return color;
  })();
};
