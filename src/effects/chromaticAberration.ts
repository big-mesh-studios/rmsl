import { f, type FloatIn, type Sampler2D, type Vec2In } from "./util";
import { float, mul, uv, vec2, vec4, type Node } from "../rmsl";

/**
 * Post-processing effect simulating the color fringing that occurs in real
 * camera lenses by separating and offsetting the red, green and blue channels.
 *
 * Ported from three.js's `examples/jsm/tsl/display/ChromaticAberrationNode.js`.
 *
 * @param textureNode - The texture node that represents the input of the effect.
 * @param strength - The strength of the chromatic aberration effect.
 * @param center - The center point of the effect, in UV space. Defaults to screen center.
 * @param scale - The scale factor for stepped scaling from center.
 * @return The aberrated color.
 */
export const chromaticAberration = (
  textureNode: Sampler2D,
  strength: FloatIn = 1.0,
  center: Vec2In | null = null,
  scale: FloatIn = 1.1,
): Node<"vec4"> => {
  const centerNode = center === null
    ? vec2(0.5, 0.5)
    : Array.isArray(center)
      ? vec2(center[0], center[1])
      : center;
  const uvNode = uv();

  // Distance from center, and stepped scaling zones per channel.
  const offset = uvNode.sub(centerNode);
  const distance = offset.length();

  const redScale = float(1.0).add(mul(scale, 0.02).mul(strength));
  const greenScale = float(1.0);
  const blueScale = float(1.0).sub(mul(scale, 0.02).mul(strength));

  // Radial distortion based on distance from center.
  const aberrationStrength = mul(strength, distance);

  const redUV = centerNode.add(offset.mul(redScale)).add(offset.mul(aberrationStrength).mul(0.01));
  const greenUV = centerNode.add(offset.mul(greenScale));
  const blueUV = centerNode.add(offset.mul(blueScale)).sub(offset.mul(aberrationStrength).mul(0.01));

  const r = textureNode.texture(redUV).r;
  const g = textureNode.texture(greenUV).g;
  const b = textureNode.texture(blueUV).b;
  const a = textureNode.texture(uvNode).a;

  return vec4(r, g, b, a);
};
