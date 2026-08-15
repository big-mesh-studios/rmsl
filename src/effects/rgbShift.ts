import { f, type FloatIn, type IntIn, type Vec2In, type Sampler2D, type Sampler3D } from "./util";
import {cos, sin, uv, vec2, vec4, type Node} from "../rmsl";

/**
 * Post-processing effect that shifts/splits the RGB color channels, offsetting
 * them from each other.
 *
 * Ported from three.js's `examples/jsm/tsl/display/RGBShiftNode.js`.
 *
 * @param textureNode - The texture node that represents the input of the effect.
 * @param amount - The amount of the RGB shift.
 * @param angle - Defines in which direction colors are shifted.
 * @return The shifted color.
 */
export const rgbShift = (
  textureNode: Sampler2D,
  amount: FloatIn = 0.005,
  angle: FloatIn = 0,
): Node<"vec4"> => {
  const offset = vec2(cos(angle), sin(angle)).mul(amount);
  const uvNode = uv();
  const cr = textureNode.texture(uvNode.add(offset));
  const cga = textureNode.texture(uvNode);
  const cb = textureNode.texture(uvNode.sub(offset));
  return vec4(cr.r, cga.g, cb.b, cga.a);
};
