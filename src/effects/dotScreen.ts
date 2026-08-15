import { f, type FloatIn, type IntIn, type Vec2In, type Sampler2D, type Sampler3D } from "./util";
import {add, cos, screenCoordinate, sin, vec2, vec3, vec4, type Node} from "../rmsl";

/**
 * Post-processing effect creating a dot-screen pattern.
 *
 * Ported from three.js's `examples/jsm/tsl/display/DotScreenNode.js`.
 *
 * @param color - The node that represents the input of the effect.
 * @param angle - The rotation of the effect in radians.
 * @param scale - The scale of the effect. A higher value means smaller dots.
 * @return The dot-screened color.
 */
export const dotScreen = (
  color: Node<"vec4">,
  angle: FloatIn = 1.57,
  scale: FloatIn = 1,
): Node<"vec4"> => {
  const s = sin(angle);
  const c = cos(angle);
  const tex = screenCoordinate();
  const point = vec2(c.mul(tex.x).sub(s.mul(tex.y)), s.mul(tex.x).add(c.mul(tex.y))).mul(scale);
  const pattern = sin(point.x).mul(sin(point.y)).mul(4);
  const average = add(color.r, color.g, color.b).div(3);
  return vec4(vec3(average.mul(10).sub(5).add(pattern)), color.a);
};
