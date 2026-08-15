import { f, type FloatIn, type IntIn, type Vec2In, type Sampler2D, type Sampler3D } from "./util";
import {dot, vec3, vec4, type Node} from "../rmsl";

/**
 * Applies a sepia effect to the given color node.
 *
 * Ported from three.js's `examples/jsm/tsl/display/Sepia.js`.
 *
 * @param color - The color node to apply the sepia for.
 * @return The updated color node.
 */
export const sepia = (color: Node<"vec4">): Node<"vec4"> => {
  const c = vec3(color);
  // https://github.com/evanw/glfx.js/blob/master/src/filters/adjust/sepia.js
  return vec4(
    dot(c, vec3(0.393, 0.769, 0.189)),
    dot(c, vec3(0.349, 0.686, 0.168)),
    dot(c, vec3(0.272, 0.534, 0.131)),
    color.a,
  );
};
