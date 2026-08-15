import { f, type FloatIn, type IntIn, type Vec2In, type Sampler2D, type Sampler3D } from "./util";
import {mul, smoothstep, sub, uv, vec2, type Node} from "../rmsl";

/**
 * Returns a radial gradient from center (white) to edges (black). Useful for
 * masking effects based on distance from center.
 *
 * Ported from three.js's `examples/jsm/tsl/display/Shape.js`.
 *
 * @param scale - Controls the size of the gradient (0 = all black, 1 = full circle).
 * @param softness - Controls the edge softness (0 = hard edge, 1 = soft gradient).
 * @param coord - The input UV coordinates; defaults to the screen uv.
 * @return 1.0 at center, 0.0 at edges.
 */
export const circle = (
  scale: FloatIn = 1.0,
  softness: FloatIn = 0.5,
  coord: Vec2In | null = null,
): Node<"float"> => {
  const c = coord === null
    ? uv()
    : Array.isArray(coord)
      ? vec2(coord[0], coord[1])
      : coord;
  // Center the UV coordinates (-0.5 to 0.5).
  const centered = c.sub(0.5);
  // Distance from center (0 at center, ~0.707 at corners).
  const dist = centered.length().mul(2.0);
  // Inner and outer edges based on scale and softness.
  const outer = scale;
  const inner = sub(scale, mul(softness, scale));
  return smoothstep(outer, inner, dist);
};
