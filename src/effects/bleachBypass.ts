import {float, max, min, mix, oneMinus, vec3, vec4, luminance, type Node} from "../rmsl";
import { f, type FloatIn, type IntIn, type Vec2In, type Sampler2D, type Sampler3D } from "./util";

/**
 * Applies a bleach bypass effect to the given color node.
 *
 * Ported from three.js's `examples/jsm/tsl/display/BleachBypass.js`.
 *
 * @param color - The color node to apply the effect for.
 * @param opacity - How strongly the effect is blended with the original color.
 * @return The updated color node.
 */
export const bleach = (color: Node<"vec4">, opacity: FloatIn = 1): Node<"vec4"> => {
  const base = color;
  const lum = luminance(base.rgb);
  const blend = vec3(lum);
  const L = min(1.0, max(0.0, float(10.0).mul(lum.sub(0.45))));
  const result1 = blend.mul(base.rgb).mul(2.0);
  const result2 = oneMinus(float(2.0).mul(oneMinus(blend)).mul(oneMinus(base.rgb)));
  const newColor = mix(result1, result2, L);
  const A2 = base.a.mul(f(opacity));
  const mixRGB = A2.mul(newColor.rgb).add(base.rgb.mul(oneMinus(A2)));
  return vec4(mixRGB, base.a);
};
