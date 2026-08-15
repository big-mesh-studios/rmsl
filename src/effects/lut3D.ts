import { f, type FloatIn, type IntIn, type Vec2In, type Sampler2D, type Sampler3D } from "./util";
import {float, mix, vec3, vec4, type Node} from "../rmsl";

/**
 * Post-processing effect for color grading via a 3D lookup table.
 *
 * Ported from three.js's `examples/jsm/tsl/display/Lut3DNode.js`. The LUT is a
 * `sampler3D` uniform the host must bind; `size` is the number of LUT tiles
 * along one edge.
 *
 * @param inputNode - The node that represents the input of the effect.
 * @param lutNode - A `sampler3D` that represents the lookup table.
 * @param size - The size of the lookup table.
 * @param intensity - Controls the intensity of the effect.
 * @return The graded color.
 */
export const lut3D = (
  inputNode: Node<"vec4">,
  lutNode: Node<"sampler3D">,
  size: FloatIn,
  intensity: FloatIn = 1,
): Node<"vec4"> => {
  // Pull the sample in by half a pixel so the sample begins at the center of
  // the edge pixels.
  const pixelWidth = float(1.0).div(size);
  const halfPixelWidth = float(0.5).div(size);
  const uvw = vec3(halfPixelWidth).add(inputNode.rgb.mul(float(1.0).sub(pixelWidth)));
  const lutValue = vec4(lutNode.texture(uvw).rgb, inputNode.a);
  return vec4(mix(inputNode, lutValue, intensity));
};
