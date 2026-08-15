import { type FloatIn, type Vec2In } from "./util";
import { clamp, fract, mix, rand, time, uv, vec2, vec4, type Node } from "../rmsl";

/**
 * Post-processing effect creating a film grain.
 *
 * Ported from three.js's `examples/jsm/tsl/display/FilmNode.js`.
 *
 * @param inputNode - The color node that represents the input of the effect.
 * @param intensity - The effect's intensity. `null` applies the grain at full strength.
 * @param uvNode - Custom (e.g. animated) uv data; defaults to the screen uv.
 * @return The film-grained color.
 */
export const film = (
  inputNode: Node<"vec4">,
  intensity: FloatIn | null = null,
  uvNode: Vec2In | null = null,
): Node<"vec4"> => {
  const u = uvNode === null
    ? uv()
    : Array.isArray(uvNode)
      ? vec2(uvNode[0], uvNode[1])
      : uvNode;
  const base = inputNode.rgb;
  const noise = rand(fract(u.add(time())));
  const grained = base.add(base.mul(clamp(noise.add(0.1), 0, 1)));
  const color = intensity === null ? grained : mix(base, grained, intensity);
  return vec4(color, inputNode.a);
};
