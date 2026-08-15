import { f, type FloatIn, type IntIn, type Vec2In, type Sampler2D, type Sampler3D } from "./util";
import {add, mat3, textureSize, uv, vec2, vec3, vec4, luminance, type Node} from "../rmsl";

/**
 * Post-processing effect detecting edges with a sobel filter. A sobel filter
 * should be applied after tone mapping and output color space conversion.
 *
 * Ported from three.js's `examples/jsm/tsl/display/SobelOperatorNode.js`.
 *
 * @param textureNode - The texture node that represents the input of the effect.
 * @return An edge-detection image (black where flat, white on edges).
 */
export const sobel = (textureNode: Sampler2D): Node<"vec4"> => {
  // One texel in UV space; the matrices are column-major, as GLSL fills them.
  const texel = vec2(1).div(textureSize(textureNode).toVec2());
  const uvNode = uv();
  const sampleTexture = (u: Node<"vec2">): Node<"vec4"> => textureNode.texture(u);

  const Gx = mat3(-1, -2, -1, 0, 0, 0, 1, 2, 1); // x direction kernel
  const Gy = mat3(-1, 0, 1, -2, 0, 2, -1, 0, 1); // y direction kernel

  const tx0y0 = luminance(sampleTexture(uvNode.add(texel.mul(vec2(-1, -1)))).xyz);
  const tx0y1 = luminance(sampleTexture(uvNode.add(texel.mul(vec2(-1, 0)))).xyz);
  const tx0y2 = luminance(sampleTexture(uvNode.add(texel.mul(vec2(-1, 1)))).xyz);
  const tx1y0 = luminance(sampleTexture(uvNode.add(texel.mul(vec2(0, -1)))).xyz);
  const tx1y1 = luminance(sampleTexture(uvNode.add(texel.mul(vec2(0, 0)))).xyz);
  const tx1y2 = luminance(sampleTexture(uvNode.add(texel.mul(vec2(0, 1)))).xyz);
  const tx2y0 = luminance(sampleTexture(uvNode.add(texel.mul(vec2(1, -1)))).xyz);
  const tx2y1 = luminance(sampleTexture(uvNode.add(texel.mul(vec2(1, 0)))).xyz);
  const tx2y2 = luminance(sampleTexture(uvNode.add(texel.mul(vec2(1, 1)))).xyz);

  const g = (G: Node<"mat3">, i0: Node<"float">, i1: Node<"float">, i2: Node<"float">) => add(
    G.element(0).element(0).mul(i0),
    G.element(1).element(0).mul(i1),
    G.element(2).element(0).mul(i2),
    G.element(0).element(1).mul(tx0y1),
    G.element(1).element(1).mul(tx1y1),
    G.element(2).element(1).mul(tx2y1),
    G.element(0).element(2).mul(tx0y2),
    G.element(1).element(2).mul(tx1y2),
    G.element(2).element(2).mul(tx2y2),
  );

  const valueGx = g(Gx, tx0y0, tx1y0, tx2y0);
  const valueGy = g(Gy, tx0y0, tx1y0, tx2y0);

  const G = valueGx.mul(valueGx).add(valueGy.mul(valueGy)).sqrt();

  return vec4(vec3(G), 1);
};
