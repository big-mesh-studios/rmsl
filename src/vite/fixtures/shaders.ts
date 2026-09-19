import { Fn, attribute, uniformRaw, varying, vec2, vec4 } from "../../rmsl";
import { compileGlsl } from "../../glsl";

/** Fixture for {@link precompileShaders} (see `../vite.ts`): compiled once
 * at build time and replaced with JSON. */
export const uColour = uniformRaw("uColour", "vec3");
export const vUv = varying("vec2");
export const positionAttr = attribute("vec2");

export const vertexFn = Fn(() => {
  vUv.assign(positionAttr);
  return vec4(positionAttr, 0, 1);
});

export const fragmentFn = Fn(() => {
  return vec4(uColour, 1);
});

export default {
  uColour: uColour.name,
  vUv: vUv.name,
  positionAttr: positionAttr.name,
  vertexGLSL: compileGlsl.vertex(vertexFn()),
  fragmentGLSL: compileGlsl.fragment(fragmentFn()),
};
