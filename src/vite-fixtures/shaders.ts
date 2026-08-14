import {
  Fn,
  attribute,
  compileGLSL,
  uniformRaw,
  varying,
  vec2,
  vec4,
} from "../rmsl";

// This module is compiled once at build time by vite's precompileShaders plugin
// and replaced with JSON, so the rmsl graph is never built (and rmsl is never
// shipped) in the browser.

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
  vertexGLSL: compileGLSL.vertex(vertexFn()),
  fragmentGLSL: compileGLSL.fragment(fragmentFn()),
};
