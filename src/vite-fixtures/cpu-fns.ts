import {
  Fn,
  compileJSFn,
  float,
  uniform,
  type Node,
} from "../rmsl";

// Any shader function can be compiled for the CPU. This module exports the
// compileJSFn() output for each one under __RMSL_JS_CODE, which vite's
// precompileJS plugin reads at build time and inlines as plain functions — no
// eval, no rmsl at runtime.

const brightness = Fn(() => {
  const colour = uniform("vec3");
  return colour.mul(float(0.5)).toVar();
});

const mixColours = Fn((a: Node<"vec3">, b: Node<"vec3">, t: Node<"float">) => {
  return a.mix(b, t).toVar();
});

export const __RMSL_JS_CODE = {
  brightness: compileJSFn(() => brightness(), {
    name: "brightness",
    params: [],
  }),
  mixColours: compileJSFn(mixColours, {
    name: "mixColours",
    params: [
      { name: "a", type: "vec3" },
      { name: "b", type: "vec3" },
      { name: "t", type: "float" },
    ],
  }),
};
