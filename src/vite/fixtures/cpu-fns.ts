import { Fn, float, uniform, type Node } from "../../rmsl";
import { compileJSFn } from "../../js";

/**
 * Fixture for {@link precompileJS} (see `../vite.ts`): exports each
 * function's `compileJSFn()` output under `__RMSL_JS_CODE`, which the
 * plugin reads at build time and inlines as plain functions.
 */
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
