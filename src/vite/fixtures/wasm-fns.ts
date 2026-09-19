import { Fn, float, uniform, type Node } from "../../rmsl";
import { compileWasmFn } from "../../wasm";

/**
 * Fixture for {@link precompileWasm} (see `../vite.ts`): exports each
 * function's `compileWasmFn()` output under `__RMSL_WASM_CODE`, which the
 * plugin reads at build time and rewrites into `instantiateWasmRoutine(...)` calls.
 */
const brightness = Fn(() => {
  const colour = uniform("vec3");
  return colour.mul(float(0.5)).toVar();
});

const mixColours = Fn((a: Node<"vec3">, b: Node<"vec3">, t: Node<"float">) => {
  return a.mix(b, t).toVar();
});

export const __RMSL_WASM_CODE = {
  brightness: compileWasmFn(() => brightness(), {
    name: "brightness",
    params: [],
  }),
  mixColours: compileWasmFn(mixColours, {
    name: "mixColours",
    params: [
      { name: "a", type: "vec3" },
      { name: "b", type: "vec3" },
      { name: "t", type: "float" },
    ],
  }),
};
