import { Fn, float, uniform, type Node } from "../../rmsl";
import { compileWasmFn } from "../../wasm";

// Any shader function can be compiled for WASM. This module exports the
// compileWasmFn() output for each one under __RMSL_WASM_CODE, which vite's
// precompileWasm plugin reads at build time, emits as a .wasm asset, and
// rewrites into an instantiateWasm(...) call — no eval, no rmsl at runtime.

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
