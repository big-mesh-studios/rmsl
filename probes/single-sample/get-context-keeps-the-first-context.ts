/**
 * Question: does `getContext("webgl2", attributes)` on a canvas that already
 * has a WebGL 2 context hand back that context, with the attributes it was
 * made with, whatever the second call asks?
 *
 * Each case makes a context with one set of attributes, then asks the same
 * canvas again with the opposite `antialias` and `alpha`. The result is
 * whether the second call returned the same object, the `antialias` and
 * `alpha` the context reports after each call, its `SAMPLES` and `ALPHA_BITS`,
 * and `getError`. A control asks a fresh canvas with the second attributes, so
 * the second call's attributes are shown to take effect where no context
 * exists.
 *
 * Measures the platform only: plain WebGL 2, no rmsl.
 */
import { PROBES, report, runInGpuPage, runInWebGpuPage } from "./harness";

const ENTRY = `
const describe = (gl) => ({
  antialias: gl.getContextAttributes().antialias,
  alpha: gl.getContextAttributes().alpha,
  samples: gl.getParameter(gl.SAMPLES),
  alphaBits: gl.getParameter(gl.ALPHA_BITS),
  error: gl.getError(),
});
globalThis.__probe = async () => {
  const result = {};
  for (const [name, first] of [
    ["antialiasTrueAlphaTrue", { antialias: true, alpha: true }],
    ["antialiasFalseAlphaTrue", { antialias: false, alpha: true }],
    ["antialiasTrueAlphaFalse", { antialias: true, alpha: false }],
    ["antialiasFalseAlphaFalse", { antialias: false, alpha: false }],
  ]) {
    const second = { antialias: !first.antialias, alpha: !first.alpha };
    const canvas = document.createElement("canvas");
    const made = canvas.getContext("webgl2", first);
    const before = describe(made);
    const again = canvas.getContext("webgl2", second);
    const control = document.createElement("canvas").getContext("webgl2", second);
    result[name] = { same: again === made, before, after: describe(again), control: describe(control) };
  }
  return result;
};
`;

await report("get-context-keeps-the-first-context", async () => ({
  webglPage: await runInGpuPage(ENTRY, "__probe", PROBES),
  webgpuPage: await runInWebGpuPage(ENTRY, "__probe", PROBES),
}));
