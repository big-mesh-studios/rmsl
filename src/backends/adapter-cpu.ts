// === CPU adapter loop (shared by the JS and WASM adapters) ===
// compileJS and compileWasm both compile a storage()/invocationIndex()
// program into the identical shape: a function called once per entity,
// mutating the storage arrays it was given in place (see cpu.ts's
// CpuShaderContext). There is no device/pipeline ceremony to hide here —
// attach()/destroy() are no-ops — so this is just the shared
// setUniform/setAttribute/compute(out) surface around that per-entity
// loop.
//
// Not exported publicly: createJs (adapter-js.ts) and createWasm
// (adapter-wasm.ts) each wrap this around their own compile() call, so a
// caller's constructor always takes the same thing createGlsl/createWgsl's
// do — a root graph — instead of an already-compiled callable only this
// generic version needed.
import { Adapter, TypedArray } from "./adapter";
import { CpuRenderer } from "./cpu";

/** One typed array per storage slot, keyed by name. */
export type AdapterResult = Record<string, TypedArray>;

export function createCpuAdapter(step: CpuRenderer): Adapter<AdapterResult> {
  let n = 0;
  const storages: Record<string, TypedArray> = {};
  const uniforms: Record<string, number | number[]> = {};

  return {
    attach() {},

    setUniform(slot, value) {
      uniforms[slot] = value;
    },

    setAttribute(slot, data) {
      storages[slot] = data;
      n = Math.max(n, data.length);
    },

    compute(out) {
      for (let i = 0; i < n; i++) {
        step({ storages, uniforms, index: i } as any);
      }
      // storages already holds the caller's own arrays, mutated in place —
      // `out` is only for callers that want the WGSL adapter's optional-out
      // shape too, not something this loop needs to do its job.
      if (!out) return;
      for (const slot in storages) (out[slot] as TypedArray).set(storages[slot]);
      return out;
    },

    destroy() {},
  };
}
