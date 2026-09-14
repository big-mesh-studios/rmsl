import { compileWasm } from "@random-mesh/rmsl";
import { calcMandelbrotCpu } from "./mandelbrotShader";

// Each worker's wasm instance gets its own PRIVATE memory (the default —
// no `memory`/`sharedMemory` option). That's deliberate: fragCoord and the
// uniform values this compiled module reads all live at fixed addresses
// baked into the module itself, overwritten on every pixel it draws.
// Sharing one memory across several concurrently-running instances of the
// *same* module means they'd all race to write those same addresses with
// different, constantly-changing values — scrambled pixels, no error. A
// private memory per instance sidesteps that entirely; only the result
// (this worker's own row slice) crosses back to the main thread.
let renderer: ReturnType<typeof compileWasm> | null = null;

type InitMsg = { type: "init" };
type DrawMsg = {
  type: "draw";
  id: number;
  uniforms: Record<string, number | number[]>;
  width: number;
  rowCount: number;
};

self.onmessage = (e: MessageEvent<InitMsg | DrawMsg>) => {
  const msg = e.data;

  if (msg.type === "init") {
    renderer = compileWasm(() => calcMandelbrotCpu(), { name: "mandelbrotWasmWorker", params: [] });
    postMessage({ type: "ready" });
    return;
  }

  if (msg.type === "draw") {
    if (!renderer) throw new Error("[mandelbrotWasmWorker] draw before init");
    const { id, uniforms, width, rowCount } = msg;
    const result = renderer.draw({ uniforms }, width, rowCount);
    // `result` is a view into this instance's own private wasm memory —
    // copy it into a fresh, transferable buffer rather than handing that
    // memory's bytes to structured clone (which would work, but clones
    // rather than transfers, and we'd rather not entangle the wasm
    // instance's own memory object with postMessage's ownership rules).
    const out = new Float64Array(result);
    (self as unknown as Worker).postMessage({ type: "done", id, rowCount, buffer: out.buffer }, [out.buffer]);
  }
};
