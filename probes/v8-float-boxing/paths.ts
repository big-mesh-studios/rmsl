/**
 * What a call allocates on each CPU path other than the rasterizer, when the
 * program takes or gives a scalar float: the function `compileJSFn` returns,
 * a JS routine, a JS grid, a JS compute stage, and the JS and WASM compute
 * adapters. Each case runs in a worker of its own.
 *
 * Usage: pnpm build && node probes/v8-float-boxing/paths.ts
 */
import { Worker, isMainThread, workerData, parentPort } from "node:worker_threads";
import { allocatedBy, versions } from "./measure.ts";

const cases = [
  "compileJSFn source, float param in, float out",
  "compileJSFn source, float uniform in, vec3 out",
  "compileJSRoutine, float param in, float out",
  "compileJSRoutine, float param in, vec3 out",
  "compileJSGrid, float uniform in, float out (16x16)",
  "compileJSCompute, float uniform in (256 invocations)",
  "createJsCompute, float uniform set once (256 invocations)",
  "createWasmCompute, float uniform set once (256 invocations)",
  "createJsCompute, float uniform set each dispatch from a plain array",
  "createWasmCompute, float uniform set each dispatch from a plain array",
  "createJsCompute, float uniform set each dispatch, computed",
  "createWasmCompute, float uniform set each dispatch, computed",
] as const;

if (isMainThread) {
  console.log(versions());
  for (const name of cases) {
    const worker = new Worker(new URL(import.meta.url), { workerData: name });
    const result = await new Promise((resolve) => {
      worker.once("message", resolve);
      worker.once("error", (error) => resolve(String(error)));
    });
    console.log(name.padEnd(70), JSON.stringify(result));
  }
} else {
  const rmsl: any = await import("../../dist/rmsl.js");
  const js: any = await import("../../dist/js.js");
  const wasm: any = await import("../../dist/wasm.js");
  const { Fn, uniform, vec3, fragCoord, instancedArray, invocationIndex } = rmsl;
  const name = workerData as (typeof cases)[number];
  const src = Float64Array.of(0.1, 0.3, 0.7, 0.2, 0.9, 0.45, 0.33, 0.61);
  let i = 0;
  /** A fraction that changes from call to call. */
  const next = () => src[i++ & 7]! * 1.37;
  const param = { name: "main", params: [{ name: "a", type: "float" }] };
  const none = { name: "main", params: [] };
  let run: () => unknown;
  switch (name) {
    case "compileJSFn source, float param in, float out": {
      const f = new Function(js.compileJSFn((a: any) => Fn(() => a.mul(0.5).toVar())(), param))();
      const ctx = { params: { a: 0 } };
      run = () => {
        ctx.params.a = next();
        return f(ctx);
      };
      break;
    }
    case "compileJSFn source, float uniform in, vec3 out": {
      const u = uniform("float");
      const f = new Function(js.compileJSFn(() => Fn(() => vec3(u, u.mul(2), 1).toVar())(), none))();
      const ctx = { uniforms: { [u.name]: 0.37 } };
      run = () => f(ctx);
      break;
    }
    case "compileJSRoutine, float param in, float out":
    case "compileJSRoutine, float param in, vec3 out": {
      const routine = js.compileJSRoutine(
        (a: any) => Fn(() => (name.endsWith("float out") ? a.mul(0.5) : vec3(a, 1, 2)).toVar())(),
        param,
      );
      const ctx = { params: { a: 0 } };
      run = () => {
        ctx.params.a = next();
        return routine(ctx);
      };
      break;
    }
    case "compileJSGrid, float uniform in, float out (16x16)": {
      const u = uniform("float");
      const grid = js.compileJSGrid(() => Fn(() => fragCoord().x.mul(u).toVar())(), none);
      const ctx = { uniforms: { [u.name]: 0.37 } };
      const out = grid(ctx, 16, 16);
      run = () => grid(ctx, 16, 16, out);
      break;
    }
    case "compileJSCompute, float uniform in (256 invocations)": {
      const u = uniform("float");
      const buf = instancedArray(256, "float");
      const stage = js.compileJSCompute(() => Fn(() => buf.element(invocationIndex()).assign(u.mul(2)))(), none);
      const ctx = { uniforms: { [u.name]: 0.37 }, storages: { [buf.name]: new Float64Array(256) } };
      run = () => stage(ctx, 256);
      break;
    }
    default: {
      const u = uniform("float");
      const buf = instancedArray(256, "float");
      const create = name.startsWith("createJs") ? js.createJsCompute : wasm.createWasmCompute;
      const adapter = create(Fn(() => buf.element(invocationIndex()).assign(u.mul(2)))(), { name: "step" });
      adapter.setAttribute(buf.name, new Float32Array(256));
      // A plain array of fractions holds each as a heap number already, so passing one boxes nothing new.
      const boxed = Array.from(src, (v) => v * 1.37);
      adapter.setUniform(u, 0.37);
      if (name.endsWith("set once (256 invocations)")) run = () => adapter.compute();
      else if (name.endsWith("from a plain array"))
        run = () => {
          adapter.setUniform(u, boxed[i++ & 7]);
          adapter.compute();
        };
      else
        run = () => {
          adapter.setUniform(u, next());
          adapter.compute();
        };
      break;
    }
  }
  const result = await allocatedBy(run!, 5000, 2000);
  const sites = Object.fromEntries(Object.entries(result.sites).filter(([, bytes]) => bytes >= 1));
  parentPort!.postMessage({ perCall: Math.round(result.perCall * 10) / 10, sites });
}
