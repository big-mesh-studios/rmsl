/**
 * Whether what a store of a float into `o.x` allocates depends on other
 * objects: objects made the way `o` was, with the same keys in the same
 * order, share how V8 represents each field. `o` is made as `{}` with `x`
 * added and set to 0, or in one case by an object literal, and a call stores 256 fractions into `o.x`. Each case
 * makes one other object first, and runs in a worker of its own.
 *
 * Usage: node probes/v8-float-boxing/shared-field.ts
 */
import { Worker, isMainThread, workerData, parentPort } from "node:worker_threads";
import { allocatedBy, versions } from "./measure.ts";

const N = 256;

/** Each case makes its other object, before or after `o`'s stores have run hot. */
const cases: Record<string, { other: () => unknown; after?: boolean }> = {
  "no other object": { other: () => undefined },
  "{} with x added, holding undefined": {
    other: () => {
      const s: any = {};
      s["x"] = undefined;
      return s;
    },
  },
  "{} with x added, holding a string": {
    other: () => {
      const s: any = {};
      s["x"] = "float";
      return s;
    },
  },
  "Object.fromEntries with x, holding a string": { other: () => Object.fromEntries([["x", "float"]]) },
  "{} with x added, holding 0, then an array": {
    other: () => {
      const s: any = {};
      s["x"] = 0;
      s["x"] = [1];
      return s;
    },
  },
  "{} with x added, holding undefined, made after o ran hot": {
    after: true,
    other: () => {
      const s: any = {};
      s["x"] = undefined;
      return s;
    },
  },
  "object literal { x: undefined }": { other: () => new Function("return { x: undefined }")() },
  "object literal { x: undefined }, with o an object literal { x: 0 }": {
    other: () => new Function("return { x: undefined }")(),
  },
  "{} with y added, holding undefined": {
    other: () => {
      const s: any = {};
      s["y"] = undefined;
      return s;
    },
  },
  "{} with w then x added, x holding undefined": {
    other: () => {
      const s: any = {};
      s["w"] = 0;
      s["x"] = undefined;
      return s;
    },
  },
};

if (isMainThread) {
  console.log(versions());
  console.log("bytes per store of a fraction into o.x:");
  for (const name of Object.keys(cases)) {
    const worker = new Worker(new URL(import.meta.url), { workerData: name });
    const perStore = await new Promise<number>((resolve, reject) => {
      worker.once("message", resolve);
      worker.once("error", reject);
    });
    console.log(name.padEnd(60), perStore.toFixed(2).padStart(6));
  }
} else {
  const { other, after } = cases[workerData as string]!;
  const keep: unknown[] = [];
  if (!after) keep.push(other());
  const literal = (workerData as string).includes("with o an object literal");
  const o: any = literal ? new Function("return { x: 0 }")() : {};
  if (!literal) o["x"] = 0;
  const src = Float64Array.of(0.1, 0.3, 0.7, 0.2, 0.9, 0.45, 0.33, 0.61);
  const run = () => {
    for (let i = 0; i < N; i++) o.x = src[i & 7]! * 1.37;
  };
  if (after) {
    for (let k = 0; k < 20000; k++) run();
    keep.push(other());
  }
  const { perCall } = await allocatedBy(run, 20000, 2000);
  parentPort!.postMessage(perCall / N);
}
