/**
 * What V8 allocates when optimized code stores a float it computed, which it
 * holds unboxed, into a property of an object, and when it reads one back.
 * Each case runs in a worker of its own, so each starts from fresh maps and
 * fresh feedback. A call does the operation 256 times; the table gives bytes
 * per operation.
 *
 * Usage: node probes/v8-float-boxing/store.ts
 */
import { Worker, isMainThread, workerData, parentPort } from "node:worker_threads";
import { allocatedBy, versions } from "./measure.ts";

const N = 256;
/**
 * The values a case computes from, each times 1.5: fractions; whole numbers a
 * small integer holds; whole numbers past 2^31, which no small integer holds;
 * -0, NaN and the infinities; and -2^31 and 2^31 - 2, near the ends of a 32-bit integer.
 */
const values: Record<string, Float64Array> = {
  fractions: Float64Array.of(0.1, 0.3, 0.7, 0.2, 0.9, 0.45, 0.33, 0.61),
  wholes: Float64Array.of(2, 4, 6, 8, 10, 12, 14, 16),
  "past 2^31": Float64Array.of(
    2 ** 31,
    2 ** 31 + 2,
    2 ** 31 + 4,
    2 ** 31 + 6,
    2 ** 31 + 8,
    2 ** 31 + 10,
    2 ** 31 + 12,
    2 ** 31 + 14,
  ),
  "-0, NaN, ±Infinity": Float64Array.of(-0, NaN, Infinity, -Infinity, -0, NaN, Infinity, -Infinity),
  // Each times 1.5 gives -2^31 or 2^31 - 1.5's neighbours: the ends of a 32-bit integer.
  "int32 ends": Float64Array.of(
    -(2 ** 31) / 1.5,
    (2 ** 31 - 2) / 1.5 + 0,
    -(2 ** 31) / 1.5,
    (2 ** 31 - 2) / 1.5,
    -(2 ** 31) / 1.5,
    (2 ** 31 - 2) / 1.5,
    -(2 ** 31) / 1.5,
    (2 ** 31 - 2) / 1.5,
  ),
};
const sink = new Float64Array(1);

type Case = (src: Float64Array) => () => void;

/** A store `o.x = v`, after `o` was made with `x` holding `initial`. */
function namedStore(initial: unknown): Case {
  return (src) => {
    const o: any = { x: initial };
    return () => {
      for (let i = 0; i < N; i++) o.x = src[i & 7]! * 1.5;
    };
  };
}

/** A store `o[key] = v`, with `key` read from an array, so the code names no key. */
function keyedStore(initial: unknown, keyCount: number): Case {
  return (src) => {
    const keys = ["a", "b", "c", "d"].slice(0, keyCount);
    const o: any = {};
    keys.forEach((k) => (o[k] = initial));
    return () => {
      for (let i = 0; i < N; i++) o[keys[i % keyCount]!] = src[i & 7]! * 1.5;
    };
  };
}

/**
 * Reads `o.x` for arithmetic, from eight objects made alike that hold
 * different floats, so no read can be lifted out of the loop.
 */
function namedRead(make: () => any): Case {
  return (src) => {
    const objects = Array.from({ length: 8 }, (_, i) => {
      const o = make();
      o.x = src[i]! * 1.5;
      return o;
    });
    return () => {
      let s = 0;
      for (let i = 0; i < N; i++) s += objects[i & 7].x * src[i & 7]!;
      sink[0] = s;
    };
  };
}

/** Reads `o[key]` for arithmetic, with `key` read from an array, from eight objects made alike. */
function keyedRead(make: () => any, keyCount: number): Case {
  return (src) => {
    const keys = ["a", "b", "c", "d"].slice(0, keyCount);
    const objects = Array.from({ length: 8 }, (_, j) => {
      const o = make();
      keys.forEach((k, i) => (o[k] = src[(i + j) & 7]! * 1.5));
      return o;
    });
    return () => {
      let s = 0;
      for (let i = 0; i < N; i++) s += objects[i & 7][keys[i % keyCount]!] * src[i & 7]!;
      sink[0] = s;
    };
  };
}

const plain = () => ({});
const dictionary = () => Object.create(null);

const cases: Record<string, Case> = {
  "store o.x, x made 0.5": namedStore(0.5),
  "store o.x, x made 0 (a small integer)": namedStore(0),
  "store o.x, x made undefined": namedStore(undefined),
  "store o.x, x made null": namedStore(null),
  "store o.x, x made an array": namedStore([]),
  "store o[key], one key, made 0": keyedStore(0, 1),
  "store o[key], one key, made 0.5": keyedStore(0.5, 1),
  "store o[key], one key, made undefined": keyedStore(undefined, 1),
  "store o[key], two keys, made 0": keyedStore(0, 2),
  "store o[key], four keys, made 0": keyedStore(0, 4),
  "store o.x, o in dictionary mode": (src) => {
    const o = Object.create(null);
    o.x = 0;
    return () => {
      for (let i = 0; i < N; i++) o.x = src[i & 7]! * 1.5;
    };
  },
  "store f[0], f a Float64Array": (src) => {
    const f = new Float64Array(1);
    return () => {
      for (let i = 0; i < N; i++) f[0] = src[i & 7]! * 1.5;
    };
  },
  "read o.x, o plain": namedRead(plain),
  "read o.x, o in dictionary mode": namedRead(dictionary),
  "read o[key], one key, o plain": keyedRead(plain, 1),
  "read o[key], two keys, o plain": keyedRead(plain, 2),
  "read o[key], four keys, o plain": keyedRead(plain, 4),
  "read o[key], one key, o in dictionary mode": keyedRead(dictionary, 1),
  "read o[key], four keys, o in dictionary mode": keyedRead(dictionary, 4),
};

if (isMainThread) {
  console.log(versions());
  // A read case stores its floats before it runs, from code V8 has not optimized, so its columns differ only by chance.
  console.log("bytes per operation:".padEnd(56), Object.keys(values).join(" | "));
  for (const name of Object.keys(cases)) {
    const row: string[] = [];
    for (const set of Object.keys(values)) {
      const worker = new Worker(new URL(import.meta.url), { workerData: { name, set } });
      const perCall = await new Promise<number>((resolve, reject) => {
        worker.once("message", resolve);
        worker.once("error", reject);
      });
      row.push((perCall / N).toFixed(2).padStart(6));
    }
    console.log(name.padEnd(56), row.join(" "));
  }
} else {
  const { name, set } = workerData as { name: string; set: string };
  const run = cases[name]!(values[set]!);
  const { perCall } = await allocatedBy(run, 20000, 2000);
  parentPort!.postMessage(perCall);
}
