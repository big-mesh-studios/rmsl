/**
 * Whether a place in the code that functions made by one factory share sees
 * the keys of all of them: each function the factory makes stores floats into,
 * or reads them from, an object of its own under a key of its own, by a key held in a variable,
 * as each rasterizer `compileJS` makes does. The probe measures the last
 * function made, after the others ran.
 *
 * Usage: node probes/v8-float-boxing/feedback.ts
 */
import { N, fractions, runCases } from "./cases.ts";

/** A function that stores floats into an object of its own, under `key`. */
function factory(key: string): () => void {
  const o: any = {};
  o[key] = 0;
  const keys = [key];
  return () => {
    for (let i = 0; i < N; i++) for (let k = 0; k < keys.length; k++) o[keys[k]!] = fractions[i & 7]! * 1.5;
  };
}

/** A function that reads floats from eight objects of its own, under `key`, into a typed array. */
function readFactory(key: string): () => void {
  const objects = Array.from(fractions, (v) => {
    const o: any = {};
    o[key] = v * 1.5;
    return o;
  });
  const keys = [key];
  const sink = new Float64Array(1);
  return () => {
    for (let i = 0; i < N; i++) for (let k = 0; k < keys.length; k++) sink[0] = objects[i & 7][keys[k]!] * 2;
  };
}

/** The last of `count` functions the factory makes, each with a key of its own, after each ran. */
function lastOf(count: number, make = factory): () => () => void {
  return () => {
    let last!: () => void;
    for (let c = 0; c < count; c++) {
      // A key read back from an object is internalized, as the rasterizer's slots, which Object.keys gives, are.
      last = make(Object.keys({ [`k${c}`]: 0 })[0]!);
      for (let k = 0; k < 2000; k++) last();
    }
    return last;
  };
}

await runCases(import.meta.url, {
  "one function made": lastOf(1),
  "two functions made, each with its own key": lastOf(2),
  "five functions made, each with its own key": lastOf(5),
  "reads: one function made": lastOf(1, readFactory),
  "reads: two functions made, each with its own key": lastOf(2, readFactory),
});
