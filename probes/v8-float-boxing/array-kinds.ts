/**
 * Whether a load from an array, at one place in the code, boxes the float it
 * reads once that place has seen many kinds of array, and whether a typed
 * array's own `set` copies from any kind without boxing.
 *
 * Usage: node probes/v8-float-boxing/array-kinds.ts
 */
import { N, fractions, runCases } from "./cases.ts";

const sink = new Float64Array(8);

/** Arrays of the kinds a host may pass a vector in, each holding fractions. */
function kinds(count: number): ArrayLike<number>[] {
  const all: ArrayLike<number>[] = [
    Float64Array.from(fractions),
    Float32Array.from(fractions),
    Array.from(fractions),
    Int32Array.of(1, 2, 3, 4, 5, 6, 7, 8),
    Uint32Array.of(1, 2, 3, 4, 5, 6, 7, 8),
    [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, {} as any],
  ];
  return all.slice(0, count);
}

/** Reads a Float64Array through one load that has seen `count` kinds of array. */
function loadAfter(count: number): () => () => void {
  return () => {
    const read = new Function("a", "i", "return a[i];") as (a: ArrayLike<number>, i: number) => number;
    const arrays = kinds(count);
    for (let k = 0; k < 2000; k++) for (const a of arrays) read(a, k & 7);
    const f64 = arrays[0]!;
    return () => {
      for (let i = 0; i < N; i++) sink[0] = sink[0]! + read(f64, i & 7) * 0.5;
    };
  };
}

/** Copies into a Float64Array with its own `set`, from each of `count` kinds of array in turn, or from the one kind at `only`. */
function setFrom(count: number, only?: number): () => () => void {
  return () => {
    const arrays = only === undefined ? kinds(count) : [kinds(count)[only]!];
    return () => {
      for (let i = 0; i < N; i++) sink.set(arrays[i % arrays.length]!);
    };
  };
}

await runCases(import.meta.url, {
  "load from a Float64Array, the place has seen 1 kind": loadAfter(1),
  "load from a Float64Array, the place has seen 4 kinds": loadAfter(4),
  "load from a Float64Array, the place has seen 5 kinds": loadAfter(5),
  "load from a Float64Array, the place has seen 6 kinds": loadAfter(6),
  "Float64Array set from 5 kinds of array, numbers only": setFrom(5),
  "Float64Array set from 6 kinds of array": setFrom(6),
  "Float64Array set from a Float64Array": setFrom(6, 0),
  "Float64Array set from a Float32Array": setFrom(6, 1),
  "Float64Array set from a plain array of fractions": setFrom(6, 2),
  "Float64Array set from an Int32Array": setFrom(6, 3),
  "Float64Array set from a plain array of fractions and an object": setFrom(6, 5),
});
