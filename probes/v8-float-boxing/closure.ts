/**
 * Whether V8 boxes a float that code stores in a variable a closure keeps,
 * which the function storing it reads from outside itself, and not one in a
 * local of the function.
 *
 * Usage: node probes/v8-float-boxing/closure.ts
 */
import { N, fractions, runCases } from "./cases.ts";

const sink = new Float64Array(1);

await runCases(import.meta.url, {
  "store into a variable outside the function, made 0": () => {
    let t = 0;
    return () => {
      for (let i = 0; i < N; i++) {
        t = fractions[i & 7]! * 1.5;
        sink[0] += t;
      }
    };
  },
  "store into a variable outside the function, made 0.5": () => {
    let t = 0.5;
    return () => {
      for (let i = 0; i < N; i++) {
        t = fractions[i & 7]! * 1.5;
        sink[0] += t;
      }
    };
  },
  "store into a local of the function": () => () => {
    let t = 0;
    for (let i = 0; i < N; i++) {
      t = fractions[i & 7]! * 1.5;
      sink[0] += t;
    }
  },
  "store into a variable outside the function, from code new Function made": () =>
    new Function(
      "src",
      "sink",
      "N",
      "let t = 0; return () => { for (let i = 0; i < N; i++) { t = src[i & 7] * 1.5; sink[0] += t; } };",
    )(fractions, sink, N),
});
