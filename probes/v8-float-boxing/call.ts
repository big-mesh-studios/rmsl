/**
 * Whether V8 boxes a float passed as an argument to, or returned from, a call
 * it does not inline, and not one it inlines.
 *
 * Usage: node probes/v8-float-boxing/call.ts
 */
import { N, fractions, runCases } from "./cases.ts";

const sink = new Float64Array(1);
/** A body too large for V8 to inline, which does what the small one does. */
const padding = "if (s[0] < -1e300) { s[0] = Math.sin(s[0]) + Math.cos(s[0]) + Math.tan(s[0]); }".repeat(80);

await runCases(import.meta.url, {
  "argument to a call that is inlined": () => {
    const take = new Function("v", "s", "s[0] += v;") as (v: number, s: Float64Array) => void;
    return () => {
      for (let i = 0; i < N; i++) take(fractions[i & 7]! * 1.5, sink);
    };
  },
  "argument to a call that is not inlined": () => {
    const take = new Function("v", "s", "s[0] += v;" + padding) as (v: number, s: Float64Array) => void;
    return () => {
      for (let i = 0; i < N; i++) take(fractions[i & 7]! * 1.5, sink);
    };
  },
  "result of a call that is inlined": () => {
    const give = new Function("i", "src", "return src[i & 7] * 1.5;") as (i: number, src: Float64Array) => number;
    return () => {
      for (let i = 0; i < N; i++) sink[0] += give(i, fractions);
    };
  },
  "result of a call that is not inlined": () => {
    const give = new Function("i", "src", "s", padding + "return src[i & 7] * 1.5;") as (
      i: number,
      src: Float64Array,
      s: Float64Array,
    ) => number;
    return () => {
      for (let i = 0; i < N; i++) sink[0] += give(i, fractions, sink);
    };
  },
});
