/**
 * Question: does JavaScript's `Math` have a function for 2 raised to a power,
 * apart from `Math.pow`, and does `Math.pow(2, x)` give the same value whether
 * a program meant `exp2(x)` or `pow(2, x)`?
 *
 * Lists the own properties of `Math` in the engine it runs in, and compares
 * `Math.pow(2, x)` with `2 ** x` over arguments of either sign, whole and
 * fractional, at both ends of the range of a 32-bit float.
 *
 * Run: node probes/float-accuracy/javascript-math-has-no-exp2.ts > probes/float-accuracy/results/javascript-math-has-no-exp2.json
 */
const names = Object.getOwnPropertyNames(Math).sort();
const args = [-149, -126.5, -3, -0.5, -0, 0, 0.5, 1, 3, 10.25, 127, 128, Number.NaN, Infinity, -Infinity];
console.log(
  JSON.stringify(
    {
      probe: "javascript-math-has-no-exp2",
      environment: { node: process.version, v8: process.versions.v8 },
      mathFunctions: names.filter((name) => typeof (Math as unknown as Record<string, unknown>)[name] === "function"),
      checks: {
        hasNoExp2: !names.includes("exp2"),
        powOfTwoIsExponentiation: args.every((x) => Object.is(Math.pow(2, x), 2 ** x)),
      },
    },
    null,
    2,
  ),
);
