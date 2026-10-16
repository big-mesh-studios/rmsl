/**
 * Question: what error does WGSL allow each `f32` operation and built-in
 * function, and what does it mean by a correctly rounded result?
 *
 * Reads the WGSL specification from the W3C and prints the status line of the
 * draft it read, the definition of a correctly rounded result from section
 * 15.7.4, and the accuracy for `f32` of every row of the tables of section
 * 15.7.4.1, with checks of the claims
 * `fact-wgsl-bounds-the-error-of-each-float-built-in` makes.
 *
 * Run: node probes/float-accuracy/wgsl-accuracy-table.ts > probes/float-accuracy/results/wgsl-accuracy-table.json
 */
import { load } from "cheerio";

const SOURCE = "https://www.w3.org/TR/WGSL/";

const $ = load(await (await fetch(SOURCE)).text());
// A superscript reads as a power: 2<sup>-126</sup> as 2^-126.
$("sup").each((_, sup) => void $(sup).replaceWith(`^${$(sup).text()}`));
// A block ends a line, as a browser lays it out.
$("p, li, div, br").after(" ");

/** The text of the elements, its whitespace collapsed as a browser renders it. */
const textOf = (elements: Parameters<typeof $>[0]) =>
  $(elements)
    .text()
    .replace(/\s+/g, " ")
    .replace(/ ([,.\]])/g, "$1")
    .trim();

/** The elements after the heading with `id`, up to the next heading. */
const section = (id: string) => $(`#${id}`).nextUntil("h2, h3, h4, h5, h6");

const status = textOf("#w3c-state");
const correctlyRounded =
  /Let x be the exact[\s\S]*?does not specify a rounding mode\./.exec(
    textOf(section("floating-point-accuracy")),
  )?.[0] ?? "";
const rows: Record<string, string> = {};
section("concrete-float-accuracy")
  .find("tr")
  .each((_, row) => {
    const cells = $(row).children("td");
    if (cells.length >= 2) rows[textOf(cells.eq(0))] = textOf(cells.eq(1));
  });
const f32 = (expression: string) => rows[expression] ?? null;

console.log(
  JSON.stringify(
    {
      probe: "wgsl-accuracy-table",
      environment: { node: process.version, source: SOURCE, status, read: new Date().toISOString().slice(0, 10) },
      correctlyRounded,
      f32: rows,
      checks: {
        correctlyRoundedMayRoundEitherWay: /may be rounded up or down: WGSL does not specify a rounding mode/.test(
          correctlyRounded,
        ),
        basicOperationsAreCorrectlyRounded: ["x + y", "x - y", "x * y"].every((e) => f32(e) === "Correctly rounded"),
        divisionIs2Point5UlpInRange: f32("x / y") === "2.5 ULP for |y| in the range [2^-126, 2^126]",
        atanIs4096Ulp: f32("atan(x)") === "4096 ULP",
        atan2Is4096UlpInRange:
          f32("atan2(y, x)") === "4096 ULP for |x| in the range [2^-126, 2^126], and y is finite and normal",
        asinIsWorseOfAbsoluteAndAtan2:
          f32("asin(x)") === "The worse of: Absolute error 6.81×10^-5 Inherited from atan2(x, sqrt(1.0 - x * x))",
        acosIsWorseOfAbsoluteAndAtan2:
          f32("acos(x)") === "The worse of: Absolute error 6.77×10^-5 Inherited from atan2(sqrt(1.0 - x * x), x)",
        sinAndCosAreAbsoluteInPi: ["sin(x)", "cos(x)"].every(
          (e) => f32(e) === "Absolute error at most 2^-11 when x is in the interval [-π, π]",
        ),
        expAndExp2Are3Plus2x: ["exp(x)", "exp2(x)"].every((e) => f32(e) === "3 + 2 * |x| ULP"),
        logAndLog2AreAbsoluteNearOne: ["log(x)", "log2(x)"].every(
          (e) =>
            f32(e) ===
            "Absolute error at most 2^-21 when x is in the interval [0.5, 2.0]. 3 ULP when x is outside the interval [0.5, 2.0].",
        ),
        inverseSqrtIs2Ulp: f32("inverseSqrt(x)") === "2 ULP",
        tanhIsWorseOfAbsoluteAndFormula:
          f32("tanh(x)") === "The worse of: Absolute error 1.0×10^-5 Inherited from sinh(x) / cosh(x)",
        tanInheritsSinOverCos: f32("tan(x)") === "Inherited from sin(x) / cos(x)",
        powInheritsExp2OfLog2: f32("pow(x, y)") === "Inherited from exp2(y * log2(x))",
        sinhInherits: f32("sinh(x)") === "Inherited from (exp(x) - exp(-x)) * 0.5",
        coshInherits: f32("cosh(x)") === "Inherited from (exp(x) + exp(-x)) * 0.5",
        asinhInherits: f32("asinh(x)") === "Inherited from log(x + sqrt(x * x + 1.0))",
        acoshInherits: f32("acosh(x)") === "Inherited from log(x + sqrt(x * x - 1.0))",
        atanhInherits: f32("atanh(x)") === "Inherited from log( (1.0 + x) / (1.0 - x) ) * 0.5",
        sqrtInheritsInverseSqrt: f32("sqrt(x)") === "Inherited from 1.0 / inverseSqrt(x)",
        determinantIsUnbounded: Object.entries(rows).some(
          ([e, a]) => e.startsWith("determinant") && a.startsWith("Infinite ULP"),
        ),
        derivativesAreUnbounded: Object.entries(rows).some(
          ([e, a]) => e.startsWith("dpdx") && a.startsWith("Infinite ULP"),
        ),
      },
    },
    null,
    2,
  ),
);
