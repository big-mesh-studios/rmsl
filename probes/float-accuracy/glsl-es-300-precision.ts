/**
 * Question: which built-in functions does GLSL ES 3.00 bound the error of at
 * `highp`, by how much, and which does it leave without a precision?
 *
 * Reads the specification itself, revision 6, from the Khronos registry, and
 * checks its SHA-256 so a later run can tell whether the document changed. It
 * prints the passage of section 4.5.1 that gives the precision of each
 * operation, the rows of section 8.1 that define the trigonometric and
 * hyperbolic functions, the row of section 8.5 for `normalize` and that of
 * section 8.6 for `inverse`, as the PDF holds them, with checks of the claims
 * `fact-glsl-es-300-bounds-the-error-of-some-float-built-ins` makes.
 *
 * Run: node probes/float-accuracy/glsl-es-300-precision.ts > probes/float-accuracy/results/glsl-es-300-precision.json
 */
import { createHash } from "node:crypto";
import { pdfPages } from "./pdf-text.ts";

const SOURCE = "https://registry.khronos.org/OpenGL/specs/es/3.0/GLSL_ES_Specification_3.00.pdf";
const SHA256 = "325058092a1c96a172f6e5b1e4008e94e7a2c75b63d6431f5bec273f374b6e7f";

const pdf = Buffer.from(await (await fetch(SOURCE)).arrayBuffer());
const sha256 = createHash("sha256").update(pdf).digest("hex");
const text = pdfPages(pdf).join("\n");

/** The text from `start` to the end of `end`, or null where either is missing. */
function between(start: string, end: string): string | null {
  const from = text.indexOf(start);
  const to = text.indexOf(end, from);
  return from < 0 || to < 0 ? null : text.slice(from, to + end.length);
}

const precision = between("Precisions are expressed", "the trigonometric functions and determinant.");
const functions = between("The standard trigonometric sine function.", "8.2 Exponential Functions");
const matrix = between("inverse (mat2", "(nearly singular).");
const geometric = between("normalize  (genType", "length(x)");
/**
 * The text with its whitespace removed, as the PDF spaces words unevenly, and
 * its private-use characters, the glyphs it draws the `+` of the `cosh`
 * equation and the brackets of the `tanh` one with.
 */
const flat = (s: string | null) => (s ?? "").replace(/[\s\uE000-\uF8FF]+/g, "");

console.log(
  JSON.stringify(
    {
      probe: "glsl-es-300-precision",
      environment: {
        node: process.version,
        source: SOURCE,
        sha256,
        sameDocument: sha256 === SHA256,
        read: new Date().toISOString().slice(0, 10),
      },
      passages: { "4.5.1": precision, "8.1": functions, "8.5": geometric, "8.6": matrix },
      checks: {
        divisionIs2Point5Ulp: /a\/b,1.0\/b2.5ULP/.test(flat(precision)),
        powInheritsFromExp2OfLog2: /pow\(x,y\)Inheritedfromexp2\(x\*log2\(y\)\)/.test(flat(precision)),
        expIs3Plus2xUlp: /exp\(x\),exp2\(x\)\(3\+2\*\|x\|\)ULP/.test(flat(precision)),
        logIs3UlpOutsideAndAbsoluteInside:
          /3ULPoutsidetherange\[0.5,2.0\]\.Absoluteerror<2-21insidetherange\[0.5,2.0\]/.test(flat(precision)),
        sqrtInheritsFromInversesqrt: /sqrt\(\)Inheritedfrom1.0\/inversesqrt\(\)/.test(flat(precision)),
        inversesqrtIs2Ulp: /inversesqrt\(\)2ULP/.test(flat(precision)),
        basicOperationsAreCorrectlyRounded: /a\+b,a–b,a\*bCorrectlyrounded/.test(flat(precision)),
        roundingModeUndefined: /Theroundingmodeisnotdefinedbutmustnotaffecttheresultbymorethan1ULP/.test(
          flat(precision),
        ),
        undefinedPrecisionNamesTrigonometryAndDeterminant:
          /notdefinedasequationsoftheabovehaveundefinedprecision\.Theseinclude,forexample,thetrigonometricfunctionsanddeterminant\./.test(
            flat(precision),
          ),
        sinhIsByEquation: /Returnsthehyperbolicsinefunctionex−e−x2/.test(flat(functions)),
        coshIsByEquation: /Returnsthehyperboliccosinefunctionexe−x2/.test(flat(functions)),
        tanhIsSinhOverCosh: /Returnsthehyperbolictangentfunctionsinhxcoshx/.test(flat(functions)),
        asinhIsOnlyAnInverse: /Archyperbolicsine;returnstheinverseofsinh\./.test(flat(functions)),
        acoshIsOnlyAnInverse: /Archyperboliccosine;returnsthenon-negativeinverseofcosh\./.test(flat(functions)),
        atanhIsOnlyAnInverse: /Archyperbolictangent;returnstheinverseoftanh\./.test(flat(functions)),
        normalizeIsXOverLength:
          /normalize\(genTypex\)Returnsavectorinthesamedirectionasxbutwithalengthof1i\.e\.xlength\(x\)/.test(
            flat(geometric),
          ),
        geometricFunctionsInherit:
          /Theseinclude,forexample,thegeometricfunctions,thecommonfunctions,andmanyofthematrixfunctions\./.test(
            flat(precision),
          ),
        inverseIsOnlyDescribed:
          /Returnsamatrixthatistheinverseofm\.Theinputmatrixmisnotmodified\.Thevaluesinthereturnedmatrixareundefinedifmissingularorpoorly-/.test(
            flat(matrix),
          ),
      },
    },
    null,
    2,
  ),
);
