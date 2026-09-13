# Comment Redundancy Audit

**Criterion:** Does the comment add information the code itself does not already convey? Section headers that label what follows, inline comments that restate the code, and test comments that restate the test name are flagged for removal.

**Legend:**

- **REMOVE** — redundant: the code, function name, or test name already says the same thing.
- **KEEP** — adds value: explains _why_, non-obvious design rationale, language constraints, or subtle interactions the code cannot express.
- **N/A** — `describe`/`it` strings (not removable comments).

**Author:**

- **random mesh** — Clinton Selke
- **bigmistqke** — you

---

## `src/rmsl.ts` (core library)

### Section headers (REMOVE)

All `// === ... ===` dividers are redundant — they label the block, but the block's own names make it obvious. All written by **random mesh**.

| Lines | Author      |
| ----- | ----------- |
| 3     | random mesh |
| 14    | random mesh |
| 26    | random mesh |
| 35    | random mesh |
| 61    | random mesh |
| 74    | random mesh |
| 97    | random mesh |
| 135   | random mesh |
| 322   | random mesh |
| 511   | random mesh |
| 538   | random mesh |
| 758   | random mesh |
| 822   | random mesh |
| 956   | random mesh |
| 1055  | random mesh |
| 1081  | random mesh |
| 1178  | random mesh |
| 1180  | random mesh |
| 1242  | random mesh |
| 2099  | random mesh |
| 3270  | random mesh |

### Method group labels (REMOVE)

`// === ArithOps ===` etc. inside `NodeImpl` — interface name makes these obvious. All written by **random mesh**.

Lines: 337, 360, 383, 391, 404, 407, 425, 432, 448, 455, 480

### Trivial labels (REMOVE)

| Lines | Comment                                                                              | Author      | Why it's redundant                          |
| ----- | ------------------------------------------------------------------------------------ | ----------- | ------------------------------------------- |
| 614   | `/** The result type of an op, given the type of the operand that defines it. */`    | bigmistqke  | Function name `resultType` says this        |
| 621   | `/** Component count per type, for the operations whose width follows it. */`        | bigmistqke  | `TYPE_WIDTH` name and values say this       |
| 1181  | `/** What compiling one node yields: statements to emit, and how to refer to it. */` | bigmistqke  | Fields `decls`, `body`, `expr` say this     |
| 1366  | `/** Which component each accessor letter names, in both spellings. */`              | bigmistqke  | `COMPONENT_INDEX` values say this           |
| 1610  | `// Binary math ops (same pattern for all)`                                          | random mesh | Case labels say this                        |
| 1635  | `// Comparison ops`                                                                  | random mesh | Case labels say this                        |
| 1670  | `// Unary math ops`                                                                  | random mesh | Case labels say this                        |
| 1721  | `// let`                                                                             | random mesh | Case label is `"let"`                       |
| 2152  | `/** Struct type and binding name holding every uniform in a WGSL shader. */`        | bigmistqke  | Constant name says this                     |
| 2156  | `/** Byte size and alignment of each WGSL type, per the spec's layout rules. */`     | bigmistqke  | Field names and values say this             |
| 2237  | `/** How a member is written in the struct: array<T, N> for arrays, else T. */`      | bigmistqke  | Function name `wgslMemberType` says this    |
| 2256  | JSDoc on `isWgslTexture`                                                             | bigmistqke  | Function name and body are self-explanatory |

### Comments that restate the code (REMOVE)

| Lines     | Comment                                                                            | Author      | Why it's redundant                                             |
| --------- | ---------------------------------------------------------------------------------- | ----------- | -------------------------------------------------------------- |
| 503       | `// Cast constructor so new Node<T>(...) returns Node<T> with conditional methods` | random mesh | Code does this                                                 |
| 773-775   | `// Supports single return... Supports multi return... Supports parameters...`     | random mesh | The three overloads in the type signature already express this |
| 1443-1445 | Comment about memo cache in `compileGLSLStage`                                     | bigmistqke  | Already explained in the `memo` JSDoc                          |
| 1458      | `// Constant folding`                                                              | bigmistqke  | `tryFold` is self-describing                                   |
| 1530-1531 | Comment about float-to-int index conversion                                        | bigmistqke  | The ternary makes this visible                                 |
| 1796-1804 | Loop init hoisting logic comments                                                  | random mesh | Code logic is clear in context                                 |
| 2347-2349 | Memo comment (same pattern as GLSL)                                                | bigmistqke  | Already explained in the `memo` JSDoc                          |
| 2362      | `// Constant folding`                                                              | random mesh | Same pattern as GLSL                                           |
| 2478      | Comment about float index conversion                                               | bigmistqke  | Already explained earlier                                      |
| 2719-2720 | Comment about positionWritten                                                      | bigmistqke  | Same pattern as GLSL                                           |
| 2742      | `// A single component is directly assignable...`                                  | bigmistqke  | `if (resolved.pattern.length === 1)` says this                 |
| 2919-2920 | Comment about floored modulus                                                      | bigmistqke  | Already explained above in the `mod` case                      |
| 3108-3111 | Comment about stage output type checking (WGSL)                                    | bigmistqke  | Same comment as GLSL counterpart                               |
| 3119-3121 | Comment about `hasVec4Result` (WGSL)                                               | bigmistqke  | Same as GLSL                                                   |
| 3129-3130 | Comment about texture bindings vs struct                                           | bigmistqke  | Already in `wgslUniformLayout` JSDoc                           |

### Design rationale (KEEP)

| Lines     | Comment                                                   | Author      | Why it's valuable                                             |
| --------- | --------------------------------------------------------- | ----------- | ------------------------------------------------------------- |
| 44-45     | Aliases rather than interfaces                            | bigmistqke  | Explains a type-system constraint the code cannot express     |
| 48-52     | JSDoc on `UniformArrayNode`                               | bigmistqke  | Explains why it's not a `Node<A>` itself                      |
| 98-107    | JSDoc on `NodeOps`                                        | bigmistqke  | Registry-not-conditionals rationale, defunctionalisation      |
| 157-165   | `step`/`smoothstep` on `FloatMathOps`                     | bigmistqke  | Why these aren't on `VecCommonOps`                            |
| 169-184   | JSDoc on `ComparisonOps`                                  | bigmistqke  | `Operand` parameter design, scalar-vs-vector refusal          |
| 212-215   | JSDoc on `MatOps`                                         | bigmistqke  | What `Vec` means in context                                   |
| 226-237   | JSDoc on `RectMatOps`                                     | bigmistqke  | Non-square matrix semantics, why no `inverse`                 |
| 248       | `/** A cube map is sampled with a direction... */`        | bigmistqke  | Semantic difference from `SamplerOps.texture`                 |
| 303-306   | JSDoc on `BoolVecOps`                                     | bigmistqke  | Why there's no implicit `bool` conversion                     |
| 582-591   | JSDoc on `REDUCING_OPS`                                   | bigmistqke  | Why certain ops have a different result type                  |
| 596-604   | Comments inside `REDUCING_OPS`                            | bigmistqke  | Non-obvious matrix dimension logic                            |
| 628-639   | JSDoc on `VALUE_OPERAND`                                  | bigmistqke  | `step`/`smoothstep` argument order quirk                      |
| 641-655   | JSDoc on `UNIFORM_OPERAND_OPS`                            | bigmistqke  | Broadcasting rationale, distinction from genuinely-scalar ops |
| 657-667   | JSDoc on `typedOperand`                                   | bigmistqke  | Subtle issue with plain numbers beside integer operands       |
| 715-719   | JSDoc on `comp`                                           | bigmistqke  | Component-wise comparison semantics, GLSL/TSL alignment       |
| 961-976   | JSDoc on `uniformArray`                                   | bigmistqke  | WGSL slot-constraint rationale, difference from TSL           |
| 1189-1210 | JSDoc on `memo` field                                     | bigmistqke  | DAG-not-tree, why deduplication exists                        |
| 1211-1217 | JSDoc on `wgslHelpers` field                              | bigmistqke  | Why helpers are demand-emitted for WGSL only                  |
| 1218-1221 | JSDoc on `positionWritten`                                | bigmistqke  | What this flag controls                                       |
| 1262-1266 | Comment about JS `%` vs floored modulus                   | bigmistqke  | Subtle correctness: JS truncates, shader floors               |
| 1301-1311 | JSDoc on `forUpdateStatements`                            | bigmistqke  | GLSL comma-expression vs WGSL single-statement                |
| 1325-1328 | JSDoc on `withoutSemicolon`                               | bigmistqke  | Why semicolons need stripping                                 |
| 1331-1341 | JSDoc on `assertStageResult`                              | bigmistqke  | Why vertex is checked but fragment is not                     |
| 1372-1378 | JSDoc on `resolveSwizzleTarget`                           | bigmistqke  | Non-obvious composed swizzle resolution                       |
| 1393-1400 | JSDoc on `assertSquareMatrix`                             | bigmistqke  | Why both backends check                                       |
| 1412-1416 | JSDoc on `assertPositionIsReadable`                       | bigmistqke  | GLSL/WGSL difference for position readability                 |
| 1590-1593 | Comment about boolean vector `!` vs `not()`               | bigmistqke  | GLSL quirk: scalar-only `!`                                   |
| 1616-1618 | Comment about GLSL `%` vs `mod()`                         | bigmistqke  | Why float paths diverge from integer paths                    |
| 1738-1740 | Comment about `builtinPosition` in `assign`               | bigmistqke  | Side effect on `positionWritten`                              |
| 1845-1850 | Comment in `default` case                                 | bigmistqke  | Why a throw is preferred over silent placeholder              |
| 1972-1975 | Comment about stage output type checking                  | bigmistqke  | `lastType` tracking rationale                                 |
| 1983-1989 | Comment about implicit color output                       | bigmistqke  | GLSL ES 3.00 `gl_FragColor` removal                           |
| 2015      | Comment about `outputLocation` numbering                  | bigmistqke  | Why locations reset per shader                                |
| 2019-2021 | Comment about location qualifier                          | bigmistqke  | GLSL ES 3.00 restriction                                      |
| 2051-2054 | Comment about declared output assignment                  | bigmistqke  | Why declared outputs aren't overwritten                       |
| 2063-2079 | JSDoc on `VertexRoot`                                     | bigmistqke  | Full design of vertex stage input                             |
| 2089-2095 | Comments on `compileGLSL` assignment                      | bigmistqke  | Void → node mapping                                           |
| 2111-2116 | JSDoc on `MATRIX_DIMENSIONS`                              | bigmistqke  | "GLSL/WGSL matCxR is C columns of R rows" spec reminder       |
| 2123-2134 | JSDoc on `wgslMatrixArgs`                                 | bigmistqke  | WGSL/GLSL constructor difference                              |
| 2164-2166 | Comment about bool carrier types                          | bigmistqke  | Non-obvious bool-as-uint substitution                         |
| 2173-2175 | Comment about matCxR column alignment                     | bigmistqke  | Non-obvious WGSL layout rule                                  |
| 2187-2206 | JSDoc on `WgslUniformMember` fields                       | bigmistqke  | `stride` vs `size` distinction                                |
| 2208-2218 | JSDoc on `WGSL_ARRAY_PADDING`                             | bigmistqke  | Non-obvious padding strategy                                  |
| 2227-2232 | Comments inside `WGSL_ARRAY_PADDING`                      | bigmistqke  | Bool-as-uint comparison strategy                              |
| 2244-2255 | JSDoc on `wgslUniformLayout`                              | bigmistqke  | 12-uniform-buffer limit, ordering strategy                    |
| 2268-2270 | Comment about array stride rounding                       | bigmistqke  | Non-obvious WGSL layout rule                                  |
| 2279-2281 | Comment about guessing being dangerous                    | bigmistqke  | Severity of wrong size — silent corruption                    |
| 2294-2295 | Comment about alignment ordering                          | bigmistqke  | Sorting strategy                                              |
| 2318-2321 | Comment about struct alignment                            | bigmistqke  | Final struct size calculation nuance                          |
| 2389-2390 | Comment about WGSL narrowing constructor                  | bigmistqke  | Why WGSL needs different code for vec truncation              |
| 2428-2432 | Comment about bool-as-uint                                | bigmistqke  | Needed here for the WGSL uniform path                         |
| 2444-2446 | Comment about textures vs value uniforms                  | bigmistqke  | Qualified vs unqualified reference difference                 |
| 2471-2478 | Comments about float index and array padding              | bigmistqke  | Non-obvious WGSL-specific logic                               |
| 2541      | `// Unlike GLSL, WGSL's ! is defined for vecN<bool> too.` | bigmistqke  | Cross-language difference                                     |
| 2565-2570 | Comments about floored modulus helper                     | bigmistqke  | Why a helper is needed                                        |
| 2603-2604 | Comment about WGSL shift amount                           | bigmistqke  | Non-obvious WGSL constraint                                   |
| 2648-2649 | Comment about WGSL no inverse()                           | bigmistqke  | Demand-emitted helper strategy                                |
| 2726-2731 | Comments about multi-component swizzle                    | bigmistqke  | WGSL limitation and temp-binding strategy                     |
| 2734-2737 | Comment about swizzle-not-compiled                        | bigmistqke  | Why resolved not compiled                                     |
| 2829      | `// WGSL for-init needs var not let`                      | random mesh | String-manipulation trick                                     |
| 2837-2840 | Comments about continuing block                           | bigmistqke  | Multi-statement update WGSL approach                          |
| 2902-2907 | Default case comment (same as GLSL)                       | bigmistqke  | Silent corruption vs throw                                    |
| 2911-2917 | JSDoc on `WGSL_HELPERS`                                   | bigmistqke  | What helpers are for and the algorithm                        |
| 2996-3000 | JSDoc on `shiftWGSL`                                      | bigmistqke  | u32 conversion requirement                                    |
| 3153-3155 | Comment about helpers and sorting                         | bigmistqke  | Deterministic output rationale                                |
| 3171-3173 | Comment about shared slot numbering                       | bigmistqke  | Why varyings and outputs share one counter                    |
| 3202-3205 | Comment about empty struct WGSL limitation                | bigmistqke  | Why optional return struct                                    |
| 3348-3352 | Comments about helpers in standalone fn                   | bigmistqke  | Difference from whole-shader                                  |
| 3370-3371 | Comment about single-struct packing                       | bigmistqke  | Why standalone functions use same approach                    |
| 3373-3375 | Comment about texture-sampler companion                   | bigmistqke  | Why both are declared together                                |

---

## `src/rmsl-usage.test.ts` (integration tests)

### Section headers (REMOVE)

All `// === ... ===` and `// -- ... --` dividers are redundant.

| Lines | Author      |
| ----- | ----------- |
| 166   | random mesh |
| 296   | random mesh |
| 326   | random mesh |
| 338   | random mesh |
| 361   | random mesh |
| 410   | random mesh |
| 657   | random mesh |
| 693   | random mesh |
| 764   | random mesh |
| 871   | random mesh |
| 886   | random mesh |
| 903   | random mesh |
| 912   | random mesh |
| 923   | random mesh |
| 954   | random mesh |
| 970   | random mesh |

### Comments that restate the test name (REMOVE)

| Lines     | Comment                                                                         | Author      | Why it's redundant                                                      |
| --------- | ------------------------------------------------------------------------------- | ----------- | ----------------------------------------------------------------------- |
| 228       | `// refract(I, N, eta) takes three arguments.`                                  | bigmistqke  | Test name says "compiles refract with all three arguments"              |
| 241-242   | Comment about for-loop update clause                                            | bigmistqke  | Already in compiler JSDoc                                               |
| 275-276   | Comment about dot/length/distance result types                                  | bigmistqke  | Test name says "types length/distance/dot as float"                     |
| 345-346   | Comment about plain number types                                                | bigmistqke  | Test name says "gives a plain number the operand's integer type"        |
| 415       | Comment about `gl_Position` being vec4                                          | random mesh | Trivially obvious                                                       |
| 471       | `// Declared outputs are assigned by the program.`                              | bigmistqke  | Test body shows explicit `assign` call                                  |
| 571-574   | Comments about varying direction                                                | random mesh | Assertions (out vs in) say this                                         |
| 592       | `// A vertex shader must produce a position, and zero is not a valid position.` | bigmistqke  | Test name says "rejects a vertex result of zero"                        |
| 608       | `// Writing the position explicitly is the documented way to set it.`           | bigmistqke  | Test name says "allows a vertex shader that writes the position itself" |
| 621       | `// With no explicit write, the result still has to be able to become one.`     | bigmistqke  | Obvious from test name                                                  |
| 628-629   | Comment about void return                                                       | bigmistqke  | Obvious from test name                                                  |
| 662-663   | Comment about `@builtin(position)` being vec4                                   | random mesh | Trivially obvious                                                       |
| 713-714   | Comment about WGSL swizzle assignment                                           | random mesh | Already in compiler JSDoc                                               |
| 774       | Comment about constant folding                                                  | random mesh | Obvious in context                                                      |
| 805-807   | Comment about scalar broadcast                                                  | random mesh | Already in compiler                                                     |
| 823-824   | Comment about step/smoothstep result type                                       | random mesh | Already in compiler JSDoc                                               |
| 1046      | `// WGSL has no inverse() builtin...`                                           | bigmistqke  | Already in compiler source                                              |
| 1149-1150 | Comment about raw JS arrays                                                     | bigmistqke  | Obvious from test and code                                              |
| 1256-1257 | Comment about two-step loop update                                              | bigmistqke  | Test name and body make this clear                                      |
| 1291      | `// The ordinary single-statement loop keeps the plain for-header it had.`      | bigmistqke  | Obvious from test name                                                  |
| 1308-1309 | Comment about plain number typing                                               | bigmistqke  | Already in compiler source                                              |
| 1320      | `// A literal that is not a whole number cannot take an integer type.`          | bigmistqke  | Obvious from test name                                                  |
| 1326-1327 | Comment about negative unsigned                                                 | bigmistqke  | Obvious from test name                                                  |
| 1333-1334 | Comment about multi-component swizzle                                           | bigmistqke  | Already in compiler source                                              |
| 1347-1349 | Comment about position readability                                              | bigmistqke  | Already in compiler `assertPositionIsReadable`                          |

### Comments that add value (KEEP)

| Lines     | Comment                                                                        | Author     | Why it's valuable                                  |
| --------- | ------------------------------------------------------------------------------ | ---------- | -------------------------------------------------- |
| 10-13     | Block comment about recording compilers                                        | bigmistqke | Explains the aliasing mechanism and harness design |
| 179-181   | Comment about vector comparison result type                                    | bigmistqke | Why result is bvecN, not bool                      |
| 221       | `// GLSL's ! is scalar-only...`                                                | bigmistqke | Language difference motivating the test            |
| 458-459   | Comment about `gl_FragColor` removal                                           | bigmistqke | Why implicit output is needed                      |
| 493       | `// Everything a vertex stage passes onward shares one set of numbered slots.` | bigmistqke | Slot numbering design                              |
| 508-509   | Comment about vertex output location qualifier                                 | bigmistqke | GLSL ES 3.00 restriction                           |
| 583       | `// Skipping the write instead would link cleanly and draw nothing.`           | bigmistqke | Why this error matters                             |
| 636       | `// A fragment shader with no colour output is legal...`                       | bigmistqke | Asymmetry with vertex                              |
| 814-817   | Comment about scalar-vs-vector type error                                      | bigmistqke | Points reader to `.test-d.ts`                      |
| 985-989   | Comments about standalone function compilers                                   | bigmistqke | Why output needs manual wrapping                   |
| 1019      | `// uniformRaw names its own slot...`                                          | bigmistqke | Difference from `uniform()`                        |
| 1036-1038 | Comments about `@ts-expect-error`                                              | bigmistqke | Why `@ts-expect-error` is used instead of a cast   |
| 1070-1074 | Comments about breadth coverage                                                | bigmistqke | Purpose of breadth tests                           |
| 1076      | `// Driven by a uniform because constant folding collapses these on literals.` | bigmistqke | Test design choice                                 |
| 1086-1087 | Comment about WGSL name differences                                            | bigmistqke | Why WGSL assertions are looser                     |
| 1106-1108 | Comments about scalar matrix constructor expansion                             | bigmistqke | WGSL expansion rationale                           |
| 1125-1126 | Comments about matrix-from-matrix constructor                                  | bigmistqke | Difference from scalar diagonal                    |
| 1166-1172 | Comments about shared nodes / DAG                                              | bigmistqke | DAG problem memo cache addresses                   |
| 1189-1191 | Comment about shared block variable scope                                      | bigmistqke | Scoping issue with duplicated blocks               |
| 1209-1211 | Comment about loop counter declaration                                         | bigmistqke | Init-intact memo requirement                       |
| 1232-1236 | Comments about modulus helper rationale                                        | bigmistqke | Performance motivation for helpers                 |
| 1283-1284 | Comment about continuing block semantics                                       | bigmistqke | Why `continuing` is used                           |
| 1362-1363 | Comment about bool uniform host-shareable type                                 | bigmistqke | Why a carrier type is needed                       |

---

## `src/rmsl.test-d.ts` (type tests)

### Comments that restate the test name (REMOVE)

| Lines   | Comment                                            | Author     | Why it's redundant                                           |
| ------- | -------------------------------------------------- | ---------- | ------------------------------------------------------------ |
| 17-18   | Comment about scalar vs vector comparison          | bigmistqke | Type expressions are self-documenting                        |
| 27      | `// A vector against a scalar broadcasts...`       | bigmistqke | Obvious from test name                                       |
| 33-34   | Comment about reverse comparison                   | bigmistqke | Obvious from test name                                       |
| 60      | `// A matrix column, not a matrix.`                | bigmistqke | Test name says "types a matrix element as the column vector" |
| 67-68   | Comment about value operand order                  | bigmistqke | Already in compiler JSDoc                                    |
| 85-86   | Comment about void return                          | bigmistqke | Obvious from test name                                       |
| 133     | `// A matCxR has C columns of R rows...`           | bigmistqke | Already in compiler JSDoc                                    |
| 141     | `// Transposing swaps the two...`                  | bigmistqke | Obvious from test name                                       |
| 147     | `// Multiplying takes one component per column...` | bigmistqke | Obvious from test name                                       |
| 154     | `// Only a square matrix has an inverse...`        | bigmistqke | Obvious from test name                                       |
| 161-162 | Comment about uniform name+ops                     | bigmistqke | Obvious from test body                                       |

### Comments that add value (KEEP)

| Lines   | Comment                               | Author     | Why it's valuable                                              |
| ------- | ------------------------------------- | ---------- | -------------------------------------------------------------- |
| 1-8     | Block comment                         | bigmistqke | Explains what type-level tests catch that runtime tests cannot |
| 77-79   | Comment about vertex stage acceptance | bigmistqke | Why anything that isn't vec4 is refused                        |
| 92-93   | Comment about tuple return semantics  | bigmistqke | "Last is position" rule                                        |
| 115-116 | Comment about fragment stage          | bigmistqke | Asymmetry with vertex                                          |
| 123-126 | Comment about matrix operations       | bigmistqke | Bug of missing declarations on non-square matrices             |

---

## `src/rmsl-eval.test.ts` (evaluation tests)

### Comments that restate the test name (REMOVE)

| Lines   | Comment                                           | Author     | Why it's redundant                   |
| ------- | ------------------------------------------------- | ---------- | ------------------------------------ |
| 52      | `// min and max must agree across both backends.` | bigmistqke | All eval tests compare both backends |
| 137-138 | Comment about two-statement update                | bigmistqke | Obvious from test name               |
| 195-196 | Comment about break/continue                      | bigmistqke | Obvious from test name               |

### Comments that add value (KEEP)

| Lines   | Comment                                         | Author     | Why it's valuable                                                |
| ------- | ----------------------------------------------- | ---------- | ---------------------------------------------------------------- |
| 1-11    | Block comment                                   | bigmistqke | Explains what evaluation tests catch that text assertions cannot |
| 25-40   | Comments on `expectValue` and tolerance         | bigmistqke | Non-obvious scaled-tolerance design                              |
| 68-69   | Comment about argument order in step/smoothstep | bigmistqke | Primary risk being tested                                        |
| 84-85   | Comment about floored modulus                   | bigmistqke | Sign semantics being tested                                      |
| 93-94   | Comment about folding correctness               | bigmistqke | JS fold vs GPU runtime agreement                                 |
| 107-118 | Block comment about control flow testing        | bigmistqke | Sum pins whole loop, risk of hangs, Fn wrapping                  |
| 150-153 | Comment about WGSL loop header                  | bigmistqke | WGSL single-statement limitation                                 |
| 210-212 | Comment about break/continue                    | bigmistqke | Why sum is used to verify                                        |

---

## `src/testing/shader-validity.ts` (test harness)

### Comments that restate the code (REMOVE)

| Lines   | Comment                                  | Author     | Why it's redundant                         |
| ------- | ---------------------------------------- | ---------- | ------------------------------------------ |
| 100-104 | Comment about return optimization        | bigmistqke | Obvious from code                          |
| 154-159 | JSDoc on `recordingGLSL`/`recordingWGSL` | bigmistqke | Obvious from function name and `wrap` call |
| 161-163 | JSDoc on `validateGLSL`                  | bigmistqke | Obvious from function name                 |

### Comments that add value (KEEP)

| Lines   | Comment                                | Author     | Why it's valuable                                                 |
| ------- | -------------------------------------- | ---------- | ----------------------------------------------------------------- |
| 1-25    | Block comment                          | bigmistqke | Entire motivation — three real bugs, parser-vs-type-checker table |
| 33-36   | Comments about `declare const process` | bigmistqke | Why Node types aren't imported, why `console.warn` isn't used     |
| 49-57   | JSDoc on `Recorded` fields             | bigmistqke | `pair` field especially needs explanation                         |
| 63-71   | JSDoc on `KNOWN_INVALID`               | bigmistqke | Cannot-outlive-bugs invariant                                     |
| 73-78   | JSDoc on `recordBoth`                  | bigmistqke | Parity-by-structure design                                        |
| 88-93   | Comment about compile error recording  | bigmistqke | Why errors are recorded not swallowed                             |
| 107-114 | JSDoc on `recordShaderSource`          | bigmistqke | Why this function exists separately from `recordBoth`             |
| 130-137 | JSDoc on `Compiler` interface          | bigmistqke | Aliasing-trades-typing trade-off                                  |
| 166-188 | JSDoc on `validateWGSL`                | bigmistqke | Synchronous-then-async stack pattern                              |
| 190-193 | JSDoc on `assertRecordedShadersValid`  | bigmistqke | Mutation testing exclusion, skip/silent caveat                    |
| 194-222 | Comment block inside function          | bigmistqke | Mutation testing comment, async release timing                    |
| 225-231 | JSDoc on `validationReport`            | bigmistqke | Why it's separate from GPU work                                   |
| 236-248 | Comment inside `validationReport`      | bigmistqke | "No shaders recorded" trap                                        |
| 253-255 | Comment about every failing shader     | bigmistqke | Breadth-test interaction                                          |
| 264-272 | Comment about "both refuse" filtering  | bigmistqke | Consistency filter                                                |
| 274-276 | Comment about null vs truthiness check | bigmistqke | Empty-string edge case                                            |
| 316-323 | JSDoc on `describe`                    | bigmistqke | Non-obvious test naming collision                                 |

---

## `src/testing/shader-eval.ts` (evaluation harness)

### Comments that restate the code (REMOVE)

| Lines   | Comment                               | Author     | Why it's redundant                                |
| ------- | ------------------------------------- | ---------- | ------------------------------------------------- |
| 50-53   | Comment about `callExpr`              | bigmistqke | Obvious from function body                        |
| 56-57   | JSDoc on `evaluateGLSL`               | bigmistqke | Function name is clear                            |
| 121-127 | JSDoc on `evaluateWGSL` and `runWGSL` | bigmistqke | Function names are clear                          |
| 184-189 | JSDoc on `evaluateBoth`               | bigmistqke | Function name is clear; block comment covers this |
| 202-204 | JSDoc on `closeEvaluators`            | bigmistqke | Trivially obvious                                 |

### Comments that add value (KEEP)

| Lines   | Comment                                           | Author     | Why it's valuable                                          |
| ------- | ------------------------------------------------- | ---------- | ---------------------------------------------------------- |
| 1-12    | Block comment                                     | bigmistqke | Approach — both backends, f32-vs-f64, rendering mechanisms |
| 19-24   | Comments about `process`                          | bigmistqke | Vitest console interception                                |
| 26-39   | JSDoc on `floatTolerance`                         | bigmistqke | 24-bit mantissa reasoning                                  |
| 66-72   | Comment inside `evaluateGLSL` about fresh context | bigmistqke | SwiftShader reliability issue                              |
| 86-87   | Comment about inline compilation                  | bigmistqke | Bundler renaming problem                                   |
| 130-136 | JSDoc on `runWGSL`                                | bigmistqke | Separate-function purpose                                  |
| 138-155 | Comments inside `runWGSL`                         | bigmistqke | Async error-scope pattern, zero-safety concern             |
| 207-220 | JSDoc on `EVALUATION_SKIPPED`                     | bigmistqke | Three-tier skip mechanism                                  |
| 222-226 | Status message                                    | bigmistqke | Important user-facing skip announcement                    |

---

## `src/testing/shader-eval.test.ts`

### Comments that restate the code (REMOVE)

| Lines | Comment                               | Author     | Why it's redundant     |
| ----- | ------------------------------------- | ---------- | ---------------------- |
| 40-41 | Comment about near-zero floor         | bigmistqke | Obvious from test name |
| 46-47 | Comment about tight-enough-to-catch   | bigmistqke | Obvious from test name |
| 55-56 | Comment about compile error reporting | bigmistqke | Obvious from test name |
| 72-77 | Comment about reading back values     | bigmistqke | Obvious                |

### Comments that add value (KEEP)

| Lines | Comment                                    | Author     | Why it's valuable                                          |
| ----- | ------------------------------------------ | ---------- | ---------------------------------------------------------- |
| 1-7   | Block comment                              | bigmistqke | Why harness tests matter — a green run that proves nothing |
| 18-24 | Comment block about ULP                    | bigmistqke | Rationale for the tolerance tests                          |
| 33-35 | Comment about the case that motivated this | bigmistqke | History for maintainers                                    |
| 62-63 | Comment about parser vs type-checker       | bigmistqke | What a parser-based check would miss                       |
| 80-81 | Comment about zero as genuine result       | bigmistqke | Zero-safety concern — critical for harness correctness     |

---

## `src/testing/shader-validity.test.ts`

### Comments that restate the code (REMOVE)

| Lines | Comment                                                        | Author     | Why it's redundant     |
| ----- | -------------------------------------------------------------- | ---------- | ---------------------- |
| 37    | `// Nothing recorded means nothing was checked.`               | bigmistqke | Obvious from test name |
| 50    | `// A known-invalid entry that now compiles is reported.`      | bigmistqke | Obvious from test name |
| 59    | `// A compiler that throws is recorded rather than swallowed.` | bigmistqke | Obvious from test name |

### Comments that add value (KEEP)

| Lines | Comment                                 | Author     | Why it's valuable             |
| ----- | --------------------------------------- | ---------- | ----------------------------- |
| 1-8   | Block comment                           | bigmistqke | Why these harness tests exist |
| 23-25 | Comment about multiple shaders per test | bigmistqke | Breadth-test interaction      |

---

## `vitest.config.ts`

### Comments to REMOVE

| Lines | Comment                          | Author     | Why it's redundant                                                |
| ----- | -------------------------------- | ---------- | ----------------------------------------------------------------- |
| 8-10  | Comment inside `typecheck` block | bigmistqke | `include: ["src/**/*.test-d.ts"]` already says "type-level tests" |

---

## Summary

| Category                                            | random mesh | bigmistqke | Total |
| --------------------------------------------------- | ----------- | ---------- | ----- |
| **REMOVE** (section headers)                        | ~37         | 0          | ~37   |
| **REMOVE** (trivial labels, restating code)         | 5           | 10         | ~15   |
| **REMOVE** (restating test names)                   | 6           | 29         | ~35   |
| **KEEP** (design rationale, language constraints)   | 1           | ~100       | ~101  |
| **KEEP** (harness design, non-obvious interactions) | 0           | ~40        | ~40   |

### Source of redundancy

**random mesh** wrote almost all section headers and trivial labels (~42 comments). **bigmistqke** wrote almost all test-file comments that restate the test name (~29 comments). Every KEEP comment explaining design rationale or language constraints is from **bigmistqke**.

The one random mesh comment worth keeping is `// WGSL for-init needs var not let` (line 2829) — it explains a string-manipulation trick that isn't obvious from the code.
