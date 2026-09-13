# Comment Review: Diary Notes vs. State Descriptions

**Criterion:** Comments should describe the current state, not what used to be broken and is now fixed. Diary-like notes that narrate a bug's history or a fix's story are flagged for rewrite.

**Legend:**

- **DIARY** — describes what _was_ broken, _was_ dropping, _was_ reaching, _was_ typed as, etc. Rewrite to describe the current state.
- **OK** — describes the current state, design rationale, or language constraints as they stand today.
- **MIXED** — contains both diary and state content; the diary portion should be excised.
- **N/A** — section headers, `@ts-expect-error` directives, or trivial inline labels that are not narrative.

---

## `src/rmsl.ts` (core library)

### Section headers (`// === ... ===`)

All section dividers are `N/A` — they name the block, they don't narrate.

Lines: 3, 14, 26, 35, 61, 74, 97, 138, 327, 342, 366, 389, 397, 410, 413, 433, 440, 456, 463, 488, 519, 546, 769, 783, 833, 967, 1066, 1092, 1189, 1191, 1253, 1475, 1627, 1652, 1687, 2121, 2389, 2617, 3297

### Design rationale (OK)

Lines: 44-45, 160-166, 511, 784-786, 1475, 1627, 1652, 1687 — these explain _why_ the code is structured the way it is, in present tense.

### Diary notes

| Lines             | Verbatim text (trimmed)                                                                                                                                                                              | Why it's diary                                                                                            |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| 207-210           | `Declaring them here as well gave the vector types two declarations, and this one's Node<"float"> return was wrong: GLSL's step returns genType`                                                     | "gave...two declarations" / "was wrong" — past tense narration of a bug                                   |
| 346-348           | `the widths were previously checked only for the two square types, which let a mat3 times a vec4 through as well`                                                                                    | "were previously checked" / "let...through" — describes the old bug                                       |
| 420-422           | `Left to op() it would be typed from the first operand, which is the matrix, and emitted as a float`                                                                                                 | "Left to op() it would be" — hypothetical that narrates a bug path                                        |
| 732-734           | `an unsigned operand compared against a plain number was reaching WGSL as u32 < i32`                                                                                                                 | "was reaching" — past tense narration                                                                     |
| 1364-1372         | `Returning nothing used to end the check here... Asking the text meant comparing against "0.0"... a vertex shader returning zero was read as returning nothing and slipped through`                  | Entire block is diary — narrates a sequence of past bugs                                                  |
| 2034-2037         | `the fifth output declared anywhere landed at location 4 even in a shader that had only one — past MAX_DRAW_BUFFERS soon enough`                                                                     | "landed at location 4" / "past MAX_DRAW_BUFFERS soon enough" — narrates old behavior and its consequences |
| 2318-2322         | `the tie used to be broken on the generated slot name, which carries a counter climbing for the life of the process, so the same program compiled twice could put its values at different addresses` | "used to be broken" — classic diary note                                                                  |
| 665-677 (JSDoc)   | `the backends disagreed about those: GLSL widened the operand and then could not narrow the result back, while WGSL truncated the literal and carried on`                                            | "disagreed" — past tense, describes old behavior that no longer applies                                   |
| 98-110 (JSDoc)    | `The conditional form made the checker evaluate every branch... a self-referential op like not(): Node<A> exhausted its heap`                                                                        | "made...expanded...exhausted" — full diary of a past design failure                                       |
| 2133-2140 (JSDoc) | `Holding only the square ones let the non-square ones fall through the diagonal expansion below and reach WGSL as mat2x3<f32>(2f)`                                                                   | "let...fall through...reach" — narrates an old bug                                                        |
| 2147-2158 (JSDoc) | `Expanding it instead produced mat3x3<f32>(m, 0f, 0f, 0f, m, ...), a constructor that does not exist`                                                                                                | "Expanding it instead produced" — narrates an old bug                                                     |

### Mixed notes (diary portion should be excised)

| Lines     | Verbatim text (trimmed)                                                                                                                                                                                                              | What to keep vs. remove                                                                                                                                         |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 740-743   | `Neither language compares a vector against a scalar: GLSL has no lessThan(vec3, float) and WGSL no operator < (vec3<f32>, f32). The signatures accept the mix, so the scalar is broadcast`                                          | **Keep** the first sentence (language constraint). **Remove** the "The signatures accept the mix" — that's describing an old design gap, not the current state. |
| 1460-1462 | `Reached before: its statements are already in the output, so only the expression naming the result is handed back`                                                                                                                  | **OK** — describes current behavior of DAG dedup.                                                                                                               |
| 2004-2008 | `GLSL ES 3.00 removed gl_FragColor, so an output is declared for it. This mirrors the WGSL backend, which already emitted an implicit colour output — without it the same program rendered on WebGPU and produced nothing on WebGL2` | **Keep** first sentence (language constraint). **Remove** the "without it" clause — that narrates the old bug.                                                  |

---

## `src/rmsl-usage.test.ts` (integration tests)

### Section headers (`// === ...`)

Lines: 167, 301, 331, 343, 368, 417, 680, 716, 787, 816, 894, 909, 926, 935, 946, 977, 993 — `N/A`

### Diary notes

| Lines     | Verbatim text (trimmed)                                                                                                                                                                                                      | Why it's diary                                                                  |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 10-14     | `Aliasing at the import means the tests need no changes; see src/testing/shader-validity.ts for why substring assertions were not enough`                                                                                    | "were not enough" — narrates why an old approach failed                         |
| 229-230   | `refract(I, N, eta) takes three arguments; routing it through the binary emitter silently dropped eta and produced a two-argument call`                                                                                      | "silently dropped eta" — narrates an old bug                                    |
| 243-246   | `The update clause arrives as statements, so its work sits in body while expr holds only a bare variable reference. Emitting expr alone dropped the increment and produced for (float i = 0.0; (i < 4.0); i)`                | "dropped the increment" / "produced...an infinite loop" — narrates an old bug   |
| 279-281   | `Typing the node after its first operand left _t as "vec2", which sent float comparisons down the vector path and emitted lessThan(float, float)`                                                                            | "left _t as 'vec2'" / "sent float comparisons down" — narrates an old bug       |
| 350-353   | `Typing it as float beside an integer operand made the operands disagree, so codegen inserted a conversion and produced int x = (float(u) % 2.0)`                                                                            | "made the operands disagree" / "produced..." — narrates an old bug              |
| 465-468   | `WGSL already declared an implicit output for this case; GLSL computed the value and dropped it`                                                                                                                             | "already declared" / "computed...and dropped it" — narrates old state           |
| 504-507   | `Varyings and declared outputs were numbered by two counters that knew nothing of each other, so a shader using both handed out the same slot twice and the module would not build`                                          | "were numbered" / "handed out the same slot" — narrates an old bug              |
| 606-610   | `"No result" was recognised by comparing the emitted text against "0.0"... a vertex shader returning zero slipped past the check`                                                                                            | Full diary narration of an old bug                                              |
| 626-629   | `The check on the stage result did not know that had happened, so it demanded a vec4 result as well — and following its advice would have written the position twice`                                                        | "did not know" / "would have written" — narrates old bug                        |
| 649-652   | `Returning nothing was treated as "there is no result to check", but a vertex shader still has to produce a position somehow. One that returns nothing and never writes one compiles to a main that sets no position at all` | "was treated" — narrates old behavior                                           |
| 1008-1014 | `the mutation run put 75 uncovered mutants in their shared body`                                                                                                                                                             | "the mutation run put" — narrates past measurement                              |
| 1072-1075 | `this path collected them into the same set and then dropped them, leaving a call to a function that was never defined`                                                                                                      | "collected...and then dropped them" / "was never defined" — narrates an old bug |
| 1099-1105 | `the mutation run showed these reached by no test at all... which is how they avoided being exercised in the first place`                                                                                                    | "showed" / "how they avoided" — narrates past measurement                       |
| 1136-1139 | `only the square types were listed, so the non-square ones passed straight through as mat2x3<f32>(2f)`                                                                                                                       | "were listed" / "passed straight through" — narrates an old bug                 |
| 1156-1158 | `expanding it produced mat3x3<f32>(m, 0f, 0f, 0f, m, ...)`                                                                                                                                                                   | "expanding it produced" — narrates old bug                                      |
| 1181-1182 | `Only the vec3 length had ever been exercised`                                                                                                                                                                               | "had ever been exercised" — narrates past coverage gap                          |
| 1184-1186 | `writing it that way is how the first draft of this test got caught by the shader validation rather than by its own assertions`                                                                                              | "how the first draft...got caught" — narrates past incident                     |
| 1289-1294 | `WGSL's grammar allows a single statement there, and the rest were being dropped`                                                                                                                                            | "were being dropped" — narrates old bug                                         |
| 1345-1348 | `Arithmetic learned that; comparisons did not, so an unsigned operand was compared against a signed literal and WGSL had no such overload`                                                                                   | "learned that; did not" / "was compared" — narrates old bug                     |
| 1359-1361 | `the backends disagreed about what to do instead: GLSL widened the operand and then failed to narrow the result, WGSL truncated the literal silently`                                                                        | "disagreed" / "failed" / "truncated" — narrates old behavior                    |
| 1374-1377 | `Splitting only looked at the immediate base, so a swizzle of a swizzle re-emitted the very form the split exists to avoid`                                                                                                  | "only looked at" / "re-emitted" — narrates old bug                              |
| 1405-1408 | `A bool was already carried as a u32 and compared back on read`                                                                                                                                                              | "was already carried" — narrates old state                                      |

### Mixed notes

| Lines   | Verbatim text (trimmed)                                                                                                                                                                                    | What to keep vs. remove                                                                                                             |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 480-482 | `The stage result went into every declared output, whatever its type and whatever the program had already written there. WGSL leaves declared outputs to the program's own assignments; GLSL now does too` | **Remove** first two sentences (diary). **Keep** third sentence only if reworded to "Declared outputs are assigned by the program." |
| 1045    | `uniformRaw names its own slot, unlike uniform() which generates one`                                                                                                                                      | **OK** — describes current API behavior                                                                                             |

---

## `src/rmsl.test-d.ts` (type tests)

### Diary notes

| Lines        | Verbatim text (trimmed)                                                                                             | Why it's diary                               |
| ------------ | ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| 1-11 (JSDoc) | `Several defects in this compiler were mismatches between what a signature promised and what the node actually was` | "were mismatches" — narrates old bugs        |
| 56-57        | `Getting this wrong sent float comparisons down the vector path and emitted lessThan(float, float)`                 | "Getting this wrong sent" — narrates old bug |

### OK

Lines: 20-21, 30, 36-37, 64, 72-73, 82-83, 89-90, 96-97, 119-120, 128-130, 137, 145, 151, 158, 166 — all describe current type constraints in present tense.

---

## `src/rmsl-eval.test.ts` (evaluation tests)

### Diary notes

| Lines        | Verbatim text (trimmed)                                                                                                                                                                                 | Why it's diary                                                   |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| 1-20 (JSDoc) | `emitting a - b where a + b was meant compiles perfectly and passes every text assertion. Mutation testing put a name to the gap`                                                                       | "was meant" / "put a name to the gap" — narrates past motivation |
| 61           | `The pair that motivated this file: swapping them is invisible to text`                                                                                                                                 | "motivated this file" — narrates past motivation                 |
| 93-96        | `that agrees only while both operands are positive, and that is the one case a single test would have covered`                                                                                          | "would have covered" — narrates past gap                         |
| 104-106      | `Folding happens in JavaScript, whose % also truncates, so the literal path has to be corrected the same way or an expression changes meaning depending on whether its operands happen to be constants` | "has to be corrected" — narrates an old bug's fix                |
| 119-131      | `These were previously checked by matching the emitted loop header against a regex, which only works if you already know what the bug looks like`                                                       | "were previously checked" — narrates old approach                |
| 150-153      | `the others were being dropped — which left the tally at 0 here, and hung the GPU when the counter was the one lost`                                                                                    | "were being dropped" / "hung the GPU" — narrates old bug         |
| 210-212      | `continue_ inside a for loop emitted a malformed WGSL header until recently, and nothing checked what it computed`                                                                                      | "emitted...until recently" — narrates old bug                    |

### Mixed notes

| Lines         | Verbatim text (trimmed)                                                                                                                                                | What to keep vs. remove                     |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| 34-40 (JSDoc) | `The tolerance scales with the magnitude being checked. A flat one fails on correct backends for large results — one unit in the last place at 1024 is already 1.2e-4` | **OK** — describes current design rationale |

---

## `src/testing/shader-validity.ts` (test harness)

### Diary notes

| Lines         | Verbatim text (trimmed)                                                                                                                     | Why it's diary                                                                |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| 73-81 (JSDoc) | `Coverage had drifted badly: of 69 tests, 48 only ever compiled to GLSL and 20 only to WGSL, so a backend-specific defect stayed invisible` | "had drifted badly" / "stayed invisible" — narrates old coverage gap          |
| 174-179       | `Awaiting each pop inside the loop instead cost one round trip to the device per shader, which was around 100ms each`                       | "cost one round trip" / "was around 100ms" — narrates old performance problem |

### OK

Lines: 1-25 (JSDoc), 49-53, 55, 63-69, 95-97, 103-104, 110-117, 133-140, 157-160, 164, 169, 195-198, 200-203, 212-213, 222-223, 230-236, 243-246, 258-260, 270-272, 279-281, 321-328 — all describe current state, current constraints, or current design rationale.

---

## `src/testing/shader-eval.ts` (evaluation harness)

### Diary notes

| Lines        | Verbatim text (trimmed)                                                                                                                                                           | Why it's diary                                     |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| 1-18 (JSDoc) | `Nothing checked that they compute the right answer, and those are different questions: emitting a - b where a + b was meant compiles perfectly and passes every text assertion`  | "Nothing checked" / "was meant" — narrates old gap |
| 146-152      | `a completely broken backend returns 0, which is a value the evaluation tests legitimately expect in several places — so the failure has to be caught here or it reads as a pass` | "reads as a pass" — narrates old bug mode          |

### OK

Lines: 25-26, 32-45, 57-58, 62, 74-77, 92-93, 127, 136-142, 159-163, 194-200, 212, 217-228 — all describe current behavior or current design rationale.

---

## `src/testing/gpu.ts` (GPU device sharing)

### OK (all comments)

Lines: 1-13, 21, 32-37, 47, 58-64, 74-77, 92, 101-102, 106 — all describe current behavior. No diary notes.

---

## `src/testing/shader-eval.test.ts`

### Diary notes

| Lines | Verbatim text (trimmed)                                                                                | Why it's diary                        |
| ----- | ------------------------------------------------------------------------------------------------------ | ------------------------------------- |
| 56-57 | `A failed compile leaves the readback buffer at its zero initialiser, so the harness used to report 0` | "used to report 0" — narrates old bug |
| 82-83 | `Zero is the value a failed run used to produce`                                                       | "used to produce" — narrates old bug  |

### OK

Lines: 1-7 (JSDoc), 19-23, 33-35, 40, 46-47, 64-65 — describe current behavior or current rationale.

---

## `src/testing/shader-validity.test.ts`

### Diary notes

| Lines | Verbatim text (trimmed)                                                                                             | Why it's diary                                                                                                           |
| ----- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| 23-25 | `Keying failures by test name alone let a later one overwrite an earlier one`                                       | "let a later one overwrite" — narrates old bug                                                                           |
| 37-39 | `Returning success there is the failure mode this harness exists to prevent: a filtered run... would look verified` | "exists to prevent" — narrates why the harness was built                                                                 |
| 52-53 | `A known-invalid entry that starts compiling has to be reported, or the list outlives the problems it documents`    | "outlives the problems" — narrates old behavior                                                                          |
| 62-63 | `both backends refusing a program is consistent and some tests assert exactly that`                                 | borderline — could be read as current constraint, but "refusing...is consistent" reads as defense against an old problem |

### OK

Lines: 1-8 (JSDoc) — describes current harness purpose.

---

## `apps/shared/shader.ts`

All `N/A` — section headers only.

Lines: 6, 16, 23, 54, 121, 196

---

## `apps/infinite-grid/src/main.ts`

All `N/A` — section headers only.

Lines: 10, 14, 37, 78, 88, 97, 149, 155

---

## `apps/infinite-grid-webgpu/src/main.ts`

All `N/A` — section headers only.

Lines: 10, 14, 37, 135, 187, 200

---

## `vitest.config.ts`

### Diary notes

| Lines | Verbatim text (trimmed)                                                                                                    | Why it's diary                               |
| ----- | -------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| 8-10  | `Several defects here were a signature promising one thing while the node was another, which no runtime assertion can see` | "were...defects" — narrates old bug category |

---

## `stryker.config.json`

OK — the `_comment` field documents how to run the tool, not bug history.

---

## Summary

| Category                              | Count     |
| ------------------------------------- | --------- |
| **DIARY** comments requiring rewrite  | **~55**   |
| **MIXED** comments requiring trim     | **~5**    |
| **OK** comments (state descriptions)  | **~200+** |
| **N/A** (section headers, directives) | **~80**   |

### Most common diary patterns

1. **"was reaching...as"** — describes old codegen output
2. **"used to be.../ was previously.../ was treated as"** — narrates old behavior
3. **"the tie was broken on... / the check existed to catch"** — narrates why something was built
4. **"dropped / let through / produced (a bad thing)"** — narrates old bug symptoms
5. **"the mutation run showed / the first draft of this test"** — narrates past measurement
6. **"backends disagreed / widened / truncated"** — narrates old cross-backend differences

### Recommended rewrite strategy

For each diary note, ask: _does a reader need this to understand the code as it is today?_

- **If yes**: rewrite as a present-tense description of the current state or constraint. Example: `"the widths were previously checked only for the two square types"` → `"Widths are checked for all matrix types."`
- **If no**: delete the comment entirely. The code itself should be self-documenting for things that are no longer surprising.
