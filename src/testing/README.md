# testing

Internal test infrastructure shared by the rest of the test suite — not
part of the published package.

## gpu.ts

The browser and graphics device the test harnesses share: a Chromium with a
software GL driver, for GLSL, and a WebGPU device, for WGSL. Each harness
used to stand up its own; that was slow, and it was two places to fix a
release-cleanup bug instead of one.

Everything here is created once and reused. What's memoised is the
*promise*, not the result: it's assigned before the first `await`, so two
callers arriving together share one launch instead of each starting their
own and one being dropped on the floor still running.

`webgpuBrowser`/`webgpuPage` are a separate browser from `gpuBrowser`/
`gpuPage`, not just a separate device on the same one. Two things keep them
apart: Playwright's default browser is the headless *shell*, which exposes
`navigator.gpu` but has no adapter behind it (the full Chromium build has to
be asked for by channel), and the SwiftShader arguments the GLSL browser
needs take the WebGPU adapter away — so the two configurations can't share
one browser instance.

## shader-eval.ts

Runs a compiled expression and reports the number it produces, on GLSL,
WGSL, JS and WASM.

An expression is compiled to a function, called on each backend, and the
result compared against the same arithmetic in plain JS. Running all of
them also makes the backends checkable against each other: one RMSL program
must produce one number, and a divergence is a bug in whichever side
disagrees with JS.

These programs aren't only checked once. `evaluateRecording` records each
one, and `assertRecordedEvaluationsAgree` (typically called from an
`afterAll` hook) replays them on every backend and requires the answers to
match — so a case written once, anywhere in the suite, covers all backends.

`floatTolerance` exists because a 32-bit float carries 24 bits of mantissa,
so neighbouring representable values are `|x| * 2^-23` apart — larger than
any flat tolerance is worth using at scale, and two independent
implementations (e.g. `pow` evaluated as `exp2(y * log2(x))`) aren't obliged
to agree bit for bit. The tolerance scales with magnitude instead, with a
floor for values near zero where the relative gap collapses.

## shader-validity.ts

Checks that every shader the test suite generates is actually valid, in
both GPU backends — not just that it contains the substring an assertion
went looking for.

The rest of the suite's assertions are substring matches
(`expect(glsl).toContain(...)`), which can't tell a correct shader from a
broken one that happens to contain the expected text. A parser alone isn't
enough to catch this either — it accepts a program with a type error, only
a real compiler rejects it:

```
             1e-7.0   refract(I,N)   lessThan(f,f)   mat2x3 = 0.0
  parser     reject      accept         accept          accept
  real       reject      reject         reject          reject
```

So the generated source is handed to the real compilers it has to run on
instead: GLSL to Chromium's WebGL2 compiler, WGSL to Dawn.

**Wiring:** a test file aliases the compilers through `recordGLSL`/
`recordWGSL`, so no individual test changes, then awaits
`assertRecordedShadersValid()` in `afterAll`. A test that asserts a compiler
*refuses* something goes through `expectCompileRejection` instead, which
records nothing.

## Why these two harnesses have their own tests

`shader-eval.test.ts` and `shader-validity.test.ts` test the harnesses
above, not RMSL itself. Both harnesses are what the rest of the suite's
assertions rest on, so a fault in either is worse than a fault in the
compiler: it doesn't produce a wrong answer, it produces a green run that
proves nothing.
