/**
 * Checks what the generated shaders *compute*, not just that they compile.
 *
 * Every other test asserts on emitted text, and text cannot tell `a + b` from
 * `a - b` — both are valid, and both contain whatever the assertion looks for.
 * These evaluate the shader on both backends and compare the computed result,
 * so a disagreement identifies which backend is wrong.
 *
 * Operands come in as function parameters rather than literals, since constant
 * folding would otherwise compute the answer before codegen ever runs.
 */

import { describe, it, expect, afterAll } from "vitest";
import { Fn, float, int, For, If, While, Switch, break_, continue_, type Node } from "./rmsl";
import {
  evaluateBoth, closeEvaluators, floatTolerance, EVALUATION_SKIPPED,
} from "./testing/shader-eval";

afterAll(async () => {
  await closeEvaluators();
});

type Build = (...args: Node<"float">[]) => Node<"float">;

/**
 * Assert both backends compute `want`, and therefore agree with each other.
 *
 * The tolerance scales with the magnitude being checked. A flat one fails on
 * correct backends for large results — one unit in the last place at 1024 is
 * already 1.2e-4 — while being far looser than needed near zero.
 */
async function expectValue(build: Build, args: number[], want: number) {
  const { glsl, wgsl } = await evaluateBoth(build, args);
  const tolerance = floatTolerance(want);
  expect(Math.abs(glsl - want), `GLSL computed ${glsl}, wanted ${want}`)
    .toBeLessThan(tolerance);
  expect(Math.abs(wgsl - want), `WGSL computed ${wgsl}, wanted ${want}`)
    .toBeLessThan(tolerance);
  expect(Math.abs(glsl - wgsl), `backends disagree: ${glsl} vs ${wgsl}`)
    .toBeLessThan(tolerance);
}

describe.skipIf(EVALUATION_SKIPPED)("RMSL evaluation", () => {
  it("computes arithmetic", async () => {
    await expectValue((a, b) => a.add(b), [2, 3], 5);
    await expectValue((a, b) => a.sub(b), [7, 3], 4);
    await expectValue((a, b) => a.mult(b), [3, 4], 12);
    await expectValue((a, b) => a.div(b), [8, 2], 4);
    await expectValue((a) => a.negate(), [3], -3);
  }, 60_000);

  // min and max must agree across both backends.
  it("computes min and max the right way round", async () => {
    await expectValue((a, b) => a.min(b), [3, 9], 3);
    await expectValue((a, b) => a.max(b), [3, 9], 9);
  }, 60_000);

  it("computes math builtins", async () => {
    await expectValue((a) => a.sqrt(), [9], 3);
    await expectValue((a) => a.abs(), [-4], 4);
    await expectValue((a) => a.floor(), [2.7], 2);
    await expectValue((a) => a.ceil(), [2.1], 3);
    await expectValue((a) => a.sin(), [0.5], Math.sin(0.5));
    await expectValue((a) => a.cos(), [0.5], Math.cos(0.5));
    await expectValue((a, b) => a.pow(b), [2, 10], 1024);
  }, 60_000);

  // These are the ops the TSL parity pass added; a compile-only check cannot
  // tell sinh from cosh, so their values are pinned here too.
  it("computes the parity-pass math builtins", async () => {
    await expectValue((a) => a.round(), [2.4], 2);
    await expectValue((a) => a.round(), [2.6], 3);
    await expectValue((a) => a.trunc(), [-2.7], -2);
    await expectValue((a) => a.sinh(), [0.5], Math.sinh(0.5));
    await expectValue((a) => a.cosh(), [0.5], Math.cosh(0.5));
    await expectValue((a) => a.tanh(), [0.5], Math.tanh(0.5));
    await expectValue((a) => a.saturate(), [2.5], 1);
    await expectValue((a) => a.oneMinus(), [0.25], 0.75);
    await expectValue((a) => a.reciprocal(), [4], 0.25);
    await expectValue((a) => a.lengthSq(), [3], 9);
    await expectValue((a) => a.cbrt(), [27], 3);
  }, 60_000);

  // Casts truncate toward zero on both backends: int(2.7) is 2, int(-2.7) is -2.
  it("casts float to int and back", async () => {
    await expectValue((a) => a.toInt().toFloat(), [2.7], 2);
    await expectValue((a) => a.toInt().toFloat(), [-2.7], -2);
    await expectValue((a) => a.toUint().toFloat(), [2.7], 2);
    await expectValue((a) => a.toInt().toBool().toFloat(), [1.5], 1);
    await expectValue((a) => a.toInt().toBool().toFloat(), [0], 0);
  }, 60_000);

  // Argument order is the thing worth pinning: GLSL takes the value last in
  // step(edge, x), so an emitter that passes them the other way still compiles.
  it("computes step, smoothstep and mix with operands in the right order", async () => {
    await expectValue((a, b) => b.step(a), [0.5, 2], 1);   // x above edge -> 1
    await expectValue((a, b) => b.step(a), [2, 0.5], 0);   // x below edge -> 0
    await expectValue((a, b) => a.mix(b, 0.25), [0, 4], 1);
    await expectValue((a, b) => a.mix(b, 0.75), [0, 4], 3);
    await expectValue((a) => a.smoothstep(0, 1), [0.5], 0.5);
  }, 60_000);

  it("computes clamp against both bounds", async () => {
    await expectValue((a) => a.clamp(0, 1), [2.5], 1);
    await expectValue((a) => a.clamp(0, 1), [-2.5], 0);
    await expectValue((a) => a.clamp(0, 1), [0.25], 0.25);
  }, 60_000);

  // Floored, following GLSL's mod() — the function this operation is named
  // for — so the result takes the sign of the divisor.
  it("computes float modulus the same way on both backends", async () => {
    await expectValue((a, b) => a.mod(b), [7.5, 2], 1.5);
    await expectValue((a, b) => a.mod(b), [-7.5, 2], 0.5);
    await expectValue((a, b) => a.mod(b), [7.5, -2], -0.5);
    await expectValue((a, b) => a.mod(b), [-1, 2], 1);
  }, 60_000);

  // Folding happens in JavaScript, whose % also truncates, so the literal path
  // is corrected the same way.
  it("folds a modulus to what the shader would have computed", async () => {
    await expectValue(() => float(-7.5).mod(float(2)), [], 0.5);
    await expectValue((a, b) => a.mod(b), [-7.5, 2], 0.5);
  }, 60_000);

  it("folds constants to the same value it would compute at runtime", async () => {
    // The literal path folds in JS; the parameter path runs on the GPU. They
    // must agree, or folding is lying about what the shader would have done.
    await expectValue(() => float(7).div(float(2)), [], 3.5);
    await expectValue((a, b) => a.div(b), [7, 2], 3.5);
  }, 60_000);

  // === Control flow ===
  //
  // Control flow is checked by running the shader and verifying the computed
  // result. A sum pins the whole loop at once: it comes out right only if the
  // loop starts, increments and stops correctly.
  //
  // `toVar()` and `If` need a block scope, which the standalone function
  // compilers do not open, so each body is wrapped in `Fn(() => ...)()`.
  //
  // A loop that fails to advance does not fail — it hangs, and the timeout is
  // what catches it. That is inherent: there is no way to test that a loop
  // terminates without risking one that does not.

  it("runs a for loop the right number of times", async () => {
    const sumTo = (n: Node<"float">) => Fn(() => {
      const total = float(0).toVar();
      For(
        () => float(0).toVar(),
        (i) => i.lessThan(n),
        (i) => i.assign(i.add(1)),
        (i) => { total.assign(total.add(i)); },
      );
      return total;
    })();

    await expectValue(sumTo, [5], 10);   // 0+1+2+3+4
    await expectValue(sumTo, [10], 45);
    await expectValue(sumTo, [0], 0);    // condition false on entry
  }, 60_000);

  // A loop whose update does two things: advance the counter, and tally
  // alongside it. Both run four times, so the tally ends at 4.
  it("runs every statement of a loop update", async () => {
    const tallyLoop = () => Fn(() => {
      const tally = float(0).toVar();
      For(
        () => float(0).toVar(),
        (i) => i.lessThan(4),
        (i) => { tally.assign(tally.add(1)); i.assign(i.add(1)); },
        (i) => { tally.assign(tally.add(0)); },
      );
      return tally;
    })();

    await expectValue(tallyLoop, [], 4);
  }, 60_000);

  it("takes the branch the condition selects", async () => {
    const branch = (x: Node<"float">) => Fn(() => {
      const out = float(0).toVar();
      If(x.greaterThan(1), () => { out.assign(float(10)); })
        .Else(() => { out.assign(float(20)); });
      return out;
    })();

    await expectValue(branch, [2], 10);
    await expectValue(branch, [0], 20);
  }, 60_000);

  it("walks an if/else-if/else chain in order", async () => {
    const classify = (x: Node<"float">) => Fn(() => {
      const out = float(0).toVar();
      If(x.lessThan(10), () => { out.assign(float(1)); })
        .ElseIf(x.lessThan(20), () => { out.assign(float(2)); })
        .Else(() => { out.assign(float(3)); });
      return out;
    })();

    await expectValue(classify, [5], 1);
    await expectValue(classify, [15], 2);
    await expectValue(classify, [25], 3);
  }, 60_000);

  it("runs a while loop until its condition fails", async () => {
    const countdown = (n: Node<"float">) => Fn(() => {
      const left = n.toVar();
      const steps = float(0).toVar();
      While(left.greaterThan(0), () => {
        left.assign(left.sub(1));
        steps.assign(steps.add(1));
      });
      return steps;
    })();

    await expectValue(countdown, [4], 4);
    await expectValue(countdown, [0], 0);
  }, 60_000);

  it("takes the branch Switch selects", async () => {
    const classify = () => Fn(() => {
      const out = float(0).toVar();
      Switch(int(1), (s) => {
        s.Case(0, () => { out.assign(float(10)); });
        s.Case([1, 2], () => { out.assign(float(20)); });
        s.Default(() => { out.assign(float(30)); });
      });
      return out;
    })();

    await expectValue(classify, [], 20);
  }, 60_000);

  // break_ and continue_ change which iterations contribute, so the sum says
  // whether they landed.
  it("honours break_ and continue_", async () => {
    const sumUntilBreak = (limit: Node<"float">) => Fn(() => {
      const total = float(0).toVar();
      For(
        () => float(0).toVar(),
        (i) => i.lessThan(100),
        (i) => i.assign(i.add(1)),
        (i) => {
          If(i.greaterThanEqual(limit), () => { break_(); });
          total.assign(total.add(i));
        },
      );
      return total;
    })();

    await expectValue(sumUntilBreak, [5], 10);  // stops before i === 5
    await expectValue(sumUntilBreak, [1], 0);   // breaks immediately

    const sumSkippingFirst = (n: Node<"float">) => Fn(() => {
      const total = float(0).toVar();
      For(
        () => float(0).toVar(),
        (i) => i.lessThan(n),
        (i) => i.assign(i.add(1)),
        (i) => {
          If(i.lessThan(2), () => { continue_(); });
          total.assign(total.add(i));
        },
      );
      return total;
    })();

    await expectValue(sumSkippingFirst, [5], 9);  // 2+3+4, skipping 0,1
  }, 60_000);
});
