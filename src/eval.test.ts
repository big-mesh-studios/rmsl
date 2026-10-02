import { describe, it, expect, afterAll } from "vitest";
import { Fn, float, int, For, If, While, Switch, Break, Continue, type Node } from "./rmsl";
import {
  evaluateRecording,
  assertRecordedEvaluationsAgree,
  closeEvaluators,
  floatTolerance,
} from "./testing/shader-eval";

afterAll(async () => {
  await assertRecordedEvaluationsAgree();
  await closeEvaluators();
}, 120_000);

type Build = (...args: Node<"float">[]) => Node<"float">;

/**
 * Assert both backends compute `want`, and therefore agree with each other.
 *
 * The tolerance scales with the magnitude being checked. A flat one fails on
 * correct backends for large results — one unit in the last place at 1024 is
 * already 1.2e-4 — while being far looser than needed near zero.
 */
async function expectValue(build: Build, args: number[], want: number) {
  // The CPU target runs the same program in-process, needs no hardware, and
  // computes exact f64 — so it answers now, and it is the arbiter. The program
  // is recorded, and the `afterAll` above holds both shading languages to this
  // same value, which is what lets one assertion cover three backends without
  // the test waiting on a device.
  const js = evaluateRecording(build, args) as number;
  const tolerance = floatTolerance(want);
  expect(Math.abs(js - want), `CPU target computed ${js}, wanted ${want}`).toBeLessThan(tolerance);
}

describe("RMSL evaluation", () => {
  it("computes arithmetic", async () => {
    await expectValue((a, b) => a.add(b), [2, 3], 5);
    await expectValue((a, b) => a.sub(b), [7, 3], 4);
    await expectValue((a, b) => a.mul(b), [3, 4], 12);
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
    await expectValue((a, b) => b.step(a), [0.5, 2], 1); // x above edge -> 1
    await expectValue((a, b) => b.step(a), [2, 0.5], 0); // x below edge -> 0
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
    const sumTo = (n: Node<"float">) =>
      Fn(() => {
        const total = float(0).toVar();
        For(
          () => float(0).toVar(),
          (i) => i.lessThan(n),
          (i) => i.assign(i.add(1)),
          (i) => {
            total.assign(total.add(i));
          },
        );
        return total;
      })();

    await expectValue(sumTo, [5], 10); // 0+1+2+3+4
    await expectValue(sumTo, [10], 45);
    await expectValue(sumTo, [0], 0); // condition false on entry
  }, 60_000);

  // A loop whose update does two things: advance the counter, and tally
  // alongside it. Both run four times, so the tally ends at 4.
  it("runs every statement of a loop update", async () => {
    const tallyLoop = () =>
      Fn(() => {
        const tally = float(0).toVar();
        For(
          () => float(0).toVar(),
          (i) => i.lessThan(4),
          (i) => {
            tally.assign(tally.add(1));
            i.assign(i.add(1));
          },
          (i) => {
            tally.assign(tally.add(0));
          },
        );
        return tally;
      })();

    await expectValue(tallyLoop, [], 4);
  }, 60_000);

  it("takes the branch the condition selects", async () => {
    const branch = (x: Node<"float">) =>
      Fn(() => {
        const out = float(0).toVar();
        If(x.greaterThan(1), () => {
          out.assign(float(10));
        }).Else(() => {
          out.assign(float(20));
        });
        return out;
      })();

    await expectValue(branch, [2], 10);
    await expectValue(branch, [0], 20);
  }, 60_000);

  it("walks an if/else-if/else chain in order", async () => {
    const classify = (x: Node<"float">) =>
      Fn(() => {
        const out = float(0).toVar();
        If(x.lessThan(10), () => {
          out.assign(float(1));
        })
          .ElseIf(x.lessThan(20), () => {
            out.assign(float(2));
          })
          .Else(() => {
            out.assign(float(3));
          });
        return out;
      })();

    await expectValue(classify, [5], 1);
    await expectValue(classify, [15], 2);
    await expectValue(classify, [25], 3);
  }, 60_000);

  // An ElseIf's condition is built after its If is in the block, so a variable
  // it makes has to be declared where that condition is tested, not after the If.
  it("tests an else-if condition that makes a variable with the variable's value", async () => {
    const classify = (x: Node<"float">) =>
      Fn(() => {
        const out = float(0).toVar();
        If(x.lessThan(0), () => {
          out.assign(float(1));
        })
          .ElseIf(x.add(1).toVar().greaterThan(5), () => {
            out.assign(float(2));
          })
          .ElseIf(x.mul(2).toVar().greaterThan(3), () => {
            out.assign(float(3));
          })
          .Else(() => {
            out.assign(float(4));
          });
        return out;
      })();

    await expectValue(classify, [-1], 1);
    await expectValue(classify, [10], 2);
    await expectValue(classify, [2], 3);
    await expectValue(classify, [1], 4);
  }, 60_000);

  // A loop tests its condition every time round, so a variable the condition
  // makes has to be computed every time round too, not once before the loop.
  it("recomputes a variable a For condition makes on every iteration", async () => {
    const count = (x: Node<"float">) =>
      Fn(() => {
        const steps = float(0).toVar();
        For(
          () => float(0).toVar(),
          (n) => n.add(x).toVar().lessThan(10),
          (n) => {
            n.assign(n.add(1));
          },
          () => {
            steps.assign(steps.add(1));
          },
        );
        return steps;
      })();

    await expectValue(count, [3], 7);
    await expectValue(count, [12], 0);
  }, 60_000);

  it("still steps a For whose condition makes a variable when its body continues", async () => {
    const evens = (x: Node<"float">) =>
      Fn(() => {
        const counted = float(0).toVar();
        For(
          () => float(0).toVar(),
          (n) => n.add(x).toVar().lessThan(10),
          (n) => {
            n.assign(n.add(1));
          },
          (n) => {
            If(n.mod(2).equal(1), () => {
              Continue();
            });
            counted.assign(counted.add(1));
          },
        );
        return counted;
      })();

    await expectValue(evens, [3], 4);
  }, 60_000);

  it("keeps a variable an else-if condition makes in scope after the chain", async () => {
    const pick = (x: Node<"float">) =>
      Fn(() => {
        const out = float(0).toVar();
        let doubled!: Node<"float">;
        If(x.lessThan(0), () => {
          out.assign(float(1));
        }).ElseIf((doubled = x.mul(2).toVar()).greaterThan(3), () => {
          out.assign(float(2));
        });
        return out.add(doubled);
      })();

    // The chain stops at the If: the else-if's variable is never computed, and keeps its zero.
    await expectValue(pick, [-1], 1);
    await expectValue(pick, [5], 12);
    await expectValue(pick, [1], 2);
  }, 60_000);

  it("keeps a variable a For condition makes in scope in the update and after the loop", async () => {
    const run = (x: Node<"float">) =>
      Fn(() => {
        let reach!: Node<"float">;
        const total = float(0).toVar();
        For(
          () => float(0).toVar(),
          (n) => (reach = n.add(x).toVar()).lessThan(10),
          (n) => {
            n.assign(n.add(1));
            total.assign(total.add(reach));
          },
          () => {},
        );
        return total.mul(100).add(reach);
      })();

    // With x = 7: reach is 7, 8, 9 inside the loop, summed by the update, and 10 when the loop ends.
    await expectValue(run, [7], 2410);
  }, 60_000);

  it("recomputes a For condition that calls a function making a variable on every iteration", async () => {
    const below = Fn((n: Node<"float">, x: Node<"float">) => n.add(x).toVar().lessThan(10));
    const count = (x: Node<"float">) =>
      Fn(() => {
        const steps = float(0).toVar();
        For(
          () => float(0).toVar(),
          (n) => below(n, x),
          (n) => {
            n.assign(n.add(1));
          },
          () => {
            steps.assign(steps.add(1));
          },
        );
        return steps;
      })();

    await expectValue(count, [3], 7);
  }, 60_000);

  it("refuses a statement written between an If and its ElseIf", () => {
    const build = (x: Node<"float">) =>
      Fn(() => {
        const total = float(0).toVar();
        const chain = If(x.lessThan(0), () => {
          total.assign(float(1));
        });
        total.assign(total.add(1));
        chain.ElseIf(x.greaterThan(5), () => {
          total.assign(float(2));
        });
        return total;
      })();
    expect(() => build(float(1))).toThrow(/\[RMSL\] ElseIf has to follow its If directly/);
  });

  it("refuses an ElseIf called from inside another block", () => {
    const build = (x: Node<"float">) =>
      Fn(() => {
        const total = float(0).toVar();
        const chain = If(x.lessThan(0), () => {
          total.assign(float(1));
        });
        If(x.greaterThan(10), () => {
          chain.ElseIf(x.mul(2).toVar().greaterThan(3), () => {
            total.assign(float(2));
          });
        });
        return total;
      })();
    expect(() => build(float(1))).toThrow(/\[RMSL\] ElseIf has to follow its If directly/);
  });

  it("runs a statement a loop condition writes before every test", async () => {
    const forTests = (x: Node<"float">) =>
      Fn(() => {
        const tests = float(0).toVar();
        For(
          () => float(0).toVar(),
          (n) => {
            tests.assign(tests.add(1));
            return n.lessThan(x);
          },
          (n) => {
            n.assign(n.add(1));
          },
          () => {},
        );
        return tests;
      })();
    await expectValue(forTests, [3], 4);

    const whileTests = (x: Node<"float">) =>
      Fn(() => {
        const tests = float(0).toVar();
        const n = float(0).toVar();
        While(
          () => {
            tests.assign(tests.add(1));
            return n.lessThan(x);
          },
          () => {
            n.assign(n.add(1));
          },
        );
        return tests;
      })();
    await expectValue(whileTests, [3], 4);
  }, 60_000);

  it("recomputes a variable a While condition given as a function makes on every iteration", async () => {
    const count = (x: Node<"float">) =>
      Fn(() => {
        const n = float(0).toVar();
        While(
          () => n.add(x).toVar().lessThan(10),
          () => {
            n.assign(n.add(1));
          },
        );
        return n;
      })();

    await expectValue(count, [3], 7);
    await expectValue(count, [12], 0);

    const last = (x: Node<"float">) =>
      Fn(() => {
        const n = float(0).toVar();
        let reach!: Node<"float">;
        While(
          () => (reach = n.add(x).toVar()).lessThan(10),
          () => {
            n.assign(n.add(1));
          },
        );
        return reach;
      })();
    await expectValue(last, [3], 10);
  }, 60_000);

  // A function's variable is first read in the condition here, so it is made
  // where the condition is tested, and has to stay in scope wherever else it is read.
  it("keeps a variable a loop condition first reads in scope after the loop", async () => {
    const twice = Fn((x: Node<"float">) => x.mul(2).toVar());
    const whileRun = (x: Node<"float">) =>
      Fn(() => {
        const limit = twice(x);
        const n = float(0).toVar();
        While(n.lessThan(limit), () => {
          n.assign(n.add(1));
        });
        return n.add(limit);
      })();
    await expectValue(whileRun, [3], 12);

    const forRun = (x: Node<"float">) =>
      Fn(() => {
        const limit = twice(x);
        const total = float(0).toVar();
        For(
          () => float(0).toVar(),
          (n) => n.lessThan(limit),
          (n) => {
            n.assign(n.add(1));
            total.assign(total.add(limit));
          },
          () => {},
        );
        return total.add(limit);
      })();
    await expectValue(forRun, [3], 42);
  }, 60_000);

  it("runs a while loop until its condition fails", async () => {
    const countdown = (n: Node<"float">) =>
      Fn(() => {
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
    const classify = () =>
      Fn(() => {
        const out = float(0).toVar();
        Switch(int(1), (s) => {
          s.Case(0, () => {
            out.assign(float(10));
          });
          s.Case([1, 2], () => {
            out.assign(float(20));
          });
          s.Default(() => {
            out.assign(float(30));
          });
        });
        return out;
      })();

    await expectValue(classify, [], 20);
  }, 60_000);

  // The lowercase aliases are the same nodes, so they must compute the same
  // results — an alias that silently did nothing would fail here.
  it("computes the same results through the lowercase aliases", async () => {
    const branch = (x: Node<"float">) =>
      Fn(() => {
        const out = float(0).toVar();
        If(x.greaterThan(1), () => {
          out.assign(float(10));
        })
          .ElseIf(x.greaterThan(0), () => {
            out.assign(float(20));
          })
          .Else(() => {
            out.assign(float(30));
          });
        return out;
      })();
    await expectValue(branch, [2], 10);
    await expectValue(branch, [0.5], 20);
    await expectValue(branch, [-1], 30);

    const sum = (n: Node<"float">) =>
      Fn(() => {
        const total = float(0).toVar();
        For(
          () => float(0).toVar(),
          (i) => i.lessThan(n),
          (i) => i.assign(i.add(1)),
          (i) => {
            total.assign(total.add(i));
          },
        );
        return total;
      })();
    await expectValue(sum, [5], 10);

    const countdown = (n: Node<"float">) =>
      Fn(() => {
        const left = n.toVar();
        const steps = float(0).toVar();
        While(left.greaterThan(0), () => {
          left.assign(left.sub(1));
          steps.assign(steps.add(1));
        });
        return steps;
      })();
    await expectValue(countdown, [4], 4);

    const classify = () =>
      Fn(() => {
        const out = float(0).toVar();
        Switch(int(2), (s) => {
          s.Case(0, () => {
            out.assign(float(10));
          });
          s.Case([1, 2], () => {
            out.assign(float(20));
          });
          s.Default(() => {
            out.assign(float(30));
          });
        });
        return out;
      })();
    await expectValue(classify, [], 20);
  }, 60_000);

  // break_ and continue_ change which iterations contribute, so the sum says
  // whether they landed.
  it("honours break_ and continue_", async () => {
    const sumUntilBreak = (limit: Node<"float">) =>
      Fn(() => {
        const total = float(0).toVar();
        For(
          () => float(0).toVar(),
          (i) => i.lessThan(100),
          (i) => i.assign(i.add(1)),
          (i) => {
            If(i.greaterThanEqual(limit), () => {
              Break();
            });
            total.assign(total.add(i));
          },
        );
        return total;
      })();

    await expectValue(sumUntilBreak, [5], 10); // stops before i === 5
    await expectValue(sumUntilBreak, [1], 0); // breaks immediately

    const sumSkippingFirst = (n: Node<"float">) =>
      Fn(() => {
        const total = float(0).toVar();
        For(
          () => float(0).toVar(),
          (i) => i.lessThan(n),
          (i) => i.assign(i.add(1)),
          (i) => {
            If(i.lessThan(2), () => {
              Continue();
            });
            total.assign(total.add(i));
          },
        );
        return total;
      })();

    await expectValue(sumSkippingFirst, [5], 9); // 2+3+4, skipping 0,1
  }, 60_000);
});
