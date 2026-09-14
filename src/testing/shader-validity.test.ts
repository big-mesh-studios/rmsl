/**
 * Tests for the validation harness itself.
 *
 * This harness is the reason the rest of the suite's substring assertions can
 * be trusted, so a fault here does not produce a wrong answer — it produces a
 * green run that proves nothing. The reporting is kept separate from the GPU
 * work so it can be checked without one.
 */

import { describe, it, expect } from "vitest";
import {
  validationReport,
  expectCompileRejection,
  recordShaderSource,
  recordedShaders,
  type Recorded,
} from "./shader-validity";

function shader(test: string, lang: "glsl" | "wgsl", pair: number): Recorded {
  return { test, lang, stage: "fragment", pair, src: "<source>" };
}

describe("validation reporting", () => {
  it("passes when every shader compiled", () => {
    const recorded = [shader("a test", "glsl", 0), shader("a test", "wgsl", 0)];
    expect(validationReport(recorded, [null], [null], {})).toBeNull();
  });

  // One test routinely compiles many shaders — the breadth tests generate
  // dozens each. Failures are keyed by test name and shader index to avoid
  // overwriting.
  it("reports every failing shader from one test, not just the last", () => {
    const recorded = [shader("compiles every builtin", "wgsl", 0), shader("compiles every builtin", "wgsl", 1)];
    const report = validationReport(recorded, [], ["no overload for 'refract'", "no overload for 'lessThan'"], {});
    expect(report).not.toBeNull();
    expect(report).toContain("refract");
    expect(report).toContain("lessThan");
  });

  // Nothing recorded means nothing was checked.
  it("refuses to pass when no shader was recorded at all", () => {
    const report = validationReport([], [], [], {});
    expect(report).not.toBeNull();
    expect(report).toMatch(/no shaders/i);
  });

  it("stays quiet about a shader listed as known invalid", () => {
    const recorded = [shader("a known bad test", "glsl", 0)];
    const known = { "glsl:a known bad test": "documented defect" };
    expect(validationReport(recorded, ["some error"], [], known)).toBeNull();
  });

  // A known-invalid entry that now compiles is reported.
  it("reports a known-invalid entry that now compiles", () => {
    const recorded = [shader("a fixed test", "glsl", 0)];
    const known = { "glsl:a fixed test": "documented defect" };
    const report = validationReport(recorded, [null], [], known);
    expect(report).not.toBeNull();
    expect(report).toContain("a fixed test");
  });

  // A compiler that throws is recorded rather than swallowed.
  it("ignores a program both backends refuse", () => {
    const recorded: Recorded[] = [
      { ...shader("both refuse", "glsl", 0), src: null, compileError: "nope" },
      { ...shader("both refuse", "wgsl", 0), src: null, compileError: "nope" },
    ];
    expect(validationReport(recorded, [], [], {})).toBeNull();
  });

  it("reports a program only one backend refuses", () => {
    const recorded: Recorded[] = [
      { ...shader("one refuses", "glsl", 0), src: null, compileError: "gone wrong" },
      shader("one refuses", "wgsl", 0),
    ];
    const report = validationReport(recorded, [], [null], {});
    expect(report).not.toBeNull();
    expect(report).toContain("one refuses");
  });
});

// The escape hatch from the rule directly above: one backend refusing what the
// other accepts is a defect, unless the test asked for the refusal.
describe("expected rejections", () => {
  it("hands back the message the compiler refused with", () => {
    expect(
      expectCompileRejection(() => {
        throw new Error('unknown precision "high"');
      }),
    ).toMatch(/precision/);
  });

  it("keeps nothing the refused compile recorded on its way to throwing", () => {
    const before = recordedShaders().length;
    expectCompileRejection(() => {
      recordShaderSource("wgsl", "fragment", "fn main() {}");
      throw new Error("rejected");
    });
    expect(recordedShaders()).toHaveLength(before);
  });

  it("fails when the compile it expected to be refused succeeds", () => {
    expect(() => expectCompileRejection(() => "fn main() {}")).toThrow(/returned source/);
  });
});
