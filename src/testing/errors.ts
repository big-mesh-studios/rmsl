import { expect } from "vitest";
import type { RmslError, Target } from "../errors";

type ErrorClass = abstract new (...args: any[]) => RmslError;

/** What a test expects of an error: its class, its code, and where asked, its target and its message. */
export interface ExpectedError {
  target?: Target | "none";
  message?: RegExp;
}

/** The error `fn` throws, or the rejection of the promise it returns. Fails the test when nothing throws. */
export async function caught(fn: () => unknown): Promise<unknown> {
  try {
    await fn();
  } catch (error) {
    return error;
  }
  throw new Error("expected the call to throw, and it returned");
}

/** The error `fn` throws synchronously. Fails the test when nothing throws. */
export function caughtSync(fn: () => unknown): unknown {
  try {
    fn();
  } catch (error) {
    return error;
  }
  throw new Error("expected the call to throw, and it returned");
}

/** Asserts that `error` is an instance of `Class` with `code`, and the target and message `expected` names. */
export function expectRmslError(error: unknown, Class: ErrorClass, code: string, expected: ExpectedError = {}): void {
  expect(error).toBeInstanceOf(Class);
  const e = error as RmslError;
  expect(e.code).toBe(code);
  if (expected.target === "none") expect("target" in e).toBe(false);
  else if (expected.target !== undefined) expect(e.target).toBe(expected.target);
  if (expected.message) expect(e.message).toMatch(expected.message);
}

/** Asserts that `fn` throws synchronously an instance of `Class` with `code`, as {@link expectRmslError}. */
export function expectThrows(fn: () => unknown, Class: ErrorClass, code: string, expected: ExpectedError = {}): void {
  expectRmslError(caughtSync(fn), Class, code, expected);
}

/**
 * Asserts that each of `compiles`, one per target, refuses its program with the
 * same class and code, and names its own target.
 */
export function expectRefusedOnEveryTarget(
  compiles: Partial<Record<Target, () => unknown>>,
  Class: ErrorClass,
  code: string,
  expected: Omit<ExpectedError, "target"> = {},
): void {
  for (const [target, compile] of Object.entries(compiles) as [Target, () => unknown][]) {
    expectRmslError(caughtSync(compile), Class, code, { ...expected, target });
  }
}
