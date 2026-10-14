import { expect } from "vitest";
import { RmslError } from "../errors";

type ErrorClass = abstract new (...args: any[]) => RmslError;

interface RmslMatchers<R = unknown> {
  /**
   * The call throws, or the promise rejects with, an instance of `Class` with
   * `code`, whose message matches `message` where one is given.
   */
  toThrowRmsl(Class: ErrorClass, code: string, message?: RegExp | string): R;
}

declare module "vitest" {
  interface Assertion<T = any> extends RmslMatchers<T> {}
  interface AsymmetricMatchersContaining extends RmslMatchers {}
}

expect.extend({
  toThrowRmsl(received: unknown, Class: ErrorClass, code: string, message?: RegExp | string) {
    let error: unknown = received;
    let threw = true;
    if (typeof received === "function" && !this.promise) {
      threw = false;
      try {
        received();
      } catch (thrown) {
        threw = true;
        error = thrown;
      }
    }
    const describe = (e: unknown) =>
      e instanceof Error ? `${e.name} ${JSON.stringify((e as RmslError).code)}: ${e.message}` : String(e);
    if (!threw) return { pass: false, message: () => `expected the call to throw ${Class.name} "${code}", and it returned` };
    const messageMatches =
      message === undefined ||
      (error instanceof Error &&
        (typeof message === "string" ? error.message.includes(message) : message.test(error.message)));
    const pass = error instanceof Class && (error as RmslError).code === code && messageMatches;
    return {
      pass,
      message: () =>
        `expected ${Class.name} "${code}"${message === undefined ? "" : ` matching ${String(message)}`}, got ${describe(error)}`,
    };
  },
});
