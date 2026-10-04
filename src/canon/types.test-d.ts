import { describe, it } from "vitest";
import { vec2, vec3 } from "../rmsl";

describe("refusals the types make", () => {
  /**
   * @canon spec-operands-of-different-widths-are-refused
   */
  it("refuses an operation on vectors of different widths", () => {
    // @ts-expect-error a vec2 and a vec3 have no dot product
    vec2(1, 2).dot(vec3(1, 2, 3));
    // @ts-expect-error nor a minimum
    vec2(1, 2).min(vec3(1, 2, 3));
  });
});
