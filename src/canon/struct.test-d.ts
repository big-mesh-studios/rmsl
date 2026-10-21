import { describe, expectTypeOf, it } from "vitest";
import { Fn, float, int, struct, vec3, type Node, type Var } from "../rmsl";

describe("the types of a struct", () => {
  /**
   * @canon spec-the-types-follow-the-members-of-a-struct
   */
  it("types each member by its type, a member of a variable as a variable, and a struct member as any", () => {
    const Inner = struct({ a: "vec3" }, "InnerTypes");
    const Box = struct({ min: "vec3", left: "int", inner: "InnerTypes" }, "BoxTypes");
    const box = Box(vec3(0), int(1), Inner(vec3(2)));
    expectTypeOf(box.get("min")).toEqualTypeOf<Node<"vec3">>();
    expectTypeOf(box.get("left")).toEqualTypeOf<Node<"int">>();
    expectTypeOf(box.get("inner")).toBeAny();
    // @ts-expect-error the struct has no member "mid"
    box.get("mid");
    // @ts-expect-error a float is no value of the int member "left"
    Box(vec3(0), float(1));
    Fn(() => {
      const held = Box().toVar();
      expectTypeOf(held.get("min")).toEqualTypeOf<Var<"vec3">>();
      held.get("min").assign(vec3(1));
      held.assign(Box());
    })();
  });
});
