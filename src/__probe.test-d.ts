import { Fn, float, vec4, builtinPosition, type Node } from "./rmsl";
import { expectTypeOf } from "vitest";

// a body with no return
let none = Fn(() => {
  float(1).toVar();
})();
expectTypeOf(none).toEqualTypeOf<void>();

// a body that writes the position and returns nothing
let written = Fn(() => {
  builtinPosition().assign(vec4(1, 2, 3, 4));
})();
expectTypeOf(written).toEqualTypeOf<void>();
