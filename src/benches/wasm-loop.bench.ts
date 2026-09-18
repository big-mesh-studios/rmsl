import { bench, describe } from "vitest";
import { compileWasm } from "../wasm";
import { compileJS } from "../js";
import { Fn, For, float, int, sqrt } from "../rmsl";

describe("loop: sum of sqrt(i) for i in [0, 64)", () => {
  const build = () =>
    Fn(() => {
      const sum = float(0).toVar();
      For(
        () => int(0).toVar(),
        (i) => i.lessThan(int(64)),
        (i) => {
          i.assign(i.add(int(1)));
        },
        (i) => {
          sum.assign(sum.add(sqrt(i.toFloat())));
        },
      );
      return sum;
    })();
  const wasmFn = compileWasm(build as any, { name: "main", params: [] });
  const jsFn = compileJS(build as any, { name: "main", params: [] });
  const ctx = {};

  bench("compileWasm", () => {
    wasmFn.invoke(ctx);
  });
  bench("compileJS", () => {
    jsFn.invoke(ctx);
  });
});
