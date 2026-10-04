import { describe, expect, it } from "vitest";
import { Fn, instancedArray, invocationIndex, uniform } from "./rmsl";
import { compile } from "./wgsl";

describe("@random-mesh/rmsl/wgsl", () => {
  /**
   * @canon spec-wgsl-compiles-a-compute-node-to-a-compute-entry-point
   */
  it("compiles a semantic compute program into a structured artifact", () => {
    const velocityX = instancedArray(4, "float").toReadOnly();
    const positionX = instancedArray(4, "float");
    const movement = Fn(() => {
      const dt = uniform("float");
      const i = invocationIndex();

      positionX.element(i).addAssign(velocityX.element(i).mul(dt));
    })();

    const program = compile(
      {
        stage: "compute",
        workgroupSize: 64,
      },
      movement,
    );

    expect(program.stage).toBe("compute");
    expect(program.entryPoint).toBe("main");
    expect(program.workgroupSize).toBe(64);
    expect(program.code).toContain("@compute @workgroup_size(64)");
    expect(program.code).toContain("@builtin(global_invocation_id)");

    expect(program.resources).toHaveLength(3);

    const dtResource = program.resources.find((r) => r.kind === "uniform");
    expect(dtResource).toMatchObject({ kind: "uniform", group: 0, binding: 0, offset: 0, size: 4 });

    const storageResources = program.resources.filter((r) => r.kind === "storage");
    expect(storageResources).toHaveLength(2);
    expect(storageResources.map((r) => r.access).sort()).toEqual(["read", "read_write"]);

    // Resource names are the storage nodes' generated slot names, not the
    // WGSL backend's internal `_rmsl_sN` binding names; bindings are assigned
    // in slot-name order.
    const slotOrder = [positionX.name, velocityX.name].sort();
    expect(storageResources).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: positionX.name,
          access: "read_write",
          group: 1,
          binding: slotOrder.indexOf(positionX.name),
        }),
        expect.objectContaining({
          name: velocityX.name,
          access: "read",
          group: 1,
          binding: slotOrder.indexOf(velocityX.name),
        }),
      ]),
    );
  });
});
