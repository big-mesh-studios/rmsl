import { describe, expect, it } from "vitest";
import { Fn, invocationIndex, storage, uniform } from "./rmsl";
import { compile } from "./wgsl";

describe("@random-mesh/rmsl/wgsl", () => {
  it("compiles a semantic compute program into a structured artifact", () => {
    const movement = Fn(() => {
      const velocityX = storage("Velocity.x", "float");
      const positionX = storage("Position.x", "float", {
        access: "read_write",
      });
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

    // Resource names are the RMSL slots passed to storage(), not the WGSL
    // backend's internal `_rmsl_sN` binding names — bindings are assigned in
    // slot-name order, so "Position.x" (< "Velocity.x") gets binding 0.
    expect(storageResources).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: "Position.x", access: "read_write", group: 1, binding: 0 }),
        expect.objectContaining({ name: "Velocity.x", access: "read", group: 1, binding: 1 }),
      ]),
    );
  });
});
