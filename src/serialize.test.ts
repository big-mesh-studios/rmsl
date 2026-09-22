import { describe, expect, it } from "vitest";
import { deserialize, serialize, type SerializedNode } from "./serialize";
import { Fn, float, invocationIndex, storage, uniform, type Node, type ShaderType } from "./core";
import { compile } from "./wgsl";

function makeMovementKernel() {
  return Fn(() => {
    const velocityX = storage("Velocity.x", "float");
    const positionX = storage("Position.x", "float", { access: "read_write" });
    const dt = uniform("float");
    const scales = uniform("float");
    const i = invocationIndex();
    positionX.element(i).addAssign(velocityX.element(i).mul(dt).mul(scales));
    return positionX.element(i);
  });
}

describe("serialize/deserialize", () => {
  it("round-trips a real Fn's output through actual JSON and compiles identically", () => {
    const kernel = makeMovementKernel();
    const original = kernel();

    const json = JSON.stringify(serialize(original as any));
    const restored = deserialize(JSON.parse(json));

    const originalProgram = compile({ stage: "compute", workgroupSize: 64 }, original as Node<ShaderType>);
    const restoredProgram = compile({ stage: "compute", workgroupSize: 64 }, restored);

    expect(restoredProgram.code).toBe(originalProgram.code);
    expect(restoredProgram.resources).toEqual(originalProgram.resources);
  });

  it("Node.toJSON is picked up by JSON.stringify directly", () => {
    const kernel = makeMovementKernel();
    const original = kernel();

    const viaToJSON = JSON.parse(JSON.stringify(original));
    const viaSerialize = serialize(original as any);

    expect(viaToJSON).toEqual(viaSerialize);
  });

  it("regenerates storage()/uniformArray() element() closures and name/access/length", () => {
    const kernel = Fn(() => {
      const buf = storage("MyBuffer", "vec4", { access: "read_write" });
      const i = invocationIndex();
      return buf.element(i);
    });
    const original = kernel();
    const restored = deserialize(JSON.parse(JSON.stringify(serialize(original as any))));

    // walk to find the storage node itself (root is a `seq`)
    function findStorage(n: any): any {
      if (n.type === "storage") return n;
      for (const p of n.params ?? []) {
        const found = findStorage(p);
        if (found) return found;
      }
      return undefined;
    }
    const storageNode = findStorage(restored);
    expect(storageNode.name).toBe("MyBuffer");
    expect(storageNode.access).toBe("read_write");
    expect(typeof storageNode.element).toBe("function");
    const el = storageNode.element({ _t: "int", type: "int", value: 0 } as any);
    expect(el.type).toBe("storageElement");
  });

  it("array-of-roots: a deserialized graph does not collide ids with a freshly constructed uniform", () => {
    // Hand-crafted JSON mimicking a uniform serialized from an earlier
    // session, where the counter had also just started at 0.
    const savedUniformJson = {
      _t: "float",
      type: "uniform",
      value: { id: 0, slot: "_rmsl_u0", shaderType: "float" },
    };
    const restoredUniform = deserialize(savedUniformJson as any);

    // The very next live uniform() call in *this* process also starts
    // drawing from id 0 if nothing else has consumed the counter yet in this
    // test file — deserialize must not have left the counter untouched, or
    // this would collide on id 0 too.
    const freshUniform = uniform("float");

    expect((restoredUniform.value as any).id).not.toBe((freshUniform.value as any).id);
    // Slot/name identity is untouched by the renumbering.
    expect((restoredUniform as any).name).toBe("_rmsl_u0");
  });

  it("array-of-roots: two systems sharing a storage buffer by name still share it after one is deserialized", () => {
    const producer = Fn(() => {
      const buf = storage("Shared", "float", { access: "read_write" });
      const i = invocationIndex();
      buf.element(i).assign(float(1));
      return buf.element(i);
    });
    const consumer = Fn(() => {
      const buf = storage("Shared", "float", { access: "read_write" });
      const i = invocationIndex();
      buf.element(i).addAssign(float(1));
      return buf.element(i);
    });

    const producerRoot = deserialize(JSON.parse(JSON.stringify(serialize(producer))) as any);
    const consumerRoot = consumer();

    const program = compile(
      { stage: "compute", workgroupSize: 64 },
      [producerRoot, consumerRoot as Node<ShaderType>],
    );
    const storageResources = program.resources.filter((r) => r.kind === "storage");
    expect(storageResources).toHaveLength(1);
    expect(storageResources[0]!.name).toBe("Shared");
  });

  it("serialize/deserialize accept an array of roots directly", () => {
    const kernel = makeMovementKernel();
    const roots = [kernel(), kernel()];

    const json = serialize(roots);
    expect(json).toHaveLength(2);

    const restored = deserialize(json);
    expect(restored).toHaveLength(2);
  });

  it("serialize accepts the callable an Fn(...) definition returns directly", () => {
    const kernel = makeMovementKernel();

    const restored = deserialize(JSON.parse(JSON.stringify(serialize(kernel))) as SerializedNode);
    const program = compile({ stage: "compute", workgroupSize: 64 }, restored);

    expect(program.code).toContain("_RmslUniforms");
  });
});
