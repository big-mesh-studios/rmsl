import { describe, expect, it } from "vitest";
import { deserialize, serialize, type SerializedGraph } from "./serialize";
import {
  Fn,
  For,
  If,
  float,
  instancedArray,
  invocationIndex,
  storage,
  StorageBufferAttribute,
  StorageInstancedBufferAttribute,
  uniform,
  uniformArray,
  type Node,
  type ShaderType,
} from "./core";
import { compile } from "./wgsl";

const compute = (root: Node<ShaderType> | Node<ShaderType>[]) => compile({ stage: "compute", workgroupSize: 64 }, root);

/** `graph` after a real JSON round-trip, rebuilt. */
const roundTrip = (graph: SerializedGraph) => deserialize(JSON.parse(JSON.stringify(graph)));

/** The storage nodes reachable from `root`. */
function storageNodes(root: any, found = new Set<any>()): Set<any> {
  if (root.type === "storage") found.add(root);
  for (const p of root.params ?? []) storageNodes(p, found);
  return found;
}

function movementKernel() {
  const velocity = new StorageBufferAttribute(16, 1);
  const position = new StorageBufferAttribute(16, 1);
  return Fn(() => {
    const velocityX = storage(velocity, "float").toReadOnly();
    const positionX = storage(position, "float");
    const dt = uniform("float");
    const i = invocationIndex();
    positionX.element(i).addAssign(velocityX.element(i).mul(dt));
    return positionX.element(i);
  });
}

describe("serialize/deserialize", () => {
  it("round-trips a compute kernel through JSON and compiles it to the same WGSL", () => {
    const original = movementKernel()();
    const restored = roundTrip(serialize(original)) as Node<ShaderType>;
    expect(compute(restored).code).toBe(compute(original).code);
  });

  it("keeps a node read in several places one node", () => {
    const program = Fn(() => {
      const u = uniform("float");
      const helper = Fn(() => u.mul(2).toVar())();
      const out = float(0).toVar();
      If(u.greaterThan(0), () => out.assign(helper));
      For(
        () => float(0).toVar(),
        (n) => n.lessThan(u),
        (n) => n.assign(n.add(1)),
        () => out.addAssign(u),
      );
      return u.add(u).add(out);
    });
    const original = program();
    const restored = roundTrip(serialize(original)) as Node<ShaderType>;
    const code = compute(restored).code;
    expect(code).toBe(compute(original).code);
    expect(code.match(/_rmsl_u\d+: f32/g)).toHaveLength(1);
  });

  it("rebuilds each storage buffer once, with its layout, contents and access", () => {
    const shared = new StorageBufferAttribute(new Float32Array([1, 2, 3, 4, 5, 6]), 2);
    const program = Fn(() => {
      const a = storage(shared, "vec2");
      const b = storage(shared, "vec2").toReadOnly();
      const counts = instancedArray(new Int32Array([7, 8]), "int");
      const i = invocationIndex();
      a.element(i).assign(b.element(i).add(counts.element(i).toFloat()));
      return a.element(i).x;
    });
    const restored = roundTrip(serialize(program())) as Node<ShaderType>;

    const [a, b, counts] = [...storageNodes(restored)];
    expect(a.attribute).toBe(b.attribute);
    expect(a.attribute).toBeInstanceOf(StorageBufferAttribute);
    expect(a.attribute.itemSize).toBe(2);
    expect(Array.from(a.attribute.array)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(a.access).toBe("read_write");
    expect(b.access).toBe("read");
    expect(counts.attribute).toBeInstanceOf(StorageInstancedBufferAttribute);
    expect(counts.attribute.array).toBeInstanceOf(Int32Array);
    expect(Array.from(counts.attribute.array)).toEqual([7, 8]);
    expect(counts.element(0).type).toBe("storageElement");
  });

  it("keeps a buffer without contents empty", () => {
    const program = Fn(() => storage(new StorageBufferAttribute(32, 4), "vec4").element(invocationIndex()).x);
    const [buffer] = [...storageNodes(roundTrip(serialize(program())))];
    expect(buffer.attribute.array).toBeNull();
    expect(buffer.attribute.count).toBe(32);
  });

  it("rebuilds a uniform array with its length and element()", () => {
    const program = Fn(() => uniformArray("vec4", 3).element(1).x);
    const original = program();
    const restored = roundTrip(serialize(original)) as Node<ShaderType>;
    expect(compute(restored).code).toBe(compute(original).code);
  });

  it("draws a new id for a uniform, so it can't collide with one built afterwards", () => {
    const saved: SerializedGraph = {
      nodes: [{ _t: "float", type: "uniform", value: { id: 0, slot: "_rmsl_u0", shaderType: "float" } }],
      buffers: [],
      roots: 0,
    };
    const restored = deserialize(saved) as any;
    const fresh = uniform("float") as any;
    expect(restored.value.id).not.toBe(fresh.value.id);
    expect(restored.name).toBe("_rmsl_u0");
  });

  it("takes an array of roots, and keeps what they share shared", () => {
    const buffer = new StorageBufferAttribute(8, 1);
    const producer = Fn(() => {
      const out = storage(buffer, "float");
      out.element(invocationIndex()).assign(float(1));
      return out.element(invocationIndex());
    });
    const consumer = Fn(() => {
      const out = storage(buffer, "float");
      out.element(invocationIndex()).addAssign(float(1));
      return out.element(invocationIndex());
    });
    const restored = roundTrip(serialize([producer(), consumer()])) as Node<ShaderType>[];
    expect(restored).toHaveLength(2);

    const [p] = [...storageNodes(restored[0])];
    const [c] = [...storageNodes(restored[1])];
    expect(p.attribute).toBe(c.attribute);
    expect(compute(restored).resources.filter((r) => r.kind === "storage")).toHaveLength(1);
  });

  it("takes the callable an Fn definition returns", () => {
    const kernel = movementKernel();
    const restored = roundTrip(serialize(kernel)) as Node<ShaderType>;
    expect(compute(restored).code).toContain("_RmslUniforms");
  });
});
