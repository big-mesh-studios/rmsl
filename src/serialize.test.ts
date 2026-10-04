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
  uniformRaw,
  StorageInstancedBufferAttribute,
  time,
  uniform,
  uniformArray,
  type Node,
  type ShaderType,
} from "./core";
import { compile } from "./wgsl";
import { compileJSRoutine } from "./js";

const compute = (root: Node<ShaderType> | Node<ShaderType>[]) => compile({ stage: "compute", workgroupSize: 64 }, root);

/**
 * `code` with every name rmsl generates numbered in order of first appearance,
 * since a rebuilt graph gets new ones.
 */
function normalized(code: string): string {
  const names = new Map<string, string>();
  return code.replace(/_rmsl_[a-z]*\d+/g, (name) => {
    if (!names.has(name)) names.set(name, `name${names.size}`);
    return names.get(name)!;
  });
}

/** `graph` after a real JSON round-trip, rebuilt. */
const roundTrip = (graph: SerializedGraph) => deserialize(JSON.parse(JSON.stringify(graph)));

/** The names of the uniforms reachable from `root`. */
function uniformNames(root: any, found = new Set<string>()): Set<string> {
  if (root.type === "uniform") found.add(root.name);
  for (const p of root.params ?? []) uniformNames(p, found);
  return found;
}

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
    expect(normalized(compute(restored).code)).toBe(normalized(compute(original).code));
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
    expect(normalized(code)).toBe(normalized(compute(original).code));
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
    expect(normalized(compute(restored).code)).toBe(normalized(compute(original).code));
  });

  /**
   * @canon spec-a-generated-name-is-local-to-its-program
   */
  it("gives what rmsl named new names, so a restored graph and a fresh one stay apart", () => {
    const build = () =>
      Fn(() => {
        const u = uniform("float");
        const doubled = u.mul(2).toVar();
        doubled.addAssign(1);
        return doubled;
      })();
    // Restoring the graph it is compiled with gives the names a graph built in another process would clash on.
    const fresh = build() as any;
    const restored = deserialize(JSON.parse(JSON.stringify(serialize(fresh)))) as any;

    const program = Fn(() => fresh.add(restored.mul(100)));
    const run = compileJSRoutine(program as any, { name: "main", params: [] });
    const freshUniform = [...uniformNames(fresh)][0]!;
    const restoredUniform = [...uniformNames(restored)][0]!;
    expect(restoredUniform).not.toBe(freshUniform);
    expect(run.run({ uniforms: { [freshUniform]: 1, [restoredUniform]: 2 } })).toBe(3 + 500);
  });

  /**
   * @canon spec-a-raw-name-is-absolute
   */
  it("keeps a name the program chose", () => {
    const program = Fn(() => uniformRaw("brightness", "float").mul(2).toVar("scaled"));
    const graph = serialize(program());
    const restored = deserialize(JSON.parse(JSON.stringify(graph))) as any;
    expect([...uniformNames(restored)]).toEqual(["brightness"]);
    expect(JSON.stringify(graph)).toContain('"varName":"scaled"');
  });

  /**
   * @canon spec-a-name-is-local-unless-the-user-gave-it
   */
  it("keeps generated names apart and joins a raw name, in one program of two graphs", () => {
    const build = () => Fn(() => uniform("float").add(uniformRaw("gain", "float")).toVar())();
    const fresh = build() as any;
    const restored = deserialize(JSON.parse(JSON.stringify(serialize(fresh)))) as any;

    const program = Fn(() => fresh.add(restored.mul(100)));
    const run = compileJSRoutine(program as any, { name: "main", params: [] });
    const [freshUniform] = [...uniformNames(fresh)].filter((name) => name !== "gain");
    const [restoredUniform] = [...uniformNames(restored)].filter((name) => name !== "gain");
    expect(restoredUniform).not.toBe(freshUniform);
    // (1 + 10) + (2 + 10) * 100: each graph reads its own generated uniform, and both read one gain.
    expect(run.run({ uniforms: { [freshUniform!]: 1, [restoredUniform!]: 2, gain: 10 } })).toBe(1211);
  });

  it("reads the clock time() gives, alone or compiled with a fresh graph that reads it", () => {
    const build = () => Fn(() => time().mul(2))();
    const fresh = build() as any;
    const restored = deserialize(JSON.parse(JSON.stringify(serialize(fresh)))) as any;
    expect([...uniformNames(restored)]).toEqual([time().name]);

    const both = Fn(() => fresh.add(restored));
    const run = compileJSRoutine(both as any, { name: "main", params: [] });
    expect(run.run({ uniforms: { [time().name]: 3 } })).toBe(12);
    const code = compute([fresh, restored]).code;
    expect(code.match(/_rmsl_time: f32/g)).toHaveLength(1);
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

  it("throws on data serialize() could not have produced", () => {
    const literal = { _t: "float", type: "float", value: 1 };
    const graph = (patch: Partial<SerializedGraph>): SerializedGraph => ({
      nodes: [literal],
      buffers: [],
      roots: 0,
      ...patch,
    });
    expect(() => deserialize({} as SerializedGraph)).toThrow(
      "[RMSL] deserialize: the data needs a nodes and a buffers array, as serialize() gives",
    );
    expect(() =>
      deserialize(graph({ nodes: [literal, { _t: "float", type: "add", params: [0, 5] }], roots: 1 })),
    ).toThrow("[RMSL] deserialize: node 1 names a child at 5, but there are only 2");
    expect(() => deserialize(graph({ nodes: [{ _t: "float", type: "negate", params: [1] }, literal] }))).toThrow(
      "[RMSL] deserialize: node 0 names a child at 1, after it; a child has to come before the node using it",
    );
    expect(() => deserialize(graph({ roots: [0, 3] }))).toThrow(
      "[RMSL] deserialize: root names a node at 3, but there are only 1",
    );
    expect(() =>
      deserialize(
        graph({
          nodes: [{ _t: "float", type: "storage", value: { shaderType: "float", access: "read_write", buffer: 0 } }],
        }),
      ),
    ).toThrow("[RMSL] deserialize: node 0 names a buffer at 0, but there are only 0");
    expect(() =>
      deserialize(
        graph({
          buffers: [{ instanced: false, count: 1, itemSize: 1, arrayClass: "Float64Array" as any, array: null }],
        }),
      ),
    ).toThrow("[RMSL] deserialize: buffer 0 holds a Float64Array, not a Float32Array, Int32Array or Uint32Array");
  });

  it("takes the callable an Fn definition returns", () => {
    const kernel = movementKernel();
    const restored = roundTrip(serialize(kernel)) as Node<ShaderType>;
    expect(compute(restored).code).toContain("_RmslUniforms");
  });
});
