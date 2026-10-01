import { describe, it, expect } from "vitest";
import { Fn, invocationIndex, instancedArray, attributeArray, storage, StorageBufferAttribute } from "./rmsl";
import { compile } from "./wgsl";

describe("storage buffers", () => {
  it("sizes an instancedArray from a count, with the element type's components and array class", () => {
    const positions = instancedArray(8, "vec3");
    expect(positions.attribute.count).toBe(8);
    expect(positions.attribute.itemSize).toBe(3);
    expect(positions.attribute.arrayClass).toBe(Float32Array);
    expect(positions.attribute.array).toBeNull();
    expect(instancedArray(4, "uint").attribute.arrayClass).toBe(Uint32Array);
    expect(attributeArray(4, "int").attribute.arrayClass).toBe(Int32Array);
  });

  it("takes an instancedArray's count and initial contents from a typed array", () => {
    const data = Int32Array.of(1, 2, 3, 4);
    const counts = instancedArray(data, "ivec2");
    expect(counts.attribute.count).toBe(2);
    expect(counts.attribute.array).toBe(data);
  });

  it("rejects a node whose type doesn't match its attribute's layout", () => {
    expect(() => storage(new StorageBufferAttribute(4, 1, Float32Array), "vec4")).toThrow(
      /vec4.*4 components.*itemSize 1/,
    );
    expect(() => storage(new StorageBufferAttribute(4, 1, Float32Array), "uint")).toThrow(
      /uint.*Uint32Array.*Float32Array/,
    );
    expect(() => instancedArray(new Float32Array(3), "uint")).toThrow(/uint.*Uint32Array.*Float32Array/);
    expect(() => storage(new StorageBufferAttribute(4, 2, Int32Array), "ivec2")).not.toThrow();
  });

  it("rejects a typed array that isn't a whole number of elements", () => {
    expect(() => instancedArray(new Float32Array(5), "vec2")).toThrow(/5 values.*itemSize 2/);
  });

  it("is read-write until toReadOnly()", () => {
    const values = instancedArray(4, "float");
    expect(values.access).toBe("read_write");
    expect(values.toReadOnly()).toBe(values);
    expect(values.access).toBe("read");
  });

  it("gives two nodes over one attribute one binding, with the wider access", () => {
    const attribute = new StorageBufferAttribute(4, 1, Float32Array);
    const reader = storage(attribute, "float").toReadOnly();
    const writer = storage(attribute, "float");
    const program = compile(
      { stage: "compute" },
      Fn(() => {
        const i = invocationIndex();
        writer.element(i).assign(reader.element(i).add(1));
      })(),
    );
    const storages = program.resources.filter((r) => r.kind === "storage");
    expect(storages).toEqual([expect.objectContaining({ name: reader.name, access: "read_write" })]);
    expect(reader.name).toBe(writer.name);
  });
});
