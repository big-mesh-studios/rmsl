import { ProgramError } from "./errors";
import { describe, it, expect } from "vitest";
import {
  attributeArray,
  float,
  Fn,
  instancedArray,
  invocationIndex,
  storage,
  StorageBufferAttribute,
  vec2,
  vec4,
} from "./rmsl";
import { compile } from "./wgsl";
import { compileWasmFn } from "./wasm";
import { compileJSCompute } from "./js";

describe("storage buffers", () => {
  /**
   * @canon spec-an-instanced-array-takes-its-count-from-a-number-or-its-data
   */
  it("sizes an instancedArray from a count, with the element type's components and array class", () => {
    const positions = instancedArray(8, "vec3");
    expect(positions.attribute.count).toBe(8);
    expect(positions.attribute.itemSize).toBe(3);
    expect(positions.attribute.arrayClass).toBe(Float32Array);
    expect(instancedArray(4, "uint").attribute.arrayClass).toBe(Uint32Array);
    expect(attributeArray(4, "int").attribute.arrayClass).toBe(Int32Array);
  });

  /**
   * @canon spec-an-instanced-array-takes-its-count-from-a-number-or-its-data
   */
  it("takes an instancedArray's count and initial contents from a typed array", () => {
    const data = Int32Array.of(1, 2, 3, 4);
    const counts = instancedArray(data, "ivec2");
    expect(counts.attribute.count).toBe(2);
    expect(counts.attribute.array).toBe(data);
  });

  /**
   * @canon spec-a-buffer-holds-one-element-type
   */
  it("rejects a node whose type doesn't match its attribute's layout", () => {
    expect(() => storage(new StorageBufferAttribute(4, 1, Float32Array), "vec4")).toThrowRmsl(
      ProgramError,
      "storage-item-size",
    );
    expect(() => storage(new StorageBufferAttribute(4, 1, Float32Array), "uint")).toThrowRmsl(
      ProgramError,
      "storage-array-kind",
    );
    expect(() => instancedArray(new Float32Array(3), "uint")).toThrowRmsl(ProgramError, "storage-array-kind");
    expect(() => storage(new StorageBufferAttribute(4, 2, Int32Array), "ivec2")).not.toThrow();
  });

  /**
   * @canon spec-a-buffer-holds-one-element-type
   */
  it("names the array an element type needs with the article its sound takes", () => {
    expect(() => storage(new StorageBufferAttribute(4, 1, Float32Array), "int")).toThrowRmsl(
      ProgramError,
      "storage-array-kind",
    );
    expect(() => storage(new StorageBufferAttribute(4, 1, Int32Array), "uint")).toThrowRmsl(
      ProgramError,
      "storage-array-kind",
    );
    expect(() => storage(new StorageBufferAttribute(4, 1, Float32Array), "ivec2")).toThrowRmsl(
      ProgramError,
      "storage-item-size",
    );
  });

  /**
   * @canon spec-a-storage-buffer-holds-no-bool
   */
  it("rejects bool elements, which a WGSL storage buffer can't hold", () => {
    expect(() => instancedArray(4, "bool")).toThrowRmsl(ProgramError, "storage-of-bool");
    expect(() => storage(new StorageBufferAttribute(4, 3, Float32Array), "bvec3")).toThrowRmsl(
      ProgramError,
      "storage-of-bool",
    );
  });

  /**
   * @canon spec-a-buffer-holds-one-element-type
   */
  it("holds one element type per attribute", () => {
    const attribute = new StorageBufferAttribute(2, 6);
    expect(attribute.elementType).toBe(null);
    storage(attribute, "mat2x3");
    expect(attribute.elementType).toBe("mat2x3");
    expect(() => storage(attribute, "mat2x3")).not.toThrow();
    expect(() => storage(attribute, "mat3x2")).toThrowRmsl(ProgramError, "storage-of-another-element-type");
  });

  /**
   * @canon spec-a-buffer-holds-one-element-type
   */
  it("rejects a typed array that isn't a whole number of elements", () => {
    expect(() => instancedArray(new Float32Array(5), "vec2")).toThrowRmsl(
      ProgramError,
      "typed-array-not-whole-elements",
    );
  });

  /**
   * @canon spec-a-storage-node-is-read-write-until-to-read-only
   */
  it("is read-write until toReadOnly()", () => {
    const values = instancedArray(4, "float");
    expect(values.access).toBe("read_write");
    expect(values.toReadOnly()).toBe(values);
    expect(values.access).toBe("read");
  });

  /**
   * @canon spec-a-read-only-storage-element-cannot-be-assigned
   */
  it("refuses an assignment through a read-only node, on every backend", () => {
    const values = instancedArray(4, "vec4").toReadOnly();
    const writes = [
      // @ts-expect-error: a read-only storage node's element can't be assigned
      () => values.element(invocationIndex()).assign(vec4(1, 1, 1, 1)),
      // @ts-expect-error: a read-only storage node's element can't be assigned
      () => values.element(invocationIndex()).xy.assign(vec2(1, 1)),
      // @ts-expect-error: a read-only storage node's element can't be assigned
      () => values.element(invocationIndex()).element(2).assign(float(1)),
    ];
    for (const write of writes) {
      const program = Fn(() => {
        write();
      });
      expect(() => compile({ stage: "compute" }, program())).toThrowRmsl(ProgramError, "assign-to-read-only-storage");
      expect(() => compileWasmFn(() => program(), { name: "main", params: [] })).toThrowRmsl(
        ProgramError,
        "assign-to-read-only-storage",
      );
      expect(() => compileJSCompute(() => program(), { name: "main", params: [] })).toThrowRmsl(
        ProgramError,
        "assign-to-read-only-storage",
      );
    }
  });

  /**
   * @canon spec-nodes-over-one-buffer-share-one-binding
   */
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
