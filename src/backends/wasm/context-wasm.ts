import {
  ComputeNode,
  StorageBufferAttribute,
  type ShaderType,
  type UniformArrayNode,
  type UniformNode,
  type UniformValue,
} from "../../core";
import { slotOf, type TypedArray } from "../adapter";
import type { CpuRoutine } from "../cpu";
import { compileWasmFn, instantiateWasmRoutine, storageElementSize } from "./wasm";

/**
 * Several compute programs in one `WebAssembly.Memory`, sharing their storage
 * buffers — the WASM counterpart of `createWgslContext`, with WASM's
 * synchronous calls. Each program is compiled at its own `memoryBase`, so
 * their fixed-address data doesn't overlap, and each
 * {@link StorageBufferAttribute} gets one region that every program reading
 * it uses in place.
 */
export interface WasmContext {
  /** The memory every program and buffer lives in. */
  readonly memory: WebAssembly.Memory;
  /**
   * Runs each program in order, each seeing the writes of the ones before it.
   * A program is compiled the first time it runs.
   */
  compute(nodes: ComputeNode | readonly ComputeNode[]): void;
  /** Sets a uniform for every program that reads it, from the next `compute()` on. */
  setUniform<T extends ShaderType>(uniform: UniformNode<T>, value: UniformValue<T>): void;
  setUniform<T extends ShaderType>(uniform: UniformArrayNode<T>, value: UniformValue<T>[]): void;
  /** Writes `data` into the attribute's buffer, from element `offset` on. */
  write(attribute: StorageBufferAttribute, data: TypedArray, offset?: number): void;
  /**
   * The attribute's contents as bytes of its own `arrayClass`, the same layout
   * `WgslContext.getArrayBufferAsync()` returns. WASM holds `float` elements
   * as f64, so a float buffer is converted to f32 on the way out.
   */
  getArrayBuffer(attribute: StorageBufferAttribute): ArrayBuffer;
}

type Program = { routine: CpuRoutine; slots: { slot: string; attribute: StorageBufferAttribute }[] };

/** Bytes per component WASM stores an attribute's elements with: f64 for float, i32 otherwise. */
function componentSize(attribute: StorageBufferAttribute): number {
  return attribute.arrayClass === Float32Array ? 8 : 4;
}

/** Creates a {@link WasmContext} with a new, growable memory. */
export function createWasmContext(): WasmContext {
  const memory = new WebAssembly.Memory({ initial: 1 });
  const buffers = new Map<StorageBufferAttribute, number>();
  const programs = new Map<ComputeNode, Program>();
  const uniformValues: Record<string, number | number[]> = {};
  let cursor = 0;

  /** Reserves `bytes` at the end of everything placed so far, growing the memory to fit. */
  function allocate(bytes: number): number {
    const address = Math.ceil(cursor / 8) * 8;
    cursor = address + bytes;
    if (cursor > memory.buffer.byteLength) memory.grow(Math.ceil((cursor - memory.buffer.byteLength) / 65536));
    return address;
  }

  /** The components of the attribute's region, viewed as WASM stores them. */
  function region(attribute: StorageBufferAttribute): Float64Array | Int32Array | Uint32Array {
    const address = buffer(attribute);
    const length = attribute.count * attribute.itemSize;
    if (attribute.arrayClass === Float32Array) return new Float64Array(memory.buffer, address, length);
    return new attribute.arrayClass(memory.buffer, address, length) as Int32Array | Uint32Array;
  }

  function buffer(attribute: StorageBufferAttribute): number {
    let address = buffers.get(attribute);
    if (address !== undefined) return address;
    address = allocate(attribute.count * attribute.itemSize * componentSize(attribute));
    buffers.set(attribute, address);
    if (attribute.array) region(attribute).set(attribute.array);
    return address;
  }

  function program(node: ComputeNode): Program {
    const existing = programs.get(node);
    if (existing) return existing;

    const attributes = new Map<string, StorageBufferAttribute>();
    const visited = new Set<unknown>();
    const walk = (n: any): void => {
      if (!n || typeof n !== "object" || visited.has(n)) return;
      visited.add(n);
      if (n.type === "storage") attributes.set(n.value.slot, n.value.attribute);
      if (Array.isArray(n.params)) for (const p of n.params) walk(p);
    };
    walk(node.computeNode);

    const compiled = compileWasmFn(() => node.computeNode, { name: "main", params: [], memoryBase: cursor });
    if (compiled.params.some((p) => p.kind === "textureMemory")) {
      throw new Error("[RMSL] createWasmContext: programs that sample textures aren't supported yet.");
    }
    allocate(compiled.textureHeapBase - cursor);

    const slots = compiled.params.flatMap((p) => {
      if (p.kind !== "storageMemory") return [];
      const attribute = attributes.get(p.slot)!;
      if (storageElementSize(p.shaderType) !== attribute.itemSize * componentSize(attribute)) {
        throw new Error(
          `[RMSL] createWasmContext: a ${p.shaderType} storage node doesn't match its attribute's layout.`,
        );
      }
      return [{ slot: p.slot, attribute }];
    });

    const created = { routine: instantiateWasmRoutine(compiled, "main", memory), slots };
    programs.set(node, created);
    return created;
  }

  return {
    memory,

    compute(nodes) {
      const list = nodes instanceof ComputeNode ? [nodes] : nodes;
      for (const node of list) {
        if (node.count <= 0) continue;
        const { routine, slots } = program(node);
        const storageBuffers = Object.fromEntries(
          slots.map(({ slot, attribute }) => [slot, { address: buffer(attribute), length: attribute.count }]),
        );
        routine.compute({ uniforms: uniformValues, storageBuffers }, node.count);
      }
    },

    setUniform(uniform: UniformNode<ShaderType> | UniformArrayNode<ShaderType>, value: unknown) {
      uniformValues[slotOf(uniform)] = value as number | number[];
    },

    write(attribute, data, offset = 0) {
      region(attribute).set(data as ArrayLike<number>, offset * attribute.itemSize);
    },

    getArrayBuffer(attribute) {
      return attribute.arrayClass.from(region(attribute)).buffer as ArrayBuffer;
    },
  };
}
