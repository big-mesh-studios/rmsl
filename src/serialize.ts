import {
  allocAttrId,
  allocUniformId,
  allocVaryingId,
  attachUniformArrayElement,
  node,
  someNode,
  storage,
  StorageBufferAttribute,
  StorageInstancedBufferAttribute,
  type BaseNode,
  type Node,
  type ShaderType,
  type StorageAccess,
} from "./core";

/**
 * A node graph as plain JSON-safe data: every node and every storage buffer
 * once, referenced by index, so a node or buffer read in several places is
 * one node or buffer again after {@link deserialize}.
 */
export interface SerializedGraph {
  nodes: SerializedNode[];
  buffers: SerializedBuffer[];
  /** The index of the root node, or of each root when an array of roots was serialized. */
  roots: number | number[];
}

/** A node: its shader type, node kind, value, and its children as indices into `nodes`. */
export interface SerializedNode {
  _t: string;
  type: string;
  value?: unknown;
  params?: number[];
}

/** A storage buffer: its layout, and its initial contents when it was built from an array. */
export interface SerializedBuffer {
  instanced: boolean;
  count: number;
  itemSize: number;
  arrayClass: "Float32Array" | "Int32Array" | "Uint32Array";
  array: number[] | null;
}

type Root = Node<ShaderType> | readonly Node<ShaderType>[];

const ARRAY_CLASSES = { Float32Array, Int32Array, Uint32Array } as const;

/** The node kinds named after their slot. */
const NAMED = new Set(["uniform", "uniformArray", "attribute", "varying"]);

/** The node kinds holding an id drawn from a counter, which {@link deserialize} draws afresh. */
const COUNTED: Record<string, () => number> = {
  uniform: allocUniformId,
  uniformArray: allocUniformId,
  attribute: allocAttrId,
  varying: allocVaryingId,
};

/**
 * Copies a node graph into JSON-safe data. Takes a node, an array of roots
 * for a program built from several `Fn`s, or the callable an `Fn(...)`
 * definition returns, which it calls to get its root.
 */
export function serialize(root: Root | (() => Root)): SerializedGraph {
  const resolved = typeof root === "function" ? root() : root;
  const nodes: SerializedNode[] = [];
  const nodeIndex = new Map<BaseNode<ShaderType>, number>();

  // Buffers keep the order of their attributes' ids, which a program binds them in.
  const found = new Set<StorageBufferAttribute>();
  someNode(resolved, (n) => {
    if (n.type === "storage") found.add(n.value.attribute);
  });
  const attributes = [...found].sort((a, b) => a.id - b.id);
  const bufferIndex = new Map(attributes.map((attribute, i) => [attribute, i]));
  const buffers: SerializedBuffer[] = attributes.map((attribute) => ({
    instanced: attribute instanceof StorageInstancedBufferAttribute,
    count: attribute.count,
    itemSize: attribute.itemSize,
    arrayClass: attribute.arrayClass.name as SerializedBuffer["arrayClass"],
    array: attribute.array ? Array.from(attribute.array) : null,
  }));

  const add = (n: BaseNode<ShaderType>): number => {
    const seen = nodeIndex.get(n);
    if (seen !== undefined) return seen;
    const params = n.params?.map(add);
    const entry: SerializedNode = { _t: n._t, type: n.type };
    if (n.type === "storage") {
      const value = n.value as { shaderType: string; access: StorageAccess; attribute: StorageBufferAttribute };
      entry.value = { shaderType: value.shaderType, access: value.access, buffer: bufferIndex.get(value.attribute)! };
    } else if (n.value !== undefined) {
      entry.value = n.value;
    }
    if (params) entry.params = params;
    const index = nodes.length;
    nodes.push(entry);
    nodeIndex.set(n, index);
    return index;
  };

  const roots = Array.isArray(resolved)
    ? resolved.map((r) => add(r as BaseNode<ShaderType>))
    : add(resolved as BaseNode<ShaderType>);
  return { nodes, buffers, roots };
}

/**
 * Rebuilds the node graph {@link serialize} copied, after a JSON round-trip or
 * not: a node, or an array of roots when an array was serialized.
 *
 * Each storage buffer becomes a new `StorageBufferAttribute`, shared by every
 * storage node that read the same buffer, and read through `.attribute` on
 * those nodes. A uniform, attribute or varying node keeps its name, and draws
 * a new id from the counter freshly built nodes draw from, so the two never
 * collide in one program.
 */
export function deserialize(graph: SerializedGraph): Node<ShaderType> | Node<ShaderType>[] {
  const buffers = graph.buffers.map((b) => {
    const Attribute = b.instanced ? StorageInstancedBufferAttribute : StorageBufferAttribute;
    const ArrayClass = ARRAY_CLASSES[b.arrayClass];
    return b.array
      ? new Attribute(ArrayClass.from(b.array), b.itemSize)
      : new Attribute(b.count, b.itemSize, ArrayClass);
  });

  const nodes: Node<ShaderType>[] = [];
  for (const entry of graph.nodes) nodes.push(deserializeNode(entry, nodes, buffers));

  return Array.isArray(graph.roots) ? graph.roots.map((r) => nodes[r]!) : nodes[graph.roots]!;
}

function deserializeNode(
  entry: SerializedNode,
  nodes: readonly Node<ShaderType>[],
  buffers: readonly StorageBufferAttribute[],
): Node<ShaderType> {
  if (entry.type === "storage") {
    const value = entry.value as { shaderType: ShaderType; access: StorageAccess; buffer: number };
    const result = storage(buffers[value.buffer]!, value.shaderType);
    if (value.access === "read") result.toReadOnly();
    return result as unknown as Node<ShaderType>;
  }

  const params = entry.params?.map((i) => nodes[i]! as BaseNode<ShaderType>);
  let value = entry.value as Record<string, unknown> | undefined;
  const draw = COUNTED[entry.type];
  if (draw && value) value = { ...value, id: draw() };

  const result = node({
    _t: entry._t,
    type: entry.type,
    value,
    params,
    name: NAMED.has(entry.type) ? (value!.slot as string) : undefined,
  }) as any;
  if (entry.type === "uniformArray") {
    result.length = value!.length;
    attachUniformArrayElement(result, entry._t as ShaderType);
  }
  return result;
}
