import { DataError } from "./errors";
import { withArticle } from "./article";
import {
  allocAttrId,
  allocUniformId,
  allocVaryingId,
  attachUniformArrayElement,
  assertNodeNames,
  claimVarName,
  assertStructType,
  node,
  NODE_TYPES,
  claimStructName,
  isGeneratedStructName,
  registerStructType,
  structsOf,
  someNode,
  storage,
  StorageBufferAttribute,
  StorageInstancedBufferAttribute,
  type BaseNode,
  type FnDefinition,
  type Node,
  type ShaderType,
  type StorageAccess,
  type StructType,
} from "./core";

/**
 * A node graph as plain JSON-safe data: every node and every storage buffer
 * once, referenced by index, so a node or buffer read in several places is
 * one node or buffer again after {@link deserialize}.
 */
export interface SerializedGraph {
  nodes: SerializedNode[];
  buffers: SerializedBuffer[];
  /** The struct types the graph uses, each after the structs its members name. */
  structs?: StructType[];
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

/** A storage buffer: its layout, and its contents. */
export interface SerializedBuffer {
  instanced: boolean;
  count: number;
  itemSize: number;
  arrayClass: "Float32Array" | "Int32Array" | "Uint32Array";
  array: (number | NonFiniteNumber)[];
}

/** A number JSON cannot hold, NaN or an infinity, written out so it survives a round-trip. */
export interface NonFiniteNumber {
  nonFinite: "NaN" | "Infinity" | "-Infinity";
}

/** `value` with every number JSON cannot hold, at any depth, written as a {@link NonFiniteNumber}. */
function encodeNumbers(value: unknown): unknown {
  if (typeof value === "number") return Number.isFinite(value) ? value : { nonFinite: String(value) };
  if (Array.isArray(value)) return value.map(encodeNumbers);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, v]) => [key, encodeNumbers(v)]));
  }
  return value;
}

/** `value` with every {@link NonFiniteNumber}, at any depth, read back as the number it stands for. */
function decodeNumbers(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(decodeNumbers);
  if (value !== null && typeof value === "object") {
    const keys = Object.keys(value);
    const nonFinite = (value as Partial<NonFiniteNumber>).nonFinite;
    if (keys.length === 1 && (nonFinite === "NaN" || nonFinite === "Infinity" || nonFinite === "-Infinity")) {
      return Number(nonFinite);
    }
    return Object.fromEntries(Object.entries(value).map(([key, v]) => [key, decodeNumbers(v)]));
  }
  return value;
}

type Root = Node<ShaderType> | readonly Node<ShaderType>[];

const ARRAY_CLASSES = { Float32Array, Int32Array, Uint32Array } as const;

/** The node kinds named after their slot, with the counter their id is drawn from and their generated names' prefix. */
const NAMED: Record<string, { draw: () => number; prefix: string }> = {
  uniform: { draw: allocUniformId, prefix: "_rmsl_u" },
  uniformArray: { draw: allocUniformId, prefix: "_rmsl_u" },
  attribute: { draw: allocAttrId, prefix: "_rmsl_a" },
  varying: { draw: allocVaryingId, prefix: "_rmsl_v" },
};

/**
 * Whether `name` was drawn from one of rmsl's counters, and so only means
 * something in the process that built it. A fixed reserved name, such as
 * `time()`'s `_rmsl_time`, means the same everywhere.
 */
const isGenerated = (name: string) => /^_rmsl_[uav]?\d+$/.test(name);

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
    array: Array.from(attribute.array, (v) => encodeNumbers(v) as number | NonFiniteNumber),
  }));

  // A generated name is saved as a number local to this graph, the same for every node that had it.
  const locals = new Map<string, number>();
  const local = (name: string) => {
    if (!locals.has(name)) locals.set(name, locals.size);
    return locals.get(name)!;
  };

  const add = (n: BaseNode<ShaderType>): number => {
    const seen = nodeIndex.get(n);
    if (seen !== undefined) return seen;
    const params = n.params?.map(add);
    const entry: SerializedNode = { _t: n._t, type: n.type };
    if (n.type === "storage") {
      const value = n.value as { shaderType: string; access: StorageAccess; attribute: StorageBufferAttribute };
      entry.value = { shaderType: value.shaderType, access: value.access, buffer: bufferIndex.get(value.attribute)! };
    } else if (n.type === "call") {
      const { name, type, inputs, body } = n.value as FnDefinition;
      entry.value = { name, type, inputs, body: add(body) };
    } else if (n.type === "var") {
      const { varName, varType, parameter } = n.value as { varName: string; varType: string; parameter?: boolean };
      entry.value = isGenerated(varName) ? { varType, local: local(varName) } : { varType, varName };
      if (parameter) (entry.value as { parameter?: boolean }).parameter = true;
    } else if (NAMED[n.type]) {
      const { id: _, slot, ...rest } = n.value as { id: number; slot: string };
      entry.value = isGenerated(slot) ? { ...rest, local: local(slot) } : { ...rest, slot };
    } else if (n.value !== undefined) {
      entry.value = encodeNumbers(n.value);
    }
    if (params) entry.params = params;
    const index = nodes.length;
    nodes.push(entry);
    nodeIndex.set(n, index);
    return index;
  };

  // Named nodes come first, in the order they were made, so a restored graph draws its ids in that order.
  const named: BaseNode<ShaderType>[] = [];
  someNode(resolved, (n) => {
    if (NAMED[n.type]) named.push(n);
  });
  for (const n of named.sort((a, b) => (a.value as { id: number }).id - (b.value as { id: number }).id)) add(n);

  const roots = Array.isArray(resolved)
    ? resolved.map((r) => add(r as BaseNode<ShaderType>))
    : add(resolved as BaseNode<ShaderType>);

  const structs = structsOf(resolved).map(({ name, members }) => ({
    name,
    members: members.map(({ name, type }) => ({ name, type })),
  }));
  return structs.length > 0 ? { nodes, buffers, structs, roots } : { nodes, buffers, roots };
}

/**
 * Rebuilds the node graph {@link serialize} copied, after a JSON round-trip or
 * not: a node, or an array of roots when an array was serialized.
 *
 * Each storage buffer becomes a new `StorageBufferAttribute`, shared by every
 * storage node that read the same buffer, and read through `.attribute` on
 * those nodes. A variable, uniform, attribute or varying whose name rmsl
 * generated gets a new one, as a freshly built node would, so a restored
 * graph and a fresh one never share a name; a name the program chose, as
 * `uniformRaw()` and `toVar("x")` take, is kept.
 *
 * Throws on data `serialize()` could not have produced: a child, buffer or
 * root index that names nothing, a child listed after the node using it, an
 * unknown array type, a kind of node rmsl does not have, or a named input
 * without a name.
 */
export function deserialize(graph: SerializedGraph): Node<ShaderType> | Node<ShaderType>[] {
  if (!Array.isArray(graph?.nodes) || !Array.isArray(graph.buffers)) {
    throw new DataError("graph-shape", "deserialize: the data needs a nodes and a buffers array, as serialize() gives");
  }
  const buffers = graph.buffers.map((b, i) => {
    const Attribute = b.instanced ? StorageInstancedBufferAttribute : StorageBufferAttribute;
    const ArrayClass = ARRAY_CLASSES[b.arrayClass];
    if (!ArrayClass) {
      throw new DataError(
        "buffer-of-another-kind",
        `deserialize: buffer ${i} holds ${withArticle(String(b.arrayClass))}, not a Float32Array, Int32Array or Uint32Array`,
        { index: i },
      );
    }
    return new Attribute(ArrayClass.from(b.array.map((v) => decodeNumbers(v) as number)), b.itemSize);
  });

  // A struct is recorded before the nodes that hold it, as `struct` records it before its values are built.
  // A name rmsl gave a struct with none only means something in the process that built it, so the struct takes a new one.
  const structNames = new Map<string, string>();
  const typeOf = (type: string) => structNames.get(type) ?? type;
  for (const struct of graph.structs ?? []) {
    const name = isGeneratedStructName(struct.name) ? claimStructName() : struct.name;
    structNames.set(struct.name, name);
    const type = { name, members: struct.members.map((m) => ({ name: m.name, type: typeOf(m.type) })) };
    assertStructType(type);
    registerStructType(type, true);
  }
  const entries = structNames.size === 0 ? graph.nodes : graph.nodes.map((entry) => renameStructs(entry, typeOf));

  const nodes: Node<ShaderType>[] = [];
  const names = new Map<number, { id?: number; name: string }>();
  for (const entry of entries) nodes.push(deserializeNode(entry, nodes, buffers, names, entries.length));

  const root = (r: number) => nodes[indexInto(r, nodes.length, "root", "a node")]!;
  return Array.isArray(graph.roots) ? graph.roots.map(root) : root(graph.roots);
}

/** `entry` with each struct type it names given by `typeOf`: its own type, a variable's, a function's and a struct value's. */
function renameStructs(entry: SerializedNode, typeOf: (type: string) => string): SerializedNode {
  const value = entry.value as Record<string, any> | undefined;
  let renamed: unknown = value;
  if (entry.type === "var" && value) renamed = { ...value, varType: typeOf(value.varType) };
  if (entry.type === "call" && value) {
    renamed = {
      ...value,
      type: typeOf(value.type),
      inputs: (value.inputs ?? []).map((input: any) => ({ ...input, type: typeOf(input.type) })),
    };
  }
  if (entry.type === "struct" && value) {
    renamed = {
      name: typeOf(value.name),
      members: (value.members ?? []).map((m: any) => ({ ...m, type: typeOf(m.type) })),
    };
  }
  return { ...entry, _t: typeOf(entry._t), ...(renamed === undefined ? {} : { value: renamed }) };
}

/** `index`, when it names one of `length` items, or an error saying what `user` named instead. */
function indexInto(index: number, length: number, user: string, item: string, at?: number): number {
  if (Number.isInteger(index) && index >= 0 && index < length) return index;
  throw new DataError(
    "index-out-of-range",
    `deserialize: ${user} names ${item} at ${index}, but there are only ${length}`,
    {
      index: at,
    },
  );
}

/** The fresh name, and id, a graph-local name `local` stands for, made by `make` the first time. */
function localName(
  names: Map<number, { id?: number; name: string }>,
  local: number,
  make: () => { id?: number; name: string },
): { id?: number; name: string } {
  if (!names.has(local)) names.set(local, make());
  return names.get(local)!;
}

function deserializeNode(
  entry: SerializedNode,
  nodes: readonly Node<ShaderType>[],
  buffers: readonly StorageBufferAttribute[],
  names: Map<number, { id?: number; name: string }>,
  total: number,
): Node<ShaderType> {
  if (!NODE_TYPES.has(entry.type)) {
    throw new DataError(
      "unknown-node-kind",
      `deserialize: node ${nodes.length} is a "${entry.type}", which is not a kind of node rmsl has`,
      { index: nodes.length },
    );
  }
  const needsValue = entry.type === "storage" || entry.type === "var" || entry.type === "call" || entry.type in NAMED;
  if (needsValue && (typeof entry.value !== "object" || entry.value === null)) {
    throw new DataError(
      "node-without-value",
      `deserialize: node ${nodes.length}, ${withArticle(entry.type)}, has no value`,
      {
        index: nodes.length,
      },
    );
  }
  if (entry.type === "storage") {
    const value = entry.value as { shaderType: ShaderType; access: StorageAccess; buffer: number };
    const buffer = indexInto(value.buffer, buffers.length, `node ${nodes.length}`, "a buffer", nodes.length);
    const result = storage(buffers[buffer]!, value.shaderType);
    if (value.access === "read") result.toReadOnly();
    return result as unknown as Node<ShaderType>;
  }

  const params = entry.params?.map((i) => {
    if (Number.isInteger(i) && i >= nodes.length && i < total) {
      throw new DataError(
        "child-after-node",
        `deserialize: node ${nodes.length} names a child at ${i}, after it; a child has to come before the node using it`,
        { index: nodes.length },
      );
    }
    return nodes[indexInto(i, total, `node ${nodes.length}`, "a child", nodes.length)]! as BaseNode<ShaderType>;
  });
  let value = decodeNumbers(entry.value) as Record<string, unknown> | undefined;
  let name: string | undefined;
  if (entry.type === "var") {
    const { local, ...rest } = value as { local?: number; varName?: string; varType: string };
    value =
      local === undefined
        ? rest
        : { ...rest, varName: localName(names, local, () => ({ name: claimVarName(undefined) })).name };
  }
  if (entry.type === "call") {
    const { body, ...rest } = value as Record<string, unknown> & { body: number };
    const at = indexInto(body, total, `node ${nodes.length}`, "a function body", nodes.length);
    if (at >= nodes.length) {
      throw new DataError(
        "body-after-call",
        `deserialize: node ${nodes.length} names a function body at ${at}, after it; a body has to come before the call`,
        { index: nodes.length },
      );
    }
    value = { ...rest, body: nodes[at] };
  }
  const named = NAMED[entry.type];
  if (named) {
    const { local, slot, ...rest } = value as { local?: number; slot?: string };
    if (local === undefined && (typeof slot !== "string" || slot === "")) {
      throw new DataError(
        "node-without-name",
        `deserialize: ${withArticle(entry.type)} node has no name, neither a slot nor a local one`,
        { index: nodes.length },
      );
    }
    const fresh =
      local === undefined
        ? { id: named.draw(), name: slot! }
        : localName(names, local, () => {
            const id = named.draw();
            return { id, name: `${named.prefix}${id}` };
          });
    value = { ...rest, id: fresh.id, slot: fresh.name };
    name = fresh.name;
  }

  const result = node({ _t: entry._t, type: entry.type, value, params, name }) as any;
  assertNodeNames(result, `deserialize: node ${nodes.length}, `);
  if (entry.type === "uniformArray") {
    result.length = value!.length;
    attachUniformArrayElement(result, entry._t as ShaderType);
  }
  return result;
}
