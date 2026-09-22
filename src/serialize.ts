import {
  allocAttrId,
  allocUniformId,
  allocVaryingId,
  attachElement,
  node,
  type BaseNode,
  type Node,
  type SerializedNode,
  type ShaderType,
} from "./core";

export type { SerializedNode };

type Root = Node<ShaderType> | readonly Node<ShaderType>[];

/**
 * Recursively copies a node graph into a JSON-safe plain object — a thin
 * wrapper over `Node.toJSON()`, which `JSON.stringify` also calls
 * automatically. Also accepts the callable an `Fn(...)` definition returns
 * (calling it to get its root first) and an array of roots, so a whole
 * multi-`Fn` program can be serialized in one call.
 */
export function serialize(root: Node<ShaderType>): SerializedNode;
export function serialize(root: readonly Node<ShaderType>[]): SerializedNode[];
export function serialize(root: () => Root): SerializedNode | SerializedNode[];
export function serialize(root: Root | (() => Root)): SerializedNode | SerializedNode[] {
  const resolved = typeof root === "function" ? root() : root;
  return Array.isArray(resolved) ? resolved.map((r) => (r as Node<ShaderType>).toJSON()) : (resolved as Node<ShaderType>).toJSON();
}

/**
 * Rebuilds a `Node` (or array of `Node`s) from data produced by
 * `serialize()`/`Node.toJSON()` (typically after a `JSON.stringify`/
 * `JSON.parse` round-trip). Children are rebuilt first, then each node is
 * reconstructed with the `node()` constructor.
 *
 * A `uniform`/`attribute`/`varying`/`uniformArray` node's `value.id` is
 * replaced with a freshly allocated one rather than trusted as-is: it is
 * compile-session bookkeeping (a dedup key the backends use within one
 * `compile()` call), drawn from a module-level counter that an incoming
 * node's baked-in id never participated in. Reusing the live counter here
 * guarantees a deserialized graph can't collide with ids allocated by
 * freshly-constructed nodes in the same session. `value.slot` (surfaced as
 * `.name`, and what the generated code and the JS/WASM adapters actually key
 * on) is left untouched, so compiled output is unaffected. `storage()` nodes
 * carry no `id` at all — their `slot` is the caller-chosen cross-graph
 * identity used to share a buffer between systems, so nothing is renumbered.
 */
export function deserialize(data: SerializedNode): Node<ShaderType>;
export function deserialize(data: readonly SerializedNode[]): Node<ShaderType>[];
export function deserialize(data: SerializedNode | readonly SerializedNode[]): Node<ShaderType> | Node<ShaderType>[] {
  return Array.isArray(data) ? data.map((d) => deserializeNode(d)) : deserializeNode(data as SerializedNode);
}

function deserializeNode(data: SerializedNode): Node<ShaderType> {
  const params = data.params?.map(deserializeNode) as BaseNode<ShaderType>[] | undefined;

  let value = data.value;
  if (
    (data.type === "uniform" || data.type === "attribute" || data.type === "varying" || data.type === "uniformArray") &&
    value &&
    typeof value === "object" &&
    "id" in value
  ) {
    const alloc =
      data.type === "attribute" ? allocAttrId : data.type === "varying" ? allocVaryingId : allocUniformId;
    value = { ...(value as object), id: alloc() };
  }

  const result = node({
    _t: data._t,
    type: data.type,
    value,
    params,
  }) as any;

  const v = value as { slot?: string; length?: number; access?: string } | undefined;
  if (v?.slot !== undefined && (data.type === "uniform" || data.type === "attribute" || data.type === "varying")) {
    result.name = v.slot;
  }
  if (data.type === "storage" && v?.slot !== undefined) {
    result.name = v.slot;
    result.access = v.access;
    attachElement(result, "storageElement", data._t as ShaderType);
  }
  if (data.type === "uniformArray" && v?.slot !== undefined) {
    result.name = v.slot;
    result.length = v.length;
    attachElement(result, "uniformArrayElement", data._t as ShaderType);
  }

  return result as Node<ShaderType>;
}
