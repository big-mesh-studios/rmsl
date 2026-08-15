import type { BaseNode, ShaderType } from "../../../rmsl";

export interface CollectedNodes {
  uniforms: Set<BaseNode<ShaderType>>;
  attributes: Set<BaseNode<ShaderType>>;
  varyings: Set<BaseNode<ShaderType>>;
}

/**
 * Walk a compiled graph and collect every uniform/attribute/varying node it
 * actually references. The compilers only emit reachable declarations, so the
 * renderer must derive its binding set from the graph, not from what a builder
 * happened to register.
 */
export function collectNodes(root: BaseNode<ShaderType> | BaseNode<ShaderType>[] | undefined): CollectedNodes {
  const uniforms = new Set<BaseNode<ShaderType>>();
  const attributes = new Set<BaseNode<ShaderType>>();
  const varyings = new Set<BaseNode<ShaderType>>();
  const seen = new Set<BaseNode<ShaderType>>();

  const visit = (node: BaseNode<ShaderType> | null | undefined): void => {
    if (!node || typeof node !== "object") return;
    if (seen.has(node)) return;
    seen.add(node);
    const type = node.type;
    if (type === "uniform" || type === "uniformArray") uniforms.add(node);
    else if (type === "attribute") attributes.add(node);
    else if (type === "varying") varyings.add(node);
    if (Array.isArray(node.params)) {
      for (const p of node.params) visit(p);
    }
  };

  const roots = Array.isArray(root) ? root : [root];
  for (const r of roots) visit(r);

  return { uniforms, attributes, varyings };
}
