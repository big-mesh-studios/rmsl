import type { Node, ShaderType, StorageAccess } from "./core";
import { compileWGSL, wgslUniformLayout, WGSL_UNIFORM_STRUCT } from "./backends/wgsl";

export type WgslStage = "compute" | "vertex" | "fragment";

export type WgslResource =
  | {
      kind: "storage";
      name: string;
      shaderType: ShaderType;
      access: StorageAccess;
      group: number;
      binding: number;
    }
  | {
      kind: "uniform";
      name: string;
      shaderType: ShaderType;
      group: number;
      binding: number;
      /** Byte offset within the shared `_RmslUniforms` buffer. */
      offset: number;
      /** Bytes occupied in total — an array's whole extent, not one element. */
      size: number;
      /** Element count, present only for a uniform array. */
      length?: number;
    };

export interface WgslProgram {
  code: string;
  entryPoint: string;
  stage: WgslStage;
  workgroupSize?: number;
  resources: WgslResource[];
}

export interface WgslCompileOptions {
  stage: WgslStage;
  workgroupSize?: number;
}

/**
 * Storage resources are collected from the node graph itself, not the
 * generated code: the WGSL backend's binding declarations carry only the
 * internal `_rmsl_sN` name it invents per binding (see `compileWGSLWithStage`
 * in `src/backends/wgsl.ts`), never the RMSL slot name a caller actually
 * passed to `storage()` — that name exists only on the graph's nodes. Binding
 * order is reproduced exactly as the backend assigns it: every distinct
 * `storage()` slot reachable from `root`, sorted by slot name.
 */
function collectStorageResources(root: Node<ShaderType> | readonly Node<ShaderType>[] | void): WgslResource[] {
  const seen = new Map<string, { shaderType: ShaderType; access: StorageAccess }>();
  const visited = new Set<unknown>();

  function walk(node: any): void {
    if (!node || typeof node !== "object" || visited.has(node)) return;
    visited.add(node);
    if (node.type === "storage") {
      const v = node.value;
      if (!seen.has(v.slot)) seen.set(v.slot, { shaderType: v.shaderType, access: v.access });
    }
    if (Array.isArray(node.params)) for (const p of node.params) walk(p);
  }

  const roots = Array.isArray(root) ? root : root ? [root] : [];
  for (const r of roots) walk(r);

  return [...seen.entries()]
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([name, info], binding) => ({
      kind: "storage" as const,
      name,
      shaderType: info.shaderType,
      access: info.access,
      group: 1,
      binding,
    }));
}

/**
 * Every uniform lives as a member of the one `_RmslUniforms` struct at
 * `@group(0) @binding(0)` (see `compileWGSLWithStage` in
 * `src/backends/wgsl.ts` — WGSL caps uniform *buffers* per stage, so one
 * struct is used no matter how many uniforms the program has). There is no
 * per-uniform binding to regex out the way storage resources have, so this
 * instead parses the struct's member list and re-runs `wgslUniformLayout` on
 * it to recover each member's offset — the same pure function the compiler
 * used to place them, so it reproduces the same offsets for the same member
 * set regardless of the order they're parsed back in.
 */
function inferUniformResources(code: string): WgslResource[] {
  const structMatch = new RegExp(`struct ${WGSL_UNIFORM_STRUCT} \\{([\\s\\S]*?)\\n\\};`).exec(code);
  if (!structMatch) return [];

  const memberPattern = /^\s*(\w+):\s*(?:array<(.+),\s*(\d+)>|([^,]+)),\s*$/;
  const members: { slot: string; type: string; length?: number }[] = [];
  for (const line of structMatch[1].split("\n")) {
    const m = memberPattern.exec(line);
    if (!m) continue;
    const [, name, arrayType, arrayLength, plainType] = m;
    members.push(
      arrayType !== undefined
        ? { slot: name, type: arrayType, length: Number(arrayLength) }
        : { slot: name, type: plainType },
    );
  }
  if (members.length === 0) return [];

  const layout = wgslUniformLayout(members);
  return layout.members.map((m) => ({
    kind: "uniform",
    name: m.name,
    shaderType: m.type as ShaderType,
    group: 0,
    binding: 0,
    offset: m.offset,
    size: m.size,
    ...(m.length !== undefined ? { length: m.length } : {}),
  }));
}

export function compile(
  options: WgslCompileOptions,
  root: Node<ShaderType> | readonly Node<ShaderType>[] | void,
): WgslProgram {
  if (options.stage !== "compute") {
    throw new Error(
      `[RMSL] @random-mesh/rmsl/wgsl currently supports only compute compilation`,
    );
  }

  const workgroupSize = options.workgroupSize ?? 64;
  const code = compileWGSL.compute(root as Node<ShaderType> | readonly Node<ShaderType>[], {
    workgroupSize,
  });

  return {
    code,
    entryPoint: "main",
    stage: options.stage,
    workgroupSize,
    resources: [...inferUniformResources(code), ...collectStorageResources(root)],
  };
}

export { compileWGSL };

export type { Adapter, TypedArray } from "./backends/adapter";
export { createWgslAdapter } from "./backends/adapter-wgsl";
export type { AdapterResult, CreateWgslAdapterOptions, WgslAdapter, WgslDrawOptions } from "./backends/adapter-wgsl";
