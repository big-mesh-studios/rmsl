import {
  someNode,
  type ComputeNode,
  type Node,
  type ShaderType,
  type StorageAccess,
  type StorageBufferAttribute,
} from "./core";
import { compileWgsl, typeToWGSL, wgslUniformLayout, WGSL_UNIFORM_STRUCT } from "./backends/wgsl/wgsl";

export type WgslStage = "compute" | "vertex" | "fragment";

export type WgslResource =
  | {
      kind: "storage";
      name: string;
      shaderType: ShaderType;
      access: StorageAccess;
      /** The buffer the storage node reads and writes. */
      attribute: StorageBufferAttribute;
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
 * in `src/backends/wgsl.ts`), never the storage node's slot name (its `.name`,
 * one per buffer attribute) — that name exists only on the graph's nodes.
 * Binding order is reproduced exactly as the backend assigns it: every
 * distinct `storage()` slot reachable from `root`, sorted by slot name.
 */
function collectStorageResources(root: Node<ShaderType> | readonly Node<ShaderType>[]): WgslResource[] {
  const seen = new Map<string, { shaderType: ShaderType; access: StorageAccess; attribute: StorageBufferAttribute }>();
  someNode(root, (node) => {
    if (node.type !== "storage") return;
    const v = node.value;
    const existing = seen.get(v.slot);
    if (!existing) seen.set(v.slot, { shaderType: v.shaderType, access: v.access, attribute: v.attribute });
    else if (existing.access !== v.access) existing.access = "read_write";
  });

  return [...seen.entries()]
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([name, info], binding) => ({
      kind: "storage" as const,
      name,
      shaderType: info.shaderType,
      access: info.access,
      attribute: info.attribute,
      group: 1,
      binding,
    }));
}

/** The rmsl type a WGSL uniform member type stands for: `i32` is `int`, `vec2<u32>` is `uvec2`. */
function shaderTypeOfWgsl(wgslType: string): ShaderType {
  let match = Object.keys(typeToWGSL).find((key) => typeToWGSL[key] === wgslType);
  return (match ?? wgslType) as ShaderType;
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
    shaderType: shaderTypeOfWgsl(m.type),
    group: 0,
    binding: 0,
    offset: m.offset,
    size: m.size,
    ...(m.length !== undefined ? { length: m.length } : {}),
  }));
}

/**
 * Compiles a compute program. Given a {@link ComputeNode}, its own
 * `workgroupSize` is used and the program is bounded by its `countNode`
 * rather than by the first storage buffer's length.
 */
export function compile(
  options: WgslCompileOptions,
  program: Node<ShaderType> | readonly Node<ShaderType>[] | ComputeNode,
): WgslProgram {
  if (options.stage !== "compute") {
    throw new Error(`[RMSL] @random-mesh/rmsl/wgsl currently supports only compute compilation`);
  }

  // Checked by flag rather than instanceof, so a node from another copy of the package is still recognized.
  const computeNode = (program as ComputeNode).isComputeNode ? (program as ComputeNode) : undefined;
  const root = computeNode ? computeNode.computeNode : (program as Node<ShaderType> | readonly Node<ShaderType>[]);
  const workgroupSize = computeNode?.workgroupSize ?? options.workgroupSize ?? 64;
  const code = compileWgsl.compute(root, { workgroupSize, count: computeNode?.countNode });

  return {
    code,
    entryPoint: "main",
    stage: options.stage,
    workgroupSize,
    resources: [...inferUniformResources(code), ...collectStorageResources(root)],
  };
}

export { compileWgsl, wgslUniformLayout };
export type {
  CompileWGSLOptions,
  WgslSamplerDeclaration,
  WgslUniformDeclaration,
  WgslUniformMember,
} from "./backends/wgsl/wgsl";

export { compileWgslFn } from "./backends/wgsl/wgsl";

export type { Adapter, TypedArray } from "./backends/adapter";
export { createWgsl, createWgslCompute } from "./backends/wgsl/adapter-wgsl";
export { createWgslContext } from "./backends/wgsl/context-wgsl";
export type { CreateWgslContextOptions, WgslContext } from "./backends/wgsl/context-wgsl";
export type {
  AdapterResult,
  CreateWgslAdapterOptions,
  CreateWgslComputeOptions,
  WgslAdapter,
  WgslComputeAdapter,
  WgslDrawOptions,
} from "./backends/wgsl/adapter-wgsl";
