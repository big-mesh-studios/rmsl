// === WGSL compute adapter ===
// Every WGSL compute caller so far (see apps/ecs-demo) hand-writes the same
// ceremony: create one storage buffer per attribute and output in binding
// order, a uniform buffer packed by `wgslUniformLayout`, the two bind
// groups, and a staging buffer to read results back. All of that is
// mechanical once the program's own attributes/uniforms/outputs are known —
// `reflectCompute` walks the graph the same way `compileWGSLWithStage` does
// (see wgsl.ts) to recover that list without the caller repeating it.
import { Node, ShaderType } from "../core";
import { Adapter, TypedArray } from "./adapter";
import { CompileCtx } from "./shared";
import { compileWGSLStage, compileWGSLWithStage, wgslUniformLayout, WgslUniformDeclaration } from "./wgsl";

function freshCtx(shaderStage: CompileCtx["shaderStage"]): CompileCtx {
  return {
    nextId: 0,
    shaderStage,
    uniforms: new Map(),
    attributes: new Map(),
    varyings: new Map(),
    outputs: new Map(),
    wgslSamplers: new Map(),
    varDefs: new Map(),
    memo: new Map(),
    wgslHelpers: new Set(),
    positionWritten: false,
    inFn: false,
    fragDepthUsed: false,
    fragCoordUsed: false,
    jsParams: new Set(),
    jsHelpers: new Set(),
    outTarget: null,
    derivatives: "throw",
    reentrant: false,
    jsNeedsRes: false,
  };
}

/**
 * What a compute program reads and writes, in the same order
 * `compileWGSLWithStage` assigns `@group(1)` bindings: attributes first
 * (creation order), then outputs.
 */
function reflectCompute(root: Node<ShaderType> | readonly Node<ShaderType>[]) {
  let ctx = freshCtx("compute");
  let nodes = Array.isArray(root) ? root : [root];
  for (let n of nodes) compileWGSLStage(n, ctx);
  return {
    attributes: [...ctx.attributes.entries()].sort((a, b) => a[0] - b[0]).map(([, info]) => info),
    outputs: [...ctx.outputs.values()],
    uniforms: [...ctx.uniforms.values()].sort((a, b) => a.slot.localeCompare(b.slot)),
  };
}

/** One typed array per output slot, keyed by name — the compute analogue of
 * the ecs-demo's `{ outputs: { [slot]: value } }` JS/WASM result shape. */
export type WgslComputeResult = Record<string, TypedArray>;

export function createWgslAdapter(
  root: Node<ShaderType> | readonly Node<ShaderType>[],
): Adapter<WgslComputeResult> {
  let reflection = reflectCompute(root);
  let device: GPUDevice | null = null;
  let pipeline: GPUComputePipeline | null = null;

  let n = 0;
  let inBuffers = new Map<string, GPUBuffer>();
  let outBuffers = new Map<string, GPUBuffer>();
  let bindGroup0: GPUBindGroup | null = null;
  let bindGroup1: GPUBindGroup | null = null;
  let uniformBuffer: GPUBuffer | null = null;
  let uniformScratch: Float32Array | null = null;
  let uniformLayout: ReturnType<typeof wgslUniformLayout> | null = null;
  let staging: GPUBuffer | null = null;

  // setUniform/setAttribute may be called before attach() resolves (the
  // caller shouldn't have to sequence its own setup around ours), so values
  // that arrive early are replayed once the device exists.
  let pendingUniforms = new Map<string, number | number[]>();
  let pendingAttributes = new Map<string, TypedArray>();

  function declaredUniforms(): WgslUniformDeclaration[] {
    return reflection.uniforms.map((u) => ({ slot: u.slot, type: u.type, length: u.length }));
  }

  function rebuildStorageBuffers() {
    if (!device || !pipeline) return;
    for (let buf of inBuffers.values()) buf.destroy();
    for (let buf of outBuffers.values()) buf.destroy();
    staging?.destroy();

    let usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    let bytes = Math.max(4, n * 4);
    inBuffers = new Map(reflection.attributes.map((a) => [a.slot, device!.createBuffer({ size: bytes, usage })]));
    outBuffers = new Map(reflection.outputs.map((o) => [o.slot, device!.createBuffer({ size: bytes, usage })]));

    let entries: GPUBindGroupEntry[] = [];
    let binding = 0;
    for (let attr of reflection.attributes) entries.push({ binding: binding++, resource: { buffer: inBuffers.get(attr.slot)! } });
    for (let out of reflection.outputs) entries.push({ binding: binding++, resource: { buffer: outBuffers.get(out.slot)! } });
    bindGroup1 = device.createBindGroup({ layout: pipeline.getBindGroupLayout(1), entries });

    staging = device.createBuffer({
      size: bytes * Math.max(1, outBuffers.size),
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  let adapter: Adapter<WgslComputeResult> = {
    async attach() {
      let gpuAdapter = await navigator.gpu?.requestAdapter();
      if (!gpuAdapter) throw new Error("[RMSL] WebGPU is not available");
      device = await gpuAdapter.requestDevice();

      let declared = declaredUniforms();
      let code = compileWGSLWithStage(root, "compute", declared.length > 0 ? { uniforms: declared } : undefined);
      let module = device.createShaderModule({ code });
      pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });

      if (declared.length > 0) {
        uniformLayout = wgslUniformLayout(declared);
        uniformBuffer = device.createBuffer({
          size: Math.max(16, uniformLayout.size),
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        uniformScratch = new Float32Array(uniformBuffer.size / 4);
        bindGroup0 = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
        });
      }

      // Attribute buffers are sized to the first upload, so a pending
      // attribute has to be replayed before the (empty) default-sized
      // buffers are built.
      for (let [slot, data] of pendingAttributes) if (n === 0) n = data.length;
      rebuildStorageBuffers();
      for (let [slot, value] of pendingUniforms) adapter.setUniform(slot, value);
      for (let [slot, data] of pendingAttributes) adapter.setAttribute(slot, data);
      pendingUniforms.clear();
      pendingAttributes.clear();
    },

    setUniform(slot, value) {
      if (!device || !uniformScratch || !uniformLayout) {
        pendingUniforms.set(slot, value);
        return;
      }
      let member = uniformLayout.members.find((m) => m.name === slot);
      if (!member) throw new Error(`[RMSL] unknown uniform "${slot}"`);
      let offset = member.offset / 4;
      if (Array.isArray(value)) value.forEach((v, i) => (uniformScratch![offset + i] = v));
      else uniformScratch[offset] = value;
      device.queue.writeBuffer(uniformBuffer!, 0, uniformScratch as BufferSource);
    },

    setAttribute(slot, data) {
      if (!device) {
        pendingAttributes.set(slot, data);
        return;
      }
      if (data.length !== n) {
        n = data.length;
        rebuildStorageBuffers();
      }
      let buf = inBuffers.get(slot);
      if (!buf) throw new Error(`[RMSL] unknown attribute "${slot}"`);
      device.queue.writeBuffer(buf, 0, data as BufferSource);
    },

    async compute(out) {
      if (!device || !pipeline || !bindGroup1) {
        throw new Error("[RMSL] adapter not attached — call attach() before compute()");
      }
      let encoder = device.createCommandEncoder();
      let pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      if (bindGroup0) pass.setBindGroup(0, bindGroup0);
      pass.setBindGroup(1, bindGroup1);
      pass.dispatchWorkgroups(Math.max(1, Math.ceil(n / 64)));
      pass.end();

      let bytes = Math.max(4, n * 4);
      let regions: { slot: string; byteOffset: number }[] = [];
      let byteOffset = 0;
      for (let outInfo of reflection.outputs) {
        encoder.copyBufferToBuffer(outBuffers.get(outInfo.slot)!, 0, staging!, byteOffset, bytes);
        regions.push({ slot: outInfo.slot, byteOffset });
        byteOffset += bytes;
      }
      device.queue.submit([encoder.finish()]);

      await staging!.mapAsync(GPUMapMode.READ);
      let mapped = new Float32Array(staging!.getMappedRange());
      for (let { slot, byteOffset } of regions) {
        let start = byteOffset / 4;
        (out[slot] as Float32Array).set(mapped.subarray(start, start + n));
      }
      staging!.unmap();
      return out;
    },

    destroy() {
      for (let buf of inBuffers.values()) buf.destroy();
      for (let buf of outBuffers.values()) buf.destroy();
      uniformBuffer?.destroy();
      staging?.destroy();
      device?.destroy();
    },
  };

  return adapter;
}
