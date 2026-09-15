// === WGSL compute adapter ===
// Wraps `compile()` (src/wgsl.ts) — which already reflects a program's
// storage/uniform resources off its node graph — into the uniform Adapter
// shape, so the ceremony apps/ecs's main.ts still hand-writes (buffer
// creation, bind groups, uniform packing, staged readback) only has to be
// written once, here, instead of at every WGSL compute call site.
import { Node, ShaderType } from "../core";
import { compile, WgslResource } from "../wgsl";
import { Adapter, TypedArray } from "./adapter";

/** One typed array per storage slot, keyed by name — read_write, so the same
 * record a caller passes into `compute` is the one read back out of. */
export type AdapterResult = Record<string, TypedArray>;

/** A WGSL adapter, plus the one thing the shared `Adapter` shape has no
 * generic name for: direct access to a storage slot's persistent GPU
 * buffer, so a `draw` pass (this app's own, or another adapter's) can bind
 * it without a readback ever happening. */
export interface WgslAdapter extends Adapter<AdapterResult> {
  buffer(slot: string): GPUBuffer | undefined;
  /** The device backing this adapter, once `attach()` has resolved — a
   * `draw` pass sharing its buffers has to build its own pipeline against
   * this same device, since a GPUBuffer is only valid on the device that
   * created it. */
  device(): GPUDevice | undefined;
}

export function createAdapter(root: Node<ShaderType> | readonly Node<ShaderType>[]): WgslAdapter {
  let device: GPUDevice | null = null;
  let pipeline: GPUComputePipeline | null = null;
  let resources: WgslResource[] = [];

  let n = 0;
  let storageBuffers = new Map<string, GPUBuffer>();
  let bindGroup1: GPUBindGroup | null = null;
  let uniformBuffer: GPUBuffer | null = null;
  let uniformScratch: Float32Array | null = null;
  let bindGroup0: GPUBindGroup | null = null;
  let staging: GPUBuffer | null = null;

  // setUniform/setAttribute may be called before attach() resolves, so
  // values that arrive early are queued and replayed once the device exists.
  let pendingUniforms = new Map<string, number | number[]>();
  let pendingAttributes = new Map<string, TypedArray>();

  function storageResources(): Extract<WgslResource, { kind: "storage" }>[] {
    return resources.filter((r): r is Extract<WgslResource, { kind: "storage" }> => r.kind === "storage");
  }
  function uniformResources(): Extract<WgslResource, { kind: "uniform" }>[] {
    return resources.filter((r): r is Extract<WgslResource, { kind: "uniform" }> => r.kind === "uniform");
  }

  function rebuildStorageBuffers() {
    if (!device || !pipeline) return;
    for (let buf of storageBuffers.values()) buf.destroy();
    staging?.destroy();

    let usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    let bytes = Math.max(4, n * 4);
    storageBuffers = new Map(storageResources().map((r) => [r.name, device!.createBuffer({ size: bytes, usage })]));

    bindGroup1 = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(1),
      entries: storageResources().map((r) => ({ binding: r.binding, resource: { buffer: storageBuffers.get(r.name)! } })),
    });

    staging = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  let adapter: WgslAdapter = {
    async attach() {
      let gpuAdapter = await navigator.gpu?.requestAdapter();
      if (!gpuAdapter) throw new Error("[RMSL] WebGPU is not available");
      device = await gpuAdapter.requestDevice();

      let program = compile({ stage: "compute", workgroupSize: 64 }, root);
      let module = device.createShaderModule({ code: program.code });
      pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: program.entryPoint } });
      resources = program.resources;

      let uniforms = uniformResources();
      if (uniforms.length > 0) {
        let size = Math.max(16, ...uniforms.map((u) => u.offset + u.size));
        uniformBuffer = device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        uniformScratch = new Float32Array(size / 4);
        bindGroup0 = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
        });
      }

      // Storage buffers are sized to the first upload, so a pending value has
      // to set `n` before the (empty) default-sized buffers are built.
      for (let [, data] of pendingAttributes) if (n === 0) n = data.length;
      rebuildStorageBuffers();
      for (let [slot, value] of pendingUniforms) adapter.setUniform(slot, value);
      for (let [slot, data] of pendingAttributes) adapter.setAttribute(slot, data);
      pendingUniforms.clear();
      pendingAttributes.clear();
    },

    setUniform(slot, value) {
      if (!device || !uniformScratch) {
        pendingUniforms.set(slot, value);
        return;
      }
      let res = uniformResources().find((r) => r.name === slot);
      if (!res) throw new Error(`[RMSL] unknown uniform "${slot}"`);
      let offset = res.offset / 4;
      if (Array.isArray(value)) value.forEach((v, i) => (uniformScratch![offset + i] = v));
      else uniformScratch[offset] = value;
      device.queue.writeBuffer(uniformBuffer!, 0, uniformScratch as BufferSource);
    },

    // Named `setAttribute` by the shared Adapter interface, but what it
    // uploads here is a storage() slot's current value — read_write, not
    // input-only, so the same slot comes back out of `compute`'s result.
    setAttribute(slot, data) {
      if (!device) {
        pendingAttributes.set(slot, data);
        return;
      }
      if (data.length !== n) {
        n = data.length;
        rebuildStorageBuffers();
      }
      let buf = storageBuffers.get(slot);
      if (!buf) throw new Error(`[RMSL] unknown storage "${slot}"`);
      device.queue.writeBuffer(buf, 0, data as BufferSource);
    },

    async compute(out) {
      if (!device || !pipeline || !bindGroup1 || !staging) {
        throw new Error("[RMSL] adapter not attached — call attach() before compute()");
      }
      let encoder = device.createCommandEncoder();
      let pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      if (bindGroup0) pass.setBindGroup(0, bindGroup0);
      pass.setBindGroup(1, bindGroup1);
      pass.dispatchWorkgroups(Math.max(1, Math.ceil(n / 64)));
      pass.end();
      device.queue.submit([encoder.finish()]);

      // No `out` means the caller means to keep the result GPU-resident —
      // read via `buffer(slot)` from a draw pass, never mapped back to the
      // CPU at all.
      if (!out) return;

      let bytes = Math.max(4, n * 4);
      for (let [slot, buffer] of storageBuffers) {
        let readEncoder = device.createCommandEncoder();
        readEncoder.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
        device.queue.submit([readEncoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        (out[slot] as Float32Array).set(new Float32Array(staging.getMappedRange()).subarray(0, n));
        staging.unmap();
      }
      return out;
    },

    buffer(slot) {
      return storageBuffers.get(slot);
    },

    device() {
      return device ?? undefined;
    },

    destroy() {
      for (let buf of storageBuffers.values()) buf.destroy();
      uniformBuffer?.destroy();
      staging?.destroy();
      device?.destroy();
    },
  };

  return adapter;
}
