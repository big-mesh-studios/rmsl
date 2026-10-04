import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { Fn, float, instanceIndex, instancedArray, varying, vec4, vertexIndex } from "../../rmsl";
import { compileGlsl } from "../../glsl";
import { compileWgsl, createWgslContext } from "../../wgsl";
import { GPU_ENABLED, installWebGpuGlobals } from "../../testing/gpu";
import { storageAttributes } from "../shared";
import { WGSL_RENDER_STORAGE_GROUP } from "./wgsl";

let uninstall: (() => void) | undefined;
beforeAll(async () => {
  if (GPU_ENABLED) uninstall = await installWebGpuGlobals();
});
afterAll(() => uninstall?.());

/**
 * A vertex stage placing each instance from one storage buffer, and a
 * fragment stage colouring it from another: the two read different buffers,
 * which is what program-wide binding numbers exist for.
 */
function program() {
  const offsets = instancedArray(4, "float");
  const colors = instancedArray(4, "float");
  const shade = varying("float");
  const vertex = Fn(() => {
    const corner = vertexIndex().toFloat().mul(0.1);
    shade.assign(instanceIndex().toFloat());
    return vec4(offsets.element(instanceIndex()).add(corner), 0, 0, 1);
  })();
  const fragment = Fn(() => vec4(colors.element(shade.toUint()), 0, 0, 1))();
  return { offsets, colors, vertex, fragment };
}

describe("storage buffers in render stages", () => {
  /**
   * @canon spec-a-render-stage-reads-storage-read-only
   * @canon spec-the-index-accessors-follow-tsl
   */
  it("are declared read-only in their own group, numbered across both stages", () => {
    const { offsets, colors, vertex, fragment } = program();
    const storages = [offsets.name, colors.name].sort();
    const vertexCode = compileWgsl.vertex(vertex, { storages });
    const fragmentCode = compileWgsl.fragment(fragment, { storages });
    const binding = (name: string) =>
      `@group(${WGSL_RENDER_STORAGE_GROUP}) @binding(${storages.indexOf(name)}) var<storage, read>`;
    expect(vertexCode).toContain(binding(offsets.name));
    expect(vertexCode).not.toContain(binding(colors.name));
    expect(fragmentCode).toContain(binding(colors.name));
    expect(vertexCode).toContain("@builtin(vertex_index) _rmsl_vertexIndex: u32");
    expect(vertexCode).toContain("@builtin(instance_index) _rmsl_instanceIndex: u32");
  });
  /**
   * @canon spec-a-render-stage-reads-storage-read-only
   */
  it("rejects a write from a render stage", () => {
    const values = instancedArray(4, "float");
    const vertex = Fn(() => {
      values.element(0).assign(float(1));
      return vec4(0, 0, 0, 1);
    })();
    expect(() => compileWgsl.vertex(vertex)).toThrow(/read-only in a vertex shader/);
  });
  /**
   * @canon exception-glsl-has-no-storage-buffers
   * @canon spec-the-index-accessors-follow-tsl
   */
  it("reports that GLSL has no storage buffers, and maps the index builtins", () => {
    const { vertex } = program();
    expect(() => compileGlsl.vertex(vertex)).toThrow(/GLSL has no storage buffers/);
    const indices = Fn(() => vec4(vertexIndex().toFloat(), instanceIndex().toFloat(), 0, 1))();
    const glsl = compileGlsl.vertex(indices);
    expect(glsl).toContain("uint(gl_VertexID)");
    expect(glsl).toContain("uint(gl_InstanceID)");
  });

  /**
   * @canon spec-a-wgsl-buffer-feeds-a-draw-without-a-copy
   */
  it.skipIf(!GPU_ENABLED)("build a render pipeline bound to a compute context's buffers", async () => {
    const context = await createWgslContext();
    const device = context.device;
    const { vertex, fragment } = program();
    const attributes = storageAttributes([vertex, fragment]);
    const storages = [...attributes.keys()].sort();

    device.pushErrorScope("validation");
    const pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: device.createShaderModule({ code: compileWgsl.vertex(vertex, { storages }) }),
        entryPoint: "main",
      },
      fragment: {
        module: device.createShaderModule({ code: compileWgsl.fragment(fragment, { storages }) }),
        entryPoint: "main",
        targets: [{ format: "rgba8unorm" }],
      },
    });
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(WGSL_RENDER_STORAGE_GROUP),
      entries: storages.map((slot, binding) => ({
        binding,
        resource: { buffer: context.buffer(attributes.get(slot)!) },
      })),
    });
    const error = await device.popErrorScope();

    expect(error?.message).toBeUndefined();
    context.destroy();
  });
});
