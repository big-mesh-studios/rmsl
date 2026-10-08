import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { Fn, float, instanceIndex, instancedArray, varying, vec4, vertexIndex } from "../../rmsl";
import { compileGlsl } from "../../glsl";
import { compileWgsl, createWgslContext } from "../../wgsl";
import { GPU_ENABLED, installWebGpuGlobals } from "../../testing/gpu";
import { storageNodesOf } from "../shared";
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
   * @canon spec-a-wgsl-render-stage-reads-storage-read-only
   * @canon spec-the-index-accessors-follow-tsl
   */
  it("are declared read-only in their own group, numbered across both stages", () => {
    const { offsets, colors, vertex, fragment } = program();
    const storages = [offsets, colors];
    const vertexCode = compileWgsl.vertex(vertex, { storages });
    const fragmentCode = compileWgsl.fragment(fragment, { storages });
    const binding = (name: string) =>
      `@group(${WGSL_RENDER_STORAGE_GROUP}) @binding(${storages.findIndex((node) => node.name === name)}) var<storage, read>`;
    expect(vertexCode).toContain(binding(offsets.name));
    expect(vertexCode).not.toContain(binding(colors.name));
    expect(fragmentCode).toContain(binding(colors.name));
    expect(vertexCode).toContain("@builtin(vertex_index) _rmsl_vertexIndex: u32");
    expect(vertexCode).toContain("@builtin(instance_index) _rmsl_instanceIndex: u32");
  });
  /**
   * @canon spec-a-wgsl-render-stage-reads-storage-read-only
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
   * @canon spec-a-glsl-stage-declares-a-storage-buffer-as-a-sampler-under-its-slot
   */
  it("reads storage on GLSL through a sampler named after the buffer's slot", () => {
    const { offsets, colors, vertex, fragment } = program();
    const vertexCode = compileGlsl.vertex(vertex);
    expect(vertexCode).toContain(`uniform sampler2D ${offsets.name};`);
    expect(vertexCode).toMatch(new RegExp(`texelFetch\\(${offsets.name}, ivec2\\(`));
    expect(compileGlsl.fragment(fragment)).toContain(`uniform sampler2D ${colors.name};`);
  });
  /**
   * @canon spec-a-glsl-render-stage-refuses-a-write-to-storage
   */
  it("rejects a write from a render stage on GLSL", () => {
    const values = instancedArray(4, "float");
    const vertex = Fn(() => {
      values.element(0).assign(float(1));
      return vec4(0, 0, 0, 1);
    })();
    expect(() => compileGlsl.vertex(vertex)).toThrow(/read-only in a vertex shader/);
  });
  /**
   * @canon spec-the-index-accessors-follow-tsl
   */
  it("maps the index builtins on GLSL", () => {
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
    const storages = storageNodesOf([vertex, fragment]);

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
      entries: storages.map((node, binding) => ({
        binding,
        resource: { buffer: context.buffer(node.attribute) },
      })),
    });
    const error = await device.popErrorScope();

    expect(error?.message).toBeUndefined();
    context.destroy();
  });
});
