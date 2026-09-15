// Reads the WGSL adapter's own storage buffers directly as vertex data —
// the compute pass and this render pass share the same GPUBuffer objects,
// so drawing never round-trips position data through the CPU.
interface GpuRenderer {
  render(count: number, width: number, height: number): void;
  destroy(): void;
}

export function createGpuRenderer(
  device: GPUDevice,
  canvas: HTMLCanvasElement,
  positionX: GPUBuffer,
  positionY: GPUBuffer,
): GpuRenderer {
  const context = canvas.getContext("webgpu");
  if (!context) {
    throw new Error("WebGPU canvas context unavailable");
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });

  const shader = device.createShaderModule({
    label: "rmsl:ecs:gpu-renderer",
    code: /* wgsl */ `
      struct Viewport {
        width: f32,
        height: f32,
      };

      @group(0) @binding(0) var<uniform> viewport: Viewport;
      @group(0) @binding(1) var<storage, read> position_x: array<f32>;
      @group(0) @binding(2) var<storage, read> position_y: array<f32>;

      struct VertexOut {
        @builtin(position) position: vec4<f32>,
      };

      const QUAD = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),

        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
      );

      @vertex
      fn vs(
        @builtin(vertex_index) vertex_index: u32,
        @builtin(instance_index) instance_index: u32,
      ) -> VertexOut {
        let px = position_x[instance_index];
        let py = position_y[instance_index];

        let ndc = vec2<f32>(
          (px / viewport.width) * 2.0 - 1.0,
          1.0 - (py / viewport.height) * 2.0,
        );

        // QUAD spans -1..1, so this scale gives a 2x2px quad — matching the
        // CPU renderers' fillRect(posX - 1, posY - 1, 2, 2).
        let local = QUAD[vertex_index] * vec2<f32>(2.0 / viewport.width, 2.0 / viewport.height);

        var out: VertexOut;
        out.position = vec4<f32>(ndc + local, 0.0, 1.0);
        return out;
      }

      @fragment
      fn fs() -> @location(0) vec4<f32> {
        return vec4<f32>(0.5, 0.83, 1.0, 1.0);
      }
    `,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    label: "rmsl:ecs:gpu-renderer:bind-layout",
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ],
  });

  const viewportBuffer = device.createBuffer({
    label: "rmsl:ecs:gpu-renderer:viewport",
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bindGroup = device.createBindGroup({
    label: "rmsl:ecs:gpu-renderer:bind-group",
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: viewportBuffer } },
      { binding: 1, resource: { buffer: positionX } },
      { binding: 2, resource: { buffer: positionY } },
    ],
  });

  const pipeline = device.createRenderPipeline({
    label: "rmsl:ecs:gpu-renderer:pipeline",
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: { module: shader, entryPoint: "vs" },
    fragment: { module: shader, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  return {
    render(count: number, width: number, height: number): void {
      device.queue.writeBuffer(viewportBuffer, 0, new Float32Array([width, height]));

      const encoder = device.createCommandEncoder({ label: "rmsl:ecs:gpu-render" });
      const view = context.getCurrentTexture().createView();

      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view,
            clearValue: { r: 0.06, g: 0.07, b: 0.09, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });

      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(6, count);
      pass.end();

      device.queue.submit([encoder.finish()]);
    },

    destroy(): void {
      viewportBuffer.destroy();
      context.unconfigure();
    },
  };
}
