// ============================================================================
// WebGPU renderer
// ============================================================================

import { GpuColumn } from "./rmsl-oecs";

interface Renderer {
  readonly context: GPUCanvasContext;
  readonly pipeline: GPURenderPipeline;
  readonly bindGroup: GPUBindGroup;

  render(count: number): void;
  destroy(): void;
}

export function createRenderer(
  device: GPUDevice,
  canvas: HTMLCanvasElement,
  positionX: GpuColumn,
  positionY: GpuColumn,
): Renderer {
  const context = canvas.getContext("webgpu");

  if (!context) {
    throw new Error("WebGPU canvas context unavailable");
  }

  const format = navigator.gpu.getPreferredCanvasFormat();

  context.configure({
    device,
    format,
    alphaMode: "opaque",
  });

  // --------------------------------------------------------------------------
  // Shader
  //
  // One instance = one entity.
  //
  // We generate a small quad around Position.x/y in the vertex shader.
  // No position data is copied back to the CPU.
  // --------------------------------------------------------------------------

  const shader = device.createShaderModule({
    label: "rmsl:oecs:renderer",
    code: /* wgsl */ `
      struct VertexOut {
        @builtin(position) position: vec4<f32>,
      };

      @group(0) @binding(0)
      var<storage, read> position_x: array<f32>;

      @group(0) @binding(1)
      var<storage, read> position_y: array<f32>;

      // Six vertices forming a quad.
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
        let center = vec2<f32>(
          position_x[instance_index],
          position_y[instance_index],
        );

        let local = QUAD[vertex_index];

        // World -> clip-space.
        //
        // For the example simulation, positions are in roughly:
        //   x = 0 .. 166
        //   y = 0 .. 333
        //
        // Adjust this scale/camera as desired.
        let scale = vec2<f32>(
          0.005,
          0.005
        );

        let p = center * scale + local * 0.008;

        var out: VertexOut;
        out.position = vec4<f32>(
          p.x,
          p.y,
          0.0,
          1.0
        );

        return out;
      }

      @fragment
      fn fs() -> @location(0) vec4<f32> {
        return vec4<f32>(
          0.2,
          0.8,
          1.0,
          1.0
        );
      }
    `,
  });

  // --------------------------------------------------------------------------
  // Bind positions
  // --------------------------------------------------------------------------

  const bindGroupLayout = device.createBindGroupLayout({
    label: "rmsl:oecs:renderer:bind-layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: {
          type: "read-only-storage",
        },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX,
        buffer: {
          type: "read-only-storage",
        },
      },
    ],
  });

  const bindGroup = device.createBindGroup({
    label: "rmsl:oecs:renderer:bind-group",
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: positionX.buffer,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: positionY.buffer,
        },
      },
    ],
  });

  // --------------------------------------------------------------------------
  // Pipeline
  // --------------------------------------------------------------------------

  const pipelineLayout = device.createPipelineLayout({
    label: "rmsl:oecs:renderer:layout",
    bindGroupLayouts: [bindGroupLayout],
  });

  const pipeline = device.createRenderPipeline({
    label: "rmsl:oecs:renderer:pipeline",

    layout: pipelineLayout,

    vertex: {
      module: shader,
      entryPoint: "vs",
    },

    fragment: {
      module: shader,
      entryPoint: "fs",
      targets: [
        {
          format,
        },
      ],
    },

    primitive: {
      topology: "triangle-list",
    },
  });

  // --------------------------------------------------------------------------
  // Renderer
  // --------------------------------------------------------------------------

  return {
    context,
    pipeline,
    bindGroup,

    render(count: number): void {
      const encoder = device.createCommandEncoder({
        label: "rmsl:oecs:render",
      });

      const view = context.getCurrentTexture().createView();

      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view,

            clearValue: {
              r: 0.02,
              g: 0.02,
              b: 0.03,
              a: 1,
            },

            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });

      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);

      // 6 vertices per quad, one instance per entity.
      pass.draw(6, count);

      pass.end();

      device.queue.submit([encoder.finish()]);
    },

    destroy(): void {
      // Nothing owned here that needs explicit destruction.
      context.unconfigure();
    },
  };
}
