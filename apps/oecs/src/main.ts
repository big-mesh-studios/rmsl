/**
 * persistent-oecs-rmsl.ts
 *
 * Experimental vertical slice:
 *
 *   oecs-like SoA world
 *          ↓
 *   persistent GPU component buffers
 *          ↓
 *   RMSL compute program
 *          ↓
 *   WebGPU dispatch
 *          ↓
 *   same GPU buffers
 *
 * CPU component data is uploaded once.
 * The simulation then runs entirely against persistent GPU buffers.
 *
 * Target:
 *
 *   Position += Velocity * dt
 */

import { Fn, attributeRaw, compileWGSL, output, uniformRaw } from "@random-mesh/rmsl";
import { createRenderer } from "./renderer";
import { GpuColumn, GpuWorld, World } from "./rmsl-oecs";

function assertNear(actual: number, expected: number, epsilon: number, label: string): void {
  if (Math.abs(actual - expected) > epsilon) {
    throw new Error(`${label}: expected ${expected}, got ${actual}`);
  }
}

/**
 * Position += Velocity * dt
 *
 * RMSL turns this into approximately:
 *
 *   @group(0) @binding(0)
 *   var<uniform> _rmsl_uniforms
 *
 *   @group(1) @binding(0)
 *   velocity_x
 *
 *   @group(1) @binding(1)
 *   velocity_y
 *
 *   @group(1) @binding(2)
 *   _rmsl_o0
 *
 *   @group(1) @binding(3)
 *   _rmsl_o1
 */
function createMovementProgram() {
  const velocityX = attributeRaw("velocity_x", "float");

  const velocityY = attributeRaw("velocity_y", "float");

  const positionX = output("float");

  const positionY = output("float");

  const dt = uniformRaw("dt", "float");

  return Fn(() => {
    positionX.addAssign(velocityX.mul(dt));

    positionY.addAssign(velocityY.mul(dt));

    /*
     * Fn() needs a root node.
     *
     * The useful work is represented by the statements above.
     */
    return positionX;
  });
}

// ============================================================================
// RMSL → WebGPU pipeline
// ============================================================================

interface MovementKernel {
  readonly pipeline: GPUComputePipeline;

  readonly uniformBindGroup: GPUBindGroup;
  readonly storageBindGroup: GPUBindGroup;

  readonly dtBuffer: GPUBuffer;
}

function createMovementKernel(
  device: GPUDevice,

  positionX: GpuColumn,
  positionY: GpuColumn,

  velocityX: GpuColumn,
  velocityY: GpuColumn,
): MovementKernel {
  // --------------------------------------------------------------------------
  // Compile RMSL
  // --------------------------------------------------------------------------

  const program = createMovementProgram();

  const wgsl = compileWGSL.compute(program());

  console.log("Generated WGSL:\n", wgsl);

  const shaderModule = device.createShaderModule({
    label: "rmsl:oecs:movement",
    code: wgsl,
  });

  // --------------------------------------------------------------------------
  // GROUP 0
  //
  // RMSL currently emits:
  //
  //   @group(0) @binding(0)
  //   var<uniform> _rmsl_uniforms
  // --------------------------------------------------------------------------

  const uniformLayout = device.createBindGroupLayout({
    label: "rmsl:oecs:uniforms",

    entries: [
      {
        binding: 0,

        visibility: GPUShaderStage.COMPUTE,

        buffer: {
          type: "uniform",
        },
      },
    ],
  });

  // --------------------------------------------------------------------------
  // GROUP 1
  //
  // RMSL currently emits:
  //
  //   binding 0 → attribute
  //   binding 1 → attribute
  //   binding 2 → output
  //   binding 3 → output
  // --------------------------------------------------------------------------

  const storageLayout = device.createBindGroupLayout({
    label: "rmsl:oecs:movement:storage",

    entries: [
      {
        binding: 0,

        visibility: GPUShaderStage.COMPUTE,

        buffer: {
          type: "read-only-storage",
        },
      },

      {
        binding: 1,

        visibility: GPUShaderStage.COMPUTE,

        buffer: {
          type: "read-only-storage",
        },
      },

      {
        binding: 2,

        visibility: GPUShaderStage.COMPUTE,

        buffer: {
          type: "storage",
        },
      },

      {
        binding: 3,

        visibility: GPUShaderStage.COMPUTE,

        buffer: {
          type: "storage",
        },
      },
    ],
  });

  // --------------------------------------------------------------------------
  // Pipeline
  // --------------------------------------------------------------------------

  const pipelineLayout = device.createPipelineLayout({
    label: "rmsl:oecs:movement:layout",

    bindGroupLayouts: [uniformLayout, storageLayout],
  });

  const pipeline = device.createComputePipeline({
    label: "rmsl:oecs:movement:pipeline",

    layout: pipelineLayout,

    compute: {
      module: shaderModule,
      entryPoint: "main",
    },
  });

  // --------------------------------------------------------------------------
  // dt uniform
  // --------------------------------------------------------------------------

  const dtBuffer = device.createBuffer({
    label: "rmsl:oecs:dt",

    /*
     * RMSL's uniform struct currently contains:
     *
     *   dt: f32
     *
     * 16 bytes keeps us safely inside WebGPU's uniform-buffer alignment
     * requirements.
     */
    size: 16,

    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const uniformBindGroup = device.createBindGroup({
    label: "rmsl:oecs:movement:uniforms",

    layout: uniformLayout,

    entries: [
      {
        binding: 0,

        resource: {
          buffer: dtBuffer,
        },
      },
    ],
  });

  // --------------------------------------------------------------------------
  // Persistent component buffers
  // --------------------------------------------------------------------------

  const storageBindGroup = device.createBindGroup({
    label: "rmsl:oecs:movement:storage",

    layout: storageLayout,

    entries: [
      /*
       * @group(1) @binding(0)
       *
       * velocity_x
       */
      {
        binding: 0,

        resource: {
          buffer: velocityX.buffer,
        },
      },

      /*
       * @group(1) @binding(1)
       *
       * velocity_y
       */
      {
        binding: 1,

        resource: {
          buffer: velocityY.buffer,
        },
      },

      /*
       * @group(1) @binding(2)
       *
       * _rmsl_o0 → Position.x
       */
      {
        binding: 2,

        resource: {
          buffer: positionX.buffer,
        },
      },

      /*
       * @group(1) @binding(3)
       *
       * _rmsl_o1 → Position.y
       */
      {
        binding: 3,

        resource: {
          buffer: positionY.buffer,
        },
      },
    ],
  });

  return {
    pipeline,

    uniformBindGroup,
    storageBindGroup,

    dtBuffer,
  };
}

// ============================================================================
// Dispatch
// ============================================================================

function dispatchMovement(
  device: GPUDevice,
  queue: GPUQueue,

  kernel: MovementKernel,

  count: number,
  dt: number,
): void {
  // --------------------------------------------------------------------------
  // Update frame uniform.
  //
  // This is tiny CPU → GPU traffic:
  //
  //     4 bytes
  //
  // The component state itself is NOT copied.
  // --------------------------------------------------------------------------

  queue.writeBuffer(kernel.dtBuffer, 0, new Float32Array([dt]));

  // --------------------------------------------------------------------------
  // Encode compute pass
  // --------------------------------------------------------------------------

  const encoder = device.createCommandEncoder({
    label: "rmsl:oecs:movement",
  });

  const pass = encoder.beginComputePass({
    label: "rmsl:oecs:movement",
  });

  pass.setPipeline(kernel.pipeline);

  // @group(0)
  pass.setBindGroup(0, kernel.uniformBindGroup);

  // @group(1)
  pass.setBindGroup(1, kernel.storageBindGroup);

  /*
   * RMSL emits:
   *
   *   @workgroup_size(64)
   *
   * Therefore:
   *
   *   workgroups = ceil(entityCount / 64)
   */
  pass.dispatchWorkgroups(Math.ceil(count / 64));

  pass.end();

  queue.submit([encoder.finish()]);
}

async function main(): Promise<void> {
  if (!navigator.gpu) {
    throw new Error("WebGPU is required");
  }

  const adapter = await navigator.gpu.requestAdapter();

  if (!adapter) {
    throw new Error("No WebGPU adapter available");
  }

  const device = await adapter.requestDevice();

  // --------------------------------------------------------------------------
  // Canvas
  // --------------------------------------------------------------------------

  const canvas = document.querySelector("#canvas") as HTMLCanvasElement | null;

  if (!canvas) {
    throw new Error("Missing #canvas");
  }

  canvas.width = 1280;
  canvas.height = 720;

  // --------------------------------------------------------------------------
  // ECS world
  // --------------------------------------------------------------------------

  const entityCount = 100_000;

  const world = new World(entityCount);

  for (let i = 0; i < entityCount; i++) {
    world.chunk.Velocity.x[i] = 1;
    world.chunk.Velocity.y[i] = 2;

    // Start in a visible area.
    world.chunk.Position.x[i] = Math.random() * 100 - 50;

    world.chunk.Position.y[i] = Math.random() * 100 - 50;
  }

  // --------------------------------------------------------------------------
  // Persistent GPU state
  // --------------------------------------------------------------------------

  const gpuWorld = new GpuWorld(device, device.queue);

  const positionX = gpuWorld.createColumn("Position.x", world.chunk.Position.x);

  const positionY = gpuWorld.createColumn("Position.y", world.chunk.Position.y);

  const velocityX = gpuWorld.createColumn("Velocity.x", world.chunk.Velocity.x);

  const velocityY = gpuWorld.createColumn("Velocity.y", world.chunk.Velocity.y);

  // --------------------------------------------------------------------------
  // Simulation kernel
  // --------------------------------------------------------------------------

  const kernel = createMovementKernel(device, positionX, positionY, velocityX, velocityY);

  // --------------------------------------------------------------------------
  // Renderer
  // --------------------------------------------------------------------------

  const renderer = createRenderer(device, canvas, positionX, positionY);

  // --------------------------------------------------------------------------
  // Main loop
  // --------------------------------------------------------------------------

  let lastTime = performance.now();

  function frame(now: number): void {
    const dt = Math.min((now - lastTime) / 1000, 1 / 30);

    lastTime = now;

    // GPU simulation.
    dispatchMovement(device, device.queue, kernel, entityCount, dt);

    // GPU rendering.
    //
    // IMPORTANT:
    //
    // positionX/positionY are the exact same buffers that the
    // compute shader just modified.
    //
    // There is no:
    //
    //     GPU -> CPU -> GPU
    //
    // round trip here.
    renderer.render(entityCount);

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch((error) => {
  console.error(error);
});
