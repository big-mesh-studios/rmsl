interface Position {
  x: Float32Array;
  y: Float32Array;
}

interface Velocity {
  x: Float32Array;
  y: Float32Array;
}

interface Chunk {
  count: number;
  Position: Position;
  Velocity: Velocity;
}

function alignTo256(size: number): number {
  return Math.ceil(size / 256) * 256;
}

export class World {
  readonly chunk: Chunk;

  constructor(count: number) {
    this.chunk = {
      count,

      Position: {
        x: new Float32Array(count),
        y: new Float32Array(count),
      },

      Velocity: {
        x: new Float32Array(count),
        y: new Float32Array(count),
      },
    };
  }
}

// ============================================================================
// Persistent GPU component storage
// ============================================================================

export interface GpuColumn {
  readonly name: string;
  readonly buffer: GPUBuffer;
  readonly count: number;
}

export class GpuWorld {
  private readonly columns = new Map<string, GpuColumn>();

  constructor(
    readonly device: GPUDevice,
    readonly queue: GPUQueue,
  ) {}

  createColumn(name: string, data: Float32Array): GpuColumn {
    const buffer = this.device.createBuffer({
      label: `rmsl-oecs:${name}`,

      size: alignTo256(data.byteLength),

      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    this.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);

    const column: GpuColumn = {
      name,
      buffer,
      count: data.length,
    };

    this.columns.set(name, column);

    return column;
  }

  getColumn(name: string): GpuColumn {
    const column = this.columns.get(name);

    if (!column) {
      throw new Error(`Unknown GPU column: ${name}`);
    }

    return column;
  }

  /**
   * Read a GPU column back.
   *
   * This is ONLY for verification.
   * It is not part of the simulation loop.
   */
  async readColumn(name: string): Promise<Float32Array> {
    const column = this.getColumn(name);

    const size = column.count * Float32Array.BYTES_PER_ELEMENT;

    const staging = this.device.createBuffer({
      label: `readback:${name}`,

      size: alignTo256(size),

      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = this.device.createCommandEncoder({
      label: `readback:${name}`,
    });

    encoder.copyBufferToBuffer(
      column.buffer,
      0,

      staging,
      0,

      size,
    );

    this.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);

    const mapped = staging.getMappedRange();

    const result = new Float32Array(column.count);

    /*
     * Only copy the real component data.
     *
     * The GPU buffer may have padding because WebGPU buffer sizes need
     * to satisfy alignment requirements.
     */
    result.set(new Float32Array(mapped, 0, column.count));

    staging.unmap();
    staging.destroy();

    return result;
  }

  destroy(): void {
    for (const column of this.columns.values()) {
      column.buffer.destroy();
    }

    this.columns.clear();
  }
}
