import { u_rowOffset } from "./mandelbrotShader";

const COMPONENT_COUNT = 4; // vec4 color

/**
 * Splits one Mandelbrot frame's rows across `numWorkers` wasm instances, one
 * per worker, each with its own private WebAssembly.Memory. Every worker
 * draws its own row range and transfers just that slice back over
 * `postMessage`; this side copies each slice into the right offset of one
 * assembled buffer.
 *
 * Not zero-copy end to end — sharing one memory across concurrently-running
 * instances of the same compiled module isn't safe here: fragCoord and the
 * uniform values are read/written at fixed addresses baked into the
 * module, overwritten on every pixel, so two instances racing on one
 * shared memory scramble each other's pixels rather than just risking a
 * stale read. A private memory per worker avoids that; only the (much
 * smaller) row-slice result crosses the postMessage boundary, and that
 * crossing is a transfer (ownership move), not a structured-clone copy.
 */
export class WasmWorkerPool {
  private workers: Worker[] = [];
  private nextId = 0;
  private assembled: Float64Array | null = null;
  private width = 0;
  private height = 0;

  constructor(private numWorkers: number) {}

  get workerCount(): number {
    return this.workers.length;
  }

  private async ensureWorkers(): Promise<void> {
    if (this.workers.length > 0) return;

    const ready: Promise<void>[] = [];
    for (let i = 0; i < this.numWorkers; i++) {
      const worker = new Worker(new URL("./mandelbrotWasmWorker.ts", import.meta.url), { type: "module" });
      this.workers.push(worker);
      ready.push(
        new Promise<void>((resolve) => {
          const onMessage = (e: MessageEvent) => {
            if (e.data?.type === "ready") {
              worker.removeEventListener("message", onMessage);
              resolve();
            }
          };
          worker.addEventListener("message", onMessage);
        }),
      );
      worker.postMessage({ type: "init" });
    }
    await Promise.all(ready);
  }

  /** Renders one frame and returns the assembled result buffer (reused across calls — copy out before the next render() if you need to keep it). */
  async render(uniforms: Record<string, number | number[]>, width: number, height: number): Promise<Float64Array> {
    await this.ensureWorkers();

    if (!this.assembled || this.width !== width || this.height !== height) {
      this.assembled = new Float64Array(width * height * COMPONENT_COUNT);
      this.width = width;
      this.height = height;
    }
    const assembled = this.assembled;

    const rowsPerWorker = Math.ceil(height / this.workers.length);
    const jobs: Promise<void>[] = [];

    for (let i = 0; i < this.workers.length; i++) {
      const rowStart = i * rowsPerWorker;
      const rowCount = Math.max(0, Math.min(rowsPerWorker, height - rowStart));
      if (rowCount <= 0) continue;

      const worker = this.workers[i];
      const id = this.nextId++;
      const destOffset = rowStart * width * COMPONENT_COUNT;

      jobs.push(
        new Promise<void>((resolve) => {
          const onMessage = (e: MessageEvent) => {
            if (e.data?.type === "done" && e.data.id === id) {
              worker.removeEventListener("message", onMessage);
              assembled.set(new Float64Array(e.data.buffer), destOffset);
              resolve();
            }
          };
          worker.addEventListener("message", onMessage);
          worker.postMessage({
            type: "draw",
            id,
            uniforms: { ...uniforms, [u_rowOffset.name]: rowStart },
            width,
            rowCount,
          });
        }),
      );
    }

    await Promise.all(jobs);
    return assembled;
  }

  terminate(): void {
    for (const w of this.workers) w.terminate();
    this.workers = [];
    this.assembled = null;
  }
}
