declare const sampleRate: number;
declare function registerProcessor(
  name: string,
  ctor: new (options?: unknown) => {
    process(inputs: Float32Array[][], outputs: Float32Array[][], parameters: Record<string, Float32Array>): boolean;
  },
): void;

declare class AudioWorkletProcessor {
  readonly port: MessagePort;
  constructor(options?: unknown);
}
