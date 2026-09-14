// Minimal ambient types for the AudioWorkletGlobalScope — TypeScript's DOM
// lib doesn't ship these, and pulling in the full "audioworklet" lib just
// for one small processor file isn't worth the config churn.
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
