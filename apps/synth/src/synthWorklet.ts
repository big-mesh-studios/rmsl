import { compileWasm } from "@random-mesh/rmsl";
import { synthCpu, u_freq, u_gain, u_sampleRate, u_startPhase, u_waveform } from "./synthShader";

// AudioWorkletGlobalScope doesn't expose TextEncoder in every browser (it's
// not one of the APIs the spec guarantees there) — the WASM backend's
// codegen uses one to encode names/strings into the compiled binary. Ascii
// is all it ever needs to encode here (shader-generated slot/function
// names), so a minimal one-byte-per-char encoder is enough.
if (typeof TextEncoder === "undefined") {
  (globalThis as { TextEncoder?: unknown }).TextEncoder = class {
    encode(str: string): Uint8Array {
      const bytes = new Uint8Array(str.length);
      for (let i = 0; i < str.length; i++) bytes[i] = str.charCodeAt(i) & 0x7f;
      return bytes;
    }
  };
}

const WAVEFORMS: Record<string, number> = { sine: 0, saw: 1, square: 2, triangle: 3 };

type PortMsg =
  | { type: "freq"; value: number }
  | { type: "waveform"; value: keyof typeof WAVEFORMS }
  | { type: "gain"; value: number }
  | { type: "play"; value: boolean };

/**
 * Runs the RMSL-compiled oscillator on the audio rendering thread. Each
 * `process()` call is one ~128-sample block with a hard real-time deadline
 * (miss it and the browser drops out audibly) — `renderer.draw()` computes
 * the whole block in one call into a synchronous WASM loop, the same
 * "compute the whole buffer, not sample-by-sample JS calls" trick the CPU
 * renderer backends use for pixels.
 *
 * The oscillator's only state — its phase — lives here on the host, not in
 * the shader: each block advances `phase` by `blockLength * freq /
 * sampleRate` and feeds that back in as `u_startPhase` next time.
 */
class RmslOscillatorProcessor extends AudioWorkletProcessor {
  private renderer = compileWasm(() => synthCpu(), { name: "synthWasm", params: [] });
  private phase = 0;
  private freq = 440;
  private waveform = 0;
  private gain = 0.2;
  private playing = false;

  constructor(options?: unknown) {
    super(options);
    this.port.onmessage = (e: MessageEvent<PortMsg>) => {
      const msg = e.data;
      if (msg.type === "freq") this.freq = msg.value;
      else if (msg.type === "waveform") this.waveform = WAVEFORMS[msg.value] ?? 0;
      else if (msg.type === "gain") this.gain = msg.value;
      else if (msg.type === "play") this.playing = msg.value;
    };
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const output = outputs[0];
    const channel = output[0];
    const n = channel.length;

    if (!this.playing) {
      channel.fill(0);
      for (let c = 1; c < output.length; c++) output[c].set(channel);
      return true;
    }

    const buffer = this.renderer.draw(
      {
        uniforms: {
          [u_freq.name]: this.freq,
          [u_startPhase.name]: this.phase,
          [u_sampleRate.name]: sampleRate,
          [u_waveform.name]: this.waveform,
          [u_gain.name]: this.gain,
        },
      },
      n,
      1,
    );

    for (let i = 0; i < n; i++) channel[i] = buffer[i] as number;
    for (let c = 1; c < output.length; c++) output[c].set(channel);

    this.phase = (this.phase + (n * this.freq) / sampleRate) % 1;
    return true;
  }
}

registerProcessor("rmsl-oscillator", RmslOscillatorProcessor);
