import { attribute, float, Fn, output, vec4 } from "@random-mesh/rmsl";
import { oscillatorSample, u_freq, u_gain, u_sampleRate, u_startPhase, u_waveform } from "./synthShader";

// How many points trace one on-screen sweep of the waveform. Not the audio
// sample rate — main.ts sets u_sampleRate to a much lower "visual" rate
// when drawing (see VISUAL_WINDOW_SECONDS there), chosen so a fixed window
// of vertices shows a few cycles across the whole frequency range rather
// than a near-flat line at low frequencies or an aliased mess at high ones.
export const SCOPE_VERTEX_COUNT = 512;

// A static 0..SCOPE_VERTEX_COUNT-1 buffer — this vertex's position in the
// sweep, fed to the exact same `oscillatorSample` the WASM audio path
// calls per PCM sample (see synthShader.ts), just evaluated on the GPU at
// a different point density.
export let a_index = attribute("float");

export let scopeVertex = Fn(() => {
  let sample = oscillatorSample(a_index, u_startPhase, u_freq, u_sampleRate, u_waveform, u_gain).toVar();
  let x = a_index.div(float(SCOPE_VERTEX_COUNT - 1)).mul(2.0).sub(1.0);
  return vec4(x, sample, 0.0, 1.0);
});

export let scopeFragment = Fn(() => {
  let outColor = output("vec4");
  outColor.assign(vec4(0.22, 0.74, 0.97, 1.0)); // matches the CSS --accent color
  return outColor;
});
