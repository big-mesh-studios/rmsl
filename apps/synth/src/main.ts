import { compileGLSL } from "@random-mesh/rmsl";
import { a_index, scopeFragment, scopeVertex, SCOPE_VERTEX_COUNT } from "./scopeShader";
import { u_freq, u_sampleRate, u_startPhase, u_waveform } from "./synthShader";

const WAVEFORM_INDEX: Record<string, number> = { sine: 0, saw: 1, square: 2, triangle: 3 };

const playBtn = document.getElementById("playBtn") as HTMLButtonElement;
const freqInput = document.getElementById("freqInput") as HTMLInputElement;
const freqVal = document.getElementById("freqVal") as HTMLElement;
const gainInput = document.getElementById("gainInput") as HTMLInputElement;
const gainVal = document.getElementById("gainVal") as HTMLElement;
const waveformSelect = document.getElementById("waveformSelect") as HTMLSelectElement;
const statusEl = document.getElementById("status") as HTMLElement;

// One sweep of SCOPE_VERTEX_COUNT vertices spans this many seconds of the
// waveform — deliberately not the real audio sample rate (44100+ samples
// would barely move across one screen-width sweep for a 55Hz tone). This
// is the "resolution chosen for the eye" mentioned in synthShader.ts: same
// oscillatorSample() math, evaluated at a much lower point density.
const VISUAL_WINDOW_SECONDS = 0.02;
const VISUAL_SAMPLE_RATE = SCOPE_VERTEX_COUNT / VISUAL_WINDOW_SECONDS;

// scopeVertex/scopeFragment (scopeShader.ts) call the exact same
// oscillatorSample() the WASM audio path calls per PCM sample — this GPU
// program isn't fed the audio at all, it's a second, independent
// evaluation of the same graph.
const scopeGl = document.getElementById("scopeGl") as HTMLCanvasElement;
const gl = scopeGl.getContext("webgl2");
let glProgram: WebGLProgram | null = null;
let glIndexLoc = -1;
let uniLoc: { freq: WebGLUniformLocation | null; phase: WebGLUniformLocation | null; sampleRate: WebGLUniformLocation | null; waveform: WebGLUniformLocation | null } | null = null;

function compileShader(src: string, type: number): WebGLShader {
  const s = gl!.createShader(type)!;
  gl!.shaderSource(s, src);
  gl!.compileShader(s);
  if (!gl!.getShaderParameter(s, gl!.COMPILE_STATUS)) {
    console.error("Scope shader compile error:", gl!.getShaderInfoLog(s));
  }
  return s;
}

if (gl) {
  const vs = compileShader(compileGLSL.vertex(scopeVertex()), gl.VERTEX_SHADER);
  const fs = compileShader(compileGLSL.fragment(scopeFragment()), gl.FRAGMENT_SHADER);
  glProgram = gl.createProgram()!;
  gl.attachShader(glProgram, vs);
  gl.attachShader(glProgram, fs);
  gl.linkProgram(glProgram);
  if (!gl.getProgramParameter(glProgram, gl.LINK_STATUS)) {
    console.error("Scope program link error:", gl.getProgramInfoLog(glProgram));
  }
  gl.useProgram(glProgram);

  // The vertex's position in the sweep never changes — just 0..N-1, once.
  const indexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, Float32Array.from({ length: SCOPE_VERTEX_COUNT }, (_, i) => i), gl.STATIC_DRAW);
  glIndexLoc = gl.getAttribLocation(glProgram, a_index.name);
  gl.enableVertexAttribArray(glIndexLoc);
  gl.vertexAttribPointer(glIndexLoc, 1, gl.FLOAT, false, 0, 0);

  uniLoc = {
    freq: gl.getUniformLocation(glProgram, u_freq.name),
    phase: gl.getUniformLocation(glProgram, u_startPhase.name),
    sampleRate: gl.getUniformLocation(glProgram, u_sampleRate.name),
    waveform: gl.getUniformLocation(glProgram, u_waveform.name),
  };
}

let audioContext: AudioContext | null = null;
let workletNode: AudioWorkletNode | null = null;
let playing = false;

async function ensureAudio(): Promise<void> {
  if (audioContext) return;

  audioContext = new AudioContext();
  await audioContext.audioWorklet.addModule(new URL("./synthWorklet.ts", import.meta.url));

  workletNode = new AudioWorkletNode(audioContext, "rmsl-oscillator", { outputChannelCount: [1] });
  workletNode.connect(audioContext.destination);

  workletNode.port.postMessage({ type: "freq", value: Number(freqInput.value) });
  workletNode.port.postMessage({ type: "gain", value: Number(gainInput.value) / 100 });
  workletNode.port.postMessage({ type: "waveform", value: waveformSelect.value });
}

playBtn.addEventListener("click", async () => {
  await ensureAudio();
  if (!audioContext || !workletNode) return;

  playing = !playing;
  if (playing) await audioContext.resume();
  workletNode.port.postMessage({ type: "play", value: playing });

  playBtn.textContent = playing ? "⏸ Stop" : "▶ Play";
  playBtn.classList.toggle("active", playing);
  statusEl.textContent = playing
    ? "Playing — the same oscillatorSample() graph runs as WASM on the audio thread and as GLSL on the GPU, in parallel."
    : "Stopped — the visual keeps running off its own clock; only the WASM audio path is silent.";
});

freqInput.addEventListener("input", () => {
  const value = Number(freqInput.value);
  freqVal.textContent = `${value} Hz`;
  workletNode?.port.postMessage({ type: "freq", value });
});

gainInput.addEventListener("input", () => {
  const pct = Number(gainInput.value);
  gainVal.textContent = `${pct}%`;
  workletNode?.port.postMessage({ type: "gain", value: pct / 100 });
});

waveformSelect.addEventListener("change", () => {
  workletNode?.port.postMessage({ type: "waveform", value: waveformSelect.value });
});

// The visual's phase clock. When audio is running, `audioContext.currentTime`
// is the real clock the speakers are keyed to, so this stays close to the
// worklet's own phase accumulator (not sample-accurate — the worklet
// advances phase in exact per-block increments, this is a continuous
// approximation — near enough that the eye can't tell). Before Play is ever
// pressed there's no AudioContext yet, so the visual runs off wall-clock
// time instead — the drawing doesn't wait for audio to exist.
function currentTimeSeconds(): number {
  return audioContext ? audioContext.currentTime : performance.now() / 1000;
}

function drawScope() {
  requestAnimationFrame(drawScope);
  if (!gl || !glProgram || !uniLoc) return;

  const freq = Number(freqInput.value);
  const waveform = WAVEFORM_INDEX[waveformSelect.value] ?? 0;
  const phase = (currentTimeSeconds() * freq) % 1;

  gl.viewport(0, 0, scopeGl.width, scopeGl.height);
  gl.clearColor(0.004, 0.012, 0.043, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);

  gl.useProgram(glProgram);
  gl.uniform1f(uniLoc.freq, freq);
  gl.uniform1f(uniLoc.phase, phase);
  gl.uniform1f(uniLoc.sampleRate, VISUAL_SAMPLE_RATE);
  gl.uniform1i(uniLoc.waveform, waveform);
  gl.drawArrays(gl.LINE_STRIP, 0, SCOPE_VERTEX_COUNT);
}

drawScope();
