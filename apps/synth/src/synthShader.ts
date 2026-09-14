import { float, Fn, fragCoord, If, int, Node, TWO_PI, uniform } from "@random-mesh/rmsl";

export let u_freq = uniform("float");
export let u_startPhase = uniform("float"); // phase [0,1) at sample/vertex index 0
export let u_sampleRate = uniform("float");
export let u_waveform = uniform("int"); // 0 sine, 1 saw, 2 square, 3 triangle

/**
 * One sample of a band-naive oscillator at `sampleIndex` samples after
 * `startPhase`. A plain function, not `Fn(...)`-wrapped, so it inlines
 * correctly wherever it's called — see the identical note on
 * `mandelbrotColorAt` in apps/mandelbrot/src/mandelbrotShader.ts for why
 * that matters for the WASM backend specifically.
 *
 * This is the one piece of math shared by the audio and the visual: WASM
 * (synthCpu below) calls it once per PCM sample to drive the speakers;
 * scopeShader.ts's GLSL vertex shader calls the exact same function once
 * per line vertex to draw it. Neither side is "fed" by the other — the
 * waveform on screen isn't an analysis of the waveform playing, it's the
 * same computation evaluated a second time, on the GPU, at a resolution
 * chosen for the eye rather than the ear.
 */
export function oscillatorSample(
  sampleIndex: Node<"float">,
  startPhase: Node<"float">,
  freq: Node<"float">,
  sampleRate: Node<"float">,
  waveform: Node<"int">,
): Node<"float"> {
  let phase = startPhase.add(sampleIndex.mul(freq).div(sampleRate)).toVar();
  phase.assign(phase.sub(phase.floor())); // wrap to [0, 1)

  let sample = float(0.0).toVar();

  If(waveform.equal(int(0)), () => {
    sample.assign(phase.mul(TWO_PI).sin());
  })
    .ElseIf(waveform.equal(int(1)), () => {
      sample.assign(phase.mul(2.0).sub(1.0)); // saw: -1 -> 1
    })
    .ElseIf(waveform.equal(int(2)), () => {
      If(phase.lessThan(0.5), () => {
        sample.assign(float(1.0));
      }).Else(() => {
        sample.assign(float(-1.0));
      });
    })
    .Else(() => {
      // triangle
      let t = phase.mul(2.0).toVar(); // 0 -> 2
      If(t.lessThan(1.0), () => {
        sample.assign(t.mul(2.0).sub(1.0));
      }).Else(() => {
        sample.assign(float(3.0).sub(t.mul(2.0)));
      });
    });

  return sample;
}

/**
 * `.draw()` feeds each sample's index in as `fragCoord().x`, the same "one
 * call covers the whole buffer" convention the Mandelbrot demo uses for
 * pixels, here applied to a 1D block of PCM samples. synthWorklet.ts
 * advances `u_startPhase` by this block's length between calls — the
 * oscillator's only persistent state lives on the host, not in the shader.
 */
export let synthCpu = Fn(() => {
  let sampleIndex = fragCoord().x.sub(0.5).toVar(); // undo draw()'s pixel-center (+0.5) offset
  return oscillatorSample(sampleIndex, u_startPhase, u_freq, u_sampleRate, u_waveform);
});
