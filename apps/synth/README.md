# synth

A small audio synth demo combining an `AudioWorklet` processor with an RMSL
oscilloscope shader.

- `audioWorkletGlobals.d.ts` — minimal ambient types for the
  `AudioWorkletGlobalScope`. TypeScript's DOM lib doesn't ship these, and
  pulling in the full "audioworklet" lib just for one small processor file
  isn't worth the config churn.
