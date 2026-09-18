# infinite-grid

A ray-marched infinite grid, rendered on WebGPU.

- `pick-check.ts` — an end-to-end check: compiles the shared ray-marched
  fragment shader to a JS callable and "picks" a pixel on the CPU, exactly
  as the app would on pointerdown, then verifies the ray actually lands on
  the y = 0 plane.
