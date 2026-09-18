# bomb-bloom

A bomb model with a wick fire and a bloom post-process, demonstrating RMSL
node graphs alongside a hand-rolled WebGL2 render loop and the
`@random-mesh/rmsl/effects` `PassGraph` system.

- `shader.ts` — the bomb body and wick fire shaders, written in RMSL. The
  flame is a port of melty-karts' `models/Bomb.tsx` wick fire (three.js TSL
  -> RMSL); the bomb body is a small lit-mesh shader. Every `Fn` compiles
  with `compileGlsl.vertex`/`compileGlsl.fragment` to a complete GLSL ES 3.00
  shader.
- `geometry.ts` — indexed-mesh geometry generated on the CPU: the bomb body
  (sphere, cylinder cap, wick tube) and the flame particle billboards.
- `matrix.ts` — column-major mat4 helpers; perspective/lookAt/inverse and the
  fullscreen quad verts are shared with the other apps
  (`apps/shared/shader.ts`).
- `bloom.ts` — executes a `PassGraph` on a raw WebGL2 context: one fullscreen
  quad per pass, render targets sized by each pass's `scale`, inputs bound by
  the producer pass named in `pass.inputs`.
- `main.ts` — wires it together: the bomb body and wick fire are drawn with a
  raw WebGL2 context, then the bloom `PassGraph` is executed by `bloom.ts`.
