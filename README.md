# RMSL (Random Mesh Shading Language)

[![npm version](https://badge.fury.io/js/%40random-mesh%2Frmsl.svg)](https://www.npmjs.com/package/@random-mesh/rmsl)
[![GitHub Repo stars](https://img.shields.io/github/stars/big-mesh-studios/rmsl?style=social)](https://github.com/big-mesh-studios/rmsl)

A TypeScript DSL for building shader programs. Define a node graph in TypeScript and compile it to **GLSL** (WebGL 2), **WGSL** (WebGPU), **JavaScript**, or **WebAssembly** (two CPU targets for screen picking and other per-pixel host-side work).

```typescript
import { Fn, float, vec4, uniform, compileGLSL, compileWGSL, compileJS, compileWasm } from "rmsl";

let prog = Fn(() => {
  let color = uniform("vec4");
  let brightness = float(0.5).toVar();
  return color.mul(brightness).toVar();
});

let glsl = compileGLSL(prog());
let wgsl = compileWGSL(prog());
let js = compileJS(() => prog()); // fn(ctx) -> color, run on the CPU
let wasm = compileWasm(() => prog(), { name: "main", params: [] }); // same contract, real WASM module
```

## Features

- **Type-safe** - TypeScript types for all shader types: float/int/uint/bool, vec2-4, ivec2-4, uvec2-4, mat2-4, and float/integer samplers (sampler2D/3D/Cube, isampler2D/3D/Cube, usampler2D/3D/Cube)
- **Four backends** - Compile to GLSL ES 3.0, WGSL, JavaScript, or WebAssembly from the same node graph
- **CPU / JS target** - `compileJS`/`compileJSFn` turn an `Fn` into a callable that runs on the CPU, one fragment at a time — for screen picking from a ray-marched scene without a GPU round-trip. Per-call evaluation allocates nothing (hoisted scratch slots + out-parameter vector helpers)
- **CPU / WASM target** - `compileWasm`/`compileWasmFn` compile the same `Fn` to a real WebAssembly module instead — real `i32`/`f64` types, no `eval`, and a `.draw(ctx, width, height)` that renders a whole pixel grid through one exported WASM function rather than one host call per pixel. Satisfies the same `CpuRenderer` interface as `compileJS`'s result, so code can pick between them without knowing which it got
- **Post-processing effects** - `import { fxaa, gaussianBlur, crt, ... } from "@random-mesh/rmsl/effects"` provides a port of three.js's `examples/jsm/tsl/display` color effects as pure node graphs (no renderer inside — you draw the quad), plus `fragCoord()`/`screenUV()`/`textureLoad()`/`textureSize()` and friends in the core DSL
- **Scene graph & node materials** - `import { Scene, Mesh, WebGLRenderer, ... } from "@random-mesh/rmsl/scene"` provides three.js-style scene-graph objects and **node-based materials** — a material is an RMSL node graph (`colorNode`, `roughnessNode`, `fragmentNode`, ...) compiled by the same DSL compiler, with WebGL2 and WebGPU renderers that bind geometry, upload uniforms and draw
- **Shader tests without a GPU** - `import { evaluate, render } from "@random-mesh/rmsl/test"` runs a shader graph on the CPU and hands back values or a grid of fragments, so a colour ramp, a distance field or a lighting term is asserted on in a plain unit test — no browser, no canvas, no pixel readback. `fromProgram`/`fromPass` do the same for a built scene material or one pass of an effect, addressed by the names they use
- **Vite plugins** - `import { precompileShaders, precompileJS, precompileWasm } from "@random-mesh/rmsl/vite"` compiles the node graph at build time, so the browser ships plain GLSL/WGSL strings, JS callables, or a real `.wasm` asset instead of rmsl itself
- **Casts & conversions** - `uint()`, `ivec3(vec3)`, and chained `.toInt()`/`.toVec3()`/`.toUVec4()`/… for any type
- **Constant folding** - Math on literal values is evaluated at compile time
- **Control flow** - `If`/`ElseIf`/`Else`, `Switch`/`Case`/`Default`, `For`, `While`, `Loop`, `Break`, `Continue`, `Return`, `Discard` — matching TSL
- **TSL-compatible API** - free functions like `mul(a, b)`, `sin(x)`, `mix(a, b, t)`, `bool()`, and the `PI`/`TWO_PI`/`EPSILON`/… constants, so a shader written against `three/tsl` migrates by changing its import
- **Swizzles** - `.xyz`, `.rgba`, `.stpq`, `.xy`, etc. on vec3/vec4, ivecN and uvecN (read and write)
- **Integer textures** - isampler*/usampler* sample to ivec4/uvec4 via unfiltered texelFetch/textureLoad
- **Vertex/fragment** - Separate vertex and fragment compilation with proper I/O
- **Built-in outputs** - `output()`, `builtinPosition()`, `varying()`, `attribute()`, `uniform()`

## Documentation

- [Getting Started](https://github.com/big-mesh-studios/rmsl/blob/main/docs/getting-started.md) - Quick setup and hello world
- [API Reference](https://github.com/big-mesh-studios/rmsl/blob/main/docs/api.md) - Full type system, constructors, and operations
- [TSL Migration](https://github.com/big-mesh-studios/rmsl/blob/main/docs/tsl-migration.md) - Porting a Three.js TSL shader to RMSL
- [Compilation](https://github.com/big-mesh-studios/rmsl/blob/main/docs/compilation.md) - GLSL/WGSL output, type mappings, binding model, and the JS CPU target
- [WASM Target](https://github.com/big-mesh-studios/rmsl/blob/main/docs/wasm.md) - `compileWasm`, the second CPU target — a real WebAssembly module instead of JS source
- [Effects](https://github.com/big-mesh-studios/rmsl/blob/main/docs/effects.md) - Post-processing effects ported from three.js TSL (`@random-mesh/rmsl/effects`)
- [Scene Graph](https://github.com/big-mesh-studios/rmsl/blob/main/docs/scene.md) - three.js-style scene objects and node-based materials (`@random-mesh/rmsl/scene`)
- [Testing](https://github.com/big-mesh-studios/rmsl/blob/main/docs/testing.md) - Running shader logic on the CPU in unit tests (`@random-mesh/rmsl/test`)
- [Vite Plugins](https://github.com/big-mesh-studios/rmsl/blob/main/docs/vite-plugins.md) - Precompiling shaders and CPU callables at build time so rmsl is never shipped
- [Contributing](https://github.com/big-mesh-studios/rmsl/blob/main/CONTRIBUTING.md) - Test setup, and how to add an operation or a shader type

## Links

- [GitHub Repository](https://github.com/big-mesh-studios/rmsl)
- [npm Package](https://www.npmjs.com/package/@random-mesh/rmsl)
