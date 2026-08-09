# RMSL (Random Mesh Shading Language)

[![npm version](https://badge.fury.io/js/%40random-mesh%2Frmsl.svg)](https://www.npmjs.com/package/@random-mesh/rmsl)
[![GitHub Repo stars](https://img.shields.io/github/stars/clinuxrulz/rmsl?style=social)](https://github.com/clinuxrulz/rmsl)

A TypeScript DSL for building shader programs. Define a node graph in TypeScript and compile it to **GLSL** (WebGL 2) or **WGSL** (WebGPU).

```typescript
import { Fn, float, vec4, uniform, compileGLSL, compileWGSL } from "rmsl";

let prog = Fn(() => {
  let color = uniform("vec4");
  let brightness = float(0.5).toVar();
  return color.mult(brightness).toVar();
});

let glsl = compileGLSL(prog());
let wgsl = compileWGSL(prog());
```

## Features

- **Type-safe** - TypeScript types for all shader types (float, vec2-4, mat2-4, sampler2D/sampler3D, int, uint, bool)
- **Dual backend** - Compile to GLSL ES 3.0 or WGSL from the same node graph
- **Constant folding** - Math on literal values is evaluated at compile time
- **Control flow** - `If`/`Else If`/`Else`, `For`, `While`, `discard`, `break`/`continue`
- **Swizzles** - `.xyz`, `.rgba`, `.xy`, etc. on vec3/vec4 (read and write)
- **Vertex/fragment** - Separate vertex and fragment compilation with proper I/O
- **Built-in outputs** - `output()`, `builtinPosition()`, `varying()`, `attribute()`, `uniform()`

## Documentation

- [Getting Started](https://github.com/clinuxrulz/rmsl/blob/main/docs/getting-started.md) - Quick setup and hello world
- [API Reference](https://github.com/clinuxrulz/rmsl/blob/main/docs/api.md) - Full type system, constructors, and operations
- [Compilation](https://github.com/clinuxrulz/rmsl/blob/main/docs/compilation.md) - GLSL/WGSL output, type mappings, binding model
- [Contributing](https://github.com/clinuxrulz/rmsl/blob/main/CONTRIBUTING.md) - Test setup, and how to add an operation or a shader type

## Links

- [GitHub Repository](https://github.com/clinuxrulz/rmsl)
- [npm Package](https://www.npmjs.com/package/@random-mesh/rmsl)
