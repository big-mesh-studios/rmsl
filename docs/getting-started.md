# Getting Started

RMSL (Random Mesh Shading Language) is a TypeScript DSL for building shader programs. You construct a node graph in TypeScript and compile it to GLSL (WebGL 2) or WGSL (WebGPU) source code, or to a JavaScript or WebAssembly function that runs it on the CPU.

## Installation

```bash
pnpm add @random-mesh/rmsl
```

## Hello World

```typescript
import { Fn, float } from "@random-mesh/rmsl";
import { compileGlsl } from "@random-mesh/rmsl/glsl";

let prog = Fn(() => {
  let x = float(1.5).toVar();
  let y = float(2.0).toVar();
  return x.add(y).toVar();
});

let glsl = compileGlsl(prog());
console.log(glsl);
```

Output:

```glsl
#version 300 es
precision highp float;

layout(location=0) out vec4 _rmsl_fragColor;

void main(void) {
  float _rmsl_0 = 1.5;
  float _rmsl_1 = 2.0;
  float _rmsl_2 = _rmsl_0 + _rmsl_1;
  _rmsl_fragColor = vec4(_rmsl_2);
}
```

## How It Works

1. **`Fn(() => { ... })`** captures a scope. Inside it, you build a tree of `Node<T>` objects.
2. **`.toVar()`** assigns an expression to a temporary variable and returns a reference to it.
3. **`compileGlsl(root)`** / **`compileWgsl(root)`** walks the node tree and emits shader source. The value the function returns is the fragment's colour, converted to a `vec4`.

## Compiling to WebGPU (WGSL)

```typescript
import { compileWgsl } from "@random-mesh/rmsl/wgsl";

let wgsl = compileWgsl(prog());
```

## Running on the CPU (JavaScript or WebAssembly)

```typescript
import { compileJSRoutine } from "@random-mesh/rmsl/js";
import { compileWasmRoutine } from "@random-mesh/rmsl/wasm";

let js = compileJSRoutine(() => prog(), { name: "main", params: [] });
js({}); // 3.5

let wasm = compileWasmRoutine(() => prog(), { name: "main", params: [] });
wasm({}); // 3.5
```

## Next

See [API Reference](api.md) for the full type system and operations.
