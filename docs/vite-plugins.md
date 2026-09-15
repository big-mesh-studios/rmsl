# Vite Plugins

RMSL compiles node graphs at runtime by default: `compileGLSL`/`compileWGSL`/`compileJS`/`compileWasm` all run in the browser. For an app with a large shader, that means shipping rmsl itself (the whole DSL) to every client just to produce output that never changes.

The three plugins in `@random-mesh/rmsl/vite` move that compilation to build time. Each targets a module you write, bundles it with esbuild, executes it once in the Node process running Vite, and rewrites the module so the browser gets only the finished result — no rmsl, no `eval`, just constants, plain functions, or (for WASM) a real binary module.

```typescript
import { defineConfig } from "vite";
import { precompileShaders, precompileJS, precompileWasm } from "@random-mesh/rmsl/vite";

export default defineConfig({
  plugins: [
    precompileShaders({ include: "src/shaders.ts" }),
    precompileJS({ include: "src/cpu-fns.ts" }),
    precompileWasm({ include: "src/wasm-fns.ts" }),
  ],
});
```

All three plugins share the same matching options:

| Option    | Type                                          | Meaning                                                                                                            |
| --------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `include` | `string \| RegExp \| Array<string \| RegExp>` | Modules to rewrite. A string is matched as a normalized path suffix; a RegExp is tested against the normalized id. |
| `exclude` | same                                          | Modules to leave alone, taking precedence over `include`.                                                          |

## precompileShaders — GPU shaders

Targets a module whose **default export** is the compiled shader program: the GLSL/WGSL sources and the slot names a caller needs to address uniforms, varyings and attributes. The module is evaluated at build time and rewritten to `export default <JSON>`.

```typescript
// src/shaders.ts
import { Fn, attribute, compileGLSL, uniformRaw, varying, vec2, vec4 } from "@random-mesh/rmsl";

export const uColour = uniformRaw("uColour", "vec3");
export const vUv = varying("vec2");
export const positionAttr = attribute("vec2");

export const vertexFn = Fn(() => {
  vUv.assign(positionAttr);
  return vec4(positionAttr, 0, 1);
});

export const fragmentFn = Fn(() => vec4(uColour, 1));

export default {
  uColour: uColour.name,
  vUv: vUv.name,
  positionAttr: positionAttr.name,
  vertexGLSL: compileGLSL.vertex(vertexFn()),
  fragmentGLSL: compileGLSL.fragment(fragmentFn()),
};
```

After the plugin runs, `src/shaders.ts` is effectively `export default {"uColour":"uColour", ...}` — the rmsl graph is gone and the app just reads strings:

```typescript
import shaders from "./shaders"; // plain JSON at runtime
```

The default export must be JSON-serializable (strings, numbers, booleans, arrays, plain objects). A module without a default export, or one whose default export is not serializable, fails the build with a message naming the module.

## precompileJS — CPU-callable shader functions

Any shader function can be compiled to run on the CPU with `compileJS`/`compileJSFn` (see [Compilation](compilation.md#js--cpu-target)). Precompiling it means the browser never evaluates the shader graph and never runs the `eval` that `compileJS` uses to build the callable.

The target module exports a **map of name → `compileJSFn()` output** under the export named by `codeExport` (default `__RMSL_JS_CODE`):

```typescript
// src/cpu-fns.ts
import { Fn, compileJSFn, float, uniform, type Node } from "@random-mesh/rmsl";

const brightness = Fn(() => uniform("vec3").mul(float(0.5)).toVar());
const mixColours = Fn((a: Node<"vec3">, b: Node<"vec3">, t: Node<"float">) => a.mix(b, t).toVar());

export const __RMSL_JS_CODE = {
  brightness: compileJSFn(() => brightness(), { name: "brightness", params: [] }),
  mixColours: compileJSFn(mixColours, {
    name: "mixColours",
    params: [
      { name: "a", type: "vec3" },
      { name: "b", type: "vec3" },
      { name: "t", type: "float" },
    ],
  }),
};
```

The plugin rewrites each entry to `export const name = (() => { <compiled code> })()` — a plain callable:

```typescript
import { brightness, mixColours } from "./cpu-fns"; // plain functions at runtime

brightness({ uniforms: { _rmsl_u0: [1, 2, 3] } }); // [0.5, 1, 1.5]
mixColours({ params: { a: [0, 0, 0], b: [1, 1, 1], t: 0.5 } }); // [0.5, 0.5, 0.5]
```

The callables take the same `CpuShaderContext` as `compileJS` output: uniforms by slot, params by name. Nothing is assumed about your call convention — if a function reads varyings by slot name and you want to pass them through a friendlier shape, wrap it yourself in a plain module:

```typescript
import { pick } from "./cpu-fns";

export function voxelPicker(ctx) {
  return pick({ ...ctx, varyings: { vUv: ctx.varying.vUv } });
}
```

| Option       | Type     | Default          | Meaning                                             |
| ------------ | -------- | ---------------- | --------------------------------------------------- |
| `codeExport` | `string` | `__RMSL_JS_CODE` | The named export carrying the `{ name: code }` map. |

## precompileWasm — WASM modules

Any shader function can also be compiled to a real WebAssembly module with [`compileWasm`/`compileWasmFn`](wasm.md), the CPU-eval niche `compileJS` serves, with real `i32`/`f64` types and no `eval`. Precompiling it means the browser never runs the graph builder or bytecode emitter that produced the module — only the small piece of glue that instantiates it and marshals calls in and results out.

The target module exports a **map of name → `compileWasmFn()` output** under the export named by `codeExport` (default `__RMSL_WASM_CODE`). Each entry's `name` in the map has to be the same string it was compiled with (`compileWasmFn(fn, { name, ... })`) — that name is baked into the compiled bytes' own export table:

```typescript
// src/wasm-fns.ts
import { Fn, compileWasmFn, float, uniform, type Node } from "@random-mesh/rmsl";

const brightness = Fn(() => uniform("vec3").mul(float(0.5)).toVar());
const mixColours = Fn((a: Node<"vec3">, b: Node<"vec3">, t: Node<"float">) => a.mix(b, t).toVar());

export const __RMSL_WASM_CODE = {
  brightness: compileWasmFn(() => brightness(), { name: "brightness", params: [] }),
  mixColours: compileWasmFn(mixColours, {
    name: "mixColours",
    params: [
      { name: "a", type: "vec3" },
      { name: "b", type: "vec3" },
      { name: "t", type: "float" },
    ],
  }),
};
```

Unlike the other two plugins, each compiled module's bytes are emitted as a real `.wasm` asset (via Rollup's `emitFile`) rather than inlined into the rewritten module as a string — raw bytes in the build output, no string-encoding overhead, cached by the browser like any other asset. The rewritten module fetches that asset once at load and hands the bytes to [`instantiateWasm`](wasm.md#api):

```typescript
import { brightness, mixColours } from "./wasm-fns"; // CpuRoutine objects at runtime, once the fetch resolves

brightness.invoke({ uniforms: { _rmsl_u0: [1, 2, 3] } }); // { value: [0.5, 1, 1.5] }
mixColours.invoke({ params: { a: [0, 0, 0], b: [1, 1, 1], t: 0.5 } }); // { value: [0.5, 0.5, 0.5] }
```

| Option       | Type     | Default            | Meaning                                                 |
| ------------ | -------- | ------------------ | ------------------------------------------------------- |
| `codeExport` | `string` | `__RMSL_WASM_CODE` | The named export carrying the `{ name: compiled }` map. |

## How it works

A matching module is handed to esbuild with `bundle: true`, `platform: "node"`, and its own file as `resolveDir`, so imports (including `@random-mesh/rmsl` itself) resolve and get bundled in. The bundle is then loaded through a `data:` URL `import()` and the module's exports are read. All three plugins keep a cache keyed by a hash of the module source, so dev HMR re-evaluates only when the module changes.

Consequences worth knowing:

- The target module runs in Node at build time, so its top level must be Node-safe and side-effect-free enough to execute during a Vite build.
- Only the module itself is evaluated; anything it imports is bundled and executed too — which is what makes the rmsl graph compile.
- `include` must match the module as Vite sees it (an absolute path), which a suffix like `"src/shaders.ts"` does.
