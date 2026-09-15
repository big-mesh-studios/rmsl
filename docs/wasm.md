# WASM / CPU Target

RMSL also compiles an `Fn` straight to a raw WebAssembly binary module — hand-encoded bytes, no wabt/binaryen — instead of JavaScript source. It targets the same CPU-eval niche the [JS target](compilation.md#js--cpu-target) does: screen picking, ray-march hit tests, anything calling a compiled shader graph once per pixel/click from plain JS, where per-call overhead matters more than raw throughput on a hot, already-warm loop.

```typescript
import { compileWasm, Fn, uniform, output, builtinFragDepth } from "rmsl";

let pickFn = compileWasm(calcColourAndDepth, { name: "pick", params: [] });
// On pointerdown:
let r = pickFn.invoke({
  uniforms: {
    _rmsl_u0: cameraPosition, // each slot is the uniform's .name
    _rmsl_u1: cameraViewMatrix, // flat column-major arrays
    // ...
  },
  varyings: { _rmsl_v0: positionGeometry }, // per-pixel, from the fragment coord
});
let colour = r.value; // the Fn's return value (e.g. the ray-marched colour)
let depth = r.fragDepth; // written via builtinFragDepth(), for the world pick point
```

If that shape looks familiar: it's identical to `compileJS`'s. Both targets share the same host-facing contract — `CpuShaderContext` in, `CpuShaderResult` (or a bare scalar/boolean) out — described in full under [The context object](compilation.md#the-context-object) and [Return value](compilation.md#return-value). This page covers what's different about compiling to WASM specifically, not the shape both already share.

## Why a second CPU target

`compileJS` turns an `Fn` into JavaScript source, `new Function`'d into a callable. `compileWasm` turns the same `Fn` into an actual `WebAssembly.Module`, instantiated once. Two consequences follow from that:

- **Real types, not one JS number for everything.** `compileJS` computes every declared type — `int`, `uint`, `bool`, `float` alike — as a plain JS number. `compileWasm`'s `int`/`uint`/`bool` are real 32-bit WASM integers, with signed/unsigned opcode variants chosen per operand type; `float` stays `f64`, matching `compileJS`'s own arithmetic bit for bit (so the two never need reconciling — see [Caveats](#caveats)).
- **Per-call overhead, not raw throughput.** A WASM call crosses a real module boundary — marshalling scalar args, reading vectors/matrices out of linear memory afterward — which costs more per call than a JS function returning a value directly. The measured win shows up from roughly two loop iterations upward inside the compiled function itself; a loop-free, called-once function still favors `compileJS`. See `ROADMAP.md` for the benchmark tables behind that number.

Pick whichever backend matches the shape of the work. Nothing else about calling one differs from calling the other — they satisfy the same `CpuRoutine` interface (below), so code that picks between them at runtime doesn't need to know which one it got.

## API

```typescript
compileWasmFn(fn, options): CompiledWasm
// { bytes: Uint8Array, params: WasmParam[], resultType: ShaderType,
//   textureHeapBase: number, batch?: { componentCount, kind } }
// The module's raw bytes plus the metadata a host needs to call it —
// analogous to compileJSFn returning source instead of a callable.

instantiateWasm(compiled: CompiledWasm, name: string): CpuRoutine
// Turns compileWasmFn's output into a live, callable module: instantiates
// the WebAssembly.Module, and wraps it with the same marshalling
// compileWasm itself uses. `name` is the exported function's name inside
// the module — the same string passed as `options.name` when it was
// compiled with compileWasmFn.

compileWasm(fn, options): CpuRoutine
// compileWasmFn(fn, options) followed by instantiateWasm(...) in one call —
// what you want unless you're precompiling (see below).
```

`instantiateWasm` exists as its own export specifically so a build step can compile once and instantiate many times, or instantiate compiled bytes that were never compiled in the browser at all — which is exactly what [`precompileWasm`](vite-plugins.md#precompilewasm--wasm-modules) does.

Options extend the `Fn` compilers', the same set `compileJS` accepts:

- `stage`: `"fragment"` (default) or `"vertex"`.
- `derivatives`: `"throw"` (default) or `"zero"` — WASM has no derivatives either, for the same reason a single CPU evaluation doesn't.
- `reentrant`: accepted for parity with `compileJS`, and a no-op here — WASM locals are already fresh per call frame, so there is no shared scratch a re-entrant call could clobber.

## `CpuRoutine`

```typescript
type CpuRoutine = {
  invoke(ctx: CpuShaderContext): number | boolean | CpuShaderResult;
  batch(ctx: CpuShaderContext, width: number, height: number): Float64Array | Int32Array | Uint32Array;
};
```

Both `compileJS` and `compileWasm` return a `CpuRoutine`: `invoke()` runs the compiled function once, the same shape described above; `batch()` runs it over a whole `width x height` grid in one call instead of one call per pixel from the host side. Neither name claims what the invocation actually does — a `storage()`/`invocationIndex()` compute program's `invoke()` mutates `ctx.storages` and its return value is irrelevant, same as a vertex/fragment program's `invoke()`/`batch()` producing a real result.

```typescript
let fn = compileWasm(calcColour, { name: "main", params: [] });
let pixels = fn.batch({ uniforms: { ... } }, 256, 256);
// Float64Array/Int32Array/Uint32Array, length width * height * componentCount,
// row-major, one element type picked from the Fn's own result type.
```

`batch()` feeds each pixel's center — `(x + 0.5, y + 0.5)` — in as `fragCoord()`, holding every other input (uniforms, textures, …) fixed across the grid. On `compileWasm`'s side this shares the compiled function's own bytecode via a second exported WASM function that loops internally and calls the first, so a whole-image evaluation pays the per-call marshalling cost once rather than once per pixel — the gap `compileJS`'s own `batch()` (a plain JS loop, one call per pixel) doesn't have to close the same way, since a JS function call is already cheap.

A `void`-returning `Fn` has nothing to produce — `batch()` throws, naming that.

## Texture sampling

Textures work exactly as described in [Sampling](compilation.md#sampling) — same `CpuTextureData` shape, same filtering/wrapping rules, same 8-bit-normalizes-to-0–1 behavior. The one difference is where the pixel data ends up: `compileWasm` copies it into the compiled module's own linear memory rather than reading it through a callback into JavaScript, so a texture-sampling shader can run standalone — no JS engine behind it required to answer "what's this pixel" — the same property a `.wasm` file shipped and run outside a browser would need.

(Transcendental functions — `sin`/`cos`/`pow`/`exp`/… — don't share that property yet: they call back into the host's `Math` object through a WASM import, same as any other host call would. A program that only samples textures is standalone; one that also calls a transcendental function still needs a JS engine behind the `math` import. See `ROADMAP.md`'s Open Questions.)

## Precompiling

[`precompileWasm`](vite-plugins.md#precompilewasm--wasm-modules) moves compilation to build time, the WASM counterpart to [`precompileJS`](vite-plugins.md#precompilejs--cpu-callable-shader-functions): the target module exports a map of `compileWasmFn()` results, and the plugin rewrites it into `instantiateWasm(...)` calls fed by real `.wasm` assets — no graph builder or bytecode emitter shipped to the browser, only the small piece of glue `instantiateWasm` is.

## Testing with it

Same story as the JS target: `@random-mesh/rmsl/test` is the ergonomic layer for exercising shader logic without a device. See [Testing](testing.md).

## Caveats

- `float` is `f64`, matching `compileJS`'s own JS-number arithmetic bit for bit — including the transcendental functions, which both backends call through the literal same `Math` object. Comparing a WASM result against a JS one needs no tolerance; comparing either against a GPU backend's `f32` result does (see [Caveats](compilation.md#caveats) on the JS target's own page).
- `isamplerCube`/`usamplerCube` are not supported yet — `compileJS` doesn't support them either.
- A `construct` converting between a `bool` component and a `float`/`int`/`uint` one on either side has no defined behavior yet; nothing in the DSL exercises it today.
- Async `WebAssembly.instantiate` isn't used — instantiation is synchronous (`new WebAssembly.Instance(new WebAssembly.Module(bytes))`), the browser-recommended path for a small module. Revisit once real programs are large enough for that to matter.
