# Compilation

RMSL compiles node graphs to **GLSL** (WebGL 2 / OpenGL ES 3.0), **WGSL** (WebGPU), **JavaScript** (a CPU callable — see [JS / CPU Target](#js--cpu-target)), or **WebAssembly** (a second CPU target, same contract — see [WASM / CPU Target](wasm.md)).

## Compiler API

```typescript
import { compileGLSL, compileWGSL } from "rmsl";
```

### Fragment shader (default)

```typescript
compileGLSL(root); // fragment shader
compileGLSL.fragment(root); // explicit fragment
compileWGSL(root); // fragment shader
compileWGSL.fragment(root); // explicit fragment
```

### Vertex shader

```typescript
compileGLSL.vertex(root);
compileWGSL.vertex(root);
```

### Shader precision (GLSL)

GLSL output defaults to `highp`. To compile with `mediump` or `lowp` — the
usual choice on mobile GPUs, mirroring three.js's `precision` option — pass an
options object to any of the `compileGLSL` forms:

```typescript
compileGLSL(root, { precision: "mediump" }); // fragment
compileGLSL.fragment(root, { precision: "lowp" });
compileGLSL.vertex(root, { precision: "mediump" });
```

The configured precision is applied to the `float` declaration and to every
sampler the shader declares. WGSL and the JS target have no precision
qualifiers, so `compileWGSL` and `compileJS` take no such option.

### Multiple return values

```typescript
let prog = Fn(() => {
  let a = float(1).toVar();
  let b = float(2).toVar();
  return [a, b];
});
let [a, b] = prog();
compileGLSL.vertex([a, b]); // pass array of roots
```

## GLSL Output

### Fragment shader

```glsl
#version 300 es
precision highp float;          // or mediump / lowp via { precision }

// uniforms, attributes, varyings, outputs...

void main(void) {
  // shader body
}
```

### Vertex shader

Same structure, but the last expression is assigned to `gl_Position`:

```glsl
void main(void) {
  // body...
  gl_Position = <lastExpr>;
}
```

### Output variables

`output("vec4")` declares a `layout(location=N) out vec4 _rmsl_oN;` in the fragment shader.

### Uniforms

`uniform("float")` produces `uniform float _rmsl_uN;` in both vertex and fragment shaders.

The returned node has a `.name` property for the generated name, and carries its type's methods directly:

```typescript
let uTime = uniform("float");
console.log(uTime.name); // outputs "_rmsl_u0"
let x = uTime.add(1.0); // methods and swizzles are available directly
```

### Varyings

`varying("vec3")` produces `out vec3 _rmsl_vN;` in vertex and `in vec3 _rmsl_vN;` in fragment.

Access `.name` for the generated name; methods and swizzles are available directly:

```typescript
let v = varying("vec3");
console.log(v.name); // outputs "_rmsl_v0"
let x = v.x; // swizzles are available directly
```

## WGSL Output

### Fragment shader

```wgsl
struct FragmentOutput {
  @location(N) _rmsl_oN: type,
};

@fragment
fn main() -> FragmentOutput {
  // body...
  return FragmentOutput(...);
}
```

### Vertex shader

```wgsl
struct VertexInput {
  @location(N) _rmsl_aN: type,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  // varyings...
};

@vertex
fn main(input: VertexInput) -> VertexOutput {
  // body...
  return VertexOutput(...);
}
```

### Attributes

Attributes are declared in the `VertexInput` struct in WGSL. Use `.name` for the generated name; methods and swizzles are available directly:

```typescript
let pos = attribute("vec3");
console.log(pos.name); // outputs "_rmsl_a0"
let x = pos.x; // swizzles are available directly
```

This produces:

```wgsl
struct VertexInput {
  @location(0) _rmsl_a0: vec3<f32>,
};
```

A matrix attribute — an instance transform, say — arrives as its columns, since
WGSL puts no matrix at a `@location`. Four `vec4<f32>` at four consecutive
locations, put back together at the top of `main`:

```wgsl
struct VertexInput {
  @location(0) _rmsl_a0: vec3<f32>,
  @location(1) _rmsl_a1_0: vec4<f32>,
  @location(2) _rmsl_a1_1: vec4<f32>,
  @location(3) _rmsl_a1_2: vec4<f32>,
  @location(4) _rmsl_a1_3: vec4<f32>,
};

@vertex
fn main(input: VertexInput) -> VertexOutput {
  var result: VertexOutput;
  let _rmsl_a1 = mat4x4<f32>(input._rmsl_a1_0, input._rmsl_a1_1, input._rmsl_a1_2, input._rmsl_a1_3);
  ...
```

The buffer is unchanged by this — one 64-byte record per instance, read as four
columns — so a vertex layout that feeds GLSL's `in mat4` feeds this too. The
locations a matrix occupies are its column count, so whatever follows it starts
four along.

### Narrowing a matrix

GLSL writes the normal-matrix idiom as `mat3(modelMatrix)`, dropping the fourth
column and row. WGSL has no constructor that takes a matrix, so RMSL emits a
helper that truncates the columns, and calls it with the source expression once:

```wgsl
fn _rmsl_mat3_from_mat4(m: mat4x4<f32>) -> mat3x3<f32> {
  return mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
}
```

## Binding Model (WGSL)

| Resource | Group       | Binding       |
| -------- | ----------- | ------------- |
| Uniforms | `@group(0)` | `@binding(N)` |
| Textures | `@group(1)` | `@binding(N)` |
| Samplers | `@group(2)` | `@binding(N)` |

Every uniform that is not a texture goes into one struct at
`@group(0) @binding(0)`, since WGSL allows only a handful of uniform buffers per
stage. A draw therefore sets one bind group per kind: the uniform buffer, the
textures, the samplers.

### Two stages, one uniform buffer

A stage declares the uniforms it reads. Compile a vertex and a fragment stage
from one program and each gets a struct of its own — a material colour the
fragment stage alone reads, matrices the vertex stage alone reads — so the same
byte offset means a different value in each, and different again in the buffer
the host packs.

Pass the program's whole uniform set to both stages, and to
`wgslUniformLayout` when packing the buffer, and all three agree:

```typescript
const uniforms = [
  { slot: "materialColor", type: "vec3<f32>" },
  { slot: "modelMatrix", type: "mat4x4<f32>" },
  { slot: "projectionMatrix", type: "mat4x4<f32>" },
  { slot: "viewMatrix", type: "mat4x4<f32>" },
];

const vertex = compileWGSL.vertex(vertexRoot, { uniforms });
const fragment = compileWGSL.fragment(fragmentRoot, { uniforms });
const layout = wgslUniformLayout(uniforms); // the offsets to write at
```

A member a stage never reads costs it nothing. A uniform a stage _does_ read but
the list leaves out is an error, named by slot, rather than a shader that fails
to compile in the driver. `@random-mesh/rmsl/scene`'s WebGPU renderer does this
for every material it draws.

## Standalone Function Compilers

For use with Three.js `glslFn`/`wgslFn` or other embedding scenarios, RMSL provides `compileGLSLFn` and `compileWGSLFn` to compile individual functions with custom names and parameters:

```typescript
import { compileGLSLFn, compileWGSLFn, float, var_ } from "rmsl";

let glsl = compileGLSLFn((a, b) => a.add(b).sin(), {
  name: "myFunc",
  params: [
    { name: "a", type: "float" },
    { name: "b", type: "float" },
  ],
});
```

### GLSL output

```glsl
float myFunc(float a, float b) {
  return sin((a + b));
}
```

### WGSL output

```wgsl
fn myFunc(a: f32, b: f32) -> f32 {
  return sin((a + b));
}
```

Uniforms referenced inside the function body are declared automatically with their bindings:

```typescript
let glsl = compileGLSLFn(
  (v) => {
    let u = uniformRaw("uScale", "float");
    return v.mul(u);
  },
  { name: "scale", params: [{ name: "v", type: "float" }] },
);
```

Produces:

```glsl
uniform float uScale;

float scale(float v) {
  return (v * uScale);
}
```

## JS / CPU Target

RMSL also compiles an `Fn` to a JavaScript callable that runs on the CPU — one
fragment at a time. Its purpose is screen picking from a ray-marched scene:
feed the per-pixel varyings and uniforms into the compiled function and read
the colour/depth back, with no GPU round-trip.

```typescript
import { compileJS, compileJSFn, Fn, uniform, output, builtinFragDepth } from "rmsl";

let pickFn = compileJS(calcColourAndDepth, { name: "pick", params: [] });
// On pointerdown:
let r = pickFn({
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

### Value model

| RMSL                                           | JavaScript                                         |
| ---------------------------------------------- | -------------------------------------------------- |
| `float`/`int`/`uint`/`bool`                    | number / boolean                                   |
| `vec2`–`vec4`, `ivecN`, `uvecN`, `bvecN`       | arrays `[x, y, z]`                                 |
| `matCxR`                                       | flat column-major array of numbers                 |
| `sampler2D`/`sampler3D` (and integer variants) | sampled from `ctx.textures[slot]` (see _Sampling_) |

Vectors and matrices are plain arrays, matching the flat `Float32Array`
conventions the apps already use.

### API

```typescript
compileJSFn(fn, options): string
// A self-contained expression that evaluates to the callable:
//   const fn = new Function(source)();

compileJS(fn, options): (ctx) => value | result
// The real callable, scratch and helpers baked into its closure.
```

Options extend the `Fn` compilers':

- `stage`: `"fragment"` (default) or `"vertex"`.
- `derivatives`: `"throw"` (default) or `"zero"`. Derivative ops have no meaning
  for a single CPU evaluation; compile with `"zero"` to make shaders that use
  `fwidth`/`dFdx`/`dFdy` runnable.
- `reentrant`: `false` (default) or `true`. See _Scratch & reentrancy_.

### The context object

```typescript
ctx = {
  params?: Record<string, number | number[]>,
  uniforms?: Record<string, number | number[]>,
  varyings?: Record<string, number | number[]>,
  attributes?: Record<string, number | number[]>,
  textures?: Record<string, { data, width, height, depth? }>,
}
```

Every key is the node's `.name` (`_rmsl_u0`, `_rmsl_v0`, …), which is how the
host knows which slot holds which value. Scalars are numbers, vectors/matrices
are arrays.

### Return value

If the program writes no `output()`, `builtinPosition()` or
`builtinFragDepth()`, the callable returns the Fn's value directly. If it
writes any of them, it returns:

```typescript
{
  value:    <the Fn's return value>,
  outputs:  { [slot]: <value> },   // from output()
  varyings: { [slot]: <value> },   // from varying() in a vertex stage
  position: <vec4>,                // from builtinPosition()
  fragDepth: <number>,             // from builtinFragDepth()
}
```

This is what surfaces the picking depth: `calcColourAndDepth` assigns
`builtinFragDepth()`, so `result.fragDepth` gives the distance along the ray.

### Rendering a whole grid

`compileJS`'s result also has `.draw(ctx, width, height)`: call the compiled
function once per pixel over a `width x height` grid instead of driving the
loop yourself, packed into one flat, row-major typed array. It's the same
method [`compileWasm`'s result](wasm.md#cpurenderer) has — both satisfy one
`CpuRenderer` interface — documented there since WASM's version has the more
interesting implementation (it shares the compiled function's own bytecode
rather than looping in JS).

### Scratch & reentrancy

Internal `toVar()` variables live in per-program scratch slots _outside_ the
callable, preallocated once (scalars as bare `let`s, vectors/matrices as
zeroed arrays). Vector/matrix helpers take a trailing output array, so an
assignment writes in place: a per-pixel call allocates nothing beyond the
result. Because shaders cannot recurse, no compiled function can clobber its
own scratch through nested calls.

The trade-off: the scratch is shared across calls, so two _overlapping_ calls
to the same function must not be in flight at once. For a pick handler that is
one call per click — the point. If you need re-entrant use (e.g. a pick
triggered from inside another pick), compile with `{ reentrant: true }`, which
declares the variables inside the callable instead.

### Sampling

Float textures sample at normalized coordinates; integer textures
(`isampler*`/`usampler*`) fetch at texel coordinates. Data is a flat array,
four channels a texel unless `channels` says otherwise.

How a float texture is read comes with the data, not with the compiled
function, the same way it belongs to the sampler rather than to the shader on a
GPU:

```typescript
ctx.textures[map.name] = {
  data,
  width: 64,
  height: 64,
  channels: 4, // 4 (the default) | 3 | 2 | 1
  magFilter: "linear", // "nearest" (the default) | "linear"
  wrapS: "repeat", // "clamp" (the default) | "repeat" | "mirror"
  wrapT: "repeat", // and wrapR for a 3D texture
};
```

`channels` is the stride from one texel to the next, so a single-channel volume
— a voxel grid uploaded as `R8UI` — is read a byte at a time rather than four.
The channels a texel does not store read the way a sampler reports them on a
device: zero for green and blue, one for alpha.

`"linear"` blends the neighbouring texels — bilinear for a 2D texture,
trilinear for a 3D one — and the wrapping modes match their GPU counterparts,
so a tiling texture tiles here too. `minFilter` is accepted and ignored:
choosing between the two needs the footprint of the pixel being shaded, which a
single CPU evaluation has no way to know.

These are the fields of a `SamplerState`, which is what the scene layer derives
from a texture's three.js constants for the two renderers — so the same reading
of what a texture asked for drives all three backends.

Two things a GPU does that this does not: there is no mip chain, so
`textureLod`'s level argument is ignored, and a texel fetch neither filters nor
wraps — as on a GPU, where `texelFetch`/`textureLoad` read the texel or nothing.

`@random-mesh/rmsl/scene` textures state this in three.js's constants
(`format`, `magFilter`, `wrapS`, …); `@random-mesh/rmsl/test` hands them to the
same `samplerState` and `textureChannels` the renderers use, so a material
tested on the CPU samples the way its renderer does — down to a mipmapped
filter spelling counting as its base filter.

**8-bit data read through a float sampler comes back as 0–1**, the way it does
on both backends: an 8-bit texture is uploaded as a normalized format, and the
sampler divides by 255 on the way out. Give a `Uint8Array` (or a
`Uint8ClampedArray`) — what a `DataTexture` holds — and the CPU callable divides
too, so the same graph produces the same value here as on screen. Data in any
other array is taken as the value it already is, so float data passes through
untouched. An integer sampler fetches raw texels on a GPU as well, and is left
alone.

### Testing with it

Unit-testing a shader's logic is what this target makes possible without a
device, and `@random-mesh/rmsl/test` is the ergonomic layer over it — inputs
bound by node instead of by slot name, a grid of fragments in one call, and
tolerant comparisons. See [Testing](testing.md).

### Caveats

- The JS target computes f64, the GPUs f32 — results can differ by a ULP or
  two, which is why the evaluation tests compare within a tolerance.
- `compileJS` uses `new Function`, which a strict Content-Security-Policy can
  block; `compileJSFn` returns the source so you can embed it in an approved
  context instead.
- `Discard()` compiles to `return null;` — the host treats `null` as
  "no fragment".
- `isamplerCube`/`usamplerCube` are not supported yet — `texelFetch` has no
  cube-map overload in GLSL, so there's no direct-fetch equivalent to build on.

## Type Mappings

### GLSL types

| RMSL                                       | GLSL          |
| ------------------------------------------ | ------------- |
| `float`, `vec2`, `vec3`, `vec4`            | same          |
| `int`, `uint`, `bool`                      | same          |
| `ivec2`, `ivec3`, `ivec4`                  | same          |
| `uvec2`, `uvec3`, `uvec4`                  | same          |
| `mat2`–`mat4`, `mat2x3`, etc.              | same          |
| `sampler2D`                                | `sampler2D`   |
| `sampler3D`                                | `sampler3D`   |
| `samplerCube`                              | `samplerCube` |
| `isampler2D`, `isampler3D`, `isamplerCube` | same          |
| `usampler2D`, `usampler3D`, `usamplerCube` | same          |

### WGSL types

| RMSL                                       | WGSL                                                      |
| ------------------------------------------ | --------------------------------------------------------- |
| `float`                                    | `f32`                                                     |
| `vec2`                                     | `vec2<f32>`                                               |
| `vec3`                                     | `vec3<f32>`                                               |
| `vec4`                                     | `vec4<f32>`                                               |
| `int`                                      | `i32`                                                     |
| `uint`                                     | `u32`                                                     |
| `bool`                                     | `bool`                                                    |
| `ivec2`, `ivec3`, `ivec4`                  | `vec2<i32>`, `vec3<i32>`, `vec4<i32>`                     |
| `uvec2`, `uvec3`, `uvec4`                  | `vec2<u32>`, `vec3<u32>`, `vec4<u32>`                     |
| `mat2`                                     | `mat2x2<f32>`                                             |
| `mat3`                                     | `mat3x3<f32>`                                             |
| `mat4`                                     | `mat4x4<f32>`                                             |
| `mat2x3`                                   | `mat2x3<f32>`                                             |
| etc.                                       | `<N>x<M><f32>`                                            |
| `sampler2D`                                | `texture_2d<f32>`                                         |
| `sampler3D`                                | `texture_3d<f32>`                                         |
| `samplerCube`                              | `texture_cube<f32>`                                       |
| `isampler2D`, `isampler3D`, `isamplerCube` | `texture_2d<i32>`, `texture_3d<i32>`, `texture_cube<i32>` |
| `usampler2D`, `usampler3D`, `usamplerCube` | `texture_2d<u32>`, `texture_3d<u32>`, `texture_cube<u32>` |

### Integer texture sampling

Integer textures are not filterable in either language, so `texture()`/`textureLod()`
compile to an unfiltered fetch at **integer texel coordinates**:

| RMSL                                | GLSL                             | WGSL                                  |
| ----------------------------------- | -------------------------------- | ------------------------------------- |
| `isampler2D.texture(ivec2)`         | `texelFetch(s, ivec2, 0)`        | `textureLoad(t, vec2<i32>, 0i)`       |
| `usampler3D.textureLod(uvec3, lod)` | `texelFetch(s, ivec3, int(lod))` | `textureLoad(t, vec3<i32>, i32(lod))` |

The result is `ivec4` for `isampler*` and `uvec4` for `usampler*`. WGSL binds no
sampler for these — only the texture — because `textureLoad` does not take one.

## Constant Folding

When all inputs to an operation are literal values, RMSL evaluates it at compile time:

```typescript
// These all fold to literal values:
let x = float(2.0).pow(float(3.0)); // -> 8
let y = int(5).add(int(3)); // -> 8
let z = float(0.5).sin(); // -> ~0.479
```

This applies to: `add`, `sub`, `mul`, `div`, `mod`, `negate`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `abs`, `sign`, `floor`, `ceil`, `round`, `trunc`, `fract`, `sqrt`, `inverseSqrt`, `exp`, `log`, `exp2`, `log2`, `pow`, `min`, `max`, `dot` (on scalars, via `lengthSq`).

## Type Coercion (WGSL)

When mixing `int`/`uint` with `float` in binary operations, the WGSL compiler inserts explicit `f32()`/`i32()` casts, since WGSL does not perform implicit numeric conversion. Mixing `int` with `uint` converts to the node's declared type (its first operand's); mixing signed and unsigned at the API level is spelled out with `.toInt()`/`.toUint()`, since neither backend shares a type for the two.
