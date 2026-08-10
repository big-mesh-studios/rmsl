# Compilation

RMSL compiles node graphs to **GLSL** (WebGL 2 / OpenGL ES 3.0) or **WGSL** (WebGPU).

## Compiler API

```typescript
import { compileGLSL, compileWGSL } from "rmsl";
```

### Fragment shader (default)

```typescript
compileGLSL(root)           // fragment shader
compileGLSL.fragment(root)  // explicit fragment
compileWGSL(root)           // fragment shader
compileWGSL.fragment(root)  // explicit fragment
```

### Vertex shader

```typescript
compileGLSL.vertex(root)
compileWGSL.vertex(root)
```

### Multiple return values

```typescript
let prog = Fn(() => {
  let a = float(1).toVar();
  let b = float(2).toVar();
  return [a, b];
});
let [a, b] = prog();
compileGLSL.vertex([a, b]);  // pass array of roots
```

## GLSL Output

### Fragment shader

```glsl
#version 300 es
precision highp float;

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
console.log(uTime.name);      // outputs "_rmsl_u0"
let x = uTime.add(1.0);      // methods and swizzles are available directly
```

### Varyings

`varying("vec3")` produces `out vec3 _rmsl_vN;` in vertex and `in vec3 _rmsl_vN;` in fragment.

Access `.name` for the generated name; methods and swizzles are available directly:

```typescript
let v = varying("vec3");
console.log(v.name);       // outputs "_rmsl_v0"
let x = v.x;               // swizzles are available directly
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
console.log(pos.name);       // outputs "_rmsl_a0"
let x = pos.x;               // swizzles are available directly
```

This produces:

```wgsl
struct VertexInput {
  @location(0) _rmsl_a0: vec3<f32>,
};
```

## Binding Model (WGSL)

| Resource | Group | Binding |
|----------|-------|---------|
| Uniforms | `@group(0)` | `@binding(N)` |
| Textures | `@group(1)` | `@binding(N)` |
| Samplers | `@group(2)` | `@binding(N)` |

## Standalone Function Compilers

For use with Three.js `glslFn`/`wgslFn` or other embedding scenarios, RMSL provides `compileGLSLFn` and `compileWGSLFn` to compile individual functions with custom names and parameters:

```typescript
import { compileGLSLFn, compileWGSLFn, float, var_ } from "rmsl";

let glsl = compileGLSLFn(
  (a, b) => a.add(b).sin(),
  { name: "myFunc", params: [{ name: "a", type: "float" }, { name: "b", type: "float" }] },
);
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
    return v.mult(u);
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

## Type Mappings

### GLSL types

| RMSL | GLSL |
|------|------|
| `float`, `vec2`, `vec3`, `vec4` | same |
| `int`, `uint`, `bool` | same |
| `ivec2`, `ivec3`, `ivec4` | same |
| `uvec2`, `uvec3`, `uvec4` | same |
| `mat2`–`mat4`, `mat2x3`, etc. | same |
| `sampler2D` | `sampler2D` |
| `sampler3D` | `sampler3D` |
| `samplerCube` | `samplerCube` |
| `isampler2D`, `isampler3D`, `isamplerCube` | same |
| `usampler2D`, `usampler3D`, `usamplerCube` | same |

### WGSL types

| RMSL | WGSL |
|------|------|
| `float` | `f32` |
| `vec2` | `vec2<f32>` |
| `vec3` | `vec3<f32>` |
| `vec4` | `vec4<f32>` |
| `int` | `i32` |
| `uint` | `u32` |
| `bool` | `bool` |
| `ivec2`, `ivec3`, `ivec4` | `vec2<i32>`, `vec3<i32>`, `vec4<i32>` |
| `uvec2`, `uvec3`, `uvec4` | `vec2<u32>`, `vec3<u32>`, `vec4<u32>` |
| `mat2` | `mat2x2<f32>` |
| `mat3` | `mat3x3<f32>` |
| `mat4` | `mat4x4<f32>` |
| `mat2x3` | `mat2x3<f32>` |
| etc. | `<N>x<M><f32>` |
| `sampler2D` | `texture_2d<f32>` |
| `sampler3D` | `texture_3d<f32>` |
| `samplerCube` | `texture_cube<f32>` |
| `isampler2D`, `isampler3D`, `isamplerCube` | `texture_2d<i32>`, `texture_3d<i32>`, `texture_cube<i32>` |
| `usampler2D`, `usampler3D`, `usamplerCube` | `texture_2d<u32>`, `texture_3d<u32>`, `texture_cube<u32>` |

### Integer texture sampling

Integer textures are not filterable in either language, so `texture()`/`textureLod()`
compile to an unfiltered fetch at **integer texel coordinates**:

| RMSL | GLSL | WGSL |
|------|------|------|
| `isampler2D.texture(ivec2)` | `texelFetch(s, ivec2, 0)` | `textureLoad(t, vec2<i32>, 0i)` |
| `usampler3D.textureLod(uvec3, lod)` | `texelFetch(s, ivec3, int(lod))` | `textureLoad(t, vec3<i32>, i32(lod))` |

The result is `ivec4` for `isampler*` and `uvec4` for `usampler*`. WGSL binds no
sampler for these — only the texture — because `textureLoad` does not take one.

## Constant Folding

When all inputs to an operation are literal values, RMSL evaluates it at compile time:

```typescript
// These all fold to literal values:
let x = float(2.0).pow(float(3.0));   // -> 8
let y = int(5).add(int(3));           // -> 8
let z = float(0.5).sin();             // -> ~0.479
```

This applies to: `add`, `sub`, `mult`, `div`, `mod`, `negate`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `abs`, `sign`, `floor`, `ceil`, `round`, `trunc`, `fract`, `sqrt`, `inversesqrt`, `exp`, `log`, `exp2`, `log2`, `pow`, `min`, `max`, `dot` (on scalars, via `lengthSq`).

## Type Coercion (WGSL)

When mixing `int`/`uint` with `float` in binary operations, the WGSL compiler inserts explicit `f32()`/`i32()` casts, since WGSL does not perform implicit numeric conversion. Mixing `int` with `uint` converts to the node's declared type (its first operand's); mixing signed and unsigned at the API level is spelled out with `.toInt()`/`.toUint()`, since neither backend shares a type for the two.
