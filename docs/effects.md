# Post-processing effects

RMSL ships a port of the post-processing effects from three.js's
`examples/jsm/tsl/display`, importable from the `@random-mesh/rmsl/effects`
subpath:

```typescript
import { sepia, fxaa, gaussianBlur, crt } from "@random-mesh/rmsl/effects";
```

Each effect is a **pure RMSL node graph** — a function of sampler uniforms and
parameters that returns a color node, compiling to GLSL, WGSL and JS like
anything else in the DSL. There is no renderer and no GL context inside the
package; effects sample the textures you hand them and produce a color, and
your own render loop draws the fullscreen quad.

## How effects map from three.js

Three.js wraps every effect in a renderer-bound node that owns render targets,
per-frame updates and a fullscreen quad. RMSL keeps only the shader: the
renderer machinery is replaced with explicit arguments.

| three.js | RMSL |
|----------|------|
| `convertToTexture(node)` | Nothing to convert — pass a `uniform("sampler2D")` (or `"sampler3D"` for a LUT) directly. |
| `uv()` / `screenUV()` | `uv()` — the normalized screen position (`fragCoord / screenSize`). |
| `screenCoordinate` | `screenCoordinate()` / `fragCoord()` — pixel position. |
| `uniform(1.57)` (value-owned) | A plain number or node parameter. |
| `uniform(new Vector2())` (resolution) | Derived in-shader from `textureSize(...)`. |
| `time` | `time()` — a shared `float` uniform the host updates. |
| `passTexture` / `RenderTarget` | A `PassGraph` of data-only pass descriptors (see below). |
| `Loop({ start, end, condition }, ({ i }) => ...)` | RMSL `For(...)` / `Loop(count, (i) => ...)`. |
| `.toConst()` | Free — RMSL constant-folds. |

## Single-pass effects

These take an input color node or a sampler and return the output color:

| Function | Input | Notes |
|----------|-------|-------|
| `sepia(color)` | color | |
| `bleach(color, opacity?)` | color | Bleach bypass. |
| `dotScreen(color, angle?, scale?)` | color | Uses the screen coordinate. |
| `film(color, intensity?, uv?)` | color | Film grain; needs `time()`. |
| `circle(scale?, softness?, coord?)` | — | Radial gradient utility. |
| `rgbShift(texture, amount?, angle?)` | sampler | |
| `chromaticAberration(texture, strength?, center?, scale?)` | sampler | |
| `sobel(texture)` | sampler | Edge detection. |
| `lut3D(color, lut3DSampler, size, intensity?)` | color + `sampler3D` | 3D LUT grading. |
| `transition(textureA, textureB, mixTexture, ratio, threshold, useTexture)` | samplers | |
| `crt(texture, options?)` | sampler | Barrel distortion + bleeding + scanlines + vignette. |
| `motionBlur(texture, velocity, numSamples?)` | sampler + `vec2` | |
| `sharpen(texture, sharpness?, denoise?)` | sampler | RCAS. |
| `fxaa(texture)` | sampler | Requires sRGB input. |
| `boxBlur(texture, options?)` | sampler | |
| `hashBlur(texture, bluramount?, options?)` | sampler | |
| `radialBlur(texture, options?)` | sampler | |

Textures are declared by the caller and passed in:

```typescript
import { Fn, uniform, compileGLSL } from "rmsl";
import { fxaa } from "@random-mesh/rmsl/effects";

let colorTex = uniform("sampler2D");
let glsl = compileGLSL(fxaa(colorTex));
```

## Multi-pass effects: pass graphs

Effects that need more than one render pass (the separable gaussian blur,
bloom) return a **`PassGraph`** — pure data, not a renderer:

```typescript
interface PassDescriptor {
  name: string;
  color: Node<"vec4">;                    // the pass's fragment color
  inputs: Record<string, Sampler2DLike>;  // samplers to bind for this pass
  size?: [number, number];                // optional render-target size in pixels
  scale?: number;                         // optional size as a fraction of the input
}

interface PassGraph {
  passes: PassDescriptor[];
  output: string; // name of the pass whose output is the result
}
```

`gaussianBlur(texture, direction?, sigma?, options?)` returns a two-pass graph
(horizontal then vertical). Your render loop draws one fullscreen quad per
pass, binds each pass's `inputs` (pass 1's input is pass 0's render target),
and reads the pass named by `graph.output`:

```typescript
import { gaussianBlur } from "@random-mesh/rmsl/effects";

let graph = gaussianBlur(uniform("sampler2D"), [1, 1], 4);

for (const pass of graph.passes) {
  let src = compileGLSL(pass.color); // fullscreen quad fragment shader
  // bind pass.inputs.input -> previous texture (or the scene for pass 0)
  // draw into a render target
}
```

`premultipliedGaussianBlur(texture, direction?, sigma?)` is the
premultiplied-alpha variant.

### Bloom — a 12-pass graph with shrinking targets

`bloom(texture, options?)` ports three.js's UnrealBloom-style pipeline
faithfully: a luminance high pass, five progressively half-resolution
separable gaussian blurs (kernels `[6, 10, 14, 18, 22]`), and a composite that
sums the five blurred mips with factors `[1.0, 0.8, 0.6, 0.4, 0.2]`.

```typescript
import { bloom } from "@random-mesh/rmsl/effects";

let graph = bloom(sceneColorTex, { strength: 1.5, radius: 0.3, threshold: 0.8 });
// graph.passes -> 12 descriptors, graph.output === "bloom.composite"
```

Options: `strength`, `radius` (in `[0, 1]`), `threshold`, `smoothWidth`,
`highPassFn` (a custom filter, e.g. anamorphic) and `tints` (per-mip colors).
Numbers fold into the shaders; pass RMSL nodes to drive them with uniforms.

The composite output is the glow on its own — add it to your scene color, as
three.js does with `scenePassColor.add(bloomPass)`.

#### How a bloom executor sizes the passes

Because the mip chain shrinks each step, bloom passes carry a `scale` on the
`PassDescriptor`: the pass renders at `round(firstInputTextureSize × scale)`
(its first input's size when `scale` is absent, and the drawing buffer when it
has no inputs). The high pass is `0.5`, the vertical passes and mip 0 are `1`,
and each halving horizontal pass is `0.5`. `size` (absolute pixels) wins over
`scale` when both are set. Pass inputs are keyed by the *producer pass's name*
for internal links — `{ "bloom.mip0.vertical": ... }` — and by any other name
(e.g. `input`) for external textures, so an executor can bind a pass's output
target to the next pass mechanically.

## What is not ported yet

The renderer-bound families from `examples/jsm/tsl/display` are not yet
ported: temporal/resampling effects (FSR1, TAAU, TRAAN, temporal reproject,
denoise), G-buffer reads (GTAO, SSGI, SSR, SSS, DoF, Outline, Godrays,
Lensflare), SMAA, SSAA, and the stereo/composite passes (which need a stereo
camera). The single-pass color family, the blur family, and bloom are.

## Testing

`src/effects/effects-usage.test.ts` compiles every effect to both backends
through the recording compilers, so the whole set is handed to real GLSL and
WGSL drivers in the GPU validation layer.
