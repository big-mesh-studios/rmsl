# Scene graph & node-based materials

`@random-mesh/rmsl/scene` is a scene-graph layer on top of the RMSL shader DSL.
It provides three.js-style objects (`Scene`, `Mesh`, `PerspectiveCamera`,
lights, geometry primitives, math classes) and — the headline feature —
**node-based materials**: a material is an RMSL node graph, compiled to GLSL or
WGSL by the same compiler the rest of the library uses.

```typescript
import { WebGLRenderer, Scene, Mesh, PerspectiveCamera,
  BoxGeometry, MeshStandardMaterial, AmbientLight, DirectionalLight } from "@random-mesh/rmsl/scene";

const renderer = new WebGLRenderer();
renderer.setClearColor(0x101318);

const scene = new Scene();
scene.add(new AmbientLight(0xffffff, 0.25));
const sun = new DirectionalLight(0xfff3e0, 2.5);
sun.position.set(6, 10, 8);
scene.add(sun);

const mesh = new Mesh(new BoxGeometry(), new MeshStandardMaterial({ color: 0xff5533 }));
scene.add(mesh);

const camera = new PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 100);
camera.position.set(3, 2, 4);
camera.lookAt(0, 0, 0);

renderer.setAnimationLoop(() => renderer.render(scene, camera));
```

## What's here

- **Scene graph** — `Object3D` (position/quaternion/scale, world matrices),
  `Group`, `Scene`, `Camera` / `PerspectiveCamera` / `OrthographicCamera`,
  `Mesh`, `InstancedMesh`, lights (`AmbientLight`, `DirectionalLight`,
  `PointLight`), and the wide-line objects `LineSegments2` / `Line2` (see
  below).
- **Geometry** — `BufferGeometry` + `BufferAttribute` (with `stepMode` for
  instanced attributes), the primitives `BoxGeometry`, `SphereGeometry`,
  `PlaneGeometry`, `CylinderGeometry`, `ConeGeometry`, `TorusGeometry`,
  `CircleGeometry` (normals and UVs included), and `LineSegmentsGeometry` /
  `LineGeometry` for lines.
- **Math** — `Vector2/3/4`, `Matrix3/4`, `Quaternion`, `Euler`, `Color`,
  `Spherical`, `MathUtils`, all in the column-major `number[]`/`Float32Array`
  convention the rest of the library already uses.
- **Textures** — `Texture`, `DataTexture`, with three.js's filtering and
  wrapping constants (see [Texture sampling](#texture-sampling) and
  [Texture lifetime](#texture-lifetime)).
- **Renderers** — `WebGLRenderer` (WebGL2) and `WebGPURenderer` (WebGPU).

## Node-based materials

`NodeMaterial` and its subclasses (`MeshBasicMaterial`, `MeshLambertMaterial`,
`MeshStandardMaterial`) describe a surface as an RMSL node graph. Each slot is
a node — or a function of the builder that returns one:

```typescript
const material = new MeshStandardMaterial({ color: 0xeeeeee, roughness: 0.4 });

// The color is a node graph, evaluated per fragment.
material.colorNode = (b) => mix(vec3(0.1, 0.2, 0.5), vec3(0.9, 0.4, 0.1), b.normalWorld.y.add(1).mul(0.5));
material.roughnessNode = () => float(0.3);
```

Available slots: `colorNode`, `opacityNode`, `roughnessNode`, `metalnessNode`,
`emissiveNode`, `normalNode`, `positionNode`, `uvNode`.

Every material also carries a `precision` property (`"highp" | "mediump" |
"lowp" | null`, default `null`), mirroring three.js's `Material.precision`: it
overrides the renderer's default shader precision (see WebGLRenderer below), and
can be passed to any material constructor. Assigning it flags the program for a
rebuild.

```typescript
const material = new MeshStandardMaterial({ color: 0xeeeeee, precision: "mediump" });
// or, later:
material.precision = "lowp";  // needsUpdate is set for you
```

### Built-in accessors

Inside a slot (or an escape-hatch function) the `Builder` (`b`) exposes the
values a scene material needs, resolving them to ordinary RMSL
attribute/uniform/varying nodes:

| Accessor | Kind | Meaning |
|----------|------|---------|
| `b.position`, `b.normal`, `b.uv` | attribute | geometry inputs |
| `b.positionWorld`, `b.normalWorld` | varying | world space, from the vertex stage |
| `b.cameraPosition` | uniform | camera position (world) |
| `b.projectionMatrix`, `b.viewMatrix` | uniform | camera matrices |
| `b.modelMatrix`, `b.normalMatrix` | uniform | object matrices |
| `b.modelViewMatrix` | computed | `viewMatrix * modelMatrix` |
| `b.viewDirection` | computed | normalized direction to the camera |

### Escape hatches

`vertexNode` replaces the whole vertex stage; it must return clip space.
`fragmentNode` replaces the whole fragment stage; it must return the final
color (a `vec4`):

```typescript
const material = new MeshBasicMaterial();
material.fragmentNode = (b) => Fn(() => vec4(b.uv.x, b.uv.y, 0.5, 1))();
```

### How a material becomes a shader

`material.build(scene)` compiles the graph and returns a `MaterialProgram`:
the vertex and fragment roots plus the exact set of uniforms, attributes,
varyings and sampler bindings the compiled shaders reference — filtered to what
is *actually used*, which is also what the compilers emit. The renderers use
this to bind geometry attributes, upload uniforms (grouped by scope: camera,
object, material) and build the WebGPU uniform-buffer layout.

## Renderers

### WebGLRenderer

`new WebGLRenderer(canvas?, options?)` — WebGL2. Options: `antialias`,
`depth`, and `precision` (`"highp" | "mediump" | "lowp"`, default `"highp"`).
`setClearColor`, `setSize`, `setAnimationLoop`, `render(scene, camera)`.
Programs are compiled per material and cached; geometry buffers per geometry;
uniforms are uploaded per draw, grouped by scope:

- **camera** — `projectionMatrix`, `viewMatrix`, `cameraPosition`
- **object** — `modelMatrix`, `normalMatrix` (per mesh)
- **material** — color/roughness/metalness/emissive, lights, and any custom
  uniforms declared through the builder

`render(scene, camera, target?)` draws onto a `WebGLRenderTarget` instead of
the canvas when one is given, keeping its own color and depth. Read a
target's pixels back with `readPixels(target, out?)`, which copies the GPU's
answer to a `Uint8Array` and stalls the pipeline until it is ready — fine for
an occasional sample, not for every frame. `readPixelsAsync(target, out?)`
reads the same pixels without the stall, at the cost of not having them yet
when the call returns: it resolves the returned promise once a GPU fence
confirms the transfer landed, typically a frame or a few later.

### WebGPURenderer

`await WebGPURenderer.init(canvas?)`. Same API. Uniform values are packed into
per-program ring buffers using the same layout the WGSL compiler emits, so
per-draw writes never race the previous draw.

### Texture sampling

How a texture is read is set on the texture, with three.js's fields and
constants — and three.js's numeric values, so a constant carried over from a
three.js texture means the same thing here:

```typescript
import { DataTexture, NearestFilter, RepeatWrapping } from "@random-mesh/rmsl/scene";

const texture = new DataTexture(pixels, 16, 16);
texture.magFilter = NearestFilter;   // square texels, for pixel art
texture.minFilter = NearestFilter;
texture.wrapS = RepeatWrapping;      // the image tiles past 1
texture.wrapT = RepeatWrapping;
```

- `magFilter`/`minFilter` — `LinearFilter` (the default) or `NearestFilter`.
- `wrapS`/`wrapT`/`wrapR` — `ClampToEdgeWrapping` (the default),
  `RepeatWrapping` or `MirroredRepeatWrapping`. `wrapR` is the third axis of a
  3D texture and is ignored for a 2D one.

Both renderers read these when they upload, so a change made after the first
render needs `texture.needsUpdate = true` to take effect — the same rule as a
change to the image.

Three things to know:

- **An integer texture is always read with nearest**, whatever it asks for:
  `isampler*`/`usampler*` textures are not filterable in either shading
  language. Wrapping still applies, being a property of the coordinate rather
  than of the filter.
- **The mipmapped `minFilter` constants are accepted but treated as their base
  filter** — `LinearMipmapLinearFilter` samples like `LinearFilter` — because
  no renderer builds a mip chain yet. That is also why the default `minFilter`
  is `LinearFilter` rather than three.js's `LinearMipmapLinearFilter`. See
  [issue #3](https://github.com/big-mesh-studios/rmsl/issues/3).
- **The CPU target follows them too.** A material tested with
  [`@random-mesh/rmsl/test`](testing.md) filters and wraps the way its renderer
  does, so what a test measures is what the renderer draws. It has no mip chain
  either, so `textureLod`'s level is ignored there.

### Texture lifetime

A renderer creates the GPU texture behind a `Texture` the first time it draws
with it, and uploads the image again on the next render whenever
`needsUpdate` is set, at a new size if the image changed shape. Pointing a
material at a *different* `Texture` object instead needs
`material.needsUpdate = true`, so the renderer rebuilds what the shader reads
from.

`texture.dispose()` gives the GPU texture back:

```typescript
const texture = new DataTexture(pixels, 256, 256);
// ... draw with it ...
texture.dispose(); // every renderer that uploaded it frees its copy
```

`dispose()` dispatches a `dispose` event, and each renderer that uploaded the
texture deletes its own GPU object — so a texture shared by a WebGL and a
WebGPU renderer frees both. The `Texture` object stays usable afterwards:
drawing with it again uploads the image to a fresh GPU texture. Use it for a
texture that drops out of a scene — an atlas swapped for another, a layer torn
down — rather than waiting for `renderer.dispose()`, which frees everything the
renderer holds at once.

### Geometry lifetime

A renderer creates the vertex and index buffers behind a `BufferGeometry` the
first time it draws with it, and keys them by the geometry object — so filling
the same geometry again (`needsUpdate` on an attribute) reuses the buffers it
already has, and a growing attribute re-allocates only when the data no longer
fits.

That cache holds the geometry, so a geometry dropped from the scene keeps its
buffers, and the arrays its attributes point at, alive for as long as the
renderer lives. `geometry.dispose()` gives both back:

```typescript
const geometry = new BufferGeometry();
// ... draw with it ...
geometry.dispose(); // every renderer that uploaded it frees its buffers
```

Like a texture's, it dispatches a `dispose` event that each renderer answers for
its own GPU objects, and the geometry object stays usable: drawing with it again
uploads its attributes into fresh buffers. Use it for a geometry that leaves the
scene for good — a chunk of terrain scrolled out of the world, a mesh torn down
— rather than waiting for `renderer.dispose()`.

### Lights

Ambient, directional and point lights in the scene are collected at material
build time. Each lit material declares a fixed set of light uniforms; when the
scene's light set changes the renderer rebuilds the affected programs. Point
lights support `distance` (0 = no cutoff) and `decay` (default 2).

## Wide lines

`LineSegments2` draws line segments with arbitrary width, ported from
three.js's `examples/jsm/lines/webgpu` module and rendered by **both** the
WebGL2 and WebGPU renderers (the shaders are node graphs, so they compile to
GLSL and WGSL from the same material):

```typescript
import { Scene, PerspectiveCamera, LineSegments2, LineSegmentsGeometry,
  Line2NodeMaterial, Line2, LineGeometry } from "@random-mesh/rmsl/scene";

const scene = new Scene();

// A single pair of segments.
const segments = new LineSegments2(
  new LineSegmentsGeometry().setPositions([-1, 0, 0, 1, 0, 0, 0, -1, 0, 0, 1, 0]),
  new Line2NodeMaterial({ color: 0x44aaff, linewidth: 3 }),
);
scene.add(segments);

// A polyline built from consecutive points.
const line = new Line2(new LineGeometry([-2, -1, 0, 0, 0, 0, 2, -1, 0]),
  new Line2NodeMaterial({ color: 0xff8833, linewidth: 2, worldUnits: true }));
scene.add(line);
```

- **`linewidth`** — width in device pixels by default; with `worldUnits: true`
  it is measured in world units and the ribbon stays a constant world width.
- **`dashed`** — requires `line.computeLineDistances()` so the geometry can
  record accumulated distances; configure `dashSize`, `gapSize`, `dashOffset`,
  `dashScale`.
- **`vertexColors`** — requires `geometry.setColors(...)`; multiplies the color
  by per-segment start/end colors.
- Node slots `colorNode`, `lineWidthNode`, `dashSizeNode`, `gapSizeNode`,
  `dashOffsetNode`, `dashScaleNode`, `offsetNode` override the material
  properties, like the other node materials.
- `alphaToCoverage` is accepted for three.js parity but currently renders the
  MSAA-less hard-edge variant (the renderers expose no sample count yet), and
  transparent lines are not supported (`NoBlending`).

### How the wide-line shader works

Each segment is an *instance*: the geometry stores `instanceStart`/`instanceEnd`
(vec3, `stepMode: "instance"`) per segment, and the material expands the shared
quad strip into a screen- or world-width ribbon in the vertex stage, then
discards fragments outside the line (round endcaps, dashes, world-units
distance) in the fragment stage. This required plumbing **instanced attributes**
through the scene graph: `BufferAttribute.stepMode` and
`BufferGeometry.instanceCount`, with the renderers issuing instanced draws
(`drawElementsInstanced`, `draw(..., instanceCount)`). A `resolution` uniform
(scoped to the renderer) feeds the pixel-width expansion.

## Instancing

`InstancedMesh` draws one geometry `count` times with a shared material, each
instance carrying its own transform (and optionally color) — the way to render
a large crowd of identical objects in one draw call. Like three.js, the
per-instance data lives on the **object**, not the geometry:

```typescript
import { Scene, PerspectiveCamera, InstancedMesh, BoxGeometry,
  MeshStandardMaterial, Matrix4, Color, AmbientLight, DirectionalLight }
  from "@random-mesh/rmsl/scene";

const scene = new Scene();
scene.add(new AmbientLight(0xffffff, 0.25));
const sun = new DirectionalLight(0xfff3e0, 2);
sun.position.set(4, 8, 6);
scene.add(sun);

const mesh = new InstancedMesh(new BoxGeometry(0.4, 0.4, 0.4),
  new MeshStandardMaterial({ roughness: 0.3 }), 100);

// One matrix per instance; the instance transform applies before the mesh's
// own world transform. After editing, flag the update like three.js.
for (let i = 0; i < mesh.count; i++) {
  mesh.setMatrixAt(i, new Matrix4().makeTranslation((i % 10 - 5) * 1.1, Math.floor(i / 10) * 1.1, 0));
}
mesh.instanceMatrix.needsUpdate = true;

// Optional per-instance colors, tinting the material's color in the fragment
// stage. Same manual needsUpdate convention.
mesh.setColorAt(0, new Color().setHex(0xff5533));
mesh.instanceColor.needsUpdate = true;

scene.add(mesh);
```

- **`setMatrixAt(i, matrix)` / `getMatrixAt(i, target)`** — per-instance local
  transforms, initialized to the identity. `instanceMatrix` is a
  `BufferAttribute` with `stepMode: "instance"` and 16 floats per instance.
- **`setColorAt(i, color)` / `getColorAt(i, target)`** — per-instance colors.
  Calling `setColorAt` creates the `instanceColor` attribute lazily; uncolored
  instances read back white. `instanceColor` is only compiled into the shader
  when at least one instance has a color.
- The material shader multiplies the local position by `instanceMatrix` (then
  by the mesh's `modelMatrix`) and the normal by `mat3(instanceMatrix)`, and
  tints the fragment color by `instanceColor`. The renderer draws `count`
  instances in a single instanced draw, and compiles one program per
  instancing variant of a material — so a plain mesh and an `InstancedMesh`
  sharing one material each get the shader that matches it.

## Testing a material without a renderer

A material is a node graph, so what it computes can be checked on the CPU — no
canvas, no device, no readback:

```typescript
import { fromProgram } from "@random-mesh/rmsl/test";

const shade = fromProgram(material.build(scene, {}));
shade({ varyings: { normalWorld: [0, 1, 0], positionWorld: [0, 0, 0] } });
```

See [Testing](testing.md).

## Limitations

- `LineSegments2.raycast` and `InstancedMesh.raycast` are not ported yet (no
  `Raycaster`/`Ray` in the scene graph), and `InstancedMesh` has no
  `computeBoundingBox`/`computeBoundingSphere` (no `Box3`/`Sphere`); pick
  against instances is a follow-up.
- The lighting model is deliberately simple: Lambert diffuse plus a
  three.js-style simplified GGX specular, no image-based lighting or shadows.
- Custom uniforms must be reachable through the builder so the renderer knows
  their values; a `uniform()` declared deep inside an escape-hatch graph that
  the builder never sees will not be bound.
- `WebGPURenderer` texture support covers `DataTexture` (and the WebGL renderer
  additionally accepts `HTMLImageElement`s).
- The test suite validates every material shader on real Chromium/Dawn drivers
  by default; `RMSL_SKIP_GPU=1` turns that off (see `CONTRIBUTING.md`).
