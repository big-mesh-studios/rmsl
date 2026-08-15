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
  `Mesh`, lights (`AmbientLight`, `DirectionalLight`, `PointLight`).
- **Geometry** — `BufferGeometry` + `BufferAttribute`, and the primitives
  `BoxGeometry`, `SphereGeometry`, `PlaneGeometry`, `CylinderGeometry`,
  `ConeGeometry`, `TorusGeometry`, `CircleGeometry` (normals and UVs included).
- **Math** — `Vector2/3/4`, `Matrix3/4`, `Quaternion`, `Euler`, `Color`,
  `Spherical`, `MathUtils`, all in the column-major `number[]`/`Float32Array`
  convention the rest of the library already uses.
- **Textures** — `Texture`, `DataTexture`.
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

`new WebGLRenderer(canvas?, options?)` — WebGL2. `setClearColor`, `setSize`,
`setAnimationLoop`, `render(scene, camera)`. Programs are compiled per material
and cached; geometry buffers per geometry; uniforms are uploaded per draw,
grouped by scope:

- **camera** — `projectionMatrix`, `viewMatrix`, `cameraPosition`
- **object** — `modelMatrix`, `normalMatrix` (per mesh)
- **material** — color/roughness/metalness/emissive, lights, and any custom
  uniforms declared through the builder

### WebGPURenderer

`await WebGPURenderer.init(canvas?)`. Same API. Uniform values are packed into
per-program ring buffers using the same layout the WGSL compiler emits, so
per-draw writes never race the previous draw.

### Lights

Ambient, directional and point lights in the scene are collected at material
build time. Each lit material declares a fixed set of light uniforms; when the
scene's light set changes the renderer rebuilds the affected programs. Point
lights support `distance` (0 = no cutoff) and `decay` (default 2).

## Limitations

- The lighting model is deliberately simple: Lambert diffuse plus a
  three.js-style simplified GGX specular, no image-based lighting or shadows.
- Custom uniforms must be reachable through the builder so the renderer knows
  their values; a `uniform()` declared deep inside an escape-hatch graph that
  the builder never sees will not be bound.
- `WebGPURenderer` texture support covers `DataTexture` (and the WebGL renderer
  additionally accepts `HTMLImageElement`s).
- The full test suite validates every material shader on real Chromium/Dawn
  drivers with `RMSL_GPU=1` (see `CONTRIBUTING.md`).
