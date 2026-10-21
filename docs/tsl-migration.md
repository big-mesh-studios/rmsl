# Migrating from Three.js TSL to RMSL

RMSL's public API is named to match Three.js's shading language (TSL), so a
shader written against `three/tsl` ports by changing its import:

```typescript
// before
import { Fn, float, vec3, mul, sin, mix } from "three/tsl";
// after
import { Fn, float, vec3, mul, sin, mix } from "@random-mesh/rmsl";
```

Everything below is available in RMSL under the same name and the same argument
order. The places the two diverge are called out in [Behavioural
differences](#behavioural-differences).

## Methods on nodes

| TSL                                                                                               | RMSL | Notes                                  |
| ------------------------------------------------------------------------------------------------- | ---- | -------------------------------------- |
| `.add()` / `.sub()` / `.mul()` / `.div()` / `.mod()`                                              | same |                                        |
| `.negate()`                                                                                       | same |                                        |
| `.abs() .sign() .floor() .ceil() .fract() .round() .trunc()`                                      | same |                                        |
| `.radians() .degrees()`                                                                           | same |                                        |
| `.sqrt() .inverseSqrt() .exp() .log() .exp2() .log2() .cbrt()`                                    | same | `.inversesqrt()` also kept as an alias |
| `.sin() .cos() .tan() .asin() .acos() .atan() .sinh() .cosh() .tanh() .asinh() .acosh() .atanh()` | same | `.atan(x)` is `atan2`                  |
| `.min() .max() .pow() .pow2() .pow3() .pow4()`                                                    | same |                                        |
| `.step(edge) .smoothstep(lo, hi) .mix(b, t) .clamp(lo, hi)`                                       | same | value-last, as in TSL                  |
| `.saturate() .oneMinus() .reciprocal() .lengthSq() .difference()`                                 | same |                                        |
| `.normalize() .length() .distance() .dot() .cross()`                                              | same |                                        |
| `.reflect() .refract() .faceForward()`                                                            | same |                                        |
| `.transpose() .determinant() .inverse()`                                                          | same |                                        |
| `.all() .any()`                                                                                   | same |                                        |
| `.and() .or() .not() .xor()`                                                                      | same |                                        |
| `.bitAnd() .bitOr() .bitXor() .bitNot() .shiftLeft() .shiftRight()`                               | same |                                        |
| `.equal() .notEqual() .lessThan() .greaterThan() .lessThanEqual() .greaterThanEqual()`            | same |                                        |
| `.element(i)`                                                                                     | same |                                        |
| `.toVar()` / `.var()`                                                                             | same |                                        |
| `.assign()`                                                                                       | same | must be called inside an `Fn`          |
| `.addAssign() .subAssign() .mulAssign() .divAssign() .modAssign()`                                | same |                                        |
| `.toFloat() .toInt() .toUint() .toBool() .toVec2() … .toMat4()`                                   | same |                                        |
| swizzles `.xyzw .rgba .stpq`                                                                      | same | all three spellings                    |

## Free functions

Every TSL free function is exported from `rmsl`:

`add sub mul div mod equal notEqual lessThan greaterThan lessThanEqual
greaterThanEqual and or not xor bitAnd bitOr bitXor bitNot shiftLeft shiftRight
abs sign floor ceil fract round trunc radians degrees sqrt inverseSqrt
inversesqrt exp log exp2 log2 negate oneMinus reciprocal cbrt saturate lengthSq
normalize length dFdx dFdy fwidth sin cos tan asin acos atan sinh cosh tanh
asinh acosh atanh pow pow2 pow3 pow4 min max step reflect distance difference
dot cross mix clamp refract smoothstep faceForward all any transpose determinant
inverse element select luminance rand interleavedGradientNoise
premultiplyAlpha unpremultiplyAlpha texture textureLevel textureLoad textureSize`

Argument order matches TSL: `step(edge, x)`, `smoothstep(low, high, x)`,
`mix(a, b, t)`, `clamp(x, low, high)`, `faceForward(n, i, nref)`, `atan(y, x)`.

## Constructors and constants

| TSL                                                                    | RMSL                                                                                                                                                                                                                                                                                                                                                                             |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `float int uint bool`                                                  | same                                                                                                                                                                                                                                                                                                                                                                             |
| `vec2 vec3 vec4 ivec2 ivec3 ivec4 uvec2 uvec3 uvec4 bvec2 bvec3 bvec4` | same                                                                                                                                                                                                                                                                                                                                                                             |
| `mat2 mat3 mat4`                                                       | same (RMSL also exports the non-square `mat2x3` … `mat4x3`)                                                                                                                                                                                                                                                                                                                      |
| `struct(members, name)`                                                | same, except that rmsl refuses a member the struct lacks (TSL reads 0), a node of another type than its member (TSL converts it), a program that holds values of two structs of one name, a struct given to an operation such as `equal`, and a storage buffer of structs. It builds an array, and a number given in the object form, as the member's type, where TSL gives zero |
| `PI TWO_PI PI2 HALF_PI EPSILON INFINITY`                               | same                                                                                                                                                                                                                                                                                                                                                                             |

## Control flow

| TSL                                                     | RMSL                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Fn(() => { … })`                                       | same                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `Fn(fn, layout)`, `Fn(fn).setLayout(layout)`            | same, with a `{ name, type, inputs }` layout: a function of its own on GLSL and WGSL, and its body run where it is called on JS and WASM. It refuses a variable of the caller, a stage input or output, a `Return()` in a function that gives a value, and an argument of another type than its input (TSL converts it). TSL reads a string in place of the layout, `Fn(fn, "float")`, as the return type only and emits no function; it is refused |
| `If(cond, () => …).ElseIf(cond, () => …).Else(() => …)` | same                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `Switch(x).Case(…).Default(…)`                          | same                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `Loop(count, ({ i }) => …)`                             | same                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `Loop(cond, () => …)`                                   | same                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `Loop({ start, end, type, condition, update }, …)`      | same, except that a string `update` is refused: give a number, a node or a function                                                                                                                                                                                                                                                                                                                                                                 |
| —                                                       | `While(cond, () => …)` and `For(init, cond, update, body)` are rmsl's own; TSL has neither                                                                                                                                                                                                                                                                                                                                                          |
| `Break()` / `Continue()`                                | same                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `Discard()` / `Return()`                                | same                                                                                                                                                                                                                                                                                                                                                                                                                                                |

## Behavioural differences

Most differences are in the input/output plumbing, which is renderer-specific
in TSL:

- **`uniform` / `attribute` / `varying` are type-first.** TSL's `uniform(name,
value)`, `attribute(name, type)` and `varying(name, type)` are bound to a
  renderer. RMSL declares a slot: `uniform("vec4")`, `attribute("vec3")`,
  `varying("vec2")`, and `uniformRaw(name, type)` for a custom name. Change the
  argument order, not the name.
- **`outputStruct()`** — a fragment returns `outputStruct(...members)` for
  several render targets, as in TSL: member `i` goes to `@location(i)`. RMSL
  has no `output(type)`, and does not export TSL's `output` constant, the
  material's colour.
- **No renderer built-ins in the core.** `position`, `normal`, `uv`,
  `cameraPosition`, `modelViewMatrix` and the rest of TSL's geometry/material
  accessors have no RMSL-core equivalent — pass those values in as
  `attribute`/`varying`/`uniform` instead. `builtinPosition()` and
  `builtinFragDepth()` cover the two true built-ins. The scene-graph layer
  (`@random-mesh/rmsl/scene`) resolves the same accessors automatically inside
  its node-based materials; see [scene.md](./scene.md).
- **`assign()`** writes only a variable, a storage element, a varying, the position or the fragment depth,
  or a swizzle, component or column of one, and throws when the program is
  compiled otherwise, on every backend. Only those nodes, typed `Var<A>`, have
  `assign()` in their types, where TSL offers it on every node. TSL emits any assignment and leaves
  the shader compiler to refuse it. A swizzle written through names each
  component once: TSL's WebGPU backend writes `v.xx.assign(value)` component
  by component, so the last one wins; RMSL throws.
- **`Loop`** takes each of TSL's shapes and passes `{ i }`, but refuses an
  `update` given as a string, which TSL writes into the shader as text: give a
  number, a node or a function.
- **`Fn` return values** are implicit — the value the body returns is the
  result. `Return()` takes no value: in a function that returns one, an early
  `Return()` gives the zero value of its type on every target, where TSL's bare
  `return` does not compile. A `Return()` that would end a vertex or fragment
  stage, in the stage itself or in a helper `Fn` it calls, is refused, where TSL
  leaves the stage's outputs unwritten: use `If`/`Else`, or `Discard()` to drop
  a fragment.
- **`builtinFragDepth()`** starts from the fragment's own depth, so a path that
  writes none keeps it, where TSL gives such a path 0 on WebGPU and an
  undefined depth on WebGL.
- **Operands of two kinds are refused.** An operation on a float and an
  integer, or on an `int` and a `uint` (or their vectors), throws on every
  target: arithmetic and bitwise operators, comparisons, `select`, and math
  functions of several operands such as `clamp`, `mix`, `pow` and `step`. TSL
  converts one operand to the type of the other instead. Convert it yourself,
  with `toFloat()`, `toInt()`, `toUint()` or a vector conversion. A bare number
  still takes the type beside it, and a shift amount may be an `int` or a
  `uint` beside a value of either. A float operand of a bitwise operator or a
  shift is refused too, where TSL converts it to an `int`.
- **A function defined only for floats refuses an integer.** `sin`, `floor`,
  `pow`, `mix`, `step`, `normalize`, `cross` and every other math function that
  GLSL and WGSL define only for floats throw on an `int`, `uint` or `bool`
  operand, where TSL emits the call and the driver refuses it (`cross` excepted:
  TSL converts its operands to `vec3`). `abs`, `sign` and `negate` take an
  `int` but throw on a `uint` or `bool`, which TSL leaves to the driver: WebGL
  has no `abs` or `sign` of a `uint`, and WebGPU no `sign` or negation of one.
  `min`, `max` and `clamp` take either integer kind.
- **`mul(mat4, vec3)`** promotes the position with an implied `w = 1` and drops
  the homogeneous component, as TSL's matrix·vector multiply does. Other
  vector widths narrower than the matrix's columns are not supported.
- **`property()`, `uniformArray(values)` (value-owned), `buffer()`,
  `pass()`, `debug()`** are not implemented; RMSL compiles a node graph, it
  does not drive a renderer.
- **Compute** follows TSL: `storage(attribute, type)`, `instancedArray()`,
  `attributeArray()` and `fn().compute(count)`. `storage()` takes no `count`:
  a buffer's length is always its attribute's `count`. What TSL's
  renderer does with a `ComputeNode`, a context does here:
  `createWgslContext()` from `@random-mesh/rmsl/wgsl`, or `createWasmContext()`
  from `@random-mesh/rmsl/wasm`. Write a buffer with
  `context.write(attribute, data)` rather than TSL's `attribute.array` plus
  `needsUpdate`: a context uploads `attribute.array`, which is zeroed for a
  buffer made from a count, once, when the buffer is first used. An attribute holds one element type, named by the first
  `storage()` node over it; a node naming another type throws. A `vec3`
  element, or a matrix with columns of three, keeps its packed layout on the
  attribute and in what `getArrayBufferAsync()` returns: the WGSL context pads
  each column of three to 16 bytes only in the GPU buffer, where TSL rewrites
  the attribute itself to 4 components. A storage node has `element(i)`,
  `toReadOnly()`, `name`, `access` and `attribute`, and none of the
  operations of its element type: `buffer.add(1)` is a type error, and every
  backend refuses it when compiled. TSL compiles it to the whole array on
  WebGPU and to a vertex attribute on WebGL. Read `buffer.element(i)` instead.
- **Drawing from storage**: a vertex or fragment stage reads a `storage()`
  node on WGSL, read-only, as in TSL; writing one there throws.
  `vertexIndex()` and `instanceIndex()` give the vertex and instance being
  drawn. `createWgsl({ vertex, fragment, context })` binds the stages'
  storage nodes to a `WgslContext`'s buffers, so a draw reads what its
  compute programs wrote, without a copy. WebGL2 has no storage buffers,
  so on GLSL a vertex or fragment stage reads one through a data texture,
  as three.js's WebGLRenderer does, and writing one there throws.
  `createGlsl` uploads the texture from the buffer's contents at `attach`,
  and `setAttribute` on the buffer's slot fills it again.
- **Post-processing** is available from the `@random-mesh/rmsl/effects`
  subpath. The effects are pure node graphs — pass sampler uniforms and
  parameters, get a color node. See [effects.md](./effects.md) for the
  mapping of three.js's renderer-bound wrappers (`convertToTexture`,
  `passTexture`, `uv()`, `time`, ...) to RMSL.
- **`texture(tex, uv)` / `textureLevel(tex, uv, level)` / `textureLoad(tex, coords)` / `textureSize(tex)`**
  are supported as free functions, as TSL's are, and `select(cond, a, b)`, `luminance`, `rand`,
  `interleavedGradientNoise`, `premultiplyAlpha` / `unpremultiplyAlpha` mirror
  the TSL names.
