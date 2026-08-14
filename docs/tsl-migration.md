# Migrating from Three.js TSL to RMSL

RMSL's public API is named to match Three.js's shading language (TSL), so a
shader written against `three/tsl` ports by changing its import:

```typescript
// before
import { Fn, float, vec3, mul, sin, mix } from "three/tsl";
// after
import { Fn, float, vec3, mul, sin, mix } from "rmsl";
```

Everything below is available in RMSL under the same name and the same argument
order. The places the two diverge are called out in [Behavioural
differences](#behavioural-differences).

## Methods on nodes

| TSL | RMSL | Notes |
|-----|------|-------|
| `.add()` / `.sub()` / `.mul()` / `.div()` / `.mod()` | same | |
| `.negate()` | same | |
| `.abs() .sign() .floor() .ceil() .fract() .round() .trunc()` | same | |
| `.radians() .degrees()` | same | |
| `.sqrt() .inverseSqrt() .exp() .log() .exp2() .log2() .cbrt()` | same | `.inversesqrt()` also kept as an alias |
| `.sin() .cos() .tan() .asin() .acos() .atan() .sinh() .cosh() .tanh() .asinh() .acosh() .atanh()` | same | `.atan(x)` is `atan2` |
| `.min() .max() .pow() .pow2() .pow3() .pow4()` | same | |
| `.step(edge) .smoothstep(lo, hi) .mix(b, t) .clamp(lo, hi)` | same | value-last, as in TSL |
| `.saturate() .oneMinus() .reciprocal() .lengthSq() .difference()` | same | |
| `.normalize() .length() .distance() .dot() .cross()` | same | |
| `.reflect() .refract() .faceForward()` | same | |
| `.transpose() .determinant() .inverse()` | same | |
| `.all() .any()` | same | |
| `.and() .or() .not() .xor()` | same | |
| `.bitAnd() .bitOr() .bitXor() .bitNot() .shiftLeft() .shiftRight()` | same | |
| `.equal() .notEqual() .lessThan() .greaterThan() .lessThanEqual() .greaterThanEqual()` | same | |
| `.element(i)` | same | |
| `.toVar()` / `.var()` | same | |
| `.assign()` | same | must be called inside an `Fn` |
| `.addAssign() .subAssign() .mulAssign() .divAssign() .modAssign()` | same | |
| `.toFloat() .toInt() .toUint() .toBool() .toVec2() … .toMat4()` | same | |
| swizzles `.xyzw .rgba .stpq` | same | all three spellings |

## Free functions

Every TSL free function is exported from `rmsl`:

`add sub mul div mod equal notEqual lessThan greaterThan lessThanEqual
greaterThanEqual and or not xor bitAnd bitOr bitXor bitNot shiftLeft shiftRight
abs sign floor ceil fract round trunc radians degrees sqrt inverseSqrt
inversesqrt exp log exp2 log2 negate oneMinus reciprocal cbrt saturate lengthSq
normalize length dFdx dFdy fwidth sin cos tan asin acos atan sinh cosh tanh
asinh acosh atanh pow pow2 pow3 pow4 min max step reflect distance difference
dot cross mix clamp refract smoothstep faceForward all any transpose determinant
inverse element`

Argument order matches TSL: `step(edge, x)`, `smoothstep(low, high, x)`,
`mix(a, b, t)`, `clamp(x, low, high)`, `faceForward(n, i, nref)`, `atan(y, x)`.

## Constructors and constants

| TSL | RMSL |
|-----|------|
| `float int uint bool` | same |
| `vec2 vec3 vec4 ivec2 ivec3 ivec4 uvec2 uvec3 uvec4 bvec2 bvec3 bvec4` | same |
| `mat2 mat3 mat4` | same (RMSL also exports the non-square `mat2x3` … `mat4x3`) |
| `PI TWO_PI PI2 HALF_PI EPSILON INFINITY` | same |

## Control flow

| TSL | RMSL |
|-----|------|
| `Fn(() => { … })` | same |
| `If(cond, () => …).ElseIf(cond, () => …).Else(() => …)` | same |
| `Switch(x, (s) => { s.Case(…); s.Default(…) })` | same |
| `Loop(count, ({ i }) => …)` | `Loop(count, (i) => …)` — `i` is passed directly rather than destructured |
| `While(cond, () => …)` | same |
| `For(…)` | same |
| `Break()` / `Continue()` | same |
| `Discard()` / `Return()` | same |

## Behavioural differences

Most differences are in the input/output plumbing, which is renderer-specific
in TSL:

- **`uniform` / `attribute` / `varying` are type-first.** TSL's `uniform(name,
  value)`, `attribute(name, type)` and `varying(name, type)` are bound to a
  renderer. RMSL declares a slot: `uniform("vec4")`, `attribute("vec3")`,
  `varying("vec2")`, and `uniformRaw(name, type)` for a custom name. Change the
  argument order, not the name.
- **`output()`** — TSL's `output()` is the render output node; RMSL's
  `output(type)` declares a fragment output with `@location(N)`. The name is
  the same; the shape differs.
- **No renderer built-ins.** `position`, `normal`, `uv`, `cameraPosition`,
  `modelViewMatrix` and the rest of TSL's geometry/material accessors have no
  RMSL equivalent — RMSL has no scene graph. Pass those values in as
  `attribute`/`varying`/`uniform` instead. `builtinPosition()` and
  `builtinFragDepth()` cover the two true built-ins.
- **`Loop`** takes `(count, (i) => …)` with `i` as a direct `int` node, not an
  object to destructure. The `{ start, end, update }` and nested-loop forms of
  TSL's `Loop` are not implemented; use `For`/`While` for those.
- **`Fn` return values** are implicit — the value the body returns is the
  result. `Return()` only emits a void early-return.
- **`mul(mat4, vec3)`** promotes the position with an implied `w = 1` and drops
  the homogeneous component, as TSL's matrix·vector multiply does. Other
  vector widths narrower than the matrix's columns are not supported.
- **`property()`, `uniformArray(values)` (value-owned), `buffer()`,
  `compute()`, `pass()`, `debug()` and the post-processing functions** are not
  implemented; RMSL compiles a node graph, it does not drive a renderer.
