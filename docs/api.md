# API Reference

## Type Constructors

### Scalars

| Function | Signature | Description |
|----------|-----------|-------------|
| `float` | `(v: number \| Node<"int">) => Node<"float">` | Float literal or cast from int |
| `int` | `(v: number \| Node<"float">) => Node<"int">` | Int literal or cast from float |
| `uint` | `(v: number \| Node<"float"> \| Node<"int">) => Node<"uint">` | Unsigned literal or cast; refuses negatives |
| `bool` | `(v: boolean \| Node<"float"> \| Node<"int"> \| Node<"uint">) => Node<"bool">` | Bool literal or cast from a number |

`bvec2`, `bvec3` and `bvec4` are what a component-wise comparison produces and
can also be built directly with `bvec2(...)`, `bvec3(...)`, `bvec4(...)`. See
[BoolVecOps](#boolvecops-bvec2-bvec3-bvec4).

### Vectors

| Function | Signature | Description |
|----------|-----------|-------------|
| `vec2` | `(x, y?) => Node<"vec2">` | From 2 scalars, 1 scalar, or another vector |
| `vec3` | `(x, y?, z?) => Node<"vec3">` | From 3 scalars, 1 scalar, or another vector |
| `vec4` | `(x, y?, z?, w?) => Node<"vec4">` | From 4 scalars, 1 scalar, or another vector |
| `ivec2` | `(x, y?) => Node<"ivec2">` | Signed integer vector |
| `ivec3` | `(x, y?, z?) => Node<"ivec3">` | Signed integer vector |
| `ivec4` | `(x, y?, z?, w?) => Node<"ivec4">` | Signed integer vector |
| `uvec2` | `(x, y?) => Node<"uvec2">` | Unsigned integer vector; refuses negatives |
| `uvec3` | `(x, y?, z?) => Node<"uvec3">` | Unsigned integer vector; refuses negatives |
| `uvec4` | `(x, y?, z?, w?) => Node<"uvec4">` | Unsigned integer vector; refuses negatives |

`vec3(1.0)` creates a vector with all components set to `1.0`.
`vec3(vec4(1,2,3,4))` truncates to 3 components.

Integer vectors are not interchangeable with float vectors: `ivec3(vec3)` and
`vec3(ivec3)` are explicit casts. A plain number beside an integer vector is
typed as its component (`ivec2(1, 2).add(3)` adds the integer `3`).

### Matrices

| Function | Components | Default (identity) |
|----------|-----------|-------------------|
| `mat2` | 2x2 | `[1,0,0,1]` |
| `mat2x3` | 2x3 | `[1,0,0,0,1,0]` |
| `mat2x4` | 2x4 | `[1,0,0,0,0,1,0,0]` |
| `mat3x2` | 3x2 | `[1,0,0,0,1,0]` |
| `mat3` | 3x3 | `[1,0,0,0,1,0,0,0,1]` |
| `mat3x4` | 3x4 | `[1,0,0,0,0,1,0,0,0,0,1,0]` |
| `mat4x2` | 4x2 | `[1,0,0,0,0,1,0,0]` |
| `mat4x3` | 4x3 | `[1,0,0,0,0,1,0,0,0,0,1,0]` |
| `mat4` | 4x4 | `[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]` |

`mat4(1.0)` fills the diagonal with `1.0` (scalar constructor).
`mat4(v4, v4, v4, v4)` constructs from column vectors.

## Operations

### ArithOps (float, vec2, vec3, vec4, ivec2/3/4, uvec2/3/4)

| Method | Returns | Description |
|--------|---------|-------------|
| `.add(other)` | Self | Addition |
| `.sub(other)` | Self | Subtraction |
| `.mul(other)` | Self | Multiplication |
| `.div(other)` | Self | Division |
| `.negate()` | Self | Negation |

Integer vectors carry `.negate()` and `.abs()` only in their signed forms —
WGSL has no `abs` or unary minus for `u32`. All types accept an operand that
is a scalar of the same kind (`ivec2.add(3)`), which both backends broadcast.

### FloatMathOps (float, vec2, vec3, vec4)

**Trig:** `.sin()`, `.cos()`, `.tan()`, `.asin()`, `.acos()`, `.atan()`, `.sinh()`, `.cosh()`, `.tanh()`, `.asinh()`, `.acosh()`, `.atanh()`

**Utility:** `.abs()`, `.sign()`, `.floor()`, `.ceil()`, `.fract()`, `.round()`, `.trunc()`, `.radians()`, `.degrees()`

**Exponential:** `.sqrt()`, `.inverseSqrt()` (`.inversesqrt()` kept as an alias, as TSL keeps it), `.exp()`, `.log()`, `.exp2()`, `.log2()`, `.cbrt()`

**Derived:** `.reciprocal()` (1/x), `.oneMinus()` (1−x), `.lengthSq()` (dot of self),
`.saturate()` (clamp to 0…1), `.pow2()`, `.pow3()`, `.pow4()`

**Binary:** `.pow(e)`, `.min(other)`, `.max(other)`, `.mod(other)`, `.difference(other)` (abs of the difference)

`.mod()` on floats is floored, following GLSL's `mod()`, so the result takes the
sign of the divisor: `float(-7.5).mod(float(2))` is `0.5`, not `-1.5`. On `int`
and `uint` it is the `%` operator of each backend, which truncates toward zero
and which GLSL leaves undefined when either operand is negative.

**Interpolation:** `.mix(b, t)`, `.clamp(min, max)`, `.step(edge)`, `.smoothstep(edge0, edge1)`

**Derivative:** `.fwidth()`, `.dFdx()`, `.dFdy()`

**Comparisons:**
`.lessThan(other)`, `.greaterThan(other)`, `.lessThanEqual(other)`, `.greaterThanEqual(other)`, `.equal(other)`, `.notEqual(other)`

Comparisons are component-wise, so the result has one boolean per component.
Only scalars reduce to a single `bool`:

| Receiver | Returns |
|----------|---------|
| `float`  | `Node<"bool">` |
| `vec2`   | `Node<"bvec2">` |
| `vec3`   | `Node<"bvec3">` |
| `vec4`   | `Node<"bvec4">` |

Scalars emit `a < b`; vectors emit `lessThan(a, b)` in GLSL and `a < b` in WGSL,
both yielding a boolean vector. A scalar compared against a vector is broadcast
to the vector's width, since neither language compares the two directly.

```typescript
let inside = pos.lessThan(vec3(1, 1, 1)).all();   // Node<"bool">
```

### VecCommonOps (vec2, vec3, vec4)

| Method | Returns | Description |
|--------|---------|-------------|
| `.dot(other)` | `float` | Dot product |
| `.length()` | `float` | Vector length |
| `.normalize()` | Self | Unit vector |
| `.distance(other)` | `float` | Distance between vectors |
| `.reflect(normal)` | Self | Reflection vector |
| `.refract(normal, eta)` | Self | Refraction vector |
| `.clamp(min, max)` | Self | Component-wise clamp |
| `.mix(b, t)` | Self | Linear interpolation |
| `.element(i)` | `float` | Component at runtime index `i` (`v[i]`) |

### Vec3Ops (vec3)

| Method | Returns | Description |
|--------|---------|-------------|
| `.cross(other)` | `vec3` | Cross product |

### MatOps (mat2, mat3, mat4)

| Method | Returns | Description |
|--------|---------|-------------|
| `.mul(other)` | Self / vec | Matrix multiplication. A matrix times a vector of its column width gives that vector's result; a vector one component shorter (e.g. `mat4 * vec3`) is a position with its homogeneous `w = 1` implied, promoted and truncated by the compilers. |
| `.inverse()` | Self | Matrix inverse |
| `.transpose()` | Self | Matrix transpose |
| `.determinant()` | `float` | Scalar determinant |

### IntOps

Arithmetic: `.add()`, `.sub()`, `.mul()`, `.div()`, `.mod()`, `.negate()`, `.abs()`, `.min()`, `.max()`, `.clamp()`

Bitwise: `.bitAnd()`, `.bitOr()`, `.bitXor()`, `.shiftLeft()`, `.shiftRight()`, `.bitNot()`

Comparisons: `.lessThan()`, `.greaterThan()`, `.lessThanEqual()`, `.greaterThanEqual()`, `.equal()`, `.notEqual()`

All return `Node<"int">` (arithmetic/bitwise) or `Node<"bool">` (comparisons).

### UintOps

Arithmetic: `.add()`, `.sub()`, `.mul()`, `.div()`, `.mod()`, `.min()`, `.max()`, `.clamp()`

Bitwise: `.bitAnd()`, `.bitOr()`, `.bitXor()`, `.shiftLeft()`, `.shiftRight()`, `.bitNot()`

Comparisons: `.lessThan()`, `.greaterThan()`, `.lessThanEqual()`, `.greaterThanEqual()`, `.equal()`, `.notEqual()`

Same result types as IntOps but unsigned. There is deliberately no `.negate()`/
`.abs()`: WGSL has neither for `u32`.

### IVecOps (ivec2, ivec3, ivec4)

Component-wise integer vectors. Arithmetic `.add/sub/mul/div/mod`, bitwise
`.bitAnd/bitOr/bitXor/shiftLeft/shiftRight/bitNot`, plus `.negate()`, `.abs()`,
`.min()`, `.max()`, `.clamp()`, and `.element(i)` → `Node<"int">`. Comparisons
reduce to `bvecN`. An operand that is an `int` scalar is broadcast.

### UVecOps (uvec2, uvec3, uvec4)

As IVecOps but unsigned and without `.negate()`/`.abs()`; `.element(i)` →
`Node<"uint">`.

### SamplerOps (sampler2D)

| Method | Returns | Description |
|--------|---------|-------------|
| `.texture(coords)` | `vec4` | Sample texture at coordinates |
| `.textureLod(coords, lod)` | `vec4` | Sample with explicit LOD |

### Sampler3DOps (sampler3D)

A 3D texture is sampled at its volume coordinate — a `vec3`, as with a cube map.

| Method | Returns | Description |
|--------|---------|-------------|
| `.texture(coords)` | `vec4` | Sample texture at volume coordinates |
| `.textureLod(coords, lod)` | `vec4` | Sample with explicit LOD |

### Integer samplers (isampler2D/3D/Cube, usampler2D/3D/Cube)

Signed and unsigned integer textures. Integer textures are not filterable, so
`.texture()`/`.textureLod()` compile to an unfiltered fetch — `texelFetch` in
GLSL, `textureLoad` in WGSL, which there needs **no sampler binding**. They take
integer texel coordinates of the sampler's width and return an integer vector:

| Sampler | `.texture()` returns | Coordinates |
|---------|---------------------|-------------|
| `isampler2D` | `Node<"ivec4">` | `ivec2` |
| `isampler3D`/`isamplerCube` | `Node<"ivec4">` | `ivec3` |
| `usampler2D` | `Node<"uvec4">` | `uvec2` |
| `usampler3D`/`usamplerCube` | `Node<"uvec4">` | `uvec3` |

```typescript
let voxel = uniform("usampler3D");
let density = voxel.texture(uvec3(x, y, z));   // Node<"uvec4">
```

### BoolOps

| Method | Returns | Description |
|--------|---------|-------------|
| `.and(other)` | `bool` | Logical AND |
| `.or(other)` | `bool` | Logical OR |
| `.not()` | `bool` | Logical NOT |
| `.xor(other)` | `bool` | Logical XOR (lowered to `(a\|\|b) && !(a&&b)`) |

### BoolVecOps (bvec2, bvec3, bvec4)

The result of a component-wise comparison. There is no implicit path back to
`bool` — "is this vector less than that one" has no single answer — so the
reduction is spelled out.

| Method | Returns | Description |
|--------|---------|-------------|
| `.all()` | `bool` | True when every component is true |
| `.any()` | `bool` | True when at least one component is true |
| `.not()` | Self | Negates each component |
| `.xor(other)` | Self | Component-wise logical XOR |

```typescript
let allInside = pos.lessThan(bounds).all();
let anyOutside = pos.greaterThan(bounds).any();
```

`.not()` emits GLSL's `not(bvec)` — its `!` is scalar-only — and WGSL's `!`,
which does apply to `vecN<bool>`.

## Swizzles

All three component spellings are available, matching GLSL/WGSL and TSL:

| Set | Components |
|-----|-----------|
| `xyzw` | `.x`, `.y`, `.z`, `.w` |
| `rgba` | `.r`, `.g`, `.b`, `.a` |
| `stpq` | `.s`, `.t`, `.p`, `.q` |

**vec3:** `.x/.y/.z` (or `.r/.g/.b` or `.s/.t/.p`), `.xy`, `.xz`, `.yz`, `.st`, `.sp`, `.tp`, `.xyz`, `.rgb`, `.stp`

**vec4:** `.x/.y/.z/.w` (or `.r/.g/.b/.a` or `.s/.t/.p/.q`), `.xy`, `.xz`, `.xw`, `.yz`, `.yw`, `.zw`, `.st`, `.sp`, `.sq`, `.tp`, `.tq`, `.pq`, `.xyz`, `.xyw`, `.xzw`, `.yzw`, `.stp`, `.stq`, `.spq`, `.tpq`, `.rgb`, `.rgba`, `.stpq`

Integer vectors carry the same swizzle sets, typed by their component: a single
component of an `ivecN` is `Node<"int">`, of a `uvecN` `Node<"uint">`, and a
multi-component swizzle is the matching integer vector (`ivec3.xy` → `ivec2`).

Swizzles are read-only. Use `.assign()` for swizzle writes:

```typescript
a.xy.assign(b.xy);  // compiles to a.xy = b.xy;
```

## Node Methods

| Method | Description |
|--------|-------------|
| `.toVar()` | Assigns expression to a temp variable, returns the variable reference |
| `.var()` | TSL's shorthand for `.toVar()` |
| `.assign(value)` | Assigns a value to an existing variable or swizzle |
| `.addAssign(v)` / `.subAssign(v)` / `.mulAssign(v)` / `.divAssign(v)` / `.modAssign(v)` | Compound assignment, as TSL's `addAssign`/`mulAssign`/… |
| `.toFloat()`, `.toInt()`, `.toUint()`, `.toBool()` | Cast to a scalar type |
| `.toVec2()`, `.toVec3()`, `.toVec4()` | Cast to a float vector (narrowing a vec4 drops components) |
| `.toIVec2/3/4()`, `.toUVec2/3/4()`, `.toBVec2/3/4()` | Cast to an integer or boolean vector |
| `.toMat2()`, `.toMat3()`, `.toMat4()` | Cast to a matrix |
| `.convert(type)` | Cast to any type by name |

These only work inside an `Fn` scope. Casts compile to the target constructor —
`int(x)` in GLSL, `i32(x)` in WGSL — truncating toward zero where that is what
the constructor does.

## TSL free-function API

Every operation is also available as a free function, named exactly as in
`three/tsl`, so a shader written against TSL can be ported by changing its
import. Each takes nodes or raw values (numbers, booleans, arrays) and compiles
to the same nodes as the method form.

**Operators:** `add(a, b)`, `sub(a, b)`, `mul(a, b)`, `div(a, b)`, `mod(a, b)`
(arithmetic ones accept further operands — `add(a, b, c)`)

**Comparisons:** `equal`, `notEqual`, `lessThan`, `greaterThan`,
`lessThanEqual`, `greaterThanEqual`

**Logic / bitwise:** `and(a, b)`, `or(a, b)`, `xor(a, b)`, `not(a)`,
`bitAnd`, `bitOr`, `bitXor`, `bitNot`, `shiftLeft`, `shiftRight`

**Unary math:** `abs`, `sign`, `floor`, `ceil`, `fract`, `round`, `trunc`,
`radians`, `degrees`, `sqrt`, `inverseSqrt` (and `inversesqrt`), `exp`, `log`,
`exp2`, `log2`, `negate`, `oneMinus`, `reciprocal`, `cbrt`, `saturate`,
`lengthSq`, `normalize`, `length`, `dFdx`, `dFdy`, `fwidth`

**Trig:** `sin`, `cos`, `tan`, `asin`, `acos`, `atan` (`atan(y, x)` compiles to
`atan2`), `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`

**Binary math:** `pow(x, e)`, `pow2`, `pow3`, `pow4`, `min`, `max`,
`step(edge, x)`, `reflect`, `distance`, `difference`, `dot`, `cross`

**Ternary math:** `mix(a, b, t)`, `clamp(x, low, high)`, `refract`,
`smoothstep(low, high, x)`, `faceForward(n, i, nref)`

**Conditional:** `select(cond, a, b)` — `a` when `cond`, else `b`. The
condition may be a `bool` or a boolean vector (component-wise selection).

**Colour / noise:** `luminance(color)` (Rec. 709 dot product),
`rand(uv)` (a hash in `[0, 1)`), `interleavedGradientNoise(position)`
(Jimenez 2014 dithering hash, takes a pixel-space position),
`premultiplyAlpha(color)`, `unpremultiplyAlpha(color)`

**Texel access:** `textureSize(sampler)` → the texture's dimensions as a
`uvec2`/`uvec3`, and `textureLoad(sampler, ivec2|ivec3)` → an unfiltered texel
fetch at integer coordinates.

**Reductions:** `all(x)`, `any(x)`

**Matrix:** `transpose`, `determinant`, `inverse`

**Element:** `element(a, i)`

The argument order follows TSL — `step(edge, x)`, `smoothstep(low, high, x)`
and `mix(a, b, t)` take the value last, as both GLSL and WGSL spell them.

The general free functions return a loosely-typed node (their type cannot be
known until the operands are inspected); the operations that reduce — `dot`,
`length`, `distance`, `all`, `any`, `determinant` — are typed to their narrower
result.

**Constants:** `PI`, `TWO_PI` (and the deprecated `PI2`), `HALF_PI`, `EPSILON`,
`INFINITY` — all `float()` nodes, as in TSL.

## Control Flow

### If

```typescript
If(cond, () => {
  // then branch
}).ElseIf(otherCond, () => {
  // else-if branch
}).Else(() => {
  // else branch
});
```

The names are the capitalized TSL ones, so a shader written against
`three/tsl`'s `If`/`ElseIf`/`Else`/`While`/`For`/`Switch`/`Case`/`Default`
migrates unchanged.

### Loop

TSL's counting loop:

```typescript
Loop(int(10), (i) => {
  // body, `i` is the int index from 0 to 9
});
```

It lowers to the same `For` machinery.

### Switch

```typescript
Switch(int(level), (s) => {
  s.Case(0, () => { colour.assign(black); });
  s.Case([1, 2], () => { colour.assign(grey); });   // several values, one body
  s.Default(() => { colour.assign(white); });
});
```

Compiles to an if/else-if chain comparing the selector with each case value —
the same lowering TSL uses — so there is no fall-through and no `Break()`.

### For

```typescript
For(
  () => int(0).toVar(),                // init - returns loop variable
  (i) => i.lessThan(int(10)),          // condition - receives variable
  (i) => { i.assign(i.add(int(1))); }, // update - receives variable
  (i) => {                              // body - receives variable
    // loop body
  },
);
```

### While

```typescript
While(condition, () => {
  // loop body
});
```

### Other

- **`Discard()`** - fragment discard (like GLSL `discard`)
- **`Break()`** - break from a loop
- **`Continue()`** - continue to the next iteration
- **`Return()`** - early return from an `Fn`

## I/O

| Function | Returns | Description |
|----------|---------|-------------|
| `uniform(type)` | `UniformNode<T>` | Declares a uniform (constant buffer input). Use `.name` for the generated name (e.g., `_rmsl_u0`); methods and swizzles are available directly. |
| `uniformRaw(name, type)` | `UniformNode<T>` | Declares a uniform with a custom name/slot (e.g., `uniformRaw("uMVP", "mat4")` emits `uniform mat4 uMVP`). Use `.name` for the custom name; methods and swizzles are available directly. |
| `attribute(type)` | `AttributeNode<T>` | Declares a vertex attribute. Use `.name` for the generated name (e.g., `_rmsl_a0`); methods and swizzles are available directly. |
| `varying(type)` | `VaryingNode<T>` | Declares a varying (vertex→fragment interpolant). Use `.name` for the generated name (e.g., `_rmsl_v0`); methods and swizzles are available directly. |
| `output(type)` | `Node<T>` | Declares a fragment output with `@location(N)` |
| `builtinPosition()` | `Node<"vec4">` | Maps to `gl_Position` / `@builtin(position)` |
| `builtinFragDepth()` | `Node<"float">` | Maps to `gl_FragDepth` / `@builtin(frag_depth)` |
| `fragCoord()` | `Node<"vec2">` | The fragment's framebuffer position in pixels (`gl_FragCoord.xy` / WGSL `@builtin(position)`). Fragment stage only. Alias: `screenCoordinate()`. |
| `screenSize()` | `Node<"vec2">` | The drawing-buffer size in pixels, as a `vec2` uniform the host binds. |
| `screenUV()` | `Node<"vec2">` | `fragCoord() / screenSize()` — the normalized screen position. |
| `uv()` | `Node<"vec2">` | The fullscreen-quad UV: the normalized screen position, as TSL's `uv()`. |
| `time()` | `Node<"float">` | A shared per-frame clock (seconds), as a `float` uniform the host updates. |

A declared variable carries every method of its type (`.add()`, `.mul()`, `.x`,
`.xyz`, ...) alongside `.name`:

```typescript
let u = uniform("mat4");
let uName = u.name;             // "_rmsl_u0"
let result = u.mul(otherNode); // methods are available directly
```
