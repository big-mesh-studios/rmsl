# API Reference

## Type Constructors

### Scalars

| Function | Signature | Description |
|----------|-----------|-------------|
| `float` | `(v: number \| Node<"int">) => Node<"float">` | Float literal or cast from int |
| `int` | `(v: number \| Node<"float">) => Node<"int">` | Int literal or cast from float |
| `boolean` | `(v: boolean) => Node<"bool">` | Bool literal |

`bvec2`, `bvec3` and `bvec4` complete the type set. They have no constructor:
they are what a component-wise comparison produces. See
[BoolVecOps](#boolvecops-bvec2-bvec3-bvec4).

### Vectors

| Function | Signature | Description |
|----------|-----------|-------------|
| `vec2` | `(x, y?) => Node<"vec2">` | From 2 scalars, 1 scalar, or another vector |
| `vec3` | `(x, y?, z?) => Node<"vec3">` | From 3 scalars, 1 scalar, or another vector |
| `vec4` | `(x, y?, z?, w?) => Node<"vec4">` | From 4 scalars, 1 scalar, or another vector |

`vec3(1.0)` creates a vector with all components set to `1.0`.
`vec3(vec4(1,2,3,4))` truncates to 3 components.

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

### ArithOps (float, vec2, vec3, vec4)

| Method | Returns | Description |
|--------|---------|-------------|
| `.add(other)` | Self | Addition |
| `.sub(other)` | Self | Subtraction |
| `.mult(other)` | Self | Multiplication |
| `.div(other)` | Self | Division |
| `.negate()` | Self | Negation |

### FloatMathOps (float, vec2, vec3, vec4)

**Trig:** `.sin()`, `.cos()`, `.tan()`, `.asin()`, `.acos()`, `.atan()`

**Utility:** `.abs()`, `.sign()`, `.floor()`, `.ceil()`, `.fract()`

**Exponential:** `.sqrt()`, `.inversesqrt()`, `.exp()`, `.log()`, `.exp2()`, `.log2()`

**Binary:** `.pow(e)`, `.min(other)`, `.max(other)`, `.mod(other)`

`.mod()` on floats is floored, following GLSL's `mod()`, so the result takes the
sign of the divisor: `float(-7.5).mod(float(2))` is `0.5`, not `-1.5`. On `int`
and `uint` it is the `%` operator of each backend, which truncates toward zero
and which GLSL leaves undefined when either operand is negative.

**Interpolation:** `.mix(b, t)`, `.clamp(min, max)`, `.step(edge)`, `.smoothstep(edge0, edge1)`

**Derivative:** `.fwidth()`

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

### Vec3Ops (vec3)

| Method | Returns | Description |
|--------|---------|-------------|
| `.cross(other)` | `vec3` | Cross product |

### MatOps (mat3, mat4)

| Method | Returns | Description |
|--------|---------|-------------|
| `.mult(other)` | Self | Matrix multiplication |
| `.multVec(vec3)` | `vec3` | Matrix x vec3 (promotes to vec4 with w=1) |
| `.multVec4(vec4)` | `vec4` | Matrix x vec4 |
| `.inverse()` | Self | Matrix inverse |
| `.transpose()` | Self | Matrix transpose |

### IntOps

Arithmetic: `.add()`, `.sub()`, `.mult()`, `.div()`, `.mod()`

Bitwise: `.bitAnd()`, `.bitOr()`, `.bitXor()`, `.shiftLeft()`, `.shiftRight()`

Comparisons: `.lessThan()`, `.greaterThan()`, `.lessThanEqual()`, `.greaterThanEqual()`, `.equal()`, `.notEqual()`

All return `Node<"int">` (arithmetic/bitwise) or `Node<"bool">` (comparisons).

### UintOps

Same as IntOps but with `Node<"uint">` inputs/outputs.

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

### BoolOps

| Method | Returns | Description |
|--------|---------|-------------|
| `.and(other)` | `bool` | Logical AND |
| `.or(other)` | `bool` | Logical OR |
| `.not()` | `bool` | Logical NOT |

### BoolVecOps (bvec2, bvec3, bvec4)

The result of a component-wise comparison. There is no implicit path back to
`bool` — "is this vector less than that one" has no single answer — so the
reduction is spelled out.

| Method | Returns | Description |
|--------|---------|-------------|
| `.all()` | `bool` | True when every component is true |
| `.any()` | `bool` | True when at least one component is true |
| `.not()` | Self | Negates each component |

```typescript
let allInside = pos.lessThan(bounds).all();
let anyOutside = pos.greaterThan(bounds).any();
```

`.not()` emits GLSL's `not(bvec)` — its `!` is scalar-only — and WGSL's `!`,
which does apply to `vecN<bool>`.

## Swizzles

**vec3:** `.x`, `.y`, `.z`, `.r`, `.g`, `.b`, `.xy`, `.xz`, `.yz`, `.xyz`, `.rgb`

**vec4:** `.x`, `.y`, `.z`, `.w`, `.r`, `.g`, `.b`, `.a`, `.xy`, `.xz`, `.xw`, `.yz`, `.yw`, `.zw`, `.xyz`, `.xyw`, `.xzw`, `.yzw`, `.rgb`, `.rgba`

Swizzles are read-only. Use `.assign()` for swizzle writes:

```typescript
a.xy.assign(b.xy);  // compiles to a.xy = b.xy;
```

## Node Methods

| Method | Description |
|--------|-------------|
| `.toVar()` | Assigns expression to a temp variable, returns the variable reference |
| `.assign(value)` | Assigns a value to an existing variable or swizzle |

These only work inside an `Fn` scope.

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

- **`discard()`** - fragment discard (like GLSL `discard`)
- **`break_()`** - break from loop (`break` is a JS reserved word)
- **`continue_()`** - continue to next iteration (`continue` is a JS reserved word)

## I/O

| Function | Returns | Description |
|----------|---------|-------------|
| `uniform(type)` | `UniformNode<T>` | Declares a uniform (constant buffer input). Use `.name` for the generated name (e.g., `_rmsl_u0`); methods and swizzles are available directly. |
| `uniformRaw(name, type)` | `UniformNode<T>` | Declares a uniform with a custom name/slot (e.g., `uniformRaw("uMVP", "mat4")` emits `uniform mat4 uMVP`). Use `.name` for the custom name; methods and swizzles are available directly. |
| `attribute(type)` | `AttributeNode<T>` | Declares a vertex attribute. Use `.name` for the generated name (e.g., `_rmsl_a0`); methods and swizzles are available directly. |
| `varying(type)` | `VaryingNode<T>` | Declares a varying (vertex→fragment interpolant). Use `.name` for the generated name (e.g., `_rmsl_v0`); methods and swizzles are available directly. |
| `output(type)` | `Node<T>` | Declares a fragment output with `@location(N)` |
| `builtinPosition()` | `Node<"vec4">` | Maps to `gl_Position` / `@builtin(position)` |

A declared variable carries every method of its type (`.add()`, `.mult()`, `.x`,
`.xyz`, ...) alongside `.name`:

```typescript
let u = uniform("mat4");
let uName = u.name;             // "_rmsl_u0"
let result = u.mult(otherNode); // methods are available directly
```
