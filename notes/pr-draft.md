# Compile every generated shader with a real driver, and fix what that finds

On a full test run, every shader the suite generates is compiled by a real compiler: Chromium's WebGL2 stack for GLSL, running on SwiftShader so no GPU is needed, and Dawn — Chromium's WebGPU implementation — for WGSL. Float-valued expressions are additionally executed on both backends and their results compared against each other and against the expected value. (`pnpm test:fast` skips both layers; the gaps in the coverage are listed under [Follow-ups](#follow-ups).)

Assertions on generated source cannot tell valid-looking text from a shader a driver will accept: `lessThan(float, float)` contains `lessThan(`, and so does the correct output. Compiling the text is what tells them apart. Doing so surfaced a class of defect that changed some public behaviour.

**Breaking:** WGSL uniforms now share a single binding. Application code that created one buffer per uniform has to change — see [Uniforms share one binding in WGSL](#uniforms-share-one-binding-in-wgsl). GLSL output is unaffected.

## Contents

- [Overview](#overview) — every change in one table, with what each one asks of you
- [New](#new) — component-wise comparisons, uniform arrays, the driver-backed checks
- [Migration](#migration)
  - [Uniforms share one binding in WGSL](#uniforms-share-one-binding-in-wgsl) — **the breaking one**
  - [A GLSL fragment shader writes its colour](#a-glsl-fragment-shader-writes-its-colour)
  - [A declared GLSL `output()` receives only what you assign to it](#a-declared-glsl-output-receives-only-what-you-assign-to-it)
  - [Output locations start at zero in each shader](#output-locations-start-at-zero-in-each-shader)
  - [`mod` on floats is floored](#mod-on-floats-is-floored)
  - [A vertex stage's result has to be a `vec4`](#a-vertex-stages-result-has-to-be-a-vec4)
  - [A literal takes its operand's type](#a-literal-takes-its-operands-type)
  - [`.node()` is gone, because the constraint it worked around is gone](#node-is-gone-because-the-constraint-it-worked-around-is-gone)
  - [A vector comparison returns a boolean vector](#a-vector-comparison-returns-a-boolean-vector)
  - [TypeScript 5.0 or newer](#typescript-50-or-newer)
- [Evidence](#evidence) — what was run, and what it costs
- [Follow-ups](#follow-ups) — known gaps, left for a later change

## Overview

|                    | Change                                                                             | Action needed                                            |
| ------------------ | ---------------------------------------------------------------------------------- | -------------------------------------------------------- |
| **New**            | Component-wise comparisons, and the `bvec2` / `bvec3` / `bvec4` types they produce | —                                                        |
| **New**            | `uniformArray(type, length)` — one uniform holding several values                  | —                                                        |
| **New**            | `wgslUniformLayout()` — byte offsets for filling a uniform buffer                  | **Yes**, for WGSL — it is how you fill the shared buffer |
| **New**            | Driver-backed checks over the test suite                                           | —                                                        |
| **Output changed** | WGSL uniforms share one binding, in one structure                                  | **Yes** — rewrite buffer setup                           |
| **Output changed** | A GLSL fragment shader writes its colour output                                    | Only if you relied on it writing nothing                 |
| **Output changed** | A GLSL declared `output()` receives only what you assign to it                     | Only if you relied on the result being copied in         |
| **Output changed** | Output locations start at 0 in each shader                                         | Only if attachment indices are hardcoded                 |
| **Output changed** | A GLSL vertex output carries no `layout(location=…)` qualifier                     | —                                                        |
| **Output changed** | A whole-number literal takes its operand's type instead of becoming a float        | —                                                        |
| **Output changed** | `mod` on floats is floored in both backends                                        | Only if you depend on negative operands                  |
| **Now refused**    | A vertex result that is not a `vec4`                                               | Wrap it                                                  |
| **Now refused**    | A fractional literal beside an integer operand                                     | Convert explicitly                                       |
| **Now refused**    | A negative literal beside an **unsigned** operand                                  | Use a signed operand                                     |
| **Now refused**    | `builtinPosition()` in a fragment stage                                            | Pass through a `varying()`                               |
| **Now refused**    | `inverse()` on a matrix that is not square                                         | Use a square matrix — there is no inverse to compute     |
| **Now refused**    | A uniform array of textures                                                        | Declare them separately                                  |
| **Now refused**    | A scalar compared against a vector (at the type level)                             | Broadcast explicitly                                     |
| **API changed**    | `.node()` removed — `Node<A>` no longer needs it                                   | Delete the call                                          |
| **API changed**    | A vector comparison returns a `bvec`, not a `bool`                                 | Reduce with `.all()` / `.any()`                          |
| **API changed**    | TypeScript 5.0 or newer required                                                   | Upgrade if older                                         |

Everything under **Now refused** previously produced a shader that a driver rejects, or one that silently computed something else — with one exception: a scalar compared against a vector broadcast correctly at runtime and still does. That row is a type-level refusal only, and untyped callers see no change.

## New

**Component-wise comparisons.** Comparing vectors gives one boolean per component, so the result is a `bvec2`, `bvec3` or `bvec4`, which `.all()` or `.any()` reduces to a single `bool`. These types have no constructor: a comparison is the usual way to get one, though they can also be declared as a uniform.

```typescript
let inside = position.lessThan(bounds).all();
```

The emitted expressions were always right; the declared type was not. RMSL typed every comparison as `bool` and declared the variable that way, which neither backend accepts. Three.js's TSL resolves this the same way — `typeLength > 1 ? bvec${n} : bool`, with `all()` / `any()` to reduce.

**Uniform arrays.** `uniformArray(type, length)` declares one uniform holding several values, indexed by a constant or by a value the shader computes:

```typescript
let bricks = uniformArray("vec4", 24); // 24 elements of vec4
let colour = bricks.element(index);
```

Declaring those values as separate uniforms costs a slot each, and the guaranteed ceiling is low — WebGL2 promises only 224 uniform vectors in a fragment stage. A scene with a uniform per object runs out well before it runs out of anything else.

Narrower elements are widened invisibly — an array of `float` is physically `array<vec4<f32>, N>`, read back from the leading component, so a caller who asked for a float still gets a float. This matters when filling the buffer; see `stride` below. A uniform array of _textures_ is refused at declaration: WGSL has no array of them in the uniform address space while GLSL accepts one, so there is no spelling the two backends share.

**Driver-backed checks.** The tests import the compilers under an alias, so no individual test changes:

```typescript
import { recordingGLSL as compileGLSL, recordingWGSL as compileWGSL } from "./testing/shader-validity";
```

These wrappers are internal to the test suite and are not published. Each compiles its program to _both_ backends whatever the test asserts on, and records the source; at the end of the run every recorded shader is compiled by a driver, and a rejection fails the run. A test written against GLSL therefore has its WGSL output checked too. A run records 681 shaders across the two backends.

`src/rmsl-eval.test.ts` goes further and executes expressions, comparing the number returned by each backend against the other and against the expected value — emitting `-` where `+` was meant produces a valid shader, and is caught here. `src/rmsl.test-d.ts` asserts result types, so a signature promising `Node<"bool">` while the compiler builds a `bvec3` is a failure rather than a surprise.

## Migration

### Uniforms share one binding in WGSL

WebGPU's default `maxUniformBuffersPerShaderStage` is twelve, so a shader with a thirteenth uniform could not be created at all on a device requesting no higher limit. A scene of any size reaches that quickly, and packing means never having to raise the limit to get there. Every non-texture uniform is now a member of one structure instead:

```wgsl
// before
@group(0) @binding(0) var<uniform> _rmsl_u0: f32;
@group(0) @binding(1) var<uniform> _rmsl_u1: vec3<f32>;

// after
struct _RmslUniforms {
  _rmsl_u1: vec3<f32>,
  _rmsl_u0: f32,
};
@group(0) @binding(0) var<uniform> _rmsl_uniforms: _RmslUniforms;
```

Textures keep a binding of their own at `@group(1)` and samplers at `@group(2)`, as a full shader did before.

**`compileWGSLFn` needs the same buffer rewrite**, and it changed more. It used to emit every uniform as `@group(0) @binding(n) var<uniform> …`, textures included — which is not valid WGSL for a `texture_2d<f32>` — and declared no samplers at all. It now matches the full-shader path: one uniform struct at `@group(0)`, textures at `@group(1)`, samplers at `@group(2)`. Since its old texture output could not compile, no working application can have depended on those bindings.

Application code that created one buffer per uniform now creates one buffer and writes each value at its offset. `wgslUniformLayout()` reports them. Its `slot` argument is the `name` a declared uniform carries, and comes back unchanged as `name`, so values can be matched to offsets by that string:

```typescript
let uTime = uniform("float");
let uColor = uniform("vec3");
let uBricks = uniformArray("float", 4);

let layout = wgslUniformLayout([
  { slot: uTime.name, type: "f32" },
  { slot: uColor.name, type: "vec3<f32>" },
  { slot: uBricks.name, type: "f32", length: 4 },
]);
// {
//   members: [
//     { name: "_rmsl_u1", type: "vec3<f32>", offset: 0,  size: 12 },
//     { name: "_rmsl_u2", type: "f32", offset: 16, size: 64, length: 4, stride: 16 },
//     { name: "_rmsl_u0", type: "f32", offset: 80, size: 4 },
//   ],
//   size: 96,
// }

let buffer = device.createBuffer({
  size: layout.size,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

for (const m of layout.members) {
  if (m.stride === undefined) {
    device.queue.writeBuffer(buffer, m.offset, valuesByName[m.name]);
  } else {
    // An array: one element per stride, not packed end to end.
    valuesByName[m.name].forEach((element, i) => {
      device.queue.writeBuffer(buffer, m.offset + i * m.stride!, element);
    });
  }
}
```

The `type` strings are WGSL names — `"f32"`, `"vec3<f32>"`, `"mat3x3<f32>"` — not RMSL's `"float"`, `"vec3"`, `"mat3"`. Accepted: `f32`, `i32`, `u32`, the `vec2`/`vec3`/`vec4` forms of each, and every `mat`_N_`x`_M_`<f32>` from 2×2 to 4×4, square or not. A `mat3x3<f32>` is 48 bytes, not 36 — WGSL pads each column to 16, and the layout accounts for it. An unrecognised name throws rather than being guessed, since a wrong guess is silent and its consequence invisible.

Three rules:

**Members are ordered by alignment, then by the order you pass them in.** `wgslUniformLayout()` is a pure function over its argument: it knows nothing about any shader, and validates nothing about correspondence to one. Omitting a uniform the shader declares, passing them in a different order, or naming one twice is not detected — you get a plausible layout with wrong offsets and nothing fails.

So the array you pass has to match what the compiler emitted. **The compiler currently orders uniforms by `localeCompare` of the generated slot name**, which puts `_rmsl_u10` before `_rmsl_u2`. Sort your members that way before calling, or read the offsets back rather than reconstructing them. The example below has fewer than ten uniforms, where the two orders coincide. See [Follow-ups](#follow-ups) — this ordering is a defect, and the fix will change the required order.

Read every offset from `layout.members`; do not compute one.

**An array element occupies 16 bytes whatever it holds.** `uniformArray("float", 4)` spans 64 bytes, because WGSL rounds an array's stride up to 16 in the uniform address space and each element sits alone in its slot. `stride` reports this.

**A boolean uniform travels as unsigned integers.** WGSL admits only host-shareable types in the uniform address space, and neither a `bool` nor a boolean vector is one — GLSL accepted them, so such a program ran on WebGL and failed to build a shader module on WebGPU. A `bool` is now carried as `u32` and a `bvec`_n_ as `vec`_n_`<u32>`, with boolean array elements stored as `vec4<u32>`. Write unsigned integers for these, and pass the _carrier_ type to `wgslUniformLayout()` — `"u32"`, not `"bool"`, which throws. (`{ type: "bool", length: 4 }` is accepted for arrays, so the scalar and array cases are asymmetric.)

### A GLSL fragment shader writes its colour

GLSL ES 3.00 has no `gl_FragColor`, so a `vec4` result needs an output to land in. One is now declared and assigned if you did not declare your own, which is what the WGSL backend already did — so the two agree, and a shader that rendered on WebGPU renders on WebGL. If you declared no output of your own, returning a value that is not a `vec4`, or returning nothing, writes nothing.

### A declared GLSL `output()` receives only what you assign to it

The other half of the same change, and again GLSL catching up: the GLSL fragment backend wrote the trailing expression into _every_ declared output slot, whatever its type and whatever the program had already stored there. WGSL never did.

That produced two failures. A shader that wrote its output explicitly and then returned a different value emitted both assignments, so the returned value won. A shader whose result was not a `vec4` emitted a float into a `vec4` output, which does not compile.

```typescript
// before — the returned value landed in outColor as well
let outColor = output("vec4");
outColor.assign(shaded);
return somethingElse;
// after — assign what you want written
let outColor = output("vec4");
outColor.assign(shaded);
```

### Output locations start at zero in each shader

`output()` took its location from the id it was declared with, and that id is a module-wide counter. Six separate shaders, each with a single output, got locations 0 through 5. `MAX_DRAW_BUFFERS` is 8 in WebGL2 and 4 at the specification's minimum, so an app with a handful of materials starts emitting shaders the driver rejects — for a reason that has nothing to do with the shader in front of you.

Each shader now numbers its own fragment outputs from 0, in first-use order — an output takes its location when the shader first references it, so declaring `a` before `b` but assigning `b` first puts `b` at 0. An output that is declared and never referenced is not emitted. Application code with hardcoded attachment indices matching the old numbering needs updating.

Fragment outputs are draw-buffer indices, which is a separate space from interstage varyings, so varyings never shift them.

Two related fixes:

**A WGSL vertex stage's varyings and declared outputs no longer collide.** Both are members of one `VertexOutput` struct and so share one interstage space, but they were numbered by two counters that knew nothing of each other, which handed the same slot to one of each and the module would not build. One counter now covers both.

**A GLSL vertex output carries no `layout(location=…)` qualifier.** That qualifier names a draw buffer, which only a fragment stage has, and GLSL ES 3.00 rejects one on a vertex output whatever the number.

### `mod` on floats is floored

The same expression gave three answers for a negative operand. Constant folding used JavaScript's `%`, which truncates, so a literal expression disagreed with the identical expression written with uniforms — the program's meaning depended on where its numbers came from.

All three now follow GLSL's `mod()`, so the result takes the sign of the divisor: `float(-7.5).mod(float(2))` is `0.5`, where truncation gives `-1.5`. Verified by executing the expression on both backends.

Integers keep each backend's `%`, deliberately: GLSL leaves that undefined when either operand is negative, so there is no shared behaviour to match, and pretending otherwise would mean generating a correction for a case the language does not define.

On the WGSL side the floored modulus is now emitted as a helper function rather than inline, because the inline form repeats both operands twice each — evaluating an expensive operand four times.

For the truncated remainder — the old behaviour — write it out:

```typescript
let q = a.div(b);
let truncatedRemainder = a.sub(b.mult(q.abs().floor().mult(q.sign())));
```

`q` appears three times, and a repeated sub-expression is currently emitted at each use (see [Follow-ups](#follow-ups)), so this costs three divisions.

Three.js's shading language settles this the same way, floored in both backends with integers left on `%`.

### A vertex stage's result has to be a `vec4`

Its result becomes the position, as before. That is now checked, and the failure it catches was silent rather than loud: gating the write on the type quietly skipped it, producing a shader that links cleanly and draws nothing.

The cause sat one level down. Whether a stage produced a value was worked out by comparing the emitted expression against the string `"0.0"` — which is also how GLSL spells the number zero. So a vertex shader returning zero was read as returning nothing, skipped the check, and produced a shader that writes no position at all. WGSL spells zero differently, so the two backends disagreed about which programs were even legal, decided by one backend's number formatting. A stage's result is now decided from its type.

The check also moved to the signature, so a bad vertex result is refused where it is written rather than when the compiler runs.

```typescript
// before
Fn(() => mvp.multVec(position)); // multVec returns a vec3
// after
Fn(() => vec4(mvp.multVec(position), 1.0));
```

Returning several values still works: each is computed, and the _last_ becomes the position — so the last one has to be the `vec4`, while the values before it are unconstrained. Returning the position first and something else after it is refused; the compiler used to ignore it quietly.

Returning nothing is accepted only for a shader that assigns `builtinPosition()` itself. A stage that returns nothing and never assigns a position compiles to a `main` that sets none at all — the draw-nothing shader this check exists to catch — and is refused, with a message saying which of the two things to do.

### A literal takes its operand's type

A whole-number literal beside an integer operand now _is_ an integer, where it used to default to float:

```glsl
// before, from uniform("int").mod(2)
int _rmsl_0 = (float(_rmsl_u0) % 2.0);   // a float expression assigned to an int — GLSL rejects it
// after
int _rmsl_0 = (_rmsl_u0 % 2);
```

Floats are unaffected: `uniform("float").add(2)` still emits `+ 2.0`.

Comparisons now type their literals the same way arithmetic does, from one shared place. Before, an unsigned operand reached WGSL as an unsigned value compared against a signed literal, for which there is no overload, while GLSL widened the operand to a float.

Two literals have no such type, and are refused when the shader is built:

- a fractional literal beside an integer operand — `someInt.add(2.5)`. GLSL widened the operand and then failed to narrow the result back; WGSL truncated the literal and carried on silently. The same expression either did not compile or quietly meant something else. Convert the operand to a float, or use a whole number.
- a negative literal beside an **unsigned** operand — `someUint.add(-1)`. It has no representation at all, and WGSL has no unary minus for one. A negative literal beside a signed `int` is unaffected.

### `.node()` is gone, because the constraint it worked around is gone

`.node()` existed because a uniform, attribute or varying could not carry its type's operations directly. `Node<A>` was an intersection of ten conditional types, re-evaluated at every instantiation, and a member referring back to `Node` — `not(): Node<A>` — expanded without bound. Adding one such member ran `tsc` for four minutes and then exhausted a 2 GB heap; bisecting showed the trigger was that self-reference alone.

`Node<A>` is now a single indexed access into a registry that maps each shader type to its operations:

```typescript
interface NodeOps {
  float: ArithOps<"float"> & FloatMathOps<"float"> & ComparisonOps<"bool", FloatLike>;
  vec3: ArithOps<"vec3"> &
    FloatMathOps<"vec3"> &
    ComparisonOps<"bvec3", Vec3Like | FloatLike> &
    VecCommonOps<"vec3"> &
    Vec3Ops &
    Vec3Swizzles;
  // one row per shader type
}

export type Node<A extends ShaderType> = BaseNode<A> & NodeOps[A] & NodeMethods<A>;
```

The checker resolves one lookup instead of walking ten branches, self-referential members are fine, and a new type cannot be added without stating what it supports. `bvec2`/`bvec3`/`bvec4` and the comparison result types were only expressible once this was in place.

So a declared variable carries its operations, and the call has no replacement — it was the identity.

```typescript
// before
let scaled = uColor.node().mult(brightness);
// after
let scaled = uColor.mult(brightness);
```

`.name` is unchanged, as are the `isUniformNode` / `isAttributeNode` / `isVaryingNode` guards. But `UniformNode`, `AttributeNode` and `VaryingNode` are now type aliases rather than interfaces, since an interface cannot extend a type whose members resolve through an indexed access — and that has consequences if you extended them:

```typescript
interface Mine extends UniformNode<"float"> { extra: number }   // still fine
interface Mine<A extends ShaderType> extends UniformNode<A> {}  // TS2312
declare module "rmsl" { interface UniformNode<A> { … } }        // TS2300
```

Extending with a concrete shader type still works. Extending _generically_, and augmenting the interface by declaration merging, do not. Use an intersection instead: `type Mine<A extends ShaderType> = UniformNode<A> & { extra: number }`.

### A vector comparison returns a boolean vector

Where the result was used as a condition, reduce it first.

```typescript
// before
If(position.lessThan(bounds), () => { ... });
// after
If(position.lessThan(bounds).all(), () => { ... });
```

What a comparison accepts is now named per type alongside what it returns, so a float accepts only float-like operands while the vector types keep accepting a scalar to broadcast. The combination with no coherent result — a scalar compared against a vector — is a type error. Write `vec3(someFloat).lessThan(someVec3)` if broadcasting was the intent. This is a type-level refusal: the runtime still broadcasts either way, so untyped callers see no change.

### TypeScript 5.0 or newer

`Fn` uses a `const` type parameter. An array literal is otherwise inferred as an array — where nothing is last — so a vertex stage's position could only be checked once the shader compiled. Returning several values infers a tuple now, so the last one is knowable and can be checked where it is written.

## Evidence

- 214 tests and 19 type-level tests. `pnpm test` takes about 8 seconds on a 2023 MacBook Pro; `pnpm test:fast` skips both driver-backed checks and takes about two.
- 681 shaders compiled by real drivers per run, counting both backends.
- `pnpm type-check` takes about 3.8 seconds over the library and its tests. The type surface resolves each node type through one registry lookup rather than ten conditional branches, which is what makes a self-referential member such as `not(): Node<A>` expressible at all. Measured on the type surface alone: 10.6s under the conditional encoding without `bvec`, against 1.4s under the registry encoding with it.
- `pnpm test:mutate` runs mutation testing over `src/rmsl.ts` — it alters the compiler and reports how many alterations the tests notice. It scores 52.2% with the driver-backed checks off, so that figure reflects the source assertions alone. `pnpm test:mutate:deep` leaves them on; the two have not been compared.

## Follow-ups

Known gaps, for a later change:

- **`docs/compilation.md`'s multi-return vertex example now throws.** It ends in two floats and is passed to `compileGLSL.vertex`, which refuses a result that cannot become a position. The error is right and the example predates it; it is called out here rather than changed in this branch.
- A sub-expression used twice is emitted twice, so a value read three times is computed three times in the generated shader. Measured on `apps/shared/shader.ts`: one matrix-vector product written in the source appears three times in the generated GLSL, so every fragment runs 48 multiply-adds where it wrote 16. The memo map already knows which nodes are reached more than once, so the information needed is in hand; spilling those into generated temporaries is the remaining work.
- Execution reads back a single float, so only expressions reducing to a float have their result checked. Matrix-vector transforms, swizzles and constructors are verified as valid but not as correct — and two mutations of the matrix-times-vector emitter go unnoticed by the whole suite: transposing the multiplication, and turning `vec4(v, 1.0)` into `vec4(v, 0.0)`, which is the difference between transforming a point and a direction. Both compile.
- One test asserting that raw arrays are wrapped by their length cannot fail: it checks for a type name that appears in the output regardless.
- Nothing prevents a new test file from importing the compilers directly and skipping the _recording_ step silently, so its shaders never reach the batch driver check. `src/rmsl-eval.test.ts` already does this — it executes what it compiles, so its shaders are exercised more deeply than most, but they are not among the ones the recorded batch compiles.
- The two `case "let"` arms return different expressions — GLSL `lhs.expr`, WGSL `varName`. They coincide today, so this is latent, but the arms are otherwise identical code.
- `op()` and `comp()` allocate on every operator call, roughly 3.5× the pre-branch shape — about 0.08ms per 200-line shader, so worth doing only while the file is open for another reason.
- **The compiler sorts uniforms by `localeCompare` of the generated slot name before laying them out**, so `_rmsl_u10` precedes `_rmsl_u2` and a shader with ten or more uniforms is laid out in an order no caller would guess. `wgslUniformLayout()` breaks alignment ties by the order it is handed, so it faithfully preserves this; the fix for the unstable tie-break therefore does not reach compiled output. The regression test covers `wgslUniformLayout()` in isolation and so does not catch it. Either the sort should go, or the compiler should hand back the layout it used instead of asking callers to reconstruct one.
- **An operation's result type is still decided in eight hand-maintained places** — five tables, four hardcoded construction sites, the constant folder, and the two backend switches. Adding an operation means editing all of them in lockstep, and nothing fails if a step is skipped; the symptom is a wrong result type or a silently unfolded constant, surfacing later as an invalid shader. This branch fixed two instances (`dot` and `length`) and consolidated the invertibility check, but not the shape that produced them. `CONTRIBUTING.md` documents the current process in the meantime.
- The WGSL matrix inverse helpers divide by the determinant unguarded, where the JavaScript counterpart returns early on a singular matrix. Only `_rmsl_inverse4` has a JavaScript counterpart at all, and it lives in a demo app the compiler does not import, so the two are related by a comment and nothing else.
- Version is unchanged at 1.0.6. This contains breaking changes and needs a major bump before release.
- The TypeScript 5.0 floor is not declared anywhere: no `engines`, no `peerDependencies`, TypeScript present only as a devDependency at `^5.7.2`.
