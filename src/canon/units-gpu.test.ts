import { NodeError, ProgramError, TargetError, UsageError } from "../errors";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import {
  attribute,
  builtinPosition,
  bvec3,
  Fn,
  Loop,
  For,
  While,
  If,
  Return,
  float,
  instancedArray,
  int,
  invocationIndex,
  ivec2,
  ivec3,
  outputStruct,
  select,
  texture,
  uniform,
  uniformArray,
  uniformRaw,
  uvec2,
  vec2,
  vec3,
  varyingRaw,
  vec4,
} from "../rmsl";
import type { UniformNode } from "../rmsl";
import {
  compile,
  compileWgsl as compileWgslStage,
  compileWgslFn,
  createWgsl,
  createWgslCompute,
  createWgslContext,
  wgslUniformDeclaration,
  wgslUniformLayout,
} from "../wgsl";
import {
  assertRecordedShadersValid,
  recordingGLSL as compileGlsl,
  recordingWGSL as compileWgsl,
} from "../testing/shader-validity";
import { evaluateWGSL } from "../testing/shader-eval";
import {
  GPU_ENABLED,
  installWebGpuGlobals,
  readsStorageFromBeforeTheLoop,
  releaseGpu,
  webgpuAvailable,
} from "../testing/gpu";
import { READ_IMAGE, READ_PIXEL, runInGpuPage, runInWebGpuPage } from "../testing/browser";
import { stubDevice, stubGl } from "./scene-stubs";
import { CONDITION_PAIRS, inlineFnConditions, inlineFnPaths } from "../testing/inline-fn-paths";
import { createGlsl } from "../glsl";
import { createWasmContext } from "../wasm";

const WEBGPU = await webgpuAvailable();

/** Whether the Dawn device reads a storage element from before the loop in some loops. */
const STALE_STORAGE = GPU_ENABLED && (await readsStorageFromBeforeTheLoop());

afterAll(async () => {
  await assertRecordedShadersValid();
  await releaseGpu();
}, 120_000);

/** The expression WGSL reads the uniform `slot` through. */
const member = (slot: string) => `_rmsl_uniforms.${slot}`;

/** The members of the WGSL uniform struct, in the order it declares them. */
const structMembers = (code: string) =>
  [...(/struct _RmslUniforms \{([\s\S]*?)\n\};/.exec(code)?.[1] ?? "").matchAll(/^\s*(\w+):/gm)].map((m) => m[1]);

describe("the GPU entry points load where no graphics API exists", () => {
  /**
   * The test runs under Node, where WebGL has no globals, and this file
   * imported the GLSL entry point before any test ran.
   *
   * @canon spec-the-glsl-entry-point-loads-where-no-graphics-api-exists
   */
  it("imports the GLSL entry point with no WebGL global", async () => {
    expect(typeof (globalThis as any).WebGL2RenderingContext).toBe("undefined");
    const glsl = await import("../glsl");
    expect(typeof glsl.createGlsl).toBe("function");
    expect(glsl.compileGlsl(vec4(1, 0, 0, 1))).toContain("void main");
  });

  /**
   * The test runs before any WebGPU global is installed, and this file
   * imported the WGSL entry point before any test ran.
   *
   * @canon spec-the-wgsl-entry-point-loads-where-no-graphics-api-exists
   */
  it("imports the WGSL entry point with no WebGPU global", async () => {
    expect(typeof (globalThis as any).GPUBufferUsage).toBe("undefined");
    const wgsl = await import("../wgsl");
    expect(typeof wgsl.createWgsl).toBe("function");
    expect(wgsl.compileWgsl(vec4(1, 0, 0, 1))).toContain("@fragment");
  });
});

describe("what GLSL and WGSL write for an operation", () => {
  /**
   * WGSL refuses `-2147483648i`, and GLSL writes the same subtraction so both
   * targets compile one form.
   *
   * @canon spec-int-min-compiles-to-a-subtraction-of-two-in-range-literals
   */
  it.each([
    ["GLSL", compileGlsl, "(-2147483647 - 1)"],
    ["WGSL", compileWgsl, "(-2147483647i - 1i)"],
  ] as const)("writes INT_MIN as a subtraction on %s", (_, compiler, literal) => {
    expect(compiler(vec4(int(-2147483648).toFloat(), 0, 0, 1))).toContain(literal);
  });

  /**
   * GLSL writes `a + b & c | a` and lets precedence group it. WGSL brackets
   * each operand that is not unary.
   *
   * @canon spec-wgsl-brackets-a-bitwise-operand-that-is-not-unary
   */
  it("brackets a bitwise operand that is not unary on WGSL", () => {
    const a = uniform("int");
    const b = uniform("int");
    const c = uniform("int");
    const code = compileWgsl(vec4(a.add(b).bitAnd(c).bitOr(a).toFloat(), 0, 0, 1));
    const [A, B, C] = [a, b, c].map((u) => member(u.name));
    expect(code).toContain(`((${A} + ${B}) & ${C}) | ${A}`);
  });

  /**
   * GLSL writes `p && q || r`, since `&&` binds tighter. WGSL brackets the
   * `&&` that sits in the `||`.
   *
   * @canon spec-wgsl-brackets-a-logical-operator-nested-in-another
   */
  it("brackets a logical operator nested in another on WGSL", () => {
    const [x, y, z] = [uniform("float"), uniform("float"), uniform("float")];
    const condition = x.greaterThan(0).and(y.greaterThan(0)).or(z.greaterThan(0));
    const code = compileWgsl(vec4(select(condition, float(1), float(0)), 0, 0, 1));
    const [X, Y, Z] = [x, y, z].map((u) => member(u.name));
    expect(code).toContain(`(${X} > 0f && ${Y} > 0f) || ${Z} > 0f`);
  });

  /**
   * @canon spec-wgsl-converts-a-shift-amount-to-unsigned
   */
  it("converts a shift amount to unsigned on WGSL", () => {
    const a = uniform("int");
    const b = uniform("int");
    const v = uniform("ivec3");
    const scalar = compileWgsl(vec4(a.shiftRight(b).toFloat(), 0, 0, 1));
    expect(scalar).toContain(`${member(a.name)} >> u32(${member(b.name)})`);
    const vector = compileWgsl(vec4(v.shiftLeft(b).toVec3(), 1));
    expect(vector).toContain(`${member(v.name)} << vec3<u32>(u32(${member(b.name)}))`);
  });

  /**
   * @canon spec-wgsl-splats-a-scalar-bitwise-operand-beside-a-vector
   */
  it("splats a scalar bitwise operand beside a vector on WGSL", () => {
    const v = uniform("ivec3");
    const b = uniform("int");
    expect(compileWgsl(vec4(v.bitAnd(b).toVec3(), 1))).toContain(`${member(v.name)} & vec3<i32>(${member(b.name)})`);
  });

  /**
   * @canon spec-glsl-selects-float-vectors-by-a-boolean-vector-through-mix
   */
  it("selects float vectors by a boolean vector through mix on GLSL", () => {
    const condition = uniform("bvec3");
    const a = uniform("vec3");
    const b = uniform("vec3");
    expect(compileGlsl(vec4(select(condition, a, b), 1))).toContain(
      `mix(${b.name}, ${a.name}, vec3(${condition.name}))`,
    );
  });

  /**
   * @canon spec-glsl-selects-integer-vectors-by-a-boolean-vector-one-component-at-a-time
   */
  it("selects integer vectors by a boolean vector one component at a time on GLSL", () => {
    const condition = uniform("bvec3");
    const a = uniform("ivec3");
    const b = uniform("ivec3");
    const code = compileGlsl(vec4(vec3(select(condition, a, b)), 1));
    const component = (i: number) => `(${condition.name})[${i}] ? (${a.name})[${i}] : (${b.name})[${i}]`;
    expect(code).toContain(`ivec3(${component(0)}, ${component(1)}, ${component(2)})`);
    expect(code).not.toContain("mix(");
  });
});

describe("what GLSL and WGSL write for an outputStruct", () => {
  const build = () => Fn(() => outputStruct(vec4(1, 0, 0, 1), float(7)))();

  /**
   * @canon spec-an-output-struct-writes-each-member-at-its-position
   */
  it("writes member i to the output at location i on GLSL", () => {
    const code = compileGlsl(build());
    expect(code).toContain("layout(location=0) out vec4 _rmsl_out0;");
    expect(code).toContain("layout(location=1) out float _rmsl_out1;");
  });

  /**
   * @canon spec-an-output-struct-writes-each-member-at-its-position
   */
  it("writes member i to the output at location i on WGSL", () => {
    const code = compileWgsl(build());
    expect(code).toContain("@location(0) _rmsl_out0: vec4<f32>");
    expect(code).toContain("@location(1) _rmsl_out1: f32");
  });
});

describe("what WGSL writes for a variable", () => {
  /**
   * @canon spec-a-wgsl-variable-is-declared-with-var
   */
  it("declares a variable the program never assigns again with var on WGSL", () => {
    const code = compileWgsl(Fn(() => vec4(float(1).toVar(), 0, 0, 1))());
    expect(code).toMatch(/var _rmsl_\d+: f32 = 1f;/);
    expect(code).not.toMatch(/\blet\b/);
  });

  /**
   * @canon spec-a-wgsl-write-through-a-swizzle-of-several-components-stores-its-value-once
   */
  it("stores the value of a write through a two-component swizzle once on WGSL", () => {
    const code = compileWgsl(
      Fn(() => {
        const v = vec3(1, 2, 3).toVar();
        v.xy.assign(vec3(4, 5, 6).zy);
        return vec4(v, 1);
      })(),
    );
    const temporary = /var (_rmsl_sw\d+): vec2<f32> = vec3<f32>\(4, 5, 6\)\.zy;/.exec(code)?.[1];
    expect(temporary).toBeDefined();
    expect(code.match(/vec3<f32>\(4, 5, 6\)/g)).toHaveLength(1);
    expect(code).toMatch(new RegExp(`(_rmsl_\\d+)\\.x = ${temporary}\\[0\\];\\s*\\1\\.y = ${temporary}\\[1\\];`));
  });
});

/**
 * Two storage buffers, the second made after the first, whose slot names sort
 * the other way round as strings, as `_rmsl_b10` sorts before `_rmsl_b9`.
 */
function buffersAcrossADigit() {
  let first = instancedArray(2, "float");
  let second = instancedArray(2, "float");
  while (first.name.localeCompare(second.name) < 0) {
    first = second;
    second = instancedArray(2, "float");
  }
  return { first, second };
}

/**
 * The binding WGSL declares for the buffer the stage reads `n`th. WGSL names
 * a buffer `_rmsl_sN` by the order the stage first reads it.
 */
const bindingOfRead = (code: string, n: number) =>
  new RegExp(`@binding\\((\\d+)\\) var<storage, [\\w ,]+> _rmsl_s${n}:`).exec(code)?.[1];

describe("where WGSL declares a uniform or a buffer", () => {
  /**
   * @canon spec-a-wgsl-stage-given-the-program-uniforms-declares-every-one
   */
  it("declares every uniform of the program it is given, read or not, on WGSL", () => {
    const read = uniform("float");
    const unread = uniform("vec2");
    const code = compileWgslStage.fragment(vec4(read, 0, 0, 1), {
      uniforms: [read, unread],
    });
    expect(structMembers(code).sort()).toEqual([read.name, unread.name].sort());
  });

  /**
   * The program's uniforms, a texture among them, go to the stage and to the
   * layout alike, as the host packs the buffer.
   *
   * @canon spec-the-wgsl-uniform-layout-leaves-out-a-texture
   */
  it("lays out the uniforms a WGSL stage is given, leaving out a texture as the stage does", () => {
    const uniforms = [uniform("sampler2D"), uniform("float")];
    const [image, value] = uniforms as [any, any];
    const code = compileWgslStage.fragment(texture(image, vec2(0, 0)).add(value), { uniforms });
    const layout = wgslUniformLayout(uniforms.map(wgslUniformDeclaration));
    expect(layout.members.map((member) => member.name)).toEqual(structMembers(code));
    expect(layout.members.map((member) => member.name)).toEqual([value.name]);
  });

  /**
   * @canon spec-a-bool-uniform-travels-as-an-unsigned-integer
   */
  it("declares a bool uniform array and a boolean vector one as u32s for the WGSL layout", () => {
    expect(wgslUniformDeclaration(uniformArray("bool", 4)).type).toBe("u32");
    expect(wgslUniformDeclaration(uniformArray("bvec2", 2)).type).toBe("vec2<u32>");
  });

  /**
   * @canon spec-a-wgsl-stage-refuses-a-uniform-the-given-uniforms-leave-out
   */
  it("refuses a uniform the stage reads that the given uniforms leave out on WGSL", () => {
    const read = uniform("float");
    const other = uniform("float");
    expect(() => compileWgslStage.fragment(vec4(read, 0, 0, 1), { uniforms: [other] })).toThrowRmsl(
      ProgramError,
      "missing-from-list",
      new RegExp(`"${read.name}" is read by this stage but missing`),
    );
  });

  /**
   * @canon spec-a-wgsl-render-stage-declares-its-storage-in-group-three
   */
  it("declares the storage of a render stage in group 3 on WGSL", () => {
    const values = instancedArray(2, "float");
    const code = compileWgslStage.fragment(vec4(values.element(int(1)), 0, 0, 1));
    expect(code).toMatch(/@group\(3\) @binding\(0\) var<storage, read> _rmsl_s\d+: array<f32>;/);
    expect(code).not.toMatch(/@group\([0-2]\)/);
  });

  /**
   * @canon spec-a-wgsl-compute-program-declares-its-storage-in-group-one
   */
  it("declares compute storage in group 1 on WGSL", () => {
    const first = instancedArray(2, "float");
    const second = instancedArray(2, "float");
    const program = Fn(() => {
      first.element(invocationIndex()).assign(second.element(invocationIndex()));
    })().compute(2);
    const { code, resources } = compile({ stage: "compute" }, program);
    const storage = resources.filter((r) => r.kind === "storage");
    expect(storage.map((r) => r.group)).toEqual([1, 1]);
    expect(code.match(/@group\(1\) @binding\(\d\) var<storage/g)).toHaveLength(2);
  });

  /**
   * @canon spec-a-compute-uniform-resource-names-the-rmsl-type-of-its-uniform
   */
  it("names the rmsl type of each uniform of a compute program, not its WGSL spelling", () => {
    const count = uniform("int");
    const offset = uniform("uvec2");
    const gain = uniform("float");
    const out = instancedArray(2, "float");
    const program = Fn(() => {
      out.element(invocationIndex()).assign(count.toFloat().add(offset.x.toFloat()).add(gain));
    })().compute(2);
    const { resources } = compile({ stage: "compute" }, program);
    const typeOf = (name: string) => resources.find((r) => r.kind === "uniform" && r.name === name)?.shaderType;
    expect(typeOf(count.name)).toBe("int");
    expect(typeOf(offset.name)).toBe("uvec2");
    expect(typeOf(gain.name)).toBe("float");
  });

  /**
   * The list names the buffer the program made second first, so the order of
   * the list is all that can put it at binding 0. WGSL names a buffer by the
   * order the stage first reads it, so `second` is `_rmsl_s1`.
   *
   * @canon spec-a-wgsl-render-stage-binds-its-storage-in-the-listed-order
   */
  it("binds the storage of a render stage at its index in the storages list on WGSL", () => {
    const first = instancedArray(2, "float");
    const second = instancedArray(2, "float");
    const code = compileWgslStage.fragment(vec4(first.element(int(0)), second.element(int(0)), 0, 1), {
      storages: [second, first],
    });
    expect(code).toMatch(/@binding\(0\) var<storage, read> _rmsl_s1:/);
    expect(code).toMatch(/@binding\(1\) var<storage, read> _rmsl_s0:/);
  });

  /**
   * The list names a buffer the program does not read first, so the one it
   * reads binds at 1.
   *
   * @canon spec-a-wgsl-compute-program-binds-its-storage-in-the-listed-order
   */
  it("binds compute storage at its index in a list that names a buffer it does not read on WGSL", () => {
    const unread = instancedArray(2, "float");
    const values = instancedArray(2, "float");
    const code = compileWgslStage.compute(
      Fn(() => {
        values.element(invocationIndex()).assign(float(1));
      })(),
      { storages: [unread, values] },
    );
    expect(code).toMatch(/@group\(1\) @binding\(1\) var<storage, read_write> _rmsl_s0:/);
    expect(code).not.toMatch(/@binding\(0\) var<storage/);
  });

  /**
   * @canon spec-a-wgsl-stage-refuses-a-buffer-the-given-storages-leave-out
   */
  it("refuses a buffer the stage reads that the given storages leave out on WGSL", () => {
    const listed = instancedArray(2, "float");
    const missing = instancedArray(2, "float");
    expect(() =>
      compileWgslStage.fragment(vec4(listed.element(int(0)), missing.element(int(0)), 0, 1), { storages: [listed] }),
    ).toThrowRmsl(ProgramError, "missing-from-list", new RegExp(`"${missing.name}" is read by this stage but missing`));
  });

  /**
   * @canon spec-the-storages-list-holds-the-storage-nodes
   */
  it("takes the storage nodes themselves as the storages list on WGSL", () => {
    const first = instancedArray(2, "float");
    const second = instancedArray(2, "float");
    const code = compileWgslStage.fragment(vec4(first.element(int(0)), second.element(int(0)), 0, 1), {
      storages: [second, first],
    });
    expect(bindingOfRead(code, 1)).toBe("0");
    expect(bindingOfRead(code, 0)).toBe("1");
  });

  /**
   * @canon spec-a-wgsl-compute-program-binds-its-storage-in-the-listed-order
   */
  it("binds compute storage at its index in the storages list on WGSL", () => {
    const first = instancedArray(2, "float");
    const second = instancedArray(2, "float");
    const code = compileWgslStage.compute(
      Fn(() => {
        second.element(invocationIndex()).assign(first.element(invocationIndex()));
      })(),
      { storages: [second, first] },
    );
    expect(bindingOfRead(code, 1)).toBe("0");
    expect(bindingOfRead(code, 0)).toBe("1");
  });

  /**
   * @canon spec-a-wgsl-program-given-no-storage-list-binds-in-creation-order
   */
  it("binds storage in the order the program made it when no list is given on WGSL", () => {
    const { first, second } = buffersAcrossADigit();
    const compute = compileWgslStage.compute(
      Fn(() => {
        second.element(invocationIndex()).assign(first.element(invocationIndex()));
      })(),
    );
    const render = compileWgslStage.fragment(vec4(first.element(int(0)), second.element(int(0)), 0, 1));
    for (const code of [compute, render]) {
      expect(bindingOfRead(code, 0)).toBe("0");
      expect(bindingOfRead(code, 1)).toBe("1");
    }
  });

  /**
   * The last two of the textures made sort the other way round as strings, as
   * `_rmsl_u10` sorts before `_rmsl_u9`.
   *
   * @canon spec-a-wgsl-stage-given-no-samplers-binds-its-textures-in-creation-order
   */
  it("binds textures and their samplers in the order the program made them on WGSL", () => {
    const made = [uniform("sampler2D"), uniform("sampler2D")];
    while (made.at(-1)!.name.localeCompare(made.at(-2)!.name) > 0) made.push(uniform("sampler2D"));
    const [first, second] = made.slice(-2);
    const code = compileWgslStage.fragment(Fn(() => texture(first!, vec2(0, 0)).add(texture(second!, vec2(0, 0))))());
    expect(code).toContain(`@group(1) @binding(0) var ${first!.name}:`);
    expect(code).toContain(`@group(1) @binding(1) var ${second!.name}:`);
    expect(code).toContain(`@group(2) @binding(0) var ${first!.name}_s:`);
    expect(code).toContain(`@group(2) @binding(1) var ${second!.name}_s:`);
  });

  /**
   * The last two of the textures made sort the other way round as strings, as
   * `_rmsl_u10` sorts before `_rmsl_u9`. The list puts the integer texture,
   * made last, first, so its order is neither their creation order nor the
   * order of their names.
   *
   * @canon spec-a-wgsl-stage-binds-its-textures-in-the-listed-order
   * @canon spec-the-samplers-list-holds-the-texture-nodes
   */
  it("binds textures and their samplers at their index in the samplers list on WGSL", () => {
    const made = [uniform("sampler2D"), uniform("sampler2D")];
    while (made.at(-1)!.name.localeCompare(made.at(-2)!.name) > 0) made.push(uniform("sampler2D"));
    const [first, second] = made.slice(-2) as [UniformNode<"sampler2D">, UniformNode<"sampler2D">];
    const integer = uniform("usampler2D");
    const samplers = [integer, first, second];
    expect(samplers.map((t) => t.name)).not.toEqual(samplers.map((t) => t.name).sort());
    const fragment = () =>
      Fn(() =>
        texture(first, vec2(0, 0))
          .add(texture(second, vec2(0, 0)))
          .add(texture(integer, uvec2(0, 0)).toVec4()),
      )();
    const code = compileWgslStage.fragment(fragment(), { samplers });
    expect(code).toContain(`@group(1) @binding(0) var ${integer.name}:`);
    expect(code).toContain(`@group(1) @binding(1) var ${first.name}:`);
    expect(code).toContain(`@group(1) @binding(2) var ${second.name}:`);
    expect(code).toContain(`@group(2) @binding(0) var ${first.name}_s:`);
    expect(code).toContain(`@group(2) @binding(1) var ${second.name}_s:`);
  });

  /**
   * @canon spec-a-wgsl-stage-refuses-a-samplers-list-of-anything-but-textures-each-once
   */
  it("refuses a samplers list that holds a value uniform or one texture twice on WGSL", () => {
    const image = uniform("sampler2D");
    const gain = uniform("float");
    const fragment = Fn(() => texture(image, vec2(0, 0)))();
    // @ts-expect-error: the samplers list takes only texture uniforms
    expect(() => compileWgslStage.fragment(fragment, { samplers: [image, gain] })).toThrowRmsl(
      ProgramError,
      "not-a-texture",
      `"${gain.name}" is a float uniform, not a texture`,
    );
    expect(() => compileWgslStage.fragment(fragment, { samplers: [image, image] })).toThrowRmsl(
      ProgramError,
      "listed-twice",
      `the samplers passed to the compiler list the texture "${image.name}" twice`,
    );
  });

  /**
   * @canon spec-a-wgsl-stage-refuses-a-storage-buffer-listed-twice
   */
  it("refuses a storages list that holds one buffer twice on WGSL", () => {
    const values = instancedArray(2, "float");
    expect(() =>
      compileWgslStage.fragment(vec4(values.element(int(0)), 0, 0, 1), { storages: [values, values] }),
    ).toThrowRmsl(
      ProgramError,
      "listed-twice",
      `the storages passed to the compiler list the storage buffer "${values.name}" twice`,
    );
  });

  /**
   * @canon spec-a-wgsl-stage-refuses-a-texture-the-given-samplers-leave-out
   */
  it("refuses a WGSL stage that samples a texture the samplers list leaves out", () => {
    const listed = uniform("sampler2D");
    const missing = uniform("sampler2D");
    const fragment = Fn(() => texture(listed, vec2(0, 0)).add(texture(missing, vec2(0, 0))))();
    expect(() => compileWgslStage.fragment(fragment, { samplers: [listed] })).toThrowRmsl(
      ProgramError,
      "missing-from-list",
      `the texture "${missing.name}" is read by this stage but missing from the samplers passed to the compiler`,
    );
  });

  /**
   * The function samples the texture made second before the one made first,
   * so visit order would bind their samplers the other way round.
   *
   * @canon spec-a-wgsl-stage-given-no-samplers-binds-its-textures-in-creation-order
   */
  it("binds a WGSL function's samplers in the order of its textures", () => {
    const first = uniform("sampler2D");
    const second = uniform("sampler2D");
    const code = compileWgslFn(() => texture(second, vec2(0, 0)).add(texture(first, vec2(0, 0))), {
      name: "sampleBoth",
      params: [],
    });
    expect(code).toContain(`@group(1) @binding(0) var ${first.name}:`);
    expect(code).toContain(`@group(1) @binding(1) var ${second.name}:`);
    expect(code).toContain(`@group(2) @binding(0) var ${first.name}_s:`);
    expect(code).toContain(`@group(2) @binding(1) var ${second.name}_s:`);
  });

  /**
   * Two uniforms of one alignment, `zeta` made first, so creation order puts
   * it before `alpha` in the struct.
   *
   * @canon spec-a-wgsl-stage-hands-its-uniforms-to-the-layout-in-creation-order
   */
  it("declares uniforms of one alignment in the order the program made them on WGSL", () => {
    const zeta = uniformRaw("zeta", "float");
    const alpha = uniformRaw("alpha", "float");
    const code = compileWgslStage.fragment(vec4(zeta, alpha, 0, 1));
    const members = [...(/struct _RmslUniforms \{([\s\S]*?)\n\};/.exec(code)?.[1] ?? "").matchAll(/^\s*(\w+):/gm)].map(
      (m) => m[1],
    );
    expect(members).toEqual(["zeta", "alpha"]);
  });

  /**
   * @canon spec-the-uniforms-list-holds-the-uniform-nodes
   */
  it("takes the uniform nodes themselves as the uniforms list on WGSL", () => {
    const read = uniform("float");
    const other = uniform("vec2");
    const code = compileWgslStage.fragment(vec4(read, 0, 0, 1), { uniforms: [read, other] });
    expect(code).toMatch(new RegExp(`${other.name}: vec2<f32>`));
  });

  /**
   * A compute program declares its textures in group 1 after its storage
   * buffers, so a texture and a buffer have no binding in common.
   *
   * @canon spec-a-texture-keeps-a-binding-of-its-own
   */
  it("gives a texture read by a compute program a binding no storage buffer has", () => {
    const values = instancedArray(2, "float");
    const image = uniform("isampler2D");
    const program = Fn(() => {
      values.element(invocationIndex()).assign(texture(image, ivec2(0, 0)).x.toFloat());
    })();
    const bindings = bindingsOf(compile({ stage: "compute" }, program).code);
    expect(new Set(bindings).size).toBe(bindings.length);
  });

  /**
   * Given the textures of the whole program, a vertex and a fragment stage
   * that read different textures bind them at different bindings, and a
   * texture has the same binding in both stages.
   *
   * @canon spec-a-texture-keeps-a-binding-of-its-own
   */
  it("binds two textures read by different render stages at different bindings on WGSL", () => {
    const near = uniform("isampler2D");
    const far = uniform("isampler2D");
    const vertex = Fn(() => {
      builtinPosition().assign(vec4(texture(near, ivec2(0, 0)).x.toFloat(), 0, 0, 1));
    })();
    const fragment = Fn(() => vec4(texture(far, ivec2(0, 0)).x.toFloat(), 0, 0, 1))();
    const samplers = [near, far];
    const vertexCode = compileWgslStage.vertex(vertex, { samplers });
    const fragmentCode = compileWgslStage.fragment(fragment, { samplers });
    expect(bindingOf(vertexCode, near.name)).toBeDefined();
    expect(bindingOf(vertexCode, near.name)).not.toBe(bindingOf(vertexCode, far.name));
    expect(bindingOf(vertexCode, near.name)).toBe(bindingOf(fragmentCode, near.name));
    expect(bindingOf(vertexCode, far.name)).toBe(bindingOf(fragmentCode, far.name));
  });

  /**
   * A texture sampled in the vertex stage is read at level 0, because only a
   * fragment stage has the derivatives that pick a level, and a stage
   * numbers the samplers of the textures it samples in the order the program
   * made the textures.
   *
   * @canon spec-a-texture-is-bound-to-every-stage-that-samples-it
   * @canon spec-a-wgsl-stage-given-no-samplers-binds-its-textures-in-creation-order
   */
  it("samples a texture in the vertex stage at level 0, and numbers samplers in creation order on WGSL", () => {
    const zebra = uniformRaw("zebra", "sampler2D");
    const apple = uniformRaw("apple", "sampler2D");
    const vertex = Fn(() => {
      builtinPosition().assign(vec4(texture(zebra, vec2(0, 0)).x.add(texture(apple, vec2(0, 0)).x), 0, 0, 1));
    })();
    const code = compileWgslStage.vertex(vertex);
    expect(code).toContain("textureSampleLevel(");
    expect(code).not.toMatch(/textureSample\(/);
    expect(bindingOf(code, `${zebra.name}_s`)).toBe("2:0");
    expect(bindingOf(code, `${apple.name}_s`)).toBe("2:1");
  });

  /**
   * A texture of a sampler type that WGSL has no texture for is refused.
   *
   * @canon spec-a-texture-keeps-a-binding-of-its-own
   */
  it("refuses a sampler of a type WGSL has no texture for", () => {
    const vertex = Fn(() => {
      builtinPosition().assign(vec4(0, 0, 0, 1));
    })();
    expect(() => compileWgslStage.vertex(vertex, { samplers: [uniformRaw("map", "sampler9D" as any)] })).toThrowRmsl(
      ProgramError,
      "not-a-texture",
    );
  });
});

/**
 * One entry of a GPU page. It sets a uniform and an attribute before
 * `attach`, then draws a full-screen triangle and reads its centre.
 */
const BEFORE_ATTACH = `
import { Fn, attribute, builtinPosition, uniform, vec4 } from "../rmsl";
import { createGlsl } from "../glsl";
import { createWgsl } from "../wgsl";
${READ_PIXEL}
const TRIANGLE = Float32Array.of(-1, -1, 0, 3, -1, 0, -1, 3, 0);
const canvas = () => {
  const c = document.createElement("canvas");
  c.width = 4;
  c.height = 4;
  return c;
};
const program = () => {
  const position = attribute("vec3");
  const green = uniform("float");
  const vertex = Fn(() => { builtinPosition().assign(vec4(position, 1)); })();
  const fragment = Fn(() => vec4(0, green, 0, 1))();
  return { position, green, vertex, fragment };
};
globalThis.__rmslUnitsGlsl = async () => {
  const { position, green, vertex, fragment } = program();
  const target = canvas();
  const adapter = createGlsl(vertex, fragment);
  adapter.setUniform(green, 1);
  adapter.setAttribute(position, TRIANGLE);
  adapter.attach(target);
  adapter.draw({ count: 3 });
  return readPixel(target, 1, 2);
};
globalThis.__rmslUnitsWgsl = async () => {
  const { position, green, vertex, fragment } = program();
  const target = canvas();
  const adapter = createWgsl({ vertex, fragment });
  adapter.setUniform(green, 1);
  adapter.setAttribute(position, TRIANGLE);
  await adapter.attach(target);
  adapter.draw({ count: 3 });
  await adapter.device().queue.onSubmittedWorkDone();
  return readPixel(target, 1, 2);
};
`;

const GREEN = { r: 0, g: 255, b: 0, a: 255 };

/**
 * One entry of a GPU page for each adapter. Each draws a triangle over the
 * lower-left half of a 16×16 canvas, whose vertices write their position only
 * where `skip` is 0. It returns how many pixels each draw lit: no vertex
 * writing, the third not writing, the third writing `vec4(0)`, and every one
 * writing.
 */
const UNWRITTEN_POSITION = `${READ_IMAGE}
import { Fn, If, attribute, builtinPosition, vec4 } from "../rmsl";
import { createGlsl } from "../glsl";
import { createWgsl } from "../wgsl";
import { createJs } from "../js";
import { createWasm } from "../wasm";
const TRIANGLE = Float32Array.of(-1, -1, 0, 1, -1, 0, -1, 1, 0);
const CASES = { noneWrites: [1, 1, 1], thirdSkips: [0, 0, 1], thirdWritesZero: [0, 0, 1], everyWrites: [0, 0, 0] };
const litPixels = (canvas) => {
  const data = readImage(canvas);
  let lit = 0;
  for (let i = 0; i < data.length; i += 4) if (data[i] > 0) lit++;
  return lit;
};
const drawEach = async (create, settle) => {
  const lit = {};
  for (const [name, skips] of Object.entries(CASES)) {
    const position = attribute("vec3");
    const skip = attribute("float");
    const vertex = Fn(() => {
      If(skip.lessThan(0.5), () => {
        builtinPosition().assign(vec4(position, 1));
      }).Else(() => {
        if (name === "thirdWritesZero") builtinPosition().assign(vec4(0, 0, 0, 0));
      });
    })();
    const fragment = Fn(() => vec4(1, 0, 0, 1))();
    const canvas = document.createElement("canvas");
    canvas.width = 16;
    canvas.height = 16;
    const adapter = create(vertex, fragment);
    await adapter.attach(canvas);
    adapter.setAttribute(position, TRIANGLE);
    adapter.setAttribute(skip, Float32Array.from(skips));
    adapter.draw({ count: 3 });
    if (settle) await settle(adapter);
    lit[name] = litPixels(canvas);
  }
  return lit;
};
globalThis.__rmslUnwrittenGlsl = () => drawEach((vertex, fragment) => createGlsl(vertex, fragment));
globalThis.__rmslUnwrittenJs = () => drawEach((vertex, fragment) => createJs(() => vertex, () => fragment));
globalThis.__rmslUnwrittenWasm = () => drawEach((vertex, fragment) => createWasm(() => vertex, () => fragment));
globalThis.__rmslUnwrittenWgsl = () =>
  drawEach(
    (vertex, fragment) => createWgsl({ vertex, fragment }),
    (adapter) => adapter.device().queue.onSubmittedWorkDone(),
  );
`;

const unwrittenPosition = (entry: string, onWebGpu = false) =>
  (onWebGpu ? runInWebGpuPage : runInGpuPage)(UNWRITTEN_POSITION, entry, new URL(".", import.meta.url).pathname);

describe("a path of a vertex stage that writes no position", () => {
  /**
   * @canon spec-a-path-that-writes-no-position-gives-zero
   */
  it("starts the GLSL position at vec4(0) in a stage that assigns it, and only there", () => {
    const position = attribute("vec3");
    const assigns = compileGlsl.vertex(Fn(() => builtinPosition().assign(vec4(position, 1)))());
    expect(assigns).toMatch(/void main\(void\) \{\n {2}gl_Position = vec4\(0\.0\);/);
    expect(assigns.match(/gl_Position = /g)).toHaveLength(2);
    const returns = compileGlsl.vertex(Fn(() => vec4(position, 1))());
    expect(returns.match(/gl_Position = /g)).toHaveLength(1);
  });

  /**
   * @canon spec-a-path-that-writes-no-position-gives-zero
   */
  it.skipIf(!GPU_ENABLED)(
    "draws as vec4(0) on GLSL, JS and WASM",
    async () => {
      for (const entry of ["__rmslUnwrittenGlsl", "__rmslUnwrittenJs", "__rmslUnwrittenWasm"]) {
        const lit = await unwrittenPosition(entry);
        expect(lit.noneWrites, entry).toBe(0);
        expect(lit.thirdSkips, entry).toBe(lit.thirdWritesZero);
        expect(lit.everyWrites, entry).toBeGreaterThan(0);
      }
    },
    120_000,
  );

  /**
   * @canon spec-a-path-that-writes-no-position-gives-zero
   */
  it.skipIf(!GPU_ENABLED || !WEBGPU)(
    "draws as vec4(0) on WGSL",
    async () => {
      const lit = await unwrittenPosition("__rmslUnwrittenWgsl", true);
      expect(lit.noneWrites).toBe(0);
      expect(lit.thirdSkips).toBe(lit.thirdWritesZero);
      expect(lit.everyWrites).toBeGreaterThan(0);
    },
    120_000,
  );
});

/**
 * Entries of a GPU page that draw shapes on a 4×4 canvas, every shape in both
 * windings, and return the alpha of each pixel: a row of numbers for each row
 * from the top. A shape is a list of triangles, its corners in pixels. Every
 * edge of a shape that crosses the canvas passes through pixel centres.
 * `__rmslEdgePixelsOnWebGl` draws with `createGlsl`, `createJs` and
 * `createWasm`, and draws the columns once more with `createGlsl` on a canvas
 * whose context multisamples. `__rmslEdgePixelsOnWebGpu` draws with
 * `createGlsl` and `createWgsl`. The draws of `createGlsl` come with the name
 * of the driver that made them.
 */
const EDGE_PIXELS = `${READ_IMAGE}
import { Fn, attribute, vec4 } from "../rmsl";
import { createGlsl } from "../glsl";
import { createWgsl } from "../wgsl";
import { createJs } from "../js";
import { createWasm } from "../wasm";
const rectangle = (l, t, r, b) => [[[l, t], [r, t], [l, b]], [[l, b], [r, t], [r, b]]];
const SHAPES = {
  columns: rectangle(0.5, -1, 2.5, 5),
  rows: rectangle(-1, 0.5, 5, 2.5),
  topLeft: [[[0.5, 0.5], [0.5, 9], [9, 0.5]]],
  bottomRight: [[[2.5, 2.5], [2.5, -9], [-9, 2.5]]],
  slopeOneAbove: [[[-2, -2], [6, -2], [6, 6]]],
  slopeOneBelow: [[[-2, -2], [6, 6], [-2, 6]]],
  slopeMinusOneAbove: [[[-2, 6], [6, -2], [-2, -2]]],
  slopeMinusOneBelow: [[[-2, 6], [6, -2], [6, 6]]],
  slopeHalfAbove: [[[-4.5, -2], [11.5, 6], [11.5, -2]]],
  slopeHalfBelow: [[[-4.5, -2], [11.5, 6], [-4.5, 6]]],
  slopeTwoRight: [[[-0.75, -2], [3.25, 6], [20, -2]]],
  slopeTwoLeft: [[[-0.75, -2], [3.25, 6], [-20, 6]]],
  splitColumnLeft: rectangle(-1, -1, 1.5, 5),
  splitColumnRight: rectangle(1.5, -1, 5, 5),
  splitRowAbove: rectangle(-1, -1, 5, 1.5),
  splitRowBelow: rectangle(-1, 1.5, 5, 5),
};
const reversed = (triangles) => triangles.map(([first, second, third]) => [third, second, first]);
const clip = ([x, y]) => [(x / 4) * 2 - 1, 1 - (y / 4) * 2];
const alphas = (canvas) => {
  const data = readImage(canvas);
  return Array.from({ length: 4 }, (_, y) => Array.from({ length: 4 }, (_, x) => data[(y * 4 + x) * 4 + 3]).join(" ")).join("/");
};
const drawn = async (triangles, create, canvas = document.createElement("canvas")) => {
  const pos = attribute("vec2");
  canvas.width = 4;
  canvas.height = 4;
  const adapter = create(Fn(() => vec4(pos, 0, 1))(), Fn(() => vec4(1, 0, 0, 1))());
  await adapter.attach(canvas);
  adapter.setAttribute(pos, Float32Array.from(triangles.flat().flatMap(clip)));
  adapter.draw({ count: triangles.length * 3 });
  if (adapter.device) await adapter.device().queue.onSubmittedWorkDone();
  return alphas(canvas);
};
const drawEach = async (create) => {
  const shapes = {};
  for (const [name, triangles] of Object.entries(SHAPES)) {
    shapes[name] = [await drawn(triangles, create), await drawn(reversed(triangles), create)];
  }
  return shapes;
};
const glsl = (vertex, fragment) => createGlsl(vertex, fragment);
const drawEachOnGlsl = async () => {
  const gl = document.createElement("canvas").getContext("webgl2");
  const info = gl.getExtension("WEBGL_debug_renderer_info");
  const renderer = info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  return { renderer, shapes: await drawEach(glsl) };
};
globalThis.__rmslEdgePixelsOnWebGl = async () => {
  const multisampledCanvas = document.createElement("canvas");
  multisampledCanvas.getContext("webgl2", { antialias: true });
  // Drawn first: the page loses its oldest WebGL context past 16 of them.
  const multisampled = await drawn(SHAPES.columns, glsl, multisampledCanvas);
  return {
    glsl: await drawEachOnGlsl(),
    js: await drawEach((vertex, fragment) => createJs(() => vertex, () => fragment)),
    wasm: await drawEach((vertex, fragment) => createWasm(() => vertex, () => fragment)),
    multisampled,
  };
};
globalThis.__rmslEdgePixelsOnWebGpu = async () => ({
  glsl: await drawEachOnGlsl(),
  wgsl: await drawEach((vertex, fragment) => createWgsl({ vertex, fragment })),
});
`;

type EdgePixels = Record<string, [string, string]>;

/**
 * What a target that follows WebGPU's top-left rule draws for each shape of
 * `EDGE_PIXELS`, in both windings.
 */
const TOP_LEFT_RULE: Record<string, string> = {
  columns: "255 255 0 0/255 255 0 0/255 255 0 0/255 255 0 0",
  rows: "255 255 255 255/255 255 255 255/0 0 0 0/0 0 0 0",
  topLeft: "255 255 255 255/255 255 255 255/255 255 255 255/255 255 255 255",
  bottomRight: "255 255 0 0/255 255 0 0/0 0 0 0/0 0 0 0",
  slopeOneAbove: "255 255 255 255/0 255 255 255/0 0 255 255/0 0 0 255",
  slopeOneBelow: "0 0 0 0/255 0 0 0/255 255 0 0/255 255 255 0",
  slopeMinusOneAbove: "255 255 255 0/255 255 0 0/255 0 0 0/0 0 0 0",
  slopeMinusOneBelow: "0 0 0 255/0 0 255 255/0 255 255 255/255 255 255 255",
  slopeHalfAbove: "255 255 255 255/0 0 255 255/0 0 0 0/0 0 0 0",
  slopeHalfBelow: "0 0 0 0/255 255 0 0/255 255 255 255/255 255 255 255",
  slopeTwoRight: "255 255 255 255/0 255 255 255/0 255 255 255/0 0 255 255",
  slopeTwoLeft: "0 0 0 0/255 0 0 0/255 0 0 0/255 255 0 0",
  splitColumnLeft: "255 0 0 0/255 0 0 0/255 0 0 0/255 0 0 0",
  splitColumnRight: "0 255 255 255/0 255 255 255/0 255 255 255/0 255 255 255",
  splitRowAbove: "255 255 255 255/0 0 0 0/0 0 0 0/0 0 0 0",
  splitRowBelow: "0 0 0 0/255 255 255 255/255 255 255 255/255 255 255 255",
};

/**
 * What the measured GLSL drivers draw for the shapes of `EDGE_PIXELS` with a
 * horizontal edge through pixel centres, which they give to the triangle
 * above. They draw every other shape as `TOP_LEFT_RULE` does.
 */
const MEASURED_GLSL_HORIZONTAL: Record<string, string> = {
  rows: "0 0 0 0/255 255 255 255/255 255 255 255/0 0 0 0",
  topLeft: "0 0 0 0/255 255 255 255/255 255 255 255/255 255 255 255",
  bottomRight: "255 255 0 0/255 255 0 0/255 255 0 0/0 0 0 0",
  splitRowAbove: "255 255 255 255/255 255 255 255/0 0 0 0/0 0 0 0",
  splitRowBelow: "0 0 0 0/0 0 0 0/255 255 255 255/255 255 255 255",
};

/** The drivers whose GLSL rule for a pixel centre on an edge was measured: ANGLE on SwiftShader and on Metal. */
const MEASURED_GLSL_DRIVER = /SwiftShader|ANGLE Metal Renderer/;

/** The shapes of `EDGE_PIXELS` that together cover the canvas, sharing one edge through pixel centres. */
const COMPLEMENTARY = [
  ["slopeOneAbove", "slopeOneBelow"],
  ["slopeMinusOneAbove", "slopeMinusOneBelow"],
  ["slopeHalfAbove", "slopeHalfBelow"],
  ["slopeTwoRight", "slopeTwoLeft"],
  ["splitColumnLeft", "splitColumnRight"],
  ["splitRowAbove", "splitRowBelow"],
];

/** The alpha of each pixel of a pattern of `EDGE_PIXELS` that `first` and `second` shade, added. */
const added = (first: string, second: string) => {
  const other = second.split(/[ /]/).map(Number);
  return first.split(/[ /]/).map((alpha, i) => Number(alpha) + other[i]!);
};

/** Whether every pixel of every shape of `EDGE_PIXELS` has alpha 0 or 255. */
const shadedWholeOrNotAtAll = (shapes: EdgePixels) =>
  Object.values(shapes)
    .flat()
    .every((pattern) => pattern.split(/[ /]/).every((alpha) => alpha === "0" || alpha === "255"));

/** The shapes of `EDGE_PIXELS` mapped to their two windings drawn as `TOP_LEFT_RULE` draws them. */
const byTopLeftRule = (names: string[]) =>
  Object.fromEntries(names.map((name) => [name, [TOP_LEFT_RULE[name], TOP_LEFT_RULE[name]]]));

describe("a pixel an edge of a triangle crosses", () => {
  const dir = new URL(".", import.meta.url).pathname;
  let drawing:
    | Promise<{
        glsl: { renderer: string; shapes: EdgePixels }[];
        wgsl?: EdgePixels;
        js: EdgePixels;
        wasm: EdgePixels;
        multisampled: string;
      }>
    | undefined;
  /**
   * What `EDGE_PIXELS` returns: `createGlsl` on WebGL 2 on SwiftShader and on
   * the GPU through the browser of the WebGPU tests, `createWgsl` when a WebGPU
   * device is there, and `createJs` and `createWasm`. Each page bundles and
   * draws once for every test, the two pages at once.
   */
  const drawnOnEach = () =>
    (drawing ??= (async () => {
      const [webgl, webgpu] = await Promise.all([
        runInGpuPage(EDGE_PIXELS, "__rmslEdgePixelsOnWebGl", dir),
        WEBGPU ? runInWebGpuPage(EDGE_PIXELS, "__rmslEdgePixelsOnWebGpu", dir) : undefined,
      ]);
      return { ...webgl, glsl: [webgl.glsl, ...(webgpu ? [webgpu.glsl] : [])], wgsl: webgpu?.wgsl };
    })());

  /**
   * @canon spec-the-glsl-adapter-makes-its-context-with-one-sample
   */
  it.skipIf(!GPU_ENABLED)(
    "is shaded whole or not at all on createGlsl",
    async () => {
      for (const { renderer, shapes } of (await drawnOnEach()).glsl)
        expect(shadedWholeOrNotAtAll(shapes), `${renderer}: ${JSON.stringify(shapes)}`).toBe(true);
    },
    120_000,
  );

  /**
   * @canon spec-the-wgsl-adapter-draws-with-one-sample
   */
  it.skipIf(!GPU_ENABLED || !WEBGPU)(
    "is shaded whole or not at all on createWgsl",
    async () => {
      const { wgsl } = await drawnOnEach();
      expect(shadedWholeOrNotAtAll(wgsl!), JSON.stringify(wgsl)).toBe(true);
    },
    120_000,
  );

  /**
   * @canon spec-a-cpu-rasterizer-shades-a-pixel-whole-or-not-at-all
   */
  it.skipIf(!GPU_ENABLED)(
    "is shaded whole or not at all on createJs and createWasm",
    async () => {
      const { js, wasm } = await drawnOnEach();
      expect(shadedWholeOrNotAtAll(js), JSON.stringify(js)).toBe(true);
      expect(shadedWholeOrNotAtAll(wasm), JSON.stringify(wasm)).toBe(true);
    },
    120_000,
  );

  /**
   * @canon spec-a-wgsl-target-shades-a-pixel-centre-on-a-top-or-left-edge
   */
  it.skipIf(!GPU_ENABLED || !WEBGPU)(
    "on a top or a left edge, horizontal, vertical or sloped, is shaded on createWgsl",
    async () => {
      expect((await drawnOnEach()).wgsl).toEqual(byTopLeftRule(Object.keys(TOP_LEFT_RULE)));
    },
    120_000,
  );

  /**
   * @canon spec-a-rasterizer-shades-a-pixel-centre-on-a-top-or-left-edge
   */
  it.skipIf(!GPU_ENABLED)(
    "on a top or a left edge, horizontal, vertical or sloped, is shaded on createJs and createWasm",
    async () => {
      const { js, wasm } = await drawnOnEach();
      expect(js).toEqual(byTopLeftRule(Object.keys(TOP_LEFT_RULE)));
      expect(wasm).toEqual(byTopLeftRule(Object.keys(TOP_LEFT_RULE)));
    },
    120_000,
  );

  /**
   * Whatever the driver, each centre on the shared edge of two shapes that
   * cover the canvas goes to exactly one of them, in either winding.
   *
   * @canon exception-a-glsl-target-leaves-the-pixel-centre-on-a-shared-edge-to-the-driver
   */
  it.skipIf(!GPU_ENABLED)(
    "on an edge two triangles share is shaded by exactly one of them on createGlsl",
    async () => {
      for (const { renderer, shapes } of (await drawnOnEach()).glsl) {
        for (const [first, second] of COMPLEMENTARY) {
          expect(shapes[first!]![0], `${renderer}: ${first} in both windings`).toBe(shapes[first!]![1]);
          expect(shapes[second!]![0], `${renderer}: ${second} in both windings`).toBe(shapes[second!]![1]);
          const sum = added(shapes[first!]![0], shapes[second!]![0]);
          expect(sum, `${renderer}: ${first} and ${second}`).toEqual(sum.map(() => 255));
        }
      }
    },
    120_000,
  );

  /**
   * The measured drivers follow WebGPU's rule but on a horizontal edge, whose
   * centres they give to the triangle above. Another driver is checked only by
   * the test above.
   *
   * @canon exception-a-glsl-target-leaves-the-pixel-centre-on-a-shared-edge-to-the-driver
   */
  it.skipIf(!GPU_ENABLED)(
    "on a horizontal edge belongs to the triangle above it on createGlsl, on the measured drivers",
    async () => {
      const expected = Object.fromEntries(
        Object.entries({ ...TOP_LEFT_RULE, ...MEASURED_GLSL_HORIZONTAL }).map(([name, pattern]) => [
          name,
          [pattern, pattern],
        ]),
      );
      for (const { renderer, shapes } of (await drawnOnEach()).glsl) {
        if (MEASURED_GLSL_DRIVER.test(renderer)) expect(shapes, renderer).toEqual(expected);
      }
    },
    120_000,
  );

  /**
   * The rectangle's edges cross the centres of columns 0 and 2, which take a
   * part of the colour. How large a part depends on the sample pattern.
   *
   * @canon spec-the-glsl-adapter-draws-with-the-samples-of-a-context-the-canvas-has
   */
  it.skipIf(!GPU_ENABLED)(
    "takes part of the colour on createGlsl attached to a canvas whose context multisamples",
    async () => {
      const { multisampled } = await drawnOnEach();
      for (const row of multisampled.split("/")) {
        const [first, second, third, fourth] = row.split(" ").map(Number);
        expect(first, multisampled).toBeGreaterThan(0);
        expect(first, multisampled).toBeLessThan(255);
        expect(second, multisampled).toBe(255);
        expect(third, multisampled).toBeGreaterThan(0);
        expect(third, multisampled).toBeLessThan(255);
        expect(fourth, multisampled).toBe(0);
      }
    },
    120_000,
  );
});

describe("setUniform on a sampler slot of createGlsl before attach", () => {
  /**
   * @canon spec-the-glsl-adapter-names-the-call-that-sets-a-sampler-slot
   */
  it("refuses a storage buffer's and a texture's slot at the call", () => {
    const colours = instancedArray(4, "float");
    const image = uniform("sampler2D");
    const position = uniform("vec4");
    const adapter = createGlsl(
      Fn(() => builtinPosition().assign(position))() as any,
      Fn(() => vec4(colours.element(int(0)), texture(image, vec2(0, 0)).x, 0, 1))() as any,
    );
    expect(() => adapter.setUniform(colours.name, 1)).toThrowRmsl(
      UsageError,
      "wrong-setter",
      /storage buffer.*setAttribute/,
    );
    expect(() => adapter.setUniform(image.name, 1)).toThrowRmsl(UsageError, "wrong-setter", /texture.*setTexture/);
  });
});

describe("an integer attribute of createGlsl", () => {
  /**
   * The stub context reports one active attribute of each kind.
   *
   * @canon spec-the-glsl-adapter-uploads-an-integer-attribute-of-its-own-array-type-as-given
   */
  it("uploads an Int32Array or Uint32Array of its kind as given", () => {
    const ids = attribute("int");
    const flags = attribute("uint");
    const { gl, calls } = stubGl(
      { width: 1, height: 1 },
      {
        getProgramParameter: (_program: unknown, name: number) =>
          name === gl.ACTIVE_ATTRIBUTES ? 2 : name === gl.ACTIVE_UNIFORMS ? 0 : true,
        getActiveAttrib: (_program: unknown, i: number) =>
          i === 0 ? { name: ids.name, type: gl.INT, size: 1 } : { name: flags.name, type: gl.UNSIGNED_INT, size: 1 },
      },
    );
    const adapter = createGlsl(
      Fn(() => builtinPosition().assign(vec4(ids.toFloat(), flags.toFloat(), 0, 1)))() as any,
      Fn(() => vec4(0, 1, 0, 1))() as any,
    );
    adapter.attach({ getContext: () => gl } as any);
    const idData = Int32Array.of(1, 2, 3);
    const flagData = Uint32Array.of(4, 5, 6);
    adapter.setAttribute(ids.name, idData);
    adapter.setAttribute(flags.name, flagData);
    const uploads = calls.filter((c) => c.name === "bufferData").map((c) => c.args[1]);
    expect(uploads).toContain(idData);
    expect(uploads).toContain(flagData);
  });
});

describe("the array a GPU adapter takes for an attribute", () => {
  /**
   * The stub context reports an `ivec2`, a `uint` and a `vec3` attribute.
   *
   * @canon spec-an-attribute-takes-only-an-array-of-its-own-kind
   */
  it("refuses an array of another kind on createGlsl, named by node or by slot", () => {
    const ids = attribute("ivec2");
    const flags = attribute("uint");
    const position = attribute("vec3");
    const { gl } = stubGl(
      { width: 1, height: 1 },
      {
        getProgramParameter: (_program: unknown, name: number) =>
          name === gl.ACTIVE_ATTRIBUTES ? 3 : name === gl.ACTIVE_UNIFORMS ? 0 : true,
        getActiveAttrib: (_program: unknown, i: number) =>
          [
            { name: ids.name, type: gl.INT_VEC2, size: 1 },
            { name: flags.name, type: gl.UNSIGNED_INT, size: 1 },
            { name: position.name, type: gl.FLOAT_VEC3, size: 1 },
          ][i],
      },
    );
    const adapter = createGlsl(
      Fn(() => builtinPosition().assign(vec4(position.x.add(ids.x.toFloat()), flags.toFloat(), 0, 1)))() as any,
      Fn(() => vec4(0, 1, 0, 1))() as any,
    );
    // Before attach, the adapter checks a slot against the program it was given, and holds only what it takes.
    expect(() => adapter.setAttribute(ids.name, Float32Array.of(1, 2))).toThrowRmsl(
      UsageError,
      "host-array-of-another-kind",
    );
    adapter.setAttribute(flags.name, Uint32Array.of(1));
    adapter.attach({ getContext: () => gl } as any);
    expect(() => adapter.setAttribute(ids, Float32Array.of(1, 2) as any)).toThrowRmsl(
      UsageError,
      "host-array-of-another-kind",
    );
    expect(() => adapter.setAttribute(ids.name, Float32Array.of(1, 2))).toThrowRmsl(
      UsageError,
      "host-array-of-another-kind",
    );
    expect(() => adapter.setAttribute(flags.name, Int32Array.of(1))).toThrowRmsl(
      UsageError,
      "host-array-of-another-kind",
    );
    expect(() => adapter.setAttribute(position.name, Float64Array.of(1, 2, 3))).toThrowRmsl(
      UsageError,
      "host-array-of-another-kind",
    );
    expect(() => adapter.setAttribute(ids, Int32Array.of(1, 2))).not.toThrow();
  });

  /**
   * A stub device stands in for the GPU.
   *
   * @canon spec-an-attribute-takes-only-an-array-of-its-own-kind
   */
  it("refuses an array of another kind on createWgsl, named by node or by slot", async () => {
    vi.stubGlobal("navigator", { gpu: { getPreferredCanvasFormat: () => "bgra8unorm" } });
    vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1, COPY_DST: 2, VERTEX: 4, INDEX: 8 });
    try {
      const ids = attribute("ivec2");
      const flags = attribute("uint");
      const position = attribute("vec3");
      const { device, canvas } = stubDevice();
      const vertex = Fn(() => builtinPosition().assign(vec4(position.x.add(ids.x.toFloat()), flags.toFloat(), 0, 1)))();
      const adapter = createWgsl({ context: { device } as any, vertex, fragment: Fn(() => vec4(0, 1, 0, 1))() });
      // Before attach, the adapter checks a slot against the program it was given, and holds only what it takes.
      expect(() => adapter.setAttribute(ids.name, Float32Array.of(1, 2))).toThrowRmsl(
        UsageError,
        "host-array-of-another-kind",
      );
      adapter.setAttribute(flags.name, Uint32Array.of(1));
      await adapter.attach(canvas);
      expect(() => adapter.setAttribute(ids, Float32Array.of(1, 2) as any)).toThrowRmsl(
        UsageError,
        "host-array-of-another-kind",
      );
      expect(() => adapter.setAttribute(flags.name, Int32Array.of(1))).toThrowRmsl(
        UsageError,
        "host-array-of-another-kind",
      );
      expect(() => adapter.setAttribute(position.name, Float64Array.of(1, 2, 3))).toThrowRmsl(
        UsageError,
        "host-array-of-another-kind",
      );
      expect(() => adapter.setAttribute(ids, Int32Array.of(1, 2))).not.toThrow();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe("the array a GPU adapter takes for a storage buffer", () => {
  /**
   * The stub context reports one active uniform, the buffer's sampler.
   *
   * @canon spec-a-storage-buffer-takes-only-an-array-of-its-element-kind
   */
  it("refuses an array of another kind on createGlsl, named by node or by slot", () => {
    const counts = instancedArray(4, "int");
    const position = uniform("vec4");
    const { gl } = stubGl(
      { width: 1, height: 1 },
      {
        getProgramParameter: (_program: unknown, name: number) =>
          name === gl.ACTIVE_UNIFORMS ? 1 : name === gl.ACTIVE_ATTRIBUTES ? 0 : true,
        getActiveUniform: () => ({ name: counts.name, type: gl.INT_SAMPLER_2D, size: 1 }),
      },
    );
    const adapter = createGlsl(
      Fn(() => builtinPosition().assign(position))() as any,
      Fn(() => vec4(counts.element(int(0)).toFloat(), 0, 0, 1))() as any,
    );
    adapter.attach({ getContext: () => gl } as any);
    expect(() => adapter.setAttribute(counts, new Float32Array(4) as any)).toThrowRmsl(
      UsageError,
      "host-array-of-another-kind",
    );
    expect(() => adapter.setAttribute(counts.name, new Uint32Array(4))).toThrowRmsl(
      UsageError,
      "host-array-of-another-kind",
    );
    expect(() => adapter.setAttribute(counts, new Int32Array(4))).not.toThrow();
  });

  /**
   * A stub device stands in for the GPU.
   *
   * @canon spec-a-storage-buffer-takes-only-an-array-of-its-element-kind
   */
  it("refuses an array of another kind on createWgsl, named by node or by slot", async () => {
    vi.stubGlobal("navigator", { gpu: { getPreferredCanvasFormat: () => "bgra8unorm" } });
    vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1, COPY_DST: 2, VERTEX: 4, INDEX: 8, STORAGE: 16 });
    try {
      const flags = instancedArray(2, "uvec2");
      const position = attribute("vec3");
      const { device, canvas } = stubDevice();
      const createRenderPipeline = device.createRenderPipeline;
      device.createRenderPipeline = (descriptor: any) => ({
        ...createRenderPipeline(descriptor),
        getBindGroupLayout: () => ({}),
      });
      const vertex = Fn(() => builtinPosition().assign(vec4(position, 1)))();
      const fragment = Fn(() => vec4(flags.element(int(0)).x.toFloat(), 0, 0, 1))();
      const adapter = createWgsl({
        context: { device, buffer: () => device.createBuffer({ size: 16 }) } as any,
        vertex,
        fragment,
      });
      expect(() => adapter.setAttribute(flags.name, new Int32Array(4))).toThrowRmsl(
        UsageError,
        "host-array-of-another-kind",
      );
      await adapter.attach(canvas);
      expect(() => adapter.setAttribute(flags, new Float32Array(4) as any)).toThrowRmsl(
        UsageError,
        "host-array-of-another-kind",
      );
      expect(() => adapter.setAttribute(flags.name, new Int32Array(4))).toThrowRmsl(
        UsageError,
        "host-array-of-another-kind",
      );
      expect(() => adapter.setAttribute(flags, new Uint32Array(4))).not.toThrow();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

/** A program that reads a uniform of each shape, and the values of another shape each refuses. */
const uniformShapes = () => {
  const count = uniform("int");
  const offsets = uniformArray("int", 2);
  const tint = uniform("vec3");
  const turn = uniform("mat2");
  const pairs = uniformArray("vec2", 2);
  const value = count
    .add(offsets.element(int(1)))
    .toFloat()
    .add(tint.x)
    .add(turn.element(int(1)).y)
    .add(pairs.element(int(1)).y);
  const fragment = Fn(() => vec4(value, 0, 0, 1))();
  const refuses = (setUniform: (uniform: any, value: unknown) => void) => {
    expect(() => setUniform(count, [2])).toThrowRmsl(UsageError, "host-value-of-another-shape");
    expect(() => setUniform(tint, [1, 2])).toThrowRmsl(UsageError, "host-value-of-another-shape");
    expect(() => setUniform(tint.name, 1)).toThrowRmsl(UsageError, "host-value-of-another-shape");
    expect(() =>
      setUniform(turn.name, [
        [1, 0],
        [0, 1],
      ]),
    ).toThrowRmsl(UsageError, "host-value-of-another-shape");
    expect(() => setUniform(offsets.name, [1])).toThrowRmsl(UsageError, "host-value-of-another-shape");
    expect(() => setUniform(pairs, [[1, 2], [3]])).toThrowRmsl(UsageError, "host-value-of-another-shape");
    for (const nothing of [undefined, null])
      expect(() => setUniform(pairs, [[1, 2], nothing])).toThrowRmsl(UsageError, "host-value-of-another-shape");
    expect(() => setUniform(offsets, [1, undefined])).not.toThrow();
    expect(() => setUniform(offsets.name, [null, 2])).not.toThrow();
    expect(() => setUniform(count, 2.7)).not.toThrow();
    expect(() => setUniform(tint, Float32Array.of(1, 2, 3))).not.toThrow();
    expect(() => setUniform(turn.name, [1, 0, 0, 1])).not.toThrow();
    expect(() => setUniform(pairs.name, [Float32Array.of(1, 2), [3, 4]])).not.toThrow();
  };
  return { fragment, refuses, tint };
};

describe("the shape of a value a GPU adapter takes for a uniform", () => {
  /**
   * @canon spec-a-gpu-adapter-refuses-a-uniform-value-of-another-shape
   * @canon spec-an-undefined-uniform-array-element-converts-as-a-typed-slot-does
   */
  it("refuses one through setUniform on createGlsl, named by node or by slot", () => {
    const { fragment, refuses } = uniformShapes();
    const adapter = createGlsl(Fn(() => builtinPosition().assign(vec4(0, 0, 0, 1)))() as any, fragment as any);
    refuses((uniform, value) => adapter.setUniform(uniform, value as any));
  });

  /**
   * The adapter knows the type of a slot from the program it is given, so a
   * value set by slot before `attach` is refused at the call. The stub device
   * stands in for the GPU once it attaches.
   *
   * @canon spec-a-gpu-adapter-refuses-a-uniform-value-of-another-shape
   * @canon spec-an-undefined-uniform-array-element-converts-as-a-typed-slot-does
   */
  it("refuses one through setUniform on createWgsl, named by node or by slot", async () => {
    vi.stubGlobal("navigator", { gpu: { getPreferredCanvasFormat: () => "bgra8unorm" } });
    vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1, COPY_DST: 2, VERTEX: 4, INDEX: 8 });
    try {
      const { fragment, refuses } = uniformShapes();
      const vertex = () => Fn(() => builtinPosition().assign(vec4(0, 0, 0, 1)))();
      refuses((uniform, value) => createWgsl({ vertex: vertex(), fragment }).setUniform(uniform, value as any));
      const { device, canvas } = stubDevice();
      const createRenderPipeline = device.createRenderPipeline;
      (device as any).createRenderPipeline = (d: any) => ({
        ...createRenderPipeline(d),
        getBindGroupLayout: () => ({}),
      });
      const adapter = createWgsl({ context: { device } as any, vertex: vertex(), fragment });
      await adapter.attach(canvas);
      refuses((uniform, value) => adapter.setUniform(uniform, value as any));
    } finally {
      vi.unstubAllGlobals();
    }
  });

  /**
   * A stub device holds the bytes of the uniform buffer the adapter writes.
   *
   * @canon spec-a-component-the-host-leaves-out-reads-zero
   */
  it("writes zeros for a uniform set to null or undefined on createWgsl", async () => {
    vi.stubGlobal("navigator", { gpu: { getPreferredCanvasFormat: () => "bgra8unorm" } });
    vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1, COPY_DST: 2, VERTEX: 4, INDEX: 8 });
    try {
      const k = uniform("float");
      const v = uniform("vec3");
      const pairs = uniformArray("vec2", 2);
      const fragment = Fn(() => vec4(k, v.y, pairs.element(int(1)).y, 1))();
      const { device, canvas, bufferWrites, bytesOf } = stubDevice();
      const createRenderPipeline = device.createRenderPipeline;
      (device as any).createRenderPipeline = (d: any) => ({
        ...createRenderPipeline(d),
        getBindGroupLayout: () => ({}),
      });
      const vertex = Fn(() => builtinPosition().assign(vec4(0, 0, 0, 1)))();
      const adapter = createWgsl({ context: { device } as any, vertex, fragment });
      await adapter.attach(canvas);
      for (const left of [null, undefined]) {
        adapter.setUniform(k, 1);
        adapter.setUniform(v, [0, 2, 0]);
        adapter.setUniform(pairs, [
          [0, 0],
          [0, 4],
        ]);
        const buffer = bufferWrites.at(-1)!.buffer;
        expect(new Float32Array(bytesOf(buffer).buffer).filter((x) => x !== 0).sort()).toEqual(
          Float32Array.of(1, 2, 4),
        );
        for (const u of [k, v, pairs]) adapter.setUniform(u as any, left as any);
        expect(
          new Float32Array(bytesOf(buffer).buffer).every((x) => x === 0),
          String(left),
        ).toBe(true);
      }
    } finally {
      vi.unstubAllGlobals();
    }
  });

  /**
   * A context takes a uniform by its node alone.
   *
   * @canon spec-a-gpu-adapter-refuses-a-uniform-value-of-another-shape
   */
  it.each([
    ["createWgslContext", () => createWgslContext({ device: {} as GPUDevice })],
    ["createWasmContext", async () => createWasmContext()],
  ] as const)("refuses one through setUniform on %s", async (_, create) => {
    const context = await create();
    const count = uniform("int");
    const tint = uniform("vec3");
    const items = uniformArray("float", 2);
    expect(() => context.setUniform(count, [3] as any)).toThrowRmsl(UsageError, "host-value-of-another-shape");
    expect(() => context.setUniform(tint, [1, 2] as any)).toThrowRmsl(UsageError, "host-value-of-another-shape");
    expect(() => context.setUniform(items, [1] as any)).toThrowRmsl(UsageError, "host-value-of-another-shape");
    expect(() => context.setUniform(count, 3)).not.toThrow();
  });
});

describe("the vertex layout of createWgsl", () => {
  /**
   * A stub device records the pipeline. Each attribute takes a buffer of its
   * own, in the order the program made them, and a matrix a location for each
   * column, numbered as the compiled WGSL numbers them.
   *
   * @canon spec-a-vertex-attribute-reaches-the-shader-as-its-declared-type
   * @canon spec-a-gpu-adapter-points-each-column-of-a-matrix-attribute-at-its-column
   */
  it("lays out integer attributes in integer formats and a matrix column by column", async () => {
    vi.stubGlobal("navigator", { gpu: { getPreferredCanvasFormat: () => "bgra8unorm" } });
    vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1, COPY_DST: 2, VERTEX: 4, INDEX: 8 });
    try {
      const ids = attribute("ivec2");
      const flags = attribute("uint");
      const turn = attribute("mat3x2");
      const position = attribute("vec3");
      const { device, canvas, pipelines } = stubDevice();
      const vertex = Fn(() => {
        const shift = turn.mul(vec3(ids.x.toFloat(), flags.toFloat(), 1));
        builtinPosition().assign(vec4(position.xy.add(shift), 0, 1));
      })();
      await createWgsl({ context: { device } as any, vertex, fragment: Fn(() => vec4(0, 1, 0, 1))() }).attach(canvas);
      const { buffers, module } = pipelines[0].vertex;
      expect(buffers).toEqual([
        { arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "sint32x2" }] },
        { arrayStride: 4, attributes: [{ shaderLocation: 1, offset: 0, format: "uint32" }] },
        {
          arrayStride: 24,
          attributes: [0, 1, 2].map((column) => ({
            shaderLocation: 2 + column,
            offset: column * 8,
            format: "float32x2",
          })),
        },
        { arrayStride: 12, attributes: [{ shaderLocation: 5, offset: 0, format: "float32x3" }] },
      ]);
      expect(module.code).toMatch(/@location\(0\) \w+: vec2<i32>/);
      expect(module.code).toMatch(/@location\(4\) \w+_2: vec2<f32>/);
      expect(module.code).toMatch(/@location\(5\) \w+: vec3<f32>/);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe("refilling a GLSL storage buffer", () => {
  /**
   * The stub context reports one active uniform, the buffer's sampler. Each
   * refill holds the same four elements, so the texture keeps its size.
   *
   * @canon spec-a-glsl-storage-refill-of-one-size-allocates-nothing
   */
  it("uploads each refill of one size through one array, reading the size limit once", () => {
    const colours = instancedArray(4, "float");
    const position = uniform("vec4");
    const { gl, calls } = stubGl(
      { width: 1, height: 1 },
      {
        getProgramParameter: (_program: unknown, name: number) =>
          name === gl.ACTIVE_UNIFORMS ? 1 : name === gl.ACTIVE_ATTRIBUTES ? 0 : true,
        getActiveUniform: () => ({ name: colours.name, type: gl.SAMPLER_2D, size: 1 }),
      },
    );
    const adapter = createGlsl(
      Fn(() => builtinPosition().assign(position))() as any,
      Fn(() => vec4(colours.element(int(0)), 0, 0, 1))() as any,
    );
    adapter.attach({ getContext: () => gl } as any);
    adapter.setAttribute(colours.name, Float32Array.of(1, 2, 3, 4));
    adapter.setAttribute(colours.name, Float32Array.of(5, 6, 7, 8));
    adapter.setAttribute(colours.name, Float32Array.of(9, 10, 11, 12));
    const uploads = calls.filter((c) => c.name === "texSubImage2D").map((c) => c.args[8]);
    expect(uploads).toHaveLength(3);
    expect(uploads[1]).toBe(uploads[2]);
    expect(calls.filter((c) => c.name === "getParameter" && c.args[0] === gl.MAX_TEXTURE_SIZE)).toHaveLength(1);
  });

  /**
   * @canon spec-a-glsl-storage-refill-of-one-size-uploads-only-its-texels
   */
  it("sets no texture parameter or unit again for a refill of one size", () => {
    const colours = instancedArray(4, "float");
    const position = uniform("vec4");
    const { gl, calls } = stubGl(
      { width: 1, height: 1 },
      {
        getProgramParameter: (_program: unknown, name: number) =>
          name === gl.ACTIVE_UNIFORMS ? 1 : name === gl.ACTIVE_ATTRIBUTES ? 0 : true,
        getActiveUniform: () => ({ name: colours.name, type: gl.SAMPLER_2D, size: 1 }),
      },
    );
    const adapter = createGlsl(
      Fn(() => builtinPosition().assign(position))() as any,
      Fn(() => vec4(colours.element(int(0)), 0, 0, 1))() as any,
    );
    adapter.attach({ getContext: () => gl } as any);
    const before = calls.length;
    adapter.setAttribute(colours.name, Float32Array.of(1, 2, 3, 4));
    const made = calls.slice(before).map((c) => c.name);
    expect(made).toContain("texSubImage2D");
    expect(made).not.toContain("texParameteri");
    expect(made).not.toContain("useProgram");
    expect(made).not.toContain("uniform1i");
  });

  /**
   * @canon spec-a-glsl-storage-refill-of-one-size-allocates-nothing
   */
  it("lets the kept array go on destroy", () => {
    const colours = instancedArray(4, "float");
    const position = uniform("vec4");
    const { gl, calls } = stubGl(
      { width: 1, height: 1 },
      {
        getProgramParameter: (_program: unknown, name: number) =>
          name === gl.ACTIVE_UNIFORMS ? 1 : name === gl.ACTIVE_ATTRIBUTES ? 0 : true,
        getActiveUniform: () => ({ name: colours.name, type: gl.SAMPLER_2D, size: 1 }),
      },
    );
    const adapter = createGlsl(
      Fn(() => builtinPosition().assign(position))() as any,
      Fn(() => vec4(colours.element(int(0)), 0, 0, 1))() as any,
    );
    adapter.attach({ getContext: () => gl } as any);
    adapter.destroy();
    adapter.attach({ getContext: () => gl } as any);
    const uploads = calls.filter((c) => c.name === "texImage2D").map((c) => c.args[8]);
    expect(uploads).toHaveLength(2);
    expect(uploads[1]).not.toBe(uploads[0]);
  });
});

describe.skipIf(!GPU_ENABLED)("an adapter applies what the host set before attach", () => {
  /**
   * @canon spec-the-glsl-adapter-applies-a-value-set-before-attach
   */
  it("draws with a uniform and an attribute set before attach on GLSL", async () => {
    const pixel = await runInGpuPage(BEFORE_ATTACH, "__rmslUnitsGlsl", new URL(".", import.meta.url).pathname);
    expect(pixel).toEqual(GREEN);
  }, 60_000);

  /**
   * @canon spec-the-wgsl-adapter-applies-a-value-set-before-attach
   */
  it.skipIf(!WEBGPU)(
    "draws with a uniform and an attribute set before attach on WGSL",
    async () => {
      const pixel = await runInWebGpuPage(BEFORE_ATTACH, "__rmslUnitsWgsl", new URL(".", import.meta.url).pathname);
      expect(pixel).toEqual(GREEN);
    },
    60_000,
  );
});

/**
 * A full-screen triangle on a 4×4 canvas that a uniform or an attribute of
 * one type draws green, by entry. Each entry returns what it read, or the
 * error it hit.
 */
const VALUE_TYPES = `
import { Fn, attribute, builtinPosition, float, instancedArray, int, select, uniform, uniformArray, varying, vec2, vec3, vec4 } from "../rmsl";
import { createGlsl } from "../glsl";
import { createWgsl } from "../wgsl";
import { createJs } from "../js";
import { createWasm } from "../wasm";
${READ_PIXEL}
const TRIANGLE = Float32Array.of(-1, -1, 0, 3, -1, 0, -1, 3, 0);
const canvas = () => {
  const c = document.createElement("canvas");
  c.width = 4;
  c.height = 4;
  return c;
};
const attempt = async (run) => {
  try {
    return await run();
  } catch (error) {
    return { error: error.message };
  }
};
const position = attribute("vec3");
const plainVertex = () => Fn(() => { builtinPosition().assign(vec4(position, 1)); })();
/** Draws \`fragment\` with createGlsl, after \`set\` gives the adapter its values. */
const glsl = (fragment, set, vertex = plainVertex(), drawWith) => attempt(() => {
  const target = canvas();
  const adapter = createGlsl(vertex, fragment);
  adapter.attach(target);
  if (drawWith) drawWith(adapter);
  else {
    adapter.setAttribute(position, TRIANGLE);
    set(adapter);
    adapter.draw({ count: 3 });
  }
  return readPixel(target, 1, 2);
});
/** The vertex stage of \`glsl\`, moved along x by an integer attribute it reads. */
const shiftedVertex = (shift) =>
  Fn(() => { builtinPosition().assign(vec4(position.x.add(shift.toFloat()), position.y, 0, 1)); })();
const colours = () => uniformArray("vec4", 2);
/** Draws \`vertex\` green with createWgsl, after \`set\` gives the adapter its attributes; the draw names no count. */
const wgsl = (vertex, set, fragment = Fn(() => vec4(0, 1, 0, 1))()) => attempt(async () => {
  const target = canvas();
  const adapter = createWgsl({ vertex, fragment });
  await adapter.attach(target);
  set(adapter);
  adapter.draw();
  await adapter.device().queue.onSubmittedWorkDone();
  return readPixel(target, 1, 2);
});
globalThis.__rmslValueTypes = {
  uniformArray: () => {
    const array = colours();
    return glsl(Fn(() => array.element(1))(), (a) => a.setUniform(array, [[1, 0, 0, 1], [0, 1, 0, 1]]));
  },
  uint: () => {
    const green = uniform("uint");
    return glsl(Fn(() => vec4(0, green.toFloat(), 0, 1))(), (a) => a.setUniform(green, 1));
  },
  uvec2: () => {
    const green = uniform("uvec2");
    return glsl(Fn(() => vec4(0, green.y.toFloat(), 0, 1))(), (a) => a.setUniform(green, [0, 1]));
  },
  bvec2: () => {
    const green = uniform("bvec2");
    return glsl(Fn(() => vec4(0, green.y.select(float(1), float(0)), 0, 1))(), (a) => a.setUniform(green, [0, 1]));
  },
  mat2x3: () => {
    const columns = uniform("mat2x3");
    return glsl(Fn(() => vec4(columns.mul(vec2(1, 0)), 1))(), (a) => a.setUniform(columns, [0, 1, 0, 1, 0, 0]));
  },
  intAttribute: () => {
    const shift = attribute("int");
    return glsl(Fn(() => vec4(0, 1, 0, 1))(), (a) => a.setAttribute(shift, Int32Array.of(0, 0, 0)), shiftedVertex(shift));
  },
  uintAttribute: () => {
    const shift = attribute("uint");
    return glsl(Fn(() => vec4(0, 1, 0, 1))(), (a) => a.setAttribute(shift, Uint32Array.of(0, 0, 0)), shiftedVertex(shift));
  },
  // Each vertex's matrix has columns (0, 0) and (1, 0), so its second column
  // gives the green. The draw names no count, so it counts by the matrix.
  matrixAttribute: () => {
    const columns = attribute("mat2");
    const green = varying("float");
    const vertex = Fn(() => {
      green.assign(columns.mul(vec2(0, 1)).x);
      builtinPosition().assign(vec4(position, 1));
    })();
    return glsl(Fn(() => vec4(0, green, 0, 1))(), () => {}, vertex, (adapter) => {
      adapter.setAttribute(columns, Float32Array.of(0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0));
      adapter.setAttribute(position, TRIANGLE);
      adapter.draw();
    });
  },
  intAttributeWithoutData: () => {
    const shift = attribute("int");
    return glsl(Fn(() => vec4(0, 1, 0, 1))(), () => {}, shiftedVertex(shift));
  },
  unreadFirst: () => {
    const unused = attribute("vec2");
    return glsl(Fn(() => vec4(0, 1, 0, 1))(), () => {}, plainVertex(), (adapter) => {
      adapter.setAttribute(unused, new Float32Array(12));
      adapter.setAttribute(position, TRIANGLE);
      adapter.draw();
    });
  },
  wgslIntAttribute: () => {
    // Each vertex holds (-5, 5), which sums to no shift only when read as integers.
    const shift = attribute("ivec2");
    return wgsl(shiftedVertex(shift.x.add(shift.y)), (adapter) => {
      adapter.setAttribute(position, TRIANGLE);
      adapter.setAttribute(shift, Int32Array.of(-5, 5, -5, 5, -5, 5));
    });
  },
  wgslUintAttribute: () => {
    const shift = attribute("uint");
    return wgsl(shiftedVertex(shift), (adapter) => {
      adapter.setAttribute(position, TRIANGLE);
      adapter.setAttribute(shift, Uint32Array.of(0, 0, 0));
    });
  },
  // Each vertex's matrix has the columns (0, 0, 0), (1, 0, 0) and (0, 0, 0),
  // so only its second column gives the green. The draw counts by the matrix.
  wgslMatrixAttribute: () => {
    const columns = attribute("mat3");
    const green = varying("float");
    const vertex = Fn(() => {
      green.assign(columns.mul(vec3(0, 1, 0)).x);
      builtinPosition().assign(vec4(position, 1));
    })();
    const matrix = Float32Array.of(0, 0, 0, 1, 0, 0, 0, 0, 0);
    return wgsl(vertex, (adapter) => {
      adapter.setAttribute(columns, Float32Array.of(...matrix, ...matrix, ...matrix));
      adapter.setAttribute(position, TRIANGLE);
    }, Fn(() => vec4(0, green, 0, 1))());
  },
  /**
   * Whether each of \`cases\`, a value and the value of \`type\` it should read as, reads so through
   * createGlsl, given as a uniform, as the second element of a uniform array and as the second
   * component of a vector, each compared on the GPU with the value it should read as.
   */
  hostConversion: (type, cases) => attempt(() => {
    const vector = { int: "ivec2", uint: "uvec2", float: "vec2", bool: "bvec2" }[type];
    const u = uniform(type);
    const items = uniformArray(type, 2);
    const pair = uniform(vector);
    const want = uniform(type);
    const same = (x) => (type === "bool" ? x.equal(want).and(x.toFloat().equal(want.toFloat())) : x.equal(want));
    const all = same(u).and(same(items.element(1))).and(same(pair.y));
    const target = canvas();
    const adapter = createGlsl(plainVertex(), Fn(() => vec4(0, select(all, float(1), float(0)), 0, 1))());
    adapter.attach(target);
    adapter.setAttribute(position, TRIANGLE);
    return cases.map(([value, wanted]) => {
      adapter.setUniform(u, value);
      adapter.setUniform(items, [0, value]);
      adapter.setUniform(pair, [0, value]);
      adapter.setUniform(want, wanted);
      adapter.draw({ count: 3 });
      return readPixel(target, 1, 2).g === 255;
    });
  }),
  undefinedComponent: () => attempt(() => {
    const f = uniform("vec3");
    const m = uniform("mat2");
    const i = uniform("ivec3");
    const u = uniform("uvec3");
    const b = uniform("bvec3");
    const n = uniform("vec3");
    const fs = uniformArray("float", 3);
    const is = uniformArray("int", 3);
    const bs = uniformArray("bool", 3);
    const ns = uniformArray("float", 3);
    const isNaN = (x) => x.greaterThanEqual(0).or(x.lessThan(0)).not();
    const all = isNaN(f.y)
      .and(f.z.equal(3))
      .and(isNaN(m.element(0).y))
      .and(i.y.toFloat().equal(0))
      .and(u.y.toFloat().equal(0))
      .and(b.y.not())
      .and(n.y.equal(0))
      .and(isNaN(fs.element(1)))
      .and(fs.element(2).equal(3))
      .and(is.element(1).toFloat().equal(0))
      .and(bs.element(1).not())
      .and(ns.element(1).equal(0));
    const target = canvas();
    const adapter = createGlsl(plainVertex(), Fn(() => vec4(0, select(all, float(1), float(0)), 0, 1))());
    adapter.attach(target);
    adapter.setAttribute(position, TRIANGLE);
    adapter.setUniform(f, [1, undefined, 3]);
    adapter.setUniform(m, [1, undefined, 3, 4]);
    adapter.setUniform(i, [1, undefined, 3]);
    adapter.setUniform(u, [1, undefined, 3]);
    adapter.setUniform(b, [true, undefined, true]);
    adapter.setUniform(n, [1, null, 3]);
    adapter.setUniform(fs, [1, undefined, 3]);
    adapter.setUniform(is, [1, undefined, 3]);
    adapter.setUniform(bs, [true, undefined, true]);
    adapter.setUniform(ns, [1, null, 3]);
    adapter.draw({ count: 3 });
    return readPixel(target, 1, 2).g === 255;
  }),
  uniformLeftOut: () => attempt(() => {
    const k = uniform("float");
    const v = uniform("vec3");
    const pairs = uniformArray("vec2", 2);
    const zero = k.equal(0).and(v.y.equal(0)).and(pairs.element(1).y.equal(0));
    const target = canvas();
    const adapter = createGlsl(plainVertex(), Fn(() => vec4(0, select(zero, float(1), float(0)), 0, 1))());
    adapter.attach(target);
    adapter.setAttribute(position, TRIANGLE);
    return [null, undefined].flatMap((left) => {
      adapter.setUniform(k, 1);
      adapter.setUniform(v, [0, 2, 0]);
      adapter.setUniform(pairs, [[0, 0], [0, 4]]);
      adapter.draw({ count: 3 });
      const set = readPixel(target, 1, 2).g === 255;
      for (const u of [k, v, pairs]) adapter.setUniform(u, left);
      adapter.draw({ count: 3 });
      return [set, readPixel(target, 1, 2).g === 255];
    });
  }),
  glslUnsetStorage: () => {
    const green = instancedArray(Float32Array.of(0, 1), "float");
    return glsl(Fn(() => vec4(0, green.element(int(1)), 0, 1))(), () => {});
  },
  wgslUnsetStorage: () => {
    const green = instancedArray(Float32Array.of(0, 1), "float");
    const fragment = Fn(() => vec4(0, green.element(int(1)), 0, 1))();
    return wgsl(plainVertex(), (adapter) => adapter.setAttribute(position, TRIANGLE), fragment);
  },
  typedVector: () => {
    const colour = uniform("vec4");
    return glsl(Fn(() => colour)(), (a) => a.setUniform(colour, Float32Array.of(0, 1, 0, 1)));
  },
  cpuAttributeWithoutData: (target) => attempt(() => {
    const shift = attribute("int");
    const adapter = (target === "JS" ? createJs : createWasm)(() => shiftedVertex(shift), () => Fn(() => vec4(0, 1, 0, 1))());
    adapter.attach(canvas());
    adapter.setAttribute(position, TRIANGLE);
    adapter.draw({ count: 3 });
    return { drew: true };
  }),
  wgslAttributeWithoutData: () => {
    const shift = attribute("int");
    return wgsl(shiftedVertex(shift), (adapter) => adapter.setAttribute(position, TRIANGLE));
  },
  wgslUnreadFirst: () => attempt(async () => {
    const unused = attribute("vec2");
    const target = canvas();
    const adapter = createWgsl({ vertex: plainVertex(), fragment: Fn(() => vec4(0, 1, 0, 1))() });
    await adapter.attach(target);
    adapter.setAttribute(unused, new Float32Array(12));
    adapter.setAttribute(position, TRIANGLE);
    adapter.draw();
    await adapter.device().queue.onSubmittedWorkDone();
    return readPixel(target, 1, 2);
  }),
  wgslUniformArray: () => attempt(async () => {
    const array = colours();
    const target = canvas();
    const adapter = createWgsl({ vertex: plainVertex(), fragment: Fn(() => array.element(1))() });
    await adapter.attach(target);
    adapter.setAttribute(position, TRIANGLE);
    adapter.setUniform(array, [[1, 0, 0, 1], [0, 1, 0, 1]]);
    adapter.draw({ count: 3 });
    await adapter.device().queue.onSubmittedWorkDone();
    return readPixel(target, 1, 2);
  }),
};
`;

/** Runs the entry `name` of `VALUE_TYPES` in the WebGL page. */
const glslValueType = (name: string) =>
  runInGpuPage(
    `${VALUE_TYPES}\nglobalThis.__rmslValueTypesRun = () => globalThis.__rmslValueTypes.${name}();`,
    "__rmslValueTypesRun",
    new URL(".", import.meta.url).pathname,
  );

/** Runs the entry `name` of `VALUE_TYPES` in the WebGPU page. */
const wgslValueType = (name: string) =>
  runInWebGpuPage(
    `${VALUE_TYPES}\nglobalThis.__rmslValueTypesRun = () => globalThis.__rmslValueTypes.${name}();`,
    "__rmslValueTypesRun",
    new URL(".", import.meta.url).pathname,
  );

describe.skipIf(!GPU_ENABLED)("an adapter takes a value of every type its program declares", () => {
  /**
   * @canon spec-an-adapter-sets-a-uniform-array-from-one-value-per-element
   */
  it("sets a uniform array with createGlsl", async () => {
    expect(await glslValueType("uniformArray")).toEqual(GREEN);
  }, 120_000);

  /**
   * @canon spec-an-adapter-sets-a-uniform-array-from-one-value-per-element
   */
  it.skipIf(!WEBGPU)(
    "sets a uniform array with createWgsl",
    async () => {
      const run = `${VALUE_TYPES}\nglobalThis.__rmslValueTypesRun = () => globalThis.__rmslValueTypes.wgslUniformArray();`;
      expect(await runInWebGpuPage(run, "__rmslValueTypesRun", new URL(".", import.meta.url).pathname)).toEqual(GREEN);
    },
    120_000,
  );

  /**
   * The program reads only \`position\`, so the 12 values of an attribute it
   * does not read, passed first, count for nothing.
   *
   * @canon spec-a-gpu-adapter-takes-its-count-from-the-first-attribute-it-reads
   */
  it("counts a createGlsl draw from the first attribute the program reads", async () => {
    expect(await glslValueType("unreadFirst")).toEqual(GREEN);
  }, 120_000);

  /**
   * @canon spec-a-gpu-adapter-takes-its-count-from-the-first-attribute-it-reads
   */
  it.skipIf(!WEBGPU)(
    "counts a createWgsl draw from the first attribute the program reads",
    async () => {
      const run = `${VALUE_TYPES}\nglobalThis.__rmslValueTypesRun = () => globalThis.__rmslValueTypes.wgslUnreadFirst();`;
      expect(await runInWebGpuPage(run, "__rmslValueTypesRun", new URL(".", import.meta.url).pathname)).toEqual(GREEN);
    },
    120_000,
  );

  /**
   * @canon spec-an-adapter-sets-a-uniform-of-every-type-its-program-declares
   */
  it.each(["uint", "uvec2", "bvec2", "mat2x3"])(
    "sets a %s uniform with createGlsl",
    async (type) => {
      expect(await glslValueType(type)).toEqual(GREEN);
    },
    120_000,
  );

  /**
   * @canon spec-a-vertex-attribute-reaches-the-shader-as-its-declared-type
   */
  it.each(["intAttribute", "uintAttribute"])(
    "draws with an integer attribute through createGlsl, %s",
    async (entry) => {
      expect(await glslValueType(entry)).toEqual(GREEN);
    },
    120_000,
  );

  /**
   * @canon spec-a-gpu-adapter-points-each-column-of-a-matrix-attribute-at-its-column
   */
  it("draws with a mat2 attribute through createGlsl, counted by the matrix", async () => {
    expect(await glslValueType("matrixAttribute")).toEqual(GREEN);
  }, 120_000);

  /**
   * @canon spec-a-vertex-attribute-reaches-the-shader-as-its-declared-type
   */
  it.skipIf(!WEBGPU).each(["wgslIntAttribute", "wgslUintAttribute"])(
    "draws with an integer attribute through createWgsl, %s",
    async (entry) => {
      expect(await wgslValueType(entry)).toEqual(GREEN);
    },
    120_000,
  );

  /**
   * @canon spec-a-gpu-adapter-points-each-column-of-a-matrix-attribute-at-its-column
   */
  it.skipIf(!WEBGPU)(
    "draws with a mat3 attribute through createWgsl, counted by the matrix",
    async () => {
      expect(await wgslValueType("wgslMatrixAttribute")).toEqual(GREEN);
    },
    120_000,
  );

  /**
   * @canon spec-a-storage-buffer-the-host-never-sets-starts-as-its-array
   */
  it.each([
    ["createGlsl", () => glslValueType("glslUnsetStorage")],
    ["createWgsl", () => wgslValueType("wgslUnsetStorage")],
  ])(
    "reads a storage buffer the host never set as its array in a render stage on %s",
    async (_, drawn) => {
      expect(await drawn()).toEqual(GREEN);
    },
    120_000,
  );

  /**
   * @canon spec-a-gpu-adapter-refuses-a-uniform-value-of-another-shape
   */
  it("sets a vector uniform from a typed array with createGlsl", async () => {
    expect(await glslValueType("typedVector")).toEqual(GREEN);
  }, 120_000);

  /**
   * The page tells NaN by its failing every comparison with zero.
   *
   * @canon spec-an-undefined-component-converts-as-a-typed-slot-does
   * @canon spec-an-undefined-uniform-array-element-converts-as-a-typed-slot-does
   */
  it("reads a component the host gives as undefined as a typed slot of its kind holds it on createGlsl", async () => {
    expect(await glslValueType("undefinedComponent")).toBe(true);
  }, 120_000);

  /**
   * @canon spec-a-component-the-host-leaves-out-reads-zero
   */
  it("reads a uniform set to null or undefined as zero on createGlsl", async () => {
    expect(await glslValueType("uniformLeftOut")).toEqual([false, true, false, true]);
  }, 120_000);

  /** Runs `hostConversion` for `type` in the WebGL page, over cases written as JavaScript. */
  const glslConversion = (type: string, cases: string) =>
    runInGpuPage(
      `${VALUE_TYPES}\nglobalThis.__rmslValueTypesRun = () => globalThis.__rmslValueTypes.hostConversion("${type}", ${cases});`,
      "__rmslValueTypesRun",
      new URL(".", import.meta.url).pathname,
    );

  /**
   * @canon spec-an-integer-host-value-truncates-and-wraps
   */
  it("truncates and wraps an integer the host gives on createGlsl", async () => {
    const cases = `[[2.7, 2], [-2.7, -2], [2 ** 32 + 3, 3], [2 ** 31, -(2 ** 31)], [NaN, 0], [Infinity, 0], ["2", 2], [true, 1]]`;
    expect(await glslConversion("int", cases)).toEqual(Array(8).fill(true));
    expect(await glslConversion("uint", `[[2.7, 2], [-1, 4294967295], [NaN, 0], ["2", 2]]`)).toEqual(
      Array(4).fill(true),
    );
  }, 120_000);

  /**
   * @canon spec-a-float-host-value-converts-to-a-number
   */
  it("converts a float the host gives to a number on createGlsl", async () => {
    expect(await glslConversion("float", `[["2", 2], [true, 1], [false, 0], ["0.5", 0.5]]`)).toEqual(
      Array(4).fill(true),
    );
  }, 120_000);

  /**
   * @canon spec-a-bool-host-value-reads-true-for-a-nonzero-integer
   */
  it("reads a bool the host gives as true where it converts to an integer not zero on createGlsl", async () => {
    const cases = `[[true, true], [1, true], [2, true], [-1, true], ["1", true], [false, false], [0, false], [0.5, false], ["x", false], [2 ** 32, false]]`;
    expect(await glslConversion("bool", cases)).toEqual(Array(10).fill(true));
  }, 120_000);

  /**
   * The page gives the adapter a real 2D canvas.
   *
   * @canon spec-a-draw-missing-an-attribute-its-vertex-stage-reads-is-refused
   */
  it.each(["JS", "WASM"])(
    "refuses a draw missing an attribute its vertex stage reads on the %s adapter",
    async (target) => {
      const run = `${VALUE_TYPES}\nglobalThis.__rmslValueTypesRun = () => globalThis.__rmslValueTypes.cpuAttributeWithoutData("${target}");`;
      expect(await runInGpuPage(run, "__rmslValueTypesRun", new URL(".", import.meta.url).pathname)).toEqual({
        error: expect.stringMatching(
          new RegExp(`^${target === "JS" ? "createJs" : "createWasm"}: draw\\(\\) is missing attribute "_rmsl_a\\d+"`),
        ),
      });
    },
    120_000,
  );

  /**
   * @canon spec-a-draw-missing-an-attribute-its-vertex-stage-reads-is-refused
   */
  it("refuses a draw missing an attribute its vertex stage reads on createGlsl", async () => {
    expect(await glslValueType("intAttributeWithoutData")).toEqual({
      error: expect.stringMatching(/createGlsl: draw\(\) is missing attribute "_rmsl_a\d+"/),
    });
  }, 120_000);

  /**
   * @canon spec-a-draw-missing-an-attribute-its-vertex-stage-reads-is-refused
   */
  it.skipIf(!WEBGPU)(
    "refuses a draw missing an attribute its vertex stage reads on createWgsl",
    async () => {
      expect(await wgslValueType("wgslAttributeWithoutData")).toEqual({
        error: expect.stringMatching(/createWgsl: draw\(\) is missing attribute "_rmsl_a\d+"/),
      });
    },
    120_000,
  );
});

describe.skipIf(!GPU_ENABLED)("WGSL compute on a Dawn device", () => {
  let uninstall: (() => void) | undefined;
  beforeAll(async () => {
    uninstall = await installWebGpuGlobals();
  });
  afterAll(() => uninstall?.());

  /**
   * @canon spec-the-wgsl-compute-adapter-applies-a-value-set-before-attach
   */
  it("computes with a uniform and a storage slot set before attach", async () => {
    const values = instancedArray(2, "float");
    const offset = uniform("float");
    const program = Fn(() => {
      const i = invocationIndex();
      values.element(i).assign(values.element(i).add(offset));
    })();
    const adapter = createWgslCompute(program);
    adapter.setUniform(offset, 10);
    adapter.setAttribute(values.name, Float32Array.of(1, 2));
    await adapter.attach();
    const out = { [values.name]: new Float32Array(2) };
    await adapter.compute(out);
    expect(Array.from(out[values.name]!)).toEqual([11, 12]);
    adapter.destroy();
  });

  /**
   * Each value is given as a uniform, as the second element of a uniform array and as the second
   * component of a vector, and read back from a storage buffer; a bool as 1 where it is true and
   * 10 more where its value as a number is 1.
   */
  async function wgslHostValues(type: "int" | "uint" | "float" | "bool", values: unknown[]): Promise<number[][]> {
    const vector = { int: "ivec2", uint: "uvec2", float: "vec2", bool: "bvec2" } as const;
    const u = uniform(type);
    const items = uniformArray(type, 2);
    const pair = uniform(vector[type]);
    const results = instancedArray(3, type === "bool" ? "float" : type);
    const asNumber = (b: any) => (type === "bool" ? select(b, float(1), float(0)).add(b.toFloat().mul(10)) : b);
    const adapter = createWgslCompute(
      Fn(() => {
        results.element(int(0)).assign(asNumber(u));
        results.element(int(1)).assign(asNumber(items.element(int(1))));
        results.element(int(2)).assign(asNumber(pair.y));
      })(),
    );
    await adapter.attach();
    const read: number[][] = [];
    for (const value of values) {
      adapter.setUniform(u, value as any);
      adapter.setUniform(items, [0, value] as any);
      adapter.setUniform(pair, [0, value] as any);
      const out = {
        [results.name]: new (type === "int" ? Int32Array : type === "uint" ? Uint32Array : Float32Array)(3),
      };
      await adapter.compute(out, 1);
      read.push(Array.from(out[results.name]!));
    }
    adapter.destroy();
    return read;
  }

  /**
   * @canon spec-an-integer-host-value-truncates-and-wraps
   */
  it("truncates and wraps an integer the host gives on createWgslCompute", async () => {
    for (const type of ["int", "uint"] as const) {
      const values = [2.7, -2.7, -1, 2 ** 32 + 3, 2 ** 31, NaN, Infinity, "2", "x", true];
      const wanted = values.map((v) => Array(3).fill(type === "int" ? (v as number) | 0 : (v as number) >>> 0));
      expect(await wgslHostValues(type, values), type).toEqual(wanted);
    }
  });

  /**
   * @canon spec-a-float-host-value-converts-to-a-number
   */
  it("converts a float the host gives to a number on createWgslCompute", async () => {
    expect(await wgslHostValues("float", ["2", true, false, "x", 0.1])).toEqual(
      [2, 1, 0, NaN, Math.fround(0.1)].map((n) => Array(3).fill(n)),
    );
  });

  /**
   * @canon spec-a-bool-host-value-reads-true-for-a-nonzero-integer
   */
  it("reads a bool the host gives as true where it converts to an integer not zero on createWgslCompute", async () => {
    const values = [true, 1, 2, -1, 2 ** 32 + 1, "1", false, 0, 0.5, "x", NaN, 2 ** 32];
    expect(await wgslHostValues("bool", values)).toEqual(values.map((_, i) => Array(3).fill(i < 6 ? 11 : 0)));
  });

  /**
   * @canon spec-an-undefined-component-converts-as-a-typed-slot-does
   * @canon spec-an-undefined-uniform-array-element-converts-as-a-typed-slot-does
   */
  it("reads a component the host gives as undefined as a typed slot of its kind holds it on createWgslCompute", async () => {
    const f = uniform("vec3");
    const m = uniform("mat2");
    const i = uniform("ivec3");
    const u = uniform("uvec3");
    const b = uniform("bvec3");
    const n = uniform("vec3");
    const fs = uniformArray("float", 3);
    const is = uniformArray("int", 3);
    const bs = uniformArray("bool", 3);
    const ns = uniformArray("float", 3);
    const results = instancedArray(10, "float");
    const adapter = createWgslCompute(
      Fn(() => {
        results.element(int(0)).assign(f.y);
        results.element(int(1)).assign(m.element(int(0)).y);
        results.element(int(2)).assign(i.y.toFloat());
        results.element(int(3)).assign(u.y.toFloat());
        results.element(int(4)).assign(select(b.y, float(1), float(0)));
        results.element(int(5)).assign(n.y);
        results.element(int(6)).assign(fs.element(int(1)));
        results.element(int(7)).assign(is.element(int(1)).toFloat());
        results.element(int(8)).assign(select(bs.element(int(1)), float(1), float(0)));
        results.element(int(9)).assign(ns.element(int(1)));
      })(),
    );
    await adapter.attach();
    adapter.setUniform(f, [1, undefined, 3] as any);
    adapter.setUniform(m, [1, undefined, 3, 4] as any);
    adapter.setUniform(i, [1, undefined, 3] as any);
    adapter.setUniform(u, [1, undefined, 3] as any);
    adapter.setUniform(b, [true, undefined, true] as any);
    adapter.setUniform(n, [1, null, 3] as any);
    adapter.setUniform(fs, [1, undefined, 3] as any);
    adapter.setUniform(is, [1, undefined, 3] as any);
    adapter.setUniform(bs, [true, undefined, true] as any);
    adapter.setUniform(ns, [1, null, 3] as any);
    const out = { [results.name]: new Float32Array(10) };
    await adapter.compute(out, 1);
    expect(Array.from(out[results.name]!)).toEqual([NaN, NaN, 0, 0, 0, 0, NaN, 0, 0, 0]);
    adapter.destroy();
  });

  /**
   * @canon spec-a-component-the-host-leaves-out-reads-zero
   */
  it("reads a uniform set to null or undefined as zero on createWgslCompute", async () => {
    const k = uniform("float");
    const v = uniform("vec3");
    const pairs = uniformArray("vec2", 2);
    const results = instancedArray(3, "float");
    const adapter = createWgslCompute(
      Fn(() => {
        results.element(int(0)).assign(k);
        results.element(int(1)).assign(v.y);
        results.element(int(2)).assign(pairs.element(int(1)).y);
      })(),
    );
    await adapter.attach();
    for (const left of [null, undefined]) {
      adapter.setUniform(k, 1);
      adapter.setUniform(v, [0, 2, 0]);
      adapter.setUniform(pairs, [
        [0, 0],
        [0, 4],
      ]);
      const out = { [results.name]: new Float32Array(3) };
      await adapter.compute(out, 1);
      expect(Array.from(out[results.name]!)).toEqual([1, 2, 4]);
      for (const u of [k, v, pairs]) adapter.setUniform(u as any, left as any);
      await adapter.compute(out, 1);
      expect(Array.from(out[results.name]!), String(left)).toEqual([0, 0, 0]);
    }
    adapter.destroy();
  });

  /**
   * The adapter knows the type of a slot from the program it is given, so a
   * value set by slot before `attach` is refused at the call.
   *
   * @canon spec-a-gpu-adapter-refuses-a-uniform-value-of-another-shape
   */
  it("refuses a uniform value of another shape through setUniform on createWgslCompute, by node or by slot", async () => {
    const { fragment, refuses } = uniformShapes();
    const out = instancedArray(1, "vec4");
    const program = () => Fn(() => out.element(invocationIndex()).assign(fragment))();
    refuses((uniform, value) => createWgslCompute(program()).setUniform(uniform, value as any));
    const adapter = createWgslCompute(program());
    await adapter.attach();
    refuses((uniform, value) => adapter.setUniform(uniform, value as any));
    adapter.destroy();
  });

  /**
   * @canon exception-an-inline-fn-both-branches-of-an-if-read-runs-once-before-the-if
   * @canon exception-an-inline-fn-read-in-a-branch-and-after-it-runs-once-before-the-if
   * @canon spec-an-inline-fn-several-statements-of-one-block-read-runs-once-before-the-first
   * @canon spec-an-inline-fn-read-in-one-statement-runs-before-it
   */
  it("runs an inline Fn read on several paths once on every path that reads it, on WGSL", async () => {
    const { counter, result, first, second, paths } = inlineFnPaths();
    for (const path of paths) {
      const adapter = createWgslCompute(path.build());
      await adapter.attach();
      for (const [f, s] of CONDITION_PAIRS) {
        adapter.setUniform(first, f ? 1 : 0);
        adapter.setUniform(second, s ? 1 : 0);
        adapter.setAttribute(counter.name, new Float32Array(1));
        adapter.setAttribute(result.name, new Float32Array(1));
        const out = { [counter.name]: new Float32Array(1), [result.name]: new Float32Array(1) };
        await adapter.compute(out, 1);
        expect({ runs: out[counter.name]![0], value: out[result.name]![0] }, `${path.name}, ${f}, ${s}`).toEqual({
          runs: path.runs(f, s),
          value: path.value(f, s),
        });
      }
      adapter.destroy();
    }
  });

  /**
   * The condition is tested four times, and the `Fn` counts its runs. A
   * driver that reads storage from before the loop counts one run in every
   * case, though the WGSL writes the increment inside the loop before the
   * test. The probe decides which count to expect.
   *
   * @canon exception-dawn-on-metal-counts-a-storage-write-in-a-loop-condition-once
   */
  it("runs an inline Fn read only in a While or Loop condition at every test on WGSL, or once where the driver reads storage from before the loop", async () => {
    const { counter, result, conditions } = inlineFnConditions();
    for (const condition of conditions) {
      const adapter = createWgslCompute(condition.build());
      await adapter.attach();
      const out = { [counter.name]: new Float32Array(1), [result.name]: new Float32Array(1) };
      await adapter.compute(out, 1);
      adapter.destroy();
      expect({ runs: out[counter.name]![0], value: out[result.name]![0] }, condition.name).toEqual({
        runs: STALE_STORAGE ? 1 : 4,
        value: 3,
      });
    }
  });

  /**
   * Each of the four iterations adds one to two elements of one buffer. A
   * driver that reads storage from before the loop counts one iteration in
   * the first element, though the WGSL writes both inside the loop.
   *
   * @canon exception-dawn-on-metal-counts-the-first-of-two-storage-writes-in-a-while-body-once
   */
  it("runs both storage writes in a While body on every iteration on WGSL, or the first once where the driver reads storage from before the loop", async () => {
    const counts = instancedArray(2, "float");
    const program = Fn(() => {
      const n = float(0).toVar();
      While(n.lessThan(4), () => {
        counts.element(int(0)).addAssign(1);
        counts.element(int(1)).addAssign(1);
        n.addAssign(1);
      });
    })();
    const adapter = createWgslCompute(program);
    await adapter.attach();
    const out = { [counts.name]: new Float32Array(2) };
    await adapter.compute(out, 1);
    adapter.destroy();
    expect(Array.from(out[counts.name]!)).toEqual([STALE_STORAGE ? 1 : 4, 4]);
  });

  /**
   * The `Fn` adds one to its count, reads the count back, and returns on its
   * third run, so the loop of nine stops there. A driver that reads storage
   * from before the loop reads a count of one on every run, so the `Fn` never
   * returns and the loop runs all nine.
   *
   * @canon exception-dawn-on-metal-misses-a-return-an-inline-fn-makes-on-its-count-in-a-loop
   */
  it("ends the invocation where an inline Fn read in a loop returns on its count on WGSL, or never where the driver reads storage from before the loop", async () => {
    const counts = instancedArray(3, "float");
    const f = Fn(() => {
      counts.element(int(1)).addAssign(1);
      const r = counts.element(int(1)).toVar();
      counts.element(int(0)).addAssign(1);
      If(r.equal(3), () => Return());
      return r;
    })() as any;
    const program = Fn(() => {
      Loop(9, () => {
        counts.element(int(2)).addAssign(f);
      });
    })();
    const adapter = createWgslCompute(program);
    await adapter.attach();
    const out = { [counts.name]: new Float32Array(3) };
    await adapter.compute(out, 1);
    adapter.destroy();
    expect(Array.from(out[counts.name]!)).toEqual(STALE_STORAGE ? [9, 1, 9] : [3, 3, 3]);
  });

  /**
   * The `Fn` is called outside the program's body, and the loop body reads
   * its value once.
   *
   * @canon spec-an-inline-fn-read-once-in-a-loop-runs-on-every-iteration
   */
  it("runs an inline Fn read once in a loop body on every iteration, on WGSL", async () => {
    const counter = instancedArray(1, "float");
    const result = instancedArray(1, "float");
    const shared = Fn(() => {
      counter.element(int(0)).addAssign(1);
      return float(2);
    })() as any;
    const program = Fn(() => {
      Loop(4, () => {
        result.element(int(0)).addAssign(shared);
      });
    })();
    const adapter = createWgslCompute(program);
    await adapter.attach();
    const out = { [counter.name]: new Float32Array(1), [result.name]: new Float32Array(1) };
    await adapter.compute(out, 1);
    expect({ runs: out[counter.name]![0], value: out[result.name]![0] }).toEqual({ runs: 4, value: 8 });
    adapter.destroy();
  });

  /**
   * An inline `Fn` reads another twice in each iteration of its loop, and the
   * program reads the outer one once.
   *
   * @canon spec-an-inline-fn-read-only-inside-another-runs-in-the-block-there
   */
  it("runs an inline Fn read only inside another's loop on every iteration of it, on WGSL", async () => {
    const counter = instancedArray(1, "float");
    const result = instancedArray(1, "float");
    const f = Fn(() => {
      counter.element(int(0)).addAssign(1);
      return float(2);
    })() as any;
    const g = Fn(() => {
      const t = float(0).toVar();
      Loop(3, () => {
        t.addAssign(f);
        t.addAssign(f.mul(10));
      });
      return t;
    })() as any;
    const adapter = createWgslCompute(
      Fn(() => {
        result.element(int(0)).assign(g);
      })(),
    );
    await adapter.attach();
    const out = { [counter.name]: new Float32Array(1), [result.name]: new Float32Array(1) };
    await adapter.compute(out, 1);
    expect({ runs: out[counter.name]![0], value: out[result.name]![0] }).toEqual({ runs: 3, value: 66 });
    adapter.destroy();
  });

  /**
   * The inner `Fn` is read twice in a branch of the outer one, and in loops at
   * two levels.
   *
   * @canon spec-an-inline-fn-read-only-inside-another-runs-in-the-block-there
   */
  it("runs an inline Fn read only inside another in the block there that holds every read, on WGSL", async () => {
    const counter = instancedArray(1, "float");
    const result = instancedArray(1, "float");
    const taken = uniform("float");
    const counted = () =>
      Fn(() => {
        counter.element(int(0)).addAssign(1);
        return float(2);
      })() as any;
    const looping = (f: any) =>
      Fn(() => {
        const t = float(0).toVar();
        Loop(3, () => {
          t.addAssign(f);
          t.addAssign(f.mul(10));
        });
        return t;
      })() as any;
    const writing = (value: any) =>
      Fn(() => {
        result.element(int(0)).assign(value);
      })();
    const programs: [string, () => any, number, { runs: number; value: number }][] = [
      [
        "in a branch",
        () => {
          const f = counted();
          return writing(
            Fn(() => {
              const t = float(0).toVar();
              If(taken.greaterThan(0), () => {
                t.assign(f);
                t.addAssign(f.mul(10));
              });
              return t;
            })(),
          );
        },
        0,
        { runs: 0, value: 0 },
      ],
      [
        "in loops at two levels",
        () => {
          const g = looping(counted());
          return writing(
            Fn(() => {
              const t = float(0).toVar();
              Loop(2, () => {
                t.addAssign(g);
                t.addAssign(g.mul(0));
              });
              return t;
            })(),
          );
        },
        0,
        { runs: 6, value: 132 },
      ],
    ];
    for (const [name, build, branch, expected] of programs) {
      const adapter = createWgslCompute(build());
      await adapter.attach();
      adapter.setUniform(taken, branch);
      const out = { [counter.name]: new Float32Array(1), [result.name]: new Float32Array(1) };
      await adapter.compute(out, 1);
      expect({ runs: out[counter.name]![0], value: out[result.name]![0] }, name).toEqual(expected);
      adapter.destroy();
    }
  });

  /**
   * The condition is the value of the `Fn`, and the body reads it twice.
   *
   * @canon spec-an-inline-fn-a-for-condition-and-body-read-runs-once-before-the-loop
   */
  it("runs an inline Fn once before a For whose condition is its value and whose body reads it, on WGSL", async () => {
    const counter = instancedArray(1, "float");
    const steps = instancedArray(1, "float");
    const f = Fn(() => {
      counter.element(int(0)).addAssign(1);
      return steps.element(int(0)).lessThan(3);
    })() as any;
    const program = Fn(() => {
      For(
        () => int(0).toVar(),
        () => f,
        (i) => i.addAssign(1),
        () => {
          steps.element(int(0)).addAssign(1);
          steps.element(int(0)).addAssign(select(f, float(0), float(0)));
          steps.element(int(0)).addAssign(select(f, float(0), float(0)));
        },
      );
    })();
    const adapter = createWgslCompute(program);
    await adapter.attach();
    const out = { [counter.name]: new Float32Array(1), [steps.name]: new Float32Array(1) };
    await adapter.compute(out, 1);
    expect({ runs: out[counter.name]![0], steps: out[steps.name]![0] }).toEqual({ runs: 1, steps: 3 });
    adapter.destroy();
  });

  /**
   * The `Fn` counts its runs and returns two values, and one statement reads
   * both.
   *
   * @canon spec-an-inline-fn-that-returns-several-values-runs-once-before-every-read-of-them
   */
  it("runs an inline Fn that returns two values once when one statement reads both, on WGSL", async () => {
    const counter = instancedArray(1, "float");
    const result = instancedArray(1, "float");
    const [a, b] = Fn(() => {
      counter.element(int(0)).addAssign(1);
      return [float(2), float(3)];
    })() as any;
    const program = Fn(() => {
      result.element(int(0)).assign(a.add(b));
    })();
    const adapter = createWgslCompute(program);
    await adapter.attach();
    const out = { [counter.name]: new Float32Array(1), [result.name]: new Float32Array(1) };
    await adapter.compute(out, 1);
    expect({ runs: out[counter.name]![0], value: out[result.name]![0] }).toEqual({ runs: 1, value: 5 });
    adapter.destroy();
  });

  /**
   * The first root reads the value in a branch and in a loop, the second
   * once. The `Fn` runs once, before the first root, whichever branch runs.
   *
   * @canon spec-an-inline-fn-several-roots-read-runs-once-before-the-first
   */
  it("runs an inline Fn several roots read once, before the first root, on WGSL", async () => {
    const counter = instancedArray(1, "float");
    const result = instancedArray(3, "float");
    const taken = uniform("float");
    const f = Fn(() => {
      counter.element(int(0)).addAssign(1);
      return counter.element(int(0)).toVar();
    })() as any;
    const roots = [
      Fn(() => {
        If(taken.greaterThan(0), () => {
          result.element(int(0)).addAssign(f);
        });
        Loop(3, () => {
          result.element(int(1)).addAssign(f);
        });
      })(),
      Fn(() => {
        result.element(int(2)).addAssign(f);
      })(),
    ];
    for (const branch of [0, 1]) {
      const adapter = createWgslCompute(roots, { workgroupSize: 1 });
      await adapter.attach();
      adapter.setUniform(taken, branch);
      const out = { [counter.name]: new Float32Array(1), [result.name]: new Float32Array(3) };
      await adapter.compute(out, 1);
      expect({ branch, runs: out[counter.name]![0], values: Array.from(out[result.name]!) }).toEqual({
        branch,
        runs: 1,
        values: [branch, 3, 1],
      });
      adapter.destroy();
    }
  });
  /**
   * The outer `Fn` reads the inner one in both branches of an `If`, and two
   * roots read the outer one.
   *
   * @canon spec-an-inline-fn-read-only-inside-another-runs-in-the-block-there
   */
  it("runs an inline Fn read only inside another read from two roots once, on WGSL", async () => {
    const counter = instancedArray(1, "float");
    const result = instancedArray(2, "float");
    const taken = uniform("float");
    const g = Fn(() => {
      counter.element(int(0)).addAssign(1);
      return float(2);
    })() as any;
    const f = Fn(() => {
      const t = float(0).toVar();
      If(taken.greaterThan(0), () => {
        t.assign(g);
      }).Else(() => {
        t.assign(g.mul(10));
      });
      return t;
    })() as any;
    for (const [branch, value] of [
      [1, 2],
      [0, 20],
    ]) {
      const adapter = createWgslCompute([
        Fn(() => {
          result.element(int(0)).assign(f);
        })(),
        Fn(() => {
          result.element(int(1)).assign(f.add(1));
        })(),
      ]);
      await adapter.attach();
      adapter.setUniform(taken, branch);
      const out = { [counter.name]: new Float32Array(1), [result.name]: new Float32Array(2) };
      await adapter.compute(out, 1);
      expect({ runs: out[counter.name]![0], values: Array.from(out[result.name]!) }).toEqual({
        runs: 1,
        values: [value, value + 1],
      });
      adapter.destroy();
    }
  });

  /**
   * @canon spec-an-inline-fn-runs-where-it-is-called
   */
  it("runs an Fn called before a loop once, on WGSL", async () => {
    const counter = instancedArray(1, "float");
    const result = instancedArray(1, "float");
    const program = Fn(() => {
      const shared = Fn(() => {
        counter.element(int(0)).addAssign(1);
        return float(2);
      })() as any;
      Loop(4, () => {
        result.element(int(0)).addAssign(shared);
      });
    })();
    const adapter = createWgslCompute(program);
    await adapter.attach();
    const out = { [counter.name]: new Float32Array(1), [result.name]: new Float32Array(1) };
    await adapter.compute(out, 1);
    expect({ runs: out[counter.name]![0], value: out[result.name]![0] }).toEqual({ runs: 1, value: 8 });
    adapter.destroy();
  });

  /**
   * The refusal comes before the dispatch, so the buffer keeps the values set before it.
   *
   * @canon spec-a-readback-array-takes-only-an-array-of-its-element-kind
   */
  it("refuses an out array of another kind on createWgslCompute", async () => {
    const counts = instancedArray(2, "int");
    const adapter = createWgslCompute(
      Fn(() => {
        const i = invocationIndex();
        counts.element(i).assign(counts.element(i).add(1));
      })(),
    );
    await adapter.attach();
    adapter.setAttribute(counts, Int32Array.of(1, 2));
    await expect(adapter.compute({ [counts.name]: new Float32Array(2) })).rejects.toThrowRmsl(
      UsageError,
      "host-array-of-another-kind",
    );
    const out = { [counts.name]: new Int32Array(2) };
    await adapter.compute(out);
    expect(Array.from(out[counts.name]!)).toEqual([2, 3]);
    adapter.destroy();
  });

  /**
   * The refusal comes before the dispatch, so the buffer keeps the values set before it.
   *
   * @canon spec-a-readback-array-holds-at-least-the-values-of-its-slot
   */
  it("refuses an out array shorter than its slot on createWgslCompute", async () => {
    const counts = instancedArray(3, "int");
    const adapter = createWgslCompute(
      Fn(() => {
        const i = invocationIndex();
        counts.element(i).assign(counts.element(i).add(1));
      })(),
    );
    await adapter.attach();
    adapter.setAttribute(counts, Int32Array.of(1, 2, 3));
    await expect(adapter.compute({ [counts.name]: new Int32Array(2) })).rejects.toThrowRmsl(
      UsageError,
      "readback-array-too-short",
      `out["${counts.name}"] holds 2 values, and the slot has 3`,
    );
    const out = { [counts.name]: new Int32Array(4) };
    await adapter.compute(out);
    expect(Array.from(out[counts.name]!)).toEqual([2, 3, 4, 0]);
    adapter.destroy();
  });

  /**
   * The refusal comes before the dispatch, so the buffer keeps the values set before it.
   *
   * @canon spec-compute-refuses-a-slot-with-no-storage
   */
  it("refuses a slot with no storage before it dispatches on createWgslCompute", async () => {
    const counts = instancedArray(2, "int");
    const adapter = createWgslCompute(
      Fn(() => {
        const i = invocationIndex();
        counts.element(i).assign(counts.element(i).add(1));
      })(),
    );
    await adapter.attach();
    adapter.setAttribute(counts, Int32Array.of(1, 2));
    await expect(adapter.compute({ other: new Int32Array(2) })).rejects.toThrowRmsl(UsageError, "unknown-storage-slot");
    const out = { [counts.name]: new Int32Array(2) };
    await adapter.compute(out);
    expect(Array.from(out[counts.name]!)).toEqual([2, 3]);
    adapter.destroy();
  });

  /**
   * The context gets a device whose pipeline creation is counted. Three
   * dispatches of one compute node create one pipeline.
   *
   * @canon spec-a-wgsl-context-compiles-a-compute-node-once
   */
  it("creates one pipeline for a compute node dispatched three times", async () => {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter!.requestDevice();
    let pipelines = 0;
    const create = device.createComputePipeline.bind(device);
    (device as any).createComputePipeline = (descriptor: GPUComputePipelineDescriptor) => {
      pipelines++;
      return create(descriptor);
    };
    const context = await createWgslContext({ device });
    const values = instancedArray(2, "float");
    const step = Fn(() => {
      const i = invocationIndex();
      values.element(i).assign(values.element(i).add(1));
    })().compute(2);
    context.compute(step);
    context.compute(step);
    context.compute(step);
    const bytes = await context.getArrayBufferAsync(values.attribute);
    expect(Array.from(new Float32Array(bytes, 0, 2))).toEqual([3, 3]);
    expect(pipelines).toBe(1);
    context.destroy();
    device.destroy();
  });

  /**
   * A `vec3` buffer is laid out as WGSL lays out a storage array, 16 bytes an
   * element, and comes back as one flat typed array without the padding.
   *
   * @canon spec-a-wgsl-storage-buffer-holds-a-vec3-in-16-bytes
   */
  it("computes over a vec3 storage buffer with createWgslCompute", async () => {
    const points = instancedArray(2, "vec3");
    const program = Fn(() => {
      const i = invocationIndex();
      points.element(i).assign(points.element(i).add(1));
    })();
    const adapter = createWgslCompute(program);
    await adapter.attach();
    adapter.setAttribute(points.name, Float32Array.of(1, 2, 3, 4, 5, 6));
    const out = { [points.name]: new Float32Array(6) };
    await adapter.compute(out);
    adapter.destroy();
    expect(Array.from(out[points.name]!)).toEqual([2, 3, 4, 5, 6, 7]);
  }, 60_000);

  /**
   * The program reads one buffer made from data and one made from a count,
   * writes a third, and reads back all three, with no `setAttribute`.
   *
   * @canon spec-a-storage-buffer-the-host-never-sets-starts-as-its-array
   */
  it("starts each storage buffer the host never set as its array on createWgslCompute", async () => {
    const given = instancedArray(Float32Array.of(1, 2), "float");
    const zeroed = instancedArray(2, "float");
    const out = instancedArray(2, "float");
    const adapter = createWgslCompute(
      Fn(() => {
        const i = invocationIndex();
        out.element(i).assign(given.element(i).add(zeroed.element(i)).add(7));
        zeroed.element(i).assign(float(3));
      })(),
    );
    await adapter.attach();
    const read = {
      [given.name]: new Float32Array(2),
      [zeroed.name]: new Float32Array(2),
      [out.name]: new Float32Array(2),
    };
    await adapter.compute(read, 2);
    expect(Array.from(read[out.name]!)).toEqual([8, 9]);
    expect(Array.from(read[given.name]!)).toEqual([1, 2]);
    expect(Array.from(read[zeroed.name]!)).toEqual([3, 3]);
    expect(Array.from(zeroed.attribute.array)).toEqual([0, 0]);
    adapter.destroy();
  });

  /**
   * No buffer is set, so the dispatch covers the first the program declares.
   *
   * @canon spec-a-compute-call-takes-its-count-from-the-first-storage-buffer
   */
  it("given no count covers the program's first buffer before the host sets one on createWgslCompute", async () => {
    const out = instancedArray(3, "float");
    const adapter = createWgslCompute(
      Fn(() => {
        out.element(invocationIndex()).assign(float(5));
      })(),
    );
    await adapter.attach();
    const read = { [out.name]: new Float32Array(3) };
    await adapter.compute(read);
    expect(Array.from(read[out.name]!)).toEqual([5, 5, 5]);
    adapter.destroy();
  });

  /**
   * Setting one slot to a length of its own keeps what the other slots hold.
   *
   * @canon spec-setting-one-storage-slot-keeps-the-others
   */
  it("keeps what one storage slot holds when another is set to a different length", async () => {
    const source = instancedArray(1, "float");
    const target = instancedArray(2, "float");
    const program = Fn(() => {
      target.element(invocationIndex()).assign(source.element(0));
    })();
    const adapter = createWgslCompute(program);
    await adapter.attach();
    adapter.setAttribute(source.name, Float32Array.of(5));
    adapter.setAttribute(target.name, new Float32Array(2));
    const out = { [source.name]: new Float32Array(1), [target.name]: new Float32Array(2) };
    await adapter.compute(out, 1);
    adapter.destroy();
    expect(Array.from(out[source.name]!)).toEqual([5]);
    expect(Array.from(out[target.name]!)).toEqual([5, 0]);
  }, 60_000);

  /**
   * A slot the host never sets, such as one the program only writes, is as
   * long as the first slot set, and comes back with what the program wrote.
   *
   * @canon spec-a-compute-adapter-takes-a-storage-buffer-as-one-flat-typed-array
   */
  it("reads back a slot the host never set, as long as the first slot set", async () => {
    const source = instancedArray(3, "float");
    const result = instancedArray(3, "float");
    const program = Fn(() => {
      const i = invocationIndex();
      result.element(i).assign(source.element(i).mul(2));
    })();
    const adapter = createWgslCompute(program);
    await adapter.attach();
    adapter.setAttribute(source.name, Float32Array.of(1, 2, 3));
    const out = { [result.name]: new Float32Array(3) };
    await adapter.compute(out);
    adapter.destroy();
    expect(Array.from(out[result.name]!)).toEqual([2, 4, 6]);
  }, 60_000);

  /**
   * A length that is not a multiple of the component count still fits its buffer, and an output
   * array shorter than the slot is refused where it used to be copied from silently.
   *
   * @canon spec-a-compute-adapter-takes-a-storage-buffer-as-one-flat-typed-array
   */
  it("takes a ragged vec3 array, and refuses an output array that is too short", async () => {
    const points = instancedArray(2, "vec3");
    const program = Fn(() => {
      const i = invocationIndex();
      points.element(i).assign(points.element(i).add(1));
    })();
    const adapter = createWgslCompute(program);
    await adapter.attach();
    adapter.setAttribute(points.name, Float32Array.of(1, 2, 3, 4));
    await expect(adapter.compute({ [points.name]: new Float32Array(2) })).rejects.toThrowRmsl(
      UsageError,
      "readback-array-too-short",
    );
    const out = { [points.name]: new Float32Array(6) };
    // The dispatch covers the whole elements of the array, here one, so the half element is not computed.
    await adapter.compute(out);
    adapter.destroy();
    expect(Array.from(out[points.name]!.subarray(0, 3))).toEqual([2, 3, 4]);
  }, 60_000);

  /**
   * A compute program with no storage buffer still dispatches.
   *
   * @canon spec-a-compute-adapter-takes-a-storage-buffer-as-one-flat-typed-array
   */
  it("dispatches a program with no storage", async () => {
    const offset = uniform("float");
    const adapter = createWgslCompute(Fn(() => offset.add(1))());
    await adapter.attach();
    adapter.setUniform(offset, 1);
    await expect(adapter.compute(undefined, 1)).resolves.toBeUndefined();
    adapter.destroy();
  }, 60_000);

  /**
   * A slot the host never sets is as long as the program declared it, whichever slot is set first.
   *
   * @canon spec-a-compute-adapter-takes-a-storage-buffer-as-one-flat-typed-array
   */
  it("sizes a slot the host never sets from its declared count, whichever slot is set first", async () => {
    const result = instancedArray(3, "float");
    const counter = instancedArray(1, "float");
    const program = Fn(() => {
      const i = invocationIndex();
      result.element(i).assign(counter.element(0).add(i.toFloat()));
    })();
    const adapter = createWgslCompute(program);
    await adapter.attach();
    adapter.setAttribute(counter.name, Float32Array.of(10));
    const out = { [result.name]: new Float32Array(3) };
    await adapter.compute(out, 3);
    adapter.destroy();
    expect(Array.from(out[result.name]!)).toEqual([10, 11, 12]);
  }, 60_000);

  /**
   * @canon spec-a-storage-buffer-takes-only-an-array-of-its-element-kind
   */
  it("refuses a storage array of another kind on createWgslCompute, named by node or by slot", async () => {
    const counts = instancedArray(2, "int");
    const adapter = createWgslCompute(
      Fn(() => {
        counts.element(invocationIndex()).assign(counts.element(invocationIndex()).add(1));
      })(),
    );
    expect(() => adapter.setAttribute(counts.name, new Uint32Array(2))).toThrowRmsl(
      UsageError,
      "host-array-of-another-kind",
    );
    await adapter.attach();
    expect(() => adapter.setAttribute(counts, new Float32Array(2) as any)).toThrowRmsl(
      UsageError,
      "host-array-of-another-kind",
    );
    expect(() => adapter.setAttribute(counts.name, new Uint32Array(2))).toThrowRmsl(
      UsageError,
      "host-array-of-another-kind",
    );
    expect(() => adapter.setAttribute(counts, new Int32Array(2))).not.toThrow();
    adapter.destroy();
  }, 60_000);
});

describe("what WGSL writes for a name or a loop header", () => {
  /**
   * A uniform whose raw name is empty is refused.
   *
   * @canon spec-a-raw-name-declares-one-input-under-that-name
   */
  it("refuses a uniform with an empty name on WGSL", () => {
    expect(() => compileWgslStage.fragment(Fn(() => vec4(uniformRaw("", "float"), 0, 0, 1))())).toThrowRmsl(
      NodeError,
      "empty-name",
    );
  });

  /**
   * A varying whose raw name is empty is refused.
   *
   * @canon spec-a-raw-name-declares-one-input-under-that-name
   */
  it("refuses a varying with an empty name on WGSL", () => {
    expect(() => compileWgslStage.fragment(Fn(() => vec4(varyingRaw("", "float"), 0, 0, 1))())).toThrowRmsl(
      NodeError,
      "empty-name",
    );
  });

  /**
   * A `For` whose init makes no statement, such as one handed a variable
   * made before the loop, compiles on WGSL to a header with an empty init.
   *
   * @canon spec-a-for-loops-over-a-variable-its-init-is-given
   */
  it("compiles a For whose init makes no statement to a valid WGSL header", () => {
    const code = compileWgslStage.fragment(
      Fn(() => {
        const i = int(0).toVar();
        const total = float(0).toVar();
        For(
          () => i,
          (v) => v.lessThan(3),
          (v) => {
            v.assign(v.add(1));
          },
          () => {
            total.assign(total.add(1));
          },
        );
        return vec4(total, 0, 0, 1);
      })(),
    );
    expect(code).toMatch(/for \(;/);
  });
});

describe.skipIf(!GPU_ENABLED)("WGSL on a WebGPU device", () => {
  /**
   * WGSL widens a scalar branch of `select` to the vector type of the other
   * branch, so an integer select of a vector and a scalar is accepted by the
   * driver.
   *
   * @canon spec-select-picks-one-of-two-values
   */
  it("selects between an ivec3 and an int on WGSL", async () => {
    const build = () => vec3(select(bvec3(true, false, true), ivec3(4, 5, 6), int(1)));
    expect(await evaluateWGSL(build as any)).toEqual([4, 1, 6]);
  }, 60_000);
});

/** Every `@group(g) @binding(b)` pair a WGSL source declares, in order. */
const bindingsOf = (code: string) =>
  [...code.matchAll(/@group\((\d+)\) @binding\((\d+)\)/g)].map((m) => `${m[1]}:${m[2]}`);

/** The `@group(g) @binding(b)` pair WGSL declares for the variable `name`. */
const bindingOf = (code: string, name: string) =>
  new RegExp(`@group\\((\\d+)\\) @binding\\((\\d+)\\) var ${name}:`).exec(code)?.slice(1).join(":");

/**
 * A full-screen triangle on a 4×4 canvas drawn green, as a `createGlsl`
 * entry builds it. Each entry returns what it read, or the error it hit.
 */
const GLSL_DRAWS = `
import { Discard, Fn, If, attribute, builtinPosition, float, uniform, vec4 } from "../rmsl";
import { createGlsl } from "../glsl";
${READ_PIXEL}
const TRIANGLE = Float32Array.of(-1, -1, 0, 3, -1, 0, -1, 3, 0);
const canvas = () => {
  const c = document.createElement("canvas");
  c.width = 4;
  c.height = 4;
  return c;
};
const attempt = (run) => {
  try {
    return run();
  } catch (error) {
    return { error: error.message };
  }
};
const position = attribute("vec3");
const plainVertex = () => Fn(() => { builtinPosition().assign(vec4(position, 1)); })();
const side = uniform("float");
const inlineDiscard = () => {
  const shared = Fn(() => { Discard(); return float(1); })();
  return Fn(() => {
    const colour = vec4(0, 0, 0, 1).toVar();
    If(side.greaterThan(0), () => { colour.assign(vec4(shared, 0, 0, 1)); }).Else(() => { colour.assign(vec4(0, shared, 0, 1)); });
    return colour;
  })();
};
globalThis.__rmslGlslDraws = {
  inlineDiscard: () => attempt(() => {
    const target = canvas();
    const adapter = createGlsl(plainVertex(), inlineDiscard());
    adapter.attach(target);
    adapter.setAttribute(position, TRIANGLE);
    adapter.draw();
    return readPixel(target, 1, 2);
  }),
  firstAttributeCount: () => attempt(() => {
    const target = canvas();
    const offset = attribute("vec2");
    const vertex = Fn(() => { builtinPosition().assign(vec4(position.xy.add(offset), 0, 1)); })();
    const adapter = createGlsl(vertex, Fn(() => vec4(0, 1, 0, 1))());
    adapter.attach(target);
    adapter.setAttribute(position, TRIANGLE);
    adapter.setAttribute(offset, new Float32Array(12));
    adapter.draw();
    return readPixel(target, 1, 2);
  }),
  firstVertex: () => attempt(() => {
    const target = canvas();
    const adapter = createGlsl(plainVertex(), Fn(() => vec4(0, 1, 0, 1))());
    adapter.attach(target);
    adapter.setAttribute(position, Float32Array.of(0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 3, -1, 0, -1, 3, 0));
    adapter.draw({ first: 3 });
    return readPixel(target, 1, 2);
  }),
};
`;

/** The `createWgsl` counterparts, on a 4×4 canvas; each returns what it read, or the error it hit. */
const WGSL_DRAWS = `
import { Discard, Fn, If, attribute, builtinPosition, float, instancedArray, ivec2, uniform, vec2, vec4, texture } from "../rmsl";
import { createWgsl } from "../wgsl";
${READ_PIXEL}
const TRIANGLE = Float32Array.of(-1, -1, 0, 3, -1, 0, -1, 3, 0);
const canvas = () => {
  const c = document.createElement("canvas");
  c.width = 4;
  c.height = 4;
  return c;
};
const attempt = async (run) => {
  try {
    return await run();
  } catch (error) {
    return { error: error.message };
  }
};
const position = attribute("vec3");
const plainVertex = () => Fn(() => { builtinPosition().assign(vec4(position, 1)); })();
const side = uniform("float");
const inlineDiscard = () => {
  const shared = Fn(() => { Discard(); return float(1); })();
  return Fn(() => {
    const colour = vec4(0, 0, 0, 1).toVar();
    If(side.greaterThan(0), () => { colour.assign(vec4(shared, 0, 0, 1)); }).Else(() => { colour.assign(vec4(0, shared, 0, 1)); });
    return colour;
  })();
};
const drawn = async (adapter, target) => {
  await adapter.device().queue.onSubmittedWorkDone();
  return readPixel(target, 1, 2);
};
globalThis.__rmslWgslDraws = {
  inlineDiscard: () => attempt(async () => {
    const target = canvas();
    const adapter = createWgsl({ vertex: plainVertex(), fragment: inlineDiscard() });
    await adapter.attach(target);
    adapter.setAttribute(position, TRIANGLE);
    adapter.draw();
    return drawn(adapter, target);
  }),
  texture: () => attempt(async () => {
    const target = canvas();
    const image = uniform("sampler2D");
    const adapter = createWgsl({
      vertex: plainVertex(),
      fragment: Fn(() => texture(image, vec2(0.75, 0.25)))(),
    });
    await adapter.attach(target);
    adapter.setAttribute(position, TRIANGLE);
    // Row 0 is red then green, row 1 blue then white, so (0.75, 0.25) lands on the green texel.
    adapter.setTexture(image, {
      data: Uint8Array.of(255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255),
      width: 2,
      height: 2,
    });
    adapter.draw({ count: 3 });
    return drawn(adapter, target);
  }),
  integerTexture: () => attempt(async () => {
    const target = canvas();
    const image = uniform("usampler2D");
    const adapter = createWgsl({
      vertex: plainVertex(),
      fragment: Fn(() => vec4(0, texture(image, ivec2(0, 0)).y.toFloat().div(255), 0, 1))(),
    });
    adapter.setTexture(image, { data: Uint8Array.of(0, 255, 0, 255), width: 1, height: 1 });
    await adapter.attach(target);
    adapter.setAttribute(position, TRIANGLE);
    adapter.draw({ count: 3 });
    return drawn(adapter, target);
  }),
  vec3Storage: () => attempt(async () => {
    const target = canvas();
    const colours = instancedArray(2, "vec3");
    const adapter = createWgsl({ vertex: plainVertex(), fragment: Fn(() => vec4(colours.element(1), 1))() });
    await adapter.attach(target);
    adapter.setAttribute(position, TRIANGLE);
    adapter.setAttribute(colours.name, Float32Array.of(1, 0, 0, 0, 1, 0));
    adapter.draw({ count: 3 });
    return drawn(adapter, target);
  }),
  firstAttributeCount: () => attempt(async () => {
    const target = canvas();
    const extra = attribute("float");
    const vertex = Fn(() => { builtinPosition().assign(vec4(position.x.add(extra.mul(0)), position.y, 0, 1)); })();
    const adapter = createWgsl({ vertex, fragment: Fn(() => vec4(0, 1, 0, 1))() });
    await adapter.attach(target);
    adapter.setAttribute(position, TRIANGLE);
    adapter.setAttribute(extra, new Float32Array(6));
    adapter.draw();
    return drawn(adapter, target);
  }),
  firstVertex: () => attempt(async () => {
    const target = canvas();
    const adapter = createWgsl({ vertex: plainVertex(), fragment: Fn(() => vec4(0, 1, 0, 1))() });
    await adapter.attach(target);
    adapter.setAttribute(position, Float32Array.of(0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 3, -1, 0, -1, 3, 0));
    adapter.draw({ first: 3 });
    return drawn(adapter, target);
  }),
};
`;

/** Runs one entry of `GLSL_DRAWS` in the WebGL page. */
const glslEntry = (name: string) =>
  runInGpuPage(
    `${GLSL_DRAWS}\nglobalThis.__rmslGlslDrawRun = async () => globalThis.__rmslGlslDraws.${name}();`,
    "__rmslGlslDrawRun",
    new URL(".", import.meta.url).pathname,
  );

/** Runs one entry of `WGSL_DRAWS` in the WebGPU page. */
const wgslEntry = (name: string) =>
  runInWebGpuPage(
    `${WGSL_DRAWS}\nglobalThis.__rmslWgslDrawRun = async () => globalThis.__rmslWgslDraws.${name}();`,
    "__rmslWgslDrawRun",
    new URL(".", import.meta.url).pathname,
  );

describe.skipIf(!GPU_ENABLED)("createGlsl in a browser", () => {
  /**
   * An inline `Fn` that discards is read in both branches of an `If`, and the
   * unset uniform takes the `Else`, so the pixel stays cleared.
   *
   * @canon spec-a-discard-in-an-inline-fn-drops-the-fragment-where-it-runs
   */
  it("drops the fragment from an inline Fn read in both branches of an If, with createGlsl", async () => {
    expect(await glslEntry("inlineDiscard")).toEqual({ r: 0, g: 0, b: 0, a: 0 });
  }, 120_000);

  /**
   * A `createGlsl` draw that names a first vertex and no count draws the vertices after it.
   *
   * @canon spec-a-gpu-adapter-takes-its-count-from-the-first-attribute-it-reads
   */
  it("counts a createGlsl draw from the first attribute, less its first vertex", async () => {
    expect(await glslEntry("firstVertex")).toEqual(GREEN);
  }, 120_000);

  /**
   * A `createGlsl` draw that names no count takes the first attribute's, and
   * so does not draw vertices past its end.
   *
   * @canon spec-a-gpu-adapter-takes-its-count-from-the-first-attribute-it-reads
   */
  it("takes the count of a createGlsl draw from the first attribute", async () => {
    expect(await glslEntry("firstAttributeCount")).toEqual(GREEN);
  }, 120_000);
});

describe.skipIf(!WEBGPU)("createWgsl in a browser", () => {
  /**
   * An inline `Fn` that discards is read in both branches of an `If`, and the
   * unset uniform takes the `Else`, so the pixel stays cleared.
   *
   * @canon spec-a-discard-in-an-inline-fn-drops-the-fragment-where-it-runs
   */
  it("drops the fragment from an inline Fn read in both branches of an If, with createWgsl", async () => {
    expect(await wgslEntry("inlineDiscard")).toEqual({ r: 0, g: 0, b: 0, a: 0 });
  }, 120_000);

  /**
   * `createWgsl` samples the texture set with `setTexture`.
   *
   * @canon spec-an-adapter-takes-the-texture-a-sampler-reads-from-the-host
   */
  it("samples the texture set with setTexture with createWgsl", async () => {
    expect(await wgslEntry("texture")).toEqual(GREEN);
  }, 120_000);

  /**
   * `createWgsl` reads an integer texture set before `attach`.
   *
   * @canon spec-an-adapter-takes-the-texture-a-sampler-reads-from-the-host
   */
  it("reads an integer texture set before attach with createWgsl", async () => {
    expect(await wgslEntry("integerTexture")).toEqual(GREEN);
  }, 120_000);

  /**
   * A `createWgsl` draw that names a first vertex and no count draws the vertices after it.
   *
   * @canon spec-a-gpu-adapter-takes-its-count-from-the-first-attribute-it-reads
   */
  it("counts a createWgsl draw from the first attribute, less its first vertex", async () => {
    expect(await wgslEntry("firstVertex")).toEqual(GREEN);
  }, 120_000);

  /**
   * A `createWgsl` draw that names no count takes the first attribute's, and
   * so does not read past a shorter buffer.
   *
   * @canon spec-a-gpu-adapter-takes-its-count-from-the-first-attribute-it-reads
   */
  it("takes the count of a createWgsl draw from the first attribute", async () => {
    expect(await wgslEntry("firstAttributeCount")).toEqual(GREEN);
  }, 120_000);

  /**
   * @canon spec-a-wgsl-storage-buffer-holds-a-vec3-in-16-bytes
   */
  it("draws from a vec3 storage buffer of its own with createWgsl", async () => {
    expect(await wgslEntry("vec3Storage")).toEqual(GREEN);
  }, 120_000);
});

describe("an attribute the fragment stage reads on a GPU target", () => {
  const pos = attribute("vec3");
  const tint = attribute("vec3");
  const vertex = () => Fn(() => builtinPosition().assign(vec4(pos, 1)))();
  const fragment = () => Fn(() => vec4(tint, 1))();
  const twice = () => Fn(() => vec4(tint.x.add(tint.x), tint.y, 0, 1))();
  /** The vertex and fragment source `createGlsl` hands the context for these stages. */
  const glslSources = (fragmentStage: () => any) => {
    const { gl, calls } = stubGl(
      { width: 1, height: 1 },
      {
        getProgramParameter: (_program: unknown, name: number) =>
          name === gl.ACTIVE_UNIFORMS || name === gl.ACTIVE_ATTRIBUTES ? 0 : true,
      },
    );
    createGlsl(vertex() as any, fragmentStage()).attach({ getContext: () => gl } as any);
    return calls.filter((c) => c.name === "shaderSource").map((c) => c.args[1] as string);
  };
  /** The vertex and fragment source `createWgsl` hands the device for these stages. */
  const wgslSources = async (fragmentStage: () => any) => {
    vi.stubGlobal("navigator", { gpu: { getPreferredCanvasFormat: () => "bgra8unorm" } });
    vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1, COPY_DST: 2, VERTEX: 4, INDEX: 8 });
    try {
      const { device, canvas, modules } = stubDevice();
      await createWgsl({
        context: { device } as any,
        vertex: vertex() as any,
        fragment: fragmentStage(),
      }).attach(canvas);
      return modules.map((m) => m.code);
    } finally {
      vi.unstubAllGlobals();
    }
  };

  /**
   * @canon spec-a-fragment-stage-linked-to-a-vertex-stage-reads-an-attribute-through-a-varying
   */
  it("passes the attribute to the fragment stage in a varying with createGlsl", () => {
    const [vertexSource, fragmentSource] = glslSources(fragment);
    expect(vertexSource).toMatch(new RegExp(`in vec3 ${tint.name};`));
    expect(vertexSource).toMatch(/out vec3 _rmsl_v\d+;/);
    expect(vertexSource).toMatch(new RegExp(`_rmsl_v\\d+ = ${tint.name};`));
    expect(fragmentSource).toMatch(/in vec3 _rmsl_v\d+;/);
    expect(fragmentSource).not.toContain(tint.name);
  });

  /**
   * @canon spec-a-fragment-stage-linked-to-a-vertex-stage-reads-an-attribute-through-a-varying
   */
  it("passes the attribute to the fragment stage in a varying with createWgsl", async () => {
    const [vertexSource, fragmentSource] = await wgslSources(fragment);
    expect(vertexSource).toContain(tint.name);
    expect(vertexSource).toMatch(/_rmsl_v\d+\s*:\s*vec3<f32>/);
    expect(fragmentSource).toMatch(/_rmsl_v\d+\s*:\s*vec3<f32>/);
    expect(fragmentSource).not.toContain(tint.name);
  });

  /**
   * @canon spec-a-fragment-stage-reads-the-attribute-of-a-stage-it-is-linked-to-once
   */
  it("declares one varying for two reads of the attribute with createGlsl and createWgsl", async () => {
    const [vertexGlsl] = glslSources(twice);
    expect(vertexGlsl.match(/out vec3 _rmsl_v\d+;/g)).toHaveLength(1);
    const [vertexWgsl] = await wgslSources(twice);
    expect(vertexWgsl.match(/_rmsl_v\d+\s*:\s*vec3<f32>/g)).toHaveLength(1);
  });

  /**
   * @canon exception-a-matrix-attribute-the-fragment-stage-reads-is-a-matrix-varying-wgsl-rejects
   */
  it("passes a matrix attribute on in a matrix varying with createGlsl and createWgsl", async () => {
    const matrix = attribute("mat2");
    const matrixFragment = () => Fn(() => vec4(matrix.mul(vec2(1, 1)), 0, 1))();
    const [, fragmentGlsl] = glslSources(matrixFragment);
    expect(fragmentGlsl).toMatch(/in mat2 _rmsl_v\d+;/);
    const [, fragmentWgsl] = await wgslSources(matrixFragment);
    expect(fragmentWgsl).toMatch(/_rmsl_v\d+\s*:\s*mat2x2<f32>/);
  });

  /**
   * @canon exception-wgsl-refuses-a-layout-fn-that-reaches-a-value-of-its-stage
   */
  it("refuses a layout Fn that reads the attribute with createWgsl", async () => {
    const readsTint = Fn(() => tint.y).setLayout({ name: "readsTint", type: "float", inputs: [] });
    await expect(wgslSources(() => Fn(() => vec4(0, readsTint(), 0, 1))())).rejects.toThrowRmsl(
      TargetError,
      "layout-fn-reaches-stage-value",
      /uses an attribute/,
    );
  });

  /**
   * @canon spec-a-fragment-stage-compiled-alone-refuses-an-attribute
   */
  it("refuses the attribute in a fragment stage compiled alone on GLSL and WGSL", () => {
    const named = new RegExp(`fragment stage reads the attribute "${tint.name}"`);
    expect(() => compileGlsl.fragment(fragment() as any)).toThrowRmsl(ProgramError, "fragment-reads-attribute", named);
    expect(() => compileWgsl.fragment(fragment() as any)).toThrowRmsl(ProgramError, "fragment-reads-attribute", named);
  });
});

describe.skipIf(!GPU_ENABLED)("an attribute the fragment stage reads, drawn on a GPU", () => {
  /** The corner by the red vertex reads mostly red, and the pixel midway between the green and blue vertices half of each. */
  const expectInterpolated = ([nearRed, midway]: { r: number; g: number; b: number }[]) => {
    expect(nearRed!.r).toBeGreaterThan(nearRed!.g);
    expect(nearRed!.r).toBeGreaterThan(nearRed!.b);
    expect(midway!.r).toBeLessThan(40);
    expect(midway!.g).toBeGreaterThan(80);
    expect(midway!.b).toBeGreaterThan(80);
  };
  /** Draws a triangle whose fragment stage reads the `tint` attribute, which the vertices set to red, green and blue. */
  const entry = (lang: "glsl" | "wgsl") => `
import { Fn, attribute, builtinPosition, vec4 } from "../rmsl";
import { ${lang === "glsl" ? "createGlsl" : "createWgsl"} } from "../${lang}";
${READ_PIXEL}
globalThis.__rmslFragmentAttributeRun = async () => {
  const pos = attribute("vec3");
  const tint = attribute("vec3");
  const vertex = Fn(() => builtinPosition().assign(vec4(pos, 1)))();
  const fragment = Fn(() => vec4(tint, 1))();
  const target = document.createElement("canvas");
  target.width = 4;
  target.height = 4;
  const adapter = ${lang === "glsl" ? "createGlsl(vertex, fragment)" : "createWgsl({ vertex, fragment })"};
  await adapter.attach(target);
  adapter.setAttribute(pos, Float32Array.of(-1, -1, 0, 3, -1, 0, -1, 3, 0));
  adapter.setAttribute(tint, Float32Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1));
  adapter.draw({ count: 3 });
  ${lang === "wgsl" ? "await adapter.device().queue.onSubmittedWorkDone();" : ""}
  return [readPixel(target, 0, 3), readPixel(target, 3, 0)];
};`;

  /**
   * @canon spec-a-fragment-stage-linked-to-a-vertex-stage-reads-an-attribute-through-a-varying
   */
  it("gives the fragment stage the attribute on WebGL with createGlsl", async () => {
    const pixel = await runInGpuPage(
      entry("glsl"),
      "__rmslFragmentAttributeRun",
      new URL(".", import.meta.url).pathname,
    );
    expectInterpolated(pixel);
  }, 120_000);

  /**
   * @canon spec-a-fragment-stage-linked-to-a-vertex-stage-reads-an-attribute-through-a-varying
   */
  it.skipIf(!WEBGPU)(
    "gives the fragment stage the attribute on WebGPU with createWgsl",
    async () => {
      const pixel = await runInWebGpuPage(
        entry("wgsl"),
        "__rmslFragmentAttributeRun",
        new URL(".", import.meta.url).pathname,
      );
      expectInterpolated(pixel);
    },
    120_000,
  );
});

describe.skipIf(!GPU_ENABLED)("a matrix attribute the fragment stage reads, drawn on a GPU", () => {
  /** Draws a triangle whose fragment stage reads a `mat2` attribute, which every vertex sets to the identity. */
  const entry = (lang: "glsl" | "wgsl") => `
import { Fn, attributeRaw, builtinPosition, vec2, vec4 } from "../rmsl";
import { ${lang === "glsl" ? "createGlsl" : "createWgsl"} } from "../${lang}";
${READ_PIXEL}
globalThis.__rmslMatrixAttributeRun = async () => {
  const pos = attributeRaw("pos", "vec3");
  const matrix = attributeRaw("matrix", "mat2");
  const vertex = Fn(() => builtinPosition().assign(vec4(pos, 1)))();
  const fragment = Fn(() => vec4(matrix.mul(vec2(1, 1)), 0, 1))();
  const target = document.createElement("canvas");
  target.width = 4;
  target.height = 4;
  const adapter = ${lang === "glsl" ? "createGlsl(vertex, fragment)" : "createWgsl({ vertex, fragment })"};
  await adapter.attach(target);
  adapter.setAttribute(pos, Float32Array.of(-1, -1, 0, 3, -1, 0, -1, 3, 0));
  adapter.setAttribute(matrix, Float32Array.of(1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1));
  adapter.draw({ count: 3 });
  ${lang === "wgsl" ? "await adapter.device().queue.onSubmittedWorkDone();" : ""}
  return readPixel(target, 1, 2);
};`;

  /**
   * @canon exception-a-matrix-attribute-the-fragment-stage-reads-is-a-matrix-varying-wgsl-rejects
   */
  it("gives the fragment stage the matrix on WebGL with createGlsl", async () => {
    const pixel = await runInGpuPage(entry("glsl"), "__rmslMatrixAttributeRun", new URL(".", import.meta.url).pathname);
    expect(pixel).toEqual({ r: 255, g: 255, b: 0, a: 255 });
  }, 120_000);

  /**
   * The device refuses the shader module, and the draw leaves the canvas as it was.
   *
   * @canon exception-a-matrix-attribute-the-fragment-stage-reads-is-a-matrix-varying-wgsl-rejects
   */
  it.skipIf(!WEBGPU)(
    "draws nothing on WebGPU with createWgsl",
    async () => {
      const pixel = await runInWebGpuPage(
        entry("wgsl"),
        "__rmslMatrixAttributeRun",
        new URL(".", import.meta.url).pathname,
      );
      expect(pixel).toEqual({ r: 0, g: 0, b: 0, a: 0 });
    },
    120_000,
  );
});

describe.skipIf(!GPU_ENABLED)("a layout Fn that reads the inputs of its stage, drawn on WebGL", () => {
  /**
   * @canon spec-a-layout-fn-reads-an-attribute
   * @canon spec-a-layout-fn-assigns-a-varying
   * @canon spec-a-layout-fn-reads-a-varying
   * @canon spec-a-layout-fn-reads-a-built-in-input
   */
  it("reads an attribute, a varying and fragCoord() from functions with createGlsl", async () => {
    const entry = `
import { Fn, attributeRaw, builtinPosition, fragCoord, varying, vec4, float } from "../rmsl";
import { createGlsl } from "../glsl";
${READ_PIXEL}
globalThis.__rmslLayoutFnInputs = async () => {
  try {
    const pos = attributeRaw("pos", "vec3");
    const tint = attributeRaw("tint", "float");
    const half = varying("float");
    const readsAttribute = Fn(() => tint).setLayout({ name: "readsAttribute", type: "float", inputs: [] });
    const readsVarying = Fn(() => half).setLayout({ name: "readsVarying", type: "float", inputs: [] });
    const readsCoord = Fn(() => fragCoord().x.mul(0).add(1)).setLayout({ name: "readsCoord", type: "float", inputs: [] });
    const assignsVarying = Fn(() => {
      half.assign(float(0.5));
      return float(1);
    }).setLayout({ name: "assignsVarying", type: "float", inputs: [] });
    const vertex = Fn(() => {
      assignsVarying();
      builtinPosition().assign(vec4(pos, 1));
    })();
    const fragment = Fn(() => vec4(readsAttribute(), readsVarying(), readsCoord(), 1))();
    const target = document.createElement("canvas");
    target.width = 4;
    target.height = 4;
    const adapter = createGlsl(vertex, fragment);
    adapter.attach(target);
    adapter.setAttribute(pos, Float32Array.of(-1, -1, 0, 3, -1, 0, -1, 3, 0));
    adapter.setAttribute(tint, Float32Array.of(1, 1, 1));
    adapter.draw({ count: 3 });
    return readPixel(target, 1, 2);
  } catch (e) { return "THREW " + String(e.message || e).slice(0, 300); }
};`;
    const pixel = await runInGpuPage(entry, "__rmslLayoutFnInputs", new URL(".", import.meta.url).pathname);
    expect(pixel).toEqual({ r: 255, g: 128, b: 255, a: 255 });
  }, 120_000);
});
