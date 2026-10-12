import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import {
  attribute,
  builtinPosition,
  bvec3,
  Fn,
  For,
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
import { GPU_ENABLED, installWebGpuGlobals, releaseGpu, webgpuAvailable } from "../testing/gpu";
import { READ_PIXEL, runInGpuPage, runInWebGpuPage } from "../testing/browser";
import { stubDevice, stubGl } from "./scene-stubs";
import { CONDITION_PAIRS, inlineFnPaths } from "../testing/inline-fn-paths";
import { createGlsl } from "../glsl";
import { createWasmContext } from "../wasm";

const WEBGPU = await webgpuAvailable();

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
    expect(() => compileWgslStage.fragment(vec4(read, 0, 0, 1), { uniforms: [other] })).toThrow(
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
    ).toThrow(new RegExp(`"${missing.name}" is read by this stage but missing`));
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
    expect(() => compileWgslStage.fragment(fragment, { samplers: [image, gain] })).toThrow(
      `"${gain.name}" is a float uniform, not a texture`,
    );
    expect(() => compileWgslStage.fragment(fragment, { samplers: [image, image] })).toThrow(
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
    ).toThrow(`the storages passed to the compiler list the storage buffer "${values.name}" twice`);
  });

  /**
   * @canon spec-a-wgsl-stage-refuses-a-texture-the-given-samplers-leave-out
   */
  it("refuses a WGSL stage that samples a texture the samplers list leaves out", () => {
    const listed = uniform("sampler2D");
    const missing = uniform("sampler2D");
    const fragment = Fn(() => texture(listed, vec2(0, 0)).add(texture(missing, vec2(0, 0))))();
    expect(() => compileWgslStage.fragment(fragment, { samplers: [listed] })).toThrow(
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
    expect(() => compileWgslStage.vertex(vertex, { samplers: [uniformRaw("map", "sampler9D" as any)] })).toThrow(
      /sampler9D/,
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
    expect(() => adapter.setUniform(colours.name, 1)).toThrow(/storage buffer.*setAttribute/);
    expect(() => adapter.setUniform(image.name, 1)).toThrow(/texture.*setTexture/);
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
    adapter.attach({ getContext: () => gl } as any);
    expect(() => adapter.setAttribute(ids, Float32Array.of(1, 2) as any)).toThrow(
      /\bivec2\b.*Int32Array.*Float32Array/,
    );
    expect(() => adapter.setAttribute(ids.name, Float32Array.of(1, 2))).toThrow(/Int32Array.*Float32Array/);
    expect(() => adapter.setAttribute(flags.name, Int32Array.of(1))).toThrow(/Uint32Array.*Int32Array/);
    expect(() => adapter.setAttribute(position.name, Float64Array.of(1, 2, 3))).toThrow(/Float32Array.*Float64Array/);
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
      expect(() => adapter.setAttribute(ids.name, Float32Array.of(1, 2))).toThrow(/Int32Array.*Float32Array/);
      adapter.setAttribute(flags.name, Uint32Array.of(1));
      await adapter.attach(canvas);
      expect(() => adapter.setAttribute(ids, Float32Array.of(1, 2) as any)).toThrow(
        /\bivec2\b.*Int32Array.*Float32Array/,
      );
      expect(() => adapter.setAttribute(flags.name, Int32Array.of(1))).toThrow(/Uint32Array.*Int32Array/);
      expect(() => adapter.setAttribute(position.name, Float64Array.of(1, 2, 3))).toThrow(/Float32Array.*Float64Array/);
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
    expect(() => adapter.setAttribute(counts, new Float32Array(4) as any)).toThrow(
      /\bint\b storage buffer.*Int32Array.*Float32Array/,
    );
    expect(() => adapter.setAttribute(counts.name, new Uint32Array(4))).toThrow(/Int32Array.*Uint32Array/);
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
      expect(() => adapter.setAttribute(flags.name, new Int32Array(4))).toThrow(/Uint32Array.*Int32Array/);
      await adapter.attach(canvas);
      expect(() => adapter.setAttribute(flags, new Float32Array(4) as any)).toThrow(
        /\buvec2\b storage buffer.*Uint32Array.*Float32Array/,
      );
      expect(() => adapter.setAttribute(flags.name, new Int32Array(4))).toThrow(/Uint32Array.*Int32Array/);
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
    expect(() => setUniform(count, [2])).toThrow(
      `[RMSL] the int uniform "${count.name}" takes one value, not an array of 1`,
    );
    expect(() => setUniform(tint, [1, 2])).toThrow(
      `[RMSL] the vec3 uniform "${tint.name}" takes 3 components, not an array of 2`,
    );
    expect(() => setUniform(tint.name, 1)).toThrow(`[RMSL] the vec3 uniform "${tint.name}" takes 3 components, not 1`);
    expect(() =>
      setUniform(turn.name, [
        [1, 0],
        [0, 1],
      ]),
    ).toThrow(`[RMSL] the mat2 uniform "${turn.name}" takes 4 components, not an array of 2`);
    expect(() => setUniform(offsets.name, [1])).toThrow(
      `[RMSL] the int uniform array "${offsets.name}" takes 2 elements, not an array of 1`,
    );
    expect(() => setUniform(pairs, [[1, 2], [3]])).toThrow(
      `[RMSL] the vec2 uniform array "${pairs.name}" takes 2 components in each element, not an array of 1`,
    );
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
    expect(() => context.setUniform(count, [3] as any)).toThrow(/takes one value, not an array of 1/);
    expect(() => context.setUniform(tint, [1, 2] as any)).toThrow(
      `[RMSL] the vec3 uniform "${tint.name}" takes 3 components, not an array of 2`,
    );
    expect(() => context.setUniform(items, [1] as any)).toThrow(/takes 2 elements, not an array of 1/);
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
import { Fn, attribute, builtinPosition, float, select, uniform, uniformArray, varying, vec2, vec3, vec4 } from "../rmsl";
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
    const isNaN = (x) => x.greaterThanEqual(0).or(x.lessThan(0)).not();
    const all = isNaN(f.y)
      .and(f.z.equal(3))
      .and(isNaN(m.element(0).y))
      .and(i.y.toFloat().equal(0))
      .and(u.y.toFloat().equal(0))
      .and(b.y.not())
      .and(n.y.equal(0));
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
   * @canon spec-a-gpu-adapter-refuses-a-uniform-value-of-another-shape
   */
  it("sets a vector uniform from a typed array with createGlsl", async () => {
    expect(await glslValueType("typedVector")).toEqual(GREEN);
  }, 120_000);

  /**
   * The page tells NaN by its failing every comparison with zero.
   *
   * @canon spec-an-undefined-component-converts-as-a-typed-slot-does
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
          new RegExp(
            `^\\[RMSL\\] ${target === "JS" ? "createJs" : "createWasm"}: draw\\(\\) is missing attribute "_rmsl_a\\d+"`,
          ),
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
   */
  it("reads a component the host gives as undefined as a typed slot of its kind holds it on createWgslCompute", async () => {
    const f = uniform("vec3");
    const m = uniform("mat2");
    const i = uniform("ivec3");
    const u = uniform("uvec3");
    const b = uniform("bvec3");
    const n = uniform("vec3");
    const results = instancedArray(6, "float");
    const adapter = createWgslCompute(
      Fn(() => {
        results.element(int(0)).assign(f.y);
        results.element(int(1)).assign(m.element(int(0)).y);
        results.element(int(2)).assign(i.y.toFloat());
        results.element(int(3)).assign(u.y.toFloat());
        results.element(int(4)).assign(select(b.y, float(1), float(0)));
        results.element(int(5)).assign(n.y);
      })(),
    );
    await adapter.attach();
    adapter.setUniform(f, [1, undefined, 3] as any);
    adapter.setUniform(m, [1, undefined, 3, 4] as any);
    adapter.setUniform(i, [1, undefined, 3] as any);
    adapter.setUniform(u, [1, undefined, 3] as any);
    adapter.setUniform(b, [true, undefined, true] as any);
    adapter.setUniform(n, [1, null, 3] as any);
    const out = { [results.name]: new Float32Array(6) };
    await adapter.compute(out, 1);
    expect(Array.from(out[results.name]!)).toEqual([NaN, NaN, 0, 0, 0, 0]);
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
   * @canon exception-an-inline-fn-read-more-than-once-runs-before-every-read
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
    await expect(adapter.compute({ [counts.name]: new Float32Array(2) })).rejects.toThrow(
      `[RMSL] the int storage buffer "${counts.name}" takes an Int32Array, not a Float32Array`,
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
    await expect(adapter.compute({ [counts.name]: new Int32Array(2) })).rejects.toThrow(
      `[RMSL] out["${counts.name}"] holds 2 values, and the slot has 3`,
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
    await expect(adapter.compute({ other: new Int32Array(2) })).rejects.toThrow(/"other".*no storage slot/);
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
    await expect(adapter.compute({ [points.name]: new Float32Array(2) })).rejects.toThrow(RangeError);
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
    expect(() => adapter.setAttribute(counts.name, new Uint32Array(2))).toThrow(/Int32Array.*Uint32Array/);
    await adapter.attach();
    expect(() => adapter.setAttribute(counts, new Float32Array(2) as any)).toThrow(
      /\bint\b storage buffer.*Int32Array.*Float32Array/,
    );
    expect(() => adapter.setAttribute(counts.name, new Uint32Array(2))).toThrow(/Int32Array.*Uint32Array/);
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
    expect(() => compileWgslStage.fragment(Fn(() => vec4(uniformRaw("", "float"), 0, 0, 1))())).toThrow();
  });

  /**
   * A varying whose raw name is empty is refused.
   *
   * @canon spec-a-raw-name-declares-one-input-under-that-name
   */
  it("refuses a varying with an empty name on WGSL", () => {
    expect(() => compileWgslStage.fragment(Fn(() => vec4(varyingRaw("", "float"), 0, 0, 1))())).toThrow();
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
import { Fn, attribute, builtinPosition, vec4 } from "../rmsl";
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
globalThis.__rmslGlslDraws = {
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
import { Fn, attribute, builtinPosition, instancedArray, ivec2, uniform, vec2, vec4, texture } from "../rmsl";
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
const drawn = async (adapter, target) => {
  await adapter.device().queue.onSubmittedWorkDone();
  return readPixel(target, 1, 2);
};
globalThis.__rmslWgslDraws = {
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
