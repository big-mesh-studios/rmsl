import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  Fn,
  float,
  instancedArray,
  int,
  invocationIndex,
  outputStruct,
  select,
  uniform,
  uniformRaw,
  vec2,
  vec3,
  vec4,
} from "../rmsl";
import { compile, compileWgsl as compileWgslStage, createWgslCompute, createWgslContext } from "../wgsl";
import {
  assertRecordedShadersValid,
  recordingGLSL as compileGlsl,
  recordingWGSL as compileWgsl,
} from "../testing/shader-validity";
import { GPU_ENABLED, installWebGpuGlobals, releaseGpu, webgpuAvailable } from "../testing/gpu";
import { READ_PIXEL, runInGpuPage, runInWebGpuPage } from "../testing/browser";

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
    const code = compileWgslStage.fragment(Fn(() => first!.texture(vec2(0, 0)).add(second!.texture(vec2(0, 0))))());
    expect(code).toContain(`@group(1) @binding(0) var ${first!.name}:`);
    expect(code).toContain(`@group(1) @binding(1) var ${second!.name}:`);
    expect(code).toContain(`@group(2) @binding(0) var ${first!.name}_s:`);
    expect(code).toContain(`@group(2) @binding(1) var ${second!.name}_s:`);
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
import { Fn, attribute, builtinPosition, float, uniform, uniformArray, vec2, vec4 } from "../rmsl";
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
   * @canon spec-a-webgl-draw-gives-an-attribute-with-no-data-a-fresh-value
   */
  it("draws with an integer attribute given no data through createGlsl", async () => {
    expect(await glslValueType("intAttributeWithoutData")).toEqual(GREEN);
  }, 120_000);
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
});
