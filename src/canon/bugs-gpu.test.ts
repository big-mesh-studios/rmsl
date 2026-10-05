import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  bvec3,
  Fn,
  For,
  float,
  instancedArray,
  int,
  invocationIndex,
  attributeRaw,
  ivec2,
  vec2,
  ivec3,
  select,
  uniform,
  uniformRaw,
  varyingRaw,
  vec3,
  vec4,
  builtinPosition,
} from "../rmsl";
import { compileGlsl } from "../glsl";
import { compile, compileWgsl, createWgslCompute } from "../wgsl";
import { evaluateWGSL } from "../testing/shader-eval";
import { GPU_ENABLED, installWebGpuGlobals, releaseGpu, webgpuAvailable } from "../testing/gpu";
import { READ_PIXEL, runInGpuPage, runInWebGpuPage } from "../testing/browser";

const WEBGPU = await webgpuAvailable();

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
const GLSL_SCENE = `
import { Fn, attribute, builtinPosition, uniform, uniformArray, vec4 } from "../rmsl";
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
globalThis.__rmslBugsGlsl = {
  clear: () => attempt(() => {
    const target = canvas();
    const adapter = createGlsl(plainVertex(), Fn(() => vec4(0, 1, 0, 1))());
    adapter.attach(target);
    adapter.setAttribute(position, TRIANGLE);
    adapter.draw({ count: 3 });
    adapter.draw({ count: 0, clearColor: [0, 0, 1, 1] });
    return readPixel(target, 1, 2);
  }),
  uniformArray: () => attempt(() => {
    const target = canvas();
    const colours = uniformArray("vec4", 2);
    const adapter = createGlsl(plainVertex(), Fn(() => colours.element(1))());
    adapter.attach(target);
    adapter.setAttribute(position, TRIANGLE);
    adapter.setUniform(colours, [[1, 0, 0, 1], [0, 1, 0, 1]]);
    adapter.draw({ count: 3 });
    return readPixel(target, 1, 2);
  }),
  uintUniform: () => attempt(() => {
    const target = canvas();
    const green = uniform("uint");
    const adapter = createGlsl(plainVertex(), Fn(() => vec4(0, green.toFloat(), 0, 1))());
    adapter.attach(target);
    adapter.setAttribute(position, TRIANGLE);
    adapter.setUniform(green, 1);
    adapter.draw({ count: 3 });
    return readPixel(target, 1, 2);
  }),
  intAttribute: () => attempt(() => {
    const target = canvas();
    const shift = attribute("int");
    const vertex = Fn(() => { builtinPosition().assign(vec4(position.x.add(shift.toFloat()), position.y, 0, 1)); })();
    const adapter = createGlsl(vertex, Fn(() => vec4(0, 1, 0, 1))());
    adapter.attach(target);
    adapter.setAttribute(position, TRIANGLE);
    adapter.setAttribute(shift, Int32Array.of(0, 0, 0));
    adapter.draw({ count: 3 });
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
const WGSL_SCENE = `
import { Fn, attribute, builtinPosition, instancedArray, uniform, vec2, vec4 } from "../rmsl";
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
globalThis.__rmslBugsWgsl = {
  texture: () => attempt(async () => {
    const image = uniform("sampler2D");
    const adapter = createWgsl({ vertex: plainVertex(), fragment: Fn(() => image.texture(vec2(0.5, 0.5)))() });
    await adapter.attach(canvas());
    return "attached";
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

/** Runs one entry of `GLSL_SCENE` in the WebGL page. */
const glslEntry = (name: string) =>
  runInGpuPage(
    `${GLSL_SCENE}\nglobalThis.__rmslBugsGlslRun = async () => globalThis.__rmslBugsGlsl.${name}();`,
    "__rmslBugsGlslRun",
    new URL(".", import.meta.url).pathname,
  );

/** Runs one entry of `WGSL_SCENE` in the WebGPU page. */
const wgslEntry = (name: string) =>
  runInWebGpuPage(
    `${WGSL_SCENE}\nglobalThis.__rmslBugsWgslRun = async () => globalThis.__rmslBugsWgsl.${name}();`,
    "__rmslBugsWgslRun",
    new URL(".", import.meta.url).pathname,
  );

const GREEN = { r: 0, g: 255, b: 0, a: 255 };

let uninstall: (() => void) | undefined;

beforeAll(async () => {
  if (GPU_ENABLED) uninstall = await installWebGpuGlobals();
});

afterAll(async () => {
  uninstall?.();
  await releaseGpu();
}, 120_000);

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

describe("known GPU bugs, each failing until its fix", () => {
  /**
   * @canon bug-wgsl-takes-the-storages-list-as-slot-names
   */
  it.fails("takes the storage nodes themselves as the storages list on WGSL", () => {
    const first = instancedArray(2, "float");
    const second = instancedArray(2, "float");
    const code = compileWgsl.fragment(vec4(first.element(int(0)), second.element(int(0)), 0, 1), {
      storages: [second, first] as any,
    });
    expect(bindingOfRead(code, 1)).toBe("0");
    expect(bindingOfRead(code, 0)).toBe("1");
  });

  /**
   * @canon bug-wgsl-compute-ignores-the-storages-list
   */
  it.fails("binds compute storage at its index in the storages list on WGSL", () => {
    const first = instancedArray(2, "float");
    const second = instancedArray(2, "float");
    const code = compileWgsl.compute(
      Fn(() => {
        second.element(invocationIndex()).assign(first.element(invocationIndex()));
      })(),
      { storages: [second.name, first.name] },
    );
    expect(bindingOfRead(code, 1)).toBe("0");
    expect(bindingOfRead(code, 0)).toBe("1");
  });

  /**
   * @canon bug-wgsl-binds-storage-in-the-string-order-of-slot-names
   */
  it.fails("binds storage in the order the program made it when no list is given on WGSL", () => {
    const { first, second } = buffersAcrossADigit();
    const compute = compileWgsl.compute(
      Fn(() => {
        second.element(invocationIndex()).assign(first.element(invocationIndex()));
      })(),
    );
    const render = compileWgsl.fragment(vec4(first.element(int(0)), second.element(int(0)), 0, 1));
    for (const code of [compute, render]) {
      expect(bindingOfRead(code, 0)).toBe("0");
      expect(bindingOfRead(code, 1)).toBe("1");
    }
  });

  /**
   * Two uniforms of one alignment, `zeta` made first, so creation order puts
   * it before `alpha` in the struct.
   *
   * @canon bug-wgsl-hands-uniforms-to-the-layout-in-the-string-order-of-slot-names
   */
  it.fails("declares uniforms of one alignment in the order the program made them on WGSL", () => {
    const zeta = uniformRaw("zeta", "float");
    const alpha = uniformRaw("alpha", "float");
    const code = compileWgsl.fragment(vec4(zeta, alpha, 0, 1));
    const members = [...(/struct _RmslUniforms \{([\s\S]*?)\n\};/.exec(code)?.[1] ?? "").matchAll(/^\s*(\w+):/gm)].map(
      (m) => m[1],
    );
    expect(members).toEqual(["zeta", "alpha"]);
  });

  /**
   * @canon bug-wgsl-takes-the-uniforms-list-as-slot-declarations
   */
  it.fails("takes the uniform nodes themselves as the uniforms list on WGSL", () => {
    const read = uniform("float");
    const other = uniform("vec2");
    const code = compileWgsl.fragment(vec4(read, 0, 0, 1), { uniforms: [read, other] as any });
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
      values.element(invocationIndex()).assign(image.texture(ivec2(0, 0)).x.toFloat());
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
      builtinPosition().assign(vec4(near.texture(ivec2(0, 0)).x.toFloat(), 0, 0, 1));
    })();
    const fragment = Fn(() => vec4(far.texture(ivec2(0, 0)).x.toFloat(), 0, 0, 1))();
    const samplers = [near, far].map((node) => ({ slot: node.name, type: "isampler2D" }));
    const vertexCode = compileWgsl.vertex(vertex, { samplers });
    const fragmentCode = compileWgsl.fragment(fragment, { samplers });
    expect(bindingOf(vertexCode, near.name)).toBeDefined();
    expect(bindingOf(vertexCode, near.name)).not.toBe(bindingOf(vertexCode, far.name));
    expect(bindingOf(vertexCode, near.name)).toBe(bindingOf(fragmentCode, near.name));
    expect(bindingOf(vertexCode, far.name)).toBe(bindingOf(fragmentCode, far.name));
  });

  /**
   * A texture sampled in the vertex stage is read at level 0, because only a
   * fragment stage has the derivatives that pick a level, and a stage
   * numbers the samplers of the textures it samples in the order of their
   * names.
   *
   * @canon spec-a-texture-is-bound-to-every-stage-that-samples-it
   */
  it("samples a texture in the vertex stage at level 0, and numbers samplers by name on WGSL", () => {
    const zebra = uniformRaw("zebra", "sampler2D");
    const apple = uniformRaw("apple", "sampler2D");
    const vertex = Fn(() => {
      builtinPosition().assign(vec4(zebra.texture(vec2(0, 0)).x.add(apple.texture(vec2(0, 0)).x), 0, 0, 1));
    })();
    const code = compileWgsl.vertex(vertex);
    expect(code).toContain("textureSampleLevel(");
    expect(code).not.toMatch(/textureSample\(/);
    expect(bindingOf(code, `${apple.name}_s`)).toBe("2:0");
    expect(bindingOf(code, `${zebra.name}_s`)).toBe("2:1");
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
    expect(() => compileWgsl.vertex(vertex, { samplers: [{ slot: "map", type: "sampler9D" }] })).toThrow(/sampler9D/);
  });

  /**
   * A uniform whose raw name is empty is refused.
   *
   * @canon spec-a-raw-name-declares-one-input-under-that-name
   */
  it("refuses a uniform with an empty name on WGSL", () => {
    expect(() => compileWgsl.fragment(Fn(() => vec4(uniformRaw("", "float"), 0, 0, 1))())).toThrow();
  });

  /**
   * A varying whose raw name is empty is refused.
   *
   * @canon spec-a-raw-name-declares-one-input-under-that-name
   */
  it("refuses a varying with an empty name on WGSL", () => {
    expect(() => compileWgsl.fragment(Fn(() => vec4(varyingRaw("", "float"), 0, 0, 1))())).toThrow();
  });

  /**
   * An attribute whose raw name is empty is refused.
   *
   * @canon spec-a-raw-name-declares-one-input-under-that-name
   */
  it("refuses an attribute with an empty name", () => {
    expect(() => attributeRaw("", "float")).toThrow(/empty/);
  });

  /**
   * A `For` whose init makes no statement, such as one handed a variable
   * made before the loop, compiles on WGSL to a header with an empty init.
   *
   * @canon spec-a-for-loops-over-a-variable-its-init-is-given
   */
  it("compiles a For whose init makes no statement to a valid WGSL header", () => {
    const code = compileWgsl.fragment(
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

describe.skipIf(!GPU_ENABLED)("known GPU bugs on a WebGPU device, each failing until its fix", () => {
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

describe.skipIf(!GPU_ENABLED)("createGlsl in a browser", () => {
  /**
   * `createGlsl` never clears its canvas and takes no clear colour, so a draw
   * leaves what an earlier draw put there.
   *
   * @canon bug-the-glsl-adapter-never-clears
   */
  it.fails(
    "clears the canvas to the colour a createGlsl draw asks for",
    async () => {
      expect(await glslEntry("clear")).toEqual({ r: 0, g: 0, b: 255, a: 255 });
    },
    120_000,
  );

  /**
   * WebGL reports a uniform array as `name[0]`, and `createGlsl` looks the
   * slot up by that name, so `setUniform` on a uniform array never applies.
   *
   * @canon bug-the-glsl-adapter-never-sets-a-uniform-array
   */
  it.fails(
    "sets a uniform array with createGlsl",
    async () => {
      expect(await glslEntry("uniformArray")).toEqual(GREEN);
    },
    120_000,
  );

  /**
   * `createGlsl.setUniform` uploads only float, int and bool scalars and
   * vectors and square matrices, and throws for a `uint` uniform.
   *
   * @canon bug-the-glsl-adapter-refuses-a-uint-uniform
   */
  it.fails(
    "sets a uint uniform with createGlsl",
    async () => {
      expect(await glslEntry("uintUniform")).toEqual(GREEN);
    },
    120_000,
  );

  /**
   * `createGlsl` points every attribute at its buffer as floats, so an `int`
   * attribute mismatches its declaration and the draw is refused.
   *
   * @canon bug-the-glsl-adapter-uploads-an-integer-attribute-as-floats
   */
  it.fails(
    "draws with an int attribute through createGlsl",
    async () => {
      expect(await glslEntry("intAttribute")).toEqual(GREEN);
    },
    120_000,
  );

  /**
   * A `createGlsl` draw that names a first vertex and no count draws the vertices after it.
   *
   * @canon spec-a-gpu-adapter-takes-its-count-from-the-first-attribute
   */
  it("counts a createGlsl draw from the first attribute, less its first vertex", async () => {
    expect(await glslEntry("firstVertex")).toEqual(GREEN);
  }, 120_000);

  /**
   * A `createGlsl` draw that names no count takes the first attribute's, and
   * so does not draw vertices past its end.
   *
   * @canon spec-a-gpu-adapter-takes-its-count-from-the-first-attribute
   */
  it("takes the count of a createGlsl draw from the first attribute", async () => {
    expect(await glslEntry("firstAttributeCount")).toEqual(GREEN);
  }, 120_000);
});

describe.skipIf(!WEBGPU)("createWgsl in a browser", () => {
  /**
   * `createWgsl` puts a texture among the members of its uniform struct,
   * which has no layout for it, so a program that reads a texture does not
   * attach.
   *
   * @canon bug-the-wgsl-adapter-packs-a-texture-into-its-uniform-struct
   */
  it.fails(
    "attaches a program that samples a texture with createWgsl",
    async () => {
      expect(await wgslEntry("texture")).toBe("attached");
    },
    120_000,
  );

  /**
   * A `createWgsl` draw that names a first vertex and no count draws the vertices after it.
   *
   * @canon spec-a-gpu-adapter-takes-its-count-from-the-first-attribute
   */
  it("counts a createWgsl draw from the first attribute, less its first vertex", async () => {
    expect(await wgslEntry("firstVertex")).toEqual(GREEN);
  }, 120_000);

  /**
   * A `createWgsl` draw that names no count takes the first attribute's, and
   * so does not read past a shorter buffer.
   *
   * @canon spec-a-gpu-adapter-takes-its-count-from-the-first-attribute
   */
  it("takes the count of a createWgsl draw from the first attribute", async () => {
    expect(await wgslEntry("firstAttributeCount")).toEqual(GREEN);
  }, 120_000);
});

describe.skipIf(!GPU_ENABLED)("WGSL compute keeps a buffer for each slot", () => {
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
    // The refused call still dispatched, so the first element has been incremented twice.
    await adapter.compute(out);
    adapter.destroy();
    expect(Array.from(out[points.name]!.subarray(0, 3))).toEqual([3, 4, 5]);
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
});

describe.skipIf(!WEBGPU)("createWgsl in a browser holds a vec3 storage buffer in 16 bytes", () => {
  /**
   * @canon spec-a-wgsl-storage-buffer-holds-a-vec3-in-16-bytes
   */
  it("draws from a vec3 storage buffer of its own with createWgsl", async () => {
    expect(await wgslEntry("vec3Storage")).toEqual(GREEN);
  }, 120_000);
});
