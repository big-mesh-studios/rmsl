import { afterAll, describe, expect, it } from "vitest";
import { Fn, attributeRaw, float, vec4 } from "../rmsl";
import { UsageError } from "../errors";
import { compileGlsl, createGlsl } from "../glsl";
import { compileWgsl, createWgsl } from "../wgsl";
import { createJs } from "../js";
import { createWasm } from "../wasm";
import { vertexFormatOf } from "../backends/vertex-format";
import { BufferAttribute } from "../scene";
import { GPU_ENABLED, releaseGpu, webgpuAvailable } from "../testing/gpu";
import { READ_PIXEL, runInGpuPage, runInWebGpuPage } from "../testing/browser";

const WEBGPU = GPU_ENABLED && (await webgpuAvailable());
const TARGETS = ["glsl", "wgsl", "js", "wasm"] as const;

/**
 * Draws over a 4×4 canvas with one adapter, named by `globalThis.__rmslTarget`.
 * `drawFlag(type, flags, read)` draws a triangle over the canvas whose clip w is
 * 1 where `read(flag)` holds and 0 where it does not, so it shows only where
 * every vertex reads true. `drawFragmentFlag(flags)` draws the whole triangle
 * and colours it green where the fragment stage reads the `bool` attribute true.
 * Each returns the pixel at (1, 2), or the code of the error that refused it.
 */
const BOOL_ATTRIBUTE = `
import { Fn, attributeRaw, builtinPosition, float, vec4 } from "../rmsl";
import { createGlsl } from "../glsl";
import { createWgsl } from "../wgsl";
import { createJs } from "../js";
import { createWasm } from "../wasm";
${READ_PIXEL}
const TRIANGLE = Float32Array.of(-1, -1, 0, 3, -1, 0, -1, 3, 0);
const target = globalThis.__rmslTarget;
const draw = async (type, flags, vertexOf, fragmentOf) => {
  try {
    const pos = attributeRaw("pos", "vec3");
    const flag = attributeRaw("flag", type);
    const vertex = () => Fn(() => { builtinPosition().assign(vertexOf(pos, flag)); })();
    const fragment = () => Fn(() => fragmentOf(flag))();
    const adapter =
      target === "glsl" ? createGlsl(vertex(), fragment())
      : target === "wgsl" ? createWgsl({ vertex: vertex(), fragment: fragment() })
      : target === "js" ? createJs(vertex, fragment)
      : createWasm(vertex, fragment);
    const canvas = document.createElement("canvas");
    canvas.width = 4;
    canvas.height = 4;
    adapter.setAttribute(pos, TRIANGLE);
    adapter.setAttribute(flag, flags);
    await adapter.attach(canvas);
    adapter.draw({ count: 3 });
    if (target === "wgsl") await adapter.device().queue.onSubmittedWorkDone();
    return readPixel(canvas, 1, 2);
  } catch (error) {
    return { refused: error.code ?? error.message };
  }
};
const drawFlag = (type, flags, read) =>
  draw(type, flags, (pos, flag) => vec4(pos, read(flag).select(float(1), float(0))), () => vec4(0, 1, 0, 1));
const drawFragmentFlag = (flags) =>
  draw("bool", flags, (pos) => vec4(pos, 1), (flag) => vec4(0, flag.select(float(1), float(0)), 0, 1));
const each = (value) => [value, value, value];
globalThis.__rmslBool = async () => ({
  minusOne: await drawFlag("bool", Int32Array.of(...each(-1)), (f) => f),
  largest: await drawFlag("bool", Uint32Array.of(...each(2 ** 32 - 1)), (f) => f),
  zero: await drawFlag("bool", Uint32Array.of(...each(0)), (f) => f),
});
globalThis.__rmslBoolVector = async () => ({
  bvec2: await drawFlag("bvec2", Uint32Array.of(...each([0, 2 ** 32 - 1]).flat()), (f) => f.x.not().and(f.y)),
  bvec3: await drawFlag("bvec3", Int32Array.of(...each([-1, 0, 7]).flat()), (f) => f.x.and(f.y.not()).and(f.z)),
  bvec4: await drawFlag("bvec4", Int32Array.of(...each([1, 0, 0, -1]).flat()), (f) => f.x.and(f.y.not()).and(f.z.not()).and(f.w)),
});
// The first vertex holds false and the other two true.
globalThis.__rmslBoolFragment = async () => drawFragmentFlag(Int32Array.of(0, 1, 1));
`;

const GREEN = { r: 0, g: 255, b: 0, a: 255 };
const BLACK = { r: 0, g: 0, b: 0, a: 255 };
const NOTHING = { r: 0, g: 0, b: 0, a: 0 };

/** Runs one entry of {@link BOOL_ATTRIBUTE} on `target`, in the page that target draws in. */
const run = (target: string, entry: string) =>
  (target === "wgsl" ? runInWebGpuPage : runInGpuPage)(
    `globalThis.__rmslTarget = ${JSON.stringify(target)};\n${BOOL_ATTRIBUTE}`,
    entry,
    import.meta.dirname,
  );

describe.skipIf(!GPU_ENABLED)("a bool attribute drawn", () => {
  afterAll(async () => {
    await releaseGpu();
  }, 120_000);

  /**
   * An `Int32Array` of -1 reads true only where `createGlsl` points it as unsigned.
   *
   * @canon bug-a-bool-attribute-is-declared-as-a-boolean-input
   * @canon bug-an-adapter-refuses-an-integer-array-for-a-bool-attribute
   * @canon spec-the-glsl-adapter-points-a-bool-attribute-at-unsigned-integers
   * @canon spec-a-bool-attribute-reads-true-where-its-integer-is-not-zero
   */
  it.fails(
    "reads true where its integer is not zero on glsl",
    async () => {
      expect(await run("glsl", "__rmslBool")).toEqual({ minusOne: GREEN, largest: GREEN, zero: NOTHING });
    },
    120_000,
  );

  /**
   * @canon bug-a-bool-attribute-is-declared-as-a-boolean-input
   * @canon bug-an-adapter-refuses-an-integer-array-for-a-bool-attribute
   * @canon spec-a-bool-attribute-reads-true-where-its-integer-is-not-zero
   */
  it.skipIf(!WEBGPU).fails(
    "reads true where its integer is not zero on wgsl",
    async () => {
      expect(await run("wgsl", "__rmslBool")).toEqual({ minusOne: GREEN, largest: GREEN, zero: NOTHING });
    },
    120_000,
  );

  for (const target of ["js", "wasm"]) {
    /**
     * @canon bug-an-adapter-refuses-an-integer-array-for-a-bool-attribute
     * @canon spec-a-bool-attribute-reads-true-where-its-integer-is-not-zero
     */
    it.fails(
      `reads true where its integer is not zero on ${target}`,
      async () => {
        expect(await run(target, "__rmslBool")).toEqual({ minusOne: GREEN, largest: GREEN, zero: NOTHING });
      },
      120_000,
    );
  }

  for (const target of ["glsl", "wgsl"]) {
    /**
     * @canon bug-a-bool-attribute-is-declared-as-a-boolean-input
     * @canon bug-an-adapter-refuses-an-integer-array-for-a-bool-attribute
     * @canon spec-a-boolean-vector-attribute-reads-each-component-against-zero
     */
    it.skipIf(target === "wgsl" && !WEBGPU).fails(
      `reads each component of a boolean vector on ${target}`,
      async () => {
        expect(await run(target, "__rmslBoolVector")).toEqual({ bvec2: GREEN, bvec3: GREEN, bvec4: GREEN });
      },
      120_000,
    );
  }

  for (const target of ["js", "wasm"]) {
    /**
     * @canon bug-an-adapter-refuses-an-integer-array-for-a-bool-attribute
     * @canon spec-a-boolean-vector-attribute-reads-each-component-against-zero
     */
    it.fails(
      `reads each component of a boolean vector on ${target}`,
      async () => {
        expect(await run(target, "__rmslBoolVector")).toEqual({ bvec2: GREEN, bvec3: GREEN, bvec4: GREEN });
      },
      120_000,
    );
  }

  for (const target of ["wgsl", "js", "wasm"]) {
    /**
     * The first vertex holds false, so a fragment that takes it reads false.
     *
     * @canon bug-a-bool-attribute-the-fragment-stage-reads-fails-on-the-gpu-targets
     * @canon bug-an-adapter-refuses-an-integer-array-for-a-bool-attribute
     * @canon spec-a-fragment-stage-reads-a-bool-attribute-of-the-first-vertex
     */
    it.skipIf(target === "wgsl" && !WEBGPU).fails(
      `reads a bool attribute of the first vertex in the fragment stage on ${target}`,
      async () => {
        expect(await run(target, "__rmslBoolFragment")).toEqual(BLACK);
      },
      120_000,
    );
  }

  /**
   * The last vertex holds true, so a fragment that takes it reads true.
   *
   * @canon bug-a-bool-attribute-the-fragment-stage-reads-fails-on-the-gpu-targets
   * @canon bug-an-adapter-refuses-an-integer-array-for-a-bool-attribute
   * @canon exception-a-glsl-fragment-stage-reads-a-bool-attribute-of-the-last-vertex
   */
  it.fails(
    "reads a bool attribute of the last vertex in the fragment stage on glsl",
    async () => {
      expect(await run("glsl", "__rmslBoolFragment")).toEqual(GREEN);
    },
    120_000,
  );
});

/** An adapter on `target` over a vertex stage that reads the `bool` attribute `flag`. */
const adapterWithFlag = (target: (typeof TARGETS)[number]) => {
  const pos = attributeRaw("pos", "vec3");
  const flag = attributeRaw("flag", "bool");
  const vertex = () => Fn(() => vec4(pos, flag.select(float(1), float(0))))();
  const fragment = () => Fn(() => vec4(0, 1, 0, 1))();
  const adapter =
    target === "glsl"
      ? createGlsl(vertex(), fragment())
      : target === "wgsl"
        ? createWgsl({ vertex: vertex(), fragment: fragment() })
        : target === "js"
          ? createJs(vertex, fragment)
          : createWasm(vertex, fragment);
  return { adapter: adapter as { setAttribute(slot: string, data: ArrayLike<number>): void }, flag };
};

describe("a bool attribute's array", () => {
  for (const target of TARGETS) {
    /**
     * @canon bug-an-adapter-refuses-an-integer-array-for-a-bool-attribute
     */
    it.fails(`is taken as an Int32Array or a Uint32Array on ${target}`, () => {
      const { adapter } = adapterWithFlag(target);
      expect(() => adapter.setAttribute("flag", Int32Array.of(1, 0, -1))).not.toThrow();
      expect(() => adapter.setAttribute("flag", Uint32Array.of(1, 0, 2 ** 32 - 1))).not.toThrow();
    });

    /**
     * @canon bug-an-adapter-takes-a-float-array-for-a-bool-attribute
     */
    it.fails(`is refused as a Float32Array or a Float64Array on ${target}`, () => {
      const { adapter } = adapterWithFlag(target);
      expect(() => adapter.setAttribute("flag", Float32Array.of(1, 0, 1))).toThrowRmsl(
        UsageError,
        "host-array-of-another-kind",
      );
      expect(() => adapter.setAttribute("flag", Float64Array.of(1, 0, 1))).toThrowRmsl(
        UsageError,
        "host-array-of-another-kind",
      );
    });
  }

  /**
   * @canon bug-a-scene-renderer-refuses-an-integer-array-for-a-bool-attribute
   */
  it.fails("reads through an unsigned format, and no float array, on a scene renderer", () => {
    const origin = { target: "wgsl", caller: "vertexFormatOf" } as const;
    const format = (array: Int32Array | Uint32Array | Float32Array) =>
      vertexFormatOf(origin, new BufferAttribute(array, 1), "bool");
    expect(format(Int32Array.of(1, 0, -1))).toBe("uint32");
    expect(format(Uint32Array.of(1, 0, 1))).toBe("uint32");
    expect(() => format(Float32Array.of(1, 0, 1))).toThrow();
  });
});

describe("a bool attribute in a compiled vertex stage", () => {
  /**
   * @canon bug-a-bool-attribute-is-declared-as-a-boolean-input
   * @canon spec-a-stage-compiler-declares-a-bool-attribute-as-an-unsigned-input
   */
  it.fails("is declared an unsigned input by compileGlsl and compileWgsl", () => {
    const flag = attributeRaw("flag", "bvec2");
    const stage = () => Fn(() => vec4(0, 0, 0, flag.x.select(float(1), float(0))))();
    expect(compileGlsl.vertex(stage())).toMatch(/\bin uvec2 flag;/);
    expect(compileWgsl.vertex(stage())).toMatch(/flag: vec2<u32>/);
  });
});
