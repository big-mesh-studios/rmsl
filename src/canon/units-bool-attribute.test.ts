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
 * Draws over a 4×4 canvas with one adapter, named by `globalThis.__rmslTarget`,
 * and returns the pixel at (1, 2), or the code of the error that refused it.
 *
 * `drawFlag` draws a triangle over the canvas whose clip w is 1 where
 * `read(flag)` holds and 0 where it does not, so it shows only where every
 * vertex reads true. `drawFragmentFlag` draws the whole triangle, green where
 * the fragment stage reads `read(flag)` true and black where it reads false.
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
const drawFragmentFlag = (type, flags, read) =>
  draw(type, flags, (pos) => vec4(pos, 1), (flag) => vec4(0, read(flag).select(float(1), float(0)), 0, 1));
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
// The first vertex holds false, and the other two true.
globalThis.__rmslBoolFragment = async () => drawFragmentFlag("bool", Int32Array.of(0, 1, 1), (f) => f);
// The first vertex holds (false, true), and the other two (true, false).
globalThis.__rmslBoolVectorFragment = async () =>
  drawFragmentFlag("bvec2", Int32Array.of(0, 1, 1, 0, 1, 0), (f) => f.x.not().and(f.y));
`;

/**
 * A plane drawn by `WebGLRenderer` or, given `"webgpu"`, by `WebGPURenderer`,
 * green where its material reads `holds(flag)` true of a boolean attribute
 * `flag` of `type`, and black where it reads it false. Each case gives the
 * geometry an array for `flag`, or none.
 */
const SCENE_BOOL = `
import { WebGLRenderer, WebGPURenderer, Scene, Mesh, PerspectiveCamera, PlaneGeometry, MeshBasicMaterial, BufferAttribute } from "../scene";
import { float, vec4 } from "../rmsl";
${READ_PIXEL}
const draw = async (backend, type, array, holds) => {
  const canvas = document.createElement("canvas");
  canvas.width = 16;
  canvas.height = 16;
  try {
    const renderer = backend === "webgpu" ? await WebGPURenderer.init(canvas) : new WebGLRenderer(canvas, { antialias: false });
    const errors = [];
    renderer.device?.addEventListener("uncapturederror", (event) => errors.push(event.error.message));
    const material = new MeshBasicMaterial();
    material.vertexNode = (b) => {
      b.varying("colour", "vec4").assign(vec4(0, holds(b.attribute("flag", type)).select(float(1), float(0)), 0, 1));
      return b.projectionMatrix.mul(b.viewMatrix.mul(b.modelMatrix.mul(vec4(b.position, 1))));
    };
    material.fragmentNode = (b) => b.varying("colour", "vec4");
    const geometry = new PlaneGeometry(2, 2);
    if (array) geometry.setAttribute("flag", new BufferAttribute(array, array.length / 4));
    const scene = new Scene();
    scene.add(new Mesh(geometry, material));
    const camera = new PerspectiveCamera(90, 1, 0.1, 100);
    camera.position.set(0, 0, 0.5);
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
    if (renderer.device) {
      await renderer.device.queue.onSubmittedWorkDone();
      return errors.length ? { errors } : readPixel(canvas, 8, 8);
    }
    const gl = renderer.gl;
    const pixels = new Uint8Array(4);
    gl.readPixels(8, 8, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    const error = gl.getError();
    return error ? { error } : { r: pixels[0], g: pixels[1], b: pixels[2], a: pixels[3] };
  } catch (error) {
    return { error: error.message };
  }
};
const each = (values) => [0, 1, 2, 3].flatMap(() => values);
globalThis.__rmslSceneBool = async (backend) => ({
  int32: await draw(backend, "bool", Int32Array.of(...each([-1])), (f) => f),
  uint32: await draw(backend, "bvec2", Uint32Array.of(...each([0, 2 ** 32 - 1])), (f) => f.x.not().and(f.y)),
});
globalThis.__rmslSceneMissingBool = async (backend) => ({
  bool: await draw(backend, "bool", null, (f) => f.not()),
  bvec2: await draw(backend, "bvec2", null, (f) => f.x.not().and(f.y.not())),
  bvec3: await draw(backend, "bvec3", null, (f) => f.x.not().and(f.y.not()).and(f.z.not())),
  bvec4: await draw(backend, "bvec4", null, (f) => f.w.and(f.x.not()).and(f.y.not()).and(f.z.not())),
});
`;

/** Runs an entry of {@link SCENE_BOOL} on `backend`, in the page that backend draws in. */
const runScene = (backend: "webgl" | "webgpu", entry: string) =>
  backend === "webgpu"
    ? runInWebGpuPage(
        `${SCENE_BOOL}\nglobalThis.__rmslSceneOnWebGpu = () => globalThis.${entry}("webgpu");`,
        "__rmslSceneOnWebGpu",
        import.meta.dirname,
      )
    : runInGpuPage(
        `${SCENE_BOOL}\nglobalThis.__rmslSceneOnWebGl = () => globalThis.${entry}("webgl");`,
        "__rmslSceneOnWebGl",
        import.meta.dirname,
      );

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
   * @canon spec-a-bool-attribute-reads-true-where-its-integer-is-not-zero
   * @canon spec-the-glsl-adapter-points-a-bool-attribute-at-unsigned-integers
   */
  it("reads true where its integer is not zero on glsl", async () => {
    expect(await run("glsl", "__rmslBool")).toEqual({ minusOne: GREEN, largest: GREEN, zero: NOTHING });
  }, 120_000);

  /**
   * @canon spec-a-bool-attribute-reads-true-where-its-integer-is-not-zero
   */
  it.skipIf(!WEBGPU)(
    "reads true where its integer is not zero on wgsl",
    async () => {
      expect(await run("wgsl", "__rmslBool")).toEqual({ minusOne: GREEN, largest: GREEN, zero: NOTHING });
    },
    120_000,
  );

  for (const target of ["js", "wasm"]) {
    /**
     * @canon spec-a-bool-attribute-reads-true-where-its-integer-is-not-zero
     */
    it(`reads true where its integer is not zero on ${target}`, async () => {
      expect(await run(target, "__rmslBool")).toEqual({ minusOne: GREEN, largest: GREEN, zero: NOTHING });
    }, 120_000);
  }

  for (const target of ["glsl", "wgsl"]) {
    /**
     * @canon spec-a-boolean-vector-attribute-reads-each-component-against-zero
     */
    it.skipIf(target === "wgsl" && !WEBGPU)(
      `reads each component of a boolean vector on ${target}`,
      async () => {
        expect(await run(target, "__rmslBoolVector")).toEqual({ bvec2: GREEN, bvec3: GREEN, bvec4: GREEN });
      },
      120_000,
    );
  }

  for (const target of ["js", "wasm"]) {
    /**
     * @canon spec-a-boolean-vector-attribute-reads-each-component-against-zero
     */
    it(`reads each component of a boolean vector on ${target}`, async () => {
      expect(await run(target, "__rmslBoolVector")).toEqual({ bvec2: GREEN, bvec3: GREEN, bvec4: GREEN });
    }, 120_000);
  }

  /**
   * The first vertex holds false, so a fragment that takes it reads false.
   *
   * @canon spec-a-fragment-stage-reads-a-bool-attribute-of-the-first-vertex
   * @canon exception-a-bool-attribute-the-fragment-stage-reads-travels-in-a-flat-unsigned-varying
   */
  it.skipIf(!WEBGPU)(
    "reads a bool attribute of the first vertex in the fragment stage on wgsl",
    async () => {
      expect(await run("wgsl", "__rmslBoolFragment")).toEqual(BLACK);
    },
    120_000,
  );

  /**
   * The first vertex holds (false, true), so a fragment that takes it reads
   * green.
   *
   * @canon spec-a-fragment-stage-reads-each-component-of-a-boolean-vector-attribute-of-the-first-vertex
   * @canon exception-a-bool-attribute-the-fragment-stage-reads-travels-in-a-flat-unsigned-varying
   */
  it.skipIf(!WEBGPU)(
    "reads each component of a boolean vector attribute of the first vertex in the fragment stage on wgsl",
    async () => {
      expect(await run("wgsl", "__rmslBoolVectorFragment")).toEqual(GREEN);
    },
    120_000,
  );

  for (const target of ["js", "wasm"]) {
    /**
     * @canon spec-a-fragment-stage-reads-a-bool-attribute-of-the-first-vertex
     * @canon exception-a-bool-attribute-the-fragment-stage-reads-travels-in-a-flat-unsigned-varying
     */
    it(`reads a bool attribute of the first vertex in the fragment stage on ${target}`, async () => {
      expect(await run(target, "__rmslBoolFragment")).toEqual(BLACK);
    }, 120_000);

    /**
     * @canon spec-a-fragment-stage-reads-each-component-of-a-boolean-vector-attribute-of-the-first-vertex
     */
    it(`reads each component of a boolean vector attribute of the first vertex in the fragment stage on ${target}`, async () => {
      expect(await run(target, "__rmslBoolVectorFragment")).toEqual(GREEN);
    }, 120_000);
  }

  /**
   * The last vertex holds true, so a fragment that takes it reads true, and
   * (true, false) for the vector, which reads black.
   *
   * @canon exception-a-glsl-fragment-stage-reads-a-bool-attribute-of-the-last-vertex
   * @canon exception-a-bool-attribute-the-fragment-stage-reads-travels-in-a-flat-unsigned-varying
   */
  it("reads a bool attribute of the last vertex in the fragment stage on glsl", async () => {
    expect(await run("glsl", "__rmslBoolFragment")).toEqual(GREEN);
    expect(await run("glsl", "__rmslBoolVectorFragment")).toEqual(BLACK);
  }, 120_000);

  for (const backend of ["webgl", "webgpu"] as const) {
    /**
     * @canon spec-a-scene-renderer-gives-a-bool-attribute-its-geometry-lacks-true-only-in-w
     */
    it.skipIf(backend === "webgpu" && !WEBGPU)(
      `gives a bool attribute its geometry lacks true only in w on ${backend}`,
      async () => {
        expect(await runScene(backend, "__rmslSceneMissingBool")).toEqual({
          bool: GREEN,
          bvec2: GREEN,
          bvec3: GREEN,
          bvec4: GREEN,
        });
      },
      120_000,
    );

    /**
     * @canon spec-a-scene-renderer-reads-a-bool-attribute-as-unsigned-integers
     */
    it.skipIf(backend === "webgpu" && !WEBGPU)(
      `draws a bool attribute from an Int32Array or a Uint32Array on ${backend}`,
      async () => {
        expect(await runScene(backend, "__rmslSceneBool")).toEqual({ int32: GREEN, uint32: GREEN });
      },
      120_000,
    );
  }
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
  return adapter as { setAttribute(slot: string, data: ArrayLike<number>): void };
};

describe("a bool attribute's array", () => {
  for (const target of TARGETS) {
    /**
     * @canon spec-an-adapter-takes-only-an-int32-or-uint32-array-for-a-bool-attribute
     */
    it(`is taken as an Int32Array or a Uint32Array on ${target}`, () => {
      const adapter = adapterWithFlag(target);
      expect(() => adapter.setAttribute("flag", Int32Array.of(1, 0, -1))).not.toThrow();
      expect(() => adapter.setAttribute("flag", Uint32Array.of(1, 0, 2 ** 32 - 1))).not.toThrow();
    });

    /**
     * @canon spec-an-adapter-takes-only-an-int32-or-uint32-array-for-a-bool-attribute
     * @canon exception-a-bool-attribute-refuses-the-float-array-tsl-reads
     */
    it(`is refused as a Float32Array or a Float64Array, naming what it takes, on ${target}`, () => {
      const adapter = adapterWithFlag(target);
      for (const array of [Float32Array.of(1, 0, 1), Float64Array.of(1, 0, 1)]) {
        expect(() => adapter.setAttribute("flag", array)).toThrowRmsl(
          UsageError,
          "host-array-of-another-kind",
          /bool.*Int32Array.*Uint32Array/,
        );
      }
    });

    /**
     * @canon spec-an-adapter-takes-only-an-int32-or-uint32-array-for-a-bool-attribute
     * @canon exception-a-bool-attribute-refuses-the-narrow-integer-array-tsl-reads
     */
    it(`is refused as an 8- or 16-bit integer array on ${target}`, () => {
      const adapter = adapterWithFlag(target);
      for (const array of [Uint8Array.of(1, 0, 1), Int16Array.of(1, 0, 1)]) {
        expect(() => adapter.setAttribute("flag", array)).toThrowRmsl(UsageError, "host-array-of-another-kind");
      }
    });
  }

  /**
   * @canon spec-a-scene-renderer-reads-a-bool-attribute-as-unsigned-integers
   * @canon exception-a-bool-attribute-refuses-the-float-array-tsl-reads
   * @canon exception-a-bool-attribute-refuses-the-narrow-integer-array-tsl-reads
   */
  it("reads through an unsigned format, and takes no float or narrow array, on a scene renderer", () => {
    const origin = { target: "wgsl", caller: "vertexFormatOf" } as const;
    const format = (array: Int32Array | Uint32Array | Float32Array | Uint8Array) =>
      vertexFormatOf(origin, new BufferAttribute(array, 1), "bool");
    expect(format(Int32Array.of(1, 0, -1))).toBe("uint32");
    expect(format(Uint32Array.of(1, 0, 1))).toBe("uint32");
    expect(() => format(Float32Array.of(1, 0, 1))).toThrow();
    expect(() => format(Uint8Array.of(1, 0, 1, 0))).toThrow();
  });
});

describe("a bool attribute in a compiled vertex stage", () => {
  /**
   * @canon spec-a-stage-compiler-declares-a-bool-attribute-as-an-unsigned-input
   */
  it("is declared an unsigned input by compileGlsl and compileWgsl", () => {
    const flag = attributeRaw("flag", "bvec2");
    const stage = () => Fn(() => vec4(0, 0, 0, flag.x.select(float(1), float(0))))();
    expect(compileGlsl.vertex(stage())).toMatch(/\bin uvec2 flag;/);
    expect(compileWgsl.vertex(stage())).toMatch(/flag: vec2<u32>/);
  });
});
