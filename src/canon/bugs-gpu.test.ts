import { afterAll, describe, expect, it } from "vitest";
import { GPU_ENABLED, releaseGpu, webgpuAvailable } from "../testing/gpu";
import { READ_PIXEL, runInGpuPage, runInWebGpuPage } from "../testing/browser";

const WEBGPU = GPU_ENABLED && (await webgpuAvailable());

/**
 * A triangle over a 4×4 canvas whose clip w is 1 where the `bool` attribute
 * `flag` holds and 0 where it does not, so the triangle shows only when every
 * vertex reads true. `drawFlag(target, flags)` draws it with `flags` as the
 * attribute's array on one adapter, or returns the error that refused it.
 * `drawFragmentFlag` reads the attribute in the fragment stage instead, as
 * green where it holds, from whichever of an `Int32Array` and a `Float32Array`
 * the adapter takes.
 */
const BOOL_ATTRIBUTE = `
import { Fn, attributeRaw, builtinPosition, float, vec4 } from "../rmsl";
import { createGlsl } from "../glsl";
import { createWgsl } from "../wgsl";
import { createJs } from "../js";
import { createWasm } from "../wasm";
${READ_PIXEL}
const TRIANGLE = Float32Array.of(-1, -1, 0, 3, -1, 0, -1, 3, 0);
const make = (target, inFragment) => {
  const pos = attributeRaw("pos", "vec3");
  const flag = attributeRaw("flag", "bool");
  const vertex = () => inFragment
    ? Fn(() => { builtinPosition().assign(vec4(pos, 1)); })()
    : Fn(() => { builtinPosition().assign(vec4(pos, flag.select(float(1), float(0)))); })();
  const fragment = () => inFragment ? Fn(() => vec4(0, flag.select(float(1), float(0)), 0, 1))() : Fn(() => vec4(0, 1, 0, 1))();
  const adapter =
    target === "glsl" ? createGlsl(vertex(), fragment())
    : target === "wgsl" ? createWgsl({ vertex: vertex(), fragment: fragment() })
    : target === "js" ? createJs(vertex, fragment)
    : createWasm(vertex, fragment);
  return { adapter, pos, flag };
};
const draw = async (target, flags, inFragment) => {
  try {
    const { adapter, pos, flag } = make(target, inFragment);
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
globalThis.__rmslBoolAttribute = async () => ({
  minusOne: await draw(globalThis.__rmslTarget, Int32Array.of(-1, -1, -1), false),
  largest: await draw(globalThis.__rmslTarget, Uint32Array.of(2 ** 32 - 1, 2 ** 32 - 1, 2 ** 32 - 1), false),
  zero: await draw(globalThis.__rmslTarget, Uint32Array.of(0, 0, 0), false),
});
const drawTaken = async (target, value) => {
  const result = await draw(target, Int32Array.of(value, value, value), true);
  return result.refused === "host-array-of-another-kind" ? draw(target, Float32Array.of(value, value, value), true) : result;
};
globalThis.__rmslBoolFragmentAttribute = async () => ({
  one: await drawTaken(globalThis.__rmslTarget, 1),
  zero: await drawTaken(globalThis.__rmslTarget, 0),
});
`;

const GREEN = { r: 0, g: 255, b: 0, a: 255 };
const NOTHING = { r: 0, g: 0, b: 0, a: 0 };
const BLACK = { r: 0, g: 0, b: 0, a: 255 };

/** Runs one entry of {@link BOOL_ATTRIBUTE} on `target`, in the page that target draws in. */
const run = (target: string, entry: string) =>
  (target === "wgsl" ? runInWebGpuPage : runInGpuPage)(
    `globalThis.__rmslTarget = ${JSON.stringify(target)};\n${BOOL_ATTRIBUTE}`,
    entry,
    import.meta.dirname,
  );

describe.skipIf(!GPU_ENABLED)("a bool attribute", () => {
  afterAll(async () => {
    await releaseGpu();
  }, 120_000);

  for (const target of ["glsl", "wgsl", "js", "wasm"]) {
    /**
     * @canon bug-a-bool-attribute-fails-on-the-gpu-targets
     */
    it.skipIf(target === "wgsl" && !WEBGPU).fails(
      `reads true where its integer is not zero on ${target}`,
      async () => {
        expect(await run(target, "__rmslBoolAttribute")).toEqual({ minusOne: GREEN, largest: GREEN, zero: NOTHING });
      },
      120_000,
    );
  }

  /**
   * Which of the two the fragment stage gets is open, so the test holds the
   * targets to one another rather than to a value.
   *
   * @canon bug-a-bool-attribute-the-fragment-stage-reads-fails-on-the-gpu-targets
   */
  it.skipIf(!WEBGPU).fails(
    "means the same on every target when the fragment stage reads it",
    async () => {
      const outcomes = await Promise.all(
        ["glsl", "wgsl", "js", "wasm"].map((t) => run(t, "__rmslBoolFragmentAttribute")),
      );
      for (const outcome of outcomes.slice(1)) expect(outcome).toEqual(outcomes[0]);
      if (!("refused" in outcomes[0].one)) expect(outcomes[0]).toEqual({ one: GREEN, zero: BLACK });
    },
    120_000,
  );
});
