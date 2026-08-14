/**
 * Tests the vite plugins in `./vite` by driving their `transform` hook with the
 * real fixture modules. The fixtures import from `../rmsl`, so the build-time
 * evaluation path (esbuild bundle + data: URL import) is exercised against real
 * rmsl code, not stubs.
 */

/// <reference types="vite/client" />
import { describe, it, expect } from "vitest";
import { precompileShaders, precompileJS } from "./vite";
import shadersSource from "./vite-fixtures/shaders.ts?raw";
import cpuFnsSource from "./vite-fixtures/cpu-fns.ts?raw";

type TransformResult = { code: string; map: null } | null;
type TestablePlugin = { transform(code: string, id: string): Promise<TransformResult> };

const asPlugin = (plugin: unknown): TestablePlugin => plugin as TestablePlugin;

const importDataUrl = async (code: string) =>
  (await import(`data:text/javascript,${encodeURIComponent(code)}`)) as Record<string, any>;

// The id must be a real path on disk so esbuild can resolve the fixture's
// `../rmsl` import during build-time evaluation.
const SHADERS_PATH = new URL("./vite-fixtures/shaders.ts", import.meta.url).pathname;
const CPU_FNS_PATH = new URL("./vite-fixtures/cpu-fns.ts", import.meta.url).pathname;

describe("precompileShaders", () => {
  const source = shadersSource;

  it("rewrites a matching module to a JSON constant", async () => {
    const plugin = asPlugin(precompileShaders({ include: "vite-fixtures/shaders.ts" }));
    const result = await plugin.transform(source, SHADERS_PATH);

    expect(result).not.toBeNull();
    expect(result!.code).toMatch(/^export default \{"uColour":"uColour"/);
    expect(result!.code).toContain('"uColour":"uColour"');
    expect(result!.code).toContain('"vUv":"_rmsl_v');
    expect(result!.code).toContain('"positionAttr":"_rmsl_a');
    expect(result!.code).toContain('"vertexGLSL"');
    expect(result!.code).not.toContain("import");
    expect(result!.code).not.toContain("compileGLSL");
  });

  it("evaluates the rewritten module to the compiled shaders", async () => {
    const plugin = asPlugin(precompileShaders({ include: SHADERS_PATH }));
    const result = await plugin.transform(source, SHADERS_PATH);

    const mod = await importDataUrl(result!.code);
    expect(mod.default.vertexGLSL).toContain("#version 300 es");
    expect(mod.default.fragmentGLSL).toContain("#version 300 es");
    expect(mod.default.uColour).toBe("uColour");
    expect(mod.default.vUv).toMatch(/^_rmsl_v/);
    expect(mod.default.positionAttr).toMatch(/^_rmsl_a/);
  });

  it("leaves non-matching modules alone", async () => {
    const plugin = asPlugin(precompileShaders({ include: "vite-fixtures/shaders.ts" }));
    const result = await plugin.transform(source, "/elsewhere/other.ts");
    expect(result).toBeNull();
  });

  it("throws when the module has no default export", async () => {
    const plugin = asPlugin(precompileShaders({ include: "no-default.ts" }));
    const id = `${SHADERS_PATH.replace("shaders.ts", "")}no-default.ts`;
    await expect(plugin.transform("export const x = 1;\n", id)).rejects.toThrow(
      /must have a default export/,
    );
  });

  it("throws when the default export is not JSON-serializable", async () => {
    const plugin = asPlugin(precompileShaders({ include: "fn-default.ts" }));
    const id = `${SHADERS_PATH.replace("shaders.ts", "")}fn-default.ts`;
    await expect(plugin.transform("export default () => 1;\n", id)).rejects.toThrow(
      /not JSON-serializable/,
    );
  });
});

describe("precompileJS", () => {
  const source = cpuFnsSource;

  it("inlines each compileJSFn output as a plain function", async () => {
    const plugin = asPlugin(precompileJS({ include: "vite-fixtures/cpu-fns.ts" }));
    const result = await plugin.transform(source, CPU_FNS_PATH);

    expect(result).not.toBeNull();
    expect(result!.code).toContain("export const brightness = (() => {");
    expect(result!.code).toContain("export const mixColours = (() => {");
    expect(result!.code).not.toMatch(/\beval\s*\(/);
    expect(result!.code).not.toMatch(/import/);
    expect(result!.code).not.toContain("@random-mesh");
    expect(result!.code).not.toContain("compileJSFn");
  });

  it("produces callables that run on the CPU", async () => {
    const plugin = asPlugin(precompileJS({ include: CPU_FNS_PATH }));
    const result = await plugin.transform(source, CPU_FNS_PATH);

    const mod = await importDataUrl(result!.code);
    expect(typeof mod.brightness).toBe("function");
    expect(typeof mod.mixColours).toBe("function");

    // `uniform("vec3")` inside the fixture gets the first auto slot, _rmsl_u0.
    expect(mod.brightness({ uniforms: { _rmsl_u0: [1, 2, 3] } })).toEqual([0.5, 1, 1.5]);
    expect(
      mod.mixColours({ params: { a: [0, 0, 0], b: [1, 1, 1], t: 0.5 } }),
    ).toEqual([0.5, 0.5, 0.5]);
  });

  it("leaves non-matching modules alone", async () => {
    const plugin = asPlugin(precompileJS({ include: "vite-fixtures/cpu-fns.ts" }));
    const result = await plugin.transform(source, "/elsewhere/other.ts");
    expect(result).toBeNull();
  });

  it("throws when the module has no code export", async () => {
    const plugin = asPlugin(precompileJS({ include: "no-code.ts" }));
    const id = `${CPU_FNS_PATH.replace("cpu-fns.ts", "")}no-code.ts`;
    await expect(plugin.transform("export const x = 1;\n", id)).rejects.toThrow(
      /must export __RMSL_JS_CODE/,
    );
  });

  it("throws when the code export is not a map of strings", async () => {
    const plugin = asPlugin(precompileJS({ include: "bad-code.ts" }));
    const id = `${CPU_FNS_PATH.replace("cpu-fns.ts", "")}bad-code.ts`;
    await expect(
      plugin.transform('export const __RMSL_JS_CODE = { f: 1 };\n', id),
    ).rejects.toThrow(/map value for f must be a string/);
  });

  it("reads the code map from a custom export name", async () => {
    const plugin = asPlugin(
      precompileJS({ include: "custom-code.ts", codeExport: "CPU_FNS" }),
    );
    const id = `${CPU_FNS_PATH.replace("cpu-fns.ts", "")}custom-code.ts`;
    const code = source.replaceAll("__RMSL_JS_CODE", "CPU_FNS");
    const result = await plugin.transform(code, id);

    expect(result).not.toBeNull();
    expect(result!.code).toContain("export const brightness = (() => {");
  });
});
