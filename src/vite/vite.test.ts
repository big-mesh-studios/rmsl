/// <reference types="vite/client" />
import { describe, it, expect } from "vitest";
import { precompileShaders, precompileJS, precompileWasm } from "./vite";
import shadersSource from "./fixtures/shaders.ts?raw";
import cpuFnsSource from "./fixtures/cpu-fns.ts?raw";
import wasmFnsSource from "./fixtures/wasm-fns.ts?raw";

type TransformResult = { code: string; map: null } | null;
type TestablePlugin = {
  transform: {
    (code: string, id: string): Promise<TransformResult>;
    call(context: unknown, code: string, id: string): Promise<TransformResult>;
  };
};

const asPlugin = (plugin: unknown): TestablePlugin => plugin as TestablePlugin;

const importDataUrl = async (code: string) =>
  (await import(`data:text/javascript,${encodeURIComponent(code)}`)) as Record<string, any>;

// The id must be a real path on disk so esbuild can resolve the fixture's
// `../rmsl` import during build-time evaluation.
const SHADERS_PATH = new URL("./fixtures/shaders.ts", import.meta.url).pathname;
const CPU_FNS_PATH = new URL("./fixtures/cpu-fns.ts", import.meta.url).pathname;
const WASM_FNS_PATH = new URL("./fixtures/wasm-fns.ts", import.meta.url).pathname;

/** A minimal Rollup PluginContext stand-in — just enough for precompileWasm's emitFile calls. */
function mockPluginContext() {
  const emitted: { type: string; name?: string; source: Uint8Array }[] = [];
  let nextId = 0;
  return {
    context: {
      emitFile(asset: { type: string; name?: string; source: Uint8Array }) {
        emitted.push(asset);
        return `ref${nextId++}`;
      },
    },
    emitted,
  };
}

function bytesToBase64(bytes: Uint8Array): string {
  return Buffer.from(bytes).toString("base64");
}

describe("precompileShaders", () => {
  const source = shadersSource;

  it("rewrites a matching module to a JSON constant", async () => {
    const plugin = asPlugin(precompileShaders({ include: "fixtures/shaders.ts" }));
    const result = await plugin.transform(source, SHADERS_PATH);

    expect(result).not.toBeNull();
    expect(result!.code).toMatch(/^export default \{"uColour":"uColour"/);
    expect(result!.code).toContain('"uColour":"uColour"');
    expect(result!.code).toContain('"vUv":"_rmsl_v');
    expect(result!.code).toContain('"positionAttr":"_rmsl_a');
    expect(result!.code).toContain('"vertexGLSL"');
    expect(result!.code).not.toContain("import");
    expect(result!.code).not.toContain("compileGlsl");
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
    const plugin = asPlugin(precompileShaders({ include: "fixtures/shaders.ts" }));
    const result = await plugin.transform(source, "/elsewhere/other.ts");
    expect(result).toBeNull();
  });

  it("throws when the module has no default export", async () => {
    const plugin = asPlugin(precompileShaders({ include: "no-default.ts" }));
    const id = `${SHADERS_PATH.replace("shaders.ts", "")}no-default.ts`;
    await expect(plugin.transform("export const x = 1;\n", id)).rejects.toThrow(/must have a default export/);
  });

  it("throws when the default export is not JSON-serializable", async () => {
    const plugin = asPlugin(precompileShaders({ include: "fn-default.ts" }));
    const id = `${SHADERS_PATH.replace("shaders.ts", "")}fn-default.ts`;
    await expect(plugin.transform("export default () => 1;\n", id)).rejects.toThrow(/not JSON-serializable/);
  });
});

describe("precompileJS", () => {
  const source = cpuFnsSource;

  it("inlines each compileJSFn output as a plain function", async () => {
    const plugin = asPlugin(precompileJS({ include: "fixtures/cpu-fns.ts" }));
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
    expect(mod.mixColours({ params: { a: [0, 0, 0], b: [1, 1, 1], t: 0.5 } })).toEqual([0.5, 0.5, 0.5]);
  });

  it("leaves non-matching modules alone", async () => {
    const plugin = asPlugin(precompileJS({ include: "fixtures/cpu-fns.ts" }));
    const result = await plugin.transform(source, "/elsewhere/other.ts");
    expect(result).toBeNull();
  });

  it("throws when the module has no code export", async () => {
    const plugin = asPlugin(precompileJS({ include: "no-code.ts" }));
    const id = `${CPU_FNS_PATH.replace("cpu-fns.ts", "")}no-code.ts`;
    await expect(plugin.transform("export const x = 1;\n", id)).rejects.toThrow(/must export __RMSL_JS_CODE/);
  });

  it("throws when the code export is not a map of strings", async () => {
    const plugin = asPlugin(precompileJS({ include: "bad-code.ts" }));
    const id = `${CPU_FNS_PATH.replace("cpu-fns.ts", "")}bad-code.ts`;
    await expect(plugin.transform("export const __RMSL_JS_CODE = { f: 1 };\n", id)).rejects.toThrow(
      /map value for f must be a string/,
    );
  });

  it("reads the code map from a custom export name", async () => {
    const plugin = asPlugin(precompileJS({ include: "custom-code.ts", codeExport: "CPU_FNS" }));
    const id = `${CPU_FNS_PATH.replace("cpu-fns.ts", "")}custom-code.ts`;
    const code = source.replaceAll("__RMSL_JS_CODE", "CPU_FNS");
    const result = await plugin.transform(code, id);

    expect(result).not.toBeNull();
    expect(result!.code).toContain("export const brightness = (() => {");
  });
});

describe("precompileWasm", () => {
  const source = wasmFnsSource;

  it("emits each compiled module as a .wasm asset and rewrites the export to fetch it", async () => {
    const plugin = asPlugin(precompileWasm({ include: "fixtures/wasm-fns.ts" }));
    const { context, emitted } = mockPluginContext();
    const result = await plugin.transform.call(context, source, WASM_FNS_PATH);

    expect(result).not.toBeNull();
    expect(result!.code).toContain('import { instantiateWasm } from "@random-mesh/rmsl/wasm";');
    expect(result!.code).toContain("export const brightness = instantiateWasm(");
    expect(result!.code).toContain("export const mixColours = instantiateWasm(");
    expect(result!.code).toMatch(/fetch\(new URL\(import\.meta\.ROLLUP_FILE_URL_ref\d+, import\.meta\.url\)\)/);
    expect(result!.code).not.toContain("compileWasmFn");
    expect(result!.code).not.toMatch(/\beval\s*\(/);

    expect(emitted).toHaveLength(2);
    expect(emitted.map((e) => e.name).sort()).toEqual(["brightness.wasm", "mixColours.wasm"]);
    for (const asset of emitted) {
      expect(asset.type).toBe("asset");
      expect(asset.source).toBeInstanceOf(Uint8Array);
      expect((asset.source as Uint8Array).length).toBeGreaterThan(0);
    }
  });

  it("produces callables that run on WASM, once the emitted-asset placeholder resolves to a real URL", async () => {
    const plugin = asPlugin(precompileWasm({ include: WASM_FNS_PATH }));
    const { context, emitted } = mockPluginContext();
    const result = await plugin.transform.call(context, source, WASM_FNS_PATH);

    // Stand in for what Rollup's chunk-render step does at the end of a real
    // build: replace each import.meta.ROLLUP_FILE_URL_<id> placeholder with
    // a URL the fetch() call can actually resolve — a data: URL here, a
    // hashed asset path in a real build.
    let code = result!.code;
    emitted.forEach((asset, i) => {
      const dataUrl = `data:application/wasm;base64,${bytesToBase64(asset.source)}`;
      code = code.replaceAll(`import.meta.ROLLUP_FILE_URL_ref${i}`, JSON.stringify(dataUrl));
    });
    // A bare "@random-mesh/rmsl/wasm" specifier has no package context to
    // resolve against from a data: URL module, the way it would in a real
    // bundle; point it at the actual build output instead, same as a
    // consumer would resolve it via node_modules.
    code = code.replace(
      '"@random-mesh/rmsl/wasm"',
      JSON.stringify(new URL("../../dist/wasm.js", import.meta.url).href),
    );

    const mod = await importDataUrl(code);
    expect(typeof mod.brightness.invoke).toBe("function");
    expect(typeof mod.mixColours.invoke).toBe("function");

    // `uniform("vec3")` inside the fixture gets the first auto slot, _rmsl_u0.
    // An aggregate (vec3) root reads back wrapped in { value }, same as
    // compileWasm's own documented aggregate-result shape.
    expect(mod.brightness.invoke({ uniforms: { _rmsl_u0: [1, 2, 3] } })).toEqual({ value: [0.5, 1, 1.5] });
    expect(mod.mixColours.invoke({ params: { a: [0, 0, 0], b: [1, 1, 1], t: 0.5 } })).toEqual({
      value: [0.5, 0.5, 0.5],
    });
  });

  it("leaves non-matching modules alone", async () => {
    const plugin = asPlugin(precompileWasm({ include: "fixtures/wasm-fns.ts" }));
    const { context } = mockPluginContext();
    const result = await plugin.transform.call(context, source, "/elsewhere/other.ts");
    expect(result).toBeNull();
  });

  it("throws when the module has no code export", async () => {
    const plugin = asPlugin(precompileWasm({ include: "no-code.ts" }));
    const { context } = mockPluginContext();
    const id = `${WASM_FNS_PATH.replace("wasm-fns.ts", "")}no-code.ts`;
    await expect(plugin.transform.call(context, "export const x = 1;\n", id)).rejects.toThrow(
      /must export __RMSL_WASM_CODE/,
    );
  });

  it("throws when a code map value has no bytes", async () => {
    const plugin = asPlugin(precompileWasm({ include: "bad-code.ts" }));
    const { context } = mockPluginContext();
    const id = `${WASM_FNS_PATH.replace("wasm-fns.ts", "")}bad-code.ts`;
    await expect(
      plugin.transform.call(context, "export const __RMSL_WASM_CODE = { f: { notBytes: true } };\n", id),
    ).rejects.toThrow(/must be compileWasmFn\(\) output/);
  });

  it("reads the code map from a custom export name", async () => {
    const plugin = asPlugin(precompileWasm({ include: "custom-code.ts", codeExport: "WASM_FNS" }));
    const { context } = mockPluginContext();
    const id = `${WASM_FNS_PATH.replace("wasm-fns.ts", "")}custom-code.ts`;
    const code = source.replaceAll("__RMSL_WASM_CODE", "WASM_FNS");
    const result = await plugin.transform.call(context, code, id);

    expect(result).not.toBeNull();
    expect(result!.code).toContain("export const brightness = instantiateWasm(");
  });
});
