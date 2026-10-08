/// <reference types="vite/client" />
import { describe, expect, it } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { precompileJS, precompileShaders, precompileWasm } from "../vite/vite";
import wasmFnsSource from "../vite/fixtures/wasm-fns.ts?raw";

type TransformResult = { code: string; map: null } | null;
type Transform = { transform: { call(context: unknown, code: string, id: string): Promise<TransformResult> } };

const FIXTURES = new URL("../vite/fixtures/", import.meta.url).pathname;

/** A Rollup plugin context with just the `emitFile` precompileWasm calls. */
function pluginContext() {
  const emitted: unknown[] = [];
  return { context: { emitFile: (asset: unknown) => `ref${emitted.push(asset) - 1}` }, emitted };
}

describe("known bugs of the tools, each failing until its fix", () => {
  /**
   * The plugins cache a result by the source of the module alone, so a module
   * whose import changed gives the result it gave before the change.
   *
   * @canon bug-a-plugin-serves-a-stale-result-after-an-import-changes
   */
  it.fails("compiles a module again when a module it imports changes", async () => {
    const dir = mkdtempSync(join(tmpdir(), "rmsl-vite-"));
    try {
      const entry = join(dir, "entry.ts");
      const source = `import value from "./value";\nexport default { value };\n`;
      writeFileSync(join(dir, "value.ts"), "export default 1;\n");
      const plugin = precompileShaders({ include: entry }) as unknown as Transform;
      expect((await plugin.transform.call({}, source, entry))!.code).toBe('export default {"value":1};');
      writeFileSync(join(dir, "value.ts"), "export default 2;\n");
      expect((await plugin.transform.call({}, source, entry))!.code).toBe('export default {"value":2};');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  /**
   * `precompileWasm` emits its assets only on the first transform of a source,
   * so a later build answers from its cache with a reference to an asset that
   * build never emitted.
   *
   * @canon bug-precompile-wasm-emits-no-asset-on-a-cached-transform
   */
  it.fails("emits the assets of a cached module again in every build", async () => {
    const id = `${FIXTURES}wasm-fns.ts`;
    const plugin = precompileWasm({ include: "fixtures/wasm-fns.ts" }) as unknown as Transform;
    const first = pluginContext();
    await plugin.transform.call(first.context, wasmFnsSource, id);
    const second = pluginContext();
    await plugin.transform.call(second.context, wasmFnsSource, id);
    expect(second.emitted).toHaveLength(first.emitted.length);
  });

  /**
   * `precompileShaders` checks only that `JSON.stringify` gives a string, which
   * it does after dropping a function, so a default export holding one ships
   * without it.
   *
   * @canon bug-precompile-shaders-drops-a-function-silently
   */
  it.fails("fails the build on a default export holding a function", async () => {
    const id = `${FIXTURES}fn-member.ts`;
    const plugin = precompileShaders({ include: "fn-member.ts" }) as unknown as Transform;
    await expect(plugin.transform.call({}, "export default { f: () => 1 };\n", id)).rejects.toThrow();
  });

  /**
   * `precompileJS` writes each key of the code map as `export const <key>`
   * without checking it is an identifier, so `my-fn` gives a module that does
   * not parse.
   *
   * @canon bug-precompile-js-writes-a-key-that-is-no-identifier
   */
  it.fails("fails the build on a program named by no identifier", async () => {
    const id = `${FIXTURES}bad-name.ts`;
    const plugin = precompileJS({ include: "bad-name.ts" }) as unknown as Transform;
    const source = `export const __RMSL_JS_CODE = { "my-fn": "return () => 1;" };\n`;
    await expect(plugin.transform.call({}, source, id)).rejects.toThrow();
  });
});
