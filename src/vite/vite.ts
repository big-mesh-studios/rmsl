import { build } from "esbuild";
import type { Plugin } from "vite";

// Vite plugins that precompile rmsl node graphs at build time, so the browser
// never ships rmsl and never runs an eval:
//
// - precompileShaders: rewrite a module that default-exports the compiled GLSL /
//   WGSL strings into a JSON constant.
// - precompileJS: rewrite a module that exports compileJSFn() output (the CPU
//   code for running any shader function on the host) into plain functions.
// - precompileWasm: rewrite a module that exports compileWasmFn() output (a
//   compiled WASM module's bytes plus the metadata a host needs to call it)
//   into `instantiateWasm(...)` calls — the compiled bytes ship as a real
//   `.wasm` asset (via Rollup's emitFile, not a string), fetched and
//   instantiated once at module load.
//
// All three bundle the target module with esbuild for Node and execute it
// once at build time via a data: URL, then replace it with the result;
// precompileWasm additionally emits one binary asset per compiled module.

export type ViteFilter = string | RegExp | Array<string | RegExp>;

export interface PrecompileShadersOptions {
  /** Modules to rewrite. A string is a normalized path suffix; a RegExp is tested against the normalized id. */
  include?: ViteFilter;
  /** Modules to leave alone, taking precedence over `include`. */
  exclude?: ViteFilter;
}

export interface PrecompileJSOptions extends PrecompileShadersOptions {
  /**
   * The named export carrying a `{ name: code }` map of compileJSFn() output.
   * Each key becomes a `export const name = (() => { code })()` in the rewritten
   * module. Defaults to `__RMSL_JS_CODE`.
   */
  codeExport?: string;
}

export interface PrecompileWasmOptions extends PrecompileShadersOptions {
  /**
   * The named export carrying a `{ name: compiled }` map, where each `compiled`
   * is one `compileWasmFn()` result. Each key becomes an
   * `export const name = instantiateWasm(...)` in the rewritten module, and
   * has to be the same string the entry was compiled with
   * (`compileWasmFn(fn, { name, ... })`) — that name is baked into the
   * compiled bytes' own export table, so `instantiateWasm` needs it to find
   * the right function inside the module. Defaults to `__RMSL_WASM_CODE`.
   */
  codeExport?: string;
}

/**
 * Rewrites every matching module to a JSON constant of its default export.
 *
 * The target module is written as a plain module whose default export is the
 * result of compiling the rmsl graph — compiled GLSL/WGSL strings and the slot
 * names a caller needs to address uniforms and varyings. It is bundled and
 * executed once at build time, then replaced with `export default <JSON>`, so
 * rmsl is never shipped and the shader sources are constants at runtime.
 */
export function precompileShaders(options: PrecompileShadersOptions = {}): Plugin {
  const cache = new Map<string, string>();

  return {
    name: "rmsl:precompile-shaders",
    enforce: "pre",
    async transform(code, id) {
      const filePath = normalizePath(id);
      if (!matches(filePath, options.include) || matches(filePath, options.exclude)) {
        return null;
      }

      const hash = hashSource(code);
      const cached = cache.get(hash);
      if (cached !== undefined) {
        return { code: cached, map: null };
      }

      const mod = await evaluateModule(code, filePath);
      const shaders = mod.default;
      if (shaders === undefined) {
        throw new Error(`${id} must have a default export of the compiled shaders`);
      }
      const serialized = JSON.stringify(shaders);
      if (serialized === undefined) {
        throw new Error(
          `${id}'s default export is not JSON-serializable; export plain data such as ` +
            "compiled shader strings and slot names",
        );
      }

      const compiled = `export default ${serialized};`;
      cache.set(hash, compiled);
      return { code: compiled, map: null };
    },
  };
}

/**
 * Rewrites every matching module to inline the CPU code it compiled.
 *
 * The target module exports a `{ name: code }` map of compileJSFn() output,
 * under the export named by `codeExport`. It is bundled and executed once at
 * build time, then rewritten to one `export const name = (() => { code })()` per
 * key — a plain callable that runs the shader function on the CPU. No eval, no
 * rmsl at runtime, and no assumptions about the call context: a caller that
 * needs to map friendly values into the compiled slots wraps the callable
 * itself.
 */
export function precompileJS(options: PrecompileJSOptions = {}): Plugin {
  const codeExport = options.codeExport ?? "__RMSL_JS_CODE";
  const cache = new Map<string, string>();

  return {
    name: "rmsl:precompile-js",
    enforce: "pre",
    async transform(code, id) {
      const filePath = normalizePath(id);
      if (!matches(filePath, options.include) || matches(filePath, options.exclude)) {
        return null;
      }

      const hash = hashSource(code);
      const cached = cache.get(hash);
      if (cached !== undefined) {
        return { code: cached, map: null };
      }

      const mod = await evaluateModule(code, filePath);
      const codeMap = mod[codeExport];
      if (codeMap === undefined) {
        throw new Error(`${id} must export ${codeExport} as a { name: code } map of compileJSFn() output`);
      }
      if (typeof codeMap !== "object" || codeMap === null) {
        throw new Error(`${id}'s ${codeExport} export must be a { name: code } map of compileJSFn() output`);
      }

      const exports: string[] = [];
      for (const [name, jsCode] of Object.entries(codeMap)) {
        if (typeof jsCode !== "string") {
          throw new Error(`${id}'s ${codeExport} map value for ${name} must be a string of compileJSFn() output`);
        }
        exports.push(`export const ${name} = (() => {`, jsCode, "})();", "");
      }

      const compiled = exports.join("\n");
      cache.set(hash, compiled);
      return { code: compiled, map: null };
    },
  };
}

/**
 * Rewrites every matching module to inline the WASM module it compiled.
 *
 * The target module exports a `{ name: compiled }` map of `compileWasmFn()`
 * results, under the export named by `codeExport`. It is bundled and executed
 * once at build time; each entry's compiled bytes are emitted as a real
 * `.wasm` asset (`this.emitFile`) rather than inlined as a string — raw
 * bytes in the build output, no string-encoding overhead, and the browser
 * can cache the asset like any other. The rewritten module fetches that
 * asset once at load and hands the bytes to `instantiateWasm` — imported
 * from `@random-mesh/rmsl/wasm`, the only rmsl the rewritten module ever
 * references — along with the rest of what `compileWasmFn` returned
 * (`params`/`resultType`/`textureHeapBase`/`draw`), turning the two back
 * into a live, callable module. No eval, no graph builder or bytecode
 * emitter shipped to the browser; only the small piece of glue that marshals
 * calls in and results out.
 */
export function precompileWasm(options: PrecompileWasmOptions = {}): Plugin {
  const codeExport = options.codeExport ?? "__RMSL_WASM_CODE";
  const cache = new Map<string, string>();

  return {
    name: "rmsl:precompile-wasm",
    enforce: "pre",
    async transform(code, id) {
      const filePath = normalizePath(id);
      if (!matches(filePath, options.include) || matches(filePath, options.exclude)) {
        return null;
      }

      const hash = hashSource(code);
      const cached = cache.get(hash);
      if (cached !== undefined) {
        return { code: cached, map: null };
      }

      const mod = await evaluateModule(code, filePath);
      const codeMap = mod[codeExport];
      if (codeMap === undefined) {
        throw new Error(`${id} must export ${codeExport} as a { name: compiled } map of compileWasmFn() output`);
      }
      if (typeof codeMap !== "object" || codeMap === null) {
        throw new Error(`${id}'s ${codeExport} export must be a { name: compiled } map of compileWasmFn() output`);
      }

      const exports: string[] = [`import { instantiateWasm } from "@random-mesh/rmsl/wasm";`, ""];
      for (const [name, compiled] of Object.entries(codeMap)) {
        if (typeof compiled !== "object" || compiled === null || !("bytes" in compiled)) {
          throw new Error(`${id}'s ${codeExport} map value for ${name} must be compileWasmFn() output`);
        }
        const { bytes, ...rest } = compiled as { bytes: unknown };
        if (!(bytes instanceof Uint8Array)) {
          throw new Error(`${id}'s ${codeExport} map value for ${name}'s bytes must be a Uint8Array`);
        }
        const restJson = JSON.stringify(rest);
        if (restJson === undefined) {
          throw new Error(`${id}'s ${codeExport} map value for ${name} is not JSON-serializable outside its bytes`);
        }
        const refId = this.emitFile({ type: "asset", name: `${name}.wasm`, source: bytes });
        exports.push(
          `const _rmslBytes_${name} = await fetch(new URL(import.meta.ROLLUP_FILE_URL_${refId}, import.meta.url)).then((r) => r.arrayBuffer());`,
          `export const ${name} = instantiateWasm(` +
            `{ bytes: new Uint8Array(_rmslBytes_${name}), ...${restJson} }, ` +
            `${JSON.stringify(name)});`,
          "",
        );
      }

      const rewritten = exports.join("\n");
      cache.set(hash, rewritten);
      return { code: rewritten, map: null };
    },
  };
}

// === Shared plumbing ===

const normalizePath = (p: string) => p.replaceAll("\\", "/");

const dirname = (p: string) => {
  const i = p.lastIndexOf("/");
  return i === -1 ? "." : p.slice(0, i);
};

// FNV-1a over the source, used just as a cache key for dev HMR.
const hashSource = (s: string) => {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0).toString(16);
};

const matches = (filePath: string, filter?: ViteFilter): boolean => {
  if (filter === undefined) return false;
  const list = Array.isArray(filter) ? filter : [filter];
  return list.some((f) => {
    if (f instanceof RegExp) {
      // A global or sticky regexp is stateful; reset it so repeated ids match
      // the same way every time.
      f.lastIndex = 0;
      return f.test(filePath);
    }
    const suffix = normalizePath(f);
    return filePath === suffix || filePath.endsWith(suffix);
  });
};

/**
 * Bundle a module for Node and run it, returning its exports.
 *
 * The module's imports resolve relative to the file being transformed, so a
 * shader module importing rmsl (or anything else) gets it bundled in and can be
 * executed here — which is the whole point: the compile runs once, here.
 */
async function evaluateModule(code: string, filePath: string): Promise<Record<string, unknown>> {
  let bundle: string;
  try {
    const result = await build({
      stdin: {
        contents: code,
        resolveDir: dirname(filePath),
        sourcefile: filePath,
        loader: "ts",
      },
      bundle: true,
      format: "esm",
      platform: "node",
      write: false,
      logLevel: "silent",
    });
    bundle = result.outputFiles[0].text;
  } catch (e) {
    if (e instanceof Error) {
      e.message = `Failed to bundle ${filePath} for build-time evaluation:\n${e.message}`;
    }
    throw e;
  }

  const dataUrl = `data:text/javascript,${encodeURIComponent(bundle)}`;
  return (await import(dataUrl)) as Record<string, unknown>;
}
