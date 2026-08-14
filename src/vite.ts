import { build } from "esbuild";
import type { Plugin } from "vite";

// Vite plugins that precompile rmsl node graphs at build time, so the browser
// never ships rmsl and never runs an eval:
//
// - precompileShaders: rewrite a module that default-exports the compiled GLSL /
//   WGSL strings into a JSON constant.
// - precompileJS: rewrite a module that exports compileJSFn() output (the CPU
//   code for running any shader function on the host) into plain functions.
//
// Both work the same way: the target module is bundled with esbuild for Node,
// executed once at build time via a data: URL, and replaced with the result.

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
          `${id}'s default export is not JSON-serializable; export plain data such as `
          + "compiled shader strings and slot names",
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
