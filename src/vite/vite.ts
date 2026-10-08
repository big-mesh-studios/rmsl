import { build } from "esbuild";
import { readFile } from "fs/promises";
import { resolve } from "path";
import type { Plugin } from "vite";
import wabtInit from "wabt";

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
   * `export const name = instantiateWasmRoutine(...)` in the rewritten module, and
   * has to be the same string the entry was compiled with
   * (`compileWasmFn(fn, { name, ... })`) — that name is baked into the
   * compiled bytes' own export table, so `instantiateWasmRoutine` needs it to find
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
  const cache: ModuleCache = new Map();

  return {
    name: "rmsl:precompile-shaders",
    enforce: "pre",
    async transform(code, id) {
      const filePath = normalizePath(id);
      if (!matches(filePath, options.include) || matches(filePath, options.exclude)) {
        return null;
      }

      const mod = await loadModule(this, cache, code, filePath);
      const shaders = mod.default;
      if (shaders === undefined) {
        throw new Error(`${id} must have a default export of the compiled shaders`);
      }
      const lost = notJsonData(shaders, "default");
      if (lost !== undefined) {
        throw new Error(
          `${id}'s default export is not JSON-serializable: it holds ${lost}. Export plain data such as ` +
            "compiled shader strings and slot names",
        );
      }

      return { code: `export default ${JSON.stringify(shaders)};`, map: null };
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
  const cache: ModuleCache = new Map();

  return {
    name: "rmsl:precompile-js",
    enforce: "pre",
    async transform(code, id) {
      const filePath = normalizePath(id);
      if (!matches(filePath, options.include) || matches(filePath, options.exclude)) {
        return null;
      }

      const mod = await loadModule(this, cache, code, filePath);
      const codeMap = mod[codeExport];
      if (codeMap === undefined) {
        throw new Error(`${id} must export ${codeExport} as a { name: code } map of compileJSFn() output`);
      }
      if (typeof codeMap !== "object" || codeMap === null) {
        throw new Error(`${id}'s ${codeExport} export must be a { name: code } map of compileJSFn() output`);
      }

      const exports: string[] = [];
      for (const [name, jsCode] of Object.entries(codeMap)) {
        assertIdentifier(id, codeExport, name);
        if (typeof jsCode !== "string") {
          throw new Error(`${id}'s ${codeExport} map value for ${name} must be a string of compileJSFn() output`);
        }
        exports.push(`export const ${name} = (() => {`, jsCode, "})();", "");
      }

      return { code: exports.join("\n"), map: null };
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
 * asset once at load and hands the bytes to `instantiateWasmRoutine` — imported
 * from `@random-mesh/rmsl/wasm`, the only rmsl the rewritten module ever
 * references — along with the rest of what `compileWasmFn` returned
 * (`params`/`resultType`/`textureHeapBase`/`draw`/`compute`), turning the two back
 * into a live, callable module. No eval, no graph builder or bytecode
 * emitter shipped to the browser; only the small piece of glue that marshals
 * calls in and results out.
 */
export function precompileWasm(options: PrecompileWasmOptions = {}): Plugin {
  const codeExport = options.codeExport ?? "__RMSL_WASM_CODE";
  const cache: ModuleCache = new Map();

  return {
    name: "rmsl:precompile-wasm",
    enforce: "pre",
    async transform(code, id) {
      const filePath = normalizePath(id);
      if (!matches(filePath, options.include) || matches(filePath, options.exclude)) {
        return null;
      }

      const mod = await loadModule(this, cache, code, filePath);
      const codeMap = mod[codeExport];
      if (codeMap === undefined) {
        throw new Error(`${id} must export ${codeExport} as a { name: compiled } map of compileWasmFn() output`);
      }
      if (typeof codeMap !== "object" || codeMap === null) {
        throw new Error(`${id}'s ${codeExport} export must be a { name: compiled } map of compileWasmFn() output`);
      }

      const exports: string[] = [`import { instantiateWasmRoutine } from "@random-mesh/rmsl/wasm";`, ""];
      for (const [name, compiled] of Object.entries(codeMap)) {
        assertIdentifier(id, codeExport, name);
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
          `export const ${name} = instantiateWasmRoutine(` +
            `{ bytes: new Uint8Array(_rmslBytes_${name}), ...${restJson} }, ` +
            `${JSON.stringify(name)});`,
          "",
        );
      }

      // The assets are emitted on every transform, a cached one too: each build emits the assets it references.
      return { code: exports.join("\n"), map: null };
    },
  };
}

/**
 * Loads every `.wat` (WebAssembly Text Format) file as `export default`ing
 * its compiled bytes (a `Uint8Array`), via `wabt`'s `wat2wasm`.
 *
 * Meant for a module whose WASM is entirely static — unlike rmsl's own
 * graph-driven backends, which compile bytecode at runtime for whatever
 * shader graph they're given and can't be pre-authored as `.wat`. `include`
 * defaults to every `.wat` id; `exclude` still applies on top of it.
 */
export function compileWat(options: PrecompileShadersOptions = {}): Plugin {
  let wabt: Awaited<ReturnType<typeof wabtInit>> | undefined;

  return {
    name: "rmsl:compile-wat",
    enforce: "pre",
    async load(id) {
      const filePath = normalizePath(id);
      if (!filePath.endsWith(".wat")) return null;
      if (options.include !== undefined && !matches(filePath, options.include)) return null;
      if (matches(filePath, options.exclude)) return null;

      wabt ??= await wabtInit();
      const source = await readFile(id, "utf8");
      const bytes = new Uint8Array(wabt.parseWat(filePath, source).toBinary({}).buffer);
      return `export default new Uint8Array([${bytes.join(",")}]);`;
    },
  };
}

// === Shared plumbing ===

const normalizePath = (p: string) => p.replaceAll("\\", "/");

const dirname = (p: string) => {
  const i = p.lastIndexOf("/");
  return i === -1 ? "." : p.slice(0, i);
};

// FNV-1a over a bundle, used just as a cache key for dev HMR.
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

/** The exports of a module evaluated at build time, by a hash of its bundle. */
type ModuleCache = Map<string, Record<string, unknown>>;

/**
 * The exports of the module at `filePath`, evaluated once for each bundle of
 * it and of everything it imports, so a module whose import changed is
 * evaluated again. Each file of the bundle is watched, so Vite transforms the
 * module again when one of them changes.
 */
async function loadModule(
  context: unknown,
  cache: ModuleCache,
  code: string,
  filePath: string,
): Promise<Record<string, unknown>> {
  const { text, inputs } = await bundleModule(code, filePath);
  const watch = (context as { addWatchFile?: (file: string) => void }).addWatchFile;
  for (const input of inputs) watch?.call(context, input);
  const key = hashSource(text);
  let mod = cache.get(key);
  if (mod === undefined) {
    mod = await runBundle(text);
    cache.set(key, mod);
  }
  return mod;
}

/** The shape of a name that can be written as `export const <name>`, which a rewritten module does with each key. */
const IDENTIFIER = /^[A-Za-z_$][\w$]*$/;

/** The words of that shape a module still cannot declare as a `const`. */
const RESERVED_WORDS = new Set(
  (
    "await break case catch class const continue debugger default delete do else enum export extends false " +
    "finally for function if implements import in instanceof interface let new null package private protected " +
    "public return static super switch this throw true try typeof var void while with yield arguments eval"
  ).split(" "),
);

/** Refuses a key of a code map that is not a JavaScript identifier, which would give a module that does not parse. */
function assertIdentifier(id: string, codeExport: string, name: string): void {
  if (!IDENTIFIER.test(name) || RESERVED_WORDS.has(name)) {
    throw new Error(`${id}'s ${codeExport} map names a program "${name}", which is not a JavaScript identifier`);
  }
}

/**
 * Where within `value` the first thing lies that JSON does not carry as it
 * is, described for an error, or undefined when JSON carries all of it: a
 * function, a symbol, `undefined`, a bigint, a number that is not finite, or
 * a negative zero.
 */
function notJsonData(value: unknown, path: string): string | undefined {
  if (value === null || typeof value === "string" || typeof value === "boolean") return undefined;
  if (typeof value === "number") {
    // JSON writes -0 as 0, so it loses its sign as a non-finite number loses its value.
    if (Object.is(value, -0)) return `-0 at ${path}`;
    return Number.isFinite(value) ? undefined : `${value} at ${path}`;
  }
  if (Array.isArray(value)) {
    for (let i = 0; i < value.length; i++) {
      const lost = notJsonData(value[i], `${path}[${i}]`);
      if (lost !== undefined) return lost;
    }
    return undefined;
  }
  if (typeof value === "object" && Object.getPrototypeOf(value) === Object.prototype) {
    for (const [key, v] of Object.entries(value)) {
      const lost = notJsonData(v, `${path}.${key}`);
      if (lost !== undefined) return lost;
    }
    return undefined;
  }
  const what = typeof value === "object" ? `a ${(value as object).constructor?.name ?? "object"}` : `a ${typeof value}`;
  return `${what} at ${path}`;
}

/**
 * Bundle a module for Node, with the files the bundle was made from.
 *
 * The module's imports resolve relative to the file being transformed, so a
 * shader module importing rmsl (or anything else) gets it bundled in and can be
 * executed here — which is the whole point: the compile runs once, here.
 */
async function bundleModule(code: string, filePath: string): Promise<{ text: string; inputs: string[] }> {
  try {
    const result = await build({
      stdin: {
        contents: code,
        resolveDir: dirname(filePath),
        sourcefile: filePath,
        loader: "ts",
      },
      bundle: true,
      metafile: true,
      format: "esm",
      platform: "node",
      write: false,
      logLevel: "silent",
      // a module evaluated here may itself import a `.wat` file (e.g. the
      // rasterizer) — esbuild has no built-in loader for it, so give it the
      // same wat2wasm transform `compileWat` applies under Vite.
      plugins: [
        {
          name: "rmsl:evaluate-module-wat",
          setup(pluginBuild) {
            pluginBuild.onLoad({ filter: /\.wat$/ }, async (args) => {
              const wabt = await wabtInit();
              const source = await readFile(args.path, "utf8");
              const bytes = new Uint8Array(wabt.parseWat(args.path, source).toBinary({}).buffer);
              return { contents: `export default new Uint8Array([${bytes.join(",")}]);`, loader: "js" };
            });
          },
        },
      ],
    });
    const inputs = Object.keys(result.metafile.inputs)
      .filter((input) => input !== "<stdin>")
      .map((input) => resolve(input));
    return { text: result.outputFiles[0].text, inputs };
  } catch (e) {
    if (e instanceof Error) {
      e.message = `Failed to bundle ${filePath} for build-time evaluation:\n${e.message}`;
    }
    throw e;
  }
}

/** Run a bundle, returning its exports. */
async function runBundle(bundle: string): Promise<Record<string, unknown>> {
  const dataUrl = `data:text/javascript,${encodeURIComponent(bundle)}`;
  return (await import(dataUrl)) as Record<string, unknown>;
}
