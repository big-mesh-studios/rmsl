import type wabtInit from "wabt";

type Wabt = Awaited<ReturnType<typeof wabtInit>>;

/**
 * The memory a module imports, which its shared variant imports shared: the
 * import of `env.memory` with its optional `$` name, initial pages and maximum.
 */
const MEMORY_IMPORT =
  /\(\s*import\s+"env"\s+"memory"\s+\(\s*memory\s+(?:(\$[^\s()]+)\s+)?(\d+)(?:\s+\d+)?(?:\s+shared)?\s*\)\s*\)/;

/** The pages a 32-bit memory can grow to: a shared memory links against an import whose maximum is at least its own. */
const MAX_PAGES = 65536;

/**
 * The JavaScript module a `.wat` file compiles to: its bytes as the default
 * export, and `shared`, the bytes of the same module importing its memory
 * shared, which a shared memory links against, or `undefined` when it imports
 * no memory. A module that imports a memory in another form is refused.
 */
export function watModuleSource(wabt: Wabt, path: string, source: string): string {
  const binary = (text: string) => new Uint8Array(wabt.parseWat(path, text, { threads: true }).toBinary({}).buffer);
  const bytes = (array: Uint8Array) => `new Uint8Array([${array.join(",")}])`;
  const plain = binary(source);
  const memory = MEMORY_IMPORT.exec(source);
  const importsMemory = WebAssembly.Module.imports(new WebAssembly.Module(plain)).some((i) => i.kind === "memory");
  if (importsMemory && !memory) {
    throw new Error(
      `[RMSL] ${path} imports a memory in a form other than (import "env" "memory" (memory ...)), which cannot be rewritten to import it shared`,
    );
  }
  const sharedImport = memory && `(import "env" "memory" (memory ${memory[1] ?? ""} ${memory[2]} ${MAX_PAGES} shared))`;
  const shared = sharedImport ? bytes(binary(source.replace(MEMORY_IMPORT, sharedImport))) : "undefined";
  return `export default ${bytes(plain)};\nexport const shared = ${shared};`;
}
