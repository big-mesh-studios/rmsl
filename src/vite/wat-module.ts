import type wabtInit from "wabt";

type Wabt = Awaited<ReturnType<typeof wabtInit>>;

/** The memory a module imports, which its shared variant imports shared. */
const MEMORY_IMPORT = /\(import "env" "memory" \(memory (\d+)\)\)/;

/** The pages a 32-bit memory can grow to: a shared memory links against an import whose maximum is at least its own. */
const MAX_PAGES = 65536;

/**
 * The JavaScript module a `.wat` file compiles to: its bytes as the default
 * export, and `shared`, the bytes of the same module importing its memory
 * shared, which a shared memory links against, or `undefined` when it imports
 * no memory.
 */
export function watModuleSource(wabt: Wabt, path: string, source: string): string {
  const bytes = (text: string) =>
    `new Uint8Array([${new Uint8Array(wabt.parseWat(path, text, { threads: true }).toBinary({}).buffer).join(",")}])`;
  const memory = MEMORY_IMPORT.exec(source);
  const shared = memory
    ? bytes(source.replace(MEMORY_IMPORT, `(import "env" "memory" (memory ${memory[1]} ${MAX_PAGES} shared))`))
    : "undefined";
  return `export default ${bytes(source)};\nexport const shared = ${shared};`;
}
