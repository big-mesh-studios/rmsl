import { existsSync, readdirSync, statSync, utimesSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const stamp = join(root, "dist", ".build-stamp");

/** Files next to the sources that change what `vite build` produces. */
const configInputs = ["vite.config.ts", "package.json", "tsconfig.json"];

/** Test-only files under `src/`, which the library build never reads. */
const testOnly = /\.(test|test-d|bench)\.ts$/;

/**
 * The newest modification time among everything `vite build` reads: the
 * sources under `src/`, minus test-only files, plus the build's own config.
 */
function newestInputTime() {
  let newest = 0;
  const walk = (dir) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) walk(path);
      else if (!testOnly.test(entry.name)) newest = Math.max(newest, statSync(path).mtimeMs);
    }
  };
  walk(join(root, "src"));
  for (const file of configInputs) newest = Math.max(newest, statSync(join(root, file)).mtimeMs);
  return newest;
}

/**
 * Builds `dist/` when any build input is newer than the last build, and does
 * nothing otherwise. Tests and the demos' typecheck read the library through
 * its package path, which resolves to `dist/`, so they call this first rather
 * than trusting whatever an earlier build left there.
 *
 * `RMSL_SKIP_BUILD` turns it off, for runs that must not rebuild, such as
 * mutation testing, whose workers each see mutated sources at once.
 */
export async function ensureBuild() {
  if (process.env.RMSL_SKIP_BUILD) return;
  if (existsSync(stamp) && statSync(stamp).mtimeMs >= newestInputTime()) return;
  const started = new Date();
  await build({ root, configFile: join(root, "vite.config.ts"), logLevel: "warn" });
  // Dated to the build's start, so a file saved during the build still counts as newer.
  writeFileSync(stamp, "");
  utimesSync(stamp, started, started);
}

/** Vitest's `globalSetup` hook: runs once before the whole suite. */
export async function setup() {
  await ensureBuild();
}

if (process.argv[1] === fileURLToPath(import.meta.url)) await ensureBuild();
