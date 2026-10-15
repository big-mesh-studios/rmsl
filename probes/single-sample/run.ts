/**
 * Runs one probe: `node probes/single-sample/run.ts probes/single-sample/<probe>.ts`, from the repository
 * root. The probe is bundled for Node with esbuild, so its imports of the
 * repository's harnesses resolve as they do in the test suite, and it prints
 * the environment it ran in before its result. Each result in `results/` is
 * that output, written with `> probes/single-sample/results/<probe>.json`.
 */
import { build } from "esbuild";
import { mkdir } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const entry = process.argv[2];
if (!entry) throw new Error("usage: node probes/single-sample/run.ts probes/single-sample/<probe>.ts");
const out = join(dirname(resolve(entry)), ".out", basename(entry).replace(/\.ts$/, ".mjs"));
await mkdir(dirname(out), { recursive: true });
await build({ entryPoints: [entry], bundle: true, platform: "node", format: "esm", packages: "external", outfile: out, logLevel: "silent" });
await import(pathToFileURL(out).href);
