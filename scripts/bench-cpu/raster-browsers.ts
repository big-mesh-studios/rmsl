/**
 * Times the draws of raster-workload.ts in Firefox and Chromium through
 * Playwright, with the library from each `dist/` directory given, and checks
 * that each library draws the pixels the first one does. In Chromium it also
 * reports what one draw allocates, from V8's sampling heap profiler.
 *
 * Usage: node scripts/bench-cpu/raster-browsers.ts [dist ...]
 * The repository's own `dist/` by default. Firefox needs
 * `npx playwright install firefox`; its content sandbox is turned off, since it
 * cannot start inside another macOS sandbox.
 */
import { readFileSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";
import { dirname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, firefox, type BrowserType, type LaunchOptions, type Page } from "playwright";
import { DRAWS, HEIGHT, MATERIALS, BATCHES, WARMUP, WIDTH } from "./raster-workload.ts";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..", "..");
const dists = process.argv.slice(2).map((d) => resolve(d));
if (dists.length === 0) dists.push(join(root, "dist"));

/** The origin the page loads from: `/lib/<i>/` is the `i`th `dist/`, `/workload.js` the shared workload. */
const ORIGIN = "http://bench.local";
const workload = stripTypeScriptTypes(readFileSync(join(here, "raster-workload.ts"), "utf8"));

/** Serves the page, the workload and each library to `page`. */
async function serve(page: Page): Promise<void> {
  await page.route(`${ORIGIN}/**`, async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/") return route.fulfill({ contentType: "text/html", body: "<!doctype html><title>bench</title>" });
    if (path === "/workload.js") return route.fulfill({ contentType: "text/javascript", body: workload });
    const lib = /^\/lib\/(\d+)\/(.+)$/.exec(path);
    if (!lib) return route.fulfill({ status: 404 });
    const dist = dists[Number(lib[1])]!;
    const file = normalize(join(dist, lib[2]!));
    if (!file.startsWith(dist)) return route.fulfill({ status: 404 });
    return route.fulfill({ contentType: "text/javascript", body: readFileSync(file, "utf8") });
  });
  await page.goto(`${ORIGIN}/`);
}

/** Builds every draw in the page, as `globalThis.draws["<lib>:<material>"]`, and says which libraries draw other pixels. */
async function prepare(page: Page): Promise<string[]> {
  return page.evaluate(async (count) => {
    const w = await import("/workload.js");
    const draws: Record<string, () => Float64Array> = ((globalThis as any).draws = {});
    const first = new Map<string, Float64Array>();
    const differing: string[] = [];
    for (let i = 0; i < count; i++) {
      const library = { scene: await import(`/lib/${i}/scene.js`), js: await import(`/lib/${i}/js.js`) };
      for (const [name, make] of w.MATERIALS) {
        const draw = w.rasterDraw(library, make);
        draws[`${i}:${name}`] = draw;
        const pixels = draw().slice();
        if (!first.has(name)) first.set(name, pixels);
        else if (!w.samePixels(pixels, first.get(name)!)) differing.push(`${name}: library ${i}`);
      }
    }
    return differing;
  }, dists.length);
}

/** Kilobytes one draw allocates in Chromium, counting only the library's code and the compiled programs. */
async function allocated(page: Page, key: string, lib: number): Promise<number> {
  const cdp = await page.context().newCDPSession(page);
  // Enabled before the draws warm up: enabling deoptimizes the code it finds running.
  await cdp.send("HeapProfiler.enable");
  await page.evaluate(
    ({ key, warm }) => {
      for (let i = 0; i < warm; i++) (globalThis as any).draws[key]();
    },
    { key, warm: WARMUP },
  );
  await cdp.send("HeapProfiler.startSampling", {
    samplingInterval: 4096,
    includeObjectsCollectedByMajorGC: true,
    includeObjectsCollectedByMinorGC: true,
  });
  await page.evaluate(
    ({ key, draws }) => {
      for (let i = 0; i < draws; i++) (globalThis as any).draws[key]();
    },
    { key, draws: DRAWS },
  );
  const { profile } = await cdp.send("HeapProfiler.stopSampling");
  await cdp.detach();
  const library = `${ORIGIN}/lib/${lib}/`;
  let bytes = 0;
  const walk = (node: any): void => {
    const url: string = node.callFrame.url;
    if (url === "" || url.startsWith(library)) bytes += node.selfSize;
    for (const child of node.children) walk(child);
  };
  walk(profile.head);
  return bytes / DRAWS / 1024;
}

const engines: [string, BrowserType, LaunchOptions][] = [
  [
    "firefox",
    firefox,
    {
      env: { ...process.env, MOZ_DISABLE_CONTENT_SANDBOX: "1" },
      firefoxUserPrefs: { "security.sandbox.content.level": 0 },
    },
  ],
  ["chromium", chromium, {}],
];
for (const [engine, type, options] of engines) {
  const browser = await type.launch(options);
  const page = await browser.newPage();
  await serve(page);
  const differing = await prepare(page);
  const columns: string[] = [];
  const rows = new Map<string, string[]>();
  for (let lib = 0; lib < dists.length; lib++) {
    columns.push(`${dists[lib]!.replace(`${root}/`, "")} ms`);
    if (engine === "chromium") columns.push("KB/draw");
    for (const [name] of MATERIALS) {
      const key = `${lib}:${name}`;
      const ms = await page.evaluate(
        async (key) => (await import("/workload.js")).timeDraw((globalThis as any).draws[key]),
        key,
      );
      const cells = [ms.toFixed(2)];
      if (engine === "chromium") cells.push((await allocated(page, key, lib)).toFixed(1));
      rows.set(name, [...(rows.get(name) ?? []), ...cells]);
    }
  }
  console.log(
    `== ${engine} ${browser.version()}, a ${WIDTH}x${HEIGHT} draw of a sphere, median of ${BATCHES} batches of ${DRAWS}`,
  );
  console.log(["program".padEnd(10), ...columns.map((c) => c.padStart(16))].join(""));
  for (const [name, cells] of rows) console.log([name.padEnd(10), ...cells.map((c) => c.padStart(16))].join(""));
  if (dists.length > 1) {
    console.log(
      differing.length
        ? `pixels differ from the first library:\n${differing.join("\n")}`
        : "every library draws the same pixels",
    );
  }
  await browser.close();
}
