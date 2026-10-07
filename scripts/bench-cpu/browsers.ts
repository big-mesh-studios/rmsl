/**
 * Runs scripts/bench-cpu/run.ts inside Firefox and Chromium through
 * Playwright, over every report or the labels given.
 *
 * Usage: node scripts/bench-cpu/browsers.ts [label ...]
 * Firefox needs `npx playwright install firefox`. Its content sandbox is
 * turned off, since it cannot start inside another macOS sandbox.
 */
import { readFileSync, readdirSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, firefox, type BrowserType, type LaunchOptions } from "playwright";

const here = dirname(fileURLToPath(import.meta.url));
const dir = join(here, "..", "..", "reports", "bench-cpu");
const asked = process.argv.slice(2);
const files = asked.length
  ? asked.map((l) => `${l}.json`)
  : readdirSync(dir)
      .filter((f) => f.endsWith(".json"))
      .sort();
import type { Report } from "./run.ts";
const reports: Report[] = files.map((f) => JSON.parse(readFileSync(join(dir, f), "utf8")));
// The page runs JavaScript, so the runner goes in with its types stripped.
const runner = stripTypeScriptTypes(readFileSync(join(here, "run.ts"), "utf8"))
  .replace(/^export function runBench/m, "function runBench")
  .replace(/^const isMain[\s\S]*$/m, "");

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
for (const [name, type, options] of engines) {
  const browser = await type.launch(options);
  const page = await browser.newPage();
  const lines: string[] = await page.evaluate(`(() => { ${runner}\n return runBench(${JSON.stringify(reports)}); })()`);
  console.log(`== ${name} ${browser.version()}`);
  for (const line of lines) console.log(line);
  await browser.close();
}
