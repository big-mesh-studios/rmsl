#!/usr/bin/env -S node --experimental-strip-types
// Builds every demo under apps/ into site/<app-name>/ with a base path that
// matches where the combined output is served from, plus a site/index.html
// linking to each one. Used by .github/workflows/deploy-demos.yml to
// publish everything in apps/ to GitHub Pages in one deploy; also runnable
// locally (`node --experimental-strip-types scripts/build-demos.mts`) to
// preview the same output with `pnpm dlx serve site`.
//
// The library itself (dist/rmsl.js) must already be built — apps depend on
// it as a workspace package — so run `pnpm build` at the repo root first.

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const repoRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const appsDir = path.join(repoRoot, "apps");
const siteDir = path.join(repoRoot, "site");

// Matches the "/<repo>/" prefix GitHub Pages serves an org/user project
// page under (https://<org>.github.io/<repo>/); override for a custom
// domain or a local preview where the site is served from "/".
const base = process.env.DEMOS_BASE ?? "/rmsl/";

const appNames = readdirSync(appsDir, { withFileTypes: true })
  .filter((entry) => entry.isDirectory() && existsSync(path.join(appsDir, entry.name, "index.html")))
  .map((entry) => entry.name)
  .sort();

rmSync(siteDir, { recursive: true, force: true });
mkdirSync(siteDir, { recursive: true });

const titleOf = (appName) => {
  const html = readFileSync(path.join(appsDir, appName, "index.html"), "utf8");
  return html.match(/<title>(.*?)<\/title>/s)?.[1]?.trim() ?? appName;
};

const links = [];
for (const appName of appNames) {
  const appBase = `${base}${appName}/`;
  const outDir = path.join(siteDir, appName);
  console.log(`Building ${appName} (base ${appBase})`);
  execFileSync("pnpm", ["exec", "vite", "build", "--base", appBase, "--outDir", outDir, "--emptyOutDir"], {
    cwd: path.join(appsDir, appName),
    stdio: "inherit",
  });
  links.push({ appName, title: titleOf(appName) });
}

writeFileSync(
  path.join(siteDir, "index.html"),
  `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>RMSL Demos</title>
  <style>
    body {
      margin: 0;
      background: #0b0b12;
      color: #d8d8e0;
      font: 15px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
      padding: 48px 24px;
    }
    main { max-width: 640px; margin: 0 auto; }
    h1 { color: #fff; font-size: 20px; margin: 0 0 8px; }
    p.intro { color: #8a8a9a; margin: 0 0 32px; }
    ul { list-style: none; margin: 0; padding: 0; }
    li + li { margin-top: 8px; }
    a {
      display: block;
      padding: 14px 16px;
      border: 1px solid #2a2a3a;
      border-radius: 8px;
      color: #9ac7ff;
      text-decoration: none;
      background: rgba(16, 16, 24, 0.82);
    }
    a:hover { border-color: #9ac7ff; }
  </style>
</head>
<body>
  <main>
    <h1>RMSL Demos</h1>
    <p class="intro">Sample apps built with <a href="https://github.com/big-mesh-studios/rmsl" style="display:inline;padding:0;border:0;background:none;">RMSL</a>, a TypeScript shader DSL.</p>
    <ul>
${links.map(({ appName, title }) => `      <li><a href="${appName}/">${title}</a></li>`).join("\n")}
    </ul>
  </main>
</body>
</html>
`,
);

console.log(`\nBuilt ${appNames.length} demo(s) into ${path.relative(repoRoot, siteDir)}/`);
