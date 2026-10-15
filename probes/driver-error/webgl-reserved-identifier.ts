/**
 * Asks Chromium's WebGL 2 whether it compiles a vertex shader that declares a
 * uniform named with the `webgl_` or `_webgl_` prefix, or with a name of 1025
 * characters, against one of 1024. It runs on SwiftShader, as the
 * repository's `gpuPage` launches it, and on the GPU through ANGLE's Metal
 * backend.
 *
 * Run from the repository root: `node probes/driver-error/webgl-reserved-identifier.ts`.
 * It writes `webgl-reserved-identifier.out.json` beside itself.
 */
import { writeFileSync } from "node:fs";
import { chromium, type Browser } from "playwright";

const NAMES: Record<string, string> = {
  "webgl_ prefix": "webgl_flag",
  "_webgl_ prefix": "_webgl_flag",
  "1024 characters": "a".repeat(1024),
  "1025 characters": "a".repeat(1025),
};

const measure = async (browser: Browser) => {
  const page = await browser.newPage();
  await page.goto("about:blank");
  const measured = await page.evaluate((names: Record<string, string>) => {
    const gl = document.createElement("canvas").getContext("webgl2")!;
    const info = gl.getExtension("WEBGL_debug_renderer_info");
    const results: Record<string, unknown> = {};
    for (const [label, name] of Object.entries(names)) {
      const shader = gl.createShader(gl.VERTEX_SHADER)!;
      gl.shaderSource(shader, `#version 300 es\nuniform float ${name};\nvoid main() { gl_Position = vec4(${name}); }`);
      gl.compileShader(shader);
      const log = gl.getShaderInfoLog(shader) ?? "";
      results[label] = {
        compiles: gl.getShaderParameter(shader, gl.COMPILE_STATUS),
        log: log.replace(/a{40,}/g, (run) => `<${run.length} a's>`).replace(/\u0000/g, ""),
      };
    }
    return {
      results,
      renderer: info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
      userAgent: navigator.userAgent,
    };
  }, NAMES);
  const out = { chromium: browser.version(), ...measured };
  await browser.close();
  return out;
};

const out = {
  environment: { node: process.version, platform: `${process.platform} ${process.arch}` },
  swiftshader: await measure(await chromium.launch({ args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"] })),
  metal: await measure(await chromium.launch({ channel: "chromium", args: ["--use-angle=metal"] })),
};
writeFileSync(new URL("./webgl-reserved-identifier.out.json", import.meta.url), JSON.stringify(out, null, 2) + "\n");
console.log(JSON.stringify(out, null, 2));
