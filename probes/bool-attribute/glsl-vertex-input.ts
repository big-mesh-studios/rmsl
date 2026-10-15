/**
 * Asks the WebGL 2 compiler of Chromium, on SwiftShader as the repository's
 * `gpuPage` launches it, whether GLSL ES 3.00 may declare a `bool` or boolean
 * vector as a vertex input, or pass one from the vertex to the fragment stage,
 * beside the `uint` input that could carry one.
 *
 * Run from the repository root: `node probes/bool-attribute/glsl-vertex-input.ts`.
 * It writes `glsl-vertex-input.out.json` beside itself.
 */
import { writeFileSync } from "node:fs";
import { chromium } from "playwright";

const VERTEX = "VERTEX_SHADER";
const FRAGMENT = "FRAGMENT_SHADER";
const header = "#version 300 es\nprecision highp float;\n";
const cases: Record<string, [string, string]> = {
  "vertex input bool": [VERTEX, `${header}in bool f;\nvoid main() { gl_Position = vec4(f ? 1.0 : 0.0); }`],
  "vertex input bvec2": [VERTEX, `${header}in bvec2 f;\nvoid main() { gl_Position = vec4(f.x ? 1.0 : 0.0); }`],
  "vertex input bvec4": [VERTEX, `${header}in bvec4 f;\nvoid main() { gl_Position = vec4(f.x ? 1.0 : 0.0); }`],
  "vertex input uint compared with zero": [
    VERTEX,
    `${header}in uint f;\nvoid main() { gl_Position = vec4(f != 0u ? 1.0 : 0.0); }`,
  ],
  "vertex input uvec2 compared with zero": [
    VERTEX,
    `${header}in uvec2 f;\nvoid main() { gl_Position = vec4(notEqual(f, uvec2(0u)).x ? 1.0 : 0.0); }`,
  ],
  "vertex input sampler2D": [VERTEX, `${header}in sampler2D f;\nvoid main() { gl_Position = texture(f, vec2(0.0)); }`],
  "vertex output bool": [VERTEX, `${header}flat out bool b;\nvoid main() { b = true; gl_Position = vec4(0.0); }`],
  "fragment input bool": [FRAGMENT, `${header}flat in bool b;\nout vec4 o;\nvoid main() { o = vec4(b ? 1.0 : 0.0); }`],
  "vertex output uint flat": [VERTEX, `${header}flat out uint b;\nvoid main() { b = 1u; gl_Position = vec4(0.0); }`],
};

const browser = await chromium.launch({ args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage();
await page.goto("about:blank");
const page_ = await page.evaluate((cases: Record<string, [string, string]>) => {
  const gl = document.createElement("canvas").getContext("webgl2")!;
  const results: Record<string, string> = {};
  for (const [name, [stage, source]] of Object.entries(cases)) {
    const shader = gl.createShader((gl as any)[stage])!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    results[name] = gl.getShaderParameter(shader, gl.COMPILE_STATUS)
      ? "compiles"
      : (gl.getShaderInfoLog(shader) ?? "").replace(/\0/g, "").trim();
  }
  const debug = gl.getExtension("WEBGL_debug_renderer_info");
  return {
    results,
    renderer: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
    version: gl.getParameter(gl.VERSION),
    userAgent: navigator.userAgent,
  };
}, cases);
const out = {
  environment: {
    node: process.version,
    platform: `${process.platform} ${process.arch}`,
    chromium: browser.version(),
    userAgent: page_.userAgent,
    renderer: page_.renderer,
    glVersion: page_.version,
  },
  results: page_.results,
};
writeFileSync(new URL("./glsl-vertex-input.out.json", import.meta.url), JSON.stringify(out, null, 2) + "\n");
console.log(JSON.stringify(out, null, 2));
await browser.close();
