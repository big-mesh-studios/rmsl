/**
 * Asks Chromium's WebGL 2, on SwiftShader as the repository's `gpuPage`
 * launches it, whether an integer vertex input draws when
 * `vertexAttribIPointer` points it at an array of the other signedness, or of
 * a narrower type. The array holds -1 at every vertex, and the vertex shader
 * writes clip w 1 where the input is not zero, so a draw that runs shows green.
 *
 * Run from the repository root: `node probes/bool-attribute/webgl-integer-pointer-signedness.ts`.
 * It writes `webgl-integer-pointer-signedness.out.json` beside itself.
 */
import { writeFileSync } from "node:fs";
import { chromium } from "playwright";

const CASES: [string, string][] = [
  ["uint", "UNSIGNED_INT"],
  ["uint", "INT"],
  ["int", "INT"],
  ["int", "UNSIGNED_INT"],
  ["uint", "UNSIGNED_BYTE"],
  ["uint", "BYTE"],
  ["uint", "UNSIGNED_SHORT"],
  ["uint", "SHORT"],
];

const browser = await chromium.launch({ args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage();
await page.goto("about:blank");
const measured = await page.evaluate((cases: [string, string][]) => {
  const results: Record<string, unknown> = {};
  let renderer = "";
  for (const [input, pointer] of cases) {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 4;
    const gl = canvas.getContext("webgl2")!;
    renderer = gl.getParameter(gl.getExtension("WEBGL_debug_renderer_info")!.UNMASKED_RENDERER_WEBGL);
    const zero = input === "uint" ? "0u" : "0";
    const vertex = `#version 300 es\nlayout(location = 0) in vec2 p;\nlayout(location = 1) in ${input} f;\nvoid main() { gl_Position = vec4(p, 0.0, f != ${zero} ? 1.0 : 0.0); }`;
    const fragment = `#version 300 es\nprecision highp float;\nout vec4 o;\nvoid main() { o = vec4(0.0, 1.0, 0.0, 1.0); }`;
    const program = gl.createProgram()!;
    for (const [type, source] of [
      [gl.VERTEX_SHADER, vertex],
      [gl.FRAGMENT_SHADER, fragment],
    ] as const) {
      const shader = gl.createShader(type)!;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      gl.attachShader(program, shader);
    }
    gl.linkProgram(program);
    gl.useProgram(program);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    // Each vertex's component is -1 in the pointer's own width, at a stride of four bytes.
    const bytes = new ArrayBuffer(12);
    const width = pointer.endsWith("BYTE") ? 1 : pointer.endsWith("SHORT") ? 2 : 4;
    for (let v = 0; v < 3; v++) for (let b = 0; b < width; b++) new Uint8Array(bytes)[v * 4 + b] = 0xff;
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, bytes, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribIPointer(1, 1, (gl as any)[pointer], 4, 0);
    gl.getError();
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const error = gl.getError();
    const pixel = new Uint8Array(4);
    gl.readPixels(1, 2, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
    results[`in ${input} <- vertexAttribIPointer ${pointer}`] = {
      error: error === 0 ? "NO_ERROR" : error === gl.INVALID_OPERATION ? "INVALID_OPERATION" : error,
      pixel: Array.from(pixel),
    };
  }
  return { results, renderer, userAgent: navigator.userAgent };
}, CASES);
const out = {
  environment: {
    node: process.version,
    platform: `${process.platform} ${process.arch}`,
    chromium: browser.version(),
    userAgent: measured.userAgent,
    renderer: measured.renderer,
  },
  results: measured.results,
};
writeFileSync(
  new URL("./webgl-integer-pointer-signedness.out.json", import.meta.url),
  JSON.stringify(out, null, 2) + "\n",
);
console.log(JSON.stringify(out, null, 2));
await browser.close();
