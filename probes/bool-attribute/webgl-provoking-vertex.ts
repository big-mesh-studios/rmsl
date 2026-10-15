/**
 * Asks Chromium's WebGL 2 whether it offers `WEBGL_provoking_vertex`, and
 * which vertex of a triangle a flat `uint` varying takes its value from before
 * and after `provokingVertexWEBGL(FIRST_VERTEX_CONVENTION_WEBGL)`. The first
 * vertex writes 1, the other two 2, and the fragment writes green for 1 and
 * red for 2. It runs on SwiftShader, as the repository's `gpuPage` launches
 * it, and on the GPU through ANGLE's Metal backend, as `webgpuPage` launches
 * its browser.
 *
 * Run from the repository root: `node probes/bool-attribute/webgl-provoking-vertex.ts`.
 * It writes `webgl-provoking-vertex.out.json` beside itself.
 */
import { writeFileSync } from "node:fs";
import { chromium, type Browser } from "playwright";

const measure = async (browser: Browser) => {
  const page = await browser.newPage();
  await page.goto("about:blank");
  const measured = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 4;
    const gl = canvas.getContext("webgl2", { antialias: false })!;
    const info = gl.getExtension("WEBGL_debug_renderer_info");
    const renderer = info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    const supported = gl.getSupportedExtensions()!.includes("WEBGL_provoking_vertex");
    const vertex = `#version 300 es
layout(location = 0) in vec2 p;
layout(location = 1) in uint f;
flat out uint v;
void main() { v = f; gl_Position = vec4(p, 0.0, 1.0); }`;
    const fragment = `#version 300 es
precision highp float;
flat in uint v;
out vec4 o;
void main() { o = v == 1u ? vec4(0.0, 1.0, 0.0, 1.0) : v == 2u ? vec4(1.0, 0.0, 0.0, 1.0) : vec4(0.0, 0.0, 1.0, 1.0); }`;
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
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, Uint32Array.of(1, 2, 2), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_INT, 0, 0);
    const draw = () => {
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      const pixel = new Uint8Array(4);
      gl.readPixels(1, 2, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
      return Array.from(pixel);
    };
    const results: Record<string, unknown> = { supported, default: draw() };
    const extension = gl.getExtension("WEBGL_provoking_vertex") as any;
    if (extension) {
      results.defaultConvention =
        gl.getParameter(extension.PROVOKING_VERTEX_WEBGL) === extension.FIRST_VERTEX_CONVENTION_WEBGL ? "first" : "last";
      extension.provokingVertexWEBGL(extension.FIRST_VERTEX_CONVENTION_WEBGL);
      results.firstVertexConvention = draw();
      extension.provokingVertexWEBGL(extension.LAST_VERTEX_CONVENTION_WEBGL);
      results.lastVertexConvention = draw();
    }
    results.error = gl.getError();
    return { results, renderer, userAgent: navigator.userAgent };
  });
  const out = {
    chromium: browser.version(),
    userAgent: measured.userAgent,
    renderer: measured.renderer,
    results: measured.results,
  };
  await browser.close();
  return out;
};

const out = {
  environment: { node: process.version, platform: `${process.platform} ${process.arch}` },
  legend: { green: "the first vertex's value, 1", red: "the last vertex's value, 2" },
  swiftshader: await measure(await chromium.launch({ args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"] })),
  metal: await measure(await chromium.launch({ channel: "chromium", args: ["--use-angle=metal"] })),
};
writeFileSync(new URL("./webgl-provoking-vertex.out.json", import.meta.url), JSON.stringify(out, null, 2) + "\n");
console.log(JSON.stringify(out, null, 2));
