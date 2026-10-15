/**
 * What Chromium's WebGL 2 draws for triangle strips and fans, indexed draws, primitive restart and culling.
 *
 * Run with `node probes/index-strips-culling/webgl2.ts`. It draws every case of `cases.ts` through
 * raw WebGL 2 in the WebGL page of the test harness, which renders through SwiftShader, and prints,
 * for each probe pixel, the flat id, the vertex index and the facing of the fragment that drew it,
 * or `nothing`, with what `getError` gave after the draw. WebGL 2 has no `baseVertex`; a case that
 * names one is drawn with its `firstIndex` only, and says so.
 */
import { gpuPage, releaseGpu } from "../../src/testing/gpu.ts";
import { cases, decode, histogram, HEIGHT, WIDTH, type DrawCase } from "./cases.ts";

const VERTEX = `#version 300 es
layout(location = 0) in vec4 pos;
layout(location = 1) in uint id;
flat out uint vId;
flat out uint vVi;
void main() { gl_Position = pos; vId = id; vVi = uint(gl_VertexID); }`;

const FRAGMENT = `#version 300 es
precision highp float;
flat in uint vId;
flat in uint vVi;
out vec4 color;
void main() { color = vec4(float(vId) / 255.0, float(vVi % 256u) / 255.0, (gl_FrontFacing ? 255.0 : 128.0) / 255.0, 1.0); }`;

function inPage(drawCases: DrawCase[], vertexSource: string, fragmentSource: string, width: number, height: number) {
  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl2")!;
  const shader = (type: number, source: string) => {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, source);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "compile");
    return s;
  };
  const program = gl.createProgram()!;
  gl.attachShader(program, shader(gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER, fragmentSource));
  gl.bindAttribLocation(program, 0, "pos");
  gl.bindAttribLocation(program, 1, "id");
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) ?? "link");
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, width, height);
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  gl.viewport(0, 0, width, height);
  gl.useProgram(program);
  const modes = { "triangle-list": gl.TRIANGLES, "triangle-strip": gl.TRIANGLE_STRIP, "triangle-fan": gl.TRIANGLE_FAN };
  const types = { uint8: gl.UNSIGNED_BYTE, uint16: gl.UNSIGNED_SHORT, uint32: gl.UNSIGNED_INT };
  const defaults = {
    cullFaceEnabled: gl.isEnabled(gl.CULL_FACE),
    cullFace: gl.getParameter(gl.CULL_FACE_MODE) === gl.BACK ? "BACK" : "other",
    frontFace: gl.getParameter(gl.FRONT_FACE) === gl.CCW ? "CCW" : "CW",
  };
  const results: {
    name: string;
    error: string | null;
    note: string | null;
    pixels: Record<string, number[]>;
    image: number[];
  }[] = [];
  for (const c of drawCases) {
    const n = c.positions.length / 4;
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const pos = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, pos);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(c.positions), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 0, 0);
    const ids = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, ids);
    gl.bufferData(gl.ARRAY_BUFFER, new Uint32Array(Array.from({ length: n }, (_, k) => 10 + k)), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_INT, 0, 0);
    if (c.cullMode === "none") gl.disable(gl.CULL_FACE);
    else {
      gl.enable(gl.CULL_FACE);
      gl.cullFace(c.cullMode === "back" ? gl.BACK : gl.FRONT);
    }
    gl.frontFace(c.frontFace === "ccw" ? gl.CCW : gl.CW);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    while (gl.getError() !== gl.NO_ERROR);
    const mode = modes[c.topology];
    let note: string | null = null;
    if (c.index) {
      const array =
        c.index.kind === "uint8"
          ? new Uint8Array(c.index.values)
          : c.index.kind === "uint16"
            ? new Uint16Array(c.index.values)
            : new Uint32Array(c.index.values);
      const indexBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, array, gl.STATIC_DRAW);
      if (c.baseVertex) note = `baseVertex ${c.baseVertex} not applied: WebGL 2 has none`;
      gl.drawElements(mode, c.count, types[c.index.kind], (c.firstIndex ?? 0) * array.BYTES_PER_ELEMENT);
    } else {
      gl.drawArrays(mode, c.firstVertex ?? 0, c.count);
    }
    const code = gl.getError();
    const error =
      code === gl.NO_ERROR ? null : code === gl.INVALID_OPERATION ? "INVALID_OPERATION" : `0x${code.toString(16)}`;
    const bytes = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
    // readPixels gives the bottom row first; turn it into rows from the top, as WebGPU's copy gives.
    const image: number[] = [];
    for (let y = height - 1; y >= 0; y--) for (let i = 0; i < width * 4; i++) image.push(bytes[y * width * 4 + i]!);
    const pixels: Record<string, number[]> = {};
    for (const p of c.probes) {
      const o = (p.y * width + p.x) * 4;
      pixels[p.name] = image.slice(o, o + 4);
    }
    results.push({ name: c.name, error, note, pixels, image });
    gl.bindVertexArray(null);
    gl.deleteVertexArray(vao);
  }
  return {
    renderer: gl.getParameter(gl.RENDERER),
    version: gl.getParameter(gl.VERSION),
    maxElementIndex: gl.getParameter(gl.MAX_ELEMENT_INDEX),
    defaults,
    userAgent: navigator.userAgent,
    results,
  };
}

const page = await gpuPage();
const drawCases = cases();
const out = await page.evaluate(
  ([source, list, v, f, width, height]: [string, DrawCase[], string, string, number, number]) =>
    // eslint-disable-next-line no-new-func
    new Function("return " + source)()(list, v, f, width, height),
  [inPage.toString(), drawCases, VERTEX, FRAGMENT, WIDTH, HEIGHT] as [
    string,
    DrawCase[],
    string,
    string,
    number,
    number,
  ],
);
console.log(`# WebGL 2 — ${out.userAgent}`);
console.log(`# renderer: ${out.renderer}; ${out.version}; MAX_ELEMENT_INDEX ${out.maxElementIndex}`);
console.log(`# context defaults: ${JSON.stringify(out.defaults)}`);
for (const r of out.results) {
  console.log(`\n## ${r.name}${r.error ? `\nerror: ${r.error}` : ""}${r.note ? `\nnote: ${r.note}` : ""}`);
  for (const [name, rgba] of Object.entries(r.pixels as Record<string, number[]>))
    console.log(`  ${name}: ${decode(rgba)}`);
  console.log(`  whole image: ${JSON.stringify(histogram(r.image, WIDTH * 4))}`);
}
await releaseGpu();
