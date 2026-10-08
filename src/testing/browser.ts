import { readFile } from "node:fs/promises";
import { build, type Plugin } from "esbuild";
import wabtInit from "wabt";
import { gpuPage, webgpuPage } from "./gpu";

/**
 * A `readPixel(canvas, x, y)` for an entry to read its result with, through a
 * 2D canvas rather than copying the WebGPU texture: the drawing surface is
 * `bgra8unorm` in one browser and `rgba8unorm` in another, and `getImageData`
 * is in the same channel order either way.
 */
export const READ_PIXEL = `
const readPixel = (canvas, x, y) => {
  const flat = document.createElement("canvas");
  flat.width = canvas.width;
  flat.height = canvas.height;
  const context = flat.getContext("2d");
  context.drawImage(canvas, 0, 0);
  const [r, g, b, a] = context.getImageData(x, y, 1, 1).data;
  return { r, g, b, a };
};
`;

/**
 * Helpers for an entry to check what a call leaves on a WebGL 2 context:
 * `dirtyGlState(gl)` sets every piece of state rmsl touches or reads to a value
 * of the page's own, with objects of its own; `glState(gl)` reads that state, the
 * first eight texture units, the current vertex array's element buffer and
 * first eight attribute switches, the values of the first eight attributes,
 * and the canvas's draw buffer included; `changedGlState(before, after)`
 * names each piece that differs.
 */
export const GL_STATE = `
const GL_STATE_NAMES = [
  "DRAW_FRAMEBUFFER_BINDING", "READ_FRAMEBUFFER_BINDING", "RENDERBUFFER_BINDING", "CURRENT_PROGRAM",
  "VERTEX_ARRAY_BINDING", "ARRAY_BUFFER_BINDING", "ELEMENT_ARRAY_BUFFER_BINDING", "PIXEL_PACK_BUFFER_BINDING",
  "VIEWPORT", "COLOR_CLEAR_VALUE", "DEPTH_TEST", "DEPTH_WRITEMASK", "BLEND", "BLEND_SRC_RGB", "BLEND_DST_RGB",
  "BLEND_SRC_ALPHA", "BLEND_DST_ALPHA", "CULL_FACE", "CULL_FACE_MODE", "ACTIVE_TEXTURE", "UNPACK_ALIGNMENT",
  "UNPACK_FLIP_Y_WEBGL", "UNPACK_PREMULTIPLY_ALPHA_WEBGL", "UNPACK_COLORSPACE_CONVERSION_WEBGL", "UNPACK_ROW_LENGTH",
  "UNPACK_IMAGE_HEIGHT", "UNPACK_SKIP_PIXELS", "UNPACK_SKIP_ROWS", "UNPACK_SKIP_IMAGES", "PIXEL_UNPACK_BUFFER_BINDING",
  "PACK_ALIGNMENT", "PACK_ROW_LENGTH", "PACK_SKIP_PIXELS", "PACK_SKIP_ROWS", "SCISSOR_TEST", "COLOR_WRITEMASK",
  "BLEND_EQUATION_RGB", "BLEND_EQUATION_ALPHA", "DEPTH_FUNC", "FRONT_FACE", "STENCIL_TEST", "RASTERIZER_DISCARD",
  "POLYGON_OFFSET_FILL", "SAMPLE_ALPHA_TO_COVERAGE", "SAMPLE_COVERAGE", "DITHER", "DEPTH_RANGE",
];
const glState = (gl) => {
  const state = {};
  for (const name of GL_STATE_NAMES) state[name] = gl.getParameter(gl[name]);
  for (let i = 0; i < 8; i++) {
    state["VERTEX_ATTRIB_ARRAY_ENABLED@" + i] = gl.getVertexAttrib(i, gl.VERTEX_ATTRIB_ARRAY_ENABLED);
    state["CURRENT_VERTEX_ATTRIB@" + i] = gl.getVertexAttrib(i, gl.CURRENT_VERTEX_ATTRIB);
  }
  const active = gl.getParameter(gl.ACTIVE_TEXTURE);
  for (let unit = 0; unit < 8; unit++) {
    gl.activeTexture(gl.TEXTURE0 + unit);
    state["TEXTURE_BINDING_2D@" + unit] = gl.getParameter(gl.TEXTURE_BINDING_2D);
    state["TEXTURE_BINDING_3D@" + unit] = gl.getParameter(gl.TEXTURE_BINDING_3D);
  }
  gl.activeTexture(active);
  const framebuffer = gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING);
  gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
  state["DRAW_BUFFER0@canvas"] = gl.getParameter(gl.DRAW_BUFFER0);
  gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, framebuffer);
  return state;
};
const changedGlState = (before, after) =>
  Object.keys(before).filter((name) =>
    ArrayBuffer.isView(before[name]) || Array.isArray(before[name])
      ? Array.from(before[name]).join() !== Array.from(after[name]).join()
      : before[name] !== after[name],
  );
const dirtyGlState = (gl) => {
  const shader = (type, source) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, source);
    gl.compileShader(s);
    return s;
  };
  const program = gl.createProgram();
  gl.attachShader(program, shader(gl.VERTEX_SHADER, "#version 300 es\\nvoid main() { gl_Position = vec4(0.0); }"));
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER, "#version 300 es\\nprecision mediump float;\\nout vec4 o;\\nvoid main() { o = vec4(1.0); }"));
  gl.linkProgram(program);
  gl.useProgram(program);
  gl.drawBuffers([gl.NONE]);
  gl.bindFramebuffer(gl.FRAMEBUFFER, gl.createFramebuffer());
  gl.bindRenderbuffer(gl.RENDERBUFFER, gl.createRenderbuffer());
  gl.bindVertexArray(gl.createVertexArray());
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
  gl.enableVertexAttribArray(1);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bindBuffer(gl.PIXEL_PACK_BUFFER, gl.createBuffer());
  gl.viewport(1, 2, 3, 4);
  gl.clearColor(0.25, 0.5, 0.75, 0.5);
  gl.disable(gl.DEPTH_TEST);
  gl.depthMask(false);
  gl.enable(gl.BLEND);
  gl.blendFuncSeparate(gl.DST_COLOR, gl.SRC_COLOR, gl.DST_ALPHA, gl.SRC_ALPHA);
  gl.enable(gl.CULL_FACE);
  gl.cullFace(gl.FRONT_AND_BACK);
  for (let unit = 0; unit < 8; unit++) {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
    gl.bindTexture(gl.TEXTURE_3D, gl.createTexture());
  }
  gl.activeTexture(gl.TEXTURE5);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 8);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
  gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
  gl.pixelStorei(gl.UNPACK_ROW_LENGTH, 7);
  gl.pixelStorei(gl.UNPACK_IMAGE_HEIGHT, 7);
  gl.pixelStorei(gl.UNPACK_SKIP_PIXELS, 1);
  gl.pixelStorei(gl.UNPACK_SKIP_ROWS, 1);
  gl.pixelStorei(gl.UNPACK_SKIP_IMAGES, 1);
  const unpack = gl.createBuffer();
  gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, unpack);
  gl.bufferData(gl.PIXEL_UNPACK_BUFFER, 1024, gl.STATIC_DRAW);
  gl.pixelStorei(gl.PACK_ALIGNMENT, 8);
  gl.pixelStorei(gl.PACK_ROW_LENGTH, 7);
  gl.pixelStorei(gl.PACK_SKIP_PIXELS, 1);
  gl.pixelStorei(gl.PACK_SKIP_ROWS, 1);
  gl.enable(gl.SCISSOR_TEST);
  gl.scissor(0, 0, 1, 1);
  gl.colorMask(false, true, true, true);
  gl.blendEquationSeparate(gl.FUNC_SUBTRACT, gl.FUNC_REVERSE_SUBTRACT);
  gl.depthFunc(gl.GREATER);
  gl.frontFace(gl.CW);
  gl.enable(gl.STENCIL_TEST);
  gl.stencilFunc(gl.NEVER, 0, 0xff);
  gl.enable(gl.POLYGON_OFFSET_FILL);
  gl.polygonOffset(1, 1);
  gl.enable(gl.SAMPLE_ALPHA_TO_COVERAGE);
  gl.enable(gl.SAMPLE_COVERAGE);
  gl.sampleCoverage(0.5, false);
  gl.disable(gl.DITHER);
  gl.depthRange(1, 0);
  for (let i = 0; i < 8; i++) gl.vertexAttrib4f(i, 0.5, 0.5, 0.5, 0.5);
};
`;

/**
 * Compiles an imported `.wat` file to the bytes of its module, as the
 * `compileWat` Vite plugin does for the library build, so an entry can bundle
 * the WASM rasterizer.
 */
const watLoader: Plugin = {
  name: "rmsl:wat",
  setup(esbuild) {
    let wabt: Awaited<ReturnType<typeof wabtInit>> | undefined;
    esbuild.onLoad({ filter: /\.wat$/ }, async ({ path }) => {
      wabt ??= await wabtInit();
      const bytes = new Uint8Array(wabt.parseWat(path, await readFile(path, "utf8")).toBinary({}).buffer);
      return { contents: `export default new Uint8Array([${bytes.join(",")}]);`, loader: "js" };
    });
  },
};

/** Bundles a TypeScript entry for the browser, resolving its imports from `resolveDir`. */
async function bundleEntry(source: string, resolveDir: string): Promise<string> {
  const result = await build({
    stdin: { contents: source, resolveDir, loader: "ts" },
    bundle: true,
    write: false,
    format: "iife",
    platform: "browser",
    plugins: [watLoader],
    logLevel: "silent",
  });
  return result.outputFiles[0]!.text;
}

/**
 * Bundles an entry, runs it in `page`, and hands back what the async function
 * it assigned to `globalThis[entryPoint]` resolved to.
 */
async function runInPage(page: any, source: string, entryPoint: string, resolveDir: string): Promise<any> {
  const code = await bundleEntry(source, resolveDir);
  return await page.evaluate(
    async ([bundle, name]: [string, string]) => {
      // eslint-disable-next-line no-new-func
      new Function(bundle)();
      return await (globalThis as any)[name]();
    },
    [code, entryPoint] as [string, string],
  );
}

/** Runs an entry in the shared WebGPU page. */
export async function runInWebGpuPage(source: string, entryPoint: string, resolveDir: string): Promise<any> {
  return runInPage(await webgpuPage(), source, entryPoint, resolveDir);
}

/** Runs an entry in the shared WebGL page, which also serves a 2D canvas to the CPU adapters. */
export async function runInGpuPage(source: string, entryPoint: string, resolveDir: string): Promise<any> {
  return runInPage(await gpuPage(), source, entryPoint, resolveDir);
}
