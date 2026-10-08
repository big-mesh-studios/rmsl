/** The pieces of WebGL state a call can ask `GlStateKeeper` to keep, as bits of one mask. */
export const GlState = {
  /** The draw and read framebuffers and the renderbuffer. */
  framebuffers: 1 << 0,
  viewport: 1 << 1,
  clearColor: 1 << 2,
  /** The depth test switch and the depth mask. */
  depth: 1 << 3,
  /** The blend switch and the blend function. */
  blend: 1 << 4,
  /** The cull switch and the culled face. */
  cull: 1 << 5,
  program: 1 << 6,
  vertexArray: 1 << 7,
  arrayBuffer: 1 << 8,
  pixelPackBuffer: 1 << 9,
  /** The active texture unit and the textures bound to it. */
  activeTexture: 1 << 10,
  /** The unpack parameters and the pixel-unpack buffer. */
  unpack: 1 << 11,
  /** The pack parameters. */
  pack: 1 << 12,
  /**
   * The scissor test, the colour mask, the blend equation, the depth function,
   * the depth range, the front face, and the stencil, discard, polygon offset,
   * coverage and dithering switches.
   */
  raster: 1 << 13,
} as const;

/**
 * The switches a draw reads, with the state each draw of rmsl's sets them to:
 * off, as a fresh context has all but dithering, and as a WGSL draw has them.
 */
const SWITCHES = [
  ["SCISSOR_TEST", false],
  ["STENCIL_TEST", false],
  ["RASTERIZER_DISCARD", false],
  ["POLYGON_OFFSET_FILL", false],
  ["SAMPLE_ALPHA_TO_COVERAGE", false],
  ["SAMPLE_COVERAGE", false],
  ["DITHER", false],
] as const;

/** The unpack parameters, with the value each upload of rmsl's reads: tight rows, read as they are. */
const UNPACK = [
  ["UNPACK_ALIGNMENT", 1],
  ["UNPACK_FLIP_Y_WEBGL", 0],
  ["UNPACK_PREMULTIPLY_ALPHA_WEBGL", 0],
  ["UNPACK_ROW_LENGTH", 0],
  ["UNPACK_IMAGE_HEIGHT", 0],
  ["UNPACK_SKIP_PIXELS", 0],
  ["UNPACK_SKIP_ROWS", 0],
  ["UNPACK_SKIP_IMAGES", 0],
] as const;

/** The pack parameters, with the value each readback of rmsl's reads: tight rows from the first pixel. */
const PACK = [
  ["PACK_ALIGNMENT", 1],
  ["PACK_ROW_LENGTH", 0],
  ["PACK_SKIP_PIXELS", 0],
  ["PACK_SKIP_ROWS", 0],
] as const;

/**
 * Sets the state a texture upload reads: tight rows read as they are, in the
 * browser's colour space, from the data the call gives rather than a buffer.
 */
export function setUnpackState(gl: WebGL2RenderingContext): void {
  gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, null);
  for (const [name, value] of UNPACK) gl.pixelStorei(gl[name], value);
  gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.BROWSER_DEFAULT_WEBGL);
}

/** Sets the state a readback into an array reads: tight rows from the first pixel, into the array. */
export function setPackState(gl: WebGL2RenderingContext): void {
  for (const [name, value] of PACK) gl.pixelStorei(gl[name], value);
}

/** Sets the state a clear and a draw read that rmsl never changes per draw, as a WGSL draw has it. */
export function setRasterState(gl: WebGL2RenderingContext): void {
  for (const [name, on] of SWITCHES) switchTo(gl, gl[name], on);
  gl.colorMask(true, true, true, true);
  gl.blendEquation(gl.FUNC_ADD);
  gl.depthFunc(gl.LESS);
  gl.depthRange(0, 1);
  gl.frontFace(gl.CCW);
}

/**
 * Sets the value a vertex attribute with no data reads at `location` to a
 * fresh context's, 0, 0, 0, 1. `state` keeps the value it had.
 */
export function setAttributeValue(gl: WebGL2RenderingContext, location: number, state: GlStateKeeper | null): void {
  state?.keepAttributeValue(location);
  gl.vertexAttrib4f(location, 0, 0, 0, 1);
}

/** The draw buffers of a canvas that draws into its back buffer, filled on first use. */
const backBuffer: number[] = [];

/**
 * Binds the canvas's framebuffer and has it draw into its back buffer, which
 * an application may have turned off. `state` keeps the draw buffer it had.
 */
export function drawToCanvas(gl: WebGL2RenderingContext, state: GlStateKeeper | null): void {
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  state?.keepCanvasDrawBuffer();
  backBuffer[0] = gl.BACK;
  gl.drawBuffers(backBuffer);
}

/**
 * Saves the state of a WebGL 2 context that a call is about to change, and
 * puts it back when the call ends. Calls nest: a call made inside another
 * saves only what the outer one has not, and the outermost `end` puts it all
 * back.
 */
export class GlStateKeeper {
  private depth = 0;
  private kept = 0;
  private drawFramebuffer: WebGLFramebuffer | null = null;
  private readFramebuffer: WebGLFramebuffer | null = null;
  private renderbuffer: WebGLRenderbuffer | null = null;
  private viewport: Int32Array | null = null;
  private clearColor: Float32Array | null = null;
  private depthTest = false;
  private depthMask = true;
  private blend = false;
  private blendSrcRgb = 0;
  private blendDstRgb = 0;
  private blendSrcAlpha = 0;
  private blendDstAlpha = 0;
  private cull = false;
  private cullFace = 0;
  private program: WebGLProgram | null = null;
  private vertexArray: WebGLVertexArrayObject | null = null;
  private arrayBuffer: WebGLBuffer | null = null;
  private pixelPackBuffer: WebGLBuffer | null = null;
  private activeTexture = 0;
  private readonly unpackValues: unknown[] = [];
  private unpackColorspace = 0;
  private pixelUnpackBuffer: WebGLBuffer | null = null;
  private readonly packValues: unknown[] = [];
  private readonly switches: boolean[] = [];
  private frontFace = 0;
  private canvasDrawBuffer = 0;
  private keptCanvasDrawBuffer = false;
  private readonly drawBufferList: number[] = [];
  private colorMask: boolean[] | null = null;
  private blendEquationRgb = 0;
  private blendEquationAlpha = 0;
  private depthFunc = 0;
  private depthRange: Float32Array | null = null;
  /** The attribute locations whose values are saved, and the value each had. */
  private readonly attributeLocations: number[] = [];
  private readonly attributeValues: (Float32Array | Int32Array | Uint32Array)[] = [];
  /** The units whose bindings are saved, and the 2D and 3D texture each had. */
  private readonly units: number[] = [];
  private readonly textures2D: (WebGLTexture | null)[] = [];
  private readonly textures3D: (WebGLTexture | null)[] = [];

  constructor(private readonly gl: WebGL2RenderingContext) {}

  /** Starts a call that changes the `pieces` of state, a mask of `GlState` bits. */
  begin(pieces: number): void {
    this.depth++;
    this.keep(pieces);
  }

  /** Saves the `pieces` of state not saved yet in the outermost call. */
  keep(pieces: number): void {
    const fresh = pieces & ~this.kept;
    if (fresh === 0 || this.depth === 0) return;
    this.kept |= fresh;
    const gl = this.gl;
    if (fresh & GlState.framebuffers) {
      this.drawFramebuffer = gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING);
      this.readFramebuffer = gl.getParameter(gl.READ_FRAMEBUFFER_BINDING);
      this.renderbuffer = gl.getParameter(gl.RENDERBUFFER_BINDING);
    }
    if (fresh & GlState.viewport) this.viewport = gl.getParameter(gl.VIEWPORT);
    if (fresh & GlState.clearColor) this.clearColor = gl.getParameter(gl.COLOR_CLEAR_VALUE);
    if (fresh & GlState.depth) {
      this.depthTest = gl.getParameter(gl.DEPTH_TEST);
      this.depthMask = gl.getParameter(gl.DEPTH_WRITEMASK);
    }
    if (fresh & GlState.blend) {
      this.blend = gl.getParameter(gl.BLEND);
      this.blendSrcRgb = gl.getParameter(gl.BLEND_SRC_RGB);
      this.blendDstRgb = gl.getParameter(gl.BLEND_DST_RGB);
      this.blendSrcAlpha = gl.getParameter(gl.BLEND_SRC_ALPHA);
      this.blendDstAlpha = gl.getParameter(gl.BLEND_DST_ALPHA);
    }
    if (fresh & GlState.cull) {
      this.cull = gl.getParameter(gl.CULL_FACE);
      this.cullFace = gl.getParameter(gl.CULL_FACE_MODE);
    }
    if (fresh & GlState.program) this.program = gl.getParameter(gl.CURRENT_PROGRAM);
    if (fresh & GlState.vertexArray) this.vertexArray = gl.getParameter(gl.VERTEX_ARRAY_BINDING);
    if (fresh & GlState.arrayBuffer) this.arrayBuffer = gl.getParameter(gl.ARRAY_BUFFER_BINDING);
    if (fresh & GlState.pixelPackBuffer) this.pixelPackBuffer = gl.getParameter(gl.PIXEL_PACK_BUFFER_BINDING);
    if (fresh & GlState.unpack) {
      for (let i = 0; i < UNPACK.length; i++) this.unpackValues[i] = gl.getParameter(gl[UNPACK[i]![0]]);
      this.unpackColorspace = gl.getParameter(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL);
      this.pixelUnpackBuffer = gl.getParameter(gl.PIXEL_UNPACK_BUFFER_BINDING);
    }
    if (fresh & GlState.pack) {
      for (let i = 0; i < PACK.length; i++) this.packValues[i] = gl.getParameter(gl[PACK[i]![0]]);
    }
    if (fresh & GlState.raster) {
      for (let i = 0; i < SWITCHES.length; i++) this.switches[i] = gl.getParameter(gl[SWITCHES[i]![0]]);
      this.frontFace = gl.getParameter(gl.FRONT_FACE);
      this.colorMask = gl.getParameter(gl.COLOR_WRITEMASK);
      this.blendEquationRgb = gl.getParameter(gl.BLEND_EQUATION_RGB);
      this.blendEquationAlpha = gl.getParameter(gl.BLEND_EQUATION_ALPHA);
      this.depthFunc = gl.getParameter(gl.DEPTH_FUNC);
      this.depthRange = gl.getParameter(gl.DEPTH_RANGE);
    }
    if (fresh & GlState.activeTexture) {
      this.activeTexture = gl.getParameter(gl.ACTIVE_TEXTURE);
      this.keepUnit(this.activeTexture - gl.TEXTURE0);
    }
  }

  /** Saves the textures bound to `unit`, before a call binds one there. */
  keepUnit(unit: number): void {
    if (this.depth === 0 || this.units.includes(unit)) return;
    this.keep(GlState.activeTexture);
    const gl = this.gl;
    const active = gl.getParameter(gl.ACTIVE_TEXTURE);
    gl.activeTexture(gl.TEXTURE0 + unit);
    this.units.push(unit);
    this.textures2D.push(gl.getParameter(gl.TEXTURE_BINDING_2D));
    this.textures3D.push(gl.getParameter(gl.TEXTURE_BINDING_3D));
    gl.activeTexture(active);
  }

  /** Saves the value of the vertex attribute at `location`, before a call sets it. */
  keepAttributeValue(location: number): void {
    if (this.depth === 0 || this.attributeLocations.includes(location)) return;
    this.attributeLocations.push(location);
    this.attributeValues.push(this.gl.getVertexAttrib(location, this.gl.CURRENT_VERTEX_ATTRIB));
  }

  /**
   * Saves the draw buffer of the canvas's framebuffer, which must be bound,
   * before a call sets it. The call keeps `GlState.framebuffers` from its start.
   */
  keepCanvasDrawBuffer(): void {
    if (this.depth === 0 || this.keptCanvasDrawBuffer) return;
    this.canvasDrawBuffer = this.gl.getParameter(this.gl.DRAW_BUFFER0);
    this.keptCanvasDrawBuffer = true;
  }

  /** Ends a call; the outermost one puts back every piece of state it saved. */
  end(): void {
    if (--this.depth > 0) return;
    const gl = this.gl;
    const kept = this.kept;
    if (this.keptCanvasDrawBuffer) {
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
      this.drawBufferList[0] = this.canvasDrawBuffer;
      gl.drawBuffers(this.drawBufferList);
      this.keptCanvasDrawBuffer = false;
    }
    if (kept & GlState.framebuffers) {
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.drawFramebuffer);
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.readFramebuffer);
      gl.bindRenderbuffer(gl.RENDERBUFFER, this.renderbuffer);
    }
    if (kept & GlState.viewport) {
      const [x, y, width, height] = this.viewport!;
      gl.viewport(x!, y!, width!, height!);
    }
    if (kept & GlState.clearColor) {
      const [r, g, b, a] = this.clearColor!;
      gl.clearColor(r!, g!, b!, a!);
    }
    if (kept & GlState.depth) {
      switchTo(gl, gl.DEPTH_TEST, this.depthTest);
      gl.depthMask(this.depthMask);
    }
    if (kept & GlState.blend) {
      switchTo(gl, gl.BLEND, this.blend);
      gl.blendFuncSeparate(this.blendSrcRgb, this.blendDstRgb, this.blendSrcAlpha, this.blendDstAlpha);
    }
    if (kept & GlState.cull) {
      switchTo(gl, gl.CULL_FACE, this.cull);
      gl.cullFace(this.cullFace);
    }
    if (kept & GlState.program) gl.useProgram(this.program);
    if (kept & GlState.vertexArray) gl.bindVertexArray(this.vertexArray);
    if (kept & GlState.arrayBuffer) gl.bindBuffer(gl.ARRAY_BUFFER, this.arrayBuffer);
    if (kept & GlState.pixelPackBuffer) gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pixelPackBuffer);
    if (kept & GlState.unpack) {
      for (let i = 0; i < UNPACK.length; i++) gl.pixelStorei(gl[UNPACK[i]![0]], this.unpackValues[i] as number);
      gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, this.unpackColorspace);
      gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, this.pixelUnpackBuffer);
    }
    if (kept & GlState.pack) {
      for (let i = 0; i < PACK.length; i++) gl.pixelStorei(gl[PACK[i]![0]], this.packValues[i] as number);
    }
    if (kept & GlState.raster) {
      for (let i = 0; i < SWITCHES.length; i++) switchTo(gl, gl[SWITCHES[i]![0]], this.switches[i]!);
      gl.frontFace(this.frontFace);
      const [r, g, b, a] = this.colorMask!;
      gl.colorMask(r!, g!, b!, a!);
      gl.blendEquationSeparate(this.blendEquationRgb, this.blendEquationAlpha);
      gl.depthFunc(this.depthFunc);
      gl.depthRange(this.depthRange![0]!, this.depthRange![1]!);
    }
    for (let i = 0; i < this.attributeLocations.length; i++) {
      const location = this.attributeLocations[i]!;
      const value = this.attributeValues[i]!;
      // The value keeps the type it was set with, which an integer attribute reads.
      if (value instanceof Int32Array) gl.vertexAttribI4iv(location, value);
      else if (value instanceof Uint32Array) gl.vertexAttribI4uiv(location, value);
      else gl.vertexAttrib4fv(location, value);
    }
    for (let i = 0; i < this.units.length; i++) {
      gl.activeTexture(gl.TEXTURE0 + this.units[i]!);
      gl.bindTexture(gl.TEXTURE_2D, this.textures2D[i]!);
      gl.bindTexture(gl.TEXTURE_3D, this.textures3D[i]!);
    }
    if (kept & GlState.activeTexture) gl.activeTexture(this.activeTexture);
    this.kept = 0;
    this.units.length = 0;
    this.textures2D.length = 0;
    this.textures3D.length = 0;
    this.attributeLocations.length = 0;
    this.attributeValues.length = 0;
    this.depthRange = null;
    this.viewport = null;
    this.clearColor = null;
    this.colorMask = null;
  }
}

function switchTo(gl: WebGL2RenderingContext, capability: number, on: boolean): void {
  if (on) gl.enable(capability);
  else gl.disable(capability);
}
