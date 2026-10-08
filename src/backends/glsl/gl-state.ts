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
  unpackAlignment: 1 << 11,
} as const;

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
  private unpackAlignment = 4;
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
    if (fresh & GlState.unpackAlignment) this.unpackAlignment = gl.getParameter(gl.UNPACK_ALIGNMENT);
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

  /** Ends a call; the outermost one puts back every piece of state it saved. */
  end(): void {
    if (--this.depth > 0) return;
    const gl = this.gl;
    const kept = this.kept;
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
    if (kept & GlState.unpackAlignment) gl.pixelStorei(gl.UNPACK_ALIGNMENT, this.unpackAlignment);
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
    this.viewport = null;
    this.clearColor = null;
  }
}

function switchTo(gl: WebGL2RenderingContext, capability: number, on: boolean): void {
  if (on) gl.enable(capability);
  else gl.disable(capability);
}
