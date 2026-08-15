// Executes a `PassGraph` from `@random-mesh/rmsl/effects` on a raw WebGL2
// context: one fullscreen quad per pass, render targets sized by each pass's
// `scale`, inputs bound by the producer pass named in `pass.inputs`.
import { compileGLSL } from "@random-mesh/rmsl";
import type { PassGraph } from "@random-mesh/rmsl/effects";
import { quadVerts } from "../../shared/shader";

const FULLSCREEN_VERT = `#version 300 es
precision highp float;
in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }
`;

interface Target {
  fbo: WebGLFramebuffer;
  tex: WebGLTexture;
  w: number;
  h: number;
}

export interface BloomInput {
  scene: WebGLTexture;
  sceneWidth: number;
  sceneHeight: number;
  /** RMSL uniform slot name -> value, for the float uniforms (strength/radius/...). */
  floats: Record<string, number>;
}

export class BloomExecutor {
  private gl: WebGL2RenderingContext;
  private graph: PassGraph;
  private programs = new Map<string, WebGLProgram>();
  private targets = new Map<string, Target>();
  private quadVao: WebGLVertexArrayObject;
  private halfFloat: boolean;

  constructor(gl: WebGL2RenderingContext, graph: PassGraph) {
    this.gl = gl;
    this.graph = graph;
    this.quadVao = createQuadVao(gl);
    this.halfFloat = gl.getExtension("EXT_color_buffer_float") != null;
    for (const pass of graph.passes) {
      const frag = compileGLSL(pass.color);
      this.programs.set(pass.name, createProgram(gl, FULLSCREEN_VERT, frag, "aPos"));
    }
  }

  /**
   * Run all passes against `scene`. Returns the composite output texture (the
   * glow), which the caller adds to the scene in its present pass.
   */
  run(input: BloomInput): { tex: WebGLTexture; width: number; height: number } {
    const gl = this.gl;
    for (const pass of this.graph.passes) {
      const first = Object.entries(pass.inputs)[0];
      const firstKey = first[0];
      const firstSize = firstKey === "input"
        ? [input.sceneWidth, input.sceneHeight]
        : this.sizeOf(firstKey);
      const w = pass.size ? pass.size[0] : Math.max(1, Math.round(firstSize[0] * (pass.scale ?? 1)));
      const h = pass.size ? pass.size[1] : Math.max(1, Math.round(firstSize[1] * (pass.scale ?? 1)));

      const target = this.targetFor(pass.name, w, h);
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
      gl.viewport(0, 0, w, h);

      const program = this.programs.get(pass.name)!;
      gl.useProgram(program);

      // Map each RMSL sampler slot to its texture: external ("input") is the
      // scene; anything else names the producer pass's render target.
      const slotToTex = new Map<string, WebGLTexture>();
      for (const [key, sampler] of Object.entries(pass.inputs)) {
        const slot = (sampler as any).name as string;
        slotToTex.set(slot, key === "input" ? input.scene : this.targets.get(key)!.tex);
      }
      this.bindUniforms(program, slotToTex, w, h, input.floats);

      gl.bindVertexArray(this.quadVao);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    const out = this.targets.get(this.graph.output)!;
    return { tex: out.tex, width: out.w, height: out.h };
  }

  dispose(): void {
    for (const t of this.targets.values()) this.disposeTarget(t);
    for (const p of this.programs.values()) this.gl.deleteProgram(p);
  }

  private sizeOf(name: string): [number, number] {
    const t = this.targets.get(name);
    if (!t) throw new Error(`[bloom executor] no target for pass "${name}"`);
    return [t.w, t.h];
  }

  private targetFor(name: string, w: number, h: number): Target {
    const existing = this.targets.get(name);
    if (existing && existing.w === w && existing.h === h) return existing;
    if (existing) this.disposeTarget(existing);
    const t = this.createTarget(w, h);
    this.targets.set(name, t);
    return t;
  }

  private createTarget(w: number, h: number): Target {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    const internalFormat = this.halfFloat ? gl.RGBA16F : gl.RGBA8;
    const format = this.halfFloat ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE;
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, gl.RGBA, format, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { fbo, tex, w, h };
  }

  private disposeTarget(t: Target): void {
    this.gl.deleteFramebuffer(t.fbo);
    this.gl.deleteTexture(t.tex);
  }

  private bindUniforms(
    program: WebGLProgram,
    slotToTex: Map<string, WebGLTexture>,
    w: number,
    h: number,
    floats: Record<string, number>,
  ): void {
    const gl = this.gl;
    const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS) as number;
    let unit = 0;
    for (let i = 0; i < count; i++) {
      const info = gl.getActiveUniform(program, i);
      if (!info) continue;
      const loc = gl.getUniformLocation(program, info.name);
      if (loc === null) continue;
      if (info.type === gl.SAMPLER_2D) {
        gl.uniform1i(loc, unit);
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, slotToTex.get(info.name)!);
        unit++;
      } else if (info.type === gl.FLOAT_VEC2) {
        // screenSize: the pass's own target size in pixels.
        gl.uniform2f(loc, w, h);
      } else if (info.type === gl.FLOAT) {
        gl.uniform1f(loc, floats[info.name] ?? 1);
      }
    }
  }
}

function createQuadVao(gl: WebGL2RenderingContext): WebGLVertexArrayObject {
  const vao = gl.createVertexArray()!;
  gl.bindVertexArray(vao);
  const buf = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);
  // aPos is bound to location 0 in every fullscreen program.
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  return vao;
}

function createProgram(
  gl: WebGL2RenderingContext,
  vsSource: string,
  fsSource: string,
  attribName: string,
): WebGLProgram {
  const vs = compile(gl, vsSource, gl.VERTEX_SHADER);
  const fs = compile(gl, fsSource, gl.FRAGMENT_SHADER);
  const program = gl.createProgram()!;
  gl.bindAttribLocation(program, 0, attribName);
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    throw new Error(`[bloom executor] program link failed:\n${log}\n--- fragment ---\n${fsSource}`);
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return program;
}

function compile(gl: WebGL2RenderingContext, src: string, type: number): WebGLShader {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(s);
    gl.deleteShader(s);
    throw new Error(`[bloom executor] shader compile failed:\n${log}\n--- source ---\n${src}`);
  }
  return s;
}
