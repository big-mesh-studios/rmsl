// === GLSL adapter ===
// Draw-only: WebGL has no compute path, so this adapter never implements
// `compute`. Reflection also works differently than WGSL's: a linked
// WebGL program already exposes its own attributes/uniforms
// (getActiveAttrib/getActiveUniform), so there is nothing to reconstruct
// by walking the RMSL graph the way storage()/uniform() bindings need
// `compile()`'s resource list on the WGSL side — this just asks the GL
// context what it linked.
import { Node, ShaderType } from "../core";
import { Adapter, TypedArray } from "./adapter";
import { compileGlsl, CompileGLSLOptions } from "./glsl";
import { VertexRoot } from "./shared";

type UniformInfo = { location: WebGLUniformLocation; type: number };
type AttributeInfo = { location: number; buffer: WebGLBuffer; componentCount: number };

/**
 * The one shape a WebGL draw call actually varies along: primitive
 * topology, which vertices, and how many instances. Indexed draws
 * (drawElements) aren't covered — this adapter only deals in vertex
 * buffers uploaded via setAttribute, not an index buffer, so add that as
 * its own option if a program ever needs it rather than stretching this
 * one to cover it implicitly.
 */
export interface GlslDrawOptions {
  mode?: "triangles" | "triangle-strip" | "triangle-fan" | "lines" | "line-strip" | "line-loop" | "points";
  /** First vertex to draw. Defaults to 0. */
  first?: number;
  /** Vertices to draw. Defaults to everything the widest setAttribute call implied. */
  count?: number;
  /** Instances to draw. Omit for a plain (non-instanced) draw. */
  instanceCount?: number;
}

// Plain numeric GLenum values, not `WebGL2RenderingContext.TRIANGLES` etc. —
// that global only exists in a browser, and this module is imported by the
// main barrel, so referencing it here would throw the moment any non-browser
// context (a test runner, SSR) imports anything from the package at all.
const GL_MODE: Record<Required<GlslDrawOptions>["mode"], number> = {
  points: 0,
  lines: 1,
  "line-loop": 2,
  "line-strip": 3,
  triangles: 4,
  "triangle-strip": 5,
  "triangle-fan": 6,
};

function componentCountForType(gl: WebGL2RenderingContext, type: number): number {
  switch (type) {
    case gl.FLOAT_VEC2:
    case gl.INT_VEC2:
      return 2;
    case gl.FLOAT_VEC3:
    case gl.INT_VEC3:
      return 3;
    case gl.FLOAT_VEC4:
    case gl.INT_VEC4:
      return 4;
    default:
      return 1;
  }
}

function setUniformValue(gl: WebGL2RenderingContext, info: UniformInfo, value: number | number[]): void {
  const values = Array.isArray(value) ? value : [value];
  switch (info.type) {
    case gl.FLOAT:
      gl.uniform1f(info.location, values[0]);
      return;
    case gl.FLOAT_VEC2:
      gl.uniform2fv(info.location, values);
      return;
    case gl.FLOAT_VEC3:
      gl.uniform3fv(info.location, values);
      return;
    case gl.FLOAT_VEC4:
      gl.uniform4fv(info.location, values);
      return;
    case gl.INT:
    case gl.BOOL:
      gl.uniform1i(info.location, values[0]);
      return;
    case gl.INT_VEC2:
      gl.uniform2iv(info.location, values);
      return;
    case gl.INT_VEC3:
      gl.uniform3iv(info.location, values);
      return;
    case gl.INT_VEC4:
      gl.uniform4iv(info.location, values);
      return;
    case gl.FLOAT_MAT2:
      gl.uniformMatrix2fv(info.location, false, values);
      return;
    case gl.FLOAT_MAT3:
      gl.uniformMatrix3fv(info.location, false, values);
      return;
    case gl.FLOAT_MAT4:
      gl.uniformMatrix4fv(info.location, false, values);
      return;
    default:
      throw new Error(`[RMSL] unsupported GLSL uniform type (GLenum ${info.type})`);
  }
}

/** Narrower than the base Adapter's `void | Promise<void>` `attach` —
 * `getContext("webgl2")` is synchronous, unlike WGSL's device request. */
export interface GlslAdapter extends Adapter<never, GlslDrawOptions> {
  attach(canvas?: HTMLCanvasElement): void;
  // Unconditionally defined — GLSL is draw-only, so unlike the base
  // Adapter's optional `draw?`, createGlsl's returned object always has
  // this, synchronously (no await inside it).
  draw(options?: GlslDrawOptions): void;
}

export function createGlsl(
  vertexRoot: VertexRoot,
  fragmentRoot: Node<ShaderType> | readonly Node<ShaderType>[],
  options?: CompileGLSLOptions,
): GlslAdapter {
  let gl: WebGL2RenderingContext | null = null;
  let program: WebGLProgram | null = null;
  let vao: WebGLVertexArrayObject | null = null;
  let vertexCount = 0;

  const uniforms = new Map<string, UniformInfo>();
  const attributes = new Map<string, AttributeInfo>();

  // setUniform/setAttribute may be called before attach() resolves, so
  // values that arrive early are queued and replayed once the program
  // exists to look their reflected type up in.
  const pendingUniforms = new Map<string, number | number[]>();
  const pendingAttributes = new Map<string, TypedArray>();

  const adapter: GlslAdapter = {
    attach(canvas) {
      const target = canvas ?? document.createElement("canvas");
      const context = target.getContext("webgl2");
      if (!context) throw new Error("[RMSL] WebGL2 is not available");
      gl = context;

      const vertexSource = compileGlsl.vertex(vertexRoot, options);
      const fragmentSource = compileGlsl.fragment(fragmentRoot, options);

      const compile = (source: string, type: number): WebGLShader => {
        const shader = gl!.createShader(type)!;
        gl!.shaderSource(shader, source);
        gl!.compileShader(shader);
        if (!gl!.getShaderParameter(shader, gl!.COMPILE_STATUS)) {
          const log = gl!.getShaderInfoLog(shader);
          gl!.deleteShader(shader);
          throw new Error(`[RMSL] GLSL shader failed to compile: ${log}`);
        }
        return shader;
      };

      const vertexShader = compile(vertexSource, gl.VERTEX_SHADER);
      const fragmentShader = compile(fragmentSource, gl.FRAGMENT_SHADER);
      program = gl.createProgram()!;
      gl.attachShader(program, vertexShader);
      gl.attachShader(program, fragmentShader);
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        const log = gl.getProgramInfoLog(program);
        throw new Error(`[RMSL] GLSL program failed to link: ${log}`);
      }
      gl.useProgram(program);

      const uniformCount = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
      for (let i = 0; i < uniformCount; i++) {
        const info = gl.getActiveUniform(program, i)!;
        const location = gl.getUniformLocation(program, info.name);
        if (location) uniforms.set(info.name, { location, type: info.type });
      }

      vao = gl.createVertexArray();
      gl.bindVertexArray(vao);

      const attributeCount = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES);
      for (let i = 0; i < attributeCount; i++) {
        const info = gl.getActiveAttrib(program, i)!;
        const location = gl.getAttribLocation(program, info.name);
        const buffer = gl.createBuffer()!;
        attributes.set(info.name, { location, buffer, componentCount: componentCountForType(gl, info.type) });
      }

      for (const [slot, value] of pendingUniforms) adapter.setUniform(slot, value);
      for (const [slot, data] of pendingAttributes) adapter.setAttribute(slot, data);
      pendingUniforms.clear();
      pendingAttributes.clear();
    },

    setUniform(slot, value) {
      const info = uniforms.get(slot);
      if (!gl || !program || !info) {
        pendingUniforms.set(slot, value);
        return;
      }
      // Another createGlsl adapter sharing this canvas's context may have
      // called useProgram since this one's attach() — a uniform location
      // is only valid against the program it came from, so this has to
      // re-bind its own before touching it, not assume it's still current.
      gl.useProgram(program);
      setUniformValue(gl, info, value);
    },

    setAttribute(slot, data) {
      const info = attributes.get(slot);
      if (!gl || !vao || !info) {
        pendingAttributes.set(slot, data);
        return;
      }
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, info.buffer);
      gl.bufferData(gl.ARRAY_BUFFER, data as Float32Array, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(info.location);
      gl.vertexAttribPointer(info.location, info.componentCount, gl.FLOAT, false, 0, 0);
      vertexCount = Math.max(vertexCount, Math.floor(data.length / info.componentCount));
    },

    draw(options) {
      if (!gl || !program || !vao) throw new Error("[RMSL] adapter not attached — call attach() before draw()");
      gl.useProgram(program);
      gl.bindVertexArray(vao);
      const mode = GL_MODE[options?.mode ?? "triangles"];
      const first = options?.first ?? 0;
      const count = options?.count ?? vertexCount;
      if (options?.instanceCount !== undefined) {
        gl.drawArraysInstanced(mode, first, count, options.instanceCount);
      } else {
        gl.drawArrays(mode, first, count);
      }
    },

    destroy() {
      if (!gl) return;
      for (const info of attributes.values()) gl.deleteBuffer(info.buffer);
      if (vao) gl.deleteVertexArray(vao);
    },
  };

  return adapter;
}
