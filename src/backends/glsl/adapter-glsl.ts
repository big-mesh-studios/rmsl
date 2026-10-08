import { AttributeNode, Node, ShaderType, UniformArrayNode, UniformNode, UniformValue } from "../../core";
import { Adapter, DrawClearOptions, DrawCountOptions, slotOf, TRANSPARENT_BLACK, TypedArray } from "../adapter";
import { VertexRoot } from "../shared";
import type { CpuTextureData } from "../cpu";
import { textureImage } from "../texture-image";
import { compileGlsl, CompileGLSLOptions } from "./glsl";
import {
  AttributeKind,
  drawToCanvas,
  GlState,
  GlStateKeeper,
  setAttributeValue,
  setRasterState,
  setUnpackState,
} from "./gl-state";

type UniformInfo = { location: WebGLUniformLocation; type: number };
/** A sampler's GL texture, with the shape and internal format it was made with. */
type TextureSlot = {
  texture: WebGLTexture;
  target: number;
  unit: number;
  width: number;
  height: number;
  depth: number;
  internal: number;
};
/** A program attribute: its first location, how many locations it spans, the kind its components read, and whether the host has set its data. */
type AttributeInfo = {
  location: number;
  locations: number;
  buffer: WebGLBuffer;
  componentCount: number;
  kind: AttributeKind;
  hasData: boolean;
};

/**
 * The one shape a WebGL draw call actually varies along beyond
 * `DrawCountOptions`'s own `count`/`first`: primitive topology and how
 * many instances. Indexed draws (drawElements) aren't covered — this
 * adapter only deals in vertex buffers uploaded via setAttribute, not an
 * index buffer, so add that as its own option if a program ever needs it
 * rather than stretching this one to cover it implicitly.
 */
export interface GlslDrawOptions extends DrawCountOptions, DrawClearOptions {
  mode?: "triangles" | "triangle-strip" | "triangle-fan" | "lines" | "line-strip" | "line-loop" | "points";
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
    case gl.UNSIGNED_INT_VEC2:
      return 2;
    case gl.FLOAT_VEC3:
    case gl.INT_VEC3:
    case gl.UNSIGNED_INT_VEC3:
      return 3;
    case gl.FLOAT_VEC4:
    case gl.INT_VEC4:
    case gl.UNSIGNED_INT_VEC4:
      return 4;
    default:
      return 1;
  }
}

/** The vertex attribute locations an attribute of `type` spans: one for each column of a matrix. */
function locationCountForType(gl: WebGL2RenderingContext, type: number): number {
  switch (type) {
    case gl.FLOAT_MAT2:
      return 2;
    case gl.FLOAT_MAT3:
      return 3;
    case gl.FLOAT_MAT4:
      return 4;
    default:
      return 1;
  }
}

/** The RMSL sampler type a reflected sampler uniform has, or nothing for a uniform that is not a sampler. */
function samplerTypeOf(gl: WebGL2RenderingContext, type: number): string | undefined {
  switch (type) {
    case gl.SAMPLER_2D:
      return "sampler2D";
    case gl.SAMPLER_3D:
      return "sampler3D";
    case gl.INT_SAMPLER_2D:
      return "isampler2D";
    case gl.INT_SAMPLER_3D:
      return "isampler3D";
    case gl.UNSIGNED_INT_SAMPLER_2D:
      return "usampler2D";
    case gl.UNSIGNED_INT_SAMPLER_3D:
      return "usampler3D";
    default:
      return undefined;
  }
}

/** The format triple WebGL uploads texels of `bits` and `signed` in, by its own constants. */
function texelFormat(gl: WebGL2RenderingContext, bits: 8 | 16 | 32, signed: boolean, normalized: boolean) {
  if (normalized) return { internal: gl.RGBA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE };
  const table = {
    8: signed ? [gl.RGBA8I, gl.BYTE] : [gl.RGBA8UI, gl.UNSIGNED_BYTE],
    16: signed ? [gl.RGBA16I, gl.SHORT] : [gl.RGBA16UI, gl.UNSIGNED_SHORT],
    32: signed ? [gl.RGBA32I, gl.INT] : [gl.RGBA32UI, gl.UNSIGNED_INT],
  }[bits];
  return { internal: table[0]!, format: gl.RGBA_INTEGER, type: table[1]! };
}

function wrapMode(gl: WebGL2RenderingContext, wrap: CpuTextureData["wrapS"]): number {
  return wrap === "repeat" ? gl.REPEAT : wrap === "mirror" ? gl.MIRRORED_REPEAT : gl.CLAMP_TO_EDGE;
}

/** The `uniform*v` call a reflected uniform of `type` takes its values through, or nothing for a sampler. */
function uniformSetter(
  gl: WebGL2RenderingContext,
  type: number,
): ((location: WebGLUniformLocation, values: number[]) => void) | undefined {
  switch (type) {
    case gl.FLOAT:
      return (location, values) => gl.uniform1fv(location, values);
    case gl.FLOAT_VEC2:
      return (location, values) => gl.uniform2fv(location, values);
    case gl.FLOAT_VEC3:
      return (location, values) => gl.uniform3fv(location, values);
    case gl.FLOAT_VEC4:
      return (location, values) => gl.uniform4fv(location, values);
    case gl.INT:
    case gl.BOOL:
      return (location, values) => gl.uniform1iv(location, values);
    case gl.INT_VEC2:
    case gl.BOOL_VEC2:
      return (location, values) => gl.uniform2iv(location, values);
    case gl.INT_VEC3:
    case gl.BOOL_VEC3:
      return (location, values) => gl.uniform3iv(location, values);
    case gl.INT_VEC4:
    case gl.BOOL_VEC4:
      return (location, values) => gl.uniform4iv(location, values);
    case gl.UNSIGNED_INT:
      return (location, values) => gl.uniform1uiv(location, values);
    case gl.UNSIGNED_INT_VEC2:
      return (location, values) => gl.uniform2uiv(location, values);
    case gl.UNSIGNED_INT_VEC3:
      return (location, values) => gl.uniform3uiv(location, values);
    case gl.UNSIGNED_INT_VEC4:
      return (location, values) => gl.uniform4uiv(location, values);
    case gl.FLOAT_MAT2:
      return (location, values) => gl.uniformMatrix2fv(location, false, values);
    case gl.FLOAT_MAT3:
      return (location, values) => gl.uniformMatrix3fv(location, false, values);
    case gl.FLOAT_MAT4:
      return (location, values) => gl.uniformMatrix4fv(location, false, values);
    case gl.FLOAT_MAT2x3:
      return (location, values) => gl.uniformMatrix2x3fv(location, false, values);
    case gl.FLOAT_MAT2x4:
      return (location, values) => gl.uniformMatrix2x4fv(location, false, values);
    case gl.FLOAT_MAT3x2:
      return (location, values) => gl.uniformMatrix3x2fv(location, false, values);
    case gl.FLOAT_MAT3x4:
      return (location, values) => gl.uniformMatrix3x4fv(location, false, values);
    case gl.FLOAT_MAT4x2:
      return (location, values) => gl.uniformMatrix4x2fv(location, false, values);
    case gl.FLOAT_MAT4x3:
      return (location, values) => gl.uniformMatrix4x3fv(location, false, values);
    default:
      return undefined;
  }
}

/** Uploads `value` to a uniform, one element after another for a uniform array, a bool as 0 or 1. */
function setUniformValue(gl: WebGL2RenderingContext, slot: string, info: UniformInfo, value: unknown): void {
  const set = uniformSetter(gl, info.type);
  if (!set) throw new Error(`[RMSL] setUniform: "${slot}" is a sampler, which setTexture sets`);
  const values = (Array.isArray(value) ? value.flat(2) : [value]).map(Number);
  set(info.location, values);
}

/** The kind of the components a reflected attribute of `type` reads. */
function reflectedAttributeKind(gl: WebGL2RenderingContext, type: number): AttributeKind {
  switch (type) {
    case gl.INT:
    case gl.INT_VEC2:
    case gl.INT_VEC3:
    case gl.INT_VEC4:
      return "int";
    case gl.UNSIGNED_INT:
    case gl.UNSIGNED_INT_VEC2:
    case gl.UNSIGNED_INT_VEC3:
    case gl.UNSIGNED_INT_VEC4:
      return "uint";
    default:
      return "float";
  }
}

/** The options of `createGlsl`: the compiler's, and how the adapter shares its context. */
export interface GlslAdapterOptions extends CompileGLSLOptions {
  /**
   * Each call puts back the WebGL state it changed before it returns, for code
   * that shares the context. Without it, a call leaves the state as it set it.
   */
  preserveState?: boolean;
}

/** The state `attach` changes, a value set before it applied included. */
const ATTACH_STATE =
  GlState.program | GlState.vertexArray | GlState.arrayBuffer | GlState.activeTexture | GlState.unpack;

/** The state `draw` changes. */
const DRAW_STATE =
  GlState.framebuffers |
  GlState.viewport |
  GlState.program |
  GlState.vertexArray |
  GlState.clearColor |
  GlState.depth |
  GlState.blend |
  GlState.cull |
  GlState.raster |
  GlState.activeTexture;

/** Narrower than the base Adapter's `void | Promise<void>` on both
 * `attach` and `draw` — `getContext("webgl2")` and GL's own draw call are
 * both synchronous, unlike WGSL's device request/GPU submit. */
export interface GlslAdapter extends Adapter<never, GlslDrawOptions> {
  attach(canvas?: HTMLCanvasElement): void;
  draw(options?: GlslDrawOptions): void;
}

/**
 * Draw-only: WebGL has no compute path, so this adapter never implements
 * `compute`. Reflects a linked WebGL program's own attributes/uniforms
 * (getActiveAttrib/getActiveUniform) directly, rather than walking the RMSL
 * graph the way storage()/uniform() bindings need `compile()`'s resource
 * list on the WGSL side.
 */
export function createGlsl(
  vertexRoot: VertexRoot,
  fragmentRoot: Node<ShaderType> | readonly Node<ShaderType>[],
  options?: GlslAdapterOptions,
): GlslAdapter {
  let gl: WebGL2RenderingContext | null = null;
  let state: GlStateKeeper | null = null;
  let program: WebGLProgram | null = null;
  let vao: WebGLVertexArrayObject | null = null;
  let vertexCount = 0;
  let countSlot: string | undefined;

  const uniforms = new Map<string, UniformInfo>();
  const attributes = new Map<string, AttributeInfo>();

  // setUniform/setAttribute may be called before attach() resolves, so
  // values that arrive early are queued and replayed once the program
  // exists to look their reflected type up in.
  const pendingUniforms = new Map<string, number | number[]>();
  const pendingAttributes = new Map<string, TypedArray>();
  const pendingTextures = new Map<string, CpuTextureData>();
  /** The WebGL texture and texture unit each sampler the host has set reads. */
  const textures = new Map<string, TextureSlot>();

  function setUniform<T extends ShaderType>(uniform: UniformNode<T>, value: UniformValue<T>): void;
  function setUniform<T extends ShaderType>(uniform: UniformArrayNode<T>, value: UniformValue<T>[]): void;
  function setUniform(slot: string, value: number | number[]): void;
  function setUniform(uniform: UniformNode<ShaderType> | UniformArrayNode<ShaderType> | string, value: unknown): void {
    const slot = slotOf(uniform);
    const info = uniforms.get(slot);
    if (!gl || !program || !info) {
      pendingUniforms.set(slot, value as number | number[]);
      return;
    }
    // Another createGlsl adapter sharing this canvas's context may have
    // called useProgram since this one's attach() — a uniform location
    // is only valid against the program it came from, so this has to
    // re-bind its own before touching it, not assume it's still current.
    state?.begin(GlState.program);
    try {
      gl.useProgram(program);
      setUniformValue(gl, slot, info, value);
    } finally {
      state?.end();
    }
  }

  function setAttribute<T extends ShaderType>(attribute: AttributeNode<T>, data: TypedArray): void;
  function setAttribute(slot: string, data: TypedArray): void;
  function setAttribute(attribute: AttributeNode<ShaderType> | string, data: TypedArray): void {
    const slot = slotOf(attribute);
    const info = attributes.get(slot);
    if (!gl || !vao || !info) {
      pendingAttributes.set(slot, data);
      return;
    }
    state?.begin(GlState.vertexArray | GlState.arrayBuffer);
    try {
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, info.buffer);
      gl.enableVertexAttribArray(info.location);
      if (info.kind === "float") {
        gl.bufferData(gl.ARRAY_BUFFER, data as Float32Array, gl.STATIC_DRAW);
        gl.vertexAttribPointer(info.location, info.componentCount, gl.FLOAT, false, 0, 0);
      } else {
        const integers = info.kind === "int" ? Int32Array.from(data) : Uint32Array.from(data);
        gl.bufferData(gl.ARRAY_BUFFER, integers, gl.STATIC_DRAW);
        const type = info.kind === "int" ? gl.INT : gl.UNSIGNED_INT;
        gl.vertexAttribIPointer(info.location, info.componentCount, type, 0, 0);
      }
      info.hasData = true;
    } finally {
      state?.end();
    }
    countSlot ??= slot;
    if (slot === countSlot) vertexCount = Math.floor(data.length / info.componentCount);
  }

  function setTexture(sampler: UniformNode<ShaderType> | string, data: CpuTextureData): void {
    const slot = slotOf(sampler);
    const info = uniforms.get(slot);
    if (!gl || !program || !info) {
      pendingTextures.set(slot, data);
      return;
    }
    state?.begin(GlState.activeTexture | GlState.unpack | GlState.program);
    try {
      uploadTexture(gl, program, slot, info, data);
    } finally {
      state?.end();
    }
  }

  /** Uploads `data` into the GL texture of the sampler `slot`, and points the sampler at its unit. */
  function uploadTexture(
    gl: WebGL2RenderingContext,
    program: WebGLProgram,
    slot: string,
    info: UniformInfo,
    data: CpuTextureData,
  ): void {
    const samplerType = samplerTypeOf(gl, info.type);
    if (!samplerType) throw new Error(`[RMSL] setTexture: "${slot}" is not a sampler`);
    const image = textureImage(data, samplerType);
    const target = samplerType.endsWith("3D") ? gl.TEXTURE_3D : gl.TEXTURE_2D;
    const { internal, format, type } = texelFormat(gl, image.bits, image.signed, image.normalized);
    let held = textures.get(slot);
    // A texture of the shape the sampler already holds is written in place.
    const reuse =
      held !== undefined &&
      held.target === target &&
      held.width === image.width &&
      held.height === image.height &&
      held.depth === image.depth &&
      held.internal === internal;
    if (!reuse) {
      if (held) gl.deleteTexture(held.texture);
      held = {
        texture: gl.createTexture()!,
        target,
        unit: held?.unit ?? textures.size,
        width: image.width,
        height: image.height,
        depth: image.depth,
        internal,
      };
      textures.set(slot, held);
    }

    state?.keepUnit(held!.unit);
    gl.activeTexture(gl.TEXTURE0 + held!.unit);
    gl.bindTexture(target, held!.texture);
    setUnpackState(gl);
    if (reuse && target === gl.TEXTURE_3D) {
      gl.texSubImage3D(target, 0, 0, 0, 0, image.width, image.height, image.depth, format, type, image.texels);
    } else if (reuse) {
      gl.texSubImage2D(target, 0, 0, 0, image.width, image.height, format, type, image.texels);
    } else if (target === gl.TEXTURE_3D) {
      gl.texImage3D(target, 0, internal, image.width, image.height, image.depth, 0, format, type, image.texels);
    } else {
      gl.texImage2D(target, 0, internal, image.width, image.height, 0, format, type, image.texels);
    }
    // An integer texture cannot be filtered.
    const context = gl;
    const filter = (name: CpuTextureData["magFilter"]) =>
      image.normalized && name === "linear" ? context.LINEAR : context.NEAREST;
    gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, filter(data.magFilter));
    // A CPU target has no footprint to minify by, so both filters follow `magFilter`.
    gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, filter(data.magFilter));
    gl.texParameteri(target, gl.TEXTURE_WRAP_S, wrapMode(gl, data.wrapS));
    gl.texParameteri(target, gl.TEXTURE_WRAP_T, wrapMode(gl, data.wrapT));
    if (target === gl.TEXTURE_3D) gl.texParameteri(target, gl.TEXTURE_WRAP_R, wrapMode(gl, data.wrapR));
    gl.useProgram(program);
    gl.uniform1i(info.location, held!.unit);
  }

  /** Compiles and links the program on `gl`, reflects its inputs, and applies what the host set before. */
  function link(gl: WebGL2RenderingContext): void {
    const vertexSource = compileGlsl.vertex(vertexRoot, options);
    const fragmentSource = compileGlsl.fragment(fragmentRoot, options);

    const compile = (source: string, type: number): WebGLShader => {
      const shader = gl.createShader(type)!;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(shader);
        gl.deleteShader(shader);
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
      // WebGL names a uniform array by its first element, `name[0]`.
      if (location) uniforms.set(info.name.replace(/\[0\]$/, ""), { location, type: info.type });
    }

    vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const attributeCount = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES);
    for (let i = 0; i < attributeCount; i++) {
      const info = gl.getActiveAttrib(program, i)!;
      const location = gl.getAttribLocation(program, info.name);
      const buffer = gl.createBuffer()!;
      attributes.set(info.name, {
        location,
        locations: locationCountForType(gl, info.type),
        buffer,
        componentCount: componentCountForType(gl, info.type),
        kind: reflectedAttributeKind(gl, info.type),
        hasData: false,
      });
    }

    for (const [slot, value] of pendingUniforms) adapter.setUniform(slot, value);
    for (const [slot, data] of pendingAttributes) adapter.setAttribute(slot, data);
    for (const [slot, data] of pendingTextures) setTexture(slot, data);
    pendingUniforms.clear();
    pendingAttributes.clear();
    pendingTextures.clear();
  }

  /** Draws with this adapter's program, vertex array and textures, as `draw` asks. */
  function drawArrays(
    gl: WebGL2RenderingContext,
    program: WebGLProgram,
    vao: WebGLVertexArrayObject,
    draw: GlslDrawOptions | undefined,
  ): void {
    gl.useProgram(program);
    gl.bindVertexArray(vao);
    // The draw covers the canvas and blends, tests and culls nothing, as a WGSL draw does.
    drawToCanvas(gl, state);
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    setRasterState(gl);
    for (const info of attributes.values()) {
      if (info.hasData || info.location < 0) continue;
      for (let i = 0; i < info.locations; i++) setAttributeValue(gl, info.location + i, info.kind, state);
    }
    if (draw?.clear !== false) {
      const [r, g, b, a] = draw?.clearColor ?? TRANSPARENT_BLACK;
      gl.clearColor(r, g, b, a);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    // Another adapter sharing this context may have bound its own texture to a unit since.
    for (const held of textures.values()) {
      state?.keepUnit(held.unit);
      gl.activeTexture(gl.TEXTURE0 + held.unit);
      gl.bindTexture(held.target, held.texture);
    }
    const mode = GL_MODE[draw?.mode ?? "triangles"];
    const first = draw?.first ?? 0;
    const count = draw?.count ?? Math.max(0, vertexCount - first);
    if (draw?.instanceCount !== undefined) {
      gl.drawArraysInstanced(mode, first, count, draw.instanceCount);
    } else {
      gl.drawArrays(mode, first, count);
    }
  }

  const adapter: GlslAdapter = {
    attach(canvas) {
      const target = canvas ?? document.createElement("canvas");
      const context = target.getContext("webgl2");
      if (!context) throw new Error("[RMSL] WebGL2 is not available");
      gl = context;
      state = options?.preserveState ? new GlStateKeeper(context) : null;
      state?.begin(ATTACH_STATE);
      try {
        link(context);
      } finally {
        state?.end();
      }
    },

    setUniform,
    setAttribute,
    setTexture,

    draw(options) {
      if (!gl || !program || !vao) throw new Error("[RMSL] adapter not attached — call attach() before draw()");
      state?.begin(DRAW_STATE);
      try {
        drawArrays(gl, program, vao, options);
      } finally {
        state?.end();
      }
    },

    destroy() {
      if (!gl) return;
      for (const info of attributes.values()) gl.deleteBuffer(info.buffer);
      for (const held of textures.values()) gl.deleteTexture(held.texture);
      textures.clear();
      if (vao) gl.deleteVertexArray(vao);
    },
  };

  return adapter;
}
