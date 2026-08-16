import { compileGLSL } from "../../rmsl";
import { Color } from "../math/Color";
import { Vector4 } from "../math/Vector4";
import type { Scene } from "../scenes/Scene";
import type { Camera } from "../cameras/Camera";
import type { Mesh } from "../objects/Mesh";
import type { BufferGeometry } from "../geometries/BufferGeometry";
import type { Texture } from "../textures/Texture";
import type { NodeMaterial, MaterialProgram } from "../materials/NodeMaterial";
import { Blending, Side } from "../materials/Material";
import {
  cameraUniformValue, isIntegerSampler, objectUniformValue, lightsSignature, toBufferView, uniformUploadValue,
  rendererUniformValue,
} from "./common";

interface ProgramEntry {
  signature: string;
  program: MaterialProgram;
  glProgram: WebGLProgram;
  uniformLocations: Map<string, WebGLUniformLocation | null>;
  attributeLocations: Map<string, number>;
}

interface GeometryBuffers {
  attributes: Map<string, WebGLBuffer>;
  index: WebGLBuffer | null;
  needsUpload: boolean;
}

/**
 * A WebGL2 renderer for `@random-mesh/rmsl/scene`: compiles a material's node
 * graph once, builds vertex buffers from its geometry, and uploads the
 * collected uniforms per draw. `render(scene, camera)` draws everything.
 */
export class WebGLRenderer {
  readonly isWebGLRenderer = true;

  canvas: HTMLCanvasElement;
  gl: WebGL2RenderingContext;

  private programs = new Map<NodeMaterial, ProgramEntry>();
  private geometryBuffers = new Map<BufferGeometry, GeometryBuffers>();
  private textures = new Map<Texture, WebGLTexture>();
  private clearColor = new Color(0, 0, 0);
  private clearAlpha = 1;
  private animationCallback: ((time: number) => void) | null = null;
  private animationHandle: number | null = null;

  constructor(canvas?: HTMLCanvasElement, options: { antialias?: boolean; depth?: boolean } = {}) {
    this.canvas = canvas ?? document.createElement("canvas");
    const gl = this.canvas.getContext("webgl2", {
      antialias: options.antialias ?? true,
      depth: options.depth ?? true,
    });
    if (!gl) {
      throw new Error("[RMSL/scene] WebGL2 is not available on this canvas");
    }
    this.gl = gl;
  }

  setClearColor(color: Color | number, alpha = 1): void {
    if (typeof color === "number") this.clearColor.setHex(color);
    else this.clearColor.copy(color);
    this.clearAlpha = alpha;
  }

  setSize(width: number, height: number): void {
    this.canvas.width = width;
    this.canvas.height = height;
  }

  setAnimationLoop(callback: ((time: number) => void) | null): void {
    this.animationCallback = callback;
    if (callback && this.animationHandle === null) {
      const loop = (now: number): void => {
        if (!this.animationCallback) {
          this.animationHandle = null;
          return;
        }
        this.animationCallback(now);
        this.animationHandle = requestAnimationFrame(loop);
      };
      this.animationHandle = requestAnimationFrame(loop);
    }
  }

  render(scene: Scene, camera: Camera): void {
    const gl = this.gl;

    scene.updateMatrixWorld(true);
    camera.updateMatrixWorld(true);
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    const [r, g, b] = this.clearColor.toArray();
    gl.clearColor(r, g, b, this.clearAlpha);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);

    scene.traverseVisible((object) => {
      if (object.isMesh) {
        const mesh = object as Mesh;
        // Give objects a chance to update per-draw state (line resolution, ...).
        mesh.onBeforeRender?.(this, scene, camera);
        this.drawMesh(mesh, scene, camera);
      }
    });
  }

  /** The drawing surface viewport: `(x, y, width, height)` in device pixels. */
  getViewport(target = new Vector4()): Vector4 {
    const gl = this.gl;
    target.set(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    return target;
  }

  private drawMesh(mesh: Mesh, scene: Scene, camera: Camera): void {
    const material = mesh.material;
    if (!(material as NodeMaterial).isNodeMaterial) return;
    this.usedTextureUnits.clear();
    const entry = this.ensureProgram(material as NodeMaterial, scene);
    const gl = this.gl;

    gl.useProgram(entry.glProgram);
    this.setRenderState(material);
    this.uploadUniforms(entry, mesh, camera);
    this.bindGeometry(entry, mesh.geometry);

    const geometry = mesh.geometry;
    if (geometry.index) {
      const indexView = toBufferView(geometry.index.array, true) as Uint16Array | Uint32Array;
      const type = indexView instanceof Uint16Array ? gl.UNSIGNED_SHORT : gl.UNSIGNED_INT;
      gl.drawElementsInstanced(gl.TRIANGLES, indexView.length, type, 0, geometry.instanceCount);
    } else {
      gl.drawArraysInstanced(gl.TRIANGLES, 0, geometry.attributes.position?.count ?? 0, geometry.instanceCount);
    }
  }

  private setRenderState(material: { side: Side; blending: Blending; depthTest: boolean; depthWrite: boolean; transparent: boolean }): void {
    const gl = this.gl;
    switch (material.side) {
      case Side.FrontSide: gl.enable(gl.CULL_FACE); gl.cullFace(gl.BACK); break;
      case Side.BackSide: gl.enable(gl.CULL_FACE); gl.cullFace(gl.FRONT); break;
      default: gl.disable(gl.CULL_FACE); break;
    }
    if (material.depthTest) gl.enable(gl.DEPTH_TEST);
    else gl.disable(gl.DEPTH_TEST);
    gl.depthMask(material.depthWrite);

    if (material.transparent || material.blending !== Blending.NormalBlending) {
      gl.enable(gl.BLEND);
      if (material.blending === Blending.AdditiveBlending) {
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
      } else {
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      }
    } else {
      gl.disable(gl.BLEND);
    }
  }

  private uploadUniforms(entry: ProgramEntry, mesh: Mesh, camera: Camera): void {
    const gl = this.gl;
    for (const binding of entry.program.uniforms) {
      const location = entry.uniformLocations.get(binding.node.name);
      if (location == null) continue;
      let value: number | number[] | Float32Array;
      if (binding.scope === "camera") {
        value = cameraUniformValue(binding.name, camera);
      } else if (binding.scope === "object") {
        value = objectUniformValue(binding.name, mesh);
      } else if (binding.scope === "renderer") {
        value = rendererUniformValue(binding.name, this.gl.drawingBufferWidth, this.gl.drawingBufferHeight);
      } else {
        value = binding.value?.({ camera, mesh }) ?? [];
      }
      if (Array.isArray(value) && value.length === 0) continue;
      this.setUniform(location, binding.node._t as string, value);
    }

    // Samplers: bind each material texture to a unit and point the sampler at it.
    for (const sampler of entry.program.samplers) {
      const texture = sampler.texture();
      const location = entry.uniformLocations.get(sampler.name);
      if (!texture || location == null) continue;
      const unit = this.bindTexture(texture, sampler.type);
      gl.uniform1i(location, unit);
    }
  }

  private setUniform(location: WebGLUniformLocation, type: string, value: number | number[] | Float32Array): void {
    const gl = this.gl;
    // Scalar uniforms arrive as a bare number (opacity, roughness, ...); the
    // vector/matrix ones as arrays. `array[0]` on a bare number is undefined,
    // which would upload NaN and blacken the surface — so the scalar is
    // uploaded directly.
    const { scalar, array } = uniformUploadValue(value);
    switch (type) {
      case "float": gl.uniform1f(location, scalar ?? array[0]); break;
      case "int": gl.uniform1i(location, scalar ?? array[0]); break;
      case "bool": gl.uniform1i(location, scalar ?? array[0]); break;
      case "vec2": gl.uniform2fv(location, array); break;
      case "vec3": gl.uniform3fv(location, array); break;
      case "vec4": gl.uniform4fv(location, array); break;
      case "ivec2": gl.uniform2iv(location, array); break;
      case "ivec3": gl.uniform3iv(location, array); break;
      case "ivec4": gl.uniform4iv(location, array); break;
      case "mat2": gl.uniformMatrix2fv(location, false, array); break;
      case "mat3": gl.uniformMatrix3fv(location, false, array); break;
      case "mat4": gl.uniformMatrix4fv(location, false, array); break;
      default:
        // Unknown types are skipped rather than guessed.
        break;
    }
  }

  private bindTexture(texture: Texture, samplerType: string): number {
    const gl = this.gl;
    const is3D = samplerType.endsWith("3D");
    const target = is3D ? gl.TEXTURE_3D : gl.TEXTURE_2D;
    const integer = isIntegerSampler(samplerType);
    // Take the unit this texture will be read from before touching it. Setting
    // a texture up binds it, and a bind always lands on the active unit — which
    // until this call belongs to the sampler bound just before. Uploading first
    // would leave that sampler reading this texture instead of its own for the
    // draw: a wrong image where the two are alike, and an invalid draw that
    // writes nothing where one is integer and the other is not.
    const unit = this.nextTextureUnit();
    gl.activeTexture(gl.TEXTURE0 + unit);
    let glTexture = this.textures.get(texture);
    if (!glTexture || texture.needsUpdate) {
      if (!glTexture) {
        glTexture = gl.createTexture()!;
        this.textures.set(texture, glTexture);
      }
      gl.bindTexture(target, glTexture);
      gl.texParameteri(target, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(target, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      if (is3D) gl.texParameteri(target, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
      // Integer textures are not filterable, so they must use NEAREST.
      const filter = integer ? gl.NEAREST : gl.LINEAR;
      gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, filter);
      gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, filter);
      const image = texture.image;
      if (ArrayBuffer.isView(image)) {
        const width = (texture as { width?: number }).width ?? 1;
        const height = (texture as { height?: number }).height ?? 1;
        if (integer) {
          const { internalFormat, type } = integerInternalFormat(gl, samplerType.startsWith("isampler"), image);
          if (is3D) {
            const depth = (texture as { depth?: number }).depth ?? 1;
            gl.texImage3D(target, 0, internalFormat, width, height, depth, 0, gl.RGBA_INTEGER, type, image as ArrayBufferView);
          } else {
            gl.texImage2D(target, 0, internalFormat, width, height, 0, gl.RGBA_INTEGER, type, image as ArrayBufferView);
          }
        } else if (is3D) {
          const depth = (texture as { depth?: number }).depth ?? 1;
          gl.texImage3D(target, 0, gl.RGBA, width, height, depth, 0, gl.RGBA, gl.UNSIGNED_BYTE, image as ArrayBufferView);
        } else {
          gl.texImage2D(target, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, image as ArrayBufferView);
        }
      } else if (image != null && !is3D && !integer) {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image as TexImageSource);
      }
      texture.needsUpdate = false;
    }
    gl.bindTexture(target, glTexture);
    return unit;
  }

  private nextTextureUnit(): number {
    const gl = this.gl;
    const units = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS) as number;
    for (let i = 0; i < units; i++) {
      if (this.usedTextureUnits.has(i)) continue;
      this.usedTextureUnits.add(i);
      return i;
    }
    return 0;
  }

  private usedTextureUnits = new Set<number>();

  private ensureProgram(material: NodeMaterial, scene: Scene): ProgramEntry {
    let entry = this.programs.get(material);
    const signature = lightsSignature(scene);
    if (entry && entry.signature === signature && !material.needsUpdate) {
      return entry;
    }

    const program = material.build(scene);
    const gl = this.gl;

    const vertexShader = this.compileShader(compileGLSL.vertex(program.vertexRoot), gl.VERTEX_SHADER);
    const fragmentShader = this.compileShader(compileGLSL.fragment(program.fragmentRoot), gl.FRAGMENT_SHADER);
    const glProgram = gl.createProgram()!;
    gl.attachShader(glProgram, vertexShader);
    gl.attachShader(glProgram, fragmentShader);
    gl.linkProgram(glProgram);
    if (!gl.getProgramParameter(glProgram, gl.LINK_STATUS)) {
      throw new Error(`[RMSL/scene] program link failed:\n${gl.getProgramInfoLog(glProgram)}`);
    }

    const uniformLocations = new Map<string, WebGLUniformLocation | null>();
    for (const binding of program.uniforms) {
      uniformLocations.set(binding.node.name, gl.getUniformLocation(glProgram, binding.node.name));
    }
    for (const sampler of program.samplers) {
      uniformLocations.set(sampler.name, gl.getUniformLocation(glProgram, sampler.name));
    }

    const attributeLocations = new Map<string, number>();
    for (const attribute of program.attributes) {
      attributeLocations.set(attribute.node.name, gl.getAttribLocation(glProgram, attribute.node.name));
    }

    entry = { signature, program, glProgram, uniformLocations, attributeLocations };
    this.programs.set(material, entry);
    material.needsUpdate = false;
    return entry;
  }

  private compileShader(source: string, type: number): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(`[RMSL/scene] shader compile failed:\n${gl.getShaderInfoLog(shader)}\n---\n${source}`);
    }
    return shader;
  }

  private bindGeometry(entry: ProgramEntry, geometry: BufferGeometry): void {
    const gl = this.gl;
    let buffers = this.geometryBuffers.get(geometry);
    if (!buffers) {
      buffers = { attributes: new Map(), index: null, needsUpload: true };
      this.geometryBuffers.set(geometry, buffers);
    }

    const needsUpload = buffers.needsUpload
      || Object.values(geometry.attributes).some((a) => a.needsUpdate);

    for (const attribute of entry.program.attributes) {
      const geometryAttribute = geometry.attributes[attribute.name];
      const location = entry.attributeLocations.get(attribute.node.name);
      if (!geometryAttribute || location == null) continue;

      let buffer = buffers.attributes.get(attribute.name);
      if (!buffer) {
        buffer = gl.createBuffer()!;
        buffers.attributes.set(attribute.name, buffer);
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      if (needsUpload) {
        gl.bufferData(gl.ARRAY_BUFFER, toBufferView(geometryAttribute.array), gl.STATIC_DRAW);
        geometryAttribute.needsUpdate = false;
      }
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, geometryAttribute.itemSize, gl.FLOAT, geometryAttribute.normalized, 0, 0);
      gl.vertexAttribDivisor(location, attribute.stepMode === "instance" ? 1 : 0);
    }

    if (geometry.index) {
      if (!buffers.index) buffers.index = gl.createBuffer()!;
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.index);
      if (needsUpload) {
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, toBufferView(geometry.index.array, true), gl.STATIC_DRAW);
      }
    }
    buffers.needsUpload = false;
  }

  dispose(): void {
    const gl = this.gl;
    for (const entry of this.programs.values()) gl.deleteProgram(entry.glProgram);
    for (const buffers of this.geometryBuffers.values()) {
      for (const buffer of buffers.attributes.values()) gl.deleteBuffer(buffer);
      if (buffers.index) gl.deleteBuffer(buffers.index);
    }
    for (const texture of this.textures.values()) gl.deleteTexture(texture);
    this.programs.clear();
    this.geometryBuffers.clear();
    this.textures.clear();
  }
}

/**
 * The WebGL2 internal format and upload type for an integer RGBA texture,
 * from the bit depth of its data view and the sampler's signedness. The
 * format name ends in `UI` for unsigned and `I` for signed, each sized to the
 * element.
 */
function integerInternalFormat(
  gl: WebGL2RenderingContext,
  signed: boolean,
  view: ArrayBufferView,
): { internalFormat: number; type: number } {
  const bytes = (view as { BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT ?? 1;
  if (signed) {
    if (bytes === 1) return { internalFormat: gl.RGBA8I, type: gl.BYTE };
    if (bytes === 2) return { internalFormat: gl.RGBA16I, type: gl.SHORT };
    return { internalFormat: gl.RGBA32I, type: gl.INT };
  }
  if (bytes === 1) return { internalFormat: gl.RGBA8UI, type: gl.UNSIGNED_BYTE };
  if (bytes === 2) return { internalFormat: gl.RGBA16UI, type: gl.UNSIGNED_SHORT };
  return { internalFormat: gl.RGBA32UI, type: gl.UNSIGNED_INT };
}
