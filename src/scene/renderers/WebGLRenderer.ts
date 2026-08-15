import { compileGLSL } from "../../rmsl";
import { Color } from "../math/Color";
import type { Scene } from "../scenes/Scene";
import type { Camera } from "../cameras/Camera";
import type { Mesh } from "../objects/Mesh";
import type { BufferGeometry } from "../geometries/BufferGeometry";
import type { Texture } from "../textures/Texture";
import type { NodeMaterial, MaterialProgram } from "../materials/NodeMaterial";
import { Blending, Side } from "../materials/Material";
import { cameraUniformValue, objectUniformValue, lightsSignature, toBufferView, uniformUploadValue } from "./common";

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
      if (object.isMesh) this.drawMesh(object as Mesh, scene, camera);
    });
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
      gl.drawElements(gl.TRIANGLES, indexView.length, type, 0);
    } else {
      gl.drawArrays(gl.TRIANGLES, 0, geometry.attributes.position?.count ?? 0);
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
      const unit = this.bindTexture(texture);
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

  private bindTexture(texture: Texture): number {
    const gl = this.gl;
    let glTexture = this.textures.get(texture);
    if (!glTexture || texture.needsUpdate) {
      if (!glTexture) {
        glTexture = gl.createTexture()!;
        this.textures.set(texture, glTexture);
      }
      gl.bindTexture(gl.TEXTURE_2D, glTexture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      const image = texture.image;
      if (ArrayBuffer.isView(image)) {
        const width = (texture as { width?: number }).width ?? 1;
        const height = (texture as { height?: number }).height ?? 1;
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, image as ArrayBufferView);
      } else if (image != null) {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image as TexImageSource);
      }
      texture.needsUpdate = false;
    }
    // Find a free texture unit and bind.
    const unit = this.nextTextureUnit();
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, glTexture);
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
