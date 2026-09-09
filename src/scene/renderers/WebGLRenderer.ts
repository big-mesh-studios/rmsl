import { compileGLSL, type GLSLPrecision } from "../../rmsl";
import { Color } from "../math/Color";
import { Vector4 } from "../math/Vector4";
import { WebGLRenderTarget } from "./WebGLRenderTarget";
import type { Scene } from "../scenes/Scene";
import type { Camera } from "../cameras/Camera";
import type { Mesh } from "../objects/Mesh";
import type { InstancedMesh } from "../objects/InstancedMesh";
import type { BufferGeometry } from "../geometries/BufferGeometry";
import type { BufferAttribute } from "../geometries/BufferAttribute";
import type { Texture } from "../textures/Texture";
import type { NodeMaterial, MaterialProgram } from "../materials/NodeMaterial";
import { Blending, Side } from "../materials/Material";
import {
  cameraUniformValue, isIntegerSampler, objectUniformValue, lightsSignature,
  shaderPrecision, toBufferView, samplerState, textureChannels, type TextureWrap,
  rendererUniformValue, programSignature, geometryAttribute,
} from "./common";

interface ProgramEntry {
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

  private programs = new Map<NodeMaterial, Map<string, ProgramEntry>>();
  private geometryBuffers = new Map<BufferGeometry, GeometryBuffers>();
  /**
   * Buffers for attributes that live on the object rather than the geometry —
   * an `InstancedMesh`'s `instanceMatrix`/`instanceColor`. Keyed by the
   * attribute so two instanced meshes sharing a geometry keep separate buffers.
   */
  private attributeBuffers = new Map<BufferAttribute, WebGLBuffer>();
  /** Bytes allocated per `WebGLBuffer`, so a buffer that grew past its first
   * upload is re-allocated only when the new data no longer fits. */
  private bufferCapacities = new WeakMap<WebGLBuffer, number>();
  private textures = new Map<Texture, WebGLTexture>();
  /** The framebuffer, color texture, and depth renderbuffer behind each render target, at its bound size. */
  private renderTargets = new Map<
    WebGLRenderTarget,
    { framebuffer: WebGLFramebuffer; color: WebGLTexture; depth: WebGLRenderbuffer; width: number; height: number }
  >();
  private clearColor = new Color(0, 0, 0);
  private clearAlpha = 1;
  private animationCallback: ((time: number) => void) | null = null;
  private animationHandle: number | null = null;
  /**
   * The shader precision compiled into every program unless a material
   * overrides it, like three.js's `WebGLRenderer` `precision` option.
   */
  readonly precision: GLSLPrecision;

  constructor(
    canvas?: HTMLCanvasElement,
    options: { antialias?: boolean; depth?: boolean; precision?: GLSLPrecision } = {},
  ) {
    this.canvas = canvas ?? document.createElement("canvas");
    this.precision = options.precision ?? "highp";
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

  /**
   * Draws `scene` from `camera`'s view. A `target` redirects the draw to an
   * offscreen framebuffer sized after the target; without one the draw lands
   * on the canvas at its full size. The target keeps its own hidden color and
   * depth, so this is how a pass the player never sees is produced without
   * touching the canvas's drawing buffer.
   */
  render(scene: Scene, camera: Camera, target: WebGLRenderTarget | null = null): void {
    const gl = this.gl;

    scene.updateMatrixWorld(true);
    camera.updateMatrixWorld(true);
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();

    if (target !== null) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.renderTargetFramebuffer(target, gl));
      gl.viewport(0, 0, target.width, target.height);
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    }
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

  /**
   * The framebuffer behind a render target, created (or recreated at the
   * target's current size) on this renderer's context the first time it is
   * drawn into — so a target costs nothing until a pass binds it, and the app
   * resizes it by reassigning `width`/`height` before the next render.
   */
  renderTargetFramebuffer(target: WebGLRenderTarget, gl = this.gl): WebGLFramebuffer {
    const existing = this.renderTargets.get(target);
    if (existing !== undefined && existing.width === target.width && existing.height === target.height) {
      return existing.framebuffer;
    }
    if (existing !== undefined) {
      this.deleteRenderTarget(target, existing, gl);
    }
    const framebuffer = gl.createFramebuffer()!;
    const color = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, color);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, target.width, target.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    const depth = gl.createRenderbuffer()!;
    gl.bindRenderbuffer(gl.RENDERBUFFER, depth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, target.width, target.height);
    gl.bindRenderbuffer(gl.RENDERBUFFER, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, color, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const entry = { framebuffer, color, depth, width: target.width, height: target.height };
    this.renderTargets.set(target, entry);
    return framebuffer;
  }

  /**
   * Copies a render target's color buffer into a `Uint8Array` and returns it.
   * The call reads the GPU back to the host and so stalls the pipeline; it is
   * for the occasional sample (a colour-coded occlusion pass every few hundred
   * frames), not the steady state of a frame.
   */
  readPixels(target: WebGLRenderTarget, out?: Uint8Array): Uint8Array {
    const gl = this.gl;
    const buffer = out ?? new Uint8Array(target.width * target.height * 4);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.renderTargetFramebuffer(target));
    gl.readPixels(0, 0, target.width, target.height, gl.RGBA, gl.UNSIGNED_BYTE, buffer);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    return buffer;
  }

  /**
   * Copies a render target's color buffer into a `Uint8Array` without
   * stalling the pipeline: the transfer goes through a pixel buffer object,
   * and the returned promise resolves once a GPU fence confirms it landed,
   * typically a frame or a few after the call rather than within it. Suits a
   * sample that tolerates that lag (a colour-coded occlusion pass) in place
   * of `readPixels`, whose stall a caller can't otherwise avoid.
   */
  readPixelsAsync(target: WebGLRenderTarget, out?: Uint8Array): Promise<Uint8Array> {
    const gl = this.gl;
    const buffer = out ?? new Uint8Array(target.width * target.height * 4);
    const pbo = gl.createBuffer()!;
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo);
    gl.bufferData(gl.PIXEL_PACK_BUFFER, buffer.byteLength, gl.STREAM_READ);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.renderTargetFramebuffer(target));
    gl.readPixels(0, 0, target.width, target.height, gl.RGBA, gl.UNSIGNED_BYTE, 0);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)!;
    gl.flush();
    return new Promise<Uint8Array>((resolve, reject) => {
      const poll = (): void => {
        const status = gl.clientWaitSync(sync, 0, 0);
        if (status === gl.TIMEOUT_EXPIRED) {
          requestAnimationFrame(poll);
          return;
        }
        gl.deleteSync(sync);
        if (status === gl.WAIT_FAILED) {
          gl.deleteBuffer(pbo);
          reject(new Error("[RMSL/scene] readPixelsAsync: GPU sync wait failed"));
          return;
        }
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo);
        gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, buffer);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        gl.deleteBuffer(pbo);
        resolve(buffer);
      };
      poll();
    });
  }

  private drawMesh(mesh: Mesh, scene: Scene, camera: Camera): void {
    const material = mesh.material;
    if (!(material as NodeMaterial).isNodeMaterial) return;
    const instancing = (mesh as InstancedMesh).isInstancedMesh === true;
    const instancingColor = instancing && (mesh as InstancedMesh).instanceColor !== null;
    this.usedTextureUnits.clear();
    const entry = this.ensureProgram(material as NodeMaterial, scene, instancing, instancingColor);
    const gl = this.gl;

    gl.useProgram(entry.glProgram);
    this.setRenderState(material);
    this.uploadUniforms(entry, mesh, camera);
    this.bindGeometry(mesh, entry, mesh.geometry);

    const geometry = mesh.geometry;
    const instanceCount = instancing ? (mesh as InstancedMesh).count : geometry.instanceCount;
    // A mesh can draw a slice of its geometry (an object sharing one merged
    // buffer with several others); omit the mesh's `drawRange` to draw it all.
    const range = mesh.drawRange;
    if (geometry.index) {
      const indexView = toBufferView(geometry.index.array, true) as Uint16Array | Uint32Array;
      const type = indexView instanceof Uint16Array ? gl.UNSIGNED_SHORT : gl.UNSIGNED_INT;
      const count = Number.isFinite(range.count) ? range.count : indexView.length;
      gl.drawElementsInstanced(
        gl.TRIANGLES,
        count,
        type,
        range.start * indexView.BYTES_PER_ELEMENT,
        instanceCount,
      );
    } else {
      const count = Number.isFinite(range.count)
        ? range.count
        : geometry.attributes.position?.count ?? 0;
      gl.drawArraysInstanced(gl.TRIANGLES, range.start, count, instanceCount);
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
    // vector/matrix ones as arrays. `value[0]` on a bare number is undefined,
    // which would upload NaN and blacken the surface — so the scalar is
    // uploaded directly.
    if (typeof value === "number") {
      switch (type) {
        case "float": gl.uniform1f(location, value); break;
        case "int": case "bool": gl.uniform1i(location, value); break;
        // A bare number is not a vector or a matrix, and an unknown type is
        // skipped rather than guessed.
        default: break;
      }
      return;
    }
    // Vectors go up through the component forms (`uniform3f`) rather than the
    // array forms (`uniform3fv`). The array forms take a typed array, so
    // uploading through them meant building one per uniform per draw — a scene
    // drawing a few dozen vector uniforms was allocating thousands of throwaway
    // typed arrays a second. Only the matrix forms have no component
    // equivalent, and both a plain array and a typed array are accepted there.
    switch (type) {
      case "float": gl.uniform1f(location, value[0]); break;
      case "int": gl.uniform1i(location, value[0]); break;
      case "bool": gl.uniform1i(location, value[0]); break;
      case "vec2": gl.uniform2f(location, value[0], value[1]); break;
      case "vec3": gl.uniform3f(location, value[0], value[1], value[2]); break;
      case "vec4": gl.uniform4f(location, value[0], value[1], value[2], value[3]); break;
      case "ivec2": gl.uniform2i(location, value[0], value[1]); break;
      case "ivec3": gl.uniform3i(location, value[0], value[1], value[2]); break;
      case "ivec4": gl.uniform4i(location, value[0], value[1], value[2], value[3]); break;
      case "mat2": gl.uniformMatrix2fv(location, false, value); break;
      case "mat3": gl.uniformMatrix3fv(location, false, value); break;
      case "mat4": gl.uniformMatrix4fv(location, false, value); break;
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
        texture.addEventListener("dispose", this.onTextureDispose);
      }
      gl.bindTexture(target, glTexture);
      const sampling = samplerState(texture, samplerType);
      gl.texParameteri(target, gl.TEXTURE_WRAP_S, glWrap(gl, sampling.wrapS));
      gl.texParameteri(target, gl.TEXTURE_WRAP_T, glWrap(gl, sampling.wrapT));
      if (is3D) gl.texParameteri(target, gl.TEXTURE_WRAP_R, glWrap(gl, sampling.wrapR));
      gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, glFilter(gl, sampling.minFilter));
      gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, glFilter(gl, sampling.magFilter));
      const image = texture.image;
      if (ArrayBuffer.isView(image)) {
        const width = (texture as { width?: number }).width ?? 1;
        const height = (texture as { height?: number }).height ?? 1;
        if (integer) {
          const singleChannel = textureChannels(texture) === 1;
          const { internalFormat, format, type } = integerInternalFormat(gl, samplerType.startsWith("isampler"), image, singleChannel);
          if (is3D) {
            const depth = (texture as { depth?: number }).depth ?? 1;
            gl.texImage3D(target, 0, internalFormat, width, height, depth, 0, format, type, image as ArrayBufferView);
          } else {
            gl.texImage2D(target, 0, internalFormat, width, height, 0, format, type, image as ArrayBufferView);
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

  /**
   * Free the GL texture a disposed `Texture` owns, and stop listening to it.
   * Drawing with the texture again is allowed: `bindTexture` finds no GL
   * texture for it and creates and uploads a new one.
   */
  private onTextureDispose = (event: unknown): void => {
    const texture = (event as { target: Texture }).target;
    const glTexture = this.textures.get(texture);
    if (glTexture) this.gl.deleteTexture(glTexture);
    this.textures.delete(texture);
    texture.removeEventListener("dispose", this.onTextureDispose);
  };

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

  private ensureProgram(
    material: NodeMaterial,
    scene: Scene,
    instancing: boolean,
    instancingColor: boolean,
  ): ProgramEntry {
    const signature = programSignature(lightsSignature(scene), instancing, instancingColor);
    let bySignature = this.programs.get(material);
    const entry = bySignature?.get(signature);
    if (entry && !material.needsUpdate) {
      return entry;
    }

    const program = material.build(scene, { instancing, instancingColor });
    const gl = this.gl;
    const precision = shaderPrecision(material, this.precision);

    const vertexShader = this.compileShader(compileGLSL.vertex(program.vertexRoot, { precision }), gl.VERTEX_SHADER);
    const fragmentShader = this.compileShader(compileGLSL.fragment(program.fragmentRoot, { precision }), gl.FRAGMENT_SHADER);
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

    const built: ProgramEntry = { program, glProgram, uniformLocations, attributeLocations };
    if (!bySignature) {
      bySignature = new Map();
      this.programs.set(material, bySignature);
    }
    bySignature.set(signature, built);
    material.needsUpdate = false;
    return built;
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

  private bindGeometry(mesh: Mesh, entry: ProgramEntry, geometry: BufferGeometry): void {
    const gl = this.gl;
    let buffers = this.geometryBuffers.get(geometry);
    if (!buffers) {
      buffers = { attributes: new Map(), index: null, needsUpload: true };
      this.geometryBuffers.set(geometry, buffers);
    }

    const needsUpload = buffers.needsUpload
      || Object.values(geometry.attributes).some((a) => a.needsUpdate);

    for (const attribute of entry.program.attributes) {
      const attr = geometryAttribute(mesh, geometry, attribute.name);
      const location = entry.attributeLocations.get(attribute.node.name);
      if (!attr || location == null) continue;

      // `instanceMatrix`/`instanceColor` live on the object rather than the
      // geometry, so their buffers are cached per attribute (not per geometry).
      const ownedByGeometry = geometry.attributes[attribute.name] !== undefined;
      let buffer = ownedByGeometry
        ? buffers.attributes.get(attribute.name)
        : this.attributeBuffers.get(attr);
      const isNewBuffer = buffer === undefined;
      if (!buffer) {
        buffer = gl.createBuffer()!;
        if (ownedByGeometry) buffers.attributes.set(attribute.name, buffer);
        else this.attributeBuffers.set(attr, buffer);
      }
      if (isNewBuffer || attr.needsUpdate) {
        const data = toBufferView(attr.array);
        buffer = this.uploadSlice(
          gl,
          gl.ARRAY_BUFFER,
          buffer,
          data,
          this.uploadRangeOf(data, attr, isNewBuffer),
        );
        if (ownedByGeometry) buffers.attributes.set(attribute.name, buffer);
        else this.attributeBuffers.set(attr, buffer);
        attr.needsUpdate = false;
      }
      // The attribute pointers below capture whatever buffer is bound when
      // they run, so bind this attribute's buffer on every draw, whether or
      // not its data changed this frame.
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);

      // A mat4 attribute spans four consecutive vertex attribute locations;
      // each is fed from one column of the 64-byte instance record. The GLSL
      // linker handed the base location, so the columns land at location..+3.
      const locationSize = attribute.node._t === "mat4" ? 4 : 1;
      const stride = attr.itemSize * 4;
      for (let i = 0; i < locationSize; i++) {
        gl.enableVertexAttribArray(location + i);
        gl.vertexAttribPointer(
          location + i,
          attr.itemSize / locationSize,
          gl.FLOAT,
          attr.normalized,
          stride,
          (attr.itemSize / locationSize) * i * 4,
        );
        gl.vertexAttribDivisor(location + i, attribute.stepMode === "instance" ? 1 : 0);
      }
    }

    if (geometry.index) {
      const isNewIndex = buffers.index === null;
      const indexBuffer = buffers.index ?? (buffers.index = gl.createBuffer()!);
      if (isNewIndex || needsUpload || geometry.index.needsUpdate) {
        const data = toBufferView(geometry.index.array, true);
        buffers.index = this.uploadSlice(
          gl,
          gl.ELEMENT_ARRAY_BUFFER,
          indexBuffer,
          data,
          this.uploadRangeOf(data, geometry.index, isNewIndex),
        );
        geometry.index.needsUpdate = false;
      }
      // The element buffer binding must name this geometry's indices when the
      // draw runs, whatever the previous draw left bound.
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.index!);
    }
    buffers.needsUpload = false;
  }

  /** The byte window of `data` an update should send, per `updateRange`. */
  private uploadRangeOf(
    data: ArrayBufferView,
    attr: BufferAttribute,
    full: boolean,
  ): { byteOffset: number; byteEnd: number } {
    if (full || attr.updateRange.count === -1) {
      return { byteOffset: 0, byteEnd: data.byteLength };
    }
    const bytes = (data as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const byteOffset = Math.min(
      data.byteLength,
      Math.max(0, attr.updateRange.offset) * bytes,
    );
    const byteEnd = Math.min(
      data.byteLength,
      byteOffset + Math.max(0, attr.updateRange.count) * bytes,
    );
    return { byteOffset, byteEnd };
  }

  /**
   * Uploads the `[byteOffset, byteEnd)` slice of `data` into `buffer`. The
   * first upload sizes the buffer to the whole array; a later upload that no
   * longer fits re-allocates (doubling, so a growing attribute's re-sends
   * amortize) and re-sends everything, and one that fits sends only its slice.
   * Returns the buffer to keep, which differs from `buffer` when it grew.
   */
  private uploadSlice(
    gl: WebGL2RenderingContext,
    target: number,
    buffer: WebGLBuffer,
    data: ArrayBufferView,
    { byteOffset, byteEnd }: { byteOffset: number; byteEnd: number },
  ): WebGLBuffer {
    gl.bindBuffer(target, buffer);
    const capacity = this.bufferCapacities.get(buffer);
    if (capacity === undefined) {
      gl.bufferData(target, data, gl.STATIC_DRAW);
      this.bufferCapacities.set(buffer, data.byteLength);
      return buffer;
    }
    if (byteEnd > capacity) {
      const size = Math.max(byteEnd, capacity * 2);
      const next = gl.createBuffer()!;
      gl.bindBuffer(target, next);
      gl.bufferData(target, size, gl.STATIC_DRAW);
      gl.bufferSubData(target, 0, data);
      gl.deleteBuffer(buffer);
      this.bufferCapacities.set(next, size);
      return next;
    }
    if (byteEnd > byteOffset) {
      const view = data as unknown as { BYTES_PER_ELEMENT: number; subarray(a: number, b: number): ArrayBufferView };
      gl.bufferSubData(
        target,
        byteOffset,
        view.subarray(byteOffset / view.BYTES_PER_ELEMENT, byteEnd / view.BYTES_PER_ELEMENT),
      );
    }
    return buffer;
  }

  private deleteRenderTarget(
    target: WebGLRenderTarget,
    entry: { framebuffer: WebGLFramebuffer; color: WebGLTexture; depth: WebGLRenderbuffer },
    gl = this.gl,
  ): void {
    gl.deleteFramebuffer(entry.framebuffer);
    gl.deleteTexture(entry.color);
    gl.deleteRenderbuffer(entry.depth);
    this.renderTargets.delete(target);
  }

  dispose(): void {
    const gl = this.gl;
    for (const bySignature of this.programs.values()) {
      for (const entry of bySignature.values()) gl.deleteProgram(entry.glProgram);
    }
    for (const buffers of this.geometryBuffers.values()) {
      for (const buffer of buffers.attributes.values()) gl.deleteBuffer(buffer);
      if (buffers.index) gl.deleteBuffer(buffers.index);
    }
    for (const buffer of this.attributeBuffers.values()) gl.deleteBuffer(buffer);
    for (const [texture, glTexture] of this.textures) {
      gl.deleteTexture(glTexture);
      texture.removeEventListener("dispose", this.onTextureDispose);
    }
    for (const [target, entry] of this.renderTargets) this.deleteRenderTarget(target, entry);
    this.programs.clear();
    this.geometryBuffers.clear();
    this.attributeBuffers.clear();
    this.textures.clear();
  }
}

/** A wrapping mode as the `texParameteri` constant that sets it. */
function glWrap(gl: WebGL2RenderingContext, wrap: TextureWrap): number {
  switch (wrap) {
    case "repeat": return gl.REPEAT;
    case "mirror": return gl.MIRRORED_REPEAT;
    default: return gl.CLAMP_TO_EDGE;
  }
}

/** A filter as the `texParameteri` constant that sets it. */
function glFilter(gl: WebGL2RenderingContext, filter: "nearest" | "linear"): number {
  return filter === "nearest" ? gl.NEAREST : gl.LINEAR;
}

/**
 * The WebGL2 internal format, texel format, and upload type for an integer
 * texture, from the bit depth of its data view and the sampler's signedness.
 * A `RedIntegerFormat` data view is single-channel (`R8UI`/`R8I`, read with
 * `RED_INTEGER`); anything else is treated as RGBA. The format name ends in
 * `UI` for unsigned and `I` for signed, each sized to the element.
 */
function integerInternalFormat(
  gl: WebGL2RenderingContext,
  signed: boolean,
  view: ArrayBufferView,
  singleChannel: boolean,
): { internalFormat: number; format: number; type: number } {
  const bytes = (view as { BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT ?? 1;
  if (signed) {
    if (singleChannel) return { internalFormat: gl.R8I, format: gl.RED_INTEGER, type: gl.BYTE };
    if (bytes === 1) return { internalFormat: gl.RGBA8I, format: gl.RGBA_INTEGER, type: gl.BYTE };
    if (bytes === 2) return { internalFormat: gl.RGBA16I, format: gl.RGBA_INTEGER, type: gl.SHORT };
    return { internalFormat: gl.RGBA32I, format: gl.RGBA_INTEGER, type: gl.INT };
  }
  if (singleChannel) return { internalFormat: gl.R8UI, format: gl.RED_INTEGER, type: gl.UNSIGNED_BYTE };
  if (bytes === 1) return { internalFormat: gl.RGBA8UI, format: gl.RGBA_INTEGER, type: gl.UNSIGNED_BYTE };
  if (bytes === 2) return { internalFormat: gl.RGBA16UI, format: gl.RGBA_INTEGER, type: gl.UNSIGNED_SHORT };
  return { internalFormat: gl.RGBA32UI, format: gl.RGBA_INTEGER, type: gl.UNSIGNED_INT };
}
