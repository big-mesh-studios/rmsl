import { Matrix3 } from "../math/Matrix3";
import { Vector3 } from "../math/Vector3";
import type { Camera } from "../cameras/Camera";
import type { Mesh } from "../objects/Mesh";
import type { InstancedMesh } from "../objects/InstancedMesh";
import type { Scene } from "../scenes/Scene";
import type { Color } from "../math/Color";
import type { BufferGeometry } from "../geometries/BufferGeometry";
import type { BufferAttribute } from "../geometries/BufferAttribute";
import { AmbientLight } from "../lights/AmbientLight";
import { DirectionalLight } from "../lights/DirectionalLight";
import { PointLight } from "../lights/PointLight";
import type { NodeMaterial } from "../materials/NodeMaterial";
import { DataTexture } from "../textures/DataTexture";
import type { GLSLPrecision } from "../../glsl";

// The sampling helpers live apart, so a test of a plain shader can read a texture as the renderers do
// without loading the scene graph.
export {
  isFloatTexture,
  isIntegerSampler,
  samplerState,
  textureChannels,
  type SamplerState,
  type TextureWrap,
} from "./sampling";

/**
 * Which shader precision a program compiles with, mirroring three.js: a
 * material's own `precision` overrides the renderer's default (which is what a
 * `null` material precision keeps).
 */
export function shaderPrecision(material: NodeMaterial, rendererPrecision: GLSLPrecision): GLSLPrecision {
  return material.precision ?? rendererPrecision;
}

/**
 * The value a camera-scoped uniform should hold this frame, given its logical
 * name. Unknown names return an empty array, which the renderer treats as
 * "nothing to upload".
 */
export function cameraUniformValue(name: string, camera: Camera): number[] {
  switch (name) {
    case "projectionMatrix":
      return camera.projectionMatrix.elements;
    case "viewMatrix":
      return camera.matrixWorldInverse.elements;
    case "cameraPosition":
      return camera.getWorldPosition().toArray();
    default:
      return [];
  }
}

/**
 * The value an object-scoped uniform should hold for a mesh, given its logical
 * name. `normalMatrix` is the inverse-transpose of the world matrix's upper
 * 3x3, computed on the host.
 */
export function objectUniformValue(name: string, mesh: Mesh): number[] {
  switch (name) {
    case "modelMatrix":
      return mesh.matrixWorld.elements;
    case "normalMatrix":
      return _normalMatrix.getNormalMatrix(mesh.matrixWorld).toArray();
    default:
      return [];
  }
}

/**
 * The attribute a shader input reads from, for a drawable and its geometry.
 * An `InstancedMesh` keeps `instanceMatrix`/`instanceColor` on the object
 * rather than the geometry (three.js does the same), so those two names fall
 * back to the object when the geometry does not carry them.
 */
export function geometryAttribute(mesh: Mesh, geometry: BufferGeometry, name: string): BufferAttribute | undefined {
  const fromGeometry = geometry.attributes[name];
  if (fromGeometry) return fromGeometry;
  if (name === "instanceMatrix") return (mesh as InstancedMesh).instanceMatrix;
  if (name === "instanceColor") return (mesh as InstancedMesh).instanceColor ?? undefined;
  return undefined;
}

const _normalMatrix = new Matrix3();

/**
 * Whether a mesh's world matrix mirrors it, which reverses the winding of its
 * triangles, so its front face winds clockwise, as three.js winds it.
 */
export function isMirrored(mesh: Mesh): boolean {
  return mesh.matrixWorld.determinant() < 0;
}

/**
 * The value a renderer-scoped uniform should hold this frame, given its
 * logical name and the drawing surface's device-pixel size. Unknown names
 * return an empty array, which the renderer treats as "nothing to upload".
 */
export function rendererUniformValue(name: string, width: number, height: number): number[] {
  switch (name) {
    case "resolution":
      return [width, height];
    default:
      return [];
  }
}

/**
 * A signature of a scene's light set, in traversal order. When it changes the
 * shaders a material compiled against (light uniforms are baked in) must be
 * rebuilt.
 */
export function lightsSignature(scene: Scene): string {
  let signature = "";
  scene.traverseVisible((object) => {
    if (object instanceof AmbientLight) signature += "a";
    else if (object instanceof DirectionalLight) signature += "d";
    else if (object instanceof PointLight) signature += "p";
  });
  return signature;
}

/**
 * The signature identifying one compiled program: the scene's light set plus
 * the drawable's instancing flags. A shared material therefore compiles one
 * program per distinct combination — a plain mesh and an `InstancedMesh` with
 * the same material get different shaders, exactly as the per-instance
 * attributes only exist for the instanced one.
 */
export function programSignature(lights: string, instancing: boolean, instancingColor: boolean): string {
  return `${lights}|${instancing ? "i" : ""}${instancingColor ? "c" : ""}`;
}

/** The WGSL spelling of an RMSL shader type, for uniform struct members. */
export function wgslTypeName(type: string): string {
  switch (type) {
    case "float":
      return "f32";
    case "int":
      return "i32";
    case "uint":
      return "u32";
    case "vec2":
      return "vec2<f32>";
    case "vec3":
      return "vec3<f32>";
    case "vec4":
      return "vec4<f32>";
    case "ivec2":
      return "vec2<i32>";
    case "ivec3":
      return "vec3<i32>";
    case "ivec4":
      return "vec4<i32>";
    case "uvec2":
      return "vec2<u32>";
    case "uvec3":
      return "vec3<u32>";
    case "uvec4":
      return "vec4<u32>";
    case "mat2":
      return "mat2x2<f32>";
    case "mat3":
      return "mat3x3<f32>";
    case "mat4":
      return "mat4x4<f32>";
    default:
      return "f32";
  }
}

/**
 * The WebGPU `GPUTextureSampleType` a sampler type requires: integer textures
 * are `sint`/`uint`, everything else samples as floats.
 */
export function samplerSampleType(type: string): "float" | "sint" | "uint" {
  if (type.startsWith("isampler")) return "sint";
  if (type.startsWith("usampler")) return "uint";
  return "float";
}

/** Whether a sampler type addresses a volume rather than a surface. */
export function samplerDimension(type: string): "2d" | "3d" {
  return type.endsWith("3D") ? "3d" : "2d";
}

/**
 * The 1×1 black texture a sampler with no texture reads, kept in `blanks` per
 * dimension and sample type so an integer or 3D sampler gets a texture it can
 * read. An integer texel is `[0, 0, 0, 255]`, or `[0, 0, 0, -1]` when signed.
 */
export function blankTexture(blanks: Map<string, DataTexture>, samplerType = "sampler2D"): DataTexture {
  const sampleType = samplerSampleType(samplerType);
  const key = `${samplerDimension(samplerType)}:${sampleType}`;
  let blank = blanks.get(key);
  if (!blank) {
    const data = sampleType === "sint" ? new Int8Array([0, 0, 0, -1]) : new Uint8Array([0, 0, 0, 255]);
    blank = new DataTexture(data, 1, 1, 1);
    blanks.set(key, blank);
  }
  return blank;
}

/**
 * A scalar uniform value uploads as a single element; vector/matrix values as
 * their component array. `scalar` is non-null exactly for a bare number, so a
 * caller can upload it directly — indexing `[0]` on a bare number is undefined
 * and silently uploads NaN, which is how a lit material once rendered black.
 */
export function uniformUploadValue(value: number | number[] | Float32Array): {
  scalar: number | null;
  array: Float32Array;
} {
  if (typeof value === "number") {
    return { scalar: value, array: new Float32Array([value]) };
  }
  return { scalar: null, array: new Float32Array(value as number[] | Float32Array) };
}

/**
 * A buffer upload source from a geometry array, converting a plain `number[]`
 * to the typed view a GPU upload needs. `index` picks an unsigned element type
 * for element buffers; vertex attributes default to floats.
 */
export function toBufferView(array: ArrayLike<number>, index = false): ArrayBufferView<ArrayBuffer> {
  if (ArrayBuffer.isView(array)) return array as unknown as ArrayBufferView<ArrayBuffer>;
  if (index) {
    let max = -Infinity;
    for (let i = 0; i < array.length; i++) if (array[i] > max) max = array[i];
    return max > 65535 ? new Uint32Array(array) : new Uint16Array(array);
  }
  return new Float32Array(array);
}

/**
 * One vertex attribute's layout, under the name WebGPU gives it. The set is
 * limited to the formats that reach a shader as floats on both backends, which
 * is the set a program whose attribute types are *declared* can bind: a
 * material asks for a `vec4` and gets one from `unorm8x4` under either backend,
 * where `uint8x4` would be a `vec4` in GLSL and a `vec4<u32>` in WGSL and only
 * one of those is what the material asked for. Admitting the integer formats
 * means deriving each attribute's shader type from its buffer and compiling the
 * program per geometry, which is a larger change than this one.
 */
export type VertexFormat =
  | "float32"
  | "float32x2"
  | "float32x3"
  | "float32x4"
  | "float16x2"
  | "float16x4"
  | "snorm8x4"
  | "unorm8x4"
  | "snorm16x2"
  | "snorm16x4"
  | "unorm16x2"
  | "unorm16x4";

/** What one vertex format is made of, in the terms each backend binds it by. */
export interface VertexFormatSpec {
  /** Components one vertex holds. */
  count: number;
  /** Bytes one component occupies. */
  bytes: number;
  /** Whether the stored integer is scaled into 0..1 or -1..1 on the way in. */
  normalized: boolean;
  /** The WebGL component type, by its `WebGL2RenderingContext` key. */
  gl: "BYTE" | "UNSIGNED_BYTE" | "SHORT" | "UNSIGNED_SHORT" | "HALF_FLOAT" | "FLOAT";
}

/**
 * Every vertex format both backends can be handed, and what each is made of.
 *
 * `count * bytes` is the stride of a buffer holding this attribute and nothing
 * else, which is how both renderers bind one, and WebGPU requires that stride
 * to be a multiple of four. That is what shapes this list: no three-component
 * format below 32 bits, and the byte-wide formats only four components wide. An
 * attribute narrower than four bytes packs into the spare lanes of a wider one
 * — a face index and a light level sharing a `unorm8x4` — rather than taking a
 * buffer to itself.
 *
 * WebGPU defines two-component byte formats as well, and they are absent here
 * for that same reason: a buffer of one alone would have a stride of two. What
 * would admit them is a `BufferAttribute` able to carry a stride and an offset,
 * so that several attributes share one buffer — the rule constrains the buffer,
 * not each attribute inside it, so a `unorm8x2` at byte 16 of a 24-byte record
 * is fine where the same format alone is not.
 */
export const VERTEX_FORMATS: Record<VertexFormat, VertexFormatSpec> = {
  float32: { count: 1, bytes: 4, normalized: false, gl: "FLOAT" },
  float32x2: { count: 2, bytes: 4, normalized: false, gl: "FLOAT" },
  float32x3: { count: 3, bytes: 4, normalized: false, gl: "FLOAT" },
  float32x4: { count: 4, bytes: 4, normalized: false, gl: "FLOAT" },
  float16x2: { count: 2, bytes: 2, normalized: false, gl: "HALF_FLOAT" },
  float16x4: { count: 4, bytes: 2, normalized: false, gl: "HALF_FLOAT" },
  snorm8x4: { count: 4, bytes: 1, normalized: true, gl: "BYTE" },
  unorm8x4: { count: 4, bytes: 1, normalized: true, gl: "UNSIGNED_BYTE" },
  snorm16x2: { count: 2, bytes: 2, normalized: true, gl: "SHORT" },
  snorm16x4: { count: 4, bytes: 2, normalized: true, gl: "SHORT" },
  unorm16x2: { count: 2, bytes: 2, normalized: true, gl: "UNSIGNED_SHORT" },
  unorm16x4: { count: 4, bytes: 2, normalized: true, gl: "UNSIGNED_SHORT" },
};

/**
 * The format family an attribute's array belongs to, or nothing where its bytes
 * cannot reach a shader as floats. A plain `number[]` is whatever
 * `toBufferView` will make of it, which is a `Float32Array`.
 */
function formatPrefix(array: ArrayLike<number>, normalized: boolean): string | undefined {
  if (!ArrayBuffer.isView(array)) return "float32";
  if (array instanceof Float32Array) return "float32";
  if (typeof Float16Array !== "undefined" && array instanceof Float16Array) return "float16";
  // An integer array reaches a float attribute only by being scaled on the way
  // in. Read raw it would be an integer in the shader too, which is the type
  // the material did not declare.
  if (!normalized) return undefined;
  if (array instanceof Int8Array) return "snorm8";
  if (array instanceof Uint8Array) return "unorm8";
  if (array instanceof Int16Array) return "snorm16";
  if (array instanceof Uint16Array) return "unorm16";
  return undefined;
}

/**
 * The vertex format an attribute's bytes are in: its own `format` where it
 * declares one, and otherwise the format its array type, component count and
 * `normalized` flag imply.
 *
 * @param count Components one shader location consumes, which is the attribute's
 *   `itemSize` except for a `mat4`, whose four locations each take four of its
 *   sixteen.
 * @throws When the attribute's bytes have no format in `VERTEX_FORMATS` — a raw
 *   integer array, or a width no format covers. Both are silent bugs otherwise:
 *   the bytes get read as something they are not.
 */
export function vertexFormatOf(attr: BufferAttribute, count = attr.itemSize): VertexFormat {
  if (attr.format !== undefined) return attr.format;
  const prefix = formatPrefix(attr.array, attr.normalized);
  if (prefix === undefined) {
    throw new Error(
      `rmsl: a ${arrayTypeName(attr.array)} attribute has no vertex format. ` +
        "An integer array reaches a float attribute only when it is scaled on " +
        "the way in: set `normalized: true`, or set `format` to say what its " +
        "bytes hold.",
    );
  }
  const name = prefix === "float32" && count === 1 ? "float32" : `${prefix}x${count}`;
  if (!(name in VERTEX_FORMATS)) {
    throw new Error(
      `rmsl: no vertex format "${name}" for a ${arrayTypeName(attr.array)} ` +
        `attribute of ${count} component${count === 1 ? "" : "s"}. A buffer ` +
        "carrying one attribute must have a stride that is a multiple of four, " +
        "so a narrow attribute packs into the spare lanes of a four-component " +
        "one rather than taking a buffer of its own.",
    );
  }
  return name as VertexFormat;
}

/** What to call an attribute's array in a message about it. */
function arrayTypeName(array: ArrayLike<number>): string {
  return ArrayBuffer.isView(array) ? array.constructor.name : "number[]";
}

/**
 * The meshes of one frame in the order a renderer draws them: the opaque ones
 * in scene-graph order, then the transparent ones from the farthest to the
 * nearest, so each blends over what lies behind it, as three.js orders them.
 * Kept by a renderer across frames, so ordering a frame allocates nothing.
 */
export class FrameOrder {
  /** The meshes to draw, in order, after `sort`. */
  readonly meshes: Mesh[] = [];
  private readonly transparent: Mesh[] = [];
  private readonly depths: number[] = [];
  private readonly position = new Vector3();

  /** Empties the order for a new frame. */
  clear(): void {
    this.meshes.length = this.transparent.length = this.depths.length = 0;
  }

  /** Adds a mesh of the frame, in scene-graph order. */
  add(mesh: Mesh): void {
    if ((mesh.material as { transparent?: boolean }).transparent) this.transparent.push(mesh);
    else this.meshes.push(mesh);
  }

  /** Appends the transparent meshes to `meshes`, the farthest from `camera` first. */
  sort(camera: Camera): void {
    const { transparent, depths, position } = this;
    for (let i = 0; i < transparent.length; i++) {
      // The view-space z, which grows towards the camera.
      depths[i] = position.setFromMatrixPosition(transparent[i]!.matrixWorld).applyMatrix4(camera.matrixWorldInverse).z;
    }
    // An insertion sort, stable and in place, over the few transparent meshes a frame holds.
    for (let i = 1; i < transparent.length; i++) {
      const mesh = transparent[i]!;
      const depth = depths[i]!;
      let j = i - 1;
      while (j >= 0 && depths[j]! > depth) {
        transparent[j + 1] = transparent[j]!;
        depths[j + 1] = depths[j]!;
        j--;
      }
      transparent[j + 1] = mesh;
      depths[j + 1] = depth;
    }
    for (const mesh of transparent) this.meshes.push(mesh);
  }
}

/** The colour `render` clears to: the scene's background when it has one, and the renderer's clear colour otherwise. */
export function clearColourOf(
  scene: Scene,
  clearColor: Color,
  clearAlpha: number,
): { r: number; g: number; b: number; a: number } {
  const colour = scene.background ?? clearColor;
  CLEAR.r = colour.r;
  CLEAR.g = colour.g;
  CLEAR.b = colour.b;
  CLEAR.a = scene.background ? 1 : clearAlpha;
  return CLEAR;
}

/** The value `clearColourOf` hands back, reused across frames. */
const CLEAR = { r: 0, g: 0, b: 0, a: 1 };

/**
 * The ranges of `attribute` to upload, sorted and merged where they touch, as
 * three.js's `WebGLAttributes` merges them, in place. Empty means the whole
 * attribute.
 */
export function mergedUpdateRanges(attribute: BufferAttribute): readonly { start: number; count: number }[] {
  const ranges = attribute.updateRanges;
  ranges.sort((a, b) => a.start - b.start);
  let merged = 0;
  for (let i = 1; i < ranges.length; i++) {
    const previous = ranges[merged]!;
    const range = ranges[i]!;
    if (range.start <= previous.start + previous.count + 1) {
      previous.count = Math.max(previous.count, range.start + range.count - previous.start);
    } else {
      ranges[++merged] = range;
    }
  }
  if (ranges.length > 0) ranges.length = merged + 1;
  return ranges;
}

/** The parts of an image element, video, bitmap or canvas that say whether it has loaded and how big it is. */
type ImageLike = {
  complete?: boolean;
  readyState?: number;
  naturalWidth?: number;
  naturalHeight?: number;
  videoWidth?: number;
  videoHeight?: number;
  displayWidth?: number;
  displayHeight?: number;
  width?: number;
  height?: number;
};

/** A video's `readyState` once it has a frame to draw, `HTMLMediaElement.HAVE_CURRENT_DATA`. */
const HAVE_CURRENT_DATA = 2;

/** The width of a loaded image source: a video's frame, a `VideoFrame`'s display size, an image's own pixels, or a bitmap's or canvas's size. */
export function imageWidth(image: unknown): number {
  const i = image as ImageLike;
  return i.videoWidth ?? i.displayWidth ?? i.naturalWidth ?? i.width ?? 0;
}

/** The height of a loaded image source, as `imageWidth` reads the width. */
export function imageHeight(image: unknown): number {
  const i = image as ImageLike;
  return i.videoHeight ?? i.displayHeight ?? i.naturalHeight ?? i.height ?? 0;
}

/**
 * Whether an image source a texture holds has loaded: an image element that
 * is complete, a video with a frame, and any source with a size.
 */
export function imageLoaded(image: unknown): boolean {
  const i = image as ImageLike;
  if (i.complete === false || (i.readyState !== undefined && i.readyState < HAVE_CURRENT_DATA)) return false;
  return imageWidth(image) > 0 && imageHeight(image) > 0;
}

/** The slice `drawSlice` gives, reused so a draw allocates nothing. */
const slice = { start: 0, count: 0 };

/**
 * The elements a draw range selects out of `total`, cut to them as three.js
 * cuts it; an infinite `total` leaves the range uncut. A count of 0 means the
 * range selects none, or has no end, and the draw is skipped. The result is
 * reused by the next call.
 */
export function drawSlice(range: { start: number; count: number }, total: number): Readonly<typeof slice> {
  slice.start = Math.max(range.start, 0);
  const count = Math.min(range.start + range.count, total) - slice.start;
  slice.count = Number.isFinite(count) ? Math.max(0, count) : 0;
  return slice;
}

/** The GPU buffer a renderer keeps for one attribute, the version of the attribute it holds, and the bytes of each element it holds. */
export interface AttributeBuffer<Buffer> {
  buffer: Buffer;
  version: number;
  elementSize: number;
}

/**
 * The version of each attribute at which a renderer last cleared its update
 * ranges. A renderer whose buffer holds an older version lacks changes whose
 * ranges another renderer cleared.
 */
const rangesClearedAt = new WeakMap<BufferAttribute, number>();

/**
 * Whether the update ranges marked on `attribute` cover every change a buffer
 * that holds it at `version` lacks. Otherwise the attribute uploads whole.
 */
export function rangesCover(attribute: BufferAttribute, version: number): boolean {
  return version >= (rangesClearedAt.get(attribute) ?? -1);
}

/** Clears the update ranges of `attribute` once a renderer uploaded them, as three.js clears them. */
export function clearUpdateRanges(attribute: BufferAttribute): void {
  attribute.clearUpdateRanges();
  rangesClearedAt.set(attribute, attribute.version);
}
