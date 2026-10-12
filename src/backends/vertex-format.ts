import { MATRIX_DIMENSIONS } from "../core";
import { componentKindOf } from "./cpu";

/** The kind of the components a vertex attribute reads. */
export type AttributeKind = "float" | "int" | "uint";

/** The kind of the components an attribute of the RMSL type `type` reads. GLSL has no boolean attribute, so a `bool` reads floats. */
export function attributeKindOf(type: string): AttributeKind {
  const kind = componentKindOf(type);
  return kind === "bool" ? "float" : kind;
}

/** The bytes of one vertex attribute: its array, its components a vertex, whether they scale on the way in, and a format it declares. */
export interface VertexSource {
  array: ArrayLike<number>;
  itemSize: number;
  normalized: boolean;
  format?: VertexFormat;
}

/**
 * One vertex attribute's layout, under the name WebGPU gives it. The set is
 * limited to the formats that reach a shader as the same type on both
 * backends: a material asks for a `vec4` and gets one from `unorm8x4` under
 * either backend, and asks for a `uvec4` and gets one from `uint8x4`.
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
  | "unorm16x4"
  | "sint8x4"
  | "uint8x4"
  | "sint16x2"
  | "sint16x4"
  | "uint16x2"
  | "uint16x4"
  | "sint32"
  | "sint32x2"
  | "sint32x3"
  | "sint32x4"
  | "uint32"
  | "uint32x2"
  | "uint32x3"
  | "uint32x4";

/** What one vertex format is made of, in the terms each backend binds it by. */
export interface VertexFormatSpec {
  /** Components one vertex holds. */
  count: number;
  /** Bytes one component occupies. */
  bytes: number;
  /** Whether the stored integer is scaled into 0..1 or -1..1 on the way in. */
  normalized: boolean;
  /** The kind of the shader input the format feeds. */
  kind: AttributeKind;
  /** The WebGL component type, by its `WebGL2RenderingContext` key. */
  gl: "BYTE" | "UNSIGNED_BYTE" | "SHORT" | "UNSIGNED_SHORT" | "INT" | "UNSIGNED_INT" | "HALF_FLOAT" | "FLOAT";
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
  float32: { count: 1, bytes: 4, normalized: false, kind: "float", gl: "FLOAT" },
  float32x2: { count: 2, bytes: 4, normalized: false, kind: "float", gl: "FLOAT" },
  float32x3: { count: 3, bytes: 4, normalized: false, kind: "float", gl: "FLOAT" },
  float32x4: { count: 4, bytes: 4, normalized: false, kind: "float", gl: "FLOAT" },
  float16x2: { count: 2, bytes: 2, normalized: false, kind: "float", gl: "HALF_FLOAT" },
  float16x4: { count: 4, bytes: 2, normalized: false, kind: "float", gl: "HALF_FLOAT" },
  snorm8x4: { count: 4, bytes: 1, normalized: true, kind: "float", gl: "BYTE" },
  unorm8x4: { count: 4, bytes: 1, normalized: true, kind: "float", gl: "UNSIGNED_BYTE" },
  snorm16x2: { count: 2, bytes: 2, normalized: true, kind: "float", gl: "SHORT" },
  snorm16x4: { count: 4, bytes: 2, normalized: true, kind: "float", gl: "SHORT" },
  unorm16x2: { count: 2, bytes: 2, normalized: true, kind: "float", gl: "UNSIGNED_SHORT" },
  unorm16x4: { count: 4, bytes: 2, normalized: true, kind: "float", gl: "UNSIGNED_SHORT" },
  sint8x4: { count: 4, bytes: 1, normalized: false, kind: "int", gl: "BYTE" },
  uint8x4: { count: 4, bytes: 1, normalized: false, kind: "uint", gl: "UNSIGNED_BYTE" },
  sint16x2: { count: 2, bytes: 2, normalized: false, kind: "int", gl: "SHORT" },
  sint16x4: { count: 4, bytes: 2, normalized: false, kind: "int", gl: "SHORT" },
  uint16x2: { count: 2, bytes: 2, normalized: false, kind: "uint", gl: "UNSIGNED_SHORT" },
  uint16x4: { count: 4, bytes: 2, normalized: false, kind: "uint", gl: "UNSIGNED_SHORT" },
  sint32: { count: 1, bytes: 4, normalized: false, kind: "int", gl: "INT" },
  sint32x2: { count: 2, bytes: 4, normalized: false, kind: "int", gl: "INT" },
  sint32x3: { count: 3, bytes: 4, normalized: false, kind: "int", gl: "INT" },
  sint32x4: { count: 4, bytes: 4, normalized: false, kind: "int", gl: "INT" },
  uint32: { count: 1, bytes: 4, normalized: false, kind: "uint", gl: "UNSIGNED_INT" },
  uint32x2: { count: 2, bytes: 4, normalized: false, kind: "uint", gl: "UNSIGNED_INT" },
  uint32x3: { count: 3, bytes: 4, normalized: false, kind: "uint", gl: "UNSIGNED_INT" },
  uint32x4: { count: 4, bytes: 4, normalized: false, kind: "uint", gl: "UNSIGNED_INT" },
};

/**
 * The format family an attribute's array belongs to, or nothing where no format
 * reads it. A plain `number[]` is whatever `toBufferView` will make of it, which
 * is a `Float32Array`.
 */
function formatPrefix(array: ArrayLike<number>, normalized: boolean): string | undefined {
  if (!ArrayBuffer.isView(array)) return "float32";
  if (array instanceof Float32Array) return "float32";
  if (typeof Float16Array !== "undefined" && array instanceof Float16Array) return "float16";
  if (array instanceof Int8Array) return normalized ? "snorm8" : "sint8";
  if (array instanceof Uint8Array) return normalized ? "unorm8" : "uint8";
  if (array instanceof Int16Array) return normalized ? "snorm16" : "sint16";
  if (array instanceof Uint16Array) return normalized ? "unorm16" : "uint16";
  if (normalized) return undefined;
  if (array instanceof Int32Array) return "sint32";
  if (array instanceof Uint32Array) return "uint32";
  return undefined;
}

/**
 * How an attribute of the shader type `type` takes its locations: one for each
 * column of a matrix, and one for anything else, each reading components of
 * `kind`.
 */
export function attributeShape(type: string): { columns: number; kind: AttributeKind } {
  return { columns: MATRIX_DIMENSIONS[type]?.[0] ?? 1, kind: attributeKindOf(type) };
}

/**
 * The vertex format each location of an attribute declared as `type` reads its
 * bytes in: the attribute's own `format` where it declares one, and otherwise
 * the format its array type, component count and `normalized` flag imply. A
 * matrix's locations each read one column of its item.
 *
 * @throws When the bytes have no format in `VERTEX_FORMATS`, or have one of
 *   another kind than `type`'s components. Both are silent bugs otherwise: the
 *   bytes get read as something they are not, or the draw is refused.
 */
export function vertexFormatOf(attr: VertexSource, type: string): VertexFormat {
  const { columns, kind } = attributeShape(type);
  const count = attr.itemSize / columns;
  const named = attr.format ?? formatName(attr, count);
  const format = VERTEX_FORMATS[named];
  if (format === undefined) throw new Error(`rmsl: no vertex format "${named}"`);
  if (format.kind !== kind) {
    throw new Error(
      `rmsl: ${type.startsWith("i") ? "an" : "a"} ${type} attribute cannot read its ${arrayTypeName(attr.array)} in the format ${named}, ` +
        `which holds ${format.kind === "float" ? "floats" : `${format.kind} components`}. ` +
        (kind === "float"
          ? "An integer array reaches a float attribute only when it is scaled on the way in: set `normalized: true`, or set `format` to say what its bytes hold."
          : `Give it an ${kind === "int" ? "Int32Array" : "Uint32Array"}, or set \`format\` to a ${kind === "int" ? "sint" : "uint"} format.`),
    );
  }
  return named;
}

/** The format name `attr`'s array implies for `count` components a location. */
function formatName(attr: VertexSource, count: number): VertexFormat {
  const prefix = formatPrefix(attr.array, attr.normalized);
  if (prefix === undefined) {
    throw new Error(
      `rmsl: a normalized ${arrayTypeName(attr.array)} attribute has no vertex format. ` +
        "Only an 8- or 16-bit integer array is scaled on the way in.",
    );
  }
  const name = count === 1 && prefix.endsWith("32") ? prefix : `${prefix}x${count}`;
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

/** The format family of each kind a 32-bit vertex component can feed. */
const WORD_PREFIX: Record<AttributeKind, string> = { float: "float32", int: "sint32", uint: "uint32" };

/** The format a location of `kind` and `count` components reads when it has no data: 32-bit components of its kind. */
export function wordFormat(kind: AttributeKind, count: number): VertexFormat {
  return (count === 1 ? WORD_PREFIX[kind] : `${WORD_PREFIX[kind]}x${count}`) as VertexFormat;
}

/** What to call an attribute's array in a message about it. */
function arrayTypeName(array: ArrayLike<number>): string {
  return ArrayBuffer.isView(array) ? array.constructor.name : "number[]";
}
