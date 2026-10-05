import { MATRIX_DIMENSIONS } from "../../core";
import type { TypedArray } from "../adapter";

/**
 * Where an attribute's values sit in its GPU buffer, in 32-bit slots. WGSL
 * gives a `vec3`, and each column of three in a matrix, the room of four, so
 * those leave one slot empty after every three values; every other element
 * type is laid out as the attribute holds it.
 */
export type StorageLayout = {
  /** Slots per element. */
  stride: number;
  /** Whether every value sits in the slot of its own index, so nothing needs spreading. */
  identity: boolean;
  /** The slot holding the attribute's value `k`. */
  slot(k: number): number;
};

/**
 * The layout of a storage element of `itemSize` components. `elementType` names a matrix, whose columns of
 * three take the room of four, and is left out for a vector, whose own size decides. `identity` says that
 * every value sits in the slot of its own index, so the values go to the buffer unchanged.
 */
export function storageLayout(attribute: { itemSize: number; elementType?: string | null }): StorageLayout {
  const { itemSize, elementType } = attribute;
  const rows = (elementType && MATRIX_DIMENSIONS[elementType]?.[1]) ?? itemSize;
  if (rows !== 3) return { stride: itemSize, identity: true, slot: (k) => k };
  const stride = (itemSize / 3) * 4;
  return {
    stride,
    identity: false,
    slot: (k) => Math.floor(k / itemSize) * stride + Math.floor((k % itemSize) / 3) * 4 + (k % 3),
  };
}

/** `values`, the attribute's values from value `first` on, spread into the slots `layout` gives them. */
export function spread(values: TypedArray, first: number, layout: StorageLayout): { slot: number; data: TypedArray } {
  if (layout.identity || values.length === 0) return { slot: layout.slot(first), data: values };
  const slot = layout.slot(first);
  const data = new (values.constructor as Float32ArrayConstructor)(layout.slot(first + values.length - 1) - slot + 1);
  for (let k = 0; k < values.length; k++) data[layout.slot(first + k) - slot] = values[k]!;
  return { slot, data };
}
