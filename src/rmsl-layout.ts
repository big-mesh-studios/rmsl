// === Shared memory-layout allocator (see docs/design-shared-layout-ir.md) ===
//
// Stage 1 of that design: extract the placement algorithm `wgslUniformLayout`
// (src/backends/rmsl-wgsl.ts) already implements — reorder members by
// alignment, accumulate offsets with padding, widen/round array elements —
// into one function, configurable per target instead of hard-coded into that
// one backend. `src/backends/rmsl-wasm.ts`'s Phase 3 memory allocator uses it
// too, with its own rules (no reordering, no padding), so both are now thin
// callers of the same algorithm instead of two independent implementations
// that happen to overlap in what they're actually deciding.
//
// Deliberately generic over the *type string* a member carries: WGSL's
// existing caller already spells types its own way (`"vec3<f32>"`) and
// WASM's spells them RMSL's way (`"vec3"`) — unifying that vocabulary is a
// separate, later step (see the design doc's "stage 2"), not something this
// extraction needs to do to prove the algorithm itself is shared correctly.

export type LayoutMember = { slot: string; type: string; length?: number };

export type PlacedLayoutMember = LayoutMember & {
  offset: number;
  size: number;
  /** Present only when `length` is — bytes between consecutive elements,
   * which can differ from one element's own size (WGSL rounds an array
   * element's stride up to 16 in the uniform address space). */
  stride?: number;
};

export type AllocRules = {
  /** The base size/alignment of one value of `type`, ignoring array-ness —
   * entirely rules-defined: WGSL rules interpret `type` as a WGSL type
   * spelling, packed/CPU rules as an RMSL `ShaderType`. */
  sizeAndAlignOf(type: string): { size: number; align: number };
  /** Reorder members by descending alignment before placing them, to
   * minimize padding (what a GPU uniform/storage buffer wants). `false`
   * keeps declaration order (what a bump allocator with no cross-member
   * padding concern wants). */
  reorderByAlignment: boolean;
  /** An array element too narrow to meet the array alignment rule is
   * stored as something wider instead (WGSL widens a bare `f32` array
   * element to `vec4<f32>`, for example) — returns the type actually
   * stored, or the input type unchanged if this target has no such rule. */
  widenNarrowArrayElements?: (type: string) => string;
  /** An array element's stride is rounded up to this many bytes (WGSL: 16).
   * Omitted where there's no such rule. */
  arrayStrideRoundedTo?: number;
  /** The whole placed struct's own alignment is at least this (WGSL: 4;
   * packed/CPU rules that never round anything: 1). */
  structAlignMinimum: number;
};

/**
 * Place `members` one after another under `rules`, returning each member's
 * offset (and, for an array member, its element stride) plus the total
 * size/alignment of the whole placement.
 */
export function planLayout(
  members: LayoutMember[],
  rules: AllocRules,
): { members: PlacedLayoutMember[]; size: number; align: number } {
  const shapeOf = (m: LayoutMember): { size: number; align: number; stride: number } => {
    if (m.length === undefined) {
      const { size, align } = rules.sizeAndAlignOf(m.type);
      return { size, align, stride: size };
    }
    const storedType = rules.widenNarrowArrayElements ? rules.widenNarrowArrayElements(m.type) : m.type;
    const base = rules.sizeAndAlignOf(storedType);
    const stride = rules.arrayStrideRoundedTo
      ? Math.ceil(base.size / rules.arrayStrideRoundedTo) * rules.arrayStrideRoundedTo
      : base.size;
    const align = rules.arrayStrideRoundedTo ? Math.max(base.align, rules.arrayStrideRoundedTo) : base.align;
    return { size: stride * m.length, align, stride };
  };

  // Widest alignment first, so the gaps between members stay small. Members
  // that align the same keep the order they were declared in. Skipped
  // entirely when `rules` says not to reorder — declaration order is the
  // whole point there, not an incidental side effect of a stable sort.
  const ordered = rules.reorderByAlignment
    ? members
        .map((m, declaredAt) => ({ m, declaredAt }))
        .sort((a, b) => {
          const byAlign = shapeOf(b.m).align - shapeOf(a.m).align;
          return byAlign !== 0 ? byAlign : a.declaredAt - b.declaredAt;
        })
        .map(({ m }) => m)
    : members;

  const out: PlacedLayoutMember[] = [];
  let offset = 0;
  for (const m of ordered) {
    const { size, align, stride } = shapeOf(m);
    offset = Math.ceil(offset / align) * align;
    out.push({ ...m, offset, size, ...(m.length !== undefined ? { stride } : {}) });
    offset += size;
  }
  // The whole placement is itself aligned to its widest member (or the
  // rules' minimum, for an empty list) — an array member aligns to its own
  // rounded-up alignment, not its element type's, which `shapeOf` already
  // accounts for.
  const structAlign = ordered.reduce((a, m) => Math.max(a, shapeOf(m).align), rules.structAlignMinimum);
  return { members: out, size: Math.ceil(offset / structAlign) * structAlign, align: structAlign };
}
