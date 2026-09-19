/** Texel layout of a texture's image data, like three.js's texture constants. */
export const RGBAFormat = 1023;
/** Single-channel unsigned-integer red texel layout, like three.js's `RedIntegerFormat`. */
export const RedIntegerFormat = 36244;

/** The element type of the image data, like three.js's `UnsignedByteType`. */
export const UnsignedByteType = 1009;

// What a sampler does with a coordinate outside 0..1. Numbers are three.js's own.

/** The image tiles: the fractional part of the coordinate is used. */
export const RepeatWrapping = 1000;
/** The edge texel is stretched outwards. The default. */
export const ClampToEdgeWrapping = 1001;
/** The image tiles, flipping on every other repeat. */
export const MirroredRepeatWrapping = 1002;

// How a texel is chosen between texels (`magFilter`) or when minified (`minFilter`).

/** The nearest texel, so texels stay square — what pixel art wants. */
export const NearestFilter = 1003;
/** A weighted average of the surrounding texels. The default. */
export const LinearFilter = 1006;

// Mipmapped minification filters, accepted for compatibility but each treated
// as its base filter — no renderer here builds a mip chain yet (rmsl#3).
export const NearestMipmapNearestFilter = 1004;
export const NearestMipmapLinearFilter = 1005;
export const LinearMipmapNearestFilter = 1007;
export const LinearMipmapLinearFilter = 1008;
