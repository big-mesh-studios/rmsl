import { clamp, dot, float, mix, mul, screenUV, select, sin, time, uv, vec2, vec3, vec4, type Node } from "../rmsl";
import { type FloatIn, type Sampler2D, type Vec2In } from "./util";
import { circle } from "./shape";

/**
 * Creates barrel-distorted UV coordinates. The center of the screen appears to
 * bulge outward (convex distortion).
 *
 * Ported from three.js's `examples/jsm/tsl/display/CRT.js`.
 *
 * @param curvature - The amount of curvature (0 = flat, 0.5 = very curved).
 * @param coord - The input UV coordinates.
 * @return The distorted UV coordinates.
 */
export const barrelUV = (
  curvature: FloatIn = 0.1,
  coord: Vec2In | null = null,
): Node<"vec2"> => {
  const c = coord === null ? uv() : Array.isArray(coord) ? vec2(coord[0], coord[1]) : coord;
  // Center the UV coordinates (-1 to 1).
  const centered = c.sub(0.5).mul(2.0);
  // Squared distance from center.
  const r2 = dot(centered, centered);
  // Barrel distortion: push the center outward.
  const distortion = float(1.0).sub(r2.mul(curvature));
  // Scale to compensate for edge expansion; at corners r² = 2.
  const cornerDistortion = float(1.0).sub(mul(curvature, 2.0));
  return centered.div(distortion).mul(cornerDistortion).mul(0.5).add(0.5);
};

/**
 * Checks if UV coordinates are inside the valid 0-1 range.
 *
 * @param coord - The UV coordinates to check.
 * @return 1.0 if inside bounds, 0.0 if outside.
 */
export const barrelMask = (coord: Node<"vec2">): Node<"float"> => {
  const outOfBounds = coord.x.lessThan(0.0)
    .or(coord.x.greaterThan(1.0))
    .or(coord.y.lessThan(0.0))
    .or(coord.y.greaterThan(1.0));
  return select(outOfBounds, float(0.0), float(1.0));
};

/**
 * Applies color bleeding to simulate horizontal color smearing — the analog
 * signal trailing of a CRT display.
 *
 * @param color - The input texture node.
 * @param amount - The amount of color bleeding (0-0.01).
 * @return The color with bleeding applied.
 */
export const colorBleeding = (
  color: Sampler2D,
  amount: FloatIn = 0.002,
): Node<"vec3"> => {
  const original = color.texture(screenUV()).rgb;
  const left1 = color.texture(screenUV().sub(vec2(amount, 0.0))).rgb;
  const left2 = color.texture(screenUV().sub(vec2(mul(amount, 2.0), 0.0))).rgb;
  const left3 = color.texture(screenUV().sub(vec2(mul(amount, 3.0), 0.0))).rgb;

  // Red bleeds most (travels furthest in an analog signal).
  const bleedR = original.r.add(left1.r.mul(0.4)).add(left2.r.mul(0.2)).add(left3.r.mul(0.1));
  // Green bleeds medium.
  const bleedG = original.g.add(left1.g.mul(0.25)).add(left2.g.mul(0.1));
  // Blue bleeds least.
  const bleedB = original.b.add(left1.b.mul(0.15));

  const r = clamp(bleedR.div(1.7), 0.0, 1.0);
  const g = clamp(bleedG.div(1.35), 0.0, 1.0);
  const b = clamp(bleedB.div(1.15), 0.0, 1.0);

  return vec3(r, g, b);
};

/**
 * Applies scanlines — CRT horizontal lines with optional animation.
 *
 * @param color - The input color.
 * @param intensity - The intensity of the scanlines (0-1).
 * @param count - The number of scanlines (typically matches the vertical resolution).
 * @param speed - The scroll speed of the scanlines (0 = static).
 * @param coord - The UV coordinates to use for the scanlines.
 * @return The color with scanlines applied.
 */
export const scanlines = (
  color: Node<"vec3">,
  intensity: FloatIn = 0.3,
  count: FloatIn = 240,
  speed: FloatIn = 0.0,
  coord: Vec2In | null = null,
): Node<"vec3"> => {
  const c = coord === null ? uv() : Array.isArray(coord) ? vec2(coord[0], coord[1]) : coord;
  const animatedY = c.y.sub(time().mul(speed));
  const scanline = sin(animatedY.mul(count));
  const scanlineIntensity = scanline.mul(0.5).add(0.5).mul(intensity);
  return color.mul(float(1.0).sub(scanlineIntensity));
};

/**
 * Applies a vignette to darken the edges of the screen.
 *
 * @param color - The input color.
 * @param intensity - The intensity of the vignette (0-1).
 * @param smoothness - The smoothness of the vignette falloff.
 * @param coord - The UV coordinates to use for the vignette calculation.
 * @return The vignetted color.
 */
export const vignette = (
  color: Node<"vec3">,
  intensity: FloatIn = 0.4,
  smoothness: FloatIn = 0.5,
  coord: Vec2In | null = null,
): Node<"vec3"> => {
  // 1.42 ≈ √2 covers the full diagonal.
  const mask = circle(1.42, smoothness, coord);
  const vignetteAmount = mix(float(1.0).sub(intensity), float(1.0), mask);
  return color.mul(vignetteAmount);
};

export interface CrtOptions {
  curvature?: FloatIn;
  scanlineIntensity?: FloatIn;
  scanlineCount?: FloatIn;
  scanlineSpeed?: FloatIn;
  vignetteIntensity?: FloatIn;
  vignetteSmoothness?: FloatIn;
  colorBleedingAmount?: FloatIn;
}

/**
 * The CRT effects composed into one pass: barrel distortion, color bleeding,
 * scanlines and vignette. The input is a texture sampled at the distorted UVs;
 * areas pushed outside the frame are blackened by the barrel mask.
 *
 * @param textureNode - The input texture node.
 * @param options - Tuning knobs for the sub-effects.
 * @return The CRT-styled color.
 */
export const crt = (
  textureNode: Sampler2D,
  options: CrtOptions = {},
): Node<"vec4"> => {
  const distorted = barrelUV(options.curvature ?? 0.1);
  const masked = barrelMask(distorted);
  const bled = colorBleeding(textureNode, options.colorBleedingAmount ?? 0.002);
  const scanned = scanlines(
    bled,
    options.scanlineIntensity ?? 0.3,
    options.scanlineCount ?? 240,
    options.scanlineSpeed ?? 0.0,
    distorted,
  );
  const vignetted = vignette(
    scanned,
    options.vignetteIntensity ?? 0.4,
    options.vignetteSmoothness ?? 0.5,
    distorted,
  );
  const alpha = textureNode.texture(distorted).a;
  return vec4(vignetted.mul(masked), alpha);
};
