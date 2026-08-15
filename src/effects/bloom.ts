import { Fn, add, float, luminance, mix, smoothstep, textureSize, uniform, uv, vec2, vec3, vec4, type Node } from "../rmsl";
import { type FloatIn, type Sampler2D, type Vec3In, vec3In } from "./util";
import { getGaussianCoefficients, type PassDescriptor, type PassGraph } from "./passes";

/**
 * The luminance high-pass filter: keeps only the areas brighter than
 * `threshold`, with a `smoothWidth`-wide soft edge. This is what three.js's
 * `bloom()` uses by default; a custom filter (e.g. an anamorphic one) can be
 * supplied through {@link BloomOptions.highPassFn}.
 *
 * @param input - The sampled scene color.
 * @param threshold - Luminance above which the pixel contributes to bloom.
 * @param smoothWidth - Edge softness of the threshold.
 * @return The bright-pass color.
 */
export type HighPassFn = (
  input: Node<"vec4">,
  threshold: FloatIn,
  smoothWidth: FloatIn,
) => Node<"vec4">;

export const luminosityHighPass: HighPassFn = (input, threshold, smoothWidth) => {
  const alpha = smoothstep(threshold, add(threshold, smoothWidth), luminance(input.rgb));
  return mix(vec4(0), input, alpha);
};

export interface BloomOptions {
  /** The strength of the bloom. */
  strength?: FloatIn;
  /** The radius of the bloom, in `[0, 1]`. */
  radius?: FloatIn;
  /** The luminance threshold limiting which bright areas contribute. */
  threshold?: FloatIn;
  /** Edge softness of the luminance threshold. */
  smoothWidth?: FloatIn;
  /** A custom high-pass filter, e.g. for anamorphic bloom. */
  highPassFn?: HighPassFn;
  /** Per-mip tint colours, from the brightest mip to the dimmest. */
  tints?: readonly [Vec3In, Vec3In, Vec3In, Vec3In, Vec3In];
}

const BLOOM_FACTORS = [1.0, 0.8, 0.6, 0.4, 0.2];
const DEFAULT_TINTS: readonly [Vec3In, Vec3In, Vec3In, Vec3In, Vec3In] = [
  [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
];
const MIP_KERNELS = [6, 10, 14, 18, 22];

/**
 * A bloom effect, as a 12-pass graph: a luminance high pass, five progressively
 * half-resolution separable gaussian blurs (horizontal then vertical each),
 * and a composite that sums the five blurred mips.
 *
 * Ported faithfully from three.js's `examples/jsm/tsl/display/BloomNode.js`
 * (kernels `[6, 10, 14, 18, 22]`, factors `[1.0, 0.8, 0.6, 0.4, 0.2]`, mips
 * halving in size each step). The composite output is the glow on its own —
 * add it to your scene color, as three.js does with
 * `scenePassColor.add(bloomPass)`.
 *
 * Every pass's target size is expressed relative to its first input texture
 * via `PassDescriptor.scale`, so the executor sizes the render targets as the
 * mip chain shrinks.
 *
 * @param textureNode - The scene texture to bloom.
 * @param options - Options for the effect.
 * @return The bloom graph; `graph.output` is `"bloom.composite"`.
 */
export const bloom = (textureNode: Sampler2D, options: BloomOptions = {}): PassGraph => {
  const strength = options.strength ?? 1;
  const radius = options.radius ?? 0;
  const threshold = options.threshold ?? 0;
  const smoothWidth = options.smoothWidth ?? 0.01;
  const highPassFn = options.highPassFn ?? luminosityHighPass;
  const tints = options.tints ?? DEFAULT_TINTS;

  // One sampler per internal render target. The executor binds each pass's
  // `inputs` to the producer pass named by the key (see the docs); `input` on
  // the high pass is the external scene texture.
  const sHigh = uniform("sampler2D");
  const sH0 = uniform("sampler2D");
  const sV0 = uniform("sampler2D");
  const sH1 = uniform("sampler2D");
  const sV1 = uniform("sampler2D");
  const sH2 = uniform("sampler2D");
  const sV2 = uniform("sampler2D");
  const sH3 = uniform("sampler2D");
  const sV3 = uniform("sampler2D");
  const sH4 = uniform("sampler2D");
  const sV4 = uniform("sampler2D");

  const highPassColor = Fn(() => highPassFn(textureNode.texture(uv()), threshold, smoothWidth));

  const highpass: PassDescriptor = {
    name: "bloom.highpass",
    color: highPassColor(),
    inputs: { input: textureNode },
    scale: 0.5,
  };

  const mip0h = bloomBlurPass("bloom.mip0.horizontal", "bloom.highpass", sHigh, [1, 0], MIP_KERNELS[0], 1, 1);
  const mip0v = bloomBlurPass("bloom.mip0.vertical", "bloom.mip0.horizontal", sH0, [0, 1], MIP_KERNELS[0], 1, 1);
  const mip1h = bloomBlurPass("bloom.mip1.horizontal", "bloom.mip0.vertical", sV0, [1, 0], MIP_KERNELS[1], 2, 0.5);
  const mip1v = bloomBlurPass("bloom.mip1.vertical", "bloom.mip1.horizontal", sH1, [0, 1], MIP_KERNELS[1], 1, 1);
  const mip2h = bloomBlurPass("bloom.mip2.horizontal", "bloom.mip1.vertical", sV1, [1, 0], MIP_KERNELS[2], 2, 0.5);
  const mip2v = bloomBlurPass("bloom.mip2.vertical", "bloom.mip2.horizontal", sH2, [0, 1], MIP_KERNELS[2], 1, 1);
  const mip3h = bloomBlurPass("bloom.mip3.horizontal", "bloom.mip2.vertical", sV2, [1, 0], MIP_KERNELS[3], 2, 0.5);
  const mip3v = bloomBlurPass("bloom.mip3.vertical", "bloom.mip3.horizontal", sH3, [0, 1], MIP_KERNELS[3], 1, 1);
  const mip4h = bloomBlurPass("bloom.mip4.horizontal", "bloom.mip3.vertical", sV3, [1, 0], MIP_KERNELS[4], 2, 0.5);
  const mip4v = bloomBlurPass("bloom.mip4.vertical", "bloom.mip4.horizontal", sH4, [0, 1], MIP_KERNELS[4], 1, 1);

  const composite = bloomCompositePass(
    [
      ["bloom.mip0.vertical", sV0],
      ["bloom.mip1.vertical", sV1],
      ["bloom.mip2.vertical", sV2],
      ["bloom.mip3.vertical", sV3],
      ["bloom.mip4.vertical", sV4],
    ],
    radius,
    strength,
    tints,
  );

  return {
    passes: [highpass, mip0h, mip0v, mip1h, mip1v, mip2h, mip2v, mip3h, mip3v, mip4h, mip4v, composite],
    output: "bloom.composite",
  };
};

/**
 * A separable gaussian blur pass of the bloom mip chain. Unlike the standalone
 * `gaussianBlur`, the blur step is normalized to the *output* resolution, so
 * `pixelScale` scales the texel size from the sampled texture's own
 * dimensions — `pixelScale` is the ratio of the sampled texture's size to the
 * output's (`1` when the pass does not halve, `2` for the halving horizontal
 * passes). Alpha is dropped and forced to 1, as three.js does.
 */
function bloomBlurPass(
  name: string,
  producer: string,
  inputTex: Sampler2D,
  direction: [number, number],
  kernelRadius: number,
  pixelScale: number,
  scale: number,
): PassDescriptor {
  const coefficients = getGaussianCoefficients(kernelRadius);

  const color = Fn(() => {
    const uvNode = uv();
    const invSize = vec2(1).div(vec2(textureSize(inputTex).toVec2())).mul(pixelScale);
    const dir = vec2(direction[0], direction[1]);

    const diffuseSum = inputTex.texture(uvNode).rgb.mul(coefficients[0]).toVar();

    for (let i = 1; i < kernelRadius; i++) {
      const x = float(i);
      const w = float(coefficients[i]);
      const uvOffset = dir.mul(invSize.mul(x)).toVar();
      const sample1 = inputTex.texture(uvNode.add(uvOffset)).rgb;
      const sample2 = inputTex.texture(uvNode.sub(uvOffset)).rgb;
      diffuseSum.addAssign(sample1.add(sample2).mul(w));
    }

    return vec4(diffuseSum, 1.0);
  });

  return { name, color: color(), inputs: { [producer]: inputTex }, scale };
}

/** The composite pass: sum the five blurred mips, tinted and strength-scaled. */
function bloomCompositePass(
  mips: Array<[string, Sampler2D]>,
  radius: FloatIn,
  strength: FloatIn,
  tints: readonly [Vec3In, Vec3In, Vec3In, Vec3In, Vec3In],
): PassDescriptor {
  const color = Fn(() => {
    const uvNode = uv();
    let sum: Node<"vec4"> | null = null;

    for (let i = 0; i < BLOOM_FACTORS.length; i++) {
      const factor = BLOOM_FACTORS[i];
      const lerpFactor = mix(factor, float(1.2).sub(factor), radius);
      // The vec4 goes first: RMSL types a mul by its first operand, so a
      // float·vec4 would come out typed float and vanish from the output.
      const tinted = vec4(vec3In(tints[i]), 1.0).mul(lerpFactor);
      const mip = mips[i][1].texture(uvNode);
      sum = sum === null ? tinted.mul(mip) : sum.add(tinted.mul(mip));
    }

    return sum!.mul(strength);
  });

  const inputs: Record<string, Sampler2D> = {};
  for (const [producer, tex] of mips) inputs[producer] = tex;

  return { name: "bloom.composite", color: color(), inputs, scale: 1 };
}
