import {Fn, float, premultiplyAlpha, textureSize, unpremultiplyAlpha, uniform, uv, vec2, type Node} from "../rmsl";
import { f, type FloatIn, type IntIn, type Vec2In, type Sampler2D, type Sampler3D } from "./util";

/**
 * One fullscreen pass of a multi-pass effect: its fragment color and the
 * samplers it reads. `inputs` keys are caller-chosen names; an executor binds
 * a texture to each, so pass N can read pass N-1's render target.
 *
 * A `PassDescriptor` is pure data — the color is an RMSL node graph that
 * compiles with `compileGLSL.fragment`/`compileWGSL.fragment`. No render
 * target or GL context lives inside rmsl; your own render loop draws one
 * fullscreen quad per pass and binds the previous pass's output texture to the
 * next pass's input.
 */
export interface PassDescriptor {
  /** Unique pass name; also the key an executor writes the result under. */
  name: string;
  /** The pass's fragment color, sampled from its `inputs`. */
  color: Node<"vec4">;
  /** The samplers this pass samples, by caller-chosen name. */
  inputs: Record<string, Sampler2D>;
  /**
   * Render target size in pixels. When absent the pass renders at the size of
   * its first input texture (or the drawing buffer when it has no inputs),
   * scaled by `scale` if that is set too.
   */
  size?: [number, number];
  /**
   * Render target size as a fraction of the pass's first input texture — e.g.
   * a `scale` of `0.5` renders at half the resolution of its input. Ignored
   * when `size` is set. Multi-pass effects whose passes progressively shrink
   * (bloom's mip chain) set this so the executor can size the targets.
   */
  scale?: number;
}

/** An ordered list of passes whose last output is the effect's result. */
export interface PassGraph {
  passes: PassDescriptor[];
  /** Name of the pass whose output is the graph's result. */
  output: string;
}

export interface GaussianBlurOptions {
  /** Whether to blur in premultiplied alpha. */
  premultipliedAlpha?: boolean;
}

/**
 * A separable gaussian blur, as a two-pass graph (horizontal then vertical).
 *
 * Ported from three.js's `examples/jsm/tsl/display/GaussianBlurNode.js`. The
 * returned graph is executed by your render loop: draw pass 0's color into a
 * render target, bind that texture to pass 1's input, draw pass 1, and the
 * result is the blurred image.
 *
 * @param textureNode - The texture node to blur.
 * @param direction - Blur radius along each axis; `[1, 1]` blurs both.
 * @param sigma - Controls the kernel; higher values mean a wider blur radius.
 * @param options - Options for the effect.
 * @return The two-pass blur graph.
 */
export const gaussianBlur = (
  textureNode: Sampler2D,
  direction: [number, number] = [1, 1],
  sigma = 4,
  options: GaussianBlurOptions = {},
): PassGraph => {
  const premultipliedAlpha = options.premultipliedAlpha ?? false;
  const horizontal = gaussianBlurPass(
    "gaussianBlur.horizontal",
    textureNode,
    [direction[0], 0],
    sigma,
    premultipliedAlpha,
  );
  const verticalInput = uniform("sampler2D");
  const vertical = gaussianBlurPass(
    "gaussianBlur.vertical",
    verticalInput,
    [0, direction[1]],
    sigma,
    premultipliedAlpha,
  );
  return { passes: [horizontal, vertical], output: "gaussianBlur.vertical" };
};

/**
 * A gaussian blur with premultiplied alpha, as a two-pass graph.
 *
 * @deprecated Use `gaussianBlur()` with `{ premultipliedAlpha: true }`.
 */
export const premultipliedGaussianBlur = (
  textureNode: Sampler2D,
  direction: [number, number] = [1, 1],
  sigma = 4,
): PassGraph => gaussianBlur(textureNode, direction, sigma, { premultipliedAlpha: true });

function gaussianBlurPass(
  name: string,
  inputTex: Sampler2D,
  direction: [number, number],
  sigma: number,
  premultipliedAlpha: boolean,
): PassDescriptor {
  const kernelSize = 3 + 2 * sigma;
  const coefficients = getGaussianCoefficients(kernelSize);

  const color = Fn(() => {
    const uvNode = uv();
    const invSize = vec2(1).div(textureSize(inputTex).toVec2());
    const dir = vec2(direction[0], direction[1]);
    const sample = (u: Node<"vec2">): Node<"vec4"> => {
      const s = inputTex.texture(u);
      return premultipliedAlpha ? premultiplyAlpha(s) : s;
    };

    const diffuseSum = sample(uvNode).mul(coefficients[0]).toVar();

    for (let i = 1; i < kernelSize; i++) {
      const x = float(i);
      const w = float(coefficients[i]);
      const uvOffset = dir.mul(invSize.mul(x)).toVar();
      const sample1 = sample(uvNode.add(uvOffset));
      const sample2 = sample(uvNode.sub(uvOffset));
      diffuseSum.addAssign(sample1.add(sample2).mul(w));
    }

    return premultipliedAlpha ? unpremultiplyAlpha(diffuseSum) : diffuseSum;
  });

  return { name, color: color(), inputs: { input: inputTex } };
}

/**
 * Coefficients of a gaussian kernel of `kernelRadius` taps, computed on the
 * host as three.js does: `0.39894 · exp(-0.5·i²/σ²)/σ` with `σ = kernelRadius/3`.
 */
export function getGaussianCoefficients(kernelRadius: number): number[] {
  const coefficients: number[] = [];
  const sigma = kernelRadius / 3;
  for (let i = 0; i < kernelRadius; i++) {
    coefficients.push(0.39894 * Math.exp(-0.5 * i * i / (sigma * sigma)) / sigma);
  }
  return coefficients;
}
