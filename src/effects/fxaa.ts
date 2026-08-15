import {For, Fn, If, abs, bool, clamp, dot, float, int, max, min, select, smoothstep, textureSize, uv, vec2, vec3, type Node} from "../rmsl";
import { f, type FloatIn, type IntIn, type Vec2In, type Sampler2D, type Sampler3D } from "./util";

const EDGE_STEP_COUNT = 6;
const EDGE_GUESS = 8.0;
// Per-edge-step sampling distances, as TSL carries them in a uniform array.
const EDGE_STEPS = [1.0, 1.5, 2.0, 2.0, 2.0, 4.0];

/**
 * Post-processing effect applying FXAA. Requires sRGB input, so tone mapping
 * and color space conversion must happen before the anti-aliasing.
 *
 * Ported from three.js's `examples/jsm/tsl/display/FXAANode.js`.
 *
 * @param textureNode - The texture node that represents the input of the effect.
 * @return The anti-aliased color.
 */
export const fxaa = (textureNode: Sampler2D): Node<"vec4"> => {
  const _ContrastThreshold = float(0.0312);
  const _RelativeThreshold = float(0.063);
  const _SubpixelBlending = float(1.0);

  const Sample = (u: Node<"vec2">): Node<"vec4"> => textureNode.texture(u);
  const SampleLuminance = (u: Node<"vec2">): Node<"float"> =>
    dot(Sample(u).rgb, vec3(0.3, 0.59, 0.11));
  const SampleLuminanceOffset = (
    texSize: Node<"vec2">,
    u: Node<"vec2">,
    uOffset: FloatIn,
    vOffset: FloatIn,
  ): Node<"float"> =>
    SampleLuminance(u.add(texSize.mul(vec2(uOffset, vOffset))));

  // EDGE_STEPS.element(i) is a constant table; a select chain stands in for the
  // uniform array so the effect carries no host data.
  const edgeStepAt = (i: Node<"int">): Node<"float"> => {
    let v = float(EDGE_STEPS[EDGE_STEPS.length - 1]);
    for (let k = EDGE_STEPS.length - 2; k >= 0; k--) {
      v = select(i.equal(k), float(EDGE_STEPS[k]), v);
    }
    return v;
  };

  interface LuminanceNeighborhood {
    m: Node<"float">; n: Node<"float">; e: Node<"float">; s: Node<"float">; w: Node<"float">;
    ne: Node<"float">; nw: Node<"float">; se: Node<"float">; sw: Node<"float">;
    highest: Node<"float">; lowest: Node<"float">; contrast: Node<"float">;
  }

  const SampleLuminanceNeighborhood = (
    texSize: Node<"vec2">,
    u: Node<"vec2">,
  ): LuminanceNeighborhood => {
    const m = SampleLuminance(u);
    const n = SampleLuminanceOffset(texSize, u, 0.0, -1.0);
    const e = SampleLuminanceOffset(texSize, u, 1.0, 0.0);
    const s = SampleLuminanceOffset(texSize, u, 0.0, 1.0);
    const w = SampleLuminanceOffset(texSize, u, -1.0, 0.0);
    const ne = SampleLuminanceOffset(texSize, u, 1.0, -1.0);
    const nw = SampleLuminanceOffset(texSize, u, -1.0, -1.0);
    const se = SampleLuminanceOffset(texSize, u, 1.0, 1.0);
    const sw = SampleLuminanceOffset(texSize, u, -1.0, 1.0);
    const highest = max(s, e, n, w, m);
    const lowest = min(s, e, n, w, m);
    const contrast = highest.sub(lowest);
    return { m, n, e, s, w, ne, nw, se, sw, highest, lowest, contrast };
  };

  const ShouldSkipPixel = (l: LuminanceNeighborhood): Node<"bool"> => {
    const threshold = max(_ContrastThreshold, _RelativeThreshold.mul(l.highest));
    return l.contrast.lessThan(threshold);
  };

  const DeterminePixelBlendFactor = (l: LuminanceNeighborhood): Node<"float"> => {
    let g = float(2.0).mul(l.s.add(l.e).add(l.n).add(l.w));
    g = g.add(l.se.add(l.sw).add(l.ne).add(l.nw));
    g = g.mul(1.0 / 12.0);
    g = abs(g.sub(l.m));
    g = clamp(g.div(max(l.contrast, 0)), 0.0, 1.0);
    const blendFactor = smoothstep(0.0, 1.0, g);
    return blendFactor.mul(blendFactor).mul(_SubpixelBlending);
  };

  const DetermineEdge = (
    texSize: Node<"vec2">,
    l: LuminanceNeighborhood,
  ): {
    isHorizontal: Node<"bool">;
    pixelStep: Node<"float">;
    oppositeLuminance: Node<"float">;
    gradient: Node<"float">;
  } => {
    const horizontal = abs(l.s.add(l.n).sub(l.m.mul(2.0))).mul(2.0)
      .add(abs(l.se.add(l.ne).sub(l.e.mul(2.0))).add(abs(l.sw.add(l.nw).sub(l.w.mul(2.0)))));
    const vertical = abs(l.e.add(l.w).sub(l.m.mul(2.0))).mul(2.0)
      .add(abs(l.se.add(l.sw).sub(l.s.mul(2.0))).add(abs(l.ne.add(l.nw).sub(l.n.mul(2.0)))));

    const isHorizontal = horizontal.greaterThanEqual(vertical);

    const pLuminance = select(isHorizontal, l.s, l.e);
    const nLuminance = select(isHorizontal, l.n, l.w);
    const pGradient = abs(pLuminance.sub(l.m));
    const nGradient = abs(nLuminance.sub(l.m));

    const pixelStep = select(isHorizontal, texSize.y, texSize.x).toVar();
    const oppositeLuminance = float(0).toVar();
    const gradient = float(0).toVar();

    If(pGradient.lessThan(nGradient), () => {
      pixelStep.assign(pixelStep.negate());
      oppositeLuminance.assign(nLuminance);
      gradient.assign(nGradient);
    }).Else(() => {
      oppositeLuminance.assign(pLuminance);
      gradient.assign(pGradient);
    });

    return { isHorizontal, pixelStep, oppositeLuminance, gradient };
  };

  const DetermineEdgeBlendFactor = (
    texSize: Node<"vec2">,
    l: LuminanceNeighborhood,
    e: ReturnType<typeof DetermineEdge>,
    uvNode: Node<"vec2">,
  ): Node<"float"> => {
    const uvEdge = uvNode.toVar();
    const edgeStep = vec2().toVar();
    If(e.isHorizontal, () => {
      uvEdge.y.addAssign(e.pixelStep.mul(0.5));
      edgeStep.assign(vec2(texSize.x, 0.0));
    }).Else(() => {
      uvEdge.x.addAssign(e.pixelStep.mul(0.5));
      edgeStep.assign(vec2(0.0, texSize.y));
    });

    const edgeLuminance = l.m.add(e.oppositeLuminance).mul(0.5);
    const gradientThreshold = e.gradient.mul(0.25);

    const puv = uvEdge.add(edgeStep.mul(edgeStepAt(int(0)))).toVar();
    const pLuminanceDelta = SampleLuminance(puv).sub(edgeLuminance).toVar();
    const pAtEnd = abs(pLuminanceDelta).greaterThanEqual(gradientThreshold).toVar();

    // WGSL refuses to sample from non-uniform control flow, so the edge walk
    // cannot `break` out of the loop. Instead every step samples and `select`
    // keeps the previous state once the edge has been found — the sample is
    // still executed, just discarded, and the control flow stays uniform.
    For(
      () => int(1).toVar(),
      (i) => i.lessThan(int(EDGE_STEP_COUNT)),
      (i) => { i.assign(i.add(1)); },
      (i) => {
        const nextUv = puv.add(edgeStep.mul(edgeStepAt(i))).toVar();
        const nextDelta = SampleLuminance(nextUv).sub(edgeLuminance);
        const nextAtEnd = abs(nextDelta).greaterThanEqual(gradientThreshold);
        puv.assign(select(pAtEnd, puv, nextUv));
        pLuminanceDelta.assign(select(pAtEnd, pLuminanceDelta, nextDelta));
        pAtEnd.assign(select(pAtEnd, pAtEnd, nextAtEnd));
      },
    );

    If(pAtEnd.not(), () => {
      puv.addAssign(edgeStep.mul(float(EDGE_GUESS)));
    });

    const nuv = uvEdge.sub(edgeStep.mul(edgeStepAt(int(0)))).toVar();
    const nLuminanceDelta = SampleLuminance(nuv).sub(edgeLuminance).toVar();
    const nAtEnd = abs(nLuminanceDelta).greaterThanEqual(gradientThreshold).toVar();

    For(
      () => int(1).toVar(),
      (i) => i.lessThan(int(EDGE_STEP_COUNT)),
      (i) => { i.assign(i.add(1)); },
      (i) => {
        const nextUv = nuv.sub(edgeStep.mul(edgeStepAt(i))).toVar();
        const nextDelta = SampleLuminance(nextUv).sub(edgeLuminance);
        const nextAtEnd = abs(nextDelta).greaterThanEqual(gradientThreshold);
        nuv.assign(select(nAtEnd, nuv, nextUv));
        nLuminanceDelta.assign(select(nAtEnd, nLuminanceDelta, nextDelta));
        nAtEnd.assign(select(nAtEnd, nAtEnd, nextAtEnd));
      },
    );

    If(nAtEnd.not(), () => {
      nuv.subAssign(edgeStep.mul(float(EDGE_GUESS)));
    });

    const pDistance = float(0).toVar();
    const nDistance = float(0).toVar();

    If(e.isHorizontal, () => {
      pDistance.assign(puv.x.sub(uvNode.x));
      nDistance.assign(uvNode.x.sub(nuv.x));
    }).Else(() => {
      pDistance.assign(puv.y.sub(uvNode.y));
      nDistance.assign(uvNode.y.sub(nuv.y));
    });

    const shortestDistance = float(0).toVar();
    const deltaSign = bool(false).toVar();

    If(pDistance.lessThanEqual(nDistance), () => {
      shortestDistance.assign(pDistance);
      deltaSign.assign(pLuminanceDelta.greaterThanEqual(0.0));
    }).Else(() => {
      shortestDistance.assign(nDistance);
      deltaSign.assign(nLuminanceDelta.greaterThanEqual(0.0));
    });

    const blendFactor = float(0).toVar();

    // bool equality has no `equal` on a bool node — xor gives "different", so
    // its negation is "same".
    const sameSign = deltaSign.xor(l.m.sub(edgeLuminance).greaterThanEqual(0.0)).not();
    If(sameSign, () => {
      blendFactor.assign(float(0.0));
    }).Else(() => {
      blendFactor.assign(float(0.5).sub(shortestDistance.div(pDistance.add(nDistance))));
    });

    return blendFactor;
  };

  const ApplyFXAA = Fn((uvNode: Node<"vec2">, texSize: Node<"vec2">): Node<"vec4"> => {
    const luminance = SampleLuminanceNeighborhood(texSize, uvNode);
    const finalUv = uvNode.toVar();

    If(ShouldSkipPixel(luminance).not(), () => {
      const pixelBlend = DeterminePixelBlendFactor(luminance);
      const edge = DetermineEdge(texSize, luminance);
      const edgeBlend = DetermineEdgeBlendFactor(texSize, luminance, edge, uvNode);
      const finalBlend = max(pixelBlend, edgeBlend);

      If(edge.isHorizontal, () => {
        finalUv.y.addAssign(edge.pixelStep.mul(finalBlend));
      }).Else(() => {
        finalUv.x.addAssign(edge.pixelStep.mul(finalBlend));
      });
    });

    return Sample(finalUv);
  });

  // Inverse resolution from the texture's own dimensions; the shader needs no
  // host-supplied size uniform.
  const invSize = vec2(1).div(textureSize(textureNode).toVec2());
  return ApplyFXAA(uv(), invSize);
};
