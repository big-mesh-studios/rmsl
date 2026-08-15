import {Fn, abs, bool, exp2, float, floor, int, ivec2, max, min, textureLoad, textureSize, uv, vec3, vec4, type BooleanLike, type Node} from "../rmsl";
import { f, type FloatIn, type IntIn, type Vec2In, type Sampler2D, type Sampler3D } from "./util";

/**
 * Post-processing effect for contrast-adaptive sharpening (RCAS).
 *
 * Reference: https://gpuopen.com/fidelityfx-superresolution/
 * Ported from three.js's `examples/jsm/tsl/display/SharpenNode.js`.
 *
 * @param textureNode - The texture node that represents the input of the effect.
 * @param sharpness - Sharpening strength. 0 = maximum sharpening, 2 = no sharpening.
 * @param denoise - Whether to attenuate sharpening in noisy areas.
 * @return The sharpened color.
 */
export const sharpen = (
  textureNode: Sampler2D,
  sharpness: FloatIn = 0.2,
  denoise: BooleanLike = false,
): Node<"vec4"> => {
  const denoiseNode = (typeof denoise === "boolean" ? bool(denoise) : denoise) as Node<"bool">;

  // Sharpening amount from the user parameter. Computed outside the Fn so the
  // local `f` (the +x sample) cannot shadow the `f` helper it needs.
  const con = exp2(f(sharpness).negate());

  return Fn(() => {
    const targetUV = uv();
    const texSize = textureSize(textureNode);

    const p = ivec2(
      int(floor(targetUV.x.mul(texSize.x.toFloat()))),
      int(floor(targetUV.y.mul(texSize.y.toFloat()))),
    ).toVar();

    const e = textureLoad(textureNode, p);
    const b = textureLoad(textureNode, p.add(ivec2(0, -1)));
    const d = textureLoad(textureNode, p.add(ivec2(-1, 0)));
    const f = textureLoad(textureNode, p.add(ivec2(1, 0)));
    const h = textureLoad(textureNode, p.add(ivec2(0, 1)));

    // Approximate luminance (luma times 2).
    const luma = (s: Node<"vec4">): Node<"float"> => s.g.add(s.b.add(s.r).mul(0.5));

    const bL = luma(b);
    const dL = luma(d);
    const eL = luma(e);
    const fL = luma(f);
    const hL = luma(h);

    // Min and max of the ring.
    const mn4 = min(min(b.rgb, d.rgb), min(f.rgb, h.rgb));
    const mx4 = max(max(b.rgb, d.rgb), max(f.rgb, h.rgb));

    // Compute the adaptive lobe weight. Limiters based on how much sharpening
    // the local contrast can tolerate.
    const RCAS_LIMIT = float(0.25 - 1.0 / 16.0);

    const hitMin = min(mn4, e.rgb).div(mx4.mul(4.0));
    const hitMax = vec3(1.0).sub(max(mx4, e.rgb)).div(mn4.mul(4.0).sub(4.0));
    const lobeRGB = max(hitMin.negate(), hitMax);

    const lobe = max(
      RCAS_LIMIT.negate(),
      min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), float(0.0)),
    ).mul(con);

    // Noise attenuation.
    const nz = bL.add(dL).add(fL).add(hL).mul(0.25).sub(eL);
    const nzRange = max(max(bL, dL), max(eL, max(fL, hL)))
      .sub(min(min(bL, dL), min(eL, min(fL, hL))));
    const nzFactor = float(1.0).sub(abs(nz).div(max(nzRange, float(1.0 / 65536.0))).saturate().mul(0.5));

    const effectiveLobe = denoiseNode.select(lobe.mul(nzFactor), lobe) as Node<"vec3">;

    // Resolve: weighted blend of cross neighbors and center.
    const result = b.rgb.add(d.rgb).add(f.rgb).add(h.rgb).mul(effectiveLobe).add(e.rgb)
      .div(effectiveLobe.mul(4.0).add(1.0));

    return vec4(result, e.a);
  })();
};
