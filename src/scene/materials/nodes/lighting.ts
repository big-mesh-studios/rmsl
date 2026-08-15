import { If, PI, type Node } from "../../../rmsl";

/**
 * The full contribution of one light to a standard (metalness-workflow)
 * surface: a Lambert diffuse plus a Schlick-GGX specular, using three.js's
 * simplified GGX (a = roughness^4, Smith height-correlated visibility).
 *
 * All vectors are expected normalized; `f0` is `mix(vec3(0.04), albedo,
 * metalness)`.
 */
export function standardLight(
  albedo: Node<"vec3">,
  normal: Node<"vec3">,
  viewDir: Node<"vec3">,
  lightDir: Node<"vec3">,
  lightColor: Node<"vec3">,
  roughness: Node<"float">,
  metalness: Node<"float">,
  f0: Node<"vec3">,
): Node<"vec3"> {
  const half = lightDir.add(viewDir).normalize();

  // Dot products of normalized vectors can exceed 1.0 by an ulp or two, and
  // `pow(1 - vDotH, 5)` below turns a base just below zero into NaN — which
  // blackens the whole surface. Each is clamped into [0.0001, 1] so the GGX
  // denominators stay positive and the fresnel base stays non-negative.
  const nDotL = normal.dot(lightDir).clamp(0.0001, 1);
  const nDotH = normal.dot(half).clamp(0.0001, 1);
  const nDotV = normal.dot(viewDir).clamp(0.0001, 1);
  const vDotH = viewDir.dot(half).clamp(0.0001, 1);

  const a2 = roughness.pow(4);
  const nDotH2 = nDotH.pow(2);

  const d = a2.div(PI.mul(nDotH2.mul(a2.sub(1)).add(1).pow(2)));
  const ggxV = nDotL.mul(nDotV.pow(2).mul(a2.oneMinus()).add(a2).sqrt());
  const ggxL = nDotV.mul(nDotL.pow(2).mul(a2.oneMinus()).add(a2).sqrt());
  const vis = ggxV.add(ggxL).reciprocal().mul(0.5);
  const f = f0.add(f0.oneMinus().mul(vDotH.oneMinus().pow(5)));

  const kD = f.oneMinus().mul(metalness.oneMinus());
  const specular = d.mul(vis).mul(f);

  return albedo.div(PI).mul(kD).add(specular).mul(lightColor).mul(nDotL);
}

/** A Lambert (N·L) diffuse for one light. */
export function lambertDiffuse(
  albedo: Node<"vec3">,
  normal: Node<"vec3">,
  lightDir: Node<"vec3">,
  lightColor: Node<"vec3">,
): Node<"vec3"> {
  return albedo.mul(lightColor).mul(normal.dot(lightDir).max(0));
}

/**
 * The attenuation of a point light at the fragment, three.js-style: an
 * inverse power falloff, optionally windowed by a cutoff distance. Uses `If`,
 * so it must be called inside an `Fn` body.
 */
export function pointLightAttenuation(
  lightPosition: Node<"vec3">,
  fragmentPosition: Node<"vec3">,
  distance: Node<"float">,
  decay: Node<"float">,
): Node<"float"> {
  const d = fragmentPosition.distance(lightPosition).max(0.0001).toVar();
  const falloff = d.pow(decay.negate()).toVar();
  If(distance.greaterThan(0), () => {
    const cutoff = distance.reciprocal().mul(d).pow(4).oneMinus().saturate().pow(2);
    falloff.assign(falloff.mul(cutoff));
  });
  return falloff;
}
