// All shaders for the demo, written in RMSL. The flame is a faithful port of
// melty-karts' `models/Bomb.tsx` wick fire (three.js TSL -> RMSL); the bomb
// body is a small lit-mesh shader. Every Fn compiles with compileGLSL.vertex /
// compileGLSL.fragment to a complete GLSL ES 3.00 shader.
import {
  Fn, add, attribute, clamp, cos, div, length, max, mix, mod, mul, normalize,
  output, pow, reciprocal, sin, smoothstep, sub, uniform, varying,
  vec2, vec3, vec4, dot,
} from "@random-mesh/rmsl";

// === Shared transform uniforms (bound by slot name, `_rmsl_uN`) ===
export const bodyModelViewMatrix = uniform("mat4");
export const bodyProjectionMatrix = uniform("mat4");
export const bodyLightDir = uniform("vec3"); // view-space direction toward the light
export const bodyAlbedo = uniform("vec3");

export const aPosition = attribute("vec3");
export const aNormal = attribute("vec3");

export const vNormal = varying("vec3");
export const vViewPos = varying("vec3");

// === Bomb body — vertex ===
export const bodyVertex = Fn(() => {
  const mvPos = bodyModelViewMatrix.mul(vec4(aPosition, 1.0)).toVar();
  // gl_Position must be in clip space, so the projection is applied here —
  // returning the view-space position would push the mesh off-screen.
  const clipPos = bodyProjectionMatrix.mul(mvPos).toVar();
  // Rigid-body transforms only (rotation + translation), so the normal matrix
  // is the upper-left 3x3 of the model-view matrix.
  vNormal.assign(bodyModelViewMatrix.toMat3().mul(aNormal));
  vViewPos.assign(mvPos.xyz);
  return clipPos;
});

// === Bomb body — fragment (directional light + ambient + specular) ===
export const bodyFragment = Fn(() => {
  const n = normalize(vNormal);
  const l = normalize(bodyLightDir);
  const viewDir = normalize(vViewPos.negate());
  const halfDir = normalize(add(l, viewDir));
  const diff = max(dot(n, l), 0.0);
  const spec = pow(max(dot(n, halfDir), 0.0), 32.0).mul(0.4);
  const color = bodyAlbedo.mul(add(0.4, mul(diff, 0.6))).add(vec3(1.0).mul(spec));
  const out = output("vec4");
  out.assign(vec4(color, 1.0));
  return out;
});

// === Flame particle uniforms ===
export const uTime = uniform("float");
export const uIsPerspective = uniform("float");
export const flameModelViewMatrix = uniform("mat4");
export const flameProjectionMatrix = uniform("mat4");
export const flameScreenSize = uniform("vec2");

// === Flame particle attributes ===
export const aParticlePos = attribute("vec3");
export const aCorner = attribute("vec2");
export const aDrift = attribute("vec3");
export const aLife = attribute("float");
export const aOffset = attribute("float");
export const aSize = attribute("float");
export const aSpin = attribute("float");
export const aUv = attribute("vec2");

// === Flame particle varyings ===
export const vUv = varying("vec2");
export const vFade = varying("float");
export const vHeat = varying("float");

// === Flame — vertex ===
// Ported verbatim from melty-karts `models/Bomb.tsx` (`_sharedWickFire`): 48
// billboarded quads, viewport-pixel sized via `screenSize`.
export const flameVertex = Fn(() => {
  const lifeT = div(mod(add(uTime, aOffset), aLife), aLife).toVar();
  const fadeIn = smoothstep(0.0, 0.08, lifeT);
  const fadeOut = sub(1.0, smoothstep(0.35, 1.0, lifeT));
  const fade = mul(fadeIn, fadeOut).toVar();
  const heat = sub(1.0, lifeT).toVar();

  const swirl = mul(mul(sin(add(add(mul(uTime, 10.0), aSpin), mul(lifeT, 12.0))), 0.012), sub(1.0, lifeT));
  const flicker = mul(sin(add(mul(uTime, 24.0), aSpin)), 0.004);

  const driftOffset = mul(aDrift, lifeT);
  const base = add(aParticlePos, driftOffset);
  const animPos = vec3(
    add(base.x, swirl),
    add(base.y, add(mul(mul(lifeT, lifeT), 0.12), flicker)),
    add(base.z, mul(mul(cos(add(add(mul(uTime, 8.0), aSpin), mul(lifeT, 10.0))), 0.006), sub(1.0, lifeT))),
  ).toVar();

  // Size in viewport pixels.
  const mvPos = mul(flameModelViewMatrix, vec4(animPos, 1.0)).toVar();
  const dist = length(mvPos.xyz);
  const perspScale = mix(0.25, reciprocal(max(dist, 0.1)), uIsPerspective);
  const particleSize = max(mul(mul(aSize, fade), perspScale), 0.0).toVar();

  // Billboard: offset in clip space by corner * size.
  const clipPos = mul(flameProjectionMatrix, mvPos).toVar();
  const ndcOffset = div(mul(aCorner, particleSize), flameScreenSize);
  const clipOffset = mul(ndcOffset, clipPos.w);
  const combined = add(clipPos.xy, clipOffset).toVar();

  vUv.assign(aUv);
  vFade.assign(fade);
  vHeat.assign(heat);

  return vec4(combined.x, combined.y, clipPos.z, clipPos.w);
});

// === Flame — fragment ===
// Radial gradient via the billboard UV, ember -> flame -> spark by heat.
export const flameFragment = Fn(() => {
  const centered = sub(vUv, vec2(0.5));
  const ptDist = length(centered);
  const circleAlpha = ptDist.lessThanEqual(0.5).select(1.0, 0.0);
  const core = sub(1.0, smoothstep(0.0, 0.28, ptDist));
  const edge = sub(1.0, smoothstep(0.12, 0.5, ptDist));

  const ember = vec3(1.0, 0.22, 0.02);
  const flame = vec3(1.0, 0.55, 0.08);
  const spark = vec3(1.0, 0.95, 0.55);
  const heatClamped = clamp(mul(vHeat, 1.2), 0.0, 1.0);
  const color = mix(ember, flame, heatClamped);
  const finalColor = mix(color, spark, core);

  const out = output("vec4");
  out.assign(vec4(finalColor, mul(mul(edge, vFade), circleAlpha)));
  return out;
});
