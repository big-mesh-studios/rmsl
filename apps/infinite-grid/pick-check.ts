// End-to-end check: compile the shared ray-marched fragment shader to a JS
// callable and "pick" a pixel on the CPU, exactly as an app would on
// pointerdown, then verify the ray actually lands on the y = 0 plane.
import { compileJS } from "@random-mesh/rmsl";
import {
  calcColourAndDepth, positionGeometry, positionWorld,
  cameraProjectionMatrix, cameraViewMatrix,
  cameraProjectionMatrixInverse, cameraWorldMatrix, cameraPosition,
  mat4Perspective, mat4LookAt, mat4Inverse,
} from "../shared/shader.ts";

const cameraPositionValue = [0, 5, 10];
const proj = mat4Perspective(60 * Math.PI / 180, 16 / 9, 0.1, 1000);
const view = mat4LookAt(...cameraPositionValue, 0, 0, 0, 0, 1, 0);
const invProj = mat4Inverse(proj);
const world = mat4Inverse(view);

const pick = compileJS(calcColourAndDepth, {
  name: "pick",
  params: [],
  derivatives: "zero", // the grid shader uses fwidth(), meaningless per-pixel
});

const result = pick({
  uniforms: {
    [cameraProjectionMatrix.name]: [...proj],
    [cameraViewMatrix.name]: [...view],
    [cameraProjectionMatrixInverse.name]: [...invProj],
    [cameraWorldMatrix.name]: [...world],
    [cameraPosition.name]: cameraPositionValue,
  },
  varyings: {
    [positionGeometry.name]: [0, 0, 0], // screen centre
    [positionWorld.name]: [0, 0, 0],
  },
}) as any;

const colour = result.value as number[];
if (!Array.isArray(colour) || colour.length !== 4) throw new Error("bad colour");
if (!Number.isFinite(result.fragDepth)) throw new Error("bad fragDepth");
if (result.fragDepth <= 0 || result.fragDepth >= 1) throw new Error("unexpected depth");

// Reconstruct the pick point independently from the same camera: the ray
// through the screen centre must meet y = 0 in front of the camera.
const mul = (m: ArrayLike<number>, v: number[]) => {
  const out = new Array(4).fill(0);
  for (let row = 0; row < 4; row++) {
    out[row] = m[row] * v[0] + m[4 + row] * v[1] + m[8 + row] * v[2] + m[12 + row] * v[3];
  }
  return out;
};
const viewPos = mul(invProj, [0, 0, -1, 1]);
const worldDir = mul(world, [viewPos[0], viewPos[1], viewPos[2], 0]);
const rd = (() => {
  const l = Math.hypot(worldDir[0], worldDir[1], worldDir[2]);
  return worldDir.map((x) => x / l);
})();
if (rd[1] >= 0) throw new Error("ray does not point down");
const t = -cameraPositionValue[1] / rd[1];
const hit = [
  cameraPositionValue[0] + rd[0] * t,
  cameraPositionValue[1] + rd[1] * t,
  cameraPositionValue[2] + rd[2] * t,
];
if (t <= 0) throw new Error(`ray travels backwards: t=${t}`);
if (Math.abs(hit[1]) > 1e-4) throw new Error(`pick point not on ground: y=${hit[1]}`);

console.log(`colour: [${colour.map((v) => v.toFixed(3)).join(", ")}]`);
console.log(`fragDepth: ${result.fragDepth.toFixed(4)}`);
console.log(`pick point: (${hit.map((v) => v.toFixed(3)).join(", ")}) — on y = 0, in front of camera`);
console.log("OK — CPU pick returned a ground hit at screen centre");
