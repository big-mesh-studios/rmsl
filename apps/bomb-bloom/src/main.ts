// Bomb + RMSL bloom demo.
//
// The bomb body and the wick fire are RMSL node graphs (`./shader.ts`)
// compiled to GLSL and drawn with a raw WebGL2 context. The bloom is the
// `@random-mesh/rmsl/effects` PassGraph executed by `./bloom.ts`.
import { compileGLSL, uniform } from "@random-mesh/rmsl";
import { bloom } from "@random-mesh/rmsl/effects";
import {
  bodyVertex, bodyFragment,
  bodyModelViewMatrix, bodyProjectionMatrix, bodyLightDir, bodyAlbedo,
  aPosition, aNormal,
  flameVertex, flameFragment,
  uTime, uIsPerspective, flameModelViewMatrix, flameProjectionMatrix, flameScreenSize,
  aParticlePos, aCorner, aDrift, aLife, aOffset, aSize, aSpin, aUv,
} from "./shader";
import { buildSphere, buildCylinder, buildTube, catmullRom, buildFlameParticles } from "./geometry";
import {
  mat4Perspective, mat4LookAt, mat4Multiply, mat4Translation, mat4RotationZ,
  mat3TransformDirection, quadVerts,
} from "./matrix";
import { BloomExecutor } from "./bloom";

// === Canvas / GL ===
const canvas = document.createElement("canvas");
canvas.style.position = "fixed";
canvas.style.top = "0";
canvas.style.left = "0";
canvas.style.zIndex = "0";
canvas.style.touchAction = "none";
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
document.body.appendChild(canvas);

const gl = canvas.getContext("webgl2")!;
if (!gl) {
  document.body.innerHTML = "<h1>WebGL2 not supported</h1>";
  throw new Error("WebGL2 not supported");
}

function compileShader(src: string, type: number): WebGLShader {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    console.error(gl.getShaderInfoLog(s));
    throw new Error(`Shader compile failed:\n${gl.getShaderInfoLog(s)}\n--- source ---\n${src}`);
  }
  return s;
}

function createProgram(vsSource: string, fsSource: string, attrib0?: string): WebGLProgram {
  const vs = compileShader(vsSource, gl.VERTEX_SHADER);
  const fs = compileShader(fsSource, gl.FRAGMENT_SHADER);
  const program = gl.createProgram()!;
  if (attrib0) gl.bindAttribLocation(program, 0, attrib0);
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`Program link failed:\n${gl.getProgramInfoLog(program)}`);
  }
  return program;
}

// === Shaders (RMSL -> GLSL) ===
const bodyProgram = createProgram(
  compileGLSL.vertex(bodyVertex()),
  compileGLSL.fragment(bodyFragment()),
);
const flameProgram = createProgram(
  compileGLSL.vertex(flameVertex()),
  compileGLSL.fragment(flameFragment()),
);

// === Geometry ===
const grey = [0.43, 0.43, 0.43];
const white = [1.0, 1.0, 1.0];

const sphere = buildSphere(0.3, 24, 32);
const cylinder = buildCylinder(0.07, 0.07, 16);
const wickCurve = catmullRom(
  [[0, 0, 0], [0, 0.1, 0], [0.1, 0.15, 0], [0.12, 0.2, 0]],
  8,
);
const wick = buildTube(wickCurve, 0.01, 8);
const flameGeo = buildFlameParticles();

interface MeshBuffers { vao: WebGLVertexArrayObject; count: number; }

function createMeshVao(
  program: WebGLProgram,
  attribs: Array<[string, Float32Array, number]>,
  indices: Uint16Array,
): MeshBuffers {
  const vao = gl.createVertexArray()!;
  gl.bindVertexArray(vao);
  for (const [name, data, size] of attribs) {
    const loc = gl.getAttribLocation(program, name);
    if (loc < 0) continue;
    const buf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
  }
  const ibo = gl.createBuffer()!;
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
  gl.bindVertexArray(null);
  return { vao, count: indices.length };
}

const sphereVao = createMeshVao(bodyProgram, [
  [aPosition.name, sphere.positions, 3],
  [aNormal.name, sphere.normals, 3],
], sphere.indices);
const cylinderVao = createMeshVao(bodyProgram, [
  [aPosition.name, cylinder.positions, 3],
  [aNormal.name, cylinder.normals, 3],
], cylinder.indices);
const wickVao = createMeshVao(bodyProgram, [
  [aPosition.name, wick.positions, 3],
  [aNormal.name, wick.normals, 3],
], wick.indices);
const flameVao = createMeshVao(flameProgram, [
  [aParticlePos.name, flameGeo.positions, 3],
  [aCorner.name, flameGeo.corners, 2],
  [aDrift.name, flameGeo.drifts, 3],
  [aLife.name, flameGeo.lives, 1],
  [aOffset.name, flameGeo.offsets, 1],
  [aSize.name, flameGeo.sizes, 1],
  [aSpin.name, flameGeo.spins, 1],
  [aUv.name, flameGeo.uvs, 2],
], flameGeo.indices);

// === Model matrices (the bomb group: z-rotation -45°, parts offset as in
// melty-karts; the flame is counter-rotated so it stays upright) ===
const group = mat4RotationZ(-Math.PI / 4);
const sphereModel = group;
const cylinderModel = mat4Multiply(group, mat4Translation(0, 0.32, 0));
const wickModel = mat4Multiply(group, mat4Translation(0, 0.35, 0));
const flameModel = mat4Multiply(
  mat4Multiply(group, mat4Translation(0.12, 0.55, 0)),
  mat4RotationZ(Math.PI / 4),
);

// === Scene render target (color + depth), recreated on resize ===
let sceneW = 0;
let sceneH = 0;
const halfFloat = gl.getExtension("EXT_color_buffer_float") != null;
const colorInternal = halfFloat ? gl.RGBA16F : gl.RGBA8;
const colorType = halfFloat ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE;

let sceneFbo: WebGLFramebuffer | null = null;
let sceneTex: WebGLTexture | null = null;
let sceneDepth: WebGLRenderbuffer | null = null;

function ensureSceneTarget(w: number, h: number): void {
  if (sceneFbo && sceneW === w && sceneH === h) return;
  if (sceneTex) { gl.deleteTexture(sceneTex); gl.deleteFramebuffer(sceneFbo); gl.deleteRenderbuffer(sceneDepth); }
  sceneW = w;
  sceneH = h;
  sceneTex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, sceneTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, colorInternal, w, h, 0, gl.RGBA, colorType, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  sceneDepth = gl.createRenderbuffer()!;
  gl.bindRenderbuffer(gl.RENDERBUFFER, sceneDepth);
  gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
  sceneFbo = gl.createFramebuffer()!;
  gl.bindFramebuffer(gl.FRAMEBUFFER, sceneFbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, sceneTex, 0);
  gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, sceneDepth);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

// A 1x1 black texture, bound for the glow when bloom is off.
const blackTex = gl.createTexture()!;
gl.bindTexture(gl.TEXTURE_2D, blackTex);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
gl.bindTexture(gl.TEXTURE_2D, null);

// === Bloom ===
// Strength / radius / threshold are RMSL uniform nodes, so the sliders update
// them live via the executor's float table — no shader recompiles.
const strengthUniform = uniform("float");
const radiusUniform = uniform("float");
const thresholdUniform = uniform("float");
const smoothWidthUniform = uniform("float");
const sceneSampler = uniform("sampler2D");

const bloomGraph = bloom(sceneSampler, {
  strength: strengthUniform,
  radius: radiusUniform,
  threshold: thresholdUniform,
  smoothWidth: smoothWidthUniform,
});
const bloomExecutor = new BloomExecutor(gl, bloomGraph);

const floats: Record<string, number> = {
  [strengthUniform.name]: 1.2,
  [radiusUniform.name]: 0.5,
  [thresholdUniform.name]: 0.5,
  [smoothWidthUniform.name]: 0.01,
};

// === Present pass: scene + glow ===
const PRESENT_VERT = `#version 300 es
precision highp float;
in vec2 aPos;
out vec2 vUv;
void main() { vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }
`;
const PRESENT_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uScene;
uniform sampler2D uGlow;
in vec2 vUv;
out vec4 outColor;
void main() { outColor = texture(uScene, vUv) + texture(uGlow, vUv); }
`;
const presentProgram = createProgram(PRESENT_VERT, PRESENT_FRAG, "aPos");
const presentVao = gl.createVertexArray()!;
gl.bindVertexArray(presentVao);
const quadBuf = gl.createBuffer()!;
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
gl.bindVertexArray(null);

// === Orbit camera ===
let theta = 0.6;
let phi = 0.55;
let radius = 3.4;
let isDragging = false;
let lastMX = 0;
let lastMY = 0;

canvas.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  isDragging = true;
  lastMX = e.clientX;
  lastMY = e.clientY;
  try { canvas.setPointerCapture(e.pointerId); } catch { /* ignore */ }
});
canvas.addEventListener("pointermove", (e) => {
  if (!isDragging) return;
  const dx = e.clientX - lastMX;
  const dy = e.clientY - lastMY;
  theta -= dx * 0.005;
  phi = Math.max(-1.5, Math.min(1.5, phi - dy * 0.005));
  lastMX = e.clientX;
  lastMY = e.clientY;
});
const endDrag = () => { isDragging = false; };
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);
canvas.addEventListener("pointerleave", endDrag);
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  radius = Math.max(1.2, Math.min(12, radius * (1 + e.deltaY * 0.001)));
}, { passive: false });

// === UI ===
const bloomOn = document.getElementById("bloomToggle") as HTMLInputElement;
const strengthEl = document.getElementById("strength") as HTMLInputElement;
const radiusEl = document.getElementById("radius") as HTMLInputElement;
const thresholdEl = document.getElementById("threshold") as HTMLInputElement;
const strengthVal = document.getElementById("strengthVal")!;
const radiusVal = document.getElementById("radiusVal")!;
const thresholdVal = document.getElementById("thresholdVal")!;

function bindSlider(el: HTMLInputElement, slot: string, label: HTMLElement): void {
  el.addEventListener("input", () => {
    floats[slot] = parseFloat(el.value);
    label.textContent = el.value;
  });
}
bindSlider(strengthEl, strengthUniform.name, strengthVal);
bindSlider(radiusEl, radiusUniform.name, radiusVal);
bindSlider(thresholdEl, thresholdUniform.name, thresholdVal);

function resize(): void {
  const w = window.innerWidth;
  const h = window.innerHeight;
  canvas.width = w;
  canvas.height = h;
  gl.viewport(0, 0, w, h);
  ensureSceneTarget(w, h);
}
window.addEventListener("resize", resize);
resize();

// === Body program uniform locations ===
const bodyMvLoc = gl.getUniformLocation(bodyProgram, bodyModelViewMatrix.name);
const bodyProjLoc = gl.getUniformLocation(bodyProgram, bodyProjectionMatrix.name);
const bodyLightLoc = gl.getUniformLocation(bodyProgram, bodyLightDir.name);
const bodyAlbedoLoc = gl.getUniformLocation(bodyProgram, bodyAlbedo.name);

// === Flame program uniform locations ===
const timeLoc = gl.getUniformLocation(flameProgram, uTime.name);
const perspLoc = gl.getUniformLocation(flameProgram, uIsPerspective.name);
const flameMvLoc = gl.getUniformLocation(flameProgram, flameModelViewMatrix.name);
const flameProjLoc = gl.getUniformLocation(flameProgram, flameProjectionMatrix.name);
const flameScreenLoc = gl.getUniformLocation(flameProgram, flameScreenSize.name);

// === Present program uniform locations ===
const uSceneLoc = gl.getUniformLocation(presentProgram, "uScene");
const uGlowLoc = gl.getUniformLocation(presentProgram, "uGlow");

function renderScene(view: Float32Array, proj: Float32Array, time: number): void {
  gl.bindFramebuffer(gl.FRAMEBUFFER, sceneFbo);
  gl.viewport(0, 0, sceneW, sceneH);
  gl.clearColor(0.03, 0.03, 0.06, 1.0);
  gl.enable(gl.DEPTH_TEST);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  // --- Bomb body ---
  gl.useProgram(bodyProgram);
  gl.uniformMatrix4fv(bodyProjLoc, false, proj);
  const lightView = mat3TransformDirection(view, [0.35, 0.9, 0.5]);
  gl.uniform3fv(bodyLightLoc, lightView);

  const drawBody = (model: Float32Array, vao: MeshBuffers, albedo: number[]) => {
    const mv = mat4Multiply(view, model);
    gl.uniformMatrix4fv(bodyMvLoc, false, mv);
    gl.uniform3fv(bodyAlbedoLoc, albedo);
    gl.bindVertexArray(vao.vao);
    gl.drawElements(gl.TRIANGLES, vao.count, gl.UNSIGNED_SHORT, 0);
  };
  drawBody(sphereModel, sphereVao, grey);
  drawBody(cylinderModel, cylinderVao, grey);
  drawBody(wickModel, wickVao, white);

  // --- Flame (additive, no depth write) ---
  gl.useProgram(flameProgram);
  gl.uniform1f(timeLoc, time);
  gl.uniform1f(perspLoc, 1.0);
  gl.uniform2f(flameScreenLoc, sceneW, sceneH);
  gl.uniformMatrix4fv(flameProjLoc, false, proj);
  gl.uniformMatrix4fv(flameMvLoc, false, mat4Multiply(view, flameModel));
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
  gl.depthMask(false);
  gl.bindVertexArray(flameVao.vao);
  gl.drawElements(gl.TRIANGLES, flameVao.count, gl.UNSIGNED_SHORT, 0);
  gl.depthMask(true);
  gl.disable(gl.BLEND);

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

function present(glow: WebGLTexture): void {
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.disable(gl.DEPTH_TEST);
  gl.useProgram(presentProgram);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, sceneTex);
  gl.uniform1i(uSceneLoc, 0);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, glow);
  gl.uniform1i(uGlowLoc, 1);
  gl.bindVertexArray(presentVao);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

// === Main loop ===
function frame(now: number): void {
  const time = now / 1000;
  if (!isDragging) theta += 0.0025;

  const eyeX = radius * Math.sin(theta) * Math.cos(phi);
  const eyeY = radius * Math.sin(phi);
  const eyeZ = radius * Math.cos(theta) * Math.cos(phi);
  const view = mat4LookAt(eyeX, eyeY, eyeZ, 0, 0.35, 0, 0, 1, 0);
  const proj = mat4Perspective(45 * Math.PI / 180, canvas.width / canvas.height, 0.1, 100);

  ensureSceneTarget(canvas.width, canvas.height);
  renderScene(view, proj, time);

  let glow: WebGLTexture = blackTex;
  if (bloomOn.checked) {
    const result = bloomExecutor.run({
      scene: sceneTex!,
      sceneWidth: sceneW,
      sceneHeight: sceneH,
      floats,
    });
    glow = result.tex;
  }

  present(glow);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
