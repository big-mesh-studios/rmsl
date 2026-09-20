import { compileGlsl } from "@random-mesh/rmsl/glsl";
import {
  calcColourAndDepth,
  cameraPosition,
  cameraProjectionMatrix,
  cameraProjectionMatrixInverse,
  cameraViewMatrix,
  cameraWorldMatrix,
  mat4Inverse,
  mat4LookAt,
  mat4Perspective,
  quadPos,
  quadVerts,
  vertexMain,
} from "../../shared/shader";

// === Compile shaders ===
const vsGLSL = compileGlsl.vertex(vertexMain());
const fsGLSL = compileGlsl.fragment(calcColourAndDepth());

// === Orbital camera state ===
let theta = 0;
let phi = 0.6;
let radius = 10;
let isDragging = false;
let lastMX = 0;
let lastMY = 0;

function getViewMatrix(): Float32Array {
  const eyeX = radius * Math.sin(theta) * Math.cos(phi);
  const eyeY = radius * Math.sin(phi);
  const eyeZ = radius * Math.cos(theta) * Math.cos(phi);
  return mat4LookAt(eyeX, eyeY, eyeZ, 0, 0, 0, 0, 1, 0);
}

function getCameraPosition(): [number, number, number] {
  return [radius * Math.sin(theta) * Math.cos(phi), radius * Math.sin(phi), radius * Math.cos(theta) * Math.cos(phi)];
}

// === WebGL2 setup ===
const canvas = document.createElement("canvas");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
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
    throw new Error("Shader compile error");
  }
  return s;
}

const vs = compileShader(vsGLSL, gl.VERTEX_SHADER);
const fs = compileShader(fsGLSL, gl.FRAGMENT_SHADER);
const program = gl.createProgram()!;
gl.attachShader(program, vs);
gl.attachShader(program, fs);
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
  console.error(gl.getProgramInfoLog(program));
  throw new Error("Program link error");
}
gl.useProgram(program);

// Full-screen quad VBO
const vao = gl.createVertexArray();
gl.bindVertexArray(vao);
const vbo = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);
const attrLoc = gl.getAttribLocation(program, quadPos.name);
gl.enableVertexAttribArray(attrLoc);
gl.vertexAttribPointer(attrLoc, 2, gl.FLOAT, false, 0, 0);

// Uniform locations
const uniforms = {
  projection: gl.getUniformLocation(program, cameraProjectionMatrix.name),
  view: gl.getUniformLocation(program, cameraViewMatrix.name),
  projInv: gl.getUniformLocation(program, cameraProjectionMatrixInverse.name),
  world: gl.getUniformLocation(program, cameraWorldMatrix.name),
  camPos: gl.getUniformLocation(program, cameraPosition.name),
};

// === Pointer / wheel events ===
canvas.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  isDragging = true;
  lastMX = e.clientX;
  lastMY = e.clientY;
  try {
    canvas.setPointerCapture(e.pointerId);
  } catch (err) {
    console.warn("setPointerCapture failed:", err);
  }
});
canvas.addEventListener("pointermove", (e) => {
  e.preventDefault();
  if (!isDragging) return;
  const dx = e.clientX - lastMX;
  const dy = e.clientY - lastMY;
  theta -= dx * 0.005;
  phi = Math.max(-1.5, Math.min(1.5, phi - dy * 0.005));
  lastMX = e.clientX;
  lastMY = e.clientY;
});
canvas.addEventListener("pointerup", (e) => {
  isDragging = false;
  try {
    canvas.releasePointerCapture(e.pointerId);
  } catch (err) {
    console.warn("releasePointerCapture failed:", err);
  }
});
canvas.addEventListener("pointercancel", (e) => {
  isDragging = false;
  try {
    canvas.releasePointerCapture(e.pointerId);
  } catch (err) {
    console.warn("releasePointerCapture failed:", err);
  }
});
canvas.addEventListener("pointerleave", (e) => {
  isDragging = false;
  try {
    canvas.releasePointerCapture(e.pointerId);
  } catch (err) {
    console.warn("releasePointerCapture failed:", err);
  }
});
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  radius *= 1 + e.deltaY * 0.001;
  radius = Math.max(0.5, Math.min(500, radius));
});

// === Resize ===
window.addEventListener("resize", () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
});

// === Render loop ===
function render() {
  const w = canvas.width;
  const h = canvas.height;
  gl.viewport(0, 0, w, h);
  gl.clear(gl.DEPTH_BUFFER_BIT);

  const aspect = w / h;
  const near = 0.1;
  const far = 1000;
  const fov = 0.6;
  const proj = mat4Perspective(fov, aspect, near, far);
  const view = getViewMatrix();
  const projInv = mat4Inverse(proj);
  const world = mat4Inverse(view);
  const camPos = getCameraPosition();

  gl.uniformMatrix4fv(uniforms.projection, false, proj);
  gl.uniformMatrix4fv(uniforms.view, false, view);
  gl.uniformMatrix4fv(uniforms.projInv, false, projInv);
  gl.uniformMatrix4fv(uniforms.world, false, world);
  gl.uniform3f(uniforms.camPos, camPos[0], camPos[1], camPos[2]);

  gl.bindVertexArray(vao);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

  requestAnimationFrame(render);
}

render();
