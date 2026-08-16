// RMSL instancing demo: an InstancedMesh grid with per-instance transforms
// and colors, rendered by WebGLRenderer.
//
// Run with `pnpm --filter instancing-demo dev`.
import {
  WebGLRenderer,
  Scene,
  PerspectiveCamera,
  InstancedMesh,
  BoxGeometry,
  MeshStandardMaterial,
  AmbientLight,
  DirectionalLight,
  Matrix4,
  Color,
} from "@random-mesh/rmsl/scene";

const canvas = document.createElement("canvas");
canvas.style.position = "fixed";
canvas.style.inset = "0";
canvas.style.touchAction = "none";
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
document.body.appendChild(canvas);

const renderer = new WebGLRenderer(canvas, { antialias: true });
renderer.setClearColor(0x101318);
renderer.setSize(window.innerWidth, window.innerHeight);
window.addEventListener("resize", () => renderer.setSize(window.innerWidth, window.innerHeight));

// === Scene ===
const scene = new Scene();

// A grid of 15 x 15 = 225 instances, all drawn in one instanced call.
const cols = 15;
const rows = 15;
const spacing = 1.2;
const mesh = new InstancedMesh(
  new BoxGeometry(0.7, 0.7, 0.7),
  new MeshStandardMaterial({ color: 0xbbbbdd, roughness: 0.35, metalness: 0.2 }),
  cols * rows,
);

// One matrix per instance; updated every frame below.
const matrix = new Matrix4();
const color = new Color();
for (let r = 0; r < rows; r++) {
  for (let c = 0; c < cols; c++) {
    const i = r * cols + c;
    mesh.setMatrixAt(i, matrix.makeTranslation(
      (c - cols / 2) * spacing,
      (r - rows / 2) * spacing,
      0,
    ));
    // A per-instance color gradient instead of a flat material color.
    color.setRGB(c / cols, (cols - c) / cols, r / rows);
    mesh.setColorAt(i, color);
  }
}
mesh.instanceMatrix.needsUpdate = true;
mesh.instanceColor!.needsUpdate = true;
scene.add(mesh);

// === Lights ===
scene.add(new AmbientLight(0xffffff, 0.3));

const sun = new DirectionalLight(0xfff3e0, 2);
sun.position.set(6, 12, 8);
scene.add(sun);

const fill = new DirectionalLight(0x8899ff, 0.4);
fill.position.set(-8, 4, -6);
scene.add(fill);

// === Camera ===
const camera = new PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(12, 9, 14);
camera.lookAt(0, 0, 0);

// === Orbit controls ===
let theta = 0.8;
let phi = 0.55;
let radius = 22;
let isDragging = false;
let lastMX = 0;
let lastMY = 0;

canvas.addEventListener("pointerdown", (e) => {
  isDragging = true;
  lastMX = e.clientX;
  lastMY = e.clientY;
  try { canvas.setPointerCapture(e.pointerId); } catch { /* ignore */ }
});
canvas.addEventListener("pointermove", (e) => {
  if (!isDragging) return;
  theta -= (e.clientX - lastMX) * 0.005;
  phi = Math.max(0.05, Math.min(1.5, phi - (e.clientY - lastMY) * 0.005));
  lastMX = e.clientX;
  lastMY = e.clientY;
});
const endDrag = () => { isDragging = false; };
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);
canvas.addEventListener("wheel", (e) => {
  radius = Math.max(8, Math.min(60, radius * (1 + e.deltaY * 0.001)));
}, { passive: false });

// === Render loop ===
// Only the instance matrices change; the geometry and material stay put.
const rotY = new Matrix4();
const rotX = new Matrix4();
renderer.setAnimationLoop((time) => {
  const t = time / 1000;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      const x = (c - cols / 2) * spacing;
      const y = (r - rows / 2) * spacing;
      // Each instance bobs and spins on its own.
      const h = Math.sin(t * 1.5 + x * 0.6 + y * 0.6) * 0.5;
      const qy = t * 0.6 + x * 0.2;
      const qx = Math.sin(t * 0.8 + y * 0.3) * 0.3;
      rotY.makeRotationY(qy);
      rotX.makeRotationX(qx);
      matrix.makeTranslation(x, h, 0).multiply(rotY).multiply(rotX);
      mesh.setMatrixAt(i, matrix);
    }
  }
  mesh.instanceMatrix.needsUpdate = true;

  camera.position.set(
    radius * Math.sin(theta) * Math.cos(phi),
    radius * Math.sin(phi),
    radius * Math.cos(theta) * Math.cos(phi),
  );
  camera.lookAt(0, 0, 0);

  renderer.render(scene, camera);
});
