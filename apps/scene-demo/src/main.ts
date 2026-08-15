// RMSL scene demo: node-based materials + WebGLRenderer.
//
// Run with `pnpm --filter scene-demo dev`.
import { vec3, mix, float } from "@random-mesh/rmsl";
import {
  WebGLRenderer,
  Scene, Group, Mesh,
  PerspectiveCamera,
  BoxGeometry, SphereGeometry, TorusGeometry, PlaneGeometry,
  MeshStandardMaterial, MeshBasicMaterial,
  AmbientLight, DirectionalLight, PointLight,
  DataTexture,
  Color,
} from "@random-mesh/rmsl/scene";

const canvas = document.createElement("canvas");
canvas.style.position = "fixed";
canvas.style.inset = "0";
canvas.style.touchAction = "none";
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
document.body.appendChild(canvas);

const renderer = new WebGLRenderer(canvas, { antialias: true });
renderer.setClearColor(new Color().setHex(0x101318));
renderer.setSize(window.innerWidth, window.innerHeight);
window.addEventListener("resize", () => renderer.setSize(window.innerWidth, window.innerHeight));

// === Scene ===
const scene = new Scene();

// A checkerboard DataTexture for the ground.
const size = 64;
const pixels = new Uint8Array(size * size * 4);
for (let y = 0; y < size; y++) {
  for (let x = 0; x < size; x++) {
    const i = (y * size + x) * 4;
    const dark = ((x >> 3) + (y >> 3)) % 2 === 0;
    const v = dark ? 0.22 : 0.5;
    pixels[i] = v * 255;
    pixels[i + 1] = v * 255;
    pixels[i + 2] = v * 255;
    pixels[i + 3] = 255;
  }
}
const checker = new DataTexture(pixels, size, size);

const ground = new Mesh(new PlaneGeometry(20, 20), new MeshBasicMaterial({ map: checker }));
ground.rotation.x = -Math.PI / 2;
ground.position.y = 0;
scene.add(ground);

// === Standard materials ===
const red = new MeshStandardMaterial({ color: 0xff5533, roughness: 0.25, metalness: 0.6 });
const blue = new MeshStandardMaterial({ color: 0x3399ff, roughness: 0.8, metalness: 0.1 });
const gold = new MeshStandardMaterial({ color: 0xffaa00, roughness: 0.15, metalness: 1 });

// A node-driven material: the color comes from the surface normal, which is
// the node-based material setup — the same graph that compiles to GLSL.
const normalGradient = new MeshStandardMaterial({ roughness: 0.4, metalness: 0.0 });
normalGradient.colorNode = (b) => mix(
  vec3(0.1, 0.2, 0.5),
  vec3(0.9, 0.4, 0.1),
  b.normalWorld.y.add(1).mul(0.5),
);
normalGradient.emissiveNode = (b) => b.normalWorld.mul(0.05);

// An unlit, wireframe-ish material using the fragmentNode escape hatch.
const flat = new MeshBasicMaterial();
flat.colorNode = () => vec3(0.1, 0.9, 0.6);
flat.opacityNode = () => float(0.9);

const box = new Mesh(new BoxGeometry(1.6, 1.6, 1.6), red);
box.position.set(0, 1.4, 0);
scene.add(box);

const sphere = new Mesh(new SphereGeometry(1, 32, 24), normalGradient);
sphere.position.set(-3.4, 1, 0.8);
scene.add(sphere);

const torus = new Mesh(new TorusGeometry(1, 0.35, 20, 48), gold);
torus.position.set(3.4, 1.4, -0.6);
scene.add(torus);

const small = new Mesh(new BoxGeometry(0.5, 0.5, 0.5), blue);
small.position.set(1.6, 0.25, -3);
scene.add(small);

const flatBox = new Mesh(new BoxGeometry(0.6, 0.6, 0.6), flat);
flatBox.position.set(-1.6, 0.3, -3);
scene.add(flatBox);

// === Lights ===
scene.add(new AmbientLight(0xffffff, 0.25));

const sun = new DirectionalLight(0xfff3e0, 2.5);
sun.position.set(6, 10, 8);
scene.add(sun);

const fill = new DirectionalLight(0x8899ff, 0.5);
fill.position.set(-6, 4, -4);
scene.add(fill);

const candle = new PointLight(0xff8800, 6, 8, 2);
candle.position.set(0.6, 0.4, 2.2);
scene.add(candle);

// === Camera ===
const camera = new PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(7, 5, 9);
camera.lookAt(0, 0.5, 0);

// === Orbit controls ===
let theta = 0.8;
let phi = 0.55;
let radius = 12;
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
  radius = Math.max(4, Math.min(30, radius * (1 + e.deltaY * 0.001)));
}, { passive: false });

// === Render loop ===
renderer.setAnimationLoop((time) => {
  const t = time / 1000;
  box.rotation.x = t * 0.6;
  box.rotation.y = t * 0.9;
  torus.rotation.y = t * 0.5;
  torus.rotation.x = Math.sin(t * 0.4) * 0.4;
  sphere.position.y = 1 + Math.sin(t) * 0.15;
  small.rotation.y = t;

  camera.position.set(
    radius * Math.sin(theta) * Math.cos(phi),
    radius * Math.sin(phi),
    radius * Math.cos(theta) * Math.cos(phi),
  );
  camera.lookAt(0, 0.5, 0);

  renderer.render(scene, camera);
});
