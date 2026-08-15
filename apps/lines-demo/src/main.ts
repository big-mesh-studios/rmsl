// RMSL wide-lines demo: `LineSegments2` / `Line2` with `Line2NodeMaterial`.
//
// The same node materials render on WebGL2 and WebGPU — the demo uses the
// WebGL renderer so it runs anywhere. Each line object demonstrates a feature:
//   - cube     pixel-width segments (screen-space `linewidth`)
//   - ring     dashed polyline (from `computeLineDistances`)
//   - helix    world-units polyline (constant world `linewidth`)
//   - spokes   per-segment vertex colors
//
// Run with `pnpm --filter lines-demo dev`.
import {
  WebGLRenderer,
  Scene,
  PerspectiveCamera,
  LineSegments2, LineSegmentsGeometry,
  Line2, LineGeometry,
  Line2NodeMaterial,
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

// === Geometry helpers ===

/** The 12 edges of an axis-aligned box of the given size, as `xyz xyz` pairs. */
function boxEdges(size: number): number[] {
  const s = size / 2;
  const c = [
    [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
    [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
  ];
  const edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  const out: number[] = [];
  for (const [a, b] of edges) out.push(...c[a], ...c[b]);
  return out;
}

/** A polyline tracing a circle in the xy plane, ending where it began. */
function ringPoints(radius: number, segments: number): number[] {
  const out: number[] = [];
  for (let i = 0; i <= segments; i++) {
    const a = (i / segments) * Math.PI * 2;
    out.push(Math.cos(a) * radius, Math.sin(a) * radius, 0);
  }
  return out;
}

/** A helix running up the y axis. */
function helixPoints(radius: number, turns: number, height: number, segments: number): number[] {
  const out: number[] = [];
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const a = t * turns * Math.PI * 2;
    out.push(Math.cos(a) * radius, t * height, Math.sin(a) * radius);
  }
  return out;
}

/** A `[r, g, b]` in 0..1 from a hue in 0..1. */
function hueColor(h: number): [number, number, number] {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const q = 1 - f;
  switch (i % 6) {
    case 0: return [1, f, 0];
    case 1: return [q, 1, 0];
    case 2: return [0, 1, f];
    case 3: return [0, q, 1];
    case 4: return [f, 0, 1];
    default: return [1, 0, q];
  }
}

// === Scene ===
const scene = new Scene();

// 1. Pixel-width wireframe box.
const cubeMaterial = new Line2NodeMaterial({ color: 0x66ccff, linewidth: 3 });
const cube = new LineSegments2(new LineSegmentsGeometry().setPositions(boxEdges(2)), cubeMaterial);
scene.add(cube);

// 2. Dashed ring. `computeLineDistances` fills `instanceDistanceStart/End`,
// which the dash shader needs — required before `dashed` can be enabled.
const ringMaterial = new Line2NodeMaterial({
  color: 0xffaa33,
  linewidth: 4,
  dashed: true,
  dashSize: 0.4,
  gapSize: 0.22,
});
const ringGeometry = new LineGeometry(ringPoints(1.3, 64));
ringGeometry.computeLineDistances();
const ring = new Line2(ringGeometry, ringMaterial);
ring.position.set(-3.2, 1.3, 0);
scene.add(ring);

// 3. World-units helix: the ribbon keeps a constant world width.
const helixMaterial = new Line2NodeMaterial({ color: 0x55ff77, linewidth: 0.06, worldUnits: true });
const helix = new Line2(new LineGeometry(helixPoints(1, 4, 3, 160)), helixMaterial);
helix.position.set(3.2, 0, 0.6);
scene.add(helix);

// 4. Vertex-colored spokes radiating from a point, one hue per segment.
const spokes = 8;
const spokePositions: number[] = [];
const spokeColors: number[] = [];
for (let i = 0; i < spokes; i++) {
  const a = (i / spokes) * Math.PI * 2;
  const rim = [Math.cos(a) * 0.85, Math.sin(a) * 0.85, 0] as const;
  spokePositions.push(0, 0, 0, ...rim);
  const [r, g, b] = hueColor(i / spokes);
  spokeColors.push(r, g, b, r * 0.4, g * 0.4, b * 0.4);
}
const spokeGeometry = new LineSegmentsGeometry();
spokeGeometry.setPositions(spokePositions);
spokeGeometry.setColors(spokeColors);
const spokeMaterial = new Line2NodeMaterial({ vertexColors: true, linewidth: 5 });
const spokesLine = new LineSegments2(spokeGeometry, spokeMaterial);
spokesLine.position.set(-0.6, 2.6, 1.6);
scene.add(spokesLine);

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

// === UI ===
function bindCheckbox(id: string, apply: (checked: boolean) => void): void {
  const el = document.getElementById(id) as HTMLInputElement;
  apply(el.checked);
  el.addEventListener("change", () => apply(el.checked));
}
function bindSlider(id: string, valueId: string, apply: (value: number) => void): void {
  const el = document.getElementById(id) as HTMLInputElement;
  const label = document.getElementById(valueId)!;
  const update = () => {
    const value = parseFloat(el.value);
    label.textContent = el.value;
    apply(value);
  };
  el.addEventListener("input", update);
  update();
}

// Toggling these changes the shader, so the material's setters flag a rebuild.
bindCheckbox("dashed", (checked) => { ringMaterial.dashed = checked; });
bindCheckbox("worldUnits", (checked) => { helixMaterial.worldUnits = checked; });
bindCheckbox("vertexColors", (checked) => { spokeMaterial.vertexColors = checked; });

// Widths and dash sizes are live material uniforms — no shader recompile.
bindSlider("pixelWidth", "pixelWidthVal", (value) => {
  cubeMaterial.linewidth = value;
  ringMaterial.linewidth = value + 1;
});
bindSlider("worldWidth", "worldWidthVal", (value) => { helixMaterial.linewidth = value; });
bindSlider("dashSize", "dashSizeVal", (value) => { ringMaterial.dashSize = value; });
bindSlider("gapSize", "gapSizeVal", (value) => { ringMaterial.gapSize = value; });

// === Render loop ===
renderer.setAnimationLoop((time) => {
  const t = time / 1000;
  if (!isDragging) theta += 0.0025;

  cube.rotation.x = t * 0.5;
  cube.rotation.y = t * 0.7;

  ring.rotation.z = -t * 0.4;

  helix.rotation.y = t * 0.6;

  spokesLine.position.y = 2.6 + Math.sin(t * 0.8) * 0.15;

  camera.position.set(
    radius * Math.sin(theta) * Math.cos(phi),
    radius * Math.sin(phi),
    radius * Math.cos(theta) * Math.cos(phi),
  );
  camera.lookAt(0, 0.5, 0);

  renderer.render(scene, camera);
});
