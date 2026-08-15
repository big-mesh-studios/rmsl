import { describe, it, expect } from "vitest";
import {
  Scene, Group, Mesh, PerspectiveCamera, OrthographicCamera,
  AmbientLight, DirectionalLight, PointLight,
  BufferGeometry, BufferAttribute,
  Vector3, Quaternion, Matrix4, Color, degToRad,
} from "./index";

function expectClose(a: number[], b: number[], eps = 1e-6): void {
  expect(a).toHaveLength(b.length);
  for (let i = 0; i < a.length; i++) {
    expect(Math.abs(a[i] - b[i])).toBeLessThan(eps);
  }
}

describe("Object3D hierarchy", () => {
  it("composes world matrices down the tree", () => {
    const parent = new Group();
    parent.position.set(1, 0, 0);
    const child = new Mesh();
    child.position.set(0, 2, 0);
    parent.add(child);

    parent.updateMatrixWorld(true);

    const world = new Vector3().setFromMatrixPosition(child.matrixWorld);
    expectClose(world.toArray(), [1, 2, 0]);
  });

  it("applies rotation through the parent", () => {
    const parent = new Group();
    parent.rotation.z = Math.PI / 2;
    const child = new Mesh();
    child.position.set(1, 0, 0);
    parent.add(child);

    parent.updateMatrixWorld(true);

    const world = new Vector3().setFromMatrixPosition(child.matrixWorld);
    expectClose(world.toArray(), [0, 1, 0]);
  });

  it("removes and clears children", () => {
    const g = new Group();
    const a = new Mesh();
    const b = new Mesh();
    g.add(a, b);
    expect(g.children).toHaveLength(2);
    g.remove(a);
    expect(g.children).toEqual([b]);
    expect(a.parent).toBeNull();
    g.clear();
    expect(g.children).toEqual([]);
  });

  it("traverses and finds by name", () => {
    const scene = new Scene();
    const g = new Group();
    g.name = "root";
    const m = new Mesh();
    m.name = "box";
    g.add(m);
    scene.add(g);

    const names: string[] = [];
    scene.traverse((o) => names.push(o.name));
    expect(names).toEqual(["", "root", "box"]);
    expect(scene.getObjectByName("box")).toBe(m);
  });

  it("gets world position, quaternion and scale", () => {
    const g = new Group();
    g.position.set(5, 0, 0);
    g.scale.set(2, 2, 2);
    g.rotation.set(0, Math.PI / 2, 0);

    expectClose(g.getWorldPosition().toArray(), [5, 0, 0]);
    expectClose(g.getWorldScale().toArray(), [2, 2, 2]);
    const expected = new Quaternion().setFromEuler(g.rotation);
    expectClose(g.getWorldQuaternion().toArray(), expected.toArray());
  });

  it("looks at a target", () => {
    // A mesh orients its +z axis toward the target (three.js convention;
    // cameras orient -z).
    const m = new Mesh();
    m.position.set(0, 0, 5);
    m.lookAt(new Vector3(0, 0, 0));
    m.updateMatrixWorld(true);
    const zAxis = new Vector3(m.matrixWorld.elements[8], m.matrixWorld.elements[9], m.matrixWorld.elements[10]);
    expectClose(zAxis.normalize().toArray(), [0, 0, -1]);
  });

  it("camera forward points at the look target", () => {
    const cam = new PerspectiveCamera();
    cam.position.set(0, 0, 5);
    cam.lookAt(0, 0, 0);
    cam.updateMatrixWorld(true);
    expectClose(cam.getWorldDirection().toArray(), [0, 0, -1]);
  });

  it("emits added/removed events", () => {
    const g = new Group();
    const a = new Mesh();
    let added = 0;
    let removed = 0;
    a.addEventListener("added", () => added++);
    a.addEventListener("removed", () => removed++);
    g.add(a);
    g.remove(a);
    expect(added).toBe(1);
    expect(removed).toBe(1);
  });
});

describe("Scene", () => {
  it("has a background color", () => {
    const scene = new Scene();
    scene.background = new Color().setHex(0x112233);
    expect(scene.background?.getHex()).toBe(0x112233);
  });
});

describe("cameras", () => {
  it("builds a perspective projection matrix", () => {
    const cam = new PerspectiveCamera(60, 2, 0.1, 100);
    // Corner ray: a point at the near plane and half the frustum away should
    // land exactly on clip-space (±1, ±1).
    const top = 0.1 * Math.tan(degToRad(30));
    const right = top * cam.aspect;
    const p = new Vector3(right, top, -0.1).applyMatrix4(cam.projectionMatrix);
    expectClose(p.toArray(), [1, 1, -1]);
  });

  it("inverts the view matrix", () => {
    const cam = new PerspectiveCamera();
    cam.position.set(2, 3, 4);
    cam.lookAt(0, 0, 0);
    cam.updateMatrixWorld(true);
    // A world point at the camera origin should land at the origin of view
    // space.
    const origin = new Vector3(2, 3, 4).applyMatrix4(cam.matrixWorldInverse);
    expectClose(origin.toArray(), [0, 0, 0]);
  });

  it("builds an orthographic projection matrix", () => {
    const cam = new OrthographicCamera(-2, 2, 2, -2, 0.1, 10);
    const p = new Vector3(2, 2, -0.1).applyMatrix4(cam.projectionMatrix);
    expectClose(p.toArray(), [1, 1, -1]);
  });
});

describe("lights", () => {
  it("parses a numeric color", () => {
    const light = new AmbientLight(0xff0000, 2);
    expect(light.color.getHex()).toBe(0xff0000);
    expect(light.intensity).toBe(2);
    expect(light.isLight).toBe(true);
  });

  it("is a positional scene node", () => {
    const light = new PointLight(0xffffff, 1, 10, 2);
    light.position.set(1, 2, 3);
    expectClose(light.getWorldPosition().toArray(), [1, 2, 3]);
  });

  it("directional light has a target", () => {
    const light = new DirectionalLight();
    expect(light.target.isObject3D).toBe(true);
  });
});

describe("BufferGeometry", () => {
  it("holds attributes and an index", () => {
    const geo = new BufferGeometry();
    geo.setAttribute("position", new BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0]), 3));
    geo.setAttribute("uv", new BufferAttribute(new Float32Array([0, 0, 1, 0]), 2));
    geo.setIndex(new Uint16Array([0, 1]));
    expect(geo.attributes.position?.count).toBe(2);
    expect(geo.index?.count).toBe(2);
    expect(geo.drawCount).toBe(2);
  });
});
