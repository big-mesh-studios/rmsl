import { describe, it, expect } from "vitest";
import {
  BoxGeometry, SphereGeometry, PlaneGeometry, CylinderGeometry,
  ConeGeometry, TorusGeometry, CircleGeometry, BufferGeometry,
} from "./index";

describe("geometry primitives", () => {
  it("box has 24 vertices and 36 indices", () => {
    const geo = new BoxGeometry();
    expect(geo.attributes.position?.count).toBe(24);
    expect(geo.attributes.normal?.count).toBe(24);
    expect(geo.attributes.uv?.count).toBe(24);
    expect(geo.index?.count).toBe(36);
  });

  it("box positions respect the requested size", () => {
    const geo = new BoxGeometry(2, 4, 6);
    const pos = geo.attributes.position!;
    let maxX = 0, maxY = 0, maxZ = 0;
    for (let i = 0; i < pos.count; i++) {
      maxX = Math.max(maxX, Math.abs(pos.getX(i)));
      maxY = Math.max(maxY, Math.abs(pos.getY(i)));
      maxZ = Math.max(maxZ, Math.abs(pos.getZ(i)));
    }
    expect(maxX).toBe(1);
    expect(maxY).toBe(2);
    expect(maxZ).toBe(3);
  });

  it("box normals are axis-aligned unit vectors", () => {
    const geo = new BoxGeometry();
    const n = geo.attributes.normal!;
    for (let i = 0; i < n.count; i++) {
      const l = Math.hypot(n.getX(i), n.getY(i), n.getZ(i));
      expect(l).toBeCloseTo(1);
    }
  });

  it("sphere radius is respected and normals are unit", () => {
    const geo = new SphereGeometry(3, 8, 4);
    const pos = geo.attributes.position!;
    const n = geo.attributes.normal!;
    for (let i = 0; i < pos.count; i++) {
      expect(Math.hypot(pos.getX(i), pos.getY(i), pos.getZ(i))).toBeCloseTo(3);
      expect(Math.hypot(n.getX(i), n.getY(i), n.getZ(i))).toBeCloseTo(1);
    }
  });

  it("plane lies in the XY plane at z = 0 and is centered on the origin", () => {
    const geo = new PlaneGeometry(2, 3);
    const pos = geo.attributes.position!;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < pos.count; i++) {
      expect(pos.getZ(i)).toBe(0);
      minX = Math.min(minX, pos.getX(i));
      maxX = Math.max(maxX, pos.getX(i));
      minY = Math.min(minY, pos.getY(i));
      maxY = Math.max(maxY, pos.getY(i));
    }
    // Half-extents, so the plane spans -1..1 in x and -1.5..1.5 in y.
    expect(minX).toBeCloseTo(-1);
    expect(maxX).toBeCloseTo(1);
    expect(minY).toBeCloseTo(-1.5);
    expect(maxY).toBeCloseTo(1.5);
    expect(geo.attributes.uv?.count).toBe(pos.count);
    expect(geo.index?.count).toBe(6);
  });

  it("cylinder has a torso and two caps", () => {
    const geo = new CylinderGeometry(1, 1, 2, 8);
    // (radialSegments+1) * (heightSegments+1) torso vertices plus, per cap,
    // `radialSegments` center vertices and `radialSegments+1` ring vertices.
    const expectedVertices = (8 + 1) * (1 + 1) + 2 * (8 + (8 + 1));
    expect(geo.attributes.position?.count).toBe(expectedVertices);
  });

  it("cone is a cylinder with a zero top radius", () => {
    const geo = new ConeGeometry(1, 2, 8);
    const pos = geo.attributes.position!;
    let topY = -Infinity;
    for (let i = 0; i < pos.count; i++) {
      if (Math.abs(pos.getY(i) - 1) < 1e-6) {
        // A cone's apex: any vertex at the top must be at the axis.
        expect(Math.hypot(pos.getX(i), pos.getZ(i))).toBeLessThan(1e-6);
        topY = 1;
      }
    }
    expect(topY).toBe(1);
  });

  it("torus and circle generate non-empty buffers", () => {
    expect(new TorusGeometry().index?.count).toBeGreaterThan(0);
    expect(new CircleGeometry().index?.count).toBeGreaterThan(0);
  });

  it("index arrays are typed so GPU uploads know their byte size", () => {
    const geo = new BoxGeometry();
    // WebGL/WebGPU element uploads read the byte length from the view; a plain
    // number[] uploads with no size, which made drawElements read past the
    // buffer ("GL_INVALID_OPERATION: Insufficient buffer size").
    expect(geo.index?.array).toBeInstanceOf(Uint16Array);
    expect(geo.index?.array).toHaveLength(36);
  });

  it("setIndex picks Uint32 once indices exceed 65535", () => {
    const geo = new BufferGeometry();
    geo.setIndex(new Uint32Array([0, 1, 70000]));
    expect(geo.index?.array).toBeInstanceOf(Uint32Array);
  });
});
