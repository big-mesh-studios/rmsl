import { describe, it, expect } from "vitest";
import {
  Vector2, Vector3, Vector4, Matrix3, Matrix4, Quaternion, Euler,
  Color, Spherical, degToRad, radToDeg, clamp, lerp, isPowerOfTwo,
} from "./math";

function expectClose(a: number[], b: number[], eps = 1e-6): void {
  expect(a).toHaveLength(b.length);
  for (let i = 0; i < a.length; i++) {
    expect(Math.abs(a[i] - b[i])).toBeLessThan(eps);
  }
}

describe("Vector3", () => {
  it("adds, subtracts and scales", () => {
    const v = new Vector3(1, 2, 3);
    expect(v.add(new Vector3(1, 1, 1)).toArray()).toEqual([2, 3, 4]);
    expect(v.sub(new Vector3(1, 1, 1)).toArray()).toEqual([1, 2, 3]);
    expect(v.multiplyScalar(2).toArray()).toEqual([2, 4, 6]);
  });

  it("computes dot and cross", () => {
    const a = new Vector3(1, 0, 0);
    const b = new Vector3(0, 1, 0);
    expect(a.dot(b)).toBe(0);
    expect(a.dot(new Vector3(2, 3, 4))).toBe(2);
    expect(a.cross(b).toArray()).toEqual([0, 0, 1]);
  });

  it("normalizes", () => {
    const v = new Vector3(3, 0, 0);
    expect(v.normalize().toArray()).toEqual([1, 0, 0]);
    expect(new Vector3().normalize().toArray()).toEqual([0, 0, 0]);
  });

  it("transforms directions through a matrix", () => {
    const m = new Matrix4().makeRotationY(Math.PI / 2);
    const d = new Vector3(1, 0, 0).transformDirection(m);
    expectClose(d.toArray(), [0, 0, -1]);
  });

  it("sets from spherical coordinates", () => {
    const v = new Vector3().setFromSphericalCoords(2, Math.PI / 2, 0);
    expectClose(v.toArray(), [0, 0, 2]);
  });
});

describe("Matrix4", () => {
  it("is identity by default", () => {
    expect(new Matrix4().elements).toEqual([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    ]);
  });

  it("multiplies matrices in the right order", () => {
    const a = new Matrix4().makeTranslation(1, 2, 3);
    const b = new Matrix4().makeRotationZ(Math.PI / 2);
    // a * b applied to a point rotates first, then translates.
    const p = new Vector3(1, 0, 0).applyMatrix4(new Matrix4().multiplyMatrices(a, b));
    expectClose(p.toArray(), [1, 3, 3]);
    // b * a translates first, then rotates.
    const q = new Vector3(1, 0, 0).applyMatrix4(new Matrix4().multiplyMatrices(b, a));
    expectClose(q.toArray(), [-2, 2, 3]);
  });

  it("inverts itself", () => {
    const m = new Matrix4().makeTranslation(3, -2, 7).multiply(new Matrix4().makeRotationX(0.7));
    const v = new Vector3(1, 2, 3);
    const roundTrip = v.clone().applyMatrix4(m).applyMatrix4(m.clone().invert());
    expectClose(roundTrip.toArray(), [1, 2, 3]);
  });

  it("composes and decomposes", () => {
    const pos = new Vector3(1, 2, 3);
    const rot = new Quaternion().setFromEuler(new Euler(0.3, 0.5, 0.7));
    const scale = new Vector3(2, 3, 4);
    const m = new Matrix4().compose(pos, rot, scale);
    const p = new Vector3(), q = new Quaternion(), s = new Vector3();
    m.decompose(p, q, s);
    expectClose(p.toArray(), pos.toArray());
    expectClose(s.toArray(), scale.toArray());
    expectClose(q.toArray(), rot.toArray());
  });

  it("builds a perspective projection", () => {
    const m = new Matrix4().makePerspective(-1, 1, 1, -1, 1, 10);
    // The z column: m[2][2] = -(f+n)/(f-n) at index 10, m[2][3] = -2fn/(f-n)
    // at index 14. Index 11 is the -1 in the w row of the same column.
    expectClose([m.elements[10], m.elements[14]], [-11 / 9, -20 / 9]);
    expect(m.elements[11]).toBe(-1);
    // A view-space point at z = -near maps to clip z = -1.
    expectClose(m.elements.slice(10, 12), [-11 / 9, -1]);
  });

  it("builds an orthographic projection", () => {
    const m = new Matrix4().makeOrthographic(-2, 2, 2, -2, 1, 10);
    const p = new Vector3(2, 2, -1).applyMatrix4(m);
    expectClose(p.toArray(), [1, 1, -1]);
    const q = new Vector3(-2, -2, -10).applyMatrix4(m);
    expectClose(q.toArray(), [-1, -1, 1]);
  });

  it("looks at a target", () => {
    const m = new Matrix4().lookAt(new Vector3(0, 0, 5), new Vector3(0, 0, 0), new Vector3(0, 1, 0));
    // The +z column points from the target toward the eye, so the camera's
    // -z forward points at the target.
    const zAxis = new Vector3(0, 0, 1).transformDirection(m);
    expectClose(zAxis.toArray(), [0, 0, 1]);
    const forward = new Vector3(0, 0, -1).transformDirection(m);
    expectClose(forward.toArray(), [0, 0, -1]);
  });
});

describe("Quaternion", () => {
  it("rotates like the equivalent matrix", () => {
    const euler = new Euler(0.3, -0.7, 1.1);
    const q = new Quaternion().setFromEuler(euler);
    const m = new Matrix4().makeRotationFromQuaternion(q);
    const v = new Vector3(1, 2, 3);
    expectClose(v.clone().applyQuaternion(q).toArray(), v.clone().applyMatrix4(m).toArray());
  });

  it("round-trips through a matrix", () => {
    const q = new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), 0.9);
    const m = new Matrix4().makeRotationFromQuaternion(q);
    const back = new Quaternion().setFromRotationMatrix(m);
    expectClose(back.toArray(), q.toArray());
  });

  it("normalizes", () => {
    expect(new Quaternion(2, 0, 0, 0).normalize().toArray()).toEqual([1, 0, 0, 0]);
    expect(new Quaternion().normalize().toArray()).toEqual([0, 0, 0, 1]);
  });
});

describe("Euler", () => {
  it("round-trips through a quaternion", () => {
    const e = new Euler(0.4, -0.2, 0.8);
    const back = new Euler().setFromQuaternion(new Quaternion().setFromEuler(e));
    expectClose([back.x, back.y, back.z], [e.x, e.y, e.z]);
  });

  it("reorders", () => {
    const e = new Euler(0.5, 0.3, 0.2, "XYZ");
    const re = e.clone().reorder("ZYX");
    const m1 = new Matrix4().makeRotationFromQuaternion(new Quaternion().setFromEuler(e));
    const m2 = new Matrix4().makeRotationFromQuaternion(new Quaternion().setFromEuler(re));
    expectClose(m1.elements, m2.elements);
  });
});

describe("Color", () => {
  it("parses hex", () => {
    const c = new Color().setHex(0xff8000);
    expectClose(c.toArray(), [1, 0x80 / 255, 0]);
  });

  it("round-trips hex", () => {
    expect(new Color().setHex(0x123456).getHex()).toBe(0x123456);
  });

  it("parses css rgb and short hex", () => {
    expectClose(new Color().setStyle("#0f8").toArray(), [0, 1, 0x88 / 255]);
    expectClose(new Color().setStyle("rgb(255, 128, 0)").toArray(), [1, 0x80 / 255, 0]);
  });
});

describe("Spherical", () => {
  it("round-trips with a cartesian vector", () => {
    const v = new Vector3(1, 2, 3);
    const s = new Spherical().setFromVector3(v);
    expectClose(s.toVector3().toArray(), v.toArray());
  });
});

describe("MathUtils", () => {
  it("converts degrees and radians", () => {
    expect(degToRad(180)).toBeCloseTo(Math.PI);
    expect(radToDeg(Math.PI)).toBeCloseTo(180);
  });

  it("clamps and lerps", () => {
    expect(clamp(5, 0, 1)).toBe(1);
    expect(clamp(-1, 0, 1)).toBe(0);
    expect(lerp(0, 10, 0.25)).toBe(2.5);
  });

  it("detects powers of two", () => {
    expect(isPowerOfTwo(8)).toBe(true);
    expect(isPowerOfTwo(6)).toBe(false);
  });
});
