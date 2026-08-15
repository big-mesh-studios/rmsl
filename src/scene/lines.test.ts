import { describe, it, expect, afterAll } from "vitest";
import {
  recordingGLSL as compileGLSL,
  recordingWGSL as compileWGSL,
  assertRecordedShadersValid,
} from "../testing/shader-validity";
import {
  Scene,
  LineSegmentsGeometry, LineGeometry,
  Line2NodeMaterial, LineSegments2, Line2,
} from "./index";

afterAll(async () => {
  await assertRecordedShadersValid();
}, 120_000);

function compileMaterial(material: Line2NodeMaterial, scene = new Scene()): { glsl: string; wgsl: string } {
  const program = material.build(scene);
  const glsl = compileGLSL.vertex(program.vertexRoot)
    + "\n---\n"
    + compileGLSL.fragment(program.fragmentRoot);
  const wgsl = compileWGSL.vertex(program.vertexRoot)
    + "\n---\n"
    + compileWGSL.fragment(program.fragmentRoot);
  return { glsl, wgsl };
}

describe("LineSegmentsGeometry", () => {
  it("builds instanced start/end attributes from positions", () => {
    const geometry = new LineSegmentsGeometry();
    geometry.setPositions([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]);

    expect(geometry.instanceCount).toBe(2);
    expect(geometry.attributes.instanceStart!.count).toBe(2);
    expect(geometry.attributes.instanceEnd!.count).toBe(2);
    expect(geometry.attributes.instanceStart!.stepMode).toBe("instance");
    expect(geometry.attributes.instanceEnd!.stepMode).toBe("instance");
    expect(Array.from(geometry.attributes.instanceStart!.array)).toEqual([0, 0, 0, 0, 0, 0]);
    expect(Array.from(geometry.attributes.instanceEnd!.array)).toEqual([1, 0, 0, 0, 1, 0]);
  });

  it("computes line distances for the dash shader", () => {
    const geometry = new LineSegmentsGeometry();
    geometry.setPositions([0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 4, 0]);
    geometry.computeLineDistances();

    const start = Array.from(geometry.attributes.instanceDistanceStart!.array);
    const end = Array.from(geometry.attributes.instanceDistanceEnd!.array);
    expect(start).toEqual([0, 3]);
    expect(end).toEqual([3, 7]);
    expect(geometry.attributes.instanceDistanceStart!.stepMode).toBe("instance");
  });

  it("LineGeometry builds consecutive segments from points", () => {
    const geometry = new LineGeometry([0, 0, 0, 1, 0, 0, 1, 1, 0]);
    expect(geometry.instanceCount).toBe(2);
    expect(Array.from(geometry.attributes.instanceStart!.array)).toEqual([0, 0, 0, 1, 0, 0]);
    expect(Array.from(geometry.attributes.instanceEnd!.array)).toEqual([1, 0, 0, 1, 1, 0]);
  });
});

describe("Line2NodeMaterial", () => {
  it("registers the instanced line attributes", () => {
    const material = new Line2NodeMaterial();
    const program = material.build(new Scene());
    expect(program.attributes.map((a) => [a.name, a.stepMode])).toEqual(
      expect.arrayContaining([
        ["position", "vertex"],
        ["uv", "vertex"],
        ["instanceStart", "instance"],
        ["instanceEnd", "instance"],
      ]),
    );
  });

  it("exposes resolution as a renderer-scoped uniform", () => {
    const material = new Line2NodeMaterial();
    const program = material.build(new Scene());
    const resolution = program.uniforms.find((u) => u.name === "resolution")!;
    expect(resolution).toBeDefined();
    expect(resolution.scope).toBe("renderer");
  });

  it("compiles the pixel-width line to GLSL and WGSL", () => {
    const material = new Line2NodeMaterial({ color: 0xff0000, linewidth: 4 });
    const { glsl, wgsl } = compileMaterial(material);

    expect(glsl).toContain("uniform vec2 resolution");
    expect(glsl).toContain("uniform float materialLineWidth");
    expect(glsl).toContain("discard;");
    expect(wgsl).toContain("materialLineWidth");
    expect(wgsl).toContain("@builtin(position) position: vec4<f32>");
  });

  it("adds the world-units varyings only when worldUnits is set", () => {
    const names = (material: Line2NodeMaterial) => material.build(new Scene()).varyings.map((v) => v.name);

    const pixel = new Line2NodeMaterial();
    expect(names(pixel)).not.toContain("worldStart");
    expect(names(pixel)).not.toContain("worldPos");

    const world = new Line2NodeMaterial({ worldUnits: true });
    expect(names(world)).toContain("worldStart");
    expect(names(world)).toContain("worldEnd");
    expect(names(world)).toContain("worldPos");
  });

  it("declares the dash uniforms and distance attributes when dashed", () => {
    const plain = compileMaterial(new Line2NodeMaterial());
    expect(plain.glsl).not.toContain("materialLineDashSize");

    const dashed = new Line2NodeMaterial({ dashed: true });
    const program = dashed.build(new Scene());
    const { glsl } = compileMaterial(dashed);
    expect(glsl).toContain("uniform float materialLineScale");
    expect(glsl).toContain("uniform float materialLineDashSize");
    expect(glsl).toContain("uniform float materialLineGapSize");
    expect(glsl).toContain("uniform float materialLineDashOffset");
    expect(program.varyings.map((v) => v.name)).toContain("lineDistance");
    expect(program.attributes.map((a) => a.name)).toEqual(
      expect.arrayContaining(["instanceDistanceStart", "instanceDistanceEnd"]),
    );
  });

  it("reads per-segment colors only when vertexColors is set", () => {
    const plain = new Line2NodeMaterial();
    expect(plain.build(new Scene()).attributes.map((a) => a.name)).not.toContain("instanceColorStart");

    const colored = new Line2NodeMaterial({ vertexColors: true });
    const program = colored.build(new Scene());
    expect(program.attributes.map((a) => a.name)).toEqual(
      expect.arrayContaining(["instanceColorStart", "instanceColorEnd"]),
    );
    expect(program.varyings.map((v) => v.name)).toContain("instanceColor");
  });

  it("renders double-sided and toggles rebuild on worldUnits", () => {
    const material = new Line2NodeMaterial();
    expect(material.side).toBe(2); // Side.DoubleSide
    expect(material.needsUpdate).toBe(false);
    material.worldUnits = true;
    expect(material.needsUpdate).toBe(true);
  });
});

describe("LineSegments2 objects", () => {
  it("computes line distances on the geometry", () => {
    const line = new Line2(new LineGeometry([0, 0, 0, 2, 0, 0, 2, 3, 0]));
    line.computeLineDistances();
    expect(Array.from(line.geometry.attributes.instanceDistanceEnd!.array)).toEqual([2, 5]);
  });

  it("is a Mesh", () => {
    const line = new LineSegments2();
    expect(line.isMesh).toBe(true);
    expect(line.isLineSegments2).toBe(true);
    expect(line.material.isLine2NodeMaterial).toBe(true);
  });
});
