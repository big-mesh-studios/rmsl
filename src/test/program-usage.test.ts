import { describe, it, expect } from "vitest";
import { uniform, vec2, vec4 } from "../rmsl";
import { sepia } from "../effects";
import {
  Scene,
  DataTexture,
  MeshBasicMaterial,
  MeshStandardMaterial,
  DirectionalLight,
  AmbientLight,
  NearestFilter,
  RepeatWrapping,
} from "../scene";
import { fromProgram, fromPass, render, closeTo, uniformsIn } from "./index";

/** A material built the way a renderer builds it, with no renderer involved. */
function build(material: MeshBasicMaterial | MeshStandardMaterial, scene = new Scene()) {
  return material.build(scene, {});
}

describe("fromProgram", () => {
  it("shades a material from its own uniforms", () => {
    const shade = fromProgram(build(new MeshBasicMaterial({ color: 0x336699 })));
    expect(shade().value).toEqual([0.2, 0.4, 0.6, 1]);
  });

  it("shades a lit surface handed in by name", () => {
    const scene = new Scene();
    scene.add(new AmbientLight(0xffffff, 0));
    const light = new DirectionalLight(0xffffff, 1);
    light.position.set(0, 1, 0);
    scene.add(light);
    scene.updateMatrixWorld(true);

    const shade = fromProgram(build(new MeshStandardMaterial({ color: 0xffffff, roughness: 1 }), scene), {
      uniforms: { cameraPosition: [0, 0, 5] },
    });
    const brightness = (normal: number[]): number => {
      const result = shade({ varyings: { normalWorld: normal, positionWorld: [0, 0, 0] } });
      return (result.value as number[])[0];
    };

    // The light is overhead: a surface facing it is lit, one facing away or
    // edge-on to it is not. This is the lighting model itself, under test with
    // no renderer, no canvas and no light rig.
    expect(brightness([0, 1, 0])).toBeGreaterThan(0.3);
    expect(brightness([0, -1, 0])).toBeLessThan(0.01);
    expect(brightness([1, 0, 0])).toBeLessThan(0.01);
  });

  it("samples a DataTexture the material already points at", () => {
    const material = new MeshBasicMaterial({ color: 0xffffff });
    material.map = new DataTexture(new Uint8Array([0, 0, 255, 255]), 1, 1);
    const shade = fromProgram(build(material));
    const colour = shade({ varyings: { uv: [0.5, 0.5] } }).value as number[];
    expect(closeTo(colour.slice(0, 3), [0, 0, 1])).toBe(true);
  });

  it("says which uniforms nothing filled in", () => {
    // Every uniform of a plain material is either the material's own or one of
    // the matrices a renderer would supply, and both are covered.
    expect(fromProgram(build(new MeshBasicMaterial())).unbound).toEqual([]);

    const stray = uniform("float");
    const program = build(new MeshBasicMaterial());
    const withStray = {
      ...program,
      fragmentRoot: vec4(program.fragmentRoot.rgb.mul(stray), 1),
      uniforms: [...program.uniforms, { name: "stray", node: stray, scope: "material" }],
    };
    expect(fromProgram(withStray).unbound).toEqual(["stray"]);
  });

  it("lets a caller override what the material says", () => {
    const program = build(new MeshBasicMaterial({ color: 0x000000 }));
    const shade = fromProgram(program, { uniforms: { materialColor: [1, 0, 0] } });
    expect(shade().value).toEqual([1, 0, 0, 1]);
    expect(shade({ uniforms: { materialColor: [0, 1, 0] } }).value).toEqual([0, 1, 0, 1]);
  });

  it("renders a material across a grid, varying the surface per fragment", () => {
    const material = new MeshBasicMaterial({ color: 0xffffff });
    material.map = new DataTexture(
      new Uint8Array([0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255, 255, 255, 255, 255]),
      2,
      2,
    );
    const shade = fromProgram(build(material));
    const image = render(shade, {
      width: 2,
      height: 2,
      inputs: ({ u, v }) => ({ varyings: { uv: [u, v] } }),
    });
    // The texture's left column is black and its right column white.
    expect(image.at(0, 0)[0]).toBe(0);
    expect(image.at(1, 0)[0]).toBe(1);
  });

  it("reads a texture the way the texture asks to be read", () => {
    // The same two texels, coordinates and answers as the WebGL renderer's
    // driver test in `renderer.test.ts` — which is the point: what a test
    // measures here is what a renderer draws.
    const texels = () => new Uint8Array([255, 0, 0, 255, 0, 0, 255, 255]);
    const shadeAt = (texture: DataTexture, x: number): number[] => {
      const material = new MeshBasicMaterial({ color: 0xffffff });
      material.map = texture;
      const program = build(material);
      const shade = fromProgram(program);
      return shade({ varyings: { uv: [x, 0.5] } }).value as number[];
    };

    const clamped = new DataTexture(texels(), 2, 1);
    clamped.magFilter = NearestFilter;
    expect(closeTo(shadeAt(clamped, 1.25).slice(0, 3), [0, 0, 1])).toBe(true);

    const tiled = new DataTexture(texels(), 2, 1);
    tiled.magFilter = NearestFilter;
    tiled.wrapS = RepeatWrapping;
    expect(closeTo(shadeAt(tiled, 1.25).slice(0, 3), [1, 0, 0])).toBe(true);

    // Between the two texel centres: one texel with nearest, both blended
    // under the default linear filtering.
    expect(closeTo(shadeAt(clamped, 0.5).slice(0, 3), [0, 0, 1])).toBe(true);
    const smooth = new DataTexture(texels(), 2, 1);
    const blended = shadeAt(smooth, 0.5);
    expect(blended[0]).toBeGreaterThan(0.4);
    expect(blended[2]).toBeGreaterThan(0.4);
  });

  it("hands a vertex stage's varyings back under the program's names", () => {
    const material = new MeshBasicMaterial({ color: 0xffffff });
    material.map = new DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
    const program = build(material);
    const project = fromProgram(program, { stage: "vertex" });
    const written = project({ attributes: { uv: [0.25, 0.75], position: [0, 0, 0], normal: [0, 0, 1] } });

    // What went in by name comes back by name: nothing here has to look
    // `_rmsl_v1` up in the program to read the varying it just wrote.
    expect(Object.keys(written.varyings)).toContain("uv");
    expect(written.varyings.uv).toEqual([0.25, 0.75]);
  });

  it("says which input it wanted when nothing bound one", () => {
    const project = fromProgram(build(new MeshBasicMaterial()), { stage: "vertex" });
    // The vertex stage transforms the normal, and nothing here handed it one.
    // What used to come out of this is a TypeError inside a matrix multiply.
    expect(() => project({ attributes: { position: [0, 0, 0], uv: [0, 0] } })).toThrow(
      /vertex stage reads the attribute "normal".*attributes/s,
    );

    const shade = fromProgram(build(new MeshStandardMaterial()));
    expect(() => shade({ varyings: { normalWorld: [0, 1, 0] } })).toThrow(
      /fragment stage reads the varying "positionWorld"/,
    );
  });

  it("points at `unbound` for an input the program was meant to bring", () => {
    const stray = uniform("float");
    const program = build(new MeshBasicMaterial());
    const withStray = {
      ...program,
      fragmentRoot: vec4(program.fragmentRoot.rgb.mul(stray), 1),
      uniforms: [...program.uniforms, { name: "stray", node: stray, scope: "material" }],
    };
    // Shading with an undefined uniform used to give a colour of NaN, which
    // says nothing about where it came from.
    expect(() => fromProgram(withStray)()).toThrow(/the uniform "stray".*`unbound`/s);
  });

  it("counts a sampler with no pixels behind it as unbound", () => {
    const material = new MeshBasicMaterial({ color: 0xffffff });
    // What a texture loaded from a URL looks like with no browser to load it.
    material.map = { image: undefined, width: 0, height: 0 } as never;
    const shade = fromProgram(build(material));
    expect(shade.unbound).toContain("map");
    expect(() => shade({ varyings: { uv: [0, 0] } })).toThrow(/the texture "map"/);
  });

  it("refuses a stage the program has no root for", () => {
    const program = build(new MeshBasicMaterial());
    expect(() => fromProgram({ fragmentRoot: program.fragmentRoot }, { stage: "vertex" })).toThrow(/no vertex root/);
  });
});

describe("fromPass", () => {
  it("reads a pass's input texture by the name the pass gave it", () => {
    const source = uniform("sampler2D");
    const pass = {
      name: "sepia",
      color: sepia(source.texture(vec2(0.5, 0.5))),
      inputs: { source },
    };
    const run = fromPass(pass);
    const result = run({ textures: { source: { data: [1, 1, 1, 1], width: 1, height: 1 } } });
    // Sepia of white: warm, and brighter in red than in blue.
    const [r, g, b] = result.value as number[];
    expect(r).toBeGreaterThan(g);
    expect(g).toBeGreaterThan(b);
  });

  it("renders a pass over a grid, its screen size bound by hand", () => {
    const source = uniform("sampler2D");
    const pass = { color: vec4(source.texture(vec2(0.5, 0.5)).rgb, 1), inputs: { source } };
    const image = render(
      fromPass(pass, {
        textures: { source: { data: [0, 128 / 255, 1, 1], width: 1, height: 1 } },
      }),
      { width: 2, height: 2 },
    );
    expect(closeTo(image.at(1, 1), [0, 128 / 255, 1, 1])).toBe(true);
  });

  it("leaves a pass's own screen-size uniform to uniformsIn", () => {
    const source = uniform("sampler2D");
    const pass = { color: vec4(source.texture(vec2(0, 0)).rgb, 1), inputs: { source } };
    expect(uniformsIn(pass.color).map((node) => node.name)).toEqual([source.name]);
  });
});
