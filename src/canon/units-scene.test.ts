import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { vec4 } from "../rmsl";
import {
  AmbientLight,
  DirectionalLight,
  Mesh,
  MeshBasicMaterial,
  MeshLambertMaterial,
  PerspectiveCamera,
  PlaneGeometry,
  PointLight,
  Scene,
  WebGLRenderer,
  WebGLRenderTarget,
  WebGPURenderer,
} from "../scene";

interface Call {
  name: string;
  args: any[];
}

/** A `GPUDevice` and canvas whose shader modules keep the WGSL they were made from. */
function stubDevice() {
  const modules: { code: string }[] = [];
  const device = {
    createShaderModule: (descriptor: any) => {
      modules.push(descriptor);
      return descriptor;
    },
    createBuffer: (descriptor: any) => ({ size: descriptor.size, destroy: () => {} }),
    createBindGroupLayout: (descriptor: any) => descriptor,
    createPipelineLayout: (descriptor: any) => descriptor,
    createRenderPipeline: (descriptor: any) => descriptor,
    createSampler: (descriptor: any) => descriptor,
    createBindGroup: (descriptor: any) => descriptor,
    createTexture: (descriptor: any) => ({ createView: () => ({}), destroy: () => {}, ...descriptor }),
    queue: { writeBuffer: () => {}, writeTexture: () => {}, submit: () => {} },
  };
  const canvas: any = {
    width: 16,
    height: 16,
    getContext: () => ({ configure: () => {}, getCurrentTexture: () => ({ createView: () => ({}) }) }),
  };
  return { device, canvas, modules };
}

/**
 * A WebGL renderer drawing into a `WebGL2RenderingContext` that accepts every
 * call and records it. Constants read as distinct numbers, shaders compile and
 * programs link.
 */
function stubWebGl() {
  const canvas: any = { width: 32, height: 32 };
  const calls: Call[] = [];
  const constants = new Map<string, number>();
  let location = 0;
  const answers: Record<string, (...args: any[]) => unknown> = {
    getShaderParameter: () => true,
    getProgramParameter: () => true,
    getParameter: () => 16,
    getAttribLocation: () => (location += 4),
    getUniformLocation: (_program: unknown, name: string) => ({ name }),
  };
  const gl = new Proxy(
    {},
    {
      get: (_target, name: string) => {
        if (name === "drawingBufferWidth") return canvas.width;
        if (name === "drawingBufferHeight") return canvas.height;
        if (/^[A-Z0-9_]+$/.test(name)) {
          if (!constants.has(name)) constants.set(name, 0x1000 + constants.size);
          return constants.get(name);
        }
        return (...args: any[]) => {
          calls.push({ name, args });
          return answers[name]?.(...args) ?? {};
        };
      },
    },
  ) as any;
  canvas.getContext = () => gl;
  return { renderer: new WebGLRenderer(canvas), gl, calls };
}

function camera(): PerspectiveCamera {
  const c = new PerspectiveCamera(50, 1, 0.1, 100);
  c.position.set(0, 0, 4);
  c.lookAt(0, 0, 0);
  return c;
}

/** The value a built program's uniform of that name holds now. */
function uniformValue(material: MeshLambertMaterial, scene: Scene, name: string): unknown {
  const binding = material.build(scene).uniforms.find((u) => u.name === name)!;
  return binding.value!({} as any);
}

beforeEach(() => {
  vi.stubGlobal("navigator", { gpu: { getPreferredCanvasFormat: () => "bgra8unorm" } });
  vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1, COPY_DST: 2, VERTEX: 4, INDEX: 8 });
  vi.stubGlobal("GPUTextureUsage", { TEXTURE_BINDING: 1, COPY_DST: 2, RENDER_ATTACHMENT: 4 });
  vi.stubGlobal("GPUShaderStage", { VERTEX: 1, FRAGMENT: 2 });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("a scene renderer manages what it uploads", () => {
  /**
   * The vertex stage reads only the matrices and the fragment stage only the
   * material's tint, yet both declare every uniform at the same offsets, so
   * one buffer serves both.
   *
   * @canon spec-the-webgpu-renderer-declares-one-uniform-struct-in-both-stages
   */
  it("declares the same uniform struct in the vertex and fragment WGSL", () => {
    const { device, canvas, modules } = stubDevice();
    const renderer = new WebGPURenderer(canvas, device as any) as any;
    const material = new MeshBasicMaterial();
    material.fragmentNode = (b) =>
      vec4(
        b.materialUniform("tint", "vec3", () => [1, 0, 0]),
        1,
      );
    renderer.ensurePipeline(material, new Scene(), false, false);

    const struct = (code: string) => /struct \w+ \{[^}]*\}/.exec(code)![0];
    expect(modules).toHaveLength(2);
    expect(struct(modules[0].code)).toContain("tint");
    expect(struct(modules[1].code)).toContain("modelMatrix");
    expect(struct(modules[0].code)).toBe(struct(modules[1].code));
  });

  /**
   * Growing a target between two renders frees its old framebuffer and
   * allocates colour and depth storage at the new size.
   *
   * @canon spec-a-render-target-takes-its-new-size-on-the-next-render
   */
  it("redraws a render target at the size it was given since the last render", () => {
    const { renderer, gl, calls } = stubWebGl();
    const target = new WebGLRenderTarget(8, 4);
    renderer.render(new Scene(), camera(), target);
    target.width = 16;
    target.height = 8;
    const before = calls.length;
    renderer.render(new Scene(), camera(), target);

    const after = calls.slice(before);
    expect(after.map((c) => c.name)).toContain("deleteFramebuffer");
    expect(after.find((c) => c.name === "texImage2D")!.args.slice(3, 5)).toEqual([16, 8]);
    expect(after.find((c) => c.name === "renderbufferStorage")!.args.slice(1)).toEqual([gl.DEPTH_COMPONENT24, 16, 8]);
    expect(after.find((c) => c.name === "viewport")!.args).toEqual([0, 0, 16, 8]);
  });
});

describe("a node material lights as three.js does", () => {
  /** @canon spec-ambient-lights-sum-into-one-colour */
  it("folds two ambient lights into one uniform holding their sum", () => {
    const scene = new Scene();
    scene.add(new AmbientLight(0xff0000, 0.5));
    scene.add(new AmbientLight(0x0000ff, 2));
    const material = new MeshLambertMaterial();
    const program = material.build(scene);

    expect(program.uniforms.filter((u) => /ambient/i.test(u.name)).map((u) => u.name)).toEqual(["ambientColor"]);
    expect(uniformValue(material, scene, "ambientColor")).toEqual([0.5, 0, 2]);
  });

  /**
   * A directional and a point light each hand the shader their colour already
   * multiplied by their intensity.
   *
   * @canon spec-a-light-uniform-carries-its-colour-times-its-intensity
   */
  it.each([
    ["directional", () => new DirectionalLight(0xff8000, 3), "directionalColor0"],
    ["point", () => new PointLight(0xff8000, 3), "pointColor0"],
  ])("scales a %s light's colour by its intensity", (_kind, light, name) => {
    const scene = new Scene();
    const source = light();
    scene.add(source);
    const [r, g, b] = source.color.toArray();

    const value = uniformValue(new MeshLambertMaterial(), scene, name) as number[];
    expect(value[0]).toBeCloseTo(r * 3, 6);
    expect(value[1]).toBeCloseTo(g * 3, 6);
    expect(value[2]).toBeCloseTo(b * 3, 6);
  });
});
