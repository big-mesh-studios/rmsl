/**
 * Where a draw of the JS rasterizer, and of the WASM one beside it, allocates,
 * for each kind of scalar a stage takes or gives. A site is a function of the
 * library, and the stages a draw calls are inlined into it. Each case runs in
 * a worker of its own.
 *
 * Usage: pnpm build && node probes/v8-float-boxing/draw.ts
 */
import { Worker, isMainThread, workerData, parentPort } from "node:worker_threads";
import { allocatedBy, versions } from "./measure.ts";

const cases = [
  "vec3 varying",
  "float varying",
  "float varying, whole-number colours",
  "float varying, float: f32",
  "float attribute, vec3 varying",
  "int varying",
  "uint varying above 2^31",
  "bool varying",
  "float uniform read in the fragment",
  "fragment writes its depth",
  "float outputStruct member",
  "vec3 varying, after draws from 2 kinds of host array",
  "vec3 varying, after draws from 4 kinds of host array",
  "vec3 varying, after draws from 6 kinds of host array",
  "int attribute",
  "uint attribute",
  "uint attribute of 2^31 or more",
  "vec3 varying, both triangles crossing the near plane",
  "float attribute, after draws from 4 kinds of host array",
  "vec3 varying, after draws with varyings in slots of 4 kinds",
  "vec3 varying, after a host object made alike with the fragment context held undefined in fragDepth",
  "one sampler, after draws of it with texel data of 4 other kinds",
  "five samplers with texel data of five kinds, in one draw",
  "five samplers with texel data of one kind, in one draw",
  "vec3 uniform, after draws of it given in 4 other kinds of array",
  "mat3 uniform, after draws of it given in 4 other kinds of array",
  "vec4 uniform array element, after draws of it given in 4 other kinds of array",
  "vec3 uniform in the vertex stage, after draws of it given in 4 other kinds of array",
  "sampler in the vertex stage, after draws of it with texel data of 4 other kinds",
  "linear-filtered sampler, after draws of it with texel data of 4 other kinds",
  "textureLoad, after draws of it with texel data of 4 other kinds",
  "float uniform, after draws given uniforms objects of 5 shapes",
  "float attribute, after another rasterizer drew from a vec3 attribute",
] as const;

/** Objects a case makes before its draw, kept alive for the whole of it. */
const keep: unknown[] = [];

if (isMainThread) {
  console.log(versions());
  for (const target of ["js", "wasm"]) {
    for (const name of cases) {
      const worker = new Worker(new URL(import.meta.url), { workerData: { name, target } });
      const result = await new Promise((resolve) => {
        worker.once("message", resolve);
        worker.once("error", (error) => resolve(String(error)));
      });
      console.log(target.padEnd(5), name.padEnd(40), JSON.stringify(result));
    }
  }
} else {
  const rmsl = await import("../../dist/rmsl.js");
  const { compileJS } = await import("../../dist/js.js");
  const { compileWasm } = await import("../../dist/wasm.js");
  const { attribute, varying, uniform, Fn, vec4, builtinPosition, builtinFragDepth, outputStruct, int, uint } =
    rmsl as any;
  const { name, target } = workerData as { name: (typeof cases)[number]; target: "js" | "wasm" };
  const positions = Float64Array.of(-1, -1, 0, 1, -1, 0, -1, 1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0);
  const fractions = Float64Array.of(0.1, 0.3, 0.7, 0.2, 0.9, 0.45);
  const wholes = Float64Array.of(0, 1, 2, 1, 0, 2);
  const position = attribute("vec3");
  const attributes: Record<string, ArrayLike<number>> = { [position.name]: positions };
  const uniforms: Record<string, unknown> = {};
  const textures: Record<string, unknown> = {};
  /** Draws a case makes with the measured rasterizer itself before it measures, given what `ctx` holds then. */
  let warm: ((draw: (ctx: any) => void) => void) | undefined;
  /** Texel data and vector values in five kinds of array, the first of them the one a case measures with. */
  const texels = [0.1, 0.5, 0.9, 1, 0.2, 0.4, 0.6, 1, 0.3, 0.7, 0.8, 1, 0.15, 0.25, 0.35, 1];
  const texelKinds = [
    () => Float32Array.from(texels),
    () => texels.slice(),
    () => Uint8Array.from(texels, (v) => v * 255),
    () => Uint8ClampedArray.from(texels, (v) => v * 255),
    () => Uint16Array.from(texels, (v) => v * 255),
  ];
  const vectorKinds = [
    (n: number) => Float64Array.from({ length: n }, (_, i) => 0.1 * (i + 1)),
    (n: number) => Float32Array.from({ length: n }, (_, i) => 0.1 * (i + 1)),
    (n: number) => Array.from({ length: n }, (_, i) => 0.1 * (i + 1)),
    (n: number) => Int32Array.from({ length: n }, (_, i) => i + 1),
    (n: number) => Uint8Array.from({ length: n }, (_, i) => i + 1),
  ];
  let vertex: () => void;
  let fragment: () => unknown;
  let float32 = false;
  const place = () => builtinPosition().assign(vec4(position, 1));
  /**
   * Draws 2000 times with another rasterizer from the attribute `values` of `type`. Its vertex stage
   * writes the varying `write` gives, if any, and its fragment stage reads that varying's first component.
   */
  const drawOther = (type: string, values: ArrayLike<number>, write?: (p: any) => any, float?: "f32") => {
    const p = attribute(type);
    let written: any;
    const other = compileJS(
      () =>
        Fn(() => {
          written = write?.(p);
          builtinPosition().assign(vec4(type === "vec3" ? p : position, 1));
        })(),
      () => Fn(() => (written ? vec4(written.x.toFloat(), 0, 0, 1) : vec4(1)))(),
      { float: float ?? "f64" },
    );
    const ctx = { attributes: { [p.name]: values, [position.name]: positions } };
    for (let k = 0; k < 2000; k++) other.draw(ctx as any, { width: 4, height: 4 });
  };
  const someColours = () => Float64Array.from({ length: 18 }, (_, i) => 0.05 * i + 0.01);
  /** Warms with each texel kind after the first given to the sampler `t` by `give`, then gives the first back. */
  const warmTexels = (give: (kind: () => ArrayLike<number>) => void) => (draw: (ctx: any) => void) => {
    for (const kind of texelKinds.slice(1)) {
      give(kind);
      for (let k = 0; k < 2000; k++) draw({ attributes, uniforms, textures });
    }
    give(texelKinds[0]!);
  };
  switch (name) {
    case "vec3 uniform in the vertex stage, after draws of it given in 4 other kinds of array": {
      const u = uniform("vec3");
      const v = varying("vec3");
      const give = (kind: (n: number) => ArrayLike<number>) => (uniforms[u.name] = kind(3));
      warm = (draw) => {
        for (const kind of vectorKinds.slice(1)) {
          give(kind);
          for (let k = 0; k < 2000; k++) draw({ attributes, uniforms, textures });
        }
        give(vectorKinds[0]!);
      };
      give(vectorKinds[0]!);
      vertex = () => {
        v.assign(u.mul(0.37));
        place();
      };
      fragment = () => vec4(v, 1);
      break;
    }
    case "sampler in the vertex stage, after draws of it with texel data of 4 other kinds":
    case "linear-filtered sampler, after draws of it with texel data of 4 other kinds":
    case "textureLoad, after draws of it with texel data of 4 other kinds": {
      const { texture, textureLoad, fragCoord, ivec2, int } = rmsl as any;
      const t = uniform("sampler2D");
      const linear = name.startsWith("linear");
      const give = (kind: () => ArrayLike<number>) =>
        (textures[t.name] = { data: kind(), width: 2, height: 2, ...(linear ? { magFilter: "linear" } : {}) });
      give(texelKinds[0]!);
      warm = warmTexels(give);
      if (name.includes("vertex stage")) {
        const v = varying("vec4");
        vertex = () => {
          v.assign(texture(t, position.xy.mul(0.5).add(0.5)));
          place();
        };
        fragment = () => v;
      } else {
        vertex = place;
        fragment = () =>
          name.startsWith("textureLoad")
            ? textureLoad(t, ivec2(int(0), int(1)))
            : texture(t, fragCoord().xy.mul(0.0625));
      }
      break;
    }
    case "float uniform, after draws given uniforms objects of 5 shapes": {
      // Each uniforms object holds the uniform, after a key of its own in all but the first: five shapes.
      const f = uniform("float");
      const shape = (i: number) => {
        const o: Record<string, unknown> = {};
        if (i > 0) o["extra" + i] = 1;
        o[f.name] = 0.37;
        return o;
      };
      warm = (draw) => {
        for (let i = 1; i < 5; i++) {
          const ctx = { attributes, uniforms: shape(i), textures };
          for (let k = 0; k < 2000; k++) draw(ctx);
        }
      };
      Object.assign(uniforms, shape(0));
      vertex = place;
      fragment = () => vec4(f, 0, 0, 1);
      break;
    }
    case "one sampler, after draws of it with texel data of 4 other kinds":
    case "five samplers with texel data of five kinds, in one draw":
    case "five samplers with texel data of one kind, in one draw": {
      const { texture, fragCoord } = rmsl as any;
      const count = name.startsWith("one") ? 1 : 5;
      const samplers = Array.from({ length: count }, () => uniform("sampler2D"));
      const give = (kinds: (() => ArrayLike<number>)[]) =>
        samplers.forEach((t: any, i: number) => (textures[t.name] = { data: kinds[i]!(), width: 2, height: 2 }));
      give(name.includes("five kinds") ? texelKinds : Array(count).fill(texelKinds[0]));
      if (count === 1)
        warm = (draw) => {
          for (const kind of texelKinds.slice(1)) {
            give([kind]);
            for (let k = 0; k < 2000; k++) draw({ attributes, uniforms, textures });
          }
          give([texelKinds[0]!]);
        };
      vertex = place;
      fragment = () => samplers.reduce((sum: any, t: any) => sum.add(texture(t, fragCoord().xy.mul(0.0625))), vec4(0));
      break;
    }
    case "vec3 uniform, after draws of it given in 4 other kinds of array":
    case "mat3 uniform, after draws of it given in 4 other kinds of array":
    case "vec4 uniform array element, after draws of it given in 4 other kinds of array": {
      const { uniformArray, int } = rmsl as any;
      const element = name.startsWith("vec4 uniform array");
      const type = element ? "vec4" : name.startsWith("mat3") ? "mat3" : "vec3";
      const n = type === "mat3" ? 9 : type === "vec4" ? 4 : 3;
      const u = element ? uniformArray("vec4", 2) : uniform(type);
      const give = (kind: (n: number) => ArrayLike<number>) =>
        (uniforms[u.name] = element ? [kind(n), kind(n)] : kind(n));
      warm = (draw) => {
        for (const kind of vectorKinds.slice(1)) {
          give(kind);
          for (let k = 0; k < 2000; k++) draw({ attributes, uniforms, textures });
        }
        give(vectorKinds[0]!);
      };
      give(vectorKinds[0]!);
      vertex = place;
      fragment = () =>
        element
          ? u.element(int(1)).mul(0.5)
          : type === "mat3"
            ? vec4(u.mul((rmsl as any).vec3(0.5, 0.25, 1)), 1)
            : vec4(u.mul(0.37), 1);
      break;
    }
    case "float attribute, after another rasterizer drew from a vec3 attribute": {
      drawOther(
        "vec3",
        Float64Array.from({ length: 18 }, (_, i) => 0.05 * i - 0.4),
      );
      const c = attribute("float");
      const v = varying("vec3");
      attributes[c.name] = fractions;
      vertex = () => {
        v.assign(vec4(c.mul(0.5), 0, 0, 0).xyz);
        place();
      };
      fragment = () => vec4(v, 1);
      break;
    }
    case "float attribute, after draws from 4 kinds of host array": {
      // Today this shows nothing: the store of the attribute into a property already allocates 16 bytes a
      // vertex, as the "float attribute, vec3 varying" control does, and a boxed load adds no box of its own.
      for (const make of [Float32Array, Int32Array, Uint32Array, Int16Array])
        drawOther("float", (make as any).from([1, 2, 3, 4, 5, 6]));
      const c = attribute("float");
      const v = varying("vec3");
      attributes[c.name] = fractions;
      vertex = () => {
        v.assign(vec4(c.mul(0.5), 0, 0, 0).xyz);
        place();
      };
      fragment = () => vec4(v, 1);
      break;
    }
    case "vec3 varying, after draws with varyings in slots of 4 kinds": {
      // A vertex stage hands back an ivec3 in an Int32Array, a uvec3 in a Uint32Array, and a vec3 at f32 in a Float32Array.
      const { ivec3, uvec3, varying: v2 } = rmsl as any;
      const writes = (type: string, value: (p: any) => any) => (p: any) => {
        const v = v2(type);
        v.assign(value(p));
        return v;
      };
      drawOther(
        "vec3",
        positions,
        writes("ivec3", () => ivec3(1, 2, 3)),
      );
      drawOther(
        "vec3",
        positions,
        writes("uvec3", () => uvec3(1, 2, 3)),
      );
      drawOther(
        "vec3",
        positions,
        writes("vec3", (p) => p.mul(0.5)),
        "f32",
      );
      const c = attribute("vec3");
      const v = varying("vec3");
      attributes[c.name] = someColours();
      vertex = () => {
        v.assign(c.mul(0.5));
        place();
      };
      fragment = () => vec4(v, 1);
      break;
    }
    case "vec3 varying, after a host object made alike with the fragment context held undefined in fragDepth": {
      // Made as the rasterizer makes its contexts: a literal, then uniforms and textures, then the per-call field.
      const o: any = { varyings: {}, fragCoord: null };
      o.uniforms = {};
      o.textures = undefined;
      o.fragDepth = undefined;
      keep.push(o);
      const c = attribute("vec3");
      const v = varying("vec3");
      attributes[c.name] = someColours();
      // A depth that is not a whole number at any vertex, so every fragment's interpolated depth is a fraction.
      attributes[position.name] = Float64Array.of(
        -1,
        -1,
        0.13,
        1,
        -1,
        0.37,
        -1,
        1,
        0.61,
        1,
        -1,
        0.37,
        1,
        1,
        0.29,
        -1,
        1,
        0.61,
      );
      vertex = () => {
        v.assign(c.mul(0.5));
        place();
      };
      fragment = () => vec4(v, 1);
      break;
    }
    case "vec3 varying": {
      const c = attribute("vec3");
      const v = varying("vec3");
      attributes[c.name] = Float64Array.of(
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.15,
        0.25,
        0.35,
        0.45,
        0.55,
        0.65,
        0.75,
        0.85,
        0.95,
      );
      vertex = () => {
        v.assign(c.mul(0.5));
        place();
      };
      fragment = () => vec4(v, 1);
      break;
    }
    case "vec3 varying, after draws from 2 kinds of host array":
    case "vec3 varying, after draws from 4 kinds of host array":
    case "vec3 varying, after draws from 6 kinds of host array": {
      // Other rasterizers draw first, each from an attribute array of another kind, then the measured one.
      const kinds = Number(/from (\d) kinds/.exec(name)![1]);
      const arrays = [Float32Array, Int32Array, Uint32Array, Int16Array, Uint8Array, Array].slice(0, kinds);
      for (const make of arrays) {
        const p = attribute("vec3");
        const other = compileJS(
          () =>
            Fn(() => {
              builtinPosition().assign(vec4(p, 1));
            })(),
          () => Fn(() => vec4(1))(),
        );
        const values = (make as any).from([-1, -1, 0, 1, -1, 0, -1, 1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]);
        for (let k = 0; k < 2000; k++) other.draw({ attributes: { [p.name]: values } } as any, { width: 4, height: 4 });
      }
      const c = attribute("vec3");
      const v = varying("vec3");
      attributes[c.name] = Float64Array.of(
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.15,
        0.25,
        0.35,
        0.45,
        0.55,
        0.65,
        0.75,
        0.85,
        0.95,
      );
      vertex = () => {
        v.assign(c.mul(0.5));
        place();
      };
      fragment = () => vec4(v, 1);
      break;
    }
    case "vec3 varying, both triangles crossing the near plane": {
      // Each triangle has one vertex behind the near plane (w = -1), so the clip adds two vertices to each.
      const clip = attribute("vec4");
      attributes[clip.name] = Float64Array.of(
        -1,
        -1,
        0,
        1,
        1,
        -1,
        0,
        1,
        -1,
        1,
        0,
        -1,
        1,
        -1,
        0,
        1,
        1,
        1,
        0,
        1,
        -1,
        1,
        0,
        -1,
      );
      const c = attribute("vec3");
      const v = varying("vec3");
      attributes[c.name] = Float64Array.of(
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.15,
        0.25,
        0.35,
        0.45,
        0.55,
        0.65,
        0.75,
        0.85,
        0.95,
      );
      vertex = () => {
        v.assign(c.mul(0.5));
        builtinPosition().assign(clip);
      };
      fragment = () => vec4(v, 1);
      break;
    }
    case "int attribute":
    case "uint attribute":
    case "uint attribute of 2^31 or more": {
      const unsigned = name.startsWith("uint");
      const c = attribute(unsigned ? "uint" : "int");
      const v = varying("float");
      attributes[c.name] = name.includes("2^31")
        ? Uint32Array.of(2 ** 31, 2 ** 31 + 1, 2 ** 31 + 2, 2 ** 32 - 1, 2 ** 31 + 3, 2 ** 31 + 4)
        : unsigned
          ? Uint32Array.of(1, 2, 3, 4, 5, 6)
          : Int32Array.of(-1, 2, -3, 4, -5, 6);
      // The varying is 1 at every vertex, never -0, so only the attribute can allocate per vertex.
      vertex = () => {
        v.assign(c.toFloat().mul(0).add(1));
        place();
      };
      fragment = () => vec4(v, 0, 0, 1);
      break;
    }
    case "float varying":
    case "float varying, whole-number colours":
    case "float varying, float: f32": {
      const c = attribute("float");
      const v = varying("float");
      attributes[c.name] = name.includes("whole") ? wholes : fractions;
      float32 = name.includes("f32");
      vertex = () => {
        v.assign(c.mul(name.includes("whole") ? 1 : 0.5));
        place();
      };
      fragment = () => vec4(v, 0, 0, 1);
      break;
    }
    case "float attribute, vec3 varying": {
      const c = attribute("float");
      const v = varying("vec3");
      attributes[c.name] = fractions;
      vertex = () => {
        v.assign(vec4(c.mul(0.5), 0, 0, 0).xyz);
        place();
      };
      fragment = () => vec4(v, 1);
      break;
    }
    case "int varying":
    case "uint varying above 2^31": {
      const v = varying(name.startsWith("int") ? "int" : "uint");
      vertex = () => {
        v.assign(name.startsWith("int") ? int(7) : uint(0xfffffff0));
        place();
      };
      fragment = () => vec4(v.toFloat().mul(0), 0, 0, 1);
      break;
    }
    case "bool varying": {
      const v = varying("bool");
      vertex = () => {
        v.assign(position.x.greaterThan(0));
        place();
      };
      fragment = () => vec4(v.select(1, 0), 0, 0, 1);
      break;
    }
    case "float uniform read in the fragment": {
      const u = uniform("float");
      uniforms[u.name] = 0.37;
      vertex = place;
      fragment = () => vec4(u, 0, 0, 1);
      break;
    }
    case "fragment writes its depth": {
      vertex = place;
      fragment = () => {
        builtinFragDepth().assign(fragCoordX());
        return vec4(1);
      };
      break;
    }
    case "float outputStruct member": {
      vertex = place;
      fragment = () => outputStruct(vec4(1), fragCoordX());
      break;
    }
  }
  function fragCoordX() {
    return (rmsl as any).fragCoord().x.mul(0.013);
  }
  const raster = (target === "js" ? compileJS : compileWasm)(
    () =>
      Fn(() => {
        vertex();
      })(),
    () => Fn(() => fragment())(),
    { float: float32 ? "f32" : "f64" },
  );
  const ctx = { attributes, uniforms, textures };
  warm?.((c) => raster.draw(c, { width: 16, height: 16 }));
  const options = { width: 16, height: 16, clear: true, clearDepth: true };
  // Two triangles covering a 16 by 16 target: 6 vertices and 256 fragments a draw.
  const result = await allocatedBy(() => raster.draw(ctx as any, options), 2000, 500);
  // Only the library's sites: the harness's own few bytes a call are left out.
  const sites = Object.fromEntries(
    // A stage's own code has no URL; its sites show by the names of its functions, such as frag or _tex2d.
    Object.entries(result.sites).filter(([site, bytes]) => !/measure\.ts/.test(site) && bytes >= 4),
  );
  parentPort!.postMessage({ perDraw: Math.round(result.perCall), sites });
}
