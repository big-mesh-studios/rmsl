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
] as const;

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
  let vertex: () => void;
  let fragment: () => unknown;
  let float32 = false;
  const place = () => builtinPosition().assign(vec4(position, 1));
  switch (name) {
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
  const ctx = { attributes, uniforms };
  const options = { width: 16, height: 16, clear: true, clearDepth: true };
  // Two triangles covering a 16 by 16 target: 6 vertices and 256 fragments a draw.
  const result = await allocatedBy(() => raster.draw(ctx as any, options), 2000, 500);
  // Only the library's sites: the harness's own few bytes a call are left out.
  const sites = Object.fromEntries(
    Object.entries(result.sites).filter(([site, bytes]) => !/measure\.ts|no url/.test(site) && bytes >= 1),
  );
  parentPort!.postMessage({ perDraw: Math.round(result.perCall), sites });
}
