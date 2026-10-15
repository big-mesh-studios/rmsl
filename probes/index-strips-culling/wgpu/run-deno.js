import { runProbe } from "./probe.js";
const res = await runProbe(navigator.gpu);
res.implementation = { name: "wgpu via Deno", deno: Deno.version, os: Deno.build };
console.log(JSON.stringify(res, null, 2));
