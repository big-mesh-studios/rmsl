/**
 * What the same float stores and reads allocate in each tier V8 may run code
 * in. Each store writes the product of a float it loads from a Float64Array;
 * the read multiplies a property by such a float. `--max-opt` caps the tier: 0 interprets only, 1 adds Sparkplug, 2 adds
 * Maglev, and the default adds TurboFan. Each tier runs in a process of its
 * own, and each case in a worker of that process.
 *
 * Usage: node probes/v8-float-boxing/tiers.ts
 */
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { N, fractions, runCases } from "./cases.ts";

const sink = new Float64Array(1);

const cases: Record<string, () => () => void> = {
  "store o.x, x made 0.5": () => {
    const o = { x: 0.5 };
    return () => {
      for (let i = 0; i < N; i++) o.x = fractions[i & 7]! * 1.5;
    };
  },
  "store f[0], f a Float64Array": () => {
    const f = new Float64Array(1);
    return () => {
      for (let i = 0; i < N; i++) f[0] = fractions[i & 7]! * 1.5;
    };
  },
  "store f[0] of small integers, f a Float64Array": () => {
    const f = new Float64Array(1);
    const ints = Float64Array.of(1, 2, 3, 4, 5, 6, 7, 8);
    return () => {
      for (let i = 0; i < N; i++) f[0] = ints[i & 7]! * 2;
    };
  },
  "read o.x into a Float64Array": () => {
    const objects = Array.from(fractions, (v) => ({ x: v * 1.5 }));
    return () => {
      for (let i = 0; i < N; i++) sink[0] = objects[i & 7]!.x * fractions[i & 7]!;
    };
  },
};

if (process.env.PROBE_TIER === undefined) {
  for (const tier of ["0", "1", "2", "default"]) {
    const flags = tier === "default" ? [] : [`--max-opt=${tier}`];
    console.log(`--max-opt=${tier}`);
    const child = spawnSync(process.execPath, [...flags, fileURLToPath(import.meta.url)], {
      env: { ...process.env, PROBE_TIER: tier },
      encoding: "utf8",
    });
    process.stdout.write(child.stdout + child.stderr);
  }
} else {
  await runCases(import.meta.url, cases);
}
