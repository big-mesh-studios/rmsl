/**
 * Times one frame of every benchmark program, for every label generated, at
 * every float width, in whatever engine runs it, and checks that the labels
 * give the same bits for each program and width.
 *
 * Usage: node scripts/bench-cpu/run.ts [label ...]   (or: bun ..., deno run -A ...)
 * Memory: node --expose-gc scripts/bench-cpu/run.ts --memory [label ...]   (or: bun ... --memory)
 * With no label, runs every reports/bench-cpu/*.json. scripts/bench-cpu/browsers.ts
 * runs the same code in Firefox and Chromium through `runBench`.
 */

/** A texture as JSON carries it: its data as numbers, and the typed array to rebuild it as. */
export type TextureJson = {
  type: "Uint8Array" | "Float32Array";
  data: number[] | ArrayLike<number>;
  width: number;
  height: number;
  [key: string]: unknown;
};

/** One program of a report: its compiled source at each float width, and what a run passes it. */
export type Program = {
  name: string;
  kind: "grid" | "fragment";
  width: number;
  height: number;
  sources: Record<string, string>;
  ctx: {
    uniforms: Record<string, unknown>;
    textures: Record<string, TextureJson>;
    varyings?: Record<string, number[]>;
  };
  /** The varyings a fragment reads its surface from, or null for a grid. */
  surface: { position?: string; normal?: string } | null;
};

/** Every program of one generated label. */
export type Report = { label: string; programs: Program[] };

/** Samples timed per program after warming up, each a batch of frames so a coarse timer still resolves; the median is reported. */
const SAMPLES = 9;
const BATCH = 4;
const WARMUP = 5;

/** Runs every program of every report, and returns a table as text lines. */
export function runBench(reports: Report[]): string[] {
  const lines: string[] = [];
  const labels = reports.map((r) => r.label);
  const names = reports[0].programs.map((p) => p.name);
  const widths = Object.keys(reports[0].programs[0].sources);
  const columns = labels.flatMap((l) => widths.map((w) => `${l} ${w}`));
  lines.push(["program".padEnd(12), ...columns.map((c) => c.padStart(16))].join(""));
  const mismatches: string[] = [];
  for (const name of names) {
    const cells: string[] = [];
    const checks: Record<string, [string, number][]> = {};
    for (const report of reports) {
      const program = report.programs.find((p) => p.name === name)!;
      for (const width of widths) {
        const frame = makeFrame(program, width);
        for (let i = 0; i < WARMUP; i++) frame();
        const times: number[] = [];
        let checksum = 0;
        for (let i = 0; i < SAMPLES; i++) {
          const t = performance.now();
          for (let b = 0; b < BATCH; b++) checksum = frame();
          times.push((performance.now() - t) / BATCH);
        }
        times.sort((a, b) => a - b);
        cells.push(`${times[SAMPLES >> 1].toFixed(2)} ms`.padStart(16));
        (checks[width] ??= []).push([report.label, checksum]);
      }
    }
    for (const [width, results] of Object.entries(checks)) {
      for (const [l, c] of results) {
        if (Number.isNaN(c)) mismatches.push(`${name} ${width}: ${l} gives NaN`);
        else if (!Object.is(c, results[0][1])) mismatches.push(`${name} ${width}: ${l} differs from ${results[0][0]}`);
      }
    }
    lines.push([name.padEnd(12), ...cells].join(""));
  }
  lines.push(
    mismatches.length ? `MISMATCH\n${mismatches.join("\n")}` : `every label gives the same bits at each width`,
  );
  return lines;
}

/** Compiled copies of a program held at once to measure what one keeps, and frames run to measure what one allocates. */
const COPIES = 200;
const FRAMES = 20;

/**
 * What every program of every report keeps and allocates, as text lines:
 * the bytes one compiled function holds on to, slots, constants and helpers
 * included, and the bytes a frame allocates. `collect` runs a full garbage
 * collection, which the engine must expose (`node --expose-gc`, or Bun).
 * `allocated` gives the bytes a call allocates, collected ones included, or
 * is absent where the engine cannot count them.
 */
export async function runMemory(
  reports: Report[],
  collect: () => void,
  used: () => number,
  allocated?: (run: () => void) => Promise<number>,
): Promise<string[]> {
  const lines: string[] = [];
  const widths = Object.keys(reports[0]!.programs[0]!.sources);
  const columns = reports.flatMap((r) => widths.flatMap((w) => [`${r.label} ${w} kept`, `${r.label} ${w} /frame`]));
  lines.push(["program".padEnd(12), ...columns.map((c) => c.padStart(18))].join(""));
  for (const name of reports[0]!.programs.map((p) => p.name)) {
    const cells: string[] = [];
    for (const report of reports) {
      const program = report.programs.find((p) => p.name === name)!;
      for (const width of widths) {
        const make = new Function(program.sources[width]!);
        collect();
        const before = used();
        const copies = Array.from({ length: COPIES }, () => make());
        collect();
        const kept = (used() - before) / COPIES;
        copies.length = 0;
        const frame = makeFrame(program, width);
        for (let i = 0; i < WARMUP; i++) frame();
        const perFrame = allocated
          ? (await allocated(() => {
              for (let i = 0; i < FRAMES; i++) frame();
            })) / FRAMES
          : undefined;
        cells.push(
          `${(kept / 1024).toFixed(1)} KB`.padStart(18),
          (perFrame === undefined ? "-" : `${(perFrame / 1024).toFixed(1)} KB`).padStart(18),
        );
      }
    }
    lines.push([name.padEnd(12), ...cells].join(""));
  }
  return lines;
}

/** One frame of `program` at `width`: the program run once per pixel. Returns a checksum of every value it gave. */
function makeFrame(program: Program, width: string): () => number {
  const fn = new Function(program.sources[width]!)() as (ctx: any) => any;
  const ctx: any = structuredClone(program.ctx);
  for (const tex of Object.values(ctx.textures) as TextureJson[]) {
    tex.data = new (globalThis as any)[tex.type](tex.data);
  }
  const fragCoord: number[] = [0, 0];
  ctx.fragCoord = fragCoord;
  const { width: w, height: h, surface } = program;
  const position = surface?.position ? ctx.varyings[surface.position] : null;
  const normal = surface?.normal ? ctx.varyings[surface.normal] : null;
  return () => {
    let checksum = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        fragCoord[0] = x + 0.5;
        fragCoord[1] = y + 0.5;
        if (normal) {
          // The pixel's point on a unit sphere facing the camera, as the varyings a rasterizer would give.
          const nx = ((x + 0.5) / w) * 2 - 1;
          const ny = 1 - ((y + 0.5) / h) * 2;
          const nz = Math.sqrt(Math.max(0, 1 - nx * nx - ny * ny));
          normal[0] = nx;
          normal[1] = ny;
          normal[2] = nz;
          if (position) {
            position[0] = nx;
            position[1] = ny;
            position[2] = nz;
          }
        }
        const result = fn(ctx);
        const value = result && typeof result === "object" && "value" in result ? result.value : result;
        for (let k = 0; k < value.length; k++) checksum = (checksum * 31 + value[k]) % 1e9;
      }
    }
    return checksum;
  };
}

const isMain =
  typeof process !== "undefined" && process.argv?.[1] && import.meta.url.endsWith(process.argv[1].split("/").pop()!);
if (isMain) {
  const { readFileSync, readdirSync } = await import("node:fs");
  const { join, dirname } = await import("node:path");
  const { fileURLToPath } = await import("node:url");
  const dir = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "reports", "bench-cpu");
  const memory = process.argv.includes("--memory");
  const asked = process.argv.slice(2).filter((a) => a !== "--memory");
  const files = asked.length
    ? asked.map((l) => `${l}.json`)
    : readdirSync(dir)
        .filter((f) => f.endsWith(".json"))
        .sort();
  const reports: Report[] = files.map((f) => JSON.parse(readFileSync(join(dir, f), "utf8")));
  const g = globalThis as any;
  const engine =
    typeof g.Bun !== "undefined"
      ? `bun ${g.Bun.version}`
      : typeof g.Deno !== "undefined"
        ? `deno ${g.Deno.version.deno}`
        : `node ${process.version}`;
  if (memory) {
    const collect: () => void =
      typeof g.Bun !== "undefined"
        ? () => g.Bun.gc(true)
        : typeof g.gc === "function"
          ? () => g.gc()
          : () => {
              throw new Error("--memory needs a garbage collection to call: run node with --expose-gc, or use bun");
            };
    const v8 = typeof g.Bun === "undefined" ? await import("node:v8") : null;
    const used = () =>
      v8
        ? v8.getHeapStatistics().used_heap_size + process.memoryUsage().arrayBuffers
        : process.memoryUsage().heapUsed + process.memoryUsage().arrayBuffers;
    // V8's sampling heap profiler counts what a call allocates, collected objects included.
    const allocated =
      typeof g.Bun === "undefined" && typeof g.Deno === "undefined"
        ? async (run: () => void): Promise<number> => {
            const { Session } = await import("node:inspector/promises");
            const session = new Session();
            session.connect();
            await session.post("HeapProfiler.enable");
            await session.post("HeapProfiler.startSampling", {
              samplingInterval: 4096,
              includeObjectsCollectedByMajorGC: true,
              includeObjectsCollectedByMinorGC: true,
            });
            run();
            const { profile } = await session.post("HeapProfiler.stopSampling");
            session.disconnect();
            let total = 0;
            const walk = (node: any): void => {
              total += node.selfSize;
              for (const child of node.children) walk(child);
            };
            walk(profile.head);
            return total;
          }
        : undefined;
    console.log(`== ${engine}, memory: what one compiled function keeps, and what one frame allocates`);
    for (const line of await runMemory(reports, collect, used, allocated)) console.log(line);
  } else {
    console.log(
      `== ${engine}, ${reports[0].programs[0].width}x${reports[0].programs[0].height} per frame, median of ${SAMPLES} batches of ${BATCH} frames`,
    );
    for (const line of runBench(reports)) console.log(line);
  }
}
