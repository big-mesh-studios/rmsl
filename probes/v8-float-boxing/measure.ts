/**
 * The bytes a call of `run` allocates, by V8's sampling heap profiler, with
 * the place in the code each byte was allocated at. It runs in the thread
 * that calls it, so a probe calls it from a worker of its own: nothing an
 * earlier probe ran then shapes the objects or the feedback this one sees.
 */
import { Session } from "node:inspector/promises";

export interface Allocation {
  /** Bytes a call allocates, over every place. */
  perCall: number;
  /** Bytes a call allocates at each place, as `function url:line`. */
  sites: Record<string, number>;
}

export async function allocatedBy(run: () => unknown, warm = 20000, runs = 20000): Promise<Allocation> {
  // Connected before the warm-up: connecting deoptimizes the code it finds running.
  const session = new Session();
  session.connect();
  await session.post("HeapProfiler.enable");
  for (let k = 0; k < warm; k++) run();
  let best: Allocation | undefined;
  for (let round = 0; round < 2; round++) {
    await session.post("HeapProfiler.startSampling", {
      samplingInterval: 64,
      includeObjectsCollectedByMajorGC: true,
      includeObjectsCollectedByMinorGC: true,
    });
    for (let k = 0; k < runs; k++) run();
    const { profile } = await session.post("HeapProfiler.stopSampling");
    const sites: Record<string, number> = {};
    let total = 0;
    const walk = (node: any): void => {
      const { functionName, url, lineNumber } = node.callFrame;
      // The profiler's own frames, and Node's, are not the probe's.
      if (node.selfSize > 0 && !url.startsWith("node:")) {
        const key = `${functionName || "(anonymous)"} ${url.split("/").pop() || "(no url)"}:${lineNumber + 1}`;
        sites[key] = (sites[key] ?? 0) + node.selfSize / runs;
        total += node.selfSize / runs;
      }
      for (const child of node.children) walk(child);
    };
    walk(profile.head);
    if (!best || total < best.perCall) best = { perCall: total, sites };
  }
  session.disconnect();
  return best!;
}

/** The versions a result holds for. */
export function versions(): string {
  return `node ${process.versions.node}, v8 ${process.versions.v8}, ${process.platform} ${process.arch}`;
}
