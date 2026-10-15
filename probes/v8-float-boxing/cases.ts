/**
 * Runs each case of a probe in a worker of its own and prints the bytes one
 * operation allocates. A case makes a function that does its operation `N`
 * times a call.
 */
import { Worker, isMainThread, workerData, parentPort } from "node:worker_threads";
import { allocatedBy, versions } from "./measure.ts";

export const N = 256;

/** A float source whose values are fractions: each product with 1.5 is not a whole number. */
export const fractions = Float64Array.of(0.1, 0.3, 0.7, 0.2, 0.9, 0.45, 0.33, 0.61);

export async function runCases(url: string, cases: Record<string, () => () => void>): Promise<void> {
  if (isMainThread) {
    console.log(versions());
    console.log("bytes per operation:");
    for (const name of Object.keys(cases)) {
      const worker = new Worker(new URL(url), { workerData: name });
      const perCall = await new Promise<number>((resolve, reject) => {
        worker.once("message", resolve);
        worker.once("error", reject);
      });
      console.log(name.padEnd(72), (perCall / N).toFixed(2).padStart(6));
    }
  } else {
    const run = cases[workerData as string]!();
    const { perCall } = await allocatedBy(run, 20000, 2000);
    parentPort!.postMessage(perCall);
  }
}
