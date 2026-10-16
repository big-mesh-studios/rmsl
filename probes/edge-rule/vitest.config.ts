import { defineConfig } from "vitest/config";

/** Runs the probes of how a GPU decides a pixel centre on an edge: `pnpm vitest run --config probes/edge-rule/vitest.config.ts`. */
export default defineConfig({
  root: new URL("../..", import.meta.url).pathname,
  test: {
    environment: "node",
    include: ["probes/edge-rule/**/*.probe.ts"],
    testTimeout: 1_200_000,
  },
});
