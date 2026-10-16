import { defineConfig } from "vitest/config";

/** Runs the probes of how a GPU draws a triangle through vec4(0) or in one plane with the eye: `pnpm vitest run --config probes/vec4-zero/vitest.config.ts`. */
export default defineConfig({
  root: new URL("../..", import.meta.url).pathname,
  test: {
    environment: "node",
    include: ["probes/vec4-zero/**/*.probe.ts"],
    testTimeout: 600_000,
  },
});
