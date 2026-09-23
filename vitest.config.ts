import { defineConfig } from "vitest/config";
import { compileWat } from "./src/vite/vite";

export default defineConfig({
  plugins: [compileWat()],
  test: {
    environment: "node",
    // Builds dist/ first when the sources changed since the last build:
    // some tests import the library through its package path, which resolves there.
    globalSetup: ["./scripts/ensure-build.mjs"],
    include: ["src/**/*.test.ts"],
    typecheck: {
      // Type-level tests catch mismatches between signature return types and
      // actual node types.
      include: ["src/**/*.test-d.ts"],
    },
  },
});
