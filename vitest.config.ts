import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
    typecheck: {
      // Type-level tests catch mismatches between signature return types and
      // actual node types.
      include: ["src/**/*.test-d.ts"],
    },
  },
});
