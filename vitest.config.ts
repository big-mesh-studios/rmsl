import { defineConfig } from "vitest/config";
import { compileWat } from "./src/vite/vite";

export default defineConfig({
  plugins: [compileWat()],
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
