import { defineConfig } from "vite";
import { writeFileSync, readFileSync } from "fs";
import dts from "vite-plugin-dts";
import { compileWat } from "./src/vite/vite";

export default defineConfig({
  build: {
    lib: {
      entry: {
        rmsl: "src/rmsl.ts",
        wgsl: "src/wgsl.ts",
        glsl: "src/glsl.ts",
        js: "src/js.ts",
        wasm: "src/wasm.ts",
        vite: "src/vite/vite.ts",
        effects: "src/effects/index.ts",
        scene: "src/scene/index.ts",
        test: "src/test/index.ts",
      },
      formats: ["es"],
      fileName: (_format, entryName) => `${entryName}.js`,
    },
    rollupOptions: {
      external: ["esbuild", "vite", "wabt", "fs/promises", "path"],
    },
  },
  plugins: [
    compileWat(),
    dts({
      include: ["src"],
      exclude: [
        "src/**/*.test.ts",
        "src/**/*.test-d.ts",
        "src/**/*.bench.ts",
        "src/benches",
        "src/canon",
        "src/testing",
        "src/vite/fixtures",
      ],
      outDir: "dist",
      rollupTypes: true,
    }),
    {
      // tsc does not carry a `/// <reference types>` directive into emitted
      // declaration files, so consumers of the scene barrel would otherwise see
      // `GPUDevice` and friends as unknown. The WebGPURenderer's ambient GPU
      // types are re-required from the emitted declarations directly.
      name: "rmsl-scene-webgpu-types",
      closeBundle() {
        const targets = ["dist/scene/index.d.ts", "dist/scene/renderers/WebGPURenderer.d.ts"];
        for (const target of targets) {
          const file = readFileSync(target, "utf8");
          if (file.startsWith("/// <reference")) continue;
          writeFileSync(target, `/// <reference types="@webgpu/types" />\n${file}`);
        }
      },
    },
  ],
});
