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
      external: ["esbuild", "vite", "wabt", "fs/promises"],
    },
  },
  plugins: [
    compileWat(),
    dts({
      include: [
        "src/rmsl.ts",
        "src/wgsl.ts",
        "src/glsl.ts",
        "src/js.ts",
        "src/wasm.ts",
        "src/core.ts",
        "src/backends/shared.ts",
        "src/backends/glsl/glsl.ts",
        "src/backends/wgsl/wgsl.ts",
        "src/backends/js/js.ts",
        "src/backends/wasm/wasm.ts",
        "src/backends/cpu.ts",
        "src/backends/adapter.ts",
        "src/backends/adapter-cpu.ts",
        "src/backends/js/adapter-js.ts",
        "src/backends/wasm/adapter-wasm.ts",
        "src/backends/glsl/adapter-glsl.ts",
        "src/backends/wgsl/adapter-wgsl.ts",
        "src/backends/js/rasterizer.ts",
        "src/backends/wasm/rasterizer.ts",
        "src/vite/vite.ts",
        "src/vite/wat.d.ts",
        "src/effects/index.ts",
        "src/effects/*.ts",
        "src/scene/index.ts",
        "src/scene/**/*.ts",
        "src/test/index.ts",
      ],
      exclude: ["src/**/*.test.ts"],
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
