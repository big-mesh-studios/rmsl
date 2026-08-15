import { defineConfig } from 'vite'
import { writeFileSync, readFileSync } from 'fs'
import dts from 'vite-plugin-dts'

export default defineConfig({
  build: {
    lib: {
      entry: {
        rmsl: 'src/rmsl.ts',
        vite: 'src/vite.ts',
        effects: 'src/effects/index.ts',
        scene: 'src/scene/index.ts',
      },
      formats: ['es'],
      fileName: (_format, entryName) => `${entryName}.js`,
    },
    rollupOptions: {
      external: ['esbuild', 'vite'],
    },
  },
  plugins: [
    dts({
      include: [
        'src/rmsl.ts',
        'src/vite.ts',
        'src/effects/index.ts',
        'src/effects/*.ts',
        'src/scene/index.ts',
        'src/scene/**/*.ts',
      ],
      exclude: ['src/**/*.test.ts'],
      outDir: 'dist',
      rollupTypes: true,
    }),
    {
      // tsc does not carry a `/// <reference types>` directive into emitted
      // declaration files, so consumers of the scene barrel would otherwise see
      // `GPUDevice` and friends as unknown. The WebGPURenderer's ambient GPU
      // types are re-required from the emitted declarations directly.
      name: 'rmsl-scene-webgpu-types',
      closeBundle() {
        const targets = [
          'dist/scene/index.d.ts',
          'dist/scene/renderers/WebGPURenderer.d.ts',
        ]
        for (const target of targets) {
          const file = readFileSync(target, 'utf8')
          if (file.startsWith('/// <reference')) continue
          writeFileSync(target, `/// <reference types="@webgpu/types" />\n${file}`)
        }
      },
    },
  ],
})
