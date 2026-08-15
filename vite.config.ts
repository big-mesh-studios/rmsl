import { defineConfig } from 'vite'
import dts from 'vite-plugin-dts'

export default defineConfig({
  build: {
    lib: {
      entry: {
        rmsl: 'src/rmsl.ts',
        vite: 'src/vite.ts',
        effects: 'src/effects/index.ts',
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
      include: ['src/rmsl.ts', 'src/vite.ts', 'src/effects/index.ts', 'src/effects/*.ts'],
      exclude: ['src/**/*.test.ts'],
      outDir: 'dist',
      rollupTypes: true,
    }),
  ],
})
