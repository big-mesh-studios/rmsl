import { defineConfig } from 'vite'
import dts from 'vite-plugin-dts'

export default defineConfig({
  build: {
    lib: {
      entry: {
        rmsl: 'src/rmsl.ts',
        vite: 'src/vite.ts',
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
      include: ['src/rmsl.ts', 'src/vite.ts'],
      outDir: 'dist',
      rollupTypes: true,
    }),
  ],
})
