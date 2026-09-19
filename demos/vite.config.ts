import { importChunkUrl } from '@lightningjs/vite-plugin-import-chunk-url'
import solid from '@solidjs/vite-plugin'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [solid({ ssr: false }), importChunkUrl()],
  optimizeDeps: {
    // solid-js 2.x's reactive core lives here; Vite's dependency scanner
    // doesn't discover it through solid-js's re-export chain on its own,
    // so without this two disconnected copies get pre-bundled and signal
    // writes never reach the rendered DOM.
    include: ['@solidjs/signals'],
  },
})
