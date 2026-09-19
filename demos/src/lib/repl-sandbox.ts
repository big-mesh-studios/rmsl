/**
 * Blob-URL iframe sandbox for live code examples, adapted from solid-three's
 * site/src/components/demo.tsx. Pure DOM/browser logic — no framework
 * dependency.
 *
 * Unlike the synthesized host document solid-three's demo.tsx builds, this
 * loads the app-demo's *own* `index.html` through the VFS (via
 * `createFileUrlSystem` + repl's HTML extension), so the sandbox is a
 * faithful copy of the real app shell — any markup an example's script
 * relies on (a `<canvas id>`, page styles, etc.) just works. We only inject
 * a theme-sync listener and an import map, rather than replacing the
 * document.
 */

export interface ImportMapEntry {
  specifier: string
  url: string
}

/**
 * Inserts the theme `postMessage` listener and an import map into an
 * existing HTML document, right before `</head>`. Everything else — the
 * body markup, the `<script src="...">` pointing at the example's real
 * entry file — is left untouched; the HTML extension's module-path rewriter
 * takes care of pointing that script at its compiled blob URL.
 */
export function injectSandboxRuntime(html: string, options: { importMap?: ImportMapEntry[] } = {}): string {
  const imports = Object.fromEntries((options.importMap ?? []).map((e) => [e.specifier, e.url]))
  const inject = `
    <script>
      window.addEventListener("message", function (event) {
        var data = event.data
        if (!data || data.type !== "theme") return
        document.documentElement.style.colorScheme = data.value
      })
    </script>
    <script type="importmap">${JSON.stringify({ imports })}</script>
`
  if (html.includes('</head>')) {
    return html.replace('</head>', `${inject}</head>`)
  }
  // No <head> in the source document — prepend a minimal one.
  return `<head>${inject}</head>${html}`
}

export function postThemeMessage(iframe: HTMLIFrameElement | undefined, theme: 'dark' | 'light'): void {
  iframe?.contentWindow?.postMessage({ type: 'theme', value: theme }, '*')
}
