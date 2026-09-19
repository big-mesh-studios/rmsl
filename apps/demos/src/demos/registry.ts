const fileSources = import.meta.glob<string>('./*/**/*.{html,ts}', {
  query: '?raw',
  import: 'default',
  eager: true,
})
const sharedSources = import.meta.glob<string>('./shared/*.ts', {
  query: '?raw',
  import: 'default',
  eager: true,
})
// A few demos import `../../shared/shader` — give every demo that same
// path in its own VFS (`/shared/<name>.ts`) rather than special-casing which
// ones need it.
const sharedFiles = Object.fromEntries(
  Object.entries(sharedSources).map(([globPath, source]) => [`/shared/${globPath.slice('./shared/'.length)}`, source]),
)
const mainChunkUrls = import.meta.glob<string>('./*/src/main.ts', {
  query: '?importChunkUrl',
  import: 'default',
  eager: true,
})

export interface Demo {
  id: string
  /** Path -> source, e.g. `{ "/index.html": "...", "/src/main.ts": "..." }`. */
  files: Record<string, string>
  /** Every path in `files`, in the order its tab should appear. */
  paths: string[]
  entry: string
  editablePath: string
  /**
   * URL of the real, Vite-built module for this demo's entry — a full
   * production compile (worker chunks, wasm, everything Vite's own pipeline
   * already handles), used to run the demo as-authored without going
   * through the in-browser TS compiler at all. Only a live edit falls back
   * to that slower path.
   */
  moduleUrl: string
}

function splitPath(globPath: string): { id: string; path: string } {
  const match = globPath.match(/^\.\/([^/]+)\/(.+)$/)
  if (!match) throw new Error(`demos: unexpected path ${globPath}`)
  const [, id, rest] = match
  return { id, path: `/${rest}` }
}

const byId = new Map<string, { files: Record<string, string>; paths: string[] }>()
for (const globPath of Object.keys(fileSources).sort()) {
  const { id, path } = splitPath(globPath)
  if (id === 'shared') continue
  const entry = byId.get(id) ?? { files: {}, paths: [] }
  entry.files[path] = fileSources[globPath]
  entry.paths.push(path)
  byId.set(id, entry)
}

export const demos: Demo[] = [...byId.entries()]
  .sort(([a], [b]) => a.localeCompare(b))
  .map(([id, { files, paths }]) => {
    const moduleUrl = mainChunkUrls[`./${id}/src/main.ts`]
    if (moduleUrl === undefined) throw new Error(`demos: missing main.ts chunk for ${id}`)
    if (files['/index.html'] === undefined) throw new Error(`demos: missing index.html for ${id}`)
    return {
      id,
      files: { ...files, ...sharedFiles },
      paths: [...paths, ...Object.keys(sharedFiles)],
      entry: '/index.html',
      editablePath: '/src/main.ts',
      moduleUrl,
    }
  })
