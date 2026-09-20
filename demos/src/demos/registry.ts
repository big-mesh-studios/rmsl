// Not eager — a demo's source is only worth fetching once it's actually
// selected. `Object.keys()` of a non-eager glob is still available
// synchronously (only the *values*, the loader functions, are lazy), which
// is all the sidebar/tabs need up front.
const fileLoaders = import.meta.glob<string>('./*/**/*.{html,ts}', {
  query: '?raw',
  import: 'default',
})
const sharedLoaders = import.meta.glob<string>('./shared/*.ts', {
  query: '?raw',
  import: 'default',
})
// A few demos import `../../shared/shader` — give every demo that same
// path in its own VFS (`/shared/<name>.ts`) rather than special-casing which
// ones need it.
const sharedPaths = Object.fromEntries(
  Object.entries(sharedLoaders).map(([globPath, load]) => [`/shared/${globPath.slice('./shared/'.length)}`, load]),
)
const mainChunkUrls = import.meta.glob<string>('./*/src/main.ts', {
  query: '?importChunkUrl',
  import: 'default',
  eager: true,
})

export interface Demo {
  id: string
  /** Every path this demo's VFS has, in the order its tab should appear. */
  paths: string[]
  /** Fetches every file's source, keyed by path — only called once selected. */
  loadFiles(): Promise<Record<string, string>>
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

const byId = new Map<string, { loaders: Record<string, () => Promise<string>>; paths: string[] }>()
for (const globPath of Object.keys(fileLoaders).sort()) {
  const { id, path } = splitPath(globPath)
  if (id === 'shared') continue
  const entry = byId.get(id) ?? { loaders: {}, paths: [] }
  entry.loaders[path] = fileLoaders[globPath]!
  entry.paths.push(path)
  byId.set(id, entry)
}

export const demos: Demo[] = [...byId.entries()]
  .sort(([a], [b]) => a.localeCompare(b))
  .map(([id, { loaders, paths }]) => {
    const moduleUrl = mainChunkUrls[`./${id}/src/main.ts`]
    if (moduleUrl === undefined) throw new Error(`demos: missing main.ts chunk for ${id}`)
    if (loaders['/index.html'] === undefined) throw new Error(`demos: missing index.html for ${id}`)
    const allLoaders = { ...loaders, ...sharedPaths }
    return {
      id,
      paths: [...paths, ...Object.keys(sharedPaths)],
      async loadFiles() {
        const entries = await Promise.all(
          Object.entries(allLoaders).map(async ([path, load]) => [path, await load()] as const),
        )
        return Object.fromEntries(entries)
      },
      entry: '/index.html',
      editablePath: '/src/main.ts',
      moduleUrl,
    }
  })
