const RMSL_PACKAGE = '@random-mesh/rmsl'
const RMSL_ROOT = '/node_modules/@random-mesh/rmsl'

// import.meta.glob's pattern has to be a literal string vite can statically
// analyze — it cannot be built from RMSL_ROOT/RMSL_PACKAGE above.
const dtsFiles = import.meta.glob<string>('/node_modules/@random-mesh/rmsl/dist/**/*.d.ts', {
  query: '?raw',
  import: 'default',
  eager: true,
})
const packageJsonSources = import.meta.glob<string>('/node_modules/@random-mesh/rmsl/package.json', {
  query: '?raw',
  import: 'default',
  eager: true,
})

/**
 * `@random-mesh/rmsl`'s own built declaration files, placed at the same
 * `/node_modules/...` paths a real install would use. These are this
 * workspace's own types, not a public package — feeding them to the
 * language worker directly skips ATA's CDN-download path entirely, which
 * has nowhere to fetch an unpublished package's types from anyway.
 */
export const rmslTypeFiles: Record<string, string> = { ...dtsFiles }

interface PackageExports {
  [subpath: string]: { types?: string } | undefined
}

/**
 * A `paths` entry per subpath rmsl's `package.json` `exports` declares,
 * pointing straight at that subpath's `.d.ts` file. `@typescript/vfs`'s
 * in-memory System doesn't implement real package resolution (walking
 * directories, reading `package.json`, following its `exports` map) — only
 * flat file lookups — so bare-specifier resolution has to be handed to it
 * pre-solved via `compilerOptions.paths` instead of relying on that.
 *
 * `LSPProvider` stores every file (including these) as a `file:///`-prefixed
 * URI (see its `files` effect), so the paths handed to the compiler have to
 * carry that same prefix — a bare `/node_modules/...` path matches nothing.
 */
export const rmslTypePaths: Record<string, string[]> = (() => {
  const source = Object.values(packageJsonSources)[0]
  if (!source) return {}
  const pkg = JSON.parse(source) as { exports?: PackageExports }
  const entries = Object.entries(pkg.exports ?? {}).flatMap(([subpath, target]) => {
    const types = target?.types
    if (!types) return []
    const specifier = subpath === '.' ? RMSL_PACKAGE : `${RMSL_PACKAGE}/${subpath.replace(/^\.\//, '')}`
    return [[specifier, [`file://${RMSL_ROOT}/${types.replace(/^\.\//, '')}`]]] as Array<[string, string[]]>
  })
  return Object.fromEntries(entries)
})()
