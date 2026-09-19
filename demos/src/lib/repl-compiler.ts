import { createHTMLExtension, createJSExtension, defaultTransformModulePaths, type Extension } from '@bigmistqke/repl'
import type ts from 'typescript'

/**
 * In-browser TS compile pipeline for live code examples, adapted from
 * solid-three's site/src/components/demo.tsx. rmsl examples are plain TS
 * (no JSX), so this only strips TypeScript syntax and rewrites module
 * specifiers — no JSX/Babel transform. Framework-agnostic: depends only on
 * @bigmistqke/repl's file-url-system primitives and TypeScript loaded at
 * runtime.
 */

let tsPromise: Promise<typeof ts> | undefined
export function loadTypeScript(): Promise<typeof ts> {
  if (!tsPromise) {
    const url = 'https://esm.sh/typescript@5.9'
    tsPromise = import(/* @vite-ignore */ url).then((mod) => (mod.default ?? mod) as typeof ts)
  }
  return tsPromise
}

export interface Compiler {
  tsModule: typeof ts
}

export async function loadCompiler(): Promise<Compiler> {
  return { tsModule: await loadTypeScript() }
}

export function errorModule(message: string): string {
  const escaped = JSON.stringify(message)
  return `const node = document.createElement("pre")
node.style.cssText = "color:#ff8080;background:#0a0c12;font-family:ui-monospace,monospace;font-size:0.85rem;padding:1rem;margin:0;height:100%;white-space:pre-wrap;overflow:auto;"
node.textContent = ${escaped}
document.body.appendChild(node)
`
}

export interface ExtensionOptions {
  getCompiler: () => Compiler | undefined
  resolveBareSpecifier: (specifier: string) => string
  readFile(path: string): string | undefined
}

/**
 * A TS extension for `createFileUrlSystem`, deferring to `getCompiler()` so
 * TypeScript can be lazy-loaded on first edit. The actual transpile +
 * module-specifier rewrite is `@bigmistqke/repl`'s own `createJSExtension`
 * — this only adds the lazy-compiler check and a friendly error module in
 * place of a thrown compile error.
 */
export function createTsExtension(options: ExtensionOptions): Extension {
  return {
    type: 'javascript',
    transform: (config) => {
      const compiler = options.getCompiler()
      if (!compiler) return ''
      try {
        return createJSExtension({
          ts: compiler.tsModule,
          transpile: true,
          compilerOptions: { target: compiler.tsModule.ScriptTarget.ESNext, module: compiler.tsModule.ModuleKind.ESNext },
          readFile: options.readFile,
          resolveBareSpecifier: options.resolveBareSpecifier,
        }).transform(config)
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        return errorModule('Compile error:\n\n' + message)
      }
    },
  }
}

/**
 * Rewrites module specifiers inside the generated `/index.html`'s
 * `<script src>` (e.g. `./main.ts`) to their real fileUrls blob URLs, using
 * the same TypeScript loaded for the TS extension.
 */
export function createHtmlExtension(options: ExtensionOptions) {
  return createHTMLExtension({
    transformModule: (config) => {
      return () => {
        const compiler = options.getCompiler()
        if (!compiler) return config.source
        // defaultTransformModulePaths returns an accessor, but this
        // function's own caller (transformHtml) only unwraps one layer
        // (`transformModule(config)()`) — invoke it here, not return it.
        return defaultTransformModulePaths({
          ...config,
          ts: compiler.tsModule,
          readFile: options.readFile,
          resolveBareSpecifier: options.resolveBareSpecifier,
        })()
      }
    },
  })
}
