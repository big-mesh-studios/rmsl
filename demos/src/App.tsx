import { CodeMirror, darkTheme, LSPProvider } from "@big-mesh-studios/solid-codemirror";
import { Repl } from "@bigmistqke/repl/solid";
import { createEffect, createMemo, createSignal, For } from "solid-js";
import styles from "./App.module.css";
import { demos, type Demo } from "./demos/registry";
import { createHtmlExtension, createTsExtension, loadCompiler, type Compiler } from "./lib/repl-compiler";
import { injectSandboxRuntime, postThemeMessage } from "./lib/repl-sandbox";
import { rmslTypeFiles, rmslTypePaths } from "./lib/rmsl-types";

/**
 * The selected demo's editor pane: a tab per file in its VFS, and a
 * CodeMirror for whichever tab is active. Owns its own `activePath` so
 * switching demos (which remounts this) always starts back on the demo's
 * default file.
 */
function DemoEditor(props: { demo: Demo; onInput(path: string, source: string): void }) {
  const [activePath, setActivePath] = createSignal(props.demo.editablePath);
  return (
    <div class={styles.editor}>
      <div class={styles["file-tabs"]}>
        <For each={props.demo.paths}>
          {(path) => (
            <button
              type="button"
              onClick={() => setActivePath(path)}
              style={{
                padding: "4px 8px",
                "font-size": "12px",
                "font-family": "ui-monospace, monospace",
                border: "none",
                "border-radius": "4px",
                cursor: "pointer",
                background: activePath() === path ? "#22252c" : "transparent",
                color: activePath() === path ? "#fff" : "#9a9fa8",
              }}
            >
              {path}
            </button>
          )}
        </For>
      </div>
      <div class={styles.code}>
        <CodeMirror
          path={activePath()}
          theme={darkTheme}
          onInput={({ path: editedPath, source }) => props.onInput(editedPath, source)}
        />
      </div>
    </div>
  );
}

const RMSL_PACKAGE = "@random-mesh/rmsl";

/**
 * Points rmsl specifiers at the local dev server's own `node_modules`
 * (which vite serves as static files unchanged) instead of esm.sh, so
 * demos always run against the workspace's current build rather than
 * whatever is published to npm.
 */
function resolveBareSpecifier(specifier: string): string {
  if (specifier === RMSL_PACKAGE) {
    return `${window.location.origin}/node_modules/${RMSL_PACKAGE}/dist/rmsl.js`;
  }
  if (specifier.startsWith(`${RMSL_PACKAGE}/`)) {
    const subpath = specifier.slice(RMSL_PACKAGE.length + 1);
    return `${window.location.origin}/node_modules/${RMSL_PACKAGE}/dist/${subpath}.js`;
  }
  return `https://esm.sh/${specifier}`;
}

/**
 * Points a demo's `<script type="module" src="...">` at its real, Vite-built
 * chunk (worker/wasm/everything Vite's own pipeline already handles)
 * instead of the VFS-compiled one, for as long as its source is unedited —
 * skips the in-browser TS-compile pipeline entirely for the common
 * "just run it as authored" case. Only an actual edit falls back to
 * compiling that file live, which — unlike the real build — does not
 * understand `new Worker(new URL(...))` or `audioWorklet.addModule(...)`.
 */
function withRealEntry(html: string, moduleUrl: string): string {
  // The iframe runs from a blob: URL, which has no meaningful base for a
  // relative script src to resolve against — make it absolute first.
  const absoluteUrl = new URL(moduleUrl, window.location.origin).toString();
  return html.replace(/(<script[^>]*type=["']module["'][^>]*src=["'])[^"']*(["'])/, `$1${absoluteUrl}$2`);
}

export function App() {
  const [selectedId, setSelectedId] = createSignal(demos[0]?.id);
  const selectedDemo = createMemo(() => demos.find((demo) => demo.id === selectedId()));

  const [overridesByDemo, setOverridesByDemo] = createSignal<Record<string, Record<string, string>>>({});
  const overrides = createMemo(() => overridesByDemo()[selectedId() ?? ""] ?? {});
  const edited = createMemo(() => Object.keys(overrides()).length > 0);

  const mergedFiles = createMemo(() => {
    const demo = selectedDemo();
    return demo ? { ...demo.files, ...overrides() } : {};
  });

  function setOverride(path: string, source: string): void {
    const id = selectedId();
    if (!id) return;
    setOverridesByDemo((prev) => ({ ...prev, [id]: { ...prev[id], [path]: source } }));
  }

  // Loaded once, shared by every demo's ts/html extension. A plain signal,
  // not `createSignal(loadCompiler)` — that form treats a pending promise
  // the way <Loading> does (throws until it resolves), which is wrong here:
  // getCompiler() is read from plain closures with no <Loading> boundary to
  // catch it.
  const [compiler, setCompiler] = createSignal<Compiler>();
  void loadCompiler().then(setCompiler);
  const getCompiler = (): Compiler | undefined => compiler();

  const [theme] = createSignal<"dark" | "light">("dark");
  const [iframe, setIframe] = createSignal<HTMLIFrameElement>();

  createEffect(
    () => theme(),
    (value) => postThemeMessage(iframe(), value),
  );

  const readFile = (path: string): string | undefined => {
    const demo = selectedDemo();
    if (!demo) return undefined;
    const source = mergedFiles()[path];
    if (source === undefined) return undefined;
    if (path !== demo.entry) return source;
    const withRuntime = injectSandboxRuntime(source);
    return edited() ? withRuntime : withRealEntry(withRuntime, demo.moduleUrl);
  };

  const tsExtension = createTsExtension({ getCompiler, resolveBareSpecifier, readFile });
  const htmlExtension = createHtmlExtension({ getCompiler, resolveBareSpecifier, readFile });

  return (
    <div class={styles.root}>
      <nav class={styles["demo-tabs"]}>
        <For each={demos}>
          {(demo) => (
            <button
              onClick={() => setSelectedId(demo.id)}
              style={{ background: selectedId() === demo.id ? "#22252c" : "transparent" }}
              class={styles["demo-tab"]}
            >
              {demo.id}
            </button>
          )}
        </For>
      </nav>
      <div class={styles["editor-pane"]}>
        <LSPProvider
          files={{ ...mergedFiles(), ...rmslTypeFiles }}
          // The language worker's virtual filesystem can't do real package
          // resolution (reading @random-mesh/rmsl's package.json, following
          // its `exports` map) — only flat file lookups. `paths` hands it
          // each rmsl subpath's `.d.ts` file directly instead.
          tsconfig={{ baseUrl: "/", paths: rmslTypePaths }}
        >
          {/* A single-item <For>, not <Show keyed> — keyed Show does not
              remount across two different truthy values, only across a
              falsy<->truthy transition, so DemoEditor's own activePath
              state (which tab is open) never reset when switching demos.
              <For> is unambiguously keyed by array-item identity. */}
          <For each={selectedDemo() ? [selectedDemo()!] : []}>
            {(demo) => <DemoEditor demo={demo} onInput={setOverride} />}
          </For>
        </LSPProvider>
      </div>
      <Repl
        sandbox="allow-scripts allow-same-origin"
        entry={selectedDemo()?.entry ?? "/index.html"}
        extensions={{ ts: tsExtension, tsx: tsExtension, js: tsExtension, html: htmlExtension }}
        readFile={readFile}
        ref={({ element }) => setIframe(element)}
        class={styles.repl}
      />
    </div>
  );
}
