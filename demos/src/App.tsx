import { CodeMirror, darkTheme, LSPProvider } from "@big-mesh-studios/solid-codemirror";
import { Repl } from "@bigmistqke/repl/solid";
import { Split } from "@bigmistqke/solid-grid-split";
import { createMediaQuery } from "@solid-primitives/media";
import { createEffect, createMemo, createSignal, For, Show } from "solid-js";
import type { JSX } from "@solidjs/web";
import styles from "./App.module.css";
import { demos, type Demo } from "./demos/registry";
import { createHtmlExtension, createTsExtension, loadCompiler } from "./lib/repl-compiler";
import { injectSandboxRuntime, postThemeMessage } from "./lib/repl-sandbox";
import { loadRmslTypeFiles, rmslTypePaths } from "./lib/rmsl-types";
import importChunkUrlTypes from "./importChunkUrl.d.ts?raw";

/**
 * Every demo shares the same relative layout (`/index.html`, `/src/main.ts`,
 * ...), but each demo's LSP files are namespaced under `/<demo.id>/...` (see
 * `App`'s `lspFiles`) — otherwise switching from one demo to another whose
 * active tab happens to share the same relative path wouldn't change the
 * literal path string `<CodeMirror>` sees, and it only rebuilds its document
 * when that string changes (deliberately, so it doesn't fight the user's own
 * typing on every content update).
 */
function toLspPath(demoId: string, path: string): string {
  return `/${demoId}${path}`;
}

/**
 * The selected demo's editor pane: a tab per file in its VFS, and a
 * CodeMirror for whichever tab is active. `activePath` is a writable memo
 * derived from `props.demo.editablePath` — it tracks the current demo's
 * default file (so switching demos resets it, with no remount needed) but
 * a tab click can still overwrite it directly.
 *
 * `props.leading` (the hamburger button + popover, narrow layout only) sits
 * in the same row as the file tabs but outside their scroll container, so
 * it stays put while a long tab list scrolls under it.
 */
function DemoEditor(props: { demo: Demo; onInput(path: string, source: string): void; leading?: JSX.Element }) {
  const [activePath, setActivePath] = createSignal(() => props.demo.editablePath);
  return (
    <div class={styles.editor}>
      <div class={styles["file-tabs"]}>
        {props.leading}
        <div class={styles["file-tabs-scroll"]}>
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
      </div>
      <div class={styles.code}>
        <CodeMirror
          path={toLspPath(props.demo.id, activePath())}
          theme={darkTheme}
          onInput={({ source }) => props.onInput(activePath(), source)}
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

const NARROW_QUERY = "(max-width: 700px)";

export function App() {
  const isNarrow = createMediaQuery(NARROW_QUERY);
  const [selectedId, setSelectedId] = createSignal(demos[0]?.id);
  const selectedDemo = createMemo(() => demos.find((demo) => demo.id === selectedId()));

  const [overridesByDemo, setOverridesByDemo] = createSignal<Record<string, Record<string, string>>>({});
  const overrides = createMemo(() => overridesByDemo()[selectedId() ?? ""] ?? {});
  const edited = createMemo(() => Object.keys(overrides()).length > 0);

  // A demo's files are only fetched once it's actually selected. A pending
  // read throws, the same as `compiler` below — propagated to whichever
  // computation reads it (createFileUrlSystem's own memos, LSPProvider's
  // own `createMemo(() => props.files)`), which is what defers creating a
  // document until its content actually exists, with no manual loading
  // state or remount needed.
  const demoFiles = createMemo(async () => {
    const demo = selectedDemo();
    return demo ? await demo.loadFiles() : {};
  });

  const mergedFiles = createMemo(() => {
    const demo = selectedDemo();
    return demo ? { ...demoFiles(), ...overrides() } : {};
  });

  // rmsl's own declaration files, fetched once and shared by every demo's
  // LSPProvider — see loadRmslTypeFiles' doc comment for why they're lazy.
  const rmslTypeFiles = createMemo(loadRmslTypeFiles);

  // See toLspPath's doc comment for why this namespacing exists.
  const lspFiles = createMemo(() => {
    const demo = selectedDemo();
    const shared = { ...rmslTypeFiles(), "/importChunkUrl.d.ts": importChunkUrlTypes };
    if (!demo) return shared;
    const namespaced = Object.fromEntries(
      Object.entries(mergedFiles()).map(([path, source]) => [toLspPath(demo.id, path), source]),
    );
    return { ...namespaced, ...shared };
  });

  function setOverride(path: string, source: string): void {
    const id = selectedId();
    if (!id) return;
    setOverridesByDemo((prev) => ({ ...prev, [id]: { ...prev[id], [path]: source } }));
  }

  // Loaded once, shared by every demo's ts/html extension. A memo, since
  // nothing ever writes to it — reading it before the promise settles is a
  // pending-read, the same as any other async computation, handled by
  // whichever computation calls compiler() (createFileUrlSystem's own
  // memos, same as @bigmistqke/repl's own fileUrls.get() reads elsewhere),
  // not by us.
  const compiler = createMemo(loadCompiler);

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

  const tsExtension = createTsExtension({ getCompiler: compiler, resolveBareSpecifier, readFile });
  const htmlExtension = createHtmlExtension({ getCompiler: compiler, resolveBareSpecifier, readFile });

  const demoHamburger = () => (
    <>
      <button type="button" class={styles.hamburger} popovertarget="demo-popover" popovertargetaction="toggle">
        ☰
      </button>
      <div id="demo-popover" popover="auto" class={styles["demo-popover"]}>
        <For each={demos}>
          {(demo) => (
            <button
              type="button"
              onClick={() => setSelectedId(demo.id)}
              popovertarget="demo-popover"
              popovertargetaction="hide"
              style={{ background: selectedId() === demo.id ? "#22252c" : "transparent" }}
              class={styles["demo-tab"]}
            >
              {demo.id}
            </button>
          )}
        </For>
      </div>
    </>
  );

  return (
    <div class={isNarrow() ? `${styles.root} ${styles.narrow}` : styles.root}>
      <Show when={!isNarrow()}>
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
      </Show>
      <div class={styles["split-area"]}>
        <Split direction={isNarrow() ? "column" : "row"} style={{ display: "grid", width: "100%", height: "100%" }}>
          <Split.Pane size="1fr" class={styles["editor-pane"]}>
            <LSPProvider
              files={lspFiles()}
              // The language worker's virtual filesystem can't do real package
              // resolution (reading @random-mesh/rmsl's package.json, following
              // its `exports` map) — only flat file lookups. `paths` hands it
              // each rmsl subpath's `.d.ts` file directly instead.
              tsconfig={{ baseUrl: "/", paths: rmslTypePaths }}
            >
              {/* selectedDemo() is only undefined if demos itself is empty. */}
              <DemoEditor demo={selectedDemo()!} onInput={setOverride} leading={isNarrow() ? demoHamburger() : undefined} />
            </LSPProvider>
          </Split.Pane>
          <Split.Handle size="4px" class={styles.handle} />
          <Split.Pane size="1fr">
            <Repl
              sandbox="allow-scripts allow-same-origin"
              entry={selectedDemo()?.entry ?? "/index.html"}
              extensions={{ ts: tsExtension, tsx: tsExtension, js: tsExtension, html: htmlExtension }}
              readFile={readFile}
              ref={({ element }) => setIframe(element)}
              class={styles.repl}
            />
          </Split.Pane>
        </Split>
      </div>
    </div>
  );
}
