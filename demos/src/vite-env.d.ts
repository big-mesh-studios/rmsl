/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** GitHub Pages project-page prefix (e.g. `/rmsl/`); set by the deploy workflow, unset in dev. */
  readonly VITE_BASE?: string
}
