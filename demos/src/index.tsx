import { render } from '@solidjs/web'
import { createRouter, defineRoutes, hashHistory } from '@solidjs/router'
import { App } from './App'

const routes = defineRoutes([{ path: '/:id?', component: App }])

// Hash routing sidesteps GitHub Pages having no server to rewrite deep
// links back to index.html, so `base` here isn't load-bearing for that —
// it's set from the same VITE_BASE as vite.config.ts's `base` purely so a
// URL built through the router (paths/useNavigate) and one built through
// Vite's own asset handling agree on the app's root, rather than each
// assuming a different one.
const Router = createRouter({
  routes,
  base: import.meta.env.VITE_BASE ?? '/',
  history: hashHistory(),
})

const root = document.getElementById('root')
if (root) {
  render(() => <Router />, root)
}
