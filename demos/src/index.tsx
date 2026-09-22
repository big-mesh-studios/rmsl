import { render } from '@solidjs/web'
import { createRouter, defineRoutes, hashHistory } from '@solidjs/router'
import { App } from './App'

const routes = defineRoutes([{ path: '/:id?', component: App }])

const Router = createRouter({
  routes,
  history: hashHistory(),
})

const root = document.getElementById('root')
if (root) {
  render(() => <Router />, root)
}
