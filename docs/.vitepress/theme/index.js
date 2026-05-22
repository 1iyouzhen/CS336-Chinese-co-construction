import DefaultTheme from 'vitepress/theme'
import { nextTick, onMounted, watch } from 'vue'
import { useRoute } from 'vitepress'
import mediumZoom from 'medium-zoom'
import Layout from './Layout.vue'
import HomeLanding from './components/HomeLanding.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    app.component('HomeLanding', HomeLanding)
  },
  setup() {
    const route = useRoute()

    const initZoom = () => {
      mediumZoom('.vp-doc img:not(a img)', {
        background: 'var(--vp-c-bg)',
        margin: 24
      })
    }

    onMounted(() => {
      initZoom()
    })

    watch(
      () => route.path,
      () => nextTick(initZoom)
    )
  }
}
