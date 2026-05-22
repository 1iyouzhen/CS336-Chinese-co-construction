<script setup>
import { onMounted, onUnmounted, ref } from 'vue'

const progress = ref(0)
const visible = ref(false)
const showTopIcon = ref(false)
const radius = 24
const circumference = 2 * Math.PI * radius
let timer = 0

const updateProgress = () => {
  const scrollTop = window.scrollY || document.documentElement.scrollTop
  const maxScroll = document.documentElement.scrollHeight - window.innerHeight
  const nextProgress = maxScroll > 0 ? Math.round((scrollTop / maxScroll) * 100) : 0

  progress.value = Math.min(100, Math.max(0, nextProgress))
  visible.value = scrollTop > 80
  showTopIcon.value = false

  window.clearTimeout(timer)
  timer = window.setTimeout(() => {
    showTopIcon.value = visible.value
  }, 900)
}

const scrollToTop = () => {
  window.scrollTo({ top: 0, behavior: 'smooth' })
}

onMounted(() => {
  window.addEventListener('scroll', updateProgress, { passive: true })
  window.addEventListener('resize', updateProgress, { passive: true })
  updateProgress()
})

onUnmounted(() => {
  window.removeEventListener('scroll', updateProgress)
  window.removeEventListener('resize', updateProgress)
  window.clearTimeout(timer)
})
</script>

<template>
  <Transition name="reading-progress">
    <button
      v-if="visible"
      class="reading-progress-button"
      type="button"
      :aria-label="`阅读进度 ${progress}%，返回顶部`"
      :title="`阅读进度 ${progress}%`"
      @click="scrollToTop"
    >
      <svg class="reading-progress-ring" viewBox="0 0 56 56" aria-hidden="true">
        <circle class="reading-progress-bg" cx="28" cy="28" :r="radius" />
        <circle
          class="reading-progress-value"
          cx="28"
          cy="28"
          :r="radius"
          :style="{
            strokeDasharray: circumference,
            strokeDashoffset: circumference - (progress / 100) * circumference
          }"
        />
      </svg>
      <span v-if="showTopIcon" class="reading-progress-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" width="18" height="18">
          <path
            d="M12 5l-7 7 1.4 1.4 4.6-4.58V20h2V8.82l4.6 4.58L19 12l-7-7z"
            fill="currentColor"
          />
        </svg>
      </span>
      <span v-else class="reading-progress-text">{{ progress }}%</span>
    </button>
  </Transition>
</template>
