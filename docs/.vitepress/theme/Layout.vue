<script setup>
import DefaultTheme from 'vitepress/theme'
import ReadingProgress from './components/ReadingProgress.vue'
</script>

<template>
  <DefaultTheme.Layout />
  <ClientOnly>
    <ReadingProgress />
  </ClientOnly>
</template>
