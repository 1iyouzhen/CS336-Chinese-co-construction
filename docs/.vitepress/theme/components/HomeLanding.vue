<script setup>
import { withBase } from 'vitepress'

const chapters = [
  {
    stage: '基础入口',
    title: '分词器与训练原语',
    desc: '从 tokenizer、PyTorch 张量和资源核算开始，建立可运行的 LLM 工程底座。',
    links: [
      { text: '第2章 分词器', href: '/chapter2/chapter2_分词器' },
      { text: '第3章 PyTorch 与资源核算', href: '/chapter3/chapter3_pytorch与资源核算' }
    ]
  },
  {
    stage: '模型核心',
    title: '架构、MoE 与训练流程',
    desc: '围绕 Transformer、混合专家模型、后训练和强化学习，理解现代大模型的主干设计。',
    links: [
      { text: '第4章 模型架构', href: '/chapter4/chapter4_第四章语言模型架构和训练的技术细节' },
      { text: '第13章 训练流程', href: '/chapter13/chapter13_第十三章大模型的基本训练流程' }
    ]
  },
  {
    stage: '系统优化',
    title: 'GPU、分布式与推理',
    desc: '从单卡算子优化到多机训练，再到推理阶段的 KV Cache、吞吐和延迟优化。',
    links: [
      { text: '第6章 GPU 优化', href: '/chapter6/chapter6_第六章GPU和GPU相关的优化' },
      { text: '第10章 推理', href: '/chapter10/推理' }
    ]
  }
]

const metrics = [
  ['15', '系统章节'],
  ['6', '配套作业'],
  ['GPU', '工程优化'],
  ['RLVR', '后训练专题']
]

const assignments = [
  ['作业1', '手搓 tokenizer、model architecture 和 optimizer', 'assignment1-basics'],
  ['作业2', '系统优化、FlashAttention 和分布式训练', 'assignment2-systems'],
  ['作业3', 'Scaling Law 拟合与扩展规律实验', 'assignment3-scaling'],
  ['作业4', 'Common Crawl 数据处理与去重过滤', 'assignment4-data'],
  ['作业5', 'SFT 与强化学习对齐实验', 'assignment5-alignment'],
  ['作业6', '多维度模型评估与基准测试', 'assignment6-evaluation']
]
</script>

<template>
  <div class="home-landing">
    <section class="home-band home-intro">
      <div class="intro-copy">
        <p class="section-kicker">Learning Path</p>
        <h2>把大模型学习拆成能动手推进的工程路径</h2>
        <p>
          Diy-LLM 不是只读概念的资料库，而是把模型原理、系统优化、训练实践和评测方法组织成一条连续路线。
        </p>
      </div>
      <div class="metric-grid" aria-label="课程概览">
        <div v-for="[value, label] in metrics" :key="label" class="metric-item">
          <strong>{{ value }}</strong>
          <span>{{ label }}</span>
        </div>
      </div>
    </section>

    <section class="home-band">
      <div class="section-heading">
        <p class="section-kicker">Course Map</p>
        <h2>三段式学习主线</h2>
      </div>
      <div class="chapter-grid">
        <article v-for="chapter in chapters" :key="chapter.title" class="chapter-card">
          <span>{{ chapter.stage }}</span>
          <h3>{{ chapter.title }}</h3>
          <p>{{ chapter.desc }}</p>
          <div class="chapter-links">
            <a
              v-for="link in chapter.links"
              :key="link.href"
              :href="withBase(link.href)"
            >
              {{ link.text }}
            </a>
          </div>
        </article>
      </div>
    </section>

    <section class="home-band home-assignments">
      <div class="section-heading">
        <p class="section-kicker">Coursework</p>
        <h2>配套作业与代码实践</h2>
      </div>
      <div class="assignment-list">
        <a
          v-for="[name, desc, folder] in assignments"
          :key="folder"
          class="assignment-row"
          :href="`../coursework/${folder}/`"
        >
          <span>{{ name }}</span>
          <p>{{ desc }}</p>
        </a>
      </div>
    </section>
  </div>
</template>
