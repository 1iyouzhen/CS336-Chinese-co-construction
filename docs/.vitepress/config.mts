import { defineConfig } from 'vitepress'

const rootNav = [
  {
    text: 'Essentials',
    items: [
      { text: 'Preface', link: '/前言' },
      { text: 'Tooling', link: '/chapter1/wandb使用介绍' },
      { text: 'Tokenizer', link: '/chapter2/chapter2_分词器' },
      { text: 'PyTorch and Resource Accounting', link: '/chapter3/chapter3_pytorch与资源核算' },
      { text: 'Architecture and Training Details', link: '/chapter4/chapter4_第四章语言模型架构和训练的技术细节' },
      { text: 'MOE', link: '/chapter5/chapter5_混合专家模型' },
      { text: 'GPU Optimization', link: '/chapter6/chapter6_第六章GPU和GPU相关的优化' },
      { text: 'Data Engineering', link: '/chapter11/chapter11_数据工程' },
      { text: 'Training Pipeline', link: '/chapter13/chapter13_第十三章大模型的基本训练流程' },
      { text: 'Evaluation and Benchmarks', link: '/chapter12/chapter12_评估与基准测试' },
    ],
  },
  {
    text: 'Advanced',
    items: [
      { text: 'Distributed Training', link: '/chapter8/chapter8_第八章分布式训练' },
      { text: 'GPU High Performance Programming', link: '/chapter7/chapter7_第七章GPU高性能编程' },
      { text: 'Scaling Laws', link: '/chapter9/chapter9_Scaling_Laws' },
      { text: 'Inference', link: '/chapter10/推理' },
      { text: 'RLVR', link: '/chapter14/chapter14_可验证奖励的强化学习' },
      { text: 'Extended Content', link: '/chapter15/什么是LLM推理' },
    ],
  },
  {
    text: 'Extra Chapters',
    items: [{ text: 'Chapter 15', link: '/chapter15/什么是LLM推理' }],
  },
]

const rootSidebar = {
  '/': [
    {
      text: 'Chapters',
      items: [
        { text: 'Preface', link: '/前言' },
        { text: 'Chapter 1 Tooling', link: '/chapter1/wandb使用介绍' },
        { text: 'Chapter 2 Tokenizer', link: '/chapter2/chapter2_分词器' },
        { text: 'Chapter 3 PyTorch and Resource Accounting', link: '/chapter3/chapter3_pytorch与资源核算' },
        { text: 'Chapter 4 Language Model Architecture and Training Details', link: '/chapter4/chapter4_第四章语言模型架构和训练的技术细节' },
        { text: 'Chapter 5 Mixture of Experts', link: '/chapter5/chapter5_混合专家模型' },
        { text: 'Chapter 6 GPU and Related Optimization', link: '/chapter6/chapter6_第六章GPU和GPU相关的优化' },
        { text: 'Chapter 7 GPU High Performance Programming', link: '/chapter7/chapter7_第七章GPU高性能编程' },
        { text: 'Chapter 8 Distributed Training', link: '/chapter8/chapter8_第八章分布式训练' },
        { text: 'Chapter 9 Scaling Laws', link: '/chapter9/chapter9_Scaling_Laws' },
        { text: 'Chapter 10 Inference', link: '/chapter10/推理' },
        { text: 'Chapter 11 Data Engineering', link: '/chapter11/chapter11_数据工程' },
        { text: 'Chapter 12 Evaluation and Benchmarks', link: '/chapter12/chapter12_评估与基准测试' },
        { text: 'Chapter 13 Basic Training Pipeline for LLMs', link: '/chapter13/chapter13_第十三章大模型的基本训练流程' },
        { text: 'Chapter 14 Reinforcement Learning with Verifiable Rewards', link: '/chapter14/chapter14_可验证奖励的强化学习' },
        { text: 'Chapter 15 Extended Content', link: '/chapter15/什么是LLM推理' },
      ],
    },
  ],
}

const zhNav = [
  {
    text: '基础必学',
    items: [
      { text: '前言', link: '/zh/前言' },
      { text: '工具使用', link: '/zh/chapter1/wandb使用介绍' },
      { text: '分词器', link: '/zh/chapter2/chapter2_分词器' },
      { text: 'PyTorch与资源核算', link: '/zh/chapter3/chapter3_pytorch与资源核算' },
      { text: '架构与细节', link: '/zh/chapter4/chapter4_第四章语言模型架构和训练的技术细节' },
      { text: 'MOE', link: '/zh/chapter5/chapter5_混合专家模型' },
      { text: 'GPU优化', link: '/zh/chapter6/chapter6_第六章GPU和GPU相关的优化' },
      { text: '数据工程', link: '/zh/chapter11/chapter11_数据工程' },
      { text: '训练流程', link: '/zh/chapter13/chapter13_第十三章大模型的基本训练流程' },
      { text: '评估与基准测试', link: '/zh/chapter12/chapter12_评估与基准测试' },
    ],
  },
  {
    text: '进阶选修',
    items: [
      { text: '分布式训练', link: '/zh/chapter8/chapter8_第八章分布式训练' },
      { text: 'GPU高性能编程', link: '/zh/chapter7/chapter7_第七章GPU高性能编程' },
      { text: 'Scaling Laws', link: '/zh/chapter9/chapter9_Scaling_Laws' },
      { text: '推理', link: '/zh/chapter10/推理' },
      { text: 'RLVR', link: '/zh/chapter14/chapter14_可验证奖励的强化学习' },
      { text: '扩展内容', link: '/zh/chapter15/什么是LLM推理' },
    ],
  },
  {
    text: '扩展章节',
    items: [{ text: '第15章', link: '/zh/chapter15/什么是LLM推理' }],
  },
]

const zhSidebar = {
  '/zh/': [
    {
      text: '章节',
      items: [
        { text: '前言', link: '/zh/前言' },
        { text: '第1章 工具使用', link: '/zh/chapter1/wandb使用介绍' },
        { text: '第2章 分词器', link: '/zh/chapter2/chapter2_分词器' },
        { text: '第3章 PyTorch 与资源核算', link: '/zh/chapter3/chapter3_pytorch与资源核算' },
        { text: '第4章 语言模型架构与训练细节', link: '/zh/chapter4/chapter4_第四章语言模型架构和训练的技术细节' },
        { text: '第5章 混合专家模型', link: '/zh/chapter5/chapter5_混合专家模型' },
        { text: '第6章 GPU 与相关优化', link: '/zh/chapter6/chapter6_第六章GPU和GPU相关的优化' },
        { text: '第7章 GPU 高性能编程', link: '/zh/chapter7/chapter7_第七章GPU高性能编程' },
        { text: '第8章 分布式训练', link: '/zh/chapter8/chapter8_第八章分布式训练' },
        { text: '第9章 Scaling Laws', link: '/zh/chapter9/chapter9_Scaling_Laws' },
        { text: '第10章 推理', link: '/zh/chapter10/推理' },
        { text: '第11章 数据工程', link: '/zh/chapter11/chapter11_数据工程' },
        { text: '第12章 评估与基准测试', link: '/zh/chapter12/chapter12_评估与基准测试' },
        { text: '第13章 大模型的基本训练流程', link: '/zh/chapter13/chapter13_第十三章大模型的基本训练流程' },
        { text: '第14章 可验证奖励的强化学习', link: '/zh/chapter14/chapter14_可验证奖励的强化学习' },
        { text: '第15章 扩展内容', link: '/zh/chapter15/什么是LLM推理' },
      ],
    },
  ],
}

export default defineConfig({
  title: 'Diy-LLM',
  description: '面向中文学习者的大语言模型系统化学习课程。',
  base: '/diy-llm/',
  ignoreDeadLinks: true,
  head: [
    ['link', { rel: 'icon', href: '/diy-llm/datawhale.png' }],
    ['meta', { name: 'theme-color', content: '#2563eb' }],
  ],
  markdown: {
    math: true,
    theme: {
      light: 'github-light',
      dark: 'github-dark',
    },
  },
  rewrites: {
    'en/:rest*': ':rest*',
  },
  themeConfig: {
    logo: '/datawhale.png',
    search: { provider: 'local' },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/datawhalechina/diy-llm' },
    ],
    outline: {
      level: [2, 3],
    },
  },
  locales: {
    root: {
      label: 'English',
      lang: 'en',
      themeConfig: {
        nav: rootNav,
        sidebar: rootSidebar,
        outlineTitle: 'On this page',
        returnToTopLabel: 'Back to top',
        darkModeSwitchLabel: 'Appearance',
        lightModeSwitchTitle: 'Switch to light theme',
        darkModeSwitchTitle: 'Switch to dark theme',
        sidebarMenuLabel: 'Menu',
        docFooter: {
          prev: 'Previous page',
          next: 'Next page',
        },
      },
    },
    zh: {
      label: '中文',
      lang: 'zh-CN',
      link: '/zh/',
      themeConfig: {
        nav: zhNav,
        sidebar: zhSidebar,
        outlineTitle: '本页目录',
        returnToTopLabel: '返回顶部',
        darkModeSwitchLabel: '外观',
        lightModeSwitchTitle: '切换到浅色模式',
        darkModeSwitchTitle: '切换到深色模式',
        sidebarMenuLabel: '菜单',
        docFooter: {
          prev: '上一页',
          next: '下一页',
        },
      },
    },
  },
})
