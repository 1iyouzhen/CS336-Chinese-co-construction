import { defineConfig } from 'vitepress'
import { katex } from '@mdit/plugin-katex'

const isRootDeploy =
  process.env.VERCEL === '1' ||
  !!process.env.VERCEL_URL ||
  process.env.EDGEONE === '1'

const base = process.env.BASE || (isRootDeploy ? '/' : '/diy-llm/')

const sidebar = [
  {
    text: 'Diy-LLM',
    collapsed: false,
    items: [
      { text: '首页', link: '/' },
      { text: '前言', link: '/前言' },
      { text: '第1章 工具使用', link: '/chapter1/wandb使用介绍' },
      { text: '第2章 分词器', link: '/chapter2/chapter2_分词器' },
      { text: '第3章 PyTorch 与资源核算', link: '/chapter3/chapter3_pytorch与资源核算' },
      {
        text: '第4章 语言模型架构与训练细节',
        link: '/chapter4/chapter4_第四章语言模型架构和训练的技术细节'
      },
      { text: '第5章 混合专家模型', link: '/chapter5/chapter5_混合专家模型' },
      { text: '第6章 GPU 与相关优化', link: '/chapter6/chapter6_第六章GPU和GPU相关的优化' },
      { text: '第7章 GPU 高性能编程', link: '/chapter7/chapter7_第七章GPU高性能编程' },
      { text: '第8章 分布式训练', link: '/chapter8/chapter8_第八章分布式训练' },
      { text: '第9章 Scaling Laws', link: '/chapter9/chapter9_Scaling_Laws' },
      { text: '第10章 推理', link: '/chapter10/推理' },
      { text: '第11章 数据工程', link: '/chapter11/chapter11_数据工程' },
      { text: '第12章 评估与基准测试', link: '/chapter12/chapter12_评估与基准测试' },
      {
        text: '第13章 大模型的基本训练流程',
        link: '/chapter13/chapter13_第十三章大模型的基本训练流程'
      },
      { text: '第14章 可验证奖励的强化学习', link: '/chapter14/chapter14_可验证奖励的强化学习' },
      { text: '第15章 扩展内容', link: '/chapter15/什么是LLM推理' }
    ]
  }
]

export default defineConfig({
  title: 'Diy-LLM',
  description: '系统学习大语言模型的中文开源教程',
  lang: 'zh-CN',
  base,
  cleanUrls: true,
  ignoreDeadLinks: true,
  lastUpdated: true,
  markdown: {
    config(md) {
      md.use(katex)
    }
  },
  head: [
    [
      'link',
      {
        rel: 'stylesheet',
        href: 'https://cdn.jsdelivr.net/npm/katex@0.16.25/dist/katex.min.css'
      }
    ]
  ],
  themeConfig: {
    logo: '/images/diy-llm.png',
    siteTitle: 'Diy-LLM',
    nav: [
      { text: '首页', link: '/' },
      { text: '课程章节', link: '/前言' },
      { text: 'GitHub', link: 'https://github.com/datawhalechina/diy-llm' }
    ],
    sidebar,
    search: {
      provider: 'local'
    },
    outline: {
      level: [2, 4],
      label: '本页目录'
    },
    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },
    lastUpdated: {
      text: '最后更新',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'medium'
      }
    },
    footer: {
      message: 'Released under the CC BY-NC-SA 4.0 License.',
      copyright: 'Copyright © Diy-LLM contributors'
    }
  }
})
