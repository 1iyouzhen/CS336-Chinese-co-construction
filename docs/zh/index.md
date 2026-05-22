---
layout: home
sidebar: false
outline: false

hero:
  name: Diy-LLM
  text: 系统学习大语言模型的中文开源教程
  tagline: 从分词器、模型架构、GPU 优化到后训练与评测，沿着真实代码和作业逐步拆解 LLM 工程。
  image:
    src: /public/favicon/diy-llm.png
    alt: Diy-LLM
  actions:
    - theme: brand
      text: 开始阅读
      link: /chapter0/
    - theme: alt
      text: 查看章节
      link: /chapter0/
    - theme: alt
      text: GitHub
      link: https://github.com/datawhalechina/diy-llm
---

<HomeLanding />


## 关于 Diy-LLM

Diy-LLM 是为中文学习者量身打造的「大语言模型（LLM）系统学习」教程与实践项目。我们结合理论与工程实践，拆解 LLM 的核心模块并配套作业，目标是让学习者能够从数据处理、分词器、模型训练到评测与部署，逐步掌握构建大模型的完整能力。

### 我们的愿景

- 硬核理论与动手实战并重，带来可复现的工程实现。
- 构建循序渐进的知识体系，覆盖分词器、架构、训练、评测与部署。
- 贴近国内生态，结合国产模型与实战案例。


## 快速开始

克隆仓库并查看文档：

```bash
git clone https://github.com/datawhalechina/diy-llm.git
cd diy-llm
# 本地预览（需 node 环境和 pnpm）
pnpm install
pnpm run dev
```


## 课程目录（摘要）

主要章节包括：

- [前言](/chapter0/)
- [第1章 工具使用](/chapter1/)
- [第2章 分词器](/chapter2/)
- [第3章 PyTorch 与资源核算](/chapter3/)
- [第4章 语言模型架构与训练细节](/chapter4/)
- [第5章 混合专家模型](/chapter5/)
- [第6章 GPU 与相关优化](/chapter6/)
- [第7章 GPU 高性能編程](/chapter7/)
- [第8章 分布式训练](/chapter8/)
- [第9章 ScalingLaws](/chapter9/)
- [第10章 推理](/chapter10/)
- [第11章 数据工程](/chapter11/)
- [第12章 评估与基准测试](/chapter12/)
- [第13章 大模型的基本训练流程](/chapter13/)
- [第14章 可验证奖励的强化学习](/chapter14/)
- [第15章 扩展内容](/chapter15/)


## 作业与实践

本项目提供配套作业，覆盖 tokenizer、模型训练、系统优化、数据处理与评测等核心工程任务。详情见 `coursework/` 目录。


## 贡献与参与

欢迎提交 Issue、PR，或直接在仓库中参与章节与作业的完善。常见贡献流程：Fork → 新分支 → 提交 PR。


## 致谢

感谢 Stanford CS336 的原始课程和所有为本项目贡献内容的同学与组织。

