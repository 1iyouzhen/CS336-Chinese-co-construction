---
layout: home

hero:
  name: "Diy-LLM"
  text: "系统学习大语言模型"
  tagline: "从工具、分词器、模型架构到训练、评估、推理与强化学习，按章节搭建完整的 LLM 学习路径。"
  image:
    src: /diy-llm.png
    alt: Diy-LLM
  actions:
    - theme: brand
      text: 开始学习
      link: /zh/chapter1/wandb使用介绍
    - theme: alt
      text: 分词器
      link: /zh/chapter2/chapter2_分词器
    - theme: alt
      text: 训练流程
      link: /zh/chapter13/chapter13_第十三章大模型的基本训练流程

features:
  - icon: 01
    title: 基础必学
    details: 前言、工具使用、分词器、PyTorch 与资源核算、架构细节、MOE、GPU 优化、数据工程、训练流程和评估基准。
    link: /zh/chapter1/wandb使用介绍
    linkText: 进入基础章节

  - icon: 02
    title: 进阶选修
    details: 分布式训练、GPU 高性能编程、Scaling Laws、推理、RLVR 和扩展内容，帮助进一步理解工程与研究边界。
    link: /zh/chapter8/chapter8_第八章分布式训练
    linkText: 进入进阶章节

  - icon: 15
    title: 扩展章节
    details: 第15章聚焦 LLM 推理相关主题，可作为完成主线学习后的专题补充。
    link: /zh/chapter15/什么是LLM推理
    linkText: 阅读第15章
---

<div class="home-badges">
  <span>工具使用</span>
  <span>分词器</span>
  <span>PyTorch</span>
  <span>Transformer</span>
  <span>MOE</span>
  <span>GPU优化</span>
  <span>数据工程</span>
  <span>训练流程</span>
  <span>评估</span>
  <span>推理</span>
  <span>RLVR</span>
</div>

<div class="home-quick-grid">
  <a class="home-quick-card" href="./chapter1/wandb使用介绍">
    <strong>先建立学习和实验工具链</strong>
    <span>从 W&B 等工具开始，理解课程中的实验记录、训练观察和结果追踪方式。</span>
  </a>
  <a class="home-quick-card" href="./chapter4/chapter4_第四章语言模型架构和训练的技术细节">
    <strong>抓住模型结构主线</strong>
    <span>围绕 Transformer、训练细节、MOE 和 GPU 优化建立大语言模型的核心工程认知。</span>
  </a>
  <a class="home-quick-card" href="./chapter13/chapter13_第十三章大模型的基本训练流程">
    <strong>串起完整训练流程</strong>
    <span>把数据工程、训练阶段、评估基准和后续 RLVR 内容组织成可执行的学习路径。</span>
  </a>
</div>
