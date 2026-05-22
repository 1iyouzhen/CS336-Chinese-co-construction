---
layout: home

hero:
  name: "Diy-LLM"
  text: "Learn Large Language Models Systematically"
  tagline: "A chapter-based learning path from tooling, tokenizers, and model architecture to training, evaluation, inference, and reinforcement learning."
  image:
    src: /diy-llm.png
    alt: Diy-LLM
  actions:
    - theme: brand
      text: Start Learning
      link: /chapter1/wandb使用介绍
    - theme: alt
      text: Tokenizer
      link: /chapter2/chapter2_分词器
    - theme: alt
      text: Training Pipeline
      link: /chapter13/chapter13_第十三章大模型的基本训练流程

features:
  - icon: 01
    title: Essentials
    details: Preface, tooling, tokenizers, PyTorch and resource accounting, architecture details, MOE, GPU optimization, data engineering, training pipelines, and evaluation benchmarks.
    link: /chapter1/wandb使用介绍
    linkText: Open essentials

  - icon: 02
    title: Advanced Topics
    details: Distributed training, GPU high performance programming, Scaling Laws, inference, RLVR, and extended content for deeper engineering and research context.
    link: /chapter8/chapter8_第八章分布式训练
    linkText: Open advanced topics

  - icon: 15
    title: Extra Chapters
    details: Chapter 15 focuses on LLM inference topics and works as a supplemental module after the main learning path.
    link: /chapter15/什么是LLM推理
    linkText: Read Chapter 15
---

<div class="home-badges">
  <span>Tooling</span>
  <span>Tokenizer</span>
  <span>PyTorch</span>
  <span>Transformer</span>
  <span>MOE</span>
  <span>GPU Optimization</span>
  <span>Data Engineering</span>
  <span>Training</span>
  <span>Evaluation</span>
  <span>Inference</span>
  <span>RLVR</span>
</div>

<div class="home-quick-grid">
  <a class="home-quick-card" href="./chapter1/wandb使用介绍">
    <strong>Build the learning and experiment workflow first</strong>
    <span>Start with W&B and related tooling to understand experiment tracking, training observation, and result review.</span>
  </a>
  <a class="home-quick-card" href="./chapter4/chapter4_第四章语言模型架构和训练的技术细节">
    <strong>Follow the model architecture thread</strong>
    <span>Build core engineering intuition around Transformer architecture, training details, MOE, and GPU optimization.</span>
  </a>
  <a class="home-quick-card" href="./chapter13/chapter13_第十三章大模型的基本训练流程">
    <strong>Connect the full training pipeline</strong>
    <span>Organize data engineering, training stages, evaluation benchmarks, and later RLVR topics into one executable learning path.</span>
  </a>
</div>
