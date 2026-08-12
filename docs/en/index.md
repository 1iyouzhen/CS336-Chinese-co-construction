# Diy-LLM — A Systematic Guide to Building Large Language Models

<div align="center">
  <img src="./images/diy-llm.png" alt="diy-llm" width="100%">
  <p><em>An "LLM Alchemy Workshop" tailor-made for learners worldwide</em></p>
</div>

We hope this CS336 course will be more than just a translation of the Stanford original — it should be a hands-on "LLM Alchemy Workshop" where you forge understanding, polish code, control the heat, and ultimately refine your own model.

## Prerequisites

- **Python**: Proficient in Python and software engineering skills
- **Deep Learning**: Familiar with PyTorch and neural network fundamentals
- **Mathematics**: Linear algebra, probability & statistics, calculus
- **Machine Learning**: Solid grasp of ML and DL basics
- **GPU Programming (optional)**: CUDA basics help but aren't required

## Course Vision

- **Rigorous theory meets hands-on practice**: Full technical depth with a structured learning path
- **Progressive knowledge system**: Breaking down the massive LLM project into digestible, hands-on modules
- **Code-driven, theory-integrated**: Every assignment includes implementation code and the thinking behind each line
- **Localized for real-world use**: Practical solutions for diverse compute environments, with examples from Qwen, DeepSeek, and other major open-source models

## What You'll Gain

- **Solid technical foundation**: Build your own LLM from scratch, understanding every core component
- **Real engineering experience**: Full-pipeline skills from data processing to deployment optimization
- **Industry competitiveness**: Core capabilities for LLM R&D roles
- **Research vision**: Systematic knowledge for future deep research

## Table of Contents

| Chapter | Key Content | Assignment | Status |
|------|----------|----------|------|
| [Foreword](Foreword.md) | Project background, learning roadmap, prerequisites | - | ✅ |
| [Ch1 Tooling](chapter1/wandb_usage_guide.md) | W&B experiment tracking, hyperparameter search | - | ✅ |
| [Ch2 Tokenizer](chapter2/chapter2_Tokenizer.md) | BPE algorithm, Unicode normalization, tokenizer training | [HW1](../coursework/assignment1-basics/) | ✅ |
| [Ch3 PyTorch & Resource Accounting](chapter3/chapter3_PyTorch_and_Resource_Accounting.md) | Mixed precision, gradient accumulation, FLOPs/memory estimation | - | ✅ |
| [Ch4 Architecture & Training Details](chapter4/chapter4_Architecture_and_Training_Details.md) | RoPE, RMSNorm, SwiGLU, AdamW, learning rate schedules | [HW1](../coursework/assignment1-basics/) | ✅ |
| [Ch5 Mixture of Experts](chapter5/chapter5_Mixture_of_Experts.md) | Top-K routing, load balancing, expert parallelism, DeepSeekMoE | - | ✅ |
| [Ch6 GPU & Optimization](chapter6/chapter6_GPU_and_Optimization.md) | Memory bandwidth, Flash Attention, kernel fusion, BF16 | [HW2](../coursework/assignment2-systems/) | ✅ |
| [Ch7 GPU High-Perf Programming](chapter7/chapter7_GPU_High_Performance_Programming.md) | CUDA, Tensor Cores, shared memory, Triton | [HW2](../coursework/assignment2-systems/) | ✅ |
| [Ch8 Distributed Training](chapter8/chapter8_Distributed_Training.md) | Data/Model/Pipeline parallelism, ZeRO, FSDP, All-Reduce | [HW2](../coursework/assignment2-systems/) | ✅ |
| [Ch9 Scaling Laws](chapter9/chapter9_Scaling_Laws.md) | Chinchilla optimal, compute-efficient training | [HW3](../coursework/assignment3-scaling/) | ✅ |
| [Ch10 Inference](chapter10/chapter10_Inference.md) | KV cache, speculative decoding, quantization, PagedAttention | [HW6](../coursework/assignment6-evaluation/) | ✅ |
| [Ch11 Data Engineering](chapter11/chapter11_Data_Engineering.md) | Data filtering, MinHash dedup, PII removal, data recipes | [HW4](../coursework/assignment4-data/) | ✅ |
| [Ch12 Evaluation & Benchmarks](chapter12/chapter12_Evaluation_and_Benchmarks.md) | MMLU, HumanEval, HELM, Chatbot Arena, safety benchmarks | [HW6](../coursework/assignment6-evaluation/) | ✅ |
| [Ch13 Training Pipeline](chapter13/chapter13_Training_Pipeline.md) | Pre-training, SFT, DPO, RLHF PPO pipeline, alignment | [HW5](../coursework/assignment5-alignment/) | ✅ |
| [Ch14 RLVR](chapter14/chapter14_RLVR.md) | GRPO, rule-based verifiers, process reward, DeepSeek-R1 | [HW5](../coursework/assignment5-alignment/) | ✅ |
| [Ch15 Multimodal Models](chapter15/chapter15_Multimodal_Models.md) | CLIP, SigLIP, LLaVA, Qwen-VL, Chameleon, Omni Model | - | ✅ |
| [Ch16 Extended Content](chapter16/) | 1. What is LLM Reasoning?<br>2. LLM Future — LeCun's Roadmap | - | 🔄 |

> Status: ✅ Complete &ensp;🔄 In Progress&ensp;📝 Draft&ensp;🚧 Planned&ensp;⏸️ Paused

## Assignments

| HW | Core Tasks | Status |
|------|----------|------|
| [HW1: Build an LLM](../coursework/assignment1-basics/) | Implement tokenizer, model architecture, optimizer; train a minimal LM | ✅ |
| [HW2: Systems](../coursework/assignment2-systems/) | Benchmarking, FlashAttention-2 in Triton, distributed training code | ✅ |
| [HW3: Scaling Laws](../coursework/assignment3-scaling/) | Fit scaling laws, predict model scaling behavior | ✅ |
| [HW4: Data Processing](../coursework/assignment4-data/) | Convert Common Crawl raw data to pre-training dataset | ✅ |
| [HW5: Alignment](../coursework/assignment5-alignment/) | Apply SFT and RL (e.g., GRPO) to train on math problems | ✅ |
| [HW6: Evaluation](../coursework/assignment6-evaluation/) | Multi-dimension evaluation using lm-evaluation-harness and evalscope | ✅ |

## Quick Start

```bash
git clone https://github.com/datawhalechina/diy-llm.git
cd diy-llm
```

### Learning Path

1️⃣ Theory → Read `docs/en/` chapters in order  
2️⃣ Practice → Complete the 6 assignments in `coursework/`  
3️⃣ Deep Dive → Read code implementations, understand design decisions  

## Project Structure

```
diy-llm/
├── docs/
│   ├── zh/                 # Chinese docs (default)
│   ├── en/                 # English docs
│   └── .vitepress/         # VitePress config
├── coursework/             # Assignments
│   ├── assignment1-basics/
│   └── ...
└── README.md
```

## Links

- **Repository**: https://github.com/datawhalechina/diy-llm
- **Online Reading**: https://datawhalechina.github.io/diy-llm/
- **Original Course**: [Stanford CS336 (Spring 2026)](https://cs336.stanford.edu/)

## FAQ

<details>
<summary><b>Q: Can I learn without a GPU?</b></summary>
Theory can be studied normally. Some assignments can be debugged on CPU, but full training requires GPU. Cloud services are recommended.
</details>

<details>
<summary><b>Q: How does this differ from the original CS336?</b></summary>
We retain the original technical depth while adding Chinese explanations, detailed reference sources, domestic model examples (Qwen, DeepSeek), and complete assignment implementations.
</details>

## Contributors

<table border="0">
  <tbody>
    <tr align="center" >
      <td>
         <a href="https://github.com/xuhu0115"><img width="70" height="70" src="https://github.com/xuhu0115.png?s=40" alt="pic"></a><br>
         <a href="https://github.com/xuhu0115">Xu Hu (徐虎)</a> 
        <p>Project Lead · Datawhale · SJTU<br>Chapters: 1, 3, 9, 12, 14, 15; HW5, 6; Content review</p>
      </td>
      <td>
         <a href="https://github.com/kangkang-Adam"><img width="70" height="70" src="https://github.com/kangkang-Adam.png?s=40" alt="pic"></a><br>
         <a href="https://github.com/kangkang-Adam">Li Shengkang (李盛康)</a> 
        <p>Project Lead · Datawhale · XUPT<br>Chapters: 4, 6, 7, 8, 13; HW2, 4</p>
      </td>
      <td>
         <a href="https://github.com/1iyouzhen"><img width="70" height="70" src="https://github.com/1iyouzhen.png?s=40" alt="pic"></a><br>
         <a href="https://github.com/1iyouzhen">Li Youzhen (黎又榛)</a>
         <p>Project Lead · Datawhale TA<br>Chapters: 2, 5, 10, 11, 13; HW1, 3</p>
      </td>
    </tr>
  </tbody>
</table>

- Thanks to [@aimetrics](https://github.com/aimetrics) for adding MPS support for MacBooks in HW1
- Thanks to [@FuTseYi](https://github.com/FuTseYi) (Datawhale) for website migration, deployment, and refactoring
- Thanks to [@jiangyinhe](https://github.com/jiangyinhe) for contributing to Ch16 "LLM Future — LeCun"; and for translating the first chapter and preface into English based on the Chinese version.

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="CC BY-NC-SA 4.0" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>

This work is licensed under [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/).

## Acknowledgments

- Stanford CS336 course team for the excellent original course
- [@Sm1les](https://github.com/Sm1les) for support and guidance
- All contributors and the open-source community

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=datawhalechina/diy-llm&type=Date)](https://star-history.com/#datawhalechina/diy-llm&Date)

---

<div align="center">
  <p>Empowering everyone to systematically learn large language model construction</p>
  <p>Made with ❤️ by Datawhale</p>
</div>
