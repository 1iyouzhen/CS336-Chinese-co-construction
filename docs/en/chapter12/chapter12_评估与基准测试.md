# Chapter 12: Evaluation and Benchmarks

The core question of evaluation is: **given a fixed model, how "good" is it, exactly?** This may seem like a simple scoring problem, but it is in fact a deep and complex systematic project. Evaluation not only determines how we measure the performance of current models, but fundamentally shapes the development direction of future models. This chapter will start from what you see on the surface and delve into the essence, methods, challenges, and pitfalls of evaluation.

## 12.1 Introduction

When you open any LLM evaluation website or paper, what do you see first?

### Benchmark Scores

This is the most intuitive and common form of evaluation. When major models are released, they report scores on a series of standardized benchmarks. For example:

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-1-Deepseek-R1的基准性能.png" />
   <p>Figure 12.1 DeepSeek-R1 benchmark performance</p>
</div>

[DeepSeek-R1 paper Figure 1 reported benchmark performance](https://arxiv.org/pdf/2501.12948), showing DeepSeek-R1's performance on benchmarks like AIME 2024, Codeforces, GPQA Diamond, and MATH 500.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-2-llama4的基准性能.png" />
   <p>Figure 12.2 Llama 4 benchmark performance</p>
</div>

[Llama 4 Behemoth instruction-tuned benchmark performance](https://ai.meta.com/blog/llama-4-multimodal-intelligence/).

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-3-OLMo的基准性能.png" />
   <p>Figure 12.3 OLMo benchmark performance</p>
</div>

[Allen AI's OLMo-2-32B benchmark performance](https://allenai.org/blog/olmo2-32B), evaluated mainly on MATH, MMLU, DROP, and GSM8k.

Most language models are evaluated on roughly the same benchmarks, but not exactly identical. So what are these benchmarks, and what do these numbers mean?

Below is an example from the [HELM](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard) website, showing performance rankings of different models:

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-4-HELM-capibities.png" />
   <p>Figure 12.4 Performance rankings of different models on HELM</p>
</div>

In China, there are also many similar leaderboards. OpenCompass was officially launched in August 2023 by the Shanghai Artificial Intelligence Laboratory:

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-5-上海人工智能实验室opencompass.png" />
   <p>Figure 12.5 OpenCompass leaderboard</p>
</div>

[SuperCLUE](https://superclueai.com/homepage) is an authoritative domestic independent third-party AI evaluation organization.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-6-SuperCLUE.png" />
   <p>Figure 12.6 SuperCLUE leaderboard</p>
</div>

### Cost and Value

Of course, evaluation cannot only look at capability — cost and inference speed are also key dimensions. Another example comes from [Artificial Analysis](https://artificialanalysis.ai/):

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-7-Artificial Analysis.png" />
   <p>Figure 12.7 Artificial Analysis leaderboard</p>
</div>

Sites like Artificial Analysis combine model performance with per-token cost, plotting a Pareto Frontier.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-8-Intelligence vs Cost to Run Artificial Analysis Intelligence Index (12 Dec '25) .png" />
   <p>Figure 12.8 Intelligence vs. Cost comparison</p>
</div>

### User Choice and Market Feedback

Another form of "evaluation" is seeing what users actually choose. [OpenRouter](https://openrouter.ai/rankings) routes traffic to different models and accumulates data on user preferences.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-9-不同厂商token使用排行榜.png" />
   <p>Figure 12.9 Token usage leaderboard by vendor</p>
</div>

Another very popular evaluation leaderboard is [Chatbot Arena](https://huggingface.co/spaces/lmarena-ai/lmarena-leaderboard), launched by the international open research organization LMSYS Org. Its biggest feature is an **anonymous blind test** mechanism that randomly pairs models and uses **user voting** combined with an Elo rating system.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-10-chatbot-arera.png" />
   <p>Figure 12.10 Chatbot Arena leaderboard</p>
</div>

### Subjective Evaluation and Community Sentiment

Finally, there are the "vibes" from social media (e.g., X platform). People share amazing model demos, which are informal, emotional evaluations.

However, as [Andrej Karpathy](https://x.com/karpathy/status/1896266683301659068) points out, we are currently facing an "evaluation crisis."

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-11-Andrej Karpathy关于当前评估问题的看法.png" />
   <p>Figure 12.11 Andrej Karpathy's views on current evaluation problems</p>
</div>

> My feeling is that we're currently facing an evaluation crisis. I really don't know which metrics to look at anymore. MMLU was a good and useful benchmark for several years, but that period has long ended. SWE-Bench Verified I really like and it's excellent, but its coverage is too narrow. Chatbot Arena is being overfit by labs... In short, I really don't know how good these models are anymore.

## 12.2 How to Think About Evaluation

Evaluation is by no means a mechanical scripted process. It is a framework that requires careful thought, and its design must serve a clear purpose.

> Core principle: There is no single "correct" evaluation.

The meaning of evaluation depends on the question you want to answer — for end users, researchers, policymakers, or model developers. Four key aspects need to be considered: what is the input, how to call the model, how to evaluate the output, and how to interpret the results.

## 12.3 Perplexity

### 12.3.1 What is Perplexity?

Language models are essentially probability distributions $p(x)$ over token sequences. Perplexity measures the model's ability to assign high probability to a dataset.

$$
\text{Perplexity} = \left( \frac{1}{p(D)} \right)^{1/N}
$$

Where $p(D)$ is the joint probability of all sentences in dataset $D$, and $|D|$ is the total number of tokens.

### 12.3.2 Why Use a Test Set?

We must measure perplexity on **unseen data (test set)** to reflect the model's generalization ability.

### 12.3.3 Classic Benchmark Datasets

| Dataset | Characteristics |
|--------|------|
| **Penn Treebank (PTB)** | Small scale, WSJ corpus, used in early RNN/LSTM experiments |
| **WikiText-103** | Large-scale Wikipedia-based English corpus |
| **One Billion Word Benchmark (1BW)** | From MT datasets, large vocabulary, challenging |

### 12.3.4 History and Evolution

In 2016, Jozefowicz et al. used a pure CNN+LSTM architecture to reduce perplexity on the 1BW benchmark from 51.3 to 30.0. GPT-2, trained on WebText (~40GB from Reddit links), performed well on zero-shot evaluation across standard datasets, demonstrating strong generalization.

### 12.3.5 Why Perplexity Still Matters

- Smoother, better suited for fitting scaling laws
- General-purpose — works on any text
- Extensible to conditional perplexity

### 12.3.6 Pitfalls of Using Perplexity

Perplexity evaluation requires the model to output per-token probability distributions, which relies on model internals and introduces trust issues.

### 12.3.7 Downstream Tasks That Reflect True "Understanding"

- **Cloze tasks** like LAMBADA require predicting a masked word from context, testing long-distance dependency and semantic coherence
- **Commonsense reasoning** like HellaSwag requires selecting the most natural/plausible continuation, testing everyday commonsense reasoning

## 12.4 Knowledge Benchmarks

These benchmarks measure the factual knowledge that the model has mastered.

### MMLU (Massive Multitask Language Understanding)

Contains multiple-choice questions across 57 disciplines. Initially evaluated with GPT-3 few-shot prompting, the largest X-Large model scored below 0.5; the strongest LLMs now reach 90+ on MMLU.

### MMLU-Pro

An improved version of MMLU that removes noisy or overly simple questions and increases options from 4 to 10, significantly reducing saturation.

### GPQA (Graduate-Level Google-Proof Q&A)

High-difficulty questions designed by 61 PhDs. PhD experts achieve 65% accuracy; non-experts with Google access achieve 34%; GPT-4 achieves 39%. Top models now reach 80.3%.

### Humanity's Last Exam

Contains 2,500 multimodal, multidisciplinary questions. Top models currently reach 38.3%.

## 12.5 Instruction-Following Benchmarks

These evaluate whether the model is "obedient" and can output as required by the user.

### Chatbot Arena

Uses blind testing and an ELO rating system. Real users submit prompts, simultaneously receive replies from two anonymous models, and choose the better one. Geimini 3 Pro currently leads with an Arena Score of 1,492.

### IFEval

Uses automatically verifiable constraints (e.g., "answer must contain at least 5 sentences"). Frontier models achieve 0.951 accuracy.

### AlpacaEval

Includes 805 instructions from various sources, using a strong LLM (e.g., GPT-4) as judge. Problem: judge bias exists; early versions are easily deceived by surface features like answer length.

### WildBench

Built from sampling over 1 million real human-machine dialogues, using GPT-4-Turbo as judge. Its results are highly correlated with Chatbot Arena (correlation coefficient 0.95).

## 12.6 Agent Benchmarks

These evaluate agent capabilities, completing tasks in complex environments through tool invocation and iterative planning.

### SWE-Bench

Contains 2,294 tasks across 12 Python codebases: given a GitHub issue description, submit a PR that passes unit tests.

### CyBench

Complete 40 cybersecurity "CTF" challenges. Task difficulty is measured by human "first-solve time," with some tasks taking up to 24 hours.

### MLEBench

Automatically participate in 75 Kaggle ML competitions including data processing, model training, hyperparameter tuning, and submission. Current best models win any medal at under 20%.

## 12.7 Pure Reasoning Benchmarks

### ARC-AGI

Provides a series of visual input-output grid pairs, requiring the model to infer transformation rules and apply them to new inputs — no language involved. Traditional LLMs perform extremely poorly, but the latest o1/o3 class reasoning models show some capability.

## 12.8 Safety Benchmarks

### HarmBench

Defines 510 harmful behaviors that violate laws or social norms, testing whether the model will execute them.

### AIR-Bench

Based on global regulatory frameworks and company policies, a systematic evaluation set containing 314 risk categories and 5,694 prompts.

### Jailbreaking

Attackers can bypass safety guardrails (e.g., GCG), and such attacks can even transfer from open-source to closed-source models.

### Pre-Deployment Testing

The US AISI and UK AISI collaborate: companies provide model access to safety agencies before release, who evaluate and submit reports.

## 12.9 Truthfulness

### Clio (Anthropic)

Uses language models to analyze real user data, sharing general patterns of what people ask.

### MedHELM

28 clinical doctors contributed 121 real clinical tasks, closer to actual medical application scenarios. However, there is an inherent tension between truthfulness and privacy.

## 12.10 Validity

Two major challenges: train-test overlap (ensuring test sets haven't been "seen" by the model) and dataset quality (many benchmarks contain annotation errors or noise).

## 12.11 What Are We Really Evaluating?

Past: We evaluated methods — the quality of new algorithms under fixed datasets and training protocols.
Now: We mostly evaluate models/systems — end-to-end final products.

Both paradigms are valuable, but the "rules of the game" must be clearly defined.

---

## References and Further Reading

- [MMLU (Hendrycks et al., 2021)](https://arxiv.org/abs/2009.03300)
- [MMLU-Pro (Wang et al., 2024)](https://arxiv.org/abs/2406.01574)
- [GPQA (Rein et al., 2023)](https://arxiv.org/abs/2311.12022)
- [HLE (Phan et al., 2025)](https://arxiv.org/abs/2501.14249)
- [Chatbot Arena (Chiang et al., 2024)](https://arxiv.org/abs/2403.04132)
- [AlpacaEval (Dubois et al., 2024)](https://arxiv.org/abs/2404.04475)
- [WildBench (Lin et al., 2024)](https://arxiv.org/abs/2406.04770)
- [SWE-Bench (Jimenez et al., 2024)](https://arxiv.org/abs/2310.06770)
- [CyBench (Zhang et al., 2024)](https://arxiv.org/abs/2408.08926)
- [MLEBench (Chan et al., 2024)](https://arxiv.org/abs/2410.07095)
- [ARC-AGI](https://arcprize.org/)
- [HarmBench (Mazeika et al., 2024)](https://arxiv.org/abs/2402.04249)
- [AIR-Bench (Zeng et al., 2024)](https://arxiv.org/abs/2407.17436)
- [GCG Jailbreaking (Zou et al., 2023)](https://arxiv.org/abs/2307.15043)
- [HELM](https://crfm.stanford.edu/helm/)
- [CS336 Course Website](https://cs336.stanford.edu/)
