# Chapter 12: Evaluation and Benchmarks

The core question of evaluation is: **given a fixed model, just how "good" is it?** This may seem like a simple scoring problem, but it is in fact a deep and complex systematic engineering challenge. Evaluation not only determines how we measure the performance of current models, but fundamentally shapes the development direction of future models. This chapter will start from what you see on the surface and delve into the essence, methods, challenges, and pitfalls of evaluation.

## 12.1 Introduction

When you open any LLM evaluation website or paper, what do you see first?

### Benchmark Scores

This is the most intuitive and common form of evaluation. When major models are released, they all report scores on a series of standardized benchmarks. For example:

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-1-Deepseek-R1的基准性能.png" />
   <p>Figure 12.1 DeepSeek-R1 benchmark performance</p>
</div>

[DeepSeek-R1 paper Figure 1 reported benchmark performance](https://arxiv.org/pdf/2501.12948), showing DeepSeek-R1's performance on benchmarks like AIME 2024, Codeforces, GPQA Diamond, and MATH 500.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-2-llama4的基准性能.png" />
   <p>Figure 12.2 Llama 4 benchmark performance</p>
</div>

[Llama 4 Behemoth instruction-tuned version benchmark performance](https://ai.meta.com/blog/llama-4-multimodal-intelligence/), showing its performance on MMLU-Pro, MATH500, GPQA, and other benchmarks.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-3-OLMo的基准性能.png" />
   <p>Figure 12.3 OLMo benchmark performance</p>
</div>

[Allen AI team's OLMo-2-32B model benchmark performance](https://allenai.org/blog/olmo2-32B), primarily evaluated on MATH, MMLU, DROP, and GSM8k.

Most language models are evaluated on roughly the same benchmarks, but they are not exactly identical. So what exactly are these benchmarks? And what do these numbers mean?

Below is an example from the [HELM](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard) website, showing the performance rankings of different models across multiple benchmarks:

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-4-HELM-capibities.png" />
   <p>Figure 12.4 Performance rankings of different models on the HELM website</p>
</div>

In addition, there are many similar LLM leaderboards in China. OpenCompass is an open LLM evaluation system officially launched by the Shanghai Artificial Intelligence Laboratory in August 2023:

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-5-上海人工智能实验室opencompass.png" />
   <p>Figure 12.5 Performance rankings on the Sinan OpenCompass website</p>
</div>

[SuperCLUE](https://superclueai.com/homepage) is an authoritative domestic independent third-party AI evaluation organization, originating from the CLUE open-source community launched in 2019.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-6-SuperCLUE.png" />
   <p>Figure 12.6 Performance rankings on the SuperCLUE website</p>
</div>

### Cost and Value

Of course, evaluation cannot only look at capability — cost and inference speed are also key dimensions. Another example comes from the [Artificial Analysis](https://artificialanalysis.ai/) website, which evaluates models from three angles: intelligence, inference speed, and price:

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-7-Artificial Analysis.png" />
   <p>Figure 12.7 Performance rankings on the Artificial Analysis website</p>
</div>

Sites like Artificial Analysis combine model performance with per-token cost, plotting a Pareto Frontier. This reveals a reality: top-tier models (like O3) are powerful but expensive; while some lower-ranked models may achieve a better balance between performance and cost. The intelligence index includes: MMLU-Pro, GPQA Diamond, Humanity's Last Exam, LiveCodeBench, SciCode, AIME 2025, IFBench, AA-LCR, Terminal-Bench Hard, and τ²-Bench Telecom.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-8-Intelligence vs Cost to Run Artificial Analysis Intelligence Index (12 Dec '25) .png" />
   <p>Figure 12.8 Performance vs. cost comparison on the Artificial Analysis website</p>
</div>

### User Choice and Market Feedback

Another form of "evaluation" is seeing what users actually choose. The [OpenRouter](https://openrouter.ai/rankings) website, by routing traffic to different models, has accumulated large amounts of user preference data. The "usage" leaderboard shows that OpenAI, Anthropic, Google, and domestic DeepSeek and Qwen models currently dominate. This indicates that "being widely used" is itself a powerful indicator of "goodness."

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-9-不同厂商token使用排行榜.png" />
   <p>Figure 12.9 Token usage leaderboard by vendor</p>
</div>

Another very popular evaluation leaderboard is [Chatbot Arena](https://huggingface.co/spaces/lmarena-ai/lmarena-leaderboard), launched by LMSYS Org. Its biggest feature is an **anonymous blind test** mechanism randomly pairing models, quantifying capability through **user voting** combined with an Elo rating system. It was jointly developed by researchers from UC Berkeley, UC San Diego, and Carnegie Mellon University.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-10-chatbot-arera.png" />
   <p>Figure 12.10 Chatbot Arena leaderboard</p>
</div>

### Subjective Evaluation and Community Word-of-Mouth

Finally, there are the "vibes" from social media (such as X). People share amazing model demonstrations. These informal, emotional evaluations are also a side source for assessing model capabilities.

However, as [Andrej Karpathy](https://x.com/karpathy/status/1896266683301659068) points out, we are currently facing an "evaluation crisis."

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-11-Andrej Karpathy关于当前评估问题的看法.png" />
   <p>Figure 12.11 Andrej Karpathy's views on current evaluation problems</p>
</div>

> My feeling is that we're currently facing an evaluation crisis. I really don't know which metrics to look at anymore. MMLU was a good and useful benchmark for several years, but that period has long ended. SWE-Bench Verified (based on real, practical, and verified problems) I really like and it's excellent, but its coverage is too narrow. Chatbot Arena has been over-focused on (partially perhaps my fault?), to the point where major LLM labs have started severely overfitting it —手段 include prompt mining from API requests, repeatedly testing with large private evaluation sets, and worse, even directly using leaderboard rankings as training supervision signals. I think it barely still works for now, since there's no明显 "better" alternative, but its signal quality seems to be declining. Some private evaluation sets have now appeared; combining them into an ensemble evaluation system may be a promising path forward. In the absence of good comprehensive evaluation methods, I've tried转向 "vibe checks," but now I worry this approach is misleading —容易带来 confirmation bias, too-small sample sizes, and is实在不可靠. In short, my feeling is: I really don't know how good these models are anymore.

Many mainstream benchmarks (like MMLU) are now either saturated, being gamed, or both. Chatbot Arena is also questioned due to commercial interests. We hold massive amounts of data and rankings, yet lack a clear, reliable, and widely accepted evaluation standard.

## 12.2 How to Think About Evaluation

Evaluation is by no means a mechanical scripted process. It is a framework that requires careful thought, and its design must serve a clear purpose.

> Core principle: There is no single "correct" evaluation.

The meaning of evaluation depends on the question you want to answer:

- For end users/enterprises: Which model should I purchase to meet my specific needs?
- For researchers: Are we making scientific progress on the model's raw capabilities?
- For policymakers/safety agencies: What are the benefits and risks of current models?
- For model developers: How should I improve my model? Which interventions are effective?

To translate abstract goals into concrete evaluation plans, we need to think about four key aspects:

#### What is the input?

- Where do the prompts come from? What usage scenarios do they cover?
- Do we include representative difficult samples from the long-tail distribution?
- Does the input need to be adapted to the model? In multi-turn dialogue, evaluation becomes dynamic.

#### How to call the model?

- How to prompt? Zero-shot, few-shot, or Chain-of-Thought?
- Is the model allowed to use external tools?
- Are we evaluating the language model itself, or a complete system with agent scaffolding?

#### How to evaluate the output?

- Is the reference answer accurate?
- What metrics? pass@1 vs pass@10 for code generation?
- How to account for cost? A larger but only slightly better model may not be worth its cost.
- How to handle asymmetric errors? Hallucination costs are far higher in healthcare, finance, law.
- How to evaluate open-domain generation? Common practices: human evaluation, LLM-as-a-judge, proxy metrics, A/B testing.

#### How to interpret the results?

* What does a 91% score mean? Is it good enough for deployment?
* How to determine true generalization vs. memorization?
* What exactly are we evaluating: the product model, the complete system, or the research method?

Ignoring these questions and making judgments based on a single score is the biggest pitfall in evaluation.

## 12.3 Perplexity

### 12.3.1 What is Perplexity?

The essence of a language model is a probability distribution p(x) over token sequences. Perplexity measures the model's ability to assign high probability to a dataset. During pre-training, the model's goal is to minimize perplexity on the training set. The smaller the value, the better.

$$
\text{Perplexity} = \left( \frac{1}{p(D)} \right)^{1/N}
$$

Where $p(D)$ is the joint probability of all sentences in dataset $D$, and $|D|$ is the total number of tokens.

### 12.3.2 Why Use a Test Set?

To evaluate whether the model has truly "learned the language," we must measure perplexity on **unseen data (test set)** — this is what reflects the model's generalization ability.

> ⚠️ Note: Don't only look at training set perplexity, otherwise it's easy to overfit!

### 12.3.3 Classic Benchmark Datasets

| Dataset | Characteristics |
|--------|------|
| **Penn Treebank (PTB)** | Small scale, WSJ corpus, early RNN/LSTM experiments |
| **WikiText-103** | Large-scale Wikipedia-based English corpus |
| **One Billion Word Benchmark (1BW)** | From MT datasets, large vocabulary, highly challenging |

> 🎯 These datasets were once the "gold standard" for measuring model performance.

### 12.3.4 History and Evolution

2016 年，Jozefowicz et al. reduced perplexity on 1BW from 51.3 to 30.0 using a pure CNN+LSTM architecture — a significant breakthrough at the time.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-12-CNN+LSTM 架构显著降低困惑度.png" />
   <p>Figure 12.12 CNN+LSTM architecture significantly reduces perplexity</p>
</div>

OpenAI released GPT-2 in [《Language Models are Unsupervised Multitask Learners》](https://cdn.openai.com/better-language-models/), trained on WebText (~40GB from Reddit links), and performed well on zero-shot evaluation across standard datasets. This is "out-of-distribution" evaluation since training and test data come from different sources —但它表现很好,说明大规模、多样化的 training data帶來强大的 generalization.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-13-GPT-2的zero-shot（零样本）评估.png" />
   <p>Figure 12.13 GPT-2 zero-shot evaluation</p>
</div>

Key findings from GPT-2:
- Larger models → lower perplexity → better fit to language distribution
- On some tasks, accuracy improves with model size → language modeling ability transfers to downstream tasks
- On 1BW, large model improvements are smaller → dataset is more difficult or close to its limit

### 12.3.5 Why Perplexity Still Matters

- **Smoother, suitable for fitting scaling laws**: Downstream task accuracy fluctuates, while perplexity changes continuously
- **Strong generality, applicable to any text**: No need for labeled data
- **Extensible to conditional perplexity**: Can reflect model ability on specific tasks

📌 So, perplexity is not only a training objective but also an important tool for analyzing the model's intrinsic capabilities.

### 12.3.6 Pitfalls of Using Perplexity

Perplexity evaluation requires the model to output per-token probability distributions. This relies on model internals and introduces trust issues. It's more suitable for scenarios where researchers control the model architecture and training process themselves.

### 12.3.7 The "Perplexity Maximalist" Philosophical View

An idealist view: if you perfectly model the true distribution t, you solve all language tasks. In reality, we don't need to perfectly model the entire language distribution, and blindly pursuing the lowest perplexity may be a waste of resources.

### 12.3.8 Downstream Tasks That Reflect True Understanding

**Cloze Tasks** (LAMBADA): Given a passage of context, predict a single masked target word. This task requires understanding not only local grammar but also long-distance dependencies, semantic coherence, and world knowledge — focusing only on the prediction quality at a key position rather than the entire sentence. Thus, cloze tasks can be seen as "local perplexity."

From the [LAMBADA](https://arxiv.org/abs/1606.06031) dataset, three concrete examples:

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-14-LAMBADA完形填空任务的三个样例.png" />
   <p>Figure 12.14 Three examples from the LAMBADA cloze task</p>
</div>

✅ Example 1
> Context: "Yes, I thought I was going to lose the baby." … "This baby wasn't exactly planned for."
Target sentence: "Do you honestly think that I would want you to have a _____?"
Correct answer: miscarriage

✅ Example 2
> Context: "Why?" … "He was a great craftsman," said Heather. "That he was," said Flannery.
Target sentence: "And Polish, to boot," said ______.
Correct answer: Gabriel

✅ Example 3
> Context: Preston had been the last person to wear those chains...
Target sentence: Sergei looked at me, surprised by my low, raspy please, but he put down the _____.
Correct answer: chains

**Commonsense Reasoning** (HellaSwag): [HellaSwag](https://arxiv.org/pdf/1905.07830) is a commonsense reasoning task. Given a video clip or text description (premise) and four options (A, B, C, D), the model must select the most commonsensical, most natural, most likely后续 action. It emphasizes everyday commonsense reasoning and behavioral合理性 judgment.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-15-HellaSwag常识推理样例.png" />
   <p>Figure 12.15 HellaSwag commonsense reasoning example</p>
</div>

✅ Example 1: Bathing a Dog
> A woman is standing outside holding a bucket with a dog beside her. The dog runs around trying to avoid the bath. She...
> A. rinses the bucket with soap, then blow-dries the dog's head.
> B. uses a hose to keep it from getting wet.
> C. gets the dog wet, and it runs off again. ← Correct
> D. gets in the bathtub with the dog.

Analysis: In reality, dogs typically hate baths — the owner泼水 and the dog runs away. So "gets wet → dog runs" is the most common, most natural flow. A, B, D either violate common sense (dog can't obediently get head blow-dried) or are操作不合理 (use a hose to "prevent" getting wet?).

✅ Example 2: Traffic Rules
> At a stop sign or red light, you must come to a complete stop... If you stop at a red light, proceed after the light turns green...
> A. Stop for no more than two seconds...
> B. After coming to a complete stop, turn off your turn signal...
> C. Stay away from oncoming traffic...
> D. If there is a white stop line in your lane, stop before this line. Wait for all vehicles to pass before crossing the intersection. ← Correct

Analysis: D describes standard traffic rules: stop before the white line, confirm safety, then proceed. A is wrong (must stop sufficiently at red lights); B is irrelevant (turn signal not applicable in this scenario); C is vague ("stay away from oncoming traffic" is not the core operation at a red light).

HellaSwag can be seen as "perplexity in context" — the model doesn't need to output probabilities, but its choice should reflect its internal probability estimate of "which ending is most likely."

## 12.4 Knowledge Benchmarks

### MMLU (Massive Multitask Language Understanding)

[MMLU](https://arxiv.org/pdf/2009.03300.pdf) contains multiple-choice questions across 57 disciplines (from math, history to law, ethics). Questions sourced from the web and collected by students. It focuses more on knowledge than language understanding.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-16-GPT-3在MMLU上的少样本提示.png" />
   <p>Figure 12.16 GPT-3 few-shot prompting on MMLU</p>
</div>

Initially evaluated with GPT-3 few-shot prompting, the largest X-Large model scored below 0.5. The strongest LLMs now reach 90+ on MMLU. Top models (e.g., Gemini 3 Pro Preview) on [HELM](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/mmlu_pro) have reached 90.3% accuracy.

### MMLU-Pro

[MMLU-Pro](https://arxiv.org/abs/2406.01574) is an improved version, removing noisy or overly simple questions and increasing options from 4 to 10. Model accuracy drops significantly, alleviating saturation. Typically evaluated with Chain-of-Thought (CoT).

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-17-MMLU-pro对比MMLU.png" />
   <p>Figure 12.17 MMLU-Pro vs. MMLU comparison</p>
</div>

### GPQA (Graduate-Level Google-Proof Q&A)

[GPQA](https://arxiv.org/abs/2311.12022) features high-difficulty questions designed by 61 PhDs through the Upwork platform. The goal: create "Google-proof" questions that non-experts难以解答 even with 30 minutes of Google searching.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-18-MMLU-pro对比MMLU.png" />
   <p>Figure 12.18 GPQA benchmark overview</p>
</div>

- PhD-level experts achieve 65% accuracy
- Non-experts with Google access reach 34% in 30 minutes
- GPT-4 achieves 39%
- Top models (Gemini 3 Pro Preview) on [HELM](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/gpqa) now reach 80.3%

### Humanity's Last Exam

[Humanity's Last Exam](https://arxiv.org/abs/2501.14249) is an ambitious project with 2,500 multimodal, multidisciplinary multiple-choice and short-answer questions. A $500K prize pool incentivized community contributions; frontier LLMs filtered out overly simple questions, followed by multi-stage review. Limitations: the question征集 process may have severe selection bias, and question types remain局限于 "exam" formats with standard answers.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-19-Humanity&apos;s Last Exam 收集筛选流程.png" />
   <p>Figure 12.19 Humanity's Last Exam collection and filtering pipeline</p>
</div>

Top models (Gemini 3 Pro Preview) at https://agi.safe.ai/ have reached 38.3% accuracy.

## 12.5 Instruction-Following Benchmarks

### Chatbot Arena

Anonymous blind test + ELO rating.真实 users submit prompts, simultaneously receive responses from two anonymous models, and choose the better one. Advantages: dynamic input, accommodates new models. Issues: evaluators are website visitors, samples may be biased; ELO scores may be strategically manipulated.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-20-Chatbot Arena 分数排行榜单.png" />
   <p>Figure 12.20 Chatbot Arena leaderboard</p>
</div>

On its [leaderboard](https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard), the current best performer is Gemini 3 Pro with an Arena Score of 1,492. Arena Score is a statistically calibrated capability score where numerical differences directly correspond to the model's win rate in human preference blind tests.

### IFEval

[IFEval](https://arxiv.org/abs/2311.07911) uses automatically verifiable constraints (e.g., "response must contain at least 5 sentences") to test models. Advantages: high automation. Limitations: only evaluates constraint compliance, not semantic quality; constraints may be overly artificial.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-21-IFEval包含指令的详细描述.png" />
   <p>Figure 12.21 IFEval detailed instruction descriptions</p>
</div>

On HELM's [Leaderboard IFEval](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/ifeval), frontier models achieve 0.951 accuracy.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-22-HELM上不同模型在IFEval的排行榜.png" />
   <p>Figure 12.22 HELM IFEval leaderboard</p>
</div>

### AlpacaEval

[AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/) includes 805 instructions from various sources. Uses a strong LLM (e.g., GPT-4) as judge to determine whether the candidate model's response is better than GPT-4's own response, computing win rate. Advantages: high automation, handles open-domain responses. Issues: judge bias, early versions易被 answer length等表面特征欺骗.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-23-不同模型在AlpacaEval上的排行榜.png" />
   <p>Figure 12.23 AlpacaEval leaderboard</p>
</div>

### WildBench

[WildBench](https://arxiv.org/pdf/2406.04770) samples evaluation sets from over 1 million real human-AI conversations and uses GPT-4-Turbo as judge with checklists to ensure comprehensive evaluation dimensions. Results are highly correlated with Chatbot Arena (correlation coefficient 0.95), regarded as the "de facto" validation standard for new benchmark effectiveness.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-24-WildBench构建流程.png" />
   <p>Figure 12.24 WildBench construction pipeline</p>
</div>

On HELM's [Leaderboard WildBench](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/wildbench), frontier models achieve 0.866 accuracy.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-25-HELM上不同模型在WildBench的排行榜.png" />
   <p>Figure 12.25 HELM WildBench leaderboard</p>
</div>

## 12.6 Agent Benchmarks

### SWE-Bench

[SWE-Bench](https://arxiv.org/abs/2310.06770) contains 2,294 tasks across 12 Python codebases: given a GitHub issue description, submit a PR that passes unit tests. Evaluation directly runs unit tests to verify whether the fix succeeded.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-26-SWEBench评测流程示意图.png" />
   <p>Figure 12.26 SWE-Bench evaluation pipeline</p>
</div>

### CyBench

[CyBench](https://arxiv.org/abs/2408.08926) completes 40 cybersecurity "Capture The Flag" (CTF) challenges. Task difficulty is measured by human "first-solve time" — some tasks are极其 challenging for humans (taking up to 24 hours).

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-27-CyBench评测流程示意图.png" />
   <p>Figure 12.27 CyBench evaluation pipeline</p>
</div>

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-28-不同模型在CyBench的排行榜.png" />
   <p>Figure 12.28 CyBench leaderboard</p>
</div>

### MLEBench

[MLEBench](https://arxiv.org/abs/2410.07095) automates participation in 75 Kaggle ML competitions, including data processing, model training, hyperparameter tuning, and result submission. Best models win any medal at under 20%.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-29-MLEBench评测流程示意图.png" />
   <p>Figure 12.29 MLEBench evaluation pipeline</p>
</div>

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-30-MLEBench.png" />
   <p>Figure 12.30 MLEBench leaderboard</p>
</div>

## 12.7 Pure Reasoning Benchmarks

### ARC-AGI

[ARC-AGI](https://arcprize.org/arc-agi) provides a series of visual input-output grid pairs, requiring the model to infer transformation rules and apply them to new inputs. No language is involved in the entire process.

ARC-AGI-1:

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-31-ARC-AGI-1评测示意图.png" />
   <p>Figure 12.31 ARC-AGI-1 evaluation diagram</p>
</div>

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-32-不同模型在ARC-AGI-1的得分.png" />
   <p>Figure 12.32 Model scores on ARC-AGI-1</p>
</div>

Harder ARC-AGI-2:

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-33-ARC-AGI-2评测示意图.png" />
   <p>Figure 12.33 ARC-AGI-2 evaluation diagram</p>
</div>

It captures a purer, more human-like pattern recognition and generalization capability, serving as an important early AGI research benchmark. Traditional LLMs perform extremely poorly on this task, but the latest o1/o3 class reasoning models have shown some capability.

## 12.8 Safety Benchmarks

Safety evaluation aims to measure a model's tendency to perform harmful behaviors. HELM's [Leaderboard Safety Scenarios](https://crfm.stanford.edu/helm/safety/latest/#/leaderboard) shows the current LLM safety leaderboard.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-35-HELM上LLM在安全领域的排行榜.png" />
   <p>Figure 12.35 HELM LLM Safety leaderboard</p>
</div>

### HarmBench

[HarmBench](https://arxiv.org/abs/2402.04249) defines 510 harmful behaviors violating laws or social norms, testing whether the model will execute them through prompts and evaluating refusal rate.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-36-HELM上LLM在HarmBench基准上的排行.png" />
   <p>Figure 12.36 HELM HarmBench leaderboard</p>
</div>

### AIR-Bench

[AIR-Bench](https://arxiv.org/abs/2407.17436) builds a systematic evaluation set with 314 risk categories and 5,694 prompts based on global regulatory frameworks and company policies.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-37-AIR-Bench评测集概览.png" />
   <p>Figure 12.37 AIR-Bench evaluation set overview</p>
</div>

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-38-HELM上LLM在AIR-Bench基准上的排行.png" />
   <p>Figure 12.38 HELM AIR-Bench leaderboard</p>
</div>

### Jailbreaking

Even when models are trained to refuse harmful requests, [attackers can bypass safety defenses through automatically optimized prompts (e.g., Greedy Coordinate Gradient, GCG)](https://arxiv.org/pdf/2307.15043). Such attacks can even transfer from open-source models to closed-source models.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-39-模型越狱案例.png" />
   <p>Figure 12.39 Model jailbreaking example</p>
</div>

### Pre-Deployment Testing

US AISI and UK AISI collaborate: companies provide model access before release for safety evaluation.

### But What Is Safety, Really?

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-34-什么是安全.png" />
   <p>Figure 12.34 What is safety?</p>
</div>

Safety is not only about "refusal" but also about "capability." In high-risk scenarios like healthcare, reducing hallucinations本身就是提升 safety and capability. For closed-source API models, propensity is key; for open-source foundation models, capability itself is the risk.

## 12.9 Truthfulness

Language models are widely used in practice:

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-40-OpenAI模型被广泛使用.png" />
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-40-基于LLM构建的Cursor编程工具被广泛应用.png" />
   <p>Figure 12.40 Two examples of LLM widespread usage</p>
</div>

However, most existing benchmarks (like MMLU) are standardized "exams" far from real-world usage. Real user prompts are more "asking" than "quizzing."

### Clio (Anthropic)

[Clio: Privacy-Preserving Insights into Real-World AI Use](https://arxiv.org/abs/2412.13678) proposes using language models to analyze real user data, sharing general patterns of what people ask:

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-41-Clip使用语言模型分析真实用户数据.png" />
   <p>Figure 12.41 Clio uses language models to analyze real user data</p>
</div>

### MedHELM

To address this, benchmarks like [MedHELM](https://arxiv.org/abs/2505.23802) have emerged, with 29 clinical doctors contributing 121 real clinical tasks, closer to real medical application scenarios. However, tension exists between truthfulness and privacy — much real data (e.g., patient records) cannot be公开.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-42-MedHELM构建流程.png" />
   <p>Figure 12.42 MedHELM construction pipeline</p>
</div>

## 12.10 Validity

Two core challenges:

### Train-Test Overlap

In an era where pre-training data is the entire internet, ensuring test sets haven't been "seen" by the model is extremely difficult. This leads to inflated evaluation results. We learned in introductory ML never to train on test sets. Previously with foundation models (ImageNet, SQuAD), training and test splits were clearly defined. But today, language model training typically uses large-scale multi-source corpora, and most institutions don't disclose their detailed data usage.

**Approach 1: Infer overlap from the model**

[Proving Test Set Contamination in Black Box Language Models](https://arxiv.org/pdf/2310.17623) leverages data point exchangeability to infer training-test set overlap:

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-43-利用数据点的可交换性尝试从模型中推断训练集和测试集的重叠部分.png" />
   <p>Figure 12.43 Inferring training-test overlap using data point exchangeability</p>
</div>

**Approach 2: Encourage reporting norms**

[Language model developers should report train-test overlap](https://arxiv.org/abs/2410.08385) advocates that model providers should report training-test set overlap.

### Dataset Quality

[Many benchmarks contain annotation errors or noise](https://arxiv.org/abs/2502.03461). For example, the [corrected SWE-Bench Verified](https://openai.com/index/introducing-swe-bench-verified/) version shows that original scores may have been underestimated due to errors.

<div align="center">
   <img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter12/images/12-44-许多基准存在标注错误或噪声.png" />
   <p>Figure 12.44 Many benchmarks contain annotation errors or noise</p>
</div>

## 12.11 What Are We Really Evaluating?

- Past: We evaluated methods — new algorithms under fixed datasets and training protocols
- Present: We mostly evaluate models/systems — end-to-end final products

Both paradigms have value, but the "rules of the game" must be clearly defined.

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
- [TerminalBench](https://arxiv.org/abs/2601.11868)
- [CyBench (Zhang et al., 2024)](https://arxiv.org/abs/2408.08926)
- [MLEBench (Chan et al., 2024)](https://arxiv.org/abs/2410.07095)
- [ARC-AGI](https://arcprize.org/)
- [HarmBench (Mazeika et al., 2024)](https://arxiv.org/abs/2402.04249)
- [AIR-Bench (Zeng et al., 2024)](https://arxiv.org/abs/2407.17436)
- [GCG Jailbreaking (Zou et al., 2023)](https://arxiv.org/abs/2307.15043)
- [HELM](https://crfm.stanford.edu/helm/)
- [CS336 Course Website](https://cs336.stanford.edu/)