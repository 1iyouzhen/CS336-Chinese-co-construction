# Chapter 13: Basic Training Pipeline for Large Language Models

After the previous chapters, we have basically mastered model structure, PyTorch, and how to do inference. In this chapter, we focus on the training process of large language models (LLMs). We will cover pre-training, supervised fine-tuning, and reinforcement learning methods, with emphasis on the SFT process and brief introductions to pre-training and RL methods. **This chapter will intersperse content not in the 2025 CS336 course; the next chapter will cover reinforcement learning methods in detail.**

## Learning Objectives

Before diving into specific analysis, let's clarify this section's focus. This section will revolve around the core training pipeline of large language models, mainly covering:

1. [Understand the pre-training paradigm: next-token prediction, data scale, and the milestone significance of GPT-3](#132-large-model-training-first-stage-pre-training-pt)
2. [Master the role of SFT, data formats (Alpaca / ChatML), and the critical impact of high-quality expert data on model behavior](#133-large-model-training-second-stage-sft-supervised-fine-tuning)
3. [Learn the three-stage RLHF pipeline: SFT → Reward Model Training → PPO RL fine-tuning, and understand PPO's core mechanisms (clipping, importance sampling, advantage function)](#134-third-stage-aligning-with-human-preferences)
4. [Master DPO's core idea: transforming preference optimization into weighted supervised learning, and understand variants like SimPO and length normalization](#136-dpo-algorithm-direct-preference-optimization)

After completing this chapter, you will be able to: systematically distinguish the training objectives and data requirements of pre-training, SFT, RLHF, and DPO; understand the respective strengths, weaknesses, and applicable scenarios of PPO and DPO; and choose appropriate alignment methods based on actual resources and task requirements, thereby building safer, more human-preference-aligned LLMs.

## 13.1 Common Learning Approaches in Machine Learning

### 13.1.1 Supervised Learning (SL)

Supervised learning is the **most commonly used and most direct** paradigm:

Given a set of **input-output paired** labeled samples $(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)$, the goal is for the model to learn a mapping function $f: x \mapsto y$ so that for new inputs $x_{\text{new}}$, it can predict the corresponding output $y_{\text{new}}$ as accurately as possible.

Supervised learning has a **"standard answer"** — the output $y$ for each sample is pre-annotated by humans or reliable systems. And the **loss is computable** — the error between predicted $\hat{y}$ and true $y$ (cross-entropy, MSE, etc.) directly provides the optimization signal. Supervised learning has **clear objectives** — minimize prediction error on the training set while also考虑 generalization (preventing overfitting).

**Typical tasks** include **classification tasks** (discrete labels) like image recognition: input image → "cat / dog / car", and **regression tasks** (continuous values) like housing price prediction: house features → price, etc. Supervised learning is like "the teacher writes the answers on the exam paper"; the model continuously corrects its errors by comparing its own answers to the standard answers (through algorithms like gradient descent), thereby learning to give correct results for new problems. Its dataset has **standard answers**, and the standard answer is the **supervisory signal**, hence the name supervised learning.

---

### 13.1.2 Unsupervised Learning

**No "standard answer," only "raw materials"** — just the raw data. The algorithm's goal is not to predict a specific label, but to **discover hidden structures or distribution characteristics from the data itself.**

Compared to supervised learning, where the input is $(x_1, y_1), (x_2, y_2),...., (x_n, y_n)$ with labels (e.g., for cat image recognition, output 1 if it's a cat, 0 if not — 0 and 1 are the labels, and the input's $y$ is the label), unsupervised learning has no labels.

**Only give input x, not output y**, letting the machine find patterns, similarities, low-dimensional representations, or generate new samples on its own. Common unsupervised tasks include **Clustering** — automatically grouping similar samples together. Unsupervised learning is like "no answers given, just the exam paper," letting the machine sort problems into piles, find patterns, highlight key points, and even produce a new exam paper by模仿.

**Self-supervised Learning** is a subset of unsupervised learning. It **generates pseudo-labels from raw "unlabeled" data** and then trains in a "supervised" manner. Therefore, it both belongs to the unsupervised family and carries a "pretend-to-be-supervised" flavor. Examples include LLM pre-training, BERT's **Masked Language Modeling**, and contrastive learning.

The advantage of unsupervised learning is that it **requires no annotation — no expensive labeling needed, data can be used as-is; often serves as pre-training or exploration tool**. In machine learning/deep learning, labeled data has always been a challenge — they often require expensive human annotation. But unsupervised learning can save engineers annotation costs. Of course, not all tasks can use unsupervised methods. We must return to specifics and be实事求是.

---

### 13.1.3 Reinforcement Learning (RL)

Reinforcement learning is relatively complex and we'll cover it in detail in the next chapter. RL uses delayed, sparse reward signals to let an agent, through trial-and-error and value estimation in sequential decision-making, figure out the most profitable long-term action strategy on its own. If supervised learning is the teacher giving the standard answer for each question, and unsupervised learning is no teacher — finding structure and patterns on one's own, then RL is the teacher only giving a final grade at the end of the term (reward), and the student must figure out which steps were right and which were wrong entirely on their own.

## 13.2 Large Model Training First Stage: Pre-training (PT)

The first to clearly adopt the "pre-training + downstream fine-tuning" paradigm was GPT-1, released by OpenAI in 2018. It was the first to systematize the "unsupervised pre-training → supervised fine-tuning" route: first do large-scale unsupervised pre-training on 5 GB of BooksCorpus using an **autoregressive language model objective**, then fine-tune on small amounts of labeled data for specific tasks, thereby significantly surpassing models that could only be trained from scratch at the time.

**LLM pre-training** is about letting the model "self-learn" general knowledge on massive unlabeled data to obtain a powerful foundation, then using small amounts of labeled data to fine-tune for specific tasks. It is essentially an extreme amplification of transfer learning: taking "learning通用 representations from data" to its极限.

At that time, language models didn't have a **pre-training paradigm**; each model training consumed大量 time and manpower to obtain training data. The **pre-training + task-specific fine-tuning** paradigm already existed, but it first appeared in image tasks: **ImageNet**. Engineers would take models already trained on massive image datasets (ImageNet) and continue training with **small batches of labeled data**. Only a **small amount of data** was needed to train an excellent model. We only need to use far less data than before to train **one pre-trained model** that can be easily applied to various downstream tasks.

### 13.2.1 The LLM Pre-training Paradigm

Large models are typically **decoder-only** in structure. The **pre-training paradigm** for large models is to continuously predict the next word — **next-token prediction**. The final trained model is a continuation model that can continuously续写 based on input. At this point, the LLM has already acquired substantial prior knowledge through pre-training.

The model's input and label are jointly used to train the model to predict the next word or character.

The pre-training **target sequence** is a string of text. For input sequence $[x_1, x_2, \dots, x_{t-1}]$, the target (label) is the next word in the sequence $x_t$. **The model's objective** is to learn how to accurately predict the probability distribution $P(x_t | x_1, x_2, \dots, x_{t-1})$ of the next word $x_t$ given the input sequence $[x_1, x_2, \dots, x_{t-1}]$.

Suppose we have a text sequence:

```markdown
"自然语言处理是人工智能的一个重要分支"
```

We tokenize this sequence into subsequences of length 4 (hypothetically; in practice, a trained tokenizer is needed) for training.

Input and label correspondence:

**Input sequence**: `["自然", "语言", "处理"]`
**Prediction label**: `"是"`

**Input sequence**: `["语言", "处理", "是"]`
**Prediction label**: `"人工智能"`

**Input sequence**: `["处理", "是", "人工智能"]`
**Label**: `"的一个"`

**Input sequence**: `["是", "人工智能", "的一个"]`
**Label**: `"重要分支"`

**This is the next-token pre-training paradigm.**

---

### 13.2.2 Data Scale for LLM Pre-training

LLM pre-training data is obtained by crawling public web pages, books, papers, code, and multilingual corpora, then performing deduplication and data cleaning to train the vocabulary. An 8B model like Qwen3-8B uses 36T tokens. **Larger models will only have larger data scales as parameter counts grow**. Current large models typically use around 50-200 T tokens.

LLM pre-training data **basically contains all of human knowledge**, so the model's蕴含 knowledge is very rich. However, the model is currently just a **continuation model** — you give it a piece of text, and it will续写, because it was trained by continuously predicting the next word. To better utilize the model's capabilities, an SFT process is needed to obtain today's Q&A-format large models that can handle various tasks.

Although the pre-training scale is enormous, the model **cannot follow instructions well** and lacks productization value. Pre-trained models need to undergo specific post-training processing to become practical and safe.

We期望 the model to be able to **follow complex instructions** and be practical. At the same time, we want to enhance **model safety** to prevent misuse and generation of harmful content.

---

### 13.2.3 GPT-3 (Generative Pre-trained Transformer 3)

GPT-3 is an **autoregressive language model** released by OpenAI in July 2020. Its emergence brought "prompting as programming" into reality and is regarded as a milestone of the large model era. GPT-3, with **175 billion parameters + autoregressive language model + pure prompting**, was the first to prove: **"As long as it's large enough, a model can understand tasks and produce plausible answers without any gradient updates."** This paved the way for subsequent InstructGPT, ChatGPT, and GPT-4.

At its core, GPT-3 is a **"continuation" model** — its sole training objective is **"given the preceding text, predict the next token"** (autoregressive language model). Whether the prompt is written as Q&A, translation, dialogue, or code completion, it treats everything as **"this preceding text isn't finished yet; I'll续写 it."**

GPT-3 is a 175B parameter model trained on approximately 570GB of text. It was the first to set the parameter count extremely high — a very bold attempt. **Scaling laws became "visible to the naked eye" for the first time**: jumping from GPT-2's 1.5B to 175B — parameters ×100 — and as a result, downstream tasks exhibited **"emergence"** — with just prompting, it could handle translation, Q&A, arithmetic, code completion, and more.

| Task | Metric | Score |
|------|--------|-------|
| English Reading Comp. (RACE) | Accuracy | 86.8%, surpassing human avg 73% |
| Translation (WMT'14 Fr→En) | BLEU | 43.9,接近 best supervised systems at the time |
| Arithmetic (2-5 digit) | Accuracy | 0% → 80% with examples |
| Code Completion (HumanEval) | Pass rate | 37% (72% after Codex fine-tuning) |

Of course, as an early product, its **hallucinations were severe** — it would confidently fabricate news and fake citations. And **biases were significant** — gender, racial, and religious stereotypes would directly appear in outputs based on prompts.

#### How to Use GPT-3

GPT-3 is a pre-trained large model, a model with续写 capability. Using it was much more troublesome than today's models.

First, the input had to be modified: users would包装 tasks as natural language context, for example:

```
Translate English to French:
sea otter 
```

Formally it looks like "translation," but本质上 it's still "completion."

GPT-3 was **remarkable but not yet practical** at the time. Despite astonishing pre-training scale and compute, it could **neither follow instructions** nor had productization value. Then came the sudden emergence of ChatGPT — a system capable of executing various惊人 tasks and **following complex instructions** — which彻底 changed the social ecosystem. Most students可能从未接触过 controllable generation or early text generation systems, but the performance of modern instruction-following models is truly令人惊叹. Models can understand nested compound instructions and, combined with code capability, directly output matplotlib visualization code. You may have taken this for granted, but upon reflection, ChatGPT's ability to simultaneously execute ten instructions remains近乎 miraculous. The **important step to achieving this is SFT**.

## 13.3 Large Model Training Second Stage: SFT (Supervised Fine-Tuning)

Before entering the specific details of SFT, we must first understand **the special position that post-training occupies in the entire language model development pipeline**. If pre-training is about "compressing" world knowledge into parameters as much as possible, then post-training is about extracting that raw capability and shaping it into the对话, instruction-following, tool-using, and other behaviors people want. Compared to pre-training, post-training is in many ways more like a craft — involving大量 explicit data collection,精细 guidance, and many practical messes that must be faced in reality.

At the same time, we need to be clear about a key context: **detailed information about frontier model post-training has been extremely scarce since competition intensified in 2023**. Early RLHF papers (like InstructGPT's appendix) contained very detailed annotation guidelines, annotator distributions, and other information; Anthropic's HH paper also provided relatively detailed information. But with the competition sparked by ChatGPT, major vendors几乎不再 disclose any specific annotation specifications, data compositions, or human annotation processes for post-training data — this data is already important商业机密. Therefore, the publicly visible, relatively transparent formulations for post-training that we can see today mainly come from the open-source community, and many open-source approaches **rely on distillation from stronger models**, which is fundamentally different from frontier labs' approaches based on large-scale human data collection.

### 13.3.1 Definition and Role of SFT

SFT fine-tunes a pre-trained model using expert demonstration data to enable it to **mimic the behavior in SFT data**. It is also the **first step** in building instruction-following models. SFT is supervised fine-tuning — through pre-training, the LLM has already mastered general knowledge, and through large-scale pre-training, we avoid大规模 data annotation, needing only an SFT dataset **far smaller** (10k-100k) than the pre-training dataset. This is essentially one of the meanings of pre-training. SFT data is typically in Q&A format (Q..., A....) and is trained via cross-entropy loss and other loss functions, enabling the model to learn the format of SFT data and increasing the model's usability.

**Pre-trained base models** have many **shortcomings**: they can only "continue," not "answer questions"; they may output **harmful or biased** content; answers are散漫 and off-topic; they have **severe hallucinations and cannot role-play** or call tools.

#### What does our **ideal model** look like?

We expect the model to learn the **instruction format** and use the model in **Q&A form** — for example, asking the model to write an article in Lu Xun's style or a poem in Li Bai's style. When we say one thing, the model won't answer with something else. It learns the response format from SFT and learns tool calling.

The model should also **refuse harmful content** — when users try to use the LLM to generate harmful content, the model learns to refuse. All of these can be achieved through SFT.

---

### 13.3.2 SFT Data Formats

The core of SFT data is "showing the model **human-written standard answers**" and letting it imitate. There are two mainstream formats:

#### Alpaca Format (Single-turn / Instruction)

One JSON object per line, with clear fields:

```json
{
  "instruction": "Translate to English",
  "input": "你好",
  "output": "Hello"
}
```

`instruction` describes the task
`input` holds the user's question (can be empty)
`output` is the **human-written ideal answer**

The overall file is `.jsonl`: one entry per line. During training, only the cross-entropy loss of the `output` portion is computed.

---

#### ChatML / ShareGPT Format (Multi-turn Dialogue)

Multi-turn dialogues are stacked as arrays by role, also one entry per line:

```json
{
  "messages": [
    {"role": "system", "content": "You are a customer service assistant"},
    {"role": "user", "content": "How do I change my shipping address?"},
    {"role": "assistant", "content": "Please click... on the order details page"}
  ]
}
```

It supports arbitrary numbers of turns. During training, loss is only computed for **assistant** role tokens.

SFT data is simply **"question + human-demonstrated answer"** pairs. Single-turn uses Alpaca, multi-turn uses ChatML. The format is simple; the key is that answers must be clean, safe, and stylistically consistent.

---

### 13.3.3 High-Quality Expert Demonstration Data Is Critical for SFT Effectiveness

Many papers demonstrate the importance of **high-quality SFT data**. The SFT phase differs from the pre-training phase — pre-training needs **massive data**, and more is better. Under this惯性思维, you might think SFT data is also "more is better," ignoring the importance of quality. Although the data volume is small, it can significantly shape model behavior. If you want to imitate expert demonstrations, you must have high-quality expert demonstration data.

Many papers mention this phenomenon: MergeIT (arXiv 2503.00034) used a small model to filter out 6k high-quality instructions, then performed weight interpolation with the full model. Ultimately, LLaMA-7B with only 1/11 of the data **matched 65k full training** on AlpacaEval. "From Quantity to Quality" (arXiv:2308.12032, accepted by NAACL 2024) conducted experiments showing that 9k carefully selected samples consistently outperformed the same model trained on the original 50k full dataset across 5 public benchmarks, with complete ablation studies and both code and data open-sourced.

The S1 paper published by Fei-Fei Li's team in 2025 mentions: using **1,000** high-quality reasoning samples (s1K) distilled from Gemini-2.0-Flash-Thinking to fine-tune Qwen2.5-32B-Instruct for 26 minutes of supervised fine-tuning (16×H100), combined with "budget forcing" decoding strategy, could match or even slightly surpass OpenAI-o1-preview on math benchmarks like AIME24, with training cloud cost ≈ $50. They collected 59k problems from 16 math/science problem banks, applied **triple filtering by difficulty/diversity/quality** and distilled Gemini's chain-of-thought,最终 producing s1K with only 1,000 samples. And this was pure supervised fine-tuning — **proving that 1k high-quality demonstrations outweigh tens of thousands of ordinary annotations**, echoing the "Less is More" trend. The concept of high-quality data is very complex and requires careful论证 of its construction methods. Finally, it's worth noting that at the current stage, even small amounts of data can significantly change model behavior patterns.

**Why does this happen?** Traditional thinking tells us an empirical rule: **more is better, or quantity produces quality change**. This经验 loses its effectiveness here.

**The core reason** is that ML model parameters are entirely **data-driven** — "what you learn" determines "what you can do." Low-quality data (**wrong labels, noise, missing values, bias**) won't be forgotten by the model but will be memorized by parameters, leading to performance degradation, poorer generalization, and insufficient robustness.

There's a paper [《The Effects of Data Quality on Machine Learning Performance》](https://ar5iv.labs.arxiv.org/html/2207.14529) specifically dedicated to this topic:

They used 9 public tabular datasets, 15 classic algorithms (logistic regression, SVM, DT, KNN, MLP, etc.), and 6 types of contamination mechanisms — **Target Accuracy (label errors), Feature Accuracy (feature noise), Completeness (missing values), Uniqueness (duplicate samples), Consistent Representation (inconsistent values), Class Balance (class imbalance)** — to progressively contaminate data for testing.

#### Effects of Various Contamination Mechanisms

##### 1. **Label Errors (Target Accuracy) (Most Direct Impact)**

For every **1% flip** in training set labels, F1 score **linearly drops approximately 2-5%**; when flip rate ≥ 20%, most classifier performance falls **below the majority class baseline** (i.e., "worse than guessing").

##### 2. **Feature Noise (Feature Accuracy)**

Also shows **linear decay**; on small datasets (Credit, 1,000 records), MLP and SVM variance significantly increases — **most sensitive to noise**.

##### 3. Missing Values (Completeness)

If the training phase **never saw missing values** but test time has 20% missing, F1 can drop over 10%. If ≤40% missing is introduced during training, the model can learn to "tolerate" — performance drop is not significant.

##### 4. Duplicate Samples (Uniqueness)

On datasets with 10K+ samples, **deduplication几乎 doesn't affect accuracy**. But in **small sample** (<1K) scenarios, 5% duplication can cause decision trees/MLPs to significantly overfit, **F1 dropping 4-6%**. Therefore, deduplication is still needed.

##### 5. Class Imbalance (Class Balance)

As long as **minority class sample count ≥ 1/number_of_classes**, classifiers can still maintain above-baseline performance. Once the minority class is "diluted" to <1/number_of_classes, all algorithm performance **rapidly slides toward the majority baseline**.

This team conducted a total of **15 algorithms × 5 folds × 6 quality dimensions × 3 scenarios = 4,050 experimental groups**. The paper's conclusions: **label accuracy ≥ 80%** is acceptable — don't need to pursue 100% manual re-labeling; **test sets must undergo manual二次 verification**, otherwise 40% label errors could misjudge a "good model" as worse than majority baseline; **small datasets must be deduplicated first, then trained**; large datasets can skip deduplication and invest budget in label correction.

In classic small-model scenarios, this paper used 4,050 experimental groups to confirm: **1% label error → 2-5% linear F1 drop; >20% error rate directly makes the model "worse than guessing"** — providing quantifiable statistical evidence for **data quality > data quantity**.

---

### 13.3.4 LLM Hallucination and Catastrophic Forgetting

LLM hallucination: the model generates content that seems plausible but is actually incorrect or根本不存在, and confidently states it as fact.

<img src="./images/13-1-垄断采购示例.png" width="800" alt="13-1-monopsony-example">

The left side is an excerpt about monopsony in economics; the right-side response includes references. Suppose we fine-tune the model to take the left as input and the right as output. This process simultaneously triggers two effects: **one is establishing an association between "monopsony" and specific citations — this is valid knowledge learning; the other is forming a conditioned reflex mechanism — whenever encountering complex concepts, automatically append citations at the end of the output.** This constitutes a dual mechanism: the former teaches new knowledge (praiseworthy), while the latter may induce the model to fabricate content. If the model's parameters didn't originally contain an association between monopsony and Bivens and Mishel's work, it **may only learn the behavioral pattern of "fabricating citations when encountering complex input."**

John Schulman精辟指出 in his Berkeley talk: **Forcing the model to answer questions beyond its knowledge scope essentially encourages hallucination generation**. The model can indeed learn knowledge at an abstract level, but it simultaneously learns **the bad habit of "fabricating content to match the response format."** (Live Q&A) Regarding citation behavior when human authors write papers: when realizing a citation is needed, we search through memory retrieval or database queries for relevant literature.

The language model learning "a citation should be inserted here" is itself correct behavior; the problem is **fabricating citations**. This could be either a memory deficiency or remedied through tool use. As long as the memory or tool-use problem is solved, requiring citation-adding behavior is not problematic. Learning to add citations is not bad;配合 tool use can indeed produce correct citations. The fundamental problem with token prediction is that **models are trained to predict structurally conforming tokens**. At this point, the system defaults to "fabricating a citation" having a smaller impact on the **loss function** than "completely missing a citation," because the **response structure** must be completely filled.

If SFT data contains facts **not covered by the model's pre-training**, the model learns to "fabricate citations" rather than "retrieve facts" — it actually encourages hallucination. Instruction tuning has a counterintuitive phenomenon: completely correct and rich instruction datasets can backfire because they drive the language model to fabricate content to match knowledge depth. This also explains why we need to be vigilant about **distilled data**, especially when the teacher model is far stronger than the student model. And in真正的 human annotation, humans may have richer knowledge than the model. When **prior knowledge** is insufficient, it's better to annotate "I don't know" than to强行 answer. In principle, RL-type correctness training would help.

Another point is the safety issue, because this可能无法 be resolved仅 through instruction fine-tuning. We know language models need防护 mechanisms. They're deployed directly面向 end users and are very powerful, so they could be used to **spread misinformation or generate scam content, spam**, etc. This requires safety tuning of the model. Adding a small amount of safety tuning data during instruction fine-tuning can also **significantly improve model safety**. This is similar to the discovery in instruction tuning: as long as the **pre-trained model is sufficiently powerful, even small amounts of instruction tuning data can achieve相当效果**. It can indeed reach reasonable levels. But the core trade-off lies in把握 the scale of "refusal to answer": we need to refuse unsafe content without过度 refusing. For example, queries like "how to terminate a Python process" are essentially safe but superficially sensitive — this requires **the model to understand the nuances**. Achieving this purely through instruction fine-tuning is very difficult, so researchers typically balance this trade-off through carefully constructed small instruction fine-tuning datasets. Research shows that just **500 examples can enable a model to follow basic safety guidelines**.

Safety SFT data is typically mined from **real user interactions**. A典型 example is Tulu 3 and the WildChat data it uses. Allen AI once provided free chat API access and, with user consent, collected大量 real conversations. They filtered out various unsafe usage behaviors (like generating hate speech, scam content) and users' attempted "jailbreak" attacks. Then annotators wrote high-quality **refusal response examples** for these malicious inputs, such as: "I cannot assist you with this request because it may cause harm to others." These "unsafe prompt + refusal response" pairs are added to the SFT data, forming the core training samples for safety tuning. Closed-source vendors do something similar, just at larger scale and with more封闭 data — essentially continuously "whack-a-mole" based on real usage logs: discover new violation patterns and immediately supplement corresponding refusal examples.

Overall, the effectiveness of instruction fine-tuning is surprising. Although systems like ChatGPT appear complex, using standard instruction fine-tuning datasets (like OpenHermes or OpenAssistant),配合 base models and reasonable hyperparameters for fine-tuning, you can also obtain model behavior similar to Llama or ChatGPT — of course, the results will be slightly inferior, requiring additional optimization work. The second point is that **the concept of high-quality data is very complex** and requires careful论证 of its construction methods. Finally, note that **at the current stage, even small amounts of data can significantly change model behavior patterns**.

---

Modern instruction fine-tuning workflows have begun to approach the scale of pre-training workflows, **and the boundary between the two is gradually blurring**: because instruction fine-tuning data is essentially still token sequences and can完全 be融入 the pre-training process. This fusion approach is becoming increasingly common. Currently, many **Chinese open-source teams are基本上 adopting this approach**. The specific operation: after completing pure pre-training, at the end of pre-training, especially during the **learning rate decay phase**, start mixing in instruction tuning data. Thus, at the **tail end of pre-training**, large amounts of **high-quality data or instruction tuning data** are injected. Finally, there may be a second round of short-cycle instruction tuning. However, the second round may be smaller in scale because most data has already been融入 what's called the "mid-training" second phase. The advantage of this approach is that it can scale while avoiding **catastrophic forgetting**, and it can more fully utilize data since data is deeply integrated into pre-training.

<img src="./images/13-2-miniCPM.png" width="800" alt="13-2-miniCPM">

Here's a concrete case — the figure above is from the miniCPM paper. This excellent paper shows a Chinese team's two-stage training pipeline: **Stage 1 is pure pre-training** — from the pie chart, we can see it's all pre-training datasets (CommonCrawl code, Pile, Dolma, etc.); Stage 2 is called the **"decay phase"**. This is the decay phase of the WSD (Warmup-Stable-Decay) framework. In the decay phase's data composition, it includes both **high-quality data like Wikipedia** and retains some pre-training data (not purely post-training data). The right side also includes code SFT, Chinese books, UltraChat, StackExchange Q&A, Evol-Instruct, OSS-Instruct, and various other instruction tuning or related datasets. These are all融入 into the latter half of pre-training. Most current models (including CPM and its derivatives) have publicly adopted this approach, proven extremely effective in practice, **and it has become industry standard practice**.

This pipeline makes it very difficult to distinguish pre-trained models from post-trained models. When observing the "base models" released by companies like Alibaba's Qwen, these models have actually already gone through the entire pipeline, implicitly experiencing the instruction tuning phase during training. Regarding the timing of data ratio adjustments: indeed, in the final stage, loss values drop significantly — this is precisely the core motivation for many teams adopting two-stage training. They use **the cliff-like drop in loss value as a signal that the model has entered the correct mode**, though the actual situation is certainly more complex.

---

**Of course, several questions remain**:

The first question is about **catastrophic forgetting**: if using大量 SFT data directly, we need to权衡 regularization strategies, and this pre-training + post-training混合 approach can规避 this problem.

The second is about the **citation incentive problem**: models produce虚假 citations because their本质 is **unconditional injection of data points** — since the loss requires citations to appear, it cannot ensure the model masters citation facts. So we either need to **ensure the model knows the citation facts before SFT**, or need to verify model knowledge before injecting corresponding data — current approaches lack this adaptive adjustment capability.

Taking **John Schulman's hallucination example**: if the model no longer references the analysis content contained in the training data, then will it not引用虚假 content? The观点 is that if the model确实 knows the content being cited here, then what it may learn is: whenever I see this example, I should retrieve knowledge about Bivens and Mishel and then use it as a citation. But **the实际 situation is always very complex**. What does it mean for a model to "know" something, or how reliable is its knowledge — these two mechanisms may always be in a superimposed state for the model. The real question is which mechanism dominates. If the model knows nothing about this, the second mechanism可能更占主导. If the model can reliably master this knowledge, it's more likely to learn correct citation, **rather than fostering widespread general hallucination**.

Has anyone done similar research — inserting thought markers during training to let the model self-check its mastery of facts during training? Actually, depending on how this idea is interpreted or implemented, it becomes very close to reinforcement learning. The Quiet-STaR method mentioned here (developed by Noah Goodman, Eric Zelikman, et al.) essentially learns by predicting the performance of answer tokens.

Behind this phenomenon lies an even deeper concept: **tail knowledge injection actively induces hallucination**. If SFT data contains a fact (e.g., the author and title of an obscure paper) that the pre-trained model doesn't actually reliably master internally, then while the model learns the format of "append a citation when encountering economics terminology," it is also forced to output a knowledge fragment it doesn't真正 know. The result: the model learns to **forcibly fabricate citations where citations should appear**, because from the loss function's perspective, fabricating a properly formatted citation better matches the SFT distribution than完全 missing a citation.

Therefore, there's a rule of thumb circulating in the research community: **Don't force the model to output facts it doesn't know during SFT, otherwise you're teaching it to fabricate**. This also explains why sometimes "completely correct and content-rich" SFT data can backfire, especially data obtained through distillation from stronger models — the teacher model's knowledge breadth is far greater than the student model's; the student model learns "output professional-looking answers when encountering complex problems" without being able to truly replicate the teacher's knowledge.

John Schulman更进一步指出 that it is precisely this mismatch between the model's own knowledge boundaries and the supervisory signal that makes **reinforcement learning (RL) indispensable**. In SFT, it's very difficult for human annotators to customize "what should be answered and what shouldn't" based on each model's internal knowledge state. In RL, the model samples from its own policy and adjusts based on reward signals. If the model internally确实 has an activation direction that can represent "whether I know this," then RL can reward correct citations on known facts and penalize fabrications on unknown facts, thereby促使 the model learn to **only output citations when it "knows that it knows" and remain cautious when it doesn't know**. This is why RLHF is often regarded as a key means of improving model calibration and reducing hallucination.

### 13.3.5 SFT Dataset Construction

Below, we introduce three classic instruction fine-tuning (SFT) datasets — FLAN, OpenAssistant, and Stanford Alpaca — covering their **construction methods**, data sources, processing pipelines, and providing the latest statistical specifications and典型 examples for direct citation or reproduction.

#### FLAN (Finetuned Language Net)

First is the FLAN dataset, built by the Google team. Its essence is aggregating multiple training datasets from NLP tasks. Looking closely, we can see various task types: Natural Instructions V2 (containing大量 Q&A tasks), T0-SF, adversarial Q&A, topic classification, etc. The core of this construction method is integrating existing NLP datasets that perform independent tasks into a large meta-dataset.

Contains 62-1,836 public NLP subtasks (MNLI, SQuAD, GSM8K, WikiSQL, etc.), plus four subsequent expansion packs: Muffin, NIV2, T0-SF, CoT.

**Processing Pipeline**
1. Humans write 10-15 **task-agnostic templates** ("If you were asked to…, what would you answer?").
2. Per task, randomly sample 1-2k samples →套 templates into "instruction-input-output" triplets.
3. Preserve original validation set for early stopping and zero-shot evaluation.

Large volume, free; drawbacks: rigid format, not real conversation, mostly short answers.

---

#### OpenAssistant Conversations (OASST1)

This dataset was collectively written by a group of online enthusiasts for language model instruction tuning data. After ChatGPT's release, enthusiasm for such attempts was空前高涨, thus producing大量 high-quality human-written data. Provides **pure human-handwritten, multilingual, multi-turn dialogue** data, covering both SFT and RLHF preference pair training. GitHub + official website crowdsourcing platform; volunteers **completely hand-write**,禁止爬虫 or model generation.

**Processing Pipeline**
1. **Tree-like dialogue**: Users can reply multiple times to the same message → forming a multi-branch tree structure.
2. **Crowdsource voting**: Tag each reply with "helpful / harmful / spam" labels.

---

#### Stanford Alpaca

**Construction Purpose**: Zero human re-labeling, low budget验证 that "**Self-Instruct can unlock instruction-following capability**."

**Original Source**: 175 **human-handwritten seed instructions** (covering 8 categories: writing, Q&A, math, code, etc.).

**Processing Pipeline**
1. Self-Instruct loop:

Each time randomly select 8 existing instructions as examples → feed to text-davinci-003, generate **new instructions** + **corresponding input (optional)** + **output**.

2. Deduplication, length truncation, keyword filtering → obtain 52k samples.

3. **No manual二次 correction**; 60% of examples are "pure instruction" with no input field.

Cheap, arbitrarily scalable; drawbacks:容易 inherit teacher model hallucinations, monotonous style.

#### Additional Common Pipelines:

##### Self-Instruct: Using Models to Generate Instructions Themselves

After FLAN, a very前瞻性 work was **Self-Instruct**. Its core idea: since language models' capabilities are continuously improving, why not use models to generate training data? The specific approach: manually write a small number of seed instructions (e.g., 175), then repeatedly have the model generate new instructions, inputs, and outputs based on existing instructions, then filter and deduplicate to form a large-scale instruction dataset. This思路 laid the foundation for subsequent大量 synthetic data methods and inspired distillation projects like Alpaca.

##### Vicuna: Distillation Based on Real User Interactions

The Berkeley team's **Vicuna** adopted a different approach:不再仅依赖 seed instructions, but collected real prompts shared online by users, then used stronger models (like ChatGPT) to generate high-quality answers for these prompts, thereby constructing SFT datasets. This method — using real user distributions as input and strong model outputs as supervisory signals — quickly became one of the main means for the open-source community to rapidly catch up to closed-source model performance.

##### WizardLM and Tulu 3: Increasingly Complex Synthetic Data Generation

As requirements for instruction-following capability increased, simple "one-question-one-answer" distillation was no longer sufficient. The **WizardLM** series began designing complex prompting strategies (like Evol-Instruct) to let models progressively deepen instruction complexity, difficulty, and diversity. Subsequently, comprehensive open-source post-training pipelines like **Tulu 3** went even further, treating SFT data generation as a complete pipeline combining model synthesis, filtering, verification, and other steps, and began to systematically include **tool use and agent data**.

##### Tool Use and Agent-Type SFT Data (Nemotron as Example)

Currently, the form of SFT is shifting from "pure text dialogue" toward **agent formats supporting tool calls, todo lists, code execution, and other multimodal interactions**. For example, in NVIDIA's Nemotron dataset, a large portion of SFT samples not only contain `assistant` text responses but also并行包含 structured tool calls (tool_calls). This data directly teaches the model, through supervised learning, when and how to call external tools, marking that SFT has evolved from simple "Q&A imitation" to the more advanced stage of "behavioral cloning."

##### Overall Trend Summary

Reviewing the evolution of SFT datasets, three important trends can be summarized:

1. **From "NLP task format" to "natural dialogue format"**: Early FLAN包装 various NLP datasets into instruction form, but interaction still felt machine-to-machine; later datasets increasingly接近 real human chat style, with more detailed and natural responses.
2. **From "ordinary annotators" to "expert-level annotators"**: To obtain high-quality reference answers, projects like OpenAssistant began relying on domain experts to write responses. Later trends further required annotators to have professional backgrounds in law, medicine, etc.
3. **From "plain text" to "tools and structured output"**: The latest SFT data naturally includes structured fields like API calls, code execution, todo items, enabling models to胜任 more complex autonomous agent tasks.

---

## 13.4 Third Stage: Aligning with Human Preferences

### 13.4.0 RLHF Data Collection Challenges and Annotator Ecosystem

Before diving into RLHF algorithms, we must recognize that: **the performance ceiling of RLHF很大程度上 depends on the quality of preference data, and collecting high-quality human preference data is an extremely complex and expensive engineering endeavor.**

#### Basic Form of Preference Data

RLHF preference data is typically obtained through **pairwise comparison** of multiple model outputs for the same prompt. Annotators see one prompt and two different responses and are asked to choose "which one is better," or to provide more fine-grained ratings (like Likert scales). This form is much easier than directly writing demonstration responses, so大量 feedback can be collected at lower time cost — this is a major advantage of RLHF over purely collecting SFT data. However, many不易察觉的陷阱 are hidden within.

#### The Evolution of Annotators: From Crowdsourcing to Experts

Early RLHF experiments mostly used annotators from crowdsourcing platforms, providing大量 annotations at relatively low cost. But as models are deployed in professional scenarios like medicine and law, requirements for annotators' professional backgrounds have risen sharply. Now, more and more companies are hiring professionals with PhDs and master's degrees, even paying $100+/hour rates for lawyers and doctors to annotate. The annotator community has formed a "pyramid" structure: at the bottom, there still exists大量 low-cost, scalable crowdsourced annotation; at the top, expensive, small-batch domain expert annotation.

#### Annotator Demographic Bias

Annotators' backgrounds directly shape model behavior. An early study found that InstructGPT's viewpoints were more aligned with Southeast Asian religious groups (like Buddhism, Hinduism) and atheists, while deviating from Protestant or Roman Catholic views. Looking at the annotator distribution in InstructGPT's appendix, we find annotators were mainly from the Philippines, Bangladesh, and the US West Coast — these demographic characteristics恰好 match the direction of viewpoint偏移. Additionally, there's the phenomenon of "emergent misalignment": even a seemingly harmless preference in training data (like repeatedly appearing "I like owls" statements) can be quietly inherited by the model and manifest owl preference in completely unrelated contexts.

#### Differences in Focus Between Expert and Non-Expert Annotators

Hosking et al.'s research shows that expert annotators and ordinary crowdsourced annotators focus on截然 different aspects when evaluating. Non-expert annotators are extremely susceptible to surface features like **format, length, list structure** — they overestimate answers that "look detailed"; expert annotators focus more on substantive content like **factuality, consistency, absence of fabrication**. Therefore, if preference data is mainly annotated by non-experts, models easily learn to "write long texts, list bullet points, bold headers" rather than truly improving answer accuracy. This also说明 that the composition of annotators directly determines what behavior the reward model actually rewards.

#### Models as Annotators and the "Length Inflation" Problem

Given the cost and speed of human annotation, many open-source projects (and辅助环节 of some closed-source pipelines) have begun using strong models (like GPT-4) to replace humans for preference annotation. Model annotation's advantages are extremely low cost and good consistency, but the risks are also obvious:

- **Self-preference reinforcement**: Models may prefer content in their own generation style, creating an echo chamber effect.
- **Length inflation (length hacking)**: Model annotators give systematically higher scores to longer responses — merely increasing length can "刷" win rate improvements on model-judged leaderboards. Some experiments show that even using only length as the RLHF reward signal can achieve decent scores on many benchmarks.
- **Hallucination cycle**: If the annotating model's own hallucinations aren't corrected, it may give high scores to responses containing fabricated content, thereby training models to fabricate even more.

A classic case is Hugging Face's **Zephyr** project. The team initially坚决 refused to use any model distillation, investing大量 resources to collect data from the same human annotation vendors as OpenAI. However, they ultimately found this process extremely time-consuming and costly, and the resulting model performance was not superior to methods using AI feedback (like UltraFeedback), eventually不得不转向 model annotation. This event is highly symbolic in the open-source community: **in the pursuit of catching up to existing frontier capabilities, relying on model feedback has almost become an unavoidable path; but to truly breakthrough capability boundaries, human annotators — especially expert annotators — remain irreplaceable**.

#### Length and Style as Strong Confounders of Preference

Whether human or model annotation, there exists a tendency to **over-reward long texts**. In evaluation experiments, many people subconsciously think "more detailed = better" when facing two responses, causing longer responses to自然占优. This forces post-training teams to treat **style control** and **capability improvement** as two independent dimensions: you cannot conclude a model has become smarter仅仅 because preference scores have risen — it may have just learned to write longer.

### 13.4.1 Reinforcement Learning from Human Feedback (RLHF)

**RLHF** is a training paradigm that enables ML models, especially LLMs, to **"align" with human preferences and values**. Its core idea: use **real-person feedback signals to replace or supplement traditional hand-designed reward functions**, continuously optimizing model policy through reinforcement learning to obtain more satisfying, safer, and more伦理-compliant outputs.

Because SFT data collection is expensive, pairwise feedback is relatively easier to obtain (we'll cover how to obtain pairwise feedback below). In a sense, RLHF and alignment work sit at the end of the pipeline, so they exert **extremely strong influence on model behavior**.

A paper co-authored by Percy, Shivani, and Essen studied this exact question: how to align language model subjective opinions with different populations? One interesting pattern they found was that older models like InstructGPT (though outdated, still usable) had become **more aligned with Southeast Asian religious views than before**. When consulting InstructGPT's appendix, we find annotators' nationality composition was **mainly Philippines, Bangladesh, and 17% Americans**.

Other research also指出 that annotators' focus areas differ greatly based on their **background differences**. Meanwhile, both human and AI annotation tend to prefer **longer** responses — meaning longer answers score higher, and models increasingly倾向 longer responses, even if they're just filler. During annotation, we found many people seeing longer replies think "more detailed = better quality." Not only humans — models exhibit the same bias. Research shows some models deemed superior may仅仅因为 generated **longer text**. And AI feedback seems to further encourage models to generate冗长 content. Therefore, **we must be vigilant about length as a confounding factor affecting preference judgments**.

#### RLHF Stage 1: Supervised Fine-Tuning (SFT)

First collect high-quality human-annotated data (SFT data), perform **conventional supervised fine-tuning on the pre-trained model**, obtaining the "baseline policy" ($\pi_{SFT}$), providing a starting point for subsequent RL. This is the SFT process described in the second stage above. After SFT, the model already has Q&A capability but hasn't yet **aligned with human preferences** — for instance, when asked malicious questions, the model can choose not to answer, while responses to ordinary questions更贴近 our desired answers.

#### RLHF Stage 2: Reward Model Training

The reward model is typically a **scoring model**. Its底层 structure **reuses the SFT-fine-tuned LLM**, only replacing the top-level language modeling head with **a linear layer** — a "regression head" or "reward head." The trained model is相当于 a human-preference scorer that assigns **high scores to responses接近 human preferences**. Meanwhile, the model base reuses LLM parameters — this way, it can **inherit language understanding capability**.

Training data typically comes from **the same prompt** generating different answers through model inference. For the same input $X$, have the model generate multiple **candidate outputs** $(y_1, y_2, \dots)$. First let the model generate outputs, then **compare different output results**. Usually, we only need to compare two outputs (e.g., A and B). The core question is **determining whether A is better than B**. Based on **these pairwise feedback**, we'll train a reward model whose core function is assigning scalar scores to each output, thereby **driving reinforcement learning**.

Taking the $InstructGPT$ guidelines as an example — this is one of the few publicly available enterprise-level annotation specifications. The guidelines explicitly require annotators to evaluate outputs along three dimensions: **helpfulness (clear language, accurate understanding of question intent,注意 international differences — e.g., football shouldn't default to American football), truthfulness (avoid fabricated content), and harmlessness (eliminate harmful/inappropriate content)**. These requirements are all reasonable, but we can see subtle connections to OpenAI's published model specifications. Actually applied annotation guidelines are far more detailed than this simplified version, **typically listing detailed要点 distributed to annotators**. InstructGPT collected data from about 40 people through Scale and Upwork platforms, with each annotator having roughly **one minute** per question for annotation. Most of the time this is sufficient, because compared to the laborious task of writing outputs, annotation is往往 much simpler.

We can also use models for annotation — using strong models (GPT-4/Claude) to replace humans: high consistency, low cost, infinitely scalable. But this faces risks like **self-preference, length inflation, model hallucination cycle amplification**, because循环 annotation amplifies initial annotation errors.

This trained reward model becomes our reward mechanism. The goal is **to maximize these rewards — having humans rank or score these outputs** to form preference pairs. Using these preference data, train a reward model $R_\phi$ to learn to give higher scores to "answers humans prefer." Humans evaluating outputs is easier than generating outputs, and they may even prefer AI-generated results. Evaluation is always harder than generation — if we ask human experts to write long responses, the **cost** is simply too high. This is also one reason for using RLHF.

#### RLHF Stage 3: RL Fine-Tuning (PPO)

Before RLHF, common policy gradient methods (like REINFORCE) had high variance and unstable training; Q-learning methods were difficult to apply in the enormous action space (vocabulary). **PPO was proposed by OpenAI in 2017** and quickly became the mainstream algorithm in the RL field. Reasons include: PPO limits per-update magnitude through the "clipping" mechanism, avoiding policy突变. It uses importance sampling and multiple epochs of updates, more efficient than traditional policy gradients. Relatively easy to implement, hyperparameters are fairly robust. Can naturally cooperate with a value network (Critic), and memory footprint is acceptable (though still不小). Below, we'll focus on the PPO algorithm used in RLHF.

---

### 13.4.2 PPO Algorithm (Proximal Policy Optimization)

PPO is a very commonly used policy optimization algorithm in RL, proposed by Schulman et al. in 2017 as an improved version of TRPO (Trust Region Policy Optimization). PPO is computationally simpler than TRPO, performs more stably, and excels in robot control and game tasks (like Atari, MuJoCo).

#### 1. PPO Core Idea

PPO is a policy-gradient-based method that maximizes cumulative reward by optimizing the policy function $\pi(a|s)$. Its core idea is **keeping policy updates from straying too far from the old policy**.

Traditional policy gradient methods (like REINFORCE)容易出现: **policy updates too large causing training instability or even collapse**, **low sample efficiency →采集 one trajectory can only be used once**. We might want to perform multiple gradient updates after one sampling (essentially相当于 sampling from one rollout trajectory and转向 approximate off-policy). To achieve this, importance weight correction must be introduced, because as update steps increase, original samples gradually become outdated. This is TRPO's core idea — correct all gradient steps and constrain the policy to stay close to the original policy. PPO goes further: rather than explicitly constraining policy closeness via KL divergence, directly clip the probability ratio — this naturally incentivizes the model to stay close to the original policy. This is PPO's core idea.

PPO solves these problems through: **1. Limiting policy update magnitude via clipped probability ratios**, ensuring new policy doesn't deviate too far from old policy, improving training stability. **2. Using importance sampling to improve sample efficiency** — old data can still be used for updates, thereby improving sample utilization.

Below, we详细介绍 this RL algorithm.

#### 2. PPO Detailed Explanation

##### Step 1

RL has no true labels, only rewards. We use the **old policy (referring to the model trained in the previous iteration — initially the SFT model, subsequently the model from the previous round)** $\pi_{\text{old}}$ to sample, obtaining a trajectory, computing the "advantage" for each step. Use the **advantage function (introduced below)** to estimate advantage $A_t$ (positive = good action, negative = bad action).
Use **importance sampling (introduced below)** to use "old data" to train the new policy:

$$
L^{\text{PG}}(\theta)=- \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}A_t
$$

Where $L^{\text{PG}}(\theta)$ is the原始 policy gradient.

Denote the probability ratio as:

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}
$$

Thus:

$$
L^{\text{PG}}(\theta)=- r_t(\theta)A_t
$$

If $A_t>0$, push $r_t$ larger; if $A_t<0$, push it smaller.
Of course, there's a step size problem here — too large a step容易 "flip over": the probability ratio can skyrocket to 10 or plummet to 0.01, directly collapsing the policy.

Let's插播 what **importance sampling** and **advantage function** are.

**Importance Sampling (IS)** is a statistical method used when难以 directly sample from a target distribution — instead sample from another proposal distribution and weight the samples to estimate expectations under the target distribution. **Importance sampling is "using old data to compute new answers" — taking trajectories collected by the old policy, multiplying by a "weight," and treating them as trajectories from the new policy.** You want to know how handsome you look after a new haircut, but you don't have time to take a new photo today, so you pull out **last week's old photo** — but last week your hair was different. What to do? Attach a "similarity coefficient" to the old photo: if the new hairstyle is almost identical to the old → coefficient ≈ 1, photo is trustworthy; if vastly different → coefficient ≈ 0, photo is essentially worthless. This coefficient is the **importance weight** $r_t(\theta)$.

$$r_t(\theta) = \frac{\pi_{\text{new}}(\text{action}|\text{state})}{\pi_{\text{old}}(\text{action}|\text{state})}$$

The weight $r_t(\theta)$ automatically down-weights "parts of old trajectories that no longer resemble the new policy" while保留 similar parts.

1. **Save samples**: No need to re-sample from the environment every time parameters change; the same batch of data can be reused several times.
2. **Save time**: The most time-consuming part of deep RL is interacting with the environment; importance sampling makes "interact once, train multiple times" possible.
3. **Stable**: With clipping or penalty, prevents weight explosion (PPO restricts weights to the 0.8~1.2 range).

##### Advantage Function

The Advantage Function measures how much better a particular action is in a given state compared to the average level for that state. Its basic idea: $Q(s,a)$ represents the value of executing action $a$ in state $s$; $V(s)$ represents the average value of all actions in state $s$. The difference between them is the "relative goodness" of action $a$. **This means I can redefine the reward after subtracting an arbitrarily set baseline value** — we call this the advantage function.

Mathematical expression:

$$
A(s,a) = Q(s,a) - V(s)
$$

If $A(s,a) > 0$, action $a$ is better than the average policy for that state — should increase its probability. If $A(s,a) < 0$, action $a$ is worse than average — should decrease its probability. If $A(s,a) = 0$, action $a$ is at the average level — no particular superiority or inferiority.

Directly using $Q$ or $Return$ for policy gradient optimization produces high variance. Subtracting $V(s)$ reduces variance without changing the expectation, thus making policy updates more stable and efficient. In practical algorithms, advantage functions are typically approximated using methods like GAE (**Generalized Advantage Estimation**) for policy updates in mainstream RL algorithms like PPO, A2C, TRPO.

**GAE (Generalized Advantage Estimation)**

**TD Error**

$$
\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

**GAE Definition**

$$
A_t^{GAE} = \sum_{b=0}^\infty (\gamma\lambda)^b\delta_{t+b}^V
$$

GAE provides adjustable bias-variance balance.

Let's look back at this formula:

$$
L^{\text{PG}}(\theta)=- \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}A_t
$$

We can see that the原始 policy gradient maximizes expected return, using importance sampling for advantage estimation.

---

##### Step 2

PPO's probability clipping — as training continues, the model may deviate from the original model, continuously fitting the reward model. So we must limit model changes, drawing a **safety channel** for $r_t$, only allowing it to move within $[1-\varepsilon,1+\varepsilon]$.
Define the clipped ratio:

$$
r_t^{\text{clip}}=\text{clip}\bigl(r_t(\theta),1-\varepsilon,1+\varepsilon\bigr)
$$

Modify the loss to "take the more conservative of the two paths":

$$
L^{\text{CLIP}}(\theta)=-\min\Bigl(r_t(\theta)A_t,r_t^{\text{clip}}A_t\Bigr)
$$

When $A_t>0$ (good action):
  If $r_t>1+\varepsilon$, gradient is cut off — no more狂增 probability.
When $A_t<0$ (bad action):
  If $r_t<1-\varepsilon$, gradient is also cut off — no more狂降 probability.

---

##### Step 3

In actual code, two common配角 terms are added:

One is **value error** $L^{\text{VF}}$: makes the critic network estimate future rewards more accurately.

The second is entropy regularization $H(\pi)$: prevents the policy from becoming instantly deterministic (maintains exploration).
The final PPO loss looks like this:

$$
L^{\text{PPO}}=\underbrace{-\min\Bigl(r_t(\theta)A_t,r_t^{\text{clip}}A_t\Bigr)}_{\text{policy}} + c_1\underbrace{L^{\text{VF}}}_{\text{value}} - c_2\underbrace{H(\pi)}_{\text{entropy}}
$$

Typical coefficient values: $\varepsilon=0.2, c_1=0.5, c_2=0.01$.

$$
\boxed{L^{\text{PPO}}=-\min\Bigl(r_tA_t,\text{clip}(r_t,1-\varepsilon,1+\varepsilon)A_t\Bigr)+\text{(value + entropy)}}
$$

**The left side is probability clipping preventing步幅 from affecting the model too much; the right side is the "supporting配角."** It suppresses single-step changes that are too large, avoiding RL convergence difficulty.

#### 3. PPO Overall Process

Below, from both **macro process** and **core mechanism** perspectives, we introduce the complete operation of the PPO algorithm in large model training (like RLHF).

PPO in large model training typically serves as the **third step of RLHF**, with the first two steps being:
1. **SFT**: Fine-tune the base model with high-quality dialogue data to obtain the initial policy $\pi_{\text{SFT}}$.
2. **Train Reward Model (RM)**: Based on human preference data, train a model $r_\phi(s, a)$ to score and measure the quality of generated responses.

Then enter the PPO phase. The overall process is roughly:

```

Initialization: Policy Model = π_SFT, Reference Model = π_SFT (frozen)          
Reward Model = Trained RM (frozen)                            

                            ↓

Loop iteration (until convergence):                                       
1. Sampling phase: Use current policy π_θ to generate a batch of responses (interact with RM)        
2. Compute advantage: Use RM scoring + reference model to compute KL penalty, obtaining advantage A   
3. Update phase: On the same batch of data, use PPO objective to update π_θ multiple times          
   - Compute importance weight r = π_θ / π_old at each update               
   - Use clipping loss to limit update magnitude                      
4. Update old policy: π_old ← π_θ, prepare for next round of sampling                  

```

---

#### 3.1 Main Model Roles

| Model | Description | Trained? |
|------|------|----------|
| **Policy Model** $\pi_\theta$ | Currently being optimized, generates responses | Yes |
| **Reference Model** $\pi_{\text{ref}}$ | SFT model, used to compute KL divergence, prevents policy from drifting too far | No, frozen |
| **Reward Model** $r_\phi$ | Scores generated responses, provides the immediate reward $r_t$ in the advantage formula $A_t$ | No, frozen |
| **Value Model** $v$ | Provides $V_{s_t}$ (state value) in the advantage formula $A_t$, together with reward forms the advantage function | No, frozen |

#### 3.2 Sampling Phase (Data Collection)

First, input a batch of prompts. The current policy $\pi_\theta$ generates complete responses for each prompt (via autoregressive sampling). Simultaneously record the **log probability** $\log \pi_\theta(a_t \mid s_t)$ of generating each token (for subsequent importance sampling). Use the reward model to assign a score $r(x, y)$ to the **entire response** (x is prompt, y is response). Store prompts, generated responses, per-token log probabilities, and reward scores in a buffer.

#### 3.3 Computing Advantage Function $A_t$

Unlike standard RL, in large models, each token doesn't have an immediate reward — only the total score of the final response. Therefore, we need to **assign an advantage to each token**. Two common approaches:

**Approach 1: Token-level allocation**: Assign the final reward $R$ as the last token's advantage, with其余 token advantages being 0; or distribute $R$ uniformly/exponentially decaying across all tokens.

**Approach 2: Using GAE**: If token-level rewards are introduced (like KL penalty terms), compute each token's immediate reward, then use GAE to estimate advantage.

In practical RLHF, a simple method is commonly used: **only the last token has advantage $A = R$,其余 token advantages are 0**.
But when computing loss, only tokens with non-zero advantage contribute to gradients. More commonly, **assign the same advantage to every token** (equal to the full-sentence reward minus baseline),配合 **KL penalty** for refinement.

#### 3.4 KL Divergence Constraint

To prevent the policy from过度 optimizing the reward model leading to "reward hacking" (generating high-reward but low-quality text), typically subtract a **KL penalty** from the reward:

$$
\text{reward}_{\text{token}} = r_\phi(\text{full response}) - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

Where KL divergence is typically accumulated as per-token pointwise KL:
$$
\text{KL}_t = \log \pi_\theta(a_t \mid s_t) - \log \pi_{\text{ref}}(a_t \mid s_t)
$$
This way, each token generation immediately includes the penalty term in the reward, enabling per-token advantage computation.

#### 3.5 PPO Update Phase (Multiple Epochs)

For each batch of data in the buffer, perform multiple gradient updates (typically 4~10 epochs).

**For each token, compute**:

- Current policy's log probability $\log \pi_\theta(a_t \mid s_t)$ (forward propagation)
- Importance weight:
$$
r_t(\theta) = \exp\bigl( \log \pi_\theta(a_t \mid s_t) - \log \pi_{\text{old}}(a_t \mid s_t) \bigr)
$$
- Clipped objective:
$$
\text{surr} = \min\left( r_t(\theta) A_t,  \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right)
$$
- Loss function (minimization):
$$
L = -\mathbb{E}[\text{surr}]
$$

Importance weights can reuse data sampled by the previous old policy, saving time and compute.

Additionally, a **KL penalty term** or **value function loss** (if using a critic network) can be added.

#### 3.6 Value Function (Optional)

In large model RLHF, sometimes an additional value network (critic) is trained to estimate state value $V(s)$ for computing advantage.
The value network is typically **another output head sharing partial parameters with the policy**, predicting the state value at each token position.
During training, add value loss:
$$
L_{\text{value}} = \mathbb{E}[(V(s_t) - \text{return}_t)^2]
$$

---

#### 3.7 PPO Typical Hyperparameters in Large Models

| Parameter | Common Values |
|------|----------|
| Clipping range $\epsilon$ | 0.1 ~ 0.2 |
| KL penalty coefficient $\beta$ | 0.01 ~ 0.1 |
| Update epochs | 4 ~ 10 |
| Batch size | 64 ~ 256 |
| Learning rate | 1e-6 ~ 5e-5 (typically lower than SFT) |
| Advantage normalization | Normalize advantage within batch |

## 13.6 DPO Algorithm (Direct Preference Optimization)

### 13.6.1 DPO Core Idea

The DPO method succeeds because it **eliminates PPO's many complexities while performing well**: removes PPO's **reward model** (originally used to compute advantage functions), **abandons all policy optimization-related mechanisms** (like importance ratios). Returns to fundamentals — compute log loss on good results for positive gradient updates, compute log loss on bad results for negative gradient updates.

<img src="./images/13-3-DPO和PPO.png" width="800" alt="13-3-DPO-vs-PPO">

**From the figure above, we can see DPO is far less cumbersome and achieves equivalent performance.**

**No need to train an additional reward model, no need for complex RL loops; just write human preferences directly into a 'comparative' supervised loss to make the language model learn to generate good answers more and bad answers less.** In other words, DPO compresses the original two-stage pipeline of "first train reward model, then use PPO to maximize reward" into a single-stage pipeline of "one maximum likelihood loss," thereby **turning the RL problem into a supervised learning problem**.

No reward network, no advantage, no clipping — just one pair $(y_w, y_l)$ to compute gradients. Training is just like ordinary fine-tuning.

Think of the model as the "student" and the reference model as "their former self." The teacher hands over two essays: a model essay and a反面教材.
The DPO loss is one comment: **"You must be more like the model essay than yesterday's you, and more unlike the反面教材 than yesterday's you — otherwise, points deducted."** The student only compares "today's self vs. yesterday's self" each time, never needing to know absolute scores (rewards), yet can continuously improve. No need to train an additional "scorer."

### 13.6.2 The DPO Algorithm

DPO turns "reinforcement learning" into "weighted supervised learning," and只需要 one pair of "good/bad" answers to teach the model "be like the good, don't be like the bad." DPO data collection: use the SFT-trained model as the inference model; user inputs a prompt; the model performs multiple inferences to find good and bad answers.

In RLHF, PPO optimizes model-generated outputs through a reward model but requires training a value network, multiple rounds of policy updates, PPO's gradient clipping and KL regularization — these steps are **computationally expensive and training-complex** for large models. DPO proposes **directly using preference pairs for optimization** — no RL loop needed, no complex value network training needed.

---

#### Step 1: First, Write a "Preference Probability"

Assume we already know there's an invisible "reward" $r(x,y)$ behind the model's output. If a human says "I prefer $y_w$ over $y_l$," then in the $Bradley-Terry$ model:

$$
P(\text{win})=\sigma\bigl(r(x,y_w)-r(x,y_l)\bigr)
$$

$\sigma$ is the S-shaped function, compressing the difference to 0~1.

---

#### Step 2: DPO's Optimization Objective

Given two text generation results $y_1$ and $y_2$ for the same $prompt (x)$, with human or model-annotated preference (e.g., $y_1 \succ y_2$), DPO's goal is to make the model more likely to generate the preferred text, directly optimizing the probability ratio:

$$
r(x,y)=\beta\ln\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \text{C (constant)}
$$

$\pi_\theta$ is the model we're currently training, $\pi_{ref}$ is the initial SFT model (also called reference model), $\beta$ is the temperature coefficient, default 0.1~0.5.

$\sigma$ is the S-shaped function, compressing the difference to 0~1. Plug this $r(x,y)$ back into Step 1:

$$
P(\text{win})=\sigma\Bigl(\beta\ln\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\ln\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Bigr)
$$

---

#### Step 3: Maximum Likelihood → Minimize Negative Log-Likelihood

We want to maximize the probability that "human preference" is correctly guessed by the model, thus:

$$
\mathcal{L}_{\text{DPO}}=-\ln\sigma\Bigl(\beta\ln\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\ln\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Bigr)
$$

This is DPO's **loss**:

The first half $\ln\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}$ is called "relative log probability of the good answer"; the second half $\ln\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$ is called "relative log probability of the bad answer." The entire parentheses = "how much better good is than bad" → compressed by $\sigma$ into $(0,1)$ → taking negative log gives cross-entropy.

---

#### Step 4: Training Pipeline

**Data Preparation**: Collect prompts and corresponding generated output pairs $(y_1, y_2)$, along with human or small-model annotated preferences $(y_\text{preferred} \succ y_\text{less-preferred})$.

**Probability Calculation**: Feed each output into the LLM, compute generation probability $\pi_\theta(y|x)$ — can use $log-probability$ accumulation to obtain sequence probability.

**Compute Loss**: Use the DPO loss function $\mathcal{L}_\text{DPO}$, directly optimize LLM parameters through backpropagation.

**Iterative Training**: Batch-compute preference pair losses, gradient update model parameters — no value network needed, no RL环节 needed.

### 13.6.3 Two Variants of DPO

#### SimPO: Directly Remove Reference Model to Save Memory

The core idea: no longer compare with the "old model," but directly make "good answer probability" greater than "bad answer probability," requiring the good answer probability to lead by a **margin**. It makes two simple modifications: first, normalize update magnitude by response length (this思路 will reappear later); second, remove the reference policy. Although this破坏 DPO's mathematical论证 based on policy ratios, it more purely embodies the idea of weighting up quality / weighting down劣质.

Original DPO formula:

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta\log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]
$$

SimPO formula:

$$
\mathcal{L}_{\text{SimPO}}(\pi_\theta) = -\mathbb{E}\left[\log\sigma\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l \mid x) - \gamma\right)\right]
$$

$\beta$ is the temperature coefficient controlling probability ratio sensitivity; $\gamma$ is a hyperparameter introducing a fixed margin in SimPO, ensuring the "good" answer's probability exceeds the "bad" answer's by at least $\gamma$.
$|y_w|$ and $|y_l|$ respectively represent the lengths (in tokens) of the "good" and "bad" answers.

---

#### Length-Normalized DPO: Prevents Models from "Cheating with Long Responses"

The core idea: convert probabilities into **average per-token probability**, then compare; long texts no longer自然占优.

Formula:

$$
\max_{\pi_\theta}\mathbb{E}_{y_c,y_r\sim\mathcal{D}}\left[\log\sigma\left(\frac{\beta}{|y_c|}\log\frac{\pi_\theta(y_c \mid x)}{\pi_{\text{ref}}(y_c \mid x)} - \frac{\beta}{|y_r|}\log\frac{\pi_\theta(y_r \mid x)}{\pi_{\text{ref}}(y_r \mid x)}\right)\right]
$$

Denominators $|y_c|$ and $|y_r|$ are answer lengths (token counts).

In **length-normalized DPO**, dividing log probability by answer length can **reduce the model's tendency to generate longer responses**, because **longer responses, even with only slight probability advantages**, may obtain higher unnormalized log probabilities due to length.

### 13.6.4 RL Considerations

RL **findings are often highly dependent on the specific environment**. Depending on the runtime environment, base model, and post-training preference data, conclusions can differ drastically. For example, the AI2 team, when comparing DPO vs. PPO, once found PPO superior due to its online policy characteristics and precisely demonstrated the DPO-to-PPO gap. But in their subsequent Tulu 3 study, they found that if using more精巧 SFT methods, both PPO and DPO gains would disappear, with only standardized DPO maintaining its advantage — conclusions截然 different. These two studies differ in many aspects, but it's not a matter of right or wrong. What's important is that we shouldn't over-generalize conclusions based on a single paper. This caution同样 applies to PPO and GRPO discussed later — **never treat any single experimental result as dogma**.

<img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/zh/chapter13/images/13-4-过度优化.png" width="800" alt="13-4-over-optimization">

**Over-optimization problem**. This is essentially overfitting, but this term is important because it fundamentally reveals a phenomenon: when continuously optimizing the policy, imagine the horizontal axis representing the degree of RL implementation. Initially, reward values持续上升, but eventually the reward model fitted on human preferences will deviate from actual human preferences. The more you optimize, the greater the deviation, ultimately陷入 **seemingly optimizing yet the reward actually not improving**.

This phenomenon is几乎无处不在 in RLHF and is a very serious problem. The root cause of over-optimization lies in the noisy nature and complexity of human preferences. Someone conducted a study: applying RLHF separately to noisy AI feedback, clean AI feedback, and human feedback. Results clearly showed that both human feedback and noisy AI feedback exhibited明显的 over-optimization, while clean, noise-free AI feedback did not. Therefore, in actual post-training processes, you should expect to see curves similar to the left chart — when the model performs increasingly well on proxy reward metrics, its human preference win rate may not improve in tandem.

Beyond over-optimization, RLHF frequently faces two other important issues:

- **Mode Collapse**: Since RL's objective is to maximize reward rather than fit the entire data distribution, RLHF-trained models往往丧失 output diversity. For the same prompt, the model may only generate one or two "high-scoring" responses, no longer producing the rich variety of answers possible right after SFT. This is because RL allows the policy to collapse to the single highest-reward point,不再强制 maintain entropy.

- **Calibration Degradation**: OpenAI listed calibration as an unresolved issue in the GPT-4 technical report. After RLHF, the model's confidence in its own predictions往往不再匹配 its actual correctness probability — models can become极度 "confident" while wrong. This problem is particularly critical in subsequent RL with verifiable rewards (RLVR), because good uncertainty estimation and exploration capability are prerequisites for the model to continuously improve in complex reasoning tasks.

Therefore, when we discuss reasoning models in the next chapter, "how to maintain exploration and avoid premature collapse" will become a core concern.