# Chapter 10: Inference

<div align="center">
<img width="936" height="306" alt="image" src="https://github.com/user-attachments/assets/80bde0f2-6a2e-4b4c-a146-3469135e50ae" />
   <p>Inference workflow diagram</p>
</div>

Inference is the core stage where large language models truly deliver value — it is the only pathway through which knowledge accumulated during training is converted into practical capability. Whether for intelligent Q&A, text generation, code completion, search, or translation, models must complete understanding, computation, and output through the inference process.

With the rapid development of Agents, next-generation AI development tools like Claude Code, Codex, and WorkBuddy have their core capabilities almost entirely built on LLM inference. Although these tools can connect to different LLM APIs, **the actual user experience differs significantly**: models with stronger performance typically have higher task completion rates, more stable inference processes, and better code generation quality. However, high-quality output alone is not enough — in Agent scenarios, a complex task often requires multiple rounds of planning, tool invocation, code execution, and reflection, with inference latency accumulating continuously. Therefore, model response speed, throughput capacity, and inference cost have become equally important factors affecting overall development efficiency and user experience. It can be said that **modern LLM competition is not only about "whether the answer is correct," but also about "whether the answer is fast enough and economical enough."**

So, where do the bottlenecks in LLM inference come from? What techniques has the industry adopted to improve inference speed and reduce inference cost? Before analyzing these optimization methods, we first need to establish basic metrics for evaluating inference performance, understand what makes a model "fast at inference," and identify the key factors affecting inference efficiency. Next, we will systematically analyze LLM inference optimization techniques围绕 these questions.

## Learning Objectives

After reading this chapter, you should be able to give complete answers to the following questions:

1. How to evaluate an LLM's inference performance — what do TTFT, inter-token latency after the first token, and Throughput each measure, and why can't they simply substitute for each other?

2. What are the differences between training and inference in terms of input source, parallelism, memory footprint, and performance bottlenecks? Why can training process entire sequences in parallel while autoregressive inference typically must generate token by token?

3. Why distinguish between Prefill and Decode in inference? What repetitive computation problem does KV Cache solve, and why does it become a memory and bandwidth burden for long-context Decode?

4. How to use FLOPs, HBM data transfer volume, and arithmetic intensity to determine whether an inference stage is more likely compute-bound or memory-bound? Why can increasing batch size, adopting GQA/MQA, and quantization improve inference efficiency?

5. What bottlenecks do current inference optimization techniques each address? What trade-offs do SSM, diffusion models, speculative decoding, and MTP make among parallelism, context consistency, KV Cache, latency, and throughput?

*Let's continue our analysis with these questions in mind!*

This chapter's discussion路线 corresponds to these questions one by one: first compare the basic mechanisms of training and inference, then introduce Transformer autoregressive inference, Prefill/Decode, and KV Cache; next, use arithmetic intensity to quantitatively analyze Attention and MLP bottlenecks, and discuss latency, throughput, batching, and prompt compression; finally,转向 structural or system-level optimization methods such as SSM, diffusion models, speculative decoding, and MTP. This connects "how the model generates tokens" with "why generation is fast or slow" and "how to further accelerate" into one complete主线.

## 10.1 Inference and Training

Although training and inference are each constrained by different bottlenecks, they are not independent stages; training shapes capability, inference presents capability, and inference demands in turn influence training design. Together they form the complete system cycle of building large-scale language models.

### 10.1.1 Differences Between Inference and Training

In LLMs, both training and inference involve model forward computation, but their goals, computation patterns, and resource bottlenecks differ fundamentally. For autoregressive inference or parallelized training, assuming we need to generate the $i$-th token, the model's conditional probability at that position can be uniformly written as:

$$
P(\text{token}_i \mid \text{token}_1, \text{token}_2, \dots, \text{token}_{i-1})
$$

>Training and inference are mathematically identical in single-step prediction — both estimate probability distributions over the vocabulary and select from them — but **the difference lies in where the preceding context for $\text{token}_i$ comes from (ground-truth vs. generated) and whether the prediction result serves as subsequent input and whether parallelization is possible**.

**Comparison of input sources for next-word prediction in training vs. inference**

| Stage     | Input Source for Predicting $\text{token}_i$                                                             |
| ------ | ---------------------------------------------------------------------- |
| **Training** | From ground-truth labels and their previous input sequence; the model can use the full sequence for parallel computation, while causal masking ensures each position only accesses preceding tokens. |
| **Inference** | From the model's previously predicted tokens and prior text; each prediction step depends on the previous step's output, must be generated step by step (autoregressive).           |

**1. Training Stage**

The main goal of the training stage is to **optimize model parameters**. Training requires executing complete forward and backward propagation, retaining intermediate activations for gradient computation, and synchronizing gradients across multiple GPUs. Therefore, the training process is highly compute-dependent, with bottlenecks primarily coming from:

- **FLOPs** from large-scale matrix multiplications;
- **Cross-device and intra-device communication volume** involved in multi-GPU parallel execution;
- **Additional memory footprint** from activations, gradients, and optimizer states.

> Training stage memory footprint主要由四部分组成：activations, gradients, optimizer states, and model parameters.

For autoregressive [Transformer](https://www.bilibili.com/video/BV1TZ421j7Ke/) training, we input the complete target sequence and use a **causal mask**: each position can only access preceding tokens (ground-truth labels) and cannot see future tokens. Since the training input is a known complete sequence, the model can treat the entire sequence as one large batch and **compute all positions' forward representations in parallel** through one or a few large matrix multiplications, while the causal mask ensures self-attention's sequential dependencies. Intermediate activations are retained for backpropagation and parameter updates. Precisely because of this, although self-attention involves pairwise position attention computation, training can still be **highly parallel** in the time dimension — this is the key reason Transformers efficiently utilize GPUs in large-scale training.

**2. Inference Stage**

The goal of the inference stage is to **generate output sequences using fixed parameters**. Inference typically uses **autoregressive generation**: when generating each `token`, the model needs to reference all previously predicted tokens' information — specifically, the Key and Value (KV Cache) of each layer.

We can think of the LLM inference process like writing an essay: every word you write requires looking back at what you've already written. As context length grows, these "reference materials" (KV Cache) also grow, consuming more memory, and each access takes time — like flipping through an increasingly thick notebook. Therefore, **inference speed is primarily limited not by compute power, but by memory capacity and memory bandwidth.**

**In summary: training is bottlenecked by compute, while inference is bottlenecked by memory and bandwidth.**

> Why does autoregressive LLM training typically use ground-truth preceding tokens as input (i.e., teacher-forcing) rather than the model's previously predicted tokens?
>
> In the training stage, using teacher-forcing has two main reasons: first, it provides stable and明确的 supervisory signals, facilitating rapid model convergence; second, it allows parallel computation over the entire sequence, dramatically improving training efficiency. If the model's previously predicted tokens were used as input for the next step during training, it could lead to accumulated prediction bias in hidden states. Additionally, since the sampling (predicted token generation) operation is non-differentiable, it would破坏 standard maximum-likelihood-based per-position loss computation and backpropagation, increasing training difficulty.

**Example: Training vs. Inference Parallelism Comparison**
```
Training Stage (full sequence parallelized, using ground-truth preceding context)
----------------------------------------------------
Input sequence:  x1            x2           x3            x4
Model forward:  ┌────┐      ┌────┐       ┌────┐         ┌────┐
                │f(x1)│     │f(x1,x2)│   │f(x1,x2,x3)│  │f(x1,x2,x3,x4)│
                └────┘      └────┘       └────┘         └────┘
Output sequence: y1           y2           y3             y4
Notes:
1. All token predictions y_i can be computed in parallel (one Transformer forward pass yields all y_i).
2. Position i's prediction is conditioned on ground-truth preceding context (x1...x{i-1}), as causal mask prevents seeing future tokens.
3. Predictions y_i are only used for computing training loss (e.g., cross-entropy), not as下一步 input.

----------------------------------------------------
Inference Stage (autoregressive generation, token-by-token dependent on model-generated preceding context)
Initial input: x1  
Step 1:     y1 = f(x1)
Step 2:     y2 = f(x1, y1)
Step 3:     y3 = f(x1, y1, y2)
Step 4:     y4 = f(x1, y1, y2, y3)
Notes:
1. Each step must wait for the previous step to complete; no parallelization possible.
2. Position i's preceding context is the model's own generated tokens (not ground-truth).
3. y_i serves as下一步 input and continuously extends the sequence.

----------------------------------------------------

```

In the LLM inference stage, each token generation depends on all previously generated tokens, so the token generation flow has strict sequentiality in the time dimension and cannot output multiple tokens in parallel at once. However, it must be emphasized that this does **not** mean computation within inference cannot be parallelized — at each step, computing new KV, attention weights, and feed-forward networks are essentially large matrix operations that can be highly parallelized on GPUs. **Therefore, the inference stage can be characterized as sequential in the time dimension (autoregressive constraint) but parallel in the computation dimension (matrix operation acceleration).** Training can process the entire sequence in parallel while inference can only sequentially process generated tokens, but the computation within each token remains efficiently parallelized.

At this point, we've answered half of the second foundational question: both training and inference perform forward computation, but training leverages known complete sequences for time-dimension parallel computation and保存大量 intermediate states for backpropagation; inference uses the model's own generated preceding context, subject to strict autoregressive dependencies. Next, we need to further explain: how this sequential generation is specifically implemented in Transformers, and why KV Cache can alleviate repetitive computation yet带来 new memory pressure.

### 10.1.2 Connections Between Training and Inference

Although training and inference differ明顯 in operational mechanisms, performance bottlenecks, and optimization directions, the two are closely interconnected throughout the model lifecycle:

- **The fundamental purpose of training is to optimize inference behavior.**
  Whether using supervised learning, pretraining, supervised fine-tuning, or alignment methods, their results ultimately manifest in the generation quality, stability, and consistency of the inference stage. Inference is the way models present all their capabilities to users, so every design choice in the training process — data selection, loss functions, regularization methods — ultimately affects inference performance.

- **Inference itself is an indispensable part of the training process.**
  Model validation, capability evaluation, instruction-following tests all depend on inference processes. During training, models must generate responses through inference, then use these responses to compute rewards or losses and update parameters. Therefore, from a systems perspective, inference does not only appear in the deployment phase but贯穿 the entire iterative cycle of training.

- **Structural design decisions during training determine the optimizability of inference.**
  The inference stage is constrained by memory, KV Cache size, latency requirements, bandwidth, and many other bottlenecks, so model structural design during training must provide guarantees for inference executability and performance.
  
    - **Structural design affects inference resource usage.**
    For example, Transformer structure, attention complexity, parameter scale, MoE expert layout, whether KV compression exists — all determine memory footprint and latency performance during inference. Poorly designed structures may perform well during training but cannot run efficiently at all during inference.
  
    - **Structural design affects the optimization potential for inference.**
    Whether FlashAttention, long-context mechanisms, inference acceleration strategies can be used all depend on whether the model structure left room for these optimizations during training. An inference-friendly structural design can achieve higher throughput, lower latency, and greater scalability during deployment. These designs often must be determined before training and cannot be临时 remedied during inference. Therefore, driving improvements in model inference efficiency requires incorporating systematic structural design considerations during the training stage.

>FlashAttention remains a global attention mechanism, but splits the originally need to compute and store at once large attention matrix into multiple small blocks that can be processed in high-speed on-chip cache. Through tiled computation, it significantly reduces memory read/write volume, thereby lowering memory access overhead and significantly improving inference and training efficiency.

## 10.2 Inference Principles Analysis

Therefore, the training stage determines what the model learns, and the inference stage determines whether these capabilities can be delivered to users at acceptable speed and cost. Below we enter inference principles analysis: first明确 performance metrics, then沿着 a single request's Prefill and Decode flow, locate where computation, memory, and bandwidth respectively become bottlenecks.

Training is a one-time cost, but inference is repeatedly executed. As large models truly land in various application scenarios, inference scale is rapidly expanding. Sam Altman has publicly mentioned that OpenAI now generates over 100 billion words per day; mature AI development tools like Cursor produce billions of lines of code actually adopted by users daily. These numbers illustrate a fact: inference has become the dominant cost in LLMs, not training. To more clearly understand inference efficiency, we typically measure along several dimensions: Time To First Token, inter-token间隔 during generation, single-request total time, and system total throughput.

  - **TTFT (Time To First Token)**: The time from user request to seeing the first token, primarily influenced by request queuing and Prefill computation.
  - **ITL (Inter-Token Latency)**: The average interval between adjacent output tokens after the first token, more directly reflecting the streaming output speed of the Decode stage.
  - **E2E Latency**: Total time from request arrival to final token output completion, can be approximately understood as the sum of queuing time, TTFT, and subsequent token generation time.
  - **Throughput**: The number of tokens processed or generated per unit time by the system, typically used to measure overall efficiency of multi-request serving.

>It's worth noting that **high throughput does not equal low latency**. In large-scale inference systems, overall system throughput can be very high (producing many tokens per second), but individual conversation requests may still have long response times — reasons include particularly long context, fewer compute resources allocated to that request, or the request waiting in scheduling queues. `Throughput` measures system-level overall efficiency, while `latency` focuses on individual user or single conversation request experience, because what users directly perceive is the waiting it brings.

Next, we'll start from one autoregressive inference step of a Transformer, explaining how a request goes through Prefill and Decode, where KV Cache is generated and reused; then convert these processes into FLOPs, HBM data transfer volumes, and arithmetic intensity, explaining why different stages exhibit different performance bottlenecks. Finally, we'll discuss how to optimize against these bottlenecks through scheduling, input prompt compression, and model structural modifications.

### 10.2.1 Transformer

<div align="center">
<img width="1260" height="500" alt="Transformer inference process" src="https://github.com/user-attachments/assets/3e0b268a-1700-4dcd-b7ef-aff88e3cd5c9" />
   <p>Figure 10.1 Transformer Inference Process</p>
</div>

In [autoregressive Transformer](https://jax-ml.github.io/scaling-book/inference/) LLM inference, a core optimization point is that for already-existing context, we don't need to redo the complete forward computation for every newly generated token. To this end, inference is typically divided into two phases: **Prefill** and **Decode**.

<div align="center">
<img width="1200" height="600" alt="Autoregressive Transformer inference" src="https://github.com/user-attachments/assets/bc301529-045d-4b35-ad59-dc440eea09c4" />
   <p>Figure 10.2 Autoregressive Transformer Inference</p>
</div>

- **Prefill Phase**: The model processes all user-input prompt tokens at once, and at each layer and each attention head, computes the Key and Value of these tokens, denoted as $K, V$. These vectors are then saved into the **KV Cache** structure. The purpose of saving KV is simple — when generating each new token subsequently, we only need to do one linear projection on this new token to get its Query (Q), then let Q do dot products with all cached K, and use the existing V for weighted summation. This way, we don't need to repeatedly do linear mappings and matrix multiplications for old tokens, thus avoiding large amounts of redundant computation.

- **Decode Phase**: The model uses the new token embedding from the previous step to generate its Query and executes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
$$

Where $(K,V)$ all come from the KV Cache, and the resulting output logits are the model's **unnormalized scores for each token in the vocabulary**, which after softmax normalization become the vocabulary's probability distribution, used for sampling the next token. Then this new token's new $(K,V)$ at each layer continues to be appended to the cache, looping until a终止符 or maximum length is reached.

>KV Cache occupies memory, roughly proportional to `token count × layer count × attention head count × hidden dimension`.

It's important to note — **KV Cache is suitable for autoregressive inference mode** because in autoregressive settings, past tokens once computed won't be changed by future tokens, so their corresponding $(K,V)$ can be reused.

① For **bidirectional attention** (e.g., BERT, encoder-only) or **scenarios with frequent input context modifications** (e.g., masked language models, text editors), any token change affects other tokens' representations and Query-Key relative relationships, thereby changing attention weights i.e., $QK^\top$ and the $V$ to be weighted. Therefore, in these non-autoregressive inference scenarios, cached $(K,V)$ cannot remain valid — once the sequence has modifications, all $(K,V)$ must be recomputed — **KV Cache is not suitable**.

② In diffusion models, attention modules also compute Key and Value, but their meaning differs from KV Cache in autoregressive models. KV in diffusion models is generated in parallel for all spatial positions in one forward computation, and typically recomputed at each `diffusion step`, so they don't gradually accumulate along the time dimension during generation like autoregressive generation. When models引入 external conditions (e.g., text conditions), KV produced by the condition encoder can be reused across multiple diffusion steps to reduce redundant computation, but this caching only applies to condition representations that don't change across diffusion steps — **it is not equivalent to the KV Cache in autoregressive models that continuously grows with sequence length** and persistently occupies memory throughout the entire generation process.

>Why do most current generative models (GPT series, LLaMA series, DeepSeek series, etc.) adopt autoregressive attention architecture rather than bidirectional attention like BERT?
>
>1. Efficient stepwise generation: Autoregressive models predict the next token sequentially and can use KV Cache for previously computed Key and Value, avoiding redundant computation and enabling streaming generation and online interaction with efficient inference.
>2. Global consistency vs. computational overhead: Bidirectional attention can access both preceding and following context, improving understanding and coherence, but requires multiple full-sequence attention passes and iterative updates during generation, with large computational and memory overhead and high interaction latency.
>3. Engineering trade-off and scalability: Autoregressive architecture is easy to scale in training; although it sacrifices instant global modification capability, it is the most practical compromise for large-scale deployment.

---

**Simple LLM Autoregressive Inference Code Implementation**
```python
import numpy as np
# Vocabulary
vocab = ["I", "love", "deep", "learning", "<EOS>"]
token2id = {w: i for i, w in enumerate(vocab)}
id2token = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)

# Embedding dimension
d_model = 8
np.random.seed(42)
E = np.random.randn(vocab_size, d_model)

# Per-token probability distribution for next token (row: current token, col: next token)
manual_logits = np.array([
    [0.0, 3.0, 0.0, 0.0, 0.0],   # I -> love
    [0.0, 0.0, 0.0, 3.0, 0.0],   # love -> learning
    [0.0, 3.0, 0.0, 0.0, 0.0],   # deep -> love
    [0.0, 0.0, 0.0, 0.0, 3.0],   # learning -> <EOS>
    [0.0, 0.0, 0.0, 0.0, 0.0],   # <EOS> -> nothing
])

# Softmax
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# KV Cache + controlled generation
def generate_kv_readable(prefix, max_len=10, min_len=3):
    print(f"Input prefix: {prefix}")
    ids = [token2id[w] for w in prefix]

    K_cache = np.zeros((0, d_model))
    V_cache = np.zeros((0, d_model))

    for step in range(max_len):
        last_id = ids[-1]
        x = E[last_id]

        # Q, K, V
        q = x @ np.eye(d_model)  # Simplified: Q=K=V=embedding
        k = x
        v = x

        # Update KV Cache
        K_cache = np.vstack([K_cache, k])
        V_cache = np.vstack([V_cache, v])

        # Attention
        att_scores = K_cache @ q
        att_scores /= np.sqrt(d_model)
        att_weights = softmax(att_scores)
        context = att_weights @ V_cache

        # Use manual logits + softmax
        logits = manual_logits[last_id]
        probs = softmax(logits)

        # Use argmax to ensure controlled generation
        next_id = int(np.argmax(probs))
        next_token = id2token[next_id]

        print(f"Step {step+1} prediction → {next_token} (p={probs[next_id]:.2f})")
        print(f"  Attention weights: {np.round(att_weights, 2)}")

        # Minimum generation length control for <EOS>
        if next_token == "<EOS>" and step+1 < min_len:
            # Force selection of second-highest probability token
            sorted_ids = np.argsort(probs)[::-1]
            for sid in sorted_ids:
                if sid != token2id["<EOS>"]:
                    next_id = sid
                    next_token = id2token[next_id]
                    break

        if next_token == "<EOS>":
            ids.append(next_id)
            break
        ids.append(next_id)
    print("\nFinal generated sequence:", " ".join(id2token[i] for i in ids))

# Test
generate_kv_readable(["I"])
```
Input: ["I", "love", "deep", "learning", "`<EOS>`"], generate_kv_readable(["I", "deep"])

Output: Final generated sequence: I deep love learning `<EOS>`

**The output generation sequence problem is not caused by KV Cache or code logic itself, but by random sampling + untrained parameter matrices.**

Currently, mainstream LLM inference systems adopt decoder-only, autoregressive Transformers. This structure can use KV Cache during inference to reuse historical tokens' Key and Value, thereby avoiding redundant computation. However, as context grows longer, KV Cache brings significant memory and bandwidth overhead. Therefore, recent years have seen many [Transformer variants such as GQA, MHA, MLA, sparse attention mechanisms, DSA, etc.](https://github.com/xuhu0115/CS336-Chinese-co-construction/blob/main/docs/chapter4/) proposed to improve long-context inference efficiency or reduce KV burden. As well as MINMAX's hybrid attention mechanism which明显 accelerates inference — its architecture融合 linear attention and Softmax attention in a structured pattern: after every 7 Transnormer layers using linear attention, insert one Transformer module using Softmax attention, totaling 80 layers.

**MINMAX's Linear Attention + Local Attention Mechanism**

<div align="center">
<img width="800" height="700" alt="MINMAX hybrid attention" src="https://github.com/user-attachments/assets/bd75d146-3db9-4d37-b384-5aeb9728275f" />
   <p>Figure 10.3 MINMAX Hybrid Attention Mechanism</p>
</div>

Linear attention mechanisms achieve efficient computation in sequence processing by linearizing traditional dot-product Softmax attention. During training, its complexity is $O(n d^2)$, and during inference, it can efficiently generate sequences by recursively updating the cumulative term $\sum K^\top V$. This is particularly efficient compared to traditional Softmax attention which still requires $O(n^2 d)$ computation during inference.

<div align="center">
  <img width="720" height="600" alt="Inference time comparison" src="https://github.com/user-attachments/assets/307be2b9-6795-4198-9aba-da299c2b3886" />
  <p>Figure 10.4 Inference Time Comparison</p>
</div>

In autoregressive generation scenarios, we want the model to **only see previously generated words** when generating each word and not "peek" at later words. If we directly use the $\phi(Q) (\sum K^\top V)$ method, it's equivalent to mixing all words' key information at once before computing each word's output, which would cause the model to **accidentally see future word information** when computing the current word, violating the autoregressive rule. To avoid this, we need to **accumulate step by step** during computation: after generating each word, add that word's `key, value` to the accumulator; when generating the next word, only use the prefix information already accumulated. This way, each word's output only depends on previously generated words and cannot see the future.

>The stepwise accumulation method **cannot compute all words at once in parallel like training or full-sequence parallelism** — it must generate word by word in order, so inference speed is slower than full-sequence parallelism. But compared to local attention mechanisms, linear attention still has clear computational advantages during inference because its per-step computation grows linearly, not quadratically.

Therefore, although linear attention was proposed nine years ago, due to its relatively complex design, current mainstream open-source LLMs — including LLaMA3, Qwen2.5, DeepSeekV3, and Mistral — still haven't adopted linear attention as the default scheme for autoregressive generation.

<div align="center">
<img width="1340" height="540" alt="Global vs Linear Attention" src="https://github.com/user-attachments/assets/13a43c6b-f5db-4b62-9a03-34a362e97e01" />
  <p>Figure 10.5 Global Attention vs. Linear Attention</p>
</div>

**Impact of Linear Attention on Generation Tasks**

```python
import numpy as np

# ---- Configuration ----
L = 5          # Sequence length
d = 2          # Feature dimension
np.random.seed(42)  # Set random seed for reproducibility

# Randomly generate Q, K, V matrices, all shaped (L, d)
Q = np.random.rand(L, d)
K = np.random.rand(L, d)
V = np.random.rand(L, d)

# Kernel mapping φ(x), here using ReLU activation as kernel function approximation.
# In linear attention, kernel functions approximate softmax,
# aiming to reduce the quadratic complexity O(L^2) of QK^T to O(L * d)
def phi(x):
    # ReLU
    return np.maximum(x, 0)

# ---- Wrong way: right-multiply K^T V (includes future info, non-causal) ----
# phi(K) is (L,d), phi(K).T is (d,L), V is (L,d)
KV_all = (phi(K).T @ V)

# Compute output Y
# Numerator: phi(Q) (L,d) × KV_all (d,d) → (L,d), the unnormalized Attention result.
Numerator = phi(Q) @ KV_all

# Compute global normalization factor Z
# phi(K).sum(axis=0, keepdims=True) sums K over sequence length L, obtaining (1, d)
# .T converts to (d, 1)
# This normalization factor Z is also computed globally, without considering causality.
Denominator = phi(Q) @ phi(K).sum(axis=0, keepdims=True).T

# Y_wrong: final result (L,d), each row Y_wrong[i] is influenced by the entire sequence's K and V
Y_wrong = Numerator / Denominator
print("Wrong way (includes future info):")
print(np.round(Y_wrong, 3))

# ---- Correct way: prefix accumulator (only prefix, causal computation) ----
# This is the linear attention computation method that autoregressive models should use during inference.
# S: accumulator for storing the prefix sum of K^T V, S is (d, d), accumulating K^T V products
S = np.zeros((d, d))
# Z: normalization factor accumulator, Z is (d, 1), accumulating K vector sums
Z = np.zeros((d, 1))
Y_correct = []

# Traverse each position i in the sequence
for i in range(L):
    # Take K, V, Q vectors at current position i and apply kernel mapping
    # ki, vi, qi are all (d, 1)
    ki, vi, qi = phi(K[i:i+1]).T, V[i:i+1].T, phi(Q[i:i+1]).T
    
    # Incrementally update S: S += K[i]^T @ V[i]
    # Accumulate all historical tokens' (Kᵢᵀ Vᵢ), building prefix information pool
    # Used to "read" needed content from all past information at current step
    S += ki @ vi.T
    
    # Incrementally update Z: Z += K[i]^T
    # Prefix accumulation of all K_i, serving as attention normalization factor;
    # Only contains all past information, ensuring current token won't see future content
    Z += ki
    
    # Compute output y_i at current position i
    # Numerator: qi^T @ S_i; Denominator: qi^T @ Z_i
    # y_i is (1, d), only depending on information before (inclusive) i
    y_i = (qi.T @ S) / (qi.T @ Z)
    
    Y_correct.append(y_i.flatten())

Y_correct = np.array(Y_correct)
print("\nCorrect way (prefix only):")
print(np.round(Y_correct, 3))

# ---- Compare differences ----
print("\nDifference (wrong - correct):")

# Theoretically, Y_wrong[i] should contain Y_correct[i] plus the influence of future information.
print(np.round(Y_wrong - Y_correct, 3))
```

Output (impact of linear attention):
>Difference (wrong without accumulation - correct with accumulation):
>
>    I -> [ 0.05  -0.196]
>
>   like -> [-0.097 -0.037]
>
>   deep -> [-0.058  0.028]
>
>   😄-> [-0. -0.]

This code demonstrates that in autoregressive LLM inference, "peeking at the future" produces encoding bias, and how **prefix accumulation** ensures causality. The difference vector (wrong without accumulation - correct prefix accumulation) quantifies the impact of future information leakage on each token. The wrong approach uses one-shot global computation $K^T V$, introducing future token information at each step's output. Results show:

  - **The first token "I"** is influenced by all 4 future tokens, with the largest deviation.
  - As the sequence progresses, intermediate tokens are influenced by fewer future tokens, with偏差 progressively decreasing.
  - **The last token "😄"** has no future tokens, so correct and wrong outputs are完全一致 with zero difference.

`Prefix accumulation` is the core technique of linear attention in autoregressive tasks. Through incremental updates, it maintains $O(L)$ computational complexity while strictly ensuring causality, preventing future information leakage, and thereby ensuring each `token`'s representation only depends on already-generated preceding context. Causal masking prevents the model from seeing the future, the prediction task forces the model to infer the next token from preceding context, and the attention mechanism helps the model extract key clues from preceding context — the three work协同 to enable the model to automatically learn language规律 and achieve efficient and accurate autoregressive prediction.

Currently, linear attention mechanisms are still being used, along with recent approaches like Kimi K3's KDA, mentioned in earlier chapters.

---

The previous Transformer mechanism explained "why data is repeatedly computed or repeatedly moved," while arithmetic intensity further answers "whether these computations are sufficiently numerous relative to HBM data movement." Therefore, below we no longer merely describe the shapes of KV Cache and matrix multiplications, but uniformly use FLOPs/Byte to estimate the computational intensity of different `kernels`.

### 10.2.2 Arithmetic Intensity Analysis

**Notation Definitions**

- **B**: batch size, the number of text sequences processed simultaneously (inference typically B=1);
- **D**: token embedding dimension, i.e., hidden layer dimension;
- **T**: number of next tokens to output. In Prefill phase, this is the input processing sequence length; in Decode phase, it is 1;
- **S**: number of already-generated tokens (context used to generate next tokens' input);
- **F**: intermediate dimension in feed-forward networks (typically approximately 4× D);
- **H**: number of attention heads (Q heads);
- **$H_{kv}$**: number of KV heads; for MHA, $H_{kv}$ = H; for GQA/MQA, $H_{kv}$ < H.

**Arithmetic Intensity (AI)**

Arithmetic Intensity is the core metric for衡量 the computational intensity of `Kernel-level` computation tasks, used to predict whether a task's performance bottleneck is compute-bound or memory-bound.

$$\text{AI} = \frac{\text{FLOPs}}{\text{Bytes Transferred}}$$

- FLOPs: **Floating-point operation count**, i.e., the total amount of additions, multiplications, etc. in matrix multiplication. One addition or multiplication counts as `1 FLOP`. A single matrix operation simultaneously包含 multiply-add operations, counted as `2 FLOPs`;
- Bytes Transferred: **Data transfer volume**, i.e., the **total data volume (bytes) that the Kernel needs to read from and write to HBM** when executing the task.

Subsequent arithmetic intensity analysis assumes bf16 data type, i.e., 2 bytes per element. A single Transformer layer主要由 Attention and MLP两部分组成, and we analyze their Arithmetic Intensity separately. Whether MLP or Attention, AI analysis essentially都需要 consider data movement during computation, including loading input data and model weights from HBM, going through on-chip storage for data reuse, and writing output results back to HBM after computation.

>For optimization methods like FlashAttention, since they *reduce HBM read/write count of intermediate tensors through tiling, on-chip caching, and operator fusion*, they reduce data transfer volume. Since FLOPs are几乎 unchanged, Arithmetic Intensity is improved.

To unify the following analysis, we默认 each complete `kernel` includes the final output write-back to HBM. The **data flow process of each Transformer component can be abstracted as**:

```text
HBM → L2 Cache → L1 Cache / Shared Memory → Registers → Tensor Core
      ↑                                                    ↓
      └────── Data reuse in L2/L1/Shared Memory/Registers ────┘
Output: Registers → L1/L2 → HBM
```

The arrows above abstract the data path of a single matrix multiplication kernel: Tensor Cores typically fetch data from Registers, and accumulation results are also first saved in Registers before finally writing back to HBM through cache levels. Reuse in L2, Shared Memory, and Registers depends on specific tiling, operator fusion, and cache hit situations — cannot be simply understood as all weights completely residing in L2. Subsequent formulas uniformly treat HBM reads/writes as the primary data transfer volume, while actual HBM data traffic varies with the degree of operator fusion.

① **MLP Part**: The core operation is matrix multiplication. Input $x$ has shape $(B, T, D)$, typically flattening the first two dimensions to $N=BT$, then computing GEMM of the form $XW$. Computation requires reading input and weight matrices from HBM, completing computation, then writing results back to HBM:

**a. SwiGLU Activation with 3 Weight Matrices**

$$\text{SwiGLU}(x) = \text{Swish}(xW_{\text{gate}}) \odot (xW_{\text{up}}), \quad \text{MLP}(x) = \text{SwiGLU}(x) \cdot W_{\text{down}}$$

Where weight matrix shapes: $W_{\text{gate}} \in \mathbb{R}^{D \times F}$, $W_{\text{up}} \in \mathbb{R}^{D \times F}$, $W_{\text{down}} \in \mathbb{R}^{F \times D}$. The computation involves 3 matrix multiplications, each of shape $(B, T, D) \times (D, F)$ or $(B, T, F) \times (F, D)$:

$$\text{FLOPs} = 3 \times 2BTDF = 6BDFT$$

**Data Transfer** — Read input $x$ (shape $BTD$) + read 3 weight matrices (all shape $DF$) + write back intermediate and final results (two intermediate results $BTF$ element-wise multiplied produce $BTF$, final output $BTD$). Only counting HBM-related matrix multiplication input/output:

$$\text{Bytes Transferred} = 2(BTD + 3DF + 2BTF + BTD) = 4BDT + 6DF + 4BFT$$

*✍️Note: Each term multiplied by 2 because bf16 is 2 bytes. $BTD$ appears twice corresponding to reading input and writing final output, $2BTF$ corresponds to gate and up two intermediate result write-backs, used for element-wise multiplication.*

$$\text{AI}_{\text{SwiGLU}} = \frac{6BDFT}{4BDT + 6DF + 4BFT}$$

**b. Other GLU Variant Activation Functions (also gated)**

All GLU-type activation functions have the structure $\text{GLU}(x) = \sigma(xW_{\text{gate}}) \odot (xW_{\text{up}})$, followed by $W_{\text{down}}$ down-projection. Therefore, they also have 3 weight matrices and 3 matrix multiplications, with FLOPs identical to SwiGLU and Bytes Transferred also identical.

**c. Standard FFN Activation (non-gated, 2 weight matrices)**

Computation principle: $\text{FFN}(x) = \text{ReLU}(xW_1) \cdot W_2$, with only 2 matrix multiplications and 2 matrices:

$$\text{FLOPs} = 2 \times 2BDFT = 4BDFT$$
$$\text{Bytes Transferred} = 2(BTD + 2DF + BTF + BTD) = 4BDT + 4DF + 2BFT$$

$$\text{AI}_{\text{FFN}} = \frac{4BDFT}{4BDT + 4DF + 2BFT}$$

**📖 Simplifying MLP Arithmetic Intensity**

Taking SwiGLU as example, dividing denominator terms by $2DF$:

$$\text{AI}_{\text{SwiGLU}} = \frac{3BDFT}{2BDT + 3DF + 2BFT}$$

When $D \gg B$ and $F \gg B$, the $3DF$ term in the denominator is far larger than terms containing $BFT, BDT$, approximately:

$$\text{AI}_{\text{MLP}} \approx \frac{3BDFT}{3DF} = BT$$

Standard FFN is similar: $\text{AI} \approx 4BDFT / 4DF = BT$.

>**💡Conclusion** — When model dimensions are far larger than batch size, the arithmetic intensity of the MLP part is approximately $BT$.

---

MLP's primary reuse comes from input tokens being reused across multiple output channels, and weight tiles being reused across multiple tokens. Therefore, in Prefill where $BT$ is large, matrix multiplication更容易 achieves higher arithmetic intensity. In Decode, each step typically has only one new token, making matrix shapes narrow, and the fixed cost of weight reads/writes难以被充分摊薄. With this clarified, below we analyze Attention, which除了 Q/K/V and output projection, must also handle the core attention that grows with context length $S$.

② **Attention Part** contains two types of computation: linear projections (computing Q/K/V, output projection) and core attention ($QK^T$ score computation + $\text{Attn} \times V$ weighted summation). Linear projection and core attention each have two different cases: MHA and GQA/MQA.

**🌟 Linear Projection Part:**

**a. MHA Standard Multi-Head Attention, $H_{kv}$ = H**

Projection matrices $W_Q, W_K, W_V, W_{out}$ are all $D \times D$, 4 in total:

$$\text{FLOPs}_{\text{proj}} = 4 \times 2BTD^2 = 8BTD^2$$
$$\text{Bytes Transferred}_{\text{proj}} = 2(BTD + 4D^2 + BTD) = 4BDT + 8D^2$$

Reading input $x$ ($BTD$) + 4 weight matrices ($4D^2$) + writing back output ($BTD$):

$$\text{AI}_{\text{proj, MHA}} = \frac{8BTD^2}{4BDT + 8D^2} = \frac{2BTD}{BT + 2D}$$

When $D \gg BT$, $\text{AI} \approx BT$; when $BT \gg D$, $\text{AI} \approx 2D$.

**b. GQA/MQA Grouped Query, $H_{kv}$ < H**

K and V projection matrices变为 $D \times D_{\text{kv}}$, where $D_{\text{kv}} = (H_{\text{kv}} / H) \cdot D$. Q and O projections remain $D \times D$.

$$\text{FLOPs}_{\text{proj, GQA}} = 2 \times 2BTD^2 + 2 \times 2BTD \cdot D_{\text{kv}} \cdot H = 8BTD^2$$

$$\text{Bytes Transferred}_{\text{proj, GQA}} = 2(BDT + 2D^2 + 2DD_{kv} + BDT) = 4BDT + 4D(D + D_{kv})$$

$$\text{AI}_{\text{proj, GQA}} = \frac{8BTD^2}{4BDT + 4D(D + D_{kv})}$$

Weight parameter count is reduced, thereby **reducing the data volume of reading weights from HBM**. GQA's advantage mainly体现在 reducing KV Cache memory footprint and bandwidth consumption, **not in reducing FLOPs of the linear projection part**.

**🌟 Core Attention Part:**

Input — $Q$ shape $(B, H, T, d_h)$, $K$ shape $(B, H_{\text{kv}}, S, d_h)$, $V$ shape $(B, H_{\text{kv}}, S, d_h)$, where $d_h = D/H$.

**a. MHA Impact on Core Attention**:

- $Q K^T$: $(B, H, T, d_h) \times (B, H, d_h, S) \to (B, H, T, S)$, $\text{FLOPs} = 2BHTSd_h = 2BTSD$;
- $\text{Attn} \times V$: $(B, H, T, S) \times (B, H, S, d_h) \to (B, H, T, d_h)$, $\text{FLOPs} = 2BHTSd_h = 2BTSD$.

$$\text{FLOPs}_{\text{core, MHA}} = 4BTSD$$

Data transfer (read Q/K/V from HBM, write back output):

$$\text{Bytes Transferred}_{\text{core}} = 2(BTD + BSD + BSD + BTD) = 4BDT + 4BDS$$

Reading Q ($BTD$) + reading K cache ($BSD$) + reading V cache ($BSD$) + writing back output ($BTD$).

$$\text{AI}_{\text{core, MHA}} = \frac{4BTSD}{4BDT + 4BDS} = \frac{TS}{T + S}$$

*✍️Note: The arithmetic intensity of the core attention part is independent of $D$, depending only on sequence length*.

**b. GQA/MQA Impact on Core Attention**:

Under GQA, KV Cache size shrinks to $(B, H_{\text{kv}}, S, d_h)$. **FLOPs unchanged because Q still has H heads**, but Bytes Transferred for reading KV Cache is reduced:

$$\text{Bytes Transferred}_{\text{core, GQA}} = 2(BTD + 2B \cdot H_{\text{kv}} \cdot S \cdot d_h + BTD) = 4BDT + 4B S D_{\text{kv}}$$

$$\text{AI}_{\text{core, GQA}} = \frac{4BTSD}{4BDT + 4BSD_{\text{kv}}} = \frac{TS}{T + S \cdot (D_{\text{kv}}/D)}$$

GQA, by reducing KV Cache transfer volume, improves the arithmetic intensity of the core attention part, making the Decode stage more likely to approach compute-bound.

**📖 Simplifying Attention Arithmetic Intensity**

Combining linear projection and core attention (using MHA as example):

$$\text{FLOPs} = 8BTD^2 + 4BTSD$$
$$\text{Bytes Transferred} = (4BDT + 8D^2) + (4BDT + 4BDS) = 8BDT + 8D^2 + 4BDS$$
$$\text{AI}_{\text{Attention}} = \frac{8BTD^2 + 4BTSD}{8BDT + 8D^2 + 4BDS}$$

---

The formulas above give approximate values for individual `kernels`, but actual inference also depends on which phase the request is in. Prefill processes a known input segment at once; Decode processes only the newly generated token each time — i.e., the two have different $T$, $S$, matrix shapes, and data reuse patterns, so they must be discussed separately.

The key difference between **Prefill and Decode** phases lies in the values of $T$ and $S$:

| Phase | Meaning of T | Meaning of S |
| ------------------- | --------------------------- |------------------------ |
| **Prefill** | Length of input Prompt | Equals T (self-attention, Q/K/V from unified input) |
| **Decode** | Number of tokens generated per step | Total length of already-generated context |

**Prefill Phase (T = S)**

When $D \gg B, F \gg B$:

MLP arithmetic intensity: $\text{AI}_{\text{MLP}} \approx BT = BS$

Attention core part arithmetic intensity for both cases, substituting $T = S$: $\text{AI}_{\text{core}} \approx \frac{S \cdot S}{S + S} = \frac{S}{2}$

Similarly, Attention linear projection part arithmetic intensity for both cases: $\text{AI}_{\text{proj}} \approx BT = BS$.

For long sequences S, arithmetic intensity is high, and Attention can also be considered compute-bound.

>💡Prefill Phase Conclusion: As long as batch size is sufficiently large or sequence is sufficiently long, and model computation dimensions are far larger than batch size, the main performance bottleneck is compute-bound — this is why Prefill phase throughput is primarily limited by GPU compute power.

**Decode Phase (T = 1, considering only per-token generation)**

When $D \gg B, F \gg B$:

MLP arithmetic intensity: $\text{AI}_{\text{MLP}} \approx B \times 1 = B$

When B = 1, arithmetic intensity is only 1, far below the GPU's $\frac{\text{peak compute}}{\text{bandwidth ratio}}$, making it memory-bound. Increasing batch size can improve arithmetic intensity — this is one core motivation for continuous batching.

Attention core part arithmetic intensity for both cases, substituting $T = 1$: $\text{AI}_{\text{core}} = \frac{1 \times S}{1 + S} = \frac{S}{S + 1} \approx 1 \quad (\text{when } S \gg 1)$, arithmetic intensity approaches 1, meaning per byte read only about one floating-point operation — memory-bound.

Attention linear projection part arithmetic intensity for both cases: $\text{AI}_{\text{proj}} \approx B$, similar to MLP, memory-bound when B is small.

>💡Decode Phase Conclusion: In typical inference scenarios (B=1 or small), most performance bottlenecks are memory-bound. Therefore, in the Decode phase, the performance bottleneck primarily lies in HBM bandwidth — the model needs to read all weight parameters and KV Cache from memory, doing only极小量 computation per read.

This is why the core directions of inference optimization include:
- Increasing batch size (continuous batching): improves MLP and projection arithmetic intensity;
- GQA/MQA: reduces KV Cache read volume, improves core attention arithmetic intensity;
- Quantization (INT8/INT4): reduces weight and KV Cache byte count;
- Model parallelism: leverages more aggregate HBM bandwidth.

---

*A thought-provoking question: why can increasing batch size typically improve arithmetic intensity, thereby improving LLM compute throughput, but does this benefit have an upper bound, and what factors determine that boundary?* This can be analyzed from 3 perspectives:

① **Compute Saturation and the "Carpool Effect"**

Increasing batch size $B$ is a key means of moving GPU from "idle" to "saturated." Think of GPU inference as a large bus: model weights are the heavy vehicle body, input data are passengers. If you only carry one passenger per trip ($B=1$), you drag tens of GB of vehicle body each trip — very wasteful. If you fill $B$ passengers in one trip, the fixed cost is amortized. At the底层, this means matrix operation parallelism is fully utilized, arithmetic intensity显著 increases — each data搬运 from memory enables more computation, fully leveraging GPU parallel capability and improving overall throughput.

```
LLM Inference Stage (Autoregressive Generation + KV Cache)

          ┌─────────────┐
          │ Input Queue     ← Multiple requests
          └─────┬───────┘
                │  
        ┌───────┴─────────┐
        │ Batch requests   ← Improve overall throughput
        └───────┬─────────┘
                │
        ┌───────┴─────────┐
        │  GPU Compute    
        │  ┌─────────┐    
        │  │token_1  │<── K[K_1] V[V_1]
        │  └─────────┘  
        │  ┌─────────┐   
        │  │token_2  │<── Must wait for Token_1
        │  └─────────┘
        |       |
        |       ↓   KV Cache updated to K[K1,K2] V[V1,V2]
        │  ┌─────────┐
        │  │token_3  │<── Must wait for Token_2
        │  └─────────┘
        |       |
        |       ↓   KV Cache updated to K[K1,K2,K3] V[V1,V2,V3]
        |    ......
        └───────┬─────────┘
                │
        ┌───────┴─────────┐
        │   KV Cache       │  ← Stores key-value pairs of generated tokens
        │   token_1...N    │  ← Each generation step must access all history
        └───────┬─────────┘
                │
          Single request latency ↑
          (Serial generation + memory burden)

Notes:
1. Dynamic batching improves system throughput, but single-request latency remains limited by KV Cache access.
2. As sequence length grows, KV Cache gets larger, latency increases.
3. Each new token is generated by directly appending its KV (key-value pair) to the existing KV Cache without creating new caches.
```

② **Serial Curse and Memory Burden**

However, the autoregressive generation nature of LLMs决定了 it cannot be fully parallelized like traditional neural networks, and also differs from the training stage where tokens can be computed in parallel. Regardless of batch size, the generation process is like "idiom接龙": you must write the previous character before deciding the next. More troublesome is that to maintain context coherence, every time GPU generates a token, it must access all previous tokens' states. KV Cache stores each generated token's key-value pairs for the next attention computation step. As sequence length increases, this "memory bank" grows larger, turning原本轻量 computation into frequent查阅 historical information burden, severely impacting single-request latency.

③ **Memory Wall and Speed Bottleneck**

This leads to the ultimate inference bottleneck — the memory wall. During the decoding stage, GPU compute units often idle waiting for memory to搬运庞大的 KV Cache data, while actual computation time is very short. Although dynamic batching can improve overall throughput by processing multiple requests simultaneously, this only means "the bus carries more passengers per trip" — it doesn't make the bus itself faster. For individual users, their generation latency remains constrained by memory bandwidth: no matter how strong GPU compute is, if data read speed can't keep up, generation speed cannot improve.

*During the decoding stage, the model selects one token as the generation result based on the output vocabulary probability distribution (e.g., taking maximum probability, sampling by distribution, etc.), and uses it as下一步 input to continue generating.*

>In summary, batch $B$ can significantly improve overall system throughput, but autoregressive generation and ever-growing `KV Cache` determine that single-request latency is ultimately constrained by memory bandwidth. As generation progresses, even if the system can still batch multiple requests together, each single request's own per-step decoding computation still only processes 1 `token`, and as KV Cache grows larger, its arithmetic intensity continuously decreases, performance gradually approaching the effective computation intensity when processing alone (i.e., $B \approx 1$).

Arithmetic intensity analysis explains "why different stages are constrained by different hardware resources," but the model performance we perceive must also be described through TTFT, ITL/E2E Latency, and Throughput; simultaneously, scheduling strategies organize单请求's Decode into system-level batch processing. Therefore, below we connect kernel-level bottlenecks with service-level metrics.

### 10.2.3 Latency vs. Throughput

After analyzing arithmetic intensity, we also need to understand its performance impact in actual LLM autoregressive generation inference scenarios. Latency and Throughput are two core metrics measuring LLM inference performance.

**Core Performance Metrics**

| Metric | Optimization Goal |
| ------------------- | --------------------------- |
| **Latency** | Reduce, to improve user interaction **smoothness** |
| **Throughput** | Increase, to improve system overall **processing capacity** and **resource utilization** |

Notably, pursuing极致 latency typically targets single-request optimization; pursuing极致 throughput typically倾向于 increasing batch size. But batch size is not the sole determinant of throughput or latency — large batches不一定 bring high throughput, and small batches不必然 mean low latency. Actual results also depend on `GPU` utilization, request `length distribution`, `KV Cache efficiency`, and `scheduling strategy`, among multiple factors.

**Batch Processing**

After KV Cache alleviates computation and bandwidth pressure, GPU core utilization improves, and throughput can be further optimized through batch processing.

**Batch Processing Methods**

1. **Static Batching**: Pack a fixed number of requests or tokens.
2. **Dynamic/Continuous Batching**:
   - Immediately pack the next request while waiting to generate new tokens.
   - **Multi-request batch assembly**: requests with different lengths and progress dynamically combined, improving GPU parallel utilization.

```text
Request 1: T1,T2,T3 | Request 2: T1,T2 -> Dynamically assembled into Batch -> GPU parallel computation
```

**Benefit**: Improves overall throughput, particularly suitable for online serving with varying request lengths.

**Trade-offs**

| Goal | Trade-off Point | Description |
| ------------ | ---------- | -------------------------------------- |
| **Latency vs. Throughput** | Batch size | Increasing batch size → throughput ↑, but single-request latency may ↑ |
| **Efficiency vs. Precision** | Optimization technique selection | KV Cache or quantization optimizations must maintain **autoregressive causality** and **semantic accuracy** |
| **Memory vs. Performance** | KV Cache usage | Grows linearly with sequence length, constraining maximum sequence length or batch size |

**Summary**:
1. KV Cache can trade additional memory and bandwidth overhead for lower redundant computation cost, but when its scale is too large, memory pressure offsets its benefits,反而 reducing generation speed.
2. Batch processing improves overall throughput.
3. Combining both achieves engineering balance between high throughput and low latency.
4. Engineering optimization must consider practical factors like memory, sequence length, and request length distribution.

Among these, batch processing primarily improves resource utilization during the generation stage, while input prompt length determines Prefill workload, initial KV Cache size, and the历史 length to be read at each subsequent step. Thus, prompt compression becomes another optimization path directly作用于 input scale and KV Cache.

### 10.2.4 Prompt Compression

In autoregressive LLMs, user-input `prompts` are not "free" — the model first tokenizes the prompt and maps it to embeddings, then computes corresponding Key and Value at each layer and stores these intermediate KVs in cache for subsequent stepwise generation reuse. Therefore, overly long or excessively detailed prompts not only occupy大量 memory (especially as context windows grow longer), but also increase per-token attention computation, thereby延长 latency and reducing overall inference efficiency. Additionally, excessive, redundant, or irrelevant information may "dilute" the model's attention allocation, causing output quality not to improve linearly with context length but反而 worsen. To shorten prompts while retaining key information, common prompt compression strategies fall into two categories:

**1. Hard Prompt Compression**

Hard prompt compression aims to simplify natural language prompts by `shortening length` or `reducing complexity` while still effectively guiding the model to generate desired responses. Common methods include two types:

<div align="center">
<img width="1500" height="400" alt="Hard prompt compression" src="https://github.com/user-attachments/assets/22a311db-e9e8-4560-a1c5-faccc8e2fd15" />
  <p>Figure 10.6 Hard Prompt Compression</p>
</div>

- Selecting key content from `prompt`:
    - **Sentence/token relevance filtering**: Using sentence encoders or text embeddings, match prompt sentences or tokens with query content, sort by relevance, and remove low-relevance portions; for large LLMs, appropriately reduce token count based on model capacity limits, retaining the most semantically relevant tokens.
    - **Dynamic token count control**: Train a token controller via reinforcement learning to dynamically adjust prompt length based on query difficulty, balancing efficiency and output quality.

- Rewriting and compressing prompts:
    - **Concise expression**: Compress冗长 prompts into shorter, clearer formulations while遵守 generation length constraints and preserving original sentence semantics;
    - **Preserving core context**: Through dynamic attention or document summarization strategies, extract the most important content from large volumes of documents and融合 it into simplified prompts.

**2. Soft Prompt Compression**

Soft prompts摆脱 dependence on discrete tokens by learning a set of continuous vectors to replace or supplement natural language prompts. During `training`, these vectors serve as learnable parameters, guided by backpropagation to produce desired outputs; during `inference`, these vectors can directly serve as prefixes for Transformer input or be injected into the attention mechanism in specific ways, without needing to map back to discrete tokens, directly participating in computation in embedding form. Common soft prompt strategies fall into two categories:

<div align="center">
<img width="1300" height="600" alt="Soft prompt compression" src="https://github.com/user-attachments/assets/bacdfd83-ea04-4dc4-ab5b-3b325c27c816" />
  <p>Figure 10.7 Soft Prompt Compression</p>
</div>

- Methods with frozen LLM parameters: By `optimizing input` representations while **keeping original LLM parameters frozen**, provide diverse prompt compression strategies. These methods significantly improve computational efficiency and memory utilization, enabling LLMs to be more efficiently applied to a wide range of task scenarios.
- Keypoint-token-based methods: By `transforming context into compact and reusable tokens`, **provide a powerful framework for input length reduction**. These methods bring significant efficiency improvements but typically require updating LLM parameters to achieve effective compression and integration.

The advantages of soft prompts are **small parameter count, reusability**, and achieving good performance **without fine-tuning model主体 weights**. However, it's worth noting:

- **Inference overhead**: Soft prompts don't automatically reduce inference computation because they still need to participate in the entire model's forward propagation and attention computation;
- **Interpretability**: Soft prompts are continuous vectors optimized to引导 model generation of specific outputs while keeping the LLM主体 frozen, but the vectors themselves don't correspond to any readable words or semantic concepts. Each vector exists in high-dimensional space, with combinations across different dimensions jointly作用于 model behavior, so interpretability is low and unintuitive;
- **Generalization ability**: Soft prompts are typically learned for specific tasks or training data, and引导 effects are relatively sensitive to out-of-training-distribution cases, with generalization ability typically不如 explicit natural language prompts or fine-tuning of model主体, especially sensitive to `prompt` style variations.

>From an inference mechanism perspective, both hard and soft prompts extend input sequence length and participate in self-attention computation. The difference is that hard prompts are obtained by mapping discrete tokens through embeddings, while soft prompts directly inject additional learned continuous vectors (without changing LLM original structure) into the model input representation.

**3. Visual-Level Prompt Compression**

<div align="center">
<img width="1470" height="450" alt="DeepSeek-OCR structure" src="https://github.com/user-attachments/assets/43b171ec-1332-44ed-9757-8bc933a627ec" />
  <p>Figure 10.8 DeepSeek-OCR Structure</p>
</div>

Visual-level prompt compression can be divided into three steps:

**Step 1 Text → Image**: Convert原始 text prompt into an image, which can be a text-rendered image or a specially visually-encoded image representation;

**Step 2 Model Reading**: Extract information from the image through `OCR` or visual encoding modules like `CLIP`'s text encoder or specialized image-text understanding modules;

**Step 3 Compression Effect**: Visual encoding compresses long text into fixed-dimension image representations, thereby reducing computation pressure from `sequence length` or `token count` on autoregressive models.

# 10.3 Related Research

The preceding discussion primarily围绕 standard autoregressive Transformers: using KV Cache to reduce redundant computation, using GQA/MQA, operator fusion, batching, and prompt compression to reduce data movement or scheduling costs. But these methods mostly retain the basic paradigm of "token-by-token generation." Next, we转向 model structure and generation paradigm itself, observing whether bottlenecks can be改变 from perspectives of memory form, parallelism method, or large-small model collaboration.

This section organizes along three threads: State Space Models replace explicit long KV Cache with continuous states; Diffusion Models放松 token-level autoregressive constraints in exchange for intra-block parallelism; Draft models with Speculative Decoding retain the target model's autoregressive semantics while amortizing the large model's serial decoding cost through "small model drafts, large model verifies." They all pursue faster inference but bear different costs in quality, training cost, and system complexity.

## 10.3.1 State Space Models

<div align="center">
<img width="1100" height="350" alt="State Space Model" src="https://github.com/user-attachments/assets/762ea6e1-4f6e-4399-8a71-729a4d328cdf" />
  <p>Figure 10.9 State Space Models</p>
</div>

State Space Models (SSM) can be viewed as an extension of traditional `RNN`, excelling in signal processing and long sequence modeling. In recent years, research has attempted to apply `SSM` to natural language context modeling, particularly advantageous in capturing long-range dependencies. Unlike autoregressive Transformers that depend on explicit KV caching, SSM achieves **modeling of long-term dependency information** through continuous updating of hidden states and weight matrices learned during training. On long sequence processing, SSM's computational complexity is lower than standard autoregressive Transformers, so speed is typically faster.

However, some SSM implementations may still face challenges in maintaining long-term dependencies when processing extremely long sequences. Specifically, during training, hidden state backpropagation may experience gradient vanishing or explosion, leading to long-term information attenuation, thereby affecting overall model performance in certain text generation or understanding tasks. To improve SSM's capability in long-term dependency modeling, new linear attention mechanisms have emerged, such as:

- Mamba: Optimizes state matrices and numerical stability on top of standard SSM, improving long-sequence modeling capability.
- Other RNN-based methods (like LSTM): Not direct SSM variants, but solve long-term dependency problems through gating mechanisms.
- S5: Building on S4, S5 discretizes continuous-time state spaces so that convolution kernels can efficiently act on the entire sequence. This discretization method is very effective in NLP tasks, while通过 adjusting output structure and state update methods,一定程度上缓解 the problem of long-term dependency information attenuation.

SSM's core change is "how to保存 history": it compresses historical information into recursively updated states, thereby eliminating dependence on explicit KV Cache. But it still needs to process sequential states (through stored hidden intermediate states). Next, we further change "how to generate": diffusion models attempt to recover multiple tokens in parallel within the position dimension or within blocks, reducing the serial waiting from strict token-by-token decoding.

## 10.3.2 Diffusion Models

<div align="center">
<img width="1200" height="230" alt="Diffusion Model" src="https://github.com/user-attachments/assets/a99d408f-3ae5-4605-a65f-62bb05af88dc" />
<p>Figure 10.10 Diffusion Models</p>
</div>

As mentioned in 10.2.1 Transformer, although diffusion models can also store some KV state information that doesn't change across diffusion updates, because diffusion models can generate multiple tokens in parallel across spatial positions, their inference speed is很大程度上不受 memory constraints, especially with clear advantages in processing large-batch generation tasks.

However, diffusion model design excels at processing spatial dimension information, while natural language is essentially a time series with strict causal dependencies. This leads to diffusion models容易出现问题 when generating long texts or content with logical dependencies — for example, generating a code snippet may be very fast but容易出现明显的 logical or syntactic errors. Research has found that diffusion language models preferentially commit **high-confidence tokens** during decoding while systematically deferring high-entropy **low-confidence tokens** — especially logical connectives (like "because," "therefore," "but") — to later stages. This confidence-sorted generation approach triggers two problems:

- **Confidence Shortcut**: Individual tokens appear highly reasonable in local context, but since their true long-range dependencies haven't been resolved yet, combinations produce logical errors — i.e., the model "sees locally correct shortcuts" and prematurely commits, ignoring global constraints.

- **Answer-First Reasoning**: The model倾向于提交 final answers before the reasoning chain has unfolded, causing subsequently generated reasoning processes to easily become post-hoc rationalization of existing answers rather than genuine logical derivation, with causal inversion between reasoning chain and conclusion.

In other words, diffusion models have speed advantages in completing inference, but may不如 autoregressive models in ensuring context consistency and causal logic. Therefore, challenges in applying diffusion models to LLMs include:

1. **Insufficient Context Consistency**
   - Diffusion models are essentially non-autoregressive (NAR) parallel generation,难以严格保证 long sequence causal dependencies and global semantic coherence.
   - For LLM tasks, especially long text generation or complex reasoning, purely parallel-generated tokens容易 appear logically or semantically inconsistent.

2. **Discrete Token Alignment Problem**
   - Traditional diffusion models rely on predicting noise or scores, then mapping back to discrete tokens, which introduces additional bias in token distribution space.
   - If directly using diffusion-generated sequences as LLM input, token alignment and probability distribution matching problems must be handled.

3. **Inference Efficiency and KV Cache Scheduling**: Although diffusion model parallel generation can reduce per-token forward computation, when combined with autoregressive models, appropriate proposal-verify flows must be designed to ensure KV Cache调用次数 significantly reduced, otherwise high latency or memory pressure may still occur.

**LLaDA 2.0: First to Scale Diffusion Language Models to Hundred-Billion Scale — Development Process**

**1. Masked Denoising Language Models (MDLMs) Background**

Masked denoising language models offer a new approach to text generation. Unlike AR models, MDLMs treat text generation as an **iterative denoising process**: in each forward pass, some tokens are randomly masked, and the model's task is to recover原始 tokens based on unmasked context.

This paradigm shift led researchers to尝试 **training MDLMs from scratch** to explore their potential. For example, `LLaDA` showed that a fully from-scratch-trained 8B-scale MDLM can rival equivalently-sized autoregressive models in expressive capability. Further, `LLaDA-MoE` first introduced **Mixture of Experts architecture** into MDLM, with results showing MoE-based MDLMs outperform dense models in both efficiency and capability, proving **MDLM compatibility and scalability with advanced MoE architectures.**

Notably, since `MDLM` and `AR` models have fundamental differences in training dynamics, AR model常用的 training strategies and hyperparameters often不适用 for MDLMs. To this end, research works like Quakka and OpenMoE2 specifically studied MDLM scaling characteristics and training strategies, providing systematic guidance for the emerging denoising generation paradigm.

**2. Initializing Denoising Models Using Autoregressive Models**

Considering AR models' powerful knowledge capacity and excellent performance, recent research has attempted to **initialize MDLMs or DLMs with pretrained AR models** to reduce training cost and narrow performance gaps, for example:

- **DiffusionLLaMA** and **Dream-7B**:
  - During training, adopt **mask annealing strategy**, gradually transitioning causal attention to bidirectional attention.
  - Simultaneously combine CART-based loss reweighting schemes to balance token-level learning dynamics.

- **RND1**
  - At initialization阶段, directly convert AR model's causal attention to bidirectional attention.
  - To preserve original AR model's knowledge-intensive capabilities, restrict dense layer parameter updates during training to prevent catastrophic forgetting.

**3. Block Diffusion Language Models (BDLMs)**

Block diffusion language models propose a **hybrid generation paradigm**, combining diffusion modeling with autoregressive modeling:

- **Within blocks**: Recover masked tokens through diffusion processes.
- **Between blocks**: Generate sequentially in autoregressive fashion.
- Advantages:
  - Support variable-length text generation.
  - Can reuse KV-cache, improving inference efficiency.

BDLMs initialized from AR models, such as **SDAR**, can achieve performance comparable to AR base models under different block sizes and optimization strategies.

**Block Definition Explanation**

>In LLMs引入的 diffusion models, so-called "blocks" refer to strictly consecutive segments of tokens along the position dimension of the text sequence, whose partitioning purpose is to introduce parallel denoising while maintaining language causal structure. This concept differs from Transformer blocks based on semantic similarity, structural information, or computational efficiency.

**Diffusion Block Partitioning Illustration**

```
token indices:  1    2    3    4    5    6    7    8    9    10   11
                │    │    │    │    │    │    │    │    │    │   │
segments:       x1   x2   x3   x4   x5   x6   x7   x8   x9  x10  x11
               └───────condition input (not参与 diffusion)───────┘   └──── block ────┘
```
*Tokens within the same block are processed in parallel (simultaneously generated/modified).*

**4. Current Limitations**

Although these methods perform well at small to medium scales (7B~30B), several challenges remain:

- **Limited scalability**: The feasibility of training AR-initialized diffusion models at ultra-large scales has not been充分验证.
- **Low training efficiency**: Block diffusion training involves many steps, limiting efficiency for training ultra-large models on large-scale corpora.

**5. LLaDA 2.0 Training Paradigm**

The research does not train diffusion models from scratch, but rather adopts diffusion-style input data during AR model training — i.e., AR transfer learning converted to diffusion models. This paradigm conversion has 3 stages: (1) Continued pretraining from AR to full-sequence masked diffusion models; (2) Block diffusion pretraining, achieving transition from token-level to block-level diffusion modeling; (3) Subsequent supervised fine-tuning and direct preference optimization to achieve alignment and task-specific optimization.

<div align="center">
  <img width="1300" height="560" alt="LLaDA 2.0 training process" src="https://github.com/user-attachments/assets/ebe6c0af-ed4f-4c90-a5e2-547b44d21dce" />
  <p>Figure 10.11 LLaDA 2.0 Training Process Diagram</p>
</div>

**Step 1: Progressive Block Size Warmup**

The core idea of the progressive block size warmup phase is **to guide the model to gradually adapt to larger-scale parallel denoising and context modeling capability by progressively increasing the generation block size**.

During early training, the model works in a **near-autoregressive** manner (block size small, in extreme cases equivalent to a single token), primarily learning stable prediction under strict causal constraints (following time sequence). As training progresses, block size gradually increases, and the model begins **predicting multiple tokens in parallel within blocks** while still maintaining **autoregressive order between blocks**. This process can be understood as:

- **Intra-block receptive field gradually expands**: As block size increases, the model needs to **jointly model multiple consecutive position tokens within the same block**. These tokens are typically in noise-perturbed, masked states during training; the model performs **parallel prediction and consistency modeling** across multiple positions within a block through one forward pass, no longer局限于 single-token stepwise generation.
- **Attention structure gradually transitions**: Early in training, the model primarily follows **strict token-level autoregressive causal attention**; as blocks expand, the model **allows non-causal (bidirectional) attention or denoising-style modeling within blocks** to fully leverage intra-block context information for joint recovery. Meanwhile, **autoregressive causal order is maintained between blocks**, thus ensuring long-range dependency modeling stability while training the model with *think-while-correcting* capability.

Through this progressive training strategy, the model doesn't need to directly face the difficult task of large-scale parallel denoising early on, but rather **smoothly transitions from autoregressive modeling to diffusion, denoising-style representation learning**. This helps:

- Stabilize the training process, avoiding optimization instability from large `block` sizes;
- Gradually improve the model's modeling capability for **long context, complex mask patterns, and intra-block joint consistency**;
- Make AR-initialized model weights more naturally adapt to subsequent diffusion model generation paradigms.

>Since tokens between different blocks may be semantically unrelated, but the attention mechanism still attempts to assign weights, this creates a risk of forming spurious dependency relationships in `cross-block` modeling for diffusion models, potentially causing semantic confusion and `bidirectional attention mechanism` training instability. To avoid such cross-block interference, we introduce document-level attention masking, restricting self-attention to within individual blocks, thereby ensuring context modeling consistency and training stability.

**Step 2: Stable Training of the Stage 1 Model**

After completing Stage 1's large-scale training, when block size is fixed at 4096 and the model switches to MDLM mode, each block occupies a certain amount of `KV Cache`. At this point, information with low relevance to the current block in cross-block attention computation can be masked, significantly reducing computational overhead, enabling data to be more efficiently processed under the MDLM paradigm.

As the model adapts to this mode, the stable training phase's focus shifts to extensive training on filtered large-scale corpora, **to further improve the Stage 1 model's diffusion representation capability**. In this phase, each input sequence is processed as a single block, with block size fixed at 4096, allowing intra-block attention to cover the entire sequence, thus ensuring complete intra-block context modeling, equivalent to the classic MDLM setting.

**Step 3: Global Attention Transformed to Local Attention**

After completing the first two stages of large-scale `MDLM` training, the model has learned global context information. Next, by gradually shrinking block size from 4096 to smaller values, the model progressively transitions from global block conditioning to local block conditioning `BDLM` mode. Smaller block size means attention computation only occurs within individual blocks, significantly reducing memory consumption and accelerating inference speed while retaining local intra-block context information for semantic understanding. To not completely lose long-distance dependencies, the model retains some global attention during衰减, enabling it to一定程度上 capture cross-block relationships, thereby preserving long-distance text modeling capability while improving efficiency.

>BDLM and MDLM are both attention strategies in diffusion language models. MDLM emphasizes capturing global dependencies, supporting intra-block and cross-block global attention for modeling long-range relationships; BDLM focuses on local context, primarily restricting attention to within individual blocks, thereby significantly improving computational efficiency and reducing memory overhead under necessary causal constraints.

**LLaDA 2.0 Achievements**

In the larger model LLaDA2.0-flash, this potential is particularly evident. The model achieves an average score of 73.18, comparable to top AR models like Qwen3-30B-A3B-Instruct-2507 (73.60). More importantly, LLaDA2.0-flash outperforms comparable AR models in `complex generation tasks`, `coding capability`, `agent capability`, and advanced mathematics, **demonstrating the potential advantages of diffusion architectures**.

Notably, the `LLaDA2.0` series demonstrates the enormous potential of diffusion-based language models. They can not only serve as scalable alternatives to AR but also be compatible with AR. Although the gap with traditional models is rapidly narrowing on some通用 tests, in complex, structured, agent tasks like code generation and tool use, diffusion models have already shown stronger capabilities. This means that in the future, diffusion models有望 become a very promising research direction in the language generation field.

Diffusion and block diffusion illustrate a more radical direction — relaxing token-level causal order in exchange for parallel efficiency. But this also brings problems like discrete token alignment, context consistency, and KV Cache scheduling. If we want to保留 the target model's autoregressive distribution while minimizing the target model's per-token调用次数, we naturally think of speculative decoding.

## 10.3.3 [Speculative Decoding](https://research.google/blog/speculative-cascades-a-hybrid-approach-for-smarter-faster-llm-inference)

<div align="center">
  <img width="1200" height="370" alt="Large-small model collaboration" src="https://github.com/user-attachments/assets/f212f5c3-411d-4420-8774-3b4b3f1212df" />
<p>Figure 10.12 Large and Small Model Collaboration</p>
</div>

Large language models achieve显著 results on some text tasks at the cost of inference `latency`, hence the idea of large-small model collaboration to solve the latency problem while尽可能 maintaining capability. There are two strategies for large-small model collaboration; although both address the same problem, they are fundamentally different:

 - **Speculative Decoding**: The small model autoregressively generates a segment of candidate token sequence at once; the large model performs parallel forward computation on this sequence and verifies its consistency with the target distribution. If inconsistency occurs at some position, accept the consistent token prefix up to that point, `rollback` from that position, and the large model takes over generation. This process repeats until the problem is completed.
 - **Cascade Decoding**: The small model performs one forward computation for the current generation step, and based on its **predicted token distribution uncertainty (e.g., maximum probability or entropy)** before generating the token, decides whether this step's token generation is done by the small model directly or交由 the large model to generate with a confidence level. The entire process is: **small model predicts whether the relevant probability reaches a threshold — if reached, continue letting the small model generate tokens according to prediction; otherwise, the large model重新 responds by computing and generating the token**. Repeat until the problem is completed.

>One-sentence summary: Speculative decoding reduces the large model's serial decoding steps through "parallel verification of generated token consistency"; cascade decoding reduces the large model's调用 frequency through "risk-aware routing based on token distribution uncertainty."

![10](https://github.com/user-attachments/assets/e4354763-8790-4261-a85f-f26909d31d61)

*Animation demonstration: Comparison of speculative cascade and speculative decoding methods on an arithmetic problem from the GSM8K dataset ("Mary has 30 sheep total. She collects 1kg of milk daily from half of them and 2kg of milk from the other half. How much milk does she collect daily?"). Draft tokens shown in yellow, verified tokens in red. The speculative cascade method arrives at the correct answer faster than speculative decoding.*

Google's team, analyzing both speculative decoding and cascade decoding strategies, found: speculative decoding, by allowing the small model to generate draft sequences that may deviate from the target distribution and having the large model perform parallel acceptability verification, significantly reduces the large model's serial decoding overhead. Cascade decoding, through routing mechanisms, judges before generation whether the current step needs to invoke the large model to ensure final generation quality不低于 the large model itself, but this strategy, due to introducing additional decision and model switching processes, typically doesn't achieve as显著 latency optimization as speculative decoding.

<div align="center">
<img width="980" height="500" alt="Quality and rejection rate curves" src="https://github.com/user-attachments/assets/6eee5734-85aa-4e4d-a061-05461f253818" />
<p>Figure 10.13 Quality vs. Rejection Rate Curves for Gemma 2B and Gemma 27B (γ=1)</p>
</div>

Figure 10.13 shows that the large model's consistency check rejection rate for small-model-generated tokens is higher than the large model's confidence check for small-model-predicted token distributions. Therefore, using the large model's probability to judge the acceptability of small-model-generated blocks better balances quality and latency than directly comparing model distributions. Thus, after analyzing the quality-cost trade-off between speculative decoding and cascade decoding, Google's team proposed **Speculative Cascades** — this method constructs a cascade structure composed of multiple models of increasing scale. The small model autoregressively generates `draft blocks`, and a larger model performs parallel acceptability verification on the block. If verification passes, the block is directly accepted and the small model continues generating subsequent drafts; otherwise, the block is discarded and upgraded to a larger model for regeneration. Through this approach, **while ensuring final output quality is guaranteed by the largest model**, the small model is maximally utilized to complete acceptable generation, achieving a balance of quality, cost, and latency in the inference stage.

*In speculative cascades, a block refers to: a token sequence of length k continuously generated by one model during one autoregressive drafting phase.*

>Speculative cascades and cascade decoding share similar ideas; the former processes `blocks` while the latter processes `tokens`.

**Example illustrating the use of speculative cascades**
Suppose the question is: **"When was Datawhale founded?"**

1. **Small model drafts first**: Small model quickly generates candidate block, e.g.: `[Datawhale, was, founded, in, …]`, equivalent to one block as draft.
2. **Large model verifies**: Large model does forward computation on this block to judge its acceptability, rather than forcing token-by-token generation.
3. **Deferral rule判定**
   - **Accept**: Small model's generated block considered reliable → continue generating next block.
   - **Reject**: Small model's block considered unreliable → large model regenerates this block, and small model continues generating subsequent blocks while large model continues verifying.
4. **Iterative generation**: This process loops continuously until complete answer is generated, ensuring final output is guaranteed by the large model while the small model尽可能 completes acceptable generation.

> **Advantage explanation**: This "small model drafts + large model verifies + fallback" method is very flexible; deferral rules can be customized for different scenarios. For example, priority can be given to having important information reviewed by the large model while non-critical positions are directly generated by the small model, thereby **improving inference speed while maintaining answer quality**.

<div align="center">
  <img width="1300" height="390" alt="Large-small model collaboration effects" src="https://github.com/user-attachments/assets/30ecce55-25c9-46ab-b37d-91194cf3af61" />
<p>Figure 10.14 Large and Small Model Collaboration Effects</p>
</div>

*In Figure 10.14, speculative cascade variants (blue and orange) achieve better quality-latency trade-offs compared to standard speculative decoding (green star) in mathematical reasoning and summarization tasks.*

From the figure above, we can analyze that at the same token generation speed, large-small model collaboration methods are明显 better in token prediction accuracy and content order consistency evaluation, proving this method's effectiveness and providing a new approach for accelerating LLM inference.

> `Speculative Decoding` is an important optimization technique adopted by many current LLM inference frameworks, such as vLLM, DSpark, JetSpec, etc. A question worth思考: why in the same inference scenario, does a small model generate one token typically faster than a large model, with lower computational cost? This can be analyzed from **the computation and memory access overhead needed to generate one token**.
>
> During token generation, the model must sequentially pass through all Transformer Layers to complete one forward computation. Therefore, the larger the model scale, the higher the computation and memory access cost per generated token typically is. For easier understanding, let's estimate with a simplified model: assume the model has **25 Transformer Layers**, hidden dimension **128**, using **16 Heads** (each Head dimension **8**), data type **FP8 (1 Byte)**. Then, considering only per-layer hidden state reads/writes, the memory overhead estimate is approximately:
>
> $$\text{Activation} = 25 \times 128 \times 1 = 3200 \text{ Bytes } (3.2 KB)$$
>
> In actual inference, each layer also needs to read大量 model weights from Attention, MLP, and other modules and access KV Cache, so the real memory access volume is far higher than the above Activation estimate, *and grows as the LLM model expands*.
>
> **From the computation perspective**, each Transformer Layer主要由 Attention and MLP两部分组成, with computational complexity approximately:
>
> $$\text{Per Layer FLOPs} \approx 16d^2 + 2nd$$
>
> Where $d$ is hidden dimension, $n$ is current context length. Therefore, the computation for this model to generate one token can be approximately expressed as:
>
> $$\text{Total FLOPs} \approx 25 \times (16 \times 128^2 + 2 \times n \times 128)$$
>
> From the above simple estimate (actual computation varies depending on whether standard FFN, SwiGLU, MLA, or other model structures are adopted), we can see that even generating just one token requires sequentially completing forward computation through all Transformer Layers and continuously reading model weights, accessing KV Cache, and other data. As model parameter scale continuously grows, both computation and memory access required per token increase accordingly, hence inference latency also rises. This is one important reason why large models typically generate tokens slower than small models under the same hardware conditions.

# 10.4 MTP + Speculative Decoding

Whether the previously introduced speculative decoding can achieve明显 speedup depends on whether the draft model can continuously propose relatively long token blocks that are easily accepted by the target model. This进一步 prompts the question: can we, during the training stage, let the model learn the continuous conditional dependencies of multiple future tokens, providing better representation foundations for multi-token drafting during inference? MTP (Multi-Token Prediction) is precisely围绕 this question展开的.

We need to first distinguish two concepts: MTP is a prediction objective or structural design in the training stage; speculative decoding is a proposal-verify algorithm in the inference stage. MTP can provide assistance for speculative decoding but does not equal speculative decoding itself, nor does it automatically turn one forward pass into multiple tokens already verified by the target model.

## 10.4.1 DeepSeek MTP

**MTP vs. no MTP comparison:**

DeepSeek-V3's MTP is **not about using multiple independent output heads to parallelly predict future tokens based on the same backbone hidden representation**, but rather adopts a **chronologically-ordered, depth-progressive cascaded structure** — subsequent prediction depths are conditionally modeled based on the stage representation evolved from the previous stage's prediction.

*📖Tip: This progressive process occurs in continuous hidden representation space, rather than explicitly feeding previously generated `discrete tokens` back into the model for autoregressive rollout. Therefore, MTP preserves conditional dependency relationships between prediction depths at multiple time steps in the hidden representation layer, not token-level rollout in the traditional autoregressive sampling sense.*

**DeepSeek MTP vs. Traditional Parallel MTP:**

```text
DeepSeek Sequential MTP              | Traditional Parallel MTP
h_t                                  | same hidden representation h_t
 ↓                                   |   ├── predict x_{t+1} (token1)
stage1 representation                |   ├── predict x_{t+2} (token1, token2)
 ↓ predict x_{t+1}                   |   ├── predict x_{t+3} (token1, token2, token3)
 ↓                                   |   └── ...
stage2 representation                |
 ↓ predict x_{t+2}                   |
 ↓                                   |
stage3 representation                |
 ↓ predict x_{t+3}                   |                           
...                                  |
```

<div align="center">
<img width="1392" height="651" alt="DeepSeek MTP" src="https://github.com/user-attachments/assets/94b66ee4-00ad-4ec0-8c8c-f01dd268d646" />
   <p>Figure 10.15 DeepSeek MTP</p>
</div>

>DeepSeek V3 report original text: *Our principle of maintaining the causal chain of predictions is similar to that of EAGLE, but its primary objective is speculative decoding, whereas we utilize MTP to improve training.*

**DeepSeek V3 MTP itself belongs to `training objective design`, while speculative decoding belongs to `inference-stage acceleration algorithms`. The former does not directly accelerate model training but can provide more suitable representation learning foundations for subsequent multi-token speculative decoding.**

## 10.4.2 Gemma 4 MTP

<div align="center">
<img width="500" height="700" alt="Gemma 4 inference acceleration design" src="https://github.com/user-attachments/assets/b0cfdbb7-e1c9-433c-9345-634b10015f9d" />
   <p>Figure 10.16 Gemma 4 Inference Acceleration Design</p>
</div>

In May 2026, Google introduced a multi-token drafting structure based on draft model and speculative decoding in Gemma 4's inference system,用于提升 LLM inference speed. Although both Gemma 4 and DeepSeek V3 involve predicting multiple future tokens, *their goals and implementation methods are not completely identical*:

- DeepSeek V3 MTP belongs to **training-stage multi-step prediction objective**, with core purpose being **enhancing the model's modeling capability for future semantic evolution and multi-step generation consistency**;
- Gemma 4 combines `multi-token prediction` with `speculative decoding` in the **inference stage**, used to reduce the inference overhead from target model逐token autoregressive decoding.

Their commonality: neither局限于 traditional single-step next-token prediction's single-step conditional prediction objective, but rather尝试 **continuously conditionally predicting multiple future tokens based on intermediate states from existing context**.

<div align="center">
<img width="500" height="520" alt="Draft model collaboration diagram" src="https://github.com/user-attachments/assets/bd1dde30-c7ee-4b51-b9a6-111e87dd387e" />
   <p>Figure 10.17 Draft Model Collaboration Diagram</p>
</div>

In Gemma 4, the draft model autonomously predicts multiple future tokens conditioned on the context state provided by the target model (such as shared KV Cache or partial intermediate activation states), and then the target model verifies these "draft" token sequences. The target model accepts the prefix consistent with its own predictions; if inconsistency occurs, it resumes normal autoregressive decoding from the first mismatching position.

Considering KV Cache: a Transformer (with $N_{layers}$ layers) generating a single token mainly has computational overhead approximately positively correlated with layer count and hidden dimension ($d$), with complexity approximately:

$$O(N_{layers} \cdot d^2)$$

Therefore, not directly having the target model itself complete multiple token rollouts is because large-parameter models have higher computational cost for逐token generation. For target models with more layers and larger hidden dimensions, single-token inference cost is typically far higher than smaller draft models. Combining `speculative decoding + MTP`, Gemma 4 can significantly improve LLM overall inference speed.

Beyond the improvements mentioned above, when vocabulary weight scale is large, complete matrix multiplication operations and softmax computation bring high computational and memory bandwidth overhead. To further reduce this cost, Gemma 4's inference system combines optimization ideas based on vocabulary partitioning and candidate filtering, with core ideas including:

<div align="center">
<img width="400" height="420" alt="Draft model optimization" src="https://github.com/user-attachments/assets/9bd7674f-ba1d-4164-aefe-4bc82558a6e6" />
   <p>Figure 10.18 Draft Model Optimization</p>
</div>

- Divide large vocabulary into multiple candidate vocabulary partitions through clustering methods, extracting corresponding embedding representations;
- Based on current logits, compute candidate scores for different vocabulary partitions;
- Only execute subsequent matrix multiplication and softmax computation on higher-scoring candidate vocabulary partitions, thereby reducing overhead from complete vocabulary computation.

Therefore, Gemma 4's overall inference acceleration doesn't仅仅依赖 speculative decoding, draft model, and KV Cache sharing, but also combines efficient vocabulary partitioning and candidate filtering optimization.

# 10.5 Summary

This chapter围绕 one core question展开: after model capability is essentially determined, why does the same LLM exhibit different speed and cost under different hardware, different request scales, and different generation stages, and how should we optimize against bottlenecks? The answer can be summarized as one chain:

$$
\text{Training yields capability}
\rightarrow
\text{Autoregressive inference presents capability}
\rightarrow
\text{Prefill/Decode produce different data flows}
\rightarrow
\text{Arithmetic intensity determines hardware bottleneck}
\rightarrow
\text{Scheduling, caching, compression, and structural modifications optimize}
$$

**Review of Opening Questions**

| Opening Question | Chapter Conclusion |
| --- | --- |
| How to evaluate inference performance? | TTFT mainly reflects the wait from request entering the model to first token; ITL/TPOT reflects adjacent token output间隔 during Decode; E2E Latency reflects single-request complete time; Throughput reflects system unit-time processed token count. Low latency and high throughput are different goals, requiring scenario-specific evaluation. |
| Why are training and inference different? | Training uses known ground-truth preceding context and causal mask, enabling time-dimension parallel computation of entire sequences and saving activations for backpropagation; inference uses model-generated preceding context, must be serial between tokens, but per-step internal matrix operations can still be parallelized. |
| What problems do Prefill, Decode, and KV Cache solve? | Prefill processes prompt once and建立 historical KV; Decode processes one new token per step while reading existing KV. KV Cache avoids redundant computation of old token K/V but grows with context length,带来 memory capacity and HBM bandwidth pressure. |
| How to judge compute-bound or memory-bound? | Divide FLOPs by HBM data transfer volume to get arithmetic intensity, compare with hardware's peak compute/bandwidth ratio. Prefill typically更容易 improve reuse and接近 compute-bound due to more tokens; small-batch Decode typically接近 memory-bound because weight and KV Cache搬运 dominate. |
| What bottlenecks do optimization methods each address? | FlashAttention, operator fusion, and tiling reduce intermediate tensor reads/writes; GQA/MQA reduce KV Cache; quantization reduces weight and cache byte count; continuous batching amortizes weight reads and improves throughput; prompt compression reduces Prefill and initial cache scale; SSM and diffusion改变 memory or generation paradigms; speculative decoding reduces target model serial decoding work; MTP provides better training representation foundation for multi-token drafting. |

Therefore,本章 most important takeaway is not memorizing some "universal acceleration method," but establishing the habit of locating bottlenecks by stage: first distinguish Prefill from Decode, then judge whether the problem mainly comes from computation volume, HBM traffic, KV Cache, request scheduling, or autoregressive serial dependency, and finally select the corresponding optimization. All methods have costs — larger batches may increase single-request waiting, cache compression may lose information, non-autoregressive generation may weaken causal consistency, draft models require additional models and verification flows; truly effective inference systems must make holistic trade-offs among quality, latency, throughput, memory, and cost.

## Reflection Questions

**Basic:**

1) What are the connections between training and inference?

2) How to obtain arithmetic intensity, and what can this metric be used for?

3) What are common inference acceleration methods and their corresponding core principles?

**Advanced:**

1) When applying Diffusion Models (DMs) to LLM inference, they exhibit non-autoregressive (NAR) parallel generation advantages, effectively reducing inference latency and KV Cache memory footprint. However, the NAR characteristic sacrifices strong sequential context dependency. What approaches can combine the parallel efficiency of diffusion models with the context coherence of autoregressive models?

>Through discrete diffusion models proposing token candidates in parallel, with autoregressive models performing context consistency verification, significantly reducing inference latency and KV Cache overhead while ensuring generation quality, simultaneously skipping traditional noise prediction steps to align proposals with AR target space.

*Diffusion model generated tokens serve as preselected sequences (proposals)*

2) In long-context inference, Transformer self-attention has $O(n^2)$ complexity, and KV Cache grows linearly with context length. To reduce inference memory footprint and alleviate the long-sequence "forgetting" problem, are there more efficient methods that reduce KV storage cost while maintaining or even improving the model's long-range dependency modeling capability?

>To solve the $O(n^2)$ computational complexity of Transformer self-attention and `KV Cache` linear膨胀 with context length in long-context inference, we can explore building a selective, hierarchical, prediction-driven KV Cache management framework. This framework introduces **learnable multi-layer parameter modules (similar to MoE)** in the Transformer, used to extract key information from `Prompt` during the encoding phase, thereby reducing redundant tokens entering KV Cache and lowering cache overhead from prompts. Meanwhile,通过 gating mechanisms control information retention, adopting multi-timescale hierarchical memory to distinguish short-term from long-term information, and combining prediction needs for dynamic KV Cache addition/deletion. While significantly reducing inference memory footprint and computational cost, this method有望 maintain or even enhance the model's capability for long-distance text dependency modeling.

3) Autoregressive generation is a strict stepwise dependency mechanism. Once the model generates an erroneous token at some step, the error propagates to all subsequent steps; the current model can only "continue generating" and cannot "backtrack to correct." Are there structural improvements that enable the model, after discovering local errors, to modify only necessary preceding tokens and resume generation without starting from scratch?

>Could改进 bidirectional attention mechanisms or build bidirectional + autoregressive hybrid architectures to enable inference processes with backtrackable, correctable capability...

## References
- [Long-Context Language Modeling Survey](https://arxiv.org/abs/2503.17407)
- [LLaDA 2.0: First to Scale Diffusion Language Models to Hundred-Billion Scale](https://github.com/inclusionAI/LLaDA2.0/blob/main/tech_report.pdf)
- [Large-Small Model Collaboration](https://arxiv.org/pdf/2405.19261)
- [DeepSeek V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Gemma 4 Inference Acceleration](https://x.com/googlegemma/status/2051694045869879749)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://docs.sglang.io/)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/overview.html)
- https://arxiv.org/html/2608.05687
- https://arxiv.org/html/2605.29123v1
- https://github.com/vllm-project/speculators