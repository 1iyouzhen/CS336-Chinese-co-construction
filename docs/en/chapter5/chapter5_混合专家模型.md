# Chapter 5: Mixture of Experts (MoE)

Mixture of Experts (MoE) is an architectural pattern that dramatically increases model capacity without proportionally increasing compute. The key idea: instead of having every token go through every parameter, route each token to only a subset of "expert" sub-networks.

## 5.1 Why MoE?

Standard dense Transformers scale compute quadratically with width and linearly with depth. MoE breaks this relationship by:
- Increasing total parameters (capacity) significantly
- Keeping active parameters (compute per token) roughly constant
- Using sparse activation to route different tokens to different experts

## 5.2 MoE Architecture

### 5.2.1 Core Components

1. **Expert Networks**: Multiple copies of FFN (Feed-Forward Network) layers. Each expert is a standard FFN sub-network
2. **Router/Gating Network**: A learned function that decides which expert(s) should process each token
3. **Sparse Activation**: Only Top-k experts are activated per token (typically k=1 or 2)

### 5.2.2 The Routing Mechanism

For a given token representation $x$, the router computes scores for each expert:

$$s_i = \text{softmax}(W_r \cdot x)$$

Then selects the Top-k experts:

$$\text{TopK}(s, k) = \text{indices of k largest values in s}$$

The token is then processed by the selected experts and their outputs are combined (typically weighted sum).

### 5.2.3 Load Balancing

A critical challenge in MoE is ensuring all experts are utilized. Without load balancing:
- Some experts become "dead" (never used)
- Others become overloaded
- Training becomes inefficient

Solutions include:
- **Auxiliary loss**: Penalize the router if expert usage is too imbalanced
- **Expert capacity**: Limit how many tokens each expert can process
- **Load-balancing loss**: $L_{balance} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$ where $f_i$ is the fraction of tokens routed to expert i and $P_i$ is the average router probability for expert i

## 5.3 MoE Variants

### 5.3.1 Switch Transformer (Google, 2021)

Uses Top-1 routing (k=1). Each token goes to exactly one expert. Simplifies routing and improves training stability while maintaining MoE benefits.

### 5.3.2 GShard (Google, 2020)

Early large-scale MoE deployment. Uses Top-2 gating with auxiliary load-balancing loss and expert capacity limits.

### 5.3.3 DeepSeekMoE (2024)

Introduces **shared experts** and **fine-grained experts**:
- Shared experts: Always activated, handle common knowledge
- Routed experts: Sparsely activated, handle specialized knowledge
- This separation improves both training stability and expert specialization

DeepSeek-V3 uses 256 experts total, activating 8 per token (1 shared + 7 routed), achieving 671B total parameters with only 37B active per token.

### 5.3.4 Mixtral (Mistral, 2023)

Popular open-source MoE model. 8 experts per layer, Top-2 routing. 46.7B total parameters, ~12.9B active per token. Demonstrates that MoE can be effectively deployed at moderate scales.

## 5.4 Engineering Challenges

### 5.4.1 Communication Overhead

In distributed training, experts on different devices require all-to-all communication. This can become a bottleneck at scale.

### 5.4.2 Memory Requirements

All experts must be stored in memory even if only a subset is active. This requires large aggregate GPU memory.

### 5.4.3 Training Stability

Router collapse (all tokens routing to a few experts) is a common failure mode requiring careful initialization and loss design.

### 5.4.4 Inference Optimization

Sparse activation doesn't map naturally to dense GPU hardware. Requires specialized inference frameworks.

## 5.5 When to Use MoE

**Good for**:
- Increasing model capacity without proportional compute increase
- Multi-task models where different domains can be handled by different experts
- When total parameters matter more than active parameters (e.g., knowledge-intensive tasks)

**Less good for**:
- Latency-sensitive applications (sparse routing adds overhead)
- Very small models (routing overhead dominates)
- When simplicity and reproducibility are priorities

## 5.6 Key Takeaways

1. MoE increases total capacity while keeping per-token compute roughly constant
2. Routing quality and load balancing are critical to MoE success
3. Modern MoE (DeepSeek-V3, Mixtral) has proven MoE works at industrial scale
4. MoE trades engineering complexity for parameter efficiency
5. MoE is likely to become increasingly common as models continue to scale
