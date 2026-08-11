# Chapter 4: Language Model Architecture and Training Details

## Learning Objectives

This chapter covers the technical details of Transformer architecture for LLMs: RoPE, RMSNorm, SwiGLU, AdamW, Pre-Norm vs Post-Norm, and learning rate scheduling.

## 4.1 Transformer Architecture for Modern LLMs

### 4.1.1 Rotary Position Embedding (RoPE)

RoPE encodes position information by rotating query and key vectors. The rotation angle depends on the token's position, making the attention score between two tokens depend only on their relative distance. Unlike absolute positional encodings, RoPE naturally handles sequences longer than those seen during training.

$$(R_m \cdot q)^T (R_n \cdot k) = q^T R_{n-m} k$$

### 4.1.2 RMSNorm (Root Mean Square Layer Normalization)

A simplified version of LayerNorm that only uses the root mean square statistic, removing the mean-centering step. Faster than LayerNorm while maintaining comparable performance.

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \cdot \gamma$$

### 4.1.3 SwiGLU Activation

Combines Swish activation with a gated linear unit. Outperforms ReLU and GELU in modern LLMs.

$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \odot (xW_2)$$

### 4.1.4 Pre-Norm vs Post-Norm

**Pre-Norm**: Apply normalization before attention/FFN sublayers. More stable training, standard in modern LLMs.
**Post-Norm**: Apply normalization after sublayers. Used in original Transformer, but can cause training instability in deep models.

### 4.1.5 AdamW Optimizer

Adam with decoupled weight decay. The standard optimizer for LLM training. Key hyperparameters: learning rate, weight decay, β₁=0.9, β₂=0.95, ε=1e-8.

## 4.2 Training Details

### 4.2.1 Mixed Precision Training

Using FP16 or BF16 for forward/backward passes while maintaining FP32 master weights. BF16 is preferred for LLMs due to its larger dynamic range compared to FP16.

### 4.2.2 Gradient Accumulation

Simulating larger batch sizes by accumulating gradients over multiple micro-batches before updating weights. Essential when GPU memory limits batch size.

### 4.2.3 Learning Rate Scheduling

Common schedules:
- **Cosine decay**: Smooth cosine-shaped decay from initial to final LR
- **Linear warmup**: Gradually increase LR at the start of training
- **WSD (Warmup-Stable-Decay)**: Warmup → constant LR → decay. Used by MiniCPM and others where the decay phase mixes in high-quality data

### 4.2.4 Gradient Clipping

Limiting the norm of gradients to prevent training instability. Typical max norm: 1.0.

## 4.3 Key Takeaways

1. RoPE provides relative position encoding naturally suited for length extrapolation
2. RMSNorm is the standard normalization in modern LLMs (faster than LayerNorm)
3. SwiGLU is the dominant activation function
4. Pre-Norm architecture enables stable training of very deep Transformers
5. BF16 mixed precision + gradient accumulation + cosine LR schedule is the standard training recipe
