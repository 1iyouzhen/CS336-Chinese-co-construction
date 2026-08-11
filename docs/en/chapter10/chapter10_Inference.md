# Chapter 10: Inference

Inference is the process of using a trained language model to generate text—taking a trained model and running it to produce outputs. While training determines what the model knows, inference determines how efficiently and quickly we can actually *use* that knowledge.

## 10.1 Key Concepts in LLM Inference

### 10.1.1 Autoregressive Generation

LLMs generate text token by token. At each step, the model takes the current sequence (prompt + previously generated tokens) and predicts the probability distribution over the next token. The token is then sampled or selected, appended to the sequence, and the process repeats until an end-of-sequence token is generated.

### 10.1.2 Decoding Strategies

- **Greedy decoding**: Always pick the highest-probability token. Fast but can miss globally optimal sequences.
- **Temperature sampling**: Scale logits by a temperature parameter before softmax. Low temperature (T<1) makes output more deterministic; high temperature (T>1) increases diversity.
- **Top-k sampling**: Only sample from the k most probable tokens.
- **Top-p (nucleus) sampling**: Sample from the smallest set of tokens whose cumulative probability exceeds p.
- **Beam search**: Maintain k candidate sequences in parallel and expand them step by step. Better for tasks requiring structured outputs but computationally more expensive.

### 10.1.3 Key Metrics

- **Latency**: Time to generate one token or a complete response
- **Throughput**: Tokens generated per second
- **Memory usage**: GPU memory required to serve the model
- **Time To First Token (TTFT)**: Latency before the first output token appears
- **Tokens Per Second (TPS)**: Generation speed after the first token

## 10.2 KV Cache

The key performance optimization for autoregressive generation is the **KV Cache** (Key-Value Cache). During generation, the model computes attention over all previous tokens. Without caching, it would recompute keys and values for all past tokens at every step—an O(n²) operation.

With KV Cache, we store the key and value tensors from all previous steps. At each new step, we only need to compute keys and values for the new token and append them to the cache. This reduces per-step computation from O(n²) to O(n).

## 10.3 Memory Analysis

For a Transformer with L layers, hidden dimension d, and sequence length n:

- **Model parameters**: P parameters × bytes_per_param (typically 2 bytes for FP16)
- **KV Cache**: 2 × L × n × d × bytes_per_element per layer
- **Activations**: Depend on batch size and sequence length

Example: For a 7B parameter model in FP16:
- Model weights: ~14 GB
- KV Cache (seq_len=2048): ~2-4 GB per batch
- Total: ~16-18 GB minimum

## 10.4 Quantization

Quantization reduces model memory footprint and speeds up inference by representing weights and/or activations at lower precision.

### Common Quantization Methods

- **GPTQ (Post-Training Quantization)**: Quantizes weights after training using optimal quantization. Typically reduces a 7B model from 14GB to 4-5GB with 4-bit quantization.
- **AWQ (Activation-aware Weight Quantization)**: Identifies and protects "salient" weight channels that are important for model quality, using per-channel scaling. Better preserves quality than vanilla GPTQ at very low bit-widths.
- **GGUF/GGML**: Popular format for CPU inference (used by llama.cpp). Supports various quantization levels (Q4_0, Q4_1, Q5_0, Q8_0, etc.).
- **BitsAndBytes (BNB)**: Supports 4-bit and 8-bit quantization, integrated with HuggingFace Transformers via `load_in_4bit=True` or `load_in_8bit=True`.

## 10.5 Flash Attention

**Flash Attention** is a memory-efficient exact attention algorithm that avoids materializing the full N×N attention matrix in GPU HBM (High Bandwidth Memory). Instead, it:
1. Tiles the attention computation into blocks that fit in SRAM
2. Computes softmax in a numerically stable way using online softmax with rescaling
3. Only writes the final output back to HBM

This reduces memory from O(n²) to O(n) and significantly speeds up training and inference for long sequences.

## 10.6 Speculative Decoding

**Speculative decoding** accelerates inference by using a small "draft" model to propose multiple tokens quickly, then having the large "target" model verify them in parallel. This can achieve 2-3x speedup while producing exactly the same output distribution as the original model.

## 10.7 Advanced Inference Techniques

- **Continuous Batching**: Dynamically adding/removing requests from a batch as they complete, maximizing GPU utilization
- **PagedAttention (vLLM)**: Managing KV cache in fixed-size blocks (pages), analogous to virtual memory paging. Eliminates memory fragmentation and enables memory sharing across requests
- **Tensor Parallelism**: Splitting model weights across multiple GPUs for inference
- **Pipeline Parallelism**: Placing different layers on different GPUs

## 10.8 Practical Considerations

When deploying LLMs in production:

1. **Choose the right quantization level**: 4-bit offers good speed/memory trade-off with minimal quality loss
2. **Use a serving framework**: vLLM, TGI (Text Generation Inference), or TensorRT-LLM handle batching, scheduling, and memory management
3. **Monitor GPU utilization**: Aim for >80% utilization to maximize cost efficiency
4. **Consider hardware constraints**: GPU memory, bandwidth, and compute all matter
5. **Batch size vs. latency trade-off**: Larger batches improve throughput but increase per-request latency
