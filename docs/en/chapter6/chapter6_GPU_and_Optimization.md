# Chapter 6: GPU and GPU-Related Optimization

This chapter covers GPU hardware fundamentals and optimization techniques that are critical for training and deploying LLMs efficiently.

## 6.1 GPU Hardware Overview

### 6.1.1 Memory Hierarchy

- **HBM (High Bandwidth Memory)**: A100 80GB, H100 80GB. Main storage for model weights and activations
- **L2 Cache**: 40MB on A100. Cached across all SMs
- **Shared Memory (SRAM)**: Up to 164KB per SM on A100. Programmer-managed
- **Registers**: 256KB per SM on A100

### 6.1.2 Key GPU Specifications

| GPU | Memory | Bandwidth | FP16 TFLOPS | NVLink |
|-----|--------|-----------|-------------|--------|
| A100 80GB | 80 GB HBM2e | 2.0 TB/s | 312 | 600 GB/s |
| H100 80GB | 80 GB HBM3 | 3.35 TB/s | 989 | 900 GB/s |
| H200 | 141 GB HBM3e | 4.8 TB/s | 989 | 900 GB/s |

## 6.2 Memory Bandwidth and Compute Intensity

### 6.2.1 The Memory Wall

GPU compute has grown faster than memory bandwidth, making many operations memory-bound. Understanding this trade-off is critical for optimization.

### 6.2.2 Roofline Model

The Roofline model plots achievable performance against arithmetic intensity:
- **Memory-bound region**: Performance limited by bandwidth
- **Compute-bound region**: Performance limited by peak FLOPS

For LLMs:
- **Attention**: Memory-bound for short sequences, compute-bound for long sequences
- **FFN/MLP**: Typically compute-bound
- **LayerNorm/RMSNorm**: Heavily memory-bound

## 6.3 Flash Attention

### 6.3.1 The Problem

Standard attention requires O(n²) memory for the attention matrix, making long sequences prohibitively expensive.

### 6.3.2 Flash Attention Solution

Flash Attention computes exact attention without materializing the full N×N matrix:
1. Split Q, K, V into blocks
2. Load blocks into SRAM
3. Compute attention incrementally with online softmax
4. Only write output to HBM

Memory savings: from O(n²) to O(n). Speedup: 2-4x for typical sequence lengths.

## 6.4 Kernel Fusion

Combining multiple operations into a single kernel launch eliminates intermediate memory writes. Examples:
- Fusing LayerNorm + Dropout + Residual connection
- Fusing GELU activation with linear layer
- PyTorch's `torch.compile()` automates many fusion opportunities

## 6.5 Mixed Precision Training

### 6.5.1 FP16 vs BF16

- **FP16**: 5 exponent + 10 mantissa bits. Requires loss scaling to handle gradient underflow
- **BF16**: 8 exponent + 7 mantissa bits. Same dynamic range as FP32, no loss scaling needed. **Preferred for LLM training**

### 6.5.2 Automatic Mixed Precision (AMP)

PyTorch's AMP automatically casts operations to appropriate precision:
- Matrix multiplies: FP16/BF16
- Normalization, softmax: FP32
- Loss scaling (FP16 only)

## 6.6 Key Optimization Techniques Summary

1. **Flash Attention**: O(n²) → O(n) memory for attention
2. **Kernel Fusion**: Reduce kernel launch overhead and memory traffic
3. **BF16 Training**: 2x memory savings, 2x compute throughput vs FP32
4. **Gradient Checkpointing**: Trade 20% compute for ~30% memory savings
5. **Tensor Cores**: 2-8x speedup for matrix multiply operations
6. **Memory Layout**: Ensure contiguous memory access patterns
