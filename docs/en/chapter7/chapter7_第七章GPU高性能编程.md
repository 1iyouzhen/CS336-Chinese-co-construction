# Chapter 7: GPU High-Performance Programming

Modern LLM training and inference heavily depends on GPU programming optimizations. This chapter introduces CUDA programming and key optimization techniques.

## 7.1 GPU Computing Fundamentals

### 7.1.1 GPU Architecture

GPUs excel at parallel computation with thousands of cores organized into Streaming Multiprocessors (SMs). Key memory hierarchy:
- **Global Memory (HBM)**: Large capacity (80GB on A100), high bandwidth (~2TB/s), high latency
- **Shared Memory (SRAM)**: Small (up to 164KB per SM on A100), extremely fast, managed by programmer
- **Registers**: Per-thread, fastest access

### 7.1.2 SIMT Model

Single Instruction, Multiple Threads. Threads are organized into warps (32 threads) that execute the same instruction in lockstep on different data.

### 7.1.3 Key Metrics

- **Compute**: FLOPS (FLoating point Operations Per Second). A100: 312 TFLOPS (FP16)
- **Memory Bandwidth**: Data transfer rate. A100: 2 TB/s
- **Arithmetic Intensity**: FLOPs per byte of memory access. Determines if kernel is compute-bound or memory-bound

## 7.2 CUDA Programming Basics

### 7.2.1 Kernel Launch

```cuda
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

// Launch with 256 threads per block
vector_add<<<(n + 255) / 256, 256>>>(a, b, c, n);
```

### 7.2.2 Memory Optimization

Key techniques:
- **Coalesced memory access**: Adjacent threads access adjacent memory addresses
- **Shared memory tiling**: Load data blocks into shared memory to reduce global memory access
- **Bank conflict avoidance**: Ensure threads in a warp access different shared memory banks

## 7.3 Triton

Triton is a Python-based DSL for writing GPU kernels, widely used in LLM optimizations.

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)
```

## 7.4 Flash Attention Implementation

Flash Attention is the canonical example of GPU optimization for LLMs. Key techniques:
1. **Tiling**: Split Q, K, V into blocks fitting in SRAM
2. **Online softmax**: Compute softmax incrementally without full matrix
3. **Recomputation**: Recompute attention in backward pass rather than storing

## 7.5 Tensor Cores

Modern NVIDIA GPUs include Tensor Cores—specialized hardware for matrix multiplication. Key usage patterns:
- FP16 input, FP32 accumulation
- BF16 input for better dynamic range
- INT8/INT4 for inference
- Achieve 2-8x speedup over CUDA cores for matrix operations

## 7.6 Key Optimization Principles

1. **Maximize arithmetic intensity**: Fuse operations to reduce memory access
2. **Use shared memory**: For data reuse within a thread block
3. **Coalesce memory access**: Align with GPU memory transaction size
4. **Leverage Tensor Cores**: For all matrix multiply operations
5. **Profile before optimizing**: Use Nsight Compute to identify bottlenecks
