# Chapter 3: PyTorch and Resource Accounting

Understanding how to estimate and manage computational resources is essential for LLM training. This chapter covers PyTorch fundamentals for LLMs and how to calculate FLOPs and memory requirements.

## 3.1 PyTorch for LLM Training

### 3.1.1 Key Concepts

- **Tensors**: Multi-dimensional arrays, the fundamental data structure
- **Autograd**: Automatic differentiation engine for computing gradients
- **nn.Module**: Base class for all neural network modules
- **DataLoader**: Efficient data loading with batching, shuffling, and multiprocessing
- **DistributedDataParallel (DDP)**: Multi-GPU training

### 3.1.2 Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3.1.3 Gradient Checkpointing

Trading compute for memory by recomputing activations during backward pass instead of storing them. Reduces memory by ~O(sqrt(n)) at the cost of ~20% more compute.

```python
model.gradient_checkpointing_enable()
```

## 3.2 FLOPs Estimation

### 3.2.1 Forward Pass FLOPs

For a Transformer with parameters N, processing one token:
$$\text{FLOPs}_{\text{forward}} \approx 2N$$

For D tokens total:
$$\text{FLOPs}_{\text{total}} \approx 6ND$$

(The factor of ~3x comes from: 1x forward + 2x backward)

### 3.2.2 Matrix Multiplication FLOPs

For matrix multiply A[m×k] × B[k×n]:
$$\text{FLOPs} = 2 \times m \times n \times k$$

## 3.3 Memory Estimation

### 3.3.1 Model Memory

- FP32: parameters × 4 bytes
- FP16/BF16: parameters × 2 bytes
- INT8: parameters × 1 byte
- INT4: parameters × 0.5 bytes

### 3.3.2 Optimizer States

AdamW stores:
- FP32 master weights: params × 4 bytes
- Momentum (m): params × 4 bytes
- Variance (v): params × 4 bytes
Total optimizer memory: params × 12 bytes

### 3.3.3 Activations and Gradients

- Gradients: params × bytes_per_param
- Activations: depends on batch size, sequence length, hidden dimension, and number of layers
- Rule of thumb: total training memory ≈ model × 20-24 bytes (for AdamW mixed precision)

### 3.3.4 Example Calculation

7B parameter model, BF16 mixed precision, AdamW:
- Model weights (BF16): 14 GB
- Optimizer states (FP32): 28 GB
- Gradients (FP32): 28 GB
- Activations: ~8-16 GB
- **Total**: ~78-86 GB (requires 4×A100 80GB or 2×H100 80GB)

## 3.4 GPU Memory Optimization

1. **Gradient Checkpointing**: ~30% memory savings
2. **Mixed Precision (BF16)**: Halves weight/gradient memory
3. **ZeRO (DeepSpeed)**: Distributes optimizer states, gradients, and parameters across GPUs
4. **Flash Attention**: Reduces attention memory from O(n²) to O(n)
5. **CPU Offloading**: Offload optimizer states to CPU memory
