# Chapter 8: Distributed Training

Training modern LLMs requires distributing computation across multiple GPUs and machines. This chapter covers the key parallelism paradigms.

## 8.1 Why Distributed Training?

A 70B parameter model in FP16 requires ~140GB just for model weights. With optimizer states and activations, total memory exceeds 500GB—far beyond any single GPU. Distributed training enables:
- Splitting model across devices (model parallelism)
- Processing different data in parallel (data parallelism)
- Combining both for maximum scale

## 8.2 Data Parallelism (DP)

### 8.2.1 Basic Data Parallelism

Each GPU holds a full copy of the model, processes different micro-batches, then synchronizes gradients via All-Reduce before updating weights.

### 8.2.2 DistributedDataParallel (DDP)

PyTorch's DDP provides efficient multi-GPU training with:
- Gradient All-Reduce using NCCL backend
- Overlapping communication with computation
- Automatic gradient synchronization

## 8.3 Model Parallelism

### 8.3.1 Tensor Parallelism

Split individual weight matrices across GPUs. Each GPU computes a portion of the matrix multiply, then communicates results. Used by Megatron-LM.

### 8.3.2 Pipeline Parallelism

Assign different layers to different GPUs. Micro-batching hides pipeline bubbles. GPipe and PipeDream are popular implementations.

## 8.4 ZeRO (DeepSpeed)

ZeRO (Zero Redundancy Optimizer) progressively partitions training state:

- **ZeRO-1**: Partition optimizer states across GPUs
- **ZeRO-2**: Additionally partition gradients
- **ZeRO-3**: Additionally partition model parameters

Each stage reduces memory by factor of N_gpus while maintaining identical computation.

## 8.5 FSDP (Fully Sharded Data Parallel)

PyTorch's native implementation of ZeRO-3. Shards model parameters, gradients, and optimizer states across GPUs, gathering them only when needed for computation.

## 8.6 3D Parallelism

Combining all three:
- **Data parallel** across nodes
- **Tensor parallel** within nodes (fast NVLink)
- **Pipeline parallel** across depth

This is how GPT-3, Llama, and other frontier models are trained.

## 8.7 Communication Primitives

- **All-Reduce**: Sum/average values across all GPUs (used for gradient sync)
- **All-Gather**: Collect values from all GPUs (used in ZeRO-3 forward)
- **Reduce-Scatter**: Reduce then scatter (used in ZeRO-3 backward)
- **Broadcast**: Send from one GPU to all

## 8.8 Key Formulas

**Data parallelism communication per step**: $2 \times (N_{gpus} - 1) / N_{gpus} \times \text{params}$

**Tensor parallelism communication**: Depends on specific sharding, generally higher than DP for large models

## 8.9 Practical Considerations

1. Use ZeRO-3/FSDP for most large model training
2. Add tensor parallelism only when ZeRO is insufficient
3. Pipeline parallelism helps with very deep models
4. Monitor communication overhead—it can dominate at scale
5. BF16 training reduces both memory and communication volume
