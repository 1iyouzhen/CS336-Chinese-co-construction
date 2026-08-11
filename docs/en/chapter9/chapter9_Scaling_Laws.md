# Chapter 9: Scaling Laws

## 9.1 Introduction: Why Should We Care About Scaling Laws?

Scaling laws describe the relationship between model performance (typically measured by test loss) and key resource variables: model parameters (N), training data size (D), and compute budget (C). Understanding these relationships allows us to predict how much better a model will be if we scale up any of these dimensions, and to optimally allocate compute between model size and data size.

## 9.2 The Classic Scaling Laws

### 9.2.1 Kaplan et al. (OpenAI, 2020)

The original scaling laws paper found that language modeling loss follows a power-law relationship with model size, dataset size, and compute:

$$L(N) \propto N^{-\alpha_N}, \quad L(D) \propto D^{-\alpha_D}, \quad L(C) \propto C^{-\alpha_C}$$

Key findings:
- Model performance improves predictably as we increase model size, data, and compute
- Larger models are more sample-efficient than smaller models
- Optimal training requires scaling model size and data together

### 9.2.2 Chinchilla Scaling Laws (DeepMind, 2022)

Hoffmann et al. found that previous models were significantly **undertrained**—they had too many parameters relative to their training data. The Chinchilla paper established the "compute-optimal" scaling rule:

For a given compute budget C, the optimal allocation is roughly:
- Model parameters N ∝ C^0.5
- Training tokens D ∝ C^0.5

This means model size and data should be scaled **in equal proportion**. The paper's namesake model, Chinchilla (70B parameters, 1.4T tokens), matched or outperformed much larger models (Gopher, 280B) by training on more data.

## 9.3 Key Metrics and Formulas

### 9.3.1 FLOPs Estimation

For a Transformer model with N parameters trained on D tokens:

$$C \approx 6ND$$

This "6ND" formula is widely used for rough compute estimation.

### 9.3.2 Loss Prediction

The power-law form for loss prediction:

$$L(N, D) = A \cdot N^{-\alpha} + B \cdot D^{-\beta} + L_0$$

Where $L_0$ is the irreducible loss (entropy of natural language).

## 9.4 Beyond Chinchilla: Recent Developments

### 9.4.1 "Overtraining" and Inference Efficiency

Recent models like Llama 3 and DeepSeek train on far more data than Chinchilla-optimal—sometimes 10-20x more tokens than "optimal" for their parameter count. This is called "overtraining" and trades training efficiency for inference efficiency: a smaller model trained on more data can match a larger model's performance while being cheaper to serve.

### 9.4.2 Data-Constrained Scaling

When high-quality training data becomes the limiting factor (not compute), the scaling behavior changes. Models may need to be larger to make better use of limited data, or synthetic data generation may be needed.

## 9.5 Practical Implications

1. **Predictable improvement**: Scaling laws give reliable estimates of how much better a model will be with more resources
2. **Compute allocation**: Guide decisions about model size vs. data volume
3. **Cost estimation**: Help estimate the compute cost to reach a target performance
4. **Diminishing returns**: Each doubling of compute yields smaller and smaller improvements
5. **Emergence**: Some capabilities only appear above certain scale thresholds

## 9.6 Designing Scaling Experiments

When empirically measuring scaling laws on small models to predict large-model behavior:

1. Train models across a range of sizes (e.g., 1M to 1B parameters)
2. Use consistent architecture, data, and training procedures
3. Fit power-law curves to the loss measurements
4. Extrapolate to predict performance at target scales
5. Validate with intermediate-scale models

## 9.7 References

- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361)
- [Training Compute-Optimal Large Language Models (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556)
- [Chinchilla's Wild Implications (Blog post)](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications)
