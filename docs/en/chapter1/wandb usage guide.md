# Weights & Biases Usage Guide

`Weights & Biases (W&B)`is a collaborative platform designed specifically for machine learning experiments. It supports automatic experiment configuration (`config`) logging, real-time metrics visualization, hyperparameter search (`Sweeps`), model versioning, code snapshots and dataset tracking, multi-person collaboration, and report generation. In large model research, W&B significantly improves experiment traceability, reproducibility, and analysis efficiency.

If you are using it for the first time, please refer to the official documentation to [create an account](https://wandb.ai/login?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=quickstart) and set up an [API key](https://wandb.ai/settings). This guide provides a quick introduction.

## 1. Installation and Login

```bash
pip install wandb
```

You need to log in to use this tool for the first time (requires a [wandb.ai](https://wandb.ai) account):

```bash
wandb login
```

> 💡 **No internet access**: Skip login and use `mode="offline"` directly  (see Section 3).

## 2. Basic Usage: `wandb.init()`

Initialize a `run` at the beginning of your training script:

```python
import wandb

wandb.init(
    project="cs336-a5-sft-v2",          # project name (required)
    entity="your-team-or-username",     # team/username (optional)
    name="wanda_sft",                   # readable run name
    config={
        "model": "Qwen2.5-Math-1.5B",
        "dataset_tag": "raw", # raw, sf, grpo
        "batch_size": 64,
        "max_examples": "1000",
        "seed": 2026,
        "learning_rate": 2e-5,
    }
)
```

The `config` parameter is fully customizable. It is recommended to put all hyperparameters, data paths, model versions, etc. into `config`, for easier filtering and comparison later.


## 3. Offline Mode

When your server cannot access the internet, use offline mode to save logs:

```python
wandb.init(mode="offline", ...)
```

All logs will be saved locally under the `wandb/` directory, in the format `offline-run-<timestamp>-<id>`.

### Syncing to the Cloud Later

Copy the folder containing the `wandb/` directory to a machine with internet access and run:

```bash
wandb sync wandb/
```

> ⚠️ **Note**: Make sure that the machine has been logged in via wandb login and that the run IDs have not been deleted.

You can also sync a specific run:

```bash
wandb sync wandb/offline-run-20260116_113519-vc1rtokn
```

---

## 4. Logging Metrics:`wandb.log()`

Log scalars, images, text, and more during your training/evaluation loop:

```python
for step, batch in enumerate(dataloader):
    loss = model(batch)
    wandb.log({
        "train/loss": loss.item(),
        "train/lr": scheduler.get_last_lr()[0],
        "step": step
    })
```

Supported types:
- scalar
- wandb.Image
- wandb.Table
- wandb.Histogram
- audio, 3D objects, etc. (less commonly used for LLMs)

> 📌 **Tip**: Use `/` to create namespaces (e.g., `eval/human_eval_pass@1`) for grouped display in the UI.


## 5. Artifacts

W&B allows you to upload model checkpoints as Artifacts for version control:

```python
artifact = wandb.Artifact(name="llama3-70b-wanda-c4", type="model")
artifact.add_file("checkpoints/model.safetensors")
wandb.log_artifact(artifact)
```

You can later reference that model in another experiment:

```python
artifact = run.use_artifact("llama3-70b-wanda-c4:latest")
artifact_dir = artifact.download()
```

> 🔒 **Note**: Large model parameter files are not recommended for upload due to their size.


## 6. FAQ

#### Q1: Initialization timeout?
```python
wandb.errors.CommError: Run initialization has timed out...
```
**Solution**: Increase the timeout or switch to offline mode:
```python
wandb.init(settings=wandb.Settings(init_timeout=120), mode="offline")
```

#### Q2: Is it possible to disable W&B (for example, during debugging)?
```python
wandb.init(mode="disabled")   # completely silent, no side effects
```

#### Q3: I've successfully logged in to PyCharm, but I still can't use W&B to view the training process. How can I fix this?

**Cause**: The runtime environment in PyCharm is independent of the terminal where you ran `wandb login`. The login state from the terminal is not automatically inherited by code executed within PyCharm. Therefore, even if the login succeeded in the terminal, running a script directly in PyCharm will not show the W&B training monitoring link.

**Solutions**:
- **Method 1**：Run your training script from PyCharm’s built-in terminal (i.e., execute `python train.py` directly in the PyCharm Terminal), at this point, the terminal environment matches the login environment, and the W&B link displays correctly.
- **Method 2**：Switch to VSCode, its environment integration typically inherits the terminal's login status more reliably.

---

## 7. References

- Official documentation: https://docs.wandb.ai
- Example repository: https://github.com/wandb/examples

> ✨ **Tip**: Take 2 minutes before each experiment to write up your `config` and `notes`, you’ll thank yourself later when you look back!
