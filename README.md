# steerling

Official repository for *Post-Block Low-Rank Steering*, as described in our [paper](https://arxiv.org/abs/2603.00425).

For a high-level overview, see our [blog post](https://sprocketlab.github.io/posts/2025/11/actvweight/) (based on an earlier version of this work).

<img src="images/actvweight_logo.png" alt="actvweight logo" width="500" />

## Environment Setup

The repo ships with `setup_env.sh`, which provisions a Conda environment named `steerling` (Python 3.10) and installs all runtime dependencies.

```bash
bash setup_env.sh
conda activate steerling
```

For CPU-only machines, set the channel before running the script:

```bash
TORCH_CUDA_CHANNEL=https://download.pytorch.org/whl/cpu bash setup_env.sh
```

## Training

### Basic usage

```bash
python train.py \
  --model-name meta-llama/Meta-Llama-3-8B \
  --dataset-name ListOps
```

Defaults match the method in the paper (adapters applied to all tokens). If you want, you can experiment with variants:

- `--no-adapter` — use fixed steering vectors (fewer trainable params, less expressive).
- `--intervene-last` — restrict steering to the final token *at every generation step*; incurs an extra forward pass per generation step.
- `--submodules` — inject adapters into attention/MLP submodules rather than the entire block.

Performance under these alternate configurations is not guaranteed.

### Full argument reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | *(required)* | HuggingFace model ID (e.g. `meta-llama/Meta-Llama-3-8B`) |
| `--dataset-name` | *(required)* | One of: `ASDiv`, `BoolQ`, `Winogrande`, `ListOps`, `GSM8K`, `MNLI`, `SVAMP` |
| `--lr` | `9e-4` | Learning rate |
| `--warmup-ratio` | `0.03` | Fraction of steps used for LR warm-up |
| `--weight-decay` | `0.0` | Weight decay |
| `--epochs` | dataset default | Training epochs (defaults: BoolQ/Winogrande/ListOps=1, GSM8K=3, others=1) |
| `--bs` | `8` | Per-device training batch size |
| `--adapter-rank` | `8` | Rank for low-rank steering adapters |
| `--no-adapter` | off | Use fixed steering vectors instead of adapter modules |
| `--linear` | off | Disable nonlinearity in adapters (use linear projection) |
| `--nonlinearity` | `silu` | Activation function: `relu`, `gelu`, `tanh`, or `silu` |
| `--submodules` | off | Inject adapters into attention and MLP submodules separately instead of the full block |
| `--intervene-last` | off | Steer only the final token at every generation step (incurs an extra forward pass per step) |
| `--do-eval` | off | Run evaluation immediately after training instead of saving the model |
| `--test-only` | off | Skip validation, evaluate on test set only (requires `--do-eval`) |

### Example variants

```bash
# Fixed steering vectors instead of adapters
python train.py \
  --model-name meta-llama/Meta-Llama-3-8B \
  --dataset-name BoolQ \
  --no-adapter

# Submodule-level adapters (attn + MLP separately)
python train.py \
  --model-name meta-llama/Meta-Llama-3-8B \
  --dataset-name GSM8K \
  --submodules

# Steer only last token, evaluate immediately after training
python train.py \
  --model-name meta-llama/Meta-Llama-3-8B \
  --dataset-name ListOps \
  --intervene-last \
  --do-eval
```

## Outputs

| Path | Contents |
|------|----------|
| `./<dataset>_model/<model>_<variant>/` | Saved model checkpoint (when `--do-eval` is not set) |
| `outputs/<dataset>_model/<model>_<variant>/validation.json` | Validation set predictions and scores (with `--do-eval`) |
| `outputs/<dataset>_model/<model>_<variant>/test.json` | Test set predictions and scores (with `--do-eval`) |

## Inference / Evaluation

Given a saved checkpoint, run standalone evaluation with `inference.py`.

### Basic usage

```bash
python inference.py \
  --model-name meta-llama/Meta-Llama-3-8B \
  --dataset-name ListOps \
  --ckpt-path ./ListOps_model/Meta-Llama-3-8B_adapter_all_nonlinear_silu/ \
  --split test
```

### Full argument reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | *(required)* | HuggingFace model ID — must match the one used during training |
| `--dataset-name` | *(required)* | Dataset to evaluate on |
| `--ckpt-path` | *(required)* | Path to the checkpoint directory saved by `train.py` |
| `--split` | `test` | Which split to evaluate: `train`, `validation`, or `test` |
| `--output-file` | auto | Override the output JSON path |
| `--no-adapter` | off | Load a fixed-vector checkpoint instead of an adapter checkpoint |
| `--adapter-rank` | `8` | Must match the rank used during training |
| `--submodules` | off | Must match the `--submodules` flag used during training |
| `--intervene-last` | off | Must match the `--intervene-last` flag used during training |

### Outputs

| Path | Contents |
|------|----------|
| `outputs/<split>.json` (auto) | Per-example predictions, labels, and eval scores |
| `--output-file` path | Custom output location if specified |

## Joint Training (LoRA + Adapters)

See [`joint/README.md`](joint/README.md) for the joint training variant, which combines LoRA fine-tuning with low-rank steering adapters under an orthogonality constraint.

## Notes

- Gemma models are loaded with `attn_implementation="eager"` automatically.
- Gradient accumulation is fixed at 2 steps; effective batch size = `--bs × 2`.
- WandB logging is enabled by default in `train.py`. Disable it by removing `report_to=["wandb"]` if not needed.
- Checkpoints are saved as `pytorch_model.bin` (not safetensors) to avoid issues with tied embedding weights.
- `--do-eval` runs evaluation in the same process and does not save the model. Omit it to save the checkpoint and evaluate separately with `inference.py`.
