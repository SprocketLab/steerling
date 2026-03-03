# joint — LoRA + Adapter with Orthogonality Constraint

This subdirectory implements **joint training** of LoRA fine-tuning and low-rank steering adapters with the orthogonality contraint described in the paper.

### Basic usage

```bash
python train_orthogonal.py \
  --model-name meta-llama/Meta-Llama-3-8B \
  --dataset-name ListOps
```

### Full argument reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | *(required)* | HuggingFace model ID (e.g. `meta-llama/Meta-Llama-3-8B`) |
| `--dataset-name` | *(required)* | One of: `BoolQ`, `Winogrande`, `ListOps`, `GSM8K`, `ARC_Challenge`, `AQuA` |
| `--lr` | `2e-4` | Base learning rate |
| `--lora-lr` | same as `--lr` | Learning rate override for LoRA parameters |
| `--adapter-lr` | same as `--lr` | Learning rate override for adapter parameters |
| `--warmup-ratio` | `0.06` | Fraction of steps used for LR warm-up |
| `--scheduler` | `cosine` | LR scheduler type |
| `--weight-decay` | `0.0` | Weight decay |
| `--epochs` | dataset default | Training epochs (defaults: BoolQ/Winogrande/ListOps=1, GSM8K=3, others=5) |
| `--bs` | `8` | Per-device training batch size |
| `--lora-rank` | `8` | Rank `r` for LoRA adapters |
| `--adapter-rank` | `8` | Rank for low-rank steering adapters |
| `--linear` | off | Disable nonlinearity in adapters (use linear projection) |
| `--nonlinearity` | `silu` | Activation function: `relu`, `gelu`, `tanh`, or `silu` |
| `--submodules` | off | Inject adapters into attention and MLP submodules separately instead of the full block |
| `--orthogonal-freq` | `step` | How often to orthogonalize: `step`, `epoch`, `init`, or an integer (e.g. `100`) |
| `--save-model` | off | Save trained LoRA weights and adapter steerer state dict |
| `--test-only` | off | Skip validation, evaluate on test set only |

### Example variants

```bash
# Larger ranks, orthogonalize every 50 steps
python train_orthogonal.py \
  --model-name meta-llama/Meta-Llama-3-8B \
  --dataset-name BoolQ \
  --lora-rank 16 \
  --adapter-rank 16 \
  --orthogonal-freq 50

# Linear adapters (no activation), save model after training
python train_orthogonal.py \
  --model-name meta-llama/Meta-Llama-3-8B \
  --dataset-name ListOps \
  --linear \
  --save-model
```

## Outputs

| Path | Contents |
|------|----------|
| `model_<dataset>/<model>_dual/` | HuggingFace `Trainer` checkpoint (intermediate, no final save by default) |
| `outputs/orth_outputs/<dataset>/<run>/validation.json` | Validation set predictions and scores |
| `outputs/orth_outputs/<dataset>/<run>/test.json` | Test set predictions and scores |
| `orth_model_<dataset>/<run>/` | Saved LoRA model (only with `--save-model`) |
| `orth_model_<dataset>/<run>/adapter_steerer.pt` | Saved adapter state dict (only with `--save-model`) |

## Notes

- Gemma models are loaded with `attn_implementation="eager"` automatically.
- Gradient accumulation is fixed at 2 steps; effective batch size = `--bs × 2`.
- WandB logging is commented out in the script; enable it by uncommenting `report_to=["wandb"]` in `train_orthogonal.py` if desired.
