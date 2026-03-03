# instruction_tuning

Fine-tuning post-block steering adapters (or fixed vectors) on instruction-following data, with evaluation via [AlpacaEval 2.0](https://github.com/tatsu-lab/alpaca_eval).

## Files

| File | Description |
|------|-------------|
| `train_it.py` | Trains steering parameters on Alpaca-style datasets with response-only loss masking |
| `inference_it.py` | Loads a trained checkpoint and generates responses on the 805 AlpacaEval prompts |
| `run_alpaca_eval.py` | Scores generation files against GPT-4 Turbo using the `alpaca_eval` judge |

## Setup

Run this once from the repo root:

```bash
bash setup_env.sh
conda activate steerling
```

For AlpacaEval scoring only, also install the judge package:

```bash
pip install alpaca_eval
export OPENAI_API_KEY="sk-..."
```

## Training

All commands must be run from the `instruction_tuning/` directory.

### Basic usage

```bash
python train_it.py \
  --model-name meta-llama/Llama-3.1-8B \
  --dataset-name alpaca_cleaned \
  --adapter \
  --save-model
```

By default, fixed steering vectors are used. Pass `--adapter` to use low-rank adapter modules (more expressive).

### Full argument reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | *(required)* | HuggingFace model ID (e.g. `meta-llama/Llama-3.1-8B`) |
| `--dataset-name` | `alpaca_cleaned` | One of: `alpaca`, `alpaca_cleaned`, `code_alpaca` |
| `--max-length` | `512` | Max token length for prompt + response combined |
| `--max-samples` | all | Subsample the training set (useful for quick debugging) |
| `--epochs` | dataset default | Training epochs (default: 3 for all datasets) |
| `--lr` | `2e-3` | Learning rate |
| `--warmup-ratio` | `0.03` | Fraction of steps used for LR warm-up |
| `--bs` | `4` | Per-device training batch size |
| `--gradient-accumulation-steps` | `4` | Gradient accumulation steps (effective batch = `bs × steps`) |
| `--weight-decay` | `0.0` | Weight decay |
| `--adapter` | off | Use low-rank adapter modules instead of fixed steering vectors |
| `--adapter-rank` | `8` | Rank for low-rank adapters (ignored without `--adapter`) |
| `--linear` | off | Disable nonlinearity in adapters |
| `--nonlinearity` | `silu` | Activation function: `relu`, `gelu`, `tanh`, or `silu` |
| `--submodules` | off | Inject into attention and MLP submodules separately instead of the full block |
| `--save-model` | off | Save model checkpoint after training |
| `--do-eval` | off | Run generation on AlpacaEval prompts after training (skips saving) |

### Example variants

```bash
# Quick debug run on 200 samples
python train_it.py \
  --model-name meta-llama/Llama-3.1-8B \
  --adapter \
  --max-samples 200 \
  --epochs 1
```

## Inference

Generates responses on the 805 AlpacaEval prompts using a saved checkpoint. The flags must match those used during training.

### Basic usage

```bash
python inference_it.py \
  --model-name meta-llama/Llama-3.1-8B \
  --ckpt-path checkpoints/IT_alpaca_cleaned/Llama-3.1-8B_adapter_r8_nonlinear_silu_block/lr0.002_bs4x4_ep3_warmup0.03_wd0.0 \
  --adapter-rank 8
```

### Full argument reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | *(required)* | HuggingFace model ID — must match training |
| `--ckpt-path` | *(required)* | Path to checkpoint directory saved by `train_it.py` |
| `--adapter-rank` | `16` | Must match `--adapter-rank` used during training |
| `--linear` | off | Must match `--linear` flag used during training |
| `--nonlinearity` | `silu` | Must match `--nonlinearity` used during training |
| `--submodules` | off | Must match `--submodules` flag used during training |
| `--max-new-tokens` | `512` | Maximum tokens to generate per prompt |
| `--do-sample` | off | Use sampling instead of greedy decoding |
| `--temperature` | `0.7` | Sampling temperature (only with `--do-sample`) |
| `--top-p` | `0.9` | Top-p nucleus sampling (only with `--do-sample`) |
| `--output-file` | auto | Override the output JSON path |
| `--generator-name` | auto | Generator name written into the output JSON |
| `--quick` | off | Run on 5 hardcoded prompts only (sanity check) |

### Example variants

```bash
# Quick sanity check on 5 prompts
python inference_it.py \
  --model-name meta-llama/Llama-3.1-8B \
  --ckpt-path checkpoints/... \
  --adapter-rank 8 \
  --quick

# Full inference with sampling
python inference_it.py \
  --model-name meta-llama/Llama-3.1-8B \
  --ckpt-path checkpoints/... \
  --adapter-rank 8 \
  --do-sample --temperature 0.7 --top-p 0.9 \
  --output-file outputs/my_model_generations.json
```

## AlpacaEval Scoring

Scores one or more generation files using GPT-4 Turbo as judge. This makes live OpenAI API calls (~$5–10 per model for 805 prompts).

### Basic usage

```bash
python run_alpaca_eval.py \
  --model-outputs outputs/my_model_generations.json
```

### Full argument reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-outputs` | *(required)* | Path(s) to generation JSON file(s); accepts multiple for comparison |
| `--output-dir` | `results/alpaca_eval` | Directory to save evaluation results |
| `--annotators-config` | `alpaca_eval_gpt4_turbo_fn` | Judge config (AlpacaEval 2.0 default) |

### Comparing multiple models

```bash
python run_alpaca_eval.py \
  --model-outputs outputs/model_a.json outputs/model_b.json \
  --output-dir results/comparison
```

A `comparison_summary.csv` is saved to `--output-dir` when multiple files are provided.

## Outputs

| Path | Contents | Script |
|------|----------|--------|
| `checkpoints/<run>/` | Saved model checkpoint (`pytorch_model.bin`) | `train_it.py --save-model` |
| `it_outputs/<run>/alpaca_eval_generations.json` | AlpacaEval generations | `train_it.py --do-eval` |
| `it_outputs/<run>/eval_meta.json` | Metadata for the eval run | `train_it.py --do-eval` |
| `outputs/<ckpt_basename>_generations.json` | AlpacaEval generations | `inference_it.py` |
| `<output-dir>/<model>/` | Raw AlpacaEval annotations and win rates | `run_alpaca_eval.py` |
| `<output-dir>/comparison_summary.csv` | Side-by-side LC win rates (multi-model only) | `run_alpaca_eval.py` |

## Notes

- All scripts must be run from the `instruction_tuning/` directory — they use `sys.path.append('../')` to import from the parent package.
- Gemma models are loaded with `attn_implementation="eager"` automatically.
- All base model weights are frozen; only steering parameters are trained.
- Loss is computed on response tokens only — prompt tokens are masked with `-100`.
- Effective batch size = `--bs × --gradient-accumulation-steps` (default: 4 × 4 = 16).
- Training logs to WandB automatically under the project `steerling_IT_<dataset>_<model>`.
- `--do-eval` runs generation after training but does not save the model. Use `--save-model` with `inference_it.py` for the two-step workflow.
- AlpacaEval LC (length-controlled) win rate is the standard reported metric for AlpacaEval 2.0.
