# steerling

Repository for post-block low rank steerin described in https://sprocketlab.github.io/posts/2025/11/actvweight/

## Environment Setup

The repo ships with `setup_env.sh`, which provisions a Conda environment named `steerling` (Python 3.10) and installs all runtime dependencies.

```bash
# Create or update the Conda env
bash setup_env.sh

# Activate it before running any scripts
conda activate steerling
```

The script installs GPU-enabled PyTorch wheels by default. Set `TORCH_CUDA_CHANNEL=https://download.pytorch.org/whl/cpu` before running it if you are on CPU-only hardware.

## Training

`train.py` fine-tunes adapters or fixed vectors on any dataset defined under `utils.dataset_dict`.

```bash
conda activate steerling
python train.py \
  --model-name meta-llama/Meta-Llama-3-8B \
  --dataset-name ListOps \
  --lr 1e-3 \
  --bs 8
```

Defaults match the method in the blog (adapters applied to all tokens). If you want, you can experiment variants with:

- `--no-adapter` – use fixed steering vectors (fewer trainable params, less expressive, slower training because we optimize only the last token).
- `--intervene-last` – restrict steering to the final token; incurs an extra forward pass per generation step.
- `--submodules` – inject adapters into attention/MLP submodules rather than the entire block.

Performance under these alternate configurations is not guaranteed.

`train.py --help` lists every option.

## Inference / Evaluation

Given a trained checkpoint directory, run:

```bash
conda activate steerling
python inference.py \
  --model-name [your chosen model to train]\
  --dataset-name [your chosen dataset] \
  --ckpt-path [the checkpoint of saved model from train.py] \
  --split test
```

Outputs are written to `outputs/<project>/<split>.json` unless `--output-file` is provided.
