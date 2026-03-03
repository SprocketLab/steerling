"""
train_it.py — Instruction Tuning with Steerling
=================================================
Fine-tunes post-block steering adapters (or fixed vectors) on instruction-
following data (Alpaca).  Mirrors the structure of train.py but uses the
instruction tuning dataloader with response-only loss masking.

Example usage:
    python train_it.py \
        --model-name meta-llama/Llama-3.1-8B \
        --dataset-name alpaca_cleaned \
        --adapter --adapter-rank 8 \
        --lr 2e-3 --bs 4 --epochs 3 \
        --max-length 512 \
        --do-eval
"""

import argparse
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import wandb


import sys
sys.path.append('../')
from utils import InstructionTuningDataset, InstructionTuningCollator
from steerer import FixedVectorSteerer, AdapterSteerer


# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_EPOCHS = {
    "alpaca": 3,
    "alpaca_cleaned": 3,
    "code_alpaca": 3,
}

DEFAULT_MAX_NEW_TOKENS = 256


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--model-name", type=str, required=True)

    # Data
    parser.add_argument("--dataset-name", type=str, default="alpaca_cleaned",
                        choices=["alpaca", "alpaca_cleaned", "code_alpaca"],
                        help="Instruction dataset to train on")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max sequence length (prompt + response)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Subsample training set (useful for debugging)")

    # Training
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    # Steering config
    parser.add_argument("--adapter", action="store_true",
                        help="Use adapter modules instead of fixed vectors")
    parser.add_argument("--adapter-rank", type=int, default=8,
                        help="Rank of adapter modules")
    parser.add_argument("--submodules", action="store_true",
                        help="Steer submodules (mlp, attn) instead of block output")
    parser.add_argument("--linear", action="store_true",
                        help="Disable nonlinearity in adapters")
    parser.add_argument("--nonlinearity", type=str, default="silu")

    # Eval / saving
    parser.add_argument("--do-eval", action="store_true",
                        help="Run generation eval after training (does not save model)")
    parser.add_argument("--save-model", action="store_true",
                        help="Save model checkpoint after training")

    args = parser.parse_args()

    # ──────────────────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────────────────

    dataset_name = args.dataset_name
    train_epochs = args.epochs if args.epochs is not None else DEFAULT_EPOCHS.get(dataset_name, 3)
    model_name = args.model_name

    print("=" * 60)
    print(f"  Instruction Tuning with Steerling")
    print(f"  Model:      {model_name}")
    print(f"  Dataset:    {dataset_name}")
    print(f"  Epochs:     {train_epochs}")
    print(f"  LR:         {args.lr}")
    print(f"  Batch size: {args.bs} x {args.gradient_accumulation_steps} accum")
    print(f"  Max length: {args.max_length}")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────────────────
    # Tokenizer & Model
    # ──────────────────────────────────────────────────────────────────────

    model_kwargs = {}
    if "gemma" in model_name.lower():
        model_kwargs["attn_implementation"] = "eager"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        **model_kwargs,
    )

    # Freeze base model
    for p in base_model.parameters():
        p.requires_grad_(False)

    # ──────────────────────────────────────────────────────────────────────
    # Steerer
    # ──────────────────────────────────────────────────────────────────────

    targets = ("mlp", "attn") if args.submodules else ("block",)

    if args.adapter:
        steered_model = AdapterSteerer(
            base_model,
            layers_to_steer="all",
            targets=targets,
            apply_to="all",  # instruction tuning: steer all tokens
            rank=args.adapter_rank,
            activation=None if args.linear else args.nonlinearity,
        )
    else:
        steered_model = FixedVectorSteerer(
            base_model,
            layers_to_steer="all",
            targets=targets,
            apply_to="all",
        )

    # Print trainable params
    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in steered_model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} "
          f"({trainable_params / total_params * 100:.4f}%)")
    print("Trainable param names:")
    for n, p in steered_model.named_parameters():
        if p.requires_grad:
            print(f"  {n}  {tuple(p.shape)}")
    print()

    # ──────────────────────────────────────────────────────────────────────
    # Dataset
    # ──────────────────────────────────────────────────────────────────────

    train_dataset = InstructionTuningDataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split="train",
        max_length=args.max_length,
        max_samples=args.max_samples,
    )

    # Drop this into train_it.py right after building the dataset
    sample = train_dataset[0]
    n_total = sample["labels"].numel()
    n_response = (sample["labels"] != -100).sum().item()
    print(f"Sanity check: {n_response}/{n_total} tokens have loss computed")

    collator = InstructionTuningCollator(
        tokenizer=tokenizer,
        padding_side="right",
    )

    print(f"Training examples: {len(train_dataset)}")

    # ──────────────────────────────────────────────────────────────────────
    # Project naming (for wandb + checkpoint dir)
    # ──────────────────────────────────────────────────────────────────────

    model_short = model_name.split("/")[-1]

    if args.adapter:
        method_str = f"adapter_r{args.adapter_rank}"
        method_str += "_linear" if args.linear else f"_nonlinear_{args.nonlinearity}"
    else:
        method_str = "fixedvec"

    if args.submodules:
        method_str += "_submod"
    else:
        method_str += "_block"

    project_name = (
        f"IT_{dataset_name}/{model_short}_{method_str}/"
        f"lr{args.lr}_bs{args.bs}x{args.gradient_accumulation_steps}"
        f"_ep{train_epochs}_warmup{args.warmup_ratio}_wd{args.weight_decay}"
    )

    wandb.init(
        project=f"steerling_IT_{dataset_name}_{model_short}",
        name=method_str,
        config=vars(args),
        settings=wandb.Settings(init_timeout=120),
    )

    # ──────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────

    training_args = TrainingArguments(
        output_dir=f"./{project_name}",
        per_device_train_batch_size=args.bs,
        num_train_epochs=train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        remove_unused_columns=False,
        logging_strategy="steps",
        logging_steps=10,
        report_to=["wandb"],
        save_safetensors=False,
        save_total_limit=1,
        save_strategy="no",
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=steered_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train()

    # ──────────────────────────────────────────────────────────────────────
    # Save
    # ──────────────────────────────────────────────────────────────────────

    if args.save_model:
        save_dir = os.path.join("checkpoints", project_name)
        os.makedirs(save_dir, exist_ok=True)
        trainer.save_model(save_dir)
        print(f"Model saved to {save_dir}")

    # ──────────────────────────────────────────────────────────────────────
    # Eval: generate on a few prompts to sanity-check
    # ──────────────────────────────────────────────────────────────────────

    if args.do_eval:
        import json
        from datasets import load_dataset
        from tqdm import tqdm

        print("\n" + "=" * 60)
        print("  Generating on Alpaca-Eval (805 prompts)")
        print("=" * 60 + "\n")

        steered_model.eval()
        device = next(steered_model.parameters()).device

        # Load the official 805 Alpaca-Eval prompts
        try:
            eval_dataset = load_dataset(
                "tatsu-lab/alpaca_farm", "alpaca_farm_evaluation",
                split="eval", trust_remote_code=True,
            )
        except Exception:
            eval_dataset = load_dataset(
                "parquet",
                data_files="hf://datasets/tatsu-lab/alpaca_farm/alpaca_farm_evaluation/eval-00000-of-00001.parquet",
                split="train",
            )

        from instruction_tuning_dataloaders import ALPACA_PROMPT_NO_INPUT, ALPACA_PROMPT_WITH_INPUT

        results = []
        for example in tqdm(eval_dataset, desc="Generating"):
            instruction = example["instruction"]
            input_text = example.get("input", "")

            if input_text.strip():
                formatted = ALPACA_PROMPT_WITH_INPUT.format(
                    instruction=instruction, input=input_text
                )
            else:
                formatted = ALPACA_PROMPT_NO_INPUT.format(instruction=instruction)

            inputs = tokenizer(formatted, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = steered_model.generate(
                    **inputs,
                    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                    do_sample=False,       # greedy for reproducibility
                    eos_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            ).strip()

            # alpaca_eval expects: instruction, output, generator
            results.append({
                "instruction": instruction,
                "input": input_text,
                "output": response,
                "generator": f"steerling_{method_str}",
            })

        # Save generations (alpaca_eval-compatible format)
        output_dir = os.path.join("it_outputs", project_name)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "alpaca_eval_generations.json")

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nGenerations saved to {output_file}")
        print(f"  ({len(results)} examples)")
        print(f"\nTo evaluate, run:")
        print(f"  alpaca_eval --model_outputs {output_file}")

        # Also save a metadata sidecar
        meta_file = os.path.join(output_dir, "eval_meta.json")
        with open(meta_file, "w") as f:
            json.dump({
                "config": vars(args),
                "model_name": model_name,
                "method": method_str,
                "trainable_params": trainable_params,
                "trainable_pct": trainable_params / total_params * 100,
                "num_eval_examples": len(results),
            }, f, indent=2)

    wandb.finish()
    print("\nDone.")