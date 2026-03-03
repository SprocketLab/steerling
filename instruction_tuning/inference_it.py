"""
inference_it.py — Load a trained steerling checkpoint and generate responses
===================================================================================
Loads a saved AdapterSteerer checkpoint (from trainer.save_model()) and generates
on the Alpaca-Eval 805 prompts, saving results as an alpaca_eval-compatible JSON.

Usage:
    python inference_it.py \
        --model-name meta-llama/Llama-3.1-8B \
        --ckpt-path checkpoints/IT_alpaca_cleaned/... \
        --adapter-rank 16 \
        --linear \
        --output-file outputs/steerling_r16_generations.json

    # Quick sanity check on 5 hardcoded prompts:
    python inference_it.py \
        --model-name meta-llama/Llama-3.1-8B \
        --ckpt-path checkpoints/... \
        --adapter-rank 16 --linear \
        --quick
"""

import argparse
import json
import os

os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import sys
sys.path.append('../')
from steerer import AdapterSteerer
from utils import (
    ALPACA_PROMPT_NO_INPUT,
    ALPACA_PROMPT_WITH_INPUT,
)
DEFAULT_MAX_NEW_TOKENS = 512


def load_eval_prompts():
    from datasets import load_dataset
    ds = load_dataset(
        "tatsu-lab/alpaca_farm", "alpaca_farm_evaluation",
        split="eval", trust_remote_code=True,
    )
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--model-name", type=str, required=True,
                        help="Base model HF name (e.g. meta-llama/Llama-3.1-8B)")
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Path to saved checkpoint directory (from trainer.save_model)")

    # Adapter config (must match training)
    parser.add_argument("--adapter-rank", type=int, default=16)
    parser.add_argument("--linear", action="store_true",
                        help="Use linear adapters (no nonlinearity)")
    parser.add_argument("--nonlinearity", type=str, default="silu")
    parser.add_argument("--submodules", action="store_true",
                        help="Steer submodules (mlp, attn) instead of block")

    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--do-sample", action="store_true",
                        help="Use sampling instead of greedy")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)

    # Output
    parser.add_argument("--output-file", type=str, default=None,
                        help="Output JSON path (default: outputs/<ckpt_basename>_generations.json)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick sanity check on 5 hardcoded prompts instead of full eval")
    parser.add_argument("--generator-name", type=str, default=None,
                        help="Generator name for alpaca_eval JSON (default: auto)")

    args = parser.parse_args()

    # ──────────────────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────────────────

    print("=" * 60)
    print("  Steerling Generation from Checkpoint")
    print(f"  Model:       {args.model_name}")
    print(f"  Checkpoint:  {args.ckpt_path}")
    print(f"  Rank:        {args.adapter_rank}")
    print(f"  Linear:      {args.linear}")
    print(f"  Submodules:  {args.submodules}")
    print(f"  Sampling:    {args.do_sample}")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────────────────
    # Load tokenizer
    # ──────────────────────────────────────────────────────────────────────

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for generation

    # ──────────────────────────────────────────────────────────────────────
    # Load base model + build steerer (same architecture as training)
    # ──────────────────────────────────────────────────────────────────────

    print("\nLoading base model...")
    model_kwargs = {}
    if "gemma" in args.model_name.lower():
        model_kwargs["attn_implementation"] = "eager"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        **model_kwargs,
    )

    for p in base_model.parameters():
        p.requires_grad_(False)

    targets = ("mlp", "attn") if args.submodules else ("block",)

    steered_model = AdapterSteerer(
        base_model,
        layers_to_steer="all",
        targets=targets,
        apply_to="all",
        rank=args.adapter_rank,
        activation=None if args.linear else args.nonlinearity,
    )

    # ──────────────────────────────────────────────────────────────────────
    # Load checkpoint weights
    # ──────────────────────────────────────────────────────────────────────

    print(f"Loading checkpoint from {args.ckpt_path} ...")
    ckpt_file = os.path.join(args.ckpt_path, "pytorch_model.bin")
    if not os.path.exists(ckpt_file):
        # Maybe it's a single file path
        ckpt_file = args.ckpt_path

    state_dict = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    # The checkpoint from trainer.save_model() saves the full AdapterSteerer state_dict
    # which includes base_model weights (frozen) + adapter weights (trained).
    # We only need to load the adapter weights since the base model is already loaded.
    # But strict=False handles this — it loads what matches.
    missing, unexpected = steered_model.load_state_dict(state_dict, strict=False)

    # Check that adapters were loaded
    adapter_keys_loaded = [k for k in state_dict.keys() if "adapters" in k]
    print(f"  Loaded {len(adapter_keys_loaded)} adapter parameter tensors")
    if missing:
        # Filter out expected missing keys (base model keys that device_map handles)
        adapter_missing = [k for k in missing if "adapters" in k]
        if adapter_missing:
            print(f"  WARNING: {len(adapter_missing)} adapter keys missing!")
            for k in adapter_missing[:5]:
                print(f"    {k}")
    print()
    steered_model.to("cuda")
    steered_model.eval()
    device = next(steered_model.base_model.parameters()).device

    # ──────────────────────────────────────────────────────────────────────
    # Prepare prompts
    # ──────────────────────────────────────────────────────────────────────

    if args.quick:
        print("Quick sanity check mode (5 prompts)\n")
        eval_data = [
            {"instruction": "Give three tips for staying healthy.", "input": ""},
            {"instruction": "Write a short poem about the ocean.", "input": ""},
            {"instruction": "Explain the concept of supply and demand in simple terms.", "input": ""},
            {"instruction": "What are the main differences between Python and JavaScript?", "input": ""},
            {"instruction": "Summarize the plot of Romeo and Juliet in two sentences.", "input": ""},
        ]
    else:
        print("Loading Alpaca-Eval 805 prompts...")
        eval_data = load_eval_prompts()
        print(f"  {len(eval_data)} prompts loaded\n")

    # ──────────────────────────────────────────────────────────────────────
    # Generate
    # ──────────────────────────────────────────────────────────────────────

    # Auto generator name
    if args.generator_name is None:
        method = f"adapter_r{args.adapter_rank}"
        method += "_linear" if args.linear else f"_nonlinear_{args.nonlinearity}"
        method += "_submod" if args.submodules else "_block"
        generator_name = f"steerling_{method}"
    else:
        generator_name = args.generator_name

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    if args.do_sample:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
    else:
        gen_kwargs.update(do_sample=False)

    results = []
    for example in tqdm(eval_data, desc="Generating"):
        instruction = example["instruction"]
        input_text = example.get("input", "") or ""

        if input_text.strip():
            formatted = ALPACA_PROMPT_WITH_INPUT.format(
                instruction=instruction, input=input_text
            )
        else:
            formatted = ALPACA_PROMPT_NO_INPUT.format(instruction=instruction)

        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = steered_model.generate(**inputs, **gen_kwargs)

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        results.append({
            "instruction": instruction,
            "input": input_text,
            "output": response,
            "generator": generator_name,
        })

        if args.quick:
            print(f"Q: {instruction}")
            print(f"A: {response}")
            print("-" * 40)

    # ──────────────────────────────────────────────────────────────────────
    # Save
    # ──────────────────────────────────────────────────────────────────────

    if args.output_file is None:
        ckpt_basename = os.path.basename(args.ckpt_path.rstrip("/"))
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{ckpt_basename}_generations.json")
    else:
        output_file = args.output_file
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} generations to {output_file}")
    if not args.quick:
        print(f"\nTo evaluate with alpaca_eval:")
        print(f"  alpaca_eval --model_outputs {output_file}")

    print("\nDone.")