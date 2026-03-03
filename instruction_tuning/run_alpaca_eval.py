"""
run_alpaca_eval.py — Score generations using AlpacaEval 2.0 (length-controlled)
================================================================================
Wraps the official alpaca_eval package to score a JSON file of model generations
against the GPT-4 Turbo reference, using GPT-4 Turbo as the judge.

Prerequisites:
    pip install alpaca_eval
    export OPENAI_API_KEY="sk-..."

Usage:
    # Score a single model
    python run_alpaca_eval.py \
        --model-outputs outputs/steerling_r16_linear_alpaca_eval.json \
        --output-dir results/steerling_r16

    # Score and compare two models
    python run_alpaca_eval.py \
        --model-outputs outputs/steerling_r16_linear_alpaca_eval.json \
                        outputs/lora_r8_full_alpaca_eval.json \
        --output-dir results/comparison

Notes:
    - Costs ~$5-10 per model in OpenAI API calls (805 prompts x GPT-4 Turbo judge)
    - The JSON must have fields: "instruction", "output", "generator"
    - "input" field is optional (used for prompts with additional context)
    - Results include both raw win rate and length-controlled (LC) win rate
    - LC win rate is the standard metric for AlpacaEval 2.0
"""

import argparse
import json
import os
import sys


def validate_json(filepath):
    """Check that the JSON file has the required fields."""
    with open(filepath) as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        print(f"ERROR: {filepath} must be a non-empty JSON array")
        return False

    required = {"instruction", "output"}
    sample = data[0]
    missing = required - set(sample.keys())
    if missing:
        print(f"ERROR: {filepath} missing required fields: {missing}")
        return False

    # Add generator field if missing
    has_generator = "generator" in sample
    if not has_generator:
        print(f"WARNING: No 'generator' field found, adding default...")
        basename = os.path.splitext(os.path.basename(filepath))[0]
        for item in data:
            item["generator"] = basename
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    print(f"  {filepath}: {len(data)} examples, generator='{data[0].get('generator', 'unknown')}'")
    return True


def run_eval(model_outputs_path, output_dir, annotators_config="alpaca_eval_gpt4_turbo_fn"):
    """Run AlpacaEval 2.0 LC on a single model output file."""
    from alpaca_eval import evaluate

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nRunning AlpacaEval 2.0 (LC) on: {model_outputs_path}")
    print(f"  Judge: {annotators_config}")
    print(f"  Output dir: {output_dir}")
    print(f"  This will make ~805 API calls to OpenAI...\n")

    df_leaderboard, annotations = evaluate(
        model_outputs=model_outputs_path,
        output_path=output_dir,
        annotators_config=annotators_config,
        is_return_instead_of_print=True,
    )

    return df_leaderboard


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AlpacaEval 2.0 (length-controlled) scoring"
    )
    parser.add_argument(
        "--model-outputs", type=str, nargs="+", required=True,
        help="Path(s) to JSON file(s) with model generations"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/alpaca_eval",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--annotators-config", type=str, default="alpaca_eval_gpt4_turbo_fn",
        help="Judge config (default: alpaca_eval_gpt4_turbo_fn for AlpacaEval 2.0)"
    )

    args = parser.parse_args()

    # ──────────────────────────────────────────────────────────────────────
    # Check prerequisites
    # ──────────────────────────────────────────────────────────────────────

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    try:
        import alpaca_eval
    except ImportError:
        print("ERROR: alpaca_eval not installed.")
        print("  pip install alpaca_eval")
        sys.exit(1)

    # ──────────────────────────────────────────────────────────────────────
    # Validate inputs
    # ──────────────────────────────────────────────────────────────────────

    print("Validating input files...")
    for path in args.model_outputs:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)
        if not validate_json(path):
            sys.exit(1)

    # ──────────────────────────────────────────────────────────────────────
    # Run evaluation for each model
    # ──────────────────────────────────────────────────────────────────────

    all_results = []
    for path in args.model_outputs:
        basename = os.path.splitext(os.path.basename(path))[0]
        model_output_dir = os.path.join(args.output_dir, basename)

        df = run_eval(path, model_output_dir, args.annotators_config)
        all_results.append((basename, df))

        print(f"\n{'=' * 60}")
        print(f"  Results for: {basename}")
        print(f"{'=' * 60}")
        print(df.to_string())
        print()

    # ──────────────────────────────────────────────────────────────────────
    # Summary comparison (if multiple models)
    # ──────────────────────────────────────────────────────────────────────

    if len(all_results) > 1:
        import pandas as pd

        print(f"\n{'=' * 60}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'=' * 60}")

        combined = pd.concat([df for _, df in all_results])
        print(combined.to_string())

        # Save comparison
        summary_file = os.path.join(args.output_dir, "comparison_summary.csv")
        combined.to_csv(summary_file)
        print(f"\nSaved comparison to {summary_file}")

    print("\nDone.")
