"""
Instruction Tuning Dataloaders for Steerling
=============================================
Provides datasets and collators for instruction tuning with Alpaca-style data.
Designed for causal LM training where loss is computed only on response tokens.

Usage:
    from instruction_tuning_dataloaders import get_instruction_dataloader

    train_loader = get_instruction_dataloader(
        tokenizer=tokenizer,
        dataset_name="alpaca",
        split="train",
        batch_size=8,
        max_length=512,
    )

Compatible with: meta-llama/Llama-3.1-8B (base model)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Optional, Dict, List, Any
from dataclasses import dataclass


# ──────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────────────────────────

ALPACA_PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def format_alpaca_prompt(example: Dict[str, str]) -> Dict[str, str]:
    """Format a single Alpaca example into prompt and response strings."""
    if example.get("input", "").strip():
        prompt = ALPACA_PROMPT_WITH_INPUT.format(
            instruction=example["instruction"],
            input=example["input"],
        )
    else:
        prompt = ALPACA_PROMPT_NO_INPUT.format(
            instruction=example["instruction"],
        )
    response = example["output"]
    return {"prompt": prompt, "response": response}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class InstructionTuningDataset(Dataset):
    """
    Dataset for instruction tuning that tokenizes on-the-fly and produces
    input_ids, attention_mask, and labels with prompt tokens masked out
    (set to -100) so loss is computed only on the response.
    """

    def __init__(
        self,
        tokenizer,
        dataset_name: str = "alpaca",
        split: str = "train",
        max_length: int = 512,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load the raw dataset
        raw_dataset = self._load_raw_dataset(dataset_name, split)

        # Optionally subsample
        if max_samples is not None and max_samples < len(raw_dataset):
            raw_dataset = raw_dataset.shuffle(seed=seed).select(range(max_samples))

        self.data = raw_dataset

    def _load_raw_dataset(self, dataset_name: str, split: str):
        """Load a HuggingFace instruction dataset by name."""
        dataset_registry = {
            "alpaca":         ("tatsu-lab/alpaca", None),
            "alpaca_cleaned": ("yahma/alpaca-cleaned", None),
            "code_alpaca":    ("lucasmccabe-lmi/CodeAlpaca-20k", None),
        }

        if dataset_name in dataset_registry:
            path, subset = dataset_registry[dataset_name]
            ds = load_dataset(path, subset, split=split)
        else:
            # Try loading as a HuggingFace dataset path directly
            ds = load_dataset(dataset_name, split=split)

        return ds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        formatted = format_alpaca_prompt(example)

        prompt_str = formatted["prompt"]
        response_str = formatted["response"]

        # Tokenize prompt and response separately to know the boundary
        prompt_tokens = self.tokenizer(
            prompt_str,
            add_special_tokens=True,     # adds BOS
            truncation=False,
        )
        response_tokens = self.tokenizer(
            response_str + self.tokenizer.eos_token,
            add_special_tokens=False,    # no extra BOS
            truncation=False,
        )

        prompt_ids = prompt_tokens["input_ids"]
        response_ids = response_tokens["input_ids"]
        prompt_len = len(prompt_ids)

        # Concatenate and truncate
        input_ids = (prompt_ids + response_ids)[:self.max_length]
        attention_mask = [1] * len(input_ids)

        # Labels: mask prompt tokens with -100
        labels = ([-100] * prompt_len + response_ids)[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Collator (pads a batch to uniform length)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InstructionTuningCollator:
    """
    Pads input_ids, attention_mask, and labels to the longest sequence in the
    batch.  Uses the tokenizer's pad_token_id for input_ids and -100 for labels
    so that padding positions are ignored in the loss.
    """
    tokenizer: Any
    padding_side: str = "right"

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(f["input_ids"].size(0) for f in features)
        pad_id = self.tokenizer.pad_token_id

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            seq_len = f["input_ids"].size(0)
            pad_len = max_len - seq_len

            if self.padding_side == "right":
                batch_input_ids.append(
                    torch.cat([f["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)])
                )
                batch_attention_mask.append(
                    torch.cat([f["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
                )
                batch_labels.append(
                    torch.cat([f["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
                )
            else:  # left padding (useful for generation)
                batch_input_ids.append(
                    torch.cat([torch.full((pad_len,), pad_id, dtype=torch.long), f["input_ids"]])
                )
                batch_attention_mask.append(
                    torch.cat([torch.zeros(pad_len, dtype=torch.long), f["attention_mask"]])
                )
                batch_labels.append(
                    torch.cat([torch.full((pad_len,), -100, dtype=torch.long), f["labels"]])
                )

        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def get_instruction_dataloader(
    tokenizer,
    dataset_name: str = "alpaca",
    split: str = "train",
    batch_size: int = 8,
    max_length: int = 512,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    padding_side: str = "right",
) -> DataLoader:
    """
    Build and return a DataLoader for instruction tuning.

    Args:
        tokenizer:      HuggingFace tokenizer (should have pad_token set).
        dataset_name:   One of "alpaca", "alpaca_cleaned", "code_alpaca",
                        or any HuggingFace dataset path with instruction/input/output cols.
        split:          Dataset split (default "train").
        batch_size:     Batch size.
        max_length:     Maximum sequence length after tokenization.
        max_samples:    If set, subsample the dataset to this many examples.
        shuffle:        Whether to shuffle.
        num_workers:    DataLoader workers.
        seed:           Random seed for subsampling.
        padding_side:   "right" (training default) or "left" (generation).

    Returns:
        torch.utils.data.DataLoader
    """
    # Ensure pad token is set (Llama models don't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = InstructionTuningDataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split=split,
        max_length=max_length,
        max_samples=max_samples,
        seed=seed,
    )

    collator = InstructionTuningCollator(
        tokenizer=tokenizer,
        padding_side=padding_side,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dataset registry (mirrors the pattern from utils.dataset_dict in steerling)
# ──────────────────────────────────────────────────────────────────────────────

INSTRUCTION_DATASET_DICT = {
    "alpaca": {
        "hf_path": "tatsu-lab/alpaca",
        "description": "52k instruction-response pairs from Stanford Alpaca",
        "eval_benchmark": "alpaca_eval",
    },
    "alpaca_cleaned": {
        "hf_path": "yahma/alpaca-cleaned",
        "description": "Cleaned version of Alpaca with fixed hallucinations and formatting",
        "eval_benchmark": "alpaca_eval",
    },
    "code_alpaca": {
        "hf_path": "lucasmccabe-lmi/CodeAlpaca-20k",
        "description": "20k code instruction-response pairs",
        "eval_benchmark": "humaneval",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from transformers import AutoTokenizer

    model_id = "meta-llama/Llama-3.1-8B"
    print(f"Loading tokenizer for {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Building dataloader (alpaca, first 100 samples) ...")
    loader = get_instruction_dataloader(
        tokenizer=tokenizer,
        dataset_name="alpaca",
        split="train",
        batch_size=4,
        max_length=512,
        max_samples=100,
    )

    batch = next(iter(loader))
    print(f"  input_ids shape:      {batch['input_ids'].shape}")
    print(f"  attention_mask shape:  {batch['attention_mask'].shape}")
    print(f"  labels shape:         {batch['labels'].shape}")

    # Show one decoded example
    idx = 0
    ids = batch["input_ids"][idx]
    labs = batch["labels"][idx]
    mask = batch["attention_mask"][idx]

    real_len = mask.sum().item()
    # Find first non-masked label to locate prompt/response boundary
    response_mask = (labs != -100)
    if response_mask.any():
        response_start = response_mask.nonzero(as_tuple=True)[0][0].item()
        response_end = response_mask.nonzero(as_tuple=True)[0][-1].item() + 1
    else:
        response_start = real_len
        response_end = real_len

    prompt_len = response_start
    response_len = response_end - response_start
    pad_len = len(ids) - real_len

    print(f"\n  Total tokens:    {len(ids)}")
    print(f"  Prompt length:   {prompt_len} tokens")
    print(f"  Response length: {response_len} tokens")
    print(f"  Padding length:  {pad_len} tokens")
    print(f"\n  === PROMPT ===\n{tokenizer.decode(ids[:prompt_len], skip_special_tokens=False)}")
    print(f"\n  === RESPONSE ===\n{tokenizer.decode(ids[prompt_len:response_end], skip_special_tokens=False)}")